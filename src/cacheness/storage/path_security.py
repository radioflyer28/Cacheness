"""Fail-closed filesystem containment helpers for managed blob payloads.

The configured storage root is resolved once by a backend instance.  A root
symlink is therefore a supported configuration alias, but every component below
that resolved root must be an ordinary directory or file.  On descriptor-capable
Unix platforms operations walk from the anchored root descriptor with
``O_NOFOLLOW``.  The portable fallback serializes operations and repeats the
containment/reparse checks immediately before filesystem calls.

The fallback cannot close the final check-to-kernel-call race against a separate
same-user or privileged process that can mutate the managed root on platforms
without descriptor-relative no-follow operations.  Deployments outside the
trusted-host boundary must protect the root with OS ACLs or use a
descriptor-capable Unix filesystem.
"""

from __future__ import annotations

import errno
import ctypes
import hashlib
from io import BytesIO
import os
import re
import stat
import sys
import threading
import uuid
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import BinaryIO, Iterator, NoReturn, Union

from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheReason,
    CacheUnsafePathError,
)


_BLOB_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")
_MAX_BLOB_ID_LENGTH = 256
_PHYSICAL_NAME_DOMAIN = b"cacheness.physical-name.v1\x00"
_LOCK_AUTHORITY_VERSION = b"cacheness.lock-authority.v1\x00"
_ROOT_AUTHORITY_XATTR_PREFIX = "user.cacheness.lifecycle-lock."
_DARWIN_ROOT_AUTHORITY_XATTR_PREFIX = "com.cacheness.lifecycle-lock."


def _platform_name() -> str:
    """Resolve filesystem topology through a narrow test seam."""
    return os.name


class _WindowsFileApi:
    """Small native Windows surface for durable lifecycle publication.

    Python's ``os.replace`` is suitable for ordinary payload replacement, but
    it cannot express the no-replace control-record transition.  These calls
    intentionally live behind a tiny injectable adapter so POSIX development
    can exercise the Windows protocol without claiming it has executed the
    actual Win32 branch.
    """

    _GENERIC_READ = 0x80000000
    _GENERIC_WRITE = 0x40000000
    _FILE_SHARE_READ = 0x00000001
    _FILE_SHARE_WRITE = 0x00000002
    _FILE_SHARE_DELETE = 0x00000004
    _OPEN_EXISTING = 3
    _FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    _ERROR_FILE_EXISTS = 80
    _ERROR_ALREADY_EXISTS = 183

    def __init__(self) -> None:
        try:
            self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        except (AttributeError, OSError) as exc:  # pragma: no cover - Windows only.
            raise CacheBlobBackendError(
                "BlobStore lifecycle durability is unavailable on this Windows runtime",
                context={"operation": "windows_lifecycle_durability"},
            ) from exc

        self._move_file_ex = self._kernel32.MoveFileExW
        self._move_file_ex.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
        self._move_file_ex.restype = ctypes.c_int
        self._create_file = self._kernel32.CreateFileW
        self._create_file.argtypes = (
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        )
        self._create_file.restype = ctypes.c_void_p
        self._flush_file_buffers = self._kernel32.FlushFileBuffers
        self._flush_file_buffers.argtypes = (ctypes.c_void_p,)
        self._flush_file_buffers.restype = ctypes.c_int
        self._close_handle = self._kernel32.CloseHandle
        self._close_handle.argtypes = (ctypes.c_void_p,)
        self._close_handle.restype = ctypes.c_int

    @staticmethod
    def _last_error() -> int:
        """Return the native error through a seam that unit tests can exercise."""
        return ctypes.get_last_error()

    @classmethod
    def _raise_last_error(cls, operation: str) -> NoReturn:
        error_number = cls._last_error()
        if error_number in {cls._ERROR_FILE_EXISTS, cls._ERROR_ALREADY_EXISTS}:
            raise FileExistsError(error_number, f"{operation} failed")
        raise OSError(error_number, f"{operation} failed")

    def rename_no_replace(self, temporary: Path, destination: Path) -> None:
        """Atomically consume ``temporary`` only when ``destination`` is absent."""
        # Omitting MOVEFILE_REPLACE_EXISTING makes MoveFileEx fail if a
        # destination already exists while consuming the source on success.
        if not self._move_file_ex(str(temporary), str(destination), 0):
            self._raise_last_error("MoveFileExW")

    def flush_directory(self, directory: Path) -> None:
        """Acknowledge the directory entry update through a directory handle."""
        handle = self._create_file(
            str(directory),
            self._GENERIC_READ | self._GENERIC_WRITE,
            self._FILE_SHARE_READ | self._FILE_SHARE_WRITE | self._FILE_SHARE_DELETE,
            None,
            self._OPEN_EXISTING,
            self._FILE_FLAG_BACKUP_SEMANTICS,
            None,
        )
        invalid_handle = ctypes.c_void_p(-1).value
        if handle == invalid_handle:
            self._raise_last_error("CreateFileW")
        flush_failure: BaseException | None = None
        try:
            if not self._flush_file_buffers(handle):
                self._raise_last_error("FlushFileBuffers")
        except BaseException as exc:
            flush_failure = exc
            raise
        finally:
            if not self._close_handle(handle) and flush_failure is None:
                self._raise_last_error("CloseHandle")


class _WindowsRegistryAuthorityApi:
    """Persist one root-inode lock binding outside mutable store pathnames.

    A registry value is attached to the current user's Windows authority
    namespace rather than to a replaceable file below the managed root.  A
    short-lived named mutex makes first publication exclusive and is abandoned
    safely by Windows when its publisher dies.  The value is only ever created
    or compared; it is never rewritten, so a later control-file swap cannot
    rebind an active store to a second advisory-lock inode.
    """

    _HKEY_CURRENT_USER = 0x80000001
    _KEY_QUERY_VALUE = 0x0001
    _KEY_SET_VALUE = 0x0002
    _REG_BINARY = 3
    _REG_OPTION_NON_VOLATILE = 0
    _ERROR_FILE_NOT_FOUND = 2
    _WAIT_OBJECT_0 = 0
    _WAIT_ABANDONED = 0x00000080
    _INFINITE = 0xFFFFFFFF

    def __init__(self) -> None:
        try:
            self._advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
            self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        except (AttributeError, OSError) as exc:  # pragma: no cover - Windows only.
            raise CacheBlobBackendError(
                "BlobStore lifecycle authority is unavailable on this Windows runtime",
                context={"operation": "windows_lifecycle_authority"},
            ) from exc

        self._reg_create = self._advapi32.RegCreateKeyExW
        self._reg_create.argtypes = (
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint32),
        )
        self._reg_create.restype = ctypes.c_long
        self._reg_query = self._advapi32.RegQueryValueExW
        self._reg_query.argtypes = (
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        )
        self._reg_query.restype = ctypes.c_long
        self._reg_set = self._advapi32.RegSetValueExW
        self._reg_set.argtypes = (
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
        )
        self._reg_set.restype = ctypes.c_long
        self._reg_close = self._advapi32.RegCloseKey
        self._reg_close.argtypes = (ctypes.c_void_p,)
        self._reg_close.restype = ctypes.c_long
        self._create_mutex = self._kernel32.CreateMutexW
        self._create_mutex.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p)
        self._create_mutex.restype = ctypes.c_void_p
        self._wait_for_single_object = self._kernel32.WaitForSingleObject
        self._wait_for_single_object.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        self._wait_for_single_object.restype = ctypes.c_uint32
        self._release_mutex = self._kernel32.ReleaseMutex
        self._release_mutex.argtypes = (ctypes.c_void_p,)
        self._release_mutex.restype = ctypes.c_int
        self._close_handle = self._kernel32.CloseHandle
        self._close_handle.argtypes = (ctypes.c_void_p,)
        self._close_handle.restype = ctypes.c_int

    @staticmethod
    def _raise_status(status: int, operation: str) -> NoReturn:
        raise OSError(status, f"{operation} failed")

    def ensure(self, name: str, value: bytes) -> None:
        """Create or verify an immutable exact binding under a crash-safe mutex."""
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
        mutex = self._create_mutex(None, False, f"Local\\CachenessAuthority-{digest}")
        if not mutex:
            _WindowsFileApi._raise_last_error("CreateMutexW")
        acquired = False
        key = ctypes.c_void_p()
        try:
            result = self._wait_for_single_object(mutex, self._INFINITE)
            if result not in {self._WAIT_OBJECT_0, self._WAIT_ABANDONED}:
                _WindowsFileApi._raise_last_error("WaitForSingleObject")
            acquired = True
            disposition = ctypes.c_uint32()
            status = self._reg_create(
                ctypes.c_void_p(self._HKEY_CURRENT_USER),
                r"Software\Cacheness\LifecycleAuthorities",
                0,
                None,
                self._REG_OPTION_NON_VOLATILE,
                self._KEY_QUERY_VALUE | self._KEY_SET_VALUE,
                None,
                ctypes.byref(key),
                ctypes.byref(disposition),
            )
            if status != 0:
                self._raise_status(status, "RegCreateKeyExW")
            value_name = digest
            value_type = ctypes.c_uint32()
            value_size = ctypes.c_uint32()
            status = self._reg_query(
                key,
                value_name,
                None,
                ctypes.byref(value_type),
                None,
                ctypes.byref(value_size),
            )
            if status == self._ERROR_FILE_NOT_FOUND:
                buffer = (ctypes.c_ubyte * len(value)).from_buffer_copy(value)
                status = self._reg_set(
                    key,
                    value_name,
                    0,
                    self._REG_BINARY,
                    ctypes.cast(buffer, ctypes.c_void_p),
                    len(value),
                )
                if status != 0:
                    self._raise_status(status, "RegSetValueExW")
                return
            if status != 0:
                self._raise_status(status, "RegQueryValueExW")
            if value_type.value != self._REG_BINARY or value_size.value != len(value):
                _unsafe_path(CacheReason.PATH_RACE)
            buffer = (ctypes.c_ubyte * value_size.value)()
            status = self._reg_query(
                key,
                value_name,
                None,
                ctypes.byref(value_type),
                ctypes.cast(buffer, ctypes.c_void_p),
                ctypes.byref(value_size),
            )
            if status != 0:
                self._raise_status(status, "RegQueryValueExW")
            if value_type.value != self._REG_BINARY or bytes(buffer) != value:
                _unsafe_path(CacheReason.PATH_RACE)
        finally:
            if key.value:
                self._reg_close(key)
            if acquired:
                self._release_mutex(mutex)
            self._close_handle(mutex)


def _windows_file_api() -> _WindowsFileApi:
    """Construct the production Win32 durability adapter only when required."""
    return _WindowsFileApi()


def _windows_registry_authority_api() -> _WindowsRegistryAuthorityApi:
    """Construct the production root-authority adapter only when required."""
    return _WindowsRegistryAuthorityApi()


def _atomic_rename_no_replace(
    parent_fd: int, temporary_name: str, name: str
) -> None:
    """Move a fully synced candidate without replacing an existing name.

    A hard-link install leaves two directory entries during the transition, so
    a process loss can strand an unaccounted temporary.  The native primitives
    below consume the temporary as part of the successful no-replace move.  A
    platform that cannot supply that property is not a truthful topology for
    durable lifecycle control evidence.
    """
    encoded_temporary = os.fsencode(temporary_name)
    encoded_name = os.fsencode(name)
    if sys.platform == "darwin":
        try:
            renameatx_np = ctypes.CDLL(None, use_errno=True).renameatx_np
        except AttributeError as exc:  # pragma: no cover - platform capability.
            raise OSError(errno.ENOTSUP, "renameatx_np is unavailable") from exc
        # Darwin's RENAME_EXCL fails when the destination already exists.
        renameatx_np.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameatx_np.restype = ctypes.c_int
        if renameatx_np(parent_fd, encoded_temporary, parent_fd, encoded_name, 0x0004) == 0:
            return
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), name)

    if sys.platform.startswith("linux"):
        try:
            renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
        except AttributeError as exc:  # pragma: no cover - old libc/kernel topology.
            raise OSError(errno.ENOTSUP, "renameat2 is unavailable") from exc
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        if renameat2(parent_fd, encoded_temporary, parent_fd, encoded_name, 1) == 0:
            return
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), name)

    raise OSError(errno.ENOTSUP, "atomic no-replace rename is unavailable")


def _unsafe_path(reason: CacheReason) -> NoReturn:
    """Raise a typed error without reflecting an untrusted locator."""
    raise CacheUnsafePathError("Unsafe managed storage path", reason=reason)


def _is_reparse_point(file_stat: os.stat_result) -> bool:
    """Return whether a Windows reparse point is present, when supported."""
    attribute = getattr(file_stat, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse_flag and attribute & reparse_flag)


def _is_link_or_reparse(file_stat: os.stat_result) -> bool:
    """Treat every link-like managed component as a race/escape boundary."""
    return stat.S_ISLNK(file_stat.st_mode) or _is_reparse_point(file_stat)


def _path_parts_contain_traversal(path_text: str) -> bool:
    """Check POSIX and Windows path grammars independent of the host OS."""
    return ".." in PurePosixPath(path_text).parts or ".." in PureWindowsPath(
        path_text
    ).parts


def validate_blob_id(blob_id: str) -> str:
    """Validate an opaque, backend-safe blob identifier.

    This deliberately does not encode or sanitize logical cache keys.  Callers
    must translate logical keys before crossing this low-level storage boundary.
    """
    if not isinstance(blob_id, str) or "\x00" in blob_id:
        _unsafe_path(CacheReason.INVALID_IDENTIFIER)

    windows_path = PureWindowsPath(blob_id)
    if blob_id.startswith(("\\\\", "//")) and windows_path.drive:
        _unsafe_path(CacheReason.PATH_UNC)
    if windows_path.drive:
        _unsafe_path(CacheReason.PATH_DRIVE)
    if blob_id.startswith("\\"):
        _unsafe_path(CacheReason.PATH_ROOTED)
    if blob_id.startswith("/"):
        _unsafe_path(CacheReason.PATH_ABSOLUTE)
    if _path_parts_contain_traversal(blob_id):
        _unsafe_path(CacheReason.PATH_TRAVERSAL)
    if len(blob_id) > _MAX_BLOB_ID_LENGTH or not _BLOB_ID_PATTERN.fullmatch(blob_id):
        _unsafe_path(CacheReason.INVALID_IDENTIFIER)
    return blob_id


def encode_physical_name(
    logical_key: str,
    prefix: str = "",
    *,
    namespace: str,
) -> str:
    """Return a deterministic opaque ID without treating caller values as paths.

    Each component is length framed in UTF-8 and bound to a versioned namespace
    before SHA-256 hashing. This keeps ``("ab", "c")`` distinct from
    ``("a", "bc")``, prevents cross-component collisions, and preserves
    Unicode byte-level distinctions without using any user value as a pathname.
    """
    if not all(isinstance(value, str) for value in (logical_key, prefix, namespace)):
        _unsafe_path(CacheReason.INVALID_IDENTIFIER)

    digest = hashlib.sha256()
    digest.update(_PHYSICAL_NAME_DOMAIN)
    for value in (namespace, prefix, logical_key):
        value_bytes = value.encode("utf-8")
        digest.update(len(value_bytes).to_bytes(8, byteorder="big"))
        digest.update(value_bytes)

    physical_name = digest.hexdigest()
    return validate_blob_id(physical_name)


def resolve_storage_root(root: Union[str, Path]) -> Path:
    """Return the canonical storage root without creating filesystem state."""
    try:
        root_path = Path(root)
    except TypeError:
        _unsafe_path(CacheReason.INVALID_IDENTIFIER)
    if "\x00" in os.fspath(root_path):
        _unsafe_path(CacheReason.INVALID_IDENTIFIER)
    return root_path.resolve(strict=False)


def _validate_locator_shape(locator_text: str) -> None:
    """Reject Windows-shaped locators even while running on POSIX hosts."""
    if "\x00" in locator_text:
        _unsafe_path(CacheReason.INVALID_IDENTIFIER)

    windows_path = PureWindowsPath(locator_text)
    if locator_text.startswith(("\\\\", "//")) and windows_path.drive:
        _unsafe_path(CacheReason.PATH_UNC)
    if windows_path.drive:
        _unsafe_path(CacheReason.PATH_DRIVE)
    if locator_text.startswith("\\"):
        _unsafe_path(CacheReason.PATH_ROOTED)
    if _path_parts_contain_traversal(locator_text):
        _unsafe_path(CacheReason.PATH_TRAVERSAL)


def _lexically_contained(root: Path, candidate: Path) -> Path:
    """Return an absolute lexical candidate only when it is below ``root``."""
    absolute_candidate = Path(os.path.abspath(os.fspath(candidate)))
    try:
        absolute_candidate.relative_to(root)
    except ValueError:
        # macOS commonly exposes the same temporary directory through both
        # ``/var`` and its resolved ``/private/var`` target.  Accept aliases
        # only after their resolved destination proves contained; subsequent
        # operations use the canonical descendant, never the alias path.
        resolved_candidate = absolute_candidate.resolve(strict=False)
        try:
            resolved_candidate.relative_to(root)
        except ValueError:
            _unsafe_path(CacheReason.PATH_OUTSIDE_ROOT)
        return resolved_candidate
    return absolute_candidate


def _reject_existing_managed_links(root: Path, candidate: Path) -> None:
    """Reject symlink and reparse-point components below the trusted root."""
    try:
        relative_parts = candidate.relative_to(root).parts
    except ValueError:
        _unsafe_path(CacheReason.PATH_OUTSIDE_ROOT)

    current = root
    for component in relative_parts:
        current = current / component
        try:
            file_stat = os.lstat(current)
        except FileNotFoundError:
            # A missing ancestor has no existing components below it to inspect.
            return
        except OSError as exc:
            raise CacheUnsafePathError(
                "Unable to verify managed storage path",
                reason=CacheReason.PATH_RACE,
            ) from exc
        if _is_link_or_reparse(file_stat):
            _unsafe_path(CacheReason.PATH_RACE)


def resolve_managed_locator(
    root: Union[str, Path],
    locator: Union[str, Path],
    *,
    operation: str,
    allow_missing_leaf: bool = False,
) -> Path:
    """Validate and return a contained managed locator without mutating state.

    Absolute locators are accepted only when they are under the resolved root;
    relative locators are interpreted beneath that root.  ``operation`` makes
    call sites explicit for auditability.  It intentionally does not alter the
    validation result, and ``allow_missing_leaf`` records operations that may
    create their final file.
    """
    del operation, allow_missing_leaf
    canonical_root = resolve_storage_root(root)
    try:
        locator_text = os.fspath(locator)
    except TypeError:
        _unsafe_path(CacheReason.INVALID_IDENTIFIER)
    if not isinstance(locator_text, str):
        _unsafe_path(CacheReason.INVALID_IDENTIFIER)
    _validate_locator_shape(locator_text)

    locator_path = Path(locator_text)
    candidate = locator_path if locator_path.is_absolute() else canonical_root / locator_path
    candidate = _lexically_contained(canonical_root, candidate)
    if candidate == canonical_root:
        _unsafe_path(CacheReason.PATH_OUTSIDE_ROOT)
    _reject_existing_managed_links(canonical_root, candidate)
    return candidate


class ManagedFileOps:
    """Perform blob filesystem operations through one containment boundary.

    ``before_operation`` is a deterministic, test-only race seam.  Production
    callers leave it as ``None``.  It is deliberately invoked after the first
    validation and before the final guarded open/stat/delete/publish step.
    """

    def __init__(self, root: Union[str, Path], *, lock: threading.RLock | None = None):
        self.root = resolve_storage_root(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Storage root is not available: {self.root}")

        root_stat = os.stat(self.root)
        self._root_identity = (root_stat.st_dev, root_stat.st_ino)
        self._lock = lock or threading.RLock()
        self.before_operation: Callable[[str, Path], None] | None = None
        self.after_control_durability_step: Callable[[str, Path], None] | None = None
        self._root_fd: int | None = None
        self._descriptor_mode = self._open_root_descriptor()
        self._lock_identities: dict[tuple[str, ...], tuple[int, int]] = {}

    def _open_root_descriptor(self) -> bool:
        """Anchor the resolved root to a descriptor when Python supports it."""
        # ``rename`` provides the same overwrite-atomic publish semantics on
        # POSIX and, unlike ``replace``, is advertised in macOS dir_fd support.
        required_functions = (os.open, os.stat, os.mkdir, os.unlink, os.rename)
        if (
            _platform_name() == "nt"
            or not hasattr(os, "O_NOFOLLOW")
            or not hasattr(os, "O_DIRECTORY")
            or not all(function in os.supports_dir_fd for function in required_functions)
        ):
            return False
        try:
            self._root_fd = os.open(
                self.root,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
        except OSError:
            self._root_fd = None
            return False
        return True

    @property
    def descriptor_mode(self) -> bool:
        """Expose whether operations are anchored by Unix directory descriptors."""
        return self._descriptor_mode

    @property
    def root_identity(self) -> tuple[int, int]:
        """Return the identity verified while this managed root was opened."""
        return self._root_identity

    def close(self) -> None:
        """Close the anchored root descriptor, if one was acquired."""
        self._lock_identities.clear()
        if self._root_fd is not None:
            os.close(self._root_fd)
            self._root_fd = None

    def _assert_root_identity(self) -> None:
        """Detect a root replacement before fallback operations use its pathname."""
        try:
            current = os.stat(self.root)
        except OSError as exc:
            raise CacheUnsafePathError(
                "Managed storage root changed during operation",
                reason=CacheReason.PATH_RACE,
            ) from exc
        if (current.st_dev, current.st_ino) != self._root_identity:
            _unsafe_path(CacheReason.PATH_RACE)

    def assert_root_identity(self) -> None:
        """Fail closed when the configured root pathname no longer names this root."""
        self._assert_root_identity()

    def _run_hook(self, operation: str, locator: Path) -> None:
        """Run the deterministic test seam without exposing it to callers."""
        if self.before_operation is not None:
            self.before_operation(operation, locator)

    def _run_control_durability_hook(self, step: str, locator: Path) -> None:
        """Expose process-loss seams only after a concrete control transition."""
        if self.after_control_durability_step is not None:
            self.after_control_durability_step(step, locator)

    def _prepare_locator(
        self,
        locator: Union[str, Path],
        *,
        operation: str,
        allow_missing_leaf: bool = False,
    ) -> Path:
        self._assert_root_identity()
        resolved = resolve_managed_locator(
            self.root,
            locator,
            operation=operation,
            allow_missing_leaf=allow_missing_leaf,
        )
        self._run_hook(operation, resolved)
        self._assert_root_identity()
        if not self._descriptor_mode:
            # The fallback's final check is intentionally repeated after the hook.
            resolved = resolve_managed_locator(
                self.root,
                resolved,
                operation=operation,
                allow_missing_leaf=allow_missing_leaf,
            )
        return resolved

    def blob_locator(self, blob_id: str, shard_chars: int) -> Path:
        """Return a validated sharded location for one opaque backend ID."""
        safe_id = validate_blob_id(blob_id)
        if shard_chars < 0:
            raise ValueError("shard_chars must be non-negative")
        relative = Path(safe_id)
        if shard_chars > 0 and len(safe_id) >= shard_chars:
            relative = Path(safe_id[:shard_chars]) / safe_id
        return resolve_managed_locator(
            self.root,
            relative,
            operation="write",
            allow_missing_leaf=True,
        )

    def _relative_parts(self, locator: Path) -> tuple[str, ...]:
        try:
            parts = locator.relative_to(self.root).parts
        except ValueError:
            _unsafe_path(CacheReason.PATH_OUTSIDE_ROOT)
        if not parts:
            _unsafe_path(CacheReason.PATH_OUTSIDE_ROOT)
        return parts

    @contextmanager
    def _descriptor_parent(
        self, parts: tuple[str, ...], *, create: bool
    ) -> Iterator[tuple[int, str]]:
        """Walk a parent directory from the anchored descriptor without links."""
        if self._root_fd is None:
            raise RuntimeError("Descriptor mode requires an anchored storage root")

        parent_fd = os.dup(self._root_fd)
        try:
            for component in parts[:-1]:
                try:
                    child_fd = os.open(
                        component,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=parent_fd,
                    )
                except FileNotFoundError:
                    if not create:
                        raise
                    try:
                        os.mkdir(component, dir_fd=parent_fd)
                    except FileExistsError:
                        pass
                    child_fd = os.open(
                        component,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=parent_fd,
                    )
                except OSError as exc:
                    self._raise_descriptor_path_error(exc)

                try:
                    if not stat.S_ISDIR(os.fstat(child_fd).st_mode):
                        _unsafe_path(CacheReason.PATH_RACE)
                except Exception:
                    os.close(child_fd)
                    raise
                os.close(parent_fd)
                parent_fd = child_fd
            yield parent_fd, parts[-1]
        finally:
            os.close(parent_fd)

    @staticmethod
    def _raise_descriptor_path_error(exc: OSError) -> NoReturn:
        """Translate no-follow traversal failures to the public typed contract."""
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise CacheUnsafePathError(
                "Managed storage path changed during operation",
                reason=CacheReason.PATH_RACE,
            ) from exc
        raise exc

    def _descriptor_open_read(self, locator: Path) -> int:
        """Open one existing managed regular file without ever blocking on a special node."""
        parts = self._relative_parts(locator)
        try:
            with self._descriptor_parent(parts, create=False) as (parent_fd, name):
                try:
                    initial_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                    self._assert_regular_single_link(initial_stat)
                    descriptor = os.open(
                        name,
                        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0),
                        dir_fd=parent_fd,
                    )
                except FileNotFoundError:
                    raise
                except OSError as exc:
                    self._raise_descriptor_path_error(exc)
                try:
                    opened_stat = os.fstat(descriptor)
                    self._assert_regular_single_link(opened_stat)
                    if (opened_stat.st_dev, opened_stat.st_ino) != (
                        initial_stat.st_dev,
                        initial_stat.st_ino,
                    ):
                        _unsafe_path(CacheReason.PATH_RACE)
                    return descriptor
                except BaseException:
                    os.close(descriptor)
                    raise
        except FileNotFoundError:
            raise FileNotFoundError(f"Blob not found: {locator}") from None

    @staticmethod
    def _assert_regular_single_link(file_stat: os.stat_result) -> None:
        """Reject every non-file and linked object at the managed I/O boundary."""
        if (
            _is_link_or_reparse(file_stat)
            or not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_nlink != 1
        ):
            _unsafe_path(CacheReason.PATH_RACE)

    def _descriptor_stat(self, locator: Path) -> os.stat_result | None:
        parts = self._relative_parts(locator)
        try:
            with self._descriptor_parent(parts, create=False) as (parent_fd, name):
                try:
                    file_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    return None
                self._assert_regular_single_link(file_stat)
                return file_stat
        except FileNotFoundError:
            return None

    def _descriptor_delete(self, locator: Path) -> bool:
        parts = self._relative_parts(locator)
        try:
            with self._descriptor_parent(parts, create=False) as (parent_fd, name):
                try:
                    file_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    return False
                self._assert_regular_single_link(file_stat)
                self._assert_root_identity()
                os.unlink(name, dir_fd=parent_fd)
                return True
        except FileNotFoundError:
            return False

    def _write_descriptor(self, locator: Path, chunks: Iterable[bytes]) -> Path:
        parts = self._relative_parts(locator)
        with self._descriptor_parent(parts, create=True) as (parent_fd, name):
            temporary_name = f".{name}.{uuid.uuid4().hex}.tmp"
            temporary_fd: int | None = None
            try:
                temporary_fd = os.open(
                    temporary_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=parent_fd,
                )
                for chunk in chunks:
                    self._write_all(temporary_fd, chunk)
                os.fsync(temporary_fd)
                os.close(temporary_fd)
                temporary_fd = None

                try:
                    file_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    file_stat = None
                if file_stat is not None and _is_link_or_reparse(file_stat):
                    _unsafe_path(CacheReason.PATH_RACE)
                self._assert_root_identity()
                os.rename(
                    temporary_name,
                    name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
            except Exception:
                if temporary_fd is not None:
                    os.close(temporary_fd)
                try:
                    os.unlink(temporary_name, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
                raise
        return locator

    @staticmethod
    def _write_all(file_descriptor: int, chunk: bytes) -> None:
        """Write one chunk completely when the OS performs a short write."""
        view = memoryview(chunk)
        while view:
            written = os.write(file_descriptor, view)
            view = view[written:]

    @staticmethod
    def _read_bounded_descriptor(file_descriptor: int, maximum: int) -> bytes:
        """Read through a descriptor despite short reads without over-allocating."""
        chunks = bytearray()
        while len(chunks) < maximum:
            chunk = os.read(file_descriptor, maximum - len(chunks))
            if not chunk:
                break
            chunks.extend(chunk)
        return bytes(chunks)

    def _ensure_fallback_parent(self, locator: Path) -> None:
        """Create missing parents only after every existing component is verified."""
        current = self.root
        for component in self._relative_parts(locator)[:-1]:
            candidate = current / component
            try:
                file_stat = os.lstat(candidate)
            except FileNotFoundError:
                self._assert_root_identity()
                candidate.mkdir()
                file_stat = os.lstat(candidate)
            if _is_link_or_reparse(file_stat) or not stat.S_ISDIR(file_stat.st_mode):
                _unsafe_path(CacheReason.PATH_RACE)
            current = candidate

    def _write_fallback(self, locator: Path, chunks: Iterable[bytes]) -> Path:
        self._ensure_fallback_parent(locator)
        locator = resolve_managed_locator(
            self.root, locator, operation="write", allow_missing_leaf=True
        )
        temporary = locator.parent / f".{locator.name}.{uuid.uuid4().hex}.tmp"
        try:
            with open(temporary, "xb") as file_handle:
                for chunk in chunks:
                    file_handle.write(chunk)
                file_handle.flush()
                os.fsync(file_handle.fileno())
            locator = resolve_managed_locator(
                self.root, locator, operation="publish", allow_missing_leaf=True
            )
            os.replace(temporary, locator)
        except Exception:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise
        return locator

    def _publish_no_replace_descriptor(
        self, parent_fd: int, temporary_name: str, name: str
    ) -> None:
        """Install a temporary through one consuming no-replace move.

        The caller owns the directory acknowledgement so fault-injection seams
        correspond to the real temporary-fsync, rename, and directory-fsync
        boundaries rather than aliases after the operation is already durable.
        """
        _atomic_rename_no_replace(parent_fd, temporary_name, name)

    def _create_bytes_durable_exclusive_descriptor(
        self, locator: Path, data: bytes
    ) -> Path:
        """Crash-atomically create control bytes through a bound candidate."""
        parts = self._relative_parts(locator)
        with self._descriptor_parent(parts, create=True) as (parent_fd, name):
            digest = hashlib.sha256(data).hexdigest()
            # This exact locator is deterministically bound to the control
            # bytes, so recovery can validate and promote an interrupted
            # pre-install candidate without relying on a loose filename glob.
            temporary_name = f".{name}.pending.{digest}.{uuid.uuid4().hex}.tmp"
            temporary_fd: int | None = None
            try:
                temporary_fd = os.open(
                    temporary_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=parent_fd,
                )
                self._write_all(temporary_fd, data)
                os.fsync(temporary_fd)
                os.close(temporary_fd)
                temporary_fd = None
                self._run_control_durability_hook("control_temp_fsynced", locator)
                self._assert_root_identity()
                self._publish_no_replace_descriptor(parent_fd, temporary_name, name)
                self._run_control_durability_hook("control_rename_completed", locator)
                # The successful move has consumed the temporary name. A
                # process loss can therefore leave either the named pending
                # candidate or the complete final record, never two links.
                os.fsync(parent_fd)
                self._run_control_durability_hook("control_directory_fsynced", locator)
            except BaseException:
                if temporary_fd is not None:
                    os.close(temporary_fd)
                try:
                    os.unlink(temporary_name, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
                raise
        return locator

    def _create_bytes_direct_exclusive_descriptor(
        self, locator: Path, data: bytes
    ) -> Path:
        """Install bounded fixed control bytes directly at their final name.

        Fixed lock files are deliberately non-authoritative: an interrupted
        write can leave a partial regular file, but no pending candidate whose
        provenance a later opener would have to guess. Lock ownership comes
        from the descriptor/inode, never from these bytes.
        """
        parts = self._relative_parts(locator)
        with self._descriptor_parent(parts, create=True) as (parent_fd, name):
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent_fd,
            )
            try:
                self._write_all(descriptor, data)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self._run_control_durability_hook("lock_file_bytes_fsynced", locator)
            os.fsync(parent_fd)
            self._run_control_durability_hook("lock_file_directory_fsynced", locator)
        return locator

    def _create_bytes_direct_exclusive_fallback(self, locator: Path, data: bytes) -> Path:
        """Create a fixed final control name on the portable/Win32 path."""
        self._ensure_fallback_parent(locator)
        locator = resolve_managed_locator(
            self.root, locator, operation="fixed_control_create", allow_missing_leaf=True
        )
        with open(locator, "xb") as file_handle:
            file_handle.write(data)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        self._run_control_durability_hook("lock_file_bytes_fsynced", locator)
        self._fsync_containing_directory(locator)
        self._run_control_durability_hook("lock_file_directory_fsynced", locator)
        return locator

    def ensure_fixed_lock_file(self, locator: Union[str, Path]) -> None:
        """Ensure a lock inode exists without producing pending control residue."""
        if self._descriptor_mode:
            prepared = self._prepare_locator(
                locator, operation="fixed_lock_create", allow_missing_leaf=True
            )
            try:
                self._create_bytes_direct_exclusive_descriptor(prepared, b"lock\n")
            except FileExistsError:
                pass
            return
        with self._lock:
            prepared = self._prepare_locator(
                locator, operation="fixed_lock_create", allow_missing_leaf=True
            )
            try:
                self._create_bytes_direct_exclusive_fallback(prepared, b"lock\n")
            except FileExistsError:
                pass

    def _lock_authority_bytes(
        self, locator: Path, identity: tuple[int, int]
    ) -> bytes:
        """Bind one accepted lock inode to the managed-root object identity."""
        relative = "/".join(self._relative_parts(locator)).encode("utf-8")
        return (
            _LOCK_AUTHORITY_VERSION
            + self._root_identity[0].to_bytes(8, "big", signed=False)
            + self._root_identity[1].to_bytes(8, "big", signed=False)
            + hashlib.sha256(relative).digest()
            + identity[0].to_bytes(8, "big", signed=False)
            + identity[1].to_bytes(8, "big", signed=False)
        )

    def _root_authority_name(self, locator: Path) -> str:
        """Return a root-object attribute name for one logical lock locator.

        The name deliberately contains only a digest.  The authoritative
        content includes the root and lock identities; attaching it to the
        anchored root object prevents a child-control pathname swap from
        creating a new authority domain.
        """
        relative = "/".join(self._relative_parts(locator)).encode("utf-8")
        digest = hashlib.sha256(
            self._root_identity[0].to_bytes(8, "big", signed=False)
            + self._root_identity[1].to_bytes(8, "big", signed=False)
            + relative
        ).hexdigest()
        prefix = (
            _DARWIN_ROOT_AUTHORITY_XATTR_PREFIX
            if sys.platform == "darwin"
            else _ROOT_AUTHORITY_XATTR_PREFIX
        )
        return f"{prefix}{digest}"

    def _root_authority_target(self) -> int | Path:
        """Address the anchored root inode rather than a replaceable child name."""
        if self._root_fd is not None:
            return self._root_fd
        return self.root

    @staticmethod
    def _is_xattr_unsupported(exc: OSError) -> bool:
        """Classify platforms/filesystems that cannot preserve root bindings."""
        unsupported = {errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS}
        return exc.errno in unsupported

    def _set_root_authority_xattr(self, name: str, authority: bytes) -> None:
        """Create one root-directory attribute without a mutable child path."""
        if hasattr(os, "setxattr"):
            os.setxattr(
                self._root_authority_target(), name, authority, os.XATTR_CREATE
            )
            return
        if sys.platform != "darwin":
            raise AttributeError("root-object extended attributes are unavailable")
        root_fd = self._root_fd
        close_root_fd = False
        if root_fd is None:
            root_fd = os.open(
                self.root,
                os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
            )
            close_root_fd = True
        libc = ctypes.CDLL(None, use_errno=True)
        setter = libc.fsetxattr
        setter.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_int,
        )
        setter.restype = ctypes.c_int
        buffer = (ctypes.c_ubyte * len(authority)).from_buffer_copy(authority)
        try:
            if setter(
                root_fd,
                os.fsencode(name),
                ctypes.cast(buffer, ctypes.c_void_p),
                len(authority),
                0,
                0x0002,  # Darwin XATTR_CREATE (0x0001 is XATTR_NOFOLLOW)
            ) != 0:
                error_number = ctypes.get_errno()
                raise OSError(error_number, os.strerror(error_number), name)
        finally:
            if close_root_fd:
                os.close(root_fd)

    def _get_root_authority_xattr(self, name: str) -> bytes:
        """Read one bounded root-directory attribute through its descriptor."""
        if hasattr(os, "getxattr"):
            return os.getxattr(self._root_authority_target(), name)
        if sys.platform != "darwin":
            raise AttributeError("root-object extended attributes are unavailable")
        root_fd = self._root_fd
        close_root_fd = False
        if root_fd is None:
            root_fd = os.open(
                self.root,
                os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
            )
            close_root_fd = True
        libc = ctypes.CDLL(None, use_errno=True)
        getter = libc.fgetxattr
        getter.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_int,
        )
        getter.restype = ctypes.c_ssize_t
        encoded_name = os.fsencode(name)
        try:
            size = getter(root_fd, encoded_name, None, 0, 0, 0)
            if size < 0:
                error_number = ctypes.get_errno()
                raise OSError(error_number, os.strerror(error_number), name)
            if size > 512:
                _unsafe_path(CacheReason.PATH_RACE)
            buffer = (ctypes.c_ubyte * size)()
            observed = getter(
                root_fd,
                encoded_name,
                ctypes.cast(buffer, ctypes.c_void_p),
                size,
                0,
                0,
            )
            if observed < 0:
                error_number = ctypes.get_errno()
                raise OSError(error_number, os.strerror(error_number), name)
            return bytes(buffer[:observed])
        finally:
            if close_root_fd:
                os.close(root_fd)

    def _sync_root_authority_binding(self) -> None:
        """Acknowledge root-object authority publication before it is usable."""
        if self._root_fd is not None:
            os.fsync(self._root_fd)

    def _ensure_root_authority_binding(self, locator: Path, authority: bytes) -> None:
        """Create or verify one exact immutable root-object lock binding.

        Descriptor-capable POSIX systems use an extended attribute on the
        already-open root directory.  The kernel publishes an xattr as one
        complete value, so a process loss leaves either no binding or the full
        binding—never a direct-written partial sidecar.  Windows uses an HKCU
        registry value guarded by an abandon-safe named mutex for the same
        create-or-compare contract.  Neither representation is a mutable path
        below the storage root.
        """
        name = self._root_authority_name(locator)
        # Xattr/registry publication is one kernel operation.  The pre/post
        # seams model process loss on either side of that indivisible boundary;
        # there is intentionally no direct final-name write that could expose
        # a partial authoritative byte sequence in between.
        self._run_control_durability_hook("lock_authority_before_publish", locator)
        if _platform_name() == "nt":
            _windows_registry_authority_api().ensure(name, authority)
            self._run_control_durability_hook("lock_authority_published", locator)
            return

        try:
            self._set_root_authority_xattr(name, authority)
        except AttributeError as exc:  # pragma: no cover - Python platform capability.
            raise CacheBlobBackendError(
                "BlobStore lifecycle authority requires root-object attributes",
                context={"operation": "lifecycle_lock_authority"},
            ) from exc
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                if self._is_xattr_unsupported(exc):
                    raise CacheBlobBackendError(
                        "BlobStore lifecycle authority requires root-object attributes",
                        context={"operation": "lifecycle_lock_authority"},
                    ) from exc
                raise
            try:
                persisted = self._get_root_authority_xattr(name)
            except OSError as read_exc:
                if self._is_xattr_unsupported(read_exc):
                    raise CacheBlobBackendError(
                        "BlobStore lifecycle authority requires root-object attributes",
                        context={"operation": "lifecycle_lock_authority"},
                    ) from read_exc
                raise
            if persisted != authority:
                _unsafe_path(CacheReason.PATH_RACE)
        self._sync_root_authority_binding()
        self._assert_root_identity()
        self._run_control_durability_hook("lock_authority_published", locator)

    def _assert_root_authority_binding(self, locator: Path, authority: bytes) -> None:
        """Reject a lock inode that no longer matches the root-bound authority."""
        name = self._root_authority_name(locator)
        if _platform_name() == "nt":
            _windows_registry_authority_api().ensure(name, authority)
            return
        try:
            persisted = self._get_root_authority_xattr(name)
        except AttributeError as exc:  # pragma: no cover - Python platform capability.
            raise CacheBlobBackendError(
                "BlobStore lifecycle authority requires root-object attributes",
                context={"operation": "lifecycle_lock_authority"},
            ) from exc
        except OSError as exc:
            if self._is_xattr_unsupported(exc):
                raise CacheBlobBackendError(
                    "BlobStore lifecycle authority requires root-object attributes",
                    context={"operation": "lifecycle_lock_authority"},
                ) from exc
            _unsafe_path(CacheReason.PATH_RACE)
        if persisted != authority:
            _unsafe_path(CacheReason.PATH_RACE)
        self._assert_root_identity()

    def ensure_lifecycle_lock(self, locator: Union[str, Path]) -> tuple[int, int]:
        """Create and persist one root-bound lock identity before it is usable.

        The accepted identity is attached to the root object itself before the
        lock is usable. Replacing any child lock/control pathname therefore
        cannot form a second authority partition while an earlier descriptor
        is active.
        """
        prepared = self._prepare_locator(
            locator, operation="lifecycle_lock_create", allow_missing_leaf=True
        )
        self.ensure_fixed_lock_file(prepared)
        identity = self.file_identity(prepared)
        authority = self._lock_authority_bytes(prepared, identity)
        self._ensure_root_authority_binding(prepared, authority)
        return identity

    def _create_bytes_durable_exclusive_fallback(
        self, locator: Path, data: bytes
    ) -> Path:
        """Use Win32's consuming no-replace move on the supported fallback path."""
        if _platform_name() == "nt":
            self._ensure_fallback_parent(locator)
            locator = resolve_managed_locator(
                self.root, locator, operation="exclusive_create", allow_missing_leaf=True
            )
            temporary = locator.parent / f".{locator.name}.pending.{hashlib.sha256(data).hexdigest()}.{uuid.uuid4().hex}.tmp"
            try:
                with open(temporary, "xb") as file_handle:
                    file_handle.write(data)
                    file_handle.flush()
                    os.fsync(file_handle.fileno())
                self._run_control_durability_hook("control_temp_fsynced", locator)
                self._assert_root_identity()
                _windows_file_api().rename_no_replace(temporary, locator)
                self._run_control_durability_hook("control_rename_completed", locator)
                _windows_file_api().flush_directory(locator.parent)
                self._run_control_durability_hook("control_directory_fsynced", locator)
                return locator
            except BaseException:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
                raise
        # A portable fallback cannot atomically consume a candidate without a
        # directory descriptor.  Check-then-rename and hard-link install are
        # intentionally not acceptable for authenticated control evidence.
        raise CacheBlobBackendError(
            "BlobStore durable lifecycle control records require descriptor-backed "
            "atomic no-replace publication",
            context={"operation": "exclusive_create"},
        )

    def _fallback_open_read(self, locator: Path) -> BinaryIO:
        """Open a verified regular fallback file without following a special node."""
        locator = resolve_managed_locator(self.root, locator, operation="stream")
        try:
            initial_stat = os.lstat(locator)
        except FileNotFoundError:
            raise FileNotFoundError(f"Blob not found: {locator}") from None
        self._assert_regular_single_link(initial_stat)
        flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(locator, flags)
        except FileNotFoundError:
            raise FileNotFoundError(f"Blob not found: {locator}") from None
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENXIO, errno.ENODEV, errno.EISDIR}:
                _unsafe_path(CacheReason.PATH_RACE)
            raise
        try:
            opened_stat = os.fstat(descriptor)
            self._assert_regular_single_link(opened_stat)
            if (opened_stat.st_dev, opened_stat.st_ino) != (
                initial_stat.st_dev,
                initial_stat.st_ino,
            ):
                _unsafe_path(CacheReason.PATH_RACE)
            return os.fdopen(descriptor, "rb")
        except BaseException:
            os.close(descriptor)
            raise

    def write_bytes(self, blob_id: str, data: bytes, *, shard_chars: int) -> Path:
        """Atomically write bytes using a validated opaque ID."""
        locator = self.blob_locator(blob_id, shard_chars)
        if self._descriptor_mode:
            locator = self._prepare_locator(
                locator, operation="write", allow_missing_leaf=True
            )
            return self._write_descriptor(locator, (data,))
        with self._lock:
            locator = self._prepare_locator(
                locator, operation="write", allow_missing_leaf=True
            )
            return self._write_fallback(locator, (data,))

    def write_stream(self, blob_id: str, stream: BinaryIO, *, shard_chars: int) -> Path:
        """Atomically write a stream using a validated opaque ID."""
        locator = self.blob_locator(blob_id, shard_chars)

        return self.write_stream_to_locator(locator, stream)

    def write_stream_to_locator(
        self, locator: Union[str, Path], stream: BinaryIO
    ) -> Path:
        """Atomically replace one already-contained locator from a stream.

        High-level reconciliation paths retain a previously validated locator,
        rather than a backend identifier, while restoring a managed payload.
        This method keeps that write inside the same no-follow boundary as
        ordinary blob-ID writes.
        """

        def chunks() -> Iterator[bytes]:
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    return
                yield chunk

        if self._descriptor_mode:
            locator = self._prepare_locator(
                locator, operation="write_stream_to_locator", allow_missing_leaf=True
            )
            return self._write_descriptor(locator, chunks())
        with self._lock:
            locator = self._prepare_locator(
                locator, operation="write_stream_to_locator", allow_missing_leaf=True
            )
            return self._write_fallback(locator, chunks())

    def _fsync_containing_directory(self, locator: Union[str, Path]) -> None:
        """Durably acknowledge a contained locator's parent-directory update."""
        if self._descriptor_mode:
            prepared = self._prepare_locator(
                locator,
                operation="directory_fsync",
                allow_missing_leaf=True,
            )
            parts = self._relative_parts(prepared)
            with self._descriptor_parent(parts, create=False) as (parent_fd, _):
                os.fsync(parent_fd)
            return

        with self._lock:
            prepared = self._prepare_locator(
                locator,
                operation="directory_fsync",
                allow_missing_leaf=True,
            )
            if _platform_name() == "nt":
                _windows_file_api().flush_directory(prepared.parent)
                return
            directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            directory_fd = os.open(prepared.parent, directory_flags)
            try:
                self._assert_root_identity()
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)

    def write_bytes_durable(self, locator: Union[str, Path], data: bytes) -> Path:
        """Atomically publish bytes and acknowledge the containing directory."""
        written = self.write_stream_to_locator(locator, BytesIO(data))
        self._fsync_containing_directory(written)
        return written

    def create_bytes_durable_exclusive(
        self, locator: Union[str, Path], data: bytes
    ) -> Path:
        """Create one contained durable locator without replacing existing evidence."""
        if self._descriptor_mode:
            prepared = self._prepare_locator(
                locator,
                operation="exclusive_create",
                allow_missing_leaf=True,
            )
            return self._create_bytes_durable_exclusive_descriptor(prepared, data)

        with self._lock:
            prepared = self._prepare_locator(
                locator,
                operation="exclusive_create",
                allow_missing_leaf=True,
            )
            return self._create_bytes_durable_exclusive_fallback(prepared, data)

    def promote_durable_pending_control(
        self,
        locator: Union[str, Path],
        data: bytes,
        *,
        pending_name: str | None = None,
    ) -> bool:
        """Promote or retire one exact interrupted control candidate.

        ``create_bytes_durable_exclusive`` uses a digest-bound pending name so
        a process loss before native no-replace installation leaves recoverable
        evidence rather than an anonymous temporary.  This helper never uses a
        pattern as deletion authority: the candidate bytes must match the
        digest encoded in its exact locator, and an existing final must match
        byte-for-byte before its pending counterpart is retired.
        """
        if not isinstance(data, bytes) or not data:
            raise ValueError("pending control evidence must be non-empty bytes")
        if not self._descriptor_mode:
            if _platform_name() != "nt":
                raise CacheBlobBackendError(
                    "BlobStore durable control recovery requires descriptor-backed publication",
                    context={"operation": "promote_pending_control"},
                )
            return self._promote_durable_pending_control_fallback(
                locator, data, pending_name=pending_name
            )
        prepared = self._prepare_locator(
            locator, operation="promote_pending_control", allow_missing_leaf=True
        )
        parts = self._relative_parts(prepared)
        digest = hashlib.sha256(data).hexdigest()
        if pending_name is None:
            temporary_name = f".{parts[-1]}.pending.{digest}.tmp"
        else:
            expected_prefix = f".{parts[-1]}.pending.{digest}."
            token = pending_name.removeprefix(expected_prefix).removesuffix(".tmp")
            if (
                not pending_name.startswith(expected_prefix)
                or len(token) != 32
                or any(character not in "0123456789abcdef" for character in token)
            ):
                _unsafe_path(CacheReason.PATH_RACE)
            temporary_name = pending_name
        with self._descriptor_parent(parts, create=False) as (parent_fd, name):
            try:
                temporary_fd = os.open(
                    temporary_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd
                )
            except FileNotFoundError:
                return False
            try:
                temporary_stat = os.fstat(temporary_fd)
                if not stat.S_ISREG(temporary_stat.st_mode) or temporary_stat.st_nlink != 1:
                    _unsafe_path(CacheReason.PATH_RACE)
                temporary_bytes = self._read_bounded_descriptor(
                    temporary_fd, len(data) + 1
                )
                if temporary_bytes != data:
                    _unsafe_path(CacheReason.PATH_RACE)
            finally:
                os.close(temporary_fd)
            try:
                final_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
            except FileNotFoundError:
                self._assert_root_identity()
                try:
                    _atomic_rename_no_replace(parent_fd, temporary_name, name)
                except FileExistsError:
                    return self.promote_durable_pending_control(
                        prepared, data, pending_name=temporary_name
                    )
                os.fsync(parent_fd)
                return True
            try:
                final_bytes = self._read_bounded_descriptor(final_fd, len(data) + 1)
            finally:
                os.close(final_fd)
            if final_bytes != data:
                _unsafe_path(CacheReason.PATH_RACE)
            os.unlink(temporary_name, dir_fd=parent_fd)
            os.fsync(parent_fd)
            return True

    def _pending_control_name(
        self,
        locator: Path,
        data: bytes,
        pending_name: str | None,
    ) -> str:
        """Validate the exact digest-bound name for one pending control record."""
        parts = self._relative_parts(locator)
        digest = hashlib.sha256(data).hexdigest()
        if pending_name is None:
            return f".{parts[-1]}.pending.{digest}.tmp"
        expected_prefix = f".{parts[-1]}.pending.{digest}."
        token = pending_name.removeprefix(expected_prefix).removesuffix(".tmp")
        if (
            not pending_name.startswith(expected_prefix)
            or len(token) != 32
            or any(character not in "0123456789abcdef" for character in token)
        ):
            _unsafe_path(CacheReason.PATH_RACE)
        return pending_name

    def _promote_durable_pending_control_fallback(
        self,
        locator: Union[str, Path],
        data: bytes,
        *,
        pending_name: str | None,
    ) -> bool:
        """Converge exact Win32 pending evidence after process loss.

        The Windows writer emits a digest-bound candidate before the consuming
        ``MoveFileExW`` transition.  Reopen verifies those exact bytes before
        either promoting the candidate or retiring it beside an identical final
        record; it never uses a glob as deletion authority.
        """
        with self._lock:
            prepared = self._prepare_locator(
                locator,
                operation="promote_pending_control",
                allow_missing_leaf=True,
            )
            temporary_name = self._pending_control_name(prepared, data, pending_name)
            pending = prepared.parent / temporary_name
            try:
                pending_bytes = self.read_bytes_bounded(pending, max_bytes=len(data))
            except FileNotFoundError:
                return False
            if pending_bytes != data:
                _unsafe_path(CacheReason.PATH_RACE)
            try:
                _windows_file_api().rename_no_replace(pending, prepared)
            except FileExistsError:
                final_bytes = self.read_bytes_bounded(prepared, max_bytes=len(data))
                if final_bytes != data:
                    _unsafe_path(CacheReason.PATH_RACE)
                try:
                    pending.unlink()
                except FileNotFoundError:
                    return False
                _windows_file_api().flush_directory(prepared.parent)
                return True
            _windows_file_api().flush_directory(prepared.parent)
            return True

    def create_stream_durable_exclusive(
        self, locator: Union[str, Path], stream: BinaryIO
    ) -> Path:
        """Create one durable contained file from ``stream`` without replacement.

        This is deliberately distinct from ``write_stream_to_locator``: an
        immutable lifecycle generation must never replace a prior candidate or
        committed payload.  A partially written file remains operation-record
        owned residue for recovery instead of becoming an untracked temporary.
        """

        def write_stream(descriptor: int) -> None:
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    return
                self._write_all(descriptor, chunk)

        if self._descriptor_mode:
            prepared = self._prepare_locator(
                locator,
                operation="exclusive_stream_create",
                allow_missing_leaf=True,
            )
            parts = self._relative_parts(prepared)
            with self._descriptor_parent(parts, create=True) as (parent_fd, name):
                descriptor = os.open(
                    name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=parent_fd,
                )
                try:
                    write_stream(descriptor)
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                os.fsync(parent_fd)
            return prepared

        with self._lock:
            prepared = self._prepare_locator(
                locator,
                operation="exclusive_stream_create",
                allow_missing_leaf=True,
            )
            self._ensure_fallback_parent(prepared)
            prepared = resolve_managed_locator(
                self.root,
                prepared,
                operation="exclusive_stream_create",
                allow_missing_leaf=True,
            )
            with open(prepared, "xb") as destination:
                while True:
                    chunk = stream.read(8192)
                    if not chunk:
                        break
                    destination.write(chunk)
                destination.flush()
                os.fsync(destination.fileno())
            self._fsync_containing_directory(prepared)
            return prepared

    def delete_durable(self, locator: Union[str, Path]) -> bool:
        """Delete a contained locator and acknowledge the directory when it existed."""
        deleted = self.delete(locator)
        if deleted:
            self._fsync_containing_directory(locator)
        return deleted

    def read_bytes(self, locator: Union[str, Path]) -> bytes:
        """Read a contained locator or raise a typed unsafe-path error."""
        if self._descriptor_mode:
            prepared = self._prepare_locator(locator, operation="read")
            file_descriptor = self._descriptor_open_read(prepared)
            with os.fdopen(file_descriptor, "rb") as file_handle:
                return file_handle.read()
        with self._lock:
            prepared = self._prepare_locator(locator, operation="read")
            with self._fallback_open_read(prepared) as file_handle:
                return file_handle.read()

    def read_bytes_bounded(self, locator: Union[str, Path], *, max_bytes: int) -> bytes:
        """Read one contained file without allocating more than ``max_bytes + 1``.

        Control evidence is attacker-influenced at the filesystem boundary.  A
        size check alone is not sufficient because a concurrent writer could
        enlarge the file after ``stat`` and before a conventional ``read()``.
        The descriptor-backed stream read therefore caps the allocation as
        well as rejecting an already-oversized descriptor before parsing.
        """
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        source = self.open_read(locator)
        try:
            if os.fstat(source.fileno()).st_size > max_bytes:
                raise ValueError("managed file exceeds the byte limit")
            data = source.read(max_bytes + 1)
        finally:
            source.close()
        if len(data) > max_bytes:
            raise ValueError("managed file exceeds the byte limit")
        return data

    def open_read(self, locator: Union[str, Path]) -> BinaryIO:
        """Open a contained locator for streaming reads."""
        if self._descriptor_mode:
            prepared = self._prepare_locator(locator, operation="read_stream")
            return os.fdopen(self._descriptor_open_read(prepared), "rb")
        with self._lock:
            prepared = self._prepare_locator(locator, operation="read_stream")
            return self._fallback_open_read(prepared)

    def copy_to_stream(
        self,
        locator: Union[str, Path],
        destination: BinaryIO,
        *,
        chunk_size: int = 8192,
    ) -> int:
        """Copy one no-follow managed read into an already-open private stream.

        This is intentionally the only high-level snapshot primitive: it opens
        the managed file once through ``open_read`` and never exposes that
        descriptor or locator to callers that deserialize payloads.
        """
        copied = 0
        with self.open_read(locator) as source:
            while chunk := source.read(chunk_size):
                destination.write(chunk)
                copied += len(chunk)
        return copied

    def delete(self, locator: Union[str, Path]) -> bool:
        """Delete a contained locator; a missing safe leaf remains a normal miss."""
        if self._descriptor_mode:
            prepared = self._prepare_locator(locator, operation="delete")
            return self._descriptor_delete(prepared)
        with self._lock:
            prepared = self._prepare_locator(locator, operation="delete")
            try:
                prepared = resolve_managed_locator(self.root, prepared, operation="delete")
                prepared.unlink()
                return True
            except FileNotFoundError:
                return False

    def exists(self, locator: Union[str, Path]) -> bool:
        """Check a contained locator while keeping unsafe input distinct from a miss."""
        if self._descriptor_mode:
            prepared = self._prepare_locator(locator, operation="exists")
            return self._descriptor_stat(prepared) is not None
        with self._lock:
            prepared = self._prepare_locator(locator, operation="exists")
            prepared = resolve_managed_locator(self.root, prepared, operation="exists")
            try:
                file_stat = os.lstat(prepared)
            except FileNotFoundError:
                return False
            self._assert_regular_single_link(file_stat)
            return True

    def file_identity(self, locator: Union[str, Path]) -> tuple[int, int]:
        """Return a contained regular file's device/inode identity.

        Authority owners retain this identity alongside a managed descriptor so
        a name replacement cannot silently move a process onto a different
        lock inode.
        """
        if self._descriptor_mode:
            prepared = self._prepare_locator(locator, operation="file_identity")
            file_stat = self._descriptor_stat(prepared)
        else:
            with self._lock:
                prepared = self._prepare_locator(locator, operation="file_identity")
                try:
                    file_stat = os.lstat(prepared)
                except FileNotFoundError:
                    file_stat = None
                if file_stat is not None and _is_link_or_reparse(file_stat):
                    _unsafe_path(CacheReason.PATH_RACE)
        if file_stat is None:
            raise FileNotFoundError(f"Blob not found: {locator}")
        self._assert_regular_single_link(file_stat)
        return file_stat.st_dev, file_stat.st_ino

    def assert_file_identity(
        self, locator: Union[str, Path], expected: tuple[int, int]
    ) -> None:
        """Prove a managed authority filename still names one verified inode."""
        if self.file_identity(locator) != expected:
            _unsafe_path(CacheReason.PATH_RACE)

    def retain_lock_identity(self, locator: Union[str, Path]) -> tuple[int, int]:
        """Bind one managed lock name to the first verified inode it may use.

        Lock names are control-plane authority.  Once a store has accepted one,
        accepting a replacement inode would partition later participants from
        work already in flight.  The bounded callers use fixed lock stripes;
        entries are released with the managed root descriptor at close.
        """
        prepared = self._prepare_locator(locator, operation="retain_lock_identity")
        parts = self._relative_parts(prepared)
        identity = self.file_identity(prepared)
        authority = self._lock_authority_bytes(prepared, identity)
        self._assert_root_authority_binding(prepared, authority)
        with self._lock:
            expected = self._lock_identities.setdefault(parts, identity)
        if expected != identity:
            _unsafe_path(CacheReason.PATH_RACE)
        return expected

    def assert_retained_lock_identity(
        self, locator: Union[str, Path], expected: tuple[int, int]
    ) -> None:
        """Verify both the root and this store's retained lock authority."""
        self.assert_root_identity()
        current = self.retain_lock_identity(locator)
        if current != expected:
            _unsafe_path(CacheReason.PATH_RACE)

    def open_verified_regular_file(self, locator: Union[str, Path]) -> BinaryIO:
        """Open one single-linked regular authority file and retain its descriptor.

        The returned descriptor is intentionally owned by the caller.  Its
        identity is checked against the managed name before the caller can use
        it as an advisory authority boundary, so a symlink, hard link, or name
        replacement cannot silently split lock participants.
        """
        handle = self.open_read(locator)
        try:
            file_stat = os.fstat(handle.fileno())
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
                _unsafe_path(CacheReason.PATH_RACE)
            identity = (file_stat.st_dev, file_stat.st_ino)
            self.assert_file_identity(locator, identity)
            return handle
        except BaseException:
            handle.close()
            raise

    def sha256_and_size(
        self,
        locator: Union[str, Path],
        *,
        expected_identity: tuple[int, int] | None = None,
    ) -> tuple[str, int]:
        """Hash one managed generation through a no-follow descriptor snapshot.

        The descriptor identity is re-bound to the locator after hashing. A
        candidate substituted between publication and compare-and-swap cannot
        donate bytes from another inode or leave a manifest pointing at a
        different pathname.
        """
        if expected_identity is not None and self.file_identity(locator) != expected_identity:
            _unsafe_path(CacheReason.PATH_RACE)
        with self.open_read(locator) as source:
            file_stat = os.fstat(source.fileno())
            self._assert_regular_single_link(file_stat)
            identity = (file_stat.st_dev, file_stat.st_ino)
            if expected_identity is not None and identity != expected_identity:
                _unsafe_path(CacheReason.PATH_RACE)
            digest = hashlib.sha256()
            byte_size = 0
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                byte_size += len(chunk)
        self.assert_file_identity(locator, expected_identity or identity)
        return digest.hexdigest(), byte_size

    def get_size(self, locator: Union[str, Path]) -> int:
        """Return the size of a contained locator, or ``-1`` for a safe miss."""
        if self._descriptor_mode:
            prepared = self._prepare_locator(locator, operation="size")
            file_stat = self._descriptor_stat(prepared)
            return -1 if file_stat is None else file_stat.st_size
        with self._lock:
            prepared = self._prepare_locator(locator, operation="size")
            prepared = resolve_managed_locator(self.root, prepared, operation="size")
            try:
                file_stat = os.lstat(prepared)
            except FileNotFoundError:
                return -1
            self._assert_regular_single_link(file_stat)
            return file_stat.st_size
