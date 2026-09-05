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
_WINDOWS_LOCAL_STORE_COORDINATION_SCOPE = "one_os_user_one_session"


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
    _DELETE = 0x00010000
    _FILE_SHARE_READ = 0x00000001
    _FILE_SHARE_WRITE = 0x00000002
    _FILE_SHARE_DELETE = 0x00000004
    _OPEN_EXISTING = 3
    _FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    _FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    _FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
    _FILE_DISPOSITION_INFO = 4
    _MOVEFILE_REPLACE_EXISTING = 0x00000001
    _MOVEFILE_WRITE_THROUGH = 0x00000008

    class _FileDispositionInfo(ctypes.Structure):
        _fields_ = [("DeleteFile", ctypes.c_int)]

    class _ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("FileAttributes", ctypes.c_uint32),
            ("CreationTimeLow", ctypes.c_uint32),
            ("CreationTimeHigh", ctypes.c_uint32),
            ("LastAccessTimeLow", ctypes.c_uint32),
            ("LastAccessTimeHigh", ctypes.c_uint32),
            ("LastWriteTimeLow", ctypes.c_uint32),
            ("LastWriteTimeHigh", ctypes.c_uint32),
            ("VolumeSerialNumber", ctypes.c_uint32),
            ("FileSizeHigh", ctypes.c_uint32),
            ("FileSizeLow", ctypes.c_uint32),
            ("NumberOfLinks", ctypes.c_uint32),
            ("FileIndexHigh", ctypes.c_uint32),
            ("FileIndexLow", ctypes.c_uint32),
        ]

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
        self._set_file_information = self._kernel32.SetFileInformationByHandle
        self._set_file_information.argtypes = (
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        )
        self._set_file_information.restype = ctypes.c_int
        self._get_file_information = self._kernel32.GetFileInformationByHandle
        self._get_file_information.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(self._ByHandleFileInformation),
        )
        self._get_file_information.restype = ctypes.c_int

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


    def replace_write_through(self, temporary: Path, destination: Path) -> None:
        """Replace one regular file through documented write-through semantics."""
        if not self._move_file_ex(
            str(temporary),
            str(destination),
            self._MOVEFILE_REPLACE_EXISTING | self._MOVEFILE_WRITE_THROUGH,
        ):
            self._raise_last_error("MoveFileExW")

    def delete_write_through(self, locator: Path) -> tuple[int, int, int]:
        """Immediately unlink one verified managed leaf through its handle.

        ``MoveFileExW(path, NULL, WRITE_THROUGH)`` is not an immediate-delete
        contract.  A ``FileDispositionInfo`` request on a reparse-safe handle
        is the documented acknowledgement boundary: success marks the exact
        opened object for deletion and it disappears when this final handle
        closes. Windows exposes no truthful directory-fsync promise.
        """
        handle = self._create_file(
            str(locator),
            self._DELETE | self._GENERIC_READ,
            self._FILE_SHARE_READ | self._FILE_SHARE_WRITE | self._FILE_SHARE_DELETE,
            None,
            self._OPEN_EXISTING,
            self._FILE_FLAG_OPEN_REPARSE_POINT,
            None,
        )
        invalid_handle = ctypes.c_void_p(-1).value
        if handle == invalid_handle:
            self._raise_last_error("CreateFileW")
        failure: BaseException | None = None
        try:
            information = self._ByHandleFileInformation()
            if not self._get_file_information(handle, ctypes.byref(information)):
                self._raise_last_error("GetFileInformationByHandle")
            if (
                information.FileAttributes & self._FILE_ATTRIBUTE_REPARSE_POINT
                or information.NumberOfLinks != 1
            ):
                _unsafe_path(CacheReason.PATH_RACE)
            # The exact object selected by this one retained handle is the
            # deletion authority.  Do not verify by pathname, close, then
            # unlink by pathname: a substitution in that gap could dispose of
            # a different managed leaf.  Keeping this identity locally also
            # makes the native adapter's contract explicit to callers/tests.
            identity = (
                int(information.VolumeSerialNumber),
                int(information.FileIndexHigh),
                int(information.FileIndexLow),
            )
            disposition = self._FileDispositionInfo(1)
            if not self._set_file_information(
                handle,
                self._FILE_DISPOSITION_INFO,
                ctypes.byref(disposition),
                ctypes.sizeof(disposition),
            ):
                self._raise_last_error("SetFileInformationByHandle")
        except BaseException as exc:
            failure = exc
            raise
        finally:
            if not self._close_handle(handle) and failure is None:
                self._raise_last_error("CloseHandle")
        return identity

    def flush_regular_file(
        self,
        locator: Path,
        *,
        expected_identity: tuple[int, int] | None = None,
    ) -> tuple[int, int, int]:
        """Flush one retained, reparse-safe regular-file handle.

        ``expected_identity`` is the portable ``stat`` identity captured by
        the caller.  Win32 file IDs use a different representation, so this
        adapter validates the opened handle's reparse/link invariants and the
        caller re-checks its portable identity after close.  Keeping both
        checks around *this* flush handle prevents a pathname-only durability
        acknowledgement from silently accepting a substituted key object.
        """
        handle = self._create_file(
            str(locator),
            self._GENERIC_WRITE,
            self._FILE_SHARE_READ | self._FILE_SHARE_WRITE | self._FILE_SHARE_DELETE,
            None,
            self._OPEN_EXISTING,
            self._FILE_FLAG_OPEN_REPARSE_POINT,
            None,
        )
        invalid_handle = ctypes.c_void_p(-1).value
        if handle == invalid_handle:
            self._raise_last_error("CreateFileW")
        flush_failure: BaseException | None = None
        try:
            information = self._ByHandleFileInformation()
            if not self._get_file_information(handle, ctypes.byref(information)):
                self._raise_last_error("GetFileInformationByHandle")
            if (
                information.FileAttributes & self._FILE_ATTRIBUTE_REPARSE_POINT
                or information.NumberOfLinks != 1
            ):
                _unsafe_path(CacheReason.PATH_RACE)
            if expected_identity is not None:
                current = os.lstat(locator)
                if (
                    (current.st_dev, current.st_ino) != expected_identity
                    or _is_link_or_reparse(current)
                    or not stat.S_ISREG(current.st_mode)
                    or current.st_nlink != 1
                ):
                    _unsafe_path(CacheReason.PATH_RACE)
            native_identity = self._native_identity(information)
            if not self._flush_file_buffers(handle):
                self._raise_last_error("FlushFileBuffers")
        except BaseException as exc:
            flush_failure = exc
            raise
        finally:
            if not self._close_handle(handle) and flush_failure is None:
                self._raise_last_error("CloseHandle")
        return native_identity

    @staticmethod
    def _native_identity(
        information: _ByHandleFileInformation,
    ) -> tuple[int, int, int]:
        """Return the immutable Win32 volume/file identity from one handle."""
        return (
            int(information.VolumeSerialNumber),
            int(information.FileIndexHigh),
            int(information.FileIndexLow),
        )

    def _descriptor_regular_file_information(
        self, file_descriptor: int
    ) -> tuple[ctypes.c_void_p, _ByHandleFileInformation]:
        """Return one checked native handle identity without reopening a path."""
        try:
            import msvcrt
        except ImportError as exc:  # pragma: no cover - Windows only.
            raise CacheBlobBackendError(
                "BlobStore lifecycle durability is unavailable on this Windows runtime",
                context={"operation": "GetFileInformationByHandle"},
            ) from exc
        handle = ctypes.c_void_p(msvcrt.get_osfhandle(file_descriptor))
        information = self._ByHandleFileInformation()
        if not self._get_file_information(handle, ctypes.byref(information)):
            self._raise_last_error("GetFileInformationByHandle")
        if (
            information.FileAttributes & self._FILE_ATTRIBUTE_REPARSE_POINT
            or information.NumberOfLinks != 1
        ):
            _unsafe_path(CacheReason.PATH_RACE)
        return handle, information

    def native_regular_file_identity(self, file_descriptor: int) -> tuple[int, int, int]:
        """Read the native identity of an already-open verified descriptor."""
        _handle, information = self._descriptor_regular_file_information(file_descriptor)
        return self._native_identity(information)

    def flush_regular_file_descriptor(
        self,
        file_descriptor: int,
        *,
        expected_native_identity: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        """Flush the exact retained file handle used for authority admission.

        Unlike the pathname form, this does not reopen or restat a name.  The
        native volume/file identity and the flush therefore apply to precisely
        the descriptor that the integrity provider identity-checked and locked.
        """
        handle, information = self._descriptor_regular_file_information(file_descriptor)
        native_identity = self._native_identity(information)
        if native_identity != expected_native_identity:
            _unsafe_path(CacheReason.PATH_RACE)
        if not self._flush_file_buffers(handle):
            self._raise_last_error("FlushFileBuffers")
        return native_identity



def _windows_file_api() -> _WindowsFileApi:
    """Construct the production Win32 durability adapter only when required."""
    return _WindowsFileApi()


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
        self._root_fd: int | None = None
        self._descriptor_mode = self._open_root_descriptor()
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
            if _platform_name() == "nt":
                _windows_file_api().replace_write_through(temporary, locator)
            else:
                os.replace(temporary, locator)
        except Exception:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise
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
                # Windows does not document FlushFileBuffers for directory
                # handles.  A surviving regular file can instead be flushed
                # through a documented ordinary file handle; deletions use
                # MoveFileExW's WRITE_THROUGH transition in delete_durable.
                _windows_file_api().flush_regular_file(prepared)
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
        if not self._descriptor_mode and _platform_name() == "nt":
            # Windows uses a reparse-safe DELETE handle and documented
            # FileDispositionInfo acknowledgement. The native adapter opens,
            # validates, identifies, disposes, and closes the *same* handle.
            # A prior pathname identity check would introduce a
            # verify-close-delete-by-name substitution gap.
            with self._lock:
                prepared = self._prepare_locator(locator, operation="delete_durable")
                try:
                    _windows_file_api().delete_write_through(prepared)
                except FileNotFoundError:
                    return False
                return True
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
