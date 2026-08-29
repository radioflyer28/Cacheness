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
import hashlib
import os
import re
import stat
import threading
import uuid
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import BinaryIO, Iterator, NoReturn, Union

from cacheness.error_handling import CacheReason, CacheUnsafePathError


_BLOB_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")
_MAX_BLOB_ID_LENGTH = 256
_PHYSICAL_NAME_DOMAIN = b"cacheness.physical-name.v1\x00"


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
            os.name == "nt"
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
        parts = self._relative_parts(locator)
        try:
            with self._descriptor_parent(parts, create=False) as (parent_fd, name):
                try:
                    return os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
                except FileNotFoundError:
                    raise
                except OSError as exc:
                    self._raise_descriptor_path_error(exc)
        except FileNotFoundError:
            raise FileNotFoundError(f"Blob not found: {locator}") from None

    def _descriptor_stat(self, locator: Path) -> os.stat_result | None:
        parts = self._relative_parts(locator)
        try:
            with self._descriptor_parent(parts, create=False) as (parent_fd, name):
                try:
                    file_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    return None
                if _is_link_or_reparse(file_stat):
                    _unsafe_path(CacheReason.PATH_RACE)
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
                if _is_link_or_reparse(file_stat):
                    _unsafe_path(CacheReason.PATH_RACE)
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

    def _fallback_read(self, locator: Path) -> bytes:
        locator = resolve_managed_locator(self.root, locator, operation="read")
        try:
            return locator.read_bytes()
        except FileNotFoundError:
            raise FileNotFoundError(f"Blob not found: {locator}") from None

    def _fallback_open_read(self, locator: Path) -> BinaryIO:
        locator = resolve_managed_locator(self.root, locator, operation="stream")
        try:
            return open(locator, "rb")
        except FileNotFoundError:
            raise FileNotFoundError(f"Blob not found: {locator}") from None

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

        def chunks() -> Iterator[bytes]:
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    return
                yield chunk

        if self._descriptor_mode:
            locator = self._prepare_locator(
                locator, operation="write_stream", allow_missing_leaf=True
            )
            return self._write_descriptor(locator, chunks())
        with self._lock:
            locator = self._prepare_locator(
                locator, operation="write_stream", allow_missing_leaf=True
            )
            return self._write_fallback(locator, chunks())

    def read_bytes(self, locator: Union[str, Path]) -> bytes:
        """Read a contained locator or raise a typed unsafe-path error."""
        if self._descriptor_mode:
            prepared = self._prepare_locator(locator, operation="read")
            file_descriptor = self._descriptor_open_read(prepared)
            with os.fdopen(file_descriptor, "rb") as file_handle:
                return file_handle.read()
        with self._lock:
            prepared = self._prepare_locator(locator, operation="read")
            return self._fallback_read(prepared)

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
            return prepared.exists()

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
                return prepared.stat().st_size
            except FileNotFoundError:
                return -1
