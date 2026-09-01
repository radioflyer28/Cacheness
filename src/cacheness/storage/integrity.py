"""Cryptographic primitives used by the canonical BlobStore manifest path."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import stat
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import BinaryIO, Protocol, runtime_checkable

from cacheness.error_handling import (
    CacheBlobManifestUnauthenticatedError,
    CacheReason,
)


HMAC_SHA256_KEY_BYTES = 32
_INITIALIZATION_GUARD = RLock()
_KEY_READY_SUFFIX = ".ready"
_MAX_READY_RECORD_BYTES = 256


class ManifestKeyError(CacheBlobManifestUnauthenticatedError, ValueError):
    """Raised when canonical manifest key material cannot be authenticated."""

    def __init__(self, message: str):
        super().__init__(
            message,
            reason=CacheReason.MANIFEST_SIGNING_KEY_INVALID,
        )


@runtime_checkable
class ManifestSigningKeyProvider(Protocol):
    """Narrow canonical signing-key source accepted by BlobStore boundaries."""

    def get_key(self) -> bytes:
        """Return exactly 32 bytes of already-authorized key material."""


@runtime_checkable
class ManifestKeyDurabilityProvider(Protocol):
    """Acknowledge first key publication before signed evidence may exist.

    An application-owned Windows keystore may implement this protocol when it
    has a stronger platform-specific namespace acknowledgement than the
    default single-user/session file provider. Implementations must verify
    ``expected_identity`` and return only after their documented persistence
    boundary succeeds.
    """

    def acknowledge_new_key(
        self,
        key_path: Path,
        expected_identity: tuple[int, int],
    ) -> None:
        """Durably acknowledge one exact newly created trust-root object."""


def sha256_and_size(path: Path) -> tuple[str, int]:
    """Return the SHA-256 digest and exact byte size of one payload file."""
    with Path(path).open("rb") as source:
        return sha256_and_size_stream(source)


def sha256_and_size_stream(source: BinaryIO) -> tuple[str, int]:
    """Hash one readable stream incrementally without loading it into memory."""
    digest = hashlib.sha256()
    byte_size = 0
    while chunk := source.read(1024 * 1024):
        digest.update(chunk)
        byte_size += len(chunk)
    return digest.hexdigest(), byte_size


def sign_hmac_sha256(payload: bytes, key: bytes) -> str:
    """Sign canonical bytes with the fixed v1 HMAC-SHA256 algorithm."""
    _validate_key(key)
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def verify_hmac_sha256(payload: bytes, signature: str, key: bytes) -> bool:
    """Verify a canonical HMAC using a constant-time comparison."""
    if not isinstance(signature, str) or not signature:
        return False
    expected = sign_hmac_sha256(payload, key)
    return hmac.compare_digest(expected, signature)


def _validate_key(key: bytes) -> None:
    if type(key) is not bytes or len(key) != HMAC_SHA256_KEY_BYTES:
        raise ManifestKeyError("Canonical manifest key must contain exactly 32 bytes")


class ManifestKeyProvider:
    """Strict file or application-supplied canonical HMAC key provider.

    Reading a key is intentionally non-mutating. A caller must explicitly invoke
    :meth:`initialize_new_store` before a fresh local store receives a generated
    key; reopens therefore cannot manufacture a replacement trust root.
    """

    def __init__(
        self,
        key_path: Path,
        key: bytes | None = None,
        *,
        durability_provider: ManifestKeyDurabilityProvider | None = None,
    ):
        self.key_path = Path(key_path)
        self._provided_key = key
        self._durability_provider = durability_provider
        if key is not None:
            _validate_key(key)

    def get_key(self) -> bytes:
        """Load already-authorized key material without writing a key file."""
        if self._provided_key is not None:
            return self._provided_key
        return self._read_existing_key(require_ready=True)

    def initialize_new_store(self) -> bytes:
        """Create or complete exactly one acknowledged local trust root.

        The key filename alone is deliberately not an authority signal.  An
        exclusive create makes the bytes visible before the platform can
        acknowledge the key's namespace entry, so every first writer (and an
        ``EEXIST`` loser) must publish the separate, identity-bound ready
        record before it can return key material to the lifecycle layer.
        """
        if self._provided_key is not None:
            return self._provided_key
        # This guard serializes same-process first-use.  Cross-process callers
        # still converge through the no-replace ready record: no caller returns
        # a visible key until that record binds the exact inode and bytes.
        with _INITIALIZATION_GUARD:
            try:
                self.key_path.parent.mkdir(parents=True, exist_ok=True)
                self._assert_safe_parent()
            except OSError as exc:
                raise ManifestKeyError(
                    "Unable to create canonical manifest key directory"
                ) from exc
            key: bytes
            identity: tuple[int, int]
            try:
                descriptor = os.open(
                    self.key_path,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                )
            except FileExistsError:
                # The owner may have crashed or failed after publication.  An
                # exact, regular, single-link key can be *acknowledged*, but
                # it can never be consumed before the ready record exists.
                key, identity = self._read_existing_key_and_identity()
            except OSError as exc:
                raise ManifestKeyError("Unable to create canonical manifest key") from exc
            else:
                created_metadata: os.stat_result | None = None
                persistence_failure: BaseException | None = None
                try:
                    created_metadata = os.fstat(descriptor)
                    self._assert_safe_key_metadata(created_metadata)
                    self._write_all(descriptor, secrets.token_bytes(HMAC_SHA256_KEY_BYTES))
                    os.fsync(descriptor)
                except (ManifestKeyError, OSError) as exc:
                    persistence_failure = exc
                    if isinstance(exc, OSError) and created_metadata is not None:
                        self._remove_partial_key(created_metadata)
                    if isinstance(exc, OSError):
                        raise ManifestKeyError(
                            "Unable to persist canonical manifest key"
                        ) from exc
                    raise
                finally:
                    try:
                        os.close(descriptor)
                    except OSError as exc:
                        if persistence_failure is None:
                            # Preserve the exact unacknowledged inode.  A
                            # later initializer can resume acknowledgement;
                            # ordinary reads reject it until then.
                            raise ManifestKeyError(
                                "Unable to close canonical manifest key"
                            ) from exc
                assert created_metadata is not None
                identity = (created_metadata.st_dev, created_metadata.st_ino)
                # Re-read only through the ordinary no-follow boundary after
                # close, never from a writable descriptor.
                key, observed_identity = self._read_existing_key_and_identity()
                if observed_identity != identity:
                    raise ManifestKeyError("Canonical manifest key changed after creation")
            self._complete_key_acknowledgement(key, identity)
            return self._read_existing_key(require_ready=True)

    @property
    def _ready_path(self) -> Path:
        """Return the private completion record paired with the key inode."""
        return self.key_path.with_name(f"{self.key_path.name}{_KEY_READY_SUFFIX}")

    def _complete_key_acknowledgement(
        self, key: bytes, expected_identity: tuple[int, int]
    ) -> None:
        """Acknowledge and publish one exact identity-bound completion record."""
        # The key itself is a retained, exact identity-checked interprocess
        # initialization mutex.  A second process that lost ``O_EXCL`` cannot
        # observe or acknowledge a different inode while the winner publishes
        # the ready record.
        with self._acknowledgement_lock(expected_identity):
            current_key, current_identity = self._read_existing_key_and_identity()
            if current_key != key or current_identity != expected_identity:
                raise ManifestKeyError("Canonical manifest key changed before acknowledgement")
            if self._ready_matches(key, expected_identity, missing_ok=True):
                return
            try:
                self._acknowledge_new_key(expected_identity)
                ready = self._ready_bytes(key, expected_identity)
                self._publish_ready_record(ready)
            except ManifestKeyError:
                raise
            except OSError as exc:
                raise ManifestKeyError(
                    "Unable to acknowledge canonical manifest key publication"
                ) from exc

    @contextmanager
    def _acknowledgement_lock(self, expected_identity: tuple[int, int]):
        """Lock the one verified key inode across first-use acknowledgement."""
        descriptor: int | None = None
        handle: BinaryIO | None = None
        try:
            descriptor = os.open(
                self.key_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
            metadata = os.fstat(descriptor)
            self._assert_safe_key_metadata(metadata)
            if (metadata.st_dev, metadata.st_ino) != expected_identity:
                raise ManifestKeyError("Canonical manifest key changed before acknowledgement")
            handle = os.fdopen(descriptor, "rb", closefd=True)
            descriptor = None
            from .coordination import interprocess_open_file_lock

            with interprocess_open_file_lock(
                handle, exclusive=True, operation="manifest_key_initialization", close_handle=True
            ):
                handle = None
                self._assert_safe_parent()
                current = os.lstat(self.key_path)
                self._assert_safe_key_metadata(current)
                if (current.st_dev, current.st_ino) != expected_identity:
                    raise ManifestKeyError("Canonical manifest key changed during acknowledgement")
                yield
        except ManifestKeyError:
            raise
        except OSError as exc:
            raise ManifestKeyError("Unable to lock canonical manifest key initialization") from exc
        finally:
            if handle is not None:
                try:
                    handle.close()
                except OSError as exc:
                    raise ManifestKeyError(
                        "Unable to close canonical manifest key initialization lock"
                    ) from exc
            elif descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError as exc:
                    raise ManifestKeyError(
                        "Unable to close canonical manifest key initialization lock"
                    ) from exc

    @staticmethod
    def _ready_bytes(key: bytes, identity: tuple[int, int]) -> bytes:
        """Encode the fixed, bounded acknowledgement binding without a parser."""
        return (
            b"cacheness-manifest-key-ready-v1\n"
            + f"{identity[0]}:{identity[1]}:".encode("ascii")
            + hashlib.sha256(key).hexdigest().encode("ascii")
            + b"\n"
        )

    def _publish_ready_record(self, ready: bytes) -> None:
        """Durably install the no-replace completion record for this key."""
        ready_path = self._ready_path
        descriptor: int | None = None
        try:
            descriptor = os.open(
                ready_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except FileExistsError:
            # Another converging initializer may have published it.  Its bytes
            # are verified by the caller before any key is returned.
            return
        except OSError as exc:
            raise ManifestKeyError("Unable to publish canonical manifest key readiness") from exc
        try:
            self._write_all(descriptor, ready)
            os.fsync(descriptor)
        finally:
            try:
                os.close(descriptor)
            except OSError as exc:
                raise ManifestKeyError(
                    "Unable to close canonical manifest key readiness"
                ) from exc
        ready_metadata = os.lstat(ready_path)
        self._assert_safe_key_metadata(ready_metadata)
        identity = (ready_metadata.st_dev, ready_metadata.st_ino)
        if os.name == "posix":
            self._fsync_parent_entry_for(ready_path, identity)
        else:
            from .path_security import _windows_file_api

            _windows_file_api().flush_regular_file(ready_path, expected_identity=identity)

    def _ready_matches(
        self,
        key: bytes,
        expected_identity: tuple[int, int],
        *,
        missing_ok: bool,
    ) -> bool:
        """Return whether a no-follow ready record binds this exact key."""
        descriptor: int | None = None
        try:
            metadata = os.lstat(self._ready_path)
            self._assert_safe_key_metadata(metadata)
            descriptor = os.open(
                self._ready_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
            opened = os.fstat(descriptor)
            self._assert_safe_key_metadata(opened)
            if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                raise ManifestKeyError("Canonical manifest key readiness changed during open")
            raw = b""
            while len(raw) <= _MAX_READY_RECORD_BYTES:
                chunk = os.read(descriptor, _MAX_READY_RECORD_BYTES + 1 - len(raw))
                if not chunk:
                    break
                raw += chunk
        except FileNotFoundError:
            if missing_ok:
                return False
            raise ManifestKeyError("Canonical manifest key is not acknowledged") from None
        except ManifestKeyError:
            raise
        except OSError as exc:
            raise ManifestKeyError("Unable to read canonical manifest key readiness") from exc
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError as exc:
                    raise ManifestKeyError(
                        "Unable to close canonical manifest key readiness"
                    ) from exc
        if raw != self._ready_bytes(key, expected_identity):
            raise ManifestKeyError("Canonical manifest key readiness does not bind its key")
        return True

    def _acknowledge_new_key(self, expected_identity: tuple[int, int]) -> None:
        """Make the sole file trust root durable before any manifest is signed.

        POSIX acknowledges the identity-checked *parent directory entry* after
        the synced key file closes. Windows has no documented directory-handle
        flush contract; under D-22's one-user/session topology the default
        primitive flushes a verified regular key handle via ``FlushFileBuffers``
        and checks the original file identity before and after that operation.
        Deployments requiring a stronger namespace acknowledgement inject a
        ``ManifestKeyDurabilityProvider`` owned by their keystore.
        """
        if self._durability_provider is not None:
            self._durability_provider.acknowledge_new_key(
                self.key_path, expected_identity
            )
            return
        if os.name == "posix":
            self._fsync_parent_entry(expected_identity)
            return
        self._flush_windows_key_entry(expected_identity)

    def _fsync_parent_entry(self, expected_identity: tuple[int, int]) -> None:
        """Fsync the same safe POSIX parent that names the generated key."""
        self._fsync_parent_entry_for(self.key_path, expected_identity)

    def _fsync_parent_entry_for(
        self, locator: Path, expected_identity: tuple[int, int]
    ) -> None:
        """Fsync one exact regular-file entry in the safe key directory."""
        self._assert_safe_parent()
        parent_before = os.lstat(locator.parent)
        current = os.lstat(locator)
        self._assert_safe_key_metadata(current)
        if (current.st_dev, current.st_ino) != expected_identity:
            raise ManifestKeyError("Canonical manifest key entry changed before directory sync")
        descriptor: int | None = None
        try:
            descriptor = os.open(
                locator.parent,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            parent_opened = os.fstat(descriptor)
            if (parent_opened.st_dev, parent_opened.st_ino) != (
                parent_before.st_dev,
                parent_before.st_ino,
            ):
                raise ManifestKeyError(
                    "Canonical manifest key directory changed during sync"
                )
            os.fsync(descriptor)
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError as exc:
                    raise ManifestKeyError(
                        "Unable to close canonical manifest key directory"
                    ) from exc
        current = os.lstat(locator)
        self._assert_safe_key_metadata(current)
        if (current.st_dev, current.st_ino) != expected_identity:
            raise ManifestKeyError("Canonical manifest key entry changed after directory sync")

    def _flush_windows_key_entry(self, expected_identity: tuple[int, int]) -> None:
        """Use the D-22 documented regular-handle flush provider on Windows."""
        current = os.lstat(self.key_path)
        self._assert_safe_key_metadata(current)
        if (current.st_dev, current.st_ino) != expected_identity:
            raise ManifestKeyError("Canonical manifest key changed before Windows flush")
        # Imported lazily to keep the integrity primitive dependency-light on
        # POSIX. The native adapter opens and flushes a reparse-safe regular
        # handle, the documented acknowledgement we can truthfully claim for
        # the D-22 one-user/session file-provider topology.
        from .path_security import _windows_file_api

        _windows_file_api().flush_regular_file(
            self.key_path, expected_identity=expected_identity
        )
        current = os.lstat(self.key_path)
        self._assert_safe_key_metadata(current)
        if (current.st_dev, current.st_ino) != expected_identity:
            raise ManifestKeyError("Canonical manifest key changed after Windows flush")

    def get_or_initialize_new_store(self) -> bytes:
        """Read a current key or atomically initialize a proven-empty store.

        Missing key material is expected while the first canonical record is
        being created, so this path deliberately avoids constructing a public
        read failure merely to use it as control flow.
        """
        if self._provided_key is not None:
            return self._provided_key
        existing, identity = self._read_existing_key_and_identity(missing_ok=True)
        if existing is not None:
            assert identity is not None
            if self._ready_matches(existing, identity, missing_ok=True):
                return existing
        # A visible but unacknowledged key is deliberately treated exactly as
        # first-use work.  ``initialize_new_store`` resumes its exact
        # acknowledgement rather than consuming it or generating a replacement.
        return self.initialize_new_store()

    @staticmethod
    def _write_all(descriptor: int, data: bytes) -> None:
        """Persist a complete key if the operating system performs a short write."""
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written == 0:
                raise OSError("Unable to write canonical manifest key")
            view = view[written:]

    def _remove_partial_key(self, created_metadata: os.stat_result) -> None:
        """Remove only the incomplete key inode created by this provider."""
        try:
            current = os.lstat(self.key_path)
            if (current.st_dev, current.st_ino) == (
                created_metadata.st_dev,
                created_metadata.st_ino,
            ):
                os.unlink(self.key_path)
        except OSError:
            # The original persistence failure remains authoritative.  A
            # later open verifies any surviving evidence fail-closed.
            pass

    def _read_existing_key(
        self, *, missing_ok: bool = False, require_ready: bool = True
    ) -> bytes | None:
        """Read key bytes only after their exact completion binding is proven."""
        key, identity = self._read_existing_key_and_identity(missing_ok=missing_ok)
        if key is None:
            return None
        if require_ready:
            self._ready_matches(key, identity, missing_ok=False)
        return key

    def _read_existing_key_and_identity(
        self, *, missing_ok: bool = False
    ) -> tuple[bytes | None, tuple[int, int] | None]:
        """Read a safe key snapshot without assigning it authority yet."""
        descriptor: int | None = None
        try:
            self._assert_safe_parent()
            before_open = os.lstat(self.key_path)
            self._assert_safe_key_metadata(before_open)
            descriptor = os.open(
                self.key_path,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
            metadata = os.fstat(descriptor)
            self._assert_safe_key_metadata(metadata)
            if (metadata.st_dev, metadata.st_ino) != (
                before_open.st_dev,
                before_open.st_ino,
            ):
                raise ManifestKeyError("Canonical manifest key changed during open")
            key = b""
            while len(key) <= HMAC_SHA256_KEY_BYTES:
                chunk = os.read(descriptor, HMAC_SHA256_KEY_BYTES + 1 - len(key))
                if not chunk:
                    break
                key += chunk
        except ManifestKeyError:
            raise
        except FileNotFoundError:
            if missing_ok:
                return None, None
            raise ManifestKeyError("Unable to read canonical manifest key") from None
        except OSError as exc:
            raise ManifestKeyError("Unable to read canonical manifest key") from exc
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError as exc:
                    # A successful read followed by a failed close has an
                    # unknown descriptor outcome; preserve the strict typed
                    # manifest boundary rather than leaking raw OSError.
                    if "key" in locals():
                        raise ManifestKeyError(
                            "Unable to close canonical manifest key"
                        ) from exc
        _validate_key(key)
        return key, (metadata.st_dev, metadata.st_ino)

    def _assert_safe_parent(self) -> None:
        """Reject a reparse/symlink key directory before creating or reopening."""
        metadata = os.lstat(self.key_path.parent)
        if not stat.S_ISDIR(metadata.st_mode) or self._is_reparse_point(metadata):
            raise ManifestKeyError("Canonical manifest key directory is unsafe")

    def _assert_safe_key_metadata(self, metadata: os.stat_result) -> None:
        """Apply strict POSIX checks and Windows same-user/session reparse checks."""
        if not stat.S_ISREG(metadata.st_mode) or self._is_reparse_point(metadata):
            raise ManifestKeyError("Canonical manifest key must be a regular file")
        if metadata.st_nlink != 1:
            raise ManifestKeyError("Canonical manifest key must not be linked")
        if os.name == "posix":
            if metadata.st_uid != os.geteuid():
                raise ManifestKeyError("Canonical manifest key owner is unsafe")
            if stat.S_IMODE(metadata.st_mode) & 0o077:
                raise ManifestKeyError("Canonical manifest key permissions are unsafe")

    @staticmethod
    def _is_reparse_point(metadata: os.stat_result) -> bool:
        """Recognize Windows reparse points without treating normal files as links."""
        return bool(getattr(metadata, "st_file_attributes", 0) & 0x0400)


__all__ = [
    "HMAC_SHA256_KEY_BYTES",
    "ManifestKeyError",
    "ManifestKeyDurabilityProvider",
    "ManifestKeyProvider",
    "ManifestSigningKeyProvider",
    "sha256_and_size",
    "sha256_and_size_stream",
    "sign_hmac_sha256",
    "verify_hmac_sha256",
]
