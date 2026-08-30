"""Cryptographic primitives used by the canonical BlobStore manifest path."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import stat
from pathlib import Path
from typing import BinaryIO, Protocol, runtime_checkable

from cacheness.error_handling import (
    CacheBlobManifestUnauthenticatedError,
    CacheReason,
)


HMAC_SHA256_KEY_BYTES = 32


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

    def __init__(self, key_path: Path, key: bytes | None = None):
        self.key_path = Path(key_path)
        self._provided_key = key
        if key is not None:
            _validate_key(key)

    def get_key(self) -> bytes:
        """Load already-authorized key material without writing a key file."""
        if self._provided_key is not None:
            return self._provided_key
        return self._read_existing_key()

    def initialize_new_store(self) -> bytes:
        """Create a key exactly once for an explicitly initialized empty store."""
        if self._provided_key is not None:
            return self._provided_key
        if os.name != "posix":
            raise ManifestKeyError(
                "File-backed canonical manifest keys are unsupported on this platform"
            )
        try:
            self.key_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ManifestKeyError("Unable to create canonical manifest key directory") from exc
        key = secrets.token_bytes(HMAC_SHA256_KEY_BYTES)
        try:
            descriptor = os.open(
                self.key_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            return
        except OSError as exc:
            raise ManifestKeyError("Unable to create canonical manifest key") from exc
        try:
            self._write_all(descriptor, key)
            os.fsync(descriptor)
        except OSError as exc:
            raise ManifestKeyError("Unable to persist canonical manifest key") from exc
        finally:
            os.close(descriptor)
        return self._read_existing_key()

    @staticmethod
    def _write_all(descriptor: int, data: bytes) -> None:
        """Persist a complete key if the operating system performs a short write."""
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written == 0:
                raise OSError("Unable to write canonical manifest key")
            view = view[written:]

    def _read_existing_key(self) -> bytes:
        if os.name != "posix":
            raise ManifestKeyError(
                "File-backed canonical manifest keys are unsupported on this platform"
            )
        descriptor: int | None = None
        try:
            descriptor = os.open(self.key_path, os.O_RDONLY | os.O_NOFOLLOW)
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ManifestKeyError("Canonical manifest key must be a regular file")
            if metadata.st_uid != os.geteuid():
                raise ManifestKeyError("Canonical manifest key owner is unsafe")
            if stat.S_IMODE(metadata.st_mode) & 0o077:
                raise ManifestKeyError("Canonical manifest key permissions are unsafe")
            key = b""
            while len(key) <= HMAC_SHA256_KEY_BYTES:
                chunk = os.read(descriptor, HMAC_SHA256_KEY_BYTES + 1 - len(key))
                if not chunk:
                    break
                key += chunk
        except ManifestKeyError:
            raise
        except OSError as exc:
            raise ManifestKeyError("Unable to read canonical manifest key") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
        _validate_key(key)
        return key


__all__ = [
    "HMAC_SHA256_KEY_BYTES",
    "ManifestKeyError",
    "ManifestKeyProvider",
    "ManifestSigningKeyProvider",
    "sha256_and_size",
    "sha256_and_size_stream",
    "sign_hmac_sha256",
    "verify_hmac_sha256",
]
