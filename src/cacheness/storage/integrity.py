"""Cryptographic primitives used by the canonical BlobStore manifest path."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import stat
from pathlib import Path
from typing import BinaryIO


HMAC_SHA256_KEY_BYTES = 32


class ManifestKeyError(ValueError):
    """Raised when the persistent canonical manifest key is unavailable."""


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
    if not isinstance(key, bytes) or len(key) != HMAC_SHA256_KEY_BYTES:
        raise ManifestKeyError("Canonical manifest key must contain exactly 32 bytes")


class ManifestKeyProvider:
    """Create once, then strictly load and validate a local manifest HMAC key."""

    def __init__(self, key_path: Path, key: bytes | None = None):
        self.key_path = Path(key_path)
        self._provided_key = key
        if key is not None:
            _validate_key(key)

    def get_key(self) -> bytes:
        """Load the configured key or create one secure persistent key once."""
        if self._provided_key is not None:
            return self._provided_key
        if not self.key_path.exists():
            self._create_key_once()
        return self._read_existing_key()

    def _create_key_once(self) -> None:
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
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
            os.write(descriptor, key)
            os.fsync(descriptor)
        except OSError as exc:
            raise ManifestKeyError("Unable to persist canonical manifest key") from exc
        finally:
            os.close(descriptor)

    def _read_existing_key(self) -> bytes:
        try:
            metadata = self.key_path.stat()
            if stat.S_IMODE(metadata.st_mode) & 0o077:
                raise ManifestKeyError("Canonical manifest key permissions are unsafe")
            key = self.key_path.read_bytes()
        except ManifestKeyError:
            raise
        except OSError as exc:
            raise ManifestKeyError("Unable to read canonical manifest key") from exc
        _validate_key(key)
        return key


__all__ = [
    "HMAC_SHA256_KEY_BYTES",
    "ManifestKeyError",
    "ManifestKeyProvider",
    "sha256_and_size",
    "sha256_and_size_stream",
    "sign_hmac_sha256",
    "verify_hmac_sha256",
]
