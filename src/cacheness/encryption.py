"""
Encryption at Rest for Blob Content
====================================

Provides AES-256-GCM authenticated encryption for blob data stored by
BlobStore.  Encryption happens *between* handler compression and blob
backend write, so handlers remain encryption-unaware.

Requires the ``cryptography`` package::

    pip install cacheness[encryption]

Security Model:
- AES-256-GCM with random 12-byte IV per blob (authenticated encryption)
- HKDF-SHA256 key derivation with domain-separated info strings
- Encrypt-then-sign: blob is encrypted, then HMAC signs the ciphertext
- Per-namespace key derivation prevents cross-namespace key reuse

Functions:
- ``encrypt_blob`` — encrypt plaintext bytes, returns (ciphertext, iv, algorithm)
- ``decrypt_blob`` — decrypt ciphertext bytes using key and IV
- ``derive_encryption_key`` — HKDF-derive a per-namespace encryption key
"""

import logging
import os

from .security import _hkdf_sha256

logger = logging.getLogger(__name__)

# Lazy import — cryptography is an optional dependency
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.exceptions import InvalidTag

    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    AESGCM = None  # type: ignore[assignment,misc]
    InvalidTag = None  # type: ignore[assignment,misc]
    CRYPTOGRAPHY_AVAILABLE = False


def derive_encryption_key(master_key: bytes, namespace_id: str) -> bytes:
    """Derive a per-namespace AES-256 encryption key via HKDF-SHA256.

    Uses a domain-separated info string so each namespace gets a unique
    key even when sharing the same master key material.

    Args:
        master_key: 32-byte master key (read from key file).
        namespace_id: Namespace identifier for domain separation.

    Returns:
        32-byte derived key suitable for AES-256-GCM.
    """
    return _hkdf_sha256(
        master_key,
        info=b"cacheness-aes-gcm-v1:" + namespace_id.encode(),
    )


def encrypt_blob(data: bytes, key: bytes) -> tuple[bytes, bytes, bytes]:
    """Encrypt blob content with AES-256-GCM.

    Generates a random 12-byte IV for each invocation.  The returned
    ciphertext includes the 16-byte GCM authentication tag appended by
    the ``cryptography`` library.

    Args:
        data: Plaintext blob bytes to encrypt.
        key: 32-byte AES-256 key (from :func:`derive_encryption_key`).

    Returns:
        Tuple of ``(ciphertext, iv, algorithm)`` where *algorithm* is
        ``b"aes-256-gcm"`` and *ciphertext* includes the GCM auth tag.
    """
    iv = os.urandom(12)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(iv, data, None)
    return (ciphertext, iv, b"aes-256-gcm")


def decrypt_blob(ciphertext: bytes, key: bytes, iv: bytes) -> bytes:
    """Decrypt AES-256-GCM encrypted blob content.

    Args:
        ciphertext: Encrypted bytes (includes 16-byte GCM auth tag).
        key: 32-byte AES-256 key (must match the key used for encryption).
        iv: 12-byte initialisation vector used during encryption.

    Returns:
        Decrypted plaintext bytes.

    Raises:
        CacheIntegrityError: If decryption fails due to wrong key or
            tampered ciphertext (GCM authentication tag mismatch).
    """
    from .error_handling import CacheIntegrityError

    aesgcm = AESGCM(key)
    try:
        return aesgcm.decrypt(iv, ciphertext, None)
    except InvalidTag:
        raise CacheIntegrityError(
            "Decryption failed — ciphertext has been tampered with or wrong key"
        )
