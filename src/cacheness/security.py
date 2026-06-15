"""
Cache Entry Security and Signing
===============================

This module provides cryptographic signing for cache metadata entries to prevent tampering.
Uses HMAC-SHA256 for fast, secure signatures of critical metadata fields.

Features:
- HMAC-based signing for metadata integrity
- Version-based signed field lists for safe evolution
- Automatic key generation and management
- Backward compatibility with unsigned and legacy entries
- Key rotation support

Security Model:
- Signs only immutable fields to prevent signature invalidation
- Uses deterministic field ordering for consistent signatures
- Stores signature alongside entry metadata
- Verifies signatures on cache retrieval
"""

import hmac
import hashlib
import os
import secrets
import subprocess
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from .interfaces import SignableFields

logger = logging.getLogger(__name__)


def staged_key_file_path(key_file_path: Path) -> Path:
    """Return the staged rotation path for an active key file."""
    return key_file_path.with_name(f"{key_file_path.name}.new")


def write_staged_key_file(key_file_path: Path, key: bytes) -> Path:
    """Write key bytes to ``<keyfile>.new`` with key-file permissions."""
    staged_path = staged_key_file_path(key_file_path)
    staged_path.parent.mkdir(parents=True, exist_ok=True)
    staged_path.write_bytes(key)
    CacheEntrySigner._set_key_file_permissions(staged_path)
    return staged_path


def _hkdf_sha256(ikm: bytes, info: bytes, length: int = 32, salt: bytes = b"") -> bytes:
    """HKDF-SHA256 key derivation (RFC 5869) using stdlib only."""
    # Extract: PRK = HMAC-SHA256(salt, IKM)
    if not salt:
        salt = b"\x00" * 32
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    # Expand: OKM = HMAC-SHA256(PRK, info || 0x01) — single block for 32 bytes
    okm = hmac.new(prk, info + b"\x01", hashlib.sha256).digest()
    return okm[:length]


class CacheEntrySigner:
    """
    HMAC-based cache entry signer for metadata integrity protection.

    Provides cryptographic signatures for cache entry metadata to detect tampering
    with the SQLite database or JSON metadata files.

    Signed fields are managed internally via version-based field lists.
    Legacy entries (no stored version) are verified with the v1 field list.
    """

    # Version-based signed field lists.
    # Each version defines exactly which fields are included in the signature.
    # Legacy entries without a stored signature_version use v1.
    SIGNED_FIELDS_BY_VERSION: Dict[int, List[str]] = {
        1: [
            "cache_key",
            "data_type",
            "file_size",
            "file_hash",
            "object_type",
            "storage_format",
            "serializer",
            "compression_codec",
            "actual_path",
            "created_at",
        ],
        2: [
            "cache_key",
            "data_type",
            "file_size",
            "file_hash",
            "object_type",
            "storage_format",
            "serializer",
            "compression_codec",
            "created_at",
        ],
        3: [
            "cache_key",
            "data_type",
            "file_size",
            "file_hash",
            "object_type",
            "storage_format",
            "serializer",
            "compression_codec",
            "created_at",
        ],
    }

    # Current version used when signing new entries.
    CURRENT_SIGNATURE_VERSION: int = 2

    def __init__(
        self,
        key_file_path: Path,
        use_in_memory_key: bool = False,
        key_fallback_policy: str = "warn",
        namespace_id: str = "default",
        use_hkdf_derivation: bool = True,
        minimum_signature_version: int = 1,
    ):
        """
        Initialize the cache entry signer.

        Args:
            key_file_path: Path to the signing key file (ignored if use_in_memory_key=True)
            use_in_memory_key: If True, use in-memory key instead of persistent file
            key_fallback_policy: What to do when key file cannot be written.
                'raise' = raise CacheSecurityError
                'warn' = log WARNING + use in-memory key
                'fallback' = silently use in-memory key
            namespace_id: Namespace identifier for HKDF key derivation
            use_hkdf_derivation: If True, derive per-namespace keys via HKDF-SHA256
            minimum_signature_version: Lowest accepted entry signature version.
        """
        if not isinstance(minimum_signature_version, int):
            raise ValueError("minimum_signature_version must be an integer")
        if minimum_signature_version < 1:
            raise ValueError("minimum_signature_version must be at least 1")

        self.key_file_path = key_file_path
        self.use_in_memory_key = use_in_memory_key
        self.key_fallback_policy = key_fallback_policy
        self.namespace_id = namespace_id
        self.use_hkdf_derivation = use_hkdf_derivation
        self.minimum_signature_version = minimum_signature_version
        self.secret_key = self._load_or_generate_key()

        # Store master key and derive per-namespace key
        self.master_key = self.secret_key
        if use_hkdf_derivation:
            self.derived_key = _hkdf_sha256(
                self.master_key,
                info=f"cacheness-ns-v1:{namespace_id}".encode("utf-8"),
            )
        else:
            self.derived_key = self.master_key

        key_type = "in-memory" if use_in_memory_key else "persistent"
        current_fields = self.SIGNED_FIELDS_BY_VERSION[self.CURRENT_SIGNATURE_VERSION]
        logger.debug(
            f"Cache signer initialized: version={self.CURRENT_SIGNATURE_VERSION}, "
            f"fields={current_fields}, key_type={key_type}, "
            f"hkdf={use_hkdf_derivation}, namespace={namespace_id}, "
            f"minimum_signature_version={minimum_signature_version}"
        )

    def _load_or_generate_key(self) -> bytes:
        """Load existing signing key or generate a new one."""
        # If using in-memory key, always generate new one
        if self.use_in_memory_key:
            logger.info("Using in-memory signing key (not persistent)")
            return secrets.token_bytes(32)

        try:
            staged_path = staged_key_file_path(self.key_file_path)
            if staged_path.exists():
                logger.error(
                    "Detected interrupted key rotation: staged key file %s remains. "
                    "The active key file %s was left unchanged; remove the staged "
                    "file after verifying cache readability or retry rotation.",
                    staged_path,
                    self.key_file_path,
                )

            if self.key_file_path.exists():
                # Load existing key
                key = self.key_file_path.read_bytes()
                if len(key) != 32:
                    if self.key_fallback_policy == "raise":
                        from .error_handling import CacheSecurityError

                        raise CacheSecurityError(
                            f"Invalid signing key length ({len(key)} bytes) "
                            f"in {self.key_file_path}. Expected 32 bytes. "
                            f"Delete the file and retry."
                        )
                    elif self.key_fallback_policy == "warn":
                        logger.warning(
                            f"Invalid key length ({len(key)} bytes) in "
                            f"{self.key_file_path}, generating new key"
                        )
                    # "fallback" mode: no log, just regenerate
                    return self._generate_new_key()
                logger.debug(f"Loaded signing key from {self.key_file_path}")
                return key
            else:
                return self._generate_new_key()
        except (OSError, ValueError) as e:
            logger.warning(f"Failed to load signing key: {e}, generating new key")
            return self._generate_new_key()

    def _generate_new_key(self) -> bytes:
        """Generate a new 32-byte signing key and save it (unless using in-memory key)."""
        # Generate 32-byte key for HMAC-SHA256
        key = secrets.token_bytes(32)

        # Skip file operations for in-memory keys
        if self.use_in_memory_key:
            logger.debug("Generated in-memory signing key")
            return key

        try:
            # Ensure directory exists
            self.key_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Save key with restrictive permissions
            self.key_file_path.write_bytes(key)

            # Set restrictive file permissions (owner-only access)
            self._set_key_file_permissions(self.key_file_path)

            logger.info(f"Generated new signing key: {self.key_file_path}")
            return key

        except OSError as e:
            if self.key_fallback_policy == "raise":
                from .error_handling import CacheSecurityError

                raise CacheSecurityError(
                    f"Failed to persist signing key to {self.key_file_path}: {e}. "
                    f"Set key_fallback_policy='warn' or 'fallback' "
                    f"to allow in-memory fallback."
                ) from e
            elif self.key_fallback_policy == "warn":
                logger.warning(
                    f"Failed to persist signing key to "
                    f"{self.key_file_path}: {e}. "
                    f"Using in-memory signing key (not persistent)."
                )
            # "fallback" mode: silent — no log output
            return key

    @staticmethod
    def _set_key_file_permissions(path: Path) -> None:
        """Set restrictive permissions on the key file (owner-only access)."""
        if sys.platform == "win32":
            # On Windows, chmod is a no-op for ACLs. Use icacls to restrict
            # the key file to the current user only.
            try:
                username = os.getlogin()
                subprocess.run(
                    [
                        "icacls",
                        str(path),
                        "/inheritance:r",
                        "/grant:r",
                        f"{username}:(R,W)",
                    ],
                    check=True,
                    capture_output=True,
                    timeout=10,
                )
            except Exception as e:
                logger.warning(f"Failed to set Windows ACL on key file: {e}")
        else:
            try:
                path.chmod(0o600)
            except OSError as e:
                logger.warning(
                    f"Failed to set restrictive permissions on key file: {e}"
                )

    def _create_signature_payload(
        self, entry_data: SignableFields, version: int
    ) -> str:
        """
        Create deterministic payload string from entry data.

        Args:
            entry_data: Complete entry data dictionary
            version: Signature version determining which fields to sign

        Returns:
            Deterministic string representation of signed fields
        """
        signed_fields = self.SIGNED_FIELDS_BY_VERSION.get(version)
        if signed_fields is None:
            raise ValueError(
                f"Unknown signature version {version}. "
                f"Known versions: {sorted(self.SIGNED_FIELDS_BY_VERSION.keys())}"
            )

        # Extract values in consistent order (sorted field names)
        values = []
        for field in sorted(signed_fields):
            value = entry_data.get(field)

            # Handle None values
            if value is None:
                value = ""
            # Convert datetime objects to ISO format
            elif isinstance(value, datetime):
                value = value.isoformat()
            # Convert to string
            else:
                value = str(value)

            values.append(f"{field}:{value}")

        payload = "|".join(values)
        logger.debug(
            f"Signature payload: {payload[:100]}..."
            if len(payload) > 100
            else f"Signature payload: {payload}"
        )
        return payload

    def sign_entry(self, entry_data: SignableFields) -> str:
        """
        Create HMAC signature for cache entry using the current version.

        The version is embedded in the returned string as ``v{N}:{hex}``.
        Legacy callers that stored bare hex signatures are treated as v1
        during verification.

        Args:
            entry_data: Complete entry data dictionary containing all fields

        Returns:
            Versioned signature string in the format ``v{N}:{hex_signature}``
        """
        try:
            version = 3 if self.use_hkdf_derivation else self.CURRENT_SIGNATURE_VERSION
            payload = self._create_signature_payload(entry_data, version)

            hex_sig = hmac.new(
                self.derived_key, payload.encode("utf-8"), hashlib.sha256
            ).hexdigest()

            versioned = f"v{version}:{hex_sig}"

            logger.debug(
                f"Created v{version} signature for entry "
                f"{entry_data.get('cache_key', 'unknown')}"
            )
            return versioned

        except (ValueError, TypeError) as e:
            logger.error(f"Failed to create signature: {e}")
            raise

    @staticmethod
    def parse_versioned_signature(stored_signature: str) -> tuple[int, str]:
        """
        Parse a stored signature into (version, hex_signature).

        New-format signatures look like ``v2:abcdef01...``.
        Legacy signatures are bare hex strings and are treated as v1.

        Returns:
            Tuple of (version, hex_signature)
        """
        if stored_signature and ":" in stored_signature:
            prefix, _, hex_sig = stored_signature.partition(":")
            if prefix.startswith("v") and prefix[1:].isdigit():
                return int(prefix[1:]), hex_sig
        # Legacy bare-hex signature → v1
        return 1, stored_signature

    def verify_entry(
        self,
        entry_data: SignableFields,
        stored_signature: str,
    ) -> bool:
        """
        Verify HMAC signature for cache entry.

        The version is extracted from the stored signature string.
        Legacy bare-hex signatures are treated as v1.

        Args:
            entry_data: Complete entry data dictionary
            stored_signature: Previously stored signature (``v{N}:{hex}`` or bare hex)

        Returns:
            True if signature is valid, False otherwise
        """
        version, hex_sig = self.parse_versioned_signature(stored_signature)
        if version < self.minimum_signature_version:
            logger.warning(
                f"Rejected signature for entry "
                f"{entry_data.get('cache_key', 'unknown')}: "
                f"signature version v{version} is below configured minimum "
                f"v{self.minimum_signature_version}"
            )
            return False

        try:
            payload = self._create_signature_payload(entry_data, version)
            # v1/v2 used the master key; v3+ uses the HKDF-derived key
            key = self.derived_key if version >= 3 else self.master_key
            expected_signature = hmac.new(
                key, payload.encode("utf-8"), hashlib.sha256
            ).hexdigest()

            is_valid = hmac.compare_digest(expected_signature, hex_sig)

            if not is_valid:
                logger.warning(
                    f"Signature verification failed for entry "
                    f"{entry_data.get('cache_key', 'unknown')} (v{version})"
                )
            else:
                logger.debug(
                    f"Signature verified for entry "
                    f"{entry_data.get('cache_key', 'unknown')} (v{version})"
                )

            return is_valid

        except (ValueError, TypeError) as e:
            logger.error(f"Failed to verify signature: {e}")
            return False

    # ------------------------------------------------------------------
    # Namespace signing
    # ------------------------------------------------------------------
    # Namespace rows are signed on immutable fields only (namespace_id,
    # display_name, created_at).  schema_version is excluded because it
    # changes during migrations.
    # ------------------------------------------------------------------

    NAMESPACE_SIGNED_FIELDS: List[str] = [
        "created_at",
        "display_name",
        "namespace_id",
    ]

    def _create_namespace_payload(self, namespace_data: Dict[str, Any]) -> str:
        """Create deterministic payload string from namespace fields."""
        values = []
        for field in self.NAMESPACE_SIGNED_FIELDS:  # already sorted
            value = namespace_data.get(field)
            if value is None:
                value = ""
            elif isinstance(value, datetime):
                value = value.isoformat()
            else:
                value = str(value)
            values.append(f"{field}:{value}")
        return "|".join(values)

    def sign_namespace(self, namespace_data: Dict[str, Any]) -> str:
        """Create HMAC signature for a namespace registry row.

        Only immutable fields are signed (``namespace_id``, ``display_name``,
        ``created_at``).  ``schema_version`` is deliberately excluded because
        it changes during migrations.

        Args:
            namespace_data: Dict with at least the keys in
                :attr:`NAMESPACE_SIGNED_FIELDS`.

        Returns:
            Versioned signature string (``ns1:{hex}``).
        """
        try:
            payload = self._create_namespace_payload(namespace_data)
            ns_version = "ns2" if self.use_hkdf_derivation else "ns1"
            hex_sig = hmac.new(
                self.derived_key, payload.encode("utf-8"), hashlib.sha256
            ).hexdigest()
            versioned = f"{ns_version}:{hex_sig}"
            logger.debug(f"Signed namespace {namespace_data.get('namespace_id', '?')}")
            return versioned
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to sign namespace: {e}")
            raise

    def verify_namespace(
        self, namespace_data: Dict[str, Any], stored_signature: str
    ) -> bool:
        """Verify HMAC signature for a namespace registry row.

        Args:
            namespace_data: Dict with namespace fields.
            stored_signature: Previously stored signature (``ns1:{hex}``).

        Returns:
            True if the signature is valid.
        """
        if not stored_signature:
            return False
        # Parse — expect "ns1:<hex>" or "ns2:<hex>"
        prefix, _, hex_sig = stored_signature.partition(":")
        if not prefix.startswith("ns") or not hex_sig:
            logger.warning(
                f"Unrecognised namespace signature format: {stored_signature[:20]}"
            )
            return False
        try:
            payload = self._create_namespace_payload(namespace_data)
            # ns1 used master key; ns2 uses HKDF-derived key
            key = self.derived_key if prefix == "ns2" else self.master_key
            expected = hmac.new(
                key, payload.encode("utf-8"), hashlib.sha256
            ).hexdigest()
            is_valid = hmac.compare_digest(expected, hex_sig)
            if not is_valid:
                logger.warning(
                    f"Namespace signature verification failed for "
                    f"{namespace_data.get('namespace_id', '?')}"
                )
            return is_valid
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to verify namespace signature: {e}")
            return False

    def get_field_info(self) -> Dict[str, Any]:
        """Get information about the current signing configuration."""
        return {
            "signature_version": self.CURRENT_SIGNATURE_VERSION,
            "signed_fields": self.SIGNED_FIELDS_BY_VERSION[
                self.CURRENT_SIGNATURE_VERSION
            ],
            "known_versions": sorted(self.SIGNED_FIELDS_BY_VERSION.keys()),
            "key_file": str(self.key_file_path),
            "key_exists": self.key_file_path.exists()
            if not self.use_in_memory_key
            else False,
            "use_in_memory_key": self.use_in_memory_key,
            "use_hkdf_derivation": self.use_hkdf_derivation,
            "minimum_signature_version": self.minimum_signature_version,
            "namespace_id": self.namespace_id,
        }


def create_cache_signer(
    cache_dir: Path,
    key_file: str = "cache_signing_key.bin",
    use_in_memory_key: bool = False,
    key_fallback_policy: str = "warn",
    namespace_id: str = "default",
    use_hkdf_derivation: bool = True,
    minimum_signature_version: int = 1,
) -> CacheEntrySigner:
    """
    Factory function to create a cache entry signer.

    Args:
        cache_dir: Cache directory where key file will be stored
        key_file: Name of the signing key file (ignored if use_in_memory_key=True)
        key_fallback_policy: What to do when key file cannot be written.
            'raise' = raise CacheSecurityError
            'warn' = log WARNING + use in-memory key
            'fallback' = silently use in-memory key
        namespace_id: Namespace identifier for HKDF key derivation
        use_hkdf_derivation: If True, derive per-namespace keys via HKDF-SHA256
        minimum_signature_version: Lowest accepted entry signature version.

    Returns:
        Configured CacheEntrySigner instance
    """
    key_file_path = cache_dir / key_file
    return CacheEntrySigner(
        key_file_path,
        use_in_memory_key,
        key_fallback_policy,
        namespace_id=namespace_id,
        use_hkdf_derivation=use_hkdf_derivation,
        minimum_signature_version=minimum_signature_version,
    )
