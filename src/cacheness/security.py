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
import secrets
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


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
            "prefix",
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
            "prefix",
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
    ):
        """
        Initialize the cache entry signer.

        Args:
            key_file_path: Path to the signing key file (ignored if use_in_memory_key=True)
            use_in_memory_key: If True, use in-memory key instead of persistent file
        """
        self.key_file_path = key_file_path
        self.use_in_memory_key = use_in_memory_key
        self.secret_key = self._load_or_generate_key()

        key_type = "in-memory" if use_in_memory_key else "persistent"
        current_fields = self.SIGNED_FIELDS_BY_VERSION[self.CURRENT_SIGNATURE_VERSION]
        logger.debug(
            f"Cache signer initialized: version={self.CURRENT_SIGNATURE_VERSION}, "
            f"fields={current_fields}, key_type={key_type}"
        )

    def _load_or_generate_key(self) -> bytes:
        """Load existing signing key or generate a new one."""
        # If using in-memory key, always generate new one
        if self.use_in_memory_key:
            logger.info("Using in-memory signing key (not persistent)")
            return secrets.token_bytes(32)

        try:
            if self.key_file_path.exists():
                # Load existing key
                key = self.key_file_path.read_bytes()
                if len(key) != 32:
                    logger.warning(
                        f"Invalid key length ({len(key)} bytes), generating new key"
                    )
                    return self._generate_new_key()
                logger.debug(f"Loaded signing key from {self.key_file_path}")
                return key
            else:
                return self._generate_new_key()
        except Exception as e:
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

            # Set restrictive file permissions (owner read/write only)
            try:
                self.key_file_path.chmod(0o600)
            except Exception as e:
                logger.warning(
                    f"Failed to set restrictive permissions on key file: {e}"
                )

            logger.info(f"Generated new signing key: {self.key_file_path}")
            return key

        except Exception as e:
            logger.error(f"Failed to generate signing key: {e}")
            # Fallback to in-memory key (not persistent)
            logger.warning("Using in-memory signing key (not persistent)")
            return key

    def _create_signature_payload(
        self, entry_data: Dict[str, Any], version: int
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

    def sign_entry(self, entry_data: Dict[str, Any]) -> str:
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
            version = self.CURRENT_SIGNATURE_VERSION
            payload = self._create_signature_payload(entry_data, version)

            hex_sig = hmac.new(
                self.secret_key, payload.encode("utf-8"), hashlib.sha256
            ).hexdigest()

            versioned = f"v{version}:{hex_sig}"

            logger.debug(
                f"Created v{version} signature for entry "
                f"{entry_data.get('cache_key', 'unknown')}"
            )
            return versioned

        except Exception as e:
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
        entry_data: Dict[str, Any],
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
        try:
            payload = self._create_signature_payload(entry_data, version)
            expected_signature = hmac.new(
                self.secret_key, payload.encode("utf-8"), hashlib.sha256
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

        except Exception as e:
            logger.error(f"Failed to verify signature: {e}")
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
        }


def create_cache_signer(
    cache_dir: Path,
    key_file: str = "cache_signing_key.bin",
    use_in_memory_key: bool = False,
) -> CacheEntrySigner:
    """
    Factory function to create a cache entry signer.

    Args:
        cache_dir: Cache directory where key file will be stored
        key_file: Name of the signing key file (ignored if use_in_memory_key=True)
        use_in_memory_key: If True, use in-memory key instead of persistent file

    Returns:
        Configured CacheEntrySigner instance
    """
    key_file_path = cache_dir / key_file
    return CacheEntrySigner(key_file_path, use_in_memory_key)
