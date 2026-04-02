"""Verification and signing mixin for UnifiedCache."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .interfaces import IntegrityReport, SignableFields

logger = logging.getLogger(__name__)


class VerificationMixin:
    """Cache entry integrity verification and cryptographic signing."""

    def _extract_signable_fields(
        self,
        cache_key: str,
        entry_data: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> SignableFields:
        """
        Extract fields for signing/verification in a consistent manner.

        This ensures that the same fields are used during both put() and get()
        to prevent signature mismatches.  The signer selects which fields to
        include based on the signature version; this method provides the
        superset of all potentially-signable fields.

        Args:
            cache_key: The cache key
            entry_data: The entry data dictionary (data_type, etc.)
            metadata: The metadata dictionary from handler result

        Returns:
            SignableFields containing all fields that may be signed
        """
        # Normalize created_at to ISO format string without timezone
        # This ensures consistent signatures regardless of database format
        created_at = entry_data.get("created_at")
        if isinstance(created_at, datetime):
            # Convert datetime to ISO string, removing timezone info for consistency
            created_at = created_at.replace(tzinfo=None).isoformat()
        elif isinstance(created_at, str):
            # Parse and re-format to ensure consistency
            try:
                dt = datetime.fromisoformat(created_at)
                created_at = dt.replace(tzinfo=None).isoformat()
            except (ValueError, TypeError):
                # If parsing fails, use as-is
                pass

        # Build complete entry data with all potentially-signable fields.
        # The signer's version-based field list determines which are actually used.
        signable_data: SignableFields = {
            "cache_key": cache_key,
            "data_type": entry_data.get("data_type"),
            "file_size": entry_data.get("file_size", 0),
            "created_at": created_at,
            "actual_path": metadata.get("actual_path", ""),
            "file_hash": metadata.get("file_hash"),
            "object_type": metadata.get("object_type"),
            "storage_format": metadata.get("storage_format"),
            "serializer": metadata.get("serializer"),
            "compression_codec": metadata.get("compression_codec"),
        }

        return signable_data

    def _calculate_file_hash(self, file_path: Path) -> Optional[str]:
        """Calculate XXH3_64 hash of a cache file.

        Delegates to BlobStore._calculate_file_hash().
        Kept for backward compatibility.
        """
        return self._blob_store._calculate_file_hash(file_path)

    def _verify_entry(
        self,
        cache_key: str,
        entry: Dict[str, Any],
        metadata: Dict[str, Any],
        file_path: Path,
        *,
        storage_mode: bool = False,
    ) -> bool:
        """Verify entry integrity (hash) and signature before loading.

        Returns ``True`` when loading should proceed, ``False`` when the
        entry must be rejected (caller should return ``None``).

        Side effects handled internally:

        * Logs warnings on all failures.
        * When *storage_mode* is ``False``, deletes corrupted or tampered
          entries according to ``config.metadata.delete_on_error`` and
          ``config.security.delete_invalid_signatures``.
        * When *storage_mode* is ``True``, entries are **never** deleted.

        The caller is responsible for recording hits/misses and returning
        ``None`` when this method returns ``False``.
        """
        # ── Integrity verification (file hash) ─────────────────────
        if self.config.metadata.verify_cache_integrity:
            stored_hash = metadata.get("file_hash")
            if stored_hash is not None:
                # For inline entries, compute hash from blob_data in memory
                if entry.get("is_inline") and entry.get("blob_data") is not None:
                    import xxhash

                    current_hash = xxhash.xxh3_64(entry["blob_data"]).hexdigest()
                else:
                    current_hash = self._blob_store._calculate_file_hash(file_path)
                if current_hash != stored_hash:
                    detail = f"stored hash {stored_hash} != current hash {current_hash}"
                    self._invoke_hook(
                        "on_integrity_failure",
                        cache_key,
                        "hash_mismatch",
                        detail,
                    )
                    if storage_mode:
                        logger.warning(
                            f"Cache integrity verification failed for {cache_key}: "
                            f"{detail}. Entry preserved (storage mode)."
                        )
                    elif self.config.metadata.delete_on_error:
                        logger.warning(
                            f"Cache integrity verification failed for {cache_key}: "
                            f"{detail}. Removing corrupted cache entry."
                        )
                        self._blob_store.delete(cache_key)
                    else:
                        logger.warning(
                            f"Cache integrity verification failed for {cache_key}: "
                            f"{detail}. Entry retained due to delete_on_error=False."
                        )
                    return False

        # ── Signature verification ──────────────────────────────────
        if self.signer and self.config.security.enable_entry_signing:
            stored_signature = metadata.get("entry_signature")
            if stored_signature is not None:
                verify_data = self._extract_signable_fields(
                    cache_key=cache_key,
                    entry_data=entry,
                    metadata=metadata,
                )
                if not self.signer.verify_entry(verify_data, stored_signature):
                    self._invoke_hook(
                        "on_integrity_failure",
                        cache_key,
                        "signature_invalid",
                        "HMAC signature verification failed",
                    )
                    if storage_mode:
                        logger.warning(
                            f"Entry signature verification failed for {cache_key}. "
                            f"Entry preserved (storage mode)."
                        )
                        return False
                    elif self.config.security.delete_invalid_signatures:
                        logger.warning(
                            f"Entry signature verification failed for {cache_key}. "
                            f"Removing potentially tampered cache entry."
                        )
                        self._blob_store.delete(cache_key)
                        return False
                    else:
                        logger.warning(
                            f"Entry signature verification failed for {cache_key}. "
                            f"Entry retained due to delete_invalid_signatures=False."
                        )
                        # Continue loading despite invalid signature

            elif not self.config.security.allow_unsigned_entries:
                self._invoke_hook(
                    "on_integrity_failure",
                    cache_key,
                    "unsigned_rejected",
                    "Entry has no signature and unsigned entries are not allowed",
                )
                if storage_mode:
                    logger.warning(
                        f"Entry {cache_key} has no signature but unsigned entries "
                        f"are not allowed. Entry preserved (storage mode)."
                    )
                else:
                    logger.warning(
                        f"Entry {cache_key} has no signature but unsigned entries "
                        f"are not allowed. Removing entry."
                    )
                    self._blob_store.delete(cache_key)
                return False

        return True

    def _sign_entry_if_enabled(
        self,
        cache_key: str,
        entry_data: Dict[str, Any],
        metadata_dict: Dict[str, Any],
    ) -> None:
        """Sign *entry_data* in-place if a signer is configured.

        Sets ``entry_data["created_at"]`` and
        ``metadata_dict["entry_signature"]`` on success.
        Logs a warning and continues without a signature on failure.
        """
        if not self.signer:
            return
        try:
            creation_timestamp = datetime.now(timezone.utc)
            entry_data["created_at"] = creation_timestamp.isoformat()
            complete_entry_data = self._extract_signable_fields(
                cache_key=cache_key,
                entry_data=entry_data,
                metadata=metadata_dict,
            )
            signature = self.signer.sign_entry(complete_entry_data)
            metadata_dict["entry_signature"] = signature
            logger.debug(f"Created signature for entry {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to sign entry {cache_key}: {e}")

    def verify_integrity(
        self, repair: bool = False, verify_hashes: bool = False
    ) -> IntegrityReport:
        """
        Verify cache integrity by cross-checking blob files and metadata entries.

        Delegates to the internal BlobStore which performs:
        - Orphaned blobs: files in cache_dir with no metadata entry
        - Dangling metadata: entries pointing to missing blob files
        - Size mismatches: metadata file_size != actual file size on disk
        - Hash mismatches: metadata file_hash != actual file hash (if verify_hashes=True)

        Args:
            repair: If True, delete orphaned blobs and remove dangling metadata entries.
            verify_hashes: If True, also verify file hashes (slower but catches corruption).

        Returns:
            IntegrityReport with orphaned_blobs, dangling_entries, size_mismatches,
            hash_mismatches (if verify_hashes), repaired (if repair).
            Supports dict-style access for backward compatibility.
        """
        return self._blob_store.verify_integrity(
            repair=repair, verify_hashes=verify_hashes
        )
