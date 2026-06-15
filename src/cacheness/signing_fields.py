"""Canonical field extraction for cache entry signatures."""

from datetime import datetime
from typing import Any, Dict, Optional, cast

from .interfaces import SignableFields


def _normalize_created_at(value: Any) -> Any:
    """Normalize created_at values for stable signature payloads."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=None).isoformat()
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
            return dt.replace(tzinfo=None).isoformat()
        except (ValueError, TypeError):
            return value
    return value


def _optional_str(value: Any) -> Optional[str]:
    """Return strings as-is while preserving None for absent optional fields."""
    if value is None:
        return None
    return str(value)


def extract_signable_fields(
    cache_key: str,
    entry_data: Dict[str, Any],
    metadata: Dict[str, Any],
) -> SignableFields:
    """
    Extract the canonical superset of fields eligible for entry signing.

    The signer decides which fields are included for each signature version.
    This helper supplies the stable field superset shared by UnifiedCache and
    BlobStore so user metadata cannot accidentally alter signed core fields.
    """
    file_size = entry_data.get("file_size", 0)
    if not isinstance(file_size, int):
        try:
            file_size = int(file_size)
        except (TypeError, ValueError):
            file_size = 0

    created_at = _normalize_created_at(entry_data.get("created_at"))
    if created_at is None:
        created_at = ""

    signable: SignableFields = {
        "cache_key": cache_key,
        "data_type": str(entry_data.get("data_type") or ""),
        "file_size": file_size,
        "created_at": str(created_at),
        "actual_path": str(metadata.get("actual_path") or ""),
        "file_hash": _optional_str(metadata.get("file_hash")),
        "object_type": _optional_str(metadata.get("object_type")),
        "storage_format": _optional_str(metadata.get("storage_format")),
        "serializer": _optional_str(metadata.get("serializer")),
        "compression_codec": _optional_str(metadata.get("compression_codec")),
    }
    return signable


def extract_legacy_blobstore_signable_fields(
    cache_key: str,
    entry_data: Dict[str, Any],
    metadata: Dict[str, Any],
) -> SignableFields:
    """
    Rebuild the pre-SEC-04 BlobStore flattened signing shape.

    Existing BlobStore entries could be signed after flattening user metadata
    over top-level entry fields. New writes must not use this shape, but reads
    keep it as an explicit compatibility verifier.
    """
    return cast(
        SignableFields,
        {
            **entry_data,
            **metadata,
            "cache_key": cache_key,
        },
    )
