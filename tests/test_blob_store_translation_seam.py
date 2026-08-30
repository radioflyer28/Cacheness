"""Public direct-read taxonomy and future cache-translation contracts."""

import pytest

from cacheness.error_handling import CacheIntegrityError, CacheReason, CacheStorageError
from cacheness.storage import (
    CacheBlobManifestMalformedError,
    CacheBlobManifestUnauthenticatedError,
    CacheBlobManifestUnsupportedVersionError,
    CacheBlobPayloadMissingError,
    CacheBlobPayloadTamperedError,
    CacheBlobPayloadUnsupportedVersionError,
    CacheBlobLifecycleConflictError,
    CacheBlobBackendError,
    CacheBlobMigrationRequiredError,
)


@pytest.mark.parametrize(
    ("error_type", "base_type", "reason"),
    [
        (
            CacheBlobManifestMalformedError,
            CacheIntegrityError,
            CacheReason.BLOB_MANIFEST_MALFORMED,
        ),
        (
            CacheBlobManifestUnauthenticatedError,
            CacheIntegrityError,
            CacheReason.BLOB_MANIFEST_UNAUTHENTICATED,
        ),
        (
            CacheBlobPayloadMissingError,
            CacheIntegrityError,
            CacheReason.BLOB_PAYLOAD_MISSING,
        ),
        (
            CacheBlobPayloadTamperedError,
            CacheIntegrityError,
            CacheReason.BLOB_PAYLOAD_TAMPERED,
        ),
        (
            CacheBlobManifestUnsupportedVersionError,
            CacheStorageError,
            CacheReason.BLOB_MANIFEST_UNSUPPORTED_VERSION,
        ),
        (
            CacheBlobPayloadUnsupportedVersionError,
            CacheStorageError,
            CacheReason.BLOB_PAYLOAD_UNSUPPORTED_VERSION,
        ),
        (
            CacheBlobLifecycleConflictError,
            CacheStorageError,
            CacheReason.BLOB_LIFECYCLE_CONFLICT,
        ),
        (
            CacheBlobBackendError,
            CacheStorageError,
            CacheReason.BLOB_BACKEND_FAILURE,
        ),
        (
            CacheBlobMigrationRequiredError,
            CacheStorageError,
            CacheReason.BLOB_MIGRATION_REQUIRED,
        ),
    ],
)
def test_blob_store_errors_have_stable_public_types_and_reasons(
    error_type, base_type, reason
):
    """Direct callers can branch on an exact public type and stable reason."""
    error = error_type("direct storage failure", context={"reason": "ignored"})

    assert isinstance(error, base_type)
    assert error.context["reason"] == reason.value


def test_storage_barrel_exports_each_direct_read_failure_type():
    """Storage users need no private codec imports to handle direct outcomes."""
    from cacheness import storage

    expected_names = {
        "CacheBlobManifestMalformedError",
        "CacheBlobManifestUnauthenticatedError",
        "CacheBlobPayloadMissingError",
        "CacheBlobPayloadTamperedError",
        "CacheBlobManifestUnsupportedVersionError",
        "CacheBlobPayloadUnsupportedVersionError",
        "CacheBlobLifecycleConflictError",
        "CacheBlobBackendError",
        "CacheBlobMigrationRequiredError",
    }

    assert expected_names.issubset(storage.__all__)
    assert all(hasattr(storage, name) for name in expected_names)
