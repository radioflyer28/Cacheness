"""Public direct-read taxonomy and future cache-translation contracts."""

import inspect

import pytest

from cacheness.error_handling import (
    CacheBlobIntegrityError,
    CacheIntegrityError,
    CacheManifestIntegrityError,
    CacheManifestUnsupportedVersionError,
    CacheReason,
    CacheStorageError,
)
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
from cacheness.storage.read_contract import (
    CacheReadFailureCategory,
    classify_cache_read_failure,
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


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (CacheBlobManifestMalformedError("malformed"), CacheReadFailureCategory.INTEGRITY),
        (
            CacheBlobManifestUnauthenticatedError("unauthenticated"),
            CacheReadFailureCategory.INTEGRITY,
        ),
        (CacheBlobPayloadMissingError("missing"), CacheReadFailureCategory.INTEGRITY),
        (CacheBlobPayloadTamperedError("tampered"), CacheReadFailureCategory.INTEGRITY),
        (
            CacheManifestIntegrityError("legacy direct integrity"),
            CacheReadFailureCategory.INTEGRITY,
        ),
        (
            CacheBlobManifestUnsupportedVersionError("manifest future"),
            CacheReadFailureCategory.MANIFEST_UNSUPPORTED_VERSION,
        ),
        (
            CacheBlobPayloadUnsupportedVersionError("payload future"),
            CacheReadFailureCategory.PAYLOAD_UNSUPPORTED_VERSION,
        ),
        (
            CacheManifestUnsupportedVersionError("generic future"),
            CacheReadFailureCategory.UNSUPPORTED_VERSION,
        ),
        (
            CacheBlobLifecycleConflictError("not committed"),
            CacheReadFailureCategory.LIFECYCLE_CONFLICT,
        ),
        (
            CacheBlobBackendError("repository failed"),
            CacheReadFailureCategory.BACKEND_FAILURE,
        ),
        (
            CacheBlobMigrationRequiredError("legacy conversion"),
            CacheReadFailureCategory.MIGRATION_REQUIRED,
        ),
        (
            CacheIntegrityError("unrelated integrity failure"),
            CacheReadFailureCategory.UNCLASSIFIED,
        ),
    ],
)
def test_classifier_exhaustively_preserves_direct_failure_categories(error, expected):
    """Only BlobStore evidence failures are eligible for a future miss policy."""
    assert classify_cache_read_failure(error) is expected


def test_classifier_keeps_absence_outside_the_failure_taxonomy():
    """A direct BlobStore absence remains None rather than an integrity result."""
    assert classify_cache_read_failure(None) is None


def test_classifier_is_pure_and_has_no_unified_cache_dependency():
    """The Phase 6 seam cannot construct or mutate cache policy in Phase 2."""
    error = CacheBlobPayloadTamperedError(
        "payload changed", context={"counter": 0, "metadata": {"hits": 0}}
    )
    context_before = {key: value.copy() if isinstance(value, dict) else value for key, value in error.context.items()}

    assert classify_cache_read_failure(error) is CacheReadFailureCategory.INTEGRITY
    assert error.context == context_before

    source = inspect.getsource(classify_cache_read_failure)
    assert "UnifiedCache" not in source
    assert "cacheness.core" not in source


def test_blob_integrity_base_remains_the_selected_translation_boundary():
    """A pure classifier chooses the public BlobStore integrity base, not all errors."""
    assert issubclass(CacheManifestIntegrityError, CacheBlobIntegrityError)
