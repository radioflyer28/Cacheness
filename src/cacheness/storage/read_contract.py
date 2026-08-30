"""Pure classification of direct BlobStore failures for future cache policy."""

from enum import Enum

from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobIntegrityError,
    CacheBlobLifecycleConflictError,
    CacheBlobManifestUnsupportedVersionError,
    CacheBlobMigrationRequiredError,
    CacheBlobPayloadUnsupportedVersionError,
    CacheManifestUnsupportedVersionError,
)


class CacheReadFailureCategory(str, Enum):
    """Closed direct-read failure categories available to future cache policy."""

    INTEGRITY = "integrity"
    MANIFEST_UNSUPPORTED_VERSION = "manifest_unsupported_version"
    PAYLOAD_UNSUPPORTED_VERSION = "payload_unsupported_version"
    UNSUPPORTED_VERSION = "unsupported_version"
    LIFECYCLE_CONFLICT = "lifecycle_conflict"
    BACKEND_FAILURE = "backend_failure"
    MIGRATION_REQUIRED = "migration_required"
    UNCLASSIFIED = "unclassified"


def classify_cache_read_failure(
    error: BaseException | None,
) -> CacheReadFailureCategory | None:
    """Classify a direct BlobStore failure without applying cache policy.

    ``None`` remains a compatible direct-storage absence. The classifier only
    reads the public error type, so callers may separately decide whether an
    integrity result should become a cache miss in a later policy layer.
    """
    if error is None:
        return None
    if isinstance(error, CacheBlobIntegrityError):
        return CacheReadFailureCategory.INTEGRITY
    if isinstance(error, CacheBlobManifestUnsupportedVersionError):
        return CacheReadFailureCategory.MANIFEST_UNSUPPORTED_VERSION
    if isinstance(error, CacheBlobPayloadUnsupportedVersionError):
        return CacheReadFailureCategory.PAYLOAD_UNSUPPORTED_VERSION
    if isinstance(error, CacheManifestUnsupportedVersionError):
        return CacheReadFailureCategory.UNSUPPORTED_VERSION
    if isinstance(error, CacheBlobLifecycleConflictError):
        return CacheReadFailureCategory.LIFECYCLE_CONFLICT
    if isinstance(error, CacheBlobBackendError):
        return CacheReadFailureCategory.BACKEND_FAILURE
    if isinstance(error, CacheBlobMigrationRequiredError):
        return CacheReadFailureCategory.MIGRATION_REQUIRED
    return CacheReadFailureCategory.UNCLASSIFIED
