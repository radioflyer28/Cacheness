"""Authenticated entry receipts, scoped snapshots, and cache failure categories."""

from enum import Enum
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable

from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobIntegrityError,
    CacheBlobLifecycleConflictError,
    CacheBlobManifestUnsupportedVersionError,
    CacheBlobMigrationRequiredError,
    CacheBlobPayloadUnsupportedVersionError,
    CacheManifestUnsupportedVersionError,
)

from .lifecycle_authority import EntryExpectation


def _freeze_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_metadata(item) for item in value)
    return value


@dataclass(frozen=True)
class BlobEntryInfo:
    """Authenticated metadata and an opaque conditional-delete expectation."""

    key: str
    generation: str
    locator: str
    expectation: EntryExpectation
    metadata: Mapping[str, Any]
    previous_locator: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


class BlobEntry:
    """A verified private snapshot, usable only inside ``open_entry``.

    A present entry whose ``read()`` returns None is distinct from absence.
    The owning BlobStore releases resources when its context exits.
    """

    def __init__(self, info: BlobEntryInfo, reader: Callable[[], Any]) -> None:
        self.info = info
        self._reader: Callable[[], Any] | None = reader

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.info.metadata

    @property
    def expectation(self) -> EntryExpectation:
        return self.info.expectation

    def read(self) -> Any:
        if self._reader is None:
            raise RuntimeError("Blob entry snapshot is closed")
        return self._reader()

    def _release(self) -> None:
        self._reader = None

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
