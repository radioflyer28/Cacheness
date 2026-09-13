"""Authenticated entry receipts, scoped snapshots, and cache failure categories."""

from enum import Enum
from collections.abc import Mapping
from dataclasses import dataclass, replace
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
class BlobReceipt:
    """The immutable semantic result for a committed BlobStore generation.

    A receipt captures the exact generation, conditional expectation, catalog
    revision, and any derived projection outcomes. It neither authorizes a
    lifecycle transition nor aliases the development-era entry-info shape.
    """

    operation_id: str
    key: str
    generation: str
    locator: str
    expectation: EntryExpectation
    catalog_revision: int
    projections: Mapping[str, Any]

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.operation_id, self.key, self.generation, self.locator)
        ):
            raise ValueError("BlobReceipt identity fields must be non-empty strings")
        if (
            not isinstance(self.catalog_revision, int)
            or isinstance(self.catalog_revision, bool)
            or self.catalog_revision < 0
        ):
            raise ValueError("BlobReceipt catalog revision must be a non-negative integer")
        if not isinstance(self.projections, Mapping):
            raise ValueError("BlobReceipt projection outcomes must be a mapping")
        if any(not isinstance(name, str) or not name for name in self.projections):
            raise ValueError("BlobReceipt projection names must be non-empty strings")
        object.__setattr__(self, "projections", _freeze_metadata(self.projections))

    @property
    def projection_outcomes(self) -> Mapping[str, Any]:
        """Expose immutable named derived outcomes without changing authority state."""
        return self.projections

    def with_projection_outcome(self, name: str, outcome: Any) -> "BlobReceipt":
        """Return a new receipt that attributes one post-commit derived outcome."""
        if not isinstance(name, str) or not name:
            raise ValueError("BlobReceipt projection name must be a non-empty string")
        outcomes = dict(self.projections)
        outcomes[name] = outcome
        return replace(self, projections=outcomes)

    def with_projection_outcomes(self, outcomes: Mapping[str, Any]) -> "BlobReceipt":
        """Return a new receipt with the supplied named derived outcomes."""
        if not isinstance(outcomes, Mapping):
            raise ValueError("BlobReceipt projection outcomes must be a mapping")
        merged = dict(self.projections)
        merged.update(outcomes)
        return replace(self, projections=merged)


class BlobEntry:
    """An authenticated entry snapshot, usable only inside ``open_entry``.

    A present entry whose ``read()`` returns ``None`` is distinct from absence.
    The reader is installed only for a live ``open_entry`` context; inspection
    calls return the same current contract without a payload reader.
    """

    def __init__(
        self,
        key: str,
        generation: str,
        locator: str,
        expectation: EntryExpectation,
        metadata: Mapping[str, Any],
        reader: Callable[[], Any] | None = None,
    ) -> None:
        self.key = key
        self.generation = generation
        self.locator = locator
        self.expectation = expectation
        self.metadata = _freeze_metadata(metadata)
        self._reader = reader

    def read(self) -> Any:
        if self._reader is None:
            raise RuntimeError("Blob entry snapshot is closed")
        return self._reader()

    def _release(self) -> None:
        self._reader = None


class PayloadTransportComparisonStatus(str, Enum):
    """Closed outcomes for one report-only transport observation comparison."""

    MATCH = "match"
    MISMATCH = "mismatch"
    UNAVAILABLE = "unavailable"
    ABSENT = "absent"


@dataclass(frozen=True)
class PayloadTransportComparison:
    """Read-only corroboration for one committed immutable generation.

    A ``MATCH`` compares persisted opaque transport evidence with one exact
    participant observation.  It never verifies payload bytes and must not be
    treated as canonical SHA-256 integrity verification.
    """

    key: str
    generation: str
    locator: str
    status: PayloadTransportComparisonStatus
    canonical_payload_integrity_verified: bool = False

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.key, self.generation, self.locator)
        ):
            raise ValueError("Payload transport comparison identity fields must be non-empty strings")
        if not isinstance(self.status, PayloadTransportComparisonStatus):
            raise ValueError("Payload transport comparison status must be typed")
        if self.canonical_payload_integrity_verified is not False:
            raise ValueError(
                "Payload transport comparison cannot claim canonical payload integrity"
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
