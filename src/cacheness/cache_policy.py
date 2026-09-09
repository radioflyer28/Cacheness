"""Immutable result contracts for the UnifiedCache policy layer."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheBlobLifecycleTimeoutError,
    CacheBlobRecoverableCleanupError,
    CacheBlobStoreClosedError,
)


class CacheOutcome(str, Enum):
    """The finite public outcome vocabulary for one cache lookup."""

    HIT = "hit"
    ABSENT = "absent"
    EXPIRED = "expired"
    CORRUPT = "corrupt"
    CONFLICT = "conflict"
    BACKEND_ERROR = "backend_error"


@dataclass(frozen=True)
class CacheLookupResult:
    """One policy interpretation of a single BlobStore entry observation.

    ``value`` intentionally has no bearing on whether the result is a hit: a
    stored ``None`` is represented as ``CacheOutcome.HIT`` with ``value=None``.
    ``cause`` preserves a typed storage exception when a later policy outcome
    needs to expose one.
    """

    outcome: CacheOutcome
    value: Any = None
    cause: BaseException | None = None
    removal: "CacheRemovalReport | None" = None


@dataclass(frozen=True)
class CacheRemovalFailure:
    """One bounded policy removal failure without payload disclosure."""

    key: str
    cause: BaseException

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("CacheRemovalFailure key must be a non-empty string")


@dataclass(frozen=True)
class CacheRemovalReport:
    """Immutable truthful accounting for one bounded removal operation.

    The report observes BlobStore lifecycle outcomes.  It is not itself a
    catalog, cursor authority, or permission to mutate storage.
    """

    attempted: int = 0
    removed: int = 0
    conflicted: int = 0
    failed: int = 0
    complete: bool = True
    continuation: str | None = None
    failures: tuple[CacheRemovalFailure, ...] = ()

    def __post_init__(self) -> None:
        counts = (self.attempted, self.removed, self.conflicted, self.failed)
        if any(not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in counts):
            raise ValueError("CacheRemovalReport counts must be non-negative integers")
        if self.attempted != self.removed + self.conflicted + self.failed:
            raise ValueError("CacheRemovalReport outcomes must account for every attempt")
        if not isinstance(self.complete, bool):
            raise ValueError("CacheRemovalReport completion must be a boolean")
        if self.complete and self.continuation is not None:
            raise ValueError("Completed removals cannot carry a continuation")
        if not self.complete and (
            not isinstance(self.continuation, str) or not self.continuation
        ):
            raise ValueError("Incomplete removals require an opaque continuation")
        if not isinstance(self.failures, tuple) or len(self.failures) != self.failed:
            raise ValueError("CacheRemovalReport failures must match the failed count")
        if any(not isinstance(failure, CacheRemovalFailure) for failure in self.failures):
            raise ValueError("CacheRemovalReport failures must be native values")

    @property
    def retryable(self) -> int:
        """Return the retryable exact-generation conflicts in this operation."""

        return self.conflicted


@dataclass(frozen=True)
class _CacheRemovalCandidate:
    """An authenticated key and exact expectation selected by policy."""

    key: str
    expectation: Any

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("Removal candidate key must be a non-empty string")
        if self.expectation is None:
            raise ValueError("Removal candidates require an exact expectation")


def execute_exact_removals(
    candidates: Iterable[_CacheRemovalCandidate],
    *,
    delete: Callable[[_CacheRemovalCandidate], bool],
    complete: bool = True,
    continuation: str | None = None,
) -> CacheRemovalReport:
    """Run bounded selected exact deletes without becoming a storage authority."""

    attempted = removed = conflicted = failed = 0
    failures: list[CacheRemovalFailure] = []
    for candidate in candidates:
        attempted += 1
        try:
            if delete(candidate):
                removed += 1
            else:
                conflicted += 1
        except (CacheBlobLifecycleConflictError, CacheBlobLifecycleTimeoutError):
            conflicted += 1
        except (
            CacheBlobBackendError,
            CacheBlobRecoverableCleanupError,
            CacheBlobStoreClosedError,
        ) as error:
            failed += 1
            failures.append(CacheRemovalFailure(candidate.key, error))
    return CacheRemovalReport(
        attempted=attempted,
        removed=removed,
        conflicted=conflicted,
        failed=failed,
        complete=complete,
        continuation=continuation,
        failures=tuple(failures),
    )


@dataclass(frozen=True)
class CacheStatistics:
    """An immutable, derived snapshot of lookup outcomes.

    These counts are cache-policy observations only. They never establish
    BlobStore membership, select a generation, or authorize lifecycle work.
    """

    hit: int = 0
    absent: int = 0
    expired: int = 0
    corrupt: int = 0
    conflict: int = 0
    backend_error: int = 0

    def __post_init__(self) -> None:
        """Reject invalid count values before exposing a statistics snapshot."""

        for count in (
            self.hit,
            self.absent,
            self.expired,
            self.corrupt,
            self.conflict,
            self.backend_error,
        ):
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError("CacheStatistics counts must be non-negative integers")

    @property
    def lookups(self) -> int:
        """Return the number of classified lookup results in this snapshot."""

        return sum(
            (
                self.hit,
                self.absent,
                self.expired,
                self.corrupt,
                self.conflict,
                self.backend_error,
            )
        )

    @property
    def total_lookups(self) -> int:
        """Return ``lookups`` under an explicit aggregate name."""

        return self.lookups

    @property
    def misses(self) -> int:
        """Return every non-hit outcome without reclassifying stored values."""

        return self.lookups - self.hit

    @property
    def hit_rate(self) -> float:
        """Return the derived hit rate, with a defined empty-snapshot value."""

        return self.hit / self.lookups if self.lookups else 0.0


class _CacheOutcomeRecorder:
    """Best-effort, non-authoritative lookup outcome observer."""

    def __init__(self) -> None:
        self._counts = {outcome: 0 for outcome in CacheOutcome}

    def record(self, outcome: CacheOutcome) -> None:
        """Record one already-classified outcome without touching storage."""

        self._counts[outcome] += 1

    def snapshot(self) -> CacheStatistics:
        """Copy current counts into a frozen public value object."""

        return CacheStatistics(
            hit=self._counts[CacheOutcome.HIT],
            absent=self._counts[CacheOutcome.ABSENT],
            expired=self._counts[CacheOutcome.EXPIRED],
            corrupt=self._counts[CacheOutcome.CORRUPT],
            conflict=self._counts[CacheOutcome.CONFLICT],
            backend_error=self._counts[CacheOutcome.BACKEND_ERROR],
        )
