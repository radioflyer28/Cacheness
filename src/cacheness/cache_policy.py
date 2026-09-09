"""Immutable result contracts for the UnifiedCache policy layer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


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
