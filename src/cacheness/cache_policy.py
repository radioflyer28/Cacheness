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

