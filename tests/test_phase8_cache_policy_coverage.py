"""Phase 8 contracts for public UnifiedCache policy boundaries."""

from __future__ import annotations

import pytest

from cacheness.cache_policy import CacheOutcome
from cacheness.error_handling import CacheBlobLifecycleConflictError


def test_cache_policy_preserves_declared_blobstore_lookup_cause(tmp_path) -> None:
    """A typed lifecycle conflict remains the public lookup cause and outcome."""

    cache = _phase8_cache(tmp_path)
    failure = CacheBlobLifecycleConflictError("exact generation changed")
    try:
        with _open_entry_failure(cache, failure):
            result = cache.lookup(cache_key="phase8-conflict")

        assert result.outcome is CacheOutcome.CONFLICT
        assert result.cause is failure
    finally:
        cache.close()
