#!/usr/bin/env python3
"""Layer explicit cache policy and a function decorator over one BlobStore."""

from __future__ import annotations

from tempfile import TemporaryDirectory

from cacheness import (
    CacheConfig,
    CacheOutcome,
    CachePolicyConfig,
    StoreTopology,
    UnifiedCache,
    cached,
)
from cacheness.config import CacheStorageConfig
from cacheness.storage import BackendRef


def memory_topology() -> StoreTopology:
    """Return the explicit same-process topology used by this cache instance."""

    return StoreTopology(
        payload=BackendRef(name="memory"),
        authority=BackendRef(name="memory"),
    )


def main() -> None:
    """Show typed lookup and explicit reuse through ``@cached``."""

    with TemporaryDirectory(prefix="cacheness-unified-cache-") as directory:
        config = CacheConfig(
            storage=CacheStorageConfig(cache_dir=directory),
            policy=CachePolicyConfig(
                max_authoritative_bytes=1_024,
                catalog_page_size=8,
                maintenance_work_cap=8,
            ),
        )
        cache = UnifiedCache(config, store=memory_topology())
        try:
            cache.initialize()
            written = cache.put({"status": "ready"}, request_id="status")
            lookup = cache.lookup(cache_key=written.receipt.key)
            assert lookup.outcome is CacheOutcome.HIT
            assert lookup.value == {"status": "ready"}

            calls = 0

            @cached(cache=cache)
            def calculate(value: int) -> int:
                nonlocal calls
                calls += 1
                return value * 2

            assert calculate(21) == 42
            assert calculate(21) == 42
            assert calculate.cache_last_lookup.outcome is CacheOutcome.HIT
            assert calls == 1

            cleared = cache.clear_all()
            assert cleared.complete
        finally:
            cache.close()

    print("UNIFIED_CACHE_EXAMPLE_OK")


if __name__ == "__main__":
    main()
