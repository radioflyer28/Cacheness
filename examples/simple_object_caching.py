"""Canonical explicit object-cache lifecycle.

Run this example directly. It uses a fresh in-process memory topology, so it
has no service credentials, network traffic, or shared application cache.
"""

from __future__ import annotations

from tempfile import TemporaryDirectory

from cacheness import (
    CacheConfig,
    CacheOutcome,
    CachePolicyConfig,
    StoreTopology,
    UnifiedCache,
)
from cacheness.config import CacheStorageConfig
from cacheness.storage import BackendRef


def memory_topology() -> StoreTopology:
    """Return the explicit one-process topology used by this example."""

    return StoreTopology(
        payload=BackendRef(name="memory"),
        authority=BackendRef(name="memory"),
    )


def main() -> None:
    """Store, inspect, remove, and close one policy-owned cache."""

    with TemporaryDirectory(prefix="cacheness-object-example-") as directory:
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
            # Cache-owned topology: initialize before sharing this cache instance.
            cache.initialize()

            put_result = cache.put(
                {"name": "Ada", "roles": ["maintainer"]}, request_id="profile"
            )
            print(f"PUT_COMMITTED={put_result.receipt.key}")
            print(f"PUT_MAINTENANCE_COMPLETE={put_result.maintenance.complete}")

            lookup = cache.lookup(cache_key=put_result.receipt.key)
            assert lookup.outcome is CacheOutcome.HIT
            assert lookup.value["name"] == "Ada"
            print(f"LOOKUP_OUTCOME={lookup.outcome.value}")

            statistics = cache.statistics()
            print(f"STATISTICS_HITS={statistics.hit}")
            print(f"STATISTICS_LOOKUPS={statistics.lookups}")

            removal = cache.invalidate(cache_key=put_result.receipt.key)
            print(
                "INVALIDATION="
                f"attempted:{removal.attempted},removed:{removal.removed},"
                f"complete:{removal.complete}"
            )

            maintenance = cache.maintain_size()
            print(
                "MAINTENANCE="
                f"complete:{maintenance.complete},retryable:{maintenance.retryable}"
            )
        finally:
            # The cache owns the BlobStore it constructed from StoreTopology.
            cache.close()

    print("CANONICAL_OBJECT_CACHE_EXAMPLE_OK")


if __name__ == "__main__":
    main()
