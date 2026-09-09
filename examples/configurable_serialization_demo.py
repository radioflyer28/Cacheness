"""Cache one optional DataFrame through the canonical policy composition.

The cache owns the :class:`BlobStore` created from the explicit topology. It
uses only temporary in-process resources, so it needs no service credentials.
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

try:
    import pandas as pd
except ImportError:  # Optional DataFrame support is an explicit capability.
    pd = None


def memory_topology() -> StoreTopology:
    """Return the supported memory/memory topology for this process only."""

    return StoreTopology(
        payload=BackendRef(name="memory"),
        authority=BackendRef(name="memory"),
    )


def main() -> None:
    """Write typed values, distinguish a present ``None``, and close cleanly."""

    with TemporaryDirectory(prefix="cacheness-serialization-example-") as directory:
        config = CacheConfig(
            storage=CacheStorageConfig(cache_dir=directory),
            policy=CachePolicyConfig(
                max_authoritative_bytes=1_024 * 1_024,
                catalog_page_size=8,
                maintenance_work_cap=8,
            ),
        )
        cache = UnifiedCache(config, store=memory_topology())
        try:
            # The StoreTopology makes this BlobStore cache-owned.
            cache.initialize()

            none_put = cache.put(None, dataset="optional", field="comment")
            none_lookup = cache.lookup(cache_key=none_put.receipt.key)
            assert none_lookup.outcome is CacheOutcome.HIT
            assert none_lookup.value is None
            print(f"PRESENCE_NONE_OUTCOME={none_lookup.outcome.value}")

            if pd is None:
                print("DATAFRAME_CAPABILITY_UNAVAILABLE=pandas is not installed")
            else:
                frame = pd.DataFrame(
                    {"name": ["Ada", "Grace"], "commits": [42, 17]}
                )
                put_result = cache.put(frame, dataset="maintainers")
                lookup = cache.lookup(cache_key=put_result.receipt.key)
                assert lookup.outcome is CacheOutcome.HIT
                assert lookup.value.equals(frame)
                print("DATAFRAME_CAPABILITY_AVAILABLE=pandas")
                print(f"DATAFRAME_PUT_COMMITTED={put_result.receipt.key}")
                print(f"DATAFRAME_LOOKUP_OUTCOME={lookup.outcome.value}")
                print(
                    "DATAFRAME_MAINTENANCE="
                    f"complete:{put_result.maintenance.complete},"
                    f"retryable:{put_result.maintenance.retryable}"
                )

            statistics = cache.statistics()
            print(f"STATISTICS_LOOKUPS={statistics.lookups}")
            print(f"STATISTICS_HITS={statistics.hit}")
        finally:
            # Closing this facade closes only the BlobStore it owns.
            cache.close()

    print("CANONICAL_SERIALIZATION_EXAMPLE_OK")


if __name__ == "__main__":
    main()
