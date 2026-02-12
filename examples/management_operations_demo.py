#!/usr/bin/env python3
"""
Cache Management Operations Demo
=================================

Demonstrates Phase 3 management operations for inspecting, updating,
and bulk-managing cache entries without full retrieval.

Operations shown:
    - get_metadata()    — inspect entries without loading data
    - update_data()     — replace data in-place (cache_key unchanged)
    - touch()           — extend TTL without reloading
    - delete_where()    — filter-based bulk delete
    - delete_matching() — keyword-based bulk delete
    - get_batch()       — retrieve multiple entries at once
    - delete_batch()    — remove multiple entries at once
    - touch_batch()     — refresh TTL for matching entries

Usage:
    python management_operations_demo.py
"""

import numpy as np
import tempfile
import shutil

from cacheness import cacheness, CacheConfig


def main():
    """Run all management operations demos."""
    # Use a temp directory so the demo is self-contained
    tmp = tempfile.mkdtemp(prefix="cacheness_mgmt_demo_")
    print("=" * 60)
    print("  Cache Management Operations Demo")
    print("=" * 60)

    try:
        config = CacheConfig(cache_dir=tmp, metadata_backend="sqlite_memory")
        cache = cacheness(config)

        demo_get_metadata(cache)
        demo_update_data(cache)
        demo_touch(cache)
        demo_bulk_delete(cache)
        demo_batch_operations(cache)
    finally:
        cache.close()
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n✅ All demos completed successfully!")


# ── 1. get_metadata() ───────────────────────────────────────────────────


def demo_get_metadata(cache):
    print("\n" + "-" * 50)
    print("1️⃣  get_metadata() — inspect without loading data")
    print("-" * 50)

    # Store a moderately large array
    data = np.random.rand(1000, 50)
    cache.put(data, experiment="big_matrix", run=1)

    # Inspect metadata without deserializing the blob
    meta = cache.get_metadata(experiment="big_matrix", run=1)
    if meta:
        print(f"  cache_key : {meta['cache_key'][:16]}…")
        print(f"  data_type : {meta.get('data_type')}")
        print(f"  file_size : {meta.get('file_size', 0):,} bytes")
        print(f"  created_at: {meta.get('created_at')}")
    else:
        print("  (entry not found)")


# ── 2. update_data() ────────────────────────────────────────────────────


def demo_update_data(cache):
    print("\n" + "-" * 50)
    print("2️⃣  update_data() — replace data, keep cache_key")
    print("-" * 50)

    # Store initial version
    v1 = {"version": 1, "score": 0.72}
    cache.put(v1, model="logistic", stage="eval")

    # Retrieve cache_key for comparison
    key_before = cache.get_metadata(model="logistic", stage="eval")["cache_key"]

    # Update the stored data in-place
    v2 = {"version": 2, "score": 0.89}
    ok = cache.update_data(v2, model="logistic", stage="eval")
    print(f"  update succeeded: {ok}")

    # Confirm cache_key is unchanged
    key_after = cache.get_metadata(model="logistic", stage="eval")["cache_key"]
    print(f"  cache_key unchanged: {key_before == key_after}")

    # Read back the updated data
    loaded = cache.get(model="logistic", stage="eval")
    print(f"  loaded data: {loaded}")


# ── 3. touch() ──────────────────────────────────────────────────────────


def demo_touch(cache):
    print("\n" + "-" * 50)
    print("3️⃣  touch() — extend TTL without reloading data")
    print("-" * 50)

    cache.put({"status": "running"}, job="long_train")

    meta_before = cache.get_metadata(job="long_train")
    ts_before = meta_before["created_at"]

    # Simulate a short delay (timestamps are ISO-formatted)
    import time

    time.sleep(0.05)

    ok = cache.touch(job="long_train")
    print(f"  touch succeeded: {ok}")

    meta_after = cache.get_metadata(job="long_train")
    ts_after = meta_after["created_at"]
    print(f"  timestamp before: {ts_before}")
    print(f"  timestamp after : {ts_after}")
    print(f"  timestamp moved forward: {ts_after >= ts_before}")


# ── 4. Bulk delete ──────────────────────────────────────────────────────


def demo_bulk_delete(cache):
    print("\n" + "-" * 50)
    print("4️⃣  delete_where() & delete_matching() — bulk delete")
    print("-" * 50)

    # Populate several entries
    for i in range(5):
        cache.put(
            np.random.rand(10),
            project="cleanup_demo",
            trial=i,
        )

    entries_before = len(cache.list_entries())
    print(f"  entries before delete: {entries_before}")

    # delete_where: remove entries whose data_type match a filter
    deleted = cache.delete_where(lambda e: e.get("data_type") == "array")
    print(f"  delete_where(numpy_array): removed {deleted}")

    entries_after = len(cache.list_entries())
    print(f"  entries after delete : {entries_after}")


# ── 5. Batch operations ────────────────────────────────────────────────


def demo_batch_operations(cache):
    print("\n" + "-" * 50)
    print("5️⃣  get_batch / delete_batch / touch_batch")
    print("-" * 50)

    # Store several entries
    for sym in ("AAPL", "GOOG", "MSFT"):
        cache.put({"price": 100 + hash(sym) % 50}, symbol=sym, source="demo")

    # get_batch — fetch multiple at once
    results = cache.get_batch(
        [
            {"symbol": "AAPL", "source": "demo"},
            {"symbol": "GOOG", "source": "demo"},
            {"symbol": "NOPE", "source": "demo"},  # doesn't exist
        ]
    )
    found = sum(1 for v in results.values() if v is not None)
    print(f"  get_batch: {found}/{len(results)} entries found")

    # touch_batch — refresh TTL for matching entries
    # The data_type field is available at the top level of list_entries()
    touched = cache.touch_batch(data_type="object")
    print(f"  touch_batch(data_type=object): refreshed {touched} entries")

    # delete_batch — remove specific entries
    deleted = cache.delete_batch(
        [
            {"symbol": "AAPL", "source": "demo"},
            {"symbol": "MSFT", "source": "demo"},
        ]
    )
    print(f"  delete_batch: removed {deleted} entries")

    remaining = len(cache.list_entries())
    print(f"  remaining entries: {remaining}")


if __name__ == "__main__":
    main()
