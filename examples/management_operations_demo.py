#!/usr/bin/env python3
"""
Cache Management Operations
============================

Inspect, update, and bulk-manage cache entries without loading data.

Operations:  get_metadata · update_data · touch · delete_where
             get_batch · delete_batch · touch_batch
Dunder:      len(cache) · key in cache · del cache[key] · iter(cache)

Usage:
    uv run python examples/management_operations_demo.py
"""

import shutil
import tempfile

import numpy as np

from cacheness import cacheness, CacheConfig


def main():
    tmp = tempfile.mkdtemp(prefix="cacheness_mgmt_demo_")

    try:
        config = CacheConfig(cache_dir=tmp, metadata_backend="sqlite_memory")
        cache = cacheness(config)

        # ── 1. Put & get_metadata ───────────────────────────────────────
        print("1. get_metadata() — inspect without loading data")
        data = np.random.rand(1000, 50)
        cache.put(data, experiment="matrix", run=1)

        meta = cache.get_metadata(experiment="matrix", run=1)
        if meta:
            print(f"   cache_key : {meta['cache_key'][:16]}…")
            print(f"   data_type : {meta.get('data_type')}")
            print(f"   file_size : {meta.get('file_size', 0):,} bytes")

        # ── 2. update_data ──────────────────────────────────────────────
        print("\n2. update_data() — replace data, keep cache key")
        cache.put({"v": 1, "score": 0.72}, model="lr", stage="eval")
        cache.update_data({"v": 2, "score": 0.89}, model="lr", stage="eval")
        print(f"   loaded: {cache.get(model='lr', stage='eval')}")

        # ── 3. touch ────────────────────────────────────────────────────
        print("\n3. touch() — refresh TTL without reloading data")
        cache.put({"status": "running"}, job="train")
        ok = cache.touch(job="train")
        print(f"   touch succeeded: {ok}")

        # ── 4. Dunder methods ───────────────────────────────────────────
        print("\n4. Dunder methods")
        print(f"   len(cache)   = {len(cache)}")

        # __contains__
        meta = cache.get_metadata(job="train")
        key = meta["cache_key"] if meta else "missing"
        print(f"   key in cache = {key in cache}")

        # __iter__
        keys = list(cache)
        print(f"   list(cache)  = {len(keys)} keys")

        # ── 5. Bulk delete ──────────────────────────────────────────────
        print("\n5. delete_where() — bulk delete by filter")
        for i in range(5):
            cache.put(np.random.rand(10), project="cleanup", trial=i)

        before = len(cache)
        deleted = cache.delete_where(lambda e: e.get("data_type") == "array")
        print(f"   {before} entries → deleted {deleted} → {len(cache)} remaining")

        # ── 6. Batch operations ─────────────────────────────────────────
        print("\n6. Batch ops: get_batch / delete_batch / touch_batch")
        for sym in ("AAPL", "GOOG", "MSFT"):
            cache.put({"price": 100 + hash(sym) % 50}, symbol=sym, src="demo")

        results = cache.get_batch(
            [
                {"symbol": "AAPL", "src": "demo"},
                {"symbol": "GOOG", "src": "demo"},
                {"symbol": "NOPE", "src": "demo"},
            ]
        )
        found = sum(1 for v in results.values() if v is not None)
        print(f"   get_batch: {found}/{len(results)} found")

        touched = cache.touch_batch(data_type="object")
        print(f"   touch_batch: refreshed {touched}")

        removed = cache.delete_batch(
            [{"symbol": "AAPL", "src": "demo"}, {"symbol": "MSFT", "src": "demo"}]
        )
        print(f"   delete_batch: removed {removed}")

    finally:
        cache.close()
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nDone.")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
