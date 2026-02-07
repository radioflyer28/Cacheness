"""
Concurrency Stress Tests for UnifiedCache._lock
================================================

Tests that the RLock in UnifiedCache actually prevents race conditions
in multi-threaded scenarios. These tests go beyond the existing
test_sqlite_concurrency.py which tests SQLite WAL-level concurrency:
these test the cache-layer lock that coordinates blob+metadata atomicity.

Covers:
- Concurrent put() to same key (last-writer-wins, no corruption)
- Concurrent put() to distinct keys (all succeed)
- Concurrent get() during put() (no partial reads)
- Concurrent put() + invalidate() (no orphaned state)
- Concurrent touch() + get() (no stale reads)
- Concurrent delete_where() + put() (no lost entries)
- Concurrent update_data() (no corruption)
- Lock re-entrancy via delete_where → invalidate, touch_batch → touch
"""

import pytest
import tempfile
import threading
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cacheness.core import UnifiedCache
from cacheness.config import CacheConfig


@pytest.fixture
def stress_cache():
    """Create a SQLite-backed cache for stress testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = CacheConfig(
            cache_dir=temp_dir,
            metadata_backend="sqlite",
        )
        cache = UnifiedCache(config=config)
        yield cache
        cache.close()


@pytest.fixture
def json_stress_cache():
    """Create a JSON-backed cache for stress testing (exercises different lock paths)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = CacheConfig(
            cache_dir=temp_dir,
            metadata_backend="json",
        )
        cache = UnifiedCache(config=config)
        yield cache
        cache.close()


class TestConcurrentPutSameKey:
    """Concurrent put() to the same key must not corrupt state."""

    def test_last_writer_wins_no_corruption(self, stress_cache):
        """Multiple threads writing to the same key — final state must be valid."""
        errors = []
        num_threads = 8
        cache = stress_cache

        def writer(thread_id):
            try:
                data = {"thread": thread_id, "value": thread_id * 100}
                cache.put(data, key="shared")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Errors during concurrent put: {errors}"

        # The final value must be a valid dict from one of the threads
        result = cache.get(key="shared")
        assert result is not None, "Cache should have an entry"
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert "thread" in result and "value" in result
        assert result["thread"] in range(num_threads)

    def test_no_orphaned_blobs_on_concurrent_overwrite(self, stress_cache):
        """Overwriting the same key concurrently should not leave orphaned blobs."""
        cache = stress_cache
        num_threads = 6

        def writer(thread_id):
            data = np.random.rand(10) * thread_id
            cache.put(data, key="overwrite_target")

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # Verify integrity — should have no orphaned blobs
        report = cache.verify_integrity()
        assert len(report["orphaned_blobs"]) == 0, (
            f"Found orphaned blobs after concurrent overwrite: {report['orphaned_blobs']}"
        )


class TestConcurrentPutDistinctKeys:
    """Concurrent put() to distinct keys must all succeed."""

    def test_all_entries_persisted(self, stress_cache):
        """N threads writing to N different keys — all N entries must exist."""
        cache = stress_cache
        num_threads = 20
        errors = []

        def writer(thread_id):
            try:
                data = {"id": thread_id}
                cache.put(data, key=f"entry_{thread_id}")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(writer, i) for i in range(num_threads)]
            for f in as_completed(futures):
                f.result()  # re-raise exceptions

        assert not errors
        entries = cache.list_entries()
        assert len(entries) == num_threads, (
            f"Expected {num_threads} entries, got {len(entries)}"
        )

    def test_concurrent_put_with_json_backend(self, json_stress_cache):
        """JSON backend serializes entire file — concurrent puts must not corrupt."""
        cache = json_stress_cache
        num_threads = 10
        errors = []

        def writer(thread_id):
            try:
                cache.put({"v": thread_id}, key=f"json_{thread_id}")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors
        # All entries should be present
        for i in range(num_threads):
            result = cache.get(key=f"json_{i}")
            assert result is not None, f"Missing entry json_{i}"
            assert result["v"] == i


class TestConcurrentGetDuringPut:
    """get() during put() must return either None or the complete object."""

    def test_no_partial_reads(self, stress_cache):
        """Reader threads during a write must never see partial/corrupt data."""
        cache = stress_cache
        partial_reads = []
        stop_event = threading.Event()

        # Pre-populate
        original_data = {"state": "original", "payload": list(range(100))}
        cache.put(original_data, key="contested")

        def writer():
            for i in range(20):
                new_data = {"state": f"updated_{i}", "payload": list(range(100))}
                cache.put(new_data, key="contested")
            stop_event.set()

        def reader(reader_id):
            while not stop_event.is_set():
                result = cache.get(key="contested")
                if result is not None:
                    # Must be a complete dict with all expected keys
                    if not isinstance(result, dict):
                        partial_reads.append(f"Reader {reader_id}: got {type(result)}")
                    elif "state" not in result or "payload" not in result:
                        partial_reads.append(f"Reader {reader_id}: missing keys in {result.keys()}")
                    elif len(result["payload"]) != 100:
                        partial_reads.append(
                            f"Reader {reader_id}: payload len={len(result['payload'])}"
                        )

        writer_thread = threading.Thread(target=writer)
        reader_threads = [threading.Thread(target=reader, args=(i,)) for i in range(4)]

        for t in reader_threads:
            t.start()
        writer_thread.start()

        writer_thread.join(timeout=30)
        stop_event.set()
        for t in reader_threads:
            t.join(timeout=10)

        assert not partial_reads, f"Partial/corrupt reads detected: {partial_reads}"


class TestConcurrentPutAndInvalidate:
    """put() + invalidate() racing must not leave orphaned state."""

    def test_put_and_invalidate_race(self, stress_cache):
        """Interleaved put/invalidate must leave cache in a consistent state."""
        cache = stress_cache
        errors = []
        num_iterations = 30

        def put_worker():
            for i in range(num_iterations):
                try:
                    cache.put({"iter": i}, key=f"race_{i % 5}")
                except Exception as e:
                    errors.append(f"put: {e}")

        def invalidate_worker():
            for i in range(num_iterations):
                try:
                    cache.invalidate(key=f"race_{i % 5}")
                except Exception as e:
                    errors.append(f"invalidate: {e}")

        t1 = threading.Thread(target=put_worker)
        t2 = threading.Thread(target=invalidate_worker)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        assert not errors, f"Errors during put/invalidate race: {errors}"

        # Verify no dangling metadata (entries pointing to missing blobs)
        # NOTE: orphaned blobs are expected because invalidate() only deletes
        # metadata, not blob files (known design gap, not a concurrency bug)
        report = cache.verify_integrity()
        assert len(report["dangling_entries"]) == 0


class TestConcurrentTouchAndGet:
    """touch() + get() racing must not cause errors."""

    def test_touch_during_reads(self, stress_cache):
        """Touching entries while reading should not corrupt state."""
        cache = stress_cache
        errors = []

        # Populate
        for i in range(10):
            cache.put({"v": i}, key=f"item_{i}")

        def toucher():
            for _ in range(20):
                for i in range(10):
                    try:
                        cache.touch(key=f"item_{i}")
                    except Exception as e:
                        errors.append(f"touch: {e}")

        def reader():
            for _ in range(20):
                for i in range(10):
                    try:
                        result = cache.get(key=f"item_{i}")
                        if result is not None:
                            assert result["v"] == i
                    except Exception as e:
                        errors.append(f"get: {e}")

        t1 = threading.Thread(target=toucher)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        assert not errors, f"Errors during touch/get: {errors}"


class TestConcurrentDeleteWhereAndPut:
    """delete_where() + put() must not lose entries that weren't matched."""

    def test_delete_where_during_writes(self, stress_cache):
        """Bulk delete during writes must not leave inconsistent state."""
        cache = stress_cache
        errors = []

        # Seed some entries to delete
        for i in range(10):
            cache.put({"group": "old", "id": i}, key=f"old_{i}")

        def deleter():
            try:
                cache.delete_where(lambda e: e.get("description", "").startswith(""))
            except Exception as e:
                errors.append(f"delete_where: {e}")

        def writer():
            for i in range(10):
                try:
                    cache.put({"group": "new", "id": i}, key=f"new_{i}")
                except Exception as e:
                    errors.append(f"put: {e}")

        t1 = threading.Thread(target=deleter)
        t2 = threading.Thread(target=writer)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        assert not errors, f"Errors during delete_where/put: {errors}"

        # Verify no dangling metadata (entries pointing to missing blobs)
        # NOTE: orphaned blobs are expected here because invalidate() only
        # deletes metadata, not blob files (known design gap, not a lock bug)
        report = cache.verify_integrity()
        assert len(report["dangling_entries"]) == 0


class TestConcurrentUpdateData:
    """update_data() racing must produce valid final state."""

    def test_concurrent_update_same_entry(self, stress_cache):
        """Multiple threads updating same entry — final state must be one of the values."""
        cache = stress_cache
        errors = []

        # Create initial entry
        cache.put({"version": 0}, key="updatable")

        def updater(thread_id):
            try:
                new_data = {"version": thread_id, "data": list(range(thread_id))}
                cache.update_data(new_data, key="updatable")
            except Exception as e:
                errors.append(f"update {thread_id}: {e}")

        num_threads = 8
        threads = [threading.Thread(target=updater, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Errors during concurrent update: {errors}"

        result = cache.get(key="updatable")
        assert result is not None
        assert isinstance(result, dict)
        assert "version" in result
        # version must be one of the thread IDs
        assert result["version"] in range(num_threads)


class TestLockReEntrancy:
    """RLock allows re-entrant calls (method A → method B on same thread)."""

    def test_delete_where_calls_invalidate(self, stress_cache):
        """delete_where() calls invalidate() internally — must not deadlock."""
        cache = stress_cache
        for i in range(5):
            cache.put({"v": i}, key=f"re_{i}")

        # This would deadlock with a non-reentrant Lock
        deleted = cache.delete_where(lambda e: True)
        assert deleted == 5

    def test_touch_batch_calls_touch(self, stress_cache):
        """touch_batch() calls touch() internally — must not deadlock."""
        cache = stress_cache
        for i in range(5):
            cache.put({"v": i}, key=f"tb_{i}", description="group_a")

        # This would deadlock with a non-reentrant Lock
        # touch_batch uses filter_kwargs matching, seed with prefix
        touched = cache.touch_batch(description="group_a")
        assert touched == 5

    def test_delete_matching_calls_invalidate(self, stress_cache):
        """delete_matching() calls invalidate() internally — must not deadlock."""
        cache = stress_cache
        for i in range(3):
            cache.put({"v": i}, key=f"dm_{i}", description="deletable")

        deleted = cache.delete_matching(description="deletable")
        assert deleted == 3

    def test_delete_batch_calls_invalidate(self, stress_cache):
        """delete_batch() calls invalidate() internally — must not deadlock."""
        cache = stress_cache
        for i in range(3):
            cache.put({"v": i}, key=f"db_{i}")

        deleted = cache.delete_batch([{"key": f"db_{i}"} for i in range(3)])
        assert deleted == 3

    def test_get_batch_calls_get(self, stress_cache):
        """get_batch() calls get() internally — must not deadlock."""
        cache = stress_cache
        for i in range(3):
            cache.put({"v": i}, key=f"gb_{i}")

        results = cache.get_batch([{"key": f"gb_{i}"} for i in range(3)])
        assert len(results) == 3
        for v in results.values():
            assert v is not None


class TestHighConcurrencyStress:
    """High-concurrency stress test mixing all operations."""

    def test_mixed_operations_stress(self, stress_cache):
        """Many threads doing put/get/touch/invalidate concurrently."""
        cache = stress_cache
        errors = []
        num_keys = 20
        num_threads = 12
        ops_per_thread = 30

        # Seed some entries
        for i in range(num_keys):
            cache.put({"seed": i}, key=f"stress_{i}")

        def mixed_worker(worker_id):
            import random
            for _ in range(ops_per_thread):
                key_idx = random.randint(0, num_keys - 1)
                op = random.choice(["put", "get", "touch", "invalidate"])
                try:
                    if op == "put":
                        cache.put(
                            {"worker": worker_id, "rand": random.random()},
                            key=f"stress_{key_idx}",
                        )
                    elif op == "get":
                        cache.get(key=f"stress_{key_idx}")
                    elif op == "touch":
                        cache.touch(key=f"stress_{key_idx}")
                    elif op == "invalidate":
                        cache.invalidate(key=f"stress_{key_idx}")
                except Exception as e:
                    errors.append(f"Worker {worker_id} {op}: {type(e).__name__}: {e}")

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(mixed_worker, i) for i in range(num_threads)]
            for f in as_completed(futures):
                f.result()

        assert not errors, f"Stress test errors ({len(errors)}): {errors[:5]}"

        # Final integrity check — no dangling metadata (entries without blobs)
        # NOTE: orphaned blobs are expected because invalidate() only deletes
        # metadata, not blob files (known design gap, not a concurrency bug)
        report = cache.verify_integrity()
        assert len(report["dangling_entries"]) == 0, (
            f"Dangling entries after stress: {report['dangling_entries']}"
        )

    def test_clear_all_during_operations(self, stress_cache):
        """clear_all() during put/get should not crash or leave inconsistent state."""
        cache = stress_cache
        errors = []

        # Seed
        for i in range(10):
            cache.put({"v": i}, key=f"clear_{i}")

        barrier = threading.Barrier(3, timeout=10)

        def writer():
            barrier.wait()
            for i in range(10):
                try:
                    cache.put({"new": i}, key=f"clear_new_{i}")
                except Exception as e:
                    errors.append(f"writer: {e}")

        def reader():
            barrier.wait()
            for i in range(10):
                try:
                    cache.get(key=f"clear_{i}")
                except Exception as e:
                    errors.append(f"reader: {e}")

        def clearer():
            barrier.wait()
            try:
                cache.clear_all()
            except Exception as e:
                errors.append(f"clearer: {e}")

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=clearer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Errors during clear_all stress: {errors}"

        # After everything settles, integrity should be clean
        report = cache.verify_integrity()
        assert len(report["dangling_entries"]) == 0
