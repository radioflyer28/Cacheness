"""Thread safety smoke tests for concurrent put()/get() across metadata backends."""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from cacheness.core import UnifiedCache as cacheness
from cacheness.config import (
    CacheConfig,
    CacheStorageConfig,
    CacheMetadataConfig,
    CompressionConfig,
)


def _make_cache(tmp_path, backend="json"):
    """Create a cache instance with the given backend."""
    config = CacheConfig(
        storage=CacheStorageConfig(cache_dir=str(tmp_path)),
        metadata=CacheMetadataConfig(metadata_backend=backend),
        compression=CompressionConfig(use_blosc2_arrays=False),
    )
    return cacheness(config)


class TestThreadSafetyJson:
    """Thread safety smoke tests with JSON metadata backend."""

    def test_concurrent_puts(self, tmp_path):
        cache = _make_cache(tmp_path, "json")
        errors = []

        def do_put(i):
            try:
                cache.put(f"value_{i}", test_key=f"key_{i}")
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(do_put, i) for i in range(20)]
            for f in as_completed(futures):
                f.result()

        assert not errors, f"Errors during concurrent puts: {errors}"
        # Verify at least some entries were stored
        stored = sum(1 for i in range(20) if cache.get(test_key=f"key_{i}") is not None)
        assert stored == 20

    def test_concurrent_gets(self, tmp_path):
        cache = _make_cache(tmp_path, "json")
        for i in range(10):
            cache.put(f"value_{i}", test_key=f"key_{i}")

        results = {}

        def do_get(i):
            val = cache.get(test_key=f"key_{i}")
            results[i] = val

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(do_get, i) for i in range(10)]
            for f in as_completed(futures):
                f.result()

        for i in range(10):
            assert results[i] == f"value_{i}"

    def test_concurrent_put_get_mixed(self, tmp_path):
        cache = _make_cache(tmp_path, "json")
        barrier = threading.Barrier(8)
        errors = []

        def writer(start):
            barrier.wait()
            for i in range(start, start + 5):
                try:
                    cache.put(f"val_{i}", test_key=f"k_{i}")
                except Exception as e:
                    errors.append(e)

        def reader(start):
            barrier.wait()
            for i in range(start, start + 5):
                try:
                    cache.get(test_key=f"k_{i}")  # May return None (not yet written)
                except Exception as e:
                    errors.append(e)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = []
            for batch in range(4):
                futures.append(pool.submit(writer, batch * 5))
                futures.append(pool.submit(reader, batch * 5))
            for f in as_completed(futures):
                f.result()

        assert not errors, f"Errors during mixed concurrent ops: {errors}"


class TestReentrantLocking:
    """Prove SqliteBackend's RLock allows nested lock acquisition without deadlock.

    These tests would deadlock with threading.Lock but succeed with threading.RLock.
    """

    def test_reentrant_lock_sqlite_cleanup_by_size(self, tmp_path):
        """Put enough entries to trigger size-based cleanup — exercises re-entrant path."""
        config = CacheConfig(
            storage=CacheStorageConfig(
                cache_dir=str(tmp_path),
                max_cache_size="1KB",
            ),
            metadata=CacheMetadataConfig(metadata_backend="sqlite"),
            compression=CompressionConfig(use_blosc2_arrays=False),
        )
        cache = cacheness(config)

        # Put several entries to exceed the 1KB limit, triggering cleanup_by_size
        # internally via _enforce_size_limit → metadata_backend.cleanup_by_size
        for i in range(10):
            cache.put(f"value_{i}" * 50, test_key=f"key_{i}")

        # No deadlock — test completes. Verify cache is functional.
        entries = cache.list_entries()
        assert len(entries) > 0

    def test_reentrant_lock_sqlite_get_stats_during_cleanup(self, tmp_path):
        """Call get_stats while holding the cache lock — simulates re-entrant backend access."""
        cache = _make_cache(tmp_path, "sqlite")
        for i in range(5):
            cache.put(f"value_{i}", test_key=f"key_{i}")

        # Acquire the cache-level lock, then call backend methods that also acquire it
        with cache._lock:
            stats = cache.metadata_backend.get_stats()
            assert stats["total_entries"] == 5

            cache.metadata_backend.cleanup_by_size(target_size_bytes=0)
            stats_after = cache.metadata_backend.get_stats()
            assert stats_after["total_entries"] < 5

    def test_reentrant_lock_sqlite_nested_calls(self, tmp_path):
        """Direct nested lock acquisition on SqliteBackend — would deadlock with Lock."""
        cache = _make_cache(tmp_path, "sqlite")
        for i in range(3):
            cache.put(f"value_{i}", test_key=f"key_{i}")

        backend = cache.metadata_backend

        # Acquire the backend lock, then call a method that also acquires it
        with backend._lock:
            stats = backend.get_stats()
            assert stats["total_entries"] == 3

            entries = backend.list_entries()
            assert len(entries) == 3

            summaries = backend.iter_entry_summaries()
            assert len(summaries) == 3


class TestThreadSafetySqlite:
    """Thread safety smoke tests with SQLite metadata backend."""

    def test_concurrent_puts(self, tmp_path):
        cache = _make_cache(tmp_path, "sqlite")
        errors = []

        def do_put(i):
            try:
                cache.put(f"value_{i}", test_key=f"key_{i}")
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(do_put, i) for i in range(20)]
            for f in as_completed(futures):
                f.result()

        assert not errors, f"Errors during concurrent puts: {errors}"
        stored = sum(1 for i in range(20) if cache.get(test_key=f"key_{i}") is not None)
        assert stored == 20

    def test_concurrent_gets(self, tmp_path):
        cache = _make_cache(tmp_path, "sqlite")
        for i in range(10):
            cache.put(f"value_{i}", test_key=f"key_{i}")

        results = {}

        def do_get(i):
            val = cache.get(test_key=f"key_{i}")
            results[i] = val

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(do_get, i) for i in range(10)]
            for f in as_completed(futures):
                f.result()

        for i in range(10):
            assert results[i] == f"value_{i}"

    def test_concurrent_put_get_mixed(self, tmp_path):
        cache = _make_cache(tmp_path, "sqlite")
        barrier = threading.Barrier(8)
        errors = []

        def writer(start):
            barrier.wait()
            for i in range(start, start + 5):
                try:
                    cache.put(f"val_{i}", test_key=f"k_{i}")
                except Exception as e:
                    errors.append(e)

        def reader(start):
            barrier.wait()
            for i in range(start, start + 5):
                try:
                    cache.get(test_key=f"k_{i}")
                except Exception as e:
                    errors.append(e)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = []
            for batch in range(4):
                futures.append(pool.submit(writer, batch * 5))
                futures.append(pool.submit(reader, batch * 5))
            for f in as_completed(futures):
                f.result()

        assert not errors, f"Errors during mixed concurrent ops: {errors}"
