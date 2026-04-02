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
