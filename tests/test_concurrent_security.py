"""
Concurrent Security Stress Tests
=================================

Thread safety stress tests for security-layer operations:
- Concurrent rotate_key() with put()/get()
- Concurrent encrypted put/get across 8+ threads
- Sustained access deadlock detection

Covers TEST-01 (thread safety under concurrent access) for v0.10.0
security features not covered by test_concurrency_stress.py (Phase 7).
"""

import secrets
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from cacheness.config import (
    CacheConfig,
    CacheMetadataConfig,
    CacheStorageConfig,
    CompressionConfig,
    SecurityConfig,
)
from cacheness.core import UnifiedCache


def _generate_key_file(path: Path) -> Path:
    """Write 32 random bytes to a file and return the path."""
    path.write_bytes(secrets.token_bytes(32))
    return path


@pytest.fixture
def signing_cache():
    """SQLite-backed cache with signing enabled (no encryption)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        key_file = Path(temp_dir) / "cache_signing_key.bin"
        _generate_key_file(key_file)
        config = CacheConfig(
            cache_dir=temp_dir,
            metadata_backend="sqlite",
            security=SecurityConfig(
                enable_entry_signing=True,
                enable_content_encryption=False,
                allow_unsigned_entries=True,
                delete_invalid_signatures=False,
            ),
        )
        cache = UnifiedCache(config=config)
        yield cache
        cache.close()


@pytest.fixture
def encrypted_cache():
    """JSON-backed cache with signing and encryption enabled.

    Uses JSON backend because encryption+SQLite has a known incompatibility
    (matching the test_encryption_at_rest.py pattern).
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        key_file = Path(temp_dir) / "cache_signing_key.bin"
        _generate_key_file(key_file)
        config = CacheConfig(
            storage=CacheStorageConfig(cache_dir=temp_dir),
            metadata=CacheMetadataConfig(metadata_backend="json"),
            compression=CompressionConfig(use_blosc2_arrays=False),
            security=SecurityConfig(
                enable_entry_signing=True,
                enable_content_encryption=True,
                encryption_key_file="cache_signing_key.bin",
                allow_unsigned_entries=True,
                delete_invalid_signatures=False,
            ),
        )
        cache = UnifiedCache(config=config)
        yield cache
        cache.close()


class TestConcurrentRotateKey:
    """Thread safety tests for rotate_key() under concurrent access."""

    def test_rotate_key_during_concurrent_puts(self, signing_cache):
        """rotate_key() while 8 writer threads are doing put() — no crashes or corruption."""
        cache = signing_cache
        temp_dir = cache.cache_dir

        # Pre-populate 20 entries
        for i in range(20):
            cache.put(f"data-{i}", key=f"entry-{i}")

        # Create new key file for rotation
        new_key_path = Path(temp_dir) / "new_key.bin"
        _generate_key_file(new_key_path)

        errors = []

        def writer(tid):
            try:
                for j in range(5):
                    cache.put(f"new-{tid}-{j}", key=f"new-{tid}-{j}")
            except Exception as e:
                errors.append(f"Writer {tid}: {e}")

        def rotator():
            try:
                cache.rotate_key(new_key_path)
            except Exception as e:
                errors.append(f"Rotator: {e}")

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
        rotate_thread = threading.Thread(target=rotator)

        for t in threads:
            t.start()
        rotate_thread.start()

        for t in threads:
            t.join(timeout=60)
        rotate_thread.join(timeout=60)

        assert not errors, f"Errors during concurrent rotate+put: {errors}"
        # Original entries should still be accessible
        val = cache.get(key="entry-0")
        assert val is not None, "Pre-existing entry lost after concurrent rotate+put"
        assert len(cache.list_entries()) >= 20

    def test_rotate_key_during_concurrent_gets(self, signing_cache):
        """rotate_key() while 8 reader threads are doing get() — no exceptions raised."""
        cache = signing_cache
        temp_dir = cache.cache_dir

        # Pre-populate 50 entries
        for i in range(50):
            cache.put(f"data-{i}", key=f"entry-{i}")

        new_key_path = Path(temp_dir) / "new_key.bin"
        _generate_key_file(new_key_path)

        errors = []

        def reader(tid):
            try:
                for _ in range(10):
                    idx = (tid * 7 + _) % 50  # deterministic spread
                    cache.get(key=f"entry-{idx}")
                    # get() may return None during rotation (entry being re-signed)
                    # but must not raise
            except Exception as e:
                errors.append(f"Reader {tid}: {e}")

        def rotator():
            try:
                cache.rotate_key(new_key_path)
            except Exception as e:
                errors.append(f"Rotator: {e}")

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(8)]
        rotate_thread = threading.Thread(target=rotator)

        for t in threads:
            t.start()
        rotate_thread.start()

        for t in threads:
            t.join(timeout=60)
        rotate_thread.join(timeout=60)

        assert not errors, f"Errors during concurrent rotate+get: {errors}"
        # After rotation completes, entries should be accessible
        val = cache.get(key="entry-0")
        assert val is not None, "Entry inaccessible after rotation completed"

    def test_rotate_key_result_consistent(self, signing_cache):
        """rotate_key() result counts must sum to total entries."""
        cache = signing_cache
        temp_dir = cache.cache_dir

        # Pre-populate 30 entries
        for i in range(30):
            cache.put(f"data-{i}", key=f"entry-{i}")

        new_key_path = Path(temp_dir) / "new_key.bin"
        _generate_key_file(new_key_path)

        result = cache.rotate_key(new_key_path)
        total = result.re_signed + result.failed + result.skipped
        assert total == 30, f"Result counts {total} != 30 entries"
        assert result.re_signed >= 1, "Expected at least some entries to be re-signed"


class TestConcurrentEncryptedAccess:
    """Thread safety tests for encrypted put/get across multiple threads."""

    def test_concurrent_encrypted_put_get(self, encrypted_cache):
        """8 writer threads put 10 entries each, then 8 reader threads verify — no corruption."""
        cache = encrypted_cache
        errors = []

        def writer(tid):
            try:
                for i in range(10):
                    cache.put(f"data-{tid}-{i}", key=f"enc-{tid}-{i}")
            except Exception as e:
                errors.append(f"Writer {tid}: {e}")

        # Write phase
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(writer, tid) for tid in range(8)]
            for f in as_completed(futures):
                f.result()  # propagate exceptions

        assert not errors, f"Write errors: {errors}"
        assert len(cache.list_entries()) == 80, (
            f"Expected 80 entries, got {len(cache.list_entries())}"
        )

        # Read phase — verify all entries round-trip
        read_errors = []

        def reader(tid):
            try:
                for i in range(10):
                    val = cache.get(key=f"enc-{tid}-{i}")
                    if val != f"data-{tid}-{i}":
                        read_errors.append(
                            f"Mismatch enc-{tid}-{i}: expected 'data-{tid}-{i}', got {val!r}"
                        )
            except Exception as e:
                read_errors.append(f"Reader {tid}: {e}")

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(reader, tid) for tid in range(8)]
            for f in as_completed(futures):
                f.result()

        assert not read_errors, f"Read errors: {read_errors}"

    def test_concurrent_encrypted_put_same_key(self, encrypted_cache):
        """8 threads writing to same encrypted key — last-writer-wins, no corruption."""
        cache = encrypted_cache
        errors = []

        def writer(tid):
            try:
                cache.put(f"data-from-{tid}", key="shared-enc")
            except Exception as e:
                errors.append(f"Writer {tid}: {e}")

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(writer, tid) for tid in range(8)]
            for f in as_completed(futures):
                f.result()

        assert not errors, f"Write errors: {errors}"
        val = cache.get(key="shared-enc")
        assert val is not None, "Shared key returned None"
        assert val.startswith("data-from-"), f"Unexpected value: {val!r}"


class TestDeadlockDetection:
    """Verify no deadlocks under sustained concurrent access with mixed operations."""

    def test_sustained_access_no_deadlock(self, signing_cache):
        """4 writers + 4 readers + 1 rotator for sustained access — must complete within 60s."""
        cache = signing_cache
        temp_dir = cache.cache_dir
        errors = []

        # Pre-populate some entries for readers
        for i in range(20):
            cache.put(f"seed-{i}", key=f"seed-{i}")

        new_key_path = Path(temp_dir) / "rotate_key.bin"
        _generate_key_file(new_key_path)

        def writer(tid):
            try:
                for i in range(10):
                    cache.put(f"w-{tid}-{i}", key=f"w-{tid}-{i}")
            except Exception as e:
                errors.append(f"Writer {tid}: {e}")

        def reader(tid):
            try:
                for i in range(10):
                    idx = (tid * 3 + i) % 20
                    cache.get(key=f"seed-{idx}")  # may return None, must not raise
            except Exception as e:
                errors.append(f"Reader {tid}: {e}")

        def rotator():
            try:
                import time

                time.sleep(1)  # let writers/readers start first
                cache.rotate_key(new_key_path)
            except Exception as e:
                errors.append(f"Rotator: {e}")

        threads = []
        for i in range(4):
            threads.append(
                threading.Thread(target=writer, args=(i,), name=f"writer-{i}")
            )
        for i in range(4):
            threads.append(
                threading.Thread(target=reader, args=(i,), name=f"reader-{i}")
            )
        threads.append(threading.Thread(target=rotator, name="rotator"))

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=60)

        alive = [t.name for t in threads if t.is_alive()]
        assert not alive, f"Deadlock detected — threads still alive: {alive}"
        assert not errors, f"Errors during sustained access: {errors}"
