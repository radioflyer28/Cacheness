"""
SQLite Concurrency Unit Tests
=============================

Unit tests to ensure the SQLite backend maintains proper concurrency capabilities
for multi-threaded applications. These tests verify that the cache can handle:

1. Multiple threads writing different data concurrently
2. High concurrency scenarios with unique data
3. Mixed read/write operations with WAL mode  
4. Proper WAL mode configuration
5. Thread safety of cache operations

These tests are critical for ensuring future development doesn't break
the multi-threading capabilities of the cache system.

Note: These tests focus on realistic concurrency patterns where different
threads work with different data, which is the common use case.
"""

import tempfile
import threading
import time
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import get_context
import sqlite3

import pytest

from cacheness.config import CacheConfig
from cacheness.core import UnifiedCache
from cacheness.error_handling import CacheBlobLifecycleTimeoutError, CacheReason
from cacheness.storage.sqlite_lifecycle_authority import SqliteLifecycleAuthority


_EXPECTED_CONTENTION_STAGES = frozenset(
    {"writer_admission", "scheduler_dispatch", "sqlite_busy"}
)
_EXTREME_WORKLOAD_TIMEOUT_SECONDS = 15.0


def _is_expected_contention_timeout(error: BaseException) -> bool:
    """Accept only the authority's explicit bounded-contention result."""
    if not isinstance(error, CacheBlobLifecycleTimeoutError):
        return False
    context = error.context
    return (
        context.get("reason") == CacheReason.BLOB_LIFECYCLE_TIMEOUT.value
        and context.get("operation") == "lifecycle_authority"
        and context.get("stage") in _EXPECTED_CONTENTION_STAGES
    )


def _extreme_operation_count(scenario: str) -> int:
    """Return the fixed number of public calls each workload worker attempts."""
    if scenario == "high_concurrency_stress":
        return 7
    if scenario == "concurrent_metadata_access":
        return 15
    if scenario == "deadlock_prevention":
        return 30
    raise AssertionError(f"Unknown extreme contention scenario: {scenario}")


def _execute_extreme_contention_child(
    cache_dir: str,
    scenario: str,
    send_connection,
) -> None:
    """Execute one finite stress schedule and return evidence to its parent."""
    cache = None
    report = {
        "workers": [],
        "hard_failures": [],
        "read_failures": [],
        "worker_alive": [],
        "registry_empty": False,
        "live_keys_match": False,
    }
    try:
        cache = UnifiedCache(
            config=CacheConfig(
                cache_dir=cache_dir,
                metadata_backend="sqlite",
                store_cache_key_params=True,
            )
        )
        worker_count = 8
        iteration_count = 5 if scenario == "high_concurrency_stress" else (8 if scenario == "concurrent_metadata_access" else 10)
        records_lock = threading.Lock()
        first_use_barrier = threading.Barrier(worker_count)

        def record_call(worker_record, name, call, put_record=None):
            worker_record["attempted"] += 1
            try:
                value = call()
                if name in {"query", "list"} and not isinstance(value, list):
                    raise AssertionError(f"{name} returned {type(value).__name__}, not list")
                worker_record["successful"] += 1
                if put_record is not None:
                    put_record["cache_key"] = value
                    worker_record["successful_puts"].append(put_record)
            except BaseException as error:
                if _is_expected_contention_timeout(error):
                    worker_record["allowed_timeouts"] += 1
                    if put_record is not None:
                        worker_record["timed_out_puts"].append(put_record)
                else:
                    worker_record["hard_failures"].append(
                        {
                            "operation": name,
                            "type": type(error).__name__,
                            "message": str(error),
                            "context": getattr(error, "context", None),
                        }
                    )

        def worker(worker_id):
            worker_record = {
                "worker_id": worker_id,
                "attempted": 0,
                "successful": 0,
                "allowed_timeouts": 0,
                "hard_failures": [],
                "successful_puts": [],
                "timed_out_puts": [],
                "elapsed": 0.0,
            }
            first_use_barrier.wait(timeout=2)
            started = time.monotonic()
            for item in range(iteration_count):
                params = {"scenario": scenario, "worker": worker_id, "item": item}
                expected = f"{scenario}-{worker_id}-{item}"
                put_record = {"params": params, "expected": expected}
                record_call(
                    worker_record,
                    "put",
                    lambda: cache.put(expected, **params),
                    put_record,
                )
                if scenario == "high_concurrency_stress" and item % 3 == 0:
                    record_call(
                        worker_record,
                        "query",
                        lambda: cache.query_meta(worker=f"int:{worker_id}"),
                    )
                elif scenario == "concurrent_metadata_access":
                    if item % 2 == 0:
                        record_call(worker_record, "list", cache.list_entries)
                    if item % 3 == 0:
                        record_call(
                            worker_record,
                            "query",
                            lambda: cache.query_meta(worker=f"int:{worker_id}"),
                        )
                elif scenario == "deadlock_prevention":
                    record_call(
                        worker_record,
                        "query",
                        lambda: cache.query_meta(worker=f"int:{worker_id}"),
                    )
                    record_call(worker_record, "list", cache.list_entries)
            worker_record["elapsed"] = time.monotonic() - started
            with records_lock:
                report["workers"].append(worker_record)

        deadline = time.monotonic() + _EXTREME_WORKLOAD_TIMEOUT_SECONDS
        threads = [
            threading.Thread(target=worker, args=(worker_id,), daemon=True)
            for worker_id in range(worker_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                thread.join(timeout=remaining)
        report["worker_alive"] = [index for index, thread in enumerate(threads) if thread.is_alive()]

        if not report["worker_alive"]:
            successful_puts = [
                record
                for worker_record in report["workers"]
                for record in worker_record["successful_puts"]
            ]
            timed_out_puts = [
                record
                for worker_record in report["workers"]
                for record in worker_record["timed_out_puts"]
            ]
            for record in successful_puts:
                try:
                    actual = cache.get(**record["params"])
                    if actual != record["expected"]:
                        raise AssertionError(
                            f"wrong value: expected {record['expected']!r}, got {actual!r}"
                        )
                except BaseException as error:
                    report["read_failures"].append(
                        {"kind": "successful_put", "type": type(error).__name__, "message": str(error)}
                    )
            for record in timed_out_puts:
                try:
                    actual = cache.get(**record["params"])
                    if actual is not None:
                        raise AssertionError(f"timed-out put became live: {actual!r}")
                except BaseException as error:
                    report["read_failures"].append(
                        {"kind": "timed_out_put", "type": type(error).__name__, "message": str(error)}
                    )
            try:
                entries = cache.list_entries()
                entry_keys = {entry["cache_key"] for entry in entries}
                successful_keys = {record["cache_key"] for record in successful_puts}
                report["live_keys_match"] = entry_keys == successful_keys
            except BaseException as error:
                report["read_failures"].append(
                    {"kind": "list_entries", "type": type(error).__name__, "message": str(error)}
                )
        report["hard_failures"] = [
            failure
            for worker_record in report["workers"]
            for failure in worker_record["hard_failures"]
        ]
    except BaseException as error:
        report["fatal"] = {
            "type": type(error).__name__,
            "message": str(error),
            "context": getattr(error, "context", None),
        }
    finally:
        if cache is not None and not report["worker_alive"]:
            try:
                cache.close()
            except BaseException as error:
                report["close_failure"] = f"{type(error).__name__}: {error}"
        report["registry_empty"] = SqliteLifecycleAuthority.admission_registry_size_for_test() == 0
        send_connection.send(report)
        send_connection.close()


def _run_extreme_contention(tmp_path: Path, scenario: str) -> dict:
    """Bound a historical contention schedule in a disposable POSIX child."""
    try:
        context = get_context("fork")
    except ValueError:
        pytest.skip("fork process context is unavailable on this platform")
    receive_connection, send_connection = context.Pipe(duplex=False)
    child = context.Process(
        target=_execute_extreme_contention_child,
        args=(str(tmp_path / scenario), scenario, send_connection),
    )
    started = time.monotonic()
    child.start()
    send_connection.close()
    child.join(timeout=_EXTREME_WORKLOAD_TIMEOUT_SECONDS)
    exceeded_deadline = child.is_alive()
    if exceeded_deadline:
        child.terminate()
        child.join(timeout=2)
    report = receive_connection.recv() if receive_connection.poll() else None
    receive_connection.close()
    assert not exceeded_deadline, "contention child exceeded its bounded completion deadline"
    assert child.exitcode == 0
    assert report is not None
    report["parent_elapsed"] = time.monotonic() - started
    return report


def _assert_extreme_contention(report: dict, scenario: str) -> None:
    """Assert progress, accounting, integrity, and cleanup from a child schedule."""
    assert "fatal" not in report, report.get("fatal")
    assert "close_failure" not in report, report.get("close_failure")
    assert report["worker_alive"] == []
    assert len(report["workers"]) == 8
    expected_attempts = 8 * _extreme_operation_count(scenario)
    attempted = sum(worker["attempted"] for worker in report["workers"])
    successful = sum(worker["successful"] for worker in report["workers"])
    timed_out = sum(worker["allowed_timeouts"] for worker in report["workers"])
    hard_failures = sum(len(worker["hard_failures"]) for worker in report["workers"])
    assert attempted == expected_attempts
    assert successful + timed_out + hard_failures == attempted
    assert report["hard_failures"] == []
    assert successful >= expected_attempts * 0.75
    assert all(worker["successful"] >= 1 for worker in report["workers"])
    assert all(worker["elapsed"] < _EXTREME_WORKLOAD_TIMEOUT_SECONDS for worker in report["workers"])
    assert report["read_failures"] == []
    assert report["live_keys_match"]
    assert report["registry_empty"]


@pytest.fixture
def temp_cache():
    """Fixture to create a temporary cache for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = CacheConfig(
            cache_dir=temp_dir,
            metadata_backend="sqlite",
            store_cache_key_params=True
        )
        cache = UnifiedCache(config=config)
        yield cache
        cache.close()


@pytest.fixture
def basic_temp_cache():
    """Fixture to create a basic temporary cache without store_cache_key_params."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = CacheConfig(
            cache_dir=temp_dir,
            metadata_backend="sqlite"
        )
        cache = UnifiedCache(config=config)
        yield cache
        cache.close()


class TestSQLiteConcurrency:
    """Test suite for SQLite backend concurrency."""

    def test_wal_mode_enabled(self, basic_temp_cache):
        """Test that WAL mode is properly enabled for concurrency."""
        cache = basic_temp_cache
        
        # Give SQLite a moment to initialize
        time.sleep(0.1)
        
        # Check WAL mode directly
        db_file = Path(cache.cache_dir) / "cache_metadata.db"
        conn = sqlite3.connect(str(db_file))
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode")
            journal_mode = cursor.fetchone()[0]
            
            # WAL mode should be enabled for concurrency
            assert journal_mode.lower() == "wal", f"Expected WAL mode, got {journal_mode}"
        finally:
            conn.close()

    def test_concurrent_put_operations(self, temp_cache):
        """Test multiple threads performing put operations concurrently."""
        cache = temp_cache
        
        results = []
        errors = []
        num_threads = 6
        operations_per_thread = 20
        
        def worker_thread(thread_id):
            try:
                thread_results = []
                for i in range(operations_per_thread):
                    key = f"thread_{thread_id}_item_{i}"
                    value = f"data_{thread_id}_{i}"
                    cache.put(key, value, thread_id=thread_id, operation=i)
                    thread_results.append(key)
                return thread_results
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")
                return []
        
        # Execute concurrent puts
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_thread, i) for i in range(num_threads)]
            for future in as_completed(futures):
                results.extend(future.result())
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == num_threads * operations_per_thread
        
        # Verify all entries are in cache
        entries = cache.list_entries()
        assert len(entries) == num_threads * operations_per_thread

    def test_concurrent_different_data_operations(self):
        """Test concurrent operations with different data types and operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(
                cache_dir=temp_dir,
                metadata_backend="sqlite",
                store_cache_key_params=True
            )
            cache = UnifiedCache(config=config)
            
            results = {'strings': 0, 'numbers': 0, 'lists': 0}
            results_lock = threading.Lock()
            errors = []
            
            def mixed_data_worker(worker_id):
                try:
                    local_results = {'strings': 0, 'numbers': 0, 'lists': 0}
                    
                    for i in range(10):
                        # Store different types of data
                        if i % 3 == 0:
                            cache.put(f"text_data_{worker_id}_{i}", worker=worker_id, dtype="string", item=i)
                            local_results['strings'] += 1
                        elif i % 3 == 1:
                            cache.put(worker_id * 100 + i, worker=worker_id, dtype="number", item=i)
                            local_results['numbers'] += 1
                        else:
                            cache.put([worker_id, i, "data"], worker=worker_id, dtype="list", item=i)
                            local_results['lists'] += 1
                        
                        # Small random delay
                        if random.random() < 0.1:
                            time.sleep(0.001)
                    
                    # Update global results thread-safely
                    with results_lock:
                        for key in results:
                            results[key] += local_results[key]
                            
                except Exception as e:
                    errors.append(f"Worker {worker_id}: {e}")
            
            # Execute mixed operations
            threads = []
            num_workers = 5
            for i in range(num_workers):
                t = threading.Thread(target=mixed_data_worker, args=(i,))
                threads.append(t)
                t.start()
            
            for t in threads:
                t.join()
            
            # Verify results
            assert len(errors) == 0, f"Errors in mixed operations: {errors}"
            assert results['strings'] > 0, "Should have string operations"
            assert results['numbers'] > 0, "Should have number operations"
            assert results['lists'] > 0, "Should have list operations"
            
            # Verify cache integrity
            entries = cache.list_entries()
            expected_total = sum(results.values())
            assert len(entries) == expected_total, f"Expected {expected_total} entries, got {len(entries)}"
            
            cache.close()

    def test_high_concurrency_stress(self, tmp_path):
        """Bounded write contention retains progress and committed-value integrity."""
        _assert_extreme_contention(
            _run_extreme_contention(tmp_path, "high_concurrency_stress"),
            "high_concurrency_stress",
        )

    def test_concurrent_query_operations(self):
        """Test concurrent query_meta operations for thread safety."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(
                cache_dir=temp_dir,
                metadata_backend="sqlite",
                store_cache_key_params=True
            )
            cache = UnifiedCache(config=config)
            
            # Pre-populate with tagged data
            for i in range(30):
                cache.put(f"query_test_{i}", f"value_{i}", 
                         category=f"cat_{i % 3}", priority=i % 5)
            
            query_results = []
            query_errors = []
            results_lock = threading.Lock()
            
            def query_worker(worker_id):
                try:
                    local_results = []
                    for i in range(8):
                        # Different types of queries
                        if i % 3 == 0:
                            results = cache.query_meta(category=f"str:cat_{i % 3}")
                        elif i % 3 == 1:
                            results = cache.query_meta(priority=f"int:{i % 5}")
                        else:
                            results = cache.query_meta()  # Get all
                        
                        assert isinstance(results, list)
                        assert results
                        local_results.append(len(results))
                        time.sleep(0.001)
                    
                    with results_lock:
                        query_results.extend(local_results)
                        
                except Exception as e:
                    query_errors.append(f"Query worker {worker_id}: {e}")
            
            # Execute concurrent queries
            num_workers = 6
            threads = []
            for i in range(num_workers):
                t = threading.Thread(target=query_worker, args=(i,))
                threads.append(t)
                t.start()
            
            for t in threads:
                t.join(timeout=5)
                assert not t.is_alive()
            
            # Verify results
            assert len(query_errors) == 0, f"Query errors: {query_errors}"
            assert len(query_results) == num_workers * 8
            assert all(count > 0 for count in query_results), "All queries should return results"
            
            cache.close()

    def test_concurrent_metadata_access(self, tmp_path):
        """Metadata contention accounts for each bounded public operation."""
        _assert_extreme_contention(
            _run_extreme_contention(tmp_path, "concurrent_metadata_access"),
            "concurrent_metadata_access",
        )

    def test_thread_safety_with_file_operations(self):
        """Test thread safety when cache files are being written/read."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(
                cache_dir=temp_dir,
                metadata_backend="sqlite"
            )
            cache = UnifiedCache(config=config)
            
            def file_worker(worker_id):
                try:
                    # Create some cache entries that will generate files
                    for i in range(8):
                        # Use different data types to create different file patterns
                        if i % 2 == 0:
                            data = {"worker": worker_id, "item": i, "data": f"content_{i}"}
                        else:
                            data = [worker_id, i, f"list_data_{i}"]
                        
                        cache.put(data, worker=worker_id, item=i)
                    
                    return True
                except Exception as e:
                    return str(e)
            
            # Multiple workers creating files concurrently
            num_workers = 6
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(file_worker, i) for i in range(num_workers)]
                results = [f.result() for f in futures]
            
            # Verify file operations succeeded
            errors = [r for r in results if r is not True]
            assert len(errors) == 0, f"File operation errors: {errors}"
            
            # Verify all cache files were created properly
            entries = cache.list_entries()
            assert len(entries) == num_workers * 8
            
            cache.close()

    def test_deadlock_prevention(self, tmp_path):
        """A bounded child detects hangs while preserving per-call accounting."""
        _assert_extreme_contention(
            _run_extreme_contention(tmp_path, "deadlock_prevention"),
            "deadlock_prevention",
        )

    def test_concurrent_get_operations(self):
        """Test multiple threads performing get operations concurrently."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(
                cache_dir=temp_dir,
                metadata_backend="sqlite"
            )
            cache = UnifiedCache(config=config)
            
            # Pre-populate cache with simple, consistent data
            test_data = {}
            for i in range(12):  # Reduced from 20 to 12
                value = f"value_{i}"
                cache.put(value, item_id=i)  # Use kwargs to generate unique cache keys
                test_data[i] = value  # Store by index for retrieval
            
            successful_reads = 0
            errors = []
            num_threads = 3  # Reduced from 4 to 3
            
            def reader_thread(thread_id):
                nonlocal successful_reads
                try:
                    local_reads = 0
                    # Each thread reads items sequentially
                    for i in range(4):  # Reduced from 5 to 4
                        item_id = i + thread_id * 4  # Each thread reads different items
                        if item_id < 12:  # Make sure we don't go out of bounds (updated from 20)
                            value = cache.get(item_id=item_id)
                            assert value == f"value_{item_id}"
                            local_reads += 1
                            time.sleep(0.002)  # Slightly longer delay
                    return local_reads
                except Exception as e:
                    errors.append(f"Reader {thread_id}: {e}")
                    return 0
            
            # Execute concurrent reads
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(reader_thread, i) for i in range(num_threads)]
                results = [f.result(timeout=5) for f in futures]
                successful_reads = sum(results)
            
            assert errors == [], f"Read errors occurred: {errors}"
            
            assert successful_reads == 12
            
            cache.close()

    def test_concurrent_cache_stats_access(self):
        """Test that cache statistics are properly updated under concurrency."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(
                cache_dir=temp_dir,
                metadata_backend="sqlite"
            )
            cache = UnifiedCache(config=config)
            
            def stats_worker(worker_id):
                try:
                    for i in range(10):
                        # Mix of cache hits and misses
                        value = f"value_{i}"
                        cache.put(value, worker=worker_id, item=i)
                        
                        # Try to get it back (should be a hit)
                        cache.get(worker=worker_id, item=i)
                        
                        # Try to get non-existent key (should be a miss)
                        cache.get(worker=worker_id, item=i+1000)  # Non-existent item
                    
                    return True
                except Exception as e:
                    return str(e)
            
            # Run concurrent stats operations
            num_workers = 5
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(stats_worker, i) for i in range(num_workers)]
                results = [f.result() for f in futures]
            
            # Verify all workers succeeded
            errors = [r for r in results if r is not True]
            assert len(errors) == 0, f"Stats test errors: {errors}"
            
            # Get final stats - check if cache has stats method
            try:
                if hasattr(cache, 'get_stats'):
                    stats = cache.get_stats()
                    assert stats is not None
                    # Check if stats contain hit/miss info (may vary by backend)
                    assert 'total_entries' in stats or 'cache_hits' in stats, "Should have some stats"
                else:
                    # If get_stats doesn't exist, just verify cache operations worked
                    entries = cache.list_entries()
                    assert len(entries) == num_workers * 10, "Should have all cached entries"
            except AttributeError:
                # If get_stats doesn't exist, just verify cache operations worked
                entries = cache.list_entries()
                assert len(entries) == num_workers * 10, "Should have all cached entries"
            
            cache.close()

    def test_thread_safety_data_integrity(self):
        """Test that data integrity is maintained under concurrent access."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(
                cache_dir=temp_dir,
                metadata_backend="sqlite"
            )
            cache = UnifiedCache(config=config)
            
            # Test data that will be written concurrently
            test_keys = [f"integrity_test_{i}" for i in range(10)]
            expected_values = {key: f"value_for_{key}" for key in test_keys}
            
            def integrity_worker(worker_id):
                try:
                    # Each worker writes the same data with unique identifiers
                    for i, key in enumerate(test_keys):
                        cache.put(expected_values[key], test_key=key, worker=worker_id)
                        time.sleep(0.001)  # Small delay to increase contention
                    return True
                except Exception as e:
                    return str(e)
            
            # Multiple workers writing the same data concurrently
            num_workers = 4
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(integrity_worker, i) for i in range(num_workers)]
                results = [f.result() for f in futures]
            
            # Verify all workers succeeded
            errors = [r for r in results if r is not True]
            assert len(errors) == 0, f"Integrity test errors: {errors}"
            
            # Verify data integrity - all values should be correct
            for i, key in enumerate(test_keys):
                actual_value = cache.get(test_key=key, worker=0)  # Get with same params as stored
                expected_value = expected_values[key]
                assert actual_value == expected_value, f"Data corruption for {key}: expected {expected_value}, got {actual_value}"
            
            cache.close()


if __name__ == "__main__":
    # Run tests individually for debugging
    # Note: This direct execution doesn't use fixtures, so we create cache instances manually
    test_instance = TestSQLiteConcurrency()
    
    print("Running SQLite concurrency unit tests...")
    
    try:
        # Create basic temp cache manually for WAL test
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(
                cache_dir=temp_dir,
                metadata_backend="sqlite"
            )
            basic_cache = UnifiedCache(config=config)
            test_instance.test_wal_mode_enabled(basic_cache)
            print("✅ WAL mode test passed")
        
        # Create temp cache manually for put operations test
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(
                cache_dir=temp_dir,
                metadata_backend="sqlite",
                store_cache_key_params=True
            )
            temp_cache = UnifiedCache(config=config)
            test_instance.test_concurrent_put_operations(temp_cache)
            print("✅ Concurrent put operations test passed")
        
        test_instance.test_concurrent_different_data_operations()
        print("✅ Concurrent different data operations test passed")
        
        test_instance.test_high_concurrency_stress()
        print("✅ High concurrency stress test passed")
        
        test_instance.test_concurrent_query_operations()
        print("✅ Concurrent query operations test passed")
        
        test_instance.test_concurrent_metadata_access()
        print("✅ Concurrent metadata access test passed")
        
        test_instance.test_thread_safety_with_file_operations()
        print("✅ Thread safety with file operations test passed")
        
        test_instance.test_deadlock_prevention()
        print("✅ Deadlock prevention test passed")
        
        print("\n🎉 All SQLite concurrency tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
