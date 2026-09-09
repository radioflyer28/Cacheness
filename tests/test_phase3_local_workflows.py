"""Finite local-store and cache-engine regressions for the direct Phase 3 pass."""

import sqlite3
import multiprocessing
import time

import pytest

from cacheness import CacheConfig
from cacheness.cache_policy import CacheOutcome
from cacheness.config import CacheStorageConfig, LifecycleLimits
from cacheness.core import UnifiedCache
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleTimeoutError,
    CacheBlobStoreClosedError,
    CacheBlobMigrationRequiredError,
    CacheBlobRecoverableCleanupError,
)
from cacheness.storage import BlobStore
from cacheness.storage.composition import BackendRef, StoreTopology
from cacheness.storage.lifecycle_authority import EntryExpectation, MutationSpec
from cacheness.storage.sqlite_lifecycle_authority import SqliteLifecycleAuthority


def prepared(authority, number):
    return authority.prepare_mutation(MutationSpec.create(
        operation_id=f"operation-{number}", key=f"key-{number}",
        generation=f"generation-{number}",
        candidate_locator=f".cacheness/generations/{number}.payload",
        expected=EntryExpectation.absent(),
    ))


def _local_topology(root):
    """Create the supported filesystem/SQLite direct-store composition."""
    return StoreTopology(
        payload=BackendRef(name="filesystem", options={"base_dir": root}),
        authority=BackendRef(name="sqlite", options={"root": root}),
    )


def _memory_topology():
    """Create the explicit same-process-only direct-store composition."""
    return StoreTopology(
        payload=BackendRef(name="memory"),
        authority=BackendRef(name="memory"),
    )


def test_memory_unpublished_abort_is_pageable(tmp_path):
    with BlobStore(_memory_topology(), cache_dir=tmp_path, config=memory_config(tmp_path)) as store:
        authority = store.lifecycle_authority
        mutation = prepared(authority, 1)
        authority.abort_mutation(mutation, candidate_persisted=False)
        assert not store.reconcile().findings
        assert not store.reconcile(apply=True).findings


def test_memory_debt_resume_survives_retirement(tmp_path):
    with BlobStore(_memory_topology(), cache_dir=tmp_path, config=memory_config(tmp_path)) as store:
        authority = store.lifecycle_authority
        authority.lifecycle_limits = LifecycleLimits(operation_page_size=1)
        for number in range(3):
            authority.abort_mutation(prepared(authority, number), candidate_persisted=True)
        snapshot = authority.reconciliation_snapshot()
        cursor = 0
        seen = []
        while cursor < snapshot.debt_high_water:
            page = authority.page_reconciliation_work(
                snapshot, mutation_cursor=snapshot.mutation_high_water,
                debt_cursor=cursor,
            )
            for work in page.works:
                if work.debt is not None:
                    seen.append(work.debt.operation_id)
                    authority.retire_cleanup_debt(work.debt)
                    authority.retire_cleanup_debt(work.debt)
            cursor = page.debt_cursor
        assert seen == [f"operation-{n}" for n in range(3)]
        assert not authority.pending_cleanup_debts()


def test_memory_public_apply_reclaims_three_paged_debts(tmp_path, monkeypatch):
    config = memory_config(tmp_path)
    config.lifecycle_limits = LifecycleLimits(operation_page_size=1, max_reconcile_actions=1)
    with BlobStore(_memory_topology(), cache_dir=tmp_path, config=config) as store:
        old_paths = []
        cleanup = store._delete_or_prove_absent
        for number in range(3):
            key = str(number)
            receipt = store.put_entry("old", key=key)
            old_paths.append(store.cache_dir / receipt.locator)

        def unavailable(_locator):
            raise OSError("defer old payload cleanup")

        monkeypatch.setattr(store, "_delete_or_prove_absent", unavailable)
        for number in range(3):
            with pytest.raises(CacheBlobRecoverableCleanupError):
                store.put("new", key=str(number))
        monkeypatch.setattr(store, "_delete_or_prove_absent", cleanup)
        report = store.reconcile(apply=True)
        for _ in range(20):
            if report.resume_token is None:
                break
            report = store.reconcile(apply=True, resume_token=report.resume_token)
        assert report.resume_token is None
        assert not store.lifecycle_authority.pending_cleanup_debts()
        assert all(not path.exists() for path in old_paths)
        assert [store.get(str(number)) for number in range(3)] == ["new"] * 3
        assert not store.reconcile(apply=True).findings


def memory_config(root):
    return CacheConfig(storage=CacheStorageConfig(cache_dir=root))


def _cache_topology(root):
    """Build the persisted local topology for one explicit cache root."""
    return _local_topology(root / ".cacheness" / "blobstore")


def test_entry_snapshot_and_receipt_preserve_generation(tmp_path):
    with BlobStore(_local_topology(tmp_path), cache_dir=tmp_path) as store:
        store.initialize()
        first = store.put_entry(None, key="key", metadata={"label": "first"})
        with store.open_entry("key") as entry:
            assert entry is not None
            assert entry.expectation == first.expectation
            store.put_entry("new", key="key", metadata={"label": "second"})
            assert entry.metadata["metadata"]["label"] == "first"
            assert entry.read() is None
        with pytest.raises(RuntimeError):
            entry.read()
        with store.open_entry("absent") as missing:
            assert missing is None


def _initialized_worker(root, key):
    with BlobStore(_local_topology(root), cache_dir=root) as store:
        store.put({"key": key}, key=key)
        assert store.get(key) == {"key": key}


def test_initialize_before_independent_workers(tmp_path):
    with BlobStore(_local_topology(tmp_path), cache_dir=tmp_path) as store:
        store.initialize()
    context = multiprocessing.get_context("spawn")
    workers = [context.Process(target=_initialized_worker, args=(tmp_path, str(n)))
               for n in range(3)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(30)
        if worker.is_alive():
            worker.terminate()
            worker.join(5)
        assert worker.exitcode == 0
    with BlobStore(_local_topology(tmp_path), cache_dir=tmp_path) as store:
        assert store.list() == ["0", "1", "2"]


def test_incomplete_catalog_is_not_adopted(tmp_path):
    authority = SqliteLifecycleAuthority.for_root(tmp_path)
    authority.path.parent.mkdir()
    authority.path.touch()
    before = authority.path.read_bytes()
    with pytest.raises(CacheBlobMigrationRequiredError):
        authority.initialize()
    assert authority.path.read_bytes() == before
    authority.close()


@pytest.mark.parametrize("code,expected", [
    (sqlite3.SQLITE_BUSY, CacheBlobLifecycleTimeoutError),
    (sqlite3.SQLITE_LOCKED, CacheBlobLifecycleTimeoutError),
    (sqlite3.SQLITE_IOERR | (3 << 8), CacheBlobBackendError),
    (sqlite3.SQLITE_FULL, CacheBlobBackendError),
    (sqlite3.SQLITE_READONLY, CacheBlobBackendError),
    (sqlite3.SQLITE_CANTOPEN, CacheBlobBackendError),
    (sqlite3.SQLITE_CORRUPT, CacheBlobMigrationRequiredError),
    (sqlite3.SQLITE_NOTADB, CacheBlobMigrationRequiredError),
    (None, CacheBlobBackendError),
])
def test_sqlite_primary_error_meaning(tmp_path, code, expected):
    authority = SqliteLifecycleAuthority.for_root(tmp_path)
    error = sqlite3.OperationalError("injected operational failure")
    if code is not None:
        error.sqlite_errorcode = code
    now = time.monotonic()
    with pytest.raises(expected) as caught:
        authority._translate_sqlite_error(
            error, operation="lifecycle_authority", stage="connection_open",
            deadline=now + 5, started_at=now,
        )
    assert caught.value.__cause__ is error
    authority.close()


def test_cache_expiry_leaves_separate_blob_store_untouched(tmp_path):
    object_root = tmp_path / "objects"
    with BlobStore(_local_topology(object_root), cache_dir=object_root) as objects:
        objects.put("durable", key="key", metadata={"label": "first"})
        objects.update_metadata("key", {"label": "second"})
        assert objects.get_metadata("key")["metadata"]["label"] == "second"
        cache = UnifiedCache(
            CacheConfig(storage=CacheStorageConfig(cache_dir=tmp_path / "cache")),
            store=_memory_topology(),
        )
        cache.initialize()
        try:
            key = cache.put("temporary", identity="key").receipt.key
            assert (
                cache.lookup(cache_key=key, ttl_hours=-1).outcome
                is CacheOutcome.EXPIRED
            )
            cache.clear_all()
            assert objects.get("key") == "durable"
        finally:
            cache.close()


def test_close_after_commit_has_declared_derived_outcome(tmp_path, monkeypatch):
    config = CacheConfig(storage=CacheStorageConfig(cache_dir=tmp_path))
    cache = UnifiedCache(config, store=_cache_topology(tmp_path))
    cache.initialize()
    put_entry = cache._cache_blob_store.put_entry

    def commit_then_close(*args, **kwargs):
        receipt = put_entry(*args, **kwargs)
        cache.close()
        return receipt

    monkeypatch.setattr(cache._cache_blob_store, "put_entry", commit_then_close)
    with pytest.raises(CacheBlobStoreClosedError):
        cache.put("committed", identity="close")
    reopened = UnifiedCache(config, store=_cache_topology(tmp_path))
    reopened.initialize()
    try:
        result = reopened.lookup(identity="close")
        assert result.outcome is CacheOutcome.HIT
        assert result.value == "committed"
    finally:
        reopened.close()
