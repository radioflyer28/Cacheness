"""Finite local-store and cache-engine regressions for the direct Phase 3 pass."""

import sqlite3
import multiprocessing
import time

import pytest

from cacheness import CacheConfig
from cacheness.config import LifecycleAuthorityTopology, LifecycleLimits
from cacheness.core import UnifiedCache
from cacheness.error_handling import (
    CacheIntegrityError, CacheBlobBackendError, CacheBlobLifecycleTimeoutError,
    CacheBlobMigrationRequiredError, CacheMetadataError,
    CacheBlobRecoverableCleanupError,
)
from cacheness.metadata import SqliteBackend
from cacheness.storage import BlobStore
from cacheness.storage.lifecycle_authority import EntryExpectation, MutationSpec
from cacheness.storage.sqlite_lifecycle_authority import SqliteLifecycleAuthority


def prepared(authority, number):
    return authority.prepare_mutation(MutationSpec.create(
        operation_id=f"operation-{number}", key=f"key-{number}",
        generation=f"generation-{number}",
        candidate_locator=f".cacheness/generations/{number}.payload",
        expected=EntryExpectation.absent(),
    ))


def test_memory_unpublished_abort_is_pageable(tmp_path):
    with BlobStore(tmp_path, backend="memory", config=memory_config(tmp_path)) as store:
        authority = store.lifecycle_authority
        mutation = prepared(authority, 1)
        authority.abort_mutation(mutation, candidate_persisted=False)
        assert not store.reconcile().findings
        assert not store.reconcile(apply=True).findings


def test_memory_debt_resume_survives_retirement(tmp_path):
    with BlobStore(tmp_path, backend="memory", config=memory_config(tmp_path)) as store:
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
    with BlobStore(tmp_path, backend="memory", config=config) as store:
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
    return CacheConfig(cache_dir=root, lifecycle_topology=LifecycleAuthorityTopology(
        durable=False, multiprocess=False, projection=False,
    ))


@pytest.mark.parametrize("value", ['{', '[]', '{"x":1,"x":2}', sqlite3.Binary(b'\xff')])
@pytest.mark.parametrize("operation", ["get_entry", "list_entries"])
def test_projection_observation_is_strict(tmp_path, value, operation):
    backend = SqliteBackend(tmp_path / "metadata.sqlite3")
    try:
        backend.put_entry("key", {"data_type": "object", "metadata": {}})
        with sqlite3.connect(tmp_path / "metadata.sqlite3") as connection:
            connection.execute("UPDATE cache_entries SET cache_key_params=?", (value,))
        with pytest.raises(CacheIntegrityError):
            getattr(backend, operation)(*(["key"] if operation == "get_entry" else []))
    finally:
        backend.close()


def test_entry_snapshot_and_receipt_preserve_generation(tmp_path):
    with BlobStore(tmp_path) as store:
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


def test_cache_reads_authority_despite_corrupt_projection(tmp_path):
    cache = UnifiedCache(CacheConfig(
        cache_dir=tmp_path, metadata_backend="sqlite",
        enable_memory_cache=False, store_cache_key_params=True,
    ))
    try:
        cache.initialize()
        key = cache.put("valid", parameter="value")
        backend = cache.metadata_backend
        with backend.engine.begin() as connection:
            connection.exec_driver_sql(
                "UPDATE cache_entries SET cache_key_params='{' WHERE cache_key=?", (key,)
            )
        assert cache.get(key) == "valid"
        with pytest.raises(CacheIntegrityError):
            backend.get_entry(key)
        assert cache._cache_blob_store.get(key) == "valid"
    finally:
        cache.close()


def _initialized_worker(root, key):
    with BlobStore(root) as store:
        store.put({"key": key}, key=key)
        assert store.get(key) == {"key": key}


def test_initialize_before_independent_workers(tmp_path):
    with BlobStore(tmp_path) as store:
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
    with BlobStore(tmp_path) as store:
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


def test_optional_export_failure_does_not_revoke_commit(tmp_path, monkeypatch):
    cache = UnifiedCache(CacheConfig(cache_dir=tmp_path, metadata_backend="sqlite"))
    try:
        def unavailable(*args, **kwargs):
            raise OSError("projection offline")
        monkeypatch.setattr(cache, "_publish_entry_projection", unavailable)
        key = cache.put("stored", identity="one")
        assert cache.get(key) == "stored"
        with pytest.raises(CacheMetadataError) as caught:
            cache.put("also stored", identity="two", custom_metadata=object())
        assert caught.value.context["committed"] is True
        assert cache.get(caught.value.context["key"]) == "also stored"
    finally:
        cache.close()


def test_cache_expiry_leaves_separate_blob_store_untouched(tmp_path):
    with BlobStore(tmp_path / "objects") as objects:
        objects.put("durable", key="key", metadata={"label": "first"})
        assert objects.list(metadata_filter={"label": "first"}) == ["key"]
        objects.update_metadata("key", {"label": "second"})
        assert objects.get_metadata("key")["metadata"]["label"] == "second"
        cache = UnifiedCache(CacheConfig(cache_dir=tmp_path / "cache"))
        try:
            key = cache.put("temporary", identity="key")
            assert cache.get(key, ttl_hours=-1) is None
            cache.clear_all()
            assert objects.get("key") == "durable"
        finally:
            cache.close()


@pytest.mark.parametrize("requested_links", [False, True])
def test_close_after_commit_has_declared_derived_outcome(tmp_path, monkeypatch, requested_links):
    cache = UnifiedCache(CacheConfig(cache_dir=tmp_path, metadata_backend="sqlite"))
    put_entry = cache._cache_blob_store.put_entry

    def commit_then_close(*args, **kwargs):
        receipt = put_entry(*args, **kwargs)
        cache.close()
        return receipt

    monkeypatch.setattr(cache._cache_blob_store, "put_entry", commit_then_close)
    if requested_links:
        with pytest.raises(CacheMetadataError) as failure:
            cache.put("committed", custom_metadata=object(), identity="close")
        assert failure.value.context["committed"] is True
        key = failure.value.context["key"]
    else:
        key = cache.put("committed", identity="close")
    reopened = UnifiedCache(CacheConfig(cache_dir=tmp_path, metadata_backend="sqlite"))
    try:
        assert reopened.get(key) == "committed"
    finally:
        reopened.close()
