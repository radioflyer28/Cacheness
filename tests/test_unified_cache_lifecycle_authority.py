"""Regression coverage for UnifiedCache's authority-backed policy boundary."""

from __future__ import annotations

from pathlib import Path
from threading import Event, Thread

import pytest

from cacheness import CacheConfig, cacheness
from cacheness.error_handling import (
    CacheBlobLifecycleConflictError,
    CacheBlobRecoverableCleanupError,
)


def _two_caches(root: Path):
    """Open independent cache-policy facades over one durable cache root."""
    configuration = CacheConfig(
        cache_dir=str(root),
        cleanup_on_init=False,
        store_cache_key_params=True,
    )
    return cacheness(configuration), cacheness(configuration)


def test_replacement_before_projection_hook_preserves_m2(
    tmp_path: Path,
) -> None:
    """An A write that captured M1 loses cleanly when B replaces M1 first."""
    first, second = _two_caches(tmp_path / "replacement-before-projection")
    paused = Event()
    resume = Event()
    first_error: list[BaseException] = []
    try:
        key = first.put({"generation": "m1"}, race_key="projection-hook")

        def pause_before_projection(boundary: str) -> None:
            if boundary == "put.candidate_published":
                paused.set()
                assert resume.wait(timeout=5)

        def write_a() -> None:
            try:
                first.put({"generation": "a"}, race_key="projection-hook")
            except BaseException as error:
                first_error.append(error)

        first._cache_blob_store.lifecycle.test_hook = pause_before_projection
        writer = Thread(target=write_a)
        writer.start()
        assert paused.wait(timeout=5)
        assert second.put({"generation": "m2"}, race_key="projection-hook") == key
        resume.set()
        writer.join(timeout=5)

        assert len(first_error) == 1
        assert isinstance(first_error[0], CacheBlobLifecycleConflictError)
        assert second.get(race_key="projection-hook") == {"generation": "m2"}
    finally:
        resume.set()
        second.close()
        first.close()


def test_distinct_key_put_finishes_while_another_payload_is_paused(
    tmp_path: Path,
) -> None:
    """Distinct keys overlap through BlobStore without a facade lock contract."""
    cache = _two_caches(tmp_path / "distinct-key-overlap")[0]
    paused = Event()
    resume = Event()
    second_finished = Event()
    try:
        def pause_first(boundary: str) -> None:
            if boundary == "put.candidate_published" and not paused.is_set():
                paused.set()
                assert resume.wait(timeout=5)

        def put_first() -> None:
            cache.put({"generation": "a"}, race_key="first")

        def put_second() -> None:
            cache.put({"generation": "b"}, race_key="second")
            second_finished.set()

        cache._cache_blob_store.lifecycle.test_hook = pause_first
        first_writer = Thread(target=put_first)
        second_writer = Thread(target=put_second)
        first_writer.start()
        assert paused.wait(timeout=5)
        second_writer.start()
        assert second_finished.wait(timeout=5)
        resume.set()
        first_writer.join(timeout=5)
        second_writer.join(timeout=5)
        assert cache.get(race_key="first") == {"generation": "a"}
        assert cache.get(race_key="second") == {"generation": "b"}
    finally:
        resume.set()
        cache.close()


def test_tombstone_cleanup_debt_is_invisible_to_all_public_cache_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A committed tombstone is absent even while reconciliation owns cleanup."""
    cache, peer = _two_caches(tmp_path / "tombstone-public-surfaces")
    try:
        key = cache.put({"generation": "old"}, race_key="tombstone")
        original_delete = cache._cache_blob_store._delete_or_prove_absent

        def leave_cleanup_debt(_locator) -> None:
            raise OSError("payload cleanup remains for reconciliation")

        monkeypatch.setattr(
            cache._cache_blob_store,
            "_delete_or_prove_absent",
            leave_cleanup_debt,
        )
        with pytest.raises(CacheBlobRecoverableCleanupError):
            cache.invalidate(cache_key=key)
        monkeypatch.setattr(
            cache._cache_blob_store,
            "_delete_or_prove_absent",
            original_delete,
        )

        def reject_payload_open(*_args, **_kwargs):
            raise AssertionError("a tombstone must not be read as payload bytes")

        monkeypatch.setattr(
            cache._cache_blob_store.guarded_handler_io,
            "open_snapshot",
            reject_payload_open,
        )
        assert cache.get(race_key="tombstone") is None
        assert cache.list_entries() == []
        stats = cache.get_stats()
        assert stats["total_entries"] == 0
        assert stats["total_size_mb"] == 0
        cache._cleanup_expired()
        cache._enforce_size_limit()
        tombstone = cache._cache_blob_store.lifecycle_authority.read_entry(key)
        assert tombstone is not None
        report = cache._cache_blob_store.reconcile(apply=False)
        assert report.findings
    finally:
        peer.close()
        cache.close()


def test_authority_read_raises_after_two_unstable_generation_observations(tmp_path, monkeypatch):
    """Generation changes during the engine snapshot exhaust one bounded retry."""
    first, second = _two_caches(tmp_path / "authority-read-conflict")
    try:
        key = first.put({"generation": "m1"}, race_key="unstable-read")
        original = first._cache_blob_store.guarded_handler_io.open_snapshot
        from contextlib import contextmanager
        versions = iter(("m2", "m3"))
        @contextmanager
        def changing_snapshot(*args, **kwargs):
            with original(*args, **kwargs) as snapshot:
                second.put({"generation": next(versions)}, race_key="unstable-read")
                yield snapshot
        monkeypatch.setattr(first._cache_blob_store.guarded_handler_io,
                            "open_snapshot", changing_snapshot)
        with pytest.raises(CacheBlobLifecycleConflictError):
            first.get(key)
    finally:
        second.close()
        first.close()


def test_clear_all_preserves_a_generation_published_after_its_snapshot(
    tmp_path: Path,
) -> None:
    """A clear snapshot must not retire a second instance's new generation."""
    first, second = _two_caches(tmp_path / "clear-race")
    try:
        key = first.put({"generation": "old"}, race_key="clear")
        published = False

        def publish_after_snapshot(boundary: str) -> None:
            nonlocal published
            if boundary == "clear.snapshot_committed" and not published:
                published = True
                assert second.put({"generation": "new"}, race_key="clear") == key

        first._cache_blob_store.lifecycle.test_hook = publish_after_snapshot
        first.clear_all()

        assert second.get(race_key="clear") == {"generation": "new"}
    finally:
        second.close()
        first.close()


def test_invalidate_refuses_to_retire_a_generation_replaced_by_another_instance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalidation operates on the observed generation, never the key alone."""
    first, second = _two_caches(tmp_path / "invalidate-race")
    try:
        key = first.put({"generation": "old"}, race_key="invalidate")

        def replace_before_removal(boundary: str) -> None:
            if boundary == "delete.intent_prepared":
                second.put({"generation": "new"}, race_key="invalidate")

        first._cache_blob_store.lifecycle.test_hook = replace_before_removal
        assert first.invalidate(cache_key=key) is None

        assert second.get(race_key="invalidate") == {"generation": "new"}
    finally:
        second.close()
        first.close()


def test_stale_read_failure_does_not_remove_a_replacement_generation(tmp_path, monkeypatch):
    """A missing payload is a non-destructive integrity miss, never read repair."""
    first, second = _two_caches(tmp_path / "read-race")
    try:
        key = first.put({"generation": "old"}, race_key="read")
        entry = first._cache_blob_store.lifecycle_authority.read_entry(key)
        assert entry is not None
        (first._cache_blob_store.cache_dir / entry.locator).unlink()
        def forbidden_delete(*args, **kwargs):
            raise AssertionError("read failure must not authorize deletion")
        monkeypatch.setattr(first._cache_blob_store, "delete", forbidden_delete)
        assert first.get(key) is None
        second.put({"generation": "new"}, race_key="read")
        assert first.get(key) == {"generation": "new"}
    finally:
        second.close()
        first.close()


def test_pre_promotion_failure_restores_projection_to_previous_authority_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reclaimed failed candidate is never left visible through metadata."""
    cache, peer = _two_caches(tmp_path / "failed-projection")
    try:
        cache.put({"generation": "old"}, race_key="failed-projection")
        original_record = cache._cache_blob_store.lifecycle_authority.record_verification

        def fail_verification(*_args, **_kwargs):
            raise OSError("verification storage unavailable")

        monkeypatch.setattr(
            cache._cache_blob_store.lifecycle_authority,
            "record_verification",
            fail_verification,
        )
        with pytest.raises(OSError, match="verification storage unavailable"):
            cache.put({"generation": "candidate"}, race_key="failed-projection")
        monkeypatch.setattr(
            cache._cache_blob_store.lifecycle_authority,
            "record_verification",
            original_record,
        )

        assert cache.get(race_key="failed-projection") == {"generation": "old"}
    finally:
        peer.close()
        cache.close()


def test_post_promotion_cleanup_failure_preserves_authority_despite_stale_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup debt preserves the winner without requiring projection publication."""
    cache, peer = _two_caches(tmp_path / "post-promotion-projection")
    try:
        key = cache.put({"generation": "old"}, race_key="post-promotion")
        old_entry = cache._cache_blob_store.lifecycle_authority.read_entry(key)
        assert old_entry is not None
        old_path = cache._cache_blob_store.cache_dir / old_entry.locator
        original_delete = cache._cache_blob_store._delete_or_prove_absent

        def fail_old_cleanup(locator):
            candidate = Path(locator)
            if not candidate.is_absolute():
                candidate = cache._cache_blob_store.cache_dir / candidate
            if candidate == old_path:
                raise OSError("old cleanup unavailable")
            return original_delete(locator)

        monkeypatch.setattr(
            cache._cache_blob_store,
            "_delete_or_prove_absent",
            fail_old_cleanup,
        )
        with pytest.raises(CacheBlobRecoverableCleanupError):
            cache.put({"generation": "new"}, race_key="post-promotion")

        authority = cache._cache_blob_store.lifecycle_authority.read_entry(key)
        assert authority is not None
        assert cache._cache_blob_store.cache_dir / authority.locator != old_path
        assert cache.get(race_key="post-promotion") == {"generation": "new"}
    finally:
        peer.close()
        cache.close()


def test_relative_cache_root_projects_an_absolute_guarded_payload_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relative public root must not be prepended to its payload path twice."""
    monkeypatch.chdir(tmp_path)
    cache = cacheness(
        CacheConfig(
            cache_dir="relative-cache",
            cleanup_on_init=False,
        )
    )
    try:
        key = cache.put({"value": "round-trip"}, relative_root=True)
        entry = cache._cache_blob_store.lifecycle_authority.read_entry(key)
        assert entry is not None
        payload_path = cache._cache_blob_store.guarded_handler_io.root / entry.locator
        assert payload_path.is_absolute()
        assert cache.get(relative_root=True) == {"value": "round-trip"}
    finally:
        cache.close()
