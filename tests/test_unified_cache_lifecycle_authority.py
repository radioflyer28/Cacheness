"""Regression coverage for UnifiedCache's authority-backed policy boundary."""

from __future__ import annotations

from pathlib import Path
from threading import Event, Thread

import pytest
from sqlalchemy import Column, String

from cacheness import CacheConfig, cacheness
from cacheness.custom_metadata import CustomMetadataBase, custom_metadata_model
from cacheness.error_handling import (
    CacheBlobLifecycleConflictError,
    CacheBlobRecoverableCleanupError,
)
from cacheness.metadata import Base


@custom_metadata_model("projection_race")
class ProjectionRaceMetadata(Base, CustomMetadataBase):
    """Minimal linked metadata used to prove projection-token ownership."""

    __tablename__ = "custom_projection_race_metadata"

    label = Column(String(100), nullable=False)


def _two_caches(root: Path):
    """Open independent cache-policy facades over one durable cache root."""
    configuration = CacheConfig(
        cache_dir=str(root), metadata_backend="json", cleanup_on_init=False
    )
    return cacheness(configuration), cacheness(configuration)


def _two_sqlite_caches(root: Path):
    """Open independent facades sharing SQLite compatibility projections."""
    configuration = CacheConfig(
        cache_dir=str(root),
        metadata_backend="sqlite",
        cleanup_on_init=False,
        store_cache_key_params=True,
    )
    return cacheness(configuration), cacheness(configuration)


def test_stale_projection_removal_is_a_noop_after_a_peer_replaces_its_token(
    tmp_path: Path,
) -> None:
    """A stale facade must not delete a peer's newer projection by cache key alone."""
    first, second = _two_sqlite_caches(tmp_path / "stale-projection-removal")
    try:
        key = first.put({"generation": "old"}, race_key="stale-removal")
        old_projection = first.metadata_backend.get_entry(key)
        assert old_projection is not None
        old_locator = old_projection["metadata"]["actual_path"]

        assert second.put({"generation": "new"}, race_key="stale-removal") == key

        outcome = first.metadata_backend.conditional_projection_mutation(
            key,
            expected_locator=old_locator,
            replacement=None,
        )

        assert outcome.status == "mismatch"
        assert second.get(race_key="stale-removal") == {"generation": "new"}
        current = second.metadata_backend.get_entry(key)
        assert current is not None
        assert current["metadata"]["actual_path"] != old_locator
    finally:
        second.close()
        first.close()


def test_stale_absence_teardown_preserves_m2_custom_metadata_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An M1-observed absence cannot remove M2's row or its newly linked data."""
    first, second = _two_sqlite_caches(tmp_path / "stale-absence-custom-links")
    try:
        key = first.put(
            {"generation": "old"},
            custom_metadata=ProjectionRaceMetadata(label="old"),
            race_key="stale-absence",
        )
        old_snapshot = first._cache_blob_store.lifecycle_authority.read_entry(key)
        assert old_snapshot is not None
        assert first._cache_blob_store.delete(key, expected=old_snapshot.expectation)
        tombstone = first._cache_blob_store.lifecycle_authority.read_entry(key)
        if tombstone is not None:
            assert first._cache_blob_store.delete(
                key, expected=tombstone.expectation
            )
        assert first._cache_blob_store.lifecycle_authority.read_entry(key) is None

        original_snapshot_manifest = first._authority_snapshot_manifest
        published = False

        def publish_m2_after_absence(cache_key: str):
            nonlocal published
            snapshot, manifest = original_snapshot_manifest(cache_key)
            if cache_key == key and not published:
                published = True
                assert second.put(
                    {"generation": "new"},
                    custom_metadata=ProjectionRaceMetadata(label="new"),
                    race_key="stale-absence",
                ) == key
            return snapshot, manifest

        monkeypatch.setattr(
            first, "_authority_snapshot_manifest", publish_m2_after_absence
        )
        outcome = first._sync_authority_projection(key)

        assert outcome.status == "mismatch"
        assert second.get(race_key="stale-absence") == {"generation": "new"}
        linked = second.get_custom_metadata_for_entry(cache_key=key)
        assert linked["projection_race"].label == "new"
        assert [entry["cache_key"] for entry in second.list_entries()] == [key]
    finally:
        second.close()
        first.close()


def test_failed_first_put_repair_preserves_peer_projection_and_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed candidate repair must not undo M2 after the authority stays absent."""
    first, second = _two_sqlite_caches(tmp_path / "failed-first-put-custom-links")
    try:
        original_record = first._cache_blob_store.lifecycle_authority.record_verification
        original_repair = first._repair_projection_after_failed_put
        published = False

        def fail_verification(*_args, **_kwargs):
            raise OSError("forced verification failure")

        def publish_before_repair(cache_key: str) -> None:
            nonlocal published
            if not published:
                published = True
                assert second.put(
                    {"generation": "m2"},
                    custom_metadata=ProjectionRaceMetadata(label="m2"),
                    race_key="failed-first-put",
                ) == cache_key
            original_repair(cache_key)

        monkeypatch.setattr(
            first._cache_blob_store.lifecycle_authority,
            "record_verification",
            fail_verification,
        )
        monkeypatch.setattr(first, "_repair_projection_after_failed_put", publish_before_repair)
        with pytest.raises(OSError, match="forced verification failure"):
            first.put({"generation": "candidate"}, race_key="failed-first-put")
        monkeypatch.setattr(
            first._cache_blob_store.lifecycle_authority,
            "record_verification",
            original_record,
        )

        assert second.get(race_key="failed-first-put") == {"generation": "m2"}
        linked = second.get_custom_metadata_for_entry(
            cache_key=second._create_cache_key({"race_key": "failed-first-put"})
        )
        assert linked["projection_race"].label == "m2"
    finally:
        second.close()
        first.close()


def test_replacement_before_projection_hook_preserves_m2(
    tmp_path: Path,
) -> None:
    """An A write that captured M1 loses cleanly when B replaces M1 first."""
    first, second = _two_sqlite_caches(tmp_path / "replacement-before-projection")
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


def test_replacement_before_custom_link_rejects_stale_metadata_insert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A promoted locator must still be current when its links are inserted."""
    first, second = _two_sqlite_caches(tmp_path / "replacement-before-custom-link")
    try:
        original_store = first._store_custom_metadata
        replaced = False

        def replace_before_link(cache_key: str, custom_metadata, **kwargs) -> None:
            nonlocal replaced
            if not replaced:
                replaced = True
                assert second.put(
                    {"generation": "m2"},
                    custom_metadata=ProjectionRaceMetadata(label="m2"),
                    race_key="custom-link",
                ) == cache_key
            original_store(cache_key, custom_metadata, **kwargs)

        monkeypatch.setattr(first, "_store_custom_metadata", replace_before_link)
        with pytest.raises(CacheBlobLifecycleConflictError):
            first.put(
                {"generation": "a"},
                custom_metadata=ProjectionRaceMetadata(label="a"),
                race_key="custom-link",
            )

        key = second._create_cache_key({"race_key": "custom-link"})
        assert second.get(race_key="custom-link") == {"generation": "m2"}
        assert second.get_custom_metadata_for_entry(cache_key=key)["projection_race"].label == "m2"
        assert [item.label for item in second.query_custom("projection_race")] == ["m2"]
    finally:
        second.close()
        first.close()


def test_distinct_key_put_finishes_while_another_payload_is_paused(
    tmp_path: Path,
) -> None:
    """The retained facade lock cannot serialize unrelated payload publication."""
    cache = _two_sqlite_caches(tmp_path / "distinct-key-overlap")[0]
    paused = Event()
    resume = Event()
    second_finished = Event()
    try:
        assert hasattr(cache, "_lock")

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
    cache, peer = _two_sqlite_caches(tmp_path / "tombstone-public-surfaces")
    try:
        key = cache.put(
            {"generation": "old"},
            custom_metadata=ProjectionRaceMetadata(label="old"),
            race_key="tombstone",
        )
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
        assert cache.query_meta(race_key="tombstone") == []
        assert cache.get_custom_metadata_for_entry(cache_key=key) == {}
        assert cache.query_custom("projection_race") == []
        with cache.query_custom_session("projection_race") as query:
            assert query.all() == []
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


def test_authority_read_repairs_an_m1_projection_then_returns_coherent_m2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One bounded retry returns a replacement, never an M1/M2 hybrid pair."""
    first, second = _two_sqlite_caches(tmp_path / "authority-read-m2")
    try:
        key = first.put({"generation": "m1"}, race_key="coherent-read")
        first.metadata_backend.remove_entry(key)
        original_sync = first._sync_authority_projection
        published = False

        def publish_m2_before_repair(cache_key: str, **kwargs):
            nonlocal published
            if cache_key == key and not published:
                published = True
                assert second.put({"generation": "m2"}, race_key="coherent-read") == key
            return original_sync(cache_key, **kwargs)

        monkeypatch.setattr(first, "_sync_authority_projection", publish_m2_before_repair)
        snapshot, projection = first._authority_snapshot_entry(key)

        assert snapshot is not None and projection is not None
        assert Path(projection["metadata"]["actual_path"]).name.startswith(
            snapshot.generation
        )
        assert first.get(race_key="coherent-read") == {"generation": "m2"}
    finally:
        second.close()
        first.close()


def test_authority_read_raises_after_two_unstable_generation_observations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forever-changing authority cannot leak a mismatched read pair."""
    first, second = _two_sqlite_caches(tmp_path / "authority-read-conflict")
    try:
        key = first.put({"generation": "m1"}, race_key="unstable-read")
        first.metadata_backend.remove_entry(key)
        original_sync = first._sync_authority_projection
        replacements = iter(("m2", "m3"))

        def replace_on_each_repair(cache_key: str, **kwargs):
            if cache_key == key:
                assert second.put(
                    {"generation": next(replacements)}, race_key="unstable-read"
                ) == key
            return original_sync(cache_key, **kwargs)

        monkeypatch.setattr(first, "_sync_authority_projection", replace_on_each_repair)
        with pytest.raises(CacheBlobLifecycleConflictError, match="did not stabilize"):
            first._authority_snapshot_entry(key)
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
        assert second.metadata_backend.get_entry(key) is not None
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
        with pytest.raises(CacheBlobLifecycleConflictError):
            first.invalidate(cache_key=key)

        assert second.get(race_key="invalidate") == {"generation": "new"}
        assert second.metadata_backend.get_entry(key) is not None
    finally:
        second.close()
        first.close()


def test_stale_read_failure_does_not_remove_a_replacement_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure cleanup may retire only the exact snapshot it failed to read."""
    first, second = _two_caches(tmp_path / "read-race")
    try:
        key = first.put({"generation": "old"}, race_key="read")
        entry = first._cache_blob_store.get_metadata(key)
        assert entry is not None
        payload = first._cache_blob_store.cache_dir / entry["metadata"]["actual_path"]
        payload.unlink()

        original_delete = first._cache_blob_store.delete

        def replace_before_cleanup(cache_key: str, **kwargs) -> bool:
            if cache_key == key:
                second.put({"generation": "new"}, race_key="read")
            return original_delete(cache_key, **kwargs)

        monkeypatch.setattr(first._cache_blob_store, "delete", replace_before_cleanup)
        assert first.get(race_key="read") is None
        assert second.get(race_key="read") == {"generation": "new"}
        assert second.metadata_backend.get_entry(key) is not None
    finally:
        second.close()
        first.close()


def test_projection_sync_never_pairs_an_old_locator_with_a_new_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A projection must render fields from one authority observation only."""
    first, second = _two_caches(tmp_path / "projection-snapshot-race")
    try:
        key = first.put({"generation": "old"}, race_key="projection")
        old_snapshot = first._cache_blob_store.lifecycle_authority.read_entry(key)
        assert old_snapshot is not None
        original_snapshot_manifest = first._authority_snapshot_manifest
        replaced = False

        def replace_after_snapshot(cache_key: str):
            nonlocal replaced
            snapshot, manifest = original_snapshot_manifest(cache_key)
            if cache_key == key and not replaced:
                replaced = True
                assert second.put({"generation": "new"}, race_key="projection") == key
            return snapshot, manifest

        monkeypatch.setattr(
            first, "_authority_snapshot_manifest", replace_after_snapshot
        )
        first._sync_authority_projection(key)

        projection = first.metadata_backend.get_entry(key)
        assert projection is not None
        metadata = projection["metadata"]
        assert not Path(metadata["actual_path"]).name.startswith(old_snapshot.generation)

        # The stale M1 observation cannot replace M2 at the projection boundary.
        assert first.get(race_key="projection") == {"generation": "new"}
        assert second.get(race_key="projection") == {"generation": "new"}
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
        key = cache.put({"generation": "old"}, race_key="failed-projection")
        before = cache.metadata_backend.get_entry(key)
        assert before is not None
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

        after = cache.metadata_backend.get_entry(key)
        assert after is not None
        assert after["metadata"]["actual_path"] == before["metadata"]["actual_path"]
        assert Path(after["metadata"]["actual_path"]).exists()
        assert cache.get(race_key="failed-projection") == {"generation": "old"}
    finally:
        peer.close()
        cache.close()


def test_post_promotion_cleanup_failure_keeps_projection_on_new_authority_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recoverable cleanup errors must not leave a projection on old bytes."""
    cache, peer = _two_caches(tmp_path / "post-promotion-projection")
    try:
        key = cache.put({"generation": "old"}, race_key="post-promotion")
        old_entry = cache.metadata_backend.get_entry(key)
        assert old_entry is not None
        old_path = Path(old_entry["metadata"]["actual_path"])
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
        projection = cache.metadata_backend.get_entry(key)
        assert authority is not None and projection is not None
        assert Path(projection["metadata"]["actual_path"]) == (
            cache._cache_blob_store.cache_dir / authority.locator
        )
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
            metadata_backend="json",
            cleanup_on_init=False,
        )
    )
    try:
        key = cache.put({"value": "round-trip"}, relative_root=True)
        entry = cache.metadata_backend.get_entry(key)
        assert entry is not None
        payload_path = Path(entry["metadata"]["actual_path"])
        assert payload_path.is_absolute()
        assert cache.get(relative_root=True) == {"value": "round-trip"}
    finally:
        cache.close()
