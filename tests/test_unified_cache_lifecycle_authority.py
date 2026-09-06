"""Regression coverage for UnifiedCache's authority-backed policy boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from cacheness import CacheConfig, cacheness
from cacheness.error_handling import (
    CacheBlobLifecycleConflictError,
    CacheBlobRecoverableCleanupError,
)


def _two_caches(root: Path):
    """Open independent cache-policy facades over one durable cache root."""
    configuration = CacheConfig(
        cache_dir=str(root), metadata_backend="json", cleanup_on_init=False
    )
    return cacheness(configuration), cacheness(configuration)


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
        assert Path(metadata["actual_path"]).name.startswith(old_snapshot.generation)
        assert metadata.get("authority_generation") == old_snapshot.generation

        # The old observation cannot be mistaken for the replacement. A normal
        # facade read refreshes the projection and must not retire M2.
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
