"""Regression coverage for UnifiedCache's authority-backed policy boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from cacheness import CacheConfig, cacheness
from cacheness.error_handling import CacheBlobLifecycleConflictError


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
