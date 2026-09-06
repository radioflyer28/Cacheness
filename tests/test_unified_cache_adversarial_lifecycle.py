"""Deterministic facade schedules for authority-owned lifecycle admission."""

from __future__ import annotations

from pathlib import Path
from threading import Event, Thread

import pytest
from sqlalchemy import Column, String

from cacheness import CacheConfig, cacheness
from cacheness.custom_metadata import CustomMetadataBase, custom_metadata_model
from cacheness.error_handling import (
    CacheBlobLifecycleConflictError,
    CacheBlobStoreClosedError,
)
from cacheness.metadata import Base
from cacheness.storage import BlobStore


@custom_metadata_model("plan15_linked_metadata")
class Plan15LinkedMetadata(Base, CustomMetadataBase):
    """One custom-metadata model used to prove M1 link ownership."""

    __tablename__ = "plan15_linked_metadata"

    label = Column(String(100), nullable=False)


def _cache(root: Path):
    """Construct one durable facade without eager compatibility cleanup."""
    return cacheness(
        CacheConfig(
            cache_dir=str(root),
            metadata_backend="sqlite",
            cleanup_on_init=False,
        )
    )


def _sqlite_cache(root: Path):
    """Construct a facade whose custom-link table uses exact SQL tokens."""
    custom_metadata_model("plan15_linked_metadata")(Plan15LinkedMetadata)
    return _cache(root)


def _join(thread: Thread) -> None:
    """Join an adversarial worker without hiding a deadlock."""
    thread.join(timeout=5)
    assert not thread.is_alive(), "worker did not finish within the bounded wait"


def test_facade_put_admission_blocks_close_until_authority_promotion_exits(
    tmp_path: Path,
) -> None:
    """An admitted facade put owns the same close drain reference as BlobStore.put."""
    cache = _cache(tmp_path / "facade-put-close")
    entered = Event()
    release = Event()
    put_finished = Event()
    close_finished = Event()
    errors: list[BaseException] = []

    def pause_before_promotion(boundary: str) -> None:
        if boundary == "put.before_promotion":
            entered.set()
            assert release.wait(timeout=5)

    def put() -> None:
        try:
            cache.put({"generation": "m1"}, race_key="facade-close")
        except BaseException as error:  # pragma: no cover - asserted below.
            errors.append(error)
        finally:
            put_finished.set()

    def close() -> None:
        try:
            cache._cache_blob_store.close()
        except BaseException as error:  # pragma: no cover - asserted below.
            errors.append(error)
        finally:
            close_finished.set()

    cache._cache_blob_store.lifecycle.test_hook = pause_before_promotion
    writer = Thread(target=put)
    closer = Thread(target=close)
    try:
        writer.start()
        assert entered.wait(timeout=5)
        closer.start()
        assert not close_finished.wait(timeout=0.2)
        release.set()
        _join(writer)
        _join(closer)
        assert put_finished.is_set()
        assert errors == []
    finally:
        release.set()
        _join(writer)
        _join(closer)
        cache.close()


def test_clear_clear_queued_clear_reopens_after_snapshot_but_cannot_overtake_close(
    tmp_path: Path,
) -> None:
    """Only the active clear owns admission; its queued peer is never counted."""
    store = BlobStore(tmp_path / "clear-admission", backend="json")
    snapshot_entered = Event()
    release_snapshot = Event()
    cleanup_entered = Event()
    release_cleanup = Event()
    unrelated_done = Event()
    close_done = Event()
    first_results: list[int] = []
    second_errors: list[BaseException] = []
    errors: list[BaseException] = []

    def pause_active_clear(boundary: str) -> None:
        if boundary == "clear.snapshot_committed":
            snapshot_entered.set()
            assert release_snapshot.wait(timeout=5)
        if boundary == "clear.before_target_delete":
            cleanup_entered.set()
            assert release_cleanup.wait(timeout=5)

    def first_clear() -> None:
        try:
            first_results.append(store.clear())
        except BaseException as error:  # pragma: no cover - asserted below.
            errors.append(error)

    def second_clear() -> None:
        try:
            store.clear()
        except BaseException as error:  # pragma: no cover - asserted below.
            second_errors.append(error)

    def unrelated_put() -> None:
        try:
            store.put("new", key="unrelated")
        except BaseException as error:  # pragma: no cover - asserted below.
            errors.append(error)
        finally:
            unrelated_done.set()

    def close() -> None:
        try:
            store.close()
        except BaseException as error:  # pragma: no cover - asserted below.
            errors.append(error)
        finally:
            close_done.set()

    store.lifecycle.test_hook = pause_active_clear
    first = Thread(target=first_clear)
    second = Thread(target=second_clear)
    unrelated = Thread(target=unrelated_put)
    closer = Thread(target=close)
    try:
        store.put("old", key="target")
        first.start()
        assert snapshot_entered.wait(timeout=5)
        second.start()
        assert store._instance_admission.in_flight == 1
        assert len(store._instance_admission._clear_queue) == 1

        release_snapshot.set()
        assert cleanup_entered.wait(timeout=5)
        unrelated.start()
        assert unrelated_done.wait(timeout=5)
        assert store.get("unrelated") == "new"

        closer.start()
        assert not close_done.wait(timeout=0.2)
        with pytest.raises(CacheBlobStoreClosedError):
            store.get("later")
        release_cleanup.set()

        _join(first)
        _join(second)
        _join(unrelated)
        _join(closer)
        assert first_results == [1]
        assert errors == []
        assert len(second_errors) == 1
        assert isinstance(second_errors[0], CacheBlobStoreClosedError)
    finally:
        release_snapshot.set()
        release_cleanup.set()
        _join(first)
        _join(second)
        _join(unrelated)
        _join(closer)
        store.close()


def test_linked_m1_survives_a_pre_promotion_verification_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M2 must not retire M1 links until its own authority promotion wins."""
    cache = _sqlite_cache(tmp_path / "linked-m1-verification-failure")
    try:
        key = cache.put(
            {"generation": "m1"},
            custom_metadata=Plan15LinkedMetadata(label="m1"),
            race_key="linked-m1",
        )
        before = cache.metadata_backend.get_entry(key)
        assert before is not None
        original_record = cache._cache_blob_store.lifecycle_authority.record_verification

        def fail_verification(*_args, **_kwargs) -> None:
            raise OSError("forced verification failure")

        monkeypatch.setattr(
            cache._cache_blob_store.lifecycle_authority,
            "record_verification",
            fail_verification,
        )
        with pytest.raises(OSError, match="forced verification failure"):
            cache.put(
                {"generation": "m2"},
                custom_metadata=Plan15LinkedMetadata(label="m2"),
                race_key="linked-m1",
            )
        monkeypatch.setattr(
            cache._cache_blob_store.lifecycle_authority,
            "record_verification",
            original_record,
        )

        after = cache.metadata_backend.get_entry(key)
        assert after is not None
        assert after["metadata"]["actual_path"] == before["metadata"]["actual_path"]
        assert cache.get(race_key="linked-m1") == {"generation": "m1"}
        assert cache.get_custom_metadata_for_entry(cache_key=key)[
            "plan15_linked_metadata"
        ].label == "m1"
    finally:
        cache.close()


def test_two_pending_candidates_preserve_m1_links_until_one_promotes(
    tmp_path: Path,
) -> None:
    """Peer candidates cannot publish or adopt each other's projection token."""
    first = _sqlite_cache(tmp_path / "two-pending-candidates")
    second = _sqlite_cache(tmp_path / "two-pending-candidates")
    first_entered = Event()
    second_entered = Event()
    release = Event()
    results: list[str] = []
    errors: list[BaseException] = []
    try:
        key = first.put(
            {"generation": "m1"},
            custom_metadata=Plan15LinkedMetadata(label="m1"),
            race_key="two-pending",
        )
        before = first.metadata_backend.get_entry(key)
        assert before is not None

        def pause_first(boundary: str) -> None:
            if boundary == "put.before_promotion":
                first_entered.set()
                assert release.wait(timeout=5)

        def pause_second(boundary: str) -> None:
            if boundary == "put.before_promotion":
                second_entered.set()
                assert release.wait(timeout=5)

        def put(cache, label: str) -> None:
            try:
                results.append(
                    cache.put(
                        {"generation": label},
                        custom_metadata=Plan15LinkedMetadata(label=label),
                        race_key="two-pending",
                    )
                )
            except BaseException as error:  # pragma: no cover - asserted below.
                errors.append(error)

        first._cache_blob_store.lifecycle.test_hook = pause_first
        second._cache_blob_store.lifecycle.test_hook = pause_second
        writer_a = Thread(target=put, args=(first, "a"))
        writer_b = Thread(target=put, args=(second, "b"))
        writer_a.start()
        writer_b.start()
        assert first_entered.wait(timeout=5)
        assert second_entered.wait(timeout=5)

        pending = first.metadata_backend.get_entry(key)
        assert pending is not None
        assert pending["metadata"]["actual_path"] == before["metadata"]["actual_path"]
        assert first.get_custom_metadata_for_entry(cache_key=key)[
            "plan15_linked_metadata"
        ].label == "m1"

        release.set()
        _join(writer_a)
        _join(writer_b)
        assert results == [key]
        assert len(errors) == 1
        assert isinstance(errors[0], CacheBlobLifecycleConflictError)
        assert first.get(race_key="two-pending") in (
            {"generation": "a"},
            {"generation": "b"},
        )
        assert first.get_custom_metadata_for_entry(cache_key=key)[
            "plan15_linked_metadata"
        ].label in {"a", "b"}
    finally:
        release.set()
        _join(writer_a)
        _join(writer_b)
        second.close()
        first.close()
