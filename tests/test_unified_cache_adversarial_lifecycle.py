"""Deterministic facade schedules for authority-owned lifecycle admission."""

from __future__ import annotations

from pathlib import Path
from threading import Event, Thread

import pytest

from cacheness import CacheConfig, cacheness


def _cache(root: Path):
    """Construct one durable facade without eager compatibility cleanup."""
    return cacheness(
        CacheConfig(
            cache_dir=str(root),
            metadata_backend="sqlite",
            cleanup_on_init=False,
        )
    )


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
