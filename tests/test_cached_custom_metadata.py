"""Cached derived-read-model contracts without runtime ORM authority state."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from cacheness.error_handling import CacheBlobCommittedPartialError
from cacheness.storage.composition import ProjectionSink
from cacheness.storage.projections import ProjectionController, ProjectionStatus


@dataclass(frozen=True)
class _Page:
    revision: int
    entries: tuple[tuple[str, str], ...]
    cursor: str | None
    exhausted: bool


class _Source:
    projection_store_id = "cached-read-model-store"

    def __init__(self, page: _Page) -> None:
        self.page = page

    def query_catalog(self, cursor: str | None) -> _Page:
        assert cursor is None
        return self.page


class _CachedReadModel:
    projection_name = "cached-read-model"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.rows: list[tuple[str, str]] = []
        self.checkpoints: list[object] = []

    def apply_projection_batch(self, batch: object) -> None:
        if self.fail:
            raise OSError("read model unavailable")
        self.rows.extend(batch.entries)

    def save_projection_checkpoint(self, checkpoint: object) -> None:
        self.checkpoints.append(checkpoint)

    def load_projection_checkpoint(self) -> object | None:
        return self.checkpoints[-1] if self.checkpoints else None


def test_cached_read_model_deduplicates_a_replayed_committed_batch() -> None:
    """Idempotency belongs to the projection sink, never the authority catalog."""
    page = _Page(1, (("key", "generation"),), None, True)
    sink = _CachedReadModel()

    ProjectionController(_Source(page), sink).apply_page(page)
    ProjectionController(_Source(page), sink).apply_page(page)

    assert isinstance(sink, ProjectionSink)
    assert sink.rows == [("key", "generation")]
    assert len(sink.checkpoints) == 2


def test_cached_read_model_best_effort_failure_preserves_the_committed_receipt() -> None:
    """A stale derived cache cannot roll back an authority-owned commit."""
    receipt = object()
    sink = _CachedReadModel(fail=True)
    controller = ProjectionController(
        _Source(_Page(1, (("key", "generation"),), "next", False)), sink
    )

    result = controller.best_effort(receipt)

    assert result.receipt is receipt
    assert result.projection_status == ProjectionStatus.DIRTY.value
    assert sink.checkpoints == []


def test_cached_read_model_refresh_reports_committed_partial_with_receipt() -> None:
    """Requested recovery reports remaining work without revoking a committed blob."""
    receipt = object()
    controller = ProjectionController(
        _Source(_Page(1, (("key", "generation"),), "next", False)),
        _CachedReadModel(fail=True),
    )

    with pytest.raises(CacheBlobCommittedPartialError) as raised:
        controller.refresh(receipt)

    assert raised.value.receipt is receipt
    assert raised.value.remaining_cursor == "next"
