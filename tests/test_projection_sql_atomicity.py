"""Derived SQL-style projection rebuild and checkpoint failure regressions."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from cacheness.storage.projections import (
    ProjectionCapabilityError,
    ProjectionController,
    ProjectionRebuildError,
    validate_rebuild_mode,
)


@dataclass(frozen=True)
class _Page:
    revision: int
    entries: tuple[tuple[str, str], ...]
    cursor: str | None
    exhausted: bool


class _Source:
    projection_store_id = "sql-projection-store"

    def __init__(self, page: _Page) -> None:
        self.page = page

    def query_catalog(self, cursor: str | None) -> _Page:
        assert cursor is None
        return self.page


class _SqlProjection:
    projection_name = "sql-read-model"
    topology_capabilities = {
        "projection_rebuild": True,
        "offline_rebuild": True,
        "online_rebuild": False,
    }

    def __init__(self, *, fail_candidate: bool = False) -> None:
        self.fail_candidate = fail_candidate
        self.rows: list[tuple[str, str]] = [("old", "generation")]
        self.checkpoints: list[object] = []
        self.published: object | None = None
        self.discarded: object | None = None

    def apply_projection_batch(self, batch: object) -> None:
        if self.fail_candidate:
            raise OSError("candidate SQL write failed")
        self.rows.extend(batch.entries)

    def save_projection_checkpoint(self, checkpoint: object) -> None:
        self.checkpoints.append(checkpoint)

    def load_projection_checkpoint(self) -> object | None:
        return self.checkpoints[-1] if self.checkpoints else None

    def begin_isolated_rebuild(self) -> "_SqlProjection":
        return _SqlProjection(fail_candidate=self.fail_candidate)

    def publish_isolated_rebuild(self, candidate: "_SqlProjection", checkpoint: object) -> None:
        self.published = (tuple(candidate.rows), checkpoint)

    def discard_isolated_rebuild(self, candidate: "_SqlProjection") -> None:
        self.discarded = candidate


def test_sql_projection_rebuild_requires_an_explicit_offline_publication_capability() -> None:
    """A local projection is never silently claimed to be online-rebuild safe."""
    validate_rebuild_mode({"projection_rebuild": True, "offline_rebuild": True}, requested="offline")

    with pytest.raises(ProjectionCapabilityError, match="online rebuild"):
        validate_rebuild_mode({"online_rebuild": False}, requested="online")


def test_sql_projection_rebuild_publishes_only_after_an_isolated_candidate_completes() -> None:
    """The old read model remains untouched until the replacement has a checkpoint."""
    sink = _SqlProjection()
    result = ProjectionController(
        _Source(_Page(4, (("new", "generation"),), None, True)), sink
    ).rebuild()

    assert result.exhausted is True
    assert sink.published is not None
    assert sink.published[0] == (("old", "generation"), ("new", "generation"))


def test_failed_sql_projection_rebuild_discards_the_candidate_without_publishing() -> None:
    """A failed derived rebuild cannot replace the prior committed read model."""
    sink = _SqlProjection(fail_candidate=True)
    controller = ProjectionController(
        _Source(_Page(4, (("new", "generation"),), None, True)), sink
    )

    with pytest.raises(ProjectionRebuildError, match="Isolated projection rebuild failed"):
        controller.rebuild()

    assert sink.published is None
    assert sink.discarded is not None
    assert sink.rows == [("old", "generation")]
