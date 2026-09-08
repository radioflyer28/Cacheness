"""Derived read-model contracts replacing runtime custom-metadata sessions."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from cacheness.storage.composition import BackendRole, ProjectionRole, ProjectionSink
from cacheness.storage.projections import (
    ProjectionCheckpoint,
    ProjectionCheckpointError,
    ProjectionController,
)


@dataclass(frozen=True)
class _Page:
    revision: int
    entries: tuple[tuple[str, str], ...]
    cursor: str | None
    exhausted: bool


class _Source:
    projection_store_id = "custom-read-model-store"

    def __init__(self, pages: tuple[_Page, ...]) -> None:
        self.pages = pages
        self.requests: list[str | None] = []

    def query_catalog(self, cursor: str | None) -> _Page:
        self.requests.append(cursor)
        return self.pages[len(self.requests) - 1]


class _ReadModel:
    """A projection-only export with no lifecycle mutation methods."""

    projection_name = "custom-read-model"

    def __init__(self) -> None:
        self.applied: list[tuple[str, str]] = []
        self.checkpoints: list[ProjectionCheckpoint] = []

    def apply_projection_batch(self, batch: object) -> None:
        self.applied.extend(batch.entries)

    def save_projection_checkpoint(self, checkpoint: object) -> None:
        self.checkpoints.append(checkpoint)

    def load_projection_checkpoint(self) -> ProjectionCheckpoint | None:
        return self.checkpoints[-1] if self.checkpoints else None


def test_custom_read_model_uses_the_shared_projection_role_and_sink_protocol() -> None:
    """An export sink is derived-only and cannot become another authority."""
    sink = _ReadModel()

    assert ProjectionRole is BackendRole.PROJECTION
    assert isinstance(sink, ProjectionSink)
    assert not hasattr(sink, "promote_catalog")
    assert not hasattr(sink, "query_complete")


def test_custom_read_model_pulls_catalog_pages_and_checkpoints_after_apply() -> None:
    """A successful projection copies committed state then records its cursor."""
    source = _Source(
        (
            _Page(2, (("key-a", "generation-a"),), "next", False),
            _Page(2, (), None, True),
        )
    )
    sink = _ReadModel()

    result = ProjectionController(source, sink, page_size=1).pull()

    assert result.exhausted is True
    assert source.requests == [None, "next"]
    assert sink.applied == [("key-a", "generation-a")]
    assert [checkpoint.cursor for checkpoint in sink.checkpoints] == ["next", None]


def test_custom_read_model_rejects_a_checkpoint_from_another_canonical_snapshot() -> None:
    """Derived state cannot resume against a changed canonical revision."""
    checkpoint = ProjectionCheckpoint(
        source_store_id="custom-read-model-store",
        store_epoch=1,
        schema_id="catalog",
        schema_fingerprint="catalog",
        query_fingerprint="all",
        revision=2,
        cursor="next",
    )

    with pytest.raises(ProjectionCheckpointError, match="stale for revision"):
        checkpoint.assert_matches(
            source_store_id="custom-read-model-store",
            store_epoch=1,
            schema_id="catalog",
            schema_fingerprint="catalog",
            query_fingerprint="all",
            revision=3,
        )
