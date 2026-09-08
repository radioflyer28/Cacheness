"""Projection checkpoint and failure contracts at the derived-only boundary."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from cacheness.storage.composition import BackendRole, ProjectionRole, ProjectionSink
from cacheness.storage.projections import ProjectionController


@dataclass(frozen=True)
class _Page:
    revision: int
    entries: tuple[tuple[str, str], ...]
    cursor: str | None
    exhausted: bool


class _Source:
    projection_store_id = "mutation-contract-store"

    def __init__(self, page: _Page) -> None:
        self.page = page

    def query_catalog(self, cursor: str | None) -> _Page:
        assert cursor is None
        return self.page


class _OrderedSink:
    projection_name = "ordered-sink"

    def __init__(self, *, fail_apply: bool = False) -> None:
        self.fail_apply = fail_apply
        self.events: list[tuple[str, object]] = []

    def apply_projection_batch(self, batch: object) -> None:
        self.events.append(("apply", batch.batch_id))
        if self.fail_apply:
            raise OSError("projection write failed")

    def save_projection_checkpoint(self, checkpoint: object) -> None:
        self.events.append(("checkpoint", checkpoint.cursor))

    def load_projection_checkpoint(self) -> None:
        return None


def test_projection_controller_uses_the_single_shared_projection_role() -> None:
    """Projection code has no alternative authority role vocabulary."""
    controller = ProjectionController(
        _Source(_Page(3, (), None, True)), _OrderedSink()
    )

    assert ProjectionRole is BackendRole.PROJECTION
    assert controller.role is BackendRole.PROJECTION
    assert isinstance(controller.sink, ProjectionSink)


def test_checkpoint_is_recorded_only_after_the_projection_batch_applies() -> None:
    """A failed derived write leaves the prior checkpoint untouched."""
    sink = _OrderedSink(fail_apply=True)
    controller = ProjectionController(
        _Source(_Page(3, (("key", "generation"),), "next", False)), sink
    )

    with pytest.raises(OSError, match="projection write failed"):
        controller.pull()

    assert [event[0] for event in sink.events] == ["apply"]


def test_projection_batch_replay_converges_without_duplicate_effects() -> None:
    """The same canonical page is safe to replay after an interrupted consumer."""
    page = _Page(3, (("key", "generation"),), None, True)
    sink = _OrderedSink()

    ProjectionController(_Source(page), sink).apply_page(page)
    ProjectionController(_Source(page), sink).apply_page(page)

    assert [event[0] for event in sink.events] == ["apply", "checkpoint", "checkpoint"]
