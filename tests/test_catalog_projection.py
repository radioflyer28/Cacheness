"""Red contracts for bounded, derived-only catalog projection synchronization."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec

import pytest


PROJECTION_MODULE = "cacheness.storage.projections"


def _projections():
    assert find_spec(PROJECTION_MODULE) is not None, (
        "Phase 4 must provide a bounded, checkpointed derived projection adapter"
    )
    return import_module(PROJECTION_MODULE)


@dataclass(frozen=True)
class _Page:
    revision: int
    entries: tuple[tuple[str, str], ...]
    cursor: str | None
    exhausted: bool


class _Authority:
    def __init__(self, pages: tuple[_Page, ...]) -> None:
        self.pages = pages
        self.requests: list[str | None] = []

    def query_catalog(self, cursor: str | None) -> _Page:
        self.requests.append(cursor)
        return self.pages[len(self.requests) - 1]


class _ProjectionSink:
    def __init__(self, *, fail_on_apply: bool = False) -> None:
        self.applied: list[tuple[str, str]] = []
        self.checkpoints: list[str | None] = []
        self.fail_on_apply = fail_on_apply
        self.published: tuple[tuple[str, str], ...] = (("old", "generation"),)

    def apply(self, entries: tuple[tuple[str, str], ...]) -> None:
        if self.fail_on_apply:
            raise OSError("projection unavailable")
        self.applied.extend(entries)

    def checkpoint(self, cursor: str | None) -> None:
        self.checkpoints.append(cursor)


def test_pull_is_bounded_idempotent_and_checkpoints_after_apply() -> None:
    projections = _projections()
    authority = _Authority((_Page(4, (("a", "1"),), "cursor-a", False), _Page(4, (), None, True)))
    sink = _ProjectionSink()

    result = projections.CatalogProjection(authority, sink, page_size=1).pull()

    assert authority.requests == [None, "cursor-a"]
    assert sink.applied == [("a", "1")]
    assert sink.checkpoints == ["cursor-a", None]
    assert result.exhausted is True


def test_duplicate_page_replay_converges_without_duplicate_projection_effects() -> None:
    projections = _projections()
    page = _Page(4, (("a", "1"),), "cursor-a", False)
    sink = _ProjectionSink()

    projections.CatalogProjection(_Authority((page,)), sink).apply_page(page)
    projections.CatalogProjection(_Authority((page,)), sink).apply_page(page)

    assert sink.applied == [("a", "1")]


def test_explicit_refresh_failure_keeps_canonical_receipt_and_remaining_work() -> None:
    projections = _projections()
    receipt = object()
    authority = _Authority((_Page(4, (("a", "1"),), "cursor-a", False),))

    with pytest.raises(projections.CommittedPartialProjectionError) as raised:
        projections.CatalogProjection(authority, _ProjectionSink(fail_on_apply=True)).refresh(receipt)

    assert raised.value.receipt is receipt
    assert raised.value.remaining_cursor == "cursor-a"


def test_best_effort_projection_failure_never_revokes_a_committed_blob() -> None:
    projections = _projections()
    receipt = object()
    authority = _Authority((_Page(4, (("a", "1"),), "cursor-a", False),))

    result = projections.CatalogProjection(authority, _ProjectionSink(fail_on_apply=True)).best_effort(receipt)

    assert result.receipt is receipt
    assert result.projection_status == "dirty"


def test_interrupted_rebuild_never_replaces_prior_published_projection() -> None:
    projections = _projections()
    sink = _ProjectionSink()
    authority = _Authority((_Page(4, (("new", "generation"),), "cursor-a", False),))

    with pytest.raises(projections.ProjectionRebuildError):
        projections.CatalogProjection(authority, sink).rebuild()

    assert sink.published == (("old", "generation"),)


def test_rebuild_mode_is_capability_qualified_not_a_universal_online_claim() -> None:
    projections = _projections()

    assert projections.validate_rebuild_mode({"offline_rebuild": True}, requested="offline") is None
    with pytest.raises(projections.ProjectionCapabilityError):
        projections.validate_rebuild_mode({"online_rebuild": False}, requested="online")


def test_checkpoint_binds_the_complete_canonical_snapshot_identity() -> None:
    projections = _projections()

    checkpoint = projections.ProjectionCheckpoint(
        source_store_id="store-a",
        store_epoch=1,
        schema_id="application",
        schema_fingerprint="schema-fingerprint",
        query_fingerprint="query-fingerprint",
        revision=4,
        cursor="cursor-a",
    )

    assert checkpoint.source_store_id == "store-a"
    assert checkpoint.revision == 4
    with pytest.raises(projections.ProjectionCheckpointError):
        checkpoint.assert_matches(
            source_store_id="store-a",
            store_epoch=1,
            schema_id="application",
            schema_fingerprint="schema-fingerprint",
            query_fingerprint="query-fingerprint",
            revision=5,
        )


def test_post_commit_projection_failure_preserves_the_authority_receipt() -> None:
    from cacheness.storage.backends.blob_backends import InMemoryBlobBackend
    from cacheness.storage.blob_store import BlobStore
    from cacheness.storage.catalog import CatalogField, CatalogQuery, CatalogSchema
    from cacheness.storage.composition import StoreTopology
    from cacheness.storage.memory_lifecycle_authority import InMemoryLifecycleAuthority

    class FailingSink:
        projection_name = "external-index"
        projection_query = CatalogQuery()
        projection_schema = CatalogSchema(
            (CatalogField("kind", "string"),), schema_id="external-index"
        )

        def apply_projection_batch(self, batch: object) -> None:
            raise OSError("projection unavailable")

        def save_projection_checkpoint(self, checkpoint: object) -> None:
            raise AssertionError("failed apply must not advance a checkpoint")

        def load_projection_checkpoint(self) -> None:
            return None

    store = BlobStore(
        StoreTopology(
            InMemoryBlobBackend(),
            InMemoryLifecycleAuthority(),
            (FailingSink(),),
        )
    )
    store.initialize()

    receipt = store.put_entry({"value": 1}, key="committed-key")

    assert receipt.projection_outcomes["external-index"].status.value == "dirty"
    assert store.get_entry_info("committed-key") is not None


def test_receipt_outcomes_are_named_and_immutable() -> None:
    from cacheness.storage.lifecycle_authority import EntryExpectation
    from cacheness.storage.read_contract import BlobReceipt

    receipt = BlobReceipt(
        operation_id="operation",
        key="key",
        generation="generation",
        locator="locator",
        expectation=EntryExpectation(
            lineage=1,
            revision=2,
            generation="generation",
            manifest_digest="0" * 64,
        ),
        catalog_revision=2,
        projections={},
    )
    outcome = object()

    attributed = receipt.with_projection_outcome("external-index", outcome)

    assert dict(receipt.projection_outcomes) == {}
    assert attributed.projection_outcomes["external-index"] is outcome
    with pytest.raises(TypeError):
        attributed.projection_outcomes["other"] = object()
