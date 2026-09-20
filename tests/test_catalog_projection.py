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


class _CustomProjectionFailure(Exception):
    """Application sink failure not represented by Cacheness exception types."""


class _ProjectionAbort(BaseException):
    """Control-flow failure that projection boundaries must not translate."""


class _ExceptionalProjectionSink(_ProjectionSink):
    def __init__(self, failure: BaseException) -> None:
        super().__init__()
        self.failure = failure

    def apply(self, entries: tuple[tuple[str, str], ...]) -> None:
        raise self.failure


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


def test_every_ordinary_projection_exception_preserves_the_exact_receipt() -> None:
    projections = _projections()
    receipt = object()
    authority = _Authority((_Page(4, (("a", "1"),), "cursor-a", False),))

    best_effort = projections.CatalogProjection(
        authority, _ExceptionalProjectionSink(_CustomProjectionFailure("sink failed"))
    ).best_effort(receipt)

    assert best_effort.receipt is receipt
    assert best_effort.projection_status == "dirty"
    assert best_effort.outcome is not None
    assert best_effort.outcome.error_type == "_CustomProjectionFailure"

    with pytest.raises(projections.CommittedPartialProjectionError) as raised:
        projections.CatalogProjection(
            _Authority((_Page(4, (("a", "1"),), "cursor-a", False),)),
            _ExceptionalProjectionSink(_CustomProjectionFailure("sink failed")),
        ).refresh(receipt)

    assert raised.value.receipt is receipt
    assert raised.value.projection_name == "_ExceptionalProjectionSink"
    assert raised.value.remaining_cursor == "cursor-a"


def test_projection_boundary_never_translates_base_exception_control_flow() -> None:
    projections = _projections()
    receipt = object()

    with pytest.raises(_ProjectionAbort):
        projections.CatalogProjection(
            _Authority((_Page(4, (("a", "1"),), "cursor-a", False),)),
            _ExceptionalProjectionSink(_ProjectionAbort()),
        ).best_effort(receipt)


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
    from cacheness.storage.blob_store import BlobStore
    from cacheness.storage.catalog import CatalogField, CatalogQuery, CatalogSchema
    from cacheness.storage.composition import StoreTopology
    from cacheness.storage.memory_lifecycle_authority import InMemoryLifecycleAuthority
    from cacheness.storage.obstore_generation_io import ObstoreGenerationIO

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

    payload = ObstoreGenerationIO.for_memory()
    authority = InMemoryLifecycleAuthority()
    authority.qualification_identity = "memory"
    store = BlobStore(StoreTopology(payload, authority, (FailingSink(),)))
    try:
        store.initialize()

        receipt = store.put_entry({"value": 1}, key="committed-key")

        assert receipt.projection_outcomes["external-index"].status.value == "dirty"
        assert store.get_entry_info("committed-key") is not None
    finally:
        store.close()
        authority.close()
        payload.close()


def test_named_rebuild_uses_its_own_sink_capabilities(tmp_path) -> None:
    from cacheness.storage.blob_store import BlobStore
    from cacheness.storage.catalog import CatalogField, CatalogQuery, CatalogSchema
    from cacheness.storage.composition import StoreTopology
    from cacheness.storage.memory_lifecycle_authority import InMemoryLifecycleAuthority
    from cacheness.storage.obstore_generation_io import ObstoreGenerationIO

    class _RebuildSink:
        projection_query = CatalogQuery()
        projection_schema = CatalogSchema(
            (CatalogField("kind", "string"),), schema_id="rebuild-schema"
        )

        def __init__(self, name: str, *, can_rebuild: bool) -> None:
            self.projection_name = name
            self.topology_capabilities = {
                "projection_rebuild": can_rebuild,
                "offline_rebuild": can_rebuild,
            }
            self.applied: list[object] = []
            self.published: object | None = None
            self.discarded: object | None = None

        def apply_projection_batch(self, batch: object) -> None:
            self.applied.append(batch)

        def save_projection_checkpoint(self, checkpoint: object) -> None:
            self.checkpoint = checkpoint

        def load_projection_checkpoint(self) -> None:
            return None

        def begin_isolated_rebuild(self):
            return _RebuildSink(f"{self.projection_name}-candidate", can_rebuild=True)

        def publish_isolated_rebuild(self, candidate: object, checkpoint: object) -> None:
            self.published = (candidate, checkpoint)

        def discard_isolated_rebuild(self, candidate: object) -> None:
            self.discarded = candidate

    capable = _RebuildSink("capable", can_rebuild=True)
    incapable = _RebuildSink("incapable", can_rebuild=False)
    payload = ObstoreGenerationIO.for_memory()
    authority = InMemoryLifecycleAuthority()
    authority.qualification_identity = "memory"
    store = BlobStore(
        StoreTopology(payload, authority, (capable, incapable)),
        cache_dir=tmp_path,
    )
    try:
        # Whole-topology reporting remains conservative, but selecting the
        # capable sink must not inherit the other sink's limitation.
        assert store.capabilities.offline_rebuild is False
        result = store.rebuild_projection("capable", requested="offline")
        assert result.exhausted is True
        assert capable.published is not None
    finally:
        store.close()
        authority.close()
        payload.close()


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


def test_builtin_json_projection_constructs_as_a_projection_sink(tmp_path):
    """The advertised JSON projection is structurally usable by composition."""
    from cacheness.storage.composition import (
        BackendRole,
        ProjectionSink,
        RoleRegistry,
    )

    sink = RoleRegistry().resolve(BackendRole.PROJECTION, "json").construct(
        {"metadata_file": tmp_path / "projection.json"}
    )

    assert isinstance(sink, ProjectionSink)


def test_json_projection_applies_reopens_and_resolves_through_topology(tmp_path) -> None:
    """The built-in JSON name is a real derived sink on the one registry path."""
    from cacheness.metadata import JsonProjection
    from cacheness.storage.catalog import CatalogEntry, CatalogPage
    from cacheness.storage.composition import BackendRef, StoreTopology
    from cacheness.storage.projections import ProjectionController

    class Source:
        def query_catalog(self, cursor):
            assert cursor is None
            return CatalogPage(
                (CatalogEntry("entry", "generation", {"kind": "derived"}),),
                revision=7,
                cursor=None,
                exhausted=True,
            )

    metadata_file = tmp_path / "derived.json"
    topology = StoreTopology(
        BackendRef(name="memory"),
        BackendRef(name="memory"),
        (BackendRef(name="json", options={"metadata_file": metadata_file}),),
    ).resolve()
    try:
        sink = topology.projections[0]
        result = ProjectionController(
            Source(),
            sink,
            page_size=1,
            source_store_id="store",
            schema_id="catalog",
            schema_fingerprint="catalog",
            query_fingerprint="all",
        ).pull()
    finally:
        topology.close()

    assert result.exhausted is True
    assert result.checkpoint is not None
    assert result.checkpoint.complete is True
    reopened = JsonProjection(metadata_file)
    assert reopened.load_projection_checkpoint() == result.checkpoint
    assert reopened.projected_entries() == (
        CatalogEntry("entry", "generation", {"kind": "derived"}),
    )


def test_json_projection_replays_a_pending_batch_after_checkpoint_failure(tmp_path) -> None:
    """A persisted pending batch remains idempotent across a fresh sink instance."""
    from cacheness.metadata import JsonProjection
    from cacheness.storage.catalog import CatalogEntry, CatalogPage
    from cacheness.storage.projections import ProjectionController

    class Source:
        def query_catalog(self, cursor):
            assert cursor is None
            return CatalogPage(
                (CatalogEntry("entry", "generation", {"kind": "derived"}),),
                revision=7,
                cursor=None,
                exhausted=True,
            )

    class CheckpointFailure(JsonProjection):
        def save_projection_checkpoint(self, checkpoint):
            raise OSError("simulated interruption after apply")

    metadata_file = tmp_path / "derived.json"
    controller_args = {
        "page_size": 1,
        "source_store_id": "store",
        "schema_id": "catalog",
        "schema_fingerprint": "catalog",
        "query_fingerprint": "all",
    }
    with pytest.raises(OSError, match="simulated interruption"):
        ProjectionController(Source(), CheckpointFailure(metadata_file), **controller_args).pull()

    result = ProjectionController(Source(), JsonProjection(metadata_file), **controller_args).pull()

    assert result.exhausted is True
    assert JsonProjection(metadata_file).projected_entries() == (
        CatalogEntry("entry", "generation", {"kind": "derived"}),
    )


def test_json_projection_rejects_incompatible_derived_documents(tmp_path) -> None:
    """Malformed derived state fails locally without a canonical fallback."""
    from cacheness.metadata import JsonProjection, JsonProjectionError

    metadata_file = tmp_path / "derived.json"
    metadata_file.write_text('{"format_version": 999}', encoding="utf-8")

    with pytest.raises(JsonProjectionError, match="incompatible shape"):
        JsonProjection(metadata_file).load_projection_checkpoint()
