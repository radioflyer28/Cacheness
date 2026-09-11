"""End-to-end contracts for explicit offline memory-store cutover."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from cacheness.storage import BackendRef, BlobStore, StoreTopology
from cacheness.error_handling import (
    CacheBlobMigrationOfflineDecisionRequiredError,
    CacheBlobMigrationPlanStaleError,
)
from cacheness.storage.migration import (
    MigrationCompatibilityEdge,
    MigrationDisposition,
    MigrationRunLimits,
    MigrationSplitRequired,
    OfflineMigrationService,
)
from cacheness.storage.migration_authority import AuthorityPublicationState
from cacheness.storage.migration_evidence import MaintenanceEvidenceState
from cacheness.storage.projections import ProjectionController, ProjectionStatus
from cacheness.storage.sqlite_lifecycle_authority import SQLITE_USER_VERSION
from cacheness.storage.manifest import (
    BlobManifest,
    CURRENT_SQLITE_USER_VERSION,
    StoreVersionDimensions,
    verify_current_manifest,
)
from cacheness.interfaces import PayloadTransformationEdge


class _SharedMemoryKeyProvider:
    """Test-only provider proving cutover preserves provider-backed signing identity."""

    def get_key(self) -> bytes:
        return b"m" * 32


class _MigrationMcapHandler:
    """Path-based custom handler with one explicitly directed native upgrade."""

    def __init__(self, *, target_version: int = 2) -> None:
        if target_version < 1:
            raise ValueError("target_version must be positive")
        self._target_version = target_version
        self.transform_calls = 0
        self.published_paths: list[Path] = []

    @property
    def data_type(self) -> str:
        return "migration-mcap"

    @property
    def payload_format(self) -> str:
        return "mcap-v2"

    @property
    def payload_format_version(self) -> int:
        return self._target_version

    def supports_payload_contract(
        self, payload_format: str, payload_format_version: int
    ) -> bool:
        return (payload_format, payload_format_version) in {
            ("mcap-v1", 1),
            ("mcap-v2", self._target_version),
        }

    def payload_transformation_edges(self) -> tuple[PayloadTransformationEdge, ...]:
        return (
            PayloadTransformationEdge("mcap-v1", 1, "mcap-v2", self._target_version),
        )

    def can_handle(self, data, config=None) -> bool:
        del config
        return isinstance(data, dict) and "mcap_version" in data

    def put(self, data, file_path: Path, config):
        del config
        version = data["mcap_version"]
        payload = f"mcap-v{version}:{data['value']}".encode("utf-8")
        actual_path = file_path.with_suffix(f".mcap{version}")
        actual_path.write_bytes(payload)
        return {
            "actual_path": str(actual_path),
            "file_size": len(payload),
            "payload_format": f"mcap-v{version}",
            "payload_format_version": version,
            "metadata": {"writer_version": version},
        }

    def get(self, file_path: Path, metadata):
        del metadata
        version_and_value = file_path.read_text(encoding="utf-8").split(":", 1)
        return {"mcap_version": int(version_and_value[0][-1]), "value": version_and_value[1]}

    def get_file_extension(self, config) -> str:
        del config
        return f".mcap{self._target_version}"

    def transform_payload(self, snapshot, edge, *, destination_io, key: str, config):
        assert edge == PayloadTransformationEdge(
            "mcap-v1", 1, "mcap-v2", self._target_version
        )
        self.transform_calls += 1
        version_and_value = snapshot.path.read_text(encoding="utf-8").split(":", 1)
        assert version_and_value[0] == "mcap-v1"
        result = destination_io.put(
            self,
            {"mcap_version": 2, "value": version_and_value[1]},
            key,
            config,
        )
        result["payload_format"] = "mcap-v2"
        result["payload_format_version"] = self._target_version
        result["metadata"] = {
            **result["metadata"],
            "runtime_transform": {"source": "mcap-v1", "target": "mcap-v2"},
        }
        self.published_paths.append(Path(result["actual_path"]))
        return result


def _memory_store(root: Path, key_provider: _SharedMemoryKeyProvider) -> BlobStore:
    """Create an initialized, explicitly same-process store for tracer tests."""
    store = BlobStore(
        StoreTopology(
            payload=BackendRef(name="memory"),
            authority=BackendRef(name="memory"),
        ),
        cache_dir=root,
        manifest_key_provider=key_provider,
    )
    store.initialize()
    return store


def _sqlite_store(root: Path, key_provider: _SharedMemoryKeyProvider) -> BlobStore:
    """Create the supported local durable topology for cutover authority tests."""
    store = BlobStore(
        StoreTopology(
            payload=BackendRef(name="filesystem", options={"base_dir": root}),
            authority=BackendRef(name="sqlite", options={"root": root}),
        ),
        cache_dir=root,
        manifest_key_provider=key_provider,
    )
    store.initialize()
    return store


def _transforming_service(
    source: BlobStore,
    destination: BlobStore,
    work_directory: Path,
    *,
    run_id: str,
    target_version: int = 2,
) -> tuple[OfflineMigrationService, _MigrationMcapHandler]:
    """Configure one store-local custom format target for migration coverage."""
    source_handler = _MigrationMcapHandler(target_version=target_version)
    source.handlers.register_handler(source_handler, priority=0)
    destination.handlers.register_handler(
        _MigrationMcapHandler(target_version=target_version), priority=0
    )
    service = OfflineMigrationService(
        source=source,
        destination=destination,
        work_directory=work_directory,
        run_id=run_id,
        stopped_workers_acknowledged=True,
        compatibility_edges=(
            MigrationCompatibilityEdge(
                source=StoreVersionDimensions(payload_format_version=1),
                destination=StoreVersionDimensions(payload_format_version=target_version),
                name="mcap-v1-to-v2",
            ),
        ),
    )
    return service, source_handler


def _service(
    source: BlobStore,
    destination: BlobStore,
    work_directory: Path,
    *,
    run_id: str = "memory-cutover-run",
) -> OfflineMigrationService:
    """Build one operator-authorized migration service with a test-only edge."""
    return OfflineMigrationService(
        source=source,
        destination=destination,
        work_directory=work_directory,
        run_id=run_id,
        stopped_workers_acknowledged=True,
        compatibility_edges=(MigrationCompatibilityEdge.current_to_current_for_test(),),
    )


def test_memory_tracer_requires_explicit_whole_store_activation(tmp_path: Path) -> None:
    """A verified candidate stays invisible until the operator activates it."""
    key_provider = _SharedMemoryKeyProvider()
    source = _memory_store(tmp_path / "source", key_provider)
    destination = _memory_store(tmp_path / "destination", key_provider)
    try:
        source.put_entry({"answer": 42}, key="entry")
        service = _service(source, destination, tmp_path / "maintenance")

        inspection = service.inspect()
        plan = service.plan(inspection)
        assert plan.assessments[0].disposition is MigrationDisposition.MIGRATABLE

        staged = service.stage(plan)
        assert staged.completed is True
        assert destination.get("entry") is None

        verified = service.verify(plan)
        assert verified.completed is True
        assert destination.get("entry") is None

        evidence = service.read_evidence()
        assert evidence.run_id == "memory-cutover-run"
        assert evidence.plan_digest == plan.digest
        assert evidence.candidate_receipt is not None

        activated = service.activate(plan)
        assert activated.completed is True
        authority = destination.lifecycle_authority
        assert authority.publication_state() is AuthorityPublicationState.ACTIVATED_OFFLINE
        with pytest.raises(CacheBlobMigrationOfflineDecisionRequiredError):
            destination.get("entry")

        resumed = service.resume(plan, run_id=service.run_id, evidence_path=service.evidence_path)
        assert resumed.completed is True
        authority.finalize_verified_candidate(run_id=service.run_id)
        assert destination.get("entry") == {"answer": 42}
        # The source remains a retained valid store; activation never deletes it.
        assert source.get("entry") == {"answer": 42}
    finally:
        source.close()
        destination.close()


def test_candidate_and_evidence_never_authorize_activation(tmp_path: Path) -> None:
    """A candidate cannot become visible merely because stage evidence exists."""
    key_provider = _SharedMemoryKeyProvider()
    source = _memory_store(tmp_path / "source", key_provider)
    destination = _memory_store(tmp_path / "destination", key_provider)
    try:
        source.put_entry({"answer": 42}, key="entry")
        service = _service(source, destination, tmp_path / "maintenance")
        plan = service.plan(service.inspect())
        service.stage(plan)

        with pytest.raises(ValueError, match="verified"):
            service.activate(plan)

        assert destination.get("entry") is None
        assert source.get("entry") == {"answer": 42}
    finally:
        source.close()
        destination.close()


def test_maintenance_request_requires_stopped_workers_and_separate_work_dir(
    tmp_path: Path,
) -> None:
    """Operator input cannot turn an ordinary store root into maintenance evidence."""
    key_provider = _SharedMemoryKeyProvider()
    source = _memory_store(tmp_path / "source", key_provider)
    destination = _memory_store(tmp_path / "destination", key_provider)
    try:
        with pytest.raises(ValueError, match="stopped-worker"):
            OfflineMigrationService(
                source=source,
                destination=destination,
                work_directory=tmp_path / "maintenance",
                run_id="not-stopped",
                stopped_workers_acknowledged=False,
                compatibility_edges=(
                    MigrationCompatibilityEdge.current_to_current_for_test(),
                ),
            )

        with pytest.raises(ValueError, match="separate"):
            _service(source, destination, tmp_path / "source")
    finally:
        source.close()
        destination.close()


def test_release_authority_baseline_exposes_only_the_resolved_publication_states() -> None:
    """RQ-01 fixes the first release schema and whole-store state vocabulary."""
    assert SQLITE_USER_VERSION == 8
    assert CURRENT_SQLITE_USER_VERSION == 8
    assert {state.value for state in AuthorityPublicationState} == {
        "idle",
        "candidate",
        "activated_offline",
        "active",
        "rolled_back",
    }


def test_sqlite_activation_seals_workers_until_explicit_finalize(tmp_path: Path) -> None:
    """A whole verified candidate activates atomically but remains offline for a decision."""
    key_provider = _SharedMemoryKeyProvider()
    source = _sqlite_store(tmp_path / "source", key_provider)
    destination = _sqlite_store(tmp_path / "destination", key_provider)
    try:
        source.put_entry({"answer": 42}, key="entry")
        service = _service(source, destination, tmp_path / "maintenance", run_id="sqlite-run")
        plan = service.plan(service.inspect())
        service.stage(plan)
        service.verify(plan)
        service.activate(plan)

        authority = destination.lifecycle_authority
        assert authority.publication_state() is AuthorityPublicationState.ACTIVATED_OFFLINE
        with pytest.raises(CacheBlobMigrationOfflineDecisionRequiredError):
            destination.get("entry")
        with pytest.raises(CacheBlobMigrationOfflineDecisionRequiredError):
            destination.put_entry({"answer": "new"}, key="new-entry")

        authority.finalize_verified_candidate(run_id="sqlite-run")
        assert authority.publication_state() is AuthorityPublicationState.ACTIVE
        assert destination.get("entry") == {"answer": 42}
    finally:
        source.close()
        destination.close()


def test_sqlite_activation_rollback_keeps_candidate_invisible(tmp_path: Path) -> None:
    """A failed authority transaction preserves candidate evidence without partial publication."""
    key_provider = _SharedMemoryKeyProvider()
    source = _sqlite_store(tmp_path / "source", key_provider)
    destination = _sqlite_store(tmp_path / "destination", key_provider)
    try:
        source.put_entry({"answer": 42}, key="entry")
        service = _service(source, destination, tmp_path / "maintenance", run_id="rollback-run")
        plan = service.plan(service.inspect())
        service.stage(plan)
        service.verify(plan)
        evidence = service.read_evidence()
        candidates = service._candidate_from_evidence(evidence, plan)
        assert evidence.candidate_receipt is not None

        authority = destination.lifecycle_authority
        authority.record_verified_candidate(
            receipt=evidence.candidate_receipt,
            entries=candidates,
        )

        def fail_before_commit(boundary: str) -> None:
            if boundary == "authority.transaction.before_commit":
                raise RuntimeError("simulated authority interruption")

        authority.set_transaction_hook_for_test(fail_before_commit)
        with pytest.raises(RuntimeError, match="simulated authority interruption"):
            authority.activate_verified_candidate(
                receipt=evidence.candidate_receipt,
                entries=candidates,
            )
        authority.set_transaction_hook_for_test(None)

        assert authority.publication_state() is AuthorityPublicationState.CANDIDATE
        assert destination.get("entry") is None
        assert source.get("entry") == {"answer": 42}
    finally:
        source.close()
        destination.close()


def test_offline_service_rolls_back_only_the_activated_receipt(tmp_path: Path) -> None:
    """Rollback restores the retained authority selection before workers restart."""
    key_provider = _SharedMemoryKeyProvider()
    source = _memory_store(tmp_path / "source", key_provider)
    destination = _memory_store(tmp_path / "destination", key_provider)
    try:
        source.put_entry({"answer": "candidate"}, key="entry")
        destination.put_entry({"answer": "prior"}, key="entry")
        service = _service(
            source,
            destination,
            tmp_path / "maintenance",
            run_id="rollback-service",
        )
        plan = service.plan(service.inspect())
        service.stage(plan)
        service.verify(plan)
        service.activate(plan)

        rollback = service.rollback(plan)

        assert rollback.run_id == service.run_id
        assert destination.lifecycle_authority.publication_state() is (
            AuthorityPublicationState.ROLLED_BACK
        )
        assert destination.get("entry") == {"answer": "prior"}
        assert service.read_evidence().state is MaintenanceEvidenceState.ROLLED_BACK
    finally:
        source.close()
        destination.close()


def test_offline_service_finalize_requires_exact_confirmation_and_seals_rollback(
    tmp_path: Path,
) -> None:
    """Finalize accepts the activated run once and makes workers safe to restart."""
    key_provider = _SharedMemoryKeyProvider()
    source = _memory_store(tmp_path / "source", key_provider)
    destination = _memory_store(tmp_path / "destination", key_provider)
    try:
        source.put_entry({"answer": "candidate"}, key="entry")
        destination.put_entry({"answer": "prior"}, key="entry")
        service = _service(
            source,
            destination,
            tmp_path / "maintenance",
            run_id="finalize-service",
        )
        plan = service.plan(service.inspect())
        service.stage(plan)
        service.verify(plan)
        service.activate(plan)

        with pytest.raises(CacheBlobMigrationOfflineDecisionRequiredError, match="confirmation"):
            service.finalize(plan, confirmation="0" * 64)

        confirmation = service.finalize_confirmation(plan)
        finalized = service.finalize(plan, confirmation=confirmation)

        assert finalized.run_id == service.run_id
        assert destination.lifecycle_authority.publication_state() is (
            AuthorityPublicationState.ACTIVE
        )
        assert destination.get("entry") == {"answer": "candidate"}
        assert service.read_evidence().state is MaintenanceEvidenceState.FINALIZED
        assert service.finalize(plan, confirmation=confirmation) == finalized
        with pytest.raises(CacheBlobMigrationOfflineDecisionRequiredError, match="finalize"):
            service.rollback(plan)
    finally:
        source.close()
        destination.close()


def test_offline_service_abort_removes_only_its_unactivated_candidate(tmp_path: Path) -> None:
    """Abort is limited to authenticated run-owned candidate effects before activation."""
    key_provider = _SharedMemoryKeyProvider()
    source = _memory_store(tmp_path / "source", key_provider)
    destination = _memory_store(tmp_path / "destination", key_provider)
    try:
        source.put_entry({"answer": "candidate"}, key="entry")
        service = _service(
            source,
            destination,
            tmp_path / "maintenance",
            run_id="abort-service",
        )
        plan = service.plan(service.inspect())
        service.stage(plan)

        aborted = service.abort(plan)

        assert aborted.run_id == service.run_id
        assert aborted.deleted_entries == 1
        assert service.read_evidence().state is MaintenanceEvidenceState.ABORTED
        assert destination.get("entry") is None

        active_service = _service(
            source,
            destination,
            tmp_path / "active-maintenance",
            run_id="active-abort-service",
        )
        active_plan = active_service.plan(active_service.inspect())
        active_service.stage(active_plan)
        active_service.verify(active_plan)
        active_service.activate(active_plan)
        with pytest.raises(CacheBlobMigrationOfflineDecisionRequiredError, match="rollback or finalize"):
            active_service.abort(active_plan)
    finally:
        source.close()
        destination.close()


def test_offline_service_purge_is_separate_idempotent_retryable_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Purge deletes only retained prior bytes and keeps final activation authoritative."""
    key_provider = _SharedMemoryKeyProvider()
    source = _memory_store(tmp_path / "source", key_provider)
    destination = _memory_store(tmp_path / "destination", key_provider)
    try:
        source.put_entry({"answer": "candidate"}, key="entry")
        destination.put_entry({"answer": "prior"}, key="entry")
        service = _service(
            source,
            destination,
            tmp_path / "maintenance",
            run_id="purge-service",
        )
        plan = service.plan(service.inspect())
        service.stage(plan)
        service.verify(plan)
        service.activate(plan)
        service.finalize(plan, confirmation=service.finalize_confirmation(plan))

        with pytest.raises(CacheBlobMigrationOfflineDecisionRequiredError, match="confirmation"):
            service.purge(plan, confirmation="0" * 64)

        original_delete = destination.delete_migration_payload
        attempts = 0

        def fail_once(locator: str) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                original_delete(locator)
                raise OSError("simulated deletion failure")
            original_delete(locator)

        monkeypatch.setattr(destination, "delete_migration_payload", fail_once)
        pending = service.purge(plan, confirmation=service.purge_confirmation(plan))

        assert pending.completed is False
        assert pending.pending_entries == 1
        assert service.read_evidence().state is MaintenanceEvidenceState.PURGE_PENDING
        assert destination.lifecycle_authority.publication_state() is AuthorityPublicationState.ACTIVE
        assert destination.get("entry") == {"answer": "candidate"}

        monkeypatch.setattr(destination, "delete_migration_payload", original_delete)
        purged = service.purge(plan, confirmation=service.purge_confirmation(plan))

        assert purged.completed is True
        assert purged.pending_entries == 0
        assert service.read_evidence().state is MaintenanceEvidenceState.PURGED
        assert destination.get("entry") == {"answer": "candidate"}
        assert service.purge(plan, confirmation=service.purge_confirmation(plan)) == purged
    finally:
        source.close()
        destination.close()


def test_sqlite_purge_uses_the_finalized_authority_retention_rows(tmp_path: Path) -> None:
    """The durable authority exposes retained prior rows only for explicit purge."""
    key_provider = _SharedMemoryKeyProvider()
    source = _sqlite_store(tmp_path / "source", key_provider)
    destination = _sqlite_store(tmp_path / "destination", key_provider)
    try:
        source.put_entry({"answer": "candidate"}, key="entry")
        destination.put_entry({"answer": "prior"}, key="entry")
        service = _service(
            source,
            destination,
            tmp_path / "maintenance",
            run_id="sqlite-purge-service",
        )
        plan = service.plan(service.inspect())
        service.stage(plan)
        service.verify(plan)
        service.activate(plan)
        service.finalize(plan, confirmation=service.finalize_confirmation(plan))

        purged = service.purge(plan, confirmation=service.purge_confirmation(plan))

        assert purged.completed is True
        assert purged.purged_entries == 1
        assert destination.lifecycle_authority.publication_state() is AuthorityPublicationState.ACTIVE
        assert destination.get("entry") == {"answer": "candidate"}
    finally:
        source.close()
        destination.close()


def test_purge_refuses_a_stale_source_without_touching_active_candidate(tmp_path: Path) -> None:
    """Purge uses the evidence-bound stopped-worker source revision before I/O."""
    key_provider = _SharedMemoryKeyProvider()
    source = _memory_store(tmp_path / "source", key_provider)
    destination = _memory_store(tmp_path / "destination", key_provider)
    try:
        source.put_entry({"answer": "candidate"}, key="entry")
        destination.put_entry({"answer": "prior"}, key="entry")
        service = _service(
            source,
            destination,
            tmp_path / "maintenance",
            run_id="stale-purge-service",
        )
        plan = service.plan(service.inspect())
        service.stage(plan)
        service.verify(plan)
        service.activate(plan)
        service.finalize(plan, confirmation=service.finalize_confirmation(plan))
        confirmation = service.purge_confirmation(plan)
        source.put_entry({"answer": "source changed"}, key="new-entry")

        with pytest.raises(CacheBlobMigrationPlanStaleError, match="source revision"):
            service.purge(plan, confirmation=confirmation)
        assert destination.get("entry") == {"answer": "candidate"}
    finally:
        source.close()
        destination.close()


def test_projection_failure_is_derived_after_memory_activation(tmp_path: Path) -> None:
    """A failed projection rebuild preserves the committed activation receipt and state."""
    key_provider = _SharedMemoryKeyProvider()
    source = _memory_store(tmp_path / "source", key_provider)
    destination = _memory_store(tmp_path / "destination", key_provider)

    class ProjectionSource:
        projection_store_id = "derived-projection-source"

        def query_catalog(self, cursor: str | None):
            assert cursor is None
            return SimpleNamespace(
                revision=1,
                entries=(("entry", "generation"),),
                cursor=None,
                exhausted=True,
            )

    class ProjectionSink:
        projection_name = "derived-projection"
        topology_capabilities = {
            "projection_rebuild": True,
            "offline_rebuild": True,
            "online_rebuild": False,
        }

        def __init__(self, *, fail: bool = False) -> None:
            self.fail = fail
            self.published = False
            self.discarded = False

        def begin_isolated_rebuild(self):
            return ProjectionSink(fail=True)

        def apply_projection_batch(self, batch) -> None:
            if self.fail:
                raise OSError("derived sink failed")

        def save_projection_checkpoint(self, checkpoint) -> None:
            return None

        def publish_isolated_rebuild(self, candidate, checkpoint) -> None:
            self.published = True

        def discard_isolated_rebuild(self, candidate) -> None:
            self.discarded = True

    try:
        source.put_entry({"answer": 42}, key="entry")
        service = _service(source, destination, tmp_path / "maintenance", run_id="projection-run")
        plan = service.plan(service.inspect())
        service.stage(plan)
        service.verify(plan)
        service.activate(plan)
        evidence = service.read_evidence()
        assert evidence.activation_receipt is not None

        sink = ProjectionSink()
        result = service.rebuild_projection(
            plan,
            ProjectionController(ProjectionSource(), sink),
        )

        assert result.receipt == evidence.activation_receipt
        assert result.projection_status == ProjectionStatus.DIRTY.value
        assert sink.published is False
        assert sink.discarded is True
        assert destination.lifecycle_authority.publication_state() is (
            AuthorityPublicationState.ACTIVATED_OFFLINE
        )
    finally:
        source.close()
        destination.close()


def test_resume_classifies_authority_activation_after_lost_evidence_observation(
    tmp_path: Path,
) -> None:
    """Resume reads exact authority state when activation committed before evidence observation."""
    key_provider = _SharedMemoryKeyProvider()
    source = _memory_store(tmp_path / "source", key_provider)
    destination = _memory_store(tmp_path / "destination", key_provider)
    try:
        source.put_entry({"answer": 42}, key="entry")
        service = _service(source, destination, tmp_path / "maintenance", run_id="lost-response-run")
        plan = service.plan(service.inspect())
        service.stage(plan)
        service.verify(plan)
        evidence = service.read_evidence()
        assert evidence.candidate_receipt is not None
        candidates = service._candidate_from_evidence(evidence, plan)
        authority = destination.lifecycle_authority
        authority.record_verified_candidate(receipt=evidence.candidate_receipt, entries=candidates)
        authority.activate_verified_candidate(receipt=evidence.candidate_receipt, entries=candidates)

        resumed = service.resume(plan, run_id=service.run_id, evidence_path=service.evidence_path)

        assert resumed.state is MaintenanceEvidenceState.ACTIVATED
        assert service.read_evidence().activation_receipt is not None
        assert authority.publication_state() is AuthorityPublicationState.ACTIVATED_OFFLINE
    finally:
        source.close()
        destination.close()


def test_migration_run_enforces_entry_byte_and_evidence_limits_with_split_plan(
    tmp_path: Path,
) -> None:
    """Oversized catalogs split before candidate publication; bounded runs checkpoint batches."""
    key_provider = _SharedMemoryKeyProvider()
    source = _memory_store(tmp_path / "source", key_provider)
    destination = _memory_store(tmp_path / "destination", key_provider)
    try:
        for key in ("entry-a", "entry-b", "entry-c"):
            source.put_entry({"value": key}, key=key)
        limits = MigrationRunLimits(
            max_entries_per_run=2,
            max_bytes_per_run=1_024,
            max_evidence_bytes_per_run=8_192,
        )
        service = OfflineMigrationService(
            source=source,
            destination=destination,
            work_directory=tmp_path / "maintenance",
            run_id="limited-run",
            stopped_workers_acknowledged=True,
            compatibility_edges=(MigrationCompatibilityEdge.current_to_current_for_test(),),
            run_limits=limits,
        )

        split = service.plan(service.inspect())

        assert isinstance(split, MigrationSplitRequired)
        assert split.reason == "split_required"
        assert [partition.keys for partition in split.partitions] == [
            ("entry-a", "entry-b"),
            ("entry-c",),
        ]
        assert destination.lifecycle_authority.identity_snapshot().revision == 0

        bounded_source = _memory_store(tmp_path / "bounded-source", key_provider)
        bounded_destination = _memory_store(tmp_path / "bounded-destination", key_provider)
        try:
            bounded_source.put_entry({"value": "one"}, key="entry-a")
            bounded_source.put_entry({"value": "two"}, key="entry-b")
            bounded_service = OfflineMigrationService(
                source=bounded_source,
                destination=bounded_destination,
                work_directory=tmp_path / "bounded-maintenance",
                run_id="bounded-run",
                stopped_workers_acknowledged=True,
                compatibility_edges=(
                    MigrationCompatibilityEdge.current_to_current_for_test(),
                ),
                run_limits=limits,
            )
            plan = bounded_service.plan(bounded_service.inspect())
            assert not isinstance(plan, MigrationSplitRequired)

            staged = bounded_service.stage(plan)

            evidence = bounded_service.read_evidence()
            assert staged.state is MaintenanceEvidenceState.STAGED
            assert evidence.candidate_batch_references
            assert evidence.candidate_entry_count == 2
            assert evidence.candidate_byte_count == plan.totals.total_bytes
            assert bounded_destination.get("entry-a") is None
        finally:
            bounded_source.close()
            bounded_destination.close()
    finally:
        source.close()
        destination.close()


def test_uncheckpointed_candidate_orphan_remains_invisible_unadopted_and_outside_exact_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A publication fault before its authority checkpoint leaves only an ignored orphan."""
    key_provider = _SharedMemoryKeyProvider()
    source = _memory_store(tmp_path / "source", key_provider)
    destination = _memory_store(tmp_path / "destination", key_provider)
    try:
        source.put_entry({"value": "orphan boundary"}, key="entry")
        service = _service(source, destination, tmp_path / "maintenance", run_id="orphan-run")
        plan = service.plan(service.inspect())
        authority = destination.lifecycle_authority
        original_record = authority.record_verified_candidate
        written_locators: list[str] = []
        original_write = destination.payload_backend.write_blob

        def capture_write(blob_id: str, payload: bytes) -> str:
            locator = original_write(blob_id, payload)
            written_locators.append(locator)
            return locator

        def fail_before_checkpoint(*, receipt, entries):
            raise RuntimeError("simulated post-publication checkpoint fault")

        monkeypatch.setattr(destination.payload_backend, "write_blob", capture_write)
        monkeypatch.setattr(authority, "record_verified_candidate", fail_before_checkpoint)
        with pytest.raises(RuntimeError, match="post-publication"):
            service.stage(plan)

        assert written_locators
        assert authority.publication_state() is AuthorityPublicationState.IDLE
        assert destination.get("entry") is None
        assert source.get("entry") == {"value": "orphan boundary"}

        monkeypatch.setattr(authority, "record_verified_candidate", original_record)
        resumed = service.resume(plan, run_id=service.run_id, evidence_path=service.evidence_path)

        attributed = authority.candidate_entries_for_run(run_id=service.run_id)
        assert resumed.state is MaintenanceEvidenceState.STAGED
        assert {entry.locator for entry in attributed}.isdisjoint(written_locators)
        assert destination.get("entry") is None
        assert source.get("entry") == {"value": "orphan boundary"}
    finally:
        source.close()
        destination.close()


def test_resume_and_abort_staging_use_only_authority_attributed_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """STAGING recovery replays only authority receipts and records exact deletion debt."""
    key_provider = _SharedMemoryKeyProvider()
    source = _memory_store(tmp_path / "source", key_provider)
    destination = _memory_store(tmp_path / "destination", key_provider)
    try:
        for key in ("entry-a", "entry-b"):
            source.put_entry({"value": key}, key=key)

        resume_service = _service(
            source, destination, tmp_path / "resume-maintenance", run_id="resume-staging"
        )
        resume_plan = resume_service.plan(resume_service.inspect())
        original_evidence_write = resume_service._write_evidence

        def interrupt_final_stage(evidence):
            if evidence.state is MaintenanceEvidenceState.STAGED:
                raise RuntimeError("interrupt after final authority checkpoint")
            return original_evidence_write(evidence)

        monkeypatch.setattr(resume_service, "_write_evidence", interrupt_final_stage)
        with pytest.raises(RuntimeError, match="final authority checkpoint"):
            resume_service.stage(resume_plan)
        monkeypatch.setattr(resume_service, "_write_evidence", original_evidence_write)

        staging_evidence = resume_service.read_evidence()
        attributed_before_resume = destination.lifecycle_authority.candidate_entries_for_run(
            run_id=resume_service.run_id
        )
        assert staging_evidence.state is MaintenanceEvidenceState.STAGING
        assert len(staging_evidence.candidate_batch_references) == 2
        assert len(attributed_before_resume) == 2

        resumed = resume_service.resume(
            resume_plan,
            run_id=resume_service.run_id,
            evidence_path=resume_service.evidence_path,
        )
        assert resumed.state is MaintenanceEvidenceState.STAGED
        assert destination.lifecycle_authority.candidate_entries_for_run(
            run_id=resume_service.run_id
        ) == attributed_before_resume
        assert resume_service.abort(resume_plan).state is MaintenanceEvidenceState.ABORTED
        assert destination.lifecycle_authority.publication_state() is AuthorityPublicationState.IDLE

        abort_service = _service(
            source, destination, tmp_path / "abort-maintenance", run_id="abort-staging"
        )
        abort_plan = abort_service.plan(abort_service.inspect())
        original_abort_evidence_write = abort_service._write_evidence

        def interrupt_abort_final_stage(evidence):
            if evidence.state is MaintenanceEvidenceState.STAGED:
                raise RuntimeError("interrupt before staged evidence")
            return original_abort_evidence_write(evidence)

        monkeypatch.setattr(abort_service, "_write_evidence", interrupt_abort_final_stage)
        with pytest.raises(RuntimeError, match="before staged evidence"):
            abort_service.stage(abort_plan)
        monkeypatch.setattr(abort_service, "_write_evidence", original_abort_evidence_write)

        original_delete = destination.delete_migration_payload
        delete_calls = 0

        def delete_then_lose_acknowledgement(locator: str) -> None:
            nonlocal delete_calls
            delete_calls += 1
            original_delete(locator)
            if delete_calls == 2:
                raise OSError("simulated deletion acknowledgement loss")

        monkeypatch.setattr(
            destination, "delete_migration_payload", delete_then_lose_acknowledgement
        )
        partial_abort = abort_service.abort(abort_plan)

        assert partial_abort.state is MaintenanceEvidenceState.STAGING
        assert len(abort_service.read_evidence().cleanup_debt) == 1
        assert destination.get("entry-a") is None
        assert source.get("entry-a") == {"value": "entry-a"}
        assert destination.lifecycle_authority.publication_state() is AuthorityPublicationState.CANDIDATE

        monkeypatch.setattr(destination, "delete_migration_payload", original_delete)
        completed_abort = abort_service.abort(abort_plan)

        assert completed_abort.state is MaintenanceEvidenceState.ABORTED
        assert destination.lifecycle_authority.publication_state() is AuthorityPublicationState.IDLE
        assert source.get("entry-b") == {"value": "entry-b"}
    finally:
        source.close()
        destination.close()


def test_migration_executes_one_handler_transform_and_publishes_destination_manifest_identity(
    tmp_path: Path,
) -> None:
    """A changed native contract uses one handler transform before candidate signing."""
    key_provider = _SharedMemoryKeyProvider()
    source = _sqlite_store(tmp_path / "source", key_provider)
    destination = _sqlite_store(tmp_path / "destination", key_provider)
    try:
        service, handler = _transforming_service(
            source,
            destination,
            tmp_path / "maintenance",
            run_id="transform-run",
        )
        source.put_entry({"mcap_version": 1, "value": "payload"}, key="entry")

        plan = service.plan(service.inspect())
        assert plan.assessments[0].disposition is MigrationDisposition.MIGRATABLE
        assert plan.assessments[0].reason.value == "directed_edge"
        service.stage(plan)

        assert handler.transform_calls == 1
        candidates = destination.lifecycle_authority.candidate_entries_for_run(
            run_id=service.run_id
        )
        assert len(candidates) == 1
        candidate_manifest = BlobManifest.from_canonical_bytes(candidates[0].manifest)
        verify_current_manifest(candidate_manifest, key_provider.get_key())
        assert candidate_manifest.payload_format == "mcap-v2"
        assert candidate_manifest.payload_format_version == 2
        assert candidate_manifest.versions.payload_format_version == 2
        assert candidate_manifest.handler_metadata["runtime_transform"] == {
            "source": "mcap-v1",
            "target": "mcap-v2",
        }
        source_manifest = BlobManifest.from_canonical_bytes(
            source.lifecycle_authority.read_entry("entry").manifest
        )
        assert candidate_manifest.digest != source_manifest.digest

        service.verify(plan)
        service.activate(plan)
        destination.lifecycle_authority.finalize_verified_candidate(run_id=service.run_id)
        assert destination.get("entry") == {"mcap_version": 2, "value": "payload"}
    finally:
        source.close()
        destination.close()


def test_same_version_different_format_uses_exact_directed_transform_and_destination_manifest(
    tmp_path: Path,
) -> None:
    """A shared version integer never turns a changed native format into a copy."""
    key_provider = _SharedMemoryKeyProvider()
    source = _sqlite_store(tmp_path / "source", key_provider)
    destination = _sqlite_store(tmp_path / "destination", key_provider)
    try:
        service, handler = _transforming_service(
            source,
            destination,
            tmp_path / "maintenance",
            run_id="same-version-transform-run",
            target_version=1,
        )
        source.put_entry({"mcap_version": 1, "value": "payload"}, key="entry")

        inspection = service.inspect()
        plan = service.plan(inspection)
        assert plan.assessments[0].disposition is MigrationDisposition.MIGRATABLE
        assert plan.assessments[0].reason.value == "directed_edge"

        service.stage(plan)

        assert handler.transform_calls == 1
        candidates = destination.lifecycle_authority.candidate_entries_for_run(
            run_id=service.run_id
        )
        assert len(candidates) == 1
        candidate_manifest = BlobManifest.from_canonical_bytes(candidates[0].manifest)
        verify_current_manifest(candidate_manifest, key_provider.get_key())
        assert candidate_manifest.payload_format == "mcap-v2"
        assert candidate_manifest.payload_format_version == 1
        assert candidate_manifest.versions.payload_format_version == 1
        assert candidate_manifest.handler_metadata["runtime_transform"] == {
            "source": "mcap-v1",
            "target": "mcap-v2",
        }
        with destination._materialize_authority_store().open_snapshot(
            candidates[0].locator, {}
        ) as snapshot:
            assert snapshot.path.read_bytes() == b"mcap-v2:payload"
        source_manifest = BlobManifest.from_canonical_bytes(
            source.lifecycle_authority.read_entry("entry").manifest
        )
        assert candidate_manifest.digest != source_manifest.digest

        service.verify(plan)
        service.activate(plan)
        destination.lifecycle_authority.finalize_verified_candidate(run_id=service.run_id)
        assert destination.get("entry") == {"mcap_version": 2, "value": "payload"}
    finally:
        source.close()
        destination.close()


def test_transformed_candidate_checkpointed_metadata_recovers_within_limits_and_uncheckpointed_orphan_remains_invisible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only checkpointed transform results resume or abort; an earlier orphan stays ignored."""
    key_provider = _SharedMemoryKeyProvider()
    source = _sqlite_store(tmp_path / "source", key_provider)
    destination = _sqlite_store(tmp_path / "destination", key_provider)
    try:
        service, handler = _transforming_service(
            source,
            destination,
            tmp_path / "maintenance",
            run_id="transform-recovery-run",
        )
        source.put_entry({"mcap_version": 1, "value": "payload"}, key="entry")
        plan = service.plan(service.inspect())
        service.stage(plan)
        assert handler.transform_calls == 1

        resumed = service.resume(
            plan, run_id=service.run_id, evidence_path=service.evidence_path
        )
        assert resumed.state is MaintenanceEvidenceState.VERIFIED
        assert handler.transform_calls == 1
        assert service.abort(plan).state is MaintenanceEvidenceState.ABORTED

        orphan_source = _sqlite_store(tmp_path / "orphan-source", key_provider)
        orphan_destination = _sqlite_store(tmp_path / "orphan-destination", key_provider)
        try:
            orphan_service, orphan_handler = _transforming_service(
                orphan_source,
                orphan_destination,
                tmp_path / "orphan-maintenance",
                run_id="transform-orphan-run",
            )
            orphan_source.put_entry({"mcap_version": 1, "value": "orphan"}, key="entry")
            orphan_plan = orphan_service.plan(orphan_service.inspect())
            authority = orphan_destination.lifecycle_authority
            original_record = authority.record_verified_candidate

            def fail_before_checkpoint(*, receipt, entries):
                raise RuntimeError("simulated transformed post-publication checkpoint fault")

            monkeypatch.setattr(authority, "record_verified_candidate", fail_before_checkpoint)
            with pytest.raises(RuntimeError, match="post-publication"):
                orphan_service.stage(orphan_plan)
            orphan_paths = tuple(orphan_handler.published_paths)
            assert orphan_paths
            assert orphan_destination.get("entry") is None
            assert authority.candidate_entries_for_run(run_id=orphan_service.run_id) == ()

            monkeypatch.setattr(authority, "record_verified_candidate", original_record)
            orphan_service.resume(
                orphan_plan,
                run_id=orphan_service.run_id,
                evidence_path=orphan_service.evidence_path,
            )
            attributed = authority.candidate_entries_for_run(run_id=orphan_service.run_id)
            assert orphan_handler.transform_calls == 2
            assert {Path(entry.locator) for entry in attributed}.isdisjoint(orphan_paths)
            assert orphan_destination.get("entry") is None
            assert orphan_service.abort(orphan_plan).state is MaintenanceEvidenceState.ABORTED
        finally:
            orphan_source.close()
            orphan_destination.close()
    finally:
        source.close()
        destination.close()
