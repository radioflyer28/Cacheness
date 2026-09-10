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
    OfflineMigrationService,
)
from cacheness.storage.migration_authority import AuthorityPublicationState
from cacheness.storage.migration_evidence import MaintenanceEvidenceState
from cacheness.storage.projections import ProjectionController, ProjectionStatus
from cacheness.storage.sqlite_lifecycle_authority import SQLITE_USER_VERSION
from cacheness.storage.manifest import CURRENT_SQLITE_USER_VERSION


class _SharedMemoryKeyProvider:
    """Test-only provider proving cutover preserves provider-backed signing identity."""

    def get_key(self) -> bytes:
        return b"m" * 32


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
