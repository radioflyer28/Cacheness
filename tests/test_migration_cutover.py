"""End-to-end contracts for explicit offline memory-store cutover."""

from __future__ import annotations

from pathlib import Path

import pytest

from cacheness.storage import BackendRef, BlobStore, StoreTopology
from cacheness.error_handling import CacheBlobMigrationOfflineDecisionRequiredError
from cacheness.storage.migration import (
    MigrationCompatibilityEdge,
    MigrationDisposition,
    OfflineMigrationService,
)
from cacheness.storage.migration_authority import AuthorityPublicationState
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
