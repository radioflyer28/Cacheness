"""End-to-end contracts for explicit offline memory-store cutover."""

from __future__ import annotations

from pathlib import Path

import pytest

from cacheness.storage import BackendRef, BlobStore, StoreTopology
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
        "candidate",
        "activated_offline",
        "active",
        "rolled_back",
    }
