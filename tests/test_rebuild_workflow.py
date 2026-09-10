"""Contracts for explicit include-all rebuild planning and confirmation."""

from __future__ import annotations

from pathlib import Path

import pytest

from cacheness.error_handling import (
    CacheBlobMigrationEvidenceMismatchError,
    CacheBlobMigrationPlanStaleError,
)
from cacheness.storage import BackendRef, BlobStore, StoreTopology
from cacheness.storage.migration import (
    MigrationCompatibilityEdge,
    MigrationPlanKind,
    MigrationPlanState,
    OfflineMigrationService,
    RebuildExclusion,
)
from cacheness.storage.migration_evidence import MaintenanceEvidenceState


class _SharedMemoryKeyProvider:
    """Provide one test-only signing identity for both offline participants."""

    def get_key(self) -> bytes:
        return b"r" * 32


def _store(root: Path, provider: _SharedMemoryKeyProvider) -> BlobStore:
    """Create an initialized same-process store for rebuild-plan contracts."""

    store = BlobStore(
        StoreTopology(
            payload=BackendRef(name="memory"),
            authority=BackendRef(name="memory"),
        ),
        cache_dir=root,
        manifest_key_provider=provider,
    )
    store.initialize()
    return store


def _service(
    source: BlobStore,
    destination: BlobStore,
    work_directory: Path,
    *,
    run_id: str = "rebuild-run",
) -> OfflineMigrationService:
    """Build one explicitly offline service without a migration fallback."""

    return OfflineMigrationService(
        source=source,
        destination=destination,
        work_directory=work_directory,
        run_id=run_id,
        stopped_workers_acknowledged=True,
        compatibility_edges=(MigrationCompatibilityEdge.current_to_current_for_test(),),
    )


def test_rebuild_plan_defaults_to_every_entry_and_cannot_use_migration_stage(
    tmp_path: Path,
) -> None:
    """A separately created rebuild plan includes the complete source by default."""

    provider = _SharedMemoryKeyProvider()
    source = _store(tmp_path / "source", provider)
    destination = _store(tmp_path / "destination", provider)
    try:
        source.put_entry({"value": "first"}, key="first")
        source.put_entry({"value": "second"}, key="second")
        service = _service(source, destination, tmp_path / "maintenance")

        plan = service.create_rebuild_plan(service.inspect())

        assert plan.plan_kind is MigrationPlanKind.REBUILD
        assert plan.state is MigrationPlanState.PLANNED
        assert plan.exclusions == ()
        assert tuple(item.entry.key for item in plan.included_entries) == ("first", "second")
        assert plan.excluded_entries == ()
        assert plan.included_totals.total_entries == 2
        assert plan.excluded_totals.total_entries == 0
        with pytest.raises(CacheBlobMigrationEvidenceMismatchError, match="rebuild"):
            service.stage(plan)
        assert destination.get("first") is None
    finally:
        source.close()
        destination.close()


def test_rebuild_exclusion_regenerates_exact_plan_and_requires_its_confirmation(
    tmp_path: Path,
) -> None:
    """An omission is accepted only through a newly bound exact rebuild plan."""

    provider = _SharedMemoryKeyProvider()
    source = _store(tmp_path / "source", provider)
    destination = _store(tmp_path / "destination", provider)
    try:
        source.put_entry({"value": "retain"}, key="retain")
        source.put_entry({"value": "omit"}, key="omit")
        service = _service(source, destination, tmp_path / "maintenance")
        inspection = service.inspect()
        include_all = service.create_rebuild_plan(inspection)
        excluded = service.create_rebuild_plan(
            inspection,
            exclusions=(
                RebuildExclusion.exact_keys(
                    ("omit",), reason="operator_confirmed_omission"
                ),
            ),
        )

        assert excluded.plan_id != include_all.plan_id
        assert excluded.digest != include_all.digest
        assert tuple(item.entry.key for item in excluded.included_entries) == ("retain",)
        assert tuple(item.entry.key for item in excluded.excluded_entries) == ("omit",)
        assert excluded.excluded_totals.total_bytes > 0
        assert excluded.exclusions[0]["reason"] == "operator_confirmed_omission"
        assert excluded.exclusions[0]["keys"] == ("omit",)

        with pytest.raises(CacheBlobMigrationEvidenceMismatchError, match="confirmation"):
            service.confirm_rebuild(
                excluded,
                confirmation=service.rebuild_confirmation(include_all),
            )
        assert service.read_evidence().state is MaintenanceEvidenceState.INSPECTED

        service.confirm_rebuild(
            excluded,
            confirmation=service.rebuild_confirmation(excluded),
        )
        evidence = service.read_evidence()
        assert evidence.state is MaintenanceEvidenceState.PLANNED
        assert evidence.plan_digest == excluded.digest
        assert evidence.acknowledgement.plan_digest == excluded.digest
        assert evidence.completed_output_digests["confirm_rebuild"] == service.rebuild_confirmation(
            excluded
        )
        assert destination.get("retain") is None
    finally:
        source.close()
        destination.close()


def test_rebuild_confirmation_rejects_source_revision_drift_before_payload_io(
    tmp_path: Path,
) -> None:
    """An inspected rebuild cannot be confirmed after its source has changed."""

    provider = _SharedMemoryKeyProvider()
    source = _store(tmp_path / "source", provider)
    destination = _store(tmp_path / "destination", provider)
    try:
        source.put_entry({"value": "before"}, key="entry")
        service = _service(source, destination, tmp_path / "maintenance")
        plan = service.create_rebuild_plan(service.inspect())
        source.put_entry({"value": "after"}, key="entry")

        with pytest.raises(CacheBlobMigrationPlanStaleError):
            service.confirm_rebuild(
                plan,
                confirmation=service.rebuild_confirmation(plan),
            )
        assert destination.get("entry") is None
    finally:
        source.close()
        destination.close()
