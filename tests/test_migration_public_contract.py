"""Public-contract coverage for explicit offline storage maintenance."""

from __future__ import annotations

import inspect
from pathlib import Path

from cacheness.storage import BackendRef, BlobStore, StoreTopology
import cacheness.storage as storage


PUBLIC_MAINTENANCE_SYMBOLS = {
    "AbortReceipt",
    "ActivationReceipt",
    "AuthorityIdentitySnapshot",
    "AuthorityPublicationState",
    "CacheBlobMigrationCleanupError",
    "CacheBlobMigrationConfirmationError",
    "CacheBlobMigrationEvidenceError",
    "CacheBlobMigrationEvidenceMismatchError",
    "CacheBlobMigrationOfflineDecisionRequiredError",
    "CacheBlobMigrationPlanStaleError",
    "CacheMigrationOrRebuildRequiredError",
    "CompatibilityDimension",
    "CompatibilityIdentity",
    "CompatibilityMatrix",
    "CompatibilityOutcome",
    "CompatibilityResult",
    "FinalizeReceipt",
    "MaintenanceEvidenceState",
    "MaintenanceRunEvidence",
    "MigrationCompatibilityEdge",
    "MigrationDisposition",
    "MigrationEntryAssessment",
    "MigrationInspection",
    "MigrationPlan",
    "MigrationPlanKind",
    "MigrationPlanState",
    "MigrationReason",
    "MigrationStepResult",
    "MigrationTotals",
    "OfflineMigrationService",
    "PriorStoreReceipt",
    "PurgeReceipt",
    "RebuildExclusion",
    "ReleaseWindow",
    "RollbackReceipt",
    "StoppedWorkerAcknowledgement",
    "VerifiedCandidateReceipt",
    "VersionEdge",
    "render_migration_report",
}


class _SharedMemoryKeyProvider:
    """Provide a test-only signing identity without exposing it in runbook output."""

    def get_key(self) -> bytes:
        return b"p" * 32


def _memory_store(root: Path, provider: _SharedMemoryKeyProvider) -> BlobStore:
    """Create the qualified same-process topology used by the public example."""
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


def test_public_storage_maintenance_exports_and_signatures_are_explicit() -> None:
    """The only supported maintenance facade exposes safe caller-facing contracts."""
    assert PUBLIC_MAINTENANCE_SYMBOLS <= set(storage.__all__)
    for name in PUBLIC_MAINTENANCE_SYMBOLS:
        assert getattr(storage, name) is not None

    constructor = inspect.signature(storage.OfflineMigrationService)
    for name in (
        "source",
        "destination",
        "work_directory",
        "run_id",
        "stopped_workers_acknowledged",
        "compatibility_edges",
    ):
        assert constructor.parameters[name].default is inspect.Parameter.empty

    for method_name, parameter_name in (
        ("resume", "run_id"),
        ("resume", "evidence_path"),
        ("confirm_rebuild", "confirmation"),
        ("finalize", "confirmation"),
        ("purge", "confirmation"),
    ):
        method = getattr(storage.OfflineMigrationService, method_name)
        assert inspect.signature(method).parameters[parameter_name].default is (
            inspect.Parameter.empty
        )


def test_documented_public_workflow_uses_one_model_and_offline_fencing(tmp_path: Path) -> None:
    """The runbook's public API sequence works without an optional live service."""
    provider = _SharedMemoryKeyProvider()
    source = _memory_store(tmp_path / "source", provider)
    destination = _memory_store(tmp_path / "destination", provider)
    try:
        source.put_entry({"answer": 42}, key="example")
        service = storage.OfflineMigrationService(
            source=source,
            destination=destination,
            work_directory=tmp_path / "maintenance",
            run_id="public-runbook-example",
            stopped_workers_acknowledged=True,
            compatibility_edges=(
                storage.MigrationCompatibilityEdge.current_to_current_for_test(),
            ),
        )

        inspection = service.inspect()
        plan = service.plan(inspection)
        encoded_plan = plan.to_canonical_bytes()
        decoded_plan = storage.MigrationPlan.from_canonical_bytes(encoded_plan)
        report = storage.render_migration_report(decoded_plan)
        assert decoded_plan.to_canonical_bytes() == encoded_plan
        assert decoded_plan.digest in report

        service.stage(decoded_plan)
        service.verify(decoded_plan)
        service.activate(decoded_plan)
        assert destination.lifecycle_authority.publication_state() is (
            storage.AuthorityPublicationState.ACTIVATED_OFFLINE
        )

        confirmation = service.finalize_confirmation(decoded_plan)
        service.finalize(decoded_plan, confirmation=confirmation)
        assert destination.get("example") == {"answer": 42}
    finally:
        source.close()
        destination.close()


def test_runbook_marks_stop_conditions_and_non_claims() -> None:
    """Documentation describes the authoritative workflow without unsafe promises."""
    guide = Path("docs/STORAGE_MIGRATION.md").read_text(encoding="utf-8")
    assert "<!-- migration-runbook:start -->" in guide
    assert "<!-- migration-runbook:end -->" in guide
    lowered = " ".join(guide.lower().split())
    for required_text in (
        "current and immediately previous released layouts",
        "stopped-worker",
        "activated_offline",
        "exact run id",
        "evidence path",
        "finalize",
        "separate confirmation",
        "purge",
        "rebuild",
        "derived projection",
        "phase 8",
        "not qualified",
        "no cli",
    ):
        assert required_text in lowered
    for required_disclaimer in (
        "no promise of automatic or seamless migration",
        "not cross-resource acid",
        "no promise of automatic or seamless migration, online-writer coordination",
        "universal payload conversion",
    ):
        assert required_disclaimer in lowered


def test_ordinary_construction_exposes_no_migration_switch_or_cli() -> None:
    """Maintenance stays a library API rather than an ordinary-open or CLI mode."""
    assert "migration" not in inspect.signature(BlobStore).parameters
    assert "migration" not in inspect.signature(BlobStore.initialize).parameters
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "[project.scripts]" not in pyproject
    assert not list(Path("src/cacheness").glob("*cli*.py"))
    initialization_guide = Path("docs/STORAGE_INITIALIZATION.md").read_text(
        encoding="utf-8"
    )
    assert "user_version = 8" in initialization_guide
    assert "wait for the Phase 7 migration tooling" not in initialization_guide
    assert "STORAGE_MIGRATION.md" in initialization_guide
