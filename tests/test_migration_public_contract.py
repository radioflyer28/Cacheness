"""Public-contract coverage for explicit offline storage maintenance."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from cacheness.storage import BackendRef, BlobStore, StoreTopology
import cacheness.storage as storage
import cacheness.storage.migration_authority as migration_authority


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


def test_migration_authority_does_not_export_duplicate_schema_versions() -> None:
    """Maintenance contracts cannot publish schema identities that drift from authority."""
    retired_names = {
        "SQLITE_MIGRATION_AUTHORITY_SCHEMA_VERSION",
        "POSTGRESQL_MIGRATION_AUTHORITY_SCHEMA_VERSION",
        "POSTGRESQL_MIGRATION_AUTHORITY_CAPABILITY",
    }

    assert retired_names.isdisjoint(migration_authority.__all__)
    assert all(not hasattr(migration_authority, name) for name in retired_names)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE_DIRECTORY = PROJECT_ROOT / ".planning/phases/07-explicit-migration-and-rebuild-cutover"
API_COVERAGE_RESULT_START = "<!-- phase7-api-coverage:detector-result:start -->\n```json\n"
API_COVERAGE_RESULT_END = "\n```\n<!-- phase7-api-coverage:detector-result:end -->"
LEGACY_PAYLOAD_SYMBOLS = frozenset(
    {
        "BlobBackend",
        "FilesystemBlobBackend",
        "InMemoryBlobBackend",
        "InMemoryHandlerIO",
        "S3BlobBackend",
        "_S3GenerationIO",
        "BOTO3_AVAILABLE",
        "register_blob_backend",
        "unregister_blob_backend",
        "get_blob_backend",
        "list_blob_backends",
        "write_blob",
        "read_blob",
        "delete_blob",
        "write_blob_stream",
        "read_blob_stream",
    }
)
LEGACY_PAYLOAD_MODULES = (
    "cacheness.storage.backends.blob_backends",
    "cacheness.storage.backends.s3_backend",
)
LEGACY_MULTIPART_SYMBOLS = frozenset(
    {
        "S3MultipartUploadEvidence",
        "S3MultipartUploadPage",
        "_publish_multipart",
        "_create_multipart_upload",
        "multipart_upload_page",
        "create_multipart_upload",
        "upload_part",
        "complete_multipart_upload",
        "abort_multipart_upload",
        "list_multipart_uploads",
        "multipart_threshold",
        "part_size",
        "max_multipart_parts",
        "max_upload_attempts",
    }
)


def _phase7_detector_scope() -> str:
    """Assemble the exact roadmap section and plan bodies sent to GSD's detector."""
    roadmap = (PROJECT_ROOT / ".planning/ROADMAP.md").read_text(encoding="utf-8")
    start = roadmap.index("### Phase 7: Explicit Migration and Rebuild Cutover")
    end = roadmap.index("### Phase 8: Production Gates and Performance Stabilization", start)
    plan_bodies: list[str] = []
    for plan_path in sorted(PHASE_DIRECTORY.glob("07-*-PLAN.md")):
        plan = plan_path.read_text(encoding="utf-8")
        parts = plan.split("\n---\n", 1)
        assert len(parts) == 2, f"{plan_path.name} must have YAML frontmatter"
        plan_bodies.append(parts[1])
    return roadmap[start:end] + "".join(plan_bodies)


def _api_coverage_detector_path() -> Path:
    """Find the active GSD detector without replacing it with test-local logic."""
    roots = []
    configured_root = os.environ.get("CODEX_HOME")
    if configured_root:
        roots.append(Path(configured_root))
    roots.append(Path.home() / ".codex")
    for root in roots:
        candidate = root / "gsd-core/bin/lib/api-coverage.cjs"
        if candidate.is_file():
            return candidate.resolve()
    raise AssertionError("the active GSD api-coverage.cjs detector is unavailable")


def _stored_detector_result() -> dict[str, object]:
    """Read the typed detector output recorded in the coverage declaration."""
    coverage = (PHASE_DIRECTORY / "07-COVERAGE.md").read_text(encoding="utf-8")
    start = coverage.index(API_COVERAGE_RESULT_START) + len(API_COVERAGE_RESULT_START)
    end = coverage.index(API_COVERAGE_RESULT_END, start)
    result = json.loads(coverage[start:end])
    assert isinstance(result, dict)
    return result


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
        source.put_entry(np.array([42]), key="example")
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
        assert np.array_equal(destination.get("example"), np.array([42]))
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
    assert "user_version = 9" in initialization_guide
    assert "wait for the Phase 7 migration tooling" not in initialization_guide
    assert "STORAGE_MIGRATION.md" in initialization_guide


def test_external_api_coverage_declaration_is_detector_backed() -> None:
    """The no-integration declaration records the real detector, not a matrix claim."""
    detector = _api_coverage_detector_path()
    completed = subprocess.run(
        ["node", str(detector), "--json"],
        input=_phase7_detector_scope(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode in {0, 1}, completed.stderr
    assert json.loads(completed.stdout) == _stored_detector_result()

    coverage = (PHASE_DIRECTORY / "07-COVERAGE.md").read_text(encoding="utf-8")
    normalized_coverage = " ".join(coverage.split())
    assert "No external API integration:" in normalized_coverage
    for required_reference in (
        "PostgresqlLifecycleAuthority",
        "tests/contracts/test_postgresql_lifecycle_authority.py",
        "tests/test_migration_remote_contract.py",
            "tests/contracts/test_s3_generation_io.py",
        "do not qualify live PostgreSQL/AWS S3",
        "Phase 8 alone",
    ):
        assert required_reference in normalized_coverage
    assert "| capability | decision | reason |" not in coverage
    assert "qualifies live PostgreSQL/AWS S3" not in coverage
    assert "supports live PostgreSQL/AWS S3" not in coverage


def test_payload_cutover_removes_legacy_runtime_mechanics_and_exports() -> None:
    """Only the obstore participant remains beneath the storage lifecycle seam."""
    source_root = PROJECT_ROOT / "src" / "cacheness"
    defined_or_imported: set[str] = set()
    imported_modules: set[str] = set()
    referenced_symbols: set[str] = set()

    for source_path in source_root.rglob("*.py"):
        text = source_path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                defined_or_imported.add(node.name)
            elif isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
                defined_or_imported.update(
                    alias.asname or alias.name.rsplit(".", 1)[-1]
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.add(node.module)
                defined_or_imported.update(alias.asname or alias.name for alias in node.names)
            elif isinstance(node, ast.Name):
                referenced_symbols.add(node.id)
            elif isinstance(node, ast.Attribute):
                referenced_symbols.add(node.attr)

    assert LEGACY_PAYLOAD_SYMBOLS.isdisjoint(defined_or_imported)
    assert not any(
        module == "boto3"
        or module.startswith("boto3.")
        or module == "botocore"
        or module.startswith("botocore.")
        for module in imported_modules
    )
    assert LEGACY_MULTIPART_SYMBOLS.isdisjoint(
        defined_or_imported | referenced_symbols
    )
    assert all(importlib.util.find_spec(module) is None for module in LEGACY_PAYLOAD_MODULES)

    script = """
import sys
import cacheness.storage as storage
import cacheness.storage.backends as backends

legacy = {
    "BlobBackend", "FilesystemBlobBackend", "InMemoryBlobBackend",
    "InMemoryHandlerIO", "S3BlobBackend", "BOTO3_AVAILABLE",
}
assert legacy.isdisjoint(storage.__all__)
assert legacy.isdisjoint(backends.__all__)
assert all(not hasattr(storage, name) for name in legacy)
assert all(not hasattr(backends, name) for name in legacy)
assert "boto3" not in sys.modules
assert "botocore" not in sys.modules
assert storage.ObstoreGenerationIO is not None
if "PostgresqlLifecycleAuthority" in backends.__all__:
    assert backends.PostgresqlLifecycleAuthority is not None
else:
    assert not hasattr(backends, "PostgresqlLifecycleAuthority")
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(part for part in sys.path if part)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
