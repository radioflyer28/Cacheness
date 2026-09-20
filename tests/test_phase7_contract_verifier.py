"""Adversarial self-tests for the fixed Phase 7 acceptance verifier."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import shutil
from types import ModuleType

import pytest


REPOSITORY_ROOT = Path(__file__).parents[1]

EXPECTED_PRODUCTION_PATHS = (
    "src/cacheness/core.py",
    "src/cacheness/error_handling.py",
    "src/cacheness/handlers.py",
    "src/cacheness/interfaces.py",
    "src/cacheness/storage/__init__.py",
    "src/cacheness/storage/blob_store.py",
    "src/cacheness/storage/lifecycle.py",
    "src/cacheness/storage/lifecycle_authority.py",
    "src/cacheness/storage/memory_lifecycle_authority.py",
    "src/cacheness/storage/migration.py",
    "src/cacheness/storage/migration_authority.py",
    "src/cacheness/storage/migration_evidence.py",
    "src/cacheness/storage/projections.py",
    "src/cacheness/storage/sqlite_lifecycle_authority.py",
    "src/cacheness/storage/backends/postgresql_lifecycle_authority.py",
    "src/cacheness/storage/obstore_generation_io.py",
    "docs/STORAGE_MIGRATION.md",
    "docs/STORAGE_INITIALIZATION.md",
    ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-COVERAGE.md",
)

EXPECTED_TEST_NODES = (
    "tests/test_migration_cutover.py",
    "tests/test_stored_compatibility.py",
    "tests/test_migration_plan_contract.py",
    "tests/test_migration_inspection.py",
    "tests/test_lifecycle_authority_contract.py",
    "tests/contracts/test_postgresql_lifecycle_authority.py",
    "tests/contracts/test_obstore_generation_io.py",
    "tests/contracts/test_topology_lifecycle.py",
    "tests/test_migration_run_evidence.py",
    "tests/test_projection_sql_atomicity.py",
    "tests/test_migration_remote_contract.py",
    "tests/test_s3_blob_backend.py",
    "tests/test_rebuild_workflow.py",
    "tests/test_handler_registration.py",
    "tests/test_blob_store_read_contract.py",
    "tests/test_blob_store_atomic_lifecycle.py",
    "tests/contracts/test_lifecycle_authority.py",
    "tests/test_migration_public_contract.py",
    "tests/test_phase7_contract_verifier.py",
)

EXPECTED_PROHIBITIONS = (
    "No ordinary constructor, open, initialize, read, cache policy, reconciliation, or cleanup path may migrate, rebuild, adopt, activate, or purge a store.",
    "No work-directory file, filesystem path, symlink, object listing, candidate presence, or derived projection may become lifecycle or cutover authority.",
    "No new lock, queue, lease, sidecar, daemon, scheduler, online writer protocol, or second lifecycle state machine may be added.",
    "No historical compatibility reader, manufactured persisted version, universal native payload converter, or force override may be introduced.",
    "No partial candidate may activate, and activation/finalize may not physically delete the prior valid store.",
    "No unauthenticated evidence, stale source fingerprint, unexplained candidate, or incomplete catalog may be adopted.",
    "No signing key bytes, raw credentials, or secret provider paths may appear in plans, reports, evidence, or logs.",
    "No unbounded inventory/evidence or Phase 8 live PostgreSQL/AWS S3/performance qualification claim may enter Phase 7.",
)


def _load_verifier() -> ModuleType:
    """Load the standalone verifier without importing ``tools`` as a package."""
    verifier_path = REPOSITORY_ROOT / "tools" / "verify_phase7_contracts.py"
    spec = spec_from_file_location("phase7_contract_verifier", verifier_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_verifier_source(source: str) -> ModuleType:
    """Load source-mutated verifier code for omission-resistance tests."""
    verifier_path = REPOSITORY_ROOT / "tools" / "verify_phase7_contracts.py"
    module = ModuleType("phase7_contract_verifier_source_mutation")
    module.__file__ = str(verifier_path)
    exec(compile(source, str(verifier_path), "exec"), module.__dict__)
    return module


def _manifest_copy(tmp_path: Path, verifier: ModuleType) -> Path:
    """Build only the fixed-manifest surface needed for hostile source mutation."""
    root = tmp_path / "repository"
    paths = (
        *verifier.PHASE7_PRODUCTION_PATHS,
        *verifier.PHASE7_TEST_NODES,
        *verifier.PHASE7_PLAN_PATHS,
        *verifier.PHASE7_CONTEXT_PATH,
    )
    for relative_path in paths:
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY_ROOT / relative_path, destination)
    return root


def test_fixed_manifest_is_complete_and_not_discovery_derived() -> None:
    """The acceptance inventory is literal, reviewed, and complete."""
    verifier = _load_verifier()

    assert verifier.PHASE7_PRODUCTION_PATHS == EXPECTED_PRODUCTION_PATHS
    assert verifier.PHASE7_TEST_NODES == EXPECTED_TEST_NODES
    assert verifier.PHASE7_QUICK_TEST_NODES == tuple(
        node
        for node in EXPECTED_TEST_NODES
        if node != "tests/contracts/test_postgresql_lifecycle_authority.py"
    )
    assert verifier.PHASE8_LIVE_UNQUALIFIED_NODES == (
        "tests/integration/test_postgresql_authority.py",
        "tests/integration/test_remote_topology.py",
        "tests/integration/test_s3_generation.py",
    )
    assert set(verifier.MIGRATION_REQUIREMENT_NODES) == {
        "MIGR-03",
        "MIGR-04",
        "MIGR-05",
        "MIGR-06",
    }
    assert set(verifier.DECISION_NODES) == {f"D-{number:02d}" for number in range(1, 23)}
    assert set(verifier.SECURITY_THREAT_NODES) == {
        *(f"T-07-{number:02d}" for number in range(1, 25)),
        *(f"T-07-{number:02d}" for number in range(26, 30)),
        *(f"T-07-{number:02d}" for number in range(31, 35)),
        *(f"T-07-{number:02d}" for number in range(36, 41)),
        *(f"T-07-{number:02d}" for number in range(42, 51)),
        *verifier.GAP_PLAN_THREAT_IDS,
    }
    assert set(verifier.FLAGGED_ASSUMPTION_NODES) == {
        "A-MIGR03",
        "A-MIGR04",
        "A-MIGR05",
        "A-MIGR06",
    }
    assert verifier.PLAN01_PROHIBITIONS == EXPECTED_PROHIBITIONS
    assert not hasattr(verifier, "discover_tests")
    assert not hasattr(verifier, "git_diff")


def test_fixed_mapping_validator_rejects_each_omission() -> None:
    """Requirements, decisions, threats, assumptions, and prohibitions cannot shrink."""
    verifier = _load_verifier()

    assert verifier.validate_mapping_inventory(
        requirements={"MIGR-03": ("x",)},
        decisions={f"D-{number:02d}": ("x",) for number in range(1, 23)},
        threats={threat: ("x",) for threat in verifier.SECURITY_THREAT_NODES},
        assumptions={assumption: ("x",) for assumption in verifier.FLAGGED_ASSUMPTION_NODES},
        prohibitions=EXPECTED_PROHIBITIONS,
    ) == (
        "MIGR mapping differs from the fixed Phase 7 set: missing=['MIGR-04', "
        "'MIGR-05', 'MIGR-06'], unexpected=[]",
    )

    valid_requirements = {requirement: ("x",) for requirement in verifier.MIGRATION_REQUIREMENT_NODES}
    valid_decisions = {f"D-{number:02d}": ("x",) for number in range(1, 23)}
    valid_threats = {threat: ("x",) for threat in verifier.SECURITY_THREAT_NODES}
    valid_assumptions = {assumption: ("x",) for assumption in verifier.FLAGGED_ASSUMPTION_NODES}
    assert verifier.validate_mapping_inventory(
        requirements=valid_requirements,
        decisions=valid_decisions,
        threats=valid_threats,
        assumptions=valid_assumptions,
        prohibitions=EXPECTED_PROHIBITIONS[:-1],
    ) == ("Plan 01 prohibition inventory differs from the fixed eight-row set",)


def test_source_mutation_cannot_drop_a_fixed_test_node() -> None:
    """The mutable execution tuple is checked against an independent fixed oracle."""
    verifier_path = REPOSITORY_ROOT / "tools" / "verify_phase7_contracts.py"
    source = verifier_path.read_text(encoding="utf-8")
    original = "PHASE7_TEST_NODES = tuple(_PHASE7_REVIEWED_TEST_NODES)"
    assert original in source
    verifier = _load_verifier_source(
        source.replace(
            original,
            "PHASE7_TEST_NODES = _PHASE7_REVIEWED_TEST_NODES[:-1]",
            1,
        )
    )

    assert verifier.validate_fixed_manifest(REPOSITORY_ROOT) == (
            "Phase 7 test inventory differs from the fixed reviewed set: "
        "missing=['tests/test_phase7_contract_verifier.py'], unexpected=[]",
    )


def test_source_mutation_cannot_drop_a_fixed_plan_path() -> None:
    """The mutable execution plan tuple cannot hide a later gap plan."""
    verifier_path = REPOSITORY_ROOT / "tools" / "verify_phase7_contracts.py"
    source = verifier_path.read_text(encoding="utf-8")
    original = "PHASE7_PLAN_PATHS = tuple(_PHASE7_REVIEWED_PLAN_PATHS)"
    assert original in source
    verifier = _load_verifier_source(
        source.replace(
            original,
            "PHASE7_PLAN_PATHS = _PHASE7_REVIEWED_PLAN_PATHS[:-1]",
            1,
        )
    )

    assert verifier.validate_fixed_manifest(REPOSITORY_ROOT) == (
        "Phase 7 plan inventory differs from the fixed reviewed set: "
        "missing=['.planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/"
        "07-23-PLAN.md'], unexpected=[]",
    )


@pytest.mark.parametrize(
    ("source", "filename", "expected"),
    [
        (
            "def initialize():\n    OfflineMigrationService(source, destination)\n",
            "src/cacheness/storage/blob_store.py",
            "ordinary lifecycle path invokes offline maintenance: OfflineMigrationService",
        ),
        (
            "class AlternateMigrationAuthority:\n    pass\n",
            "src/cacheness/storage/migration.py",
            "second migration authority: AlternateMigrationAuthority",
        ),
        (
            "import threading\nmigration_lock = threading.Lock()\n",
            "src/cacheness/storage/migration.py",
            "migration coordination primitive: threading.Lock",
        ),
        (
            "def activate():\n    candidate = os.listdir(path)\n    return activate_verified_candidate(candidate)\n",
            "src/cacheness/storage/migration.py",
            "candidate/listing adoption reaches activation",
        ),
        (
            "def rebuild():\n    handler.read(snapshot)\n    verify_manifest(snapshot)\n",
            "src/cacheness/storage/migration.py",
            "handler read precedes integrity verification",
        ),
        (
            "def inspect():\n    return authority.inventory_page()\n",
            "src/cacheness/storage/migration.py",
            "unbounded inventory/evidence operation: inventory_page",
        ),
        (
            "def render(signing_key):\n    print(signing_key)\n",
            "src/cacheness/storage/migration_evidence.py",
            "secret value flows to output: signing_key",
        ),
    ],
)
def test_architecture_audit_rejects_each_executable_prohibition(
    source: str, filename: str, expected: str
) -> None:
    """Every architectural and secrecy check has an adversarial executable fixture."""
    verifier = _load_verifier()

    assert verifier.audit_source(source, filename) == (expected,)


def test_architecture_audit_ignores_comments_strings_and_bounded_safe_code() -> None:
    """Comments, strings, and correctly bounded calls cannot create false positives."""
    verifier = _load_verifier()

    source = """
# OfflineMigrationService and threading.Lock are forbidden in ordinary paths.
message = 'candidate/listing adoption reaches activation'
def inspect(authority):
    return authority.inventory_page(page_size=10, work_cap=100)
"""
    assert verifier.audit_source(source, "src/cacheness/storage/migration.py") == ()


def test_document_and_coverage_audits_reject_false_qualification_and_secrets() -> None:
    """Phase 7 text cannot turn deterministic evidence into remote or platform support."""
    verifier = _load_verifier()

    assert verifier.audit_phase7_text("Live PostgreSQL and AWS S3 are qualified in Phase 7") == (
        "false Phase 7 live-service qualification claim",
    )
    assert verifier.audit_phase7_text("performance distribution guarantees every run") == (
        "Phase 7 performance qualification claim",
    )
    assert verifier.audit_phase7_text("credential_path=/tmp/provider.key") == (
        "secret provider path appears in rendered text",
    )
    assert verifier.audit_phase7_text("Phase 8 qualifies live PostgreSQL/AWS S3.") == ()
    assert verifier.audit_phase7_text(
        "obstore conditional create is bounded-memory in Phase 7"
    ) == ("false obstore bounded-memory conditional-publication claim",)
    assert verifier.audit_phase7_text(
        "Phase 7 automatically selected silent multipart copy publication"
    ) == ("silent multipart-copy publication policy claim",)
    assert verifier.audit_phase7_text(
        "The implementation exposes participant handles to handlers"
    ) == ("handler exposure of participant handle or managed locator",)

    good_coverage = (REPOSITORY_ROOT / ".planning" / "milestones" / "v1.0-phases" / "07-explicit-migration-and-rebuild-cutover" / "07-COVERAGE.md").read_text(encoding="utf-8")
    assert verifier.audit_coverage_document(good_coverage) == ()
    assert verifier.audit_coverage_document(good_coverage.replace('"detected": true', '"detected": false')) == (
        "coverage detector result no longer preserves the recorded public-API signal",
    )


def test_root_safe_manifest_rejects_escape_and_missing_artifacts(tmp_path: Path) -> None:
    """Fixed artifact paths cannot escape the supplied repository root."""
    verifier = _load_verifier()

    assert verifier.validate_paths(tmp_path, ("../outside.py",)) == (
        "manifest path is not repository-relative: ../outside.py",
    )
    assert verifier.validate_paths(tmp_path, ("tests/missing.py",)) == (
        "manifest artifact is missing: tests/missing.py",
    )


def test_main_never_renders_failed_requirement_as_pass(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The result view must not hide a failing requirement behind a green label."""
    verifier = _load_verifier()
    monkeypatch.setattr(
        verifier,
        "verify_repository",
        lambda _root, quick: (False, ("MIGR-04: pytest exited 1",)),
    )

    exit_code = verifier.main(["--quick", "--repo-root", str(REPOSITORY_ROOT)])
    output = capsys.readouterr()

    assert exit_code == 1
    assert "MIGR-04: see diagnostics" in output.out
    assert "MIGR-04: PASS" not in output.out


def test_fixed_manifest_requires_exact_path_and_test_name_selectors() -> None:
    """Every reviewed claim must execute a specific test, never a whole file."""
    verifier = _load_verifier()

    mappings = (
        verifier.MIGRATION_REQUIREMENT_NODES,
        verifier.DECISION_NODES,
        verifier.SECURITY_THREAT_NODES,
        verifier.FLAGGED_ASSUMPTION_NODES,
        verifier.PLAN01_PROHIBITION_NODES,
    )
    for mapping in mappings:
        for selectors in mapping.values():
            assert selectors
            assert len(selectors) == len(set(selectors))
            for selector in selectors:
                path, test_name = selector.split("::", maxsplit=1)
                assert path.startswith("tests/")
                assert path in verifier.PHASE7_TEST_NODES
                assert test_name.startswith("test_")

    assert set(verifier.FLAGGED_ASSUMPTION_NODES) == {
        "A-MIGR03",
        "A-MIGR04",
        "A-MIGR05",
        "A-MIGR06",
    }
    assert {
        "tests/test_migration_cutover.py::test_migration_run_enforces_entry_byte_and_evidence_limits_with_split_plan",
        "tests/test_migration_cutover.py::test_uncheckpointed_candidate_orphan_remains_invisible_unadopted_and_outside_exact_cleanup",
        "tests/test_migration_cutover.py::test_resume_and_abort_staging_use_only_authority_attributed_batches",
    }.issubset(verifier.MIGRATION_REQUIREMENT_NODES["MIGR-04"])
    assert {
        "tests/test_blob_store_atomic_lifecycle.py::test_blobstore_maintenance_canonical_put_replays_projection_free_receipt_after_response_loss",
        "tests/test_rebuild_workflow.py::test_rebuild_response_loss_rederives_operation_id_and_replays_authority_receipt",
        "tests/test_rebuild_workflow.py::test_projection_equipped_rebuild_replays_canonical_receipt_without_preacceptance_or_duplicate_derived_work",
    }.issubset(verifier.MIGRATION_REQUIREMENT_NODES["MIGR-05"])


def test_fixed_manifest_includes_every_gap_plan_threat_exactly_once() -> None:
    """The fixed 54-row gap inventory is exact, owned, and independently checked."""
    verifier = _load_verifier()

    assert len(verifier.GAP_PLAN_THREAT_IDS) == 54
    assert len(set(verifier.GAP_PLAN_THREAT_IDS)) == 54
    assert len(verifier._DECLARED_PHASE7_THREAT_IDS) == 100
    assert set(verifier.GAP_PLAN_THREAT_IDS).issubset(
        verifier.SECURITY_THREAT_NODES
    )
    assert verifier.validate_gap_plan_threat_inventory(REPOSITORY_ROOT) == ()


def test_fixed_manifest_maps_rebuild_cleanup_debt_fence_exactly() -> None:
    """The direct rebuild-debt fence owns the final MIGR-05 evidence."""

    verifier = _load_verifier()
    forged_debt = (
        "tests/test_rebuild_workflow.py::"
        "test_rebuild_cleanup_retry_rejects_forged_debt_without_payload_access"
    )
    forward_fence = (
        "tests/test_rebuild_workflow.py::"
        "test_rebuild_cleanup_debt_fences_forward_methods_and_resume_settles_exact_receipts"
    )
    accepted_evidence = (
        "tests/test_rebuild_workflow.py::"
        "test_rebuild_evidence_rejects_accepted_cleanup_debt"
    )
    verifier_selector = (
        "tests/test_phase7_contract_verifier.py::"
        "test_fixed_manifest_maps_rebuild_cleanup_debt_fence_exactly"
    )

    assert verifier.PHASE7_PLAN_PATHS == tuple(
        ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/"
        f"07-{number:02d}-PLAN.md"
        for number in range(1, 24)
    )
    assert len(verifier.GAP_PLAN_THREAT_IDS) == 54
    assert len(verifier._DECLARED_PHASE7_THREAT_IDS) == 100
    assert verifier.GAP_PLAN_THREAT_OWNERS[
        ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-23-PLAN.md"
    ] == ("T-07-23-01", "T-07-23-02", "T-07-23-03", "T-07-23-04")
    assert verifier.SECURITY_THREAT_NODES["T-07-23-01"] == (forward_fence,)
    assert verifier.SECURITY_THREAT_NODES["T-07-23-02"] == (forward_fence,)
    assert verifier.SECURITY_THREAT_NODES["T-07-23-03"] == (accepted_evidence,)
    assert verifier.SECURITY_THREAT_NODES["T-07-23-04"] == (verifier_selector,)
    assert verifier.SECURITY_THREAT_NODES["T-07-21-03"] == (
        forged_debt,
        forward_fence,
    )
    assert forward_fence in verifier.MIGRATION_REQUIREMENT_NODES["MIGR-05"]
    for decision in ("D-16", "D-19", "D-20", "D-21"):
        assert forward_fence in verifier.DECISION_NODES[decision]
    assert forward_fence in verifier.FLAGGED_ASSUMPTION_NODES["A-MIGR05"]


def test_fixed_manifest_maps_current_three_gap_repairs_exactly() -> None:
    """The latest repair set cannot be satisfied by adjacent whole-file tests."""
    verifier = _load_verifier()

    transform = (
        "tests/test_migration_cutover.py::"
        "test_same_version_different_format_uses_exact_directed_transform_and_destination_manifest"
    )
    s3_abort = (
        "tests/test_migration_remote_contract.py::"
        "test_s3_abort_typed_operational_failures_checkpoint_exact_debt_and_retry"
    )
    abort_count = (
        "tests/test_migration_cutover.py::"
        "test_partial_abort_receipt_counts_only_deleted_or_proven_absent_candidates"
    )
    rebuild_settlement = (
        "tests/test_rebuild_workflow.py::"
        "test_rebuild_cleanup_debt_stays_resumable_until_exact_settlement"
    )
    terminal_evidence = (
        "tests/test_rebuild_workflow.py::"
        "test_rebuild_evidence_rejects_terminal_aborted_cleanup_debt"
    )
    forged_debt = (
        "tests/test_rebuild_workflow.py::"
        "test_rebuild_cleanup_retry_rejects_forged_debt_without_payload_access"
    )
    forward_fence = (
        "tests/test_rebuild_workflow.py::"
        "test_rebuild_cleanup_debt_fences_forward_methods_and_resume_settles_exact_receipts"
    )
    changed_owner = (
        "tests/test_rebuild_workflow.py::"
        "test_rebuild_cleanup_retry_preserves_changed_current_ownership"
    )

    assert verifier.PHASE7_PLAN_PATHS == tuple(
        ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/"
        f"07-{number:02d}-PLAN.md"
        for number in range(1, 24)
    )
    assert len(verifier.GAP_PLAN_THREAT_IDS) == 54
    assert len(verifier._DECLARED_PHASE7_THREAT_IDS) == 100
    assert {
        "T-07-20-01",
        "T-07-20-02",
        "T-07-20-03",
        "T-07-20-04",
        "T-07-21-01",
        "T-07-21-02",
        "T-07-21-03",
        "T-07-21-04",
        "T-07-22-01",
        "T-07-22-02",
        "T-07-22-03",
        "T-07-22-04",
    }.issubset(verifier.GAP_PLAN_THREAT_IDS)
    assert verifier.SECURITY_THREAT_NODES["T-07-20-01"] == (transform,)
    assert verifier.SECURITY_THREAT_NODES["T-07-20-02"] == (s3_abort,)
    assert verifier.SECURITY_THREAT_NODES["T-07-20-03"] == (s3_abort,)
    assert verifier.SECURITY_THREAT_NODES["T-07-20-04"] == (abort_count,)
    assert verifier.SECURITY_THREAT_NODES["T-07-21-01"] == (rebuild_settlement,)
    assert verifier.SECURITY_THREAT_NODES["T-07-21-02"] == (terminal_evidence,)
    assert verifier.SECURITY_THREAT_NODES["T-07-21-03"] == (
        forged_debt,
        forward_fence,
    )
    assert verifier.SECURITY_THREAT_NODES["T-07-21-04"] == (changed_owner,)
    assert verifier.SECURITY_THREAT_NODES["T-07-22-01"] == (
        "tests/test_phase7_contract_verifier.py::"
        "test_fixed_manifest_maps_current_three_gap_repairs_exactly",
    )
    assert verifier.SECURITY_THREAT_NODES["T-07-22-02"] == (
        "tests/test_phase7_contract_verifier.py::"
        "test_fixed_manifest_includes_every_gap_plan_threat_exactly_once",
    )
    assert verifier.SECURITY_THREAT_NODES["T-07-22-03"] == (
        "tests/test_phase7_contract_verifier.py::"
        "test_main_never_renders_failed_requirement_as_pass",
    )
    assert verifier.SECURITY_THREAT_NODES["T-07-22-04"] == (
        "tests/test_phase7_contract_verifier.py::"
        "test_document_and_coverage_audits_reject_false_qualification_and_secrets",
    )
    assert transform in verifier.MIGRATION_REQUIREMENT_NODES["MIGR-04"]
    assert s3_abort in verifier.MIGRATION_REQUIREMENT_NODES["MIGR-04"]
    assert {s3_abort, abort_count, rebuild_settlement, terminal_evidence, forged_debt, changed_owner}.issubset(
        verifier.MIGRATION_REQUIREMENT_NODES["MIGR-05"]
    )
    assert transform in verifier.DECISION_NODES["D-03"]
    assert transform in verifier.DECISION_NODES["D-10"]
    assert s3_abort in verifier.DECISION_NODES["D-16"]
    assert rebuild_settlement in verifier.DECISION_NODES["D-19"]
    assert forged_debt in verifier.DECISION_NODES["D-20"]
    assert changed_owner in verifier.DECISION_NODES["D-21"]
    assert transform in verifier.FLAGGED_ASSUMPTION_NODES["A-MIGR04"]
    assert rebuild_settlement in verifier.FLAGGED_ASSUMPTION_NODES["A-MIGR05"]


def test_fixed_manifest_maps_unified_remote_participant_threats_exactly() -> None:
    """Remote threat coverage follows the unified participant, not retired S3 seams."""
    verifier = _load_verifier()

    exact_create = (
        "tests/contracts/test_obstore_generation_io.py::"
        "test_mocked_s3_collision_and_lost_create_response_settle_only_by_exact_object"
    )
    redacted_diagnostic = (
        "tests/contracts/test_topology_lifecycle.py::"
        "test_remote_inventory_without_snapshot_attribution_is_indeterminate"
    )
    bounded_evidence = (
        "tests/contracts/test_obstore_generation_io.py::"
        "test_mocked_s3_enforces_direct_put_bounds_and_preserves_exact_maintenance_scope"
    )

    assert len(verifier._DECLARED_PHASE7_THREAT_IDS) == 100
    assert verifier.SECURITY_THREAT_NODES["T-07-27"] == (exact_create,)
    assert verifier.SECURITY_THREAT_NODES["T-07-28"] == (redacted_diagnostic,)
    assert verifier.SECURITY_THREAT_NODES["T-07-29"] == (bounded_evidence,)


def test_fixed_gap_supply_chain_threats_map_to_frozen_command_contracts() -> None:
    """Gap-plan supply-chain evidence binds only to frozen command validation."""
    verifier = _load_verifier()

    expected = (
        "T-07-13-SC",
        "T-07-17-SC",
        "T-07-18-SC",
        "T-07-19-SC",
    )
    selector = (
        "tests/test_phase7_contract_verifier.py::"
        "test_fixed_gap_supply_chain_threats_map_to_frozen_command_contracts"
    )
    for threat_id in expected:
        assert verifier.SECURITY_THREAT_NODES[threat_id] == (selector,)
    assert verifier.audit_gap_plan_command_contracts(REPOSITORY_ROOT) == ()


def test_fixed_manifest_rejects_removed_mapped_test_function_while_file_remains(
    tmp_path: Path,
) -> None:
    """Removing a claimed function fails even when unrelated file tests remain."""
    verifier = _load_verifier()
    root = _manifest_copy(tmp_path, verifier)
    target = root / "tests/test_phase7_contract_verifier.py"
    source = target.read_text(encoding="utf-8")
    marker = "def test_fixed_manifest_rejects_removed_mapped_test_function_while_file_remains("
    start = source.index(marker)
    end = source.index("\ndef ", start + len(marker))
    target.write_text(source[:start] + source[end + 1 :], encoding="utf-8")

    errors = verifier.validate_fixed_manifest(root)

    assert (
        "threat mapping references missing test selector: T-07-18-02: "
        "tests/test_phase7_contract_verifier.py::"
        "test_fixed_manifest_rejects_removed_mapped_test_function_while_file_remains"
    ) in errors


def test_fixed_manifest_rejects_removed_current_gap_behavior_while_file_remains(
    tmp_path: Path,
) -> None:
    """Current gap repairs cannot replace exact selectors with passing modules."""
    verifier = _load_verifier()
    cases = (
        (
            "tests/test_migration_cutover.py",
            "def test_same_version_different_format_uses_exact_directed_transform_"
            "and_destination_manifest(",
            "T-07-20-01",
            "tests/test_migration_cutover.py::"
            "test_same_version_different_format_uses_exact_directed_transform_"
            "and_destination_manifest",
        ),
        (
            "tests/test_rebuild_workflow.py",
            "def test_rebuild_cleanup_debt_fences_forward_methods_and_resume_"
            "settles_exact_receipts(",
            "T-07-23-01",
            "tests/test_rebuild_workflow.py::"
            "test_rebuild_cleanup_debt_fences_forward_methods_and_resume_"
            "settles_exact_receipts",
        ),
    )
    for index, (relative_path, marker, threat_id, selector) in enumerate(cases):
        root = _manifest_copy(tmp_path / str(index), verifier)
        target = root / relative_path
        source = target.read_text(encoding="utf-8")
        start = source.index(marker)
        end = source.index("\ndef ", start + len(marker))
        target.write_text(source[:start] + source[end + 1 :], encoding="utf-8")

        errors = verifier.validate_fixed_manifest(root)

        assert (
            f"threat mapping references missing test selector: {threat_id}: {selector}"
        ) in errors


def test_fixed_manifest_rejects_renamed_mapped_test_function_while_file_remains(
    tmp_path: Path,
) -> None:
    """Renaming one claimed function cannot fall back to a passing module."""
    verifier = _load_verifier()
    root = _manifest_copy(tmp_path, verifier)
    target = root / "tests/test_phase7_contract_verifier.py"
    target.write_text(
        target.read_text(encoding="utf-8").replace(
            "test_fixed_manifest_rejects_renamed_mapped_test_function_while_file_remains",
            "test_renamed_contract_evidence",
            1,
        ),
        encoding="utf-8",
    )

    errors = verifier.validate_fixed_manifest(root)

    assert (
        "threat mapping references missing test selector: T-07-18-02: "
        "tests/test_phase7_contract_verifier.py::"
        "test_fixed_manifest_rejects_renamed_mapped_test_function_while_file_remains"
    ) in errors


@pytest.mark.parametrize(
    "threat_id",
    ("T-07-23-01", "T-07-23-02", "T-07-23-03", "T-07-23-04"),
)
def test_fixed_manifest_rejects_gap_threat_without_exact_selector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, threat_id: str
) -> None:
    """Every Plan 23 threat row and selector fails closed before pytest runs."""
    verifier = _load_verifier()
    mutated_mapping = dict(verifier.SECURITY_THREAT_NODES)
    mutated_mapping[threat_id] = ()
    monkeypatch.setattr(verifier, "SECURITY_THREAT_NODES", mutated_mapping)
    assert (
        f"threat mapping has no executable evidence: {threat_id}"
        in verifier.validate_fixed_manifest(REPOSITORY_ROOT)
    )

    root = _manifest_copy(tmp_path, verifier)
    plan_path = ".planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-23-PLAN.md"
    plan = root / plan_path
    source = plan.read_text(encoding="utf-8")
    replacement = f"{threat_id}-X"
    plan.write_text(
        source.replace(f"| {threat_id} |", f"| {replacement} |", 1),
        encoding="utf-8",
    )
    errors = verifier.validate_gap_plan_threat_inventory(root)
    assert (
        "gap threat inventory missing: "
        f"{plan_path}: {threat_id}"
    ) in errors
    assert (
        "gap threat inventory unexpected: "
        f"{plan_path}: {replacement}"
    ) in errors

    duplicate_root = _manifest_copy(tmp_path / "duplicate", verifier)
    duplicate_plan = duplicate_root / plan_path
    duplicate_source = duplicate_plan.read_text(encoding="utf-8")
    row = next(
        line
        for line in duplicate_source.splitlines()
        if line.startswith(f"| {threat_id} |")
    )
    duplicate_plan.write_text(
        duplicate_source.replace(row, f"{row}\n{row}", 1), encoding="utf-8"
    )
    assert (
        "gap threat inventory duplicate: "
        f"{plan_path}: {threat_id}"
    ) in verifier.validate_gap_plan_threat_inventory(duplicate_root)


def test_fixed_pytest_execution_uses_the_validated_selector_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The verifier passes its exact, de-duplicated nodes to pytest unchanged."""
    verifier = _load_verifier()
    executed: list[tuple[str, ...]] = []

    def record_pytest(
        _root: Path, nodes: tuple[str, ...], _label: str, *, options: tuple[str, ...] = ()
    ) -> tuple[bool, str]:
        assert options == ()
        executed.append(nodes)
        return True, "recorded"

    monkeypatch.setattr(verifier, "_run_pytest", record_pytest)
    passed, errors = verifier.verify_repository(REPOSITORY_ROOT, quick=True)

    assert passed, errors
    assert executed == [verifier.fixed_pytest_nodes(True)]
