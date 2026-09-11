"""Adversarial self-tests for the fixed Phase 7 acceptance verifier."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
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
    "src/cacheness/storage/backends/s3_backend.py",
    "docs/STORAGE_MIGRATION.md",
    "docs/STORAGE_INITIALIZATION.md",
    "docs/BACKEND_SELECTION.md",
    ".planning/phases/07-explicit-migration-and-rebuild-cutover/07-COVERAGE.md",
)

EXPECTED_TEST_NODES = (
    "tests/test_migration_cutover.py",
    "tests/test_stored_compatibility.py",
    "tests/test_migration_plan_contract.py",
    "tests/test_migration_inspection.py",
    "tests/test_lifecycle_authority_contract.py",
    "tests/contracts/test_postgresql_lifecycle_authority.py",
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

    good_coverage = (REPOSITORY_ROOT / ".planning" / "phases" / "07-explicit-migration-and-rebuild-cutover" / "07-COVERAGE.md").read_text(encoding="utf-8")
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
    """The fixed 38-row gap inventory is exact, owned, and independently checked."""
    verifier = _load_verifier()

    assert len(verifier.GAP_PLAN_THREAT_IDS) == 38
    assert len(set(verifier.GAP_PLAN_THREAT_IDS)) == 38
    assert len(verifier._DECLARED_PHASE7_THREAT_IDS) == 84
    assert set(verifier.GAP_PLAN_THREAT_IDS).issubset(
        verifier.SECURITY_THREAT_NODES
    )
    assert verifier.validate_gap_plan_threat_inventory(REPOSITORY_ROOT) == ()


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
