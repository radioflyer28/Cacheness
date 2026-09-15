"""Adversarial contracts for the fixed Phase 8 acceptance verifier."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import subprocess
from types import ModuleType

import pytest


REPOSITORY_ROOT = Path(__file__).parents[1]
VERIFIER_PATH = REPOSITORY_ROOT / "tools" / "verify_phase8_contracts.py"

EXPECTED_PLAN_PATHS = tuple(
    ".planning/phases/08-production-gates-and-performance-stabilization/"
    f"08-{number:02d}-PLAN.md"
    for number in range(1, 16)
)
EXPECTED_DECISIONS = {f"D-{number:02d}" for number in range(1, 24)}
EXPECTED_REQUIREMENTS = {
    "BACK-05",
    "QUAL-01",
    "QUAL-02",
    "QUAL-03",
    "QUAL-04",
    "QUAL-05",
    "QUAL-07",
}

DEFERRED_PERFORMANCE_REQUIREMENT = "QUAL-06"
SEED006_PATH = ".planning/seeds/SEED-006-qualify-controlled-linux-performance.md"


def _load_verifier() -> ModuleType:
    """Load the standalone verifier without making ``tools`` a package."""
    specification = spec_from_file_location("phase8_contract_verifier", VERIFIER_PATH)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _load_source_mutation(source: str) -> ModuleType:
    """Execute one hostile verifier copy to prove its inventory is closed."""
    module = ModuleType("phase8_contract_verifier_source_mutation")
    module.__file__ = str(VERIFIER_PATH)
    exec(compile(source, str(VERIFIER_PATH), "exec"), module.__dict__)
    return module


def test_fixed_manifest_covers_full_phase_decision_requirement_and_threat_sets() -> (
    None
):
    """Planning files cannot redefine the verifier's reviewed evidence surface."""
    verifier = _load_verifier()

    assert verifier.PHASE8_PLAN_PATHS == EXPECTED_PLAN_PATHS
    assert set(verifier.DECISION_NODES) == EXPECTED_DECISIONS
    assert set(verifier.PHASE8_REQUIREMENTS) == EXPECTED_REQUIREMENTS
    assert set(verifier.THREAT_NODES) == set(verifier.PHASE8_THREATS)
    assert {
        threat: verifier.THREAT_NODES[threat]
        for threat in ("T-08-13-01", "T-08-13-02", "T-08-13-03")
    } == {
        threat: (
            "tests/performance/test_phase8_benchmarks.py::"
            "test_preflight_runner_emits_only_the_bounded_eligibility_record",
        )
        for threat in ("T-08-13-01", "T-08-13-02", "T-08-13-03")
    }
    assert verifier.validate_fixed_manifest(REPOSITORY_ROOT) == ()
    assert not hasattr(verifier, "discover_tests")


def test_fixed_manifest_binds_live_configuration_preflight_gap() -> None:
    """Plan 15 keeps its protected-live prerequisite boundary executable."""
    verifier = _load_verifier()

    assert verifier.PHASE8_PLAN_PATHS == EXPECTED_PLAN_PATHS
    assert {
        threat: verifier.THREAT_NODES[threat]
        for threat in (
            "T-08-15-01",
            "T-08-15-02",
            "T-08-15-03",
            "T-08-15-04",
            "T-08-15-05",
        )
    } == {
        "T-08-15-01": (
            "tests/qualification/test_phase8_evidence.py::test_preflight_accepts_sanitized_configuration_without_external_effects",
        ),
        "T-08-15-02": (
            "tests/qualification/test_phase8_evidence.py::test_preflight_rejects_missing_invalid_and_disallowed_configuration_without_external_effects",
        ),
        "T-08-15-03": (
            "tests/qualification/test_phase8_evidence.py::test_preflight_requires_clean_source_and_reviewed_cleanup_contract",
        ),
        "T-08-15-04": (
            "tests/qualification/test_phase8_evidence.py::test_preflight_requires_clean_source_and_reviewed_cleanup_contract",
        ),
        "T-08-15-05": (
            "tests/qualification/test_phase8_evidence.py::test_preflight_cli_never_runs_or_writes_qualification_evidence",
        ),
    }
    assert verifier.validate_fixed_manifest(REPOSITORY_ROOT) == ()


def test_fixed_manifest_rejects_source_mutation_that_drops_plan_or_threat() -> None:
    """The executable inventory cannot silently follow a deleted declaration."""
    source = VERIFIER_PATH.read_text(encoding="utf-8")
    verifier = _load_source_mutation(
        source.replace(
            "PHASE8_PLAN_PATHS = tuple(_REVIEWED_PLAN_PATHS)",
            "PHASE8_PLAN_PATHS = _REVIEWED_PLAN_PATHS[:-1]",
            1,
        )
    )
    assert verifier.validate_fixed_manifest(REPOSITORY_ROOT)

    verifier = _load_source_mutation(
        source.replace(
            "THREAT_NODES = dict(_REVIEWED_THREAT_NODES)",
            "THREAT_NODES = dict(tuple(_REVIEWED_THREAT_NODES.items())[:-1])",
            1,
        )
    )
    assert verifier.validate_fixed_manifest(REPOSITORY_ROOT)


def test_selector_validation_rejects_removed_renamed_duplicate_and_unowned_nodes() -> (
    None
):
    """A selector is syntax-checked against a module-level test before execution."""
    verifier = _load_verifier()
    selector = verifier.DECISION_NODES["D-01"][0]
    path, name = selector.split("::", 1)

    assert verifier.validate_selector(selector, REPOSITORY_ROOT) == ()
    assert verifier.validate_selector(f"{path}::{name}_renamed", REPOSITORY_ROOT)
    assert verifier.validate_selector("../outside.py::test_escape", REPOSITORY_ROOT)
    assert verifier.validate_mapping_inventory(
        decisions={"D-01": (selector, selector)},
        threats=verifier.THREAT_NODES,
        plans=verifier.PHASE8_PLAN_PATHS,
    )


@pytest.mark.parametrize(
    ("source", "filename", "expected"),
    [
        (
            "import boto3\n",
            "src/cacheness/storage/runtime.py",
            "production boto3 import",
        ),
        (
            "from .legacy_payload import LegacyPayload\n",
            "src/cacheness/storage/composition.py",
            "legacy payload mechanic",
        ),
        (
            "from threading import Lock\nlock = Lock()\n",
            "src/cacheness/storage/lifecycle.py",
            "lifecycle coordination primitive",
        ),
        (
            "class AlternateLifecycleAuthority:\n    pass\n",
            "src/cacheness/storage/lifecycle.py",
            "parallel lifecycle authority",
        ),
        (
            "digest = xxh3_64_hexdigest(payload)\n",
            "src/cacheness/storage/blob_store.py",
            "canonical digest change",
        ),
        (
            "status = 'PASS' if unavailable else 'UNAVAILABLE'\n",
            "tools/run_phase8_qualification.py",
            "unavailable-to-pass conversion",
        ),
    ],
)
def test_ast_source_audit_rejects_prohibited_architecture_regressions(
    source: str, filename: str, expected: str
) -> None:
    """Comments and strings do not trigger audits; executable regressions do."""
    verifier = _load_verifier()

    assert expected in verifier.audit_source(source, filename)


def test_external_statuses_are_explicit_nonclaims() -> None:
    """No local result can fill the Windows, performance, or live proof slots."""
    verifier = _load_verifier()

    report = verifier.render_external_statuses({})

    assert "controlled_performance: DEFERRED" in report
    assert "live_services: UNAVAILABLE" in report
    assert "windows: NOT_QUALIFIED" in report


def test_all_mode_uses_fixed_local_gate_commands_before_external_reporting(
    tmp_path: Path,
) -> None:
    """All-mode cannot report only external statuses after skipping local proof."""
    verifier = _load_verifier()

    commands = verifier.local_gate_commands(tmp_path)

    assert [
        command[command.index("tools/run_phase8_local_gates.py") + 1]
        for command in commands
    ] == ["all", "platform"]
    assert all(
        command[:8]
        == (
            "uv",
            "run",
            "--isolated",
            "--all-extras",
            "--group",
            "dev",
            "--frozen",
            "python",
        )
        for command in commands
    )


def test_all_mode_reports_unavailable_local_evidence_without_stopping(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unavailable packaging/platform rows remain blocking nonclaims, not failures."""
    verifier = _load_verifier()

    def fake_run(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        if "pytest" in command:
            return subprocess.CompletedProcess(command, 0)
        gate = command[command.index("tools/run_phase8_local_gates.py") + 1]
        if gate == "all":
            output = Path(command[command.index("--output-dir") + 1]) / "packaging.json"
            output.parent.mkdir(parents=True)
            output.write_text(
                json.dumps({"evidence_class": "packaging", "status": "UNAVAILABLE"}),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 1)
        output = Path(command[command.index("--output") + 1])
        output.write_text(
            json.dumps({"evidence_class": "platform", "status": "UNAVAILABLE"}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(verifier, "_run", fake_run)

    assert verifier._run_local("all") == (
        0,
        {"packaging": "UNAVAILABLE", "platform": "UNAVAILABLE"},
    )
    assert verifier.main(["--all"]) == 2
    report = capsys.readouterr().out
    assert "packaging: UNAVAILABLE" in report
    assert "platform: UNAVAILABLE" in report


def test_fixed_manifest_covers_deferred_performance_decision() -> None:
    """Plan 14 closes current evidence without treating QUAL-06 as complete."""
    verifier = _load_verifier()

    assert any(path.endswith("08-14-PLAN.md") for path in verifier.PHASE8_PLAN_PATHS)
    assert verifier.PHASE8_REQUIREMENTS == tuple(
        requirement
        for requirement in verifier._REVIEWED_REQUIREMENTS
        if requirement != DEFERRED_PERFORMANCE_REQUIREMENT
    )
    assert verifier.DEFERRED_REQUIREMENTS == (DEFERRED_PERFORMANCE_REQUIREMENT,)
    assert "D-23" in verifier.DECISION_NODES
    assert {f"T-08-14-{number:02d}" for number in range(1, 5)}.issubset(
        verifier.THREAT_NODES
    )
    assert verifier.validate_fixed_manifest(REPOSITORY_ROOT) == ()


def test_deferred_performance_status_is_nonblocking_but_not_qualified() -> None:
    """The controlled runner is visible as a nonclaim, never a synthetic pass."""
    verifier = _load_verifier()

    report = verifier.render_external_statuses({})

    assert "controlled_performance: DEFERRED" in report
    assert "qualification: NOT_QUALIFIED" in report
    assert DEFERRED_PERFORMANCE_REQUIREMENT in report
    assert SEED006_PATH in report
    assert (
        verifier.external_status_is_blocking("controlled_performance", "DEFERRED")
        is False
    )
    assert (
        verifier.external_status_is_blocking("controlled_performance", "NOT_QUALIFIED")
        is False
    )
    assert verifier.external_status_is_blocking("live_services", "UNAVAILABLE") is True


def test_all_mode_still_blocks_on_unavailable_current_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deferral cannot cure a missing required packaging or live result."""
    verifier = _load_verifier()

    monkeypatch.setattr(
        verifier,
        "_run_local",
        lambda _mode: (0, {"packaging": "UNAVAILABLE", "platform": "PASS"}),
    )

    assert verifier.main(["--all"]) == 2


def test_release_documentation_preserves_seed006_nonclaim() -> None:
    """Public qualification text must preserve the retained, diagnostic-only harness."""
    document = (REPOSITORY_ROOT / "docs" / "RELEASE_QUALIFICATION.md").read_text(
        encoding="utf-8"
    )

    assert "DEFERRED" in document
    assert "NOT_QUALIFIED" in document
    assert SEED006_PATH in document
    assert "macOS timings are diagnostic" in document
    assert "do not establish Linux equivalence" in document
    assert "not collected or published" in document
