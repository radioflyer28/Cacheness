"""Adversarial contracts for the fixed Phase 8 acceptance verifier."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest


REPOSITORY_ROOT = Path(__file__).parents[1]
VERIFIER_PATH = REPOSITORY_ROOT / "tools" / "verify_phase8_contracts.py"

EXPECTED_PLAN_PATHS = tuple(
    ".planning/phases/08-production-gates-and-performance-stabilization/"
    f"08-{number:02d}-PLAN.md"
    for number in range(1, 13)
)
EXPECTED_DECISIONS = {f"D-{number:02d}" for number in range(1, 23)}
EXPECTED_REQUIREMENTS = {
    "BACK-05",
    "QUAL-01",
    "QUAL-02",
    "QUAL-03",
    "QUAL-04",
    "QUAL-05",
    "QUAL-06",
    "QUAL-07",
}


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


def test_fixed_manifest_covers_full_phase_decision_requirement_and_threat_sets() -> None:
    """Planning files cannot redefine the verifier's reviewed evidence surface."""
    verifier = _load_verifier()

    assert verifier.PHASE8_PLAN_PATHS == EXPECTED_PLAN_PATHS
    assert set(verifier.DECISION_NODES) == EXPECTED_DECISIONS
    assert set(verifier.PHASE8_REQUIREMENTS) == EXPECTED_REQUIREMENTS
    assert set(verifier.THREAT_NODES) == set(verifier.PHASE8_THREATS)
    assert verifier.validate_fixed_manifest(REPOSITORY_ROOT) == ()
    assert not hasattr(verifier, "discover_tests")


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


def test_selector_validation_rejects_removed_renamed_duplicate_and_unowned_nodes() -> None:
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

    assert "controlled_performance: UNAVAILABLE" in report
    assert "live_services: UNAVAILABLE" in report
    assert "windows: NOT_QUALIFIED" in report
