"""Adversarial contracts for the fixed Phase 07.1 acceptance verifier."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest


REPOSITORY_ROOT = Path(__file__).parents[1]
VERIFIER_PATH = REPOSITORY_ROOT / "tools" / "verify_phase071_contracts.py"

EXPECTED_PLAN_PATHS = tuple(
    ".planning/milestones/v1.0-phases/07.1-obstore-payload-participant-unification/"
    f"07.1-{number:02d}-PLAN.md"
    for number in range(1, 12)
)
EXPECTED_DECISIONS = {f"D-{number:02d}" for number in range(1, 17)}
EXPECTED_PHASE8_NON_CLAIMS = (
    "real AWS",
    "live PostgreSQL",
    "native platforms",
    "full independent advertised optional-group packaging matrix",
    "RSS/performance budgets",
    "SHA-256-versus-XXH3 benchmarks",
)


def _load_verifier() -> ModuleType:
    """Load the standalone verifier without treating tools as a package."""
    spec = spec_from_file_location("phase071_contract_verifier", VERIFIER_PATH)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_verifier_source(source: str) -> ModuleType:
    """Load one source-mutated verifier for hostile omission tests."""
    module = ModuleType("phase071_contract_verifier_source_mutation")
    module.__file__ = str(VERIFIER_PATH)
    exec(compile(source, str(VERIFIER_PATH), "exec"), module.__dict__)
    return module


def test_fixed_manifest_covers_all_phase_plans_decisions_and_non_claims() -> None:
    """The verifier has a literal closed inventory rather than test discovery."""
    verifier = _load_verifier()

    assert verifier.PHASE071_PLAN_PATHS == EXPECTED_PLAN_PATHS
    assert set(verifier.DECISION_NODES) == EXPECTED_DECISIONS
    assert verifier.PHASE8_NON_CLAIMS == EXPECTED_PHASE8_NON_CLAIMS
    assert verifier.validate_fixed_manifest(REPOSITORY_ROOT) == ()
    assert (
        "tests/test_phase10_sqlcache_removal.py::"
        "test_public_names_and_module_are_naturally_absent"
        in verifier.PHASE071_ALL_NODES
    )
    assert (
        "tests/test_public_api_contract.py::"
        "test_optional_sqlcache_surface_remains_separate_when_dependency_is_blocked"
        not in verifier.PHASE071_ALL_NODES
    )
    assert not hasattr(verifier, "discover_tests")
    assert not hasattr(verifier, "git_diff")


def test_selector_and_inventory_validation_fail_closed_before_pytest() -> None:
    """A removed, renamed, duplicated, malformed, or unowned node is not green."""
    verifier = _load_verifier()
    selector = next(iter(verifier.DECISION_NODES["D-01"]))
    node_path, test_name = selector.split("::", 1)

    assert verifier.validate_selector(selector, REPOSITORY_ROOT) == ()
    assert verifier.validate_selector(f"{node_path}::{test_name}_renamed", REPOSITORY_ROOT)
    assert verifier.validate_selector(f"{node_path}::", REPOSITORY_ROOT)
    assert verifier.validate_selector("../escape.py::test_escape", REPOSITORY_ROOT)
    assert verifier.validate_mapping_inventory(
        decisions={"D-01": (selector, selector)},
        threats=verifier.THREAT_NODES,
        plans=verifier.PHASE071_PLAN_PATHS,
    )


def test_source_mutation_cannot_drop_a_reviewed_plan_or_threat() -> None:
    """Execution tuples cannot become their own mutable manifest oracle."""
    source = VERIFIER_PATH.read_text(encoding="utf-8")
    verifier = _load_verifier_source(
        source.replace(
            "PHASE071_PLAN_PATHS = tuple(_REVIEWED_PLAN_PATHS)",
            "PHASE071_PLAN_PATHS = _REVIEWED_PLAN_PATHS[:-1]",
            1,
        )
    )
    assert verifier.validate_fixed_manifest(REPOSITORY_ROOT)

    verifier = _load_verifier_source(
        source.replace(
            "THREAT_NODES = dict(_REVIEWED_THREAT_NODES)",
            "THREAT_NODES = dict(tuple(_REVIEWED_THREAT_NODES.items())[:-1])",
            1,
        )
    )
    assert verifier.validate_fixed_manifest(REPOSITORY_ROOT)


def test_evidence_parser_threat_runs_every_fail_closed_parser_contract() -> None:
    """T-07.1-05-02 cannot be satisfied by unrelated opaque-ETag evidence."""
    verifier = _load_verifier()

    assert verifier.THREAT_NODES["T-07.1-05-02"] == (
        "tests/test_payload_transport_evidence.py::test_tampered_evidence_fails_closed",
        "tests/test_payload_transport_evidence.py::test_malformed_or_unknown_evidence_is_rejected_before_verification",
        "tests/test_payload_transport_evidence.py::test_noncanonical_transport_evidence_bytes_are_rejected",
    )
    assert (
        "tests/test_payload_transport_evidence.py::test_opaque_etag_is_preserved_without_digest_interpretation"
        in verifier.THREAT_NODES["T-07.1-07-02"]
    )


@pytest.mark.parametrize(
    ("source", "filename", "expected"),
    [
        ("import boto3\n", "src/cacheness/runtime.py", "production boto3 import"),
        (
            "payloads = {}\n",
            "src/cacheness/storage/legacy_payload.py",
            "custom payload dictionary",
        ),
        (
            "def publish():\n    use_multipart = True\n",
            "src/cacheness/storage/obstore_generation_io.py",
            "multipart payload lifecycle",
        ),
        (
            "def select():\n    return legacy_backend\n",
            "src/cacheness/storage/composition.py",
            "runtime payload selector or fallback",
        ),
        (
            "class AlternateLifecycleAuthority:\n    pass\n",
            "src/cacheness/storage/lifecycle.py",
            "duplicate lifecycle authority",
        ),
        (
            "class Handler:\n    def put(self, data, stream, config):\n        pass\n",
            "src/cacheness/handlers.py",
            "handler signature drift",
        ),
    ],
)
def test_ast_source_audit_rejects_each_prohibited_shape(
    source: str, filename: str, expected: str
) -> None:
    """Source gates remain executable and do not trust prose or comments."""
    verifier = _load_verifier()

    assert expected in verifier.audit_source(source, filename)


def test_documentation_states_current_d16_integrity_and_phase8_boundaries() -> None:
    """Public docs cannot revive owner pinning or a transport-hash claim."""
    verifier = _load_verifier()

    assert verifier.audit_documentation(REPOSITORY_ROOT) == ()
