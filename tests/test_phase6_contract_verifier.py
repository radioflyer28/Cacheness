"""Adversarial self-tests for the fixed Phase 6 contract verifier."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).parents[1]


def _load_verifier():
    """Load the standalone verifier without treating ``tools`` as a package."""
    verifier_path = REPOSITORY_ROOT / "tools" / "verify_phase6_contracts.py"
    spec = spec_from_file_location("phase6_contract_verifier", verifier_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_is_fixed_and_contains_the_strict_projection_node() -> None:
    """The verifier must keep the finite Phase 6 evidence surface explicit."""
    verifier = _load_verifier()

    assert verifier.PHASE6_CONTRACT_NODES == (
        "tests/test_phase6_lookup_contract.py",
        "tests/test_phase6_removal_contract.py",
        "tests/test_phase6_statistics.py",
        "tests/test_phase6_decorator_contract.py",
        "tests/test_phase6_public_api_contract.py",
        "tests/test_phase6_policy_contract.py",
        "tests/contracts/test_phase6_topology_policy.py",
        "tests/test_catalog_projection.py::"
        "test_json_projection_rejects_incompatible_derived_documents",
        "tests/test_sql_cache.py",
    )
    assert set(verifier.RETAINED_LIFECYCLE_NODES) == {
        "tests/test_phase3_local_workflows.py",
        "tests/test_blob_store_atomic_lifecycle.py",
        "tests/test_blob_store_integrity.py",
        "tests/test_blob_store_reconciliation.py",
        "tests/contracts/test_topology_lifecycle.py",
        "tests/test_lifecycle_authority_contract.py",
        "tests/test_supported_topologies.py",
    }
    assert verifier.FIXED_REGRESSION_NODES == {
        "CACH-07 SqlCache regression": ("tests/test_sql_cache.py",),
        "Canonical decorator and key regressions": (
            "tests/test_decorators.py",
            "tests/test_cache_key_consistency.py",
        ),
    }
    assert verifier.REMOTE_EVIDENCE_LABEL == "mocked-candidate; BACK-05 remains Phase 8"


@pytest.mark.parametrize(
    ("source", "filename", "expected"),
    [
        (
            "class AlternateLifecycleEngine:\n    pass\n",
            "src/cacheness/storage/alternate.py",
            ("second lifecycle engine: AlternateLifecycleEngine",),
        ),
        (
            "class CacheLifecycleCoordinator:\n    pass\n",
            "src/cacheness/core.py",
            ("second lifecycle coordinator: CacheLifecycleCoordinator",),
        ),
        (
            "import threading\nlifecycle_lock = threading.Lock()\n",
            "src/cacheness/core.py",
            ("lifecycle lock: threading.Lock",),
        ),
        (
            "import queue\nadmission_queue = queue.Queue()\n",
            "src/cacheness/cache_policy.py",
            ("lifecycle admission queue: queue.Queue",),
        ),
        (
            "store._materialize_authority_store().delete_or_prove_absent(path)\n",
            "src/cacheness/core.py",
            ("direct resource deletion from cache policy",),
        ),
        (
            "def cache_function():\n    return None\n",
            "src/cacheness/decorators.py",
            ("retired compatibility route: cache_function",),
        ),
        (
            "_global_cache = object()\n",
            "src/cacheness/core.py",
            ("hidden global cache: _global_cache",),
        ),
        (
            "store.query_catalog(schema=schema)\n",
            "src/cacheness/core.py",
            ("unbounded cache catalog query",),
        ),
    ],
)
def test_architecture_audit_rejects_executable_phase6_regressions(
    source: str, filename: str, expected: tuple[str, ...]
) -> None:
    """AST checks reject prohibited code shapes with invariant-specific messages."""
    verifier = _load_verifier()

    assert verifier.audit_source(source, filename) == expected


@pytest.mark.parametrize(
    "source",
    [
        "# class AlternateLifecycleEngine: pass\n",
        "message = 'threading.Lock and cache_function are forbidden'\n",
        "store.query_catalog(schema=schema, page_size=1, work_cap=1)\n",
    ],
)
def test_architecture_audit_ignores_comments_strings_and_bounded_calls(source: str) -> None:
    """The fail-closed rules inspect executable structure rather than prose."""
    verifier = _load_verifier()

    assert verifier.audit_source(source, "src/cacheness/core.py") == ()


def test_document_checks_reject_inflated_lifecycle_guarantees() -> None:
    """Cache documentation cannot upgrade topology-limited safety into availability."""
    verifier = _load_verifier()

    assert verifier.audit_contract_text("Every contender succeeds within 10ms.") == (
        "cache contract promises universal contender success",
    )
    assert verifier.audit_contract_text("SQLite and S3 are one ACID transaction.") == (
        "cache contract promises cross-resource ACID",
    )
    assert verifier.audit_contract_text("A 50ms benchmark is a runtime deadline.") == (
        "cache contract turns benchmark into runtime correctness deadline",
    )
    assert verifier.audit_contract_text("Cache policy is topology-specific.") == ()


def test_main_renders_a_failed_strict_projection_check_without_a_false_pass(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The human-readable strict-projection status uses its execution label."""

    verifier = _load_verifier()
    failure = f"{verifier.STRICT_PROJECTION_LABEL}: pytest exited 1"
    monkeypatch.setattr(
        verifier,
        "verify_repository",
        lambda _root: (False, (failure,)),
    )

    exit_code = verifier.main(["--repo-root", str(REPOSITORY_ROOT)])
    output = capsys.readouterr()

    assert exit_code == 1
    assert f"{verifier.STRICT_PROJECTION_LABEL}: see diagnostics" in output.out
    assert f"{verifier.STRICT_PROJECTION_LABEL}: PASS" not in output.out


def test_main_renders_an_architecture_failure_without_a_false_cach_pass(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Mapped static diagnostics cannot contradict a CACH requirement label."""

    verifier = _load_verifier()
    monkeypatch.setattr(
        verifier,
        "verify_repository",
        lambda _root: (False, ("second lifecycle engine: AlternateLifecycleEngine",)),
    )

    exit_code = verifier.main(["--repo-root", str(REPOSITORY_ROOT)])
    output = capsys.readouterr()

    assert exit_code == 1
    assert "CACH-01: see diagnostics" in output.out
    assert "CACH-01: PASS" not in output.out
