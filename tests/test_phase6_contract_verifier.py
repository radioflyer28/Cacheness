"""Adversarial self-tests for the fixed Phase 6 contract verifier."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

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


def _load_verifier_source(source: str) -> ModuleType:
    """Execute a source-mutated verifier as import-time inventory evidence."""

    verifier_path = REPOSITORY_ROOT / "tools" / "verify_phase6_contracts.py"
    module = ModuleType("phase6_contract_verifier_source_mutation")
    module.__file__ = str(verifier_path)
    exec(compile(source, str(verifier_path), "exec"), module.__dict__)
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
    )
    assert tuple(verifier.CACH_REQUIREMENT_NODES) == (
        "CACH-01",
        "CACH-02",
        "CACH-03",
        "CACH-04",
        "CACH-05",
        "CACH-06",
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
        "Canonical decorator and key regressions": (
            "tests/test_decorators.py",
            "tests/test_cache_key_consistency.py",
        ),
    }
    assert verifier._CUTOVER_REQUIREMENT_LABELS == tuple(
        verifier.CACH_REQUIREMENT_NODES
    )
    assert verifier.CANONICAL_CUTOVER_NODES == (
        "tests/test_blob_manifest.py",
        "tests/test_filesystem_containment.py",
        "tests/test_cache_signing.py",
        "tests/test_legacy_array_security.py",
        "tests/test_public_api_contract.py",
        "tests/test_query_meta.py",
        "tests/test_store_cache_key_params_config.py",
        "tests/test_query_meta_security.py",
        "tests/test_phase1_quality_gates.py",
        "tests/test_phase6_suite_isolation.py",
        "tests/test_full_suite_environment.py",
        "tests/test_phase3_gap_acceptance.py",
    )
    assert verifier.REMOTE_EVIDENCE_LABEL == "mocked-candidate; BACK-05 remains Phase 8"


def test_cutover_audit_rejects_positive_legacy_calls() -> None:
    """Migrated nodes cannot silently reintroduce a retired cache route."""
    verifier = _load_verifier()

    findings = verifier.audit_cutover_source(
        """
config = CacheConfig(cache_dir='legacy')
cache = UnifiedCache(config)
cache.get('entry')
""",
        "tests/test_cache_signing.py",
    )

    assert findings == (
        "canonical cutover audit: tests/test_cache_signing.py: retired CacheConfig "
        "keyword: cache_dir",
        "canonical cutover audit: tests/test_cache_signing.py: UnifiedCache "
        "construction omits store=",
        "canonical cutover audit: tests/test_cache_signing.py: removed UnifiedCache "
        "surface: get()",
    )


@pytest.mark.parametrize(
    "source",
    (
        """
cache = UnifiedCache(config, store=store)
alias = cache
alias.get('entry')
""",
        """
def configured_cache():
    return UnifiedCache(config, store=store)

cache = configured_cache()
cache.get('entry')
""",
    ),
)
def test_cutover_audit_rejects_cache_aliases_and_known_helper_returns(
    source: str,
) -> None:
    """Retired calls cannot hide behind a proven cache alias or helper."""

    verifier = _load_verifier()

    assert verifier.audit_cutover_source(source, "tests/test_cache_signing.py") == (
        "canonical cutover audit: tests/test_cache_signing.py: removed UnifiedCache "
        "surface: get()",
    )


def test_cutover_audit_does_not_treat_an_unproven_receiver_as_a_cache() -> None:
    """A same-named unrelated value stays outside the targeted cache audit."""

    verifier = _load_verifier()

    assert verifier.audit_cutover_source(
        """
class Mapping:
    def get(self, key):
        return key

cache = Mapping()
cache.get('entry')
""",
        "tests/test_cache_signing.py",
    ) == ()


def test_cutover_audit_allows_only_structural_negative_assertions() -> None:
    """TypeError and explicit signature-absence negatives remain legal evidence."""
    verifier = _load_verifier()

    findings = verifier.audit_cutover_source(
        """
import inspect
import pytest

with pytest.raises(TypeError):
    CacheConfig(cache_dir='legacy')
with pytest.raises(TypeError):
    UnifiedCache(CacheConfig())
assert 'cache_dir' not in inspect.signature(CacheConfig).parameters
""",
        "tests/test_public_api_contract.py",
    )

    assert findings == ()


def test_cutover_inventory_rejects_an_omitted_migration_node() -> None:
    """An import-time execution omission remains distinct from the Plan oracle."""

    verifier_path = REPOSITORY_ROOT / "tools" / "verify_phase6_contracts.py"
    source = verifier_path.read_text(encoding="utf-8")
    original = "CANONICAL_CUTOVER_NODES = tuple(_PLAN_09_11_CANONICAL_CUTOVER_NODES)"
    assert original in source
    verifier = _load_verifier_source(
        source.replace(
            original,
            "CANONICAL_CUTOVER_NODES = _PLAN_09_11_CANONICAL_CUTOVER_NODES[:-1]",
            1,
        )
    )

    assert verifier._validate_cutover_inventory() == (
        "canonical cutover inventory differs from the fixed Plan 09-11 set: "
        "missing=['tests/test_phase3_gap_acceptance.py'], unexpected=[]",
    )


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
