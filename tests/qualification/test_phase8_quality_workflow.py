"""Static contracts for the non-live Phase 8 quality workflow.

The workflow remains a thin orchestrator.  The reviewed Python gate tools own
all selectors, coverage comparisons, and structural bounds; this module checks
that CI cannot omit or substitute one of those evidence classes.
"""

from __future__ import annotations

from pathlib import Path
import re


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPOSITORY_ROOT / ".github" / "workflows" / "quality.yml"
RUNNER_PATH = REPOSITORY_ROOT / "tools" / "run_phase8_local_gates.py"
RELEASE_GUIDE_PATH = REPOSITORY_ROOT / "docs" / "RELEASE_QUALIFICATION.md"


def _workflow() -> str:
    """Read the workflow as source without adding a YAML parser dependency."""

    return WORKFLOW_PATH.read_text(encoding="utf-8")


def _job_block(workflow: str, job_name: str, next_job: str | None = None) -> str:
    """Return one literal job block for narrow no-drift assertions."""

    start = workflow.index(f"  {job_name}:\n")
    end = workflow.index(f"  {next_job}:\n", start) if next_job else len(workflow)
    return workflow[start:end]


def test_local_gate_runner_exposes_only_fixed_quality_classes() -> None:
    """CI can select a class but cannot provide arbitrary selectors or commands."""

    runner = RUNNER_PATH.read_text(encoding="utf-8")

    for gate in (
        "deterministic",
        "packaging",
        "platform",
        "coverage",
        "structural",
        "core",
    ):
        assert f'"{gate}"' in runner
    assert "GATE_CHOICES" in runner
    assert "tools/run_phase8_packaging.py" in runner
    assert "tools/run_phase8_platform_gates.py" in runner
    assert "tools/verify_phase8_coverage.py" in runner
    assert "tools/run_phase8_scale_gates.py" in runner
    assert "subprocess.run(" in runner
    assert "shell=True" not in runner


def test_quality_workflow_has_exact_non_live_matrix_and_least_permissions() -> None:
    """PR and trusted push rows cover stable Linux and macOS boundary roles."""

    workflow = _workflow()

    assert "pull_request:" in workflow
    assert "push:" in workflow
    assert "pull_request_target:" not in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "fail-fast: false" in workflow
    for minor in ("3.11", "3.12", "3.13", "3.14"):
        assert minor in workflow
    for macos_minor in ("3.11", "3.14"):
        assert macos_minor in workflow
    assert "continue-on-error: true" in workflow


def test_tensorflow_profile_is_absent_from_quality_workflow() -> None:
    """Retiring TensorFlow cannot remove the retained CI evidence structure."""

    workflow = _workflow()

    assert "tensorflow" not in workflow.casefold()
    assert "tools/run_phase8_local_gates.py core" in workflow
    assert "tests/test_phase9_examples.py" in workflow
    for retained_job in (
        "macos-boundary",
        "prerelease-advisory",
        "release-candidate",
        "release-candidate-packaging",
    ):
        assert f"  {retained_job}:\n" in workflow


def test_quality_workflow_keeps_live_and_controlled_performance_out_of_pr_jobs() -> (
    None
):
    """Untrusted CI cannot receive live inputs or turn timing into a local gate."""

    workflow = _workflow().casefold()

    assert "cacheness_test_postgres_dsn" not in workflow
    assert "cacheness_test_s3_bucket" not in workflow
    assert "secrets." not in workflow
    assert "live_qualification" not in workflow
    assert "run_phase8_qualification.py" not in workflow
    assert "performance.yml" not in workflow
    assert "phase8_benchmarks.py" not in workflow
    assert "controlled-performance" not in workflow


def test_release_candidate_binds_one_detached_sha_and_uploads_fixed_class_artifacts() -> (
    None
):
    """The trusted path cannot mix artifacts from an implicit or later revision."""

    workflow = _workflow()
    release_candidate = _job_block(workflow, "release-candidate")

    assert re.search(r"candidate_sha:\n\s+description:.*40-character", workflow)
    assert re.search(r"candidate_sha:\n\s+description:.*\n\s+required: true", workflow)
    assert "environment: release-qualification" in release_candidate
    assert "CANDIDATE_SHA: ${{ inputs.candidate_sha }}" in release_candidate
    assert "^[0-9a-f]{40}$" in release_candidate
    assert "ref: ${{ inputs.candidate_sha }}" in release_candidate
    assert "persist-credentials: false" in release_candidate
    assert "git rev-parse HEAD" in release_candidate
    assert "candidate checkout must be detached" in release_candidate
    assert release_candidate.index(
        "Prove detached source identity"
    ) < release_candidate.index("Run release-candidate quality gates")
    for artifact in (
        "phase8-deterministic-envelope",
        "phase8-packaging-envelope",
        "phase8-platform-envelope",
        "phase8-coverage-envelope",
        "phase8-structural-envelope",
    ):
        assert f"name: {artifact}" in release_candidate
    assert "GITHUB_RUN_ID" in release_candidate
    assert "retention-days: 30" in release_candidate


def test_quality_actions_are_pinned_and_commands_use_frozen_isolated_uv() -> None:
    """Workflow dependency resolution and actions remain reviewable and reproducible."""

    workflow = _workflow()
    actions = re.findall(r"uses: ([^@\s]+)@([^\s]+)", workflow)

    assert actions
    assert all(re.fullmatch(r"[0-9a-f]{40}", revision) for _action, revision in actions)
    assert "uv sync --all-extras --group dev --frozen" in workflow
    assert "uv run --isolated --all-extras --group dev --frozen python" in workflow
    assert "timeout-minutes:" in workflow
    assert "run_phase8_local_gates.py" in workflow


def test_documentation_states_all_evidence_classes_and_nonclaims() -> None:
    """Operators receive one explicit qualification matrix, not an omnibus pass."""

    guide = RELEASE_GUIDE_PATH.read_text(encoding="utf-8")

    for evidence_class in (
        "Deterministic/local",
        "Packaging",
        "Platform",
        "Coverage/quality",
        "Structural",
        "Controlled performance",
        "Live service",
        "Publication",
    ):
        assert evidence_class in guide
    for command in (
        "tools/run_phase8_local_gates.py",
        "tools/run_phase8_packaging.py",
        "tools/run_phase8_platform_gates.py",
        "tools/verify_phase8_coverage.py",
        "tools/run_phase8_scale_gates.py",
        "benchmarks/phase8_benchmarks.py",
        "tools/run_phase8_qualification.py",
        "tools/verify_phase8_release.py",
    ):
        assert command in guide
    for nonclaim in (
        "Windows remains `UNAVAILABLE` / `NOT_QUALIFIED`",
        "cross-resource ACID",
        "every concurrent contender succeeds",
        "unattributed pre-checkpoint orphan",
        "runtime deadline",
        "diagnostic only",
    ):
        assert nonclaim in guide
    assert "Linux" in guide and "macOS" in guide
    assert "TensorFlow" not in guide
    assert "current and immediately previous" in guide
    assert "candidate SHA" in guide and "source digest" in guide
    assert "30 days" in guide
