"""Static security contracts for protected Phase 8 live qualification workflow."""

from __future__ import annotations

from pathlib import Path
import re


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPOSITORY_ROOT / ".github" / "workflows" / "live_qualification.yml"


def _workflow() -> str:
    """Read the reviewed workflow as source so the contracts need no YAML dependency."""
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def _job_block(workflow: str, job_name: str, next_job: str | None = None) -> str:
    """Return one literal top-level job block for narrow source assertions."""
    start = workflow.index(f"  {job_name}:\n")
    end = workflow.index(f"  {next_job}:\n", start) if next_job else len(workflow)
    return workflow[start:end]


def test_live_workflow_has_only_protected_rc_and_scheduled_triggers() -> None:
    """Untrusted pull requests cannot invoke the workflow or receive live secrets."""
    workflow = _workflow()

    assert "pull_request:" not in workflow
    assert "pull_request_target:" not in workflow
    assert "workflow_dispatch:" in workflow
    assert "schedule:" in workflow
    assert re.search(r"candidate_sha:\n\s+description:.*40-character", workflow)
    assert re.search(r"candidate_sha:\n\s+description:.*\n\s+required: true", workflow)
    assert "contents: read" in workflow


def test_rc_job_checks_out_the_exact_detached_sha_before_services() -> None:
    """The protected RC path has a candidate-bound source identity before secrets."""
    workflow = _workflow()
    release_candidate = _job_block(workflow, "release-candidate", "scheduled-drift")

    assert "environment: live-qualification" in release_candidate
    assert "runs-on: ubuntu-latest" in release_candidate
    assert "CANDIDATE_SHA: ${{ inputs.candidate_sha }}" in release_candidate
    assert "^\"[0-9a-f]{40}\"$" not in release_candidate
    assert "^[0-9a-f]{40}$" in release_candidate
    assert "ref: ${{ inputs.candidate_sha }}" in release_candidate
    assert "persist-credentials: false" in release_candidate
    assert "git rev-parse HEAD" in release_candidate
    assert "candidate checkout must be detached" in release_candidate
    assert release_candidate.index("Prove detached source identity") < release_candidate.index(
        "Run frozen real-service qualification"
    )


def test_live_secrets_are_scoped_to_the_fixed_runner_step() -> None:
    """All four live inputs are present only where the fixed runner executes."""
    workflow = _workflow()
    release_candidate = _job_block(workflow, "release-candidate", "scheduled-drift")
    runner_step = release_candidate[
        release_candidate.index("Run frozen real-service qualification") : release_candidate.index(
            "Upload release-candidate live evidence"
        )
    ]

    for name in (
        "CACHENESS_TEST_POSTGRES_DSN",
        "CACHENESS_TEST_S3_BUCKET",
        "CACHENESS_TEST_MANIFEST_KEY_B64",
        "CACHENESS_TEST_AWS_REGION",
    ):
        assert runner_step.count(f"{name}:") == 1
        assert release_candidate.count(f"{name}:") == 1
        assert f"secrets.{name}" in runner_step
    assert "CACHENESS_PHASE8_QUALIFICATION_ROLE: release_candidate" in runner_step
    assert "--role release_candidate" in runner_step


def test_live_artifacts_are_fixed_retained_and_run_id_addressable() -> None:
    """The controller can address the RC artifact by exact workflow run rather than latest."""
    workflow = _workflow()
    release_candidate = _job_block(workflow, "release-candidate", "scheduled-drift")

    assert "name: phase8-live-qualification-envelope" in release_candidate
    assert "path: build/phase8/live_qualification.json" in release_candidate
    assert "retention-days: 30" in release_candidate
    assert "if: always()" in release_candidate
    assert "GITHUB_RUN_ID" in release_candidate
    assert "remote latency" in workflow.lower()
    assert "p50" not in workflow.lower()
    assert "p99" not in workflow.lower()
    assert "threshold" not in workflow.lower()


def test_scheduled_drift_is_diagnostic_and_cannot_emit_rc_qualification() -> None:
    """Only a clean manual RC run is eligible for immutable release attachment."""
    workflow = _workflow()
    scheduled = _job_block(workflow, "scheduled-drift")

    assert "environment: live-qualification" in scheduled
    assert "CACHENESS_PHASE8_QUALIFICATION_ROLE: scheduled_diagnostic" in scheduled
    assert "--role scheduled_diagnostic" in scheduled
    assert "name: phase8-live-scheduled-diagnostic" in scheduled
    assert "phase8-live-qualification-envelope" not in scheduled
    assert "retention-days: 30" in scheduled
    assert "eligible for release" in scheduled.lower()


def test_actions_are_pinned_to_reviewable_full_commit_shas() -> None:
    """Workflow action references are immutable reviewed revisions, not mutable tags."""
    workflow = _workflow()
    actions = re.findall(r"uses: ([^@\s]+)@([^\s]+)", workflow)

    assert actions
    assert all(re.fullmatch(r"[0-9a-f]{40}", revision) for _action, revision in actions)
