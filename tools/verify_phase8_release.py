#!/usr/bin/env python3
"""Collect only exact-SHA Phase 8 workflow artifacts for release qualification.

The collector intentionally has no "latest" mode.  It derives a new workflow
run from a before/after run-ID set, proves its identity, waits for that exact
run, and downloads only an explicitly declared artifact name.  It records only
safe references and digests; envelope aggregation and publication checks are
added by the following task in this plan.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MAX_ARTIFACT_BYTES = 256 * 1024
MAX_RUN_LIST_ITEMS = 100
_SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class ReleaseEvidenceError(RuntimeError):
    """Raised when release evidence is absent, ambiguous, or untrusted."""


@dataclass(frozen=True)
class WorkflowSpec:
    """One trusted workflow and its fixed artifact/evidence responsibility."""

    workflow: str
    workflow_name: str
    run_id: int
    artifacts: tuple[str, ...]


WORKFLOW_SPECS = {
    "quality.yml": WorkflowSpec(
        workflow="quality.yml",
        workflow_name="Phase 8 deterministic quality",
        run_id=1001,
        artifacts=(
            "phase8-deterministic-envelope",
            "phase8-packaging-envelope",
            "phase8-platform-envelope",
            "phase8-coverage-envelope",
            "phase8-structural-envelope",
        ),
    ),
    "performance.yml": WorkflowSpec(
        workflow="performance.yml",
        workflow_name="Controlled performance qualification",
        run_id=1002,
        artifacts=("controlled-performance-envelope",),
    ),
    "live_qualification.yml": WorkflowSpec(
        workflow="live_qualification.yml",
        workflow_name="Protected live PostgreSQL and Amazon S3 qualification",
        run_id=1003,
        artifacts=("phase8-live-qualification-envelope",),
    ),
}
ARTIFACT_EVIDENCE_CLASS = {
    "phase8-deterministic-envelope": "deterministic",
    "phase8-packaging-envelope": "packaging",
    "phase8-platform-envelope": "platform",
    "phase8-coverage-envelope": "coverage",
    "phase8-structural-envelope": "structural",
    "controlled-performance-envelope": "controlled_performance",
    "phase8-live-qualification-envelope": "live_services",
}


@dataclass(frozen=True)
class CollectedArtifact:
    """Sanitized provenance for one fixed downloaded artifact."""

    workflow: str
    run_id: int
    artifact_name: str
    evidence_class: str
    path: Path
    revision: str
    source_digest: str
    artifact_sha256: str


CommandExecutor = Callable[[tuple[str, ...]], subprocess.CompletedProcess[str]]


def _run(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )


def _require_success(
    command: tuple[str, ...], execute: CommandExecutor
) -> subprocess.CompletedProcess[str]:
    try:
        completed = execute(command)
    except (OSError, subprocess.SubprocessError) as error:
        raise ReleaseEvidenceError(
            "GitHub CLI command could not be completed"
        ) from error
    if completed.returncode != 0:
        raise ReleaseEvidenceError("GitHub CLI command failed")
    return completed


def _load_json(value: str, *, label: str) -> object:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise ReleaseEvidenceError(f"{label} returned invalid JSON") from error
    return decoded


def _workflow_runs(
    spec: WorkflowSpec, execute: CommandExecutor
) -> dict[int, Mapping[str, object]]:
    completed = _require_success(
        (
            "gh",
            "run",
            "list",
            "--workflow",
            spec.workflow,
            "--event",
            "workflow_dispatch",
            "--limit",
            str(MAX_RUN_LIST_ITEMS),
            "--json",
            "databaseId,headSha,workflowName,event,status,conclusion",
        ),
        execute,
    )
    value = _load_json(completed.stdout, label="workflow run list")
    if not isinstance(value, list) or len(value) > MAX_RUN_LIST_ITEMS:
        raise ReleaseEvidenceError("workflow run list is malformed or unbounded")
    normalized: dict[int, Mapping[str, object]] = {}
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {
            "databaseId",
            "headSha",
            "workflowName",
            "event",
            "status",
            "conclusion",
        }:
            raise ReleaseEvidenceError("workflow run record has an unexpected shape")
        run_id = item["databaseId"]
        if type(run_id) is not int or run_id <= 0 or run_id in normalized:
            raise ReleaseEvidenceError("workflow run record has an invalid ID")
        normalized[run_id] = item
    return normalized


def _require_run_identity(
    value: Mapping[str, object],
    *,
    spec: WorkflowSpec,
    candidate_sha: str,
    terminal: bool,
) -> None:
    if value.get("workflowName") != spec.workflow_name:
        raise ReleaseEvidenceError("workflow name does not match declared workflow")
    if value.get("event") != "workflow_dispatch":
        raise ReleaseEvidenceError("workflow event is not release dispatch")
    if value.get("headSha") != candidate_sha:
        raise ReleaseEvidenceError("workflow head SHA does not match candidate SHA")
    if terminal and (
        value.get("status") != "completed" or value.get("conclusion") != "success"
    ):
        raise ReleaseEvidenceError(
            "workflow run did not reach a successful terminal state"
        )


def _run_view(
    spec: WorkflowSpec,
    run_id: int,
    candidate_sha: str,
    execute: CommandExecutor,
    *,
    terminal: bool,
) -> None:
    completed = _require_success(
        (
            "gh",
            "run",
            "view",
            str(run_id),
            "--json",
            "databaseId,headSha,workflowName,event,status,conclusion",
        ),
        execute,
    )
    value = _load_json(completed.stdout, label="workflow run view")
    if not isinstance(value, Mapping) or set(value) != {
        "databaseId",
        "headSha",
        "workflowName",
        "event",
        "status",
        "conclusion",
    }:
        raise ReleaseEvidenceError("workflow run view has an unexpected shape")
    if value.get("databaseId") != run_id:
        raise ReleaseEvidenceError("workflow run view returned a different run ID")
    _require_run_identity(
        value, spec=spec, candidate_sha=candidate_sha, terminal=terminal
    )


def _artifact_payload(
    path: Path, evidence_class: str, candidate_sha: str
) -> tuple[str, str]:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > MAX_ARTIFACT_BYTES
    ):
        raise ReleaseEvidenceError(
            "downloaded artifact file is unsafe or exceeds its bound"
        )
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseEvidenceError("downloaded artifact is not safe JSON") from error
    if not isinstance(value, Mapping):
        raise ReleaseEvidenceError("downloaded artifact has an invalid root")
    revision = value.get("revision")
    source_digest = value.get("source_digest")
    if revision != candidate_sha:
        raise ReleaseEvidenceError("artifact revision does not match candidate SHA")
    if not isinstance(source_digest, str) or not _DIGEST_PATTERN.fullmatch(
        source_digest
    ):
        raise ReleaseEvidenceError("artifact source digest is invalid")
    if evidence_class == "controlled_performance":
        expected_schema, expected_class = (
            "cacheness-phase8-performance-v1",
            "controlled-performance",
        )
        if (
            value.get("schema") != expected_schema
            or value.get("evidence_class") != expected_class
        ):
            raise ReleaseEvidenceError(
                "controlled performance artifact has an invalid identity"
            )
    elif evidence_class == "live_services":
        if (
            value.get("schema") != "phase8-live-qualification-v1"
            or value.get("status") != "QUALIFIED"
        ):
            raise ReleaseEvidenceError("live artifact is not qualifying evidence")
    elif value.get("evidence_class") != evidence_class or value.get("status") != "PASS":
        raise ReleaseEvidenceError("local evidence artifact has an invalid identity")
    return revision, source_digest


def _download_one_artifact(
    *,
    spec: WorkflowSpec,
    run_id: int,
    artifact_name: str,
    candidate_sha: str,
    output_directory: Path,
    execute: CommandExecutor,
) -> CollectedArtifact:
    destination = output_directory / artifact_name
    destination.mkdir(parents=True, exist_ok=False)
    _require_success(
        (
            "gh",
            "run",
            "download",
            str(run_id),
            "-n",
            artifact_name,
            "-D",
            str(destination),
        ),
        execute,
    )
    files = [item for item in destination.rglob("*") if item.is_file()]
    if len(files) != 1 or any(item.is_symlink() for item in files):
        raise ReleaseEvidenceError(
            "artifact download must contain exactly one regular file"
        )
    evidence_class = ARTIFACT_EVIDENCE_CLASS[artifact_name]
    artifact = files[0]
    revision, source_digest = _artifact_payload(artifact, evidence_class, candidate_sha)
    return CollectedArtifact(
        workflow=spec.workflow,
        run_id=run_id,
        artifact_name=artifact_name,
        evidence_class=evidence_class,
        path=artifact,
        revision=revision,
        source_digest=source_digest,
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
    )


def _dispatch_and_identify(
    spec: WorkflowSpec, candidate_sha: str, execute: CommandExecutor
) -> int:
    before = _workflow_runs(spec, execute)
    _require_success(
        (
            "gh",
            "workflow",
            "run",
            spec.workflow,
            "--ref",
            candidate_sha,
            "-f",
            f"candidate_sha={candidate_sha}",
        ),
        execute,
    )
    after = _workflow_runs(spec, execute)
    candidates = [run_id for run_id in after if run_id not in before]
    if len(candidates) != 1:
        raise ReleaseEvidenceError(
            "expected exactly one newly dispatched run for candidate SHA"
        )
    run_id = candidates[0]
    _require_run_identity(
        after[run_id], spec=spec, candidate_sha=candidate_sha, terminal=False
    )
    _run_view(spec, run_id, candidate_sha, execute, terminal=False)
    _require_success(("gh", "run", "watch", str(run_id), "--exit-status"), execute)
    _run_view(spec, run_id, candidate_sha, execute, terminal=True)
    return run_id


def collect_workflow_evidence(
    *,
    candidate_sha: str,
    output_directory: Path,
    execute: CommandExecutor | None = None,
) -> tuple[CollectedArtifact, ...]:
    """Dispatch trusted workflows and collect only their exact candidate artifacts."""
    if not _SHA_PATTERN.fullmatch(candidate_sha):
        raise ReleaseEvidenceError("candidate SHA must be lowercase and 40 characters")
    if execute is None:
        execute = _run
    output_directory.mkdir(parents=True, exist_ok=True)
    if any(output_directory.iterdir()):
        raise ReleaseEvidenceError("collection output directory must be empty")
    collected: list[CollectedArtifact] = []
    try:
        for spec in WORKFLOW_SPECS.values():
            run_id = _dispatch_and_identify(spec, candidate_sha, execute)
            for artifact_name in spec.artifacts:
                collected.append(
                    _download_one_artifact(
                        spec=spec,
                        run_id=run_id,
                        artifact_name=artifact_name,
                        candidate_sha=candidate_sha,
                        output_directory=output_directory,
                        execute=execute,
                    )
                )
    except Exception:
        # Do not return a partial collection as reusable release evidence.
        raise
    if {item.artifact_name for item in collected} != set(ARTIFACT_EVIDENCE_CLASS):
        raise ReleaseEvidenceError(
            "collection did not contain the fixed artifact inventory"
        )
    return tuple(collected)


def main(arguments: Sequence[str] | None = None) -> int:
    """Expose only the bounded exact-SHA collection operation in this task."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect",))
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parsed = parser.parse_args(arguments)
    try:
        collected = collect_workflow_evidence(
            candidate_sha=parsed.candidate_sha, output_directory=parsed.output_dir
        )
    except ReleaseEvidenceError as error:
        print(f"release collection failed: {error}", file=sys.stderr)
        return 2
    print(f"collected {len(collected)} exact-SHA artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
