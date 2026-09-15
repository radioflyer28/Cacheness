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
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile


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
# Full workflow capability is retained for SEED-006, but current release
# qualification is deliberately a smaller literal inventory under D-23.
CURRENT_RELEASE_WORKFLOW_NAMES = ("quality.yml", "live_qualification.yml")
CURRENT_RELEASE_WORKFLOW_SPECS = {
    "quality.yml": WORKFLOW_SPECS["quality.yml"],
    "live_qualification.yml": WORKFLOW_SPECS["live_qualification.yml"],
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
CURRENT_ARTIFACT_EVIDENCE_CLASS = {
    "phase8-deterministic-envelope": "deterministic",
    "phase8-packaging-envelope": "packaging",
    "phase8-platform-envelope": "platform",
    "phase8-coverage-envelope": "coverage",
    "phase8-structural-envelope": "structural",
    "phase8-live-qualification-envelope": "live_services",
}
CURRENT_RELEASE_EVIDENCE_CLASSES = frozenset(CURRENT_ARTIFACT_EVIDENCE_CLASS.values())
DEFERRED_PERFORMANCE_RECORD = {
    "decision": "D-23",
    "qualification": "NOT_QUALIFIED",
    "requirement": "QUAL-06",
    "seed": ".planning/seeds/SEED-006-qualify-controlled-linux-performance.md",
    "status": "DEFERRED",
}
COLLECTION_SCHEMA = "cacheness-phase8-collection-v2"
COLLECTION_MANIFEST_NAME = "phase8-collection.json"
RELEASE_MANIFEST_SCHEMA = "cacheness-phase8-release-qualification-v2"
_FORBIDDEN_MANIFEST_TEXT = re.compile(
    r"://|access[_-]?key|credential|password|secret|token|private[_-]?key",
    re.IGNORECASE,
)


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


def _current_release_inventory() -> None:
    """Fail closed if mutable exports no longer describe the reviewed D-23 set."""
    if tuple(CURRENT_RELEASE_WORKFLOW_SPECS) != CURRENT_RELEASE_WORKFLOW_NAMES:
        raise ReleaseEvidenceError("current release workflow inventory was mutated")
    if set(CURRENT_ARTIFACT_EVIDENCE_CLASS) != {
        artifact
        for spec in CURRENT_RELEASE_WORKFLOW_SPECS.values()
        for artifact in spec.artifacts
    }:
        raise ReleaseEvidenceError("current release artifact inventory was mutated")
    if CURRENT_RELEASE_EVIDENCE_CLASSES != frozenset(
        CURRENT_ARTIFACT_EVIDENCE_CLASS.values()
    ):
        raise ReleaseEvidenceError("current release evidence classes were mutated")


def _validate_trusted_workflow_definition(spec: WorkflowSpec) -> None:
    """Require the reviewed local definition for a remotely visible workflow."""
    path = REPOSITORY_ROOT / ".github" / "workflows" / spec.workflow
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > MAX_ARTIFACT_BYTES
    ):
        raise ReleaseEvidenceError(
            "trusted workflow definition is unavailable or unsafe"
        )
    try:
        definition = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ReleaseEvidenceError(
            "trusted workflow definition is unreadable"
        ) from error
    required_fragments = (
        f"name: {spec.workflow_name}",
        "workflow_dispatch:",
        "candidate_sha:",
        "git rev-parse HEAD",
    )
    if any(fragment not in definition for fragment in required_fragments):
        raise ReleaseEvidenceError(
            "trusted workflow definition lacks release safeguards"
        )


def preflight_collection(
    *, candidate_sha: str, execute: CommandExecutor | None = None
) -> dict[str, object]:
    """Read only current dispatch prerequisites without starting a workflow."""
    if not _SHA_PATTERN.fullmatch(candidate_sha):
        raise ReleaseEvidenceError("candidate SHA must be lowercase and 40 characters")
    if execute is None:
        execute = _run
    _current_release_inventory()
    _require_success(
        ("git", "rev-parse", "--verify", f"{candidate_sha}^{{commit}}"), execute
    )
    _require_success(("gh", "auth", "status"), execute)
    for spec in CURRENT_RELEASE_WORKFLOW_SPECS.values():
        _validate_trusted_workflow_definition(spec)
        _require_success(("gh", "workflow", "view", spec.workflow, "--yaml"), execute)
    _require_success(
        (sys.executable, "tools/verify_phase8_contracts.py", "--quick"), execute
    )
    return {
        "candidate_sha": candidate_sha,
        "workflows": list(CURRENT_RELEASE_WORKFLOW_NAMES),
        "artifacts": sorted(CURRENT_ARTIFACT_EVIDENCE_CLASS),
        "local_contract": "PASS",
    }


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
    evidence_class = CURRENT_ARTIFACT_EVIDENCE_CLASS[artifact_name]
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
    _current_release_inventory()
    preflight_collection(candidate_sha=candidate_sha, execute=execute)
    output_directory.mkdir(parents=True, exist_ok=True)
    if any(output_directory.iterdir()):
        raise ReleaseEvidenceError("collection output directory must be empty")
    collected: list[CollectedArtifact] = []
    try:
        for spec in CURRENT_RELEASE_WORKFLOW_SPECS.values():
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
    if {item.artifact_name for item in collected} != set(
        CURRENT_ARTIFACT_EVIDENCE_CLASS
    ):
        raise ReleaseEvidenceError(
            "collection did not contain the fixed artifact inventory"
        )
    return tuple(collected)


def _canonical_json(value: Mapping[str, object]) -> str:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    )


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    serialized = _canonical_json(value)
    if len(
        serialized.encode("utf-8")
    ) > MAX_ARTIFACT_BYTES or _FORBIDDEN_MANIFEST_TEXT.search(serialized):
        raise ReleaseEvidenceError(
            "release record is oversized or contains unsafe text"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as staged:
        staged.write(serialized)
        staged.flush()
        staged_path = Path(staged.name)
    try:
        staged_path.replace(path)
    finally:
        if staged_path.exists():
            staged_path.unlink()


def write_collection_manifest(
    collection_directory: Path, collected: Sequence[CollectedArtifact]
) -> Path:
    """Persist sanitized run/artifact provenance beside an exact collection."""
    if not collected:
        raise ReleaseEvidenceError("collection cannot be empty")
    if {item.artifact_name for item in collected} != set(
        CURRENT_ARTIFACT_EVIDENCE_CLASS
    ) or len(collected) != len(CURRENT_ARTIFACT_EVIDENCE_CLASS):
        raise ReleaseEvidenceError(
            "collection does not contain the exact artifact inventory"
        )
    revisions = {item.revision for item in collected}
    if len(revisions) != 1:
        raise ReleaseEvidenceError("collection has conflicting revisions")
    document: dict[str, object] = {
        "schema": COLLECTION_SCHEMA,
        "revision": next(iter(revisions)),
        "artifacts": [
            {
                "workflow": item.workflow,
                "run_id": item.run_id,
                "artifact_name": item.artifact_name,
                "evidence_class": item.evidence_class,
                "path": item.path.relative_to(collection_directory).as_posix(),
                "revision": item.revision,
                "source_digest": item.source_digest,
                "artifact_sha256": item.artifact_sha256,
            }
            for item in sorted(collected, key=lambda item: item.artifact_name)
        ],
    }
    destination = collection_directory / COLLECTION_MANIFEST_NAME
    _atomic_write_json(destination, document)
    return destination


def _load_collection_manifest(collection_directory: Path) -> list[dict[str, object]]:
    path = collection_directory / COLLECTION_MANIFEST_NAME
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > MAX_ARTIFACT_BYTES
    ):
        raise ReleaseEvidenceError("collection manifest is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseEvidenceError("collection manifest is unreadable") from error
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "revision",
        "artifacts",
    }:
        raise ReleaseEvidenceError("collection manifest has an unexpected shape")
    if value.get("schema") != COLLECTION_SCHEMA or not isinstance(
        value.get("artifacts"), list
    ):
        raise ReleaseEvidenceError("collection manifest has an invalid identity")
    artifacts = value["artifacts"]
    if len(artifacts) != len(CURRENT_ARTIFACT_EVIDENCE_CLASS):
        raise ReleaseEvidenceError(
            "collection must contain exactly one artifact per declared evidence class"
        )
    normalized: list[dict[str, object]] = []
    required_keys = {
        "workflow",
        "run_id",
        "artifact_name",
        "evidence_class",
        "path",
        "revision",
        "source_digest",
        "artifact_sha256",
    }
    for item in artifacts:
        if not isinstance(item, Mapping) or set(item) != required_keys:
            raise ReleaseEvidenceError(
                "collection artifact record has an unexpected shape"
            )
        normalized.append(dict(item))
    return normalized


def _record_path(collection_directory: Path, record: Mapping[str, object]) -> Path:
    raw_path = record.get("path")
    if not isinstance(raw_path, str):
        raise ReleaseEvidenceError("collection artifact path is invalid")
    relative = PurePosixPath(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ReleaseEvidenceError("collection artifact path escapes its collection")
    candidate = collection_directory / relative
    if candidate.is_symlink() or not candidate.is_file():
        raise ReleaseEvidenceError("collected artifact file is missing or unsafe")
    return candidate


def _validate_recorded_artifact(
    record: Mapping[str, object], *, collection_directory: Path, candidate_sha: str
) -> None:
    artifact_name = record.get("artifact_name")
    evidence_class = record.get("evidence_class")
    if (
        not isinstance(artifact_name, str)
        or CURRENT_ARTIFACT_EVIDENCE_CLASS.get(artifact_name) != evidence_class
        or record.get("revision") != candidate_sha
    ):
        raise ReleaseEvidenceError(
            "collection record does not bind a declared candidate artifact"
        )
    source_digest = record.get("source_digest")
    recorded_digest = record.get("artifact_sha256")
    if not isinstance(source_digest, str) or not _DIGEST_PATTERN.fullmatch(
        source_digest
    ):
        raise ReleaseEvidenceError("collection record source digest is invalid")
    if not isinstance(recorded_digest, str) or not _DIGEST_PATTERN.fullmatch(
        recorded_digest
    ):
        raise ReleaseEvidenceError("collection record artifact digest is invalid")
    path = _record_path(collection_directory, record)
    raw = path.read_bytes()
    if (
        len(raw) > MAX_ARTIFACT_BYTES
        or hashlib.sha256(raw).hexdigest() != recorded_digest
    ):
        raise ReleaseEvidenceError(
            "collected artifact digest does not match its record"
        )
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseEvidenceError("collected artifact is not JSON") from error
    if not isinstance(value, Mapping) or value.get("revision") != candidate_sha:
        raise ReleaseEvidenceError(
            "collected artifact revision does not match candidate SHA"
        )
    if value.get("source_digest") != source_digest:
        raise ReleaseEvidenceError(
            "collected artifact source digest does not match record"
        )
    if evidence_class == "controlled_performance":
        if (
            value.get("schema") != "cacheness-phase8-performance-v1"
            or value.get("evidence_class") != "controlled-performance"
            or value.get("runner_identity") != "cacheness-perf-linux-x64"
        ):
            raise ReleaseEvidenceError(
                "controlled performance evidence has an invalid runner or schema"
            )
    elif evidence_class == "live_services":
        if (
            value.get("schema") != "phase8-live-qualification-v1"
            or value.get("status") != "QUALIFIED"
            or value.get("cleanup_status") != "CLEAN"
        ):
            raise ReleaseEvidenceError("live service evidence is not QUALIFIED/CLEAN")
    elif value.get("evidence_class") != evidence_class or value.get("status") != "PASS":
        raise ReleaseEvidenceError(
            "deterministic evidence class is not a passing exact match"
        )


def aggregate_collection(
    *, candidate_sha: str, collection_directory: Path, output: Path
) -> dict[str, object]:
    """Write a canonical aggregate only from one complete exact candidate collection."""
    if not _SHA_PATTERN.fullmatch(candidate_sha):
        raise ReleaseEvidenceError("candidate SHA must be lowercase and 40 characters")
    _current_release_inventory()
    records = _load_collection_manifest(collection_directory)
    if {record.get("artifact_name") for record in records} != set(
        CURRENT_ARTIFACT_EVIDENCE_CLASS
    ):
        raise ReleaseEvidenceError(
            "collection does not contain the exact artifact inventory"
        )
    classes = [record.get("evidence_class") for record in records]
    if set(classes) != CURRENT_RELEASE_EVIDENCE_CLASSES or len(classes) != len(
        CURRENT_ARTIFACT_EVIDENCE_CLASS
    ):
        raise ReleaseEvidenceError(
            "collection must contain exactly one artifact per evidence class"
        )
    source_digests = {record.get("source_digest") for record in records}
    if len(source_digests) != 1:
        raise ReleaseEvidenceError(
            "collection source digest does not bind one reviewed source identity"
        )
    for record in records:
        _validate_recorded_artifact(
            record,
            collection_directory=collection_directory,
            candidate_sha=candidate_sha,
        )
    source_digest = next(iter(source_digests))
    assert isinstance(source_digest, str)
    evidence = {
        str(record["evidence_class"]): {
            "workflow": record["workflow"],
            "run_id": record["run_id"],
            "artifact_name": record["artifact_name"],
            "artifact_sha256": record["artifact_sha256"],
        }
        for record in sorted(records, key=lambda item: str(item["evidence_class"]))
    }
    manifest: dict[str, object] = {
        "schema": RELEASE_MANIFEST_SCHEMA,
        "revision": candidate_sha,
        "source_digest": source_digest,
        "evidence": evidence,
        "deferred_requirements": [dict(DEFERRED_PERFORMANCE_RECORD)],
    }
    validate_release_aggregate(manifest, candidate_sha=candidate_sha)
    _atomic_write_json(output, manifest)
    return manifest


def validate_release_aggregate(
    aggregate: Mapping[str, object], *, candidate_sha: str
) -> None:
    """Validate the closed current-release aggregate before it can be published."""
    if not _SHA_PATTERN.fullmatch(candidate_sha):
        raise ReleaseEvidenceError("candidate SHA must be lowercase and 40 characters")
    if set(aggregate) != {
        "schema",
        "revision",
        "source_digest",
        "evidence",
        "deferred_requirements",
    }:
        raise ReleaseEvidenceError("release aggregate has an unexpected shape")
    if aggregate.get("schema") != RELEASE_MANIFEST_SCHEMA:
        raise ReleaseEvidenceError("release aggregate has an invalid schema")
    if aggregate.get("revision") != candidate_sha:
        raise ReleaseEvidenceError(
            "release aggregate revision does not match candidate SHA"
        )
    source_digest = aggregate.get("source_digest")
    if not isinstance(source_digest, str) or not _DIGEST_PATTERN.fullmatch(
        source_digest
    ):
        raise ReleaseEvidenceError("release aggregate source digest is invalid")
    evidence = aggregate.get("evidence")
    if (
        not isinstance(evidence, Mapping)
        or set(evidence) != CURRENT_RELEASE_EVIDENCE_CLASSES
    ):
        raise ReleaseEvidenceError("release aggregate evidence inventory is invalid")
    deferred = aggregate.get("deferred_requirements")
    if deferred != [DEFERRED_PERFORMANCE_RECORD]:
        raise ReleaseEvidenceError("release aggregate deferred requirement is invalid")


def verify_published_release(
    *,
    candidate_sha: str,
    tag: str,
    assets: Sequence[Path],
    execute: CommandExecutor | None = None,
) -> dict[str, object]:
    """Read only the exact tag/release/asset state; never repair a release."""
    if (
        not _SHA_PATTERN.fullmatch(candidate_sha)
        or not tag
        or any(character.isspace() for character in tag)
    ):
        raise ReleaseEvidenceError("candidate SHA or release tag is invalid")
    if not assets:
        raise ReleaseEvidenceError("published release requires qualifying assets")
    if execute is None:
        execute = _run
    expected: dict[str, str] = {}
    for asset in assets:
        if asset.is_symlink() or not asset.is_file() or asset.name in expected:
            raise ReleaseEvidenceError("local release asset is unsafe or duplicated")
        expected[asset.name] = hashlib.sha256(asset.read_bytes()).hexdigest()
    target = _require_success(
        ("git", "rev-parse", "--verify", f"{tag}^{{commit}}"), execute
    ).stdout.strip()
    if target != candidate_sha:
        raise ReleaseEvidenceError("release tag does not resolve to candidate SHA")
    completed = _require_success(
        ("gh", "release", "view", tag, "--json", "tagName,isDraft,isImmutable,assets"),
        execute,
    )
    value = _load_json(completed.stdout, label="release view")
    if not isinstance(value, Mapping) or set(value) != {
        "tagName",
        "isDraft",
        "isImmutable",
        "assets",
    }:
        raise ReleaseEvidenceError("release view has an unexpected shape")
    if value.get("tagName") != tag:
        raise ReleaseEvidenceError("release view returned a different tag")
    if value.get("isDraft") is True:
        raise ReleaseEvidenceError("release remains a draft")
    if value.get("isImmutable") is not True:
        raise ReleaseEvidenceError("release is not immutable")
    api_assets = value.get("assets")
    if not isinstance(api_assets, list):
        raise ReleaseEvidenceError("release assets are invalid")
    by_name: dict[str, Mapping[str, object]] = {}
    for item in api_assets:
        if not isinstance(item, Mapping) or set(item) != {"name", "state", "digest"}:
            raise ReleaseEvidenceError("release asset has an unexpected shape")
        name = item.get("name")
        if not isinstance(name, str) or name in by_name:
            raise ReleaseEvidenceError("release asset name is invalid")
        by_name[name] = item
    if set(by_name) != set(expected):
        raise ReleaseEvidenceError(
            "release asset inventory does not match qualification assets"
        )
    _require_success(("gh", "release", "verify", tag), execute)
    for asset in assets:
        api_asset = by_name[asset.name]
        if (
            api_asset.get("state") != "uploaded"
            or api_asset.get("digest") != f"sha256:{expected[asset.name]}"
        ):
            raise ReleaseEvidenceError(
                "release asset state or SHA-256 digest does not match"
            )
        _require_success(("gh", "release", "verify-asset", tag, str(asset)), execute)
    return {
        "tag": tag,
        "revision": candidate_sha,
        "immutable": True,
        "assets": sorted(expected),
    }


def _copy_live_evidence(
    *,
    collection_directory: Path,
    collected: Sequence[CollectedArtifact],
    destination: Path,
) -> None:
    """Expose the exact collected live envelope without selecting another artifact."""
    live = [item for item in collected if item.evidence_class == "live_services"]
    if len(live) != 1:
        raise ReleaseEvidenceError("collection has no unique live-service artifact")
    source = live[0].path
    if destination.exists() or destination.is_symlink():
        raise ReleaseEvidenceError("live evidence destination must not already exist")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if hashlib.sha256(destination.read_bytes()).hexdigest() != live[0].artifact_sha256:
        raise ReleaseEvidenceError(
            "copied live evidence digest does not match collection"
        )


def _require_matching_live_input(
    *, collection_directory: Path, live_evidence: Path | None
) -> None:
    """Reject a caller-supplied live path unless it is the recorded exact artifact."""
    if live_evidence is None:
        return
    records = _load_collection_manifest(collection_directory)
    matches = [
        record for record in records if record.get("evidence_class") == "live_services"
    ]
    if len(matches) != 1 or not live_evidence.is_file() or live_evidence.is_symlink():
        raise ReleaseEvidenceError("live evidence input is unavailable or ambiguous")
    expected = matches[0].get("artifact_sha256")
    if (
        not isinstance(expected, str)
        or hashlib.sha256(live_evidence.read_bytes()).hexdigest() != expected
    ):
        raise ReleaseEvidenceError(
            "live evidence input does not match the recorded workflow artifact"
        )


def main(arguments: Sequence[str] | None = None) -> int:
    """Run bounded collection, aggregation, or read-only publication inspection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("preflight-collection", "collect", "aggregate", "verify-published"),
    )
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--live-output", type=Path)
    parser.add_argument("--live-evidence", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--asset", type=Path, action="append", default=[])
    parsed = parser.parse_args(arguments)
    try:
        if parsed.command == "preflight-collection":
            if (
                any(
                    value is not None
                    for value in (
                        parsed.output_dir,
                        parsed.evidence_dir,
                        parsed.live_output,
                        parsed.live_evidence,
                        parsed.output,
                        parsed.tag,
                    )
                )
                or parsed.asset
            ):
                parser.error("preflight-collection requires only --candidate-sha")
            preflight_collection(candidate_sha=parsed.candidate_sha)
            print("current release collection preflight passed")
            return 0
        if parsed.command == "collect":
            if (
                parsed.output_dir is None
                or any(
                    value is not None
                    for value in (
                        parsed.evidence_dir,
                        parsed.live_evidence,
                        parsed.output,
                        parsed.tag,
                    )
                )
                or parsed.asset
            ):
                parser.error(
                    "collect requires only --candidate-sha, --output-dir, and optional --live-output"
                )
            collected = collect_workflow_evidence(
                candidate_sha=parsed.candidate_sha, output_directory=parsed.output_dir
            )
            write_collection_manifest(parsed.output_dir, collected)
            if parsed.live_output is not None:
                _copy_live_evidence(
                    collection_directory=parsed.output_dir,
                    collected=collected,
                    destination=parsed.live_output,
                )
            print(f"collected {len(collected)} exact-SHA artifacts")
            return 0
        if parsed.command == "aggregate":
            if (
                parsed.evidence_dir is None
                or parsed.output is None
                or parsed.output_dir is not None
            ):
                parser.error(
                    "aggregate requires --candidate-sha, --evidence-dir, and --output"
                )
            _require_matching_live_input(
                collection_directory=parsed.evidence_dir,
                live_evidence=parsed.live_evidence,
            )
            aggregate_collection(
                candidate_sha=parsed.candidate_sha,
                collection_directory=parsed.evidence_dir,
                output=parsed.output,
            )
            print("aggregated exact-SHA release evidence")
            return 0
        if (
            parsed.tag is None
            or not parsed.asset
            or parsed.output_dir is not None
            or parsed.evidence_dir is not None
        ):
            parser.error(
                "verify-published requires --candidate-sha, --tag, and one or more --asset paths"
            )
        verify_published_release(
            candidate_sha=parsed.candidate_sha,
            tag=parsed.tag,
            assets=parsed.asset,
        )
        print("published immutable release verified")
        return 0
    except ReleaseEvidenceError as error:
        print(f"release verification failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
