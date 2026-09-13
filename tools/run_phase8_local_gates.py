#!/usr/bin/env python3
"""Run exact, fail-closed local qualification gates for Phase 8."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import subprocess
import sys
from typing import Callable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPOSITORY_ROOT / "tools" / "phase8_evidence.py"
DETERMINISTIC_COMMAND = (
    "uv",
    "run",
    "--all-extras",
    "--group",
    "dev",
    "--frozen",
    "python",
    "tools/verify_phase071_contracts.py",
    "--all",
)
CHILD_TIMEOUT_SECONDS = 900
DETERMINISTIC_FAILURE_EXIT_CODE = 1
UNAVAILABLE_EXTERNAL_EVIDENCE_EXIT_CODE = 2
RELEVANT_SOURCE_PATHS = (
    "pyproject.toml",
    "uv.lock",
    "docs/adr/0001-topology-specific-storage-guarantees.md",
    "src/cacheness",
    "tests",
    "tools/phase8_evidence.py",
    "tools/run_phase8_local_gates.py",
    "tools/verify_phase071_contracts.py",
)
_INCOMPLETE_MARKERS = ("skipped", "xfailed", "xpassed", "no tests ran", "0 passed")
_PASS_ATTESTATION = "phase 07.1 all contract passed"


def _load_evidence_module():
    """Load the sibling utility when this file is run directly or in a test."""
    specification = importlib.util.spec_from_file_location("phase8_evidence", EVIDENCE_PATH)
    if specification is None or specification.loader is None:
        raise RuntimeError("Phase 8 evidence utility is unavailable")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


phase8_evidence = _load_evidence_module()


@dataclass(frozen=True)
class SourceIdentity:
    """The exact repository identity checked before and after one child gate."""

    revision: str
    source_digest: str
    clean: bool


def _git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return revision if len(revision) == 40 and all(character in "0123456789abcdef" for character in revision) else None


def _relevant_sources_are_clean() -> bool:
    try:
        completed = subprocess.run(
            (
                "git",
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                *RELEVANT_SOURCE_PATHS,
            ),
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return not completed.stdout.strip()


def current_source_identity() -> SourceIdentity | None:
    """Return the current reviewed identity without inspecting unrelated files."""
    revision = _git_revision()
    if revision is None:
        return None
    try:
        source_digest = phase8_evidence.relevant_source_digest(
            REPOSITORY_ROOT, RELEVANT_SOURCE_PATHS
        )
    except phase8_evidence.EvidenceValidationError:
        return None
    return SourceIdentity(
        revision=revision,
        source_digest=source_digest,
        clean=_relevant_sources_are_clean(),
    )


def _run_child(
    command: tuple[str, ...], timeout: int
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout,
    )


def _child_result(completed: subprocess.CompletedProcess[str]) -> str:
    """Classify fixed child output without storing untrusted output in evidence."""
    output = f"{completed.stdout}\n{completed.stderr}".casefold()
    if completed.returncode != 0:
        return "failed"
    if any(marker in output for marker in _INCOMPLETE_MARKERS):
        return "skipped" if "skipped" in output else "incomplete"
    if _PASS_ATTESTATION not in output:
        return "incomplete"
    return "passed"


def _payload(result: str) -> dict[str, object]:
    claims = {
        "integrity": "EVIDENCED" if result == "passed" else "NOT_QUALIFIED",
        "recovery": "EVIDENCED" if result == "passed" else "NOT_QUALIFIED",
        "progress": "EVIDENCED" if result == "passed" else "NOT_QUALIFIED",
        "performance": "NOT_QUALIFIED",
    }
    return {
        "command": ["tools/verify_phase071_contracts.py", "--all"],
        "result": result,
        "claim_categories": claims,
        "non_qualifying_classes": [
            evidence_class
            for evidence_class in phase8_evidence.EVIDENCE_CLASSES
            if evidence_class != "deterministic"
        ],
        "subjects": list(phase8_evidence.QUALIFIED_SUBJECTS),
    }


def _write_result(
    output: Path, identity: SourceIdentity, result: str
) -> None:
    status = "PASS" if result == "passed" else "NOT_QUALIFIED"
    envelope = phase8_evidence.make_envelope(
        evidence_class="deterministic",
        status=status,
        revision=identity.revision,
        source_digest=identity.source_digest,
        payload=_payload(result),
    )
    phase8_evidence.write_envelope(output, envelope)


def run_deterministic(
    *,
    output: Path,
    run_child: Callable[[tuple[str, ...], int], subprocess.CompletedProcess[str]] | None = None,
) -> int:
    """Execute only the inherited Phase 07.1 all-mode verifier for one clean SHA."""
    if run_child is None:
        run_child = _run_child
    before = current_source_identity()
    if before is None:
        return 1
    if not before.clean:
        _write_result(output, before, "source_dirty")
        return 1
    try:
        child_result = _child_result(run_child(DETERMINISTIC_COMMAND, CHILD_TIMEOUT_SECONDS))
    except subprocess.TimeoutExpired:
        child_result = "timed_out"
    after = current_source_identity()
    if after is None:
        _write_result(output, before, "source_changed")
        return 1
    if not after.clean:
        _write_result(output, after, "source_dirty")
        return 1
    if after != before:
        _write_result(output, after, "source_changed")
        return 1
    _write_result(output, before, child_result)
    return 0 if child_result == "passed" else 1


def render_evidence_report(envelope) -> str:
    """Render one class-by-class report without promoting absent evidence."""
    lines = ["Phase 8 qualification evidence:"]
    for evidence_class in phase8_evidence.EVIDENCE_CLASSES:
        status = (
            envelope.status
            if evidence_class == envelope.evidence_class
            else "UNAVAILABLE"
        )
        lines.append(f"{evidence_class}: {status}")
    return "\n".join(lines)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the fixed deterministic gate without user-controlled child selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate", choices=("deterministic",))
    parser.add_argument("--output", required=True, type=Path)
    parsed = parser.parse_args(arguments)
    exit_code = run_deterministic(output=parsed.output)
    if not parsed.output.is_file():
        return DETERMINISTIC_FAILURE_EXIT_CODE
    try:
        envelope = phase8_evidence.load_envelope(parsed.output)
    except phase8_evidence.EvidenceValidationError:
        return DETERMINISTIC_FAILURE_EXIT_CODE
    print(render_evidence_report(envelope))
    if exit_code != 0 or envelope.status != "PASS":
        return DETERMINISTIC_FAILURE_EXIT_CODE
    return UNAVAILABLE_EXTERNAL_EVIDENCE_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
