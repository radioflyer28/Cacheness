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
    "tools/run_phase8_packaging.py",
    "tools/run_phase8_platform_gates.py",
    "tools/run_phase8_scale_gates.py",
    "tools/verify_phase8_coverage.py",
    "tools/verify_phase071_contracts.py",
)
_INCOMPLETE_MARKERS = ("skipped", "xfailed", "xpassed", "no tests ran", "0 passed")
_PASS_ATTESTATION = "phase 07.1 all contract passed"
GATE_CHOICES = (
    "deterministic",
    "packaging",
    "platform",
    "coverage",
    "structural",
    "core",
    "all",
)
PACKAGING_TOOL = REPOSITORY_ROOT / "tools" / "run_phase8_packaging.py"
PLATFORM_TOOL = REPOSITORY_ROOT / "tools" / "run_phase8_platform_gates.py"
COVERAGE_TOOL = REPOSITORY_ROOT / "tools" / "verify_phase8_coverage.py"
STRUCTURAL_TOOL = REPOSITORY_ROOT / "tools" / "run_phase8_scale_gates.py"
COVERAGE_REPORT = REPOSITORY_ROOT / "build" / "phase8" / "coverage.json"
COVERAGE_XML = REPOSITORY_ROOT / "build" / "phase8" / "coverage.xml"
COVERAGE_BASELINE = (
    REPOSITORY_ROOT / "tests" / "qualification" / "phase8_coverage_baseline.json"
)
COVERAGE_MEASUREMENT_COMMAND = (
    sys.executable,
    "-m",
    "pytest",
    "-q",
    "-o",
    "log_cli=false",
    "-m",
    "not (live_postgresql or live_aws_s3 or live_remote)",
    "-x",
    "--cov=cacheness",
    "--cov-branch",
    f"--cov-report=json:{COVERAGE_REPORT.relative_to(REPOSITORY_ROOT)}",
    f"--cov-report=xml:{COVERAGE_XML.relative_to(REPOSITORY_ROOT)}",
)


def _load_evidence_module():
    """Load the sibling utility when this file is run directly or in a test."""
    specification = importlib.util.spec_from_file_location(
        "phase8_evidence", EVIDENCE_PATH
    )
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
    return (
        revision
        if len(revision) == 40
        and all(character in "0123456789abcdef" for character in revision)
        else None
    )


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


def _write_result(output: Path, identity: SourceIdentity, result: str) -> None:
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
    run_child: Callable[[tuple[str, ...], int], subprocess.CompletedProcess[str]]
    | None = None,
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
        child_result = _child_result(
            run_child(DETERMINISTIC_COMMAND, CHILD_TIMEOUT_SECONDS)
        )
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


def _run_required_child(command: tuple[str, ...], *, timeout: int) -> int:
    """Run one reviewed child command without allowing shell interpolation."""

    try:
        completed = _run_child(command, timeout)
    except subprocess.TimeoutExpired:
        return 1
    return 0 if completed.returncode == 0 else 1


def _validate_class_envelope(output: Path, evidence_class: str) -> int:
    """Require the child to write one matching passing envelope."""

    try:
        envelope = phase8_evidence.load_envelope(output)
    except phase8_evidence.EvidenceValidationError:
        return 1
    return int(not phase8_evidence.is_qualification_evidence(envelope, evidence_class))


def run_packaging(*, output: Path) -> int:
    """Run the fixed isolated-wheel qualification with no feature overrides."""

    return (
        _validate_class_envelope(output, "packaging")
        if not _run_required_child(
            (sys.executable, str(PACKAGING_TOOL), "--output", str(output)),
            timeout=CHILD_TIMEOUT_SECONDS,
        )
        else 1
    )


def run_platform(
    *,
    output: Path,
    expected_os: str,
    python_minor: str,
    feature_profile: str,
    advisory: bool,
) -> int:
    """Delegate one identity-bound platform row to its fixed runner."""

    command = (
        sys.executable,
        str(PLATFORM_TOOL),
        "--expected-os",
        expected_os,
        "--python-minor",
        python_minor,
        "--feature-profile",
        feature_profile,
        "--output",
        str(output),
    )
    if advisory:
        command = (*command, "--advisory")
    return (
        _validate_class_envelope(output, "platform")
        if not _run_required_child(command, timeout=CHILD_TIMEOUT_SECONDS)
        else 1
    )


def _class_payload(evidence_class: str) -> dict[str, object]:
    """Return the common safe payload for one completed local evidence class."""

    return {
        "result": "passed",
        "claim_categories": {
            "integrity": "EVIDENCED",
            "recovery": "EVIDENCED",
            "progress": "EVIDENCED",
            "performance": "NOT_QUALIFIED",
        },
        "non_qualifying_classes": [
            candidate
            for candidate in phase8_evidence.EVIDENCE_CLASSES
            if candidate != evidence_class
        ],
        "subjects": list(phase8_evidence.QUALIFIED_SUBJECTS),
    }


def _write_local_envelope(
    *, output: Path, identity: SourceIdentity, evidence_class: str
) -> None:
    """Write one validated coverage-class envelope after its child succeeds."""

    phase8_evidence.write_envelope(
        output,
        phase8_evidence.make_envelope(
            evidence_class=evidence_class,
            status="PASS",
            revision=identity.revision,
            source_digest=identity.source_digest,
            payload=_class_payload(evidence_class),
        ),
    )


def run_coverage(*, output: Path) -> int:
    """Measure branch coverage, verify its ratchet, and emit one clean envelope."""

    before = current_source_identity()
    if before is None or not before.clean:
        return 1
    if _run_required_child(COVERAGE_MEASUREMENT_COMMAND, timeout=CHILD_TIMEOUT_SECONDS):
        return 1
    if _run_required_child(
        (
            sys.executable,
            str(COVERAGE_TOOL),
            "--report",
            str(COVERAGE_REPORT.relative_to(REPOSITORY_ROOT)),
            "--baseline",
            str(COVERAGE_BASELINE.relative_to(REPOSITORY_ROOT)),
            "--ruff",
        ),
        timeout=CHILD_TIMEOUT_SECONDS,
    ):
        return 1
    after = current_source_identity()
    if after is None or not after.clean or after != before:
        return 1
    _write_local_envelope(output=output, identity=before, evidence_class="coverage")
    return 0


def run_structural(*, output: Path) -> int:
    """Run formula/RSS self-tests and collect their raw structural envelope."""

    test_command = (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-o",
        "log_cli=false",
        "tests/performance/test_complexity_contracts.py",
        "tests/performance/test_memory_bounds.py",
        "-x",
    )
    if _run_required_child(test_command, timeout=CHILD_TIMEOUT_SECONDS):
        return 1
    return (
        _validate_class_envelope(output, "structural")
        if not _run_required_child(
            (
                sys.executable,
                str(STRUCTURAL_TOOL),
                "--collect",
                "--output",
                str(output),
            ),
            timeout=CHILD_TIMEOUT_SECONDS,
        )
        else 1
    )


def run_core(*, output_directory: Path) -> int:
    """Produce every non-package local evidence class in a fixed order."""

    output_directory.mkdir(parents=True, exist_ok=True)
    commands = (
        ("deterministic", lambda path: run_deterministic(output=path)),
        ("coverage", lambda path: run_coverage(output=path)),
        ("structural", lambda path: run_structural(output=path)),
    )
    for name, runner in commands:
        if runner(output_directory / f"{name}.json"):
            return 1
    return 0


def run_all(*, output_directory: Path) -> int:
    """Produce every local class, including packaging on a compatible Python."""

    if run_core(output_directory=output_directory):
        return 1
    return run_packaging(output=output_directory / "packaging.json")


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
    """Run one fixed local evidence producer without caller-selected selectors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate", choices=GATE_CHOICES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-os", choices=("Linux", "Darwin", "Windows"))
    parser.add_argument("--python-minor")
    parser.add_argument("--feature-profile", choices=("core",))
    parser.add_argument("--advisory", action="store_true")
    parsed = parser.parse_args(arguments)
    if parsed.gate in {"core", "all"}:
        if parsed.output is not None or parsed.output_dir is None:
            parser.error(f"{parsed.gate} requires --output-dir and rejects --output")
        if parsed.gate == "core":
            return run_core(output_directory=parsed.output_dir)
        return run_all(output_directory=parsed.output_dir)
    if parsed.output is None or parsed.output_dir is not None:
        parser.error("a single gate requires --output and rejects --output-dir")
    if parsed.gate == "packaging":
        return run_packaging(output=parsed.output)
    if parsed.gate == "platform":
        if (
            parsed.expected_os is None
            or parsed.python_minor is None
            or parsed.feature_profile is None
        ):
            parser.error(
                "platform requires expected OS, Python minor, and feature profile"
            )
        return run_platform(
            output=parsed.output,
            expected_os=parsed.expected_os,
            python_minor=parsed.python_minor,
            feature_profile=parsed.feature_profile,
            advisory=parsed.advisory,
        )
    if (
        parsed.expected_os
        or parsed.python_minor
        or parsed.feature_profile
        or parsed.advisory
    ):
        parser.error("platform options are valid only for the platform gate")
    if parsed.gate == "coverage":
        return run_coverage(output=parsed.output)
    if parsed.gate == "structural":
        return run_structural(output=parsed.output)
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
