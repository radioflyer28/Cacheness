#!/usr/bin/env python3
"""Produce fail-closed platform evidence for the Phase 8 support matrix.

One invocation records one executing interpreter and operating-system row.  It
does not let a current host impersonate another row, and it deliberately keeps
matrix aggregation separate from an individual row's command result.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import platform
import subprocess
import sys
from typing import Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPOSITORY_ROOT / "tools" / "phase8_evidence.py"
LOCAL_GATE_PATH = REPOSITORY_ROOT / "tools" / "run_phase8_local_gates.py"
STABLE_PYTHON_MINORS = ("3.11", "3.12", "3.13", "3.14")
ADVISORY_PYTHON_MINORS = ("3.15",)
TENSORFLOW_COMPATIBLE_MINORS = ("3.11", "3.12", "3.13")
MACOS_BOUNDARY_MINORS = ("3.11", "3.14")
WINDOWS_BACKLOG_PHASE = "999.1"
ADR_PROGRESS_OUTCOMES = ("success", "conflict", "typed_retryable")
FEATURE_PROFILES = frozenset({"core", "non_tensorflow", "tensorflow"})
ROW_STATUSES = frozenset({"PASS", "FAIL", "SKIPPED", "UNAVAILABLE"})
_ROW_KEYS = frozenset(
    {
        "expected_os",
        "actual_os",
        "python_minor",
        "actual_python_minor",
        "feature_profile",
        "platform_role",
        "advisory",
        "command_profile",
        "command_status",
    }
)
_PLATFORM_ROLE_BY_OS = {
    "Linux": "linux_full",
    "Darwin": "macos_boundary",
    "Windows": "windows_non_native",
}
MACOS_BOUNDARY_SMOKE_SOURCE = """
from pathlib import Path
from tempfile import TemporaryDirectory

from cacheness.storage import BackendRef, BlobStore, StoreTopology


def round_trip(topology, root):
    store = BlobStore(topology, cache_dir=root)
    store.initialize()
    try:
        store.put({"boundary": "smoke"}, key="boundary")
        assert store.get("boundary") == {"boundary": "smoke"}
    finally:
        store.close()


with TemporaryDirectory() as temporary:
    root = Path(temporary)
    round_trip(
        StoreTopology(
            payload=BackendRef(name="memory"), authority=BackendRef(name="memory")
        ),
        root / "memory",
    )
    round_trip(
        StoreTopology(
            payload=BackendRef(name="filesystem", options={"base_dir": root / "payload"}),
            authority=BackendRef(name="sqlite", options={"root": root / "authority"}),
        ),
        root / "filesystem",
    )
"""


def _load_evidence_module():
    """Load the sibling evidence contract when invoked outside package import."""
    specification = importlib.util.spec_from_file_location("phase8_evidence", EVIDENCE_PATH)
    if specification is None or specification.loader is None:
        raise RuntimeError("Phase 8 evidence utility is unavailable")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


phase8_evidence = _load_evidence_module()


def current_python_minor() -> str:
    """Return the current interpreter's major.minor identity."""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def is_feature_profile_compatible(feature_profile: str, python_minor: str) -> bool:
    """Return whether the published profile is eligible for this stable minor."""
    if feature_profile not in FEATURE_PROFILES:
        return False
    if feature_profile == "tensorflow":
        return python_minor in TENSORFLOW_COMPATIBLE_MINORS
    return python_minor in STABLE_PYTHON_MINORS


def _role_for_os(expected_os: str) -> str:
    try:
        return _PLATFORM_ROLE_BY_OS[expected_os]
    except KeyError as error:
        raise ValueError("unsupported expected operating system") from error


def build_row(
    *,
    expected_os: str,
    python_minor: str,
    feature_profile: str,
    command_status: str,
    advisory: bool = False,
    actual_os: str | None = None,
    actual_python_minor: str | None = None,
    platform_role: str | None = None,
    command_profile: str | None = None,
) -> dict[str, object]:
    """Build one row for a runner or contract test before aggregation."""
    if feature_profile not in FEATURE_PROFILES:
        raise ValueError("unsupported feature profile")
    if command_status not in ROW_STATUSES:
        raise ValueError("invalid command status")
    role = platform_role or _role_for_os(expected_os)
    return {
        "expected_os": expected_os,
        "actual_os": actual_os or expected_os,
        "python_minor": python_minor,
        "actual_python_minor": actual_python_minor or python_minor,
        "feature_profile": feature_profile,
        "platform_role": role,
        "advisory": advisory,
        "command_profile": command_profile or role,
        "command_status": command_status,
    }


def validate_row(row: Mapping[str, object]) -> dict[str, object]:
    """Return an exact row or reject substitutions before aggregation."""
    if set(row) != _ROW_KEYS:
        raise ValueError("platform row has an invalid shape")
    validated = dict(row)
    for field in (
        "expected_os",
        "actual_os",
        "python_minor",
        "actual_python_minor",
        "feature_profile",
        "platform_role",
        "command_profile",
        "command_status",
    ):
        if not isinstance(validated[field], str) or not validated[field]:
            raise ValueError(f"platform row has invalid {field}")
    if validated["expected_os"] not in _PLATFORM_ROLE_BY_OS:
        raise ValueError("platform row has an unsupported operating system")
    if validated["platform_role"] != _role_for_os(str(validated["expected_os"])):
        raise ValueError("platform row has a contradictory role")
    if validated["feature_profile"] not in FEATURE_PROFILES:
        raise ValueError("platform row has an unsupported feature profile")
    if validated["command_status"] not in ROW_STATUSES:
        raise ValueError("platform row has an invalid command status")
    if not isinstance(validated["advisory"], bool):
        raise ValueError("platform row has invalid advisory status")
    if (
        not validated["advisory"]
        and validated["command_status"] == "PASS"
        and (
            validated["expected_os"] != validated["actual_os"]
            or validated["python_minor"] != validated["actual_python_minor"]
        )
    ):
        raise ValueError("platform row attempts to qualify a mismatched runtime")
    return validated


def _required_linux_slots(feature_profile: str) -> tuple[tuple[str, str, str], ...]:
    if feature_profile == "tensorflow":
        minors = TENSORFLOW_COMPATIBLE_MINORS
    elif feature_profile in {"core", "non_tensorflow"}:
        minors = STABLE_PYTHON_MINORS
    else:
        raise ValueError("unsupported feature profile")
    return tuple(("Linux", minor, feature_profile) for minor in minors)


def _aggregate_required_rows(
    rows: Sequence[Mapping[str, object]], *, feature_profile: str
) -> dict[str, object]:
    required_slots = _required_linux_slots(feature_profile)
    required_set = set(required_slots)
    seen: dict[tuple[str, str, str], dict[str, object]] = {}
    advisory_rows = 0

    for candidate in rows:
        row = validate_row(candidate)
        slot = (
            str(row["expected_os"]),
            str(row["python_minor"]),
            str(row["feature_profile"]),
        )
        if row["advisory"]:
            if row["python_minor"] in STABLE_PYTHON_MINORS:
                raise ValueError("stable row cannot be advisory")
            advisory_rows += 1
            continue
        if row["feature_profile"] == "tensorflow" and (
            row["python_minor"] not in TENSORFLOW_COMPATIBLE_MINORS
        ):
            raise ValueError("TensorFlow row is not compatible with this Python minor")
        if slot not in required_set:
            raise ValueError("row is outside the published qualification matrix")
        if slot in seen:
            raise ValueError("duplicate required qualification row")
        seen[slot] = row

    missing = required_set.difference(seen)
    if missing:
        raise ValueError("missing required qualification rows")
    for slot in required_slots:
        if seen[slot]["command_status"] != "PASS":
            raise ValueError("required qualification row did not pass")
    return {
        "status": "QUALIFIED",
        "qualified_slots": ["/".join(slot) for slot in required_slots],
        "advisory_rows": advisory_rows,
    }


def aggregate_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Require an exact Linux core matrix; advisory rows are nonqualifying."""
    return _aggregate_required_rows(rows, feature_profile="core")


def aggregate_feature_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Require the documented TensorFlow-compatible Linux subset only."""
    return _aggregate_required_rows(rows, feature_profile="tensorflow")


def validate_adr_progress_outcome(outcome: str) -> str:
    """Accept only the ADR's success/conflict/retryable progress vocabulary."""
    if outcome not in ADR_PROGRESS_OUTCOMES:
        raise ValueError("invalid ADR progress outcome")
    return outcome


def aggregate_platform_roles(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate full Linux, boundary macOS, and non-native Windows evidence."""
    validated = [validate_row(row) for row in rows]
    linux_rows = [
        row
        for row in validated
        if row["expected_os"] == "Linux" and not row["advisory"]
    ]
    linux = aggregate_rows(linux_rows)

    macos_slots = {
        ("Darwin", minor, "core") for minor in MACOS_BOUNDARY_MINORS
    }
    macos_seen: dict[tuple[str, str, str], dict[str, object]] = {}
    windows_rows: list[dict[str, object]] = []
    for row in validated:
        if row["advisory"] or row["expected_os"] == "Linux":
            continue
        slot = (
            str(row["expected_os"]),
            str(row["python_minor"]),
            str(row["feature_profile"]),
        )
        if row["expected_os"] == "Darwin":
            if slot not in macos_slots:
                raise ValueError("macOS row is outside the boundary smoke matrix")
            if row["actual_os"] != "Darwin" or row["actual_python_minor"] != row["python_minor"]:
                raise ValueError("macOS boundary row has mismatched runtime identity")
            if slot in macos_seen:
                raise ValueError("duplicate macOS boundary row")
            if row["command_status"] != "PASS":
                raise ValueError("macOS boundary row did not pass")
            macos_seen[slot] = row
        elif row["expected_os"] == "Windows":
            windows_rows.append(row)
        else:
            raise ValueError("platform row is outside the published roles")

    missing_macos = macos_slots.difference(macos_seen)
    if missing_macos:
        raise ValueError("missing required macOS boundary rows")
    if len(windows_rows) != 1:
        raise ValueError("exactly one Windows nonclaim row is required")
    windows = windows_rows[0]
    if (
        windows["feature_profile"] != "core"
        or windows["python_minor"] != "3.11"
        or windows["command_status"] != "UNAVAILABLE"
    ):
        raise ValueError("Windows row contradicts the Phase 8 native-evidence nonclaim")

    return {
        "status": "NOT_QUALIFIED",
        "linux_status": linux["status"],
        "macos_boundary_slots": ["/".join(slot) for slot in sorted(macos_slots)],
        "windows_status": "UNAVAILABLE",
        "windows_backlog_phase": WINDOWS_BACKLOG_PHASE,
        "progress_outcomes": list(ADR_PROGRESS_OUTCOMES),
    }


def _git_revision() -> str:
    try:
        completed = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("repository revision is unavailable") from error
    revision = completed.stdout.strip()
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise RuntimeError("repository revision is invalid")
    return revision


def _source_digest() -> str:
    return phase8_evidence.relevant_source_digest(
        REPOSITORY_ROOT,
        (
            "tools/phase8_evidence.py",
            "tools/run_phase8_local_gates.py",
            "tools/run_phase8_platform_gates.py",
        ),
    )


def _source_is_clean() -> bool:
    try:
        completed = subprocess.run(
            (
                "git",
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                "tools/phase8_evidence.py",
                "tools/run_phase8_local_gates.py",
                "tools/run_phase8_platform_gates.py",
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


def _command_for_linux() -> tuple[str, ...]:
    return (sys.executable, str(LOCAL_GATE_PATH), "--all")


def macos_boundary_smoke_command() -> tuple[str, ...]:
    """Return the fixed public topology smoke command for macOS boundaries."""
    return (sys.executable, "-c", MACOS_BOUNDARY_SMOKE_SOURCE)


def _run_fixed_command(command: Sequence[str], timeout: int) -> str:
    try:
        completed = subprocess.run(
            tuple(command),
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "timed_out"
    except OSError:
        return "unavailable"
    if completed.returncode == 0:
        return "passed"
    if completed.returncode == 2:
        return "unavailable"
    return "failed"


def _evidence_payload(
    *,
    status: str,
    result: str,
    expected_os: str,
    expected_python_minor: str,
    feature_profile: str,
    role: str,
    advisory: bool,
    reason: str,
) -> dict[str, object]:
    claims = (
        {
            "integrity": "EVIDENCED",
            "recovery": "EVIDENCED",
            "progress": "EVIDENCED",
            "performance": "NOT_QUALIFIED",
        }
        if status == "PASS"
        else dict.fromkeys(phase8_evidence.CLAIM_CATEGORIES, status)
    )
    return {
        "result": result,
        "claim_categories": claims,
        "non_qualifying_classes": [
            evidence_class
            for evidence_class in phase8_evidence.EVIDENCE_CLASSES
            if evidence_class != "platform"
        ],
        "subjects": list(phase8_evidence.QUALIFIED_SUBJECTS),
        "expected_os": expected_os,
        "actual_os": platform.system(),
        "expected_python_minor": expected_python_minor,
        "actual_python_minor": current_python_minor(),
        "feature_profile": feature_profile,
        "role": role,
        "advisory": advisory,
        "command_profile": role,
        "backlog_phase": (
            WINDOWS_BACKLOG_PHASE if role == "windows_non_native" else "not_applicable"
        ),
        "reason": reason,
    }


def _write_evidence(
    *,
    output: Path,
    status: str,
    result: str,
    expected_os: str,
    expected_python_minor: str,
    feature_profile: str,
    advisory: bool,
    reason: str,
) -> None:
    role = _role_for_os(expected_os)
    envelope = phase8_evidence.make_envelope(
        evidence_class="platform",
        status=status,
        revision=_git_revision(),
        source_digest=_source_digest(),
        payload=_evidence_payload(
            status=status,
            result=result,
            expected_os=expected_os,
            expected_python_minor=expected_python_minor,
            feature_profile=feature_profile,
            role=role,
            advisory=advisory,
            reason=reason,
        ),
    )
    phase8_evidence.write_envelope(output, envelope)


def run_platform_gate(
    *,
    expected_os: str,
    python_minor: str,
    feature_profile: str,
    output: Path,
    advisory: bool = False,
    timeout: int = 900,
) -> int:
    """Run one fixed row, recording only its actual runtime identity."""
    role = _role_for_os(expected_os)
    if feature_profile not in FEATURE_PROFILES:
        raise ValueError("unsupported feature profile")
    actual_os = platform.system()
    actual_minor = current_python_minor()
    if role == "windows_non_native":
        _write_evidence(
            output=output,
            status="UNAVAILABLE",
            result="unavailable",
            expected_os=expected_os,
            expected_python_minor=python_minor,
            feature_profile=feature_profile,
            advisory=advisory,
            reason="native_windows_phase_999_1_required",
        )
        return 2
    if actual_os != expected_os or actual_minor != python_minor:
        _write_evidence(
            output=output,
            status="UNAVAILABLE",
            result="unavailable",
            expected_os=expected_os,
            expected_python_minor=python_minor,
            feature_profile=feature_profile,
            advisory=advisory,
            reason="runtime_identity_mismatch",
        )
        return 2
    if not is_feature_profile_compatible(feature_profile, python_minor):
        _write_evidence(
            output=output,
            status="UNAVAILABLE",
            result="unavailable",
            expected_os=expected_os,
            expected_python_minor=python_minor,
            feature_profile=feature_profile,
            advisory=advisory,
            reason="feature_profile_incompatible",
        )
        return 2
    if not _source_is_clean():
        _write_evidence(
            output=output,
            status="NOT_QUALIFIED",
            result="source_dirty",
            expected_os=expected_os,
            expected_python_minor=python_minor,
            feature_profile=feature_profile,
            advisory=advisory,
            reason="reviewed_sources_dirty",
        )
        return 1
    if role == "macos_boundary":
        if python_minor not in MACOS_BOUNDARY_MINORS:
            _write_evidence(
                output=output,
                status="UNAVAILABLE",
                result="unavailable",
                expected_os=expected_os,
                expected_python_minor=python_minor,
                feature_profile=feature_profile,
                advisory=advisory,
                reason="macos_minor_outside_boundary_smoke",
            )
            return 2
        result = _run_fixed_command(macos_boundary_smoke_command(), timeout)
        status = {
            "passed": "PASS",
            "unavailable": "UNAVAILABLE",
        }.get(result, "NOT_QUALIFIED")
        _write_evidence(
            output=output,
            status=status,
            result=result,
            expected_os=expected_os,
            expected_python_minor=python_minor,
            feature_profile=feature_profile,
            advisory=advisory,
            reason="fixed_macos_boundary_smoke_completed",
        )
        return {"PASS": 0, "NOT_QUALIFIED": 1, "UNAVAILABLE": 2}[status]
    if role != "linux_full":
        _write_evidence(
            output=output,
            status="UNAVAILABLE",
            result="unavailable",
            expected_os=expected_os,
            expected_python_minor=python_minor,
            feature_profile=feature_profile,
            advisory=advisory,
            reason="platform_role_not_implemented",
        )
        return 2

    result = _run_fixed_command(_command_for_linux(), timeout)
    status = {
        "passed": "PASS",
        "unavailable": "UNAVAILABLE",
    }.get(result, "NOT_QUALIFIED")
    _write_evidence(
        output=output,
        status=status,
        result=result,
        expected_os=expected_os,
        expected_python_minor=python_minor,
        feature_profile=feature_profile,
        advisory=advisory,
        reason="fixed_linux_gate_completed",
    )
    return {"PASS": 0, "NOT_QUALIFIED": 1, "UNAVAILABLE": 2}[status]


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the bounded one-row command-line contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-os", required=True, choices=sorted(_PLATFORM_ROLE_BY_OS))
    parser.add_argument("--python-minor", required=True)
    parser.add_argument("--feature-profile", required=True, choices=sorted(FEATURE_PROFILES))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--advisory", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run one platform row without a shell or caller-selected test command."""
    namespace = parse_arguments(arguments)
    if namespace.timeout <= 0:
        raise ValueError("timeout must be positive")
    return run_platform_gate(
        expected_os=namespace.expected_os,
        python_minor=namespace.python_minor,
        feature_profile=namespace.feature_profile,
        output=namespace.output,
        advisory=namespace.advisory,
        timeout=namespace.timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
