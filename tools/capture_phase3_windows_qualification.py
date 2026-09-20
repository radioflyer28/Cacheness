"""Capture and attest the current host's unavailable Windows qualification.

This tool owns one non-overridable command.  It is deliberately an evidence
boundary rather than a Windows test runner: an ``UNAVAILABLE`` result records the
absence of native proof and can never become a support claim.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Callable, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION_ARGV = (
    "uv",
    "run",
    "--python",
    "3.11",
    "--frozen",
    "python",
    "verify_platform.py",
    "--phase3",
    "--require-system",
    "Windows",
    "--require-python",
    "3.11",
)
REPOSITORY_RUNTIME_COMMAND = ".venv/bin/python verify_platform.py --phase3"
QUALIFICATION_COMMAND = " ".join(QUALIFICATION_ARGV)
MAX_DOCUMENT_BYTES = 256 * 1024
MAX_STDOUT_BYTES = 32 * 1024


class AttestationError(RuntimeError):
    """Raised when qualification evidence is malformed or contradictory."""


Scalar = str | int | bool
SubprocessRunner = Callable[..., subprocess.CompletedProcess[bytes]]


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_bytes(path: Path, *, label: str) -> bytes:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise AttestationError(f"could not read {label}: {path}") from error
    if len(content) > MAX_DOCUMENT_BYTES:
        raise AttestationError(f"{label} exceeds the bounded document size")
    return content


def _frontmatter(path: Path, *, label: str) -> str:
    try:
        document = _read_bytes(path, label=label).decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise AttestationError(f"{label} is not valid UTF-8") from error
    if not document.startswith("---\n"):
        raise AttestationError(f"{label} lacks YAML frontmatter")
    closing = document.find("\n---\n", 4)
    if closing == -1:
        raise AttestationError(f"{label} has unterminated YAML frontmatter")
    return document[4:closing]


def _scalar(raw: str, *, label: str) -> Scalar:
    value = raw.strip()
    if not value:
        raise AttestationError(f"{label} has an empty scalar")
    if value.startswith('"'):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise AttestationError(f"{label} has an invalid quoted scalar") from error
        if not isinstance(parsed, str):
            raise AttestationError(f"{label} must be a string scalar")
        return parsed
    if value == "true":
        return True
    if value == "false":
        return False
    if re.fullmatch(r"0|[1-9][0-9]*", value):
        return int(value)
    if value.startswith(("[", "{", "- ")) or "\x00" in value:
        raise AttestationError(f"{label} is not a supported scalar")
    return value


def _mapping_fields(
    path: Path,
    mapping_name: str,
    expected_types: Mapping[str, type[Scalar]],
    *,
    label: str,
) -> dict[str, Scalar]:
    """Read only named direct scalar fields from one bounded frontmatter mapping."""
    lines = _frontmatter(path, label=label).splitlines()
    try:
        start = lines.index(f"{mapping_name}:") + 1
    except ValueError as error:
        raise AttestationError(f"{label} lacks {mapping_name} mapping") from error

    values: dict[str, Scalar] = {}
    for line in lines[start:]:
        if not line:
            continue
        if not line.startswith(" "):
            break
        if not line.startswith("  ") or line.startswith("   "):
            continue
        match = re.fullmatch(r"  ([a-z][a-z0-9_]*): (.+)", line)
        if match is None:
            continue
        key, raw_value = match.groups()
        if key not in expected_types:
            continue
        if key in values:
            raise AttestationError(f"{label} repeats {mapping_name}.{key}")
        values[key] = _scalar(raw_value, label=f"{label} {mapping_name}.{key}")

    missing = set(expected_types) - set(values)
    if missing:
        raise AttestationError(
            f"{label} lacks required {mapping_name} fields: {', '.join(sorted(missing))}"
        )
    for key, expected_type in expected_types.items():
        value = values[key]
        if type(value) is not expected_type:
            raise AttestationError(
                f"{label} {mapping_name}.{key} has unexpected scalar type"
            )
    return values


def _expect(value: object, expected: object, *, label: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise AttestationError(f"{label} is contradictory")


def _validate_runner(stdout: bytes, return_code: int) -> dict[str, Any]:
    if return_code != 2:
        raise AttestationError("fixed qualification command must exit 2")
    if not stdout or len(stdout) > MAX_STDOUT_BYTES:
        raise AttestationError("runner stdout is empty or exceeds the bounded size")
    try:
        decoded = stdout.decode("utf-8", "strict")
        parsed = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AttestationError("runner stdout is not exactly one UTF-8 JSON object") from error
    if not isinstance(parsed, dict):
        raise AttestationError("runner stdout must contain an object")

    expected_top_level = {
        "schema_version",
        "status",
        "host",
        "python",
        "sqlite",
        "filesystem",
        "topology",
        "requirements",
        "test_target",
        "offline_provisioning",
        "reason",
    }
    if set(parsed) != expected_top_level:
        raise AttestationError("runner object has an unexpected evidence shape")
    _expect(parsed["schema_version"], 1, label="runner schema version")
    _expect(parsed["status"], "UNAVAILABLE", label="runner status")
    requirements = parsed["requirements"]
    test_target = parsed["test_target"]
    if not isinstance(requirements, dict) or not isinstance(test_target, dict):
        raise AttestationError("runner object has malformed nested evidence")
    _expect(requirements.get("system"), "Windows", label="runner required system")
    _expect(requirements.get("python"), "3.11", label="runner required Python")
    _expect(
        test_target.get("native_windows"),
        "UNAVAILABLE",
        label="runner native evidence",
    )
    return parsed


def run_fixed_command(
    *, subprocess_runner: SubprocessRunner = subprocess.run
) -> subprocess.CompletedProcess[bytes]:
    """Run the one fixed qualification argv without a shell or caller override."""
    environment = os.environ.copy()
    # The helper itself may run from the repository's pinned .venv.  Leaving that
    # activation marker in place makes uv reject the required Python 3.11 selector
    # before it can invoke the fixed runner.
    environment.pop("VIRTUAL_ENV", None)
    return subprocess_runner(
        QUALIFICATION_ARGV,
        cwd=REPOSITORY_ROOT,
        shell=False,
        check=False,
        capture_output=True,
        text=False,
        env=environment,
    )


def _validate_input_contracts(
    *,
    runner_summary: Path,
    contract_artifact: Path,
    contract_summary: Path,
    validation: Path,
) -> dict[str, str]:
    runner = _mapping_fields(
        runner_summary,
        "platform-evidence",
        {
            "command": str,
            "non_windows_status": str,
            "unavailable_exit_code": int,
            "windows_qualified": bool,
            "backlog_phase": str,
        },
        label="Plan 03-09 summary",
    )
    contract = _mapping_fields(
        contract_summary,
        "platform-evidence-contract",
        {
            "contract_artifact_sha256": str,
            "source_summary_sha256": str,
            "repository_runtime_command": str,
            "qualification_attestation_command": str,
            "expected_status": str,
            "expected_exit_code": int,
            "milestone_status": str,
            "native_evidence": bool,
            "backlog_phase": str,
            "future_required_status": str,
            "future_required_exit_code": int,
        },
        label="Plan 03-12 summary",
    )
    addendum_repository = _mapping_fields(
        contract_artifact,
        "repository-runtime-evidence",
        {"command": str},
        label="platform evidence addendum",
    )
    addendum_qualification = _mapping_fields(
        contract_artifact,
        "qualification-attestation",
        {
            "command": str,
            "non_overridable": bool,
            "expected_status": str,
            "expected_exit_code": int,
            "milestone_status": str,
            "native_evidence": bool,
            "backlog_phase": str,
            "future_required_status": str,
            "future_required_exit_code": int,
        },
        label="platform evidence addendum",
    )
    source_binding = _mapping_fields(
        contract_artifact,
        "source-binding",
        {"source_summary_sha256": str},
        label="platform evidence addendum",
    )
    validation_contract = _mapping_fields(
        validation,
        "windows-qualification",
        {
            "current_host_status": str,
            "milestone_status": str,
            "native_evidence": bool,
            "windows_qualified": bool,
            "backlog_phase": str,
            "future_required_status": str,
            "future_required_exit_code": int,
        },
        label="Phase 3 validation",
    )

    _expect(runner["command"], REPOSITORY_RUNTIME_COMMAND, label="03-09 command")
    _expect(runner["non_windows_status"], "UNAVAILABLE", label="03-09 status")
    _expect(runner["unavailable_exit_code"], 2, label="03-09 exit code")
    _expect(runner["windows_qualified"], False, label="03-09 Windows qualification")
    _expect(runner["backlog_phase"], "999.1", label="03-09 backlog")

    _expect(
        contract["repository_runtime_command"], runner["command"], label="command roles"
    )
    _expect(
        addendum_repository["command"], runner["command"], label="addendum runtime command"
    )
    _expect(
        contract["qualification_attestation_command"],
        QUALIFICATION_COMMAND,
        label="qualification command",
    )
    _expect(
        addendum_qualification["command"],
        QUALIFICATION_COMMAND,
        label="addendum qualification command",
    )
    _expect(addendum_qualification["non_overridable"], True, label="command override")

    for key, expected in {
        "expected_status": "UNAVAILABLE",
        "expected_exit_code": 2,
        "milestone_status": "NOT_QUALIFIED",
        "native_evidence": False,
        "backlog_phase": "999.1",
        "future_required_status": "PASS",
        "future_required_exit_code": 0,
    }.items():
        _expect(contract[key], expected, label=f"03-12 {key}")
        _expect(addendum_qualification[key], expected, label=f"addendum {key}")

    for key, expected in {
        "current_host_status": "UNAVAILABLE",
        "milestone_status": "NOT_QUALIFIED",
        "native_evidence": False,
        "windows_qualified": False,
        "backlog_phase": "999.1",
        "future_required_status": "PASS",
        "future_required_exit_code": 0,
    }.items():
        _expect(validation_contract[key], expected, label=f"validation {key}")

    runner_digest = _sha256_bytes(_read_bytes(runner_summary, label="Plan 03-09 summary"))
    addendum_digest = _sha256_bytes(
        _read_bytes(contract_artifact, label="platform evidence addendum")
    )
    contract_digest = _sha256_bytes(
        _read_bytes(contract_summary, label="Plan 03-12 summary")
    )
    validation_digest = _sha256_bytes(_read_bytes(validation, label="Phase 3 validation"))
    _expect(contract["source_summary_sha256"], runner_digest, label="03-12 source digest")
    _expect(source_binding["source_summary_sha256"], runner_digest, label="addendum source digest")
    _expect(
        contract["contract_artifact_sha256"], addendum_digest, label="03-12 addendum digest"
    )
    return {
        "runner_summary_sha256": runner_digest,
        "contract_artifact_sha256": addendum_digest,
        "contract_summary_sha256": contract_digest,
        "validation_sha256": validation_digest,
    }


def _canonical_runner_bytes(runner: Mapping[str, Any]) -> bytes:
    return json.dumps(runner, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _build_artifact(
    *, stdout: bytes, runner: Mapping[str, Any], bindings: Mapping[str, str]
) -> bytes:
    canonical_runner = _canonical_runner_bytes(runner)
    qualification: dict[str, Scalar] = {
        "command": QUALIFICATION_COMMAND,
        "return_code": 2,
        "runner_status": "UNAVAILABLE",
        "stdout_base64": base64.b64encode(stdout).decode("ascii"),
        "stdout_sha256": _sha256_bytes(stdout),
        "runner_json_base64": base64.b64encode(canonical_runner).decode("ascii"),
        "runner_json_sha256": _sha256_bytes(canonical_runner),
        "milestone_status": "NOT_QUALIFIED",
        "native_evidence": False,
        "backlog_phase": "999.1",
        "future_required_status": "PASS",
        "future_required_exit_code": 0,
    }
    lines = ["---", "schema_version: 1", "qualification:"]
    lines.extend(
        f"  {key}: {json.dumps(value) if isinstance(value, str) else str(value).lower()}"
        for key, value in qualification.items()
    )
    lines.append("bindings:")
    lines.extend(f"  {key}: {value}" for key, value in bindings.items())
    lines.extend(
        [
            "---",
            "",
            "# Phase 3 Windows Qualification Evidence",
            "",
            "This record is a current-host UNAVAILABLE attestation. It is not native Windows support.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _fsync_parent(parent: Path) -> None:
    descriptor = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_publish(destination: Path, content: bytes) -> None:
    if not destination.parent.is_dir():
        raise AttestationError("qualification artifact parent does not exist")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_parent(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_qualification_artifact(path: Path) -> dict[str, dict[str, Scalar]]:
    """Parse and internally authenticate a qualification artifact."""
    qualification = _mapping_fields(
        path,
        "qualification",
        {
            "command": str,
            "return_code": int,
            "runner_status": str,
            "stdout_base64": str,
            "stdout_sha256": str,
            "runner_json_base64": str,
            "runner_json_sha256": str,
            "milestone_status": str,
            "native_evidence": bool,
            "backlog_phase": str,
            "future_required_status": str,
            "future_required_exit_code": int,
        },
        label="qualification artifact",
    )
    bindings = _mapping_fields(
        path,
        "bindings",
        {
            "runner_summary_sha256": str,
            "contract_artifact_sha256": str,
            "contract_summary_sha256": str,
            "validation_sha256": str,
        },
        label="qualification artifact",
    )
    for key, expected in {
        "command": QUALIFICATION_COMMAND,
        "return_code": 2,
        "runner_status": "UNAVAILABLE",
        "milestone_status": "NOT_QUALIFIED",
        "native_evidence": False,
        "backlog_phase": "999.1",
        "future_required_status": "PASS",
        "future_required_exit_code": 0,
    }.items():
        _expect(qualification[key], expected, label=f"artifact {key}")
    try:
        stdout = base64.b64decode(qualification["stdout_base64"], validate=True)
        canonical_runner = base64.b64decode(
            qualification["runner_json_base64"], validate=True
        )
    except (ValueError, TypeError) as error:
        raise AttestationError("artifact has invalid base64 evidence") from error
    _expect(
        _sha256_bytes(stdout), qualification["stdout_sha256"], label="artifact stdout digest"
    )
    _expect(
        _sha256_bytes(canonical_runner),
        qualification["runner_json_sha256"],
        label="artifact parsed runner digest",
    )
    runner = _validate_runner(stdout, 2)
    if canonical_runner != _canonical_runner_bytes(runner):
        raise AttestationError("artifact parsed runner bytes do not match stdout")
    return {"qualification": qualification, "bindings": bindings}


def _verify_optional_summary(
    *, artifact: Path, contract_summary: Path, qualification_summary: Path
) -> None:
    summary = _mapping_fields(
        qualification_summary,
        "windows-qualification",
        {
            "artifact_path": str,
            "artifact_sha256": str,
            "contract_summary_path": str,
            "contract_summary_sha256": str,
            "qualification_attestation_command": str,
            "runner_status": str,
            "runner_exit_code": int,
            "milestone_status": str,
            "native_evidence": bool,
            "backlog_phase": str,
            "future_required_status": str,
            "future_required_exit_code": int,
        },
        label="Plan 03-11 summary",
    )
    expected_artifact_path = artifact.resolve()
    expected_contract_path = contract_summary.resolve()
    artifact_path = Path(summary["artifact_path"])
    contract_path = Path(summary["contract_summary_path"])
    if not artifact_path.is_absolute():
        artifact_path = (REPOSITORY_ROOT / artifact_path).resolve()
    if not contract_path.is_absolute():
        contract_path = (REPOSITORY_ROOT / contract_path).resolve()
    if artifact_path != expected_artifact_path or contract_path != expected_contract_path:
        raise AttestationError("Plan 03-11 summary references a contradictory artifact")
    expected = {
        "artifact_sha256": _sha256_bytes(_read_bytes(artifact, label="qualification artifact")),
        "contract_summary_sha256": _sha256_bytes(
            _read_bytes(contract_summary, label="Plan 03-12 summary")
        ),
        "qualification_attestation_command": QUALIFICATION_COMMAND,
        "runner_status": "UNAVAILABLE",
        "runner_exit_code": 2,
        "milestone_status": "NOT_QUALIFIED",
        "native_evidence": False,
        "backlog_phase": "999.1",
        "future_required_status": "PASS",
        "future_required_exit_code": 0,
    }
    for key, value in expected.items():
        _expect(summary[key], value, label=f"Plan 03-11 summary {key}")


def capture(
    *,
    artifact: Path,
    runner_summary: Path,
    contract_artifact: Path,
    contract_summary: Path,
    validation: Path,
    subprocess_runner: SubprocessRunner = subprocess.run,
) -> None:
    """Capture fixed-command evidence and atomically publish a self-checking record."""
    bindings = _validate_input_contracts(
        runner_summary=runner_summary,
        contract_artifact=contract_artifact,
        contract_summary=contract_summary,
        validation=validation,
    )
    completed = run_fixed_command(subprocess_runner=subprocess_runner)
    runner = _validate_runner(completed.stdout, completed.returncode)
    content = _build_artifact(stdout=completed.stdout, runner=runner, bindings=bindings)
    _atomic_publish(artifact, content)
    published = read_qualification_artifact(artifact)
    if published["qualification"]["stdout_base64"] != base64.b64encode(
        completed.stdout
    ).decode("ascii"):
        raise AttestationError("post-publication stdout read-back differs")


def verify(
    *,
    artifact: Path,
    runner_summary: Path,
    contract_artifact: Path,
    contract_summary: Path,
    validation: Path,
    qualification_summary: Path | None = None,
    subprocess_runner: SubprocessRunner = subprocess.run,
) -> None:
    """Freshly rerun and structurally attest a published unavailable record."""
    bindings = _validate_input_contracts(
        runner_summary=runner_summary,
        contract_artifact=contract_artifact,
        contract_summary=contract_summary,
        validation=validation,
    )
    record = read_qualification_artifact(artifact)
    if record["bindings"] != bindings:
        raise AttestationError("artifact bindings are stale or contradictory")
    completed = run_fixed_command(subprocess_runner=subprocess_runner)
    fresh_runner = _validate_runner(completed.stdout, completed.returncode)
    stored_stdout = base64.b64decode(
        record["qualification"]["stdout_base64"], validate=True
    )
    if completed.stdout != stored_stdout:
        raise AttestationError("artifact stdout is stale or differs from fresh evidence")
    if _canonical_runner_bytes(fresh_runner) != base64.b64decode(
        record["qualification"]["runner_json_base64"], validate=True
    ):
        raise AttestationError("artifact parsed runner record differs from fresh evidence")
    if qualification_summary is not None:
        _verify_optional_summary(
            artifact=artifact,
            contract_summary=contract_summary,
            qualification_summary=qualification_summary,
        )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="mode", required=True)
    for mode in ("capture", "verify"):
        command = subcommands.add_parser(mode)
        command.add_argument("--artifact", type=Path, required=True)
        command.add_argument("--runner-summary", type=Path, required=True)
        command.add_argument("--contract-artifact", type=Path, required=True)
        command.add_argument("--contract-summary", type=Path, required=True)
        command.add_argument("--validation", type=Path, required=True)
        if mode == "verify":
            command.add_argument("--qualification-summary", type=Path)
    return parser.parse_args()


def main() -> int:
    """Run capture or verify and report only a bounded public failure message."""
    arguments = _arguments()
    kwargs = {
        "artifact": arguments.artifact,
        "runner_summary": arguments.runner_summary,
        "contract_artifact": arguments.contract_artifact,
        "contract_summary": arguments.contract_summary,
        "validation": arguments.validation,
    }
    try:
        if arguments.mode == "capture":
            capture(**kwargs)
        else:
            verify(**kwargs, qualification_summary=arguments.qualification_summary)
    except AttestationError as error:
        print(f"qualification attestation failed: {error}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
