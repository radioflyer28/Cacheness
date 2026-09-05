"""Adversarial coverage for the Phase 3 unavailable Windows attestation."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = REPOSITORY_ROOT / "tools" / "capture_phase3_windows_qualification.py"


def _load_helper():
    specification = importlib.util.spec_from_file_location(
        "phase3_windows_qualification", HELPER_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load Phase 3 Windows qualification helper")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _runner_bytes(*, reason: str = "required_system_unavailable") -> bytes:
    return (
        json.dumps(
            {
                "schema_version": 1,
                "status": "UNAVAILABLE",
                "host": {
                    "system": "Darwin",
                    "release": "redacted",
                    "architecture": "arm64",
                },
                "python": {"implementation": "CPython", "major_minor": "3.11"},
                "sqlite": {"journal_mode": "delete", "synchronous": "extra"},
                "filesystem": {
                    "class": "non-windows-local",
                    "root_provisioned": None,
                },
                "topology": {
                    "commit_authority": "sqlite",
                    "custom_win32_lock_authority": False,
                    "logon_sid_class": "S-1-5-5-X-Y",
                    "different_token": "NOT_RUN",
                },
                "requirements": {"system": "Windows", "python": "3.11"},
                "test_target": {
                    "focused_suite": "NOT_RUN",
                    "native_windows": "UNAVAILABLE",
                },
                "offline_provisioning": "offline-only",
                "reason": reason,
            },
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _completed_inputs(tmp_path: Path) -> dict[str, Path]:
    runner_summary = tmp_path / "03-09-SUMMARY.md"
    _write(
        runner_summary,
        """---
platform-evidence:
  command: \".venv/bin/python verify_platform.py --phase3\"
  non_windows_status: UNAVAILABLE
  unavailable_exit_code: 2
  windows_qualified: false
  backlog_phase: 999.1
---
""",
    )
    contract_artifact = tmp_path / "03-09-PLATFORM-EVIDENCE-ADDENDUM.md"
    _write(
        contract_artifact,
        f"""---
repository-runtime-evidence:
  command: \".venv/bin/python verify_platform.py --phase3\"
qualification-attestation:
  command: \"uv run --python 3.11 --frozen python verify_platform.py --phase3 --require-system Windows --require-python 3.11\"
  non_overridable: true
  expected_status: UNAVAILABLE
  expected_exit_code: 2
  milestone_status: NOT_QUALIFIED
  native_evidence: false
  backlog_phase: 999.1
  future_required_status: PASS
  future_required_exit_code: 0
source-binding:
  source_summary_sha256: {_digest(runner_summary)}
---
""",
    )
    contract_summary = tmp_path / "03-12-SUMMARY.md"
    _write(
        contract_summary,
        f"""---
platform-evidence-contract:
  contract_artifact_sha256: {_digest(contract_artifact)}
  source_summary_sha256: {_digest(runner_summary)}
  repository_runtime_command: \".venv/bin/python verify_platform.py --phase3\"
  qualification_attestation_command: \"uv run --python 3.11 --frozen python verify_platform.py --phase3 --require-system Windows --require-python 3.11\"
  expected_status: UNAVAILABLE
  expected_exit_code: 2
  milestone_status: NOT_QUALIFIED
  native_evidence: false
  backlog_phase: 999.1
  future_required_status: PASS
  future_required_exit_code: 0
---
""",
    )
    validation = tmp_path / "03-VALIDATION.md"
    _write(
        validation,
        """---
windows-qualification:
  current_host_status: UNAVAILABLE
  milestone_status: NOT_QUALIFIED
  native_evidence: false
  windows_qualified: false
  backlog_phase: 999.1
  future_required_status: PASS
  future_required_exit_code: 0
---
""",
    )
    return {
        "runner_summary": runner_summary,
        "contract_artifact": contract_artifact,
        "contract_summary": contract_summary,
        "validation": validation,
        "artifact": tmp_path / "qualification.md",
    }


def _fake_runner(stdout: bytes, returncode: int = 2):
    def run(argv, **kwargs):
        assert tuple(argv) == _load_helper().QUALIFICATION_ARGV
        assert kwargs["shell"] is False
        return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr=b"secret")

    return run


def _capture(helper, paths: dict[str, Path], stdout: bytes | None = None) -> None:
    helper.capture(
        artifact=paths["artifact"],
        runner_summary=paths["runner_summary"],
        contract_artifact=paths["contract_artifact"],
        contract_summary=paths["contract_summary"],
        validation=paths["validation"],
        subprocess_runner=_fake_runner(stdout or _runner_bytes()),
    )


def _verify(helper, paths: dict[str, Path], stdout: bytes | None = None, **kwargs) -> None:
    helper.verify(
        artifact=paths["artifact"],
        runner_summary=paths["runner_summary"],
        contract_artifact=paths["contract_artifact"],
        contract_summary=paths["contract_summary"],
        validation=paths["validation"],
        subprocess_runner=_fake_runner(stdout or _runner_bytes()),
        **kwargs,
    )


def test_capture_and_verify_bind_fixed_command_and_exact_bytes(tmp_path: Path) -> None:
    helper = _load_helper()
    paths = _completed_inputs(tmp_path)

    _capture(helper, paths)
    _verify(helper, paths)

    record = helper.read_qualification_artifact(paths["artifact"])
    assert record["qualification"]["command"] == " ".join(helper.QUALIFICATION_ARGV)
    assert record["qualification"]["return_code"] == 2
    assert record["qualification"]["runner_status"] == "UNAVAILABLE"
    assert record["qualification"]["stdout_base64"] == base64.b64encode(
        _runner_bytes()
    ).decode("ascii")
    assert "secret" not in paths["artifact"].read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("stdout", "returncode"),
    [
        (b"not json", 2),
        (b"{}{}", 2),
        (b"[1]", 2),
        (b"\xff", 2),
        (json.dumps({"status": "PASS"}).encode("utf-8"), 2),
        (_runner_bytes(), 0),
    ],
)
def test_capture_fails_closed_for_bad_runner_evidence(
    tmp_path: Path, stdout: bytes, returncode: int
) -> None:
    helper = _load_helper()
    paths = _completed_inputs(tmp_path)

    with pytest.raises(helper.AttestationError):
        helper.capture(
            artifact=paths["artifact"],
            runner_summary=paths["runner_summary"],
            contract_artifact=paths["contract_artifact"],
            contract_summary=paths["contract_summary"],
            validation=paths["validation"],
            subprocess_runner=_fake_runner(stdout, returncode),
        )
    assert not paths["artifact"].exists()


def test_verify_rejects_tampering_and_stale_stdout(tmp_path: Path) -> None:
    helper = _load_helper()
    paths = _completed_inputs(tmp_path)
    _capture(helper, paths)
    original = paths["artifact"].read_text(encoding="utf-8")

    paths["artifact"].write_text(
        original.replace("stdout_sha256:", "stdout_sha256: broken #"), encoding="utf-8"
    )
    with pytest.raises(helper.AttestationError):
        _verify(helper, paths)

    _capture(helper, paths)
    with pytest.raises(helper.AttestationError):
        _verify(helper, paths, _runner_bytes(reason="a-different-current-output"))


@pytest.mark.parametrize(
    ("file_name", "needle", "replacement"),
    [
        ("contract_summary", "repository_runtime_command:", "repository_runtime_command: broken #"),
        ("contract_summary", "qualification_attestation_command:", "qualification_attestation_command: broken #"),
        ("contract_artifact", "source_summary_sha256:", "source_summary_sha256: broken #"),
        ("validation", "native_evidence: false", "native_evidence: true"),
        ("validation", "windows_qualified: false", "windows_qualified: true"),
    ],
)
def test_verify_rejects_structural_binding_or_support_claim_changes(
    tmp_path: Path, file_name: str, needle: str, replacement: str
) -> None:
    helper = _load_helper()
    paths = _completed_inputs(tmp_path)
    _capture(helper, paths)
    changed = paths[file_name].read_text(encoding="utf-8")
    paths[file_name].write_text(changed.replace(needle, replacement), encoding="utf-8")

    with pytest.raises(helper.AttestationError):
        _verify(helper, paths)


def test_atomic_write_failure_preserves_existing_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    helper = _load_helper()
    paths = _completed_inputs(tmp_path)
    paths["artifact"].write_bytes(b"existing record")
    monkeypatch.setattr(helper.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("no")))

    with pytest.raises(OSError):
        _capture(helper, paths)

    assert paths["artifact"].read_bytes() == b"existing record"
    assert not list(tmp_path.glob(".qualification.md.*.tmp"))


def test_verify_optional_summary_must_mirror_the_qualification_record(tmp_path: Path) -> None:
    helper = _load_helper()
    paths = _completed_inputs(tmp_path)
    _capture(helper, paths)
    summary = tmp_path / "03-11-SUMMARY.md"
    _write(
        summary,
        f"""---
windows-qualification:
  artifact_sha256: {_digest(paths['artifact'])}
  contract_summary_sha256: {_digest(paths['contract_summary'])}
  qualification_attestation_command: \"{' '.join(helper.QUALIFICATION_ARGV)}\"
  runner_status: UNAVAILABLE
  runner_exit_code: 2
  milestone_status: NOT_QUALIFIED
  native_evidence: false
  backlog_phase: 999.1
  future_required_status: PASS
  future_required_exit_code: 0
---
""",
    )
    _verify(helper, paths, qualification_summary=summary)

    summary.write_text(
        summary.read_text(encoding="utf-8").replace("native_evidence: false", "native_evidence: true"),
        encoding="utf-8",
    )
    with pytest.raises(helper.AttestationError):
        _verify(helper, paths, qualification_summary=summary)


def test_real_current_host_fixed_command_is_repeatable() -> None:
    helper = _load_helper()

    first = helper.run_fixed_command()
    second = helper.run_fixed_command()

    assert tuple(first.args) == helper.QUALIFICATION_ARGV
    assert first.returncode == second.returncode == 2
    assert first.stdout == second.stdout
