"""Contracts for Phase 8 Python and operating-system qualification evidence."""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
from pathlib import Path

import pytest


RUNNER_PATH = (
    Path(__file__).resolve().parents[2] / "tools" / "run_phase8_platform_gates.py"
)
LOCAL_GATE_RUNNER_PATH = (
    Path(__file__).resolve().parents[2] / "tools" / "run_phase8_local_gates.py"
)


@pytest.fixture()
def runner():
    """Load the standalone platform evidence runner."""
    spec = importlib.util.spec_from_file_location("phase8_platform_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def local_gate_runner():
    """Load the local gate CLI without importing project package code."""
    spec = importlib.util.spec_from_file_location(
        "phase8_local_gate_runner", LOCAL_GATE_RUNNER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stable_rows(runner, *, profile: str = "core") -> list[dict[str, object]]:
    return [
        runner.build_row(
            expected_os="Linux",
            python_minor=minor,
            feature_profile=profile,
            command_status="PASS",
        )
        for minor in runner.STABLE_PYTHON_MINORS
    ]


def test_python_stable_matrix_is_exact_and_complete(runner) -> None:
    """Every supported stable Python minor has exactly one Linux core row."""
    assert runner.STABLE_PYTHON_MINORS == ("3.11", "3.12", "3.13", "3.14")
    evidence = runner.aggregate_rows(_stable_rows(runner))
    assert evidence["status"] == "QUALIFIED"
    assert evidence["qualified_slots"] == [
        f"Linux/{minor}/core" for minor in runner.STABLE_PYTHON_MINORS
    ]


def test_python_missing_or_duplicate_stable_rows_fail_closed(runner) -> None:
    """Omissions and duplicate rows cannot be hidden by an aggregate pass."""
    rows = _stable_rows(runner)
    with pytest.raises(ValueError, match="missing required"):
        runner.aggregate_rows(rows[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        runner.aggregate_rows([*rows, rows[0]])


def test_python_advisory_result_cannot_satisfy_or_invalidate_stable_slot(
    runner,
) -> None:
    """Prerelease evidence is retained as advisory without changing stable outcome."""
    rows = _stable_rows(runner)
    rows.append(
        runner.build_row(
            expected_os="Linux",
            python_minor="3.15",
            feature_profile="core",
            command_status="FAIL",
            advisory=True,
        )
    )
    assert runner.aggregate_rows(rows)["status"] == "QUALIFIED"

    contradictory = _stable_rows(runner)
    contradictory[0]["advisory"] = True
    with pytest.raises(ValueError, match="stable row cannot be advisory"):
        runner.aggregate_rows(contradictory)

    unpublished = _stable_rows(runner)
    unpublished.append(
        runner.build_row(
            expected_os="Linux",
            python_minor="3.16",
            feature_profile="core",
            command_status="PASS",
            advisory=True,
        )
    )
    with pytest.raises(ValueError, match="published advisory"):
        runner.aggregate_rows(unpublished)


@pytest.mark.parametrize("retired_profile", ("tensorflow", "non_tensorflow"))
def test_core_is_the_only_feature_profile(
    runner, local_gate_runner, tmp_path: Path, retired_profile: str
) -> None:
    """Both platform tools reject retired profiles before emitting evidence."""
    assert runner.FEATURE_PROFILES == frozenset({"core"})
    assert not runner.is_feature_profile_compatible(retired_profile, "3.11")
    with pytest.raises(ValueError, match="unsupported feature profile"):
        runner.build_row(
            expected_os="Linux",
            python_minor="3.11",
            feature_profile=retired_profile,
            command_status="PASS",
        )
    with pytest.raises(SystemExit):
        local_gate_runner.main(
            [
                "platform",
                "--expected-os",
                "Linux",
                "--python-minor",
                "3.11",
                "--feature-profile",
                retired_profile,
                "--output",
                str(tmp_path / f"{retired_profile}.json"),
            ]
        )


def test_python_runtime_identity_mismatch_is_rejected(runner, tmp_path: Path) -> None:
    """A caller cannot label one executing interpreter as another row."""
    current_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    wrong_minor = "3.11" if current_minor != "3.11" else "3.12"
    output = tmp_path / "evidence.json"

    exit_code = runner.run_platform_gate(
        expected_os=platform.system(),
        python_minor=wrong_minor,
        feature_profile="core",
        output=output,
    )

    assert exit_code == 2
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["status"] == "UNAVAILABLE"
    assert evidence["payload"]["reason"] == "runtime_identity_mismatch"


def test_linux_platform_row_runs_only_the_fixed_deterministic_contract(runner) -> None:
    """A platform row cannot fail because an unrelated optional extra is absent."""

    command = runner._command_for_linux()

    assert command[0] == runner.sys.executable
    assert command[1].endswith("tools/verify_phase071_contracts.py")
    assert command[2:] == ("--all",)


def _platform_role_rows(runner) -> list[dict[str, object]]:
    return [
        *_stable_rows(runner),
        runner.build_row(
            expected_os="Darwin",
            python_minor="3.11",
            feature_profile="core",
            command_status="PASS",
        ),
        runner.build_row(
            expected_os="Darwin",
            python_minor="3.14",
            feature_profile="core",
            command_status="PASS",
        ),
        runner.build_row(
            expected_os="Windows",
            actual_os="Darwin",
            python_minor="3.11",
            feature_profile="core",
            command_status="UNAVAILABLE",
        ),
    ]


def test_platform_linux_matrix_and_macos_boundaries_are_distinct(runner) -> None:
    """macOS boundary smoke complements but never replaces full Linux evidence."""
    result = runner.aggregate_platform_roles(_platform_role_rows(runner))

    assert runner.MACOS_BOUNDARY_MINORS == ("3.11", "3.14")
    assert result["linux_status"] == "QUALIFIED"
    assert result["macos_boundary_slots"] == ["Darwin/3.11/core", "Darwin/3.14/core"]
    assert result["status"] == "NOT_QUALIFIED"

    macos_only = _platform_role_rows(runner)[4:6]
    with pytest.raises(ValueError, match="missing required"):
        runner.aggregate_platform_roles(macos_only)


def test_platform_windows_is_non_native_unavailable_without_substitute_execution(
    runner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 8 cannot manufacture a Windows support claim from another host."""
    monkeypatch.setattr(
        runner,
        "_run_fixed_command",
        lambda *_args, **_kwargs: pytest.fail("Windows must not execute a substitute"),
    )
    output = tmp_path / "windows.json"

    assert (
        runner.run_platform_gate(
            expected_os="Windows",
            python_minor="3.11",
            feature_profile="core",
            output=output,
        )
        == 2
    )
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["status"] == "UNAVAILABLE"
    assert evidence["payload"]["backlog_phase"] == "999.1"
    assert evidence["payload"]["reason"] == "native_windows_phase_999_1_required"


def test_platform_macos_boundary_uses_fixed_public_smoke(
    runner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A qualifying macOS boundary row invokes only the fixed public smoke."""
    seen: list[tuple[str, ...]] = []
    monkeypatch.setattr(runner.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runner, "current_python_minor", lambda: "3.11")
    monkeypatch.setattr(runner, "_source_is_clean", lambda: True)
    monkeypatch.setattr(
        runner,
        "_run_fixed_command",
        lambda command, _timeout: seen.append(tuple(command)) or "passed",
    )
    output = tmp_path / "macos.json"

    assert (
        runner.run_platform_gate(
            expected_os="Darwin",
            python_minor="3.11",
            feature_profile="core",
            output=output,
        )
        == 0
    )
    assert seen == [runner.macos_boundary_smoke_command()]
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS"
    assert evidence["payload"]["command_profile"] == "macos_boundary"


@pytest.mark.parametrize("outcome", ("success", "conflict", "typed_retryable"))
def test_platform_adr_progress_outcomes_remain_valid(outcome: str, runner) -> None:
    """Contention progress states are not portability or corruption failures."""
    assert runner.validate_adr_progress_outcome(outcome) == outcome
