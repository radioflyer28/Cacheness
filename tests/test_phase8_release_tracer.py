"""End-to-end contracts for the Phase 8 deterministic evidence tracer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPOSITORY_ROOT / "tools" / "phase8_evidence.py"
RUNNER_PATH = REPOSITORY_ROOT / "tools" / "run_phase8_local_gates.py"


def _load_module(name: str, path: Path):
    """Load a standalone qualification tool without making ``tools`` a package."""
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _load_evidence():
    return _load_module("phase8_evidence_test", EVIDENCE_PATH)


def _load_runner():
    return _load_module("phase8_local_gates_test", RUNNER_PATH)


def _clean_identity(runner, revision: str = "a" * 40):
    return runner.SourceIdentity(
        revision=revision,
        source_digest="b" * 64,
        clean=True,
    )


def _passing_child() -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="21 passed\nPhase 07.1 all contract passed in 1.00s\n",
        stderr="",
    )


def test_tracer_writes_one_validated_exact_commit_deterministic_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The literal Phase 07.1 all-mode command produces one PASS envelope."""
    evidence = _load_evidence()
    runner = _load_runner()
    output = tmp_path / "deterministic.json"
    identity = _clean_identity(runner)
    observed: list[tuple[tuple[str, ...], int]] = []

    def run_child(command: tuple[str, ...], timeout: int) -> subprocess.CompletedProcess[str]:
        observed.append((command, timeout))
        return _passing_child()

    monkeypatch.setattr(runner, "current_source_identity", lambda: identity)

    assert runner.run_deterministic(output=output, run_child=run_child) == 0
    assert observed == [(runner.DETERMINISTIC_COMMAND, runner.CHILD_TIMEOUT_SECONDS)]

    envelope = evidence.load_envelope(output)
    assert envelope.evidence_class == "deterministic"
    assert envelope.status == "PASS"
    assert envelope.revision == "a" * 40
    assert envelope.source_digest == "b" * 64
    assert envelope.payload["claim_categories"] == {
        "integrity": "EVIDENCED",
        "recovery": "EVIDENCED",
        "progress": "EVIDENCED",
        "performance": "NOT_QUALIFIED",
    }
    assert envelope.payload["non_qualifying_classes"] == [
        "packaging",
        "platform",
        "coverage",
        "structural",
        "controlled_performance",
        "live_services",
    ]


@pytest.mark.parametrize(
    "child",
    [
        subprocess.CompletedProcess([], 1, "1 failed", ""),
        subprocess.CompletedProcess([], 0, "20 passed, 1 skipped", ""),
        subprocess.CompletedProcess([], 0, "no tests ran", ""),
        subprocess.CompletedProcess([], 0, "20 passed", ""),
    ],
)
def test_tracer_rejects_failed_skipped_incomplete_or_unattested_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    child: subprocess.CompletedProcess[str],
) -> None:
    """No incomplete child result can manufacture a deterministic PASS."""
    evidence = _load_evidence()
    runner = _load_runner()
    output = tmp_path / "deterministic.json"
    monkeypatch.setattr(runner, "current_source_identity", lambda: _clean_identity(runner))

    assert runner.run_deterministic(output=output, run_child=lambda *_args: child) == 1
    envelope = evidence.load_envelope(output)
    assert envelope.status == "NOT_QUALIFIED"
    assert envelope.payload["result"] != "passed"


def test_tracer_rejects_dirty_or_revision_drifted_source_after_child_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child pass cannot qualify sources that were dirty or changed during it."""
    evidence = _load_evidence()
    runner = _load_runner()
    output = tmp_path / "deterministic.json"
    identities = iter(
        (
            _clean_identity(runner, "a" * 40),
            _clean_identity(runner, "c" * 40),
        )
    )
    monkeypatch.setattr(runner, "current_source_identity", lambda: next(identities))

    assert runner.run_deterministic(output=output, run_child=lambda *_args: _passing_child()) == 1
    envelope = evidence.load_envelope(output)
    assert envelope.status == "NOT_QUALIFIED"
    assert envelope.payload["result"] == "source_changed"


def test_tracer_cli_has_no_selectable_child_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public command accepts only the deterministic mode and output path."""
    runner = _load_runner()
    output = tmp_path / "deterministic.json"
    monkeypatch.setattr(runner, "current_source_identity", lambda: _clean_identity(runner))
    monkeypatch.setattr(runner, "_run_child", lambda *_args: _passing_child())

    assert runner.main(["deterministic", "--output", str(output)]) == 0
    assert output.is_file()

