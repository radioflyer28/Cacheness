"""End-to-end contracts for the Phase 8 deterministic evidence tracer."""

from __future__ import annotations

import importlib.util
import json
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
    """The public command reports unavailable external classes without a false pass."""
    runner = _load_runner()
    output = tmp_path / "deterministic.json"
    monkeypatch.setattr(runner, "current_source_identity", lambda: _clean_identity(runner))
    monkeypatch.setattr(runner, "_run_child", lambda *_args: _passing_child())

    assert runner.main(["deterministic", "--output", str(output)]) == 2
    assert output.is_file()


def test_non_passing_terminal_states_are_truthful_but_not_qualification() -> None:
    """Unavailable and failed evidence remain diagnostic instead of release proof."""
    evidence = _load_evidence()
    expected_non_qualifying = [
        evidence_class
        for evidence_class in evidence.EVIDENCE_CLASSES
        if evidence_class != "deterministic"
    ]

    for status, result, claim_state in (
        ("UNAVAILABLE", "unavailable", "UNAVAILABLE"),
        ("NOT_QUALIFIED", "failed", "NOT_QUALIFIED"),
    ):
        envelope = evidence.make_envelope(
            evidence_class="deterministic",
            status=status,
            revision="a" * 40,
            source_digest="b" * 64,
            generated_at_utc="2026-09-13T00:00:00+00:00",
            payload={
                "command": ["tools/verify_phase071_contracts.py", "--all"],
                "result": result,
                "claim_categories": {
                    claim: claim_state for claim in evidence.CLAIM_CATEGORIES
                },
                "non_qualifying_classes": expected_non_qualifying,
                "subjects": list(evidence.QUALIFIED_SUBJECTS),
            },
        )

        assert envelope.status == status
        assert not evidence.is_qualification_evidence(envelope, "deterministic")


@pytest.mark.parametrize(
    ("claim", "state"),
    [("performance", "EVIDENCED"), ("progress", "DIAGNOSTIC")],
)
def test_evidence_rejects_terminal_state_and_claim_category_contradictions(
    claim: str, state: str
) -> None:
    """A PASS cannot relabel a benchmark or contention result as a lifecycle claim."""
    evidence = _load_evidence()
    payload = {
        "command": ["tools/verify_phase071_contracts.py", "--all"],
        "result": "passed",
        "claim_categories": {
            "integrity": "EVIDENCED",
            "recovery": "EVIDENCED",
            "progress": "EVIDENCED" if claim != "progress" else state,
            "performance": "NOT_QUALIFIED" if claim != "performance" else state,
        },
        "non_qualifying_classes": [
            evidence_class
            for evidence_class in evidence.EVIDENCE_CLASSES
            if evidence_class != "deterministic"
        ],
        "subjects": list(evidence.QUALIFIED_SUBJECTS),
    }

    with pytest.raises(evidence.EvidenceValidationError):
        evidence.make_envelope(
            evidence_class="deterministic",
            status="PASS",
            revision="a" * 40,
            source_digest="b" * 64,
            payload=payload,
        )


def test_evidence_rejects_forged_source_digest_for_the_required_identity(
    tmp_path: Path,
) -> None:
    """A digest-looking value is insufficient when it disagrees with reviewed sources."""
    evidence = _load_evidence()
    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    envelope = evidence.make_envelope(
        evidence_class="deterministic",
        status="PASS",
        revision="a" * 40,
        source_digest="b" * 64,
        payload={
            "command": ["tools/verify_phase071_contracts.py", "--all"],
            "result": "passed",
            "claim_categories": {
                "integrity": "EVIDENCED",
                "recovery": "EVIDENCED",
                "progress": "EVIDENCED",
                "performance": "NOT_QUALIFIED",
            },
            "non_qualifying_classes": [
                evidence_class
                for evidence_class in evidence.EVIDENCE_CLASSES
                if evidence_class != "deterministic"
            ],
            "subjects": list(evidence.QUALIFIED_SUBJECTS),
        },
    )

    with pytest.raises(evidence.EvidenceValidationError, match="source digest"):
        evidence.validate_source_identity(
            envelope,
            revision="a" * 40,
            root=tmp_path,
            paths=("source.py",),
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update({"unexpected": "field"}),
        lambda value: value["payload"].update({"result": "unavailable"}),
    ],
)
def test_evidence_rejects_unknown_keys_and_contradictory_canonical_json(
    tmp_path: Path, mutate
) -> None:
    """Untrusted envelope files have no extension keys or alternate terminal results."""
    evidence = _load_evidence()
    envelope = evidence.make_envelope(
        evidence_class="deterministic",
        status="PASS",
        revision="a" * 40,
        source_digest="b" * 64,
        generated_at_utc="2026-09-13T00:00:00+00:00",
        payload={
            "command": ["tools/verify_phase071_contracts.py", "--all"],
            "result": "passed",
            "claim_categories": {
                "integrity": "EVIDENCED",
                "recovery": "EVIDENCED",
                "progress": "EVIDENCED",
                "performance": "NOT_QUALIFIED",
            },
            "non_qualifying_classes": [
                evidence_class
                for evidence_class in evidence.EVIDENCE_CLASSES
                if evidence_class != "deterministic"
            ],
            "subjects": list(evidence.QUALIFIED_SUBJECTS),
        },
    )
    value = envelope.to_mapping()
    mutate(value)
    path = tmp_path / "evidence.json"
    path.write_text(
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(evidence.EvidenceValidationError):
        evidence.load_envelope(path)


def test_evidence_rejects_duplicate_unsafe_and_oversized_file_inputs(
    tmp_path: Path,
) -> None:
    """Evidence loading bounds hostile bytes before later release tooling inspects them."""
    evidence = _load_evidence()
    envelope = evidence.make_envelope(
        evidence_class="deterministic",
        status="PASS",
        revision="a" * 40,
        source_digest="b" * 64,
        generated_at_utc="2026-09-13T00:00:00+00:00",
        payload={
            "command": ["tools/verify_phase071_contracts.py", "--all"],
            "result": "passed",
            "claim_categories": {
                "integrity": "EVIDENCED",
                "recovery": "EVIDENCED",
                "progress": "EVIDENCED",
                "performance": "NOT_QUALIFIED",
            },
            "non_qualifying_classes": [
                evidence_class
                for evidence_class in evidence.EVIDENCE_CLASSES
                if evidence_class != "deterministic"
            ],
            "subjects": list(evidence.QUALIFIED_SUBJECTS),
        },
    )
    canonical = json.dumps(
        envelope.to_mapping(), ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8") + b"\n"

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(
        canonical.replace(
            b'"revision":', b'"revision":"c"' * 40 + b',"revision":', 1
        )
    )
    unsafe = tmp_path / "unsafe.json"
    unsafe.write_bytes(canonical.replace(b"BlobStore", b"password!", 1))
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"x" * (evidence.MAX_EVIDENCE_BYTES + 1))

    for path in (duplicate, unsafe, oversized):
        with pytest.raises(evidence.EvidenceValidationError):
            evidence.load_envelope(path)


def test_unavailable_evidence_rejects_a_non_unavailable_claim() -> None:
    """Missing prerequisites cannot silently become a partial claim."""
    evidence = _load_evidence()

    with pytest.raises(evidence.EvidenceValidationError):
        evidence.make_envelope(
            evidence_class="deterministic",
            status="UNAVAILABLE",
            revision="a" * 40,
            source_digest="b" * 64,
            payload={
                "command": ["tools/verify_phase071_contracts.py", "--all"],
                "result": "unavailable",
                "claim_categories": {
                    "integrity": "UNAVAILABLE",
                    "recovery": "UNAVAILABLE",
                    "progress": "NOT_QUALIFIED",
                    "performance": "UNAVAILABLE",
                },
                "non_qualifying_classes": [
                    evidence_class
                    for evidence_class in evidence.EVIDENCE_CLASSES
                    if evidence_class != "deterministic"
                ],
                "subjects": list(evidence.QUALIFIED_SUBJECTS),
            },
        )


def test_report_names_each_unproduced_evidence_class_as_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The local command distinguishes a passing contract from absent external proof."""
    runner = _load_runner()
    output = tmp_path / "deterministic.json"
    monkeypatch.setattr(runner, "current_source_identity", lambda: _clean_identity(runner))
    monkeypatch.setattr(runner, "_run_child", lambda *_args: _passing_child())

    assert runner.main(["deterministic", "--output", str(output)]) == 2
    report = capsys.readouterr().out
    assert "deterministic: PASS" in report
    for evidence_class in (
        "packaging",
        "platform",
        "coverage",
        "structural",
        "controlled_performance",
        "live_services",
    ):
        assert f"{evidence_class}: UNAVAILABLE" in report
