"""Fail-closed contracts for Phase 8 exact-SHA release collection."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).parents[2]
RELEASE_PATH = REPOSITORY_ROOT / "tools" / "verify_phase8_release.py"
EVIDENCE_PATH = REPOSITORY_ROOT / "tools" / "phase8_evidence.py"


def _load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _load_release():
    return _load_module("phase8_release_verifier", RELEASE_PATH)


def _load_evidence():
    return _load_module("phase8_release_evidence", EVIDENCE_PATH)


def _envelope_bytes(evidence_class: str, revision: str) -> bytes:
    if evidence_class == "controlled_performance":
        return json.dumps(
            {
                "schema": "cacheness-phase8-performance-v1",
                "evidence_class": "controlled-performance",
                "revision": revision,
                "source_digest": "c" * 64,
                "runner_identity": "cacheness-perf-linux-x64",
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    if evidence_class == "live_services":
        return json.dumps(
            {
                "schema": "phase8-live-qualification-v1",
                "status": "QUALIFIED",
                "revision": revision,
                "source_digest": "c" * 64,
                "cleanup_status": "CLEAN",
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    evidence = _load_evidence()
    payload = {
        "result": "passed",
        "claim_categories": {
            "integrity": "NOT_QUALIFIED"
            if evidence_class == "packaging"
            else "EVIDENCED",
            "recovery": "NOT_QUALIFIED"
            if evidence_class == "packaging"
            else "EVIDENCED",
            "progress": "NOT_QUALIFIED"
            if evidence_class == "packaging"
            else "EVIDENCED",
            "performance": "NOT_QUALIFIED",
        },
        "non_qualifying_classes": [
            item for item in evidence.EVIDENCE_CLASSES if item != evidence_class
        ],
        "subjects": list(evidence.QUALIFIED_SUBJECTS),
    }
    if evidence_class == "deterministic":
        payload["command"] = ["tools/verify_phase071_contracts.py", "--all"]
    if evidence_class == "platform":
        payload.update(
            {
                "expected_os": "Linux",
                "actual_os": "Linux",
                "expected_python_minor": "3.13",
                "actual_python_minor": "3.13",
                "feature_profile": "core",
                "role": "stable_linux",
                "advisory": False,
                "command_profile": "fixed",
                "backlog_phase": "none",
                "reason": "qualified",
            }
        )
    if evidence_class == "packaging":
        payload.update(
            {
                "wheel_sha256": "b" * 64,
                "python": "3.12.0",
                "platform": "Linux",
                "probes": ["base", "optional"],
                "optional_groups": [
                    "recommended",
                    "dataframes",
                    "tensorflow",
                    "s3",
                    "postgresql",
                    "cloud",
                ],
                "compatibility": [
                    "recommended:COMPATIBLE",
                    "dataframes:COMPATIBLE",
                    "tensorflow:COMPATIBLE",
                    "s3:COMPATIBLE",
                    "postgresql:COMPATIBLE",
                    "cloud:COMPATIBLE",
                ],
                "non_live_groups": ["s3", "postgresql", "cloud"],
            }
        )
    if evidence_class == "structural":
        payload.update(
            {
                "environment": {"os": "Linux", "rss_unit": "bytes"},
                "observations": [
                    {
                        "operation": "catalog",
                        "seeded_entries": 1,
                        "page_size": 1,
                        "work_cap": 1,
                        "selected_entries": 1,
                        "workload_bytes": 1,
                        "peak_rss_bytes": 1,
                        "counters": {
                            "authority_pages": 1,
                            "authority_reads": 1,
                            "authority_writes": 0,
                            "participant_head": 0,
                            "participant_open": 0,
                            "participant_delete": 0,
                            "participant_list": 0,
                        },
                    }
                ],
            }
        )
    envelope = evidence.make_envelope(
        evidence_class=evidence_class,
        status="PASS",
        revision=revision,
        source_digest="c" * 64,
        generated_at_utc="2026-09-13T00:00:00+00:00",
        payload=payload,
    )
    return (
        json.dumps(
            envelope.to_mapping(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


class FakeGh:
    """Tiny command seam that creates fixed named artifacts on download."""

    def __init__(
        self, *, revision: str, duplicate: bool = False, wrong_head: bool = False
    ) -> None:
        self.revision = revision
        self.duplicate = duplicate
        self.wrong_head = wrong_head
        self.commands: list[tuple[str, ...]] = []
        self.dispatched: set[str] = set()

    def __call__(self, command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        if command[:3] == ("gh", "run", "list"):
            workflow = command[command.index("--workflow") + 1]
            if workflow not in self.dispatched:
                return subprocess.CompletedProcess(command, 0, "[]", "")
            spec = _load_release().WORKFLOW_SPECS[workflow]
            run = {
                "databaseId": spec.run_id,
                "headSha": "d" * 40 if self.wrong_head else self.revision,
                "workflowName": spec.workflow_name,
                "event": "workflow_dispatch",
                "status": "completed",
                "conclusion": "success",
            }
            values = (
                [run, {**run, "databaseId": spec.run_id + 1}]
                if self.duplicate
                else [run]
            )
            return subprocess.CompletedProcess(command, 0, json.dumps(values), "")
        if command[:3] == ("gh", "workflow", "run"):
            self.dispatched.add(command[3])
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[:3] == ("gh", "run", "view"):
            run_id = int(command[3])
            spec = next(
                spec
                for spec in _load_release().WORKFLOW_SPECS.values()
                if spec.run_id == run_id
            )
            value = {
                "databaseId": run_id,
                "headSha": "d" * 40 if self.wrong_head else self.revision,
                "workflowName": spec.workflow_name,
                "event": "workflow_dispatch",
                "status": "completed",
                "conclusion": "success",
            }
            return subprocess.CompletedProcess(command, 0, json.dumps(value), "")
        if command[:3] == ("gh", "run", "watch"):
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[:3] == ("gh", "run", "download"):
            artifact = command[command.index("-n") + 1]
            destination = Path(command[command.index("-D") + 1])
            destination.mkdir(parents=True, exist_ok=True)
            evidence_class = _load_release().ARTIFACT_EVIDENCE_CLASS[artifact]
            (destination / f"{evidence_class}.json").write_bytes(
                _envelope_bytes(evidence_class, self.revision)
            )
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(f"unexpected command: {command!r}")


def test_dispatch_requires_exact_candidate_sha_and_records_one_run_id(
    tmp_path: Path,
) -> None:
    """Each declared workflow starts from one explicit lowercase full revision."""
    release = _load_release()
    candidate = "a" * 40
    fake = FakeGh(revision=candidate)

    collected = release.collect_workflow_evidence(
        candidate_sha=candidate, output_directory=tmp_path, execute=fake
    )

    assert {item.workflow for item in collected} == set(release.WORKFLOW_SPECS)
    assert {item.revision for item in collected} == {candidate}
    assert all(
        item.run_id > 0 and len(item.artifact_sha256) == 64 for item in collected
    )
    dispatches = [
        command for command in fake.commands if command[:3] == ("gh", "workflow", "run")
    ]
    assert len(dispatches) == len(release.WORKFLOW_SPECS)
    for command in dispatches:
        assert "--ref" in command and command[command.index("--ref") + 1] == candidate
        assert "candidate_sha=" + candidate in command


@pytest.mark.parametrize("candidate", ["A" * 40, "a" * 39, "a" * 41, "main"])
def test_dispatch_rejects_noncanonical_candidate_sha_before_gh(
    candidate: str, tmp_path: Path
) -> None:
    """A branch, abbreviated, or uppercase ref cannot select release evidence."""
    release = _load_release()
    fake = FakeGh(revision="a" * 40)

    with pytest.raises(release.ReleaseEvidenceError, match="candidate SHA"):
        release.collect_workflow_evidence(
            candidate_sha=candidate, output_directory=tmp_path, execute=fake
        )

    assert fake.commands == []


def test_run_id_collection_rejects_duplicate_latest_and_wrong_sha_substitutions(
    tmp_path: Path,
) -> None:
    """Collector derives one new candidate run rather than choosing a latest run."""
    release = _load_release()
    candidate = "a" * 40

    with pytest.raises(release.ReleaseEvidenceError, match="one newly dispatched run"):
        release.collect_workflow_evidence(
            candidate_sha=candidate,
            output_directory=tmp_path / "duplicate",
            execute=FakeGh(revision=candidate, duplicate=True),
        )
    with pytest.raises(release.ReleaseEvidenceError, match="head SHA"):
        release.collect_workflow_evidence(
            candidate_sha=candidate,
            output_directory=tmp_path / "wrong-head",
            execute=FakeGh(revision=candidate, wrong_head=True),
        )


def test_artifact_collection_requires_run_id_fixed_name_and_one_bounded_file(
    tmp_path: Path,
) -> None:
    """Filename-only downloads and arbitrary artifact trees cannot enter a release."""
    release = _load_release()
    candidate = "a" * 40
    fake = FakeGh(revision=candidate)

    collected = release.collect_workflow_evidence(
        candidate_sha=candidate, output_directory=tmp_path, execute=fake
    )

    downloads = [
        command for command in fake.commands if command[:3] == ("gh", "run", "download")
    ]
    assert downloads
    assert all(
        command[3].isdigit() and "-n" in command and "-D" in command
        for command in downloads
    )
    assert all(
        item.artifact_name in release.ARTIFACT_EVIDENCE_CLASS for item in collected
    )


def test_aggregate_requires_one_same_revision_same_digest_qualifying_class(
    tmp_path: Path,
) -> None:
    """One class cannot impersonate another or inherit an earlier candidate pass."""
    release = _load_release()
    candidate = "a" * 40
    collected = release.collect_workflow_evidence(
        candidate_sha=candidate,
        output_directory=tmp_path / "collected",
        execute=FakeGh(revision=candidate),
    )
    release.write_collection_manifest(tmp_path / "collected", collected)

    manifest = release.aggregate_collection(
        candidate_sha=candidate,
        collection_directory=tmp_path / "collected",
        output=tmp_path / "aggregate.json",
    )

    assert manifest["revision"] == candidate
    assert set(manifest["evidence"]) == set(release.ARTIFACT_EVIDENCE_CLASS.values())
    assert (tmp_path / "aggregate.json").is_file()


def test_aggregate_rejects_stale_duplicate_and_nonqualifying_evidence(
    tmp_path: Path,
) -> None:
    """A stale source, duplicate class, or unclean live evidence blocks release."""
    release = _load_release()
    candidate = "a" * 40
    collected = release.collect_workflow_evidence(
        candidate_sha=candidate,
        output_directory=tmp_path / "collected",
        execute=FakeGh(revision=candidate),
    )
    release.write_collection_manifest(tmp_path / "collected", collected)
    manifest_path = tmp_path / "collected" / release.COLLECTION_MANIFEST_NAME
    document = json.loads(manifest_path.read_text(encoding="utf-8"))

    document["artifacts"][0]["source_digest"] = "d" * 64
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(release.ReleaseEvidenceError, match="source digest"):
        release.aggregate_collection(
            candidate_sha=candidate,
            collection_directory=tmp_path / "collected",
            output=tmp_path / "bad.json",
        )

    document["artifacts"][0]["source_digest"] = "c" * 64
    document["artifacts"].append(dict(document["artifacts"][0]))
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(release.ReleaseEvidenceError, match="exactly one"):
        release.aggregate_collection(
            candidate_sha=candidate,
            collection_directory=tmp_path / "collected",
            output=tmp_path / "duplicate.json",
        )


class FakeReleaseInspection:
    """Read-only Git/GitHub release state adapter for immutable-publication tests."""

    def __init__(
        self, *, candidate: str, assets: list[dict[str, object]], draft: bool = False
    ) -> None:
        self.candidate = candidate
        self.assets = assets
        self.draft = draft
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        if command[:3] == ("git", "rev-parse", "--verify"):
            return subprocess.CompletedProcess(command, 0, self.candidate + "\n", "")
        if command[:3] == ("gh", "release", "view"):
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "tagName": "v1.0.0",
                        "isDraft": self.draft,
                        "isImmutable": True,
                        "assets": self.assets,
                    }
                ),
                "",
            )
        if command[:3] in {
            ("gh", "release", "verify"),
            ("gh", "release", "verify-asset"),
        }:
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(f"unexpected command: {command!r}")


def _sha256(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def test_publication_verifier_requires_exact_published_immutable_assets(
    tmp_path: Path,
) -> None:
    """Tag target, API asset state, digest, and GitHub verification all agree."""
    release = _load_release()
    candidate = "a" * 40
    assets = []
    asset_paths: list[Path] = []
    for name, content in (
        ("phase8-release-qualification.json", b"{}\n"),
        ("phase8-live.json", b"{}\n"),
    ):
        path = tmp_path / name
        path.write_bytes(content)
        asset_paths.append(path)
        assets.append(
            {"name": name, "state": "uploaded", "digest": "sha256:" + _sha256(content)}
        )
    fake = FakeReleaseInspection(candidate=candidate, assets=assets)

    report = release.verify_published_release(
        candidate_sha=candidate, tag="v1.0.0", assets=asset_paths, execute=fake
    )

    assert report["revision"] == candidate
    assert report["immutable"] is True
    assert any(
        command[:3] == ("gh", "release", "verify-asset") for command in fake.commands
    )


def test_publication_verifier_rejects_draft_extra_and_digest_mismatch(
    tmp_path: Path,
) -> None:
    """A local file or draft release is never equivalent to immutable publication."""
    release = _load_release()
    candidate = "a" * 40
    asset = tmp_path / "phase8-release-qualification.json"
    asset.write_bytes(b"{}\n")
    api_asset = {
        "name": asset.name,
        "state": "uploaded",
        "digest": "sha256:" + _sha256(asset.read_bytes()),
    }
    with pytest.raises(release.ReleaseEvidenceError, match="draft"):
        release.verify_published_release(
            candidate_sha=candidate,
            tag="v1.0.0",
            assets=[asset],
            execute=FakeReleaseInspection(
                candidate=candidate, assets=[api_asset], draft=True
            ),
        )
    with pytest.raises(release.ReleaseEvidenceError, match="asset inventory"):
        release.verify_published_release(
            candidate_sha=candidate,
            tag="v1.0.0",
            assets=[asset],
            execute=FakeReleaseInspection(
                candidate=candidate,
                assets=[api_asset, {**api_asset, "name": "diagnostic.json"}],
            ),
        )
