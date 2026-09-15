"""Fail-closed contracts for Phase 8 exact-SHA release collection."""

from __future__ import annotations

import importlib.util
import hashlib
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
        if command[:3] == ("git", "rev-parse", "--verify"):
            return subprocess.CompletedProcess(command, 0, self.revision + "\n", "")
        if command[:3] == ("gh", "auth", "status"):
            return subprocess.CompletedProcess(command, 0, "authenticated\n", "")
        if command[:3] == ("gh", "workflow", "view"):
            return subprocess.CompletedProcess(command, 0, "workflow: visible\n", "")
        if command[1:] == ("tools/verify_phase8_contracts.py", "--quick"):
            return subprocess.CompletedProcess(command, 0, "", "")
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

    assert {item.workflow for item in collected} == set(
        release.CURRENT_RELEASE_WORKFLOW_SPECS
    )
    assert {item.revision for item in collected} == {candidate}
    assert all(
        item.run_id > 0 and len(item.artifact_sha256) == 64 for item in collected
    )
    dispatches = [
        command for command in fake.commands if command[:3] == ("gh", "workflow", "run")
    ]
    assert len(dispatches) == len(release.CURRENT_RELEASE_WORKFLOW_SPECS)
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
        item.artifact_name in release.CURRENT_ARTIFACT_EVIDENCE_CLASS
        for item in collected
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
    assert set(manifest["evidence"]) == set(
        release.CURRENT_ARTIFACT_EVIDENCE_CLASS.values()
    )
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


def test_preflight_collection_reports_exact_current_inventory(tmp_path: Path) -> None:
    """Read-only preflight proves only quality/live evidence is currently dispatchable."""
    release = _load_release()
    candidate = "a" * 40
    fake = FakeGh(revision=candidate)

    report = release.preflight_collection(candidate_sha=candidate, execute=fake)

    assert report["candidate_sha"] == candidate
    assert report["workflows"] == ["quality.yml", "live_qualification.yml"]
    assert report["artifacts"] == sorted(release.CURRENT_ARTIFACT_EVIDENCE_CLASS)
    assert not any(
        command[:3] == ("gh", "workflow", "run") for command in fake.commands
    )
    assert ("gh", "auth", "status") in fake.commands


def test_required_release_collection_excludes_deferred_performance(
    tmp_path: Path,
) -> None:
    """Current release collection retains, but never dispatches, performance capability."""
    release = _load_release()
    candidate = "a" * 40
    fake = FakeGh(revision=candidate)

    collected = release.collect_workflow_evidence(
        candidate_sha=candidate, output_directory=tmp_path, execute=fake
    )

    assert {item.workflow for item in collected} == {
        "quality.yml",
        "live_qualification.yml",
    }
    assert "performance.yml" in release.WORKFLOW_SPECS
    assert "controlled-performance-envelope" in release.ARTIFACT_EVIDENCE_CLASS
    assert all(
        command[3] != "performance.yml"
        for command in fake.commands
        if command[:3] == ("gh", "workflow", "run")
    )


def test_aggregate_records_exact_deferred_performance_nonclaim(tmp_path: Path) -> None:
    """The current aggregate is complete only with its immutable QUAL-06 nonclaim."""
    release = _load_release()
    candidate = "a" * 40
    collected = release.collect_workflow_evidence(
        candidate_sha=candidate,
        output_directory=tmp_path / "collected",
        execute=FakeGh(revision=candidate),
    )
    release.write_collection_manifest(tmp_path / "collected", collected)

    aggregate = release.aggregate_collection(
        candidate_sha=candidate,
        collection_directory=tmp_path / "collected",
        output=tmp_path / "aggregate.json",
    )

    assert set(aggregate["evidence"]) == set(
        release.CURRENT_ARTIFACT_EVIDENCE_CLASS.values()
    )
    assert aggregate["deferred_requirements"] == [
        {
            "decision": "D-23",
            "qualification": "NOT_QUALIFIED",
            "requirement": "QUAL-06",
            "seed": ".planning/seeds/SEED-006-qualify-controlled-linux-performance.md",
            "status": "DEFERRED",
        }
    ]


def test_local_readiness_rejects_remote_or_publication_substitution() -> None:
    """A local record has no route to qualify live services or publication."""
    release = _load_release()
    record = release.build_local_readiness(
        revision="a" * 40,
        source_digest="b" * 64,
        observed_host={"machine": "arm64", "os": "Darwin", "python": "3.13.0"},
        evidence={
            evidence_class: {
                "result": "passed",
                "revision": "a" * 40,
                "source_digest": "b" * 64,
                "status": "PASS",
            }
            for evidence_class in ("deterministic", "coverage", "structural")
        }
        | {
            "base_wheel": {
                "probes": list(release.BASE_WHEEL_PROBES),
                "revision": "a" * 40,
                "source_digest": "b" * 64,
                "status": "PASS",
                "wheel_sha256": "c" * 64,
            }
        },
    )

    release.validate_local_readiness(record, revision="a" * 40, source_digest="b" * 64)
    assert record["deferred_requirements"] == [
        release.DEFERRED_PERFORMANCE_RECORD,
        release.DEFERRED_LIVE_RECORD,
    ]
    assert record["publication"] == release.DEFERRED_PUBLICATION_RECORD

    record["evidence"]["live_services"] = {"status": "QUALIFIED"}
    with pytest.raises(release.ReleaseEvidenceError, match="unexpected shape"):
        release.validate_local_readiness(
            record, revision="a" * 40, source_digest="b" * 64
        )


def test_deferred_performance_artifacts_and_macos_diagnostics_are_rejected(
    tmp_path: Path,
) -> None:
    """No timing artifact or altered deferral can enter the current release record."""
    release = _load_release()
    candidate = "a" * 40
    collection = tmp_path / "collected"
    collected = release.collect_workflow_evidence(
        candidate_sha=candidate,
        output_directory=collection,
        execute=FakeGh(revision=candidate),
    )
    release.write_collection_manifest(collection, collected)
    manifest_path = collection / release.COLLECTION_MANIFEST_NAME
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_artifacts = list(document["artifacts"])
    for forbidden_name in (
        "controlled-performance-envelope",
        "macos-performance-diagnostic.json",
    ):
        document["artifacts"] = original_artifacts + [
            {
                **original_artifacts[0],
                "artifact_name": forbidden_name,
                "evidence_class": "controlled_performance",
            }
        ]
        manifest_path.write_text(json.dumps(document), encoding="utf-8")

        with pytest.raises(release.ReleaseEvidenceError, match="exactly one artifact"):
            release.aggregate_collection(
                candidate_sha=candidate,
                collection_directory=collection,
                output=tmp_path / f"{forbidden_name}.json",
            )

    aggregate = {
        "schema": "cacheness-phase8-release-qualification-v2",
        "revision": candidate,
        "source_digest": "c" * 64,
        "evidence": {
            evidence_class: {"artifact_name": artifact_name}
            for artifact_name, evidence_class in release.CURRENT_ARTIFACT_EVIDENCE_CLASS.items()
        },
        "deferred_requirements": [
            {
                "decision": "D-23",
                "qualification": "NOT_QUALIFIED",
                "requirement": "QUAL-06",
                "seed": ".planning/seeds/SEED-006-qualify-controlled-linux-performance.md",
                "status": "QUALIFIED",
            }
        ],
    }
    with pytest.raises(release.ReleaseEvidenceError, match="deferred requirement"):
        release.validate_release_aggregate(aggregate, candidate_sha=candidate)


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


def _publication_inputs(tmp_path: Path):
    """Create one complete local non-deferred aggregate and its exact assets."""
    release = _load_release()
    candidate = "a" * 40
    asset_directory = tmp_path / "evidence"
    asset_directory.mkdir()
    evidence: dict[str, dict[str, object]] = {}
    for evidence_class in sorted(release.CURRENT_RELEASE_EVIDENCE_CLASSES):
        path = asset_directory / f"{evidence_class}.json"
        raw = _envelope_bytes(evidence_class, candidate)
        path.write_bytes(raw)
        evidence[evidence_class] = {
            "workflow": "quality.yml",
            "run_id": 1001,
            "artifact_name": f"phase8-{evidence_class}-envelope",
            "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        }
    aggregate = {
        "schema": release.RELEASE_MANIFEST_SCHEMA,
        "revision": candidate,
        "source_digest": "c" * 64,
        "evidence": evidence,
        "deferred_requirements": [dict(release.DEFERRED_PERFORMANCE_RECORD)],
    }
    aggregate_path = tmp_path / "release_qualification.json"
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
    return release, candidate, aggregate_path, asset_directory


class FakePublicationGh:
    """Mutable release seam with exact uploaded-state and digest reporting."""

    def __init__(
        self,
        *,
        candidate: str,
        mismatch_after_publish: bool = False,
        immutable_after_publish: bool = True,
    ) -> None:
        self.candidate = candidate
        self.mismatch_after_publish = mismatch_after_publish
        self.immutable_after_publish = immutable_after_publish
        self.commands: list[tuple[str, ...]] = []
        self.assets: list[dict[str, object]] = []
        self.created = False
        self.published = False

    def __call__(self, command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        if command[:3] == ("git", "rev-parse", "--verify"):
            return subprocess.CompletedProcess(command, 0, self.candidate + "\n", "")
        if command[:3] == ("gh", "auth", "status"):
            return subprocess.CompletedProcess(command, 0, "authenticated\n", "")
        if command[:3] == ("gh", "release", "view"):
            if not self.created:
                return subprocess.CompletedProcess(command, 1, "", "release not found")
            assets = list(self.assets)
            if self.published and self.mismatch_after_publish:
                assets.append(
                    {
                        "name": "macos-performance-diagnostic.json",
                        "state": "uploaded",
                        "digest": "sha256:" + "d" * 64,
                    }
                )
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "tagName": command[3],
                        "isDraft": not self.published,
                        "isImmutable": self.published and self.immutable_after_publish,
                        "assets": assets,
                    }
                ),
                "",
            )
        if command[:3] == ("gh", "release", "create"):
            self.created = True
            local_assets = [Path(value) for value in command[command.index("--") + 1 :]]
            self.assets = [
                {
                    "name": path.name,
                    "state": "uploaded",
                    "digest": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in local_assets
            ]
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[:3] == ("gh", "release", "edit"):
            self.published = True
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[:3] in {
            ("gh", "release", "verify"),
            ("gh", "release", "verify-asset"),
        }:
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(f"unexpected command: {command!r}")


def test_publication_preflight_fails_before_mutation(tmp_path: Path) -> None:
    """A bad deferral, evidence, tag, or permission stops before draft creation."""
    release, candidate, aggregate, asset_directory = _publication_inputs(tmp_path)
    fake = FakePublicationGh(candidate=candidate)

    aggregate_value = json.loads(aggregate.read_text(encoding="utf-8"))
    aggregate_value["deferred_requirements"][0]["status"] = "QUALIFIED"
    aggregate.write_text(json.dumps(aggregate_value), encoding="utf-8")

    with pytest.raises(release.ReleaseEvidenceError, match="deferred requirement"):
        release.publication_preflight(
            tag="v1.0.0",
            aggregate_path=aggregate,
            asset_directory=asset_directory,
            execute=fake,
        )

    assert not fake.created
    assert not any(
        command[:3] == ("gh", "release", "create") for command in fake.commands
    )


def test_prepare_draft_uploads_exact_non_deferred_asset_set(tmp_path: Path) -> None:
    """Draft upload includes the aggregate and exact current assets, never diagnostics."""
    release, candidate, aggregate, asset_directory = _publication_inputs(tmp_path)
    fake = FakePublicationGh(candidate=candidate)

    report = release.prepare_draft(
        tag="v1.0.0",
        aggregate_path=aggregate,
        asset_directory=asset_directory,
        output=tmp_path / "prepublication.json",
        execute=fake,
    )

    assert fake.created and not fake.published
    assert report["revision"] == candidate
    assert {item["name"] for item in report["assets"]} == {
        aggregate.name,
        *(f"{item}.json" for item in release.CURRENT_RELEASE_EVIDENCE_CLASSES),
    }
    assert all("performance" not in item["name"] for item in report["assets"])


def test_prepare_draft_verifies_uploaded_states_digests_and_content(
    tmp_path: Path,
) -> None:
    """A prepublication report follows only verified uploaded API content."""
    release, candidate, aggregate, asset_directory = _publication_inputs(tmp_path)
    fake = FakePublicationGh(candidate=candidate)

    report = release.prepare_draft(
        tag="v1.0.0",
        aggregate_path=aggregate,
        asset_directory=asset_directory,
        output=tmp_path / "prepublication.json",
        execute=fake,
    )

    assert report["report_sha256"] == release.prepublication_report_digest(report)
    assert all(item["state"] == "uploaded" for item in report["assets"])
    assert all(item["sha256"].startswith("sha256:") for item in report["assets"])


def test_publish_requires_exact_approved_prepublication_digest(tmp_path: Path) -> None:
    """A stale or altered report cannot authorize the one-way draft transition."""
    release, candidate, aggregate, asset_directory = _publication_inputs(tmp_path)
    fake = FakePublicationGh(candidate=candidate)
    prepublication = tmp_path / "prepublication.json"
    report = release.prepare_draft(
        tag="v1.0.0",
        aggregate_path=aggregate,
        asset_directory=asset_directory,
        output=prepublication,
        execute=fake,
    )

    with pytest.raises(release.ReleaseEvidenceError, match="approved prepublication"):
        release.publish_and_verify(
            tag="v1.0.0",
            prepublication=prepublication,
            approved_report_sha256="0" * 64,
            output=tmp_path / "publication.json",
            execute=fake,
        )
    assert not fake.published

    result = release.publish_and_verify(
        tag="v1.0.0",
        prepublication=prepublication,
        approved_report_sha256=report["report_sha256"],
        output=tmp_path / "publication.json",
        execute=fake,
    )
    assert result["revision"] == candidate


def test_publish_and_verify_requires_immutable_exact_remote_state(
    tmp_path: Path,
) -> None:
    """A published mutable release is not equivalent to immutable qualification."""
    release, candidate, aggregate, asset_directory = _publication_inputs(tmp_path)
    fake = FakePublicationGh(candidate=candidate, immutable_after_publish=False)
    prepublication = tmp_path / "prepublication.json"
    report = release.prepare_draft(
        tag="v1.0.0",
        aggregate_path=aggregate,
        asset_directory=asset_directory,
        output=prepublication,
        execute=fake,
    )

    with pytest.raises(release.ReleaseEvidenceError, match="immutable"):
        release.publish_and_verify(
            tag="v1.0.0",
            prepublication=prepublication,
            approved_report_sha256=report["report_sha256"],
            output=tmp_path / "publication.json",
            execute=fake,
        )


def test_postpublication_mismatch_records_unqualified_incident(tmp_path: Path) -> None:
    """A remote mismatch records a nonrepairing incident instead of a false pass."""
    release, candidate, aggregate, asset_directory = _publication_inputs(tmp_path)
    fake = FakePublicationGh(candidate=candidate, mismatch_after_publish=True)
    prepublication = tmp_path / "prepublication.json"
    output = tmp_path / "publication.json"
    report = release.prepare_draft(
        tag="v1.0.0",
        aggregate_path=aggregate,
        asset_directory=asset_directory,
        output=prepublication,
        execute=fake,
    )

    with pytest.raises(release.ReleaseEvidenceError, match="asset inventory"):
        release.publish_and_verify(
            tag="v1.0.0",
            prepublication=prepublication,
            approved_report_sha256=report["report_sha256"],
            output=output,
            execute=fake,
        )

    incident = json.loads(output.read_text(encoding="utf-8"))
    assert incident["status"] == "NOT_QUALIFIED"
    assert incident["revision"] == candidate
