"""Contract tests for the Phase 8 controlled-performance harness."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

_BENCHMARKS_DIR = Path(__file__).parents[2] / "benchmarks"
sys.path.insert(0, str(_BENCHMARKS_DIR))

_WORKLOADS = importlib.import_module("phase8_workloads")
LAYERS = _WORKLOADS.LAYERS
access_labels = _WORKLOADS.access_labels
build_layer_workload = _WORKLOADS.build_layer_workload
reviewed_workloads = _WORKLOADS.reviewed_workloads

_BENCHMARKS = importlib.import_module("phase8_benchmarks")
BaselineVerificationError = _BENCHMARKS.BaselineVerificationError
HASH_SIZES = _BENCHMARKS.HASH_SIZES
atomic_replace_baseline = _BENCHMARKS.atomic_replace_baseline
build_baseline_document = _BENCHMARKS.build_baseline_document
distribution = _BENCHMARKS.distribution
measure_hashes = _BENCHMARKS.measure_hashes
verify_baseline = _BENCHMARKS.verify_baseline


def _eligible_preflight_fingerprint() -> dict[str, object]:
    """Return a schema-valid, deliberately non-identifying runner fingerprint."""
    return {
        "os": {"system": "Linux", "release": "6.8.0", "machine": "x86_64"},
        "cpu": {
            "model": "Generic x86 CPU",
            "logical_cpus": 8,
            "scaling_governors": ["performance"],
        },
        "filesystem": {"type": "ext4"},
        "python": {"implementation": "CPython", "version": "3.13.0"},
        "uv": {"version": "0.9.0"},
        "sqlite": {"library_version": "3.50.0"},
    }


def _preflight_arguments(*, revision: str = "a" * 40) -> list[str]:
    return [
        "--preflight-runner",
        "--expect-label",
        "cacheness-perf-linux-x64",
        "--revision",
        revision,
        "--runner-identity",
        "cacheness-perf-linux-x64",
    ]


def _install_eligible_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    """Install deterministic read-only probes for CLI contract tests."""
    fingerprint = _eligible_preflight_fingerprint()
    monkeypatch.setattr(_BENCHMARKS, "_machine_fingerprint", lambda: fingerprint)
    monkeypatch.setattr(
        _BENCHMARKS,
        "_repository_preflight",
        lambda revision: {"head_state": "detached", "worktree_state": "clean"},
    )
    monkeypatch.setattr(_BENCHMARKS, "_validate_controlled_platform", lambda: None)
    return fingerprint


def _tree_snapshot(root: Path) -> dict[str, tuple[int, int, str]]:
    """Capture the bounded Git/worktree surface used by the preflight contract."""
    tracked = [root / "benchmarks" / "phase8_benchmarks.py"]
    git_path = root / ".git"
    if git_path.is_file():
        tracked.append(git_path)
    else:
        tracked.extend(
            path
            for path in (
                git_path / "HEAD",
                git_path / "index",
                git_path / "config",
                git_path / "packed-refs",
            )
            if path.exists()
        )
    snapshot: dict[str, tuple[int, int, str]] = {}
    for path in tracked:
        if path.is_file():
            stat = path.stat()
            snapshot[str(path)] = (
                stat.st_mode,
                stat.st_mtime_ns,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
    return snapshot


def test_workloads_inventory_is_representative_without_topology_cross_product() -> None:
    """Each reviewed tier occurs once and names its intended topology."""
    workloads = reviewed_workloads()

    assert {workload.name for workload in workloads} == {
        "small-generic-object",
        "numpy-npz-medium",
        "numpy-npz-large",
        "pandas-parquet",
        "polars-parquet",
    }
    assert {workload.layer for workload in workloads} == set(LAYERS)
    assert {workload.access for workload in workloads} == set(access_labels())
    assert {workload.topology for workload in workloads} == {
        "memory-memory",
        "sqlite-filesystem",
    }

    canonical_sizes = {workload.name: workload.canonical_size for workload in workloads}
    canonical_units = {workload.name: workload.size_unit for workload in workloads}
    assert canonical_sizes["small-generic-object"] == 4 * 1024
    assert canonical_sizes["numpy-npz-medium"] == 16 * 1024 * 1024
    assert canonical_sizes["numpy-npz-large"] == 128 * 1024 * 1024
    assert canonical_sizes["pandas-parquet"] == 100_000
    assert canonical_sizes["polars-parquet"] == 100_000
    assert canonical_units["numpy-npz-large"] == "bytes"
    assert canonical_units["pandas-parquet"] == "rows"
    assert all("blosc2" not in workload.name for workload in workloads)


def test_reduced_workloads_use_current_public_composition() -> None:
    """Reduced fixtures remain deterministic and exercise public layers only."""
    descriptor = next(
        workload
        for workload in reviewed_workloads()
        if workload.name == "small-generic-object"
        and workload.layer == "blobstore"
        and workload.access == "cold-put"
    )

    first = build_layer_workload(descriptor, reduced=True)
    second = build_layer_workload(descriptor, reduced=True)
    try:
        assert first.fixture_digest == second.fixture_digest
        assert first.fixture_digest == hashlib.sha256(first.fixture_bytes).hexdigest()
        assert first.layer == "blobstore"
        assert first.uses_public_composition is True
        assert first.timed_callback() is not None
    finally:
        first.close()
        second.close()


def test_workloads_layers_and_access_labels_remain_explicit() -> None:
    """Cold and warm operations are not conflated in one timing sample."""
    assert LAYERS == ("handler", "blobstore", "unified-cache")
    assert access_labels() == ("cold-put", "warm-get")


def _baseline_document() -> dict[str, object]:
    return build_baseline_document(
        revision="a" * 40,
        source_digest="b" * 64,
        runner_identity="cacheness-perf-linux-x64",
        environment={"platform": "Linux", "machine": "x86_64"},
        distributions={
            "small-generic-object__blobstore__cold-put": distribution(
                [100, 110, 120, 130, 140]
            )
        },
        envelopes={
            "small-generic-object__blobstore__cold-put": {
                "median_relative_limit": 1.2,
                "tail_relative_limit": 1.3,
            }
        },
    )


def test_distribution_preserves_raw_samples_and_deterministic_tails() -> None:
    """The reviewed record keeps derivations traceable to raw nanoseconds."""
    result = distribution([1, 2, 3, 4, 5])

    assert result["samples_ns"] == [1, 2, 3, 4, 5]
    assert result["p50_ns"] == 3.0
    assert result["p95_ns"] == 4.8
    assert result["p99_ns"] == 4.96


def test_hash_measurements_cover_canonical_sizes_without_changing_sha256() -> None:
    """XXH3 is comparative evidence while SHA-256 remains the stored digest."""
    assert HASH_SIZES == (
        4 * 1024,
        1 * 1024 * 1024,
        16 * 1024 * 1024,
        128 * 1024 * 1024,
    )

    observations = measure_hashes(
        size_bytes=4 * 1024,
        lifecycle_distribution=distribution([1_000, 1_200, 1_400]),
        loops=2,
    )

    assert {observation["algorithm"] for observation in observations} == {
        "sha256",
        "xxh3_64",
    }
    assert all(
        observation["throughput_bytes_per_second"] > 0 for observation in observations
    )
    assert all(0 < observation["lifecycle_share_p50"] for observation in observations)
    assert _BENCHMARKS.PAYLOAD_DIGEST_ALGORITHM == "sha256"


def test_verify_baseline_rejects_mismatched_identity_revision_environment_and_regression() -> (
    None
):
    """Only one exact controlled environment can qualify its reviewed envelope."""
    baseline = _baseline_document()
    current = _baseline_document()
    assert verify_baseline(baseline, current) == []

    mismatched_runner = _baseline_document()
    mismatched_runner["runner_identity"] = "ordinary-linux"
    with pytest.raises(BaselineVerificationError, match="runner identity"):
        verify_baseline(baseline, mismatched_runner)

    stale_revision = _baseline_document()
    stale_revision["revision"] = "c" * 40
    with pytest.raises(BaselineVerificationError, match="revision"):
        verify_baseline(baseline, stale_revision)

    environment_drift = _baseline_document()
    environment_drift["environment"] = {"platform": "Linux", "machine": "arm64"}
    with pytest.raises(BaselineVerificationError, match="environment"):
        verify_baseline(baseline, environment_drift)

    regressed = _baseline_document()
    record = regressed["distributions"]["small-generic-object__blobstore__cold-put"]
    record["samples_ns"] = [200, 220, 240, 260, 280]
    record.update(distribution(record["samples_ns"]))
    with pytest.raises(BaselineVerificationError, match="median"):
        verify_baseline(baseline, regressed)


def test_baseline_capture_and_recalibration_are_explicit_and_atomic(
    tmp_path: Path,
) -> None:
    """Verify reads evidence; reviewed capture/recalibration own all mutations."""
    destination = tmp_path / "phase8_baseline.json"
    baseline = _baseline_document()

    atomic_replace_baseline(destination, baseline, mode="capture")
    original = destination.read_bytes()
    assert verify_baseline(destination, baseline) == []
    assert destination.read_bytes() == original

    with pytest.raises(BaselineVerificationError, match="already exists"):
        atomic_replace_baseline(destination, baseline, mode="capture")
    with pytest.raises(BaselineVerificationError, match="justification"):
        atomic_replace_baseline(destination, baseline, mode="recalibrate")

    atomic_replace_baseline(
        destination,
        baseline,
        mode="recalibrate",
        justification="controlled runner kernel update",
    )
    recalibrated = json.loads(destination.read_text(encoding="utf-8"))
    assert recalibrated["baseline_change"] == {
        "mode": "recalibrate",
        "justification": "controlled runner kernel update",
    }


def test_controlled_workflow_requires_exact_sha_and_named_linux_runner() -> None:
    """Only a detached exact SHA on the reviewed runner can block performance."""
    workflow = (
        Path(__file__).parents[2] / ".github" / "workflows" / "performance.yml"
    ).read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert re.search(
        r"candidate_sha:\s*\n\s*description:.*\n\s*required: true", workflow
    )
    assert "cacheness-perf-linux-x64" in workflow
    assert "environment: controlled-performance" in workflow
    assert "ref: ${{ inputs.candidate_sha }}" in workflow
    assert 'ACTUAL_SHA="$(git rev-parse HEAD)"' in workflow
    assert '"$ACTUAL_SHA" = "$CANDIDATE_SHA"' in workflow
    assert "--verify-baseline benchmarks/phase8_baseline.json" in workflow
    assert "--all-workloads" in workflow
    assert "--runner-identity cacheness-perf-linux-x64" in workflow
    assert "remote PostgreSQL/S3 timing" in workflow
    assert "ordinary macOS timing" in workflow


def test_controlled_workflow_pins_actions_and_uploads_one_sanitized_envelope() -> None:
    """The qualifying artifact has a fixed name and bounded diagnostic retention."""
    workflow = (
        Path(__file__).parents[2] / ".github" / "workflows" / "performance.yml"
    ).read_text(encoding="utf-8")

    action_references = re.findall(r"uses:\s+[^@\s]+@([^\s]+)", workflow)
    assert action_references
    assert all(
        re.fullmatch(r"[0-9a-f]{40}", reference) for reference in action_references
    )
    assert "name: controlled-performance-envelope" in workflow
    assert "retention-days: 30" in workflow
    assert "build/phase8/controlled-performance.json" in workflow
    assert "build/phase8/raw-performance.json" in workflow


def test_preflight_runner_emits_only_the_bounded_eligibility_record(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A successful preflight is identity evidence, never a timing document."""
    fingerprint = _install_eligible_preflight(monkeypatch)

    assert _BENCHMARKS.main(_preflight_arguments()) == 0

    record = json.loads(capsys.readouterr().out)
    assert set(record) == {
        "schema",
        "status",
        "revision",
        "runner_label",
        "repository",
        "machine_fingerprint",
        "machine_fingerprint_sha256",
    }
    assert record["schema"] == "cacheness.phase8.runner-preflight.v1"
    assert record["status"] == "eligible"
    assert record["revision"] == "a" * 40
    assert record["runner_label"] == "cacheness-perf-linux-x64"
    assert record["repository"] == {
        "head_state": "detached",
        "worktree_state": "clean",
    }
    assert record["machine_fingerprint"] == fingerprint
    assert record[
        "machine_fingerprint_sha256"
    ] == _BENCHMARKS.machine_fingerprint_digest(fingerprint)
    assert _BENCHMARKS.validate_preflight_record(record) == record


def test_preflight_runner_canonical_fingerprint_rejects_drift_and_disclosure() -> None:
    """Every allowed family contributes to a canonical bounded digest only."""
    fingerprint = _eligible_preflight_fingerprint()
    expected = _BENCHMARKS.machine_fingerprint_digest(fingerprint)
    reordered = {
        "sqlite": fingerprint["sqlite"],
        "uv": fingerprint["uv"],
        "python": fingerprint["python"],
        "filesystem": fingerprint["filesystem"],
        "cpu": fingerprint["cpu"],
        "os": fingerprint["os"],
    }
    assert _BENCHMARKS.machine_fingerprint_digest(reordered) == expected

    drifted_fingerprints = []
    for family, replacement in (
        ("os", {"system": "Linux", "release": "6.9.0", "machine": "x86_64"}),
        (
            "cpu",
            {
                "model": "Generic x86 CPU v2",
                "logical_cpus": 8,
                "scaling_governors": ["performance"],
            },
        ),
        ("filesystem", {"type": "xfs"}),
        ("python", {"implementation": "CPython", "version": "3.13.1"}),
        ("uv", {"version": "0.9.1"}),
        ("sqlite", {"library_version": "3.50.1"}),
    ):
        changed = deepcopy(fingerprint)
        changed[family] = replacement
        drifted_fingerprints.append(changed)
    assert all(
        _BENCHMARKS.machine_fingerprint_digest(changed) != expected
        for changed in drifted_fingerprints
    )

    malformed = deepcopy(fingerprint)
    malformed["hostname"] = "never-allowed"
    with pytest.raises(BaselineVerificationError, match="machine fingerprint keys"):
        _BENCHMARKS.machine_fingerprint_digest(malformed)

    malformed = deepcopy(fingerprint)
    del malformed["uv"]
    with pytest.raises(BaselineVerificationError, match="machine fingerprint keys"):
        _BENCHMARKS.machine_fingerprint_digest(malformed)

    malformed = deepcopy(fingerprint)
    malformed["os"] = {"system": "Linux", "release": "\u2603", "machine": "x86_64"}
    with pytest.raises(BaselineVerificationError, match="release"):
        _BENCHMARKS.machine_fingerprint_digest(malformed)

    malformed = deepcopy(fingerprint)
    malformed["cpu"]["logical_cpus"] = "8"
    with pytest.raises(BaselineVerificationError, match="logical CPU count"):
        _BENCHMARKS.machine_fingerprint_digest(malformed)

    malformed = deepcopy(fingerprint)
    malformed["cpu"]["scaling_governors"] = ["performance", "performance"]
    with pytest.raises(BaselineVerificationError, match="scaling governors"):
        _BENCHMARKS.machine_fingerprint_digest(malformed)

    malformed = deepcopy(fingerprint)
    malformed["cpu"]["model"] = "x" * 257
    with pytest.raises(BaselineVerificationError, match="CPU model"):
        _BENCHMARKS.machine_fingerprint_digest(malformed)

    record = {
        "schema": "cacheness.phase8.runner-preflight.v1",
        "status": "eligible",
        "revision": "a" * 40,
        "runner_label": "cacheness-perf-linux-x64",
        "repository": {"head_state": "detached", "worktree_state": "clean"},
        "machine_fingerprint": fingerprint,
        "machine_fingerprint_sha256": expected,
    }
    hostile = deepcopy(record)
    hostile["environment"] = {"TOKEN": "never-allowed"}
    with pytest.raises(BaselineVerificationError, match="preflight record keys"):
        _BENCHMARKS.validate_preflight_record(hostile)


def test_preflight_runner_uses_optional_lock_free_git_reads_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Git source checks are bounded reads and never surface command diagnostics."""
    calls: list[tuple[str, ...]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(tuple(command))
        assert kwargs["check"] is False
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["timeout"] == _BENCHMARKS.PREFLIGHT_GIT_TIMEOUT_SECONDS
        if command[-2:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="a" * 40 + "\n", stderr="")
        if command[-3:] == ["symbolic-ref", "-q", "HEAD"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        assert command[-3:] == ["status", "--untracked-files=all", "--porcelain=v1"]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(_BENCHMARKS.subprocess, "run", fake_run)

    assert _BENCHMARKS._repository_preflight("a" * 40) == {
        "head_state": "detached",
        "worktree_state": "clean",
    }
    assert len(calls) == 3
    assert all("--no-optional-locks" in command for command in calls)
    assert all("-C" in command for command in calls)


@pytest.mark.parametrize(
    ("head", "symbolic_ref", "status", "reason"),
    [
        ("b" * 40, (1, ""), "", "repository revision"),
        ("a" * 40, (0, "refs/heads/main\n"), "", "repository head"),
        ("a" * 40, (1, ""), " M benchmarks/phase8_benchmarks.py\n", "repository state"),
        ("a" * 40, (1, ""), "?? unexpected-file\n", "repository state"),
    ],
)
def test_preflight_runner_rejects_non_clean_or_attached_repository_state(
    monkeypatch: pytest.MonkeyPatch,
    head: str,
    symbolic_ref: tuple[int, str],
    status: str,
    reason: str,
) -> None:
    """Exact source proof rejects wrong HEAD, branches, tracked, and untracked state."""

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        del kwargs
        if command[-2:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout=f"{head}\n", stderr="")
        if command[-3:] == ["symbolic-ref", "-q", "HEAD"]:
            return SimpleNamespace(
                returncode=symbolic_ref[0], stdout=symbolic_ref[1], stderr=""
            )
        return SimpleNamespace(returncode=0, stdout=status, stderr="")

    monkeypatch.setattr(_BENCHMARKS.subprocess, "run", fake_run)

    with pytest.raises(BaselineVerificationError, match=reason):
        _BENCHMARKS._repository_preflight("a" * 40)


@pytest.mark.parametrize(
    ("system", "machine"), [("Darwin", "arm64"), ("Linux", "arm64")]
)
def test_preflight_runner_rejects_nonqualifying_platforms(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    system: str,
    machine: str,
) -> None:
    """macOS and non-x86 Linux are explicit nonqualifying runner boundaries."""
    monkeypatch.setattr(
        _BENCHMARKS, "_machine_fingerprint", _eligible_preflight_fingerprint
    )
    monkeypatch.setattr(
        _BENCHMARKS,
        "_repository_preflight",
        lambda revision: {"head_state": "detached", "worktree_state": "clean"},
    )
    monkeypatch.setattr(_BENCHMARKS.platform, "system", lambda: system)
    monkeypatch.setattr(_BENCHMARKS.platform, "machine", lambda: machine)

    assert _BENCHMARKS.main(_preflight_arguments()) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == (
        "runner preflight rejected: controlled runner platform is not eligible"
    )


def test_preflight_runner_never_measures_or_mutates(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Both accepted and rejected runner checks leave repository state untouched."""
    _install_eligible_preflight(monkeypatch)
    before = _tree_snapshot(Path(__file__).parents[2])

    def fail_if_called(*args: object, **kwargs: object) -> object:
        raise AssertionError("preflight must not enter a measurement or write seam")

    monkeypatch.setattr(_BENCHMARKS, "_measurement_document", fail_if_called)
    monkeypatch.setattr(_BENCHMARKS, "atomic_replace_baseline", fail_if_called)
    monkeypatch.setattr(_BENCHMARKS, "verify_baseline", fail_if_called)

    assert _BENCHMARKS.main(_preflight_arguments()) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "eligible"
    assert _tree_snapshot(Path(__file__).parents[2]) == before

    monkeypatch.setattr(
        _BENCHMARKS,
        "_repository_preflight",
        lambda revision: (_ for _ in ()).throw(
            BaselineVerificationError("repository state is not eligible")
        ),
    )
    assert _BENCHMARKS.main(_preflight_arguments()) == 2
    assert capsys.readouterr().out == ""
    assert _tree_snapshot(Path(__file__).parents[2]) == before


@pytest.mark.parametrize(
    ("arguments", "reason"),
    [
        (_preflight_arguments(revision="A" * 40), "requested revision is invalid"),
        (
            [
                "--preflight-runner",
                "--expect-label",
                "ordinary-linux",
                "--revision",
                "a" * 40,
                "--runner-identity",
                "ordinary-linux",
            ],
            "runner label is not eligible",
        ),
        (
            [
                "--preflight-runner",
                "--expect-label",
                "cacheness-perf-linux-x64",
                "--revision",
                "a" * 40,
            ],
            "runner label is not eligible",
        ),
    ],
)
def test_preflight_runner_rejects_uncontrolled_input_without_success_record(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    reason: str,
) -> None:
    """Malformed revision and any non-fixed runner label fail with safe diagnostics."""
    _install_eligible_preflight(monkeypatch)

    assert _BENCHMARKS.main(arguments) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == f"runner preflight rejected: {reason}"
