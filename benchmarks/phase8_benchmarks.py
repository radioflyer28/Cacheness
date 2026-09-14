#!/usr/bin/env python3
"""Controlled, distribution-based Phase 8 performance evidence harness.

This module records benchmark evidence; it does not configure runtime timeouts
or alter cache, lifecycle, or manifest behavior.  A reviewed baseline is a
controlled-Linux release gate only.  Ordinary local and remote measurements are
diagnostic evidence and must never be interpreted as progress guarantees.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

import xxhash

from cacheness.storage.manifest import PAYLOAD_DIGEST_ALGORITHM

from phase8_workloads import LayerWorkload, WorkloadDescriptor, reviewed_workloads


BENCHMARK_SCHEMA_VERSION = 1
BENCHMARK_SCHEMA = f"cacheness-phase8-performance-v{BENCHMARK_SCHEMA_VERSION}"
CONTROLLED_RUNNER_IDENTITY = "cacheness-perf-linux-x64"
HASH_SIZES = (4 * 1024, 1 * 1024 * 1024, 16 * 1024 * 1024, 128 * 1024 * 1024)
DEFAULT_WARMUPS = 2
DEFAULT_SAMPLES = 9
MAX_CHILD_TIMEOUT_SECONDS = 300
_RELEVANT_SOURCE_PATHS = (
    "benchmarks/phase8_workloads.py",
    "benchmarks/phase8_benchmarks.py",
    "src/cacheness/storage/manifest.py",
    "src/cacheness/storage/blob_store.py",
    "src/cacheness/core.py",
    ".github/workflows/performance.yml",
)


class BaselineVerificationError(RuntimeError):
    """Raised for malformed, stale, or regressed controlled evidence."""


def _percentile(samples: Sequence[int], percentile: float) -> float:
    """Return the deterministic linear percentile used in all envelope checks."""
    if not samples:
        raise BaselineVerificationError("distribution requires at least one sample")
    ordered = sorted(float(sample) for sample in samples)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * fraction)


def distribution(samples: Sequence[int]) -> dict[str, Any]:
    """Preserve raw nanosecond samples with deterministic median and tails."""
    normalized = list(samples)
    if not normalized or any(type(item) is not int or item <= 0 for item in normalized):
        raise BaselineVerificationError("distribution samples must be positive integers")
    return {
        "samples_ns": normalized,
        "sample_count": len(normalized),
        "mean_ns": statistics.fmean(normalized),
        "p50_ns": _percentile(normalized, 0.50),
        "p95_ns": _percentile(normalized, 0.95),
        "p99_ns": _percentile(normalized, 0.99),
    }


def _git_revision() -> str:
    """Return the exact Git revision being measured, failing closed if absent."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise BaselineVerificationError("unable to determine benchmark source revision") from error
    revision = result.stdout.strip()
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise BaselineVerificationError("benchmark source revision must be a 40-character SHA")
    return revision


def relevant_source_digest(
    repository_root: Path | None = None, *, require_complete: bool = True
) -> str:
    """Hash the reviewed benchmark and measured-architecture source inventory."""
    root = repository_root or Path(__file__).parents[1]
    digest = hashlib.sha256()
    for relative_path in _RELEVANT_SOURCE_PATHS:
        source_path = root / relative_path
        if not source_path.is_file() and require_complete:
            raise BaselineVerificationError(
                f"relevant benchmark source is missing: {relative_path}"
            )
        if not source_path.is_file():
            continue
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def environment_fingerprint() -> dict[str, str | int]:
    """Capture stable, non-secret machine facts needed for envelope comparison."""
    filesystem = os.statvfs(Path.cwd())
    return {
        "implementation": platform.python_implementation(),
        "machine": platform.machine().lower(),
        "platform": platform.system(),
        "python": platform.python_version(),
        "python_implementation": sys.implementation.name,
        "filesystem_block_size": filesystem.f_bsize,
        "filesystem_name_max": filesystem.f_namemax,
    }


def _validate_hex(value: object, *, name: str, length: int) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise BaselineVerificationError(f"{name} must be a {length}-character hexadecimal string")
    if any(character not in "0123456789abcdef" for character in value):
        raise BaselineVerificationError(f"{name} must be a {length}-character hexadecimal string")
    return value


def _validate_environment(value: object) -> dict[str, str | int]:
    if not isinstance(value, Mapping) or not value:
        raise BaselineVerificationError("environment must be a non-empty mapping")
    normalized: dict[str, str | int] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise BaselineVerificationError("environment keys must be non-empty strings")
        if type(item) not in {str, int}:
            raise BaselineVerificationError("environment values must be strings or integers")
        normalized[key] = item
    return normalized


def _validate_distribution(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BaselineVerificationError(f"distribution for {name} must be a mapping")
    raw_samples = value.get("samples_ns")
    if not isinstance(raw_samples, list):
        raise BaselineVerificationError(f"distribution for {name} must retain raw samples")
    expected = distribution(raw_samples)
    for field in ("sample_count", "mean_ns", "p50_ns", "p95_ns", "p99_ns"):
        if value.get(field) != expected[field]:
            raise BaselineVerificationError(
                f"distribution for {name} has a malformed derived {field}"
            )
    return dict(expected)


def _validate_envelopes(
    value: object, distributions: Mapping[str, object]
) -> dict[str, dict[str, float]]:
    if not isinstance(value, Mapping) or set(value) != set(distributions):
        raise BaselineVerificationError("reviewed envelopes must match benchmark distributions")
    normalized: dict[str, dict[str, float]] = {}
    for name, envelope in value.items():
        if not isinstance(name, str) or not isinstance(envelope, Mapping):
            raise BaselineVerificationError("reviewed envelopes must be named mappings")
        median = envelope.get("median_relative_limit")
        tail = envelope.get("tail_relative_limit")
        if (
            type(median) not in {int, float}
            or type(tail) not in {int, float}
            or not math.isfinite(float(median))
            or not math.isfinite(float(tail))
            or float(median) < 1.0
            or float(tail) < 1.0
        ):
            raise BaselineVerificationError(
                f"reviewed envelope for {name} must contain finite relative limits >= 1"
            )
        normalized[name] = {
            "median_relative_limit": float(median),
            "tail_relative_limit": float(tail),
        }
    return normalized


def _validate_document(value: object) -> dict[str, Any]:
    """Validate a complete controlled evidence record before comparing it."""
    if not isinstance(value, Mapping):
        raise BaselineVerificationError("benchmark document must be a mapping")
    if value.get("schema") != BENCHMARK_SCHEMA:
        raise BaselineVerificationError("benchmark document schema is unsupported")
    if value.get("evidence_class") != "controlled-performance":
        raise BaselineVerificationError("benchmark document is not controlled-performance evidence")
    runner_identity = value.get("runner_identity")
    if not isinstance(runner_identity, str) or not runner_identity:
        raise BaselineVerificationError("benchmark runner identity is missing")
    distributions = value.get("distributions")
    if not isinstance(distributions, Mapping) or not distributions:
        raise BaselineVerificationError("benchmark document must contain distributions")
    normalized_distributions = {
        name: _validate_distribution(item, name=name)
        for name, item in distributions.items()
        if isinstance(name, str) and name
    }
    if len(normalized_distributions) != len(distributions):
        raise BaselineVerificationError("benchmark distribution names must be non-empty strings")
    normalized = {
        "schema": BENCHMARK_SCHEMA,
        "evidence_class": "controlled-performance",
        "revision": _validate_hex(value.get("revision"), name="revision", length=40),
        "source_digest": _validate_hex(
            value.get("source_digest"), name="source digest", length=64
        ),
        "runner_identity": runner_identity,
        "environment": _validate_environment(value.get("environment")),
        "distributions": normalized_distributions,
        "envelopes": _validate_envelopes(value.get("envelopes"), normalized_distributions),
    }
    baseline_change = value.get("baseline_change")
    if baseline_change is not None:
        if not isinstance(baseline_change, Mapping):
            raise BaselineVerificationError("baseline change record must be a mapping")
        mode = baseline_change.get("mode")
        if mode not in {"capture", "recalibrate"}:
            raise BaselineVerificationError("baseline change record has an invalid mode")
        change_record: dict[str, str] = {"mode": mode}
        justification = baseline_change.get("justification")
        if mode == "recalibrate":
            if not isinstance(justification, str) or not justification.strip():
                raise BaselineVerificationError(
                    "baseline recalibration record requires a justification"
                )
            change_record["justification"] = justification.strip()
        normalized["baseline_change"] = change_record
    return normalized


def build_baseline_document(
    *,
    revision: str,
    source_digest: str,
    runner_identity: str,
    environment: Mapping[str, str | int],
    distributions: Mapping[str, Mapping[str, Any]],
    envelopes: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Build and validate a reviewed controlled-performance baseline record."""
    document = {
        "schema": BENCHMARK_SCHEMA,
        "evidence_class": "controlled-performance",
        "revision": revision,
        "source_digest": source_digest,
        "runner_identity": runner_identity,
        "environment": dict(environment),
        "distributions": {name: dict(item) for name, item in distributions.items()},
        "envelopes": {name: dict(item) for name, item in envelopes.items()},
    }
    _validate_document(document)
    return document


def _load_document(source: Mapping[str, Any] | Path) -> dict[str, Any]:
    if isinstance(source, Path):
        try:
            loaded = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BaselineVerificationError("baseline is unreadable or invalid JSON") from error
        return _validate_document(loaded)
    return _validate_document(source)


def verify_baseline(
    baseline: Mapping[str, Any] | Path, current: Mapping[str, Any] | Path
) -> list[str]:
    """Compare controlled records without changing the reviewed baseline file."""
    reviewed = _load_document(baseline)
    observed = _load_document(current)
    if reviewed["runner_identity"] != observed["runner_identity"]:
        raise BaselineVerificationError("benchmark runner identity does not match baseline")
    if reviewed["revision"] != observed["revision"]:
        raise BaselineVerificationError("benchmark revision does not match baseline")
    if reviewed["source_digest"] != observed["source_digest"]:
        raise BaselineVerificationError("benchmark source digest does not match baseline")
    if reviewed["environment"] != observed["environment"]:
        raise BaselineVerificationError("benchmark environment does not match baseline")
    if set(reviewed["distributions"]) != set(observed["distributions"]):
        raise BaselineVerificationError("benchmark workload inventory does not match baseline")

    for name, baseline_distribution in reviewed["distributions"].items():
        current_distribution = observed["distributions"][name]
        envelope = reviewed["envelopes"][name]
        if current_distribution["p50_ns"] > (
            baseline_distribution["p50_ns"] * envelope["median_relative_limit"]
        ):
            raise BaselineVerificationError(f"benchmark median regressed for {name}")
        if current_distribution["p99_ns"] > (
            baseline_distribution["p99_ns"] * envelope["tail_relative_limit"]
        ):
            raise BaselineVerificationError(f"benchmark tail regressed for {name}")
    return []


def atomic_replace_baseline(
    destination: Path,
    document: Mapping[str, Any],
    *,
    mode: str,
    justification: str | None = None,
) -> None:
    """Capture or explicitly recalibrate a baseline with one atomic replacement."""
    normalized = _validate_document(document)
    if mode not in {"capture", "recalibrate"}:
        raise BaselineVerificationError("baseline mode must be capture or recalibrate")
    if mode == "capture" and destination.exists():
        raise BaselineVerificationError("baseline already exists; use recalibrate")
    if mode == "recalibrate" and (not isinstance(justification, str) or not justification.strip()):
        raise BaselineVerificationError("baseline recalibration requires a justification")
    normalized["baseline_change"] = {"mode": mode}
    if mode == "recalibrate":
        assert justification is not None
        normalized["baseline_change"]["justification"] = justification.strip()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(normalized, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, delete=False
    ) as staged:
        staged.write(payload)
        staged.flush()
        os.fsync(staged.fileno())
        staged_path = Path(staged.name)
    try:
        os.replace(staged_path, destination)
    except OSError:
        staged_path.unlink(missing_ok=True)
        raise


def _measure_callback(workload: LayerWorkload, *, loops: int) -> int:
    """Time only a prepared callback and clean every sample outside the timer."""
    workload.prepare_for_sample()
    try:
        started = time.perf_counter_ns()
        for iteration in range(loops):
            workload.timed_callback()
            if iteration + 1 < loops:
                workload.cleanup_after_sample()
                workload.prepare_for_sample()
        elapsed = time.perf_counter_ns() - started
    finally:
        workload.cleanup_after_sample()
    return max(1, elapsed // loops)


def _calibrated_loop_count(workload: LayerWorkload) -> int:
    """Bound loop calibration without adding an external benchmark dependency."""
    elapsed = _measure_callback(workload, loops=1)
    target_ns = 10_000_000
    return max(1, min(32, target_ns // max(1, elapsed)))


def measure_workload(
    descriptor: WorkloadDescriptor,
    *,
    reduced: bool,
    warmups: int,
    samples: int,
) -> dict[str, Any]:
    """Collect one isolated layer distribution through the reviewed public path."""
    if warmups < 0 or samples <= 0:
        raise ValueError("warmups must be non-negative and samples must be positive")
    workload = LayerWorkload(descriptor, reduced=reduced)
    try:
        for _ in range(warmups):
            _measure_callback(workload, loops=1)
        loops = _calibrated_loop_count(workload)
        result = distribution(
            [_measure_callback(workload, loops=loops) for _ in range(samples)]
        )
    finally:
        workload.close()
    return {
        "descriptor": {
            "access": descriptor.access,
            "format": descriptor.format_name,
            "layer": descriptor.layer,
            "size": descriptor.canonical_size,
            "size_unit": descriptor.size_unit,
            "tier": descriptor.tier,
            "topology": descriptor.topology,
        },
        "calibrated_loops": loops,
        "stability": {
            "p99_to_p50_ratio": result["p99_ns"] / result["p50_ns"],
            "raw_samples_retained": True,
        },
        "distribution": result,
    }


def _worker_command(
    descriptor: WorkloadDescriptor, *, reduced: bool, warmups: int, samples: int
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__)),
        "--worker",
        descriptor.identifier,
        "--warmups",
        str(warmups),
        "--samples",
        str(samples),
    ]
    if reduced:
        command.append("--reduced")
    return command


def _measure_in_worker(
    descriptor: WorkloadDescriptor, *, reduced: bool, warmups: int, samples: int
) -> dict[str, Any]:
    """Run one workload in a bounded child so setup state cannot leak across tiers."""
    try:
        result = subprocess.run(
            _worker_command(descriptor, reduced=reduced, warmups=warmups, samples=samples),
            check=True,
            capture_output=True,
            text=True,
            timeout=MAX_CHILD_TIMEOUT_SECONDS,
        )
        payload = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise BaselineVerificationError(
            f"benchmark worker failed for {descriptor.identifier}"
        ) from error
    if not isinstance(payload, Mapping) or payload.get("identifier") != descriptor.identifier:
        raise BaselineVerificationError("benchmark worker returned mismatched workload evidence")
    record = payload.get("record")
    if not isinstance(record, Mapping):
        raise BaselineVerificationError("benchmark worker omitted its record")
    _validate_distribution(record.get("distribution"), name=descriptor.identifier)
    return dict(record)


def measure_hashes(
    *, size_bytes: int, lifecycle_distribution: Mapping[str, Any], loops: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compare raw SHA-256 and XXH3 throughput against one lifecycle boundary."""
    if size_bytes not in HASH_SIZES:
        raise ValueError("hash size is not a reviewed Phase 8 tier")
    if loops <= 0:
        raise ValueError("hash loops must be positive")
    if PAYLOAD_DIGEST_ALGORITHM != "sha256":
        raise BaselineVerificationError(
            "Phase 8 comparison requires the canonical SHA-256 payload digest"
        )
    lifecycle = _validate_distribution(lifecycle_distribution, name="lifecycle")
    payload = bytes(index % 251 for index in range(size_bytes))
    algorithms = (
        ("sha256", lambda: hashlib.sha256(payload).digest()),
        ("xxh3_64", lambda: xxhash.xxh3_64(payload).digest()),
    )
    observations: list[dict[str, Any]] = []
    for name, digest in algorithms:
        samples: list[int] = []
        for _ in range(loops):
            started = time.perf_counter_ns()
            digest()
            samples.append(max(1, time.perf_counter_ns() - started))
        measured = distribution(samples)
        throughput = (size_bytes * 1_000_000_000) / measured["p50_ns"]
        observations.append(
            {
                "algorithm": name,
                "distribution": measured,
                "size_bytes": size_bytes,
                "throughput_bytes_per_second": throughput,
                "lifecycle_share_p50": measured["p50_ns"] / lifecycle["p50_ns"],
            }
        )
    return tuple(observations)  # type: ignore[return-value]


def _selected_workloads(identifiers: Sequence[str], *, all_workloads: bool) -> tuple[WorkloadDescriptor, ...]:
    inventory = {descriptor.identifier: descriptor for descriptor in reviewed_workloads()}
    if identifiers:
        try:
            return tuple(inventory[identifier] for identifier in identifiers)
        except KeyError as error:
            raise BaselineVerificationError("unknown reviewed workload identifier") from error
    if all_workloads:
        return tuple(inventory.values())
    return (inventory["small-generic-object__blobstore__cold-put"],)


def _controlled_runner_preflight(runner_identity: str) -> None:
    if runner_identity != CONTROLLED_RUNNER_IDENTITY:
        raise BaselineVerificationError("controlled capture requires cacheness-perf-linux-x64")
    environment = environment_fingerprint()
    if environment["platform"] != "Linux" or environment["machine"] not in {"x86_64", "amd64"}:
        raise BaselineVerificationError("controlled capture requires a Linux x86_64 runner")


def _measurement_document(
    *,
    descriptors: Sequence[WorkloadDescriptor],
    reduced: bool,
    warmups: int,
    samples: int,
    runner_identity: str,
    controlled: bool,
) -> dict[str, Any]:
    records = {
        descriptor.identifier: _measure_in_worker(
            descriptor, reduced=reduced, warmups=warmups, samples=samples
        )
        for descriptor in descriptors
    }
    distributions = {
        identifier: record["distribution"] for identifier, record in records.items()
    }
    hash_observations = [
        observation
        for size_bytes in HASH_SIZES
        for observation in measure_hashes(
            size_bytes=size_bytes,
            lifecycle_distribution=next(iter(distributions.values())),
            loops=max(2, samples),
        )
    ]
    envelopes = {
        identifier: {"median_relative_limit": 1.20, "tail_relative_limit": 1.30}
        for identifier in distributions
    }
    document: dict[str, Any] = {
        "schema": BENCHMARK_SCHEMA,
        "evidence_class": "controlled-performance" if controlled else "diagnostic",
        "revision": _git_revision(),
        "source_digest": relevant_source_digest(require_complete=controlled),
        "runner_identity": runner_identity,
        "environment": environment_fingerprint(),
        "distributions": distributions,
        "envelopes": envelopes,
        "hash_observations": hash_observations,
        "records": records,
        "sampling": {"samples": samples, "warmups": warmups, "worker_subprocesses": True},
    }
    if controlled:
        _validate_document(document)
    return document


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--diagnostic", action="store_true")
    mode.add_argument("--capture-baseline", type=Path)
    mode.add_argument("--recalibrate-baseline", type=Path)
    mode.add_argument("--verify-baseline", type=Path)
    parser.add_argument("--output", type=Path, default=Path("build/phase8/performance.json"))
    parser.add_argument("--justification")
    parser.add_argument("--runner-identity", default=os.environ.get("CACHENESS_PERF_RUNNER", "uncontrolled-local"))
    parser.add_argument("--workload", action="append", default=[])
    parser.add_argument("--all-workloads", action="store_true")
    parser.add_argument("--reduced", action="store_true")
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    """Run one diagnostic/capture/recalibration/verify benchmark operation."""
    arguments = _parse_arguments()
    if arguments.worker:
        descriptor = next(
            (
                item
                for item in reviewed_workloads()
                if item.identifier == arguments.worker
            ),
            None,
        )
        if descriptor is None:
            raise SystemExit("unknown reviewed workload identifier")
        print(
            json.dumps(
                {
                    "identifier": descriptor.identifier,
                    "record": measure_workload(
                        descriptor,
                        reduced=arguments.reduced,
                        warmups=arguments.warmups,
                        samples=arguments.samples,
                    ),
                },
                sort_keys=True,
            )
        )
        return 0

    mode = "diagnostic"
    destination: Path | None = None
    if arguments.capture_baseline is not None:
        mode, destination = "capture", arguments.capture_baseline
    elif arguments.recalibrate_baseline is not None:
        mode, destination = "recalibrate", arguments.recalibrate_baseline
    elif arguments.verify_baseline is not None:
        mode, destination = "verify", arguments.verify_baseline
    if mode in {"capture", "recalibrate", "verify"}:
        _controlled_runner_preflight(arguments.runner_identity)
        if not arguments.all_workloads:
            raise SystemExit("controlled baseline operations require --all-workloads")

    document = _measurement_document(
        descriptors=_selected_workloads(arguments.workload, all_workloads=arguments.all_workloads),
        reduced=arguments.reduced,
        warmups=arguments.warmups,
        samples=arguments.samples,
        runner_identity=arguments.runner_identity,
        controlled=mode != "diagnostic",
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if mode == "verify":
        assert destination is not None
        verify_baseline(destination, document)
        print(f"verified controlled performance baseline: {destination}")
    elif mode in {"capture", "recalibrate"}:
        assert destination is not None
        atomic_replace_baseline(
            destination,
            document,
            mode=mode,
            justification=arguments.justification,
        )
        print(f"{mode}d controlled performance baseline: {destination}")
    else:
        print(f"wrote diagnostic performance evidence: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
