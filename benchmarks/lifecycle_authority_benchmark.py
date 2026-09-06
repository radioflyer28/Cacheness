#!/usr/bin/env python3
"""Measure bounded SQLite lifecycle-authority operations on the production schema.

The checked-in baseline is evidence, not a source of runtime policy by itself.
Use ``--record-baseline`` only for the first capture and
``--recalibrate-baseline`` for an intentional replacement.  ``--verify-baseline``
repeats the production-path measurements, checks the recorded formulas, and
enforces the deliberately roomy release envelopes without mutating the baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable
from uuid import uuid4

from cacheness.config import LifecycleLimits
from cacheness.error_handling import CacheBlobLifecycleTimeoutError
from cacheness.storage.blob_store import BlobStore
from cacheness.storage.lifecycle_authority import (
    EntryExpectation,
    MutationSpec,
    VerificationProof,
)
from cacheness.storage.sqlite_lifecycle_authority import (
    SCHEMA_VERSION,
    SQLITE_APPLICATION_ID,
    SqliteLifecycleAuthority,
)


BENCHMARK_SCHEMA_VERSION = 1
REPETITIONS = 5
WARMUPS = 1
CLEAR_TARGET_COUNTS = (16, 64)
RECONCILIATION_ITEMS = 16
RECONCILIATION_MANIFEST_BYTES = 4096
BENCHMARK_LIMITS = LifecycleLimits(
    authority_busy_timeout_seconds=0.075,
    operation_page_size=RECONCILIATION_ITEMS * 2,
)

# These margins preserve a release-envelope signal while avoiding an anecdotal
# latency target.  Runtime limits are derived separately below.
MULTIPLIERS = {
    "busy_deadline": 2.0,
    "reconciliation_rows": 2.0,
    "reconciliation_actions": 2.0,
    "reconciliation_bytes": 2.0,
    "reconciliation_time": 2.0,
    "authority_transition_latency": 4.0,
    "clear_snapshot_latency": 4.0,
    "distinct_key_overlap": 0.8,
}


class BenchmarkVerificationError(RuntimeError):
    """Raised when a baseline is malformed or current measurements regress."""


def _percentile(samples: list[float | int], percentile: float) -> float:
    """Return a deterministic linear percentile without NumPy dependency."""
    if not samples:
        raise ValueError("benchmark distributions require at least one sample")
    ordered = sorted(float(sample) for sample in samples)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _distribution(samples: list[float | int]) -> dict[str, Any]:
    """Preserve raw samples and the percentiles used by every derivation."""
    if not samples:
        raise ValueError("benchmark distributions require at least one sample")
    return {
        "samples": list(samples),
        "p05": _percentile(samples, 0.05),
        "p50": _percentile(samples, 0.50),
        "p95": _percentile(samples, 0.95),
        "p99": _percentile(samples, 0.99),
    }


def _ceil_int(value: float) -> int:
    """Convert a measured upper bound into a positive integral limit."""
    return max(1, math.ceil(value))


def _ceil_milliseconds(value: float) -> float:
    """Round a deadline upward to the next whole millisecond."""
    return math.ceil(value * 1000.0) / 1000.0


def _git_revision() -> str:
    """Return the source revision that produced the measurement evidence."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _seed_entry(
    authority: SqliteLifecycleAuthority,
    *,
    key: str,
    manifest: bytes,
) -> None:
    """Exercise the real prepare/verify/promote transition for one entry."""
    existing = authority.read_entry(key)
    expected = existing.expectation if existing is not None else EntryExpectation.absent()
    operation_id = uuid4().hex
    generation = uuid4().hex
    prepared = authority.prepare_mutation(
        MutationSpec.create(
            operation_id=operation_id,
            key=key,
            generation=generation,
            candidate_locator=f".cacheness/generations/{generation}.payload",
            expected=expected,
            manifest=manifest,
        )
    )
    authority.record_verification(
        prepared,
        VerificationProof("0" * 64, len(manifest), manifest),
    )
    authority.promote_mutation(prepared)


def _with_authority(
    callback: Callable[[SqliteLifecycleAuthority], Any],
    *,
    limits: LifecycleLimits = BENCHMARK_LIMITS,
) -> Any:
    """Run one controlled local measurement and always close its authority."""
    with tempfile.TemporaryDirectory(prefix="cacheness-lifecycle-benchmark-") as temp_dir:
        authority = SqliteLifecycleAuthority.for_root(temp_dir, lifecycle_limits=limits)
        try:
            return callback(authority)
        finally:
            authority.close()


def _measure_transition() -> float:
    """Measure a short complete transition after schema creation is warmed."""

    def measure(authority: SqliteLifecycleAuthority) -> float:
        _seed_entry(authority, key="warmup", manifest=b"warmup")
        started = time.perf_counter()
        _seed_entry(authority, key="measured", manifest=b"measured")
        return time.perf_counter() - started

    return _with_authority(measure)


def _measure_clear_snapshot(target_count: int) -> float:
    """Measure the production ``INSERT ... SELECT`` clear-membership transition."""

    def measure(authority: SqliteLifecycleAuthority) -> float:
        for index in range(target_count):
            _seed_entry(
                authority,
                key=f"clear-{index:04d}",
                manifest=f"clear-{index}".encode("ascii"),
            )
        started = time.perf_counter()
        authority.begin_clear()
        return time.perf_counter() - started

    return _with_authority(measure)


def _measure_busy_wait() -> float:
    """Measure one held independent writer until the authority deadline expires."""

    def measure(authority: SqliteLifecycleAuthority) -> float:
        _seed_entry(authority, key="busy", manifest=b"busy")
        blocker = sqlite3.connect(authority.path, isolation_level=None)
        try:
            blocker.execute("BEGIN IMMEDIATE")
            started = time.perf_counter()
            try:
                authority.begin_clear()
            except CacheBlobLifecycleTimeoutError:
                elapsed = time.perf_counter() - started
            else:
                raise BenchmarkVerificationError(
                    "held SQLite writer did not enforce the authority busy deadline"
                )
        finally:
            if blocker.in_transaction:
                blocker.execute("ROLLBACK")
            blocker.close()
        return elapsed

    return _with_authority(measure)


def _measure_reconciliation_page() -> tuple[int, int, int, float]:
    """Measure bounded indexed reconciliation work over production rows."""

    def measure(authority: SqliteLifecycleAuthority) -> tuple[int, int, int, float]:
        manifest = b"r" * RECONCILIATION_MANIFEST_BYTES
        for index in range(RECONCILIATION_ITEMS):
            prepared = authority.prepare_mutation(
                MutationSpec.create(
                    operation_id=uuid4().hex,
                    key=f"reconcile-{index:04d}",
                    generation=uuid4().hex,
                    candidate_locator=f".cacheness/generations/reconcile-{index}.payload",
                    expected=EntryExpectation.absent(),
                    manifest=manifest,
                )
            )
            if prepared.operation_id == "":  # pragma: no cover - contract guard
                raise AssertionError("prepared operation must remain identifiable")
        snapshot = authority.reconciliation_snapshot()
        started = time.perf_counter()
        page = authority.page_reconciliation_work(
            snapshot,
            mutation_cursor=0,
            debt_cursor=0,
        )
        elapsed = time.perf_counter() - started
        rows = len(page.works)
        actions = sum(1 for work in page.works if work.source in {"mutation", "debt"})
        byte_count = sum(
            len(work.mutation.spec.manifest)
            for work in page.works
            if work.mutation is not None
        )
        if rows != RECONCILIATION_ITEMS or actions != rows:
            raise BenchmarkVerificationError("reconciliation benchmark did not return one bounded page")
        return rows, actions, byte_count, elapsed

    return _with_authority(measure)


def _put_pair(store: BlobStore, prefix: str) -> None:
    """Store two independent keys through the public payload lifecycle."""
    store.put(f"{prefix}-a", key=f"{prefix}-a")
    store.put(f"{prefix}-b", key=f"{prefix}-b")


def _measure_distinct_key_overlap() -> float:
    """Measure barrier-synchronized distinct-key public payload overlap ratio."""
    with tempfile.TemporaryDirectory(prefix="cacheness-lifecycle-overlap-") as temp_dir:
        sequential = BlobStore(Path(temp_dir) / "sequential", backend="json")
        try:
            started = time.perf_counter()
            _put_pair(sequential, "sequential")
            sequential_seconds = time.perf_counter() - started
        finally:
            sequential.close()

        concurrent = BlobStore(Path(temp_dir) / "concurrent", backend="json")
        barrier = threading.Barrier(2)
        errors: list[BaseException] = []

        def pause_at_candidate(boundary: str) -> None:
            if boundary == "put.candidate_published":
                barrier.wait(timeout=5)

        concurrent.lifecycle.test_hook = pause_at_candidate

        def put(key: str) -> None:
            try:
                concurrent.put(key, key=key)
            except BaseException as error:  # pragma: no cover - propagated below
                errors.append(error)

        threads = [
            threading.Thread(target=put, args=("concurrent-a",)),
            threading.Thread(target=put, args=("concurrent-b",)),
        ]
        try:
            started = time.perf_counter()
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
            concurrent_seconds = time.perf_counter() - started
            if any(thread.is_alive() for thread in threads):
                raise BenchmarkVerificationError("distinct-key overlap benchmark did not complete")
            if errors:
                raise BenchmarkVerificationError(
                    f"distinct-key overlap benchmark failed: {errors[0]}"
                )
        finally:
            concurrent.close()

    if concurrent_seconds <= 0:
        raise BenchmarkVerificationError("distinct-key overlap duration must be positive")
    return sequential_seconds / concurrent_seconds


def _warm(callback: Callable[[], Any]) -> None:
    """Run one deliberate unrecorded warmup for a measurement scenario."""
    for _ in range(WARMUPS):
        callback()


def collect_measurements() -> dict[str, Any]:
    """Collect repeated production-schema distributions with no baseline writes."""
    _warm(_measure_transition)
    transition_samples = [_measure_transition() for _ in range(REPETITIONS)]

    clear_samples: dict[str, dict[str, Any]] = {}
    for target_count in CLEAR_TARGET_COUNTS:
        _warm(lambda count=target_count: _measure_clear_snapshot(count))
        clear_samples[str(target_count)] = _distribution(
            [_measure_clear_snapshot(target_count) for _ in range(REPETITIONS)]
        )

    _warm(_measure_busy_wait)
    busy_samples = [_measure_busy_wait() for _ in range(REPETITIONS)]

    _warm(_measure_reconciliation_page)
    reconciliation_samples = [_measure_reconciliation_page() for _ in range(REPETITIONS)]

    _warm(_measure_distinct_key_overlap)
    overlap_samples = [_measure_distinct_key_overlap() for _ in range(REPETITIONS)]

    return {
        "authority_transition": {"seconds": _distribution(transition_samples)},
        "clear_snapshot": {"seconds_by_target_count": clear_samples},
        "busy_wait": {"seconds": _distribution(busy_samples)},
        "reconciliation_page": {
            "rows": _distribution([sample[0] for sample in reconciliation_samples]),
            "actions": _distribution([sample[1] for sample in reconciliation_samples]),
            "bytes": _distribution([sample[2] for sample in reconciliation_samples]),
            "seconds": _distribution([sample[3] for sample in reconciliation_samples]),
        },
        "distinct_key_overlap": {"ratio": _distribution(overlap_samples)},
    }


def derive_configuration(
    metrics: dict[str, Any],
    multipliers: dict[str, float],
) -> dict[str, float | int]:
    """Derive only the four runtime bounds authorized by D-29."""
    reconciliation = metrics["reconciliation_page"]
    busy_seconds = metrics["busy_wait"]["seconds"]["p99"]
    reconciliation_seconds = reconciliation["seconds"]["p99"]
    return {
        "operation_page_size": _ceil_int(
            reconciliation["rows"]["p99"] * multipliers["reconciliation_rows"]
        ),
        "max_reconcile_actions": _ceil_int(
            reconciliation["actions"]["p99"] * multipliers["reconciliation_actions"]
        ),
        "max_operation_record_bytes": _ceil_int(
            reconciliation["bytes"]["p99"] * multipliers["reconciliation_bytes"]
        ),
        "authority_busy_timeout_seconds": _ceil_milliseconds(
            max(
                busy_seconds * multipliers["busy_deadline"],
                reconciliation_seconds * multipliers["reconciliation_time"],
            )
        ),
    }


def derive_release_envelopes(
    metrics: dict[str, Any],
    multipliers: dict[str, float],
) -> dict[str, Any]:
    """Keep noisy latency and overlap evidence outside runtime configuration."""
    return {
        "authority_transition_max_seconds": (
            metrics["authority_transition"]["seconds"]["p99"]
            * multipliers["authority_transition_latency"]
        ),
        "clear_snapshot_max_seconds_by_target_count": {
            target_count: distribution["p99"] * multipliers["clear_snapshot_latency"]
            for target_count, distribution in metrics["clear_snapshot"][
                "seconds_by_target_count"
            ].items()
        },
        "busy_wait_max_seconds": (
            metrics["busy_wait"]["seconds"]["p99"] * multipliers["busy_deadline"]
        ),
        "distinct_key_overlap_min_ratio": (
            metrics["distinct_key_overlap"]["ratio"]["p05"]
            * multipliers["distinct_key_overlap"]
        ),
    }


def _filesystem_provenance(path: Path) -> dict[str, int | str]:
    """Record portable local-filesystem facts without host-specific secrets."""
    values = os.statvfs(path)
    return {
        "kind": "local temporary directory",
        "block_size": values.f_bsize,
        "fragment_size": values.f_frsize,
        "name_max": values.f_namemax,
    }


def build_baseline() -> dict[str, Any]:
    """Capture complete measurement evidence and all auditable derivations."""
    with tempfile.TemporaryDirectory(prefix="cacheness-lifecycle-provenance-") as temp_dir:
        filesystem = _filesystem_provenance(Path(temp_dir))
    metrics = collect_measurements()
    derived_configuration = derive_configuration(metrics, MULTIPLIERS)
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark": {
            "command": (
                ".venv/bin/python benchmarks/lifecycle_authority_benchmark.py "
                "--record-baseline benchmarks/lifecycle_authority_baseline.json"
            ),
            "harness": "production-sqlite-lifecycle-authority",
            "source_commit": _git_revision(),
            "authority_schema": {
                "application_id": SQLITE_APPLICATION_ID,
                "user_version": SCHEMA_VERSION,
            },
            "environment": {
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "sqlite": sqlite3.sqlite_version,
                "platform": platform.platform(),
                "filesystem": filesystem,
            },
            "cardinalities": {
                "clear_target_counts": list(CLEAR_TARGET_COUNTS),
                "reconciliation_items": RECONCILIATION_ITEMS,
                "reconciliation_manifest_bytes": RECONCILIATION_MANIFEST_BYTES,
            },
            "repetitions": REPETITIONS,
            "warmups": WARMUPS,
        },
        "metrics": metrics,
        "multipliers": dict(MULTIPLIERS),
        "multiplier_rationale": {
            "busy_deadline": "2x p99 leaves bounded scheduler and filesystem jitter margin.",
            "reconciliation_rows": "2x p99 preserves a whole measured indexed page margin.",
            "reconciliation_actions": "2x p99 preserves page-action headroom.",
            "reconciliation_bytes": "2x p99 leaves headroom for real canonical records.",
            "reconciliation_time": "2x p99 bounds page execution within the authority deadline.",
            "authority_transition_latency": "4x p99 is a benchmark-only release envelope.",
            "clear_snapshot_latency": "4x p99 is a benchmark-only clear-snapshot envelope.",
            "distinct_key_overlap": "80% of p05 is a benchmark-only minimum overlap envelope.",
        },
        "derived": {
            "configuration": derived_configuration,
            "release_envelopes": derive_release_envelopes(metrics, MULTIPLIERS),
        },
    }


def _require_mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkVerificationError(f"baseline field {path} must be an object")
    return value


def _validate_distribution(value: object, path: str) -> None:
    distribution = _require_mapping(value, path)
    samples = distribution.get("samples")
    if not isinstance(samples, list) or not samples or any(
        isinstance(sample, bool) or not isinstance(sample, (int, float)) or sample < 0
        for sample in samples
    ):
        raise BenchmarkVerificationError(f"baseline distribution {path} has invalid samples")
    expected = _distribution(samples)
    for name in ("p05", "p50", "p95", "p99"):
        if distribution.get(name) != expected[name]:
            raise BenchmarkVerificationError(f"baseline distribution {path}.{name} is inconsistent")


def validate_baseline(baseline: object) -> dict[str, Any]:
    """Reject unauditable evidence before it influences release verification."""
    document = _require_mapping(baseline, "root")
    if document.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise BenchmarkVerificationError("baseline schema version is unsupported")
    benchmark = _require_mapping(document.get("benchmark"), "benchmark")
    for name in ("command", "harness", "source_commit", "environment", "cardinalities"):
        if not benchmark.get(name):
            raise BenchmarkVerificationError(f"baseline benchmark.{name} is required")
    authority_schema = _require_mapping(benchmark.get("authority_schema"), "authority_schema")
    if authority_schema != {
        "application_id": SQLITE_APPLICATION_ID,
        "user_version": SCHEMA_VERSION,
    }:
        raise BenchmarkVerificationError("baseline authority schema does not match production")
    if not isinstance(benchmark.get("repetitions"), int) or benchmark["repetitions"] < 2:
        raise BenchmarkVerificationError("baseline repetitions must preserve repeated samples")
    if not isinstance(benchmark.get("warmups"), int) or benchmark["warmups"] < 1:
        raise BenchmarkVerificationError("baseline warmups must be explicit")

    metrics = _require_mapping(document.get("metrics"), "metrics")
    _validate_distribution(metrics.get("authority_transition", {}).get("seconds"), "authority_transition.seconds")
    clear = _require_mapping(metrics.get("clear_snapshot"), "clear_snapshot")
    clear_distributions = _require_mapping(
        clear.get("seconds_by_target_count"), "clear_snapshot.seconds_by_target_count"
    )
    if not clear_distributions:
        raise BenchmarkVerificationError("baseline must contain clear target cardinalities")
    for target_count, distribution in clear_distributions.items():
        if not str(target_count).isdigit():
            raise BenchmarkVerificationError("clear target count must be numeric")
        _validate_distribution(distribution, f"clear_snapshot.{target_count}")
    _validate_distribution(metrics.get("busy_wait", {}).get("seconds"), "busy_wait.seconds")
    reconciliation = _require_mapping(metrics.get("reconciliation_page"), "reconciliation_page")
    for field in ("rows", "actions", "bytes", "seconds"):
        _validate_distribution(reconciliation.get(field), f"reconciliation_page.{field}")
    _validate_distribution(
        metrics.get("distinct_key_overlap", {}).get("ratio"),
        "distinct_key_overlap.ratio",
    )

    multipliers = _require_mapping(document.get("multipliers"), "multipliers")
    if set(MULTIPLIERS) != set(multipliers) or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
        for value in multipliers.values()
    ):
        raise BenchmarkVerificationError("baseline multipliers are incomplete or invalid")
    derived = _require_mapping(document.get("derived"), "derived")
    expected_configuration = derive_configuration(metrics, multipliers)
    if derived.get("configuration") != expected_configuration:
        raise BenchmarkVerificationError("baseline configuration derivation is inconsistent")
    expected_envelopes = derive_release_envelopes(metrics, multipliers)
    if derived.get("release_envelopes") != expected_envelopes:
        raise BenchmarkVerificationError("baseline release envelope derivation is inconsistent")
    return document


def verify_regression(baseline: dict[str, Any], current: dict[str, Any]) -> None:
    """Compare fresh measurements with evidence-derived release envelopes."""
    envelopes = baseline["derived"]["release_envelopes"]
    if (
        current["authority_transition"]["seconds"]["p99"]
        > envelopes["authority_transition_max_seconds"]
    ):
        raise BenchmarkVerificationError("authority transition p99 exceeded its release envelope")
    for target_count, envelope in envelopes[
        "clear_snapshot_max_seconds_by_target_count"
    ].items():
        current_distribution = current["clear_snapshot"]["seconds_by_target_count"].get(
            target_count
        )
        if current_distribution is None or current_distribution["p99"] > envelope:
            raise BenchmarkVerificationError(
                f"clear snapshot p99 exceeded its release envelope for {target_count} targets"
            )
    if current["busy_wait"]["seconds"]["p99"] > envelopes["busy_wait_max_seconds"]:
        raise BenchmarkVerificationError("busy wait p99 exceeded its release envelope")
    if (
        current["distinct_key_overlap"]["ratio"]["p05"]
        < envelopes["distinct_key_overlap_min_ratio"]
    ):
        raise BenchmarkVerificationError("distinct-key overlap p05 fell below its release envelope")


def _atomic_write(path: Path, document: dict[str, Any]) -> None:
    """Publish recalibration evidence without leaving a partial JSON baseline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return validate_baseline(json.load(handle))
    except OSError as error:
        raise BenchmarkVerificationError(f"cannot read baseline {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise BenchmarkVerificationError(f"baseline {path} is not valid JSON") from error


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record-baseline", type=Path)
    mode.add_argument("--recalibrate-baseline", type=Path)
    mode.add_argument("--verify-baseline", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run explicit capture, recalibration, or non-mutating verification."""
    arguments = _parse_args(argv)
    if arguments.verify_baseline is not None:
        baseline = _load(arguments.verify_baseline)
        current = collect_measurements()
        verify_regression(baseline, current)
        print(f"verified lifecycle authority baseline: {arguments.verify_baseline}")
        return 0

    destination = arguments.record_baseline or arguments.recalibrate_baseline
    assert destination is not None
    if arguments.record_baseline is not None and destination.exists():
        raise BenchmarkVerificationError(
            "baseline already exists; use --recalibrate-baseline for an explicit replacement"
        )
    baseline = build_baseline()
    _atomic_write(destination, baseline)
    print(f"recorded lifecycle authority baseline: {destination}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkVerificationError as error:
        print(f"benchmark verification failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
