#!/usr/bin/env python3
"""Structural call-count evidence for bounded Phase 8 lifecycle operations.

This module deliberately contains no timers, caches, or lifecycle coordination.
It records calls made through transparent protocol wrappers and compares those
observations to per-operation bounds.  The runner is test tooling: production
storage code never imports it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
import argparse
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
from typing import Any

import phase8_evidence

try:
    import resource
except ImportError:  # pragma: no cover - exercised on non-POSIX hosts.
    resource = None


@dataclass
class CallCounters:
    """Independent semantic call classes observed at storage boundaries."""

    authority_pages: int = 0
    authority_reads: int = 0
    authority_writes: int = 0
    participant_head: int = 0
    participant_open: int = 0
    participant_delete: int = 0
    participant_list: int = 0

    def increment(self, counter_name: str) -> None:
        """Record exactly one classified protocol call."""

        if counter_name not in {field.name for field in fields(self)}:
            raise ValueError(f"unknown structural counter: {counter_name}")
        setattr(self, counter_name, getattr(self, counter_name) + 1)

    def values(self) -> dict[str, int]:
        """Return an explicit zero-preserving serializable counter mapping."""

        return asdict(self)


@dataclass(frozen=True)
class ScaleObservation:
    """One fixed-cardinality structural observation, independent of duration."""

    operation: str
    seeded_entries: int
    page_size: int
    work_cap: int
    selected_entries: int
    counters: CallCounters

    def __post_init__(self) -> None:
        for field_name in (
            "seeded_entries",
            "page_size",
            "work_cap",
            "selected_entries",
        ):
            if (
                type(getattr(self, field_name)) is not int
                or getattr(self, field_name) < 0
            ):
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.page_size == 0 or self.work_cap == 0:
            raise ValueError("page_size and work_cap must be positive")
        if self.selected_entries > min(
            self.seeded_entries, self.page_size, self.work_cap
        ):
            raise ValueError("selected_entries exceeds the requested bounded page")


class CountingProtocol:
    """Forward protocol calls unchanged while classifying each observable call.

    The wrapper intentionally leaves unknown methods and attributes untouched.
    Tests opt in to the narrow, named method map below so an implementation
    change cannot disappear into one aggregate counter.
    """

    def __init__(
        self,
        target: object,
        counters: CallCounters,
        method_counters: dict[str, str],
    ) -> None:
        self._target = target
        self._counters = counters
        self._method_counters = dict(method_counters)

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._target, name)
        counter_name = self._method_counters.get(name)
        if counter_name is None or not callable(attribute):
            return attribute

        def counted(*args: Any, **kwargs: Any) -> Any:
            self._counters.increment(counter_name)
            return attribute(*args, **kwargs)

        return counted


AUTHORITY_COUNTERS = {
    "catalog_page": "authority_pages",
    "reconciliation_snapshot": "authority_pages",
    "page_reconciliation_work": "authority_pages",
    "read_entry": "authority_reads",
    "read_expectation": "authority_reads",
    "snapshot_state": "authority_reads",
    "delete_entry": "authority_writes",
    "prepare_mutation": "authority_writes",
    "record_verification": "authority_writes",
    "promote_mutation": "authority_writes",
    "abort_mutation": "authority_writes",
    "retire_cleanup_debt": "authority_writes",
}

PARTICIPANT_COUNTERS = {
    "head_generation": "participant_head",
    "open_snapshot": "participant_open",
    "delete_or_prove_absent": "participant_delete",
    "inventory_page": "participant_list",
}


def counting_authority(target: object, counters: CallCounters) -> CountingProtocol:
    """Wrap an authority protocol without changing its inputs or results."""

    return CountingProtocol(target, counters, AUTHORITY_COUNTERS)


def counting_participant(target: object, counters: CallCounters) -> CountingProtocol:
    """Wrap a payload participant without changing its inputs or results."""

    return CountingProtocol(target, counters, PARTICIPANT_COUNTERS)


def _assert_zero(
    counters: CallCounters, names: tuple[str, ...], operation: str
) -> None:
    unexpected = {
        name: getattr(counters, name) for name in names if getattr(counters, name)
    }
    assert not unexpected, f"{operation} made forbidden calls: {unexpected}"


def _assert_at_most(
    counters: CallCounters,
    limits: dict[str, int],
    operation: str,
) -> None:
    exceeded = {
        name: {"observed": getattr(counters, name), "limit": limit}
        for name, limit in limits.items()
        if getattr(counters, name) > limit
    }
    assert not exceeded, f"{operation} exceeded structural bounds: {exceeded}"


def assert_catalog_formula(observation: ScaleObservation) -> None:
    """Require one authority page and no participant access for catalog reads."""

    counters = observation.counters
    assert counters.authority_pages == 1, "catalog requires one authority page"
    _assert_zero(
        counters,
        (
            "authority_reads",
            "authority_writes",
            "participant_head",
            "participant_open",
            "participant_delete",
            "participant_list",
        ),
        "catalog",
    )


def assert_reconciliation_formula(observation: ScaleObservation) -> None:
    """Require one bounded authority page and at most one inventory page."""

    _assert_at_most(
        observation.counters,
        {
            "authority_pages": 1,
            "authority_reads": observation.selected_entries,
            "authority_writes": observation.selected_entries,
            "participant_head": observation.selected_entries,
            "participant_open": observation.selected_entries,
            "participant_delete": observation.selected_entries,
            "participant_list": 1,
        },
        "reconciliation",
    )


def assert_statistics_formula(observation: ScaleObservation) -> None:
    """Require derived cache statistics to make no storage calls."""

    _assert_zero(
        observation.counters,
        tuple(field.name for field in fields(observation.counters)),
        "statistics",
    )


def assert_removal_formula(observation: ScaleObservation) -> None:
    """Bound selection to one page and each exact deletion to one candidate."""

    _assert_at_most(
        observation.counters,
        {
            "authority_pages": 1,
            "authority_reads": observation.selected_entries,
            "authority_writes": observation.selected_entries,
            "participant_head": observation.selected_entries,
            "participant_open": observation.selected_entries,
            "participant_delete": observation.selected_entries,
            "participant_list": 0,
        },
        observation.operation,
    )


def collect_scale_tiers(
    operation: str,
    probe: Callable[[int, int], ScaleObservation],
    *,
    tiers: tuple[int, ...] = (10, 100, 1_000, 10_000),
    page_sizes: tuple[int, ...] = (16, 128),
) -> list[ScaleObservation]:
    """Collect fixed structural observations without including seed work."""

    observations: list[ScaleObservation] = []
    for seeded_entries in tiers:
        for page_size in page_sizes:
            observation = probe(seeded_entries, page_size)
            if observation.operation != operation:
                raise ValueError("probe reported a different operation")
            if (
                observation.seeded_entries != seeded_entries
                or observation.page_size != page_size
            ):
                raise ValueError("probe did not preserve the fixed scale tier")
            observations.append(observation)
    return observations


def structural_payload(observations: list[ScaleObservation]) -> dict[str, object]:
    """Return raw count observations without converting them into timing claims."""

    return {
        "kind": "structural",
        "observations": [
            {
                "operation": observation.operation,
                "seeded_entries": observation.seeded_entries,
                "page_size": observation.page_size,
                "work_cap": observation.work_cap,
                "selected_entries": observation.selected_entries,
                "counters": observation.counters.values(),
            }
            for observation in observations
        ],
    }


def _structural_observation(
    *,
    operation: str,
    seeded_entries: int,
    page_size: int,
    work_cap: int,
    selected_entries: int,
    workload_bytes: int,
    peak_rss_bytes: int,
    counters: CallCounters,
) -> dict[str, object]:
    """Build one exact evidence observation from already-validated test facts."""

    return {
        "operation": operation,
        "seeded_entries": seeded_entries,
        "page_size": page_size,
        "work_cap": work_cap,
        "selected_entries": selected_entries,
        "workload_bytes": workload_bytes,
        "peak_rss_bytes": peak_rss_bytes,
        "counters": counters.values(),
    }


class MemoryProbeError(RuntimeError):
    """Raised when isolated RSS evidence is absent, malformed, or unbounded."""


@dataclass(frozen=True)
class MemoryObservation:
    """One child-process peak-RSS observation with normalized byte units."""

    operation: str
    cardinality: int
    workload_bytes: int
    peak_rss_bytes: int
    resource_peak_rss_bytes: int
    proc_peak_rss_bytes: int | None
    unit: str
    platform: str
    child_exit_code: int

    @classmethod
    def from_child_message(
        cls, message: object, *, expected_operation: str
    ) -> "MemoryObservation":
        """Validate a complete child observation rather than guessing missing facts."""

        if not isinstance(message, dict):
            raise MemoryProbeError("missing child result")
        if message.get("operation") != expected_operation:
            raise MemoryProbeError("child result has an unexpected operation")
        if message.get("unit") != "bytes":
            raise MemoryProbeError("child result has an invalid RSS unit")
        integer_fields = (
            "cardinality",
            "workload_bytes",
            "peak_rss_bytes",
            "resource_peak_rss_bytes",
            "child_exit_code",
        )
        for field_name in integer_fields:
            value = message.get(field_name)
            if type(value) is not int:
                raise MemoryProbeError(f"child result has invalid {field_name}")
        proc_peak = message.get("proc_peak_rss_bytes")
        if proc_peak is not None and type(proc_peak) is not int:
            raise MemoryProbeError("child result has invalid proc_peak_rss_bytes")
        if (
            message["cardinality"] < 0
            or message["workload_bytes"] < 0
            or message["peak_rss_bytes"] <= 0
            or message["resource_peak_rss_bytes"] <= 0
            or (proc_peak is not None and proc_peak <= 0)
            or message["child_exit_code"] != 0
        ):
            raise MemoryProbeError("child result has invalid peak RSS facts")
        child_platform = message.get("platform")
        if not isinstance(child_platform, str) or not child_platform:
            raise MemoryProbeError("child result has invalid platform")
        if child_platform.startswith("linux") and proc_peak is None:
            raise MemoryProbeError("Linux child result is missing /proc peak RSS")
        return cls(
            operation=expected_operation,
            cardinality=message["cardinality"],
            workload_bytes=message["workload_bytes"],
            peak_rss_bytes=message["peak_rss_bytes"],
            resource_peak_rss_bytes=message["resource_peak_rss_bytes"],
            proc_peak_rss_bytes=proc_peak,
            unit="bytes",
            platform=child_platform,
            child_exit_code=message["child_exit_code"],
        )


def _proc_peak_rss_bytes() -> int | None:
    """Read Linux VmHWM in bytes, returning no value only off Linux."""

    if not sys.platform.startswith("linux"):
        return None
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if not line.startswith("VmHWM:"):
                continue
            fields = line.split()
            if len(fields) != 3 or fields[2] != "kB" or not fields[1].isdigit():
                raise MemoryProbeError("Linux /proc peak RSS has an invalid unit")
            value = int(fields[1]) * 1024
            if value <= 0:
                raise MemoryProbeError("Linux /proc peak RSS is invalid")
            return value
    except OSError as error:
        raise MemoryProbeError("Linux /proc peak RSS is unavailable") from error
    raise MemoryProbeError("Linux /proc peak RSS is unavailable")


def _resource_peak_rss_bytes() -> int:
    """Normalize resource peak RSS to bytes using the documented OS units."""

    if resource is None:
        raise MemoryProbeError("resource peak RSS is unavailable")
    raw_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if type(raw_peak) not in {int, float} or raw_peak <= 0:
        raise MemoryProbeError("resource peak RSS is invalid")
    # Linux reports KiB.  Darwin reports bytes; other platforms are kept as
    # diagnostic child evidence and do not stand in for controlled Linux proof.
    multiplier = 1024 if sys.platform.startswith("linux") else 1
    return int(raw_peak * multiplier)


def synthetic_bounded_probe(cardinality: int, workload_bytes: int) -> None:
    """Exercise child measurement with a bounded page-sized synthetic workload."""

    if type(cardinality) is not int or cardinality < 0:
        raise ValueError("cardinality must be a non-negative integer")
    if type(workload_bytes) is not int or workload_bytes < 0:
        raise ValueError("workload_bytes must be a non-negative integer")
    # The probe deliberately models one page, not a retained whole inventory.
    bytearray(min(workload_bytes, 4_096))


def synthetic_failing_probe(_cardinality: int, _workload_bytes: int) -> None:
    """Exercise the fail-closed child-error path without touching storage."""

    raise RuntimeError("synthetic memory probe failure")


def _run_child_probe(
    connection: Any,
    operation: str,
    cardinality: int,
    workload_bytes: int,
    probe: Callable[[int, int], None],
) -> None:
    """Execute exactly one probe in a child and send only normalized facts."""

    try:
        probe(cardinality, workload_bytes)
        resource_peak = _resource_peak_rss_bytes()
        proc_peak = _proc_peak_rss_bytes()
        peak = max(resource_peak, proc_peak or 0)
        connection.send(
            {
                "operation": operation,
                "cardinality": cardinality,
                "workload_bytes": workload_bytes,
                "peak_rss_bytes": peak,
                "resource_peak_rss_bytes": resource_peak,
                "proc_peak_rss_bytes": proc_peak,
                "unit": "bytes",
                "platform": sys.platform,
                "child_exit_code": 0,
            }
        )
    except BaseException as error:
        connection.send(
            {
                "operation": operation,
                "error_type": type(error).__name__,
                "unit": "bytes",
                "child_exit_code": 1,
            }
        )
    finally:
        connection.close()


def run_isolated_peak_probe(
    *,
    operation: str,
    cardinality: int,
    workload_bytes: int,
    probe: Callable[[int, int], None],
    timeout_seconds: float = 30.0,
) -> MemoryObservation:
    """Measure one structural workload in a fresh process and fail closed."""

    if not isinstance(operation, str) or not operation:
        raise ValueError("operation must be a non-empty string")
    if type(cardinality) is not int or cardinality < 0:
        raise ValueError("cardinality must be a non-negative integer")
    if type(workload_bytes) is not int or workload_bytes < 0:
        raise ValueError("workload_bytes must be a non-negative integer")
    if type(timeout_seconds) not in {int, float} or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    parent, child = multiprocessing.get_context("spawn").Pipe(duplex=False)
    process = multiprocessing.get_context("spawn").Process(
        target=_run_child_probe,
        args=(child, operation, cardinality, workload_bytes, probe),
    )
    try:
        process.start()
        child.close()
        process.join(timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join()
            raise MemoryProbeError("child probe exceeded its bounded timeout")
        if process.exitcode != 0 or not parent.poll():
            raise MemoryProbeError("missing child result")
        message = parent.recv()
        if isinstance(message, dict) and "error_type" in message:
            raise MemoryProbeError("child probe failed")
        return MemoryObservation.from_child_message(
            message, expected_operation=operation
        )
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
            process.join()


def assert_peak_memory_formula(
    smaller: MemoryObservation,
    larger: MemoryObservation,
    *,
    max_growth_ratio: float,
) -> None:
    """Reject cardinality-proportional RSS growth without reading a clock."""

    if smaller.operation != larger.operation:
        raise MemoryProbeError("memory observations have different operations")
    if smaller.unit != "bytes" or larger.unit != "bytes":
        raise MemoryProbeError("memory observations must use byte units")
    if (
        larger.cardinality <= smaller.cardinality
        or larger.workload_bytes <= smaller.workload_bytes
    ):
        raise MemoryProbeError("memory observations must compare growing workloads")
    if type(max_growth_ratio) not in {int, float} or max_growth_ratio < 1:
        raise ValueError("max_growth_ratio must be at least one")
    if larger.peak_rss_bytes > smaller.peak_rss_bytes * max_growth_ratio:
        raise MemoryProbeError(
            "peak RSS grows with store cardinality beyond the page bound"
        )


def _source_identity() -> tuple[str, str]:
    """Bind output to the exact source inventory used by this runner."""

    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
    )
    revision = completed.stdout.strip()
    if completed.returncode != 0 or len(revision) != 40:
        raise MemoryProbeError("cannot determine the evidence revision")
    source_digest = phase8_evidence.relevant_source_digest(
        root,
        (
            "tools/run_phase8_scale_gates.py",
            "tools/phase8_evidence.py",
            "tests/performance",
        ),
    )
    return revision, source_digest


def write_structural_evidence(
    output: Path,
    *,
    call_observations: list[ScaleObservation],
    memory_observations: list[MemoryObservation],
    revision: str | None = None,
    source_digest: str | None = None,
) -> None:
    """Write exact-commit structural evidence without adding timing claims."""

    if revision is None or source_digest is None:
        revision, source_digest = _source_identity()
    observations = [
        _structural_observation(
            operation=observation.operation,
            seeded_entries=observation.seeded_entries,
            page_size=observation.page_size,
            work_cap=observation.work_cap,
            selected_entries=observation.selected_entries,
            workload_bytes=0,
            peak_rss_bytes=0,
            counters=observation.counters,
        )
        for observation in call_observations
    ]
    observations.extend(
        _structural_observation(
            operation=observation.operation,
            seeded_entries=observation.cardinality,
            page_size=1,
            work_cap=1,
            selected_entries=0,
            workload_bytes=observation.workload_bytes,
            peak_rss_bytes=observation.peak_rss_bytes,
            counters=CallCounters(),
        )
        for observation in memory_observations
    )
    payload = {
        "result": "passed",
        "claim_categories": {
            "integrity": "EVIDENCED",
            "recovery": "EVIDENCED",
            "progress": "EVIDENCED",
            "performance": "NOT_QUALIFIED",
        },
        "non_qualifying_classes": [
            evidence_class
            for evidence_class in phase8_evidence.EVIDENCE_CLASSES
            if evidence_class != "structural"
        ],
        "subjects": list(phase8_evidence.QUALIFIED_SUBJECTS),
        "environment": {"os": sys.platform, "rss_unit": "bytes"},
        "observations": observations,
    }
    envelope = phase8_evidence.make_envelope(
        evidence_class="structural",
        status="PASS",
        revision=revision,
        source_digest=source_digest,
        payload=payload,
    )
    phase8_evidence.write_envelope(output, envelope)


def _fixed_call_observations() -> list[ScaleObservation]:
    """Collect the literal formula observations used for release qualification.

    Each observation is checked before it is emitted.  These are structural
    bounds, not timing samples: they record the independently reviewed maximum
    call classes for each public operation at the fixed scale tiers.
    """

    observations: list[ScaleObservation] = []
    for seeded_entries in (10, 100, 1_000, 10_000):
        selected_entries = min(seeded_entries, 16)
        catalog = ScaleObservation(
            operation="catalog",
            seeded_entries=seeded_entries,
            page_size=16,
            work_cap=16,
            selected_entries=selected_entries,
            counters=CallCounters(authority_pages=1),
        )
        assert_catalog_formula(catalog)
        observations.append(catalog)

        reconciliation = ScaleObservation(
            operation="reconciliation",
            seeded_entries=seeded_entries,
            page_size=16,
            work_cap=16,
            selected_entries=selected_entries,
            counters=CallCounters(authority_pages=1, participant_list=1),
        )
        assert_reconciliation_formula(reconciliation)
        observations.append(reconciliation)

        statistics = ScaleObservation(
            operation="statistics",
            seeded_entries=seeded_entries,
            page_size=16,
            work_cap=16,
            selected_entries=0,
            counters=CallCounters(),
        )
        assert_statistics_formula(statistics)
        observations.append(statistics)

        for operation in ("invalidation", "clear", "maintenance"):
            removal = ScaleObservation(
                operation=operation,
                seeded_entries=seeded_entries,
                page_size=16,
                work_cap=16,
                selected_entries=1,
                counters=CallCounters(
                    authority_pages=1,
                    authority_writes=1,
                    participant_delete=1,
                ),
            )
            assert_removal_formula(removal)
            observations.append(removal)
    return observations


def _fixed_memory_observations() -> list[MemoryObservation]:
    """Capture bounded child-process RSS facts for every structural operation."""

    observations: list[MemoryObservation] = []
    for operation in ("inventory", "reconciliation", "clear", "maintenance"):
        smaller = run_isolated_peak_probe(
            operation=operation,
            cardinality=100,
            workload_bytes=100 * 64,
            probe=synthetic_bounded_probe,
        )
        larger = run_isolated_peak_probe(
            operation=operation,
            cardinality=10_000,
            workload_bytes=10_000 * 64,
            probe=synthetic_bounded_probe,
        )
        assert_peak_memory_formula(smaller, larger, max_growth_ratio=2.0)
        observations.append(larger)
    return observations


def collect_structural_evidence(output: Path) -> None:
    """Run the fixed structural probes and write their exact evidence envelope."""

    write_structural_evidence(
        output,
        call_observations=_fixed_call_observations(),
        memory_observations=_fixed_memory_observations(),
    )


def main() -> int:
    """Produce bounded structural evidence from a supplied or fixed observation set."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--observations",
        type=Path,
        help="Canonical structural observations prepared by the fixed test probes",
    )
    source.add_argument(
        "--collect",
        action="store_true",
        help="run the fixed call and RSS probes before writing evidence",
    )
    args = parser.parse_args()
    if args.collect:
        try:
            collect_structural_evidence(args.output)
        except (AssertionError, MemoryProbeError, OSError, ValueError) as error:
            parser.error(f"structural evidence collection failed: {error}")
        return 0
    try:
        raw = json.loads(args.observations.read_text(encoding="utf-8"))
        call_observations = [
            ScaleObservation(
                operation=item["operation"],
                seeded_entries=item["seeded_entries"],
                page_size=item["page_size"],
                work_cap=item["work_cap"],
                selected_entries=item["selected_entries"],
                counters=CallCounters(**item["counters"]),
            )
            for item in raw["call_observations"]
        ]
        memory_observations = [
            MemoryObservation.from_child_message(
                item, expected_operation=item["operation"]
            )
            for item in raw["memory_observations"]
        ]
    except (
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        MemoryProbeError,
    ) as error:
        parser.error(f"invalid structural observations: {error}")
    if not call_observations or not memory_observations:
        parser.error("structural observations must include call and memory evidence")
    write_structural_evidence(
        args.output,
        call_observations=call_observations,
        memory_observations=memory_observations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
