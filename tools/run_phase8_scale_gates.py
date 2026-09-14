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
from pathlib import Path
from typing import Any


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
            if type(getattr(self, field_name)) is not int or getattr(self, field_name) < 0:
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


def _assert_zero(counters: CallCounters, names: tuple[str, ...], operation: str) -> None:
    unexpected = {name: getattr(counters, name) for name in names if getattr(counters, name)}
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
            if observation.seeded_entries != seeded_entries or observation.page_size != page_size:
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


def main() -> int:
    """Write caller-supplied observations only; production probes stay in tests."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(structural_payload([]), sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
