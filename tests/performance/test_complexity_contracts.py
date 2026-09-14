"""Structural Phase 8 scale contracts for bounded lifecycle operations.

These contracts intentionally measure semantic backend calls instead of elapsed
time.  A slow or fast host therefore cannot hide an accidental N+1 access
pattern.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


_RUNNER_PATH = Path(__file__).parents[2] / "tools" / "run_phase8_scale_gates.py"
_RUNNER_SPEC = importlib.util.spec_from_file_location("phase8_scale_gates", _RUNNER_PATH)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_RUNNER_SPEC)
sys.modules[_RUNNER_SPEC.name] = _RUNNER
_RUNNER_SPEC.loader.exec_module(_RUNNER)

CallCounters = _RUNNER.CallCounters
ScaleObservation = _RUNNER.ScaleObservation
assert_catalog_formula = _RUNNER.assert_catalog_formula
assert_reconciliation_formula = _RUNNER.assert_reconciliation_formula
assert_removal_formula = _RUNNER.assert_removal_formula
assert_statistics_formula = _RUNNER.assert_statistics_formula
collect_scale_tiers = _RUNNER.collect_scale_tiers
counting_authority = _RUNNER.counting_authority
counting_participant = _RUNNER.counting_participant


@pytest.mark.parametrize("seeded_entries", [10, 100, 1_000, 10_000])
@pytest.mark.parametrize("page_size", [16, 128])
def test_catalog_formula_is_scale_invariant(
    seeded_entries: int, page_size: int
) -> None:
    """One catalog page must not inspect payloads or perform hidden N+1 reads."""

    observation = ScaleObservation(
        operation="catalog",
        seeded_entries=seeded_entries,
        page_size=page_size,
        work_cap=page_size,
        selected_entries=min(seeded_entries, page_size),
        counters=CallCounters(authority_pages=1),
    )

    assert_catalog_formula(observation)


def test_catalog_formula_rejects_payload_access_and_extra_authority_pages() -> None:
    """Independent participant counters make an N+1 regression observable."""

    observation = ScaleObservation(
        operation="catalog",
        seeded_entries=10_000,
        page_size=16,
        work_cap=16,
        selected_entries=16,
        counters=CallCounters(authority_pages=2, participant_open=1),
    )

    with pytest.raises(AssertionError, match="catalog"):
        assert_catalog_formula(observation)


@pytest.mark.parametrize("seeded_entries", [10, 100, 1_000, 10_000])
def test_reconciliation_formula_has_independent_authority_and_inventory_bounds(
    seeded_entries: int,
) -> None:
    """Reconciliation may inspect one authority page and one inventory page."""

    observation = ScaleObservation(
        operation="reconciliation",
        seeded_entries=seeded_entries,
        page_size=32,
        work_cap=32,
        selected_entries=min(seeded_entries, 32),
        counters=CallCounters(authority_pages=1, participant_list=1),
    )

    assert_reconciliation_formula(observation)


def test_reconciliation_formula_rejects_extra_call_classes() -> None:
    """A write or exact deletion needs an explicit selected candidate budget."""

    observation = ScaleObservation(
        operation="reconciliation",
        seeded_entries=100,
        page_size=32,
        work_cap=32,
        selected_entries=1,
        counters=CallCounters(
            authority_pages=1,
            authority_writes=2,
            participant_list=1,
            participant_delete=2,
        ),
    )

    with pytest.raises(AssertionError, match="reconciliation"):
        assert_reconciliation_formula(observation)


def test_counting_wrappers_preserve_results_and_keep_call_classes_separate() -> None:
    """Instrumentation only observes a protocol call; it cannot alter it."""

    class Authority:
        def catalog_page(self, cursor: str) -> tuple[str, str]:
            return ("catalog", cursor)

    class Participant:
        def inventory_page(self, cursor: str) -> tuple[str, str]:
            return ("inventory", cursor)

        def head_generation(self, locator: str) -> str:
            return locator

    counters = CallCounters()
    authority = counting_authority(Authority(), counters)
    participant = counting_participant(Participant(), counters)

    assert authority.catalog_page("catalog-token") == ("catalog", "catalog-token")
    assert participant.inventory_page("inventory-token") == ("inventory", "inventory-token")
    assert participant.head_generation("generations/a.payload") == "generations/a.payload"
    assert counters.values() == {
        "authority_pages": 1,
        "authority_reads": 0,
        "authority_writes": 0,
        "participant_head": 1,
        "participant_open": 0,
        "participant_delete": 0,
        "participant_list": 1,
    }


def test_statistics_formula_requires_no_authority_or_participant_access() -> None:
    """Statistics is a policy-derived observer, not a storage scan."""

    observation = ScaleObservation(
        operation="statistics",
        seeded_entries=10_000,
        page_size=16,
        work_cap=16,
        selected_entries=0,
        counters=CallCounters(),
    )

    assert_statistics_formula(observation)

    with pytest.raises(AssertionError, match="statistics"):
        assert_statistics_formula(
            ScaleObservation(
                operation="statistics",
                seeded_entries=10_000,
                page_size=16,
                work_cap=16,
                selected_entries=0,
                counters=CallCounters(authority_reads=1),
            )
        )


@pytest.mark.parametrize("operation", ["invalidation", "clear", "maintenance"])
@pytest.mark.parametrize("seeded_entries", [10, 100, 1_000, 10_000])
def test_exact_removal_formulas_are_scale_invariant(
    operation: str, seeded_entries: int
) -> None:
    """One selected candidate permits at most one exact deletion sequence."""

    observation = ScaleObservation(
        operation=operation,
        seeded_entries=seeded_entries,
        page_size=16,
        work_cap=16,
        selected_entries=1,
        counters=CallCounters(authority_pages=1, authority_writes=1, participant_delete=1),
    )

    assert_removal_formula(observation)


def test_fixed_scale_collector_never_includes_seed_calls_in_observations() -> None:
    """Tiers are fixed before invocation, so setup cannot mask an N+1 result."""

    seeded: list[int] = []

    def probe(seeded_entries: int, page_size: int) -> ScaleObservation:
        seeded.append(seeded_entries)
        return ScaleObservation(
            operation="catalog",
            seeded_entries=seeded_entries,
            page_size=page_size,
            work_cap=page_size,
            selected_entries=min(seeded_entries, page_size),
            counters=CallCounters(authority_pages=1),
        )

    observations = collect_scale_tiers("catalog", probe)

    assert len(observations) == 8
    assert seeded == [10, 10, 100, 100, 1_000, 1_000, 10_000, 10_000]
    assert all(observation.counters.authority_pages == 1 for observation in observations)
