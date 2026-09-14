"""Structural Phase 8 scale contracts for bounded lifecycle operations.

These contracts intentionally measure semantic backend calls instead of elapsed
time.  A slow or fast host therefore cannot hide an accidental N+1 access
pattern.
"""

from __future__ import annotations

import pytest

from tools.run_phase8_scale_gates import (
    CallCounters,
    ScaleObservation,
    assert_catalog_formula,
    assert_reconciliation_formula,
)


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
