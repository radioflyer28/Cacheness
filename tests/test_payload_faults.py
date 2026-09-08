"""Deterministic payload-boundary fault contracts for the sole lifecycle engine.

These tests deliberately classify integrity, recovery, progress, and
performance separately.  They use the engine's named test boundaries instead
of timing, polling, or a second coordination mechanism.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _lifecycle_test_support import (
    BoundaryHooks,
    InjectedLifecycleFault,
    authority_whole_state,
)
from cacheness.error_handling import CacheBlobRecoverableCleanupError
from cacheness.storage import BlobStore
from cacheness.storage.composition import BackendRef, StoreTopology


def _store(root: Path) -> BlobStore:
    """Build the qualified durable local topology for deterministic faults."""
    return BlobStore(
        StoreTopology(
            payload=BackendRef(name="filesystem", options={"base_dir": root}),
            authority=BackendRef(name="sqlite", options={"root": root}),
        ),
        cache_dir=root,
    )


@pytest.mark.parametrize(
    ("boundary", "committed_generation", "has_cleanup_debt", "has_intent"),
    (
        ("put.before_stage", "old", False, False),
        ("put.after_stage", "old", False, False),
        ("put.intent_prepared", "old", False, True),
        ("put.before_candidate_publish", "old", False, True),
        ("put.candidate_published", "old", False, True),
        ("put.candidate_verified", "old", False, True),
        ("put.before_promotion", "old", False, True),
        ("put.promoted", "new", True, True),
        ("cleanup.before_payload_delete", "new", True, True),
        ("cleanup.after_payload_delete", "new", True, True),
        ("put.cleanup_retired", "new", False, True),
    ),
)
def test_integrity_and_recovery_put_boundaries_converge_to_one_complete_generation(
    tmp_path: Path,
    boundary: str,
    committed_generation: str,
    has_cleanup_debt: bool,
    has_intent: bool,
) -> None:
    """Integrity/recovery faults preserve old-or-new authority state only."""
    root = tmp_path / boundary.replace(".", "-")
    seeded = _store(root)
    try:
        seeded.put({"generation": "old"}, key="fault-key")
        previous = seeded.lifecycle_authority.read_entry("fault-key")
        assert previous is not None
    finally:
        seeded.close()

    hooks = BoundaryHooks()
    active = _store(root)
    try:
        active.lifecycle.fault_hook = hooks.reach
        hooks.arm_fault(boundary)
        expected_error = (
            CacheBlobRecoverableCleanupError
            if boundary.startswith("cleanup.")
            else InjectedLifecycleFault
        )
        with pytest.raises(expected_error) as raised:
            active.put({"generation": "new"}, key="fault-key")
        if isinstance(raised.value, CacheBlobRecoverableCleanupError):
            assert isinstance(raised.value.__cause__, InjectedLifecycleFault)
        assert boundary in hooks.reached
    finally:
        active.close()

    reopened = _store(root)
    try:
        state = authority_whole_state(reopened.lifecycle_authority)
        current = reopened.lifecycle_authority.read_entry("fault-key")
        assert current is not None
        if committed_generation == "old":
            assert current == previous
            assert reopened.get("fault-key") == {"generation": "old"}
        else:
            assert current.generation != previous.generation
            assert reopened.get("fault-key") == {"generation": "new"}
        if has_intent:
            assert state.mutation_states
        if has_cleanup_debt:
            assert len(state.cleanup_debt) == 1
            # Recovery is authority-driven: it reclaims exact debt but does
            # not roll back the generation whose promotion already committed.
            assert reopened.reconcile(apply=True).applied is True
            assert reopened.get("fault-key") == {"generation": "new"}
            assert reopened.lifecycle_authority.pending_cleanup_debts() == ()
        else:
            assert state.cleanup_debt == ()
    finally:
        reopened.close()


@pytest.mark.parametrize(
    ("boundary", "tombstone_committed"),
    (
        ("delete.intent_prepared", False),
        ("delete.before_tombstone_promotion", False),
        ("cleanup.before_payload_delete", True),
        ("cleanup.after_payload_delete", True),
    ),
)
def test_recovery_delete_faults_are_intent_or_cleanup_debt_not_rollback(
    tmp_path: Path,
    boundary: str,
    tombstone_committed: bool,
) -> None:
    """Deletion faults retain attributed state and resume deterministically."""
    store = _store(tmp_path / boundary.replace(".", "-"))
    try:
        store.put({"generation": "old"}, key="delete-key")
        hooks = BoundaryHooks()
        store.lifecycle.fault_hook = hooks.reach
        hooks.arm_fault(boundary)
        expected_error = (
            CacheBlobRecoverableCleanupError
            if boundary.startswith("cleanup.")
            else InjectedLifecycleFault
        )
        with pytest.raises(expected_error):
            store.delete("delete-key")
        assert boundary in hooks.reached
        state = authority_whole_state(store.lifecycle_authority)
        if tombstone_committed:
            assert len(state.cleanup_debt) == 1
            assert store.get("delete-key") is None
        else:
            assert state.cleanup_debt == ()
            assert store.get("delete-key") == {"generation": "old"}
        store.lifecycle.fault_hook = None
        assert store.delete("delete-key") is True
        assert store.get("delete-key") is None
        assert store.lifecycle_authority.pending_cleanup_debts() == ()
    finally:
        store.close()


def test_progress_and_performance_fault_contract_has_no_success_deadline(
    tmp_path: Path,
) -> None:
    """Progress accepts declared outcomes; performance is not a correctness oracle."""
    store = _store(tmp_path / "progress-profile")
    try:
        profile = store.topology.qualified_profile
        assert profile is not None
        assert profile.requirements.progress_outcomes == {
            "success",
            "conflict",
            "retryable_timeout",
        }
        assert store.put({"generation": "one"}, key="progress-key") == "progress-key"
    finally:
        store.close()

