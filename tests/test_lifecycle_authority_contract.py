"""Wave 0 compatibility and deterministic lifecycle-authority contracts."""

from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from _lifecycle_test_support import (
    BoundaryHooks,
    InjectedLifecycleFault,
    authority_root_snapshot,
    classify_authority_evidence,
    run_python_subprocess,
)
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobCloseTimeoutError,
    CacheBlobIntegrityError,
    CacheBlobLifecycleConflictError,
    CacheBlobReconciliationCheckpointError,
    CacheBlobReconciliationConflictError,
    CacheBlobReconciliationError,
    CacheBlobRecoverableCleanupError,
    CacheBlobStoreClosedError,
    CacheBlobLifecycleTimeoutError,
    CacheIntegrityError,
    CacheManifestIntegrityError,
    CacheReason,
    CacheStorageError,
)
from cacheness.storage.lifecycle_authority import EntryExpectation
from cacheness.storage.reconciliation import (
    ReconciliationAction,
    ReconciliationFinding,
    ReconciliationReport,
    ReconciliationStatus,
)


def test_lifecycle_reason_values_and_typed_error_bases_are_frozen() -> None:
    """Authority work cannot rename the direct BlobStore failure contract."""
    expected_reasons = {
        "BLOB_LIFECYCLE_CONFLICT": "blob_lifecycle_conflict",
        "BLOB_BACKEND_FAILURE": "blob_backend_failure",
        "BLOB_BACKEND_CAPABILITY_UNSUPPORTED": "blob_backend_capability_unsupported",
        "BLOB_MIGRATION_REQUIRED": "blob_migration_required",
        "BLOB_RECOVERABLE_CLEANUP": "blob_recoverable_cleanup",
        "BLOB_RECONCILIATION_BLOCKED": "blob_reconciliation_blocked",
        "BLOB_RECONCILIATION_CONFLICT": "blob_reconciliation_conflict",
        "BLOB_RECONCILIATION_CHECKPOINT_INVALID": "blob_reconciliation_checkpoint_invalid",
        "BLOB_STORE_CLOSED": "blob_store_closed",
        "BLOB_CLOSE_TIMEOUT": "blob_close_timeout",
        "BLOB_LIFECYCLE_TIMEOUT": "blob_lifecycle_timeout",
    }

    assert {name: CacheReason[name].value for name in expected_reasons} == expected_reasons
    assert issubclass(CacheBlobIntegrityError, CacheIntegrityError)
    assert issubclass(CacheManifestIntegrityError, CacheBlobIntegrityError)
    assert issubclass(CacheBlobLifecycleConflictError, CacheStorageError)
    assert issubclass(CacheBlobBackendError, CacheStorageError)
    assert issubclass(CacheBlobRecoverableCleanupError, CacheStorageError)
    assert issubclass(CacheBlobReconciliationError, CacheStorageError)
    assert issubclass(CacheBlobReconciliationConflictError, CacheBlobReconciliationError)
    assert issubclass(CacheBlobReconciliationCheckpointError, CacheBlobIntegrityError)
    assert issubclass(CacheBlobStoreClosedError, CacheStorageError)
    assert issubclass(CacheBlobCloseTimeoutError, CacheStorageError)
    assert issubclass(CacheBlobLifecycleTimeoutError, CacheStorageError)


def test_reconciliation_v1_dictionary_and_human_summary_are_exact() -> None:
    """The zero-argument dictionaries stay the legacy v1 compatibility view."""
    finding = ReconciliationFinding(
        status=ReconciliationStatus.SAFE,
        action=ReconciliationAction.DELETE_CANDIDATE,
        reason="candidate_unpublished",
        evidence_id="operation-1",
        evidence_digest="evidence-digest",
        manifest_digest="manifest-digest",
        key_fingerprint="key-fingerprint",
        locator_fingerprint="locator-fingerprint",
    )
    report = ReconciliationReport(
        findings=(finding,),
        resume_token="resume-1",
        applied=False,
        manifest_records_seen=2,
        operation_records_seen=3,
    )

    assert finding.to_dict() == {
        "status": "safe",
        "action": "delete_candidate",
        "reason": "candidate_unpublished",
        "evidence_id": "operation-1",
        "evidence_digest": "evidence-digest",
        "manifest_digest": "manifest-digest",
        "key_fingerprint": "key-fingerprint",
        "locator_fingerprint": "locator-fingerprint",
    }
    assert report.to_dict() == {
        "applied": False,
        "findings": [finding.to_dict()],
        "manifest_records_seen": 2,
        "operation_records_seen": 3,
        "resume_token": "resume-1",
        "human_summary": (
            "Reconciliation (dry-run): 1 finding(s); safe=1, blocked=0, "
            "requires_confirmation=0, resume=yes"
        ),
    }
    assert report.human_summary == report.to_dict()["human_summary"]


@pytest.mark.parametrize(
    ("artifact", "expected"),
    [
        (None, "empty"),
        ("payload.bin", "payload_without_authority"),
        ("cache_metadata.json", "metadata_without_authority"),
        ("provenance.json", "legacy_without_authority"),
        (".cacheness-inventory-v2", "scheduler_without_authority"),
        ("lifecycle-authority-v2.sqlite3", "future_authority"),
        ("lifecycle-authority-v1.sqlite3", "corrupt_authority"),
        ("mixed", "mixed_without_authority"),
    ],
)
def test_established_authority_missing_evidence_is_not_empty(
    tmp_path: Path, artifact: str | None, expected: str
) -> None:
    """Classification is read-only and keeps every migration/rebuild case distinct."""
    root = tmp_path / "store"
    if artifact is not None:
        root.mkdir()
        if artifact == "mixed":
            (root / "payload.bin").write_bytes(b"payload")
            (root / "cache_metadata.json").write_text("{}", encoding="utf-8")
        elif artifact == ".cacheness-inventory-v2":
            (root / artifact).mkdir()
        else:
            (root / artifact).write_bytes(b"unrecognized")

    before = authority_root_snapshot(root)
    assert classify_authority_evidence(root) == expected
    assert authority_root_snapshot(root) == before


def test_wrong_object_and_deterministic_boundary_observers_are_explicit(
    tmp_path: Path,
) -> None:
    """Ordering evidence uses hooks, never sleeps or timing races."""
    wrong_root = tmp_path / "not-a-directory"
    wrong_root.write_bytes(b"not a store")
    before = authority_root_snapshot(wrong_root)

    assert classify_authority_evidence(wrong_root) == "wrong_root_object"
    assert authority_root_snapshot(wrong_root) == before

    hooks = BoundaryHooks()
    observed: list[str] = []
    hooks.add_observer(observed.append)
    for boundary in (
        "authority.transaction.begin",
        "payload.stage",
        "payload.publish",
        "payload.verify",
        "payload.cleanup",
        "authority.transaction.end",
    ):
        hooks.reach(boundary)

    assert observed == [
        "authority.transaction.begin",
        "payload.stage",
        "payload.publish",
        "payload.verify",
        "payload.cleanup",
        "authority.transaction.end",
    ]

    hooks.arm_fault("payload.stage")
    with pytest.raises(InjectedLifecycleFault, match="payload.stage"):
        hooks.reach("payload.stage")

    subprocess_result = run_python_subprocess("-c", "print('lifecycle-probe')")
    assert subprocess_result.returncode == 0
    assert subprocess_result.stdout == "lifecycle-probe\n"


# =============================================================================
# Transactional authority tracer (Plan 03-02)
# =============================================================================


def test_sqlite_authority_tracer_persists_intent_promotes_and_reopens(
    tmp_path: Path,
) -> None:
    """One short authority transaction is the sole visibility switch."""
    from cacheness.storage.lifecycle_authority import (
        EntryExpectation,
        MutationSpec,
        VerificationProof,
    )
    from cacheness.storage.sqlite_lifecycle_authority import SqliteLifecycleAuthority

    root = tmp_path / "tracer-store"
    authority = SqliteLifecycleAuthority.for_root(root)
    assert not (root / ".cacheness" / "lifecycle-authority-v1.sqlite3").exists()

    prepared = authority.prepare_mutation(
        MutationSpec.create(
            operation_id="op-1",
            key="tracer-key",
            generation="generation-1",
            candidate_locator="generations/generation-1.trace",
            expected=EntryExpectation.absent(),
        )
    )
    assert authority.open_write_transactions == 0
    assert authority.read_entry("tracer-key") is None
    authority.record_verification(
        prepared,
        VerificationProof(digest="a" * 64, byte_size=7),
    )
    promoted = authority.promote_mutation(prepared)
    assert promoted.entry.key == "tracer-key"
    assert authority.open_write_transactions == 0
    authority.close()

    reopened = SqliteLifecycleAuthority.for_root(root)
    entry = reopened.read_entry("tracer-key")
    assert entry is not None
    assert entry.manifest == promoted.entry.manifest
    assert entry.generation == "generation-1"
    reopened.close()


def test_sqlite_authority_rejects_aba_stale_absence_preparation(tmp_path: Path) -> None:
    """An absence observed before create/delete cannot promote after ABA."""
    from cacheness.error_handling import CacheBlobLifecycleConflictError
    from cacheness.storage.lifecycle_authority import EntryExpectation, MutationSpec
    from cacheness.storage.sqlite_lifecycle_authority import SqliteLifecycleAuthority

    authority = SqliteLifecycleAuthority.for_root(tmp_path / "aba-store")
    stale = authority.prepare_mutation(
        MutationSpec.create(
            operation_id="stale",
            key="key",
            generation="stale-generation",
            candidate_locator="generations/stale.trace",
            expected=EntryExpectation.absent(),
        )
    )
    winner = authority.prepare_mutation(
        MutationSpec.create(
            operation_id="winner",
            key="key",
            generation="winner-generation",
            candidate_locator="generations/winner.trace",
            expected=EntryExpectation.absent(),
        )
    )
    authority.record_verification_for_test(winner)
    authority.promote_mutation(winner)
    authority.delete_entry("key", expected=authority.read_entry("key").expectation)

    authority.record_verification_for_test(stale)
    with pytest.raises(CacheBlobLifecycleConflictError):
        authority.promote_mutation(stale)


def test_authority_empty_inspection_is_lazy_and_zero_mutation(tmp_path: Path) -> None:
    """Absent compatible roots stay absent through read-only lifecycle calls."""
    from cacheness.storage import BlobStore
    from cacheness.storage.sqlite_lifecycle_authority import SqliteLifecycleAuthority

    root = tmp_path / "missing-parent" / "empty-store"
    before = authority_root_snapshot(root)
    store = BlobStore(root, lifecycle_authority=SqliteLifecycleAuthority.for_root(root))
    assert store.get("missing") is None
    assert store.get_metadata("missing") is None
    assert store.exists("missing") is False
    store.close()
    assert authority_root_snapshot(root) == before

    reopened = BlobStore(root, lifecycle_authority=SqliteLifecycleAuthority.for_root(root))
    assert reopened.get("missing") is None
    reopened.close()
    assert authority_root_snapshot(root) == before


def test_established_missing_authority_fails_unchanged(tmp_path: Path) -> None:
    """Payload evidence without authority is never silently interpreted as empty."""
    from cacheness.storage import BlobStore
    from cacheness.storage.sqlite_lifecycle_authority import SqliteLifecycleAuthority
    from cacheness.error_handling import CacheBlobMigrationRequiredError

    root = tmp_path / "established-store"
    root.mkdir()
    (root / "payload.bin").write_bytes(b"existing payload")
    before = authority_root_snapshot(root)
    store = BlobStore(root, lifecycle_authority=SqliteLifecycleAuthority.for_root(root))
    with pytest.raises(CacheBlobMigrationRequiredError):
        store.get("missing")
    store.close()
    assert authority_root_snapshot(root) == before


@pytest.mark.parametrize("adapter", ("memory", "sqlite"))
def test_common_authority_transition_contract(
    tmp_path: Path, adapter: str
) -> None:
    """Memory and SQLite share exact create/overwrite/conflict semantics."""
    from cacheness.error_handling import CacheBlobLifecycleConflictError
    from cacheness.storage.lifecycle_authority import (
        EntryExpectation,
        MutationSpec,
        VerificationProof,
    )
    from cacheness.storage.memory_lifecycle_authority import InMemoryLifecycleAuthority
    from cacheness.storage.sqlite_lifecycle_authority import SqliteLifecycleAuthority

    authority = (
        InMemoryLifecycleAuthority()
        if adapter == "memory"
        else SqliteLifecycleAuthority.for_root(tmp_path / "sqlite-contract")
    )
    first = authority.prepare_mutation(
        MutationSpec.create(
            operation_id="create",
            key="shared-key",
            generation="generation-1",
            candidate_locator="generations/one",
            expected=EntryExpectation.absent(),
            manifest=b"manifest-1",
        )
    )
    authority.record_verification(first, VerificationProof("1" * 64, 1))
    created = authority.promote_mutation(first).entry
    assert created.manifest == b"manifest-1"

    replacement = authority.prepare_mutation(
        MutationSpec.create(
            operation_id="overwrite",
            key="shared-key",
            generation="generation-2",
            candidate_locator="generations/two",
            expected=created.expectation,
            manifest=b"manifest-2",
        )
    )
    authority.record_verification(replacement, VerificationProof("2" * 64, 2))
    replaced = authority.promote_mutation(replacement).entry
    assert replaced.manifest == b"manifest-2"

    with pytest.raises(CacheBlobLifecycleConflictError):
        authority.delete_entry("shared-key", expected=created.expectation)
    authority.close()


def test_memory_authorities_are_isolated_and_truthful_about_capabilities() -> None:
    """Only injection of one memory authority can share same-process state."""
    from cacheness.storage.memory_lifecycle_authority import InMemoryLifecycleAuthority

    first = InMemoryLifecycleAuthority()
    second = InMemoryLifecycleAuthority()
    assert first.capabilities.durable is False
    assert first.capabilities.multiprocess is False
    assert first.read_entry("missing") is None
    assert second.read_entry("missing") is None
    first.close()
    second.close()


# =============================================================================
# Complete transition atomicity (Plan 03-03)
# =============================================================================


def _prepare_verified_mutation(
    authority: object,
    *,
    operation_id: str,
    expected: object,
    generation: str,
    locator: str,
):
    """Prepare one deterministic authority promotion through its public seam."""
    from cacheness.storage.lifecycle_authority import MutationSpec, VerificationProof

    prepared = authority.prepare_mutation(
        MutationSpec.create(
            operation_id=operation_id,
            key="atomic-key",
            generation=generation,
            candidate_locator=locator,
            expected=expected,
            manifest=f"manifest-{generation}".encode(),
        )
    )
    authority.record_verification(
        prepared,
        VerificationProof(digest=(generation[-1] * 64), byte_size=1),
    )
    return prepared


def test_sqlite_promotion_rolls_back_every_participating_authority_row(
    tmp_path: Path,
) -> None:
    """A promotion fault leaves entry, mutation, debt, projection, and revision intact."""
    from cacheness.storage.sqlite_lifecycle_authority import SqliteLifecycleAuthority

    authority = SqliteLifecycleAuthority.for_root(tmp_path / "atomic")
    first = _prepare_verified_mutation(
        authority,
        operation_id="first",
        expected=EntryExpectation.absent(),
        generation="generation-1",
        locator="generations/one",
    )
    previous = authority.promote_mutation(first).entry
    replacement = _prepare_verified_mutation(
        authority,
        operation_id="replacement",
        expected=previous.expectation,
        generation="generation-2",
        locator="generations/two",
    )
    before = authority.snapshot_state()
    hooks = BoundaryHooks()
    authority.set_transaction_hook_for_test(hooks.reach)
    hooks.arm_fault("promote.after_entry")

    with pytest.raises(InjectedLifecycleFault, match="promote.after_entry"):
        authority.promote_mutation(replacement)

    assert authority.read_entry("atomic-key") == previous
    assert authority.snapshot_state() == before

    promoted = authority.promote_mutation(replacement)
    after = authority.snapshot_state()
    assert promoted.entry.generation == "generation-2"
    assert promoted.cleanup_debt[0].locator == "generations/one"
    assert after.revision == before.revision + 1
    assert after.projection_dirty is True
    assert authority.promote_mutation(replacement) == promoted


def test_sqlite_classifies_an_uncertain_commit_by_reopening_exact_operation_state(
    tmp_path: Path,
) -> None:
    """A post-commit SQLite error returns the proven committed promotion, never retries."""
    from cacheness.storage.sqlite_lifecycle_authority import SqliteLifecycleAuthority

    authority = SqliteLifecycleAuthority.for_root(tmp_path / "uncertain")
    prepared = _prepare_verified_mutation(
        authority,
        operation_id="uncertain-operation",
        expected=EntryExpectation.absent(),
        generation="generation-1",
        locator="generations/one",
    )
    hooks = BoundaryHooks()
    authority.set_transaction_hook_for_test(hooks.reach)
    hooks.arm_fault("authority.transaction.committed", sqlite3.OperationalError("uncertain"))

    promoted = authority.promote_mutation(prepared)

    assert promoted.entry.generation == "generation-1"
    assert authority.read_entry("atomic-key") == promoted.entry
    assert authority.snapshot_state().mutation_states == (("uncertain-operation", "promoted"),)


@pytest.mark.parametrize("adapter", ("memory", "sqlite"))
def test_complete_clear_reconciliation_and_projection_transitions_use_authority_state(
    tmp_path: Path, adapter: str
) -> None:
    """Both adapters expose real complete-state methods rather than placeholders."""
    from cacheness.storage.memory_lifecycle_authority import InMemoryLifecycleAuthority
    from cacheness.storage.sqlite_lifecycle_authority import SqliteLifecycleAuthority

    authority = (
        InMemoryLifecycleAuthority()
        if adapter == "memory"
        else SqliteLifecycleAuthority.for_root(tmp_path / "complete-transitions")
    )
    prepared = _prepare_verified_mutation(
        authority,
        operation_id="create",
        expected=EntryExpectation.absent(),
        generation="generation-1",
        locator="generations/one",
    )
    authority.promote_mutation(prepared)

    clear = authority.begin_clear()
    assert tuple(entry.key for entry in authority.page_clear(clear)) == ("atomic-key",)
    authority.checkpoint_clear(clear)

    reconciliation = authority.begin_reconciliation()
    assert authority.page_reconciliation(reconciliation) == ()
    authority.checkpoint_reconciliation(reconciliation)

    revision = authority.snapshot_state().revision
    assert authority.compare_and_mark_projection(None).value == revision
    assert authority.snapshot_state().projection_dirty is False
