"""Durable SQLite LifecycleAuthority contracts for Phase 3."""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import time
import json

import pytest

from cacheness.config import LifecycleLimits
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleTimeoutError,
    CacheBlobMigrationRequiredError,
    CacheBlobStoreClosedError,
)
from cacheness.storage.lifecycle_authority import (
    EntryExpectation,
    MutationSpec,
    VerificationProof,
)
from cacheness.storage.sqlite_lifecycle_authority import (
    AUTHORITY_RELATIVE_PATH,
    SQLITE_APPLICATION_ID,
    SCHEMA_VERSION,
    SqliteLifecycleAuthority,
)


LIFECYCLE_BASELINE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "lifecycle_authority_baseline.json"
)


def _measured_busy_deadline() -> float:
    """Read the configured busy deadline from checked-in benchmark evidence."""
    with LIFECYCLE_BASELINE_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)["derived"]["configuration"][
            "authority_busy_timeout_seconds"
        ]


def _spec(operation_id: str = "operation-1") -> MutationSpec:
    """Create one bounded authority transition input."""
    return MutationSpec.create(
        operation_id=operation_id,
        key="authority-key",
        generation="generation-1",
        candidate_locator="generations/generation-1.native",
        expected=EntryExpectation.absent(),
        manifest=b"canonical-manifest",
    )


def _create_database(root: Path) -> Path:
    """Materialize one valid authority database through its public mutation seam."""
    authority = SqliteLifecycleAuthority.for_root(root)
    authority.prepare_mutation(_spec())
    authority.close()
    return root / AUTHORITY_RELATIVE_PATH


def test_sqlite_authority_reads_back_required_pragmas_and_identity(
    tmp_path: Path,
) -> None:
    """Every configured connection exposes the durable authority contract."""
    authority = SqliteLifecycleAuthority.for_root(tmp_path / "store")
    authority.prepare_mutation(_spec())

    diagnostics = authority.diagnostics()

    assert diagnostics["journal_mode"] == "delete"
    assert diagnostics["synchronous"] == "extra"
    assert diagnostics["foreign_keys"] is True
    assert diagnostics["trusted_schema"] is False
    assert diagnostics["application_id"] == SQLITE_APPLICATION_ID
    assert diagnostics["user_version"] == SCHEMA_VERSION
    assert len(diagnostics["store_identity"]) == 32
    assert diagnostics["integrity_check"] == ("ok",)
    assert diagnostics["foreign_key_check"] == ()


def test_sqlite_authority_reopens_exact_transport_evidence(tmp_path: Path) -> None:
    """One verified proof keeps its signed transport bytes through reopen and replay."""
    root = tmp_path / "transport-evidence"
    authority = SqliteLifecycleAuthority.for_root(root)
    spec = _spec("transport-evidence-operation")
    proof = VerificationProof(
        "a" * 64,
        len(spec.manifest),
        spec.manifest,
        b"signed-transport-evidence",
    )
    prepared = authority.prepare_mutation(spec)
    authority.record_verification(prepared, proof)
    promoted = authority.promote_mutation(prepared)
    authority.close()

    reopened = SqliteLifecycleAuthority.for_root(root)
    try:
        entry = reopened.read_entry(spec.key)
        replay = reopened.read_mutation(spec.operation_id)

        assert entry is not None
        assert entry.transport_evidence == proof.transport_evidence
        assert replay is not None
        assert replay.verification == proof
        assert replay.promotion == promoted
    finally:
        reopened.close()


def test_sqlite_schema_eight_requires_explicit_offline_migration(
    tmp_path: Path,
) -> None:
    """An older development authority remains untouched by an ordinary open."""
    root = tmp_path / "schema-eight"
    database = _create_database(root)
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA user_version = 8")
    connection.commit()
    connection.close()
    before = database.read_bytes()

    with pytest.raises(CacheBlobMigrationRequiredError):
        SqliteLifecycleAuthority.for_root(root).read_entry("authority-key")

    assert database.read_bytes() == before


@pytest.mark.parametrize("change", ("application_id", "future_version"))
def test_sqlite_authority_rejects_wrong_identity_without_mutating(
    tmp_path: Path, change: str
) -> None:
    """A wrong application identity or future schema is never adopted or rebuilt."""
    root = tmp_path / change
    database = _create_database(root)
    connection = sqlite3.connect(database)
    if change == "application_id":
        connection.execute("PRAGMA application_id = 123")
    else:
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    connection.commit()
    connection.close()
    before = database.read_bytes()

    authority = SqliteLifecycleAuthority.for_root(root)
    with pytest.raises(CacheBlobMigrationRequiredError):
        authority.read_entry("authority-key")

    assert database.read_bytes() == before


def test_sqlite_authority_rejects_reserved_symlink_and_wrong_objects_unchanged(
    tmp_path: Path,
) -> None:
    """Only a contained non-symlink directory and regular database are eligible."""
    root = tmp_path / "store"
    root.mkdir()
    reserved = root / ".cacheness"
    reserved.write_bytes(b"not a directory")
    before = reserved.read_bytes()

    with pytest.raises(CacheBlobMigrationRequiredError):
        SqliteLifecycleAuthority.for_root(root).prepare_mutation(_spec())
    assert reserved.read_bytes() == before

    if not hasattr(os, "symlink"):
        pytest.skip("symlink support is unavailable")
    symlink_root = tmp_path / "symlink-root"
    symlink_root.mkdir()
    target = tmp_path / "reserved-target"
    target.mkdir()
    (symlink_root / ".cacheness").symlink_to(target, target_is_directory=True)
    with pytest.raises(CacheBlobMigrationRequiredError):
        SqliteLifecycleAuthority.for_root(symlink_root).prepare_mutation(_spec("symlink"))
    assert not (target / "lifecycle-authority-v1.sqlite3").exists()


def test_sqlite_authority_uses_one_absolute_busy_deadline_and_preserves_cause(
    tmp_path: Path,
) -> None:
    """An independently held writer maps to a typed deadline-bounded timeout."""
    root = tmp_path / "contended"
    database = _create_database(root)
    blocker = sqlite3.connect(database, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")
    limits = LifecycleLimits(authority_busy_timeout_seconds=0.04)
    authority = SqliteLifecycleAuthority.for_root(root, lifecycle_limits=limits)
    started = time.monotonic()
    try:
        with pytest.raises(CacheBlobLifecycleTimeoutError) as captured:
            authority.prepare_mutation(_spec("contended-operation"))
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
    elapsed = time.monotonic() - started

    assert elapsed < 0.2
    assert isinstance(captured.value.__cause__, sqlite3.OperationalError)
    assert authority.open_write_transactions == 0


def test_default_busy_deadline_is_the_caller_policy_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default is caller policy, not a benchmark-derived contract."""
    authority = SqliteLifecycleAuthority.for_root(tmp_path / "default-deadline")
    monkeypatch.setattr(
        "cacheness.storage.sqlite_lifecycle_authority.time.monotonic",
        lambda: 100.0,
    )

    assert authority._deadline(None) == 100.0 + 5.0


def test_sqlite_authority_rolls_back_every_row_for_a_before_commit_fault(
    tmp_path: Path,
) -> None:
    """A deterministic in-transaction observer fault leaves no partial intent."""
    authority = SqliteLifecycleAuthority.for_root(tmp_path / "rollback")
    before = authority.snapshot_state()

    def fail_before_commit(boundary: str) -> None:
        if boundary == "authority.transaction.before_commit":
            raise RuntimeError("force transactional rollback")

    authority.set_transaction_hook_for_test(fail_before_commit)
    with pytest.raises(RuntimeError, match="force transactional rollback"):
        authority.prepare_mutation(_spec("rollback-operation"))

    assert authority.snapshot_state() == before
    assert authority.read_entry("authority-key") is None
    assert authority.pending_mutations() == ()
    assert authority.pending_cleanup_debts() == ()


def test_sqlite_authority_rejects_inherited_process_use_and_close_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Authority resources remain owned by their creating process and instance."""
    authority = SqliteLifecycleAuthority.for_root(tmp_path / "ownership")
    monkeypatch.setattr(authority, "_owner_pid", os.getpid() - 1)
    with pytest.raises(CacheBlobBackendError):
        authority.prepare_mutation(_spec())

    fresh = SqliteLifecycleAuthority.for_root(tmp_path / "fresh")
    fresh.prepare_mutation(_spec())
    fresh.close()
    fresh.close()
    with pytest.raises(CacheBlobStoreClosedError):
        fresh.read_entry("authority-key")


def test_sqlite_authority_rejects_malformed_row_values_before_entry_exposure(
    tmp_path: Path,
) -> None:
    """Malformed indexed row data is not converted into a usable authority entry."""
    root = tmp_path / "malformed"
    database = _create_database(root)
    connection = sqlite3.connect(database)
    connection.execute(
        "INSERT INTO entries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("bad", "generation", "locator", b"manifest", "0" * 64, -1, 1, None),
    )
    connection.commit()
    connection.close()

    with pytest.raises(CacheBlobBackendError):
        SqliteLifecycleAuthority.for_root(root).read_entry("bad")
