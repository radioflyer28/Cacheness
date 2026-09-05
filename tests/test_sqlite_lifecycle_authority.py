"""Durable SQLite LifecycleAuthority contracts for Phase 3."""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import time

import pytest

from cacheness.config import LifecycleLimits
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleTimeoutError,
    CacheBlobMigrationRequiredError,
    CacheBlobStoreClosedError,
)
from cacheness.storage.lifecycle_authority import EntryExpectation, MutationSpec
from cacheness.storage.sqlite_lifecycle_authority import (
    AUTHORITY_RELATIVE_PATH,
    SQLITE_APPLICATION_ID,
    SCHEMA_VERSION,
    SqliteLifecycleAuthority,
)


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
    target = tmp_path / "target-root"
    target.mkdir()
    symlink_root.symlink_to(target, target_is_directory=True)
    with pytest.raises(CacheBlobMigrationRequiredError):
        SqliteLifecycleAuthority.for_root(symlink_root).prepare_mutation(_spec("symlink"))
    assert not (target / ".cacheness").exists()


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

