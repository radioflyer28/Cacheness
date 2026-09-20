"""Participant-only payload inventory contracts for reconciliation."""

from __future__ import annotations

from pathlib import Path

from obstore.store import MemoryStore

from cacheness.storage import BackendRef, BlobStore, StoreTopology
from cacheness.storage.guarded_handler_io import GuardedHandlerIO
from cacheness.storage.memory_lifecycle_authority import InMemoryLifecycleAuthority
from cacheness.storage.obstore_generation_io import (
    ObstoreGenerationIO,
    ObstoreInventoryPage,
)


class _DeterministicPostgresqlAuthority(InMemoryLifecycleAuthority):
    """Test-only authority identity; this is not service qualification evidence."""

    qualification_identity = "postgresql"


class _SharedKeyProvider:
    """Supply the explicit signing key required by the remote-shaped topology."""

    def get_key(self) -> bytes:
        return b"p" * 32


class _PagedParticipant(ObstoreGenerationIO):
    """Return opaque bounded inventory cursors without exposing lifecycle mutation."""

    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True)
        super().__init__(
            MemoryStore(), GuardedHandlerIO(root), qualification_identity="s3"
        )
        self.inventory_calls = 0

    def inventory_page(self, continuation: str | None = None, *, max_objects: int | None = None):
        del max_objects
        self.inventory_calls += 1
        return ObstoreInventoryPage(
            objects=(), next_offset=None if continuation is not None else "next-page"
        )


def test_reconciliation_preserves_participant_inventory_cursor_as_report_only(
    tmp_path: Path,
) -> None:
    """A participant cursor is resumable evidence, never lifecycle authority."""
    participant = _PagedParticipant(tmp_path / "private-stage")
    store = BlobStore(
        StoreTopology(
            payload=BackendRef(instance=participant, transfer_ownership=True),
            authority=BackendRef(
                instance=_DeterministicPostgresqlAuthority(), transfer_ownership=True
            ),
        ),
        cache_dir=tmp_path / "store",
        manifest_key_provider=_SharedKeyProvider(),
    )
    try:
        store.initialize()
        before = store.lifecycle_authority.snapshot_state()

        report = store.reconcile()

        assert report.inventory_cursor == "next-page"
        assert report.resume_token is not None
        assert participant.inventory_calls == 1
        assert store.lifecycle_authority.snapshot_state() == before
    finally:
        store.close()
