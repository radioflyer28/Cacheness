"""Tier-aware contracts for BlobStore payload-generation participants.

The assertions intentionally distinguish integrity, recovery, progress, and
performance.  They exercise the five-method participant seam used by
``BlobStore`` without making a second lifecycle coordinator in the test.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from cacheness.storage import BlobStore
from cacheness.storage.composition import (
    BackendRef,
    PayloadGenerationIOProvider,
    StoreTopology,
)
from cacheness.storage.integrity import sha256_and_size
from cacheness.storage.manifest import BlobManifest, verify_current_manifest
from cacheness.storage.obstore_generation_io import ObstoreGenerationIO


@pytest.fixture(params=("filesystem", "memory"), ids=("filesystem", "memory"))
def local_store(request: pytest.FixtureRequest, tmp_path: Path):
    """Build one of the two locally qualified topologies at its true tier."""
    payload_name = str(request.param)
    if payload_name == "filesystem":
        topology = StoreTopology(
            payload=BackendRef(
                name="filesystem", options={"base_dir": tmp_path / "payloads"}
            ),
            authority=BackendRef(name="sqlite", options={"root": tmp_path / "store"}),
        )
    else:
        topology = StoreTopology(
            payload=BackendRef(name="memory"),
            authority=BackendRef(name="memory"),
        )
    store = BlobStore(topology, cache_dir=tmp_path / "store")
    try:
        yield payload_name, store
    finally:
        store.close()


def _stage_and_publish(
    store: BlobStore,
    *,
    value: object,
    locator: Path,
) -> dict[str, object]:
    """Publish one handler-owned native file through the selected participant."""
    handler = store.handlers.get_handler(value)
    guarded_io = store._materialize_authority_store()
    with guarded_io.stage(handler, value, store.config) as staged:
        return guarded_io.publish_generation(staged, locator)


def test_integrity_payload_generation_participants_publish_immutable_verified_snapshots(
    local_store: tuple[str, BlobStore],
) -> None:
    """Each local participant exposes only immutable, private native payloads."""
    payload_name, store = local_store
    provider = store.payload_backend
    guarded_io = store._materialize_authority_store()
    locator = Path("generations") / "participant-contract" / "native-generation"
    value = {"payload": payload_name, "generation": 1}

    assert isinstance(provider, PayloadGenerationIOProvider)
    assert isinstance(provider, ObstoreGenerationIO)
    assert store.guarded_handler_io is guarded_io
    assert all(
        callable(getattr(guarded_io, method, None))
        for method in (
            "stage",
            "publish_generation",
            "open_snapshot",
            "delete_or_prove_absent",
            "close",
        )
    )
    assert store.topology.qualified_profile is not None
    assert store.topology.qualified_profile.payload_identity == payload_name

    published = _stage_and_publish(store, value=value, locator=locator)
    with guarded_io.open_snapshot(locator, dict(published.get("metadata", {}))) as snapshot:
        first_bytes = snapshot.path.read_bytes()
        digest, byte_size = sha256_and_size(snapshot.path)
        assert stat.S_IMODE(snapshot.path.stat().st_mode) == 0o600
        assert snapshot.path != locator
        assert byte_size == published["file_size"]
        assert len(digest) == 64

    # Integrity: publication is exclusive.  Reusing an immutable generation
    # locator may fail, but it must never replace the already published bytes.
    with pytest.raises(FileExistsError):
        _stage_and_publish(
            store,
            value={"payload": payload_name, "generation": 2},
            locator=locator,
        )
    with guarded_io.open_snapshot(locator, dict(published.get("metadata", {}))) as snapshot:
        assert snapshot.path.read_bytes() == first_bytes

    # The real engine must select this exact participant and authenticate its
    # signed manifest before passing the private snapshot to the handler.
    receipt = store.put_entry(value, key=f"{payload_name}-round-trip")
    committed = store.lifecycle_authority.read_entry(receipt.key)
    assert committed is not None
    manifest = BlobManifest.from_canonical_bytes(committed.manifest)
    verify_current_manifest(manifest, store._authority_manifest_key())
    with guarded_io.open_snapshot(manifest.locator, manifest.handler_metadata) as snapshot:
        snapshot_digest, snapshot_size = sha256_and_size(snapshot.path)
        assert (snapshot_digest, snapshot_size) == (manifest.digest, manifest.byte_size)
    assert store.get(receipt.key) == value


def test_recovery_payload_generation_delete_is_exact_and_idempotent(
    local_store: tuple[str, BlobStore],
) -> None:
    """Recovery: cleanup removes only one exact immutable generation or proves absence."""
    _payload_name, store = local_store
    guarded_io = store._materialize_authority_store()
    locator = Path("generations") / "participant-contract" / "cleanup-generation"

    published = _stage_and_publish(store, value={"cleanup": True}, locator=locator)
    with guarded_io.open_snapshot(locator, dict(published.get("metadata", {}))) as snapshot:
        assert snapshot.path.is_file()

    guarded_io.delete_or_prove_absent(locator)
    guarded_io.delete_or_prove_absent(locator)
    with pytest.raises(FileNotFoundError):
        with guarded_io.open_snapshot(locator, dict(published.get("metadata", {}))):
            pass


def test_progress_and_performance_are_declared_without_universal_deadlines(
    local_store: tuple[str, BlobStore],
) -> None:
    """Progress is profile-specific; this contract makes no timing assertion."""
    payload_name, store = local_store
    profile = store.topology.qualified_profile
    assert profile is not None

    if payload_name == "filesystem":
        assert profile.requirements.progress_outcomes == {
            "success",
            "conflict",
            "retryable_timeout",
        }
        assert store.capabilities.durable is True
    else:
        assert profile.requirements.progress_outcomes == {"success", "conflict"}
        assert store.capabilities.durable is False
