"""Public contracts for obstore-backed payload generation participants."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from obstore.store import LocalStore

from cacheness.storage import BlobStore
from cacheness.storage.composition import BackendRef, StoreTopology
from cacheness.storage.guarded_handler_io import GuardedHandlerIO
from cacheness.storage.memory_lifecycle_authority import InMemoryLifecycleAuthority
from cacheness.storage.obstore_generation_io import ObstoreGenerationIO


def test_local_store_provider_round_trips_native_npz_through_blob_store(
    tmp_path: Path,
) -> None:
    """A LocalStore participant keeps native NPZ handlers path-only and private."""
    payload_root = tmp_path / "payloads"
    provider = ObstoreGenerationIO(
        LocalStore(payload_root, mkdir=True),
        GuardedHandlerIO(payload_root),
        qualification_identity="memory",
    )
    store = BlobStore(
        StoreTopology(
            payload=BackendRef(instance=provider),
            authority=BackendRef(instance=InMemoryLifecycleAuthority()),
        ),
        cache_dir=tmp_path / "store",
    )
    value = np.asarray([1, 2, 3], dtype=np.int64)
    try:
        receipt = store.put_entry(value, key="local-npz")

        assert store.get(receipt.key).tolist() == [1, 2, 3]
    finally:
        store.close()
