#!/usr/bin/env python3
"""Store and retrieve one object through the explicit memory BlobStore topology."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from cacheness.storage import BackendRef, BlobStore, StoreTopology


def memory_topology() -> StoreTopology:
    """Return the supported, same-process-only memory topology."""

    return StoreTopology(
        payload=BackendRef(name="memory"),
        authority=BackendRef(name="memory"),
    )


def main() -> None:
    """Commit and read one direct object-storage entry without a cache policy."""

    with TemporaryDirectory(prefix="cacheness-memory-blob-") as temporary:
        root = Path(temporary)
        store = BlobStore(memory_topology(), cache_dir=root)
        try:
            store.initialize()
            expected = {"owner": "Ada", "roles": ["maintainer"]}
            receipt = store.put_entry(
                expected,
                key="profile-ada",
                metadata={"purpose": "direct-object-storage"},
            )

            assert receipt.key == "profile-ada"
            assert store.get(receipt.key) == expected
        finally:
            store.close()

    print("MEMORY_BLOB_STORE_EXAMPLE_OK")


if __name__ == "__main__":
    main()
