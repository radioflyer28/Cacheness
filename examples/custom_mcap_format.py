#!/usr/bin/env python3
"""Register a store-local MCAP-style native format without a second I/O surface."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from cacheness.storage import BackendRef, BlobStore, FormatHandler, StoreTopology


class MCAPBytesHandler(FormatHandler):
    """A compact MCAP-style bytes handler with stable stored identities."""

    _HEADER = b"MCAP0"

    @property
    def data_type(self) -> str:
        """Return the persisted handler identifier for this application format."""

        return "example_mcap_bytes"

    @property
    def payload_format(self) -> str:
        """Return the stable native container identity independent of class names."""

        return "example-mcap"

    @property
    def payload_format_version(self) -> int:
        """Return the first Cacheness payload contract for the example container."""

        return 1

    def can_handle(self, data: Any) -> bool:
        """Claim raw bytes before the generic object handler is selected."""

        return isinstance(data, bytes)

    def get_file_extension(self, config: Any) -> str:
        """Publish exactly one static, safe suffix for the staging boundary."""

        del config
        return ".mcap"

    def put(self, data: Any, file_path: Path, config: Any) -> dict[str, Any]:
        """Write only the handler-provided private stage and return its artifact."""

        del config
        if not isinstance(data, bytes):
            raise TypeError("MCAPBytesHandler accepts bytes only")
        artifact = file_path.with_suffix(".mcap")
        artifact.write_bytes(self._HEADER + data)
        return {
            "storage_format": "mcap",
            "payload_format": self.payload_format,
            "payload_format_version": self.payload_format_version,
            "file_size": artifact.stat().st_size,
            "actual_path": str(artifact),
            "metadata": {"header": self._HEADER.decode("ascii")},
        }

    def get(self, file_path: Path, metadata: dict[str, Any]) -> bytes:
        """Read only the private verified snapshot supplied by guarded handler I/O."""

        del metadata
        payload = file_path.read_bytes()
        if not payload.startswith(self._HEADER):
            raise ValueError("MCAP payload has an unexpected header")
        return payload[len(self._HEADER) :]


def memory_topology() -> StoreTopology:
    """Build the disposable topology used to demonstrate one custom format."""

    return StoreTopology(
        payload=BackendRef(name="memory"),
        authority=BackendRef(name="memory"),
    )


def main() -> None:
    """Register the handler on one store and round-trip MCAP-style bytes."""

    with TemporaryDirectory(prefix="cacheness-mcap-") as temporary:
        root = Path(temporary)
        store = BlobStore(memory_topology(), cache_dir=root)
        try:
            store.initialize()
            handler = MCAPBytesHandler()
            assert handler.data_type == "example_mcap_bytes"
            assert handler.payload_format == "example-mcap"
            assert handler.payload_format_version == 1
            assert handler.get_file_extension(store.config) == ".mcap"
            store.handlers.register_handler(handler, priority=0)
            expected = b"recording-payload"
            receipt = store.put_entry(expected, key="recording-001")

            assert store.get(receipt.key) == expected
        finally:
            store.close()

    print("CUSTOM_MCAP_FORMAT_EXAMPLE_OK")


if __name__ == "__main__":
    main()
