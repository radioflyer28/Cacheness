# Add a custom file format

Use one store-local `FormatHandler` when an application has a native file
format that Cacheness does not already handle. This guide follows the complete,
executable [MCAP-style example](../examples/custom_mcap_format.py). It keeps
the handler responsible for encoding and decoding one format while `BlobStore`
continues to own payload publication and catalog lifecycle.

## The durable format contract

A handler declares three persisted payload identities:

- `data_type` identifies the handler's application-facing data kind.
- `payload_format` identifies the native on-disk container.
- `payload_format_version` identifies the container contract.

These persisted payload identities, rather than the Python protocol name or a
handler class name, determine whether a stored payload is compatible with a
reader. Choose stable values before storing data. The example uses
`example_mcap_bytes`, `example-mcap`, and version `1`.

The handler must also return one static, safe suffix. For an MCAP-style
container that suffix is `.mcap`; it is not selected from input data or a key.

## Implement the handler

Import the storage protocol and write only the private path supplied by the
store:

```python
from pathlib import Path
from typing import Any

from cacheness.storage import FormatHandler


class MCAPBytesHandler(FormatHandler):
    _HEADER = b"MCAP0"

    @property
    def data_type(self) -> str:
        return "example_mcap_bytes"

    @property
    def payload_format(self) -> str:
        return "example-mcap"

    @property
    def payload_format_version(self) -> int:
        return 1

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, bytes)

    def get_file_extension(self, config: Any) -> str:
        del config
        return ".mcap"

    def put(self, data: Any, file_path: Path, config: Any) -> dict[str, Any]:
        del config
        artifact = file_path.with_suffix(".mcap")
        artifact.write_bytes(self._HEADER + data)
        return {
            "storage_format": "mcap",
            "payload_format": self.payload_format,
            "payload_format_version": self.payload_format_version,
            "actual_path": str(artifact),
        }

    def get(self, file_path: Path, metadata: dict[str, Any]) -> bytes:
        del metadata
        payload = file_path.read_bytes()
        if not payload.startswith(self._HEADER):
            raise ValueError("unexpected MCAP payload header")
        return payload[len(self._HEADER) :]
```

`put()` receives a private staging path, not a managed payload locator. Write
the artifact below that stage and return its actual path. Before publication,
Cacheness verifies that the result is a contained regular file with the declared
safe suffix. It rejects a path outside the private stage, a directory, a
symlink, or an unexpected artifact. `get()` receives a private verified
snapshot path and reads only that file.

Do not create another client or transport layer in the handler. The format code
does not need an obstore object, a managed locator, catalog authority access, or
a second publication rule.

## Register on the store and assert a round trip

Create the store's topology as usual, initialize it, then register the handler
only on that store. `priority=0` makes this bytes handler run before the generic
object handler for this application instance.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from cacheness.storage import BackendRef, BlobStore, StoreTopology


def memory_topology() -> StoreTopology:
    return StoreTopology(
        payload=BackendRef(name="memory"),
        authority=BackendRef(name="memory"),
    )


with TemporaryDirectory(prefix="cacheness-mcap-") as temporary:
    store = BlobStore(memory_topology(), cache_dir=Path(temporary))
    try:
        store.initialize()
        store.handlers.register_handler(MCAPBytesHandler(), priority=0)

        expected = b"recording-payload"
        receipt = store.put_entry(expected, key="recording-001")
        assert store.get(receipt.key) == expected  # round trip
    finally:
        store.close()
```

That assertion is the minimum extension check: the store selected the handler,
the private staging artifact passed the contained regular-file boundary, and
the declared format read back to the original value. The runnable example adds
the same identity and suffix assertions.

## Keep the boundary small

Register each handler with `store.handlers.register_handler(...)`; a handler
does not become process-wide by default. Use a separate store if a different
application format selection or priority is needed. Keep format evolution
explicit by changing persisted identities/version only when readers and an
offline maintenance plan support the change.

For storage guarantees, payload bounds, and external qualification limits, see
[Release qualification](RELEASE_QUALIFICATION.md). This tutorial is solely the
path-based serialization/deserialization seam.
