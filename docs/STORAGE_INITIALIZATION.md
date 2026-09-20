# Initialize a local store deliberately

Initialization is an application boundary, not a hidden side effect of a
shared first request. Before workers use a durable local store, one initializer
creates or validates its current layout. Ordinary opens then validate the
existing root; they do not silently bootstrap a new shared store, migrate a
layout, or adopt unexplained files.

## Initialize before sharing

For the qualified local topology, pair a filesystem payload participant with a
SQLite authority:

```python
from pathlib import Path

from cacheness.storage import BackendRef, BlobStore, StoreTopology


def local_topology(root: Path) -> StoreTopology:
    return StoreTopology(
        payload=BackendRef(name="filesystem", options={"base_dir": root}),
        authority=BackendRef(name="sqlite", options={"root": root}),
    )


root = Path("./objects")
initializer = BlobStore(local_topology(root), cache_dir=root)
try:
    initializer.initialize()  # Once, before independent workers begin.
finally:
    initializer.close()

# A worker validates and uses the same initialized root.
worker = BlobStore(local_topology(root), cache_dir=root)
try:
    worker.initialize()
    worker.put_entry({"answer": 42}, key="result")
finally:
    worker.close()
```

The memory-plus-memory topology is ephemeral and limited to one process. Do
not use it as a substitute for a durable local catalog.

## What initialization does not do

`initialize()` creates a fresh current store or validates a current one. It
does not silently upgrade an existing layout, choose a new backend, repair an
unknown root, or start an offline maintenance operation. An unsupported or
incomplete store fails explicitly; stop ordinary workers and use the
[migration and rebuild guide](STORAGE_MIGRATION.md) to decide what happens
next.

The current supported SQLite authority boundary is
`.cacheness/lifecycle-authority-v2.sqlite3` with application ID `0x43414348`
and SQLite `user_version = 9`. This database schema identifier is separate from
the public store and payload-format versions: validation does not turn it into
an ordinary-open migration switch.

Run initialization before creating independent workers. Concurrent first
creation is not an availability guarantee. Under supported contention, a typed
retryable result can be correct when it preserves the store's integrity and
recovery boundary.

## Storage authority and policy boundary

In the local durable topology, the SQLite authority selects the visible
generation. Immutable payload creation happens before promotion, and cleanup
after promotion is recorded for reconciliation. This is crash-consistent
recovery across the filesystem boundary, not a cross-resource ACID transaction.

`UnifiedCache` is optional policy over one `BlobStore`. A cache constructed
from a topology creates and initializes its private store when its own
`initialize()` is called. When an application injects a direct `BlobStore`, the
application keeps responsibility for that store's initialization and close
boundary. Use separate roots or namespaces when direct persistence and cache
policy have different retention needs.

## Respond to failures without creating another authority

Treat malformed metadata, unsafe paths, integrity errors, and incompatible
layouts as typed failures. Do not infer lifecycle state from filesystem paths,
payload presence, timing, or a process-local lock. Preserve the root and its
sidecar files while investigating; an unexplained payload is not a visible
entry.

For the complete guarantee matrix and retained nonclaims, use
[release qualification](RELEASE_QUALIFICATION.md). For the storage maintenance
workflow, use [migration and rebuild](STORAGE_MIGRATION.md).
