# Obstore Replacement Inventory

This inventory is responsibility-based. Line counts are context, not a claim
that every line in a listed module disappears.

| Current responsibility | Current location | Obstore replacement | Cacheness responsibility retained |
|---|---|---|---|
| Filesystem byte/stream CRUD, stat, listing, and exclusive object creation | `storage/backends/blob_backends.py::FilesystemBlobBackend`; portions of `storage/path_security.py::ManagedFileOps` | `LocalStore.put/get/head/delete/list` and streaming response/input APIs | Root configuration, managed locator validation, topology declaration, domain errors, durability policy |
| In-process byte dictionary and separate `InMemoryHandlerIO` lifecycle adapter | `storage/backends/blob_backends.py::InMemoryBlobBackend`, `InMemoryHandlerIO` | `MemoryStore` behind the same participant adapter used for local/S3 | Ephemeral topology declaration, limits, domain errors; MemoryStore necessarily owns payload-sized RAM |
| S3 requests, conditional single PUT, multipart upload orchestration, part upload/complete/abort, downloads, head, delete, and object listing | `storage/backends/s3_backend.py::_S3GenerationIO` | `S3Store.put/get/head/delete/list`; automatic multipart upload; configured multipart copy-if-absent where chosen | Bucket/prefix policy, credentials policy, locator validation, digests and byte limits, exception normalization, cleanup debt, bounded inventory/reconciliation evidence |
| Handler private temp directory, artifact containment, symlink rejection, descriptor/inode validation, safe suffix, result normalization | `storage/guarded_handler_io.py` | Not replaced | Retain and make backend-neutral; give obstore the already validated open descriptor, and materialize remote reads into private suffix-preserving snapshots |
| Filesystem publication and snapshot implementation inside guarded handler I/O | `GuardedHandlerIO.publish_generation`, `delete_or_prove_absent`, `open_snapshot`; `ManagedFileOps` calls | One backend-neutral obstore participant for local, memory, and S3 | Pre/postcondition checks, exact absence proof, integrity verification, handler snapshot lifetime |
| Lifecycle intent, visibility/promotion, cleanup debt, reconciliation | `storage/lifecycle.py`, lifecycle authorities, `storage/reconciliation.py` | Not replaced | Must remain solely authority-owned under ADR 0001; obstore is evidence/effect, never visibility |
| Path-based format serialization, deserialization, and custom registration | `interfaces.py`, `handlers.py::HandlerRegistry` | Not replaced | Preserve unchanged; explicitly reject lazy path-retaining and multi-file handlers |

## Important non-replacements

- Obstore does not validate Cacheness's managed namespace or handler suffixes.
- Obstore does not provide metadata-plus-payload atomicity.
- Obstore's delete API has no e-tag/version condition.
- Direct conditional S3 put is non-multipart and therefore materializes its
  input. Bounded-memory conditional publication requires the more complex
  temporary multipart object plus configured multipart copy-if-absent path.
- Multipart upload/copy can leave hidden abandoned uploads, so an S3 lifecycle
  rule and/or reconciliation inventory remains necessary.

