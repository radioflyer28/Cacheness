# Cacheness Examples

These are the supported local journeys for the current development version.
Each script creates private temporary storage, asserts its own result, and
leaves no files behind. Continuous integration runs these exact files unchanged.

## Canonical journeys

- [In-memory BlobStore](memory_blob_store.py) stores and retrieves an object
  through an in-memory payload and catalog topology.
- [Durable catalog store](durable_catalog_store.py) uses local filesystem
  payloads with a SQLite lifecycle catalog and caller-supplied catalog metadata.
- [UnifiedCache policy](unified_cache.py) shows explicit cache operations and
  the `@cached` decorator over the BlobStore-backed cache engine.
- [Custom MCAP-style format](custom_mcap_format.py) registers a store-local
  `FormatHandler` with stable native identities and a contained `.mcap` suffix.

Run an example from a checked-out repository after the project environment is
ready with `uv`. These examples demonstrate qualified local behavior only; see
the repository's release qualification guide for topology and platform limits.
