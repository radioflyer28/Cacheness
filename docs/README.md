# Cacheness documentation

Start with the [local-ready gateway](../README.md). The guides below are
organized by the work you want to do; each uses the current explicit API.

## Store objects

- [Direct BlobStore and catalog metadata](BLOB_STORE.md) — store, query, update,
  close, and reopen local objects.

## Cache function results

- [UnifiedCache policy](CACHE_POLICY.md) — layer TTL, lookup outcomes, removal,
  and `@cached(cache=cache)` over one selected store.

## Add a file format

- [FormatHandler tutorial](PLUGIN_DEVELOPMENT.md) — register one custom format
  on the store that uses it.

## Operate or migrate a store

- [Initialization and failure boundaries](STORAGE_INITIALIZATION.md) — create or
  validate a local store deliberately before workers share it.
- [Offline migration and rebuild](STORAGE_MIGRATION.md) — inspect, copy/verify,
  switch, or explicitly rebuild with workers stopped.

## Reference

- [API reference](API_REFERENCE.md)
- [Security guide](SECURITY.md)
- [Release qualification](RELEASE_QUALIFICATION.md) — the sole detailed owner
  of topology, platform, payload-bound, performance, and publication claims.

Configuration follows the job it affects: construct a direct store from the
topology in [Direct BlobStore and catalog metadata](BLOB_STORE.md), or add
cache policy with [UnifiedCache policy](CACHE_POLICY.md). Use the qualification
guide to determine which topology claims have evidence for this checkout.

The [example directory](../examples/README.md) contains the exact executable
local journeys used by the quality workflow.
