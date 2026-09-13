---
id: SEED-004
status: dormant
planted: 2026-09-13
planted_during: Phase 07.1 obstore-payload-participant-unification
trigger_when: when relevant
scope: unknown
---

# SEED-004: Reintroduce byte-preserving BlobStore file import/export helpers with handler-bound metadata and copy/move semantics

## Why This Matters

Existing datasets should be easy to ingest into Cacheness and export back to the
filesystem without applications loading them into Python data structures or
rewriting their native file formats. The earlier storage design contemplated
file-oriented transfer helpers, but the current public `BlobStore` exposes only
object-oriented `put()` and `get()` operations.

The selected handler still matters as persisted interpretation metadata: an
imported Parquet, NPZ, Blosc2, MCAP, or custom-format file must retain the stable
handler identity and native format information needed for a later `get()` to
deserialize a verified snapshot correctly. The handler should not serialize or
rewrite bytes during import, and no handler should run during byte-preserving
export.

## When to Surface

**Trigger:** when relevant

Surface when planning the public file-transfer API, large-dataset ingestion and
export, or the first post-obstore storage ergonomics phase. The implementation
should build on the single obstore payload-participant seam after Phase 07.1,
without reopening backend-specific filesystem/S3 mechanics.

This seed will surface during `$gsd-new-milestone` when the milestone scope matches.

## Scope Estimate

**Unknown** — run `$gsd-capture --seed --enrich SEED-004` to estimate effort.

## Breadcrumbs

- `src/cacheness/storage/blob_store.py` — public lifecycle facade where import/export operations should live.
- `src/cacheness/storage/guarded_handler_io.py` — retained-descriptor staging and verified suffix-preserving snapshots.
- `src/cacheness/storage/read_contract.py` — authenticated entry and receipt contracts.
- `.planning/phases/07.1-obstore-payload-participant-unification/07.1-CONTEXT.md` — locks handler identity, path containment, and the unified obstore participant boundary.
- `.planning/phases/07.1-obstore-payload-participant-unification/07.1-03-PLAN.md` — bounded stream publication and snapshot mechanics that can support the helpers.
- `docs/DEVELOPMENT_PLANNING.md` in historical commits such as `a22f4b4` — prior `write_blob_from_file` / `read_blob_to_file` design sketches.

## Notes

Required capability matrix:

| Direction | Non-destructive | Destructive |
| --- | --- | --- |
| Filesystem into BlobStore | copy/import file | move/import file, deleting the source only after committed storage success |
| BlobStore out to filesystem | copy/export file | move/export file, deleting the exact stored generation only after a durable destination succeeds |

- Transfer file bytes exactly and in bounded memory; do not parse, deserialize,
  or reserialize merely to move a file.
- Require or otherwise unambiguously resolve a registered handler on import and
  persist its stable `data_type`, storage format, and validated native suffix so
  ordinary `get()` can later invoke the correct handler.
- Export should use the committed entry's authenticated handler/format metadata
  to select or validate the destination suffix, but should copy the verified
  payload snapshot without invoking handler deserialization.
- Route every operation through `BlobStore`, `AuthorityLifecycleEngine`, and the
  configured payload participant. Never expose or copy directly to managed local
  paths, S3 keys, or obstore objects.
- Preserve immutable-generation and compare-and-swap semantics. A destructive
  export must delete only the exact generation that was exported; a concurrent
  replacement must survive.
- Define overwrite refusal, destination atomic publication, source deletion
  failure, retry/idempotency, returned receipt/result, metadata/catalog inputs,
  original filename handling, suffix validation, and cleanup behavior explicitly.
- Keep object-oriented `put()` / `get()` unchanged. These helpers are a first-class
  file-transfer surface, not a second lifecycle authority or a backend escape hatch.

