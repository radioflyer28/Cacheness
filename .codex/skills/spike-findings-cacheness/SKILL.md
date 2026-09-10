---
name: spike-findings-cacheness
description: Implementation blueprint from verified obstore payload-participant spikes for Cacheness. Auto-load for storage lifecycle, handler I/O, LocalStore, MemoryStore, S3, immutable publication, reconciliation, deletion, or streaming work.
---

<context>
## Project: cacheness

Evaluate whether `obstore` can replace Cacheness's backend-specific payload
mechanics behind one unified payload-participant seam while preserving the
existing path-based handler ecosystem and ADR 0001's single-authority lifecycle.

Spike session wrapped: 2026-09-10
</context>

<requirements>
## Requirements

- Preserve `CacheHandler.put(data, Path, config)` and
  `CacheHandler.get(Path, metadata)` for built-in and user-registered handlers.
- Keep handler staging private; handlers never receive managed filesystem or
  object-store locators.
- Publish payloads at deterministic, immutable generation locators using an
  atomic create-if-absent primitive.
- Treat payload publication atomicity separately from metadata-plus-payload
  crash consistency; metadata authority remains the visibility point.
- Enforce safe suffixes and normalized locators below one managed namespace.
- Delete exact immutable generations and rely on authority-owned cleanup debt
  and reconciliation rather than object presence or listings.
</requirements>

<findings_index>
## Feature Areas

| Area | Reference | Key Finding |
|------|-----------|-------------|
| Handler integration | `references/handler-integration.md` | Existing path-based built-ins and custom MCAP handlers work through private staging and snapshots without learning about obstore. |
| Lifecycle and recovery | `references/lifecycle-and-recovery.md` | Obstore provides atomic payload effects; SQLite/PostgreSQL authority still owns visibility, intent, and reconciliation. |
| Backend mechanics | `references/backend-mechanics.md` | Obstore can replace most Local/Memory/S3 mechanics, but strict conditional publication conflicts with bounded-memory direct uploads. |

## Source Files

Original spike source files are preserved in `sources/` for complete reference.
</findings_index>

<metadata>
## Processed Spikes

- 001-handler-boundary
- 002-immutable-publication
- 003-authority-boundary
- 004-containment-deletion
- 005-streaming-replacement
</metadata>

