---
created: 2026-04-03T20:10:58.493Z
title: Tiered pull-through cache
area: general
files:
  - docs/FUTURE_IMPROVEMENTS.md
  - docs/ARCHITECTURE.md
---

## Problem

Users sharing caches across teams need a way to compose a fast local cache backed by a remote shared store. Currently, Cacheness only supports a single UnifiedCache instance — either local or remote, not both.

## Solution

Implement a `TieredCache` class (~200 LOC) that composes two `UnifiedCache` instances:
- **Local tier:** SQLite + filesystem (microsecond reads, LRU eviction, size-capped)
- **Remote tier:** PostgreSQL/libSQL + S3 (durable, shared, signed)

Pull-through on miss: `local.get()` → miss → `remote.get()` → `local.put()` → return.

Invalidation strategies: TTL-based, metadata-version check (cheap with libSQL embedded replicas), or no invalidation.

For storage mode users, this becomes a local workspace pattern — work locally, persist to remote. Like git's local/remote model.

See FUTURE_IMPROVEMENTS.md section 9 and ARCHITECTURE.md "Tiered Cache Composition" for full design.

### Prerequisites from 2026-06-12 code review (`docs/CODE_REVIEW_FINDINGS.md`)

A tiered design composes local + remote tiers, so these findings become load-bearing and should be fixed first:

- **CachedMetadataBackend staleness** (metadata/base.py:518): serves stale reads vs. external writes for up to its TTL — a tiered cache makes multi-writer the *normal* case; invalidation strategy must account for it.
- **U2** (SQLite/PG silently drop user metadata that JSON preserves): tier promotion (`remote.get()` → `local.put()`) across different backends would silently lose metadata until TASK-10 lands.
- **R13** (eviction skips remote `://` blobs): local-tier LRU eviction is the core mechanism here — TASK-8 must land first.
- **U4** (divergent signing schemes UnifiedCache vs BlobStore): entries promoted between tiers must verify under one scheme (see SEED on signing unification).
