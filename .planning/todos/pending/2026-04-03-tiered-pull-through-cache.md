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
