---
created: 2026-04-03T19:44:09.447Z
title: Store cacheness version in metadata for migration support
area: database
files:
  - src/cacheness/metadata/base.py
  - src/cacheness/metadata/sqlite_backend.py
  - src/cacheness/metadata/json_backend.py
  - src/cacheness/core.py
  - src/cacheness/_storage_mode_mixin.py
---

## Problem

When Cacheness is upgraded, the serialization format or handler behavior may change. Currently there is no record of which Cacheness version (or serialization format version) wrote a given cache entry. This means:

1. **Decorator mode:** A format-incompatible entry causes a cache miss and re-execution — acceptable but wasteful when a migration could preserve it.
2. **Storage mode (critical):** There is no function to re-execute. If an upgrade changes how data is serialized/deserialized and old entries can't be read, that's **data loss**, not just a performance hit.

Users discussed including version in the cache key vs. the metadata database. Version-in-key is blunt (invalidates everything on upgrade even if format didn't change). Version-in-metadata enables targeted, per-entry migration decisions.

## Solution

1. **Add `cacheness_version` (or `serialization_format_version`) field to metadata entries** — written on every `put()` call.
2. **Schema migration** for existing backends (SQLite: ALTER TABLE, JSON: field addition on read, PostgreSQL: migration).
3. **On read:** Check entry's format version vs. current — if mismatched, either:
   - Re-deserialize with a compat path, or
   - Flag for migration
4. **`migrate()` API** on UnifiedCache — walks entries and re-serializes stale ones in bulk.
5. **Lazy migration option** — entries re-written in new format on first access rather than requiring big-bang migration.
6. **Storage mode emphasis** — this is the primary motivator since stored data is the source of truth with no fallback.

### Design considerations
- Version granularity: `cacheness.__version__` vs. a separate `serialization_format_version` that only bumps when format actually changes (reduces unnecessary migrations).
- Handler-level versioning: each handler (parquet for DataFrames, blosc2 for NumPy, pickle for objects) may evolve independently — consider per-handler format versions.
- Backward compatibility window: define how many prior format versions must be readable without explicit migration.
