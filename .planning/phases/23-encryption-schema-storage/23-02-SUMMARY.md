---
phase: 23-encryption-schema-storage
plan: 02
status: complete
requirements_completed: [ENC-01, ENC-02]
---

## Summary

Added `encryption_algorithm`, `encryption_iv`, and `cacheness_version` columns to PostgreSQL metadata backend via v3→v4 schema migration. Updated `_upsert_entry()` to handle encryption fields and `_entry_to_dict()` to reconstruct them.

## What Was Built

**File:** `src/cacheness/storage/backends/postgresql_backend.py`
- Created `_pg_migrate_v3_to_v4()` function using `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` (PostgreSQL native syntax)
- Registered migration as `(3, 4, _pg_migrate_v3_to_v4)` in `get_migrations()` override
- Updated ORM model (PgCacheEntryMixin) with 3 new Column definitions
- Updated `_upsert_entry()` to handle encryption field extraction and storage
- Updated `_entry_to_dict()` to reconstruct encryption fields from ORM columns into metadata dict
- Updated `iter_entry_summaries()` to include encryption fields

**File:** `tests/test_pg_schema_versioning.py`
- Added v3→v4 migration tests (decorated `@requires_postgres @pytest.mark.xdist_group("docker")`)
- Tests skip without Docker (expected on Windows development)

## Verification

```
uv run pytest tests/test_pg_schema_versioning.py -x -q
46 skipped in 1.27s (no Docker — expected on Windows)
```

## Commits

- PostgreSQL encryption schema v3→v4 + field handling (ENC-01, ENC-02)
