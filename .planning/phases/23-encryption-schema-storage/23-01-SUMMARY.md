---
phase: 23-encryption-schema-storage
plan: 01
status: complete
requirements_completed: [ENC-01, ENC-02]
---

## Summary

Added `encryption_algorithm`, `encryption_iv`, and `cacheness_version` columns to SQLite metadata backend via v3→v4 schema migration. Updated `put_entry()` to pop and store encryption fields in dedicated columns, and `get_entry()` to reconstruct them into the metadata dict.

## What Was Built

**File:** `src/cacheness/metadata/_compat.py`
- Added 3 Column definitions to CacheEntryMixin: `encryption_algorithm` (Text, nullable), `encryption_iv` (Text, nullable), `cacheness_version` (Text, nullable)

**File:** `src/cacheness/metadata/sqlite_backend.py`
- Created `_sqlite_migrate_v3_to_v4()` function using `PRAGMA table_info()` idempotency checks + `ALTER TABLE ADD COLUMN` for each new column
- Registered migration as `(3, 4, _sqlite_migrate_v3_to_v4)` in `get_migrations()`
- Updated `create_namespace()` CREATE TABLE SQL to include 3 new columns
- Updated `put_entry()` to pop `encryption_algorithm`, `encryption_iv`, `cacheness_version` from metadata dict and store in dedicated columns
- Updated `get_entry()` to reconstruct these fields from row columns into metadata dict
- Updated `iter_entry_summaries()` to include encryption fields in SELECT
- Updated `update_entry_metadata()` to handle encryption field updates

**File:** `tests/test_sqlite_schema_versioning.py`
- Added `TestSqliteV3ToV4Migration` class with 7 tests: migration creates columns, idempotent re-run, encrypted roundtrip, iter_entry_summaries inclusion, update_entry_metadata handling
- Added `TestSqliteV3ToV4RealDataMigration` class with 2 tests: existing data preserved, custom metadata preserved

## Verification

```
uv run pytest tests/test_sqlite_schema_versioning.py -x -q
52 passed in 23.12s
```

## Commits

- SQLite encryption schema v3→v4 + field handling (ENC-01, ENC-02)
