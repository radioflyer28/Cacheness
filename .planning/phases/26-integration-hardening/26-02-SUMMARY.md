---
phase: 26-integration-hardening
plan: 02
status: completed
started: 2025-04-07
completed: 2025-04-07
requirements_completed: [HARD-02]
---

## Summary

Added real-data v3-to-v4 schema migration tests for both SQLite and PostgreSQL backends, verifying that existing cached entries (plain + custom metadata) survive migration intact.

## Changes

### tests/test_sqlite_schema_versioning.py
- Added `TestSqliteV3ToV4RealDataMigration` class with 2 tests:
  - `test_v3_to_v4_migration_preserves_existing_data`: Creates a manual v3 SQLite DB with namespace table, cache_entries (plain + custom_metadata entries), cache_stats, and all v3 indexes. Opens with SqliteBackend to trigger auto-migration. Verifies schema_version=4, v4 columns exist, both entries survive with correct data.
  - `test_v3_to_v4_migration_preserves_custom_metadata`: Creates v3 DB with 5 entries each having distinct custom metadata (batch_id, run_name). Migrates and verifies all 5 entries and their metadata survive.

### tests/test_pg_schema_versioning.py
- Added `TestPgV3ToV4RealDataMigration` class with 4 tests (decorated `@requires_postgres @pytest.mark.xdist_group("docker")`):
  - `test_v3_style_entries_survive_on_v4_schema`: Plain entries without encryption fields work on v4 schema
  - `test_v3_style_custom_metadata_preserved`: Custom metadata in metadata_dict preserved
  - `test_v3_and_v4_entries_coexist`: v3-style (no encryption) and v4-style (with encryption) entries coexist
  - `test_iter_entry_summaries_v3_style_entries`: iter_entry_summaries includes v3-style entries

## Deviations

None. Implementation follows plan exactly.

## Key Files

- `tests/test_sqlite_schema_versioning.py` — 2 real-data migration tests
- `tests/test_pg_schema_versioning.py` — 4 v3/v4 coexistence tests

## Self-Check: PASSED
- SQLite tests: 2 passed
- PG tests: 4 skipped (no Docker, expected) — would pass with PG available
- Full suite: 1773 passed, 122 skipped, 0 failures
