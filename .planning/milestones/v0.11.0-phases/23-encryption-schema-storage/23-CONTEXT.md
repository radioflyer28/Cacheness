# Phase 23: Encryption Schema & Storage - Context

**Gathered:** 2026-04-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix the data-loss bug where SQLite and PostgreSQL backends silently discard `encryption_algorithm` and `encryption_iv` metadata fields, making encrypted blobs permanently unreadable. Add dedicated schema columns for these fields, implement v3→v4 schema migrations for both backends, and fix `put_entry()`/`get_entry()` to preserve and reconstruct encryption metadata. Bundle `cacheness_version` tracking into the same migration.

</domain>

<decisions>
## Implementation Decisions

### Storage Strategy
- **D-01:** Use **dedicated columns** for `encryption_algorithm` and `encryption_iv` in both SQLite and PostgreSQL — uniform approach, no backend divergence.
- **D-02:** Column names match metadata dict keys exactly: `encryption_algorithm` (VARCHAR/TEXT) and `encryption_iv` (VARCHAR/TEXT). No abbreviation.
- **D-03:** Both columns are nullable (existing entries have no encryption data).

### Migration Infrastructure
- **D-04:** Add `get_migrations()` override to PostgreSQL backend, following the same pattern as SQLite's existing migration system (uses base class `run_all_migrations()` mechanism).
- **D-05:** Version numbers aligned across backends: both SQLite and PostgreSQL migrate from **v3→v4**. PostgreSQL tables created before this change are assumed to be at v3.
- **D-06:** Bundle a `cacheness_version` column (TEXT, nullable) into the v3→v4 migration for both backends. This satisfies the "store cacheness version in metadata" todo.

### Residual Metadata Scope
- **D-07:** Fix **only** encryption fields + cacheness_version — targeted fix. Pop `encryption_algorithm`, `encryption_iv`, `cacheness_version` from metadata dict and store in dedicated columns. Other unknown residual fields continue to be discarded (existing behavior unchanged).

### Agent's Discretion
- Column type widths (VARCHAR vs TEXT) — use whatever is idiomatic for each backend
- Migration idempotency approach (e.g., `IF NOT EXISTS` for columns, `PRAGMA table_info` checks)
- Whether to add indexes on encryption columns (likely not needed — rarely queried)

### Folded Todos
- **Encryption at rest for metadata and blobs** — superseded by this phase's work (ENC-01/ENC-02 requirements)
- **Store cacheness version in metadata for migration support** — bundled into v3→v4 migration as `cacheness_version` column (D-06)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Metadata Backend Internals
- `src/cacheness/metadata/sqlite_backend.py` — SQLite put_entry (L685-798), get_entry (L585-683), schema (L356-385), migrations (L38-170, L318-327)
- `src/cacheness/metadata/json_backend.py` — JSON put_entry (L200-254), get_entry (L163-179) — reference for correct behavior
- `src/cacheness/storage/backends/postgresql_backend.py` — PG put_entry/_upsert_entry (L892-1104), get_entry/_entry_to_dict (L878-891, L1055-1110), ORM schema (L143-195)

### Encryption Flow
- `src/cacheness/storage/blob_store.py` — _write_blob encryption (L490-520), get decryption (L615-628)
- `src/cacheness/encryption.py` — encrypt_blob(), decrypt_blob(), derive_encryption_key()

### Migration Patterns
- `src/cacheness/metadata/sqlite_backend.py` L38-170 — existing v1→v2 and v2→v3 migration functions (follow this pattern)
- `src/cacheness/metadata/base.py` — MetadataBackend ABC with get_schema_version(), set_schema_version(), get_migrations(), run_all_migrations()

### Test References
- `tests/test_encryption_at_rest.py` — existing encryption tests (all hardcode JSON backend — Phase 24 will parametrize)
- `tests/test_sqlite_schema_versioning.py` — existing SQLite migration tests (follow this pattern)

### Research
- `.planning/research/SUMMARY.md` — v0.11.0 research summary with 4 critical bugs identified

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Migration framework** (`MetadataBackend.run_all_migrations()`): Existing base class handles version tracking and sequential migration application. SQLite already uses it with 2 migrations.
- **SQLite migration pattern** (`_sqlite_migrate_v1_to_v2`, `_sqlite_migrate_v2_to_v3`): Clean pattern using `ALTER TABLE ADD COLUMN` with `PRAGMA table_info()` idempotency checks. Follow exactly.
- **PostgreSQL JSONB helpers** (`_ensure_jsonb_value()`): PG backend has JSON conversion utilities, but we're using dedicated columns so these aren't directly needed.

### Established Patterns
- **put_entry field extraction**: Both SQLite and PG pop known fields from `metadata` dict into variables, then build SQL INSERT with those variables. New fields need: (1) pop in extraction section, (2) parameter in SQL INSERT, (3) column in schema.
- **get_entry reconstruction**: Both backends read columns and repopulate a `metadata` dict. New fields need: (1) column read, (2) conditional add to metadata dict if not None.
- **Schema versioning**: `cacheness_namespaces` table tracks `schema_version` per namespace. Migrations run on namespace init.

### Integration Points
- `put_entry()` in both backends — add encryption field extraction to the pop list
- `get_entry()` / `_entry_to_dict()` in both backends — add encryption field reconstruction
- `get_migrations()` — add v3→v4 entry in SQLite, add entire override in PG
- `create_namespace()` / ORM model — add columns to CREATE TABLE schema
- `blob_store.py` — no changes needed (already writes encryption fields correctly)

</code_context>

<specifics>
## Specific Ideas

- Follow the existing SQLite v2→v3 migration (which added 6 columns) as the direct template for v3→v4
- PostgreSQL migration should use `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` (PostgreSQL native syntax, cleaner than SQLite's PRAGMA-based checks)
- The `cacheness_version` column should store the library version string (e.g., "0.11.0") — populated on new entries, NULL for existing entries

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

### Reviewed Todos (not folded)
None — both matched todos were folded into this phase.

</deferred>

---

*Phase: 23-encryption-schema-storage*
*Context gathered: 2026-04-06*
