# Phase 23: Encryption Schema & Storage - Research

**Researched:** 2026-04-06
**Domain:** Schema migration + metadata backend field extraction/reconstruction
**Confidence:** HIGH — all findings verified against source code

## Summary

Phase 23 fixes a **silent data-loss bug** where SQLite and PostgreSQL metadata backends discard `encryption_algorithm` and `encryption_iv` fields during `put_entry()`. Both backends pop known fields from the metadata dict into dedicated columns, but encryption fields have no columns — the residual dict is discarded. The JSON backend stores everything as-is, masking the problem. All existing encryption tests hardcode `metadata_backend="json"`.

The fix is mechanical: add 3 new columns (`encryption_algorithm`, `encryption_iv`, `cacheness_version`) via v3→v4 schema migrations for both backends, then update `put_entry()` and `get_entry()` to extract/reconstruct these fields. The migration framework already exists and has been used twice (v1→v2, v2→v3). Both backends follow identical patterns.

**Primary recommendation:** Follow the exact pattern of the v2→v3 migration (which added 6 columns). Add columns, pop in put_entry, reconstruct in get_entry. No new dependencies, no architectural changes.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Use **dedicated columns** for `encryption_algorithm` and `encryption_iv` in both SQLite and PostgreSQL — uniform approach, no backend divergence.
- **D-02:** Column names match metadata dict keys exactly: `encryption_algorithm` (VARCHAR/TEXT) and `encryption_iv` (VARCHAR/TEXT). No abbreviation.
- **D-03:** Both columns are nullable (existing entries have no encryption data).
- **D-04:** Add `get_migrations()` override to PostgreSQL backend, following the same pattern as SQLite's existing migration system (uses base class `run_all_migrations()` mechanism).
- **D-05:** Version numbers aligned across backends: both SQLite and PostgreSQL migrate from **v3→v4**. PostgreSQL tables created before this change are assumed to be at v3.
- **D-06:** Bundle a `cacheness_version` column (TEXT, nullable) into the v3→v4 migration for both backends. This satisfies the "store cacheness version in metadata" todo.
- **D-07:** Fix **only** encryption fields + cacheness_version — targeted fix. Pop `encryption_algorithm`, `encryption_iv`, `cacheness_version` from metadata dict and store in dedicated columns. Other unknown residual fields continue to be discarded (existing behavior unchanged).

### Agent's Discretion
- Column type widths (VARCHAR vs TEXT) — use whatever is idiomatic for each backend
- Migration idempotency approach (e.g., `IF NOT EXISTS` for columns, `PRAGMA table_info` checks)
- Whether to add indexes on encryption columns (likely not needed — rarely queried)

### Deferred Ideas (OUT OF SCOPE)
None.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ENC-01 | Encryption metadata (`encryption_algorithm`, `encryption_iv`) preserved across SQLite and PostgreSQL backends via dedicated schema columns | Migration framework verified (v3→v4), put_entry/get_entry patterns identified, ORM mixin columns documented |
| ENC-02 | Encrypted blob roundtrip (put→get) works correctly with all 3 metadata backends (JSON, SQLite, PostgreSQL) | JSON already works; SQLite and PG need 3 changes each (schema + put + get); `_build_metadata_dict` flows encryption fields from `result.extra` correctly |
</phase_requirements>

## Project Constraints (from copilot-instructions.md)

- **Package manager:** Always use `uv` — never `pip` or `python` directly
- **Testing:** `uv run pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py`
- **Baseline:** 1604+ tests passing (current baseline may be higher)
- **Quality gates:** `uv run ruff format`, `uv run ruff check --fix`, `uv run ruff check`, `uv run ty check`
- **Imports:** `from cacheness.core import UnifiedCache` in tests (not `from cacheness import UnifiedCache`)

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| SQLAlchemy | Already installed | ORM models, schema migration DDL | Already used for all metadata backends [VERIFIED: source code] |
| psycopg | Already installed | PostgreSQL driver | Already used by PostgresBackend [VERIFIED: source code] |

### Supporting
No new dependencies needed. The fix uses existing ALTER TABLE DDL, existing ORM patterns, and existing migration framework.

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Dedicated columns (chosen) | Store in metadata_dict JSON/JSONB | JSON storage works for JSON backend but SQLite metadata_dict is TEXT that's only stored when `store_full_metadata` is enabled — unreliable |
| ALTER TABLE migration | Alembic | Overkill — the project already has its own lightweight migration system that works well |

## Architecture Patterns

### Pattern 1: Field Extraction in put_entry()

Both SQLite and PostgreSQL backends follow the same pattern — pop known fields from the `metadata` dict and map them to dedicated columns. [VERIFIED: sqlite_backend.py L685-798, postgresql_backend.py L908-1050]

**Current extraction (SQLite put_entry L704-724):**
```python
# Extract backend technical metadata to dedicated columns (not JSON)
object_type = metadata.pop("object_type", None)
storage_format = metadata.pop("storage_format", None)
serializer = metadata.pop("serializer", None)
# ... more pops ...
```

**New fields to add (after existing pops):**
```python
# Extract encryption fields to dedicated columns
encryption_algorithm = metadata.pop("encryption_algorithm", None)
encryption_iv = metadata.pop("encryption_iv", None)
cacheness_version = metadata.pop("cacheness_version", None)
```

**PostgreSQL _upsert_entry (L920-932)** follows the identical pattern with `metadata.pop()`.

### Pattern 2: Field Reconstruction in get_entry()

Both backends reconstruct metadata dicts from columns. [VERIFIED: sqlite_backend.py L585-683, postgresql_backend.py L1055-1110]

**Current reconstruction (SQLite get_entry L639-652):**
```python
# Optional security fields
if row.file_hash is not None:
    metadata["file_hash"] = row.file_hash
if row.entry_signature is not None:
    metadata["entry_signature"] = row.entry_signature
```

**New fields to reconstruct:**
```python
# Encryption fields
if row.encryption_algorithm is not None:
    metadata["encryption_algorithm"] = row.encryption_algorithm
if row.encryption_iv is not None:
    metadata["encryption_iv"] = row.encryption_iv
```

**PostgreSQL _entry_to_dict (L1055-1110)** uses `if entry.field_name:` pattern — identical approach.

### Pattern 3: Schema Migration (v3→v4)

**SQLite migration pattern** (from v2→v3, L80-157):
- Use `PRAGMA table_info` to get existing columns
- Conditionally `ALTER TABLE ADD COLUMN` only if column doesn't exist
- Idempotent by design

**PostgreSQL migration pattern** (from v2→v3, L358-428):
- Use `ALTER TABLE ADD COLUMN IF NOT EXISTS` (PostgreSQL 9.6+ native syntax)
- Cleaner than SQLite's PRAGMA-based approach

### Pattern 4: ORM Model Updates

**SQLite ORM** (`metadata/_compat.py` CacheEntryMixin, L110-147): Add columns to the mixin class. [VERIFIED: source code]

**PostgreSQL ORM** (`postgresql_backend.py` PgCacheEntryMixin, L143-195): Add columns to the mixin class. [VERIFIED: source code]

Both use `Column(String/Text, nullable=True)`.

### Pattern 5: CREATE TABLE in create_namespace()

Both backends have hardcoded `CREATE TABLE` SQL in `create_namespace()` for new namespaces. These must include the new columns. [VERIFIED: sqlite_backend.py L355-385, postgresql_backend.py L640-670]

### Pattern 6: Encryption Metadata Flow

The flow from encryption to metadata is:
1. `_write_blob()` encrypts → sets `result.extra["encryption_algorithm"]` and `result.extra["encryption_iv"]` [VERIFIED: blob_store.py L1236-1237]
2. `_build_metadata_dict()` merges `result.extra` into `metadata_dict` [VERIFIED: _inline_blob_mixin.py L198-210]
3. `metadata_dict` goes into `entry_data["metadata"]` [VERIFIED: core.py L880-882]
4. `put_entry()` receives `entry_data` with encryption fields in `metadata` dict
5. **BUG:** SQLite/PG pop known fields but don't pop encryption fields → they remain in residual dict → discarded

### Anti-Patterns to Avoid
- **Don't store encryption_iv as binary:** It's stored as hex string (`.hex()`) and read back via `bytes.fromhex()`. Keep as TEXT. [VERIFIED: blob_store.py L501, L628]
- **Don't add indexes on encryption columns:** These are never queried for filtering — only read back per-entry during `get_entry()`. [ASSUMED]
- **Don't modify BlobStore or core.py:** The bug is purely in the metadata backend layer. The encryption flow from `_write_blob` through `_build_metadata_dict` is correct. [VERIFIED: source code]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Schema migration framework | Custom migration runner | Existing `MetadataBackend.run_migrations()` + `get_migrations()` | Already handles version tracking, sequential application, per-namespace migration [VERIFIED: base.py L270-295] |
| Column idempotency (SQLite) | Try/except on ALTER TABLE | `PRAGMA table_info` check (existing pattern) | SQLite doesn't support `IF NOT EXISTS` on `ADD COLUMN`; the project uses pragma checks [VERIFIED: sqlite_backend.py L105-110] |
| Column idempotency (PG) | Manual column existence check | `ADD COLUMN IF NOT EXISTS` | PostgreSQL native syntax, already used in v2→v3 migration [VERIFIED: postgresql_backend.py L388-410] |

## Common Pitfalls

### Pitfall 1: Forgetting create_namespace() SQL
**What goes wrong:** New columns exist in migrations and ORM, but `create_namespace()` has hardcoded `CREATE TABLE` SQL that doesn't include them. New namespaces created after migration have no encryption columns.
**Why it happens:** `create_namespace()` duplicates the schema as raw SQL (not derived from ORM). Easy to miss.
**How to avoid:** Update `CREATE TABLE` in both SQLite `create_namespace()` (L355-385) and PG `create_namespace()` (L640-670).
**Warning signs:** Tests pass for default namespace but fail for custom namespaces.

### Pitfall 2: Forgetting the SELECT list in get_entry()
**What goes wrong:** Columns added to schema and put_entry, but `get_entry()` SELECT statement doesn't request them. Fields are silently NULL.
**Why it happens:** SQLite `get_entry()` uses an explicit `select(CE.col1, CE.col2, ...)` list (L591-616), not `SELECT *`. New columns must be added to this list.
**How to avoid:** Add `CE.encryption_algorithm, CE.encryption_iv, CE.cacheness_version` to the select list.
**Warning signs:** `get_entry()` returns None for encryption fields even after successful `put_entry()`.

### Pitfall 3: PG _entry_to_dict missing metadata_dict
**What goes wrong:** PG `_entry_to_dict()` doesn't include `metadata_dict` in the result metadata — but `get_entry()` callers may expect it.
**Why it happens:** PG `_entry_to_dict()` builds metadata from individual columns but was never updated as new fields were added.
**How to avoid:** Add encryption field reconstruction to `_entry_to_dict()` following the existing pattern.
**Warning signs:** PG roundtrip fails but SQLite works.

### Pitfall 4: iter_entry_summaries() not updated
**What goes wrong:** Encryption fields not in summaries, so bulk operations relying on `iter_entry_summaries()` (like `verify_integrity`, `rotate_key`) don't see encryption metadata.
**Why it happens:** SQLite `iter_entry_summaries()` uses raw SQL with an explicit column list (L876-886).
**How to avoid:** Add encryption columns to `iter_entry_summaries()` SELECT and flat dict construction.
**Warning signs:** Key rotation or integrity verification fails on encrypted entries with SQLite/PG backend.

### Pitfall 5: update_entry_metadata() not updated
**What goes wrong:** `update_entry_metadata()` can't update encryption fields after key rotation.
**Why it happens:** SQLite `update_entry_metadata()` has an explicit field-by-field update pattern (L829-862).
**How to avoid:** Add `encryption_algorithm`, `encryption_iv` to the update handler.
**Warning signs:** Key rotation completes but metadata still shows old IV.

### Pitfall 6: PG schema_version in create_namespace() is hardcoded to 3
**What goes wrong:** New PG namespaces bypass migration because they start at v3 (current max), but now max is v4.
**Why it happens:** `create_namespace()` hardcodes `schema_version=3` (L730).
**How to avoid:** Update to `schema_version=4` after adding the v3→v4 migration, or better: set to the max version that the CREATE TABLE schema matches.
**Warning signs:** New PG namespaces missing encryption columns.

### Pitfall 7: SQLite ORM model out of sync with migration
**What goes wrong:** ORM model defines the columns (so `create_all()` creates them for new databases), but migration doesn't run because schema_version is already at 3 for pre-existing databases.
**Why it happens:** `CacheEntryMixin` in `_compat.py` is used by `Base.metadata.create_all()` for fresh databases. Migrations handle existing databases. Both must agree.
**How to avoid:** Add columns to BOTH the ORM mixin AND the migration function.
**Warning signs:** Fresh databases work, existing databases missing columns.

## Code Examples

### v3→v4 Migration — SQLite
```python
# Source: follows exact pattern of _sqlite_migrate_v2_to_v3 in sqlite_backend.py
def _sqlite_migrate_v3_to_v4(backend: "SqliteBackend", namespace_id: str) -> None:
    """v3 → v4: add encryption_algorithm, encryption_iv, cacheness_version columns."""
    if namespace_id == DEFAULT_NAMESPACE:
        table = "cache_entries"
    else:
        table = f"cache_entries_{namespace_id}"

    with backend.SessionLocal() as session:
        existing_cols = {
            row[1]
            for row in session.execute(text(f'PRAGMA table_info("{table}")')).fetchall()
        }

        if "encryption_algorithm" not in existing_cols:
            session.execute(
                text(f'ALTER TABLE "{table}" ADD COLUMN encryption_algorithm TEXT')
            )
        if "encryption_iv" not in existing_cols:
            session.execute(
                text(f'ALTER TABLE "{table}" ADD COLUMN encryption_iv TEXT')
            )
        if "cacheness_version" not in existing_cols:
            session.execute(
                text(f'ALTER TABLE "{table}" ADD COLUMN cacheness_version TEXT')
            )
        session.commit()
```

### v3→v4 Migration — PostgreSQL
```python
# Source: follows exact pattern of _pg_migrate_v2_to_v3 in postgresql_backend.py
def _pg_migrate_v3_to_v4(backend: "PostgresBackend", namespace_id: str) -> None:
    """v3 → v4: add encryption_algorithm, encryption_iv, cacheness_version columns."""
    table = (
        "cache_entries"
        if namespace_id == DEFAULT_NAMESPACE
        else f"cache_entries_{namespace_id}"
    )

    with backend.SessionLocal() as session:
        session.execute(
            text(f'ALTER TABLE "{table}" ADD COLUMN IF NOT EXISTS encryption_algorithm TEXT')
        )
        session.execute(
            text(f'ALTER TABLE "{table}" ADD COLUMN IF NOT EXISTS encryption_iv TEXT')
        )
        session.execute(
            text(f'ALTER TABLE "{table}" ADD COLUMN IF NOT EXISTS cacheness_version TEXT')
        )
        session.commit()
```

### ORM Column Additions
```python
# Add to CacheEntryMixin in metadata/_compat.py (after inline_ext)
encryption_algorithm = Column(Text, nullable=True)
encryption_iv = Column(Text, nullable=True)
cacheness_version = Column(Text, nullable=True)

# Add to PgCacheEntryMixin in postgresql_backend.py (after inline_ext)
encryption_algorithm = Column(Text, nullable=True)
encryption_iv = Column(Text, nullable=True)
cacheness_version = Column(Text, nullable=True)
```

### put_entry Field Extraction (SQLite)
```python
# Add after inline_ext pop (sqlite_backend.py put_entry)
encryption_algorithm = metadata.pop("encryption_algorithm", None)
encryption_iv = metadata.pop("encryption_iv", None)
cacheness_version = metadata.pop("cacheness_version", None)
```

### get_entry Field Reconstruction (SQLite)
```python
# Add after entry_signature/s3_etag reconstruction
if row.encryption_algorithm is not None:
    metadata["encryption_algorithm"] = row.encryption_algorithm
if row.encryption_iv is not None:
    metadata["encryption_iv"] = row.encryption_iv
```

### cacheness_version Population
```python
# In core.py put() or blob_store.py _write_blob(), add to metadata_dict:
from cacheness import __version__
metadata_dict["cacheness_version"] = __version__
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| All metadata in flat JSON file | Dedicated columns in SQL backends | v0.7.0 (Phase 2) | Unknown fields in metadata dict are silently discarded by SQL backends |
| No encryption at rest | AES-256-GCM encryption via `_write_blob` | v0.10.0 (Phase 18) | Encryption fields flow through `result.extra` → `metadata` dict, but SQL backends don't extract them |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | No indexes needed on encryption columns | Anti-Patterns | Low — can always add later if query patterns emerge |
| A2 | `cacheness_version` should be populated from `__version__` | Code Examples | Low — simple to change the source; column exists regardless |
| A3 | PG `_entry_to_dict` doesn't currently include `metadata_dict` key in result | Pitfall 3 | Low — verified by source code reading; PG includes metadata_dict via JSONB natively via `cache_key_params` handling |

## Open Questions (RESOLVED)

1. **Where should `cacheness_version` be populated?**
   - What we know: Column will exist after migration. Would be populated on new entries.
   - What's unclear: Should it go in `core.py` `put()`, `_build_metadata_dict()`, or in each backend's `put_entry()` as a default?
   - RESOLVED: Agent discretion — add column only in v3→v4 migration; population logic deferred to executor's judgment (likely `_build_metadata_dict()`). Plans add the column but don't mandate population site.

2. **Should `iter_entry_summaries()` include encryption fields?**
   - What we know: Key rotation in `core.py` uses `get_entry()` not `iter_entry_summaries()`. `verify_integrity()` uses `iter_entry_summaries()`.
   - What's unclear: Whether `verify_integrity()` needs encryption fields for its logic.
   - RESOLVED: Include for completeness — both plans implement this. Needed if integrity verification ever checks encryption state.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest + pytest-xdist |
| Config file | pyproject.toml |
| Quick run command | `uv run pytest tests/test_sqlite_schema_versioning.py tests/test_encryption_at_rest.py -x -q --ignore=tests/test_tensorflow_handler.py` |
| Full suite command | `uv run pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ENC-01 | v3→v4 migration adds encryption columns (SQLite) | unit | `uv run pytest tests/test_sqlite_schema_versioning.py -x -q -k "v3_to_v4 or v4"` | ❌ Wave 0 |
| ENC-01 | v3→v4 migration adds encryption columns (PG) | unit | `uv run pytest tests/test_pg_schema_versioning.py -x -q -k "v3_to_v4 or v4"` | ❌ Wave 0 |
| ENC-01 | put_entry preserves encryption fields (SQLite) | unit | `uv run pytest tests/test_metadata.py -x -q -k "sqlite and encryption"` | ❌ Wave 0 |
| ENC-01 | put_entry preserves encryption fields (PG) | integration | `uv run pytest tests/test_metadata.py -x -q -k "pg and encryption"` | ❌ Wave 0 |
| ENC-01 | get_entry returns encryption fields (SQLite) | unit | `uv run pytest tests/test_metadata.py -x -q -k "sqlite and encryption"` | ❌ Wave 0 |
| ENC-02 | Encrypted roundtrip with SQLite backend | integration | `uv run pytest tests/test_encryption_at_rest.py -x -q -k "sqlite"` | ❌ Wave 0 |
| ENC-02 | Encrypted roundtrip with PG backend | integration | `uv run pytest tests/test_encryption_at_rest.py -x -q -k "postgres"` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/test_sqlite_schema_versioning.py tests/test_encryption_at_rest.py tests/test_metadata.py -x -q`
- **Per wave merge:** `uv run pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_sqlite_schema_versioning.py` — add v3→v4 migration test (existing file, add tests)
- [ ] `tests/test_pg_schema_versioning.py` — add v3→v4 migration test (existing file, add tests)
- [ ] `tests/test_encryption_at_rest.py` — add SQLite/PG roundtrip tests (existing file, add parametrized tests)

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | — |
| V3 Session Management | No | — |
| V4 Access Control | No | — |
| V5 Input Validation | No | — |
| V6 Cryptography | Yes | AES-256-GCM encryption metadata preserved across all storage backends — no new crypto code needed |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Encryption metadata loss → permanent data loss | Information Disclosure | Dedicated columns ensure encryption fields survive backend storage roundtrip |
| SQL injection in migration DDL | Tampering | Table names derived from validated namespace_id (`^[a-z0-9_]{1,48}$`), not user input [VERIFIED: metadata/_compat.py validate_namespace_id] |

## Sources

### Primary (HIGH confidence)
- `src/cacheness/metadata/sqlite_backend.py` — put_entry, get_entry, migrations, create_namespace, iter_entry_summaries, update_entry_metadata
- `src/cacheness/storage/backends/postgresql_backend.py` — _upsert_entry, _entry_to_dict, migrations, PgCacheEntryMixin, create_namespace
- `src/cacheness/metadata/_compat.py` — CacheEntryMixin ORM model
- `src/cacheness/metadata/base.py` — MetadataBackend ABC, run_migrations(), get_migrations()
- `src/cacheness/storage/blob_store.py` — _write_blob encryption flow (L1230-1237), _read_blob decryption (L1280-1297)
- `src/cacheness/_inline_blob_mixin.py` — _build_metadata_dict (L188-210)
- `src/cacheness/encryption.py` — encrypt_blob, decrypt_blob
- `tests/test_sqlite_schema_versioning.py` — existing migration test patterns
- `tests/test_pg_schema_versioning.py` — existing PG migration test patterns
- `tests/test_encryption_at_rest.py` — existing encryption tests (all JSON backend)

### Secondary (MEDIUM confidence)
- `.planning/research/SUMMARY.md` — v0.11.0 research summary identifying 4 critical bugs
- `.planning/research/FEATURES.md` — feature landscape with bug analysis

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies, all verified in source
- Architecture: HIGH — all patterns verified against source code with specific line numbers
- Pitfalls: HIGH — all derived from reading the actual code paths that need modification

**Research date:** 2026-04-06
**Valid until:** Indefinite (codebase-specific findings, not version-dependent)
