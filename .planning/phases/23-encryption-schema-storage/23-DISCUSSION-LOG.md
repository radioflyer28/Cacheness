# Phase 23: Encryption Schema & Storage - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-06
**Phase:** 23-Encryption Schema & Storage
**Areas discussed:** Storage strategy for encryption fields, PostgreSQL migration infrastructure, Residual metadata preservation scope

---

## Storage Strategy for Encryption Fields

| Option | Description | Selected |
|--------|-------------|----------|
| Dedicated columns (uniform) | Add encryption_algorithm VARCHAR(20) and encryption_iv VARCHAR(48) columns to both SQLite and PG. Uniform behavior, explicit pop/reconstruct. Requires ALTER TABLE migration for both. | ✓ |
| Dedicated columns for SQLite, JSONB for PG | Add columns for SQLite (required since metadata_dict is TEXT). For PG, store in metadata_dict JSONB (natively queryable, no schema change needed). Simpler PG migration but divergent code paths. | |
| metadata_dict for both | Store encryption fields in metadata_dict for both backends. SQLite would need to serialize/deserialize JSON in get_entry. No schema change needed. Least disruptive but fragile. | |

**User's choice:** Dedicated columns (uniform)
**Notes:** Keeps both SQLite and PG backends consistent. Explicit columns are queryable and self-documenting.

### Follow-up: Column Naming

| Option | Description | Selected |
|--------|-------------|----------|
| encryption_algorithm + encryption_iv (exact match) | Match the metadata dict keys from blob_store.py exactly. VARCHAR(48) accommodates hex-encoded 16-byte IV (32 chars) with margin. | ✓ |
| enc_algorithm + enc_iv (abbreviated) | Shorter column names, but diverges from the metadata dict key names used in blob_store.py. | |

**User's choice:** encryption_algorithm + encryption_iv (exact match)
**Notes:** Consistency with blob_store.py field names.

---

## PostgreSQL Migration Infrastructure

| Option | Description | Selected |
|--------|-------------|----------|
| Add get_migrations() override (same pattern as SQLite) | Override get_migrations() in PgBackend, just like SqliteBackend. Uses the same base class run_all_migrations() mechanism. Most maintainable long-term. | ✓ |
| Ad-hoc ALTER TABLE on connect | No formal migration system. Just run ALTER TABLE ADD COLUMN IF NOT EXISTS on connect. Simpler but loses migration ordering/versioning. | |
| Schema change on table creation only | Include new columns in CREATE TABLE only. Existing PG databases would need manual migration. Simplest but breaks existing deployments. | |

**User's choice:** Add get_migrations() override (same pattern as SQLite)
**Notes:** Consistent migration infrastructure across both backends.

### Follow-up: Version Numbering

| Option | Description | Selected |
|--------|-------------|----------|
| Start PG at v1, migrate to v2 | PG has no version history. Starting at v1 is PG-specific. | |
| Align with SQLite: v3→v4 | Match SQLite versioning. Keeps version numbers aligned across backends. | ✓ |

**User's choice:** Align with SQLite: v3→v4
**Notes:** Aligned versioning makes cross-backend reasoning simpler.

### Follow-up: Bundling Version Tracking Todo

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, bundle version tracking into v3→v4 migration | Add cacheness_version column during same migration. Natural fit since we're already touching schema. | ✓ |
| No, defer version tracking | Keep Phase 23 focused on encryption only. | |

**User's choice:** Yes, bundle version tracking into v3→v4 migration
**Notes:** Avoids a future v4→v5 migration for a single column.

---

## Residual Metadata Preservation Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Fix only encryption + version fields (targeted) | Pop encryption_algorithm, encryption_iv, cacheness_version from metadata dict. Store in dedicated columns. Other unknown fields continue to be discarded. Minimal change surface. | ✓ |
| Preserve all residual fields in metadata_dict (broad fix) | After popping known fields, serialize whatever remains into metadata_dict column. Future-proofs against any new metadata fields. Wider change. | |
| Fix encryption fields + add warning for other discarded fields | Same as targeted, but log a warning when unknown fields are about to be discarded. Helps detect future field-loss bugs. | |

**User's choice:** Fix only encryption + version fields (targeted)
**Notes:** Minimal scope, no surprise behavior changes for existing users.

---

## Agent's Discretion

- Column type widths (VARCHAR vs TEXT)
- Migration idempotency approach
- Whether to add indexes on encryption columns

## Deferred Ideas

None — discussion stayed within phase scope.
