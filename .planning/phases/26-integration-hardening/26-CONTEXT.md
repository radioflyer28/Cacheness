# Phase 26 — Integration & Hardening: Context

**Phase goal:** Harden cross-component integration with config validation and real-data migration tests.

**Requirements:** HARD-01 (config validation), HARD-02 (schema migration on real databases)

**Depends on:** Phase 23 (schema), Phase 25 (inline blob encryption)

---

## Decisions

### D-01: Validation strictness — Hard errors only

All bad config combinations raise `ConfigValidationError` at construction time. No warnings-only path. Fail fast so users discover misconfigurations immediately, not after data is written.

**Rationale:** Silent misconfiguration is worse than a startup crash. Matches Phase 26 success criterion #1 ("raises a clear error at construction time").

### D-02: Five bad combos to detect

| # | Bad combination | Why it's bad |
|---|----------------|--------------|
| 1 | encryption enabled + no key file/source | Can't encrypt without a key |
| 2 | encryption + `allow_unsigned_entries=True` | Encrypted data without integrity signing is risky |
| 3 | `use_in_memory_key=True` + encryption | In-memory keys don't persist — cache won't survive restart |
| 4 | `enable_entry_signing=True` + `allow_unsigned_entries=True` | Contradictory: enabling signing but allowing unsigned defeats the purpose |
| 5 | JSON backend + inline blobs | JSON backend doesn't support inline blob storage (Phase 25 D-04) |

**Rationale:** Covers all known footguns. Combo #5 prevents a silent failure discovered in Phase 25.

### D-03: Migration test backends — SQLite + PostgreSQL

Both backends get v3→v4 real-data migration tests. PostgreSQL tests use Docker (existing `@pytest.mark.xdist_group("docker")` pattern).

**Rationale:** Both backends ship as production options. Untested migration on either is a risk.

### D-04: Migration data types to verify

Migration tests verify these entry types survive v3→v4:
- **Plain cached entries** — standard function results with metadata
- **Entries with custom metadata** — entries using the `custom_metadata` parameter

Encrypted entries and TTL entries are NOT tested in migration (encryption writes new v4 columns directly; TTL is metadata-only and unaffected by schema changes).

**Rationale:** Plain + custom_metadata cover the realistic "existing production cache" scenario. Encrypted entries only exist post-v4 schema, so migrating them is impossible.

### D-05: Exception type — ConfigValidationError

Use the existing `ConfigValidationError` for all config validation failures. No new exception classes.

**Rationale:** Already exists in the codebase. Semantic and specific. Avoids import complexity.

### D-06: Actionable error messages

Every `ConfigValidationError` message includes a fix suggestion. Format:

```
"<what's wrong>. <how to fix it>"
```

Example: `"Encryption is enabled but no key source is configured. Set encryption_key_file to a path, or set use_in_memory_key=True for ephemeral caches."`

**Rationale:** Matches Phase 26 success criterion #2 ("actionable error messages"). Users shouldn't need to search docs to fix a config error.

---

## Prior decisions carried forward

- **Phase 23 D-04:** PostgreSQL uses same migration pattern as SQLite (`get_migrations()` override)
- **Phase 23 D-05:** Version numbers aligned — both backends migrate v3→v4
- **Phase 23 D-06:** `cacheness_version` column bundled into v3→v4 migration
- **Phase 25 D-04:** `blob_data` stored as raw bytes; JSON backend doesn't support inline blobs
