# Phase 26 — Discussion Log

**Date:** 2025-07-06
**Mode:** Interactive discuss (no flags)

## Areas Discussed

### 1. Config Validation Scope

**Question:** Which bad combos to detect? How strict?

**Decision:** Hard errors only (D-01). All five identified combos validated (D-02):
1. encryption + no key source
2. encryption + allow_unsigned
3. in-memory key + encryption
4. signing + allow_unsigned
5. JSON backend + inline blobs

User selected all combos and hard-error strictness. No debate needed.

### 2. Migration Test Strategy

**Question:** Which backends? What data types?

**Decision:** Both SQLite + PostgreSQL (D-03). Verify plain entries + custom_metadata entries (D-04).

User confirmed both backends need coverage. Encrypted/TTL entries excluded — encrypted entries can't pre-exist in v3 schema, TTL is metadata-only.

### 3. Error Experience

**Question:** Exception type and message format?

**Decision:** ConfigValidationError (D-05) with actionable fix suggestions (D-06).

User selected existing exception type and actionable messages. Aligns with success criteria.

## Codebase Scout Summary

- **SecurityConfig.__post_init__** (config.py L439-459): Validates cryptography presence and key_fallback_policy. Hook point for D-02 combos #1-4.
- **CacheConfig._validate_backend_compatibility()** (config.py L774-800): Existing pattern for cross-component validation. Hook point for D-02 combo #5.
- **ConfigValidationError**: Already exists in codebase — ready to use.
- **test_sqlite_schema_versioning.py**: Has v1→v4 chain with real data, but v3→v4 only on fresh DB. Gap to fill.
- **test_pg_schema_versioning.py**: No v3→v4 real-data test. Gap to fill.
