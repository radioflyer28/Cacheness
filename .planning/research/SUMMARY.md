# Project Research Summary

**Project:** Cacheness v0.11.0 "Cross-Backend Hardening"
**Domain:** Python disk caching library — cross-backend encryption, inline blob encryption, test parity
**Researched:** 2026-04-06
**Confidence:** HIGH — all findings verified against source code

## Executive Summary

v0.11.0 is a **critical bug-fix and hardening milestone**. The v0.10.0 encryption-at-rest feature has a silent data loss bug: SQLite and PostgreSQL backends discard `encryption_algorithm` and `encryption_iv` fields because no dedicated columns exist. This means encrypted blobs stored via these backends become permanently unreadable. The JSON backend masks this because it stores the entire metadata dict as-is.

Three additional bugs exist: (1) inline blob direct path bypasses encryption entirely (plaintext stored in metadata db), (2) inline blob read path doesn't decrypt, (3) key rotation skips inline entries. All encryption tests hardcode `metadata_backend="json"`, providing zero coverage of these backend-specific issues.

**No new dependencies are needed.** The existing stack (cryptography, SQLAlchemy, psycopg) provides everything. The fixes are schema migrations, code path corrections, and test parametrization.

## Key Findings

### Stack

No new dependencies. The fix is entirely an integration gap — adding columns, migrations, and code path corrections. See [STACK.md](STACK.md).

### Critical Bugs

| Bug | Impact | Root Cause |
|-----|--------|------------|
| Encryption metadata dropped by SQLite/PG | **Data loss** — encrypted blobs permanently unreadable | `put_entry()` pops known fields; encryption fields not in the pop list, residual dict discarded |
| Direct inline bypasses encryption | **Security hole** — plaintext stored when `max_inline_size > 0` + encryption enabled | `_try_direct_inline()` skips `_write_blob()` where encryption happens |
| Inline read doesn't decrypt | **Read failure** — ciphertext passed to handler | `_read_inline_blob()` has no decryption step |
| Key rotation skips inline entries | **Orphaned ciphertext** — inline entries untouched during rotation | `rotate_key()` filters on `actual_path`, which is `None` for inline |

### Feature Categories

**Table Stakes (must fix):**
- Encryption metadata preserved across all backends (schema migration)
- Cross-backend encrypted roundtrip
- Encryption test parity across backends

**Differentiators (should fix):**
- Encryption config validation at init (fail loudly for bad combos)
- Encrypted inline blobs (encrypt before inline)
- Key rotation for inline blobs

**Anti-Features (do NOT build):**
- Per-entry encryption keys (complexity without benefit)
- Transparent re-encryption on backend switch
- Encrypting metadata field values
- Multiple encryption algorithm support

### Build Order

```
1. Schema migration (add encryption columns to SQLite/PG)
   ↓
2. put_entry/get_entry extraction (preserve encryption fields)
   ↓
3. Cross-backend test parity (parametrize encryption tests)
   ↓
4. Inline blob + encryption (encrypt direct inline, decrypt on read)
   ↓
5. Key rotation for inline entries
```

### Watch Out For

1. **Dual write paths** — both `BlobStore.put()` and `_write_blob()` need fixes
2. **Schema version coordination** — verify current version is v3 before writing v3→v4 migration
3. **PostgreSQL requires Docker** — encryption tests need `@pytest.mark.xdist_group("docker")`
4. **Migration testing** — must test ALTER TABLE on existing databases with data
5. **Inline size threshold** — encrypted data is larger than plaintext; `max_inline_size` check may need adjustment

## Roadmap Implications

**4 phases recommended:**

1. **Schema + Storage** — Add `encryption_algorithm` and `encryption_iv` columns to SQLite and PostgreSQL, schema migrations, update put_entry/get_entry. Foundation that unblocks everything.
2. **Cross-Backend Test Parity** — Parametrize all encryption tests across JSON/SQLite/PostgreSQL. Verify the schema fix actually works end-to-end.
3. **Inline Blob Encryption** — Encrypt direct inline path, add decryption to read path, or validate+reject the config combination. Address key rotation for inline entries.
4. **Integration & Hardening** — Config validation, migration testing, edge cases, and final verification.

## Open Questions

1. Should inline blob + encryption be a hard error (reject config) or should encrypted inline blobs be implemented?
2. Should pending todos (version metadata, property-based testing) be folded into this milestone or deferred?
