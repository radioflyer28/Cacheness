# Feature Landscape

**Domain:** Cross-backend encryption hardening for a disk caching library
**Researched:** 2026-04-06

## Table Stakes

Features users expect when encryption at rest is advertised as "supported." Missing = security vulnerability or silent data loss.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Encryption metadata preserved across all backends | Encrypted entries unreadable if `encryption_algorithm`/`encryption_iv` are lost | **High** | **Critical bug:** SQLite and PostgreSQL `put_entry()` silently drop encryption fields — no dedicated columns exist, and remaining metadata dict entries are discarded after popping known fields. JSON backend stores everything as-is, masking the problem. |
| Cross-backend encrypted roundtrip | Users switching from JSON→SQLite (recommended upgrade path) will lose encryption metadata for all entries | Medium | SQLite `get_entry()` reconstructs metadata only from dedicated columns; encryption fields never appear in output |
| PostgreSQL encrypted roundtrip | Same issue as SQLite — `_upsert_entry()` pops known fields, encryption_algorithm/encryption_iv remain in the residual metadata dict which is discarded | Medium | Identical structural problem to SQLite; PgCacheEntryMixin also lacks encryption columns |
| Encryption test parity across backends | Current tests use `metadata_backend="json"` exclusively (all 7 test references). Zero coverage of SQLite/PostgreSQL with encryption. | Low | Existing test helpers (`_make_encrypted_cache`, `_make_encrypted_blobstore`) hardcode `backend="json"` |
| Inline blob encryption consistency | `_try_direct_inline()` bypasses `_write_blob()` entirely — no encryption is applied. Plaintext stored in metadata row. | **High** | When `max_inline_size > 0` AND `enable_content_encryption=True`, direct inline path stores cleartext blob_data in the database — a **security hole** |
| Disk-then-inline encrypted blob readback | `_try_inline_blob()` reads back bytes from a file encrypted by `_write_blob()`. The blob_data stored is ciphertext. On read, `_read_inline_blob()` tries `handler.get_bytes()` which fails on ciphertext, then falls back to `_read_blob()` which CAN decrypt — but only if encryption metadata fields survived the backend roundtrip. | Medium | Works with JSON backend only. Fails silently with SQLite/PG because encryption metadata is dropped. |

## Differentiators

Features that set the product apart from competitors. Not strictly expected, but significantly improve security posture.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Encryption config validation at init | Fail loudly when `enable_content_encryption=True` with inline blobs enabled, alerting the user to the interaction instead of silently storing plaintext | Low | Simple validation in `CacheConfig.__post_init__()` or `SecurityConfig.__post_init__()`. Most caching libraries don't warn about configuration conflicts. |
| Encrypted inline blobs (encrypt before inline) | Apply encryption to inline blob_data so metadata DB rows don't contain plaintext even when inlining | Medium | Requires encrypting in `_try_direct_inline()` and `_try_inline_blob()`, and decrypting in `_read_inline_blob()`. Orthogonal to the backend column problem. |
| Key rotation for inline blobs | `rotate_key()` currently skips inline entries entirely — it only processes entries with `actual_path` pointing to a file | Medium | Needs to decrypt blob_data with old key, re-encrypt with new key, update metadata in-place |
| Backend migration preserves encryption metadata | When an entry is migrated (e.g., JSON→SQLite upgrade path), encryption metadata should survive | Low | Falls out naturally if the dedicated columns fix is done correctly |

## Anti-Features

Features to explicitly NOT build. These add complexity without proportional value.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Per-entry encryption keys | Massive key management complexity, no real benefit over per-namespace HKDF derivation which already provides isolation | Keep existing per-namespace HKDF derivation (already implemented in `derive_encryption_key()`) |
| Transparent re-encryption on backend switch | Automatically re-encrypting all entries when metadata backend changes is fragile and slow | Document that users should export/re-import, or provide a migration utility in a future milestone |
| Encrypting metadata fields themselves | Encrypting field values in metadata columns adds huge query complexity for marginal gain | Blob content encryption is sufficient; metadata fields (data_type, description, file_size) are not secrets |
| Supporting multiple encryption algorithms simultaneously | AES-256-GCM is the standard; supporting alternatives (ChaCha20-Poly1305, etc.) adds test surface without clear demand | Store algorithm field for forward-compat but only implement AES-256-GCM |

## Feature Dependencies

```
Encryption metadata columns (SQLite/PG) → Cross-backend encrypted roundtrip
Cross-backend encrypted roundtrip → Cross-backend encryption test parity
Inline blob encryption consistency → Encrypted inline blobs (differentiator)
Encryption metadata columns → Key rotation for inline blobs
```

**Critical path:** The SQLite/PG dedicated column fix is the foundation — without it, no other cross-backend encryption feature can work.

## Detailed Analysis of Critical Bugs

### Bug 1: SQLite/PG silently drop encryption_algorithm and encryption_iv

**Root cause:** Both `sqlite_backend.py:put_entry()` (line ~710-730) and `postgresql_backend.py:_upsert_entry()` (line ~910-940) pop recognized fields from the metadata dict into dedicated columns. The encryption fields (`encryption_algorithm`, `encryption_iv`) are NOT in the recognized set. After all pops, the residual metadata dict is discarded — there's no generic "extra_metadata" column.

**Evidence:** SQLite `put_entry()` pops: `object_type`, `storage_format`, `serializer`, `compression_codec`, `actual_path`, `file_hash`, `entry_signature`, `s3_etag`, `cache_key_params`, `metadata_dict`, `inline_ext`, `data_type`. PostgreSQL `_upsert_entry()` pops the same set. Neither pops `encryption_algorithm` or `encryption_iv`.

**Impact:** Any entry written with encryption enabled through SQLite or PostgreSQL backend will have its encryption metadata silently dropped. On read, `_read_blob()` checks `metadata.get("encryption_algorithm")` — finds None — treats the blob as unencrypted — attempts to deserialize ciphertext — fails — deletes the entry (if `delete_on_error=True`, which is the default). **Silent, destructive data loss.**

**Fix approach:** Add `encryption_algorithm` (String) and `encryption_iv` (String) dedicated columns to both SQLite and PostgreSQL models, with corresponding migration functions. Pop them in `put_entry()`, reconstruct in `get_entry()`.

### Bug 2: _try_direct_inline() skips encryption entirely

**Root cause:** The `_try_direct_inline()` method in `_inline_blob_mixin.py` serializes data to bytes via `handler.put_bytes()` entirely in-memory, bypassing `_write_blob()`. Since encryption is applied inside `_write_blob()`, the direct inline path never encrypts. The plaintext blob_data is stored directly in the metadata row.

**Impact:** When `max_inline_size > 0` and `enable_content_encryption=True`, small entries bypass encryption. This is a security hole — the user expects all data to be encrypted, but inline entries are stored as cleartext in the metadata database.

**Mitigation (current):** `max_inline_size` defaults to 0, so this only affects users who explicitly opt into inlining. But the config docs recommend `max_inline_size=4000` for SQLite, making this a realistic scenario.

### Bug 3: rotate_key() ignores inline entries

**Root cause:** Both `core.py:rotate_key()` and `blob_store.py:rotate_key()` iterate entries and skip those without `actual_path` (`if not actual_path_str: continue`). Inline entries have `actual_path=None`.

**Impact:** After key rotation, inline entries remain encrypted with the old key (if they were encrypted at all). Subsequent reads with the new derived key will fail with `CacheIntegrityError`.

## MVP Recommendation

Prioritize:
1. **Encryption metadata columns for SQLite and PostgreSQL** — fixes silent data loss (table stakes, HIGH priority)
2. **Cross-backend encryption test parity** — parameterize existing tests across all 3 backends (table stakes, LOW complexity)
3. **Encryption + inline blob config validation** — fail or warn at init when both are enabled simultaneously (table stakes, LOW complexity)
4. **Encrypted inline blobs** — encrypt blob_data before storing in metadata row (differentiator, MEDIUM complexity)

Defer:
- **Key rotation for inline blobs** — only relevant after encrypted inline blobs are implemented
- **Backend migration preserving encryption** — falls out naturally from the column fix

## Sources

- Source code analysis: `src/cacheness/metadata/sqlite_backend.py` (put_entry/get_entry), `src/cacheness/storage/backends/postgresql_backend.py` (_upsert_entry/_entry_to_dict), `src/cacheness/encryption.py`, `src/cacheness/storage/blob_store.py` (_write_blob/_read_blob), `src/cacheness/_inline_blob_mixin.py`, `src/cacheness/core.py` (put/get flows)
- Test analysis: `tests/test_encryption_at_rest.py` — all 7 backend references are `"json"`
- Config analysis: `src/cacheness/config.py` — `max_inline_size` default 0, `BlobStorageConfig` recommends 4000 for SQLite
