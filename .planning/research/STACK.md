# Technology Stack

**Project:** Cacheness v0.11.0 — Cross-Backend Encryption Hardening
**Researched:** 2026-04-06

## Recommendation: No New Dependencies

**No new packages, libraries, or version bumps are needed.** The existing stack already provides every primitive required for cross-backend encryption. The problem is entirely an integration gap — the SQLite and PostgreSQL backends silently drop encryption metadata fields because they lack dedicated columns for them.

## Existing Stack (Already Sufficient)

### Encryption Library
| Technology | Version | Purpose | Status |
|------------|---------|---------|--------|
| `cryptography` | ≥41.0.0 | AES-256-GCM encryption, HKDF-SHA256 key derivation | ✅ Already handles all encryption operations |

AES-256-GCM with random 12-byte IV and 16-byte auth tag — no additional cryptographic primitives needed. The `AESGCM` class from `cryptography.hazmat.primitives.ciphers.aead` and `InvalidTag` exception cover the full encrypt/decrypt/authenticate cycle.

### Database Backends
| Technology | Version | Purpose | Status |
|------------|---------|---------|--------|
| SQLAlchemy | ≥2.0.0 | ORM for SQLite and PostgreSQL metadata backends | ✅ Supports `ALTER TABLE ADD COLUMN` migrations |
| psycopg[binary] | ≥3.1.0 | PostgreSQL adapter | ✅ Handles `BYTEA` and `VARCHAR` seamlessly |

### Binary Data Handling
| Backend | Column Type | Binary Support | Encryption Compatibility |
|---------|-------------|----------------|--------------------------|
| JSON | Native dict | Full (stores entire metadata dict) | ✅ Works today — encryption_algorithm and encryption_iv preserved |
| SQLite | `BLOB` (blob_data), `TEXT`/`VARCHAR` (metadata fields) | Full — SQLite BLOB stores arbitrary bytes | ⚠️ **Broken** — missing columns for encryption metadata |
| PostgreSQL | `BYTEA` (blob_data), `VARCHAR` (metadata fields) | Full — BYTEA stores arbitrary bytes | ⚠️ **Broken** — missing columns for encryption metadata |

## Root Cause: Missing Schema Columns

### What's Broken

Both SQLite and PostgreSQL `put_entry()` methods extract known metadata fields into dedicated columns by **popping** them from the metadata dict:

```
object_type, storage_format, serializer, compression_codec, actual_path,
file_hash, entry_signature, s3_etag, cache_key_params, metadata_dict, inline_ext
```

After these pops, `encryption_algorithm` and `encryption_iv` remain in the leftover dict — which is then **discarded**. Neither backend has columns for these fields, and neither stores leftover metadata.

On `get_entry()`, only known column values are reconstructed into the returned metadata dict — so `encryption_algorithm` and `encryption_iv` are never returned to the caller.

**Result:** `_read_blob()` sees `encryption_algorithm = None` → treats blob as plaintext → handler receives ciphertext → crash or corruption.

### What Needs Adding

Two new columns per cache_entries table (both SQLite and PostgreSQL):

| Column | Type | Purpose |
|--------|------|---------|
| `encryption_algorithm` | `VARCHAR(20)` / `String(20)` | Algorithm identifier (e.g. `"aes-256-gcm"`) |
| `encryption_iv` | `VARCHAR(32)` / `String(32)` | Hex-encoded 12-byte IV (24 hex chars) |

These are small, fixed-size strings — no binary handling needed. The IV is already hex-encoded by `encrypt_blob()`.

### Schema Migration Path

**SQLite:** Add a `v3 → v4` migration (following the existing `_sqlite_migrate_v2_to_v3` pattern) that runs `ALTER TABLE ADD COLUMN` for both fields. Fully idempotent with `IF NOT EXISTS` column check via `PRAGMA table_info`.

**PostgreSQL:** Add a `v3 → v4` migration using `ALTER TABLE ADD COLUMN IF NOT EXISTS`.

**ORM Models:** Add `encryption_algorithm = Column(String(20), nullable=True)` and `encryption_iv = Column(String(32), nullable=True)` to:
- `CacheEntryMixin` in `src/cacheness/metadata/_compat.py`
- `PgCacheEntryMixin` in `src/cacheness/storage/backends/postgresql_backend.py`

## Second Issue: Inline Blob + Encryption Interaction

### `_try_direct_inline` Bypasses Encryption

The zero-disk inline path (`_try_direct_inline`) calls `handler.put_bytes()` and stores the result directly as `blob_data` — it **never** calls `_write_blob()`, so encryption is never applied. Inline blobs stored via this path are plaintext even when `enable_content_encryption=True`.

**Fix options (no new deps needed):**
1. Apply `encrypt_blob()` to the bytes in `_try_direct_inline` before storing as `blob_data` — sets encryption_algorithm/encryption_iv in metadata
2. Skip `_try_direct_inline` when encryption is enabled (force disk path which encrypts correctly)

Option 2 is simpler and avoids subtle correctness bugs. Option 1 is more performant but requires careful `_read_inline_blob` changes.

### `_try_inline_blob` Partially Works

This path reads from a file already encrypted by `_write_blob`, so encrypted ciphertext is stored as `blob_data`. On read, `_read_inline_blob` tries the `get_bytes()` fast path first — which would fail since it receives ciphertext. The fallback to temp file + `_read_blob()` does decrypt correctly.

**Fix:** Either skip the `get_bytes()` fast path when encryption metadata is present, or decrypt the bytes before calling `get_bytes()`.

## What NOT to Add

| Not Needed | Reason |
|------------|--------|
| New encryption library | `cryptography>=41.0.0` already provides everything |
| SQLite encryption extension (SEE/sqlcipher) | Overkill — we encrypt blobs at the application layer, not the database |
| Additional Python DB drivers | psycopg3 and SQLAlchemy already handle all binary/VARCHAR operations |
| Key management service integration | Out of scope — local key file approach is sufficient |
| New serialization format for encrypted metadata | Plain VARCHAR columns for algo+IV are sufficient |

## Installation

No changes to `pyproject.toml` dependencies. Existing extras already cover all needs:

```bash
# Encryption support (already defined)
pip install cacheness[encryption]    # → cryptography>=41.0.0

# PostgreSQL support (already defined)
pip install cacheness[postgresql]    # → psycopg[binary]>=3.1.0, sqlalchemy>=2.0.0
```

## Confidence Assessment

| Finding | Confidence | Source |
|---------|------------|--------|
| No new deps needed | HIGH | Direct code inspection of encryption.py, config.py, blob_store.py |
| Missing columns root cause | HIGH | Traced put_entry/get_entry in both sqlite_backend.py and postgresql_backend.py |
| Inline blob interaction bugs | HIGH | Direct code inspection of _inline_blob_mixin.py, _write_blob, _read_blob |
| Migration path (ALTER TABLE) | HIGH | Existing v2→v3 migration pattern already demonstrates this exact approach |
| SQLite BLOB/VARCHAR handles encryption metadata | HIGH | SQLite natively supports these types, confirmed by existing blob_data column |
| PostgreSQL BYTEA/VARCHAR handles encryption metadata | HIGH | PG model already has LargeBinary and String columns, confirmed in postgresql_backend.py |

## Sources

- `src/cacheness/encryption.py` — encryption primitives (lines 1-104)
- `src/cacheness/storage/blob_store.py` — encryption integration in `_write_blob` (L1196-1260) and `_read_blob` (L1262-1330)
- `src/cacheness/metadata/sqlite_backend.py` — `put_entry` (L685-805), `get_entry` (L585-685)
- `src/cacheness/storage/backends/postgresql_backend.py` — `_upsert_entry` (L920-1050), `_entry_to_dict` (L1050-1100)
- `src/cacheness/metadata/_compat.py` — `CacheEntryMixin` column definitions (L110-148)
- `src/cacheness/_inline_blob_mixin.py` — inline blob paths (L1-200)
- `tests/test_encryption_at_rest.py` — confirms JSON-only encryption tests (L1-80)
- `tests/test_concurrent_security.py` — documents known SQLite+encryption incompatibility (L63)
