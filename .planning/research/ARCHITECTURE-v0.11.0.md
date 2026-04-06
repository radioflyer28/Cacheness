# Architecture Patterns: Cross-Backend Encryption Hardening

**Domain:** Python disk caching library — encryption at rest across metadata backends
**Researched:** 2026-04-06

## Recommended Architecture

### Current Data Flow (Write Path)

```
UnifiedCache.put(data)
  ├─ _try_direct_inline(data, handler)      ← zero-disk path (fast)
  │    └─ handler.put_bytes() → blob_bytes
  │         (encryption NOT applied here — gap #1)
  │
  └─ _write_blob(data, base_path)           ← disk path
       └─ handler.put(data, path)
       └─ encrypt_blob(plaintext, key)      ← encryption applied here
       └─ blob_backend.write_blob_from_path()
       └─ result.extra["encryption_algorithm"] = "aes-256-gcm"
       └─ result.extra["encryption_iv"] = iv.hex()
  │
  ├─ _build_metadata_dict(result, file_hash)
  │    └─ merges result.extra (incl. encryption fields) into metadata_dict
  │
  ├─ _try_inline_blob(result, file_hash, cleanup)  ← post-disk inline
  │    └─ reads blob bytes AFTER encryption
  │         (inlines ciphertext — gap #2: never decrypted on read)
  │
  └─ metadata_backend.put_entry(cache_key, entry_data)
       ├─ JSON: stores nested metadata dict as-is → encryption fields preserved ✓
       ├─ SQLite: pops known fields from metadata → encryption fields DROPPED ✗
       └─ PostgreSQL: pops known fields → encryption fields DROPPED ✗
```

### Current Data Flow (Read Path)

```
UnifiedCache.get(cache_key)
  └─ metadata_backend.get_entry(cache_key)
       ├─ JSON: returns metadata dict as-is → encryption fields available ✓
       ├─ SQLite: reconstructs metadata from columns → no encryption fields ✗
       └─ PostgreSQL: reconstructs metadata from columns → no encryption fields ✗
  │
  ├─ is_inline? → _read_inline_blob(entry, data_type, metadata)
  │    └─ handler.get_bytes(blob_bytes, metadata)
  │         (no decryption — gap #2: blob_bytes may be ciphertext)
  │    └─ fallback: _read_blob(tmp_path, ...) → decryption WOULD trigger
  │
  └─ _read_blob(file_path, data_type, metadata)
       └─ checks metadata["encryption_algorithm"]
       └─ decrypt_blob(ciphertext, key, iv)    ← requires encryption fields
            (fails for SQLite/PG — fields not in metadata)
```

### Component Boundaries

| Component | Responsibility | Encryption Awareness |
|-----------|---------------|---------------------|
| `encryption.py` | AES-256-GCM primitives, key derivation | Fully aware (core module) |
| `BlobStore._write_blob()` | Serialize → encrypt → persist blob | Encrypts, injects `encryption_*` into `result.extra` |
| `BlobStore._read_blob()` | Decrypt → deserialize blob | Decrypts using `metadata["encryption_algorithm/iv"]` |
| `BlobStore.put()` | Standalone blob store write | Own encryption path (parallel, works with JSON) |
| `BlobStore.get()` | Standalone blob store read | Own decryption path (parallel, works with JSON) |
| `InlineBlobMixin._try_inline_blob()` | Post-disk inlining | Reads bytes after encryption — inlines ciphertext |
| `InlineBlobMixin._try_direct_inline()` | Zero-disk inlining | **No encryption** — bypasses blob pipeline entirely |
| `InlineBlobMixin._read_inline_blob()` | Read inline blob | **No decryption** — passes raw bytes to handler |
| `_build_metadata_dict()` | Build metadata from HandlerResult | Passes through `result.extra` (encryption fields) |
| `SqliteBackend.put_entry()` | Store metadata in SQLite | **Drops** unknown fields from metadata dict |
| `SqliteBackend.get_entry()` | Load metadata from SQLite | **Cannot reconstruct** encryption fields |
| `PostgresBackend._upsert_entry()` | Store metadata in PostgreSQL | **Drops** unknown fields from metadata dict |
| `PostgresBackend._entry_to_dict()` | Load metadata from PostgreSQL | **Cannot reconstruct** encryption fields |
| `JsonBackend.put_entry()` | Store metadata in JSON | Preserves entire metadata dict ✓ |

## Identified Gaps

### Gap 1: SQLite/PostgreSQL Drop Encryption Metadata Fields

**Root cause:** Both `SqliteBackend.put_entry()` and `PostgresBackend._upsert_entry()` extract known fields from the `metadata` dict into dedicated columns via `.pop()`, then **discard the remaining fields**. The encryption fields `encryption_algorithm` and `encryption_iv` are not in the known-fields list.

**Evidence:**
- SQLite `put_entry()` (sqlite_backend.py L685) pops: `object_type`, `storage_format`, `serializer`, `compression_codec`, `actual_path`, `file_hash`, `entry_signature`, `s3_etag`, `cache_key_params`, `metadata_dict`, `inline_ext`, `data_type`, `_full_metadata`
- PostgreSQL `_upsert_entry()` (postgresql_backend.py L907) pops the same set
- Neither pops or stores `encryption_algorithm` or `encryption_iv`
- No dedicated columns exist for these fields in either ORM model

**Impact:** Encrypted entries stored via SQLite/PostgreSQL cannot be read back — `_read_blob()` checks `metadata["encryption_algorithm"]` and finds nothing, so it reads ciphertext as plaintext, causing handler deserialization failure.

### Gap 2: Inline Blob + Encryption Interaction

**Two sub-issues:**

**2a: `_try_inline_blob()` inlines ciphertext, not plaintext.**
After `_write_blob()` encrypts and writes ciphertext to disk, `_try_inline_blob()` reads back those bytes and stores them as `blob_data`. The inlined bytes are **ciphertext**. On read, `_read_inline_blob()` passes these bytes to `handler.get_bytes()` without decryption, causing deserialization failure.

The fallback path (`_read_blob()` via temp file) **would** decrypt because `_read_blob()` checks `encryption_algorithm`, but only if the encryption metadata fields are present — which circles back to Gap 1 for SQLite/PG.

**2b: `_try_direct_inline()` bypasses encryption entirely.**
The zero-disk path calls `handler.put_bytes()` directly and stores plaintext as `blob_data`. No encryption is applied. This means inline entries via the direct path contain **unencrypted data** in the metadata row, defeating encryption-at-rest when the threat model includes metadata store compromise.

### Gap 3: Key Rotation Skips Inline Blobs

Both `UnifiedCache.rotate_key()` and `BlobStore.rotate_key()` iterate entries, read encrypted blobs from disk, decrypt with old key, re-encrypt with new key, and write back. However:
- Entries with `is_inline=1` have `actual_path=None` — the rotation code skips them (`if not actual_path_str: continue`)
- Inline ciphertext is never re-encrypted during key rotation

### Gap 4: BlobStore.put() and BlobStore._write_blob() Have Parallel Encryption Paths

The standalone `BlobStore.put()` method has its own encryption logic independent of `_write_blob()`:
- `put()`: reads handler output, encrypts, writes back, sets `encryption_meta` dict, stores in `custom_metadata`
- `_write_blob()`: reads handler output, encrypts, writes back, sets `result.extra` dict

Both work, but the dual-path maintenance burden is a risk. Changes to one may not be reflected in the other.

## Patterns to Follow

### Pattern 1: Add Dedicated Encryption Columns

**What:** Add `encryption_algorithm VARCHAR(20)` and `encryption_iv VARCHAR(32)` columns to all three backend schemas.

**When:** This is the primary fix for Gap 1.

**Why not use the JSON overflow field:** The encryption fields are critical for data retrieval, not optional metadata. They must survive the put_entry/get_entry round-trip reliably. Dedicated columns ensure:
1. Fields cannot be silently dropped
2. Query capability (`WHERE encryption_algorithm IS NOT NULL`)
3. Consistent with existing pattern (file_hash, entry_signature, s3_etag are all dedicated columns)

**Where to change:**

| File | Change |
|------|--------|
| `metadata/_compat.py` `CacheEntryMixin` | Add 2 columns |
| `metadata/_compat.py` `_get_namespace_models()` | Already dynamic — inherits from mixin |
| `metadata/sqlite_backend.py` | New migration (v3→v4), update `put_entry`/`get_entry` |
| `storage/backends/postgresql_backend.py` `PgCacheEntryMixin` | Add 2 columns |
| `storage/backends/postgresql_backend.py` | New migration (v3→v4), update `_upsert_entry`/`_entry_to_dict` |
| `metadata/json_backend.py` | No change needed — stores everything |

### Pattern 2: Decrypt Inline Blobs on Read

**What:** In `_read_inline_blob()`, check for `encryption_algorithm` in metadata. If present, decrypt `blob_bytes` before passing to handler.

**When:** After Gap 1 is fixed (encryption metadata must be available).

```python
# In _read_inline_blob():
blob_bytes = entry["blob_data"]
enc_algo = metadata.get("encryption_algorithm")
if enc_algo and self._blob_store._encryption_key:
    from .encryption import decrypt_blob
    iv = bytes.fromhex(metadata["encryption_iv"])
    blob_bytes = decrypt_blob(blob_bytes, self._blob_store._encryption_key, iv)
```

### Pattern 3: Encrypt Direct-Inline Blobs

**What:** In `_try_direct_inline()`, if encryption is enabled, encrypt the serialized bytes before returning them for inline storage.

**When:** After Pattern 2 is in place (so encrypted inline blobs can be read back).

```python
# In _try_direct_inline(), after handler.put_bytes():
if self._blob_store._encryption_key is not None:
    from .encryption import encrypt_blob
    blob_bytes, iv, algo = encrypt_blob(blob_bytes, self._blob_store._encryption_key)
    # Return encryption metadata alongside blob
    return {
        "blob_data": blob_bytes,
        ...
        "encryption_algorithm": algo.decode(),
        "encryption_iv": iv.hex(),
    }
```

### Pattern 4: Key Rotation for Inline Entries

**What:** During key rotation, handle `is_inline=1` entries by decrypting `blob_data` with old key and re-encrypting with new key in-place.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Storing Encryption Fields in JSON Overflow Column

**What:** Keeping encryption fields in `metadata_dict` JSON text/JSONB column instead of dedicated columns.
**Why bad:** `metadata_dict` is the user-facing kwargs column for query support. Mixing system-critical fields here creates parsing fragility, makes queries harder, and breaks the existing pattern where all system fields have dedicated columns.
**Instead:** Dedicated columns with schema migrations.

### Anti-Pattern 2: Disabling Inline Blobs When Encryption Is Enabled

**What:** Skipping inlining entirely to avoid the encrypt/decrypt complexity.
**Why bad:** Inline blobs provide significant performance benefits for small entries. Disabling them for encrypted caches penalizes the common case.
**Instead:** Properly encrypt inline blobs and decrypt on read.

### Anti-Pattern 3: Duplicating Encryption Logic in a Third Location

**What:** Adding encryption handling to `_read_inline_blob()` and `_try_direct_inline()` as standalone implementations separate from `_write_blob()`/`_read_blob()`.
**Why bad:** Three independent call sites for encrypt/decrypt increases inconsistency risk.
**Instead:** Consider extracting shared encrypt/decrypt helpers that both the blob path and inline path can call.

## Recommended Build Order

Based on dependency analysis, the following order minimizes risk:

### Phase A: Schema + Storage (fixes Gap 1)
1. Add `encryption_algorithm` and `encryption_iv` columns to ORM models (`_compat.py`, `postgresql_backend.py`)
2. Add schema migration for SQLite (v3→v4 or current→next)
3. Add schema migration for PostgreSQL
4. Update `SqliteBackend.put_entry()` to pop/store encryption fields
5. Update `SqliteBackend.get_entry()` to reconstruct encryption fields in metadata
6. Update `PostgresBackend._upsert_entry()` to store encryption fields
7. Update `PostgresBackend._entry_to_dict()` to reconstruct encryption fields
8. **Test:** Encrypted put/get roundtrip with SQLite and PostgreSQL backends

**This phase alone unblocks basic encryption for all backends.**

### Phase B: Inline Blob Encryption (fixes Gaps 2a, 2b)
1. Update `_read_inline_blob()` to decrypt ciphertext inline blobs
2. Update `_try_direct_inline()` to encrypt before storing
3. Update `_try_inline_blob()` — already stores ciphertext, just ensure metadata flows
4. Propagate encryption metadata through `_build_metadata_dict` for inline paths
5. **Test:** Encrypted inline blob put/get roundtrip (all backends)

### Phase C: Key Rotation for Inline (fixes Gap 3)
1. Update `UnifiedCache.rotate_key()` to re-encrypt inline blob_data
2. Update `BlobStore.rotate_key()` to re-encrypt inline blob_data
3. **Test:** Key rotation with mix of inline and file-backed encrypted entries

### Phase D: Cross-Backend Test Parity
1. Parameterize existing encryption tests across JSON/SQLite backends
2. Add PostgreSQL encryption tests (Docker-gated)
3. Add inline+encryption integration tests for all backends
4. Add key rotation tests with inline entries

## New vs Modified Components

| Component | Status | Changes |
|-----------|--------|---------|
| `encryption.py` | **No change** | Primitives are backend-agnostic |
| `metadata/_compat.py` | **Modified** | +2 columns on `CacheEntryMixin` |
| `metadata/sqlite_backend.py` | **Modified** | +migration, +put/get encryption fields |
| `metadata/json_backend.py` | **No change** | Already preserves all metadata fields |
| `storage/backends/postgresql_backend.py` | **Modified** | +migration, +put/get encryption fields |
| `_inline_blob_mixin.py` | **Modified** | Encrypt direct-inline, decrypt on read |
| `core.py` | **Modified** | Key rotation handles inline entries |
| `storage/blob_store.py` | **Modified** | Key rotation handles inline entries |
| `config.py` | **No change** | SecurityConfig already complete |
| `interfaces.py` | **No change** | No new interfaces needed |

## Scalability Considerations

| Concern | Current | After Hardening |
|---------|---------|-----------------|
| Schema migration | v3 (all backends) | v4 with 2 new nullable columns — backward compatible |
| Inline blob size | Unchanged (ciphertext ~16 bytes larger than plaintext due to GCM tag) | No meaningful size impact |
| Read latency | One decrypt call for file blobs | Same + decrypt for inline blobs (in-memory, ~microseconds) |
| Key rotation | Skips inline entries | Handles all entry types — slightly longer rotation time |
| Storage overhead | 0 | 2 new nullable VARCHAR columns (~0 bytes when NULL) |

## Sources

All findings from direct codebase analysis (HIGH confidence — no external sources needed):
- `src/cacheness/storage/blob_store.py` — `_write_blob()`, `_read_blob()`, `put()`, `get()`, `rotate_key()`
- `src/cacheness/core.py` — `put()`, `get()`, `rotate_key()`, `_init_blob_store()`
- `src/cacheness/_inline_blob_mixin.py` — `_try_inline_blob()`, `_try_direct_inline()`, `_read_inline_blob()`
- `src/cacheness/metadata/sqlite_backend.py` — `put_entry()`, `get_entry()`, schema migrations
- `src/cacheness/storage/backends/postgresql_backend.py` — `_upsert_entry()`, `_entry_to_dict()`, schema migrations
- `src/cacheness/metadata/json_backend.py` — `put_entry()` stores metadata as-is
- `src/cacheness/metadata/_compat.py` — ORM column definitions
- `src/cacheness/encryption.py` — AES-256-GCM primitives
- `src/cacheness/config.py` — `SecurityConfig`
- `tests/test_encryption_at_rest.py` — existing tests (JSON-only)
- `tests/test_concurrent_security.py` — documents known SQLite incompatibility
