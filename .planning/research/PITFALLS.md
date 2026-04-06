# Domain Pitfalls

**Domain:** Cross-backend encryption hardening for a disk caching library
**Researched:** 2026-04-06
**Confidence:** HIGH — all findings verified against source code

## Critical Pitfalls

Mistakes that cause data loss, silent corruption, or false test confidence.

### Pitfall 1: Encryption metadata silently dropped by SQLite and PostgreSQL backends

**What goes wrong:** `encryption_algorithm` and `encryption_iv` are stored in the nested `metadata` dict by `_write_blob()` via `result.extra`. When SQLite's `put_entry()` processes this dict, it `pop()`s known fields (`object_type`, `storage_format`, `serializer`, `compression_codec`, `actual_path`, `file_hash`, `entry_signature`, `s3_etag`, `cache_key_params`, `metadata_dict`, `inline_ext`) into dedicated columns. `encryption_algorithm` and `encryption_iv` are NOT in this list — they remain in the dict after all pops, but that residual dict is never written to any column. The INSERT statement has no column for remaining metadata. These fields are silently discarded.

The same issue exists in PostgreSQL's `_upsert_entry()` — identical pop list, no remaining-metadata storage.

The JSON backend works because it stores the entire `metadata` dict as-is in a JSON file.

**Why it happens:** The encryption feature (v0.10.0) added metadata fields without adding corresponding columns to the SQLite/PostgreSQL schemas. The column-extraction pattern (`metadata.pop("field")`) was designed before encryption existed, and encryption fields were never added to the extraction list or schema.

**Consequences:**
- Encrypted blobs stored via SQLite/PG backends become **permanently unreadable** — the IV needed for AES-GCM decryption is lost
- `_read_blob()` checks `metadata.get("encryption_algorithm")` — returns `None` → skips decryption → handler receives ciphertext → returns garbage or raises
- No error is raised at write time — data appears to store successfully
- Key rotation's re-encrypt loop skips entries where `meta.get("encryption_algorithm") is None` — entries become orphaned ciphertext

**Prevention:**
1. Add `encryption_algorithm VARCHAR(20)` and `encryption_iv VARCHAR(24)` columns to SQLite and PostgreSQL schemas
2. Add schema migration (v3→v4) to ALTER TABLE for existing databases
3. Update `put_entry()` pop lists to extract these fields
4. Update `get_entry()` reconstruction to include these fields in the returned metadata dict
5. Add a cross-backend roundtrip test: encrypt with backend X, verify `encryption_algorithm` and `encryption_iv` survive `put_entry()`→`get_entry()` for ALL backends

**Detection:** Any test that does `put()` with encryption + SQLite/PG backend, then `get()` and checks the result. Currently undetected because all encryption tests hardcode `metadata_backend="json"`.

**Severity:** DATA LOSS — this is the #1 priority fix.

### Pitfall 2: Inline blob storage bypasses encryption on the direct path

**What goes wrong:** `_try_direct_inline()` serializes data in-memory via `handler.put_bytes()` — this path never calls `encrypt_blob()`. The resulting plaintext bytes are stored directly in `blob_data`. When encryption is enabled, `_try_direct_inline()` stores **plaintext** in the metadata row, defeating the purpose of encryption at rest.

**Why it happens:** The direct inline path was designed for performance (zero disk I/O). Encryption was added to `_write_blob()` only, which the direct inline path deliberately skips.

**Consequences:**
- Small values (≤ `max_inline_size`) are stored in plaintext in the metadata database even when `enable_content_encryption=True`
- SQLite databases and JSON metadata files contain unencrypted secrets
- The `encryption_algorithm` field is not set in metadata, so `_read_inline_blob()` doesn't attempt decryption — reads work correctly, masking the bug
- Security audit would reveal plaintext in database

**Prevention:**
1. Either encrypt the blob bytes before storing in `blob_data` (add encryption to `_try_direct_inline`)
2. Or disable direct inline when encryption is enabled (`if self._blob_store._encryption_key is not None: return None` at the top of `_try_direct_inline`)
3. Option 2 is simpler and safer — encryption users are already paying a performance cost

**Detection:** Write a test that enables encryption, stores a small value (will inline), then reads the raw `blob_data` from the metadata entry and asserts it's not equal to the plaintext serialization.

### Pitfall 3: Disk-based inline blob stores encrypted bytes without decryption on read

**What goes wrong:** `_try_inline_blob()` reads blob bytes AFTER `_write_blob()` encrypts them. So `blob_data` contains encrypted ciphertext. On read, `_read_inline_blob()` first tries the fast path `handler.get_bytes(blob_bytes, metadata)` — passing encrypted bytes to deserializer → failure or garbage. The slow path writes to temp file and calls `_read_blob()`, which DOES check `encryption_algorithm` and decrypt — but only if that metadata field survives the backend round-trip (see Pitfall 1).

**Why it happens:** The inline blob path and the encryption path were developed independently. `_try_inline_blob` doesn't know it's reading encrypted bytes; `_read_inline_blob` doesn't know it needs to decrypt before passing to handler.

**Consequences:**
- With JSON backend: slow path works (metadata preserved), fast path fails (handler gets ciphertext)
- With SQLite/PG: both paths fail (encryption metadata lost per Pitfall 1)
- Handler may raise cryptic deserialization errors or return corrupt data

**Prevention:**
1. `_read_inline_blob()` should check for `encryption_algorithm` in metadata and decrypt `blob_bytes` before passing to handler
2. Alternatively, `_try_inline_blob()` should decrypt before inlining (store plaintext in blob_data), but this weakens encryption-at-rest since plaintext is in metadata
3. Best approach: disable disk-based inline when encryption is enabled (same as Pitfall 2 prevention)

**Detection:** Test encrypted put with `max_inline_size > 0`, verify roundtrip works. Currently untested.

## Moderate Pitfalls

### Pitfall 4: Test fixtures hardcode JSON backend — false confidence

**What goes wrong:** `_make_encrypted_cache()` and `_make_encrypted_blobstore()` in `test_encryption_at_rest.py` both hardcode `metadata_backend="json"`. The `encrypted_cache` fixture in `test_concurrent_security.py` explicitly notes: *"Uses JSON backend because encryption+SQLite has a known incompatibility."* All encryption tests pass because JSON preserves all metadata fields. This creates false confidence that encryption "works" when it silently fails for 2 of 3 backends.

**Prevention:**
1. Parametrize encryption test fixtures across all three backends: `@pytest.mark.parametrize("backend", ["json", "sqlite"])`
2. Add PostgreSQL parametrization gated behind `postgres_available` fixture
3. The moment you parametrize, Pitfall 1 becomes immediately visible as test failures
4. After fixing schema, keep parametrized tests as regression guards

**Detection:** Review test file — search for `metadata_backend=` in encryption test fixtures. If all say `"json"`, coverage is incomplete.

### Pitfall 5: Key rotation re-encryption loop skips entries with lost encryption metadata

**What goes wrong:** Both `core.py` and `blob_store.py` `rotate_key()` methods check `if meta.get("encryption_algorithm") is None: continue` to skip non-encrypted entries. But with SQLite/PG backends, encrypted entries also have `encryption_algorithm == None` because the field was never stored (Pitfall 1). These entries are silently skipped during re-encryption — they remain encrypted with the OLD key, which is now overwritten. They become permanently unreadable.

**Why it happens:** The skip logic correctly handles the "mixed encrypted/unencrypted entries" case, but doesn't account for backends that lose the encryption_algorithm field.

**Consequences:**
- After key rotation with SQLite/PG, all previously encrypted entries are permanently lost
- No error reported — `RotationResult.re_encrypted` shows 0, which looks like "no entries needed re-encryption"
- This is the worst-case data loss scenario

**Prevention:** Fix Pitfall 1 first (store encryption metadata in all backends). Then key rotation automatically works correctly.

**Detection:** Test key rotation with SQLite backend, assert `result.re_encrypted > 0` when entries were encrypted.

### Pitfall 6: PostgreSQL BYTEA vs. SQLite BLOB binary handling differences

**What goes wrong:** Encrypted ciphertext is raw binary data. When stored as inline blob_data or when encryption metadata (IV as hex string) flows through PostgreSQL's text columns, encoding assumptions may differ. PostgreSQL uses UTF-8 by default; SQLite is encoding-agnostic. If any layer converts bytes through a text encoding path (JSON serialization of metadata, logging, etc.), binary data gets corrupted.

**Prevention:**
1. Ensure `encryption_iv` is always stored as hex string (currently done: `iv.hex()`) — never raw bytes in text columns
2. Ensure `blob_data` column is BLOB/BYTEA (currently correct in both schemas)
3. Test that binary ciphertext survives a full put/get cycle through PostgreSQL's BYTEA column
4. Never serialize `blob_data` through JSON (JSON backend already notes this: "blob_data is intentionally NOT stored in JSON backend")

**Detection:** Integration test with PostgreSQL: store encrypted data, retrieve blob_data column directly, compare bytes.

### Pitfall 7: `_entry_to_dict()` / `get_entry()` metadata reconstruction doesn't include encryption fields

**What goes wrong:** Even after adding columns for encryption metadata (Pitfall 1 fix), the `get_entry()` reconstruction code in both SQLite and PostgreSQL backends must be updated to include these fields in the returned `metadata` dict. The current pattern reconstructs metadata from an explicit list of column values — any new column not added to this reconstruction is silently omitted.

In SQLite (`get_entry`): metadata is built from explicit `if row.X is not None: metadata["X"] = row.X` checks. In PostgreSQL (`_entry_to_dict`): same pattern via `if entry.X: metadata["X"] = entry.X`.

**Prevention:**
1. When adding columns, always update BOTH `put_entry()` (pop + insert) AND `get_entry()` (reconstruction)
2. Write a "metadata roundtrip" test: create entry_data with all known fields, put_entry, get_entry, assert all fields present in returned dict

**Detection:** Unit test for each backend: `put_entry()` with encryption fields → `get_entry()` → assert `metadata["encryption_algorithm"]` exists.

## Minor Pitfalls

### Pitfall 8: `verify_integrity()` doesn't account for encryption in file hash verification

**What goes wrong:** The file hash is computed on ciphertext (correct — hash-of-ciphertext is the intended design per the encryption architecture). But `verify_integrity()` re-computes the hash from the file on disk. If the file is correctly encrypted, this works. However, if inline blob entries have their `blob_data` encrypted but `file_hash` was computed from plaintext (or vice versa), integrity checks produce false positives/negatives.

**Prevention:** Ensure hash computation timing is consistent: always hash after encryption (current design). Verify this holds for inline blobs too.

### Pitfall 9: Schema migration testing gap for encryption columns

**What goes wrong:** Adding encryption columns requires a v3→v4 migration for both SQLite and PostgreSQL. If the migration is only tested with fresh databases, existing production databases that upgrade may fail. The migration must handle: (1) ALTER TABLE ADD COLUMN, (2) databases that already have entries with no encryption fields (nullable columns), (3) databases that were migrated from v2→v3 (already have inline columns).

**Prevention:**
1. Test migration with pre-populated v3 databases
2. Encryption columns MUST be nullable (existing entries won't have encryption data)
3. Test that non-encrypted entries still work after migration (encryption_algorithm = NULL)

### Pitfall 10: `BlobStore.put()` duplicates encryption logic from `_write_blob()`

**What goes wrong:** `BlobStore.put()` (the standalone API) has its own encryption block that duplicates `_write_blob()`'s encryption logic. `UnifiedCache.put()` calls `_write_blob()`. When fixing encryption metadata handling, changes must be applied to BOTH paths. Missing one path creates inconsistency — one of the two write paths works while the other doesn't.

The repo memory confirms this: *"UnifiedCache.put() does NOT call BlobStore.put() — it uses _write_blob() instead."*

**Prevention:**
1. Any encryption fix must grep for ALL occurrences of `encrypt_blob` / `encryption_algorithm` / `encryption_iv` and update every site
2. Consider refactoring `BlobStore.put()` to call `_write_blob()` internally (but this is a larger refactor)
3. At minimum, test both paths: `BlobStore.put()` + `BlobStore.get()` AND `UnifiedCache.put()` + `UnifiedCache.get()` for each backend

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Add encryption columns to SQLite/PG | Missing `get_entry()` reconstruction (Pitfall 7) | Update both put and get in same commit |
| Schema migration v3→v4 | Forgetting nullable constraint | All new columns MUST be nullable |
| Parametrize encryption tests | Tests fail because inline path broken | Disable inline for encrypted caches first, or fix inline+encryption |
| Fix inline + encryption interaction | Direct inline stores plaintext | Disable inline when encryption enabled (simplest) |
| Key rotation with SQLite/PG | Silent skip of encrypted entries (Pitfall 5) | Fix Pitfall 1 first, verify rotation in same PR |
| BlobStore.put() standalone encryption | Fixing _write_blob but not put() | Grep for all encrypt_blob call sites |

## Recommended Fix Order

1. **Pitfall 1** (schema + put/get) — unblocks everything else
2. **Pitfall 7** (get_entry reconstruction) — part of same commit as Pitfall 1
3. **Pitfall 4** (parametrize tests) — validates fix for 1+7, catches regression
4. **Pitfall 2+3** (inline + encryption) — either fix or disable inline when encrypted
5. **Pitfall 5** (key rotation) — automatically fixed by Pitfall 1, but add test
6. **Pitfall 9** (migration tests) — validates schema migration robustness

## Sources

- [blob_store.py](src/cacheness/storage/blob_store.py) — `_write_blob()` L1196-1260, `_read_blob()` L1262-1330, `put()` L455-555, `get()` L560-680
- [sqlite_backend.py](src/cacheness/metadata/sqlite_backend.py) — `put_entry()` L686-810, `get_entry()` L590-680, schema L357-385
- [postgresql_backend.py](src/cacheness/storage/backends/postgresql_backend.py) — `_upsert_entry()` L908-1050, `_entry_to_dict()` L1050-1110, schema L639-665
- [json_backend.py](src/cacheness/metadata/json_backend.py) — `put_entry()` L124-175 (preserves full metadata dict)
- [_inline_blob_mixin.py](src/cacheness/_inline_blob_mixin.py) — `_try_inline_blob()` L18-80, `_try_direct_inline()` L80-145, `_read_inline_blob()` L145-195
- [core.py](src/cacheness/core.py) — `put()` L810-920, `rotate_key()` L440-530
- [test_encryption_at_rest.py](tests/test_encryption_at_rest.py) — all fixtures hardcode JSON backend
- [test_concurrent_security.py](tests/test_concurrent_security.py) — `encrypted_cache` fixture L60-90, documents "known incompatibility"
- [encryption.py](src/cacheness/encryption.py) — `encrypt_blob()`, `decrypt_blob()`, `derive_encryption_key()`