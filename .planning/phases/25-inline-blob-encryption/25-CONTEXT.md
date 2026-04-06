# Phase 25: Inline Blob Encryption - Context

**Gathered:** 2026-04-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Encrypt inline blobs at rest, just like file-backed blobs. Currently `_try_direct_inline()` stores plaintext bytes in `blob_data` metadata (bypassing the encryption in `_write_blob()`), and `_read_inline_blob()` has no decryption logic. Key rotation (`rotate_key()`) skips inline entries entirely (`if not actual_path_str: continue`). This phase closes all three gaps: encrypt on write, decrypt on read, rotate inline entries.

</domain>

<decisions>
## Implementation Decisions

### Write Path
- **D-01:** Encrypt inside `_try_direct_inline()` — after `handler.put_bytes()` produces raw bytes, call `encrypt_blob()` on those bytes when `self._blob_store._encryption_key` is not None. Return ciphertext as `blob_data` with `encryption_algorithm` and `encryption_iv` in the returned dict. Self-contained — caller doesn't need to know about encryption.
- **D-05:** `file_hash` is computed on the plaintext bytes (before encryption), consistent with file-backed behavior where hash represents original data integrity.

### Read Path
- **D-02:** Decrypt upfront in `_read_inline_blob()` — check `entry` metadata for `encryption_algorithm`. If present and encryption key is available, decrypt `blob_data` to plaintext before dispatching to fast path (`handler.get_bytes()`) or slow path (temp file → `_read_blob()`). One decrypt point, both paths get clean plaintext.

### Key Rotation
- **D-03:** Add inline branch in `rotate_key()` — when `actual_path` is None and entry has `encryption_algorithm` in metadata, read `blob_data` from full entry, decrypt with old key, re-encrypt with new key, update `blob_data` + `encryption_iv` in entry, call `put_entry()` back. Same loop, minimal code change.

### Storage Format
- **D-04:** Keep `blob_data` as raw bytes — no explicit base64 encoding. Each backend handles byte storage natively. JSON backend does not support inline blobs (only SQLite/PG have binary `blob_data` columns), so this is not a concern.

### Agent's Discretion
- Whether to add a guard clause in `_read_inline_blob()` when encryption metadata is present but no key is configured (return None vs raise)
- Error handling strategy for decryption failures in inline path (log + return None vs raise)
- Whether `_try_direct_inline()` size check should compare against `max_inline_size` before or after encryption (ciphertext is slightly larger than plaintext due to GCM tag)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Inline Blob Mixin
- `src/cacheness/_inline_blob_mixin.py` — `_try_direct_inline()` (L80-145), `_read_inline_blob()` (L145-195)

### Encryption Infrastructure
- `src/cacheness/encryption.py` — `encrypt_blob()`, `decrypt_blob()`, `derive_encryption_key()`
- `src/cacheness/storage/blob_store.py` — `_write_blob()` encryption pattern (L1196-1260), `_read_blob()` decryption pattern (L1262-1320)

### Key Rotation
- `src/cacheness/core.py` — `rotate_key()` re-encryption loop (~L347+), specifically the `if not actual_path_str: continue` line that skips inline entries

### Metadata Backends (blob_data handling)
- `src/cacheness/metadata/sqlite_backend.py` — `put_entry()` blob_data column handling
- `src/cacheness/storage/backends/postgresql_backend.py` — `_upsert_entry()` blob_data handling
- `src/cacheness/metadata/json_backend.py` L139 — confirms JSON backend does NOT support blob_data

### Test References
- `tests/test_write_intent.py` L148 — `test_inline_put_skips_intent()` confirms inline bypasses write intent
- `tests/test_handler_bytes_protocol.py` L260 — `test_inline_entry_has_no_blob_file()` confirms no disk file
- `tests/test_encryption_at_rest.py` — existing encryption tests (file-backed only)
- `tests/test_backend_parity.py` — cross-backend encryption parity tests (Phase 24)

### Phase 23 Context
- `.planning/phases/23-encryption-schema-storage/23-CONTEXT.md` — Schema decisions, encryption column layout

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`encrypt_blob()` / `decrypt_blob()`** in `encryption.py`: Already used by `_write_blob()` and `_read_blob()`. Same functions apply to inline bytes.
- **`_write_blob()` encryption pattern**: Read plaintext → `encrypt_blob()` → write ciphertext + set `encryption_algorithm`/`encryption_iv` in result.extra. Mirror this in `_try_direct_inline()`.
- **`_read_blob()` decryption pattern**: Check `encryption_algorithm` → if present, `decrypt_blob()` → pass plaintext to handler. Mirror this in `_read_inline_blob()`.

### Established Patterns
- **Encryption key access**: `self._blob_store._encryption_key` (from `_inline_blob_mixin.py` context, which is a mixin on UnifiedCache that has `self._blob_store`)
- **Encryption metadata fields**: `encryption_algorithm` (string, e.g. "AES-256-GCM") and `encryption_iv` (hex string) — stored in dedicated SQLite/PG columns since Phase 23
- **GCM overhead**: Ciphertext is ~28 bytes larger than plaintext (16-byte auth tag + 12-byte nonce). May affect `max_inline_size` threshold comparison.

### Integration Points
- `_try_direct_inline()` in `_inline_blob_mixin.py` — add encryption after `handler.put_bytes()`
- `_read_inline_blob()` in `_inline_blob_mixin.py` — add decryption before handler dispatch
- `rotate_key()` in `core.py` — add inline branch to the re-encryption loop
- No changes needed to metadata backends (blob_data column already exists, encryption columns already exist from Phase 23)

</code_context>

<specifics>
## Specific Ideas

- The `_try_direct_inline()` encryption should happen after the `len(blob_bytes) > max_inline` check on plaintext size, since the user's intention with `max_inline_size` is about original data size, not ciphertext size
- In `_read_inline_blob()`, if encryption metadata is present but no key is configured, follow the same pattern as `_read_blob()` which logs a warning and returns None
- For `rotate_key()` inline branch, update `file_hash` after re-encryption using the new plaintext (decrypt with new key, hash, store), and re-sign the entry

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

### Reviewed Todos (not folded)
- **Encryption at rest for metadata and blobs** — already folded into Phase 23 (ENC-01/ENC-02)
- **Store cacheness version in metadata** — already folded into Phase 23 (D-06)
- **Property-based stress testing for cache key serialization** — unrelated to inline encryption
- **Tiered pull-through cache** — unrelated to inline encryption

</deferred>

---

*Phase: 25-inline-blob-encryption*
*Context gathered: 2026-04-06*
