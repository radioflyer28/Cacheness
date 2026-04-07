---
phase: 25-inline-blob-encryption
plan: 01
status: complete
requirements_completed: [INLINE-01, INLINE-02]
---

## Summary

Added encryption/decryption to inline blob write and read paths in `_inline_blob_mixin.py`, plus propagated encryption metadata in all three callers of `_try_direct_inline()`.

## What Was Built

**File:** `src/cacheness/_inline_blob_mixin.py` (~30 lines added)

### `_try_direct_inline()` — Encrypt on write (D-01, D-05, INLINE-01)
- After `put_bytes()` serialization + xxhash computation, encrypts `blob_bytes` via `encrypt_blob()` when `self._blob_store._encryption_key` is not None
- `file_hash` computed on plaintext BEFORE encryption (D-05 consistency)
- `max_inline_size` check uses plaintext size (before encryption)
- Returns `encryption_algorithm` and `encryption_iv` in the dict via `**encryption_meta` spread

### `_read_inline_blob()` — Decrypt on read (D-02, INLINE-02)
- Checks `metadata.get("encryption_algorithm")` upfront
- If encrypted: decrypts `blob_data` via `decrypt_blob()` before handler dispatch
- Both fast path (`get_bytes`) and slow path (temp file) receive plaintext
- Returns None with warning when encrypted but no key configured (matches `_read_blob()` pattern)

**File:** `src/cacheness/core.py` (~5 lines added)
- Propagates `encryption_algorithm` and `encryption_iv` from `direct` dict into `metadata_dict` in the put flow

**File:** `src/cacheness/_storage_mode_mixin.py` (~5 lines added)
- Same encryption metadata propagation for storage mode put path

**File:** `src/cacheness/_update_mixin.py` (~5 lines added)
- Same encryption metadata propagation for update path

## Key Decisions

- Encryption metadata stored as `dict[str, str]` (`encryption_meta`), spread into return dict — empty when encryption disabled, preserving backward compat
- `_try_inline_blob()` (file-path-based inlining) NOT modified — it reads bytes from files already encrypted by `_write_blob()`
- All three callers of `_try_direct_inline()` updated to propagate encryption fields

## Verification

```
uv run pytest tests/test_inline_blobs.py tests/test_core.py -x -q --override-ini="addopts="
75 passed in 13.91s
```

## Commits

- `538635c` — feat(INLINE-01,INLINE-02): encrypt inline blob write/read paths
