---
phase: 17-key-rotation-api
plan: 01
status: complete
started: 2026-04-03T19:30:00Z
completed: 2026-04-03T19:35:00Z
---

## Summary

Implemented the `rotate_key()` API on both `UnifiedCache` and `BlobStore`, plus the `RotationResult` return type. Users can now rotate signing keys and re-sign all existing entries without data loss.

## Tasks Completed

### Task 1: RotationResult dataclass + UnifiedCache.rotate_key()
- Added `RotationResult` dataclass to `interfaces.py` with `total`, `re_signed`, `failed`, `skipped`, `failures` fields
- Added `TYPE_CHECKING` import guard for `RotationResult` in `core.py`
- Implemented `UnifiedCache.rotate_key(new_key_file)`:
  - Validates signer enabled, key file exists, key is 32 bytes
  - Holds `self._lock` for entire operation
  - Overwrites current key file, creates new signer via `create_cache_signer()`
  - Iterates all entries via `iter_entry_summaries()`, re-signs each with `_extract_signable_fields()` → `sign_entry()` → `put_entry()`
  - Re-signs namespace registry row (D-12)
  - Replaces `self.signer` and `self._blob_store.signer`
  - Returns `RotationResult` with accurate counts
- **Commit:** `c1678a5`

### Task 2: BlobStore.rotate_key()
- Added `TYPE_CHECKING` import guard for `RotationResult` in `blob_store.py`
- Implemented `BlobStore.rotate_key(new_key_file)`:
  - Same validation and lock pattern as UnifiedCache
  - Uses BlobStore's flatten-for-signing pattern (matching `put()`/`get()`)
  - Stores signature in both top-level and nested metadata
  - Does NOT re-sign namespace (that's UnifiedCache's responsibility)
- **Commit:** `c1678a5` (same commit as Task 1)

## Key Files

### Modified
- `src/cacheness/interfaces.py` — Added `RotationResult` dataclass
- `src/cacheness/core.py` — Added `rotate_key()` method, `TYPE_CHECKING` import
- `src/cacheness/storage/blob_store.py` — Added `rotate_key()` method, `TYPE_CHECKING` import

## Deviations from Plan

**[Ruff F821] String-quoted return type triggered undefined name error** — Added `TYPE_CHECKING` import guards for `RotationResult` in both `core.py` and `blob_store.py` to satisfy ruff's `F821` check.

**Total deviations:** 1 auto-fixed (lint). **Impact:** Minimal — import structure only.

## Test Results

- **Tier 1:** 112 passed (key rotation + core + blob store)
- **Full suite:** 1691 passed, 101 skipped, 0 failures

## Self-Check: PASSED
