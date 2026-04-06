---
phase: 08-crash-safe-write-intent-logging
plan: 01
status: complete
---

# Summary: Crash-Safe Write Intent Journal

## What Was Built

File-based write intent journal that prevents orphaned blobs after crashes. Before a blob is written, an intent file is created in `{cache_dir}/.intents/`. After metadata commit succeeds, the intent file is removed. On cache init (when `cleanup_on_init=True`), stale intents older than a configurable threshold trigger deletion of orphaned blobs.

## Files Created/Modified

| File | Change |
|------|--------|
| `src/cacheness/write_intent.py` | NEW — `WriteIntentJournal` class with `record_intent()`, `clear_intent()`, `cleanup_stale_intents()` |
| `src/cacheness/config.py` | Added `stale_intent_threshold_seconds` (default 300s) to `CacheStorageConfig` |
| `src/cacheness/core.py` | Integrated journal into `__init__()` (cleanup on init) and `put()` (record/clear lifecycle) |
| `src/cacheness/_storage_mode_mixin.py` | Integrated journal into `_storage_mode_put()` (record/clear lifecycle) |
| `tests/test_write_intent.py` | NEW — 12 tests (6 unit + 6 integration) |

## Key Decisions

- **File-based intent storage** (not metadata table): Backend-agnostic, crash-safe (individual files vs shared JSON), no schema migration needed
- **xxhash-based filenames**: Consistent with existing blob naming, fast and collision-resistant
- **Lazy directory creation**: `.intents/` dir only created when first intent is recorded
- **Non-inline path only**: Inline mode stores data in metadata directly (no blob), so no intent needed

## Test Results

- 12 new tests pass (6 unit, 6 integration)
- Full suite: 1631 passed, 101 skipped, 0 failures

## Artifacts

- Intent files: `{cache_dir}/.intents/{xxhash}.intent` containing JSON `{cache_key, blob_path, created_at}`
- Config: `stale_intent_threshold_seconds` controls cleanup threshold
