---
created: 2026-06-12T15:45:54.188Z
title: Fix write-intent journal path resolution and safety checks
area: storage
files:
  - src/cacheness/write_intent.py:93
  - src/cacheness/core.py:884
  - src/cacheness/core.py:153
  - src/cacheness/_storage_mode_mixin.py:75
---

## Problem

Three bugs in write-intent crash recovery (code review findings **R2** + **R17**, 🔴 HIGH):

1. `record_intent()` stores a relative path (`default/abc.pkl`); `cleanup_stale_intents()` resolves it against the **process CWD**, not cache_dir — orphans are never cleaned, and a same-named CWD file could be deleted by mistake.
2. Cleanup deletes the blob without checking whether the metadata commit succeeded — crash between `put_entry()` and `clear_intent()` ⇒ a valid, referenced blob gets deleted on next init (data loss).
3. Storage mode forces `cleanup_on_init=False`, so stale-intent cleanup NEVER runs in storage mode — intents accumulate forever; a later cache-mode open of the same dir could (pre-fix) delete valid durable blobs.

## Solution

Execute **TASK-2** in `docs/CODE_REVIEW_ACTIONS.md` — resolve paths against cache_dir, add `entry_exists` callback guard, run stale-intent cleanup unconditionally in `__init__` (safe only with the guard). Acceptance tests incl. storage-mode cases are specified there. Tier-1: `uv run pytest tests/test_atomic_writes.py tests/test_core.py tests/test_storage_mode.py -x -q --ignore=tests/test_tensorflow_handler.py`
