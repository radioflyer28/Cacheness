---
created: 2026-06-12T15:45:54.188Z
title: Fix clear_all() not deleting blob files
area: storage
resolves_phase: 28
files:
  - src/cacheness/storage/blob_store.py:1321
  - src/cacheness/core.py:1155
  - src/cacheness/core.py:1201
---

## Problem

`_clear_blob_files()` globs only `cache_dir/*.{ext}` (root-level, non-recursive), but blobs are stored under `cache_dir/{namespace}/`. So `UnifiedCache.clear_all()` → `BlobStore.clear()` deletes **metadata only** — every blob file survives as an orphan. `clear_all_namespaces()` is affected too. Likely a regression from the move to per-namespace blob directories. Secondary: the hardcoded extension whitelist misses custom-handler extensions.

Code review finding **R1** (🔴 HIGH — silent disk leak / broken guarantee).

## Solution

Execute **TASK-1** in `docs/CODE_REVIEW_ACTIONS.md` — it contains the verify-first repro, exact change spec (enumerate via `blob_backend.list_blobs()` per namespace or rglob with reserved-file exclusions), the do-not-delete list, and acceptance tests. Tier-1: `uv run pytest tests/test_blob_store.py tests/test_core.py tests/test_storage_mode.py -x -q --ignore=tests/test_tensorflow_handler.py`
