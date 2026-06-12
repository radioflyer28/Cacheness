---
created: 2026-06-12T15:45:54.188Z
title: JSON backend - propagate save failures, preserve corrupt files
area: database
resolves_phase: 28
files:
  - src/cacheness/metadata/json_backend.py
---

## Problem

Code review findings **R3** + **R4** (🔴 HIGH — silent data loss; worst in storage mode):

1. `JsonBackend._save_to_disk()` catches ALL exceptions, logs, and returns — `put_entry()` reports success on disk-full/permission errors while nothing was persisted. In-memory state diverges from disk; everything since the failure is lost on process exit with no error surfaced.
2. `_load_from_disk()` on a corrupt JSON file logs a warning and "starts fresh" — all metadata silently discarded, no backup of the corrupt file, every blob orphaned.

## Solution

Execute **TASK-3** in `docs/CODE_REVIEW_ACTIONS.md` — add `raise_on_error=True` for `put_entry`/`remove_entry` (keep best-effort for stats-only writes), mkdir-retry for missing parent dir, rename corrupt file to `*.corrupt-<ts>` and log at ERROR. Existing tests asserting swallow behavior must be updated to the new contract. Tier-1: `uv run pytest tests/test_metadata.py tests/test_json_schema_versioning.py -x -q --ignore=tests/test_tensorflow_handler.py`
