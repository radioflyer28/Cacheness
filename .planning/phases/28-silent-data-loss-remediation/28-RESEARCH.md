# Phase 28: Silent Data-Loss Remediation - Research

**Researched:** 2026-06-12
**Mode:** inline Codex execution of the GSD researcher role
**Status:** Complete

## Source Contract

Phase 28 is bounded by `.planning/phases/28-silent-data-loss-remediation/28-CONTEXT.md`.
The binding source documents are `docs/CODE_REVIEW_ACTIONS.md` and
`docs/CODE_REVIEW_FINDINGS.md`; the action document is the concrete
implementation contract and the findings document supplies rationale and
guardrails.

The phase covers only Wave 1 silent data-loss and broken-guarantee work:

- TASK-1 / R1: recursive namespace blob cleanup for `clear_all()` and
  `clear_all_namespaces()`.
- TASK-2 / R2 and R17: write-intent path resolution, committed-entry guard,
  and storage-mode stale-intent cleanup.
- SEED-006 / R8: record write intent before blob write as a companion follow-up
  to TASK-2.
- TASK-3 / R3 and R4: JSON backend persistence failures and corrupt-file
  preservation.
- TASK-4 / U1 plus KEY-02 todo: stable persistent cache-key fallback behavior
  and property-based regression coverage.

The tiered pull-through cache todo is explicitly out of scope for Phase 28.

## Code Findings

### TASK-1: `clear_all()` Blob Cleanup

Current `src/cacheness/storage/blob_store.py::_clear_blob_files()` builds
root-level glob patterns under `self.cache_dir` and deletes only known
extensions. Blobs are stored below namespace directories such as
`cache_dir/default/` and may be sharded further by the blob backend.

`src/cacheness/core.py::clear_all()` delegates to `BlobStore.clear()`.
`clear_all_namespaces()` calls `_clear_blob_files()` directly for the default
namespace before dropping metadata and removing non-default namespace dirs.

Plan implication: implement backend-aware enumeration or careful recursive
filesystem deletion while preserving reserved files. Because
`FilesystemBlobBackend.list_blobs()` currently returns all known blob extensions
for the active namespace, the executor should verify whether backend enumeration
is enough for all namespaces before choosing it. A cautious `rglob` with the
TASK-1 exclusion list may be the lower-risk local fix.

### TASK-2: Write-Intent Recovery

Current `WriteIntentJournal` only stores `cache_dir / ".intents"`; it does not
retain `cache_dir` for resolving relative paths. `cleanup_stale_intents()` uses
`Path(blob_path)` directly, so relative paths resolve against process CWD.

`UnifiedCache.__init__()` currently runs `_cleanup_stale_intents()` only inside
the `cleanup_on_init` block. Storage mode flips `cleanup_on_init=False` in
`config.py`, so stale intents accumulate there.

`UnifiedCache._cleanup_stale_intents()` passes no metadata-existence callback,
so the journal cannot distinguish an orphan from a committed entry whose intent
file survived a crash. This is the R2 data-loss window.

Plan implication: add cache-dir-relative resolution in the journal, add an
optional `entry_exists(cache_key)` callback, run stale-intent cleanup
unconditionally from `UnifiedCache.__init__()`, and keep expired-entry cleanup
inside the existing `cleanup_on_init` guard.

### SEED-006: Intent Before Blob Write

Despite the module docstring saying intents are recorded before blob writes,
the current non-inline write paths call `_write_blob()` first, then
`record_intent(cache_key, result.actual_path)` in both `UnifiedCache.put()` and
`_storage_mode_put()`.

Plan implication: after TASK-2 lands, record a planned path intent before the
blob write in both cache and storage mode. The executor must account for intents
that point at paths that were never created because the process crashed before
or during handler I/O. Cleanup should remove the intent and tolerate a missing
blob.

### TASK-3: JSON Backend Persistence

`JsonBackend._save_to_disk()` catches all exceptions and logs an error without
re-raising. `put_entry()` and `remove_entry()` therefore report success even
when data-critical metadata persistence failed. `update_access_time()`,
`increment_hits()`, and `increment_misses()` also call `_save_to_disk()` but are
stats-only paths and may remain best-effort.

`_load_from_disk()` logs a warning and starts fresh on corrupted JSON without
preserving the bad file.

Plan implication: add `_save_to_disk(raise_on_error: bool = False)`, call it
with `True` from `put_entry()` and `remove_entry()`, retry parent-directory
creation once when missing, and preserve corrupt metadata as
`*.corrupt-<timestamp>` with error-level logging.

### TASK-4 and KEY-02: Cache-Key Stability

`serialization.py::_serialize_with_config()` falls through large tuples to the
hashable fallback. The fallback uses Python `hash(obj)`, which is randomized
across `PYTHONHASHSEED` for string-containing objects and address-based for
default object hashes. The string fallback uses raw `str(obj)`, which can embed
memory addresses for default repr objects.

Existing tests already include `tests/test_serialization.py`,
`tests/test_cache_key_consistency.py`, and Hypothesis coverage in
`tests/test_property_based.py`. The current tests still expect
`"hashed:tuple:"` for large tuples, so TASK-4 must update tests to the new
deterministic contract.

Plan implication: make large tuple serialization deterministic with recursive
element serialization plus `xxhash.xxh3_64`; restrict `hash()` fallback to
stable scalar types and Enum names; replace default memory-address repr with a
stable low-quality marker and warning. Add a cross-subprocess
`PYTHONHASHSEED` regression and extend property tests for determinism,
collision resistance, control-parameter stripping through `UnifiedCache`, and
argument-order normalization.

## Test Strategy

Use the task-specific Tier-1 commands from `docs/CODE_REVIEW_ACTIONS.md`.
Always include `--ignore=tests/test_tensorflow_handler.py` on Windows. Follow
`.planning/codebase/TESTING.md` for test placement:

- Blob cleanup: `tests/test_blob_store.py`, `tests/test_core.py`,
  `tests/test_storage_mode.py`.
- Write intent: prefer `tests/test_write_intent.py` for journal behavior and
  include `tests/test_atomic_writes.py`, `tests/test_core.py`,
  `tests/test_storage_mode.py` for acceptance coverage.
- JSON backend: `tests/test_metadata.py` and
  `tests/test_json_schema_versioning.py`.
- Cache keys: `tests/test_serialization.py`, `tests/test_cache_key_consistency.py`,
  `tests/test_decorators.py`, `tests/test_core.py`, and
  `tests/test_property_based.py`.

Every plan with a verify-first step must run that repro before implementation
and stop if the observed behavior does not match the review document.

## Recommended Plan Shape

Create five ordered Wave 1 plan files:

1. `28-01-PLAN.md` - TASK-1 / REL-01.
2. `28-02-PLAN.md` - TASK-2 / REL-02 through REL-04.
3. `28-03-PLAN.md` - SEED-006 / STRG-03-adjacent follow-up scoped to Phase 28
   because it is tied to TASK-2; it should not expand TASK-2 acceptance.
4. `28-04-PLAN.md` - TASK-3 / REL-05 through REL-06.
5. `28-05-PLAN.md` - TASK-4 plus KEY-02 / KEY-01 through KEY-02.

Each plan should become one beads issue and one atomic code commit during
execute-phase. TASK-4 must include a `CHANGELOG.md` compatibility note and a
commit message that mentions affected cache keys change.

## RESEARCH COMPLETE

