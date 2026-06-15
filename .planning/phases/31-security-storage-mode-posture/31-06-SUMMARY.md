---
phase: 31-security-storage-mode-posture
plan: 06
subsystem: storage
tags: [storage-mode, durability, fsync, atomic-writes, write-intent]

requires:
  - phase: 31-security-storage-mode-posture
    provides: STRG-01 destructive API warning policy, STRG-03 pre-blob write-intent coverage, and SEC-03 local atomic publication patterns
provides:
  - CacheStorageConfig.fsync_on_write with compatibility default false
  - Opt-in local fsync for filesystem blob writes and BlobStore write_blob_from_path paths
  - Opt-in local fsync for JSON metadata saves and write-intent files
  - Storage-mode transaction documentation distinguishing atomic rename from power-loss durability
affects: [storage-mode, json-metadata, blob-store, write-intent, phase-31]

tech-stack:
  added: []
  patterns:
    - Shared local durability helper for file fsync plus best-effort parent directory fsync
    - Opt-in durability policy propagated from CacheStorageConfig to local file-writing components

key-files:
  created:
    - src/cacheness/_durability.py
    - .planning/phases/31-security-storage-mode-posture/31-06-SUMMARY.md
  modified:
    - src/cacheness/config.py
    - src/cacheness/core.py
    - src/cacheness/metadata/__init__.py
    - src/cacheness/metadata/json_backend.py
    - src/cacheness/storage/backends/blob_backends.py
    - src/cacheness/storage/blob_store.py
    - src/cacheness/write_intent.py
    - docs/TRANSACTION_GUARANTEES.md
    - tests/test_config_options.py
    - tests/test_config_validation.py
    - tests/test_atomic_writes.py
    - tests/test_write_intent.py
    - tests/test_blob_store.py
    - tests/test_metadata.py

key-decisions:
  - "STRG-02 keeps fsync_on_write defaulting to False so cache-mode and storage-mode write performance is unchanged unless users opt in."
  - "Local file fsync failures propagate when fsync_on_write is enabled; parent directory fsync is best-effort for Windows and filesystems that do not support it."
  - "The fsync policy is local-only; remote blob stores and database backends rely on their own durability contracts."

patterns-established:
  - "Use cacheness._durability.flush_and_fsync for local file descriptors whose fsync failures should be visible."
  - "Use cacheness._durability.fsync_parent_dir after local atomic rename, treating unsupported directory fsync as non-fatal."

requirements-completed: [STRG-02]

duration: 11min
completed: 2026-06-15
---

# Phase 31 Plan 06: Transaction Guarantees and Opt-In Fsync Summary

**Storage-mode durability is now explicit, with opt-in local fsync for JSON metadata, filesystem blobs, and write-intent files.**

## Performance

- **Duration:** 11 min
- **Started:** 2026-06-15T03:24:52Z
- **Completed:** 2026-06-15T03:35:50Z
- **Tasks:** 2
- **Files modified:** 15

## Accomplishments

- Added `CacheStorageConfig.fsync_on_write` and flat `CacheConfig(fsync_on_write=True)` support while preserving the default `False`.
- Added regressions proving local fsync hooks are invoked for JSON metadata saves, filesystem blob writes, BlobStore local writes, and write-intent files only when enabled.
- Added `src/cacheness/_durability.py` to centralize file flush/fsync and best-effort parent directory fsync behavior.
- Updated `docs/TRANSACTION_GUARANTEES.md` to distinguish atomic rename crash consistency from power-loss durability and describe the local-only fsync policy.

## Task Commits

1. **Task 1: Add fsync_on_write and durability-doc regressions** - `077fb2f` (test)
2. **Task 2: Wire opt-in local fsync and update transaction docs** - `5eaafb8` (feat)

**Plan metadata:** pending final docs commit or skipped by GSD commit helper.

## Files Created/Modified

- `src/cacheness/_durability.py` - Shared helper for flushing file descriptors and best-effort parent directory fsync.
- `src/cacheness/config.py` - Adds and validates `CacheStorageConfig.fsync_on_write`; supports flat `CacheConfig` construction.
- `src/cacheness/core.py` - Propagates storage fsync policy to JSON metadata backend creation and write-intent journal construction.
- `src/cacheness/metadata/__init__.py` - Threads `fsync_on_write` into JSON backend creation and auto fallback.
- `src/cacheness/metadata/json_backend.py` - Fsyncs JSON metadata and namespace registry temp files before atomic move when enabled.
- `src/cacheness/storage/backends/blob_backends.py` - Fsyncs filesystem blob temp files and streamed writes before rename, with best-effort parent directory fsync after rename.
- `src/cacheness/storage/blob_store.py` - Passes the fsync policy into local JSON metadata and filesystem blob backend construction.
- `src/cacheness/write_intent.py` - Fsyncs intent files and parent directory entries when enabled.
- `docs/TRANSACTION_GUARANTEES.md` - Documents storage-mode atomic rename versus power-loss durability caveats and the opt-in local fsync policy.
- `tests/test_config_options.py`, `tests/test_config_validation.py`, `tests/test_atomic_writes.py`, `tests/test_write_intent.py`, `tests/test_blob_store.py`, `tests/test_metadata.py` - Add STRG-02 regressions.

## Decisions Made

- Kept `fsync_on_write` under `CacheStorageConfig`, because the same storage durability policy needs to reach JSON metadata, local blobs, and write intents.
- Propagated the policy through constructors rather than global state, preserving explicit per-cache configuration.
- Kept directory fsync best-effort because Windows and some filesystems reject directory handles; file fsync failures still propagate when users opt in.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added shared durability helper and propagation files**
- **Found during:** Task 2 (Wire opt-in local fsync and update transaction docs)
- **Issue:** The plan's task file list did not include `src/cacheness/core.py`, `src/cacheness/storage/blob_store.py`, `src/cacheness/metadata/__init__.py`, or a shared helper file, but without these the configured policy would not reliably reach UnifiedCache JSON metadata, BlobStore JSON metadata, BlobStore local filesystem backends, or write-intent journals.
- **Fix:** Added `src/cacheness/_durability.py` and threaded `fsync_on_write` through the relevant constructors and factories.
- **Files modified:** `src/cacheness/_durability.py`, `src/cacheness/core.py`, `src/cacheness/storage/blob_store.py`, `src/cacheness/metadata/__init__.py`
- **Verification:** Controlled plan gate passed, 165 passed.
- **Committed in:** `5eaafb8`

---

**Total deviations:** 1 auto-fixed (Rule 2)
**Impact on plan:** Required for STRG-02 correctness. No package dependencies, public breaking changes, remote-backend semantics, or always-on fsync behavior were introduced.

## Issues Encountered

- The literal plan pytest command failed before collection because xdist tried to use the ACL-denied Windows temp root `C:\Users\akriz\AppData\Local\Temp\pytest-of-akriz`. Controlled equivalent runs cleared addopts, disabled pytest cache, and used repo-local `--basetemp`.
- `ty check` still reports pre-existing diagnostics in `config.py`, `core.py`, `json_backend.py`, `blob_store.py`, `tests/test_config_validation.py`, and `tests/test_metadata.py`. New negative-type tests were annotated with `# type: ignore`; no diagnostics were reported for `src/cacheness/_durability.py` or the new fsync logic.

## Verification

- RED config gate: `uv run --python 3.12 pytest -o addopts='' -p no:cacheprovider --basetemp .uv-cache\pytest-31-06-red-config tests/test_config_options.py tests/test_config_validation.py -x -q --ignore=tests/test_tensorflow_handler.py` - FAILED as expected on missing `CacheStorageConfig.fsync_on_write`.
- RED I/O gate: `uv run --python 3.12 pytest -o addopts='' -p no:cacheprovider --basetemp .uv-cache\pytest-31-06-red-io tests/test_atomic_writes.py tests/test_write_intent.py tests/test_blob_store.py tests/test_metadata.py -x -q --ignore=tests/test_tensorflow_handler.py` - FAILED as expected on missing `FilesystemBlobBackend(fsync_on_write=...)`.
- Task 2 config gate: `uv run --python 3.12 pytest -o addopts='' -p no:cacheprovider --basetemp .uv-cache\pytest-31-06-postformat-config tests/test_config_options.py tests/test_config_validation.py -x -q --ignore=tests/test_tensorflow_handler.py` - PASSED, 63 passed.
- Task 2 I/O gate: `uv run --python 3.12 pytest -o addopts='' -p no:cacheprovider --basetemp .uv-cache\pytest-31-06-postformat-io tests/test_atomic_writes.py tests/test_write_intent.py tests/test_blob_store.py tests/test_metadata.py -x -q --ignore=tests/test_tensorflow_handler.py` - PASSED, 102 passed.
- Plan gate: `uv run --python 3.12 pytest -o addopts='' -p no:cacheprovider --basetemp .uv-cache\pytest-31-06-plan-gate tests/test_config_options.py tests/test_config_validation.py tests/test_atomic_writes.py tests/test_write_intent.py tests/test_blob_store.py tests/test_metadata.py -x -q --ignore=tests/test_tensorflow_handler.py` - PASSED, 165 passed.
- Literal plan gate: `uv run --python 3.12 pytest tests/test_config_options.py tests/test_config_validation.py tests/test_atomic_writes.py tests/test_write_intent.py tests/test_blob_store.py tests/test_metadata.py -x -q --ignore=tests/test_tensorflow_handler.py` - FAILED before collection on Windows xdist temp ACL.
- `rg -n "fsync_on_write|power-loss|atomic rename|storage mode" docs/TRANSACTION_GUARANTEES.md` - PASSED, expected terms present.
- `uv run --python 3.12 ruff format ...` - PASSED, 12 files reformatted and 2 unchanged.
- `uv run --python 3.12 ruff check --fix ...` - PASSED.
- `uv run --python 3.12 ruff check ...` - PASSED.
- `uv run --python 3.12 ty check ...` - FAILED on pre-existing diagnostics documented above.

## Known Stubs

None. Stub scan only matched ordinary optional defaults, existing fallback log strings, and test helper empty collections.

## Threat Flags

None. This plan addresses the configured storage durability trust boundary from the threat model; it does not add endpoints, auth paths, schema changes, package dependencies, or remote backend durability changes.

## User Setup Required

None - no external service configuration required.

## TDD Gate Compliance

- RED commit exists: `077fb2f` (`test(31-06): add fsync durability regressions`)
- GREEN commit exists after RED: `5eaafb8` (`feat(31-06): wire opt-in local fsync writes`)

## Next Phase Readiness

STRG-02 is complete. Phase 31 now has all planned security and storage-mode posture requirements implemented, with the remaining dirty/untracked working tree items left untouched because they were pre-existing or unrelated.

## Self-Check: PASSED

- Verified `31-06-SUMMARY.md` exists.
- Verified key source and docs files exist.
- Verified task commits `077fb2f` and `5eaafb8` are present in git history.
- Verified task commits did not delete tracked files.

---
*Phase: 31-security-storage-mode-posture*
*Completed: 2026-06-15*
