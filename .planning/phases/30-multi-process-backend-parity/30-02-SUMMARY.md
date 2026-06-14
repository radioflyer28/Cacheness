---
phase: 30-multi-process-backend-parity
plan: 02
subsystem: metadata-backends
tags: [blobstore, sqlite, postgresql, metadata, parity]

requires:
  - phase: 29-ttl-eviction-consistency
    provides: SQLite and PostgreSQL overwrite/access-count preservation
provides:
  - BlobStore user metadata parity regression across JSON, SQLite, and PostgreSQL
  - SQLite leftover user metadata preservation through metadata_dict
  - PostgreSQL leftover user metadata preservation through JSONB metadata_dict
  - Public BlobStore metadata filtering over flat, nested metadata, and metadata_dict fields
affects: [phase-30, par-02, metadata-backends, blobstore-filtering]

tech-stack:
  added: []
  patterns:
    - Leftover BlobStore metadata is merged into metadata_dict after backend technical fields are popped
    - Existing metadata_dict keys win over leftover user metadata on conflicts

key-files:
  created:
    - .planning/phases/30-multi-process-backend-parity/30-02-SUMMARY.md
  modified:
    - tests/test_backend_parity.py
    - src/cacheness/metadata/sqlite_backend.py
    - src/cacheness/storage/backends/postgresql_backend.py
    - src/cacheness/storage/blob_store.py

key-decisions:
  - "SQLite and PostgreSQL store leftover BlobStore user metadata in metadata_dict while preserving existing metadata_dict keys on conflict."
  - "BlobStore.list(metadata_filter=...) now checks nested metadata and metadata_dict fields so JSON and SQL backends share the public filtering contract."

patterns-established:
  - "Read/list paths expose metadata_dict user keys without overwriting technical metadata fields."
  - "PostgreSQL parity tests remain behind the existing availability skip gate."

requirements-completed: [PAR-02]

duration: 19min
completed: 2026-06-14
---

# Phase 30 Plan 02: User Metadata Backend Parity Summary

**SQLite and PostgreSQL now preserve BlobStore user metadata in metadata_dict and expose it through public read/filter paths with JSON parity.**

## Performance

- **Duration:** 19 min
- **Started:** 2026-06-14T14:25:28Z
- **Completed:** 2026-06-14T14:44:27Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Added a public BlobStore parity regression for `metadata={"experiment": "x42"}` across JSON, SQLite, and PostgreSQL.
- Implemented SQLite and PostgreSQL leftover metadata merging into `metadata_dict`, preserving existing `metadata_dict` values on conflicts.
- Exposed deserialized user metadata on backend read, summary, and list paths without overwriting technical fields.
- Fixed the public `BlobStore.list(metadata_filter=...)` path to check nested metadata and metadata_dict values, which was required for JSON parity.

## Task Commits

1. **Task 1: Add public BlobStore user-metadata parity regression** - `2bf78d0` (test)
2. **Task 2: Preserve user metadata in SQLite and PostgreSQL backends** - `3e85c18` (feat)

**Plan metadata:** pending docs commit

## Files Created/Modified

- `tests/test_backend_parity.py` - Adds JSON/SQLite/PostgreSQL BlobStore custom metadata round-trip and filter coverage.
- `src/cacheness/metadata/sqlite_backend.py` - Merges leftover user metadata into serialized `metadata_dict` and flattens read/list exposure.
- `src/cacheness/storage/backends/postgresql_backend.py` - Applies equivalent JSONB metadata_dict merge/read/summary behavior.
- `src/cacheness/storage/blob_store.py` - Checks nested metadata and metadata_dict during public metadata filtering.

## Verification

- RED: `uv run --python 3.12 pytest --override-ini testpaths= tests/test_backend_parity.py -x -q --ignore=tests/test_tensorflow_handler.py --basetemp .tmp\pytest -p no:cacheprovider`
  - Result: failed before implementation. JSON preserved nested metadata but did not filter through `BlobStore.list`; direct SQLite probe failed with `KeyError: 'experiment'`.
- PASS: `uv run --python 3.12 pytest -o addopts="" --override-ini testpaths= "tests/test_backend_parity.py::TestBlobStoreUserMetadataBackendParity::test_user_metadata_round_trips_and_filters" -q --ignore=tests/test_tensorflow_handler.py --basetemp .tmp\pytest -p no:cacheprovider`
  - Result: 2 passed, 1 skipped.
- PASS: `uv run --python 3.12 ruff format src/cacheness/metadata/sqlite_backend.py src/cacheness/storage/backends/postgresql_backend.py src/cacheness/storage/blob_store.py tests/test_backend_parity.py`
- PASS: `uv run --python 3.12 ruff check --fix src/cacheness/metadata/sqlite_backend.py src/cacheness/storage/backends/postgresql_backend.py src/cacheness/storage/blob_store.py tests/test_backend_parity.py`
- PASS: `uv run --python 3.12 ruff check src/cacheness/metadata/sqlite_backend.py src/cacheness/storage/backends/postgresql_backend.py src/cacheness/storage/blob_store.py tests/test_backend_parity.py`
- BASELINE FAIL: `uv run --python 3.12 ty check src/cacheness/metadata/sqlite_backend.py src/cacheness/storage/backends/postgresql_backend.py src/cacheness/storage/blob_store.py tests/test_backend_parity.py`
  - Result: failed with existing typed-SQLAlchemy, BlobStore, and pytest typing diagnostics in touched files; no type-only cleanup was attempted outside plan scope.
- PASS: `uv run --python 3.12 pytest --override-ini testpaths= tests/test_backend_parity.py -x -q --ignore=tests/test_tensorflow_handler.py --basetemp .tmp\pytest -p no:cacheprovider`
  - Result: 39 passed, 13 skipped.
- PASS: `uv run --python 3.12 pytest --override-ini testpaths= tests/test_backend_parity.py tests/test_blob_store.py tests/test_metadata.py tests/test_postgresql_backend.py -x -q --ignore=tests/test_tensorflow_handler.py --basetemp .tmp\pytest -p no:cacheprovider`
  - Result: 114 passed, 33 skipped.
- PASS: `uv run --python 3.12 pytest --override-ini testpaths= tests/test_blob_store.py tests/test_metadata.py tests/test_backend_parity.py -x -q --ignore=tests/test_tensorflow_handler.py --basetemp .tmp\pytest -p no:cacheprovider`
  - Result: 102 passed, 13 skipped.
- PASS: `uv run --python 3.12 pytest --override-ini testpaths= tests/test_blob_namespace.py tests/test_blob_store.py tests/test_backend_parity.py tests/test_fault_injection.py tests/test_storage_mode.py tests/test_cache_integrity_verification.py -x -q --ignore=tests/test_tensorflow_handler.py --basetemp .tmp\pytest -p no:cacheprovider`
  - Result: 182 passed, 13 skipped.
- FULL SUITE FAIL: `uv run --python 3.12 pytest --override-ini testpaths= tests/ -x -q --ignore=tests/test_tensorflow_handler.py --basetemp .tmp\pytest-full-30-02 -p no:cacheprovider`
  - First sandboxed run failed in `tests/test_compress_pickle.py` because `blosc2` could not read `C:\Users\akriz\AppData\Local\blosc\python-blosc2\Cache\cpuinfo.json`.
  - Escalated rerun reached 993 passed, 67 skipped, then failed unrelated tests `tests/test_decorators.py::TestCacheIfDecorator::test_cache_if_supports_ttl_parameter` and `tests/test_dunder_methods.py::TestDunderMethods::test_contains_expired_key`.

## Decisions Made

- Preserved backend technical metadata precedence by using `dict.setdefault()` when exposing user metadata on read paths.
- Kept PostgreSQL service coverage explicit and skip-gated through `_skip_if_pg_unavailable`.
- Added a small public filter-path fix in `BlobStore.list()` instead of changing or bypassing the public API in tests.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Public BlobStore metadata filtering ignored nested JSON metadata**
- **Found during:** Task 1 RED verification
- **Issue:** The plan expected JSON to demonstrate `BlobStore.list(metadata_filter=...)`, but JSON preserved user metadata only inside nested `metadata`, while `BlobStore.list()` checked only flat entry fields.
- **Fix:** Taught `BlobStore.list()` to check flat fields, nested `metadata`, and dict-valued `metadata_dict` values in order.
- **Files modified:** `src/cacheness/storage/blob_store.py`
- **Verification:** Focused parity test passed with 2 passed, 1 PostgreSQL skip; targeted and phase quick pytest commands passed.
- **Committed in:** `3e85c18`

---

**Total deviations:** 1 auto-fixed (1 Rule 1 bug)
**Impact on plan:** Required to satisfy the plan's public metadata-filtering contract. No new public API was added.

## Issues Encountered

- `uv sync --all-groups` initially failed against the global uv cache; rerunning with workspace-local `UV_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR` required escalation and restored the locked environment.
- `ty check` remains red on existing baseline diagnostics in touched files.
- Full-suite verification remains red outside this plan scope, as documented in Verification.
- PostgreSQL tests used existing skip behavior because `CACHENESS_TEST_POSTGRES_URL` was not configured.
- `.planning/config.json` had a pre-existing unrelated modification and was preserved unstaged.
- Concurrent Wave 1 work landed `30-01` commits while this plan executed; those files were not staged or committed by this plan.

## Auth Gates

None.

## Known Stubs

None. Stub scan hits were existing literal/default text only: PostgreSQL column defaults and PostgreSQL skip text.

## Threat Flags

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

PAR-02 is complete for JSON and SQLite, with PostgreSQL coverage explicit and skip-gated. Phase 30 Wave 2 can consume the public metadata filtering behavior without adding a new list API.

## Self-Check: PASSED

- Found `.planning/phases/30-multi-process-backend-parity/30-02-SUMMARY.md`.
- Found task commit `2bf78d0`.
- Found task commit `3e85c18`.
- Confirmed no committed file deletions in task commits.

---
*Phase: 30-multi-process-backend-parity*
*Completed: 2026-06-14*
