---
phase: 29-ttl-eviction-consistency
plan: 03
subsystem: cache-metadata
tags: [ttl, expiry, metadata, sqlite, postgresql, signing]

requires:
  - phase: 29-ttl-eviction-consistency
    provides: TTL-01 stored expires_at semantics and TTL-02 public cleanup routing
provides:
  - SQLite and PostgreSQL same-key overwrite access-count preservation
  - JSON, SQLite, and PostgreSQL metadata-only created_at and TTL-field preservation
  - Signed-entry regression for metadata-only update preservation
affects: [phase-29, ttl-03, metadata-backends, signing, update-data]

tech-stack:
  added: []
  patterns:
    - Backend metadata-only updates preserve provenance and TTL fields by default
    - Content updates pass explicit timestamp changes instead of relying on backend side effects

key-files:
  created:
    - .planning/phases/29-ttl-eviction-consistency/29-03-SUMMARY.md
  modified:
    - .gitignore
    - src/cacheness/_update_mixin.py
    - src/cacheness/metadata/json_backend.py
    - src/cacheness/metadata/sqlite_backend.py
    - src/cacheness/storage/backends/postgresql_backend.py
    - tests/test_update_operations.py
    - tests/test_backend_parity.py
    - tests/test_cache_signing.py
    - tests/test_postgresql_backend.py

key-decisions:
  - "SQLite put_entry now uses ON CONFLICT DO UPDATE and omits access_count from existing-row updates."
  - "PostgreSQL _upsert_entry omits access_count from existing-row updates to match SQLite semantics."
  - "update_data now passes explicit created_at/accessed_at updates so content updates keep their prior timestamp-refresh semantics."

patterns-established:
  - "Metadata-only update_entry_metadata calls do not mutate created_at, ttl_seconds, or expires_at unless those keys are explicitly present in updates."
  - "Generated workspace-local pytest temp output belongs under .tmp/ and is ignored."

requirements-completed: [TTL-03]

duration: 17min
completed: 2026-06-13T22:23:11Z
---

# Phase 29 Plan 03: Metadata Preservation Summary

**Backend overwrites and metadata-only updates now preserve counters, provenance timestamps, TTL fields, and signed-entry readability.**

## Performance

- **Duration:** 17 min
- **Started:** 2026-06-13T22:06:04Z
- **Completed:** 2026-06-13T22:23:11Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments

- Added verify-first TTL-03 regressions for metadata-only provenance/TTL preservation, SQLite overwrite access-count preservation, PostgreSQL overwrite access-count preservation, and signed reads after backend metadata-only updates.
- Replaced SQLite `INSERT OR REPLACE` with `INSERT ... ON CONFLICT(cache_key) DO UPDATE` while omitting `access_count` from the update set.
- Updated PostgreSQL existing-row upsert to preserve the existing `access_count`.
- Removed implicit `created_at` resets from JSON, SQLite, and PostgreSQL `update_entry_metadata()`.
- Kept `update_data()` content-update semantics intact by passing explicit `created_at` and `accessed_at` updates.

## Task Commits

1. **Task 1 RED tests:** `3b712ce` - `test(29-03): add ttl metadata preservation regressions`
2. **Task 2 implementation:** `1473230` - `feat(29-03): preserve metadata provenance on backend updates`
3. **Rule 3 cleanup:** `1e9e681` - `chore(29-03): ignore local pytest temp output`

## Files Created/Modified

- `.gitignore` - Ignores `.tmp/` workspace-local pytest temp output.
- `src/cacheness/_update_mixin.py` - Sends explicit content-update timestamps to backend metadata updates.
- `src/cacheness/metadata/json_backend.py` - Preserves metadata-only `created_at`/TTL fields by default and honors explicit timestamp/TTL updates.
- `src/cacheness/metadata/sqlite_backend.py` - Preserves overwrite `access_count`; preserves metadata-only `created_at`/TTL fields by default.
- `src/cacheness/storage/backends/postgresql_backend.py` - Preserves overwrite `access_count`; preserves metadata-only `created_at`/TTL fields by default.
- `tests/test_update_operations.py` - Adds JSON/SQLite metadata-only provenance and TTL-field preservation coverage.
- `tests/test_backend_parity.py` - Adds SQLite same-key overwrite access-count preservation coverage.
- `tests/test_cache_signing.py` - Adds signed-entry metadata-only update regression.
- `tests/test_postgresql_backend.py` - Adds PostgreSQL same-key overwrite access-count preservation coverage with existing skip behavior.

## Verification

- RED: `uv run --python 3.12 pytest --override-ini testpaths= tests/test_metadata.py tests/test_update_operations.py tests/test_backend_parity.py tests/test_cache_signing.py tests/test_cache_integrity_verification.py tests/test_postgresql_backend.py -x -q --ignore=tests/test_tensorflow_handler.py --basetemp .tmp\pytest -p no:cacheprovider`
  - Result: failed as expected at `tests/test_update_operations.py::TestUpdateEntryMetadataBackends::test_metadata_only_update_preserves_provenance_and_ttl_fields[sqlite]` because `created_at` reset to the current timestamp.
- PASS: `uv run --python 3.12 ruff format src/cacheness/_update_mixin.py src/cacheness/metadata/json_backend.py src/cacheness/metadata/sqlite_backend.py src/cacheness/storage/backends/postgresql_backend.py tests/test_update_operations.py tests/test_backend_parity.py tests/test_cache_signing.py tests/test_postgresql_backend.py`
- PASS: `uv run --python 3.12 ruff check --fix src/cacheness/_update_mixin.py src/cacheness/metadata/json_backend.py src/cacheness/metadata/sqlite_backend.py src/cacheness/storage/backends/postgresql_backend.py tests/test_update_operations.py tests/test_backend_parity.py tests/test_cache_signing.py tests/test_postgresql_backend.py`
- PASS: `uv run --python 3.12 pytest --override-ini testpaths= tests/test_metadata.py tests/test_update_operations.py tests/test_backend_parity.py tests/test_cache_signing.py tests/test_cache_integrity_verification.py tests/test_postgresql_backend.py -x -q --ignore=tests/test_tensorflow_handler.py --basetemp .tmp\pytest -p no:cacheprovider`
  - Result: 170 passed, 34 skipped.
- PASS: `uv run --python 3.12 ruff check src/cacheness/_update_mixin.py src/cacheness/metadata/json_backend.py src/cacheness/metadata/sqlite_backend.py src/cacheness/storage/backends/postgresql_backend.py tests/test_update_operations.py tests/test_backend_parity.py tests/test_cache_signing.py tests/test_postgresql_backend.py`
- BASELINE FAIL: `uv run --python 3.12 ty check src/cacheness/_update_mixin.py src/cacheness/metadata/json_backend.py src/cacheness/metadata/sqlite_backend.py src/cacheness/storage/backends/postgresql_backend.py tests/test_update_operations.py tests/test_backend_parity.py tests/test_cache_signing.py tests/test_postgresql_backend.py`
  - Result: failed with existing touched-file diagnostics, including mixin attributes unknown to `ty`, broad JSON metadata typing, pytest `skip` typing, existing nullable test metadata access, and PostgreSQL custom metadata model base typing.
- PASS: `uv run --python 3.12 pytest --override-ini testpaths= tests/test_core.py tests/test_metadata.py tests/test_backend_parity.py tests/test_update_operations.py tests/test_storage_mode.py tests/test_blob_store.py tests/test_cache_signing.py tests/test_cache_integrity_verification.py -x -q --ignore=tests/test_tensorflow_handler.py --basetemp .tmp\pytest -p no:cacheprovider`
  - Result: 300 passed, 12 skipped.

## Decisions Made

- Preserved `update_data()` behavior per D-17 by making timestamp refresh explicit at the content-update boundary instead of keeping implicit backend resets.
- Kept same-key overwrite behavior focused on access-count preservation; overwrite content metadata still updates from the new entry.
- Used existing PostgreSQL availability skip behavior; no new service setup was introduced.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Kept update_data timestamp refresh explicit**
- **Found during:** Task 2
- **Issue:** Removing implicit backend `created_at` resets caused `update_data()` metadata tests to fail because content updates previously relied on that side effect.
- **Fix:** Added explicit `created_at` and `accessed_at` updates in `UpdateMixin.update_data()` and taught JSON, SQLite, and PostgreSQL backends to honor explicit timestamp updates while preserving metadata-only calls by default.
- **Files modified:** `src/cacheness/_update_mixin.py`, `src/cacheness/metadata/json_backend.py`, `src/cacheness/metadata/sqlite_backend.py`, `src/cacheness/storage/backends/postgresql_backend.py`
- **Verification:** Targeted pytest passed with 170 passed, 34 skipped; focused Phase 29 command passed with 300 passed, 12 skipped.
- **Committed in:** `1473230`

**2. [Rule 3 - Blocking] Ignored generated pytest temp output**
- **Found during:** Task 2 close-out
- **Issue:** The workspace-local `.tmp\pytest` directory created by pytest xdist had sandbox-owned ACLs and made `git status` warn while traversing untracked generated output. Removal attempts were denied.
- **Fix:** Added `.tmp/` to `.gitignore`, matching its use as local verification temp output.
- **Files modified:** `.gitignore`
- **Verification:** `git status --short --untracked-files=all` no longer warns about `.tmp\pytest`.
- **Committed in:** `1e9e681`

---

**Total deviations:** 2 auto-fixed (1 Rule 1 bug, 1 Rule 3 blocker)
**Impact on plan:** Both were required to preserve the plan's behavioral boundary and complete clean verification. No public API or out-of-phase TTL-04 behavior was added.

## Issues Encountered

- `ty check` remains red due existing touched-file baseline diagnostics; no new `ty` cleanup was attempted outside the plan scope.
- PostgreSQL database tests used existing skip behavior because `CACHENESS_TEST_POSTGRES_URL` was not configured in this environment.
- `.planning/config.json` had a pre-existing unrelated newline modification before execution and was preserved unstaged.

## Auth Gates

None.

## Known Stubs

None.

## Threat Flags

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

TTL-03 is implemented and covered. Plan 29-04 can proceed with remote URI blob deletion during size eviction.

## Self-Check: PASSED

- Found `.planning/phases/29-ttl-eviction-consistency/29-03-SUMMARY.md`.
- Found task commit `3b712ce`.
- Found task commit `1473230`.
- Found cleanup commit `1e9e681`.

---
*Phase: 29-ttl-eviction-consistency*
*Completed: 2026-06-13*
