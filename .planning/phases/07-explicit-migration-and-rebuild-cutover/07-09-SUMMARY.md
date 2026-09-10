---
phase: 07-explicit-migration-and-rebuild-cutover
plan: "09"
subsystem: storage-migration
tags: [blobstore, offline-rebuild, handler-registry, evidence, integrity]
requires:
  - phase: 07-explicit-migration-and-rebuild-cutover
    provides: authenticated offline inspection, evidence, and cutover boundaries
provides:
  - Include-all rebuild plans with exact, reconfirmed exclusions
  - Store-local handler transformation contracts
  - Explicit handler-backed rebuild through BlobStore lifecycle ownership
affects: [offline-migration, blobstore, projections, handler-registration]
actuals:
  tokens: 21320
  tasks: 3
  commits: 7
tech-stack:
  added: []
  patterns:
    - Exact rebuild scope and confirmation are evidence-bound immutable inputs.
    - Registered handlers own payload compatibility and transformation eligibility.
    - Rebuild writes use BlobStore's existing lifecycle authority.
key-files:
  created:
    - tests/test_rebuild_workflow.py
  modified:
    - src/cacheness/storage/migration.py
    - src/cacheness/storage/migration_evidence.py
    - src/cacheness/interfaces.py
    - src/cacheness/handlers.py
    - tests/test_handler_registration.py
key-decisions:
  - "Rebuild defaults to the complete inspected inventory; every omission resolves to exact keys in a regenerated, separately confirmed plan."
  - "Payload conversion remains handler-owned through store-local exact directed edges, with no coordinator native-format switch."
  - "Destination entries are written through BlobStore and only become rebuild-accepted after exact verification; projections remain derived post-acceptance work."
patterns-established:
  - "Offline rebuild: authenticate and verify private source payload bytes before handler deserialization."
  - "Failure cleanup: only lifecycle receipts produced by the active rebuild are eligible for deletion."
requirements-completed: [MIGR-03, MIGR-05, MIGR-06]
coverage:
  - id: D1
    description: Include-all rebuild plans reject unconfirmed or open-ended exclusions.
    requirement: MIGR-03
    verification:
      - kind: unit
        ref: tests/test_rebuild_workflow.py
        status: pass
    human_judgment: false
  - id: D2
    description: Registered handlers resolve only exact declared payload transformations.
    requirement: MIGR-05
    verification:
      - kind: unit
        ref: tests/test_handler_registration.py
        status: pass
    human_judgment: false
  - id: D3
    description: Authenticated source entries rebuild through destination BlobStore lifecycle with explicit acceptance.
    requirement: MIGR-06
    verification:
      - kind: integration
        ref: tests/test_rebuild_workflow.py
        status: pass
      - kind: integration
        ref: tests/test_blob_store_read_contract.py
        status: pass
    human_judgment: false
duration: 19min
completed: 2026-09-10
status: complete
---

# Phase 07 Plan 09: Explicit Handler-Backed Rebuild Summary

**Explicit include-all rebuilds now verify source payloads before registered handler reads, preserve authenticated catalog values, and write accepted output through destination BlobStore lifecycle ownership.**

## Performance

- **Duration:** 19 min
- **Started:** 2026-09-10T03:52:04Z
- **Completed:** 2026-09-10T04:11:22Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments

- Added separately confirmed rebuild plans that include every inspected entry by default and bind any exact exclusion to regenerated scope, totals, and digest.
- Added handler-owned exact transformation edges and a store-local resolver that rejects wildcard, reverse, ambiguous, or unsupported conversion requests.
- Rebuilt verified source entries through destination `BlobStore.put_entry`, preserved authenticated catalog values, isolated failed run-owned output, and required destination verification before acceptance or projection rebuilds.

## Task Commits

Each TDD task was committed atomically:

1. **Task 1: Add explicit include-all rebuild planning and confirmation** - `95df1ab` (test), `bd392f8` (feat)
2. **Task 2: Add registered handler payload transformation contract** - `4308617` (test), `7cc246c` (feat)
3. **Task 3: Rebuild through registered source handlers and destination BlobStore** - `4b46038` (test), `dd15b17` (feat)

**Plan metadata:** final documentation commit records this summary, roadmap, and state update.

## Files Created/Modified

- `src/cacheness/storage/migration.py` - Rebuild scope, confirmation, evidence gating, handler-backed staging, verification, acceptance, and post-acceptance projection control.
- `src/cacheness/storage/migration_evidence.py` - Explicit rebuild state transitions.
- `src/cacheness/interfaces.py` - Exact directed handler payload transformation edge contract.
- `src/cacheness/handlers.py` - Store-local transformation resolver and handler registration validation.
- `tests/test_rebuild_workflow.py` - Include-all, confirmation, custom-handler, integrity, catalog, and cleanup coverage.
- `tests/test_handler_registration.py` - Exact directed transformation registration and rejection coverage.

## Decisions Made

- Rebuild is a separate lifecycle from same-backend migration: it never invokes physical staging or deletes source data.
- Exact handler compatibility stays on ordinary verified reads and writes; a format change must have one declared source-handler edge.
- Maintenance evidence corroborates offline acceptance but does not become another storage authority; `BlobStore` remains the destination lifecycle owner.

## Verification

- `uv run --frozen pytest -q tests/test_rebuild_workflow.py tests/test_handler_registration.py tests/test_blob_store_read_contract.py -x -o log_cli=false` — 16 passed.
- `uv run --frozen pytest -q tests/test_migration_plan_contract.py tests/test_migration_inspection.py tests/test_migration_run_evidence.py tests/test_migration_cutover.py -x -o log_cli=false` — 33 passed.
- `uv run --frozen ruff check src/cacheness/interfaces.py src/cacheness/handlers.py src/cacheness/storage/migration.py src/cacheness/storage/migration_evidence.py tests/test_handler_registration.py tests/test_rebuild_workflow.py --output-format concise` — passed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Preserved stopped-worker acknowledgement gating for refused rebuild actions.**
- **Found during:** Task 1
- **Issue:** The new rebuild plan action sequence could have weakened the existing acknowledgement requirement for a rebuild-only inspection result.
- **Fix:** Kept acknowledgement as mandatory for every rebuild action shape while retaining actionable plans' separate confirmation requirement.
- **Files modified:** `src/cacheness/storage/migration.py`
- **Verification:** Focused migration plan and rebuild workflow tests passed.
- **Committed in:** `bd392f8`

---

**Total deviations:** 1 auto-fixed (Rule 2).
**Impact on plan:** The adjustment preserves the plan's required offline safety gate without expanding the architecture.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

The storage migration surface now offers evidence-bound, include-all rebuild for cross-backend and incompatible payload sources. Future work can rely on explicit handler contracts and accepted destination authority without introducing a universal converter or another lifecycle authority.

## Self-Check: PASSED

- Summary file exists and all six TDD task commits are present in git history.
