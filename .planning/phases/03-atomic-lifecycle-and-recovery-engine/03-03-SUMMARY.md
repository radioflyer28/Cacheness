---
phase: 03-atomic-lifecycle-and-recovery-engine
plan: "03"
subsystem: storage-lifecycle
tags: [blobstore, lifecycle, operation-evidence, hmac, paging, recovery]
requires:
  - phase: 03-01
    provides: immutable generation publication and lifecycle tracer seams
  - phase: 03-02
    provides: exact-record manifest authority transitions
provides:
  - Bounded, domain-separated, topology-bound operation evidence records
  - Caller-owned lifecycle limits propagated unchanged to BlobStore runtime consumers
  - Stable bounded operation pages with exact conditional evidence checkpoints and retirement
affects: [03-04-delete-lifecycle, 03-05-clear-core, 03-07-reconciliation, 03-09-close-admission, BlobStore]
actuals:
  tokens: 19337
  tasks: 3
  commits: 6
tech-stack:
  added: []
  patterns:
    - Opaque operation evidence remains unauthoritative until lifecycle authentication and topology binding
    - Config-owned frozen limits are passed by identity to lifecycle repositories and recovery
    - Operation pages retain at most page-size plus one identifiers while returning exact raw bytes
key-files:
  created:
    - tests/test_blob_store_reconciliation.py
  modified:
    - src/cacheness/config.py
    - src/cacheness/storage/operation_record.py
    - src/cacheness/storage/operation_repository.py
    - src/cacheness/storage/lifecycle.py
    - src/cacheness/storage/blob_store.py
key-decisions:
  - "Lifecycle operation evidence is a strict control record with a dedicated HMAC domain, never a payload wrapper or ordinary-read authority."
  - "LifecycleLimits is declared once in cacheness.config and is preserved by identity from CacheConfig through BlobStore, LifecycleEngine, and the operation repository."
  - "Stale operation checkpoints and retirement fail on an exact-byte mismatch; elapsed age only gates already-authenticated pre-authority recovery after revalidation."
patterns-established:
  - "Recovery repository pages use opaque lexical cursors and bounded ID selection rather than materializing a full operation inventory."
  - "Lifecycle evidence cleanup uses exact-byte conditional retirement after the associated destructive cleanup succeeds."
requirements-completed: [STOR-04, STOR-06]
coverage:
  - id: D1
    description: Bounded operation records reject malformed, forged, cross-domain, or provenance-invalid control bytes before recovery can mutate managed storage.
    requirement: STOR-04
    verification:
      - kind: integration
        ref: tests/test_blob_store_atomic_lifecycle.py tests/test_blob_store_reconciliation.py
        status: pass
    human_judgment: false
  - id: D2
    description: Caller-configured lifecycle limits drive operation paging and orphan-grace eligibility while operation evidence remains retained until exact terminal cleanup.
    requirement: STOR-06
    verification:
      - kind: integration
        ref: tests/test_blob_store_reconciliation.py#test_operation_repository_uses_configured_bounded_stable_pages
        status: pass
    human_judgment: false
duration: 18 min
completed: 2026-08-30
status: complete
---

# Phase 03 Plan 03: Lifecycle Evidence and Limits Summary

**BlobStore now carries one caller-owned lifecycle policy into bounded, authenticated operation recovery, so untrusted evidence cannot cause cleanup and stale evidence cannot overwrite or retire newer recovery state.**

## Performance

- **Duration:** 18 min
- **Started:** 2026-08-30T18:45:16Z
- **Completed:** 2026-08-30T19:03:42Z
- **Tasks:** 3/3
- **Files modified:** 8

## Accomplishments

- Completed a versioned operation-evidence codec with strict byte, field, depth, node, identifier, topology, timestamp, and monotonic-checkpoint validation under a domain-separated HMAC.
- Added frozen `LifecycleLimits` to the public configuration model and preserved the exact supplied `CacheConfig` and limits objects through BlobStore, LifecycleEngine, and its evidence repository.
- Replaced materialized operation iteration with stable bounded pages, exact conditional checkpoints and retirement, action-bound recovery, and authenticated pre-authority orphan grace.

## Task Commits

1. **Task 1: Complete the authenticated bounded operation-record contract**
   - `e67d47c` `test(03-03): add failing operation evidence contract`
   - `ac75035` `feat(03-03): harden lifecycle operation evidence`
2. **Task 2: Own lifecycle policy in the configuration model**
   - `f0e74d0` `test(03-03): add failing lifecycle limits coverage`
   - `a5d9333` `feat(03-03): own lifecycle policy in cache config`
3. **Task 3: Apply explicit limits to operation retention and stable paging**
   - `cc4ccfa` `test(03-03): add failing bounded operation paging coverage`
   - `9c3da1e` `feat(03-03): bound lifecycle operation recovery`

## Files Created/Modified

- `src/cacheness/config.py` — declares validated immutable `LifecycleLimits`, composes it into `CacheConfig`, and preserves it through config serialization.
- `src/cacheness/__init__.py` — re-exports `LifecycleLimits` from the established public configuration surface.
- `src/cacheness/storage/operation_record.py` — validates complete bounded topology-bound lifecycle evidence and checkpoint transitions.
- `src/cacheness/storage/operation_repository.py` — provides opaque stable operation pages plus exact-byte conditional checkpoint and retirement operations.
- `src/cacheness/storage/lifecycle.py` — authenticates evidence before grace-gated recovery and consumes configured page/action limits by identity.
- `src/cacheness/storage/blob_store.py` — accepts a compatible keyword-only `CacheConfig` path without rebuilding its lifecycle policy.
- `tests/test_config_validation.py` — covers public defaults, invalid values, import identity, and constructor identity.
- `tests/test_blob_store_reconciliation.py` — covers hostile evidence preservation, bounded paging, exact transitions, and frozen-clock orphan grace.

## Decisions Made

- Operation evidence binds only lifecycle control and provenance; it does not introduce a custom payload header or wrapper around native formats.
- Evidence records become cleanup-eligible only after authentication, topology and locator revalidation, and configured grace; age alone has no authority.
- A small fixed set of per-operation evidence lock stripes protects in-process exact compare-and-transition work without serializing payload or manifest operations globally.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The repository-wide unused-import/local Ruff baseline still affects the plan's full lint command in pre-existing areas of `__init__.py`, `config.py`, and configuration tests. Focused lint of the changed paths passes with only those established F401/F841 findings excluded; no new lint finding was introduced.
- The separately tracked legacy clear-recovery operational-failure translation remains assigned to Plan 03-06 and was not broadened into this plan.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 03-04 can build delete lifecycle work on bounded authenticated evidence and exact conditional retirement.
- Plans 03-05, 03-07, and 03-09 must continue passing this same `CacheConfig.lifecycle_limits` object into manifest paging, reconciliation, and close admission rather than constructing another policy value.

## Self-Check: PASSED

- Verified the summary and all listed code/test artifacts exist on disk.
- Verified task commits `e67d47c`, `ac75035`, `f0e74d0`, `a5d9333`, `cc4ccfa`, and `9c3da1e` exist in git history.
