---
phase: 02-canonical-storage-and-integrity-contract
plan: "06"
subsystem: storage
tags: [blobstore, canonical-manifest, authenticated-mutation, integrity]
requires:
  - phase: 02-05
    provides: authenticated canonical BlobStore get pipeline and strict signing
provides:
  - Authenticated committed-manifest semantics for every direct read-only BlobStore API
  - Re-signed user-metadata patches that reject canonical structural changes
  - Manifest-first delete and whole-set clear preflight before destructive mutation
affects: [phase-03-lifecycle, phase-06-unified-cache, BlobStore]
actuals:
  tokens: 8239
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - One authenticated manifest loader shared by every direct BlobStore public operation
    - Containment-only validation of legacy backend projections after canonical authentication
key-files:
  created: []
  modified:
    - src/cacheness/storage/blob_store.py
    - tests/test_blob_store_read_contract.py
key-decisions:
  - "Direct BlobStore reads and mutations derive authoritative state only from authenticated committed manifests."
  - "Metadata patching changes only user_metadata and signs a full replacement manifest."
  - "Clear builds recovery mappings from canonical locators after authenticating the entire selected set."
patterns-established:
  - "Use _load_authenticated_manifest before exposing metadata, resolving handlers, snapshotting payloads, or mutating locators."
  - "Treat legacy backend locator projections as containment tripwires only; never as the source of an operation's locator."
requirements-completed: [STOR-02, STOR-08, SECU-03, SECU-04, SECU-05, SECU-08]
coverage:
  - id: D1
    description: "get_metadata, exists, and list expose only authenticated committed canonical records without read-side mutation."
    requirement: STOR-02
    verification:
      - kind: integration
        ref: "tests/test_blob_store_read_contract.py -k 'get_metadata or exists or list'"
        status: pass
    human_judgment: false
  - id: D2
    description: "Direct read-only operations preserve absence, integrity, lifecycle-conflict, and payload-tampering outcomes."
    requirement: STOR-08
    verification:
      - kind: integration
        ref: "tests/test_blob_store_read_contract.py tests/test_filesystem_containment.py"
        status: pass
    human_judgment: false
  - id: D3
    description: "update_metadata permits only re-signed user metadata while delete and clear authenticate locators before mutation."
    requirement: SECU-03
    verification:
      - kind: integration
        ref: "tests/test_blob_store_read_contract.py tests/test_clear_recovery.py tests/test_filesystem_containment.py"
        status: pass
    human_judgment: false
duration: 6min
completed: 2026-08-30
status: complete
---

# Phase 02 Plan 06: Direct BlobStore Operation Contract Summary

**Every direct BlobStore surface now authenticates a committed canonical manifest before exposing state, verifying payloads, or mutating a locator.**

## Performance

- **Duration:** 6min
- **Started:** 2026-08-30T14:47:17Z
- **Completed:** 2026-08-30T14:53:34Z
- **Tasks:** 2/2
- **Files modified:** 2

## Accomplishments

- Centralized committed manifest parsing, signature verification, lifecycle checks, payload-contract resolution, and locator validation for direct BlobStore calls.
- Made `get_metadata`, `exists`, and `list` return authenticated canonical truth; `exists` verifies a single snapshot's digest and size without deserializing.
- Restricted `update_metadata` to a bounded user-metadata patch followed by full manifest re-signing, and preflighted `delete` and `clear` before destructive work.

## Task Commits

1. **Task 1: Enforce committed authenticated semantics on read-only public operations** - `75f6d6d` (RED), `588fde9` (GREEN)
2. **Task 2: Constrain metadata patches and authenticate mutation locators** - `e74d300` (RED), `df14dc6` (GREEN)

## Files Created/Modified

- `src/cacheness/storage/blob_store.py` - Shared authenticated manifest boundary across direct reads, overwrite, patch, delete, and clear.
- `tests/test_blob_store_read_contract.py` - Direct operation matrix for successful, tampered, conflicted, and no-side-effect outcomes.

## Decisions Made

- Direct calls treat only a missing raw manifest as absence; all present records pass the same committed/authenticated boundary.
- Backend-shaped metadata can signal an unsafe legacy locator, but canonical manifests remain the only authoritative locator source.
- `clear` leaves corrupt, unknown, and conflicted manifest evidence untouched by completing all preflight before the recovery coordinator runs.

## TDD Gate Compliance

- RED commits: `75f6d6d`, `e74d300`
- GREEN commits: `588fde9`, `df14dc6`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Integration] Retained containment-only validation for legacy backend projections.**
- **Found during:** Task 2 verification
- **Issue:** Existing containment regression coverage requires list and clear to reject a hostile legacy locator projection before any sibling operation proceeds.
- **Fix:** Authenticate canonical manifests first, then validate backend projections only as non-authoritative containment tripwires.
- **Files modified:** `src/cacheness/storage/blob_store.py`
- **Verification:** `uv run pytest -q -o log_cli=false tests/test_blob_store_read_contract.py tests/test_clear_recovery.py tests/test_filesystem_containment.py -x`
- **Committed in:** `df14dc6`

---

**Total deviations:** 1 auto-fixed (1 Rule 1 integration repair).
**Impact on plan:** The repair preserves canonical manifest authority and adds no lifecycle, recovery, or cache-policy scope.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 3 can add lifecycle/CAS behavior on a complete direct-operation authenticated boundary.
- Phase 6 can consume the same typed direct outcomes without weakening BlobStore integrity behavior.

## Self-Check: PASSED

- Confirmed both modified source/test files and the Summary exist on disk.
- Confirmed all four Task 1/2 RED/GREEN commits exist in git history.

---
*Phase: 02-canonical-storage-and-integrity-contract*
*Completed: 2026-08-30*
