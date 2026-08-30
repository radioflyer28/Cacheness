---
phase: 02-canonical-storage-and-integrity-contract
plan: "03"
subsystem: storage
tags: [handlers, payload-contract, npz, parquet, pickle, dill, blosc2]
requires:
  - phase: 02-01
    provides: Canonical manifests with independent handler and payload identity fields
  - phase: 02-02
    provides: Exact local canonical-manifest persistence for direct BlobStore composition
provides:
  - Stable handler-owned payload format and version identities on built-in writes
  - Non-deserializing handler contract resolution for canonical payload declarations
  - Native-container evidence for NPZ, Parquet, pickle, dill, and read-only legacy Blosc2
affects: [02-05-integrity-pipeline, BlobStore, handler-registry, migration-compatibility]
actuals:
  tokens: 4867
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - Handler-owned native payload identities are explicit in guarded write results
    - Canonical payload support is checked through handler declarations without opening payload bytes
key-files:
  created: []
  modified:
    - src/cacheness/interfaces.py
    - src/cacheness/handlers.py
    - tests/test_blob_manifest.py
key-decisions:
  - "Payload format versions describe Cacheness handler contracts, never installed library versions."
  - "Canonical support resolution is a pure registry check; payload IO remains outside the handler-identity boundary."
  - "Legacy Blosc2 remains read-only and is selected only when its explicit payload identity is declared."
patterns-established:
  - "Built-in writes return payload_format and payload_format_version alongside compatibility storage_format metadata."
  - "Native payload tests inspect each owning library container directly instead of adding a Cacheness framing format."
requirements-completed: [STOR-01, SECU-03, MIGR-02, MIGR-07]
coverage:
  - id: D1
    description: Built-in handlers publish stable native payload identities and resolve declared support without deserializing payload bytes.
    requirement: MIGR-02
    verification:
      - kind: unit
        ref: tests/test_blob_manifest.py#test_builtin_writes_publish_explicit_payload_format_identity
        status: pass
      - kind: unit
        ref: tests/test_blob_manifest.py#test_handler_identity_resolution_is_independent_of_payload_bytes
        status: pass
    human_judgment: false
  - id: D2
    description: Native NPZ, Parquet, pickle, dill, and legacy Blosc2 containers remain directly consumable while unsupported identities are rejected before handler invocation.
    requirement: MIGR-07
    verification:
      - kind: integration
        ref: tests/test_blob_manifest.py native-container and unsupported-identity contracts
        status: pass
      - kind: integration
        ref: tests/test_stored_compatibility.py
        status: pass
    human_judgment: false
duration: 6min
completed: 2026-08-30
status: complete
---

# Phase 02 Plan 03: Native Handler Payload Contracts Summary

**Built-in handlers now publish independently versioned native payload identities, while direct tests prove Cacheness leaves NPZ, Parquet, pickle, dill, and legacy Blosc2 containers under their owning libraries.**

## Performance

- **Duration:** 6min
- **Started:** 2026-08-30T14:07:50Z
- **Completed:** 2026-08-30T14:13:34Z
- **Tasks:** 2/2
- **Files modified:** 3

## Accomplishments

- Added `payload_format` and `payload_format_version` to guarded write results and stable built-in handler declarations.
- Added pure registry resolution for exact `(handler_type, payload_format, payload_format_version)` declarations, returning a typed unsupported-version result without opening payload bytes.
- Kept ordinary arrays in pickle-disabled NPZ, preserved read-only legacy Blosc2 routing, and proved native Parquet, pickle, and dill containers remain directly library-consumable.

## Task Commits

1. **Task 1: Add explicit native payload identity to handler contracts** - `fea9672` (RED), `7560810` (GREEN)
2. **Task 2: Prove native containers and reject unknown payload contracts** - `85865a8` (RED), `fd28b40` (GREEN)

## Files Created/Modified

- `src/cacheness/interfaces.py` - Extends handler and guarded-write contracts with explicit native payload identities.
- `src/cacheness/handlers.py` - Declares built-in payload formats, checks supported contracts without IO, and dispatches legacy arrays from declared payload format.
- `tests/test_blob_manifest.py` - Covers independent identity versions, native container signatures/round trips, legacy Blosc2 dispatch, and unsupported-contract no-handler events.

## Decisions Made

- Kept payload-format versions as stable Cacheness contract numbers, separate from manifest schema and dependency versions.
- Kept every payload container native: no Cacheness header, wrapper, or format guessing was introduced.
- Treated the registry as the non-deserializing support boundary that Plan 02-05 will call before its guarded snapshot pipeline.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Test fixture] Corrected native magic-byte assertions.**
- **Found during:** Task 2 RED
- **Issue:** The first assertions encoded backslashes literally rather than asserting the NPZ ZIP and pickle protocol bytes.
- **Fix:** Replaced the escaped-literal fixtures with the real byte sequences.
- **Files modified:** `tests/test_blob_manifest.py`
- **Verification:** Native-container contract tests then reached the intended legacy-dispatch failure.
- **Committed in:** `85865a8`

**2. [Rule 1 - Test lint] Replaced a lambda-only dill fixture with a local function.**
- **Found during:** Task 2 verification
- **Issue:** Ruff's active E731 rule rejects lambda assignment.
- **Fix:** Used an equally non-pickleable local function to retain dill-fallback coverage.
- **Files modified:** `tests/test_blob_manifest.py`
- **Verification:** `uv run pytest -q -o log_cli=false tests/test_blob_manifest.py tests/test_handlers.py tests/test_stored_compatibility.py -x`
- **Committed in:** `fd28b40`

---

**Total deviations:** 2 auto-fixed (2 Rule 1 test corrections).
**Impact on plan:** Both fixes sharpen the planned native-format assertions without expanding storage lifecycle scope.

## Issues Encountered

- The plan's exact Ruff command reports three pre-existing `F401` imports in `src/cacheness/handlers.py` (`CacheHandlerError`, `CacheFormatError`, and `verify_dill_serializable`). The planned tests pass, and Ruff passes for the changed interface/test files and for all changed files when those existing `F401` findings are excluded. The unrelated baseline imports were left unchanged.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plan 02-05 can use `HandlerRegistry.resolve_payload_contract()` before opening its one guarded snapshot. The handler layer now supplies exact native identities, while strict direct `BlobStore` authentication and snapshot ordering remain intentionally owned by Plan 02-05.

## Self-Check: PASSED

- Confirmed all three planned files exist on disk.
- Confirmed TDD RED/GREEN commits `fea9672`, `7560810`, `85865a8`, and `fd28b40` exist in git history.

---
*Phase: 02-canonical-storage-and-integrity-contract*
*Completed: 2026-08-30*
