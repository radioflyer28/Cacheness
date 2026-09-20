---
phase: 02-canonical-storage-and-integrity-contract
plan: "07"
subsystem: storage
tags: [blobstore, legacy-compatibility, migration-required, non-mutation, regression]
requires:
  - phase: 02-06
    provides: authenticated committed-manifest semantics across direct BlobStore operations
provides:
  - Exact in-memory identities for the eight Phase 1 compatibility fixtures
  - Typed migration-required outcomes for every direct BlobStore operation on legacy evidence
  - Cross-surface failure regressions for malformed, future-version, lifecycle, and backend outcomes
affects: [phase-03-lifecycle, phase-06-unified-cache, phase-07-migration, BlobStore]
actuals:
  tokens: 18178
  tasks: 2
  commits: 3
tech-stack:
  added: []
  patterns:
    - Bounded fixed-name legacy recognition without canonical fallback or persistence
    - Immutable SQLite URI inspection prevents WAL/SHM sidecars on historical evidence
    - All direct read APIs preserve typed failure outcomes and stored evidence
key-files:
  created:
    - src/cacheness/storage/legacy_manifest.py
    - tests/test_blob_store_legacy_contract.py
  modified:
    - src/cacheness/storage/blob_store.py
    - tests/test_blob_store_read_contract.py
key-decisions:
  - "Exact Phase 1 fixture trees attach in-memory identities and return migration-required; Phase 7 alone owns migration execution."
  - "Legacy SQLite inspection uses immutable read-only mode so compatibility detection cannot create journal sidecars."
  - "Every direct read API preserves malformed, future-version, lifecycle, and local-backend failures rather than collapsing them into absence."
patterns-established:
  - "Recognize only the fixed provenance and exact metadata shape; never enumerate, infer, or write legacy layouts."
  - "Treat evidence-preservation checks as cross-surface contracts, including source/copy SHA-256 values and mtimes."
requirements-completed: [STOR-01, STOR-02, STOR-08, SECU-03, SECU-04, SECU-05, SECU-08, MIGR-02, MIGR-07]
coverage:
  - id: D1
    description: Exact legacy fixture identities are read-only and direct BlobStore operations report typed migration-required outcomes without evidence mutation.
    requirement: MIGR-07
    verification:
      - kind: integration
        ref: tests/test_blob_store_legacy_contract.py
        status: pass
      - kind: other
        ref: uv run python tests/fixtures/compat/validate_corpus.py --expected-through sqlite-columns-v0314
        status: pass
    human_judgment: false
  - id: D2
    description: Canonical direct read surfaces retain distinct malformed, unsupported-version, lifecycle-conflict, and backend-failure outcomes while preserving payload and manifest evidence.
    requirement: STOR-08
    verification:
      - kind: integration
        ref: tests/test_blob_store_read_contract.py#test_every_direct_read_surface_preserves_ordered_typed_failures
        status: pass
      - kind: integration
        ref: tests/test_blob_store_read_contract.py#test_every_direct_read_surface_propagates_a_local_backend_failure
        status: pass
    human_judgment: false
  - id: D3
    description: All Wave 0 contracts, Phase 1 compatibility/security/recovery regressions, the independent corpus validator, and Phase 2 Ruff gate pass together.
    requirement: SECU-03
    verification:
      - kind: integration
        ref: uv run pytest -q -o log_cli=false
        status: pass
      - kind: other
        ref: Phase 2 targeted Ruff command from 02-07-PLAN.md
        status: pass
    human_judgment: false
duration: 10min
completed: 2026-08-30
status: complete
---

# Phase 02 Plan 07: Compatibility Edge and Complete Contract Summary

**Exact read-only Phase 1 fixture recognition now gives direct BlobStore callers typed migration-required outcomes while the complete canonical integrity contract remains green.**

## Performance

- **Duration:** 10min
- **Started:** 2026-08-30T14:59:48Z
- **Completed:** 2026-08-30T15:09:54Z
- **Tasks:** 2/2
- **Files modified:** 4

## Accomplishments

- Added an exact, bounded legacy recognizer for all eight immutable Phase 1 fixture identities, with no tree traversal, canonical fallback, key creation, migration, or read-time write.
- Attached recognized legacy identity only in memory and made every direct BlobStore operation return the stable typed migration-required outcome.
- Added fixture source/copy digest and mtime assertions, including immutable SQLite inspection that cannot create WAL or SHM sidecars.
- Extended the direct-read matrix to verify all public read APIs retain malformed, future-version, lifecycle-conflict, and local-backend outcomes without mutating manifest or payload evidence.
- Passed the six Wave 0 suites, Phase 1 compatibility/security/recovery regressions, independent compatibility-corpus validator, full pytest suite, and complete Phase 2 targeted Ruff command.

## Task Commits

1. **Task 1: Recognize exact legacy records and preserve every fixture byte** - `cb5dbb9` (RED), `0d4d372` (GREEN)
2. **Task 2: Close the full canonical, integrity, and non-mutation contract** - `3ecf8e6` (cross-surface regression coverage)

## Files Created/Modified

- `src/cacheness/storage/legacy_manifest.py` - Fixed-name, bounded, read-only identity recognition for the exact Phase 1 layouts.
- `src/cacheness/storage/blob_store.py` - Attaches legacy identity before initialization and blocks public operations with a typed migration-required result.
- `tests/test_blob_store_legacy_contract.py` - Eight-fixture identity, read-surface, malformed-lookalike, signature-shape, digest, and mtime invariance coverage.
- `tests/test_blob_store_read_contract.py` - Full direct-read outcome matrix for malformed, future-version, lifecycle, and backend failures.

## Decisions Made

- Exact legacy layouts never participate in canonical manifest decoding or persistence; they remain inspectable only until the explicit Phase 7 migration workflow.
- SQLite evidence is inspected with `mode=ro&immutable=1` to prevent journal sidecars and protect the only historical copy.
- A present raw record is always a typed direct outcome; only an actually absent raw record remains a compatible `None`/`False`/empty-list result.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Prevented SQLite compatibility inspection from creating journal sidecars**
- **Found during:** Task 1: Recognize exact legacy records and preserve every fixture byte
- **Issue:** A nominally read-only SQLite URI created `-wal` and `-shm` files while inspecting copied historical evidence, violating the no-mutation contract.
- **Fix:** Used SQLite immutable read-only mode and added evidence assertions that catch any additional files, digest changes, or mtime drift.
- **Files modified:** `src/cacheness/storage/legacy_manifest.py`, `tests/test_blob_store_legacy_contract.py`
- **Verification:** Legacy suite, both high-severity legacy subsets, corpus validator, full pytest, and targeted Ruff passed.
- **Committed in:** `0d4d372`

**2. [Rule 1 - Test fixture] Constructed unsupported schema evidence as raw bytes**
- **Found during:** Task 2: Close the full canonical, integrity, and non-mutation contract
- **Issue:** The immutable schema model correctly rejects future schema versions during construction, so the initial regression could not reach the direct read boundary it was intended to test.
- **Fix:** Created the future-version fixture by changing canonical raw bytes after a valid write, preserving the intended boundary test and non-mutation assertions.
- **Files modified:** `tests/test_blob_store_read_contract.py`
- **Verification:** Full direct-read contract suite, full pytest, and targeted Ruff passed.
- **Committed in:** `3ecf8e6`

---

**Total deviations:** 2 auto-fixed (2 Rule 1).
**Impact on plan:** Both fixes tightened the required no-mutation and unknown-version guarantees without adding migration execution, general lifecycle reconciliation, backend expansion, or UnifiedCache rewiring.

## Issues Encountered

None - all required automated gates passed. Full pytest reported the existing collection warning for `TestDataClassForConsistency` and expected optional/platform skips only.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 2 now has its complete direct BlobStore canonical, authenticated, version-aware, non-mutating contract. Phase 3 can add lifecycle/CAS behavior, Phase 6 can consume the typed seam, and Phase 7 can own explicit migration execution without relying on read-time conversion.

## Self-Check: PASSED

- Confirmed all four planned source/test artifacts and this Summary exist on disk.
- Confirmed task commits `cb5dbb9`, `0d4d372`, and `3ecf8e6` exist in git history.
