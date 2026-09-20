---
phase: 02-canonical-storage-and-integrity-contract
plan: "04"
subsystem: storage
tags: [blobstore, direct-read, errors, integrity, cache-policy]
requires:
  - phase: 02-01
    provides: Canonical manifest direct-read failures and stable public error bases
  - phase: 02-03
    provides: Independently versioned native payload contracts
provides:
  - Precise public BlobStore direct-read exception subtypes with stable reasons
  - Pure closed classification for a future cache-policy translation boundary
affects: [02-05-integrity-pipeline, 02-07-legacy-migration, phase-06-unified-cache]
actuals:
  tokens: 7662
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - Precise public BlobStore leaf errors preserve broad compatible cache bases
    - Pure exception-to-category classification keeps cache policy out of storage
key-files:
  created:
    - src/cacheness/storage/read_contract.py
    - tests/test_blob_store_translation_seam.py
  modified:
    - src/cacheness/error_handling.py
    - src/cacheness/storage/__init__.py
key-decisions:
  - "Precise BlobStore subtypes retain the existing broad error bases and stable direct reason codes for compatibility."
  - "The future cache seam classifies only explicit BlobStore failures and returns None for compatible direct absence."
patterns-established:
  - "Direct storage failure types are the stable caller-branching interface; prose remains non-contractual."
  - "Future cache translation starts with a pure classifier and cannot change counters, metadata, TTL, eviction, or cache construction."
requirements-completed: [STOR-08, SECU-08, MIGR-07]
coverage:
  - id: D1
    description: Public BlobStore callers receive precise compatible direct-read exception types with stable reasons and storage-barrel exports.
    requirement: STOR-08
    verification:
      - kind: unit
        ref: tests/test_blob_store_translation_seam.py#test_blob_store_errors_have_stable_public_types_and_reasons
        status: pass
      - kind: unit
        ref: tests/test_blob_store_translation_seam.py#test_storage_barrel_exports_each_direct_read_failure_type
        status: pass
    human_judgment: false
  - id: D2
    description: A pure classifier distinguishes each direct BlobStore failure and leaves absence outside the taxonomy.
    requirement: SECU-08
    verification:
      - kind: unit
        ref: tests/test_blob_store_translation_seam.py#test_classifier_exhaustively_preserves_direct_failure_categories
        status: pass
      - kind: unit
        ref: tests/test_blob_store_translation_seam.py#test_classifier_is_pure_and_has_no_unified_cache_dependency
        status: pass
    human_judgment: false
  - id: D3
    description: Manifest and payload version outcomes remain distinct and migration-required outcomes remain observable for later read-only compatibility handling.
    requirement: MIGR-07
    verification:
      - kind: unit
        ref: tests/test_blob_store_translation_seam.py#test_classifier_exhaustively_preserves_direct_failure_categories
        status: pass
    human_judgment: false
duration: 6 min
completed: 2026-08-30
status: complete
---

# Phase 02 Plan 04: Direct Failure Taxonomy and Translation Seam Summary

**BlobStore now exposes compatible, reason-coded direct-read failure subtypes and a pure future cache-translation classifier with no UnifiedCache wiring.**

## Performance

- **Duration:** 6 min
- **Started:** 2026-08-30T14:18:49Z
- **Completed:** 2026-08-30T14:26:02Z
- **Tasks:** 2/2
- **Files modified:** 4

## Accomplishments

- Added public manifest-malformed, unauthenticated, payload-missing, payload-tampered, and manifest/payload-version error subtypes while retaining compatible integrity and storage base catches.
- Exported the supported direct-storage errors from `cacheness.storage` so callers need no codec-internal imports.
- Added a closed, side-effect-free direct-read classifier that preserves non-integrity categories and returns `None` for compatible absence.

## Task Commits

1. **Task 1: Freeze public BlobStore exception types and stable reasons** - `d4ab545` (RED), `fbe606a` (GREEN)
2. **Task 2: Implement the pure future cache-translation classifier** - `ba17f01` (RED), `ec8ded3` (GREEN)

## Files Created/Modified

- `src/cacheness/error_handling.py` - Defines compatible BlobStore direct-read error subtypes and stable reason aliases.
- `src/cacheness/storage/__init__.py` - Re-exports the supported direct-storage failure types.
- `src/cacheness/storage/read_contract.py` - Classifies direct exceptions without importing or invoking cache policy.
- `tests/test_blob_store_translation_seam.py` - Exhaustively covers types, reasons, absence, classifier categories, and purity.

## Decisions Made

- Precise BlobStore subtypes retain the existing broad error bases and stable reason codes so callers can adopt exact branches without losing compatibility.
- The future cache seam treats only direct BlobStore integrity failures as its integrity category; it leaves absence unclassified and preserves version, lifecycle, backend, and migration categories.

## TDD Gate Compliance

- RED commits: `d4ab545`, `ba17f01`
- GREEN commits: `fbe606a`, `ec8ded3`

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plan 02-05 can raise the precise direct failure subtypes from its integrity pipeline. Phase 6 can consume `classify_cache_read_failure()` without changing current UnifiedCache policy, counters, metadata, TTL, or eviction behavior.

## Self-Check: PASSED

- Confirmed all four planned source and test files plus this SUMMARY exist on disk.
- Confirmed TDD RED/GREEN commits `d4ab545`, `fbe606a`, `ba17f01`, and `ec8ded3` exist in git history.

---
*Phase: 02-canonical-storage-and-integrity-contract*
*Completed: 2026-08-30*
