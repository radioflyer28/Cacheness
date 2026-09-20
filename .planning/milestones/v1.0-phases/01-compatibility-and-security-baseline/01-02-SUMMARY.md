---
phase: 01-compatibility-and-security-baseline
plan: "02"
subsystem: storage-security
tags: [filesystem, containment, symlink-safety, no-follow, blob-storage]
requires:
  - phase: 01-01
    provides: CacheReason and CacheUnsafePathError public contract
provides:
  - Strict opaque backend-ID validation and contained-locator validation
  - Anchored-root, no-follow managed filesystem operations
  - FilesystemBlobBackend routing for every direct filesystem entry point
affects: [01-03, BlobStore, UnifiedCache, filesystem-backends]
tech-stack:
  added: []
  patterns:
    - Shared standard-library containment guard at the lowest filesystem boundary
    - Descriptor-relative no-follow operations with a locked portable fallback
key-files:
  created:
    - src/cacheness/storage/path_security.py
    - tests/test_filesystem_containment.py
  modified:
    - src/cacheness/storage/backends/blob_backends.py
    - tests/test_directory_sharding.py
    - tests/test_blob_backend_registry.py
key-decisions:
  - "FilesystemBlobBackend accepts only opaque IDs; Plan 03 owns logical-key encoding."
  - "Resolved roots are anchored once, while every managed descendant is revalidated per operation."
patterns-established:
  - "Filesystem operations must use ManagedFileOps rather than construct paths directly."
  - "Unsafe locators raise CacheUnsafePathError rather than becoming cache misses."
requirements-completed: [SECU-01]
actuals:
  tokens: 10540
  tasks: 2
  commits: 3
coverage:
  - id: D1
    description: Strict opaque-ID and locator containment boundary for filesystem blobs
    requirement: SECU-01
    verification:
      - kind: integration
        ref: "uv run pytest -q -o log_cli=false tests/test_filesystem_containment.py tests/test_directory_sharding.py tests/test_blob_backend_registry.py -x"
        status: pass
      - kind: unit
        ref: "uv run ruff check src/cacheness/storage/path_security.py src/cacheness/storage/backends/blob_backends.py tests/test_filesystem_containment.py tests/test_directory_sharding.py tests/test_blob_backend_registry.py"
        status: pass
    human_judgment: false
duration: 14min
completed: 2026-08-29
status: complete
---

# Phase 01 Plan 02: Filesystem Containment Summary

**A fail-closed, root-anchored filesystem blob boundary rejects hostile IDs and locators while retaining atomic sharded storage.**

## Performance

- **Duration:** 14 min
- **Started:** 2026-08-29T20:06:38Z
- **Completed:** 2026-08-29T20:20:27Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Added a standard-library-only containment module that validates cross-platform path shapes, anchors resolved roots, rejects managed links/reparse points, and uses descriptor-relative no-follow operations when available.
- Routed filesystem bytes, streams, existence, size, and deletion through `ManagedFileOps`; unsafe input now raises typed errors instead of returning miss-like values.
- Replaced traversal sanitization and path-shaped direct ID behavior with executable strict-opaque-ID contracts, including root aliases and pathname-swap stress coverage.

## Task Commits

Each task was committed atomically:

1. **Task 1: Define the operation-scoped containment and no-follow contract**
   - `3478a57` `test(01-02): add failing containment contract tests`
   - `c2329a3` `feat(01-02): enforce managed filesystem containment`
2. **Task 2: Route every FilesystemBlobBackend operation through the guard**
   - `553ace6` `feat(01-02): guard filesystem blob operations`

## Files Created/Modified

- `src/cacheness/storage/path_security.py` - strict ID/locator validation plus managed no-follow operations.
- `src/cacheness/storage/backends/blob_backends.py` - all filesystem backend operations delegate to the shared guard.
- `tests/test_filesystem_containment.py` - hostile path, link, retarget, root-anchor, and backend-operation corpus.
- `tests/test_directory_sharding.py` - valid canonical-root sharding and typed traversal-rejection expectations.
- `tests/test_blob_backend_registry.py` - strict direct-ID expectation for legacy nested paths.

## Decisions Made

- Kept direct `FilesystemBlobBackend` identifiers restricted to `[A-Za-z0-9][A-Za-z0-9._-]{0,255}`; Plan 03 will encode public logical keys before this boundary.
- Resolved the configured root once and retained a descriptor on capable Unix systems, so later root-alias retargeting cannot redirect an instance.
- Allowed a resolved absolute alias only when it canonically lands below the anchored root; all actual operations use the canonical descendant path.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Preserved safe locator behavior with canonical macOS root aliases and missing leaves**
- **Found during:** Task 2
- **Issue:** A valid `/var` temporary-directory locator resolved beneath the anchored `/private/var` root but was rejected lexically; descriptor reads also translated a normal missing leaf into `path_race`.
- **Fix:** Canonicalized aliases only after containment succeeds and preserved normal `FileNotFoundError` behavior for safe missing reads.
- **Files modified:** `src/cacheness/storage/path_security.py`, `tests/test_directory_sharding.py`
- **Verification:** Focused filesystem, sharding, and backend-registry tests passed.
- **Committed in:** `553ace6`

**2. [Rule 2 - Critical contract coverage] Updated an unlisted registry test for the strict direct-ID contract**
- **Found during:** Task 2
- **Issue:** `tests/test_blob_backend_registry.py` still asserted that a slash-delimited direct filesystem ID succeeded, contradicting D-12 and the new fail-closed backend boundary.
- **Fix:** Replaced the nested-path success assertion with typed `CacheUnsafePathError` rejection coverage.
- **Files modified:** `tests/test_blob_backend_registry.py`
- **Verification:** `tests/test_blob_backend_registry.py` passed in the focused verification command.
- **Committed in:** `553ace6`

**Total deviations:** 2 auto-fixed (1 Rule 1, 1 Rule 2)

**Impact on plan:** Both changes are required to preserve valid backend behavior while enforcing the planned security contract; no lifecycle, metadata, or migration scope was added.

## Issues Encountered

- The macOS temporary-directory alias exposed a canonical-path compatibility edge case; it is covered by the guarded locator contract and valid sharding tests.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 03 can provide the high-level `encode_physical_name` translation knowing direct backend IDs are strictly validated.
- BlobStore and UnifiedCache callers can adopt `ManagedFileOps` without translating unsafe locators into misses.

## Self-Check: PASSED

- All five implementation/test files and this summary exist.
- All three Task 1/2 TDD commits (`3478a57`, `c2329a3`, `553ace6`) are present in git history.

---
*Phase: 01-compatibility-and-security-baseline*
*Completed: 2026-08-29*
