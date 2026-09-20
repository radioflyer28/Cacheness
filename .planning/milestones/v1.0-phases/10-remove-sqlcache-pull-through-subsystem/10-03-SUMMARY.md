---
phase: 10-remove-sqlcache-pull-through-subsystem
plan: 03
subsystem: packaging
tags: [wheel, pytest, uv, dependency-metadata, sqlcache-removal]
requires:
  - phase: 09-adoption-and-release-surface-closure
    provides: One-artifact, source-free wheel qualification harness
provides:
  - ZIP-member, retired-import, and installed-metadata checks bound to one WheelArtifact digest
  - Synthetic packaging contracts for stale SqlCache, DuckDB, and sql-extra artifacts
affects: [phase-10-cutover, dependency-pruning, release-qualification]
actuals:
  tokens: 2479
  tasks: 2
  commits: 2
tech-stack:
  added: []
  patterns:
    - Artifact digest validation before ZIP inspection and isolated installation
    - Generated-probe prelude tests for installed metadata failure paths
key-files:
  created:
    - .planning/phases/10-remove-sqlcache-pull-through-subsystem/10-03-SUMMARY.md
  modified:
    - tools/run_phase8_packaging.py
    - tests/packaging/test_wheel_matrix.py
key-decisions:
  - "Use the existing WheelArtifact path and SHA-256 for ZIP inspection, source-free installation, and retained local round trips instead of adding another build harness."
  - "Keep real wheel execution fail-closed and intentionally red until the physical SqlCache and DuckDB cut removes the stale artifact surface."
patterns-established:
  - "Retired package surfaces are checked both in wheel ZIP members and installed distribution metadata."
requirements-completed: [CACH-07]
coverage:
  - id: D1
    description: "The sole wheel harness rejects a stale SqlCache member and verifies its digest before isolated installation."
    requirement: CACH-07
    verification:
      - kind: unit
        ref: "tests/packaging/test_wheel_matrix.py::test_base_probe_rejects_a_retired_wheel_member_before_installation"
        status: pass
      - kind: other
        ref: "uv run --isolated --group dev --frozen ruff check tools/run_phase8_packaging.py tests/packaging/test_wheel_matrix.py"
        status: pass
    human_judgment: true
    rationale: "The real fresh-wheel run is intentionally red until the later physical source and dependency cut removes the currently shipped member."
  - id: D2
    description: "The generated isolated probe rejects DuckDB requirements, the sql extra, and retired top-level SqlCache exports while retaining the existing local journey inventory."
    requirement: CACH-07
    verification:
      - kind: unit
        ref: "tests/packaging/test_wheel_matrix.py::test_base_probe_rejects_retired_installed_metadata_and_exports"
        status: pass
      - kind: other
        ref: "uv run --isolated --all-extras --group dev --frozen pytest -q --collect-only tests/packaging/test_wheel_matrix.py"
        status: pass
    human_judgment: true
    rationale: "The installed probe must remain red against the pre-cut wheel and become a release proof only after the source and manifest removal plan lands."
duration: 9 min
completed: 2026-09-17
status: complete
---

# Phase 10 Plan 03: Artifact-First Packaging Contract Summary

**The existing one-wheel harness now fails closed on shipped SqlCache members, DuckDB metadata, and retired imports while preserving BlobStore and UnifiedCache local journeys.**

## Performance

- **Duration:** 9 min
- **Started:** 2026-09-17T16:48:41Z
- **Completed:** 2026-09-17T16:58:10Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Bound ZIP-member inspection, source-free installation, installed metadata, and retained local round trips to one digest-carrying `WheelArtifact`.
- Moved `SqlCache` and `SqlCacheAdapter` into the literal retired inventory and required their natural import absence.
- Added synthetic tests for stale wheel membership, DuckDB requirements, the `sql` extra, and a surviving retired top-level export without creating a second build path.

## Task Commits

1. **Task 1: Bind retired members and installed metadata to one WheelArtifact** - `e250165` (feat)
2. **Task 2: Freeze wheel absence and retained-round-trip behavior in integration tests** - `55b9f56` (test)

## Files Created/Modified

- `tools/run_phase8_packaging.py` - Validates the immutable wheel digest, ZIP member inventory, retired imports, and installed distribution metadata in the existing base probe.
- `tests/packaging/test_wheel_matrix.py` - Exercises synthetic stale artifact and installed-metadata failures while retaining the real one-wheel journey test.

## Decisions Made

- Used the one existing `WheelArtifact` rather than a parallel build harness, so its path and recorded SHA-256 bind ZIP inspection and isolated probes.
- Kept the real package test as a mandatory post-cut proof rather than skipping or xfail-marking it before source/dependency removal.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Replaced stale fake wheel bytes in runner tests with digest-bound ZIP artifacts**

- **Found during:** Task 2 (Freeze wheel absence and retained-round-trip behavior in integration tests)
- **Issue:** Existing fake subprocess tests used arbitrary bytes and placeholder hashes, which the new digest-and-ZIP validation correctly rejected before their intended assertions.
- **Fix:** Added a minimal synthetic-wheel helper that creates a valid ZIP archive and computes the runner's actual SHA-256.
- **Files modified:** `tests/packaging/test_wheel_matrix.py`
- **Verification:** The ten synthetic/non-real wheel tests pass with the new artifact guard active.
- **Committed in:** `55b9f56`

**2. [Rule 1 - Bug] Isolated generated-probe tests from an already imported checkout module**

- **Found during:** Task 2 (Freeze wheel absence and retained-round-trip behavior in integration tests)
- **Issue:** The test process could retain `cacheness.sql_cache` in `sys.modules`, making the synthetic installed-probe prelude detect the checkout module before reaching the metadata case under test.
- **Fix:** The mock installed package explicitly removes that module from `sys.modules` before executing the generated prelude.
- **Files modified:** `tests/packaging/test_wheel_matrix.py`
- **Verification:** The DuckDB-requirement, `sql`-extra, and retired-export cases each exercise their intended failing assertion.
- **Committed in:** `55b9f56`

---

**Total deviations:** 2 auto-fixed (2 Rule 1 bugs).
**Impact on plan:** Both changes keep the new artifact boundary tests faithful to their target condition; no production, dependency, or lifecycle scope expanded.

## Issues Encountered

The real fresh-wheel base probe intentionally fails before installation because the current wheel still contains `cacheness/sql_cache.py`. This is the planned pre-cut failure and is not masked by a skip or xfail; the later physical source and dependency cut must make it green.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- The direct product/dependency removal can now use a single artifact proof that detects stale source members and distribution metadata.
- Scoped Ruff and collection are green; the real retained-round-trip probe remains deliberately red only against the current pre-cut artifact contents.

## Self-Check: PASSED

- Confirmed both modified packaging files exist and scoped Ruff passes.
- Confirmed task commits `e250165` and `55b9f56` exist in Git history.
- Confirmed all 12 packaging tests collect; the ten synthetic/non-real checks pass, while the real fresh-wheel probe fails at the expected stale `cacheness/sql_cache.py` member.

---
*Phase: 10-remove-sqlcache-pull-through-subsystem*
*Completed: 2026-09-17*
