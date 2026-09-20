---
phase: 10-remove-sqlcache-pull-through-subsystem
plan: 08
subsystem: documentation
tags: [sqlcache-removal, blobstore, unifiedcache, migration-boundaries]
requires:
  - phase: 10-remove-sqlcache-pull-through-subsystem
    provides: Inverted documentation ownership contracts and dedicated-surface removal
provides:
  - Bounded canonical cutover guidance for object/function cache policy and direct persistence
  - Current dataframe and platform guidance without retired SqlCache ownership claims
affects: [phase-10-closure, documentation, public-api-guidance]
actuals:
  tokens: 2736
  tasks: 2
  commits: 2
tech-stack:
  added: []
  patterns:
    - Keep retired-product wording in the three canonical owners only
    - Remove obsolete ownership references from mixed guides without rewriting retained guidance
key-files:
  created:
    - .planning/phases/10-remove-sqlcache-pull-through-subsystem/10-08-SUMMARY.md
  modified:
    - docs/API_REFERENCE.md
    - docs/STORAGE_MIGRATION.md
    - docs/README.md
    - docs/PANDAS_API_AUDIT.md
    - docs/CROSS_PLATFORM_GUIDE.md
key-decisions:
  - "UnifiedCache is documented as object/function cache policy over BlobStore; BlobStore is documented as direct object persistence."
  - "Cacheness documents no in-package range-aware SQL pull-through replacement and no caller-table migration or cleanup authority."
  - "Pandas/Parquet and isolated-suite guidance stay intact while only retired ownership claims are removed."
patterns-established:
  - "Current-facing cutover guidance distinguishes supported alternatives from a feature-equivalent replacement claim."
requirements-completed: [CACH-07]
coverage:
  - id: D1
    description: Canonical API, migration, and documentation-index owners give the same bounded SqlCache cutover guidance.
    requirement: CACH-07
    verification:
      - kind: unit
        ref: uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase9_documentation.py::test_canonical_cutover_notes_have_exact_owners_and_boundaries tests/test_phase9_documentation.py::test_navigation_has_no_pre_cutover_configuration_or_backend_branches -x
        status: pass
    human_judgment: false
  - id: D2
    description: Mixed pandas and platform documents preserve handler, Parquet, and isolated-suite guidance without a current retired-product claim.
    requirement: CACH-07
    verification:
      - kind: unit
        ref: uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase9_documentation.py -x
        status: pass
    human_judgment: false
metrics:
  duration: 14 min
  completed: 2026-09-17
status: complete
---

# Phase 10 Plan 08: Bounded Documentation Cutover Summary

**Canonical guidance now routes object/function caching to UnifiedCache over BlobStore and direct persistence to BlobStore, without inventing a SQL pull-through replacement.**

## Performance

- **Duration:** 14 min
- **Completed:** 2026-09-17
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Added the exact bounded cutover note to the API reference, storage migration guide, and documentation index.
- States that Cacheness has no in-package range-aware SQL pull-through replacement; caller-owned SQL tables remain untouched, unsupported, and outside migration/rebuild tooling.
- Removed only the retired SqlCache/test ownership rows from the pandas audit and one stale SQL-cache diagnostic claim from the platform guide, retaining handler, Parquet, and isolated-suite content.

## Task Commits

1. **Task 1: Add the concise canonical cutover notes** - `767cc60` (docs)
2. **Task 2: Surgically clean mixed dataframe and platform guidance** - `26dc238` (docs)

## Files Created/Modified

- `docs/API_REFERENCE.md` - Canonical use-case routing and bounded no-replacement statement.
- `docs/STORAGE_MIGRATION.md` - States that maintenance tooling does not touch caller-owned SQL tables.
- `docs/README.md` - Publishes the same cutover boundary in the documentation index.
- `docs/PANDAS_API_AUDIT.md` - Retains pandas/Parquet handler coverage without retired test or product ownership.
- `docs/CROSS_PLATFORM_GUIDE.md` - Retains isolated-suite explanation without the stale SQL-cache diagnosis.

## Decisions Made

- Kept the concise statement in exactly the three canonical owners protected by the documentation contract.
- Did not create a removal guide, replacement cache API, or caller-database migration/export/cleanup tooling.
- Preserved all dated/history artifacts and retained dataframe/platform material.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Documentation contract] Kept required cutover text as literal searchable phrases**

- **Found during:** Task 1
- **Issue:** Markdown line wrapping split assertions that deliberately inspect exact canonical wording.
- **Fix:** Reflowed the concise note so its required sentences and boundary phrases remain contiguous while preserving the same meaning.
- **Files modified:** `docs/API_REFERENCE.md`, `docs/STORAGE_MIGRATION.md`, `docs/README.md`
- **Verification:** Exact-owner documentation assertions pass.
- **Committed in:** `767cc60`

## Issues Encountered

- The combined Task 1 verification command also runs `tests/test_public_api_contract.py`, which was still red on the four SqlCache-only `CacheReason` values until Plan 10-05 removes them. This was the planned source-cut dependency, not a documentation failure. The canonical-owner selectors and complete documentation suite passed after both documentation tasks.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Current-facing documentation provides the intended bounded product story: UnifiedCache is object/function cache policy over BlobStore; BlobStore is direct persistence; no in-package range-aware SQL pull-through replacement exists.
- Caller SQL tables remain untouched and unsupported. Plan 10-05 must complete source/error removal before the combined public-contract run becomes green.

## Self-Check: PASSED

- Confirmed all five documentation files exist and the mixed-document edits preserve Parquet and isolated-suite guidance.
- Confirmed task commits `767cc60` and `26dc238` exist in Git history.
- Confirmed `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase9_documentation.py -x` passes (10 tests).

---
*Phase: 10-remove-sqlcache-pull-through-subsystem*
*Completed: 2026-09-17*
