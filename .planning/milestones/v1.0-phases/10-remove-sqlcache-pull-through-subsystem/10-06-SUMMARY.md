---
phase: 10-remove-sqlcache-pull-through-subsystem
plan: 06
subsystem: packaging
tags: [uv, wheel, duckdb-removal, metadata, pytest]
requires:
  - phase: 10-remove-sqlcache-pull-through-subsystem
    provides: Direct SqlCache source removal and a digest-bound source-free wheel harness
provides:
  - Exact DuckDB and uv sql-group dependency removal
  - Fresh-wheel metadata proof for the six retained installable extras
affects: [phase-10-cutover, release-qualification, packaging]
actuals:
  tokens: 3103
  tasks: 2
  commits: 3
tech-stack:
  added: []
  patterns:
    - Authoritative manifest changes followed by plain uv lock convergence
    - Source-free wheel probes validate both retired and retained installed metadata
key-files:
  created:
    - .planning/phases/10-remove-sqlcache-pull-through-subsystem/10-06-SUMMARY.md
  modified:
    - pyproject.toml
    - uv.lock
    - tools/run_phase8_packaging.py
    - tests/packaging/test_wheel_matrix.py
key-decisions:
  - "Remove only DuckDB and duckdb-engine plus the retired uv sql group; retain SQLAlchemy, pandas, PyArrow, psycopg, and all six advertised extras."
  - "Require installed-wheel extras to match the reviewed inventory exactly, not merely omit the retired sql extra."
patterns-established:
  - "Artifact qualification checks absence and exact preservation in the installed distribution metadata."
requirements-completed: [CACH-07]
coverage:
  - id: D1
    description: "The authoritative manifest and derived lock omit DuckDB, duckdb-engine, and the retired uv sql group while preserving retained dependency owners and package version."
    requirement: CACH-07
    verification:
      - kind: unit
        ref: "tests/test_phase10_sqlcache_removal.py::test_manifest_lock_dependency_contract"
        status: pass
      - kind: other
        ref: "uv lock --check"
        status: pass
    human_judgment: false
  - id: D2
    description: "A freshly built source-free wheel omits the retired module and metadata while retaining the exact six approved extras and local BlobStore/UnifiedCache round trips."
    requirement: CACH-07
    verification:
      - kind: integration
        ref: "tests/packaging/test_wheel_matrix.py"
        status: pass
      - kind: other
        ref: "uv run --isolated --group dev --frozen ruff check tools/run_phase8_packaging.py tests/packaging/test_wheel_matrix.py"
        status: pass
    human_judgment: false
duration: 5 min
completed: 2026-09-17
status: complete
---

# Phase 10 Plan 06: Dependency and Fresh-Wheel Cutover Summary

**The release manifest and isolated wheel now remove DuckDB and SQL pull-through metadata while retaining the complete BlobStore/UnifiedCache packaging surface.**

## Performance

- **Duration:** 5 min
- **Started:** 2026-09-17T17:41:33Z
- **Completed:** 2026-09-17T17:47:09Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Removed the DuckDB engine from both supported `recommended` declarations and deleted the retired uv `sql` dependency group; a plain `uv lock` removed only `duckdb-engine` and transitive `duckdb`.
- Retained the exact six installable extras plus SQLAlchemy, pandas, PyArrow, and psycopg ownership for supported lifecycle, PostgreSQL, and dataframe paths.
- Extended the digest-bound fresh-wheel probe so its installed distribution must expose exactly the retained extras, alongside existing retired-member/import/DuckDB checks and local BlobStore/UnifiedCache round trips.

## Task Commits

1. **Task 1: Remove the exact DuckDB and sql-group dependency surface** - `e7603f6` (chore)
2. **Task 2: Run the fresh source-free wheel cutover proof** - `772b25f` (test), `937db61` (feat)

## Files Created/Modified

- `pyproject.toml` - Removes DuckDB from the two recommended declarations and the obsolete uv-only `sql` group.
- `uv.lock` - Plain-lock convergence removes DuckDB package records and every direct group/metadata edge.
- `tools/run_phase8_packaging.py` - Requires installed wheel extras to equal the fixed six-extra inventory.
- `tests/packaging/test_wheel_matrix.py` - Adds a red/green regression for unexpected installed optional-extra metadata.

## Decisions Made

- Removed only the dependencies proven exclusive to the deleted product: `duckdb-engine`, transitive `duckdb`, and the uv `sql` group.
- Made retained-extra preservation a distribution-level invariant, so an accidental extra loss or addition cannot pass an absence-only wheel check.

## Verification

- `uv lock --check` — pass
- `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase10_sqlcache_removal.py::test_manifest_lock_dependency_contract tests/test_phase10_sqlcache_removal.py::test_package_version_remains_unchanged -x` — 2 passed
- `uv run --isolated --all-extras --group dev --frozen pytest -q tests/packaging/test_wheel_matrix.py -x` — 13 passed; includes fresh source-free wheel construction, ZIP inspection, installed metadata checks, and local round trips
- `uv run --isolated --group dev --frozen ruff check tools/run_phase8_packaging.py tests/packaging/test_wheel_matrix.py` — pass
- Frozen inverse ownership confirms SQLAlchemy remains owned by `recommended`, `postgresql`, and `cloud`; pandas/PyArrow remain owned by `recommended` and `dataframes`; psycopg remains owned by `postgresql` and `cloud`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Proved exact retained extras in installed wheel metadata**
- **Found during:** Task 2 (Run the fresh source-free wheel cutover proof)
- **Issue:** The initial probe rejected retired DuckDB and `sql` metadata but did not detect a missing or unexpected retained installable extra in the built distribution.
- **Fix:** Added a red/green test and required installed `Provides-Extra` metadata to exactly match the reviewed six-extra inventory.
- **Files modified:** `tools/run_phase8_packaging.py`, `tests/packaging/test_wheel_matrix.py`
- **Verification:** All 13 wheel qualification tests and scoped Ruff pass.
- **Committed in:** `772b25f`, `937db61`

---

**Total deviations:** 1 auto-fixed (1 Rule 2 missing critical packaging assertion).
**Impact on plan:** The new assertion enforces D-07/D-16 preservation without adding a package, product surface, or lifecycle behavior.

## Issues Encountered

None. The first direct dependency-tree read needed the existing user-scoped uv cache, then completed normally with approved environment access.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- The source, manifest, lock, and fresh artifact agree that the retired SQL pull-through dependency surface is absent.
- Phase-level closure can reuse the fresh wheel test as a distribution proof while retaining its explicit non-live remote-service boundary.

## Self-Check: PASSED

- Confirmed `pyproject.toml`, `uv.lock`, wheel runner, test contract, and this summary exist.
- Confirmed commits `e7603f6`, `772b25f`, and `937db61` exist in Git history.
- Confirmed the task commits contain no tracked-file deletion and leave user-owned `.claude/`, `.planning/config.json`, and `.planning/milestone.lock` untouched.

---
*Phase: 10-remove-sqlcache-pull-through-subsystem*
*Completed: 2026-09-17*
