---
phase: 32-small-fixes-release-polish
plan: 07
subsystem: package-release
tags: [python, packaging, uv, version, release-polish]

requires:
  - phase: 32-small-fixes-release-polish
    provides: Phase 32 package-surface release polish context
provides:
  - Package metadata version aligned to 0.12.0
  - Runtime cacheness.__version__ aligned to 0.12.0
affects: [package-metadata, runtime-imports, release-polish]

tech-stack:
  added: []
  patterns:
    - Keep pyproject.toml project.version, uv.lock editable package version, and cacheness.__version__ synchronized for releases.

key-files:
  created:
    - .planning/phases/32-small-fixes-release-polish/32-07-SUMMARY.md
  modified:
    - pyproject.toml
    - src/cacheness/__init__.py
    - uv.lock

key-decisions:
  - "Set package metadata and runtime __version__ to 0.12.0 while leaving CHANGELOG.md marked Unreleased."
  - "Included uv.lock because uv refreshed the editable package version during required verification."

patterns-established:
  - "Release metadata alignment includes pyproject.toml, runtime __version__, and lockfile package metadata when uv refreshes it."

requirements-completed: [POL-04]

duration: approx. 30 min
completed: 2026-06-15
---

# Phase 32 Plan 07: Package Version Alignment Summary

**Package release metadata and runtime version now both identify the v0.12.0 line.**

## Performance

- **Duration:** approximately 30 min
- **Started:** 2026-06-15T19:45:00Z
- **Completed:** 2026-06-15T20:16:00Z
- **Tasks:** 1
- **Files modified:** 3

## Accomplishments

- Updated `pyproject.toml` project version from `0.6.0` to `0.12.0`.
- Updated `src/cacheness/__init__.py` `__version__` from `0.6.0` to `0.12.0`.
- Kept `CHANGELOG.md` unchanged; release-date finalization remains out of scope.
- Refreshed `uv.lock` editable package metadata to `0.12.0`.

## Task Commits

1. **Task 32-07-01: Align package version to 0.12.0** - `02299c0` (chore)

**Plan metadata:** this docs commit

## Files Created/Modified

- `pyproject.toml` - Project package metadata version set to `0.12.0`.
- `src/cacheness/__init__.py` - Runtime `cacheness.__version__` set to `0.12.0`.
- `uv.lock` - Editable package lock metadata refreshed to `0.12.0`.
- `.planning/phases/32-small-fixes-release-polish/32-07-SUMMARY.md` - Execution summary for this plan.

## Verification

- Pre-change verify-first command: `uv run python -c "import cacheness; print(cacheness.__version__)"`
  - Initial default uv run failed due `C:\Users\akriz\AppData\Local\uv\cache` filesystem conflict.
  - Re-run with repo-local `UV_CACHE_DIR` / `UV_PYTHON_INSTALL_DIR` and escalation passed, output: `0.6.0`.
- Post-change smoke: `uv run python -c "import pathlib, tomllib, cacheness; data = tomllib.loads(pathlib.Path('pyproject.toml').read_text()); assert data['project']['version'] == cacheness.__version__ == '0.12.0'; print(cacheness.__version__)"`
  - Passed, output: `0.12.0`.
- Quality: `uv run ruff format src/cacheness/__init__.py`
  - Passed, output: `1 file reformatted`.
- Quality: `uv run ruff check --fix src/cacheness/__init__.py`
  - Passed, output: `All checks passed!`.
- Quality: `uv run ruff check src/cacheness/__init__.py`
  - Passed, output: `All checks passed!`.
- Quality: `uv run ty check src/cacheness/__init__.py`
  - Passed, output: `All checks passed!`.

No pytest command was added or run for this version-only smoke plan.

## Decisions Made

- Set the release metadata directly to `0.12.0` because Phase 32 targets the v0.12.0 milestone and `CHANGELOG.md` already has `[0.12.0] - Unreleased`.
- Left `CHANGELOG.md` unchanged because the plan explicitly excluded release-date and release-notes finalization.
- Included `uv.lock` in the implementation commit because uv updated the editable package version during the required verification workflow.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Used repo-local uv environment for verification**
- **Found during:** Task 32-07-01
- **Issue:** The default uv cache path failed with `Cannot create a file when that file already exists`; the sandboxed repo-local run then failed to query the interpreter with access denied.
- **Fix:** Used repo-local `UV_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR`, then escalated the uv verification and quality commands.
- **Files modified:** None for the environment workaround.
- **Verification:** All required uv smoke and quality commands passed.
- **Committed in:** `02299c0` for the version changes; no separate environment files were staged.

**2. [Rule 3 - Blocking] Included uv.lock package version refresh**
- **Found during:** Task 32-07-01
- **Issue:** Required uv verification refreshed the editable package version in `uv.lock` from `0.6.0` to `0.12.0`.
- **Fix:** Included `uv.lock` in the implementation commit so package metadata remains internally consistent.
- **Files modified:** `uv.lock`
- **Verification:** Post-change smoke confirmed pyproject and runtime version both report `0.12.0`.
- **Committed in:** `02299c0`

---

**Total deviations:** 2 auto-fixed blocking issues.
**Impact on plan:** Scope remained release metadata only; no changelog finalization or unrelated dirty files were touched.

## Issues Encountered

- The repository's active `pre-commit` hook is beads-only. The implementation commit used `--no-verify` to comply with the explicit instruction not to run `bd` or beads commands. The plan's required uv quality checks were run manually before commit.
- Existing unrelated dirty files remained untouched: `.planning/config.json` and several test files.

## Known Stubs

None.

## Threat Flags

None - this plan changed package metadata only and introduced no new trust boundary surface beyond the planned version metadata alignment.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Ready for Phase 32 Plan 08. `src/cacheness/__init__.py` now has the required `0.12.0` version outcome that Plan 08 must preserve while adding root import ergonomics.

## Self-Check: PASSED

- Confirmed `.planning/phases/32-small-fixes-release-polish/32-07-SUMMARY.md`, `pyproject.toml`, `src/cacheness/__init__.py`, and `uv.lock` exist.
- Confirmed implementation commit `02299c0` exists in git history.
- Confirmed the implementation commit did not delete tracked files.

---
*Phase: 32-small-fixes-release-polish*
*Completed: 2026-06-15*
