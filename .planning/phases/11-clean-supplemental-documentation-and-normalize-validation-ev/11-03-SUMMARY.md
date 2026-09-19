---
phase: 11
plan: 03
subsystem: package-runtime
tags: [tensorflow-removal, handlers, packaging, uv-lock, wheel-qualification]
requires:
  - phase: 11-01
    provides: "Digest-bound TensorFlow-removal wheel and retained-round-trip contracts"
  - phase: 11-02
    provides: "Five-extra release fixture and core-only qualification contracts"
provides:
  - "TensorFlow-free handler, configuration, compatibility-export, and dedicated-test surface"
  - "Five-extra package manifest, generated lock, and packaging-evidence inventory"
affects:
  - "Phase 11 documentation, CI/profile, validation, and final acceptance plans"
tech-stack:
  added: []
  patterns:
    - "Direct pre-production surface removal without compatibility aliases or tombstones"
    - "Literal retained-extra inventories enforced in source-free wheel probes and evidence parsing"
key-files:
  created: []
  modified:
    - src/cacheness/handlers.py
    - src/cacheness/config.py
    - src/cacheness/storage/handlers/__init__.py
    - pyproject.toml
    - uv.lock
    - tools/run_phase8_packaging.py
    - tools/phase8_evidence.py
    - tests/packaging/test_wheel_matrix.py
    - tests/test_phase10_sqlcache_removal.py
  deleted:
    - tests/test_tensorflow_handler.py
decisions:
  - "Removed the dormant TensorFlow handler and persisted identity directly, leaving unsupported identities to ordinary registry lookup failure."
  - "Kept exactly recommended, dataframes, s3, postgresql, and cloud as published extras; dev remains a local dependency group."
  - "Preserved one digest-bound WheelArtifact and the retained BlobStore/UnifiedCache local probes rather than adding a cutover-specific harness."
metrics:
  duration: 15m 2s
  completed: 2026-09-19
status: complete
actuals:
  tokens: 31662
  tasks: 2
  commits: 2
requirements-completed: [D-05, D-08, D-18]
coverage:
  - id: D1
    description: "The retired TensorFlow handler, configuration, export, dedicated test, package extras, resolver graph, and qualification profile paths are absent."
    requirement: D-05
    verification:
      - kind: unit
        ref: "tests/test_handler_registration.py; tests/test_config_validation.py; tests/packaging/test_wheel_matrix.py::test_tensorflow_surface_is_absent_from_built_wheel_and_metadata"
        status: pass
    human_judgment: false
  - id: D2
    description: "Retained handler boundaries, installed-wheel metadata, BlobStore, and UnifiedCache local round trips remain intact after the package cutover."
    requirement: D-08
    verification:
      - kind: integration
        ref: "tests/packaging/test_wheel_matrix.py; tests/qualification/test_phase8_release.py"
        status: pass
    human_judgment: false
---

# Phase 11 Plan 03: TensorFlow Handler and Package Cutover Summary

**Directly removed the dormant TensorFlow handler and its five-extra package surface while preserving retained handler registry and installed BlobStore/UnifiedCache journeys.**

## Performance

- **Duration:** 15m 2s
- **Started:** 2026-09-19T17:40:06Z
- **Completed:** 2026-09-19T17:55:08Z
- **Tasks:** 2/2
- **Files modified:** 10

## Accomplishments

- Deleted the TensorFlow lazy loader, handler, registry identity, configuration flag, compatibility export, and dedicated skipped test without retaining an alias, tombstone, or re-enable path.
- Pruned the TensorFlow published/local dependency groups, converged `uv.lock` using plain `uv lock`, and removed the corresponding package-probe compatibility branches.
- Kept the packaging evidence parser, one-artifact source-free wheel contract, and retained BlobStore/UnifiedCache local round trips aligned to the literal five-extra inventory.

## Task Commits

Each task was committed atomically:

1. **Task 1: Physically remove the runtime, config, export, and dedicated-test surface** — `5c0b6b4` (`feat`)
2. **Task 2: Converge manifest, lockfile, and package evidence to the retained five-extra surface** — `99ca0e8` (`feat`)

## Files Created/Modified

- `src/cacheness/handlers.py` — removes the dormant TensorFlow implementation and all registry reachability while retaining every supported handler path.
- `src/cacheness/config.py` — removes the retired handler flag and priority identity.
- `src/cacheness/storage/handlers/__init__.py` — removes the retired compatibility export probe and public symbol.
- `tests/test_tensorflow_handler.py` — deleted retired dedicated test module.
- `pyproject.toml` and `uv.lock` — define and resolve the retained five-extra package surface.
- `tools/run_phase8_packaging.py` and `tools/phase8_evidence.py` — qualify and validate only the retained package groups.
- `tests/packaging/test_wheel_matrix.py` and `tests/test_phase10_sqlcache_removal.py` — retain executable source-free cutover and current-reference contracts.

## Decisions Made

- Removed unsupported TensorFlow persisted-format identities directly; an unsupported identity now follows the existing ordinary handler-registry failure path.
- Used the manifest-first, plain-`uv lock` workflow and accepted its removal-only resolution rather than selectively hand-pruning generated lock entries.
- Kept the existing digest-bound wheel harness as the single installed-package authority.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Created the source-free wheel probe workspace before subprocess execution.**

- **Found during:** Task 2
- **Issue:** The TensorFlow-absence wheel test used a non-existent temporary directory as its subprocess `cwd`, failing before it could inspect the installed wheel.
- **Fix:** Created the private probe directory and passed that existing path to the subprocess.
- **Files modified:** `tests/packaging/test_wheel_matrix.py`
- **Verification:** The targeted source-free TensorFlow-absence wheel contract passes.
- **Committed in:** `99ca0e8`

**2. [Rule 1 - Bug] Corrected the Phase 10 retired-reference literal count.**

- **Found during:** Task 2
- **Issue:** The exact current-surface allowlist expected twelve `duckdb` literals in its own test module, while the reviewed source contains eleven.
- **Fix:** Updated the intentional inventory count to eleven without changing the SqlCache removal contract.
- **Files modified:** `tests/test_phase10_sqlcache_removal.py`
- **Verification:** The full package, Phase 10, and release-contract cluster passes.
- **Committed in:** `99ca0e8`

---

**Total deviations:** 2 auto-fixed (2 Rule 1 bugs)
**Impact on plan:** Both fixes made existing contract tests exercise the intended package surface; neither changed runtime behavior or expanded scope.

## Issues Encountered

- Scoped Ruff on Task 1 reported three pre-existing `F401` findings for the retained optional pandas/Polars re-export imports in `src/cacheness/storage/handlers/__init__.py`. The TensorFlow-removal diff did not touch those imports; the deferred note is recorded in `deferred-items.md`.

## Verification

- PASS — `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_handler_registration.py tests/test_config_validation.py -x` (17 passed).
- PASS — `uv lock --check` after plain lock convergence.
- PASS — `uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false tests/packaging/test_wheel_matrix.py tests/test_phase10_sqlcache_removal.py tests/qualification/test_phase8_release.py -x` (59 passed).
- PASS — scoped Ruff for all Task 2 Python changes; Task 1 removal introduced no new Ruff finding.
- PASS — direct manifest checks confirm published extras are exactly `recommended`, `dataframes`, `s3`, `postgresql`, and `cloud`, with `recommended`, `dataframes`, and `dev` as local dependency groups.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- The direct runtime and package cutover is complete for Plans 11-04 onward to remove the remaining current documentation, CI/profile, validation, and audit references.
- The retained handler registry, private staging/snapshot contract, `BlobStore`, `UnifiedCache`, lifecycle authorities, and topology behavior remain untouched.

## Self-Check: PASSED

- Found all nine retained modified files and the plan summary on disk; confirmed `tests/test_tensorflow_handler.py` is deleted.
- Found Task 1 commit `5c0b6b4` and Task 2 commit `99ca0e8` in Git history.

---
*Phase: 11-clean-supplemental-documentation-and-normalize-validation-ev*
*Completed: 2026-09-19*
