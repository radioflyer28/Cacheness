---
phase: 01-compatibility-and-security-baseline
plan: "12"
subsystem: stored-compatibility
tags: [compatibility, metadata, hmac, decorators, json, sqlite, numpy]
requires:
  - phase: 01-03
    provides: Guarded private snapshots and pre-deserialization authorization ordering
  - phase: 01-04
    provides: Native pickle-disabled array reads and bounded legacy-array compatibility
  - phase: 01-11
    provides: Complete immutable eight-record compatibility corpus
provides:
  - Exact read-only production adapters for the supported split-map JSON and metadata_json SQLite layouts
  - Current-first plus exact six-field historical HMAC verification on one guarded snapshot
  - A one-candidate pre-unified decorator lookup with no metadata enumeration
affects: [phase-02, migration-inputs, metadata-backends, UnifiedCache, decorators]
actuals:
  tokens: 23198.5
  tasks: 2
  commits: 5
tech-stack:
  added: []
  patterns:
    - Historical schemas must match a complete discriminator before read-only normalization.
    - Compatibility signature checks use the live guarded snapshot and never authorize an invalid current signature as unsigned.
    - Historical decorator lookup is one computed candidate after a current-key probe, never a metadata scan.
key-files:
  created:
    - tests/test_stored_compatibility.py
  modified:
    - src/cacheness/metadata.py
    - src/cacheness/security.py
    - src/cacheness/core.py
    - src/cacheness/decorators.py
key-decisions:
  - Exact recognized legacy metadata layouts remain read-only and record successful reads only in process-local compatibility counters.
  - A current signature is checked before the only supported historical six-field HMAC candidate; all other failures remain fail-closed.
  - The 0.3.13 decorator adapter derives one documented historical key after a current miss and rebases its payload only in an in-memory guarded-read entry.
patterns-established:
  - Copy immutable fixture evidence, consume it through production entry points, and prove source/copy bytes plus SQLite observations stay unchanged.
  - Suppress only the exact post-success legacy read-only bookkeeping outcome; propagate every other typed or operational failure.
requirements-completed: [MIGR-01]
coverage:
  - id: D1
    description: Exact legacy JSON and SQLite metadata layouts normalize through production backends as deprecated read-only entries without changing evidence.
    requirement: MIGR-01
    verification:
      - kind: integration
        ref: tests/test_stored_compatibility.py
        status: pass
    human_judgment: false
  - id: D2
    description: Signed 0.3.8 data verifies current-first and then against one constant-time six-field candidate on the same guarded snapshot.
    requirement: MIGR-01
    verification:
      - kind: integration
        ref: tests/test_stored_compatibility.py#test_signed_split_map_verifies_current_then_exact_legacy_on_one_snapshot
        status: pass
    human_judgment: false
  - id: D3
    description: Pre-unified 0.3.13 decorator data is recovered only through one historical key after current-key miss, with no metadata enumeration.
    requirement: MIGR-01
    verification:
      - kind: integration
        ref: tests/test_stored_compatibility.py#test_decorator_tries_one_historical_candidate_after_current_miss_without_scan
        status: pass
    human_judgment: false
duration: 40min
completed: 2026-08-29
status: complete
---

# Phase 01 Plan 12: Production Stored-Compatibility Adapters Summary

**All eight immutable compatibility fixtures now read through bounded production adapters, with legacy metadata kept read-only, signatures verified before deserialization, and decorators limited to one historical key fallback.**

## Performance

- **Duration:** 40min
- **Started:** 2026-08-29T22:27:35Z
- **Completed:** 2026-08-29T23:07:10Z
- **Tasks:** 2/2
- **Files modified:** 5

## Accomplishments

- Normalized only the complete 0.3.7/0.3.8 split-map JSON and 0.3.9 metadata_json SQLite layouts through real read-only metadata backends, retaining fixture bytes, rows, schema, and data-version evidence.
- Added current-first and exact six-field 0.3.8 HMAC verification on the same guarded snapshot, returning a typed `invalid_legacy_signature` before handler access or evidence deletion when the key is wrong.
- Recovered the documented 0.3.13 decorator key through one derived candidate after a current-key probe; legacy hits warn and neither enumerate metadata nor persist legacy bookkeeping.
- Proved all eight records through the independent corpus validator and production public flows, including raw-array, JSON, SQLite, signed, decorator, and current-control paths.

## Task Commits

Each TDD task was committed atomically:

1. **Task 1: Normalize exact historical JSON and SQLite layouts read-only**
   - `a1dedbc` `test(01-12): specify legacy metadata compatibility`
   - `b109517` `feat(01-12): add read-only metadata compatibility`
2. **Task 2: Add exact signature/decorator adapters and prove all eight production reads**
   - `2b64832` `test(01-12): specify signature and decorator compatibility`
   - `cbf471b` `feat(01-12): add production signature and decorator adapters`
3. **Wave 6 integration correction: Reject unrecognized legacy-signature metadata**
   - `ff4b84f` `fix(01-12): reject unrecognized legacy signatures`

## Files Created/Modified

- `src/cacheness/metadata.py` - exact split-map and metadata_json detection, normalization, read-only mutation rejection, and process-local legacy bookkeeping.
- `src/cacheness/security.py` - constant-time verification for the one historical six-field signature payload.
- `src/cacheness/core.py` - current-first signature dispatch, typed legacy failure propagation, and guarded legacy decorator read support.
- `src/cacheness/decorators.py` - exact 0.3.13 candidate-key fallback after current-key miss without scanning.
- `tests/test_stored_compatibility.py` - copied-fixture semantic, ordering, reason-selectivity, invariance, and current-control coverage.

## Decisions Made

- Retained compatibility only for fully identified layouts and one documented historical key/signature candidate; unknown variants do not receive heuristic parsing or lookup.
- Kept legacy success bookkeeping out of persisted evidence and suppressed only the exact post-success `read_only_legacy_store` condition.
- Rebased the pre-unified decorator payload only in the private fallback entry, keeping normal persisted locator containment strict.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Test setup] Deferred decorator-fixture cleanup to the bounded legacy fallback**
- **Found during:** Task 2 review
- **Issue:** The fixture's historical absolute payload locator correctly fails generic startup preflight before the exact decorator-key fallback can construct its in-memory guarded entry.
- **Fix:** Disabled startup cleanup only in that copied decorator-fixture test; the private fallback alone rebases the exact payload filename under the configured root before guarded access.
- **Files modified:** `tests/test_stored_compatibility.py`, `src/cacheness/core.py`
- **Verification:** Full corpus validator and required production compatibility suite passed with no metadata mutation.
- **Committed in:** `cbf471b`

**2. [Rule 1 - Security] Preserved fail-closed invalid current signatures**
- **Found during:** Task 2 final review
- **Issue:** Current-first fallback control flow could otherwise have treated an invalid current signature as unsigned when unsigned entries were permitted.
- **Fix:** Return failure immediately for an invalid current signature unless the entry carries the exact signed split-map compatibility discriminator.
- **Files modified:** `src/cacheness/core.py`
- **Verification:** Required signature ordering and wrong-key tests passed for both invalid-signature deletion settings.
- **Committed in:** `cbf471b`

**3. [Rule 2 - Quality gate] Made the phase-owned compatibility test Ruff-clean**
- **Found during:** Task 2 review
- **Issue:** Two assigned lambda helpers in the phase-created test violated the Phase 01-07 zero-findings requirement.
- **Fix:** Replaced them with named local invariance helpers without changing behavior.
- **Files modified:** `tests/test_stored_compatibility.py`
- **Verification:** `uv run ruff check tests/test_stored_compatibility.py` passed.
- **Committed in:** `cbf471b`

**4. [Rule 1 - Security] Rejected unknown legacy-signature-shaped metadata**
- **Found during:** Wave 6 full-suite integration gate
- **Issue:** An entry with `legacy_entry_signature` but no exact compatibility discriminator could fall through to the unsigned-entry policy.
- **Fix:** Reject every unrecognized legacy-signature-shaped entry before current-signature or unsigned authorization and add a no-handler regression.
- **Files modified:** `src/cacheness/core.py`, `tests/test_stored_compatibility.py`
- **Verification:** The reported containment regression, focused signature tests, corpus validator, required Plan 12 suite, and phase-owned Ruff check passed.
- **Committed in:** `ff4b84f`

---

**Total deviations:** 4 auto-fixed (3 Rule 1, 1 Rule 2)
**Impact on plan:** The corrections preserve strict locator/signature security and phase-owned test quality without broadening supported formats or storage behavior.

## Issues Encountered

None - no unresolved blockers or unrun required verification remain.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 2 can consume exact normalized legacy metadata as migration input while retaining immutable fixture evidence.
- Future compatibility work must keep the one-layout/one-candidate model and preserve current-first authentication on one guarded snapshot.

## Self-Check: PASSED

- `01-12-SUMMARY.md` exists at the phase output path.
- All five Plan 01-12 commits (`a1dedbc`, `b109517`, `2b64832`, `cbf471b`, and `ff4b84f`) are present in git history.

---
*Phase: 01-compatibility-and-security-baseline*
*Completed: 2026-08-29*
