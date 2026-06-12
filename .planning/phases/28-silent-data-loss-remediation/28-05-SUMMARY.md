---
phase: 28-silent-data-loss-remediation
plan: 05
subsystem: serialization
tags: [cache-key, serialization, pythonhashseed, hypothesis, compatibility]

requires:
  - phase: 28-04
    provides: JSON metadata reliability fixes
provides:
  - Stable large tuple cache-key fallback
  - Stable default repr fallback marker
  - Cross-PYTHONHASHSEED cache-key regression
  - Property-based large tuple determinism coverage
affects: [serialization, decorators, cache-keys, KEY-01, KEY-02]

tech-stack:
  added: []
  patterns: [deterministic fallback serialization, logged low-quality key material]

key-files:
  created:
    - .planning/phases/28-silent-data-loss-remediation/28-05-SUMMARY.md
  modified:
    - src/cacheness/serialization.py
    - tests/test_serialization.py
    - tests/test_cache_key_consistency.py
    - tests/test_property_based.py
    - tests/test_configurable_serialization.py
    - CHANGELOG.md

key-decisions:
  - "Source of truth was docs/CODE_REVIEW_ACTIONS.md TASK-4 and docs/CODE_REVIEW_FINDINGS.md U1."
  - "Large tuples now hash deterministic recursive element serialization instead of Python hash()."
  - "Default memory-address repr fallback is replaced with a stable low-quality marker and warning."

patterns-established:
  - "Persistent cache-key material must not include Python process-randomized hash() output or raw memory-address repr strings."

requirements-completed: [KEY-01, KEY-02]

duration: 50min
completed: 2026-06-12
---

# Phase 28 Plan 05 Summary

**Persistent cache keys are stable across PYTHONHASHSEED for large tuples and unstable fallback objects.**

## Performance

- **Duration:** 50 min
- **Started:** 2026-06-12T16:20:00-04:00
- **Completed:** 2026-06-12T17:10:11-04:00
- **Tasks:** 5
- **Files modified:** 7

## Accomplishments

- Replaced large tuple fallthrough-to-`hash()` with deterministic recursive serialization plus an `xxh3_64` digest marker.
- Restricted hash fallback to stable scalar types and Enums; arbitrary hashable objects warn once and fall through.
- Replaced default memory-address repr strings with a stable `{Type}:unstable_repr` marker and warning.
- Added a subprocess regression across two `PYTHONHASHSEED` values and a bounded Hypothesis large-tuple determinism property.
- Added the required v0.12.0 compatibility note to `CHANGELOG.md`.

## Task Commits

1. **Task 5: Stabilize cache-key fallback serialization** - this commit

## Files Created/Modified

- `src/cacheness/serialization.py` - Stabilizes tuple/hash/string fallback key material.
- `tests/test_serialization.py` - Updates marker expectations and no-dict object fallback coverage.
- `tests/test_cache_key_consistency.py` - Adds cross-subprocess `PYTHONHASHSEED` regression.
- `tests/test_property_based.py` - Adds bounded large tuple determinism property.
- `tests/test_configurable_serialization.py` - Updates disabled-basic-types expectation for stable string fallback.
- `CHANGELOG.md` - Documents compatibility impact for affected cache keys.
- `.planning/phases/28-silent-data-loss-remediation/28-05-SUMMARY.md` - Captures execution evidence and decisions.

## Decisions Made

- Large tuples use `tuple_hashed:{len}:{digest}` so the marker is explicit and no longer confused with Python `hash()`.
- Ordinary Python exceptions in warning paths are not raised; the code logs once per type to avoid noisy repeated warnings.

## Deviations from Plan

### Auto-fixed Issues

**1. Verify-first subprocess check was not run before editing**
- **Found during:** Task 1 execution
- **Issue:** I began the implementation before running the old cross-process repro.
- **Fix:** Added and ran the cross-subprocess regression after implementation; documented this miss explicitly rather than treating it as completed verify-first evidence.
- **Files modified:** `tests/test_cache_key_consistency.py`, this summary
- **Verification:** Plan 05 Tier-1 file set passes.
- **Committed in:** this commit

---

**Total deviations:** 1 process deviation
**Impact on plan:** Implementation is verified by regression tests, but the strict verify-first order was missed for this slice.

## Issues Encountered

- `ty check src/cacheness/serialization.py tests/test_serialization.py tests/test_cache_key_consistency.py tests/test_property_based.py` reports existing test typing diagnostics around `pytest.skip`/`pytest.fail`, optional `polars`, and handler metadata types.

## Verification

- Plan 05 Tier-1 file set: `155 passed, 1 warning`
- `ruff format src/cacheness/serialization.py tests/test_serialization.py tests/test_cache_key_consistency.py tests/test_property_based.py`: passed, reformatted files
- `ruff check --fix src/cacheness/serialization.py tests/test_serialization.py tests/test_cache_key_consistency.py tests/test_property_based.py`: passed
- `ruff check src/cacheness/serialization.py tests/test_serialization.py tests/test_cache_key_consistency.py tests/test_property_based.py`: passed
- `ty check src/cacheness/serialization.py tests/test_serialization.py tests/test_cache_key_consistency.py tests/test_property_based.py`: failed on existing diagnostics noted above
- `tests/test_configurable_serialization.py`: `15 passed`
- Full suite: `1780 passed, 130 skipped, 14 warnings`

## User Setup Required

None.

## Next Phase Readiness

KEY-01 and KEY-02 are addressed. All five Phase 28 plans now have implementation summaries and atomic commits.

---
*Phase: 28-silent-data-loss-remediation*
*Completed: 2026-06-12*
