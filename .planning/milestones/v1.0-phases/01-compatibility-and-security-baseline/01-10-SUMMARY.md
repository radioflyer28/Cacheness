---
phase: 01-compatibility-and-security-baseline
plan: "10"
subsystem: compatibility-fixtures
tags: [compatibility, fixtures, sqlite, json, npz, decorators, xxh3, provenance]
requires:
  - phase: 01-08
    provides: Independent staged compatibility-fixture validator and raw-array corpus prefix
  - phase: 01-09
    provides: Four-record JSON/NPZ corpus prefix with immutable provenance conventions
provides:
  - Immutable 0.3.9 SQLite metadata_json and numeric NPZ compatibility evidence
  - Immutable 0.3.13 pre-unified decorator-key JSON/NPZ compatibility evidence
  - Six-record corpus prefix with non-enumerating decorator candidate validation
affects: [01-11, 01-12, stored-compatibility, migration-adapters]
actuals:
  tokens: 3136
  tasks: 2
  commits: 2
tech-stack:
  added: []
  patterns:
    - Generate historical fixture evidence in disposable detached worktrees using writer-only paths
    - Recompute one recorded legacy key candidate and direct-lookup its derived entry rather than enumerate metadata
key-files:
  created:
    - tests/fixtures/compat/sqlite-metadata-json-v039/metadata.sqlite3
    - tests/fixtures/compat/sqlite-metadata-json-v039/payload.npz
    - tests/fixtures/compat/sqlite-metadata-json-v039/provenance.json
    - tests/fixtures/compat/decorator-key-v0313/metadata.json
    - tests/fixtures/compat/decorator-key-v0313/payload.npz
    - tests/fixtures/compat/decorator-key-v0313/provenance.json
  modified:
    - tests/fixtures/compat/manifest.json
    - tests/fixtures/compat/validate_corpus.py
key-decisions:
  - The legacy SQLite fixture records only read-only schema/data-version inspection and source/copy digests before evidence is committed.
  - The pre-unified decorator fixture records one stable function/call tuple, historical serialization strings, old XXH3-64 candidate, and derived storage key.
  - The independent validator directly recomputes and looks up that sole candidate; it never searches metadata for a fallback.
patterns-established:
  - Historical fixture provenance remains constrained to a stable base schema, with a fixture-specific extension only where a non-enumerating key proof is required.
  - Each fixture task appends one manifest record and validates the entire accumulated prefix before committing.
requirements-completed: [MIGR-01]
coverage:
  - id: D1
    description: The 0.3.9 SQLite metadata_json fixture has the exact legacy schema, fixed numeric NPZ semantics, source/copy digests, and read-only inspection invariance.
    requirement: MIGR-01
    verification:
      - kind: unit
        ref: uv run python tests/fixtures/compat/validate_corpus.py --expected-through sqlite-metadata-json-v039
        status: pass
    human_judgment: false
  - id: D2
    description: The 0.3.13 decorator fixture validates one exact historical candidate and derived entry key without scanning metadata, while the complete six-record corpus remains immutable.
    requirement: MIGR-01
    verification:
      - kind: unit
        ref: uv run python tests/fixtures/compat/validate_corpus.py --expected-through decorator-key-v0313
        status: pass
    human_judgment: false
duration: 5min
completed: 2026-08-29
status: complete
---

# Phase 01 Plan 10: SQLite and Decorator Compatibility Fixtures Summary

**Pinned 0.3.9 SQLite metadata_json and 0.3.13 decorator-key fixtures now extend the immutable corpus through a six-record, independently validated prefix.**

## Performance

- **Duration:** 5min
- **Started:** 2026-08-29T21:57:45Z
- **Completed:** 2026-08-29T22:02:48Z
- **Tasks:** 2/2
- **Files modified:** 8

## Accomplishments

- Generated 0.3.9 SQLite metadata_json evidence from the exact historical writer, preserving the full legacy column sequence, a read-only `PRAGMA data_version` value of `2`, and equal source/copy SHA-256 values.
- Generated the pre-unified 0.3.13 decorator evidence from one stable function call, recording its module, qualname, positional and keyword inputs, prefix, historical serialized arguments, XXH3 candidate, and derived entry key.
- Appended the fifth and sixth normative fixture records and revalidated source/copy digests, safe file lists, fixed numeric NPZ semantics, SQLite read-only invariance, and provenance schemas.
- Added the missing fixture-specific validator check that recomputes exactly one decorator candidate and directly looks up the derived entry without enumerating metadata.

## Task Commits

Each fixture task was committed atomically:

1. **Task 1: Generate the exact metadata_json SQLite fixture** - `0653d07` (`feat`)
2. **Task 2: Generate the exact pre-unified decorator-key fixture** - `b8da058` (`feat`)

## Files Created/Modified

- `tests/fixtures/compat/sqlite-metadata-json-v039/` - 0.3.9 SQLite metadata, numeric NPZ payload, and immutable provenance.
- `tests/fixtures/compat/decorator-key-v0313/` - 0.3.13 nested JSON metadata, numeric NPZ payload, and exact decorator-key provenance.
- `tests/fixtures/compat/manifest.json` - insertion-ordered six-record corpus index.
- `tests/fixtures/compat/validate_corpus.py` - fixture-specific direct recomputation and lookup of the sole allowed decorator fallback candidate.

## Decisions Made

- Inspected SQLite evidence only via a `mode=ro` URI and captured the data-version invariant in provenance rather than opening the historical database through Cacheness.
- Chose `compat_fixture_v0313.fixture_array(6, offset=0)` as the single stable decorator input; its candidate is a capability-like exact lookup input, not a basis for scanning arbitrary metadata.
- Preserved the generic provenance contract for all existing fixtures and extended it only for the decorator fixture’s required exact key material.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical Functionality] Added independent decorator candidate verification**
- **Found during:** Task 2 (Generate the exact pre-unified decorator-key fixture)
- **Issue:** The Plan 08 validator confirmed structural JSON/NPZ provenance but did not recompute the 0.3.13 decorator candidate or prove that only its direct entry lookup succeeds; this left the plan's non-enumeration acceptance requirement unenforced.
- **Fix:** Added a fixture-specific provenance extension and validator routine that reconstructs the historical serialization/base string, verifies the XXH3-64 candidate and derived cache-entry key, and directly checks that one entry's description.
- **Files modified:** `tests/fixtures/compat/validate_corpus.py`, `tests/fixtures/compat/decorator-key-v0313/provenance.json`
- **Verification:** `uv run python tests/fixtures/compat/validate_corpus.py --expected-through decorator-key-v0313`
- **Committed in:** `b8da058` (part of Task 2)

---

**Total deviations:** 1 auto-fixed (Rule 2 - missing critical functionality)
**Impact on plan:** The correction enforces the stated one-candidate safety contract without production changes, arbitrary metadata enumeration, or a new storage format.

## Issues Encountered

- The pinned historical package manifest omits NumPy, so an isolated historical environment could not execute the writer. The writer instead used the repository's existing locked test environment with the detached historical `src/` first on `PYTHONPATH`; no package was added or substituted.
- Targeted Ruff reports one pre-existing unused `os` import in `validate_corpus.py`. The fixture-specific validator change adds no lint finding; the baseline import remains out of scope.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 01-11 can append the current-family fixture records while retaining the same exact-prefix and immutable-provenance protocol.
- Stored-compatibility adapters can use the 0.3.9 schema discriminator and 0.3.13 single-candidate decorator evidence without release guessing or metadata scans.

## Self-Check: PASSED

- All six fixture evidence files, the manifest, validator, and this summary exist.
- Both fixture commits (`0653d07` and `b8da058`) are present in git history.
- The fixture/provenance/validator stub scan found no placeholders, skipped tests, or unrun required verification.

---
*Phase: 01-compatibility-and-security-baseline*
*Completed: 2026-08-29*
