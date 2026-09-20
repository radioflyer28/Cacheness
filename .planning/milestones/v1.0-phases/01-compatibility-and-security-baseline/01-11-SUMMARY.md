---
phase: 01-compatibility-and-security-baseline
plan: "11"
subsystem: compatibility-fixtures
tags: [compatibility, fixtures, json, sqlite, npz, provenance, xxh3]
requires:
  - phase: 01-10
    provides: Six-record immutable corpus prefix and independent fixture-validator conventions
provides:
  - Immutable 0.3.14 nested-JSON control with a directly validated unified cache key
  - Immutable 0.3.14 denormalized-SQLite control with read-only schema and data-version evidence
  - Complete eight-record compatibility corpus ready for production adapter coverage
affects: [01-12, stored-compatibility, migration-adapters]
actuals:
  tokens: 4604
  tasks: 2
  commits: 2
tech-stack:
  added: []
  patterns:
    - Generate exact-source fixture evidence in disposable detached worktrees through writer-only paths
    - Validate one declared unified key directly and inspect SQLite only through a copied read-only URI
key-files:
  created:
    - tests/fixtures/compat/json-nested-v0314/metadata.json
    - tests/fixtures/compat/json-nested-v0314/payload.npz
    - tests/fixtures/compat/json-nested-v0314/provenance.json
    - tests/fixtures/compat/sqlite-columns-v0314/metadata.sqlite3
    - tests/fixtures/compat/sqlite-columns-v0314/payload.npz
    - tests/fixtures/compat/sqlite-columns-v0314/provenance.json
  modified:
    - tests/fixtures/compat/manifest.json
    - tests/fixtures/compat/validate_corpus.py
key-decisions:
  - The current JSON control records and independently recomputes one declared unified cache key without metadata enumeration.
  - The current SQLite control records the full denormalized column inventory, metadata_json absence, and stable read-only data_version before artifact commit.
patterns-established:
  - Current-family controls use the same fixed int32 semantic input, source/copy SHA-256 provenance, safe relative file inventory, and staged manifest-prefix validation as historical fixtures.
  - Fixture-specific provenance extensions are admitted only when the validator needs exact non-enumerating proof beyond the base schema.
requirements-completed: [MIGR-01]
coverage:
  - id: D1
    description: The 0.3.14 nested JSON control preserves the exact writer schema, signed direct unified entry, safe NPZ payload, and immutable provenance.
    requirement: MIGR-01
    verification:
      - kind: unit
        ref: uv run python tests/fixtures/compat/validate_corpus.py --expected-through json-nested-v0314
        status: pass
    human_judgment: false
  - id: D2
    description: The 0.3.14 SQLite control preserves the full denormalized columns, metadata_json absence, stable read-only data_version, safe NPZ payload, and immutable provenance across all eight records.
    requirement: MIGR-01
    verification:
      - kind: unit
        ref: uv run python tests/fixtures/compat/validate_corpus.py --expected-through sqlite-columns-v0314
        status: pass
    human_judgment: false
duration: 4min
completed: 2026-08-29
status: complete
---

# Phase 01 Plan 11: Current JSON and SQLite Compatibility Controls Summary

**Pinned 0.3.14 nested-JSON and denormalized-SQLite controls complete the immutable eight-record corpus with exact current-writer provenance.**

## Performance

- **Duration:** 4min
- **Started:** 2026-08-29T22:11:26Z
- **Completed:** 2026-08-29T22:15:38Z
- **Tasks:** 2/2
- **Files modified:** 8

## Accomplishments

- Generated the 0.3.14 nested JSON control from audited commit `a22f4b4575cb8213d9783ed388d2a70727563db1` using only `UnifiedCache.put`, in-memory signing, and native pickle-disabled NPZ output.
- Recorded and independently recomputed the exact current unified key `6d9e6b0907ec795a`, then checked only that entry directly without scanning metadata.
- Generated the 0.3.14 denormalized SQLite control from the same writer, preserving the exact 15-column `cache_entries` schema, absence of `metadata_json`, and a stable read-only `PRAGMA data_version` of `2`.
- Appended the seventh and eighth normative manifest records; the independent validator now passes every raw, split JSON, legacy SQLite, decorator-key, and current control fixture.

## Task Commits

Each fixture task was committed atomically:

1. **Task 1: Generate current nested JSON control** - `dc6ce49` (`feat`)
2. **Task 2: Generate current denormalized SQLite control** - `c6301a6` (`feat`)

## Files Created/Modified

- `tests/fixtures/compat/json-nested-v0314/` - audited 0.3.14 nested JSON metadata, native NPZ payload, and exact unified-key provenance.
- `tests/fixtures/compat/sqlite-columns-v0314/` - audited 0.3.14 denormalized SQLite metadata, native NPZ payload, and schema/data-version provenance.
- `tests/fixtures/compat/manifest.json` - complete insertion-ordered eight-record compatibility index.
- `tests/fixtures/compat/validate_corpus.py` - independent direct current-key and current-SQLite inspection checks.

## Decisions Made

- Used the final audited source commit and a disposable detached worktree for both controls; no historical or production Cacheness reader was invoked to create or inspect evidence.
- Kept the base provenance contract for ordinary records and added narrowly scoped extensions only for the exact unified-key and SQLite inspection facts the plan requires.
- Treated the SQLite metadata file as opaque evidence: all schema/data-version checks copy it first and open only the copy with a `mode=ro` URI.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical Functionality] Added exact current unified-key verification**
- **Found during:** Task 1 (Generate current nested JSON control)
- **Issue:** The validator checked the current JSON container but did not prove the required unified key or direct entry lookup.
- **Fix:** Added constrained unified-key provenance and a validator routine that recomputes the exact key, validates top-level key order, and looks up only the declared entry with no metadata enumeration.
- **Files modified:** `tests/fixtures/compat/validate_corpus.py`, `tests/fixtures/compat/json-nested-v0314/provenance.json`
- **Verification:** `uv run python tests/fixtures/compat/validate_corpus.py --expected-through json-nested-v0314`
- **Committed in:** `dc6ce49` (part of Task 1)

**2. [Rule 2 - Missing Critical Functionality] Made current SQLite inspection evidence independently enforceable**
- **Found during:** Task 2 (Generate current denormalized SQLite control)
- **Issue:** The validator checked current SQLite columns and temporary-copy stability but did not require the recorded schema inventory, absence of `metadata_json`, or claimed read-only `data_version` values to agree with a fresh inspection.
- **Fix:** Added constrained SQLite-inspection provenance and a validator routine that copies the fixture, opens only the copy through a read-only URI, and compares the schema and data-version values with provenance.
- **Files modified:** `tests/fixtures/compat/validate_corpus.py`, `tests/fixtures/compat/sqlite-columns-v0314/provenance.json`
- **Verification:** `uv run python tests/fixtures/compat/validate_corpus.py --expected-through sqlite-columns-v0314`
- **Committed in:** `c6301a6` (part of Task 2)

---

**Total deviations:** 2 auto-fixed (Rule 2 - missing critical functionality)
**Impact on plan:** Both corrections make the plan's required provenance claims executable without changing production formats, readers, or adapter behavior.

## Issues Encountered

- The pinned source manifest does not resolve NumPy by itself. The detached historical source was therefore executed with the repository's existing locked test environment and its `src/` first on `PYTHONPATH`; no dependency was added or substituted.
- The audited source emitted a non-fatal `SqliteBackend.__del__` cleanup `TypeError` after its writer-only process exited. The writer had already completed, and copied source/current hashes plus full corpus validation confirm that the fixture evidence was unaffected. This historical cleanup behavior is out of scope for the fixture corpus task.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 01-12 can implement current and legacy production metadata/signature/decorator adapters against a complete immutable corpus.
- The independent validator now enforces safe parsing, exact staged matrix membership, native pickle-disabled NPZ semantics, direct current-key proof, and read-only SQLite schema/data-version invariance.

## Self-Check: PASSED

- All six current-control evidence files, the manifest, validator, and this summary exist.
- Both fixture commits (`dc6ce49` and `c6301a6`) are present in git history.
- The fixture/provenance/validator stub scan found no placeholders, skipped tests, or unrun required verification.

---
*Phase: 01-compatibility-and-security-baseline*
*Completed: 2026-08-29*
