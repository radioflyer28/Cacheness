---
phase: 01-compatibility-and-security-baseline
plan: "08"
subsystem: compatibility-fixtures
tags: [compatibility, fixtures, provenance, blosc2, sha256, safe-parsing]
requires:
  - phase: 01-01
    provides: Public compatibility baseline and migration characterization contract
provides:
  - Immutable 0.3.5 `blosc2.compress` and 0.3.7 `blosc2.compress2` raw-array fixtures
  - An independent staged validator for the complete normative eight-fixture matrix
  - Writer-only provenance, safe-path validation, and non-mutating raw-frame inspection
affects: [01-04, 01-09, 01-10, 01-11, 01-12]
actuals:
  tokens: 7941
  tasks: 3
  commits: 3
tech-stack:
  added: []
  patterns:
    - Historical fixture writers run only in disposable detached worktrees.
    - Fixture validation verifies digests and bounded framing before any content inspection.
key-files:
  created:
    - tests/fixtures/compat/validate_corpus.py
    - tests/fixtures/compat/manifest.json
    - tests/fixtures/compat/README.md
    - tests/fixtures/compat/array-raw-v035-compress/payload.b2nd
    - tests/fixtures/compat/array-raw-v037-compress2/payload.b2nd
  modified: []
key-decisions:
  - "The production-independent validator owns the full eight-variant normative matrix, while staged commands require an exact accumulated prefix."
  - "Raw fixtures are checked with bounded UTF-8 framing and decompressed-byte comparison, never metadata evaluation or historical readers."
  - "Every copied payload records matching source/copy SHA-256 values and an independent manifest digest for its provenance document."
patterns-established:
  - "Future fixture plans append only the matrix's next record and rerun the validator through that fixture ID."
  - "SQLite validation copies evidence before opening a read-only URI; NPZ inspection always disables pickle."
requirements-completed: [MIGR-01]
coverage:
  - id: D1
    description: "Pinned 0.3.5 and 0.3.7 raw-array frames carry exact commit, toolchain, semantic-input, and source/copy digest provenance."
    requirement: MIGR-01
    verification:
      - kind: unit
        ref: "uv run python tests/fixtures/compat/validate_corpus.py --expected-through array-raw-v037-compress2"
        status: pass
    human_judgment: false
  - id: D2
    description: "The independent validator enforces staged manifest/provenance, safe relative paths, SHA-256 invariance, bounded raw framing, and direct semantic-byte checks."
    requirement: MIGR-01
    verification:
      - kind: unit
        ref: "uv run python tests/fixtures/compat/validate_corpus.py --expected-through array-raw-v037-compress2"
        status: pass
    human_judgment: false
duration: 3min
completed: 2026-08-29
status: complete
---

# Phase 01 Plan 08: Raw Historical Fixture Corpus Summary

**Two pinned legacy Blosc2 raw-array frames now have immutable writer-only provenance and a standalone safety validator that future compatibility fixtures extend.**

## Performance

- **Duration:** 3min
- **Started:** 2026-08-29T16:42:31-04:00
- **Completed:** 2026-08-29T16:45:27-04:00
- **Tasks:** 3/3
- **Files modified:** 7

## Accomplishments

- Generated the 0.3.5 `blosc2.compress` and 0.3.7 `blosc2.compress2` evidence using only their pinned historical writers in disposable detached worktrees.
- Recorded exact commits, toolchains, fixed `int32` `(2, 3)` semantic input, and matching source/copy SHA-256 values for each payload.
- Added a production-independent validator that owns the full eight-fixture matrix and performs staged, non-mutating pre-read gates.
- Documented the immutable test-only index, writer-only process, safe relative paths, and subsequent fixture-plan ownership.

## Task Commits

Each task was committed atomically:

1. **Task 1: Generate the 0.3.5 compress fixture** - `14712e6` (`feat`)
2. **Task 2: Generate the 0.3.7 compress2 fixture** - `db0f2a6` (`feat`)
3. **Task 3: Index and safely inspect both raw fixtures** - `dfcaece` (`docs`)

## Files Created/Modified

- `tests/fixtures/compat/validate_corpus.py` - Complete normative-matrix validator with safe path, schema, digest, raw-frame, NPZ, JSON, and SQLite checks.
- `tests/fixtures/compat/manifest.json` - Insertion-ordered two-record immutable fixture index.
- `tests/fixtures/compat/README.md` - Corpus scope, provenance, validation, and ownership protocol.
- `tests/fixtures/compat/array-raw-v035-compress/` - 0.3.5 `compress` payload and provenance.
- `tests/fixtures/compat/array-raw-v037-compress2/` - 0.3.7 `compress2` payload and provenance.

## Decisions Made

- The validator defines the eight-fixture matrix in code rather than inferring compatibility facts from files on disk, so every staged manifest must be an exact prefix.
- Raw metadata is inspected with bounded framing and direct decompressed-byte comparison; no header evaluation, array reconstruction, or historical reader is used.
- `manifest.json` independently digests each provenance file, while provenance records equal source and copied digests for generated evidence.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

The runtime sandbox does not permit direct writes to the primary Git index. Each verified task commit was therefore handed to the orchestrator at its atomic boundary; this did not change the implementation or verification protocol.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plans 01-09 through 01-11 can append the remaining normative fixture records and rerun the staged validator.
- Plans 01-04 and 01-12 can use the two raw frames as immutable production-reader evidence without invoking historical readers.

---
*Phase: 01-compatibility-and-security-baseline*
*Completed: 2026-08-29*

## Self-Check: PASSED

- All seven fixture/index/validator files and this summary exist.
- All three task commits (`14712e6`, `db0f2a6`, and `dfcaece`) are present in Git history.
- `uv run python tests/fixtures/compat/validate_corpus.py --expected-through array-raw-v037-compress2` passed after the final task commit.
