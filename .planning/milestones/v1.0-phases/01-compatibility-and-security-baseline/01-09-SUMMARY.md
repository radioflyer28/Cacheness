---
phase: 01-compatibility-and-security-baseline
plan: "09"
subsystem: compatibility-fixtures
tags: [compatibility, fixtures, json, npz, hmac, provenance, sha256]
requires:
  - phase: 01-08
    provides: Independent staged compatibility-fixture validator and raw-frame corpus prefix
provides:
  - Immutable unsigned 0.3.7 split-map JSON/NPZ compatibility evidence
  - Immutable fixed-key signed 0.3.8 split-map JSON/NPZ compatibility evidence
  - Four-record corpus prefix with source/copy digest provenance
affects: [01-04, 01-10, 01-11, 01-12]
actuals:
  tokens: 3100
  tasks: 2
  commits: 2
tech-stack:
  added: []
  patterns:
    - Historical JSON fixture writers run in disposable detached worktrees.
    - Signed test evidence records fixed non-production key inputs and an expected wrong-key failure without checking in a key file.
key-files:
  created:
    - tests/fixtures/compat/json-split-unsigned-v037/metadata.json
    - tests/fixtures/compat/json-split-unsigned-v037/payload.npz
    - tests/fixtures/compat/json-split-unsigned-v037/provenance.json
    - tests/fixtures/compat/json-split-signed-v038/metadata.json
    - tests/fixtures/compat/json-split-signed-v038/payload.npz
    - tests/fixtures/compat/json-split-signed-v038/provenance.json
  modified:
    - tests/fixtures/compat/manifest.json
key-decisions:
  - "The signed split-map fixture uses a fixed, disclosed test-only 32-byte HMAC input and records a distinct wrong-key verification failure expectation in provenance."
  - "The corpus manifest remains an insertion-ordered test index, while the independent validator enforces the complete signed and unsigned split-map discriminators."
patterns-established:
  - "Append exactly one normative fixture record per task and validate the complete staged prefix before committing."
  - "Copy only payload and metadata evidence from a historical writer worktree; retain source/copy SHA-256 equality in provenance."
requirements-completed: [MIGR-01]
coverage:
  - id: D1
    description: "Pinned 0.3.7 unsigned split-map JSON metadata and numeric NPZ payload carry complete discriminators and byte-invariant source/copy provenance."
    requirement: MIGR-01
    verification:
      - kind: unit
        ref: "uv run python tests/fixtures/compat/validate_corpus.py --expected-through json-split-signed-v038"
        status: pass
    human_judgment: false
  - id: D2
    description: "Pinned 0.3.8 signed split-map JSON metadata records a fixed test-only HMAC input, wrong-key expectation, and numeric NPZ evidence."
    requirement: MIGR-01
    verification:
      - kind: unit
        ref: "uv run python tests/fixtures/compat/validate_corpus.py --expected-through json-split-signed-v038"
        status: pass
    human_judgment: false
duration: 8min
completed: 2026-08-29
status: complete
---

# Phase 01 Plan 09: Split-map JSON Fixture Corpus Summary

**Pinned 0.3.7 unsigned and fixed-key 0.3.8 signed split-map JSON/NPZ evidence now extends the corpus through its validated four-record prefix.**

## Performance

- **Duration:** 8min
- **Started:** 2026-08-29T21:09:07Z
- **Completed:** 2026-08-29T21:17:25Z
- **Tasks:** 2/2
- **Files modified:** 7

## Accomplishments

- Generated the 0.3.7 unsigned split-map JSON metadata and numeric NPZ payload from only the pinned historical writer in a disposable detached worktree.
- Generated the 0.3.8 signed split-map evidence with a fixed test-only 32-byte HMAC input, plus documented correct-key success and wrong-key failure inputs without copying the key file.
- Appended both records in normative order and revalidated the raw and split-map prefix for safe paths, exact discriminators, SHA-256 invariance, and pickle-disabled NPZ semantics.

## Task Commits

Each task was committed atomically:

1. **Task 1: Generate unsigned 0.3.7 split-map JSON/NPZ** - `3f47777` (`feat`)
2. **Task 2: Generate fixed-key signed 0.3.8 split-map JSON/NPZ** - `d0fda8e` (`feat`)

## Files Created/Modified

- `tests/fixtures/compat/json-split-unsigned-v037/` - Pinned unsigned JSON metadata, numeric NPZ payload, and source/copy digest provenance.
- `tests/fixtures/compat/json-split-signed-v038/` - Pinned signed JSON metadata, numeric NPZ payload, and fixed test-key expectation provenance.
- `tests/fixtures/compat/manifest.json` - Four-record, insertion-ordered immutable corpus index.

## Decisions Made

- The signed evidence records a test-only fixed HMAC input and a distinct wrong-key failure expectation in the required `generator_command` provenance field, keeping the key file itself out of the corpus.
- The validator, rather than fixture discovery, remains authoritative for the exact signed and unsigned split-map schema and accumulated prefix.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

The primary checkout is protected, so the parent executor performed each verified atomic task commit after the worker staged its exact boundary. This did not alter fixture generation or validation.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plans 01-10 and 01-11 can append the next normative SQLite and decorator-key evidence while retaining the same staged-validator protocol.
- Plans 01-04 and 01-12 can use the two split-map variants to exercise safe legacy metadata and signature adapters.

---
*Phase: 01-compatibility-and-security-baseline*
*Completed: 2026-08-29*

## Self-Check: PASSED

- All seven declared fixture/index files and this summary exist.
- Both task commits (`3f47777` and `d0fda8e`) are present in Git history.
- `uv run python tests/fixtures/compat/validate_corpus.py --expected-through json-split-signed-v038` passed after the final task commit.
