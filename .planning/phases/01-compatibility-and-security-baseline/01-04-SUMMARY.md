---
phase: 01-compatibility-and-security-baseline
plan: "04"
subsystem: array-security
tags: [numpy, npz, blosc2, safe-parsing, integrity, signatures, pickle]
requires:
  - phase: 01-01
    provides: Stable CacheReason values, CacheLegacyFormatError, and compatibility configuration conventions
  - phase: 01-03
    provides: Guarded snapshot verification before high-level handler deserialization
  - phase: 01-08
    provides: Immutable historical raw-array fixture corpus and provenance
provides:
  - Bounded non-evaluating reader for declared legacy Blosc2 raw-array frames
  - Native pickle-disabled NPZ writes for ordinary arrays and array dictionaries
  - Explicit integrity-enforced ObjectHandler routing for object-dtype arrays
affects: [handlers, UnifiedCache, compatibility-fixtures, Phase 01 quality gate]
actuals:
  tokens: 8123
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - Legacy metadata is parsed with bounded manual grammars and checked arithmetic before reconstruction
    - Executable serializers require configuration-time trust policy and snapshot-time authenticity verification
    - New ordinary formats use their native owner while historical formats remain read-only compatibility inputs
key-files:
  created:
    - tests/test_legacy_array_security.py
  modified:
    - src/cacheness/config.py
    - src/cacheness/handlers.py
key-decisions:
  - Legacy raw-array headers remain readable only through bounded tuple parsing, exact decompressed-byte validation, and declared-format dispatch.
  - New ordinary arrays always use native NPZ with pickle disabled; Cacheness no longer exposes a custom raw-array writer.
  - Object-dtype arrays route to ObjectHandler only after explicit opt-in requires signing, payload integrity verification, and rejection of unsigned entries.
patterns-established:
  - Treat retained invalid signature evidence as non-authorizing: verification failure returns a miss before any executable handler runs.
  - Test historical frames from copied immutable evidence and assert source/copy digests remain unchanged.
requirements-completed: [SECU-02]
coverage:
  - id: D1
    description: Bounded legacy raw-array parsing validates framing, tuple grammar, dtype, decompression, checked byte counts, and no-sidecar fallback.
    requirement: SECU-02
    verification:
      - kind: unit
        ref: uv run pytest -q -o log_cli=false tests/test_legacy_array_security.py -x
        status: pass
    human_judgment: false
  - id: D2
    description: Ordinary arrays write native pickle-disabled NPZ, while object arrays require strict trusted ObjectHandler routing before deserialization.
    requirement: SECU-02
    verification:
      - kind: integration
        ref: uv run pytest -q -o log_cli=false tests/test_legacy_array_security.py tests/test_handlers.py tests/test_config_validation.py -x
        status: pass
    human_judgment: false
duration: 14min
completed: 2026-08-29
status: complete
---

# Phase 01 Plan 04: Legacy Array Security Summary

**Legacy raw arrays now use a bounded non-evaluating compatibility reader, while new arrays write native NPZ and object arrays require strict authenticated ObjectHandler routing.**

## Performance

- **Duration:** 14min
- **Started:** 2026-08-29T21:22:41Z
- **Completed:** 2026-08-29T21:36:19Z
- **Tasks:** 2/2
- **Files modified:** 3

## Accomplishments

- Replaced metadata-controlled `eval` with a bounded ASCII tuple parser, length-framed dtype decoding, checked shape arithmetic, exact decompressed-byte validation, and typed legacy-format failures.
- Kept the historical 0.3.5 `compress` and 0.3.7 `compress2` frames readable through the production reader without changing their source/copy SHA-256 evidence.
- Removed the Cacheness-specific raw array writer: all new numeric arrays and dictionaries use native NPZ with pickle disabled, and the historical flag emits a deprecation warning.
- Added an explicit `allow_trusted_object_arrays` configuration gate requiring object pickle, entry signing, integrity verification, and unsigned-entry rejection before routing to `ObjectHandler`.
- Proved wrong signatures, payload-hash mismatches, and unsigned object-array entries stop before executable deserialization, including when invalid-signature evidence is retained.

## Task Commits

Each TDD task was committed atomically:

1. **Task 1: Specify the bounded legacy array grammar and declared-format failures**
   - `d71d4cf` `test(01-04): specify legacy array security contract`
   - `24ea798` `feat(01-04): secure legacy array decoding`
2. **Task 2: Stop custom array writes and require explicit trusted-object routing**
   - `dcf3f83` `test(01-04): specify trusted object array routing`
   - `1c80d4a` `feat(01-04): require trusted object array routing`

## Files Created/Modified

- `src/cacheness/handlers.py` - bounded legacy frame reader, declared-format-only dispatch, native NPZ persistence, and strict object-array handler selection.
- `src/cacheness/config.py` - explicit trusted-object opt-in plus configuration-time integrity/signing validation.
- `tests/test_legacy_array_security.py` - immutable fixture, parser abuse, NPZ, and authenticity-before-deserialization coverage.

## Decisions Made

- Made the custom Blosc2 raw format read-only compatibility input, rather than inventing a replacement Cacheness-specific container.
- Required the complete signing and integrity policy at configuration time so a trusted object array cannot accidentally weaken read authorization.
- Kept `delete_invalid_signatures=False` as evidence retention only; it never authorizes `ObjectHandler` deserialization.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The requested Ruff scope reports five pre-existing unused imports/locals in `src/cacheness/config.py` and `src/cacheness/handlers.py`. The plan introduced none, so the baseline findings remain out of scope.
- A sandboxed `uv` invocation could not access its existing cache; the same verification passed after approval to access that cache.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 01 now has safe, fixed-format legacy array reads and a sealed ordinary/object-array trust boundary for subsequent metadata and quality-gate plans.
- Future storage work must preserve the declared-format-only dispatch and pre-deserialization snapshot authorization patterns.

## Self-Check: PASSED

- All three implementation/test artifacts and this summary exist.
- All four Task 1/Task 2 TDD commits (`d71d4cf`, `24ea798`, `dcf3f83`, and `1c80d4a`) are present in git history.
- The introduced-code stub scan found no rendering-affecting hardcoded placeholder or empty data path.

---
*Phase: 01-compatibility-and-security-baseline*
*Completed: 2026-08-29*
