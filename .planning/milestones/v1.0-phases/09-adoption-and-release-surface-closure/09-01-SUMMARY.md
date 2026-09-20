---
phase: 09-adoption-and-release-surface-closure
plan: 01
subsystem: public-format-extension-surface
tags: [format-handlers, public-api, handler-registry, storage-barrels]
requires:
  - phase: 08-production-gates-and-performance-stabilization
    provides: BlobStore-first local-readiness baseline and guarded handler I/O
provides:
  - Alias-free FormatHandler and FormatHandlerError public contracts
  - Storage-oriented handler exports with unchanged registry and payload identities
affects: [09-02, 09-03, 09-09, phase-10]
actuals:
  tokens: 4625
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - Native format protocol names are independent of persisted handler and payload identities.
    - Storage barrels expose one generic extension protocol without retired aliases.
key-files:
  created: []
  modified:
    - src/cacheness/interfaces.py
    - src/cacheness/handlers.py
    - src/cacheness/error_handling.py
    - src/cacheness/storage/__init__.py
    - src/cacheness/storage/handlers/__init__.py
    - tests/test_error_handling.py
    - tests/test_interfaces.py
    - tests/test_handler_registration.py
key-decisions:
  - "Renamed both independent handler error bases to FormatHandlerError without compatibility aliases."
  - "Preserved data_type, payload_format, payload_format_version, registry priority, and guarded handler I/O behavior."
patterns-established:
  - "Custom formats continue to register through store.handlers.register_handler(...); no second registry surface exists."
requirements-completed: [CACH-06]
coverage:
  - id: D1
    description: Alias-free FormatHandler protocol and FormatHandlerError hierarchy preserve the existing handler contract.
    requirement: CACH-06
    verification:
      - kind: unit
        ref: tests/test_interfaces.py and tests/test_error_handling.py
        status: pass
    human_judgment: false
  - id: D2
    description: Both storage barrels export only FormatHandler and FormatHandlerError while preserving HandlerRegistry registration semantics.
    requirement: CACH-06
    verification:
      - kind: integration
        ref: tests/test_handler_registration.py and storage import assertion
        status: pass
    human_judgment: false
duration: 5min
completed: 2026-09-17
status: complete
---

# Phase 9 Plan 01: FormatHandler Public Cutover Summary

**Alias-free `FormatHandler` and `FormatHandlerError` contracts now present the BlobStore-first extension surface without changing stored handler or payload identities.**

## Performance

- **Duration:** 5 min
- **Started:** 2026-09-17T02:44:05Z
- **Completed:** 2026-09-17T02:49:35Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- Replaced the independent cross-cutting handler error base with `FormatHandlerError`, preserving `CacheError` inheritance, decorator conversion, causes, and context.
- Renamed the core protocol and interface error base, all built-in subclasses, registry annotations, and both storage barrels to `FormatHandler`/`FormatHandlerError` with no compatibility alias.
- Preserved `data_type`, payload format/version contracts, transformation validation, handler priorities, and the existing `store.handlers.register_handler(...)` extension seam.
- Removed import-time dataframe availability logging while retaining request-time optional-capability errors.

## Task Commits

1. **Task 1: Cut the independent cross-cutting error vocabulary to FormatHandlerError**
   - `38b3cc4` — `test(09-01): add failing FormatHandlerError contract`
   - `5368c9d` — `feat(09-01): rename cross-cutting handler error`
2. **Task 2: Atomically cut the core protocol, interface error, both barrels, and importing tests**
   - `a3c405e` — `test(09-01): add failing FormatHandler protocol cutover`
   - `b1281f3` — `feat(09-01): cut handler protocol to FormatHandler`

## Verification

- `uv run pytest -q -o log_cli=false tests/test_error_handling.py -x` — passed (38 tests).
- `uv run pytest -q -o log_cli=false tests/test_interfaces.py tests/test_handler_registration.py -x` — passed (46 tests).
- `uv run pytest -q -o log_cli=false tests/test_interfaces.py tests/test_handler_registration.py tests/test_stored_compatibility.py -x` — passed (57 tests).
- Storage barrel import assertion confirmed both new names resolve identically and all retired attributes are absent.
- Source scan confirmed the retired names are absent from the live implementation files changed by this plan.

## Decisions Made

- The Python protocol noun changes, but persisted `handler_type` remains the handler's existing `data_type`; no class name enters a manifest.
- `CacheWriteError`, `CacheReadError`, `CacheFormatError`, and `CacheValidationError` retain their failure-kind names below `FormatHandlerError`.
- Optional dataframe integrations stay quiet at import time and report clear errors only when their handlers are requested.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- A focused Ruff scan reports eight pre-existing findings in the optional handler re-export and historical interface tests. The plan's required verification passed; no unrelated lint cleanup or policy change was made.

## Known Stubs

None.

## Next Phase Readiness

- Plan 09-02 can prove source-free wheel behavior and durable reopen identity continuity against the renamed protocol.
- No lifecycle, concurrency, backend-composition, manifest, schema, or migration behavior changed.

## Self-Check: PASSED

- Summary file exists and all four TDD commits are present in Git history.

*Phase: 09-adoption-and-release-surface-closure*
*Completed: 2026-09-17*
