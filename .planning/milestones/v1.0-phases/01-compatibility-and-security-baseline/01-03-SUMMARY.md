---
phase: 01-compatibility-and-security-baseline
plan: "03"
subsystem: storage-security
tags: [filesystem, containment, handler-io, integrity, signatures, blob-store]
requires:
  - phase: 01-02
    provides: Anchored roots, contained locator validation, and ManagedFileOps
provides:
  - Guarded private staging and one-open snapshots for high-level handlers
  - Domain-separated SHA-256 physical names for BlobStore keys and cache prefixes
  - Complete locator preflight before high-level reads, cleanup, and metadata mutation
affects: [BlobStore, UnifiedCache, filesystem-backends, 01-12]
actuals:
  tokens: 11271
  tasks: 3
  commits: 3
tech-stack:
  added: []
  patterns:
    - Handlers receive only private staging or snapshot paths, never managed locators
    - Logical identifiers are preserved as metadata while physical names are opaque hashes
    - Multi-entry storage operations preflight all locators before mutating safe siblings
key-files:
  created:
    - src/cacheness/storage/guarded_handler_io.py
  modified:
    - src/cacheness/interfaces.py
    - src/cacheness/storage/path_security.py
    - src/cacheness/storage/blob_store.py
    - src/cacheness/core.py
    - tests/test_filesystem_containment.py
    - tests/test_core.py
key-decisions:
  - "High-level handlers serialize in private stages and deserialize only from a live private snapshot copied once through ManagedFileOps."
  - "BlobStore keys and UnifiedCache prefixes are exact caller-visible metadata, while physical payload IDs are versioned, length-framed SHA-256 names."
  - "Unsafe locator preflight runs across each complete affected set before reads, list/clear exposure, expiry cleanup, size cleanup, overwrite, or invalidation can mutate state."
patterns-established:
  - "Use GuardedHandlerIO for every high-level handler boundary and keep signature/integrity checks inside its snapshot context."
  - "Re-raise CacheUnsafePathError before broad cache miss, cleanup, or evidence-removal policy."
requirements-completed: [SECU-01]
coverage:
  - id: D1
    description: "Guarded private handler staging, no-follow snapshots, and deterministic physical-name encoding"
    requirement: SECU-01
    verification:
      - kind: integration
        ref: "uv run pytest -q -o log_cli=false tests/test_filesystem_containment.py -k 'high_level or prefix or handler_io' -x"
        status: pass
    human_judgment: false
  - id: D2
    description: "BlobStore and UnifiedCache contain hostile persisted locators and preflight multi-entry mutation paths"
    requirement: SECU-01
    verification:
      - kind: integration
        ref: "uv run pytest -q -o log_cli=false tests/test_filesystem_containment.py tests/test_core.py tests/test_blob_backend_registry.py tests/test_handlers.py -x"
        status: pass
    human_judgment: false
duration: 12min
completed: 2026-08-29
status: complete
---

# Phase 01 Plan 03: Guarded High-Level I/O Summary

**BlobStore and UnifiedCache now encode caller-visible identities into opaque physical names and pass handlers only verified private payload snapshots.**

## Performance

- **Duration:** 12 min
- **Started:** 2026-08-29T20:52:24Z
- **Completed:** 2026-08-29T21:04:47Z
- **Tasks:** 3/3
- **Files modified:** 7

## Accomplishments

- Added `GuardedHandlerIO`, which stages handler writes privately, publishes through `ManagedFileOps`, and yields one private no-follow read snapshot without deserializing it.
- Replaced raw logical paths with versioned, length-framed, domain-separated SHA-256 physical names while retaining exact BlobStore keys and UnifiedCache prefixes in public metadata.
- Routed BlobStore and UnifiedCache payload access through guarded snapshots; integrity and current/compatibility-signature rejection now occurs before handler invocation.
- Added complete affected-entry preflight for high-level read, overwrite, list, clear, expiry, size-cleanup, and invalidation paths so an unsafe record cannot mutate or expose safe siblings.

## Task Commits

Each task was committed atomically:

1. **Task 1: Specify high-level I/O, locator, prefix, and race behavior**
   - `00c7cd4` `test(01-03): specify guarded high-level storage paths`
2. **Task 2: Implement the guarded handler contract and safe physical-name encoder**
   - `117bc08` `feat(01-03): add guarded handler I/O boundary`
3. **Task 3: Wire BlobStore and UnifiedCache exclusively through guarded handler I/O**
   - `84f653c` `feat(01-03): guard high-level payload I/O`

## Files Created/Modified

- `src/cacheness/interfaces.py` - documents guarded read/write result shapes without changing public handler signatures.
- `src/cacheness/storage/path_security.py` - adds opaque physical-name encoding and one-open stream-copy support.
- `src/cacheness/storage/guarded_handler_io.py` - stages writes privately and provides live private read snapshots.
- `src/cacheness/storage/blob_store.py` - preserves logical keys while routing all payload I/O and locator checks through the guard.
- `src/cacheness/core.py` - encodes prefixes, holds snapshots through verification, and preflights affected metadata sets.
- `tests/test_filesystem_containment.py` - covers high-level path isolation, ordered verification, hostile locators, preflight, and stage-link rejection.
- `tests/test_core.py` - updates the filename contract to assert opaque physical IDs.

## Decisions Made

- Kept public `CacheHandler.put/get` signatures unchanged and inserted containment entirely through the `GuardedHandlerIO` adapter.
- Used lowercase SHA-256 names over explicitly length-framed namespace, prefix, and logical-key UTF-8 bytes to make physical naming deterministic and non-ambiguous.
- Reserved `_verify_legacy_entry_signature` as a fail-closed compatibility seam for Plan 12; no unknown legacy signature can authorize deserialization now.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Test setup] Created the managed root before constructing the direct guard test fixture**
- **Found during:** Task 2
- **Issue:** `ManagedFileOps` correctly rejects a non-existent root, while the new direct adapter test had not created its fixture directory.
- **Fix:** Created the fixture root before constructing `GuardedHandlerIO`.
- **Files modified:** `tests/test_filesystem_containment.py`
- **Verification:** Focused guarded-I/O tests passed.
- **Committed in:** `117bc08`

**2. [Rule 2 - Compatibility contract] Updated filename assertions for opaque physical IDs**
- **Found during:** Task 3
- **Issue:** An existing core test asserted that caller keys and prefixes appeared in physical paths, contradicting D-12.
- **Fix:** Asserted deterministic physical-ID derivation and absence of logical key/prefix text instead.
- **Files modified:** `tests/test_core.py`
- **Verification:** Full containment/core/blob/handler suite passed.
- **Committed in:** `84f653c`

**3. [Rule 2 - Security hardening] Rejected handler-created symlink ancestors inside private stages**
- **Found during:** Task 3 security assessment
- **Issue:** A handler could return an ordinary file reachable through a symlinked private-stage ancestor, allowing external bytes to be published.
- **Fix:** Validated every stage component with `lstat` and rejected symlink ancestors before publication.
- **Files modified:** `src/cacheness/storage/guarded_handler_io.py`, `tests/test_filesystem_containment.py`
- **Verification:** The new stage-symlink regression and the full focused suite passed.
- **Committed in:** `84f653c`

**Total deviations:** 3 auto-fixed (1 Rule 1, 2 Rule 2)

**Impact on plan:** All changes enforce the planned containment and compatibility contract without adding transactions, manifests, reconciliation, or migration behavior.

## Issues Encountered

- The requested Ruff scope still reports three pre-existing unused-import `F401` findings in legacy custom-metadata/test code. The functional suite passes; this unrelated lint debt was left for orchestrator capture rather than expanded into this task.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 2 lifecycle work can rely on a single guarded payload boundary and opaque physical IDs.
- Plan 12 can install its exact legacy-signature verifier through the fail-closed core seam.

## Self-Check: PASSED

- All seven implementation/test files and this summary exist.
- All three Task commits (`00c7cd4`, `117bc08`, `84f653c`) are present in git history.

---
*Phase: 01-compatibility-and-security-baseline*
*Completed: 2026-08-29*
