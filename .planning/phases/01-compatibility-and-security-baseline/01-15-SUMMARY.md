---
phase: 01-compatibility-and-security-baseline
plan: "15"
subsystem: storage-publication-and-recovery
tags: [blob-store, unified-cache, candidate-publication, clear-recovery, security]
requires:
  - phase: 01-14
    provides: bounded, topology-bound clear recovery for local metadata adapters
provides:
  - candidate-safe BlobStore and UnifiedCache publication paths that preserve the last committed entry on pre-commit failure
  - UnifiedCache reuse of the bounded clear-recovery coordinator rather than a second recovery state machine
  - measured CR-01 through CR-06 execution evidence for independent orchestrator review
affects: [phase-03, phase-04, phase-06, orchestrator-review]
tech-stack:
  added: []
  patterns:
    - candidate locator ownership ends only after metadata publication succeeds
    - shared clear-only coordinator with owner-specific physical-name adapters
    - executor evidence is distinct from independent review and validation approval
key-files:
  created:
    - .planning/phases/01-compatibility-and-security-baseline/01-15-EXECUTION-EVIDENCE.md
  modified:
    - src/cacheness/storage/blob_store.py
    - src/cacheness/core.py
    - src/cacheness/storage/clear_recovery.py
    - tests/test_filesystem_containment.py
    - tests/test_cache_integrity.py
    - tests/test_clear_recovery.py
key-decisions:
  - Candidate payloads remain private and owned through the complete pre-commit region; only a successful metadata write authorizes the replacement.
  - UnifiedCache adapts the Plan 01-14 clear coordinator rather than introducing a second journal or recovery protocol.
  - The clear primitive stays clear-only; Phase 3 must absorb it into a general lifecycle engine rather than extending it ad hoc.
requirements-completed: [SECU-01]
actuals:
  tokens: 14866
  tasks: 3
  commits: 5
completed: 2026-08-30
status: complete
---

# Phase 01 Plan 15: Candidate Publication and UnifiedCache Recovery Summary

Candidate-safe BlobStore and UnifiedCache writes now preserve the last committed value, while UnifiedCache global clear reuses the bounded recovery coordinator and leaves approval to independent review.

## Completed Work

### Task 1: Private BlobStore candidates

- Wrote first-write and cross-format overwrite RED coverage before implementing candidate publication.
- Made the candidate authoritative only after metadata persistence succeeds; failed publication removes only the candidate and preserves the prior payload, metadata, handler format, and readable value.
- Kept prior-payload cleanup distinct from the commit boundary, so a post-commit cleanup error cannot restore stale metadata.

### Task 2: UnifiedCache ownership and shared clear recovery

- Extended the same pre-commit candidate ownership rule through UnifiedCache serialization, snapshots, digests, signatures, and metadata publication.
- Routed `UnifiedCache.clear_all()` through Plan 01-14's admission, topology, journal, rollback, and reopen-recovery primitive without duplicating the state machine.
- Covered JSON, SQLite, and in-memory exact snapshot restoration; unsupported PostgreSQL and custom topology paths reject before staging.

### Task 3: Executor evidence handoff

- Recorded base `494d661`, execution head `61aa46d`, measured commands, pass/skip/finding counts, and CR-01 through CR-06 mappings in [01-15-EXECUTION-EVIDENCE.md](./01-15-EXECUTION-EVIDENCE.md).
- Kept `01-VALIDATION.md` draft/pending and did not author or overwrite `01-REVIEW.md` or `01-REVIEW-FIX.md`.

## Verification

- `uv run pytest -q -o log_cli=false tests/test_filesystem_containment.py -k "blob_store and (candidate or metadata or overwrite)" -x` — passed for Task 1 candidate publication coverage.
- `uv run pytest -q -o log_cli=false tests/test_cache_integrity.py tests/test_clear_recovery.py -x` — 112 passed.
- Focused CR matrix — 282 passed, 1 expected Windows-junction fixture skip.
- Legacy/native ownership regression — 122 passed.
- Full suite — 1,119 passed, 27 skipped, with 1 pre-existing collection warning.
- Phase quality gate — 7 passed; the raw Ruff diagnostic recorded 118 existing findings and no finding in the Phase 1-created Python-file inventory.
- `git diff --check` — passed.

## Task Commits

1. **Task 1 RED: candidate-safe BlobStore publication** — `68e47e8` (`test`)
2. **Task 1 GREEN: private BlobStore candidate publication** — `8c616c5` (`fix`)
3. **Task 2 RED: UnifiedCache recovery ownership** — `965fdc1` (`test`)
4. **Task 2 GREEN: UnifiedCache shared recovery ownership** — `61aa46d` (`feat`)
5. **Task 3: execution evidence** — `a73d880` (`docs`)

## TDD Gate Compliance

- Task 1 RED/GREEN: `68e47e8` / `8c616c5`.
- Task 2 RED/GREEN: `965fdc1` / `61aa46d`.

## Deviations from Plan

### Auto-fixed Issues

1. **[Rule 1 - Security] Kept clear recovery compatible with exact candidate locators**
   - **Found during:** Task 1 GREEN integration review.
   - **Issue:** The coordinator accepted only legacy physical locator forms, which would reject a committed candidate payload during clear or recovery.
   - **Fix:** Added the exact candidate-marker and UUID grammar alongside the legacy form, while retaining independent suffix grammar, the 96-byte suffix cap, and full blob-ID validation.
   - **Files modified:** `src/cacheness/storage/clear_recovery.py`, `tests/test_clear_recovery.py`.
   - **Verification:** Candidate clearability/recovery and malformed-marker, malformed-UUID, and overlong-suffix hostile tests passed.
   - **Committed in:** `8c616c5`.

2. **[Rule 2 - Correctness] Added a narrow coordinator physical-name adapter**
   - **Found during:** Task 2 GREEN.
   - **Issue:** UnifiedCache's namespaced physical locators need the same coordinator without weakening BlobStore's exact default locator identity.
   - **Fix:** Added an owner-supplied `physical_name` callback with BlobStore's identity as the default; UnifiedCache supplies its exact namespace adapter rather than a duplicate state machine.
   - **Files modified:** `src/cacheness/storage/clear_recovery.py`, `src/cacheness/core.py`, `tests/test_clear_recovery.py`.
   - **Verification:** The supported JSON/SQLite/in-memory recovery matrix and adversarial topology checks passed.
   - **Committed in:** `61aa46d`.

3. **[Rule 1 - Concurrency] Extended clear admission over preflight work**
   - **Found during:** Task 2 GREEN review.
   - **Issue:** Reading and preflighting entries outside coordinator admission allowed a contender to observe/mutate the clear operation before the lock's intended boundary.
   - **Fix:** Held coordinator admission across listing, preflight, mapping construction, and the clear transaction; added contention coverage that fails before list/preflight callbacks for a blocked contender.
   - **Files modified:** `src/cacheness/core.py`, `tests/test_clear_recovery.py`.
   - **Verification:** Focused cache-integrity/recovery and full evidence suites passed.
   - **Committed in:** `61aa46d`.

**Total deviations:** 3 auto-fixed (2 Rule 1, 1 Rule 2). All were narrow correctness/security integrations required to preserve the plan's locked boundaries.

## Independent Approval Status

Execution is complete, but Phase 1 is **not** complete. `01-VALIDATION.md` remains draft/pending, and only the orchestrator may run the renewed code/security review and decide whether validation can be finalized. This summary and the execution-evidence file are evidence, not approval.

## INCOMPLETE Downstream Handoff

Only `SECU-01` is complete for this plan. Plan 15 adds no canonical manifest, CAS or generation model, or general lifecycle/reconciliation engine. The clear recovery primitive remains deliberately clear-only and must be absorbed by Phase 3 rather than expanded in place.

| Requirement | Status | Required handoff |
| --- | --- | --- |
| STOR-03, STOR-04 | INCOMPLETE | Phase 3 must provide the complete canonical write/overwrite commit model and recoverable residue handling. |
| STOR-05, STOR-06 | INCOMPLETE | Phase 3 must extend beyond global clear to delete, overwrite, close, inventory, repair policy, and general resumable reconciliation. |
| CACH-03 | INCOMPLETE | Phase 6 must route TTL, eviction, predicate, decorator, and single-key invalidations through complete entry removal. |
| BACK-03 | INCOMPLETE | Phase 4 must establish caller-injected and registered metadata-backend composition. |
| BACK-06 | INCOMPLETE | Phase 4 must define advertised topology and durability capabilities beyond the local rejection boundary. |

## Known Stubs

None.

## Self-Check: PASSED

- Confirmed all five Task 1–3 commits exist in repository history.
- Confirmed the evidence handoff and this summary exist.
