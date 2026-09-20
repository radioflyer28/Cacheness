---
phase: 01-compatibility-and-security-baseline
plan: "14"
subsystem: storage-recovery
tags: [blob-store, json, sqlite, durability, recovery, filesystem-security]
requires:
  - phase: 01-13
    provides: guarded filesystem identities and phase quality gates
provides:
  - bounded, topology-bound clear recovery for exact local JSON, SQLite, and in-memory metadata adapters
  - durable JSON metadata publication with acknowledged rollback behavior
  - root-scoped admission and hostile-journal rejection before mutation
affects: [01-15, phase-03, phase-04, phase-06]
tech-stack:
  added: []
  patterns:
    - exact backend identity adapters
    - bounded fail-closed recovery journals
    - root-scoped in-process and advisory admission locks
key-files:
  created:
    - src/cacheness/storage/clear_recovery.py
    - tests/test_clear_recovery.py
  modified:
    - src/cacheness/metadata.py
    - src/cacheness/storage/path_security.py
    - src/cacheness/storage/blob_store.py
key-decisions:
  - Clear recovery is an internal, clear-only primitive rather than a general lifecycle protocol.
  - Durable reopen recovery accepts only exact local JSON and SQLite metadata identities; memory has a deterministic process-loss rule and remote/custom identities reject before staging.
  - Every persisted original, tombstone, and snapshot locator is topology-bound and validated before any restoration or deletion.
requirements-completed: [SECU-01]
actuals:
  tokens: 23183
  tasks: 2
  commits: 4
completed: 2026-08-30
status: complete
---

# Phase 01 Plan 14: Durable Clear Recovery Summary

Implemented a bounded, fail-closed BlobStore clear-recovery coordinator with acknowledged JSON durability, exact backend topology handling, and hostile-journal protection.

## Completed Work

### Task 1: Durable JSON and BlobStore tracer

- Added durable, same-directory JSON metadata persistence with fsync barriers, recoverable backup/rollback behavior, and truthful acknowledgement.
- Added the internal recovery coordinator and root-scoped admission around BlobStore construction and clear operations.
- Added fault coverage for prepared recovery, metadata durability, and clear rollback behavior.

### Task 2: Exact topology, bounded journal, and adversarial recovery contract

- Added exact JSON, SQLite, and in-memory adapter semantics, including complete entry plus counter snapshots and deterministic memory process-loss handling.
- Enforced exclusive bounded journal creation, cardinality and byte limits, same-root locking, and unsupported-topology rejection before callbacks or mutation.
- Rejected hostile metadata, lock, unrelated payload, forged tombstone, invalid handler-suffix, and malformed nested-snapshot journals before mutation.

## Verification

- `uv run pytest -q -o log_cli=false tests/test_metadata.py tests/test_clear_recovery.py -x` — 71 passed.
- `uv run pytest -q -o log_cli=false tests/test_filesystem_containment.py -x` — 84 passed; 1 expected Windows-junction fixture skip.
- `uv run pytest -q -o log_cli=false tests/test_phase1_quality_gates.py -x` — 7 passed.
- `uv run ruff check src/cacheness/storage/clear_recovery.py tests/test_clear_recovery.py src/cacheness/metadata.py src/cacheness/storage/path_security.py src/cacheness/storage/blob_store.py tests/test_metadata.py --output-format concise` — the raw baseline reports seven pre-existing F401 findings in `src/cacheness/metadata.py`; phase-created files are clean under the Phase 1 quality gate.
- `git diff --check` — passed before closeout documentation edits.
- Confirmed `01-VALIDATION.md` remains `draft`/pending and the downstream requirements remain unchecked; neither artifact was modified.

## TDD Gate Compliance

- RED: `a2db71e` and `86fc7d6`.
- GREEN: `973098d` and `b1a41d9`.

## Deviations from Plan

### Auto-fixed Issues

1. **[Rule 1 - Bug] Accepted the canonical in-memory metadata backend interface**
   - **Found during:** Task 2 GREEN
   - **Issue:** BlobStore's narrow metadata-interface check rejected the established in-memory implementation, preventing the plan's required deterministic memory behavior.
   - **Fix:** Accepted the two established metadata base interfaces while retaining exact adapter identity checks for recovery.
   - **Files modified:** `src/cacheness/storage/blob_store.py`, `tests/test_clear_recovery.py`
   - **Commit:** `b1a41d9`

2. **[Rule 1 - Security] Made hostile journal locators and snapshots fail closed**
   - **Found during:** Task 2 GREEN review
   - **Issue:** A forged but syntactically valid journal could target a metadata/control file, unrelated payload, forged tombstone, oversized handler suffix, or malformed nested snapshot.
   - **Fix:** Bound each original to its cache key's exact physical locator and valid handler suffix, required exact tombstone and snapshot forms, and rejected invalid journal content before any mutation.
   - **Files modified:** `src/cacheness/storage/clear_recovery.py`, `tests/test_clear_recovery.py`
   - **Commit:** `b1a41d9`

3. **[Rule 1 - Bug] Retained recoverable evidence after a staged clear fault**
   - **Found during:** Task 1 GREEN review
   - **Issue:** A failure after tombstone creation could leave cleanup behavior inconsistent with the journal-backed rollback contract.
   - **Fix:** Kept prepared evidence and performed only bounded, ownership-checked cleanup until recovery can restore a determinate state.
   - **Files modified:** `src/cacheness/storage/clear_recovery.py`, `tests/test_clear_recovery.py`
   - **Commit:** `973098d`

## INCOMPLETE Downstream Handoff

Only `SECU-01` is complete for this plan. The coordinator is intentionally clear-only and must be absorbed by Phase 3 rather than treated as the canonical lifecycle engine.

| Requirement | Status | Required handoff |
| --- | --- | --- |
| STOR-03, STOR-04 | INCOMPLETE | Phase 3 must provide the complete write/overwrite commit model and residue recovery. |
| STOR-05, STOR-06 | INCOMPLETE | Phase 3 must extend beyond clear to delete/overwrite/close, inventory, repair policy, and general resumable reconciliation. |
| CACH-03 | INCOMPLETE | Plan 01-15 must integrate the clear prerequisite into UnifiedCache; Phase 6 retains cache invalidation ownership. |
| BACK-03 | INCOMPLETE | Phase 4 must establish correct caller-injected and registered backend composition. |
| BACK-06 | INCOMPLETE | Phase 4 must define general durability and topology capabilities; current rejection only prevents overclaiming. |

CR-05 is evidenced for the bounded BlobStore clear path. CR-04 remains open until Plan 01-15 integrates this prerequisite into UnifiedCache.

## Known Stubs

None.

## Self-Check: PASSED

- Confirmed all four task commits exist in the repository history.
- Confirmed the recovery coordinator, test suite, and this summary are present.
