---
phase: 07-explicit-migration-and-rebuild-cutover
reviewed: 2026-09-12T00:23:53Z
depth: standard
files_reviewed: 8
files_reviewed_list:
  - src/cacheness/storage/migration.py
  - src/cacheness/storage/migration_evidence.py
  - src/cacheness/storage/memory_lifecycle_authority.py
  - tests/test_migration_cutover.py
  - tests/test_migration_remote_contract.py
  - tests/test_rebuild_workflow.py
  - tools/verify_phase7_contracts.py
  - tests/test_phase7_contract_verifier.py
findings:
  critical: 1
  warning: 1
  info: 0
  total: 2
status: issues_found
---

# Phase 7: Code Review Report

**Reviewed:** 2026-09-12T00:23:53Z
**Depth:** standard
**Files Reviewed:** 8
**Status:** issues_found

## Summary

The destination-format and typed S3 abort repairs are correctly scoped, and the
normal `resume()` rebuild-debt test converges through authenticated receipts.
However, rebuild cleanup debt does not actually fence the other public rebuild
methods. A caller can continue staging, verify, and accept a rebuild while old
cleanup debt remains. A later `resume()` then deletes an accepted destination
entry before failing to checkpoint the illegal `REBUILD_ACCEPTED -> ABORTED`
transition. This is a demonstrated data-loss path and blocks shipment.

The review does not reopen ADR 0001's accepted invisible pre-checkpoint orphan,
request cross-resource ACID, or recommend another lock, queue, lease, journal,
sidecar, authority, listing/adoption mechanism, or obstore integration.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01 [BLOCKER]: Cleanup debt does not fence direct rebuild progression, allowing accepted data to be deleted

**Files:** `src/cacheness/storage/migration.py:2929-2996`,
`src/cacheness/storage/migration.py:3018-3102`,
`src/cacheness/storage/migration.py:3124-3145`,
`src/cacheness/storage/migration.py:4334-4337`, and
`src/cacheness/storage/migration_evidence.py:719-725`

**Issue:** `_abort_rebuild_after_failure()` now correctly leaves failed exact
cleanup in nonterminal evidence, but `stage_rebuild()`, `verify_rebuild()`, and
`accept_rebuild()` all carry `evidence.cleanup_debt` forward instead of refusing
progress until it is settled. `MaintenanceRunEvidence` prohibits debt only in
`ABORTED`, so it permits `REBUILD_ACCEPTED` evidence with outstanding debt.
Meanwhile, `resume()` dispatches on `cleanup_debt` before checking the evidence
state.

This is exploitable through the documented public methods without forging
evidence. I reproduced the following sequence with the existing test fixtures:

1. Stage entry one; fail staging entry two; fail cleanup before deleting entry one.
2. Call `stage_rebuild(plan)` directly after restoring the participant. It reaches
   `REBUILD_STAGED` while retaining cleanup debt.
3. Call `verify_rebuild(plan)` and `accept_rebuild(plan)`. Both succeed while debt
   remains, producing `REBUILD_ACCEPTED` evidence.
4. Call `resume(...)`. It deletes the accepted `first` entry, then evidence
   checkpointing rejects the terminal transition. The observed final state was
   `REBUILD_ACCEPTED` with `destination.get("first") is None`.

The external delete preceding the failed checkpoint makes this a real data-loss
and metadata/payload-disagreement bug, not a request for stronger cross-resource
atomicity. It also violates the intended rule that cleanup debt must settle before
terminal acceptance or abort.

**Fix:** Add one narrow debt guard at the existing maintenance coordinator seam.
Reject direct `stage_rebuild()`, `verify_rebuild()`, and `accept_rebuild()` calls
whenever authenticated rebuild cleanup debt exists, directing the caller to the
explicit receipt-bound `resume()` settlement path. In `resume()`, allow debt
settlement only from the defined nonterminal rebuild cleanup states and reject
debt in `REBUILD_ACCEPTED`. Strengthen `MaintenanceRunEvidence.__post_init__()` so
`REBUILD_ACCEPTED` cannot contain cleanup debt. No new state machine or lifecycle
authority is required.

## Warnings

### WR-01 [WARNING]: The fixed verifier can report MIGR-05 PASS without exercising the direct-method debt fence

**Files:** `tools/verify_phase7_contracts.py:146-159` and
`tests/test_phase7_contract_verifier.py:371-455`

**Issue:** The new MIGR-05 selector set proves the happy `resume()` settlement,
terminal-ABORTED validation, forged-debt rejection, and changed-owner handling,
but no selected test attempts `stage_rebuild()`, `verify_rebuild()`, or
`accept_rebuild()` while cleanup debt exists. Consequently the all-mode verifier
can pass even though CR-01 permits accepted evidence to retain debt and later lose
an accepted payload. The manifest self-test checks that named selectors exist and
are mapped; it does not cover this missing behavioral dimension.

**Fix:** Add an exact regression that creates authentic cleanup debt before the
external delete, asserts all forward rebuild methods fail before any payload or
authority mutation, and asserts `resume()` alone settles the debt. Map that exact
selector to MIGR-05, D-19/D-21, A-MIGR05, and the applicable Plan 21 threat row so
the verifier cannot render PASS if the fence regresses.

## Review Evidence

- The 12 focused Phase 7 repair tests passed.
- Scoped Ruff passed for all eight reviewed files.
- A separate fixture-based reproduction demonstrated the CR-01 sequence and
  ended with `REBUILD_ACCEPTED` evidence while the accepted `first` entry was
  absent after the failed resume checkpoint.

---

_Reviewed: 2026-09-12T00:23:53Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
