---
phase: 03
fixed_at: 2026-09-04T00:18:41Z
review_path: .planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.md
iteration: 25
archived_as: iteration-25-output
findings_in_scope: 2
fixed: 2
skipped: 0
status: all_fixed
---

# Phase 03: Code Review Fix Report

**Fixed at:** 2026-09-04T00:18:41Z  
**Source review:** `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.md`  
**Iteration:** 25

## Summary

- Findings in scope: 2
- Fixed: 2
- Skipped: 0

## Fixed Issues

### CR-01: Accepted inventory heads did not bind their terminal event at read or append time

**Files modified:** `src/cacheness/storage/manifest_repository.py`, `src/cacheness/storage/operation_repository.py`, `tests/test_manifest_repository_cas.py`  
**Commit:** `8035847`

Every accepted manifest tail now verifies its exact signed terminal event before a
page is accepted or a subsequent append advances it. The same check applies to
the operation repository's acknowledgement chain for primary, sidecar, and
pending control records. Missing, malformed, replayed, or differently signed
terminals fail closed after reopening; direct evidence reads remain available.
Compacted manifest terminals require an authenticated witness bound to the
terminal tuple and exact skip-marker proof.

### CR-02: Ordinary scheduler appends serialized distinct keys behind durable I/O

**Files modified:** `src/cacheness/storage/operation_repository.py`, `tests/test_blob_store_concurrency.py`, `tests/test_blob_store_reconciliation.py`, `tests/test_manifest_repository_cas.py`  
**Commit:** `8035847`

Normal inventory scheduling no longer holds the initialization or family
transition lease across event, tail, or head persistence. A current-writer
epoch publishes immutable sequence-specific event, tail-receipt, and
head-receipt records; collision recovery validates and helps the existing
winner before retrying. Each receipt chain authenticates the exact terminal
tuple, so an accepted later receipt cannot cover a gap. Bounded receipt
discovery is charged to 125 control records of at most 4 KiB each, and signed
retirement anchors compact old receipt records in persisted bounded windows.

Public same-process and independent-process probes pause a writer at event,
tail, and head durability boundaries and verify that an unrelated key completes
before the paused writer resumes, including reopen/recovery paths.

## Verification

All commands ran in the isolated `rf-03-iter25` worktree; the full-suite rerun
used normal-host permissions so the repository's Unix-socket and existing
`uv`/Ruff-cache checks could run.

- `git diff --check` — passed
- Changed-file `py_compile` and `ruff check` — passed
- Focused Phase 3 lifecycle suite (manifest/blob manifest/blob store/clear
  recovery/CAS/concurrency/reconciliation contracts) — passed
- Python 3.11.16 `compileall -q src/cacheness` — passed
- Normal-host `pytest -q -o log_cli=false` — passed (expected optional-service
  and platform skips only)

---

_Fixed: 2026-09-04T00:18:41Z_  
_Fixer: the agent (gsd-code-fixer)_  
_Iteration: 25_
