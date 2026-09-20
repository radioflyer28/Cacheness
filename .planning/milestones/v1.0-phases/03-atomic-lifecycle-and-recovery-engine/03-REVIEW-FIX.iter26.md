---
phase: 03-atomic-lifecycle-and-recovery-engine
fixed_at: 2026-09-02T23:20:47Z
review_path: .planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.md
iteration: 24
archived_as: iteration-24-output
findings_in_scope: 2
fixed: 2
skipped: 0
status: all_fixed
---

# Phase 03: Code Review Fix Report

**Fixed at:** 2026-09-02T23:20:47Z  
**Source review:** `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.md`  
**Iteration:** 24

## Summary

- Findings in scope: 2
- Fixed: 2
- Skipped: 0

## Fixed Issues

### CR-01: Tail-ahead recovery replaces a missing committed event with a different member

**Files modified:** `src/cacheness/storage/manifest_repository.py`,
`src/cacheness/storage/operation_repository.py`,
`tests/test_manifest_repository_cas.py`  
**Commit:** `7b70a94`

**Applied fix:** Version-2 append tails bind the terminal sequence to its
immutable key/name and digest. Before any exclusive create, a one-event-ahead
tail must authenticate and match its already allocated terminal event; missing,
malformed, or mismatched members fail closed rather than being replaced. This
covers JSON manifest inventories and primary, sidecar, and pending operation
families across reopen and a later append.

### CR-02: Stale speculative events can permanently starve primary lifecycle recovery

**Files modified:** `src/cacheness/storage/operation_repository.py`,
`src/cacheness/storage/lifecycle.py`,
`tests/test_blob_store_reconciliation.py`  
**Commit:** `7b70a94`

**Applied fix:** The authenticated primary inventory head now persists bounded
recovery continuation (`recovery_high_water` and `recovery_next_sequence`).
Each caller-bounded recovery pass exact-revalidates the next fixed-locator page
and advances it durably, then resets only after the captured high water is
exhausted. It does not compact an event whose exact record publication remains
in flight; normal reads remain mutation-free and take no global lock.

**Coverage:** With page, inventory, and action limits all set to one, fresh
reopens advance past multiple stale predecessors and converge the live record.
A concurrent publication test advances a stale prefix while record publication
is held, then proves the next recovery page reaches the live record safely.

## Verification

Verification ran from the isolated review-fix worktree. The shared main `.venv`
was used read-only with `PYTHONPATH=src`; no shared environment was modified.

- Focused append-tail regressions: passed (`15` manifest-tail and `24`
  operation-tail selected tests).
- Focused reconciliation continuation regressions: passed (`2` tests).
- `tests/test_manifest_repository_cas.py` and
  `tests/test_blob_store_reconciliation.py`: passed together.
- Full Phase 3 lifecycle suite: passed on the normal host. The in-sandbox
  attempt was blocked only by the Unix-domain-socket fixture being denied a
  bind; normal-host execution passed with expected Windows junction and
  device-node skips.
- Literal full normal-host suite: `pytest -q -o log_cli=false` exited `0`.
  Expected optional-platform/PostgreSQL/TensorFlow skips and the existing
  dataclass collection warning remain.
- Compatibility corpus: passed through `sqlite-columns-v0314`.
- Python 3.11.16 import: passed against this worktree's `src`.
- Changed-file Ruff, Python AST parsing, and `git diff --check`: passed.

---

_Fixed: 2026-09-02T23:20:47Z_  
_Fixer: the agent (gsd-code-fixer)_  
_Iteration: 24_
