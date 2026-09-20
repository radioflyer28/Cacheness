---
phase: 03-atomic-lifecycle-and-recovery-engine
fixed_at: 2026-09-02T20:15:11Z
review_path: .planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.md
iteration: 21
archived_as: iteration-21-output
findings_in_scope: 3
fixed: 3
skipped: 0
status: all_fixed
---

# Phase 03: Code Review Fix Report

**Fixed at:** 2026-09-02T20:15:11Z  
**Source review:** `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.md`  
**Iteration:** 21

**Summary:**

- Findings in scope: 3
- Fixed: 3
- Skipped: 0

## Fixed Issues

### CR-01: Sibling head/marker ambiguity

**Files modified:** `src/cacheness/storage/blob_store.py`, `src/cacheness/storage/lifecycle.py`, `src/cacheness/storage/operation_repository.py`, `tests/test_blob_store_reconciliation.py`, `tests/test_manifest_repository_cas.py`

**Commits:** `42bdbe2`, `842ebc9`, `0b47d7c`, `3b83e6c`, `4b0c034`

**Applied fix:** Added signed v3 all-family initialization provenance, serialized it with every family’s first publication, and treat unsigned marker/head crash shapes as typed migration states. Constructor recovery now performs a read-only all-family compatibility classification before maintenance can create locks or inspect absent-family state. Existing provenance is verified only with a strict key read; a missing key cannot be replaced during verification and recovery remains non-mutating until authentication is possible.

### CR-02: Repeated post-authority manifest compaction failures

**Files modified:** `src/cacheness/storage/manifest_repository.py`, `tests/test_blob_store_reconciliation.py`, `tests/test_manifest_repository_cas.py`

**Commits:** `67c5b49`, `95c4077`, `0b47d7c`

**Applied fix:** Replaced live-event rehoming with durable, bounded sparse-successor markers and exact compaction debt. Original event sequence identities and old high-water cursors are preserved; pages jump proven stale runs only after marker and current-event validation. Malformed markers fail closed, normal reads do not mutate state, and recovery advances a bounded durable continuation across memory/JSON reopens.

### CR-03: Aggregate clear/pinned floor and discarded primary continuation

**Files modified:** `src/cacheness/storage/lifecycle.py`, `src/cacheness/storage/operation_repository.py`, `tests/test_blob_store_reconciliation.py`

**Commits:** `0d53e85`, `107455a`, `cb9544d`, `3b83e6c`

**Applied fix:** Persisted bounded operation-inventory maintenance targets and live floors, advancing them through authenticated exact revalidation after terminal lifecycle boundaries and during mutating recovery. Clear continuation remains serialized across constructors without granting aggregate lease authority to child records; the clear lease has its own reentrancy state so each child retains independent exact-CAS exclusion. Large clears and pinned operations therefore converge without permanently stranding stale sparse slots.

## Verification

All commands below ran from the isolated review-fix worktree on the normal host, except the explicitly temporary Python 3.11 environment. The shared main `.venv` was not modified.

- Targeted provenance, compaction, cursor, and clear-race regressions: passed.
- `tests/test_blob_store_reconciliation.py`: passed.
- Full Phase 3 selection (the 11 lifecycle, reconciliation, integrity, concurrency, containment, configuration, CAS, and public-contract modules): passed, with 2 expected platform skips (Windows junction and device-node fixtures).
- Full repository suite: passed, with expected optional PostgreSQL/TensorFlow and platform skips; its only warning was the pre-existing `TestDataClassForConsistency` collection warning.
- Changed-file Ruff, Python AST parsing, and `git diff --check`: passed.
- Python 3.11.16 import smoke test in a temporary recommended-dependency environment: `import cacheness` passed (`0.3.14`).

---

_Fixed: 2026-09-02T20:15:11Z_  
_Fixer: the agent (gsd-code-fixer)_  
_Iteration: 21_
