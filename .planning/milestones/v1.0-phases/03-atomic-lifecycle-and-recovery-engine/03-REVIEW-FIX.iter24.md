---
phase: 03-atomic-lifecycle-and-recovery-engine
fixed_at: 2026-09-02T21:10:00Z
review_path: .planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.md
iteration: 22
archived_as: iteration-22-output
findings_in_scope: 3
fixed: 3
skipped: 0
status: all_fixed
---

# Phase 03: Code Review Fix Report

**Fixed at:** 2026-09-02T21:10:00Z  
**Source review:** `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.md`  
**Iteration:** 22

## Fixed Issues

### CR-01: Store-bound initialization provenance

**Files modified:** `src/cacheness/storage/operation_repository.py`, `tests/test_manifest_repository_cas.py`  
**Commit:** `7874902`

**Applied fix:** Version-5 initialization evidence has a stable store identity,
topology-bound family provenance, and a random epoch. Replayed markers, heads,
and raw v1 crash shapes fail closed; v3/v4 records are explicit migration
evidence rather than a new-store signal.

### CR-02: Authenticated lifecycle scheduling

**Files modified:** `src/cacheness/storage/blob_store.py`,
`src/cacheness/storage/manifest_repository.py`,
`src/cacheness/storage/operation_repository.py`,
`tests/test_blob_store_read_contract.py`,
`tests/test_blob_store_reconciliation.py`,
`tests/test_manifest_repository_cas.py`  
**Commits:** `7532e3d`, `0173a42`, `d0d5c99`, `492b5b2`, `6765596`

**Applied fix:** Manifest and operation scheduling heads/events now carry an
HMAC over store identity, family, epoch, sequence/range, and exact evidence.
Sparse successor markers commit a chained digest of the exact revalidated
skipped run. Old cursors, reopens, copied heads, primary/sidecar/pending
records, substitution, and two-repository concurrent compaction are covered.
Normal pages cache only a verified epoch and do not mutate storage.

### CR-03: Tombstone delete-operation binding

**Files modified:** `src/cacheness/storage/lifecycle.py`,
`tests/test_blob_store_atomic_lifecycle.py`  
**Commit:** `0415335`

**Applied fix:** Tombstones carry a signed direct delete-operation reference.
Recovery authenticates and follows that exact record rather than scanning
unrelated operation inventory, preserving bounded work and distinguishing
evidence absence from an exhausted search budget.

## Verification

All checks ran from the isolated review-fix worktree. The shared main `.venv`
was used read-only; the Python 3.11 smoke test used a separate temporary UV
environment.

- Focused manifest scheduling/compaction tests: passed.
- Full Phase 3 selection: passed in complete file-group runs, including the
  new concurrent-compaction test and 20 repetitions of the timing-sensitive
  same-instance JSON clear/put race.
- Compatibility corpus through `sqlite-columns-v0314`: passed.
- Full repository suite: passed on the normal host (`pytest -q -o
  log_cli=false`), with expected platform, PostgreSQL, and TensorFlow skips
  plus the existing dataclass collection warning.
- Changed-file Ruff, Python AST parsing, and `git diff --check`: passed.
- Python 3.11.16 temporary-environment import: passed (`cacheness 0.3.14`).

---

_Fixed: 2026-09-02T21:10:00Z_  
_Fixer: the agent (gsd-code-fixer)_  
_Iteration: 22_
