---
phase: 03-atomic-lifecycle-and-recovery-engine
fixed_at: 2026-09-02T22:48:00Z
review_path: .planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.md
iteration: 23
archived_as: iteration-23-output
findings_in_scope: 3
fixed: 3
skipped: 0
status: all_fixed
---

# Phase 03: Code Review Fix Report

**Fixed at:** 2026-09-02T22:48:00Z  
**Source review:** `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.md`  
**Iteration:** 23

## Summary

- Findings in scope: 3
- Fixed: 3
- Skipped: 0

## Fixed Issues

### CR-01: Signed head rollback can hide later immutable inventory events

**Files modified:** `src/cacheness/storage/manifest_repository.py`,
`src/cacheness/storage/operation_repository.py`,
`tests/test_manifest_repository_cas.py`  
**Commit:** `709a03f`

**Applied fix:** Added bounded, independently signed append-tail anchors for
manifest inventory and each operation family (primary, sidecar, pending). A
non-empty signed head must match its store/family/epoch-bound tail high water;
head-only replay, tail-only replay, malformed tails, and tail/head partial
commits fail closed for pages, recovery, and reconciliation. Only the writer's
strict one-event append-resumption path accepts a tail exactly one sequence
ahead of its head, then acknowledges it. JSON collision recovery rebuilds the
current sequence-bound event before its head acknowledgement.

**Trust boundary:** Under D-21, Cacheness detects substitution or rollback of a
member of the authenticated live scheduling control set. A local principal that
can deliberately restore a mutually consistent earlier set of *all* durable
control members (head and tail) is outside this filesystem-only trust boundary;
distinguishing that case requires an external non-rollbackable monotonic
authority. The tail records remain store-, epoch-, and family-bound so copied
or partially replayed controls fail closed without filename probing or an
unbounded scan.

### CR-02: Missing allocated events were silently treated as compacted gaps

**Files modified:** `src/cacheness/storage/manifest_repository.py`,
`src/cacheness/storage/operation_repository.py`,
`tests/test_manifest_repository_cas.py`,
`tests/test_blob_store_reconciliation.py`  
**Commit:** `709a03f`

**Applied fix:** Manifest list and compaction now reject a missing allocated
event unless the sequence has an authenticated manifest sparse-successor proof.
Valid markers are consumed directly during recovery, preserving the existing
write-marker-before-unlink crash protocol. Operation inventories now fail
closed on any missing allocated primary, sidecar, or pending event and retain
stale immutable pending events rather than deleting them without a signed skip
proof. Signed floors continue to bound normal paging without mutating reads.

**Coverage:** first/middle/last deletion for manifest and all operation
families; JSON reopen; old cursor traversal of a still-pending missing member;
valid marker-based clear/recovery; stale pending compaction; bounded
reconciliation and clear recovery regressions.

### CR-03: JSON event collision could publish bytes signed for the wrong sequence

**Files modified:** `src/cacheness/storage/manifest_repository.py`,
`tests/test_manifest_repository_cas.py`  
**Commits:** `709a03f`, `07ebdda`

**Applied fix:** JSON manifest publication now reconstructs and signs the event
inside every exclusive-create retry. It validates the colliding immutable event:
an exact matching key/digest is acknowledged at that sequence, while an
unrelated event is acknowledged as stale and the intended record is signed at a
fresh sequence. Tests cover one/multiple collision paths, matching and
unrelated unacknowledged events, and later page visibility.

## Post-fix concurrency regression

**Files modified:** `src/cacheness/storage/operation_repository.py`,
`tests/test_blob_store_reconciliation.py`  
**Commit:** `968e615`

The first tail-protected scheduling gate exposed a real lock-order cycle in
clear-vs-put races: a clear-resume lease shared the in-process stripe namespace
with ordinary inventory transitions, and checkpoint scheduling could acquire
inventory stripes while holding a record-CAS stripe. The fix gives the separate
durable clear-resume lock its own bounded in-process lock namespace and
publishes prospective primary/sidecar scheduling members before the exact
record-CAS lease. A contender that subsequently loses CAS leaves only a
digest-bound stale event, which paging already revalidates as non-authoritative.
Deterministic tests force the formerly colliding lock identities and assert that
both clear/resume and inventory append complete; another test verifies that
neither checkpoint path nests inventory scheduling under a record lease.

## Verification

Verification ran from the isolated review-fix worktree. The shared main
`.venv` was used read-only with `PYTHONPATH=src`; no shared environment was
modified. Python 3.11 used a temporary environment.

- `tests/test_manifest_repository_cas.py`: passed, 111 tests.
- `tests/test_blob_store_atomic_lifecycle.py`: passed, 73 tests.
- `tests/test_blob_store_reconciliation.py`: passed, 114 tests.
- Concurrency and close tests: passed, 32 tests.
- Manifest/read/integrity/containment tests: passed with expected platform skips.
- `tests/test_clear_recovery.py`: passed; both prior race cases passed 20 repeated independent-process attempts after `968e615`.
- Full Phase 3 suite: passed outside the workspace sandbox for the Unix-domain-socket fixture.
- Literal normal-host suite: `pytest -q -o log_cli=false` exited 0.
- Compatibility corpus through `sqlite-columns-v0314`: passed.
- Changed-file Ruff, Python AST parsing, and `git diff --check`: passed.
- Python 3.11.16 import: passed (`cacheness 0.3.14`).

---

_Fixed: 2026-09-02T22:48:00Z_  
_Fixer: the agent (gsd-code-fixer)_  
_Iteration: 23_
