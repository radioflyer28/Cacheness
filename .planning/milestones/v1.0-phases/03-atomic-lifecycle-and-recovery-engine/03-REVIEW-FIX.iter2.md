---
phase: 03
fixed_at: 2026-08-31T13:31:39Z
review_path: /Users/akriz/code/cacheness/.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.md
iteration: 1
findings_in_scope: 7
fixed: 7
skipped: 0
status: all_fixed
---

# Phase 03: Code Review Fix Report

**Fixed at:** 2026-08-31T13:31:39Z  
**Source review:** `03-REVIEW.md`  
**Iteration:** 1

**Summary:**

- Findings in scope: 7
- Fixed: 7
- Skipped: 0

## Fixed Issues

### CR-01: Exact evidence transitions were only process-local

**Files modified:** `operation_repository.py`, `test_blob_store_reconciliation.py`  
**Commit:** `6905569`  
**Applied fix:** Exact checkpoint, retirement, reconciliation, and clear-control
transitions now take narrow root-scoped advisory locks in addition to shared
in-process stripes. Independent repository instances race on identical evidence
in a deterministic regression test and have exactly one CAS winner.

### CR-02: Clear admission did not establish a cross-process snapshot

**Files modified:** `coordination.py`, `lifecycle.py`, `blob_store.py`, `test_blob_store_concurrency.py`  
**Commit:** `6905569`  
**Applied fix:** Ordinary and aggregate admission now use a root-scoped POSIX
shared/exclusive lock. Lifecycle initialization recovers under aggregate
admission, and clear continuation has a narrow per-operation lease, preventing
an independent process from resuming an active clear. The regression test holds
an exact clear snapshot while a spawned process attempts a later write; the
write survives clear.

### CR-03: Valid large clear snapshots could exceed the page codec bound

**Files modified:** `operation_record.py`, `operation_repository.py`, `lifecycle.py`, `test_blob_store_atomic_lifecycle.py`  
**Commit:** `6905569`  
**Applied fix:** Clear inventory pages are admitted by encoded byte size. A
valid manifest that cannot fit inline is stored as bounded immutable exact-byte
sidecar evidence, referenced and digest-bound by the signed page, then resolved
and authenticated before deletion. Four 200 KiB manifests clear and reopen
without page-codec failure.

### CR-04: Lifecycle codecs and repository reads bypassed caller limits

**Files modified:** `path_security.py`, `operation_record.py`, `operation_repository.py`, `lifecycle.py`, `reconciliation.py`, `test_blob_store_reconciliation.py`  
**Commit:** `6905569`  
**Applied fix:** The same caller-owned `LifecycleLimits` now flow through
operation, clear-page, clear-checkpoint, and reconciliation codecs. Repository
reads check descriptor size and read at most `max + 1` bytes before parsing;
the regression test proves an over-limit evidence file is rejected before the
legacy unbounded reader can run.

### CR-05: Reconciliation cursors used deterministic reversible masking

**Files modified:** `reconciliation.py`, `pyproject.toml`, `uv.lock`, `test_blob_store_reconciliation.py`  
**Commit:** `6905569`  
**Applied fix:** Resume cursors now use fresh-nonce ChaCha20-Poly1305 AEAD with
a domain-separated HMAC-derived key. Encoded and decoded bounds are checked
before base64/JSON processing; tests cover nonce uniqueness, round-trip,
tampering, and oversized-token rejection before decode.

### CR-06: `exists` and `list` did not use the typed bounded read contract

**Files modified:** `blob_store.py`, `test_blob_store_concurrency.py`, `test_blob_store_read_contract.py`  
**Commit:** `6905569`  
**Applied fix:** `exists` now performs M1/snapshot/M2 with one proven-generation
retry and a typed exhaustion conflict. `list` pages manifests and turns a
selected disappearance/change into `CacheBlobLifecycleConflictError` rather
than an assertion or silent omission.

### WR-01: Successful clear left page/checkpoint control evidence behind

**Files modified:** `operation_repository.py`, `lifecycle.py`, `test_blob_store_atomic_lifecycle.py`  
**Commit:** `6905569`  
**Applied fix:** Terminal clear retires authenticated completed checkpoints,
sidecar references, and pages using exact CAS; it retires the main operation
record last. Missing artifacts are idempotent completion only after the signed
terminal evidence authorizes the cleanup. An injected interruption after
control retirement reopens and finishes cleanup exactly.

## Verification

Verification ran in the **isolated review-fix worktree**
`/Users/akriz/code/cacheness/.claude/worktrees/rf-03-78561-1788181012`.

- `ruff check` on every changed source and test file: passed.
- Python syntax compilation of every changed source module: passed.
- Phase 3 lifecycle gate: passed (one expected Windows-only containment skip).
- `uv run pytest -q -o log_cli=false`: completed through 93% without failures
  before the command environment's 30-second output limit; the remaining final
  test modules were then run separately and passed. The pre-limit full-suite
  failure in legacy migration tests was fixed by ensuring legacy fixtures fail
  before an admission sidecar is created.

---

_Fixed: 2026-08-31T13:31:39Z_  
_Fixer: the agent (gsd-code-fixer)_  
_Iteration: 1_
