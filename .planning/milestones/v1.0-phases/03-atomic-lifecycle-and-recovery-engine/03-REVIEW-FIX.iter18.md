---
phase: 03-atomic-lifecycle-and-recovery-engine
fixed_at: 2026-09-02T04:42:30Z
review_path: .planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.md
iteration: 16
findings_in_scope: 5
fixed: 5
skipped: 0
status: all_fixed
---

# Phase 3: Code Review Fix Report

**Fixed at:** 2026-09-02T04:42:30Z  
**Source review:** `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.md`  
**Iteration:** 16

## Summary

- Findings in scope: 5
- Fixed: 5
- Skipped: 0

The fixes preserve D-21/D-22: scheduling inventories remain non-authoritative,
all action paths revalidate exact current bytes, and the Windows lock contract is
explicitly one-user/session with required nonblocking acquisition semantics.

## Fixed Issues

### CR-01: Clear cannot consume valid high-water inventory order or stale-only pages

**Files modified:** `src/cacheness/storage/lifecycle.py`

**Commit:** `a78c30f`

**Applied fix:** Clear target pages canonicalize target keys separately from their
signed source cursor and advance across valid empty nonterminal manifest windows.
This retains publication high-water order while satisfying durable target-page
ordering and restart semantics.

### CR-02: Reconciliation continuation loops on partial and empty high-water pages

**Files modified:** `src/cacheness/storage/reconciliation.py`,
`src/cacheness/storage/manifest_repository.py`,
`src/cacheness/storage/operation_repository.py`,
`tests/test_blob_store_reconciliation.py`

**Commits:** `a78c30f`, `33fedde`

**Applied fix:** Manifest, primary, sidecar, and pending pages now carry exact
post-entry sequence cursors. Empty nonterminal pages advance to their page
cursor; partial consumption retains the last consumed source cursor rather than
reconstructing a lexical position. Pending-recovery checkpoints persist the v2
high-water/sequence continuation.

### CR-03: The inventories remain whole-history, permanently capped, and incomplete for legacy/pending evidence

**Files modified:** `src/cacheness/storage/manifest_repository.py`,
`src/cacheness/storage/operation_repository.py`,
`src/cacheness/storage/path_security.py`,
`tests/test_manifest_repository_cas.py`,
`tests/test_blob_store_reconciliation.py`

**Commits:** `a78c30f`, `33fedde`

**Applied fix:** Operation primary, sidecar, and pending scheduling now use a
durable v2 fixed-size head plus immutable sparse sequence events. JSON manifest
inventories use the same external durable design; in-memory uses bounded sparse
state and SQLite compacts stale sequence slots through backend-native tables.
Compaction deletes only digest-mismatched/absent events and keeps sparse sequence
positions valid for existing high-water tokens. Legacy v1/pre-index evidence is
either safely bootstrapped for bounded pending controls or rejected with
`CacheBlobMigrationRequiredError`; physical legacy directory inspection now has
an independent bound, including unrelated entries.

### CR-04: An exact apply conflict can return a terminal token while safe debt remains

**Files modified:** `src/cacheness/storage/reconciliation.py`,
`tests/test_blob_store_reconciliation.py`

**Commit:** `a78c30f`

**Applied fix:** A reached exact CAS consumes action budget but no longer implies
completed source progression. Conflict handling re-reads and reclassifies exact
primary/sidecar bytes, retains a continuation whenever debt remains, and still
allows later independently authenticated actions to proceed.

### CR-05: Both key-admission loops can acquire after the absolute deadline

**Files modified:** `src/cacheness/storage/integrity.py`,
`src/cacheness/storage/coordination.py`,
`tests/test_blob_store_integrity.py`,
`tests/test_blob_store_close_contract.py`,
`tests/test_blob_store_concurrency.py`

**Commits:** `a78c30f`, `0bb710d`

**Applied fix:** Both admission loops allow only their first immediate
nonblocking probe; every retry checks the shared absolute deadline before lock
acquisition. Win32 adapters that cannot accept nonblocking acquisition are
rejected with the capability error rather than falling back to a blocking call.
Regression tests cover a fake release-after-expiry clock, refcount cleanup, and
the required adapter contract.

## Verification

All focused commands below completed in the main checkout with terminal exit 0:

- `.venv/bin/pytest -q -o log_cli=false tests/test_blob_store_reconciliation.py -x` — 65 passed.
- `.venv/bin/pytest -q -o log_cli=false tests/test_blob_store_atomic_lifecycle.py -x` — 69 passed.
- `.venv/bin/pytest -q -o log_cli=false tests/test_manifest_repository_cas.py -x` — 59 passed.
- `.venv/bin/pytest -q -o log_cli=false tests/test_blob_store_integrity.py -x` — 35 passed.
- `.venv/bin/pytest -q -o log_cli=false tests/test_blob_store_close_contract.py tests/test_blob_store_concurrency.py tests/test_manifest_repository_cas.py tests/test_blob_store_integrity.py -x` — 126 passed.
- `uv run --isolated --python 3.11 --extra recommended python -c "import cacheness; print(cacheness.__version__)"` — imported successfully on Python 3.11.
- `uv lock --check` — passed.
- Changed-path `.venv/bin/ruff check ...` — passed.
- `git diff --check` — passed.
- Normal-host `uv run pytest -q -o log_cli=false` — terminal exit 0. The suite retained its documented optional-environment skips and one existing pytest collection warning.

An earlier sandboxed full-suite attempt exited 1 only because the sandbox denied
the Unix-socket fixture and access to the UV cache. It is not used as passing
evidence; the normal-host run above is the authoritative full-suite result.

## Residuals

None known for the five Iteration 16 findings. The report itself is intentionally
left uncommitted for the review-fix orchestrator to record.

---

_Fixed: 2026-09-02T04:42:30Z_  
_Fixer: the agent (gsd-code-fixer)_  
_Iteration: 16_
