---
phase: 03-atomic-lifecycle-and-recovery-engine
fixed_at: 2026-09-01T23:18:00-04:00
review_path: .planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.md
iteration: 15
findings_in_scope: 4
fixed: 4
skipped: 0
status: all_fixed
---

# Phase 03: Code Review Fix Report

**Fixed at:** 2026-09-01T23:18:00-04:00  
**Source review:** `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.md`  
**Iteration:** 15

## Summary

- Findings in scope: 4
- Fixed: 4
- Skipped: 0
- Source/test commits: `afde03f` (`fix(03): bound lifecycle inventory and apply continuation`), `aafa79b` (`fix(03): preserve high-water clear cursors`), and `8a502a0` (`test(03): accept fail-closed clear preflight order`)

## Fixed Issues

### CR-01: Same-process initialization guard exceeded the configured deadline

**Files modified:** `src/cacheness/storage/integrity.py`, `tests/test_blob_store_integrity.py`

**Applied fix:** Establishes one monotonic deadline before local or kernel admission. The authority-scoped registry now uses nonblocking bounded acquisition, retires references on every timeout/error path, and passes the same absolute deadline to the POSIX/Win32 authority loop. Regressions cover a live same-process winner timeout, release before expiry, and zero retained guard state.

### CR-02: Total inventory ceilings denied valid stores and mixed evidence families

**Files modified:** `src/cacheness/storage/manifest_repository.py`, `src/cacheness/storage/operation_repository.py`, `src/cacheness/storage/path_security.py`, `src/cacheness/storage/clear_recovery.py`, `tests/test_manifest_repository_cas.py`, `tests/test_blob_store_reconciliation.py`

**Applied fix:** Replaces manifest lexical/capped scans with versioned high-water inventories for memory, JSON, and SQLite repositories. File lifecycle evidence now has separate primary and sidecar inventory families; filtered directory accounting no longer allows unrelated controls to exhaust another family's budget. Existing clear-journal snapshots validate and preserve the versioned manifest inventory field.

### CR-03: Mutable lexical cursors could skip pre-cursor insertions

**Files modified:** `src/cacheness/storage/manifest_repository.py`, `src/cacheness/storage/operation_repository.py`, `src/cacheness/storage/reconciliation.py`, `tests/test_manifest_repository_cas.py`

**Applied fix:** New cursors bind a monotonic inventory high-water mark and next sequence position. Later create/replace events are outside an existing snapshot; deleted/replaced source positions are consumed as stale rather than reclassified as new members. Resume tokens now carry version-4 sequence cursors. v1/v2/v3 cursors still decode, conservatively restarting their source under the new snapshot semantics instead of carrying unsafe lexical state forward.

**Follow-up correction:** Clear target pages now use an explicitly versioned v2 control schema. Their existing `source_cursor` and `next_cursor` key fields remain v1-compatible, while signed high-water and next-sequence fields preserve the full manifest cursor. Page IDs bind those fields for v2 pages. Manifest pages expose an exact post-entry cursor, allowing an encoded-size split to resume from the actual next sequence instead of reconstructing a mutable lexical position. v1 clear pages remain readable and retain their original IDs.

### CR-04: Apply advanced before exact revalidation outcomes were known

**Files modified:** `src/cacheness/storage/reconciliation.py`, `tests/test_blob_store_reconciliation.py`

**Applied fix:** Apply now records private structured outcomes for attempted/completed/conflicted/stale work and derives source continuation afterward. The earliest stale or unapplied primary/sidecar source is retained; actual repository attempts alone consume the action budget, while independent later work can still progress.

## Verification

All gates ran from the **main checkout** after the follow-up source/test commits.

- `.venv/bin/pytest -q tests/test_blob_store_atomic_lifecycle.py::test_clear_resume_does_not_repeat_completed_targets_after_reopen -o log_cli=false`: passed to terminal completion. This was the previously deterministic hang.
- `.venv/bin/pytest -q tests/test_blob_store_atomic_lifecycle.py -o log_cli=false`: passed to terminal completion.
- `.venv/bin/pytest -q tests/test_blob_store_integrity.py tests/test_manifest_repository_cas.py tests/test_blob_store_reconciliation.py tests/test_clear_recovery.py tests/test_blob_store_atomic_lifecycle.py -o log_cli=false`: passed to terminal completion.
- `.venv/bin/pytest -q -o log_cli=false`: passed to terminal completion under normal host permissions (expected optional-platform skips only).
- `.venv/bin/python -m py_compile` passed for changed storage modules and touched tests.
- CPython 3.11.16 `compileall` and package import with the declared `recommended` group passed.
- Changed-path Ruff passed for each follow-up change.
- `uv lock --check` and `git diff --check` passed.

The regression uses `manifest_page_size=1`, records the exact cursor sequence `(None, None, None) → ("first", 2, 2)`, and has a hard page-call budget. The adjacent prepared-clear interruption asserts that no second high-water page is read after the fault. A repeat of the first page now fails deterministically rather than looping.

## Residuals

- Native Windows remains unavailable on this host; injected nonblocking Win32 adapter coverage remains the available local evidence.
- Indexes are lifecycle scheduling data, not alternate authority: every yielded manifest/evidence byte remains re-read, hash-bound, and authenticated/validated by the existing lifecycle path before it can authorize an action.

---

_Fixed: 2026-09-01T23:18:00-04:00_  
_Fixer: the agent (gsd-code-fixer)_  
_Iteration: 15_
