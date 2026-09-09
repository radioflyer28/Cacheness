---
phase: 03-atomic-lifecycle-and-recovery-engine
reviewed: 2026-09-02T16:58:30Z
depth: deep
head: 02d3837
files_reviewed: 35
files_reviewed_list:
  - docs/SECURITY.md
  - docs/WINDOWS_COMPATIBILITY.md
  - pyproject.toml
  - uv.lock
  - src/cacheness/config.py
  - src/cacheness/error_handling.py
  - src/cacheness/metadata.py
  - src/cacheness/serialization.py
  - src/cacheness/storage/__init__.py
  - src/cacheness/storage/blob_store.py
  - src/cacheness/storage/clear_recovery.py
  - src/cacheness/storage/coordination.py
  - src/cacheness/storage/guarded_handler_io.py
  - src/cacheness/storage/integrity.py
  - src/cacheness/storage/lifecycle.py
  - src/cacheness/storage/manifest.py
  - src/cacheness/storage/manifest_repository.py
  - src/cacheness/storage/operation_record.py
  - src/cacheness/storage/operation_repository.py
  - src/cacheness/storage/path_security.py
  - src/cacheness/storage/reconciliation.py
  - tests/test_blob_store_atomic_lifecycle.py
  - tests/test_blob_store_close_contract.py
  - tests/test_blob_store_concurrency.py
  - tests/test_blob_store_integrity.py
  - tests/test_blob_store_read_contract.py
  - tests/test_blob_store_reconciliation.py
  - tests/test_blob_manifest.py
  - tests/test_blob_manifest_backends.py
  - tests/test_clear_recovery.py
  - tests/test_config_validation.py
  - tests/test_filesystem_containment.py
  - tests/test_manifest_repository_cas.py
  - tests/test_phase1_quality_gates.py
  - tests/test_public_api_contract.py
findings:
  critical: 1
  warning: 1
  info: 0
  total: 2
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-09-02T16:58:30Z  
**Depth:** deep  
**HEAD:** `02d3837`  
**Status:** issues_found

## Summary

This iteration independently rechecked the complete Phase 3 lifecycle and the
iteration-17 fixes. The zero-target clear bridge is signed, page-bound, grants
no target authority, and traverses correctly after a post-snapshot interruption.
The new before-first primary and sidecar cursors remain generation-bound and
eventually converge after first-source conflicts. Current record read failures
remain visible through operation paging and compaction, and the pending
`first_live_sequence` floor is durable, conservative for old heads/cursors, and
bounded under the tested stale-prefix/reopen paths.

Two actionable issues remain. Manifest compaction can still destroy the only
scheduling event for malformed live authority and make reconciliation report a
false terminal result. Separately, an inventory-event filesystem failure escapes
the public API as a raw `OSError` instead of a Cacheness backend exception.

## Critical Issues

### CR-01: Manifest compaction erases malformed live authority from reconciliation

**Files:** `src/cacheness/storage/manifest_repository.py:692-717`,
`src/cacheness/storage/manifest_repository.py:842-917`

**Issue:** The page reader now correctly distinguishes absent authority from an
invalid current projection and fails closed. The compactor does not. In
`_compact_inventory_window()`, `_raw_from_entry()` failures (`TypeError`,
`ValueError`, or migration-required projection state) are converted to
`current = None`; the event is then deleted as though absence had been proven.
Compaction runs after an unrelated overwrite or removal, so malformed authority
can silently lose its only route into every later bounded inventory page.

A bounded deterministic probe reproduced this for both `InMemoryBackend` and
`JsonBackend`:

1. Put `broken` and `other` (inventory sequences 1 and 2).
2. Change only `broken.metadata.canonical_manifest_v1` to invalid base64.
3. Overwrite `other`, which appends sequence 3 and runs compaction.
4. Observe sequence 1 removed while the `broken` entry remains.
5. Reconcile and receive `resume_token=None` with only
   `manifest_has_no_reconciliation_debt` for `other`; `broken` is not inspected.

Calling `get_raw("broken")` before step 3 raises the expected typed backend
error, proving that compaction—not the page decoder—is what converts the defect
to false absence. SQLite does not use this compatibility projection path: its
manifest compactor compares opaque bytes in the canonical manifest table.

This violates the fail-closed integrity boundary and STOR-06. A clean terminal
token is materially false while current authority remains unreadable, so the
finding is Critical.

**Fix:** Never delete a manifest scheduling event when decoding the current
projection failed or migration is required. Preserve that event and continue
bounded compaction so the next page deterministically raises the typed failure;
do not propagate a post-publication maintenance error from an otherwise
successful unrelated write. Delete only when current absence is proven or when
a successfully decoded current record has a different digest. Add memory and
JSON regressions for unrelated overwrite and removal, reopen, small compaction
windows, later valid events, and repeated reconciliation, asserting the invalid
authority can never become a terminal clean report. Retain SQLite coverage to
show its opaque canonical-table path is unaffected.

## Warnings

### WR-01: Operation inventory-event I/O failures leak raw filesystem exceptions

**Files:** `src/cacheness/storage/operation_repository.py:756-787`,
`src/cacheness/storage/operation_repository.py:897-962`

**Issue:** `_read_inventory_event()` handles `FileNotFoundError` as an intentional
sparse gap and translates malformed JSON, but it does not translate other
`OSError` failures from `ManagedFileOps.read_bytes_bounded()`. `_inventory_page()`
has no repository exception boundary around that call. Injecting a
`PermissionError` for an indexed primary event made public
`BlobStore.reconcile()` raise raw `PermissionError: injected inventory event
denial` with no Cacheness cause/context. The analogous JSON manifest page
correctly raises `CacheBlobBackendError` with the original `PermissionError` as
its cause. A bounded-read `ValueError` for an oversized/corrupt event can leak
through the same seam.

The operation fails rather than reporting false cleanliness, so this is not an
integrity bypass. It nevertheless violates the documented backend-neutral error
contract and prevents callers from handling storage failures uniformly.

**Fix:** In `_read_inventory_event()`, keep `FileNotFoundError` as the only sparse
gap, translate other `OSError` to `CacheBlobBackendError` with operation,
family, and sequence context while preserving the cause, and translate the
bounded-size failure to the appropriate typed bounds/backend error. Test
primary, sidecar, and pending page reads plus compaction through both repository
and public reconciliation entry points; assert exact type, reason/context,
cause, no event deletion, and no terminal report.

## Verification Performed

- Deterministic memory and JSON compaction probes reproduced CR-01 and confirmed
  the malformed entry survives after its scheduling event is removed.
- Deterministic primary-event `PermissionError` injection reproduced WR-01;
  the equivalent manifest-event probe confirmed correct typed translation.
- Eleven focused tests covering empty clear bridges, prepared-clear authority,
  chunked references, first-sidecar conflict continuation, current read
  failures, pending recovery paging, and lifetime-prefix compaction passed.
- A bounded post-snapshot interruption probe with a leading stale-only manifest
  window reopened and completed clear on JSON and SQLite with no control residue.
- A sole first-primary action conflict emitted an authenticated before-first
  cursor (`high_water=1`, `next_sequence=1`) and converged without repeating a
  destructive delete; the existing sole-sidecar equivalent also passed.
- Relevant changed-path Ruff checks passed. The orchestrator already has a clean
  full-suite run at this HEAD, so it was not duplicated in this iteration.
- `uv lock --check` could not initialize the sandboxed user cache in this agent;
  this is an observation limitation, not a product finding.

No further actionable reliability, concurrency, durability, compatibility, or
performance issue was found in the bounded adjacent checks. Known unrelated
repository-wide lint debt, shutdown-only destructor behavior, and base-install
NumPy packaging were excluded.

---

_Reviewer: independent deep rereview, iteration 18_
