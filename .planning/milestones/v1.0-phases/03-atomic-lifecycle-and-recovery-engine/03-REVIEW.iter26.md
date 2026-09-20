---
phase: 03-atomic-lifecycle-and-recovery-engine
reviewed: 2026-09-04T02:59:41Z
depth: deep
iteration: 27
archived_as: 03-REVIEW.iter26.md
head: 33a8a44
base: 8035847
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
  critical: 3
  warning: 1
  info: 0
  total: 4
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-09-04T02:59:41Z
**Depth:** deep
**Iteration:** 27
**HEAD:** `33a8a44`
**Base:** `8035847`
**Files Reviewed:** 35
**Status:** issues_found

## Summary

The iteration-26 changes repair the previously reported delayed-writer,
predecessor-gap, anchor-rollback, and lifetime-retention cases after a receipt
sequence is committed. They do not make the new receipt-v2 lifecycle crash
complete. A process loss after any immutable event/tail/head create but before
the fixed head replace leaves durable, authenticated control objects outside
the only bounded enumeration path. Fresh recovery reports success and normal
paging reports a terminal empty result for all three inventory families.

The new compaction anchor also has an unrecoverable publication window. If the
anchor replace is durable and the following fixed-head witness replace does not
happen, constructor recovery validates the old head against the forward anchor
before invoking the helper designed to bind that anchor. Reopen therefore
fails permanently on a valid protocol-produced crash state.

Finally, receipt commitment still serializes distinct keys across the durable
fixed-head replace. Although the lock name includes the predecessor digest,
all writers in one family observe the same digest and contend for the same
lease for that generation. A public two-key probe held key A in the fixed-head
fsync and proved key B could not finish until A was released. This contradicts
the explicit no-global/no-family-lease and STOR-07 nonblocking requirement.

One additional typed-error defect remains at the bounded scheduler-head read:
an oversized head raises a raw `ValueError` from managed file I/O instead of a
domain integrity/backend error. The bound itself is enforced, so this is a
robustness/API warning rather than an integrity bypass.

The changed-file Ruff check and `git diff --check` pass. The focused Phase 3
suite reached 86% before the sandbox rejected creation of its Unix-domain
socket fixture with `EPERM`; a complete rerun excluding only that
environment-incompatible parametrized test is recorded below.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: Crashes before fixed-head commitment leave invisible authenticated residues

**Classification:** BLOCKER

**Files:** `/Users/akriz/code/cacheness/.claude/worktrees/rf-03-14929-1788481870/src/cacheness/storage/operation_repository.py:2465-2535`,
`/Users/akriz/code/cacheness/.claude/worktrees/rf-03-14929-1788481870/src/cacheness/storage/operation_repository.py:2537-2583`,
`/Users/akriz/code/cacheness/.claude/worktrees/rf-03-14929-1788481870/src/cacheness/storage/operation_repository.py:3007-3068`,
`/Users/akriz/code/cacheness/.claude/worktrees/rf-03-14929-1788481870/src/cacheness/storage/operation_repository.py:3277-3335`

**Issue:** In receipt-v2, event creation is the allocation CAS, but allocation
is not represented in the signed fixed head until `_write_inventory_head()` at
the end of `_advance_inventory_receipt_commitment()`. `_read_inventory()` and
recovery derive their complete bounded namespace solely from that fixed head.
They never inspect the exact `next_sequence` slot when the head still says the
inventory is empty. Consequently, a crash after the durable event create, tail
receipt create, or head receipt create strands valid control evidence outside
the authenticated high water. Neither `_receipt_next_sequence()` nor
`compact_inventory_for_recovery()` sees or removes it.

A deterministic nine-case fault matrix injected `BaseException` immediately
after each durable event, tail, and head create for primary, sidecar, and
pending inventories. On a fresh reopen, every case retained
`next_sequence == 1`; the corresponding page returned zero entries and a
terminal `next_cursor`; and `compact_inventory_for_recovery()` returned
`True`. The filesystem still held the event in all cases, event plus tail in
the tail cases, and event plus tail plus head in the head cases. This is a
false-clean recovery result and violates STOR-04/STOR-06.

**Fix:** Make allocation itself an authenticated, boundedly discoverable fixed
head state (for example, an `allocated/awaiting_commit` exact successor), or
make mutation/recovery validate the single exact successor slot and either
help it through receipt/head commitment or publish an authenticated abort
before declaring completion. Preserve predecessor/signature/store/epoch
checks and apply the same protocol to all three families. Add fresh-process
crash tests after event, tail, and head durability and before/after the fixed
head replace; assert that recovery cannot return clean while any residue is
unaccounted for.

### CR-02: A crash between anchor durability and its fixed-head witness cannot be recovered

**Classification:** BLOCKER

**Files:** `/Users/akriz/code/cacheness/.claude/worktrees/rf-03-14929-1788481870/src/cacheness/storage/operation_repository.py:2300-2363`,
`/Users/akriz/code/cacheness/.claude/worktrees/rf-03-14929-1788481870/src/cacheness/storage/operation_repository.py:2365-2443`,
`/Users/akriz/code/cacheness/.claude/worktrees/rf-03-14929-1788481870/src/cacheness/storage/operation_repository.py:2632-2686`,
`/Users/akriz/code/cacheness/.claude/worktrees/rf-03-14929-1788481870/src/cacheness/storage/operation_repository.py:2873-2890`,
`/Users/akriz/code/cacheness/.claude/worktrees/rf-03-14929-1788481870/src/cacheness/storage/operation_repository.py:3303-3335`

**Issue:** `_compact_inventory_window()` durably replaces
`receipt-anchor.json` before replacing the signed fixed head with the anchor
witness. That ordering is defensible only if recovery can finish the valid
forward transition. It cannot: recovery first enters
`_inventory_head_transition()`, whose `_read_inventory()` calls
`_receipt_next_sequence()`. With the old head's anchor sequence still zero,
line 2328 detects the new anchor and raises "receipt anchor exceeds its retained
witness". `_help_unbound_inventory_anchor()` is therefore unreachable from
constructor/recovery in precisely the crash state it claims to repair.

A deterministic probe created and retired a primary record, then injected
`BaseException` in the fixed-head write immediately after the new anchor was
durable. The exact record was gone and the signed forward anchor remained.
Fresh `BlobStore(root)` construction failed with
`CacheManifestIntegrityError: Lifecycle inventory receipt anchor exceeds its
retained witness`; repeated reopen cannot progress. The implementation is
generic across primary, sidecar, and pending families.

**Fix:** At explicit mutation/recovery admission, acquire the exact observed
head-generation transition and read the head plus anchor in a mode that permits
only one authenticated forward successor. Validate its previous-anchor digest,
terminal receipt digest, store identity, epoch, sequence bound, and signature,
then bind it into the fixed head before performing the ordinary strict read.
Normal reads may continue to fail closed. Add process-loss tests at anchor
replace, witness replace, each unlink/cursor update, and the next head
generation for every family.

### CR-03: The fixed-head fsync remains a family-wide durability lease

**Classification:** BLOCKER

**Files:** `/Users/akriz/code/cacheness/.claude/worktrees/rf-03-14929-1788481870/src/cacheness/storage/operation_repository.py:531-563`,
`/Users/akriz/code/cacheness/.claude/worktrees/rf-03-14929-1788481870/src/cacheness/storage/operation_repository.py:2537-2583`,
`/Users/akriz/code/cacheness/.claude/worktrees/rf-03-14929-1788481870/src/cacheness/storage/operation_repository.py:2585-2605`

**Issue:** `_advance_inventory_receipt_commitment()` locks
`inventory-head:{family}:{predecessor_digest}` and retains that interprocess
lease through `_write_inventory_head()`, including its durable replace/fsync.
Every concurrent writer in a family reads the same predecessor generation, so
the digest-qualified name does not make the critical section key-specific: it
is one family-generation lease. `_inventory_head_transition()` uses the same
lease and can retain it across bounded maintenance and exact-target waits.

A public two-key `BlobStore.put()` probe paused key A inside
`_write_inventory_head()` when the primary head advanced to sequence 2. Key B
started after A reached that durability point but did not finish during a
0.5-second observation window. Releasing A allowed both calls to finish and
both values were readable. Existing concurrency tests pause immutable
event/tail/head creates, but do not pause the fixed-name head replace where the
shared lease is actually held. This violates STOR-07 and the explicit review
contract requiring distinct keys to remain nonblocking at every durability
point with no global or family lease.

**Fix:** Replace the locked fixed-name compare/read/fsync sequence with a
helpable immutable head-generation protocol or a backend primitive providing
true atomic conditional replace without holding a shared family lease across
durable I/O. Writers must be able to publish independent candidate generations
and help the exact winning predecessor chain. Bound retries and validate
signature/store/epoch/predecessor on every generation. Add two-process tests
that block each fixed-head replace/fsync and prove another key can complete,
including one and many intervening generations and concurrent maintenance.

## Warnings

### WR-01: Oversized scheduler heads leak a raw `ValueError` across the public boundary

**Classification:** WARNING

**File:** `/Users/akriz/code/cacheness/.claude/worktrees/rf-03-14929-1788481870/src/cacheness/storage/operation_repository.py:1185-1212`

**Issue:** `_read_present_inventory_head()` handles missing files and typed
`CacheUnsafePathError`, but does not translate the `ValueError` raised by
`ManagedFileOps.read_bytes_bounded()` when the fixed head exceeds 4096 bytes.
Other bounded readers in this module translate the same condition. A probe
replaced the primary head with 4097 bytes and reopened the public store; the
constructor raised raw `ValueError("managed file exceeds the byte limit")`
with no lifecycle context. This breaks the Phase 3 typed failure contract and
makes callers distinguish an internal helper exception from domain failures.

**Fix:** Catch `ValueError` at the head boundary and raise
`CacheBlobBackendError` or `CacheManifestIntegrityError`, preserving the cause
and adding `operation`, `family`, and a stable reason. Add oversized tests for
all fixed heads on constructor recovery, paging, and mutation paths.

## Verification Performed

- Deterministic event/tail/head pre-commit crash matrix for primary, sidecar,
  and pending inventories: reproduced false-clean recovery in all nine cases.
- Deterministic anchor-before-witness crash and fresh reopen: reproduced
  permanent typed integrity failure rather than recovery.
- Public two-key concurrent put with primary fixed-head fsync paused:
  reproduced cross-key blocking until the paused writer was released.
- Oversized 4097-byte primary head and public reopen: reproduced raw
  `ValueError` without domain context.
- `ruff check` on the four files changed from `8035847..33a8a44`: passed.
- `git diff --check 8035847..33a8a44`: passed.
- Focused Phase 3 suite: first run reached 86%, then the sandbox denied the
  Unix socket fixture's `bind()` with `EPERM`; rerun excluded only
  `test_managed_reads_reject_special_nodes_before_they_can_block` and passed
  at 100% (two unrelated platform fixtures skipped).

---

_Reviewed: 2026-09-04T02:59:41Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: deep_
