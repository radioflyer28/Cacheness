---
phase: 03-atomic-lifecycle-and-recovery-engine
reviewed: 2026-09-02T22:55:12Z
depth: deep
iteration: 24
archived_as: iteration-24-input
head: 968e615
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
  critical: 2
  warning: 0
  info: 0
  total: 2
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-09-02T22:55:12Z
**Depth:** deep
**Iteration:** 24
**HEAD:** `968e615`
**Files Reviewed:** 35
**Status:** issues_found

## Summary

The append-tail controls detect ordinary head/tail rollback, but the permitted
tail-ahead writer path does not require the event already committed by that tail
to exist. A partial substitution can therefore delete the allocated event and
replay the immediately preceding head; the next writer silently fills that
already-anchored sequence with a different member, permanently omitting the
still-live original authority/evidence from inventory.

The lock-order correction also moved prospective operation scheduling before
record CAS without providing durable progress over stale speculative members.
Primary recovery restarts from a fresh cursor on every invocation. With a
supported one-item work limit, one stale event before a live operation prevents
that operation from ever reaching recovery, even across unlimited reopen calls.

Both blockers were reproduced deterministically at integrated HEAD `968e615`.
The complete manifest-CAS and reconciliation suites pass, demonstrating that
these are coverage gaps rather than already-failing checked-in contracts.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: Tail-ahead recovery replaces a missing committed event with a different member

**Classification:** BLOCKER

**Files:** `src/cacheness/storage/manifest_repository.py:909-970`,
`src/cacheness/storage/manifest_repository.py:1179-1238`,
`src/cacheness/storage/operation_repository.py:1319-1378`,
`src/cacheness/storage/operation_repository.py:1549-1619`

**Issue:** A tail one sequence ahead of its head is accepted by append callers,
but each tail authenticates only `next_sequence`, not the event identity at the
allocated position. The append loops then attempt exclusive creation before
checking whether this is a tail-ahead resumption. If the allocated event was
deleted, exclusive creation succeeds with the new caller's key/name and digest;
`_write_inventory_tail()` sees the existing equal high-water and returns, and
the head acknowledges the replacement event.

A deterministic JSON-manifest probe published `a`, saved its head, published
`lost`, restored the saved head, deleted only event 2, and retained the newer
signed tail plus canonical `lost` authority. Publishing `b` then succeeded.
`get_raw("lost")` still returned `b"L"`, but `list_page()` returned only `a`
and `b`. The equivalent primary-operation probe returned only the first and
third records while the exact second record remained readable. The same generic
operation append implementation serves primary, sidecar, and pending families.

This is not D-21's out-of-scope coordinated replay of all authenticated live
controls: the newer tail remains observable while only the head and event are
partially substituted. The contract requires that state to fail closed.

**Fix:** Treat tail-ahead as a distinct resumption branch before creating any
event. Require the already-allocated event to exist, authenticate it, and only
then acknowledge that exact member; if it is missing, malformed, or inconsistent,
raise a typed integrity failure. Bind the tail to the terminal event identity
(sequence plus event digest/key or name) if needed so equality cannot authorize
replacement. Add head-replay-plus-last-event-deletion tests for JSON manifest
and all three operation families, with reopen and a subsequent append.

### CR-02: Stale speculative events can permanently starve primary lifecycle recovery

**Classification:** BLOCKER

**Files:** `src/cacheness/storage/operation_repository.py:1463-1547`,
`src/cacheness/storage/operation_repository.py:1621-1679`,
`src/cacheness/storage/operation_repository.py:1681-1783`,
`src/cacheness/storage/operation_repository.py:2087-2120`,
`src/cacheness/storage/operation_repository.py:2886-2914`,
`src/cacheness/storage/lifecycle.py:1369-1443`

**Issue:** Checkpoint scheduling now appends a prospective event before the
exact record CAS. A losing CAS, process loss before record publication, or an
ordinary completed checkpoint leaves a stale predecessor position. Maintenance
is charged only by the later retirement path, so interrupted/live operations
can have `maintenance_target_sequence == 0` while stale positions precede their
current record. `compact_inventory_for_recovery()` then reports complete without
moving the floor. `LifecycleEngine.recover()` starts `cursor = None` on every
call and caps inventory inspection by `max_reconcile_actions`; when that budget
is consumed by the stale prefix, it discards `page.next_cursor` and returns.

With the supported configuration `operation_page_size=1`,
`max_inventory_items=1`, and `max_reconcile_actions=1`, a public BlobStore probe
completed an old put, interrupted an overwrite immediately after its
`AUTHORITY_PUBLISHED` checkpoint, and reopened the store three times. Every
reopen read the new winner, but all three left both generation payloads and the
same operation record intact: the superseded payload was never reclaimed and
the evidence was never retired. A repository-level probe isolated the cause:
each fresh pass reported maintenance complete, inspected only the first stale
checkpoint event, returned zero entries with continuation 2, and left the later
current record unseen. The same ordering occurs on every normal operation
checkpoint and on every pre-CAS loser introduced by the lock-order fix.

**Fix:** Give primary recovery durable authenticated continuation across its
caller-bounded invocations, or make speculative scheduling and exact record
publication share a lock/order protocol that can safely charge and compact every
unsettled event. A recovery mutation may exact-revalidate and advance a signed
floor, but it must not race an event whose record publication is still in flight.
Test a stale pre-CAS event and multiple stale checkpoint predecessors before one
live primary record with all three limits set to one; repeated recovery and
fresh reopen must reach and converge the live record in a bounded number of
calls. Cover concurrent checkpoint publication while the stale prefix advances.

## Warnings

None.

## Verification Performed

- `tests/test_manifest_repository_cas.py` and
  `tests/test_blob_store_reconciliation.py`: passed completely at HEAD
  `968e615`.
- JSON manifest partial-substitution probe reproduced `list_page()` omitting a
  still-readable canonical authority after the next append reused its slot.
- Primary operation partial-substitution probe reproduced the same omission for
  exact evidence bytes.
- Primary stale-prefix probe with all work/page limits set to one reproduced the
  same empty first page and continuation on three fresh passes while the later
  current record remained readable.
- A public interrupted-overwrite probe reopened three times and retained both
  the superseded and winning generation files plus the exact post-authority
  operation record on every pass.
- Existing append-tail tests cover head/tail replay, present tail-ahead events,
  and missing events during reads, but do not combine tail-ahead with a missing
  allocated member or exercise forward progress across fresh primary recovery
  invocations.
- Known repository-wide Ruff debt, shutdown-only SQLite destructor behavior,
  and the base-install NumPy packaging issue were excluded as previously
  recorded concerns.

---

_Reviewed: 2026-09-02T22:55:12Z_
_Reviewer: the agent (gsd-code-reviewer), independent iteration 24_
_Depth: deep_
