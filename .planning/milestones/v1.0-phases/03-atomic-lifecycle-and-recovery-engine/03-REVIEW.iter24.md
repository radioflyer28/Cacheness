---
phase: 03-atomic-lifecycle-and-recovery-engine
reviewed: 2026-09-03T23:24:53Z
depth: deep
iteration: 25
archived_as: iteration-25-input
head: 7b70a94
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

**Reviewed:** 2026-09-03T23:24:53Z
**Depth:** deep
**Iteration:** 25
**HEAD:** `7b70a94`
**Files Reviewed:** 35
**Status:** issues_found

## Summary

The v2 append tail closes only the one-event-ahead crash window. Once the head
has acknowledged the same high water, readers compare only `next_sequence` and
never prove that the tail's terminal identity still matches the terminal event.
A validly signed but different terminal event is therefore accepted as stale
scheduling data; a subsequent append overwrites the only tail that exposed the
mismatch. Canonical manifest authority or exact operation evidence remains
readable by direct key/ID while disappearing permanently from every inventory
consumer, including clear, recovery, and reconciliation.

The operation scheduler also takes a store-wide exclusive
`inventory:initialize` lease for every normal primary, sidecar, and pending
append, even after initialization is complete. This directly contradicts the
phase's repeated no-global-lock invariant and STOR-07. A deterministic public
probe paused one put during its primary event fsync and observed an unrelated
key's put unable to complete until the store-wide scheduler lease was released.

Both defects were reproduced at integrated HEAD `7b70a94`. The checked-in
Phase 3 suites remain green, so they are missing these adversarial cases. The
new signed primary recovery cursor otherwise survived the checked-in
limit-one/reopen and concurrent record-publication probes: its fixed high water,
durable continuation, reset behavior, and exact record revalidation showed no
separate correctness failure in this review.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: An acknowledged tail never proves that its terminal event is the member it signed

**Classification:** BLOCKER

**Files:** `src/cacheness/storage/manifest_repository.py:943-970`,
`src/cacheness/storage/manifest_repository.py:1266-1336`,
`src/cacheness/storage/operation_repository.py:1393-1424`,
`src/cacheness/storage/operation_repository.py:1675-1754`

**Issue:** `_verify_inventory_tail()` returns immediately whenever head and tail
have equal `next_sequence`. The new `_tail_matches_inventory_event()` check is
used only while acknowledging a tail that is one event ahead, or while writing
the same tail value. It is never applied to the ordinary acknowledged state.

A deterministic JSON manifest probe published `a` and `b`, replaced only event
2 with another correctly signed event for `x` at sequence 2, and left the signed
head and v2 tail untouched. `list_page()` returned only `a`, while
`get_raw("b")` still returned `b`'s canonical record. Publishing `c` then
advanced the tail to sequence 3 and permanently normalized the inconsistent
state; subsequent pages returned `a,c` and continued to omit live authority
`b`. The same primary-operation probe kept exact record `b...b` directly
readable while pages returned only the first and third records. The generic
operation implementation applies identically to primary, sidecar, and pending
families.

This is an observable partial control substitution, not D-21's excluded
deletion/rebinding of every live authority object. The head, tail, canonical
manifest/record, initialization proof, and all unrelated events remain in
place. Clear can consequently miss a committed key, and recovery or
reconciliation can lose the only bounded route to live cleanup evidence.

**Fix:** Validate the acknowledged terminal binding before accepting a head or
advancing the tail. If compaction may retire the terminal event, retain a direct
authenticated terminal witness or make the sparse-run proof verifiably commit
to the exact tail tuple; an absent terminal without that proof and any
mismatched terminal must fail closed. Add acknowledged-head tests for matching,
missing, malformed, and differently signed terminal events in JSON/memory
manifest inventories and all three operation families, across reopen and a
subsequent append. Assert the canonical key/record never becomes inventory-
invisible.

### CR-02: Every ordinary lifecycle append is serialized by a store-wide exclusive lock

**Classification:** BLOCKER

**File:** `src/cacheness/storage/operation_repository.py:1086-1132`,
`src/cacheness/storage/operation_repository.py:1675-1754`

**Issue:** `_append_inventory_event()` first calls `initialize_new_store()`,
which takes `inventory:initialize`, then reacquires that same store-wide
transition around every future family append. The nested critical section spans
event creation, event fsync, tail publication, tail fsync, head publication,
and head fsync. Normal create/replace/delete/checkpoint traffic therefore
depends on one global exclusive advisory lock before its exact per-operation
CAS, regardless of logical key or inventory family.

A deterministic public BlobStore probe paused key `a` inside
`create_bytes_durable_exclusive()` for its primary inventory event while those
locks were held. A concurrent `put()` for distinct key `b` did not complete
during the pause and completed immediately after release. The existing
distinct-key test pauses later at `manifest_publish`, after the scheduler locks
have already been released, so it cannot detect this serialization.

This violates D-17/D-20, every Phase 3 plan's explicit invariant that ordinary
operations are not serialized by one global lock, and STOR-07's requirement
that distinct keys are not globally serialized. It also makes an unrelated
key's availability depend on a stalled control-file fsync for another key.

**Fix:** Restrict the store-wide initialization transition to the one-time
freshness proof plus atomic publication of the initialization record and empty
family heads. Once that authenticated epoch exists, publish ordinary membership
through a truthfully bounded/sharded or backend-atomic scheduler that does not
take a store-global normal-operation lease; retain exact per-key/record CAS as
the authority boundary. If mixed-version writers can create raw evidence
without honoring the new protocol, reject that topology with explicit version
fencing rather than serializing all current writers behind a lock old writers
do not acquire. Add a deterministic test that pauses one key inside scheduler
event/tail/head durability and proves an unrelated key can still finish.

## Warnings

None.

## Verification Performed

- Full focused Phase 3 lifecycle selection (manifest CAS, reconciliation,
  atomic lifecycle, concurrency, integrity, read contract, and close contract):
  passed at HEAD `7b70a94`.
- Tail-ahead, missing-terminal, limit-one durable recovery continuation, and
  concurrent in-flight publication regressions: `8 passed`.
- JSON manifest acknowledged-terminal substitution probe: reproduced a directly
  readable canonical key disappearing from pages, including after a later
  append.
- Primary operation acknowledged-terminal substitution probe: reproduced a
  directly readable exact record disappearing from pages, including after a
  later append.
- Distinct-key public scheduler-lock probe: reproduced key `b` blocking while
  key `a` was paused inside its inventory-event durable create.
- Changed-file Ruff and `git diff --check 968e615..HEAD`: passed.
- Normal page/read paths were traced separately from mutating recovery. No
  normal-read mutation was found. The signed recovery head rejects malformed
  and out-of-range cursor fields; its fixed high-water pass resets and revisits
  entries that raced record publication. No additional cursor defect was proven.
- Known repository-wide Ruff debt, shutdown-only SQLite destructor behavior,
  and the base-install NumPy packaging issue were excluded as previously
  captured concerns.

---

_Reviewed: 2026-09-03T23:24:53Z_
_Reviewer: the agent (gsd-code-reviewer), independent iteration 25_
_Depth: deep_
