---
phase: 03-atomic-lifecycle-and-recovery-engine
reviewed: 2026-09-04T00:27:57Z
depth: deep
iteration: 26
archived_as: iteration-26-input
head: 8035847
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
  critical: 4
  warning: 0
  info: 0
  total: 4
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-09-04T00:27:57Z
**Depth:** deep
**Iteration:** 26
**HEAD:** `8035847`
**Files Reviewed:** 35
**Status:** issues_found

## Summary

The receipt scheduler removes the previous store-wide append lease, but its new
proof and retirement protocol is not safe. A live writer can be classified as
stale after its event/receipts are durable but before its operation record is
created; when that writer resumes, it publishes durable recovery evidence below
the signed live floor. A public `BlobStore.put()` probe reproduced a directly
readable operation record which every inventory/recovery page reported as
terminally absent.

The high-water proof has two independent rollback/gap defects. Replaying one
older, correctly signed receipt anchor truncates a newer inventory even while
the current fixed head, events, and exact operation records remain untouched.
Separately, the exponential/binary discovery algorithm assumes that receipt
existence is monotonic, but immutable receipts contain no predecessor/range
commitment. Removing only receipt 3 from an otherwise valid 1..8 sequence was
not detected: discovery returned 9 and normal paging accepted the range.

Finally, receipt retirement does not bound the scheduler namespace. It removes
tail/head receipts but deliberately retains every immutable operation event
forever. Forty completed operations with a one-item lifecycle limit converged
to an empty live inventory while leaving forty event files. This grows with
lifetime mutation/checkpoint count and can exhaust the control filesystem.

The complete checked-in focused Phase 3 suite and changed-file Ruff checks pass,
so none of these adversarial states is currently covered. The manifest
repository's signed terminal witness, sparse-marker ordering, and ordinary-read
non-mutation did not produce a separate failure in this pass.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: A delayed writer can publish live evidence below the compacted inventory floor

**Classification:** BLOCKER

**Files:** `src/cacheness/storage/operation_repository.py:2270-2308`,
`src/cacheness/storage/operation_repository.py:2398-2447`,
`src/cacheness/storage/operation_repository.py:2610-2633`,
`src/cacheness/storage/operation_repository.py:3594-3602`

**Issue:** `create_exclusive()` completes `_append_inventory_event()` before it
creates the corresponding operation record. Receipt mode acknowledges that
event through immutable tail/head receipts and then returns, with no lease or
durable publication-state token protecting the interval before line 3600.
Meanwhile, retirement of an unrelated operation runs
`_compact_inventory_window()`. An absent current record is treated as a stale
event, and a window covering the live floor advances `first_live_sequence`
past it. Every later page begins at `max(requested_position,
first_live_sequence)`.

A deterministic public probe paused writer A immediately before the primary
operation-record create, after its scheduler append had returned. Writer B for
another logical key completed and retired normally, advancing the primary
floor to `next_sequence`. A was then allowed to create its operation record and
an injected `OSError` modeled process loss. `get_raw(A)` returned the exact
durable record, while `list_page()` returned `entries == ()` and
`next_cursor is None`; the signed state was `first_live_sequence ==
next_sequence`. Reopen recovery and reconciliation therefore have no route to
the recoverable residue. The same ordering exists for sidecar and pending
control creation.

This violates D-05/D-07, STOR-04, and the recovery invariant that durable
operation evidence remains detectable. It also contradicts the new helping
protocol's stated guarantee: helping receipts proves scheduler publication,
not that the indexed record can no longer appear.

**Fix:** Add an authenticated per-sequence publication state which distinguishes
reserved/in-flight membership from a completed record publication or an
explicitly abandoned member. Compaction must never advance the live floor over
a merely absent reserved member. Publish the completion proof only after the
exact record is durable; publish an abort proof through a deterministic
recovery/grace transition for a dead writer. Keep these immutable transitions
outside a global durable-I/O lease, and make pages/recovery fail closed on an
unresolved reservation. Add event-controlled public and independent-process
tests that pause before and after event, tail receipt, head receipt, and record
durability, then let unrelated retirements/clear/reconcile run before the
original writer resumes or crashes.

### CR-02: Replaying one older signed receipt anchor silently truncates newer live evidence

**Classification:** BLOCKER

**File:** `src/cacheness/storage/operation_repository.py:1852-1948`,
`src/cacheness/storage/operation_repository.py:1964-2020`,
`src/cacheness/storage/operation_repository.py:2569-2591`

**Issue:** `_read_inventory_receipt_anchor()` authenticates the anchor and its
terminal event, but never compares its sequence with the retained signed fixed
head or another monotonic witness. `_receipt_next_sequence()` treats the
anchor's sequence as its base. Once a later anchor has retired intervening
receipts, replaying an older valid anchor makes the first post-anchor receipt
look absent and returns that old position as the current next sequence.
`_read_inventory()` overwrites the fixed head's newer `next_sequence` with this
result, and pages derive their snapshot high water from the regressed value.

A deterministic four-record probe saved the valid anchor at sequence 2, added
records 3 and 4, advanced/compacted the anchor through 4, then replaced only
the anchor with the saved signed bytes. Before replay, paging returned all four
records. After replay, exact records 3 and 4 remained directly readable and all
events plus the newer signed fixed head remained in place, but `list_page()`
returned only records 1 and 2 with a terminal cursor; `_read_inventory()`
reported `next_sequence == 3`.

This is not D-21's excluded coordinated deletion/rebinding of every lifecycle
authority object. One mutable control object was replayed while the current
head, events, evidence, initialization record, and signing key were untouched.
The result is a false-clean recovery/reconciliation inventory.

**Fix:** Make the retained fixed head (or another independently retained
monotonic witness) a lower bound that receipt discovery must prove and may
never undercut. A replayed anchor whose post-anchor proof cannot bridge that
bound must fail closed. Bind each new anchor to the previous anchor/head
commitment and enforce monotonic replacement under exact compare-and-swap.
Add same-root rollback tests for every family, including old anchors captured
before and during partial retirement, reopen, old cursors, and later appends;
assert that replay is rejected before any evidence is omitted or overwritten.

### CR-03: Binary high-water discovery skips internal receipt gaps

**Classification:** BLOCKER

**File:** `src/cacheness/storage/operation_repository.py:1964-2020`

**Issue:** Exponential probing followed by binary search is valid only for a
monotonic predicate. Receipt existence is not such a predicate under the
reviewed fault model: each sequence-specific head receipt is independent and
contains no hash/link to its predecessor or an authenticated range summary.
The algorithm probes powers of two and then assumes any present later receipt
proves all earlier receipts are present.

A deterministic primary-family probe created valid receipts and live exact
records at sequences 1 through 8, then removed only
`head-00000000000000000003.json`. The initialization record, fixed head,
anchor state, all events, all other heads/tails, and exact records remained
untouched. `_receipt_next_sequence()` returned 9 and `list_page()` accepted all
eight entries instead of reporting the internal control gap. The same generic
implementation serves primary, sidecar, and pending inventories. Missing one
tail receipt has the same blind spot whenever its head sequence is not probed.

The iteration-25 fix report explicitly claims that an accepted later receipt
cannot cover a gap; the implementation disproves that claim. A single control
loss or partial substitution is observable and must fail closed under D-21,
not be normalized into a valid contiguous high water.

**Fix:** Do not binary-search independent receipt existence. Either make every
head receipt commit to its predecessor and retain authenticated skip/range
proofs that allow logarithmic verification, or use a bounded sequential
continuation which verifies every sequence before advancing high water. Charge
the actual nested head/tail/event reads and bytes to an explicit durable
discovery budget. Add missing/malformed/substituted head and tail gaps at
non-power and power boundaries, far sparse valid receipts, 2^k-1/2^k/2^k+1
high waters, concurrent helping, and reopen tests for all three families.

### CR-04: Retirement leaves one immutable event file for every lifetime mutation

**Classification:** BLOCKER

**File:** `src/cacheness/storage/operation_repository.py:2133-2227`,
`src/cacheness/storage/operation_repository.py:2241-2313`

**Issue:** `_compact_inventory_receipts()` deletes bounded tail and head receipt
windows, but `_compact_inventory_window()` deliberately performs no deletion
for a stale event. It advances `first_live_sequence` while retaining the event
file, and there is no other event-retirement path. Consequently the
`operations/.cacheness-inventory-v2/<family>/event-*.json` namespace grows with
every operation-record checkpoint and pending/sidecar control publication,
even after all corresponding records and receipts are gone.

A deterministic run with `max_inventory_items=1` created and retired forty
primary records and drove maintenance to completion. The signed state correctly
reported `first_live_sequence == next_sequence == 41`; tail/head receipt counts
were both zero, but all forty event files remained. Repeating the workload adds
another file per scheduler append without any retention ceiling. Production
checkpoint-heavy operations add multiple events per logical operation, so the
growth is not bounded by live key or live recovery-debt cardinality.

This violates the phase's bounded lifecycle-evidence/resource contract and
creates a local storage-exhaustion failure mode in the control namespace. The
signed anchor currently proves only its terminal event and cannot authorize
retirement of the preceding event range, so the apparent receipt compaction is
not a complete bounded-namespace solution.

**Fix:** After resolving CR-01's in-flight state, extend the authenticated
retirement structure to commit to every exact event in a retired contiguous
range (for example, a chained/Merkle accumulator with an exact terminal and
prior-anchor commitment). Persist an event-retirement cursor before unlinking,
delete at most the configured window per call, and advance the anchor only
after each idempotent durable deletion. Retain enough signed range proof for
old cursors and rollback detection without retaining every event. Test crash at
each anchor/cursor/unlink boundary, delayed writers, live members inside stale
ranges, reopen, and a long-running workload whose event-file count remains a
documented function of live debt plus one bounded maintenance window.

## Warnings

None.

## Verification Performed

- Focused Phase 3 lifecycle suite (`manifest_repository_cas`, atomic lifecycle,
  concurrency, reconciliation, integrity, read contract, and close contract):
  passed at HEAD `8035847`.
- Changed-file Ruff and `git diff --check 7b70a94..HEAD`: passed.
- Public delayed-writer probe: reproduced a durable exact primary operation
  record beneath `first_live_sequence`, with terminally empty inventory paging.
- Same-root signed-anchor rollback probe: reproduced direct records 3/4 omitted
  from terminal paging after replaying only the valid sequence-2 anchor.
- Internal-gap probe: removing only head receipt 3 from valid sequences 1..8
  still produced discovered `next_sequence == 9` and an accepted eight-entry
  page.
- Retention probe: forty completed/retired primary records with one-item
  maintenance converged to `first_live_sequence == next_sequence == 41` while
  leaving forty event files and zero head/tail receipts.
- Manifest acknowledged-tail binding, compacted-tail witness, sparse-marker
  ordering, old-cursor behavior, and normal-read mutation paths were traced;
  no separate manifest-family defect was proven in this iteration.
- Known repository-wide Ruff debt, shutdown-only SQLite destructor behavior,
  and the base-install NumPy packaging issue were excluded as previously
  captured concerns.

---

_Reviewed: 2026-09-04T00:27:57Z_
_Reviewer: the agent (gsd-code-reviewer), independent iteration 26_
_Depth: deep_
