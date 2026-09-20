---
phase: 03-atomic-lifecycle-and-recovery-engine
reviewed: 2026-09-02T21:33:42Z
depth: deep
iteration: 23
archived_as: iteration-23-input
head: 6765596
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
  warning: 0
  info: 0
  total: 3
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-09-02T21:33:42Z
**Depth:** deep
**Iteration:** 23
**HEAD:** `6765596`
**Files Reviewed:** 35
**Status:** issues_found

## Summary

The iteration-22 fixes bind initialization and scheduler objects to a store and
authenticate their contents, but authentication alone does not establish a
complete monotonic inventory. Old valid signed heads can be replayed to omit
later live authority, and allocated event slots that disappear without an
authenticated sparse-run proof are silently treated as compacted gaps. In
addition, JSON manifest publication mishandles the documented crash window in
which an event is durable but its head acknowledgement is not: after the
collision it writes the old sequence-bound event bytes at the next locator.

All three issues were reproduced deterministically at integrated HEAD
`6765596`. The existing manifest CAS suite passes, so these are coverage gaps,
not failures already exposed by the checked-in tests.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: Replaying an older valid scheduler head hides later live authority

**Classification:** BLOCKER

**Files:** `src/cacheness/storage/manifest_repository.py:664-752`,
`src/cacheness/storage/manifest_repository.py:1431-1442`,
`src/cacheness/storage/operation_repository.py:832-909`,
`src/cacheness/storage/operation_repository.py:1522-1546`

**Issue:** Heads are HMAC-authenticated and store/epoch-bound, but there is no
anti-rollback relationship to the immutable event tail or another monotonic
authority. A previously valid head from the same epoch therefore still
verifies after later events and authority records have been published. Paging
trusts its smaller `next_sequence` as the snapshot high-water and never
inspects the omitted tail.

A deterministic JSON-manifest probe published `a`, saved its signed head,
published `b`, and restored the saved head. `get_raw("b")` still returned
`b"B"`, while `list_page()` returned only `a`. The same probe against the
primary operation inventory restored its signed head after a second indexed
record: `get_raw()` returned the second record, while the operation page
returned only the first. This permits false-clean reconciliation and incomplete
clear snapshots using only previously authentic same-store control bytes.

**Fix:** Anchor scheduler progress in a non-rollbackable backend CAS record or
an authenticated append chain whose terminal claim is derived and checked
against the immutable tail. At minimum, reject a head when an allocated
successor event exists beyond its claimed high-water; a production design must
also prevent rollback to a head before an omitted tail whose filenames are not
guessed. Cover same-epoch head replay after one and many publications for
manifest plus primary/sidecar/pending inventories, including reopen and old
cursors.

### CR-02: Missing allocated events are accepted as proven compacted gaps

**Classification:** BLOCKER

**Files:** `src/cacheness/storage/manifest_repository.py:857-877`,
`src/cacheness/storage/manifest_repository.py:1454-1467`,
`src/cacheness/storage/operation_repository.py:1224-1232`,
`src/cacheness/storage/operation_repository.py:1563-1572`

**Issue:** Both repositories return `None` for a missing event inside the
signed head's allocated range, and page readers silently continue. For the
manifest inventory, legitimate compaction publishes an authenticated skip
marker before deleting events, so a missing event without that marker is not a
proof of staleness. For operation inventories, the current compactor retains
stale events and does not publish skip proofs at all. In both cases, absence is
ambiguous control-data loss and must fail closed rather than erase inventory
membership.

A deterministic manifest probe published one live record, removed only
`event-...1.json`, and retained the signed head and canonical metadata. The
page returned an empty terminal result while `get_raw("live")` returned the
record. An equivalent primary-operation probe returned
`OperationPage(entries=(), next_cursor=None)` while the exact indexed operation
record remained readable. No signature forgery, malformed payload, or normal
read mutation was involved.

**Fix:** Distinguish an authenticated compacted run from unexplained event
absence. Manifest paging may skip only through a verified sparse marker that
commits the missing slot; otherwise raise a stable integrity/backend error.
Operation paging needs equivalent authenticated gap evidence, or it must treat
every missing position below the signed high-water as corruption. Add missing
first/middle/last event tests for every family, with current and stale source
records, compaction crash boundaries, reopen, reconcile, and clear.

### CR-03: JSON event-collision recovery writes an event under the wrong sequence

**Classification:** BLOCKER

**File:** `src/cacheness/storage/manifest_repository.py:1028-1071`

**Issue:** `_append_inventory_event()` constructs and signs `event` and
`encoded` once before its JSON collision loop. If the target event already
exists because a process died after durable event creation but before head
acknowledgement, the handler advances `state["next_sequence"]` and retries a
new locator but reuses bytes whose signed `sequence` is still the old value.
It then advances the head and can publish the canonical manifest successfully;
the next inventory read rejects the newly written event because its embedded
sequence does not match its filename/position.

The deterministic probe published sequence 1, injected a correctly signed
sequence-2 event without advancing the head, and then published the matching
second manifest. Publication succeeded and `get_raw("b")` returned `b"B"`,
but `list_page()` raised `CacheBlobBackendError`; the sequence-3 locator
contained a signed record with `"sequence":2`. This is the exact crash window
the `FileExistsError` branch claims to recover.

**Fix:** Rebuild and sign the event inside the loop after each refreshed
`state["next_sequence"]`, and validate the existing collision separately. Do
not acknowledge a colliding event as the caller's publication unless its full
store/epoch/sequence/key/digest tuple matches the intended member; otherwise
advance past it and create freshly encoded bytes for the new sequence. Add
fault/reopen tests for one and multiple unacknowledged events, both matching
and unrelated, and assert subsequent paging, reconciliation, and clear remain
usable.

## Warnings

None.

## Verification Performed

- `tests/test_manifest_repository_cas.py`: passed completely at `HEAD 6765596`.
- Same-store signed-head rollback probes reproduced omitted live manifest and
  primary-operation records.
- Missing-event probes reproduced terminal empty pages with the exact live
  manifest/operation bytes still present.
- JSON event/head crash probe reproduced a sequence-3 event carrying a signed
  sequence-2 identity and a subsequent typed page failure.
- Known repository-wide Ruff debt, shutdown-only SQLite destructor behavior,
  and the base-install NumPy packaging issue were excluded as previously
  recorded concerns.

---

_Reviewed: 2026-09-02T21:33:42Z_
_Reviewer: the agent (gsd-code-reviewer), independent iteration 23_
_Depth: deep_
