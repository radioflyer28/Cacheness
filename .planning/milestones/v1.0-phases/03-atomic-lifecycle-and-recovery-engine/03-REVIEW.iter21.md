---
phase: 03-atomic-lifecycle-and-recovery-engine
reviewed: 2026-09-02T20:23:52Z
depth: deep
iteration: 22
archived_as: iteration-22-input
head: 4b0c034
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

**Reviewed:** 2026-09-02T20:23:52Z
**Depth:** deep
**Iteration:** 22
**HEAD:** `4b0c034`
**Files Reviewed:** 35
**Status:** issues_found

## Summary

Iteration 21 added bounded maintenance continuations and a signed all-family
initialization record, but the resulting recovery index is still not a
fail-closed authority boundary. The initialization signature is replayable
between different stores that share a configured key; manifest sparse markers
and inventory heads are shape-validated but unauthenticated even though they
can suppress live authority; and repeated delete performs a bounded global
scan with no key-specific continuation, so an unrelated retained operation can
make the matching tombstone permanently unreachable to that API call.

All three failures were reproduced at `HEAD 4b0c034` with deterministic local
JSON stores. They do not require malformed handler payloads, filename
guessing, unsupported backends, or normal-read mutation.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: Signed v3 initialization provenance is replayable across stores

**Classification:** BLOCKER

**Files:** `src/cacheness/storage/operation_repository.py:603-613`,
`src/cacheness/storage/operation_repository.py:848-894`,
`src/cacheness/storage/operation_repository.py:1005-1011`

**Issue:** `_initialization_signing_bytes()` signs only the schema version and
the constant family-name list. It does not bind the proof to the managed root,
store identity, topology, or an initialization nonce. Consequently, two
independent stores configured with the same signing key accept the same
`initialized-v3.json`. Once replayed, `_read_inventory()` treats every missing
family as an authenticated empty v2 family and skips the legacy/raw evidence
checks the record was meant to protect.

A deterministic probe initialized store A, copied only A's signed v3 record to
store B, configured both with the same valid 32-byte key, and placed an exact
`operations/reconcile-action-<32hex>.json` legacy sidecar in B before opening
it. Store B constructed successfully; its sidecar inventory returned an empty
terminal page and `reconcile()` returned no findings while the raw sidecar
remained present. Shared application-managed keys are a supported configuration,
so a constant-domain HMAC is not store provenance.

**Fix:** Version the initialization record again and include the same stable
store/topology identity used by lifecycle records (or a durable per-store
random identity created only after the all-family proof) in both the record and
HMAC preimage. Verify that identity before a missing head can mean empty. Treat
the existing replayable v3 form as explicit migration evidence. Add cross-root
replay tests using one injected key, plus copied-marker/head/raw-v1 crash shapes.

### CR-02: Unauthenticated scheduling heads and sparse markers can hide live authority

**Classification:** BLOCKER

**Files:** `src/cacheness/storage/manifest_repository.py:559-640`,
`src/cacheness/storage/manifest_repository.py:803-875`,
`src/cacheness/storage/manifest_repository.py:1276-1307`,
`src/cacheness/storage/operation_repository.py:753-838`

**Issue:** The new sparse-successor marker contains only `version`,
`start_sequence`, and `next_sequence`; `_validate_inventory_skip()` checks its
shape but no signature, store identity, inventory epoch, or commitment to the
events it skips. `list_page()` consumes the marker before reading or
exact-revalidating the event at its start. The manifest and operation inventory
heads are likewise unsigned even though `first_live_sequence`, high-water, and
maintenance fields determine which evidence can be observed. Thus valid-shaped
corruption or replay of scheduling metadata can create a false terminal page.
The signed v3 initialization record authenticates only the original
all-family decision; it does not authenticate any head, event, or sparse proof.

A one-key JSON store had an intact signed committed manifest and event at
sequence 1. Writing the valid-shaped marker
`{"version":2,"start_sequence":1,"next_sequence":2}` at the expected skip
locator left `get("victim")` readable, but `manifest_repository.list_page()`
returned an empty terminal page, `reconcile()` returned zero findings, and
`clear()` returned `0` while the victim remained stored. This is precisely the
false-clean/false-empty outcome D-15 and the Phase 3 threat model prohibit for
untrusted persisted control metadata.

**Fix:** Authenticate every scheduling record that is allowed to omit source
positions. At minimum, bind store identity, inventory family/epoch, start,
successor, and an exact proof/commitment for the skipped immutable run under
the store key; authenticate heads and event membership or replace mutable head
claims with an authenticated append/checkpoint chain. On any missing,
unauthenticated, replayed, or inconsistent proof, fail closed and inspect no
skipped authority as absent. Test valid-shaped marker/head substitution,
cross-store replay, old cursors, reopen, and concurrent compaction for manifest,
primary, sidecar, and pending inventories.

### CR-03: Repeated delete cannot reach its tombstone behind one unrelated live operation

**Classification:** BLOCKER

**Files:** `src/cacheness/storage/lifecycle.py:1325-1370`,
`src/cacheness/storage/lifecycle.py:1681-1689`

**Issue:** A tombstoned manifest carries no direct authenticated reference to
its delete operation. `_find_tombstone_record()` therefore scans the global
primary operation inventory from `None` on every call and stops after
`max_reconcile_actions`. It persists no lookup continuation and performs no
maintenance progress. If an earlier unrelated operation remains live, every
repeated `delete(key)` spends its whole supported budget on that same record,
returns `None`, and raises “tombstone has no matching authenticated operation.”
The matching signed delete record and tombstone remain intact but unreachable.

The probe used valid caller-owned limits
`operation_page_size=max_inventory_items=max_reconcile_actions=1`. An
interrupted post-authority overwrite retained one valid earlier operation;
then an interrupted tombstone cleanup retained the victim's matching delete
record later in the inventory. Three consecutive `delete("victim")` calls all
raised `CacheBlobLifecycleConflictError` without changing either record. This
violates D-09's repeated-delete resume contract and makes cleanup convergence
depend on unrelated global operation ordering.

**Fix:** Put an authenticated direct operation reference in the signed
tombstone, or maintain a durable store-bound `(key, tombstone_generation) ->
operation_id` index published before tombstone authority and exact-revalidated
against the signed operation record. If bounded scanning is retained as a
compatibility fallback, persist a key/generation-bound cursor so repeated calls
make monotonic progress and distinguish “budget exhausted” from “no matching
operation.” Add pinned-valid-record, limits-one/two, repeated-call, reopen, and
cross-process tests.

## Warnings

None.

## Verification Performed

- Full Phase 3 selection reached the final containment module with all prior
  tests passing; the POSIX Unix-socket fixture then failed because the managed
  sandbox forbids `AF_UNIX.bind`, an environment restriction rather than an
  implementation failure.
- Focused CAS, lifecycle, reconciliation, concurrency, and close modules were
  rerun separately; their terminal result is recorded by the parent workflow.
- Cross-store provenance replay probe: a v3 record from store A authenticated
  in store B under the same injected key and hid B's exact raw sidecar.
- Sparse-marker probe: one valid-looking marker hid an intact live manifest
  from paging, reconciliation, and clear while ordinary `get` still returned
  the payload.
- Repeated-delete probe: with supported limits of one, an unrelated retained
  operation permanently preceded the matching signed tombstone record and
  every repeated delete failed at the same lookup boundary.
- Known repository-wide Ruff debt, shutdown-only SQLite destructor behavior,
  and the base-install NumPy packaging issue were excluded as previously
  recorded out-of-scope concerns.

---

_Reviewed: 2026-09-02T20:23:52Z_
_Reviewer: the agent (gsd-code-reviewer), independent iteration 22_
_Depth: deep_
