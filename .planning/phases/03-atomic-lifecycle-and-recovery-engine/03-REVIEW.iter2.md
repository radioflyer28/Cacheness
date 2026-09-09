---
phase: 03-atomic-lifecycle-and-recovery-engine
reviewed: 2026-08-31T12:51:19Z
depth: deep
files_reviewed: 25
files_reviewed_list:
  - src/cacheness/__init__.py
  - src/cacheness/config.py
  - src/cacheness/error_handling.py
  - src/cacheness/storage/__init__.py
  - src/cacheness/storage/blob_store.py
  - src/cacheness/storage/clear_recovery.py
  - src/cacheness/storage/coordination.py
  - src/cacheness/storage/guarded_handler_io.py
  - src/cacheness/storage/lifecycle.py
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
  - tests/test_clear_recovery.py
  - tests/test_config_validation.py
  - tests/test_filesystem_containment.py
  - tests/test_manifest_repository_cas.py
  - tests/test_public_api_contract.py
findings:
  critical: 6
  warning: 1
  info: 0
  total: 7
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-08-31T12:51:19Z  
**Depth:** deep  
**Files Reviewed:** 25  
**Status:** issues_found

## Summary

The phase-focused suite passes (all reviewed test modules passed, with one Windows-only skip), but the implementation has six shipping blockers. The most serious defects are non-atomic operation-evidence CAS, a process-local-only clear snapshot boundary, and clear pages that can exceed their own evidence format for entirely valid manifests. The configured evidence limits are also not enforced at the repository boundary, the reconciliation cursor codec is neither size-bounded nor semantically confidential, and two public read surfaces break their typed concurrency contract. Successful clears additionally leak permanent control evidence.

This review does not count the documented repository-wide Ruff baseline or the known shutdown-only `SqliteBackend.__del__` issue.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: BLOCKER — “Exact” operation-evidence CAS is only local to one repository object

**Files:** `src/cacheness/storage/operation_repository.py:111-123`, `src/cacheness/storage/operation_repository.py:208-231`, `src/cacheness/storage/operation_repository.py:348-372`, `src/cacheness/storage/operation_repository.py:430-480`, `src/cacheness/storage/lifecycle.py:82-98`, `src/cacheness/storage/reconciliation.py:446-465`

**Issue:** `FileOperationRecordRepository` allocates its 32 conditional locks per instance. Every purported conditional update then performs a separate read, comparison, and durable replace/delete under only that instance-local lock. Two `BlobStore` instances create different repositories and therefore can both observe the same expected bytes and both succeed. This is also unprotected across processes. A checkpoint can overwrite another checkpoint, a stale retirement can delete newer evidence after its comparison, or a stale checkpoint can recreate evidence after another worker retired it. Clear progress can regress and recovery/reconciliation can lose or resurrect the evidence that is supposed to authorize cleanup.

The current tests prove stale rejection only by invoking transitions sequentially on one repository (`tests/test_blob_store_reconciliation.py:256-283` and `tests/test_blob_store_atomic_lifecycle.py:448-496`), so they do not exercise the actual independent-store boundary.

**Fix:** Put compare-plus-replace/delete behind a store-wide, cross-process conditional primitive. For the filesystem repository, use a fixed per-operation advisory lock (with shared in-process identity) around the exact read and durable replace/delete, or move operation evidence into a backend that provides transactional conditional mutation. Add deterministic tests using two independently constructed repositories for checkpoint-vs-checkpoint and checkpoint-vs-retire races.

### CR-02: BLOCKER — Clear’s target “snapshot” is not a cross-process snapshot and can delete post-start writes

**Files:** `src/cacheness/storage/coordination.py:20-47`, `src/cacheness/storage/lifecycle.py:263-332`, `src/cacheness/storage/manifest_repository.py:431-467`, `src/cacheness/storage/manifest_repository.py:740-780`, `tests/test_blob_store_atomic_lifecycle.py:517-570`

**Issue:** `StoreAdmissionBarrier` explicitly coordinates only the current process. Clear then builds its inventory by issuing multiple live lexical page queries. A writer in another process can commit between those page queries. If its key sorts after the current cursor, it becomes part of the clear inventory and is deleted even though it was created after clear began. A stream of increasing keys can also prevent the supposed finite snapshot from reaching a stable end. Manifest CAS protects individual keys; it does not establish an aggregate snapshot boundary.

The concurrency test uses two instances in one process and explicitly asserts that they share the same process-local barrier, so it cannot detect this defect.

**Fix:** Establish one backend-level stable snapshot for the inventory. SQLite can page within one consistent read transaction; JSON/in-memory need an immutable snapshot identity or a cross-process admission protocol that excludes publication only while the finite inventory is persisted. Add a subprocess test that commits a lexically later key between clear pages and proves the key is outside the target set.

### CR-03: BLOCKER — Valid manifests can overflow a clear target page and permanently block reopen

**Files:** `src/cacheness/config.py:361-365`, `src/cacheness/storage/lifecycle.py:274-305`, `src/cacheness/storage/operation_record.py:301-318`, `src/cacheness/storage/operation_record.py:386-417`

**Issue:** Manifest paging is bounded only by entry count. Clear embeds every exact manifest again as base64 inside one control page, but `_canonical_clear_bytes` caps the entire page at 1 MiB. Four valid manifests of about 200 KiB each already produce a control page over 1 MiB and make `clear()` raise `CacheManifestIntegrityError`. The main clear operation record has already been persisted at that point. On reopen, constructor recovery retries the same oversized page and fails again, making the store unavailable without manual intervention or a configuration change.

This was reproduced with four successful `put` calls whose metadata each contained a 200,000-character string; each canonical manifest was about 200,834 bytes, and `clear()` failed with `Clear control evidence exceeds the byte limit`.

**Fix:** Page by encoded byte budget as well as count. Build pages incrementally, accounting for base64 expansion and page/signature overhead, and persist a cursor only after a non-empty page fits. If one valid manifest cannot fit, use a bounded reference to separately persisted exact bytes rather than nesting the full record. Add a reopen test using several individually valid large manifests.

### CR-04: BLOCKER — Caller-owned evidence byte limits are unused and hostile files are fully read before rejection

**Files:** `src/cacheness/config.py:353-391`, `src/cacheness/storage/operation_record.py:36-40`, `src/cacheness/storage/operation_repository.py:169-177`, `src/cacheness/storage/operation_repository.py:313-322`, `src/cacheness/storage/operation_repository.py:408-418`, `src/cacheness/storage/operation_repository.py:571-585`

**Issue:** `LifecycleLimits.max_operation_record_bytes` and `max_operation_field_bytes` are validated but never consumed by the evidence codecs or repository. The codecs instead use fixed module constants. Smaller caller policy therefore has no effect. Worse, all operation, clear-page, clear-checkpoint, and reconciliation-checkpoint reads call `read_bytes()` before checking any size, so an attacker-controlled evidence file can force an unbounded allocation before the parser’s fixed 1 MiB check runs. Startup recovery and public reconciliation both traverse this path.

The oversized-evidence test calls `LifecycleOperationRecord.from_canonical_bytes` directly, after bytes are already resident, and therefore does not validate the repository boundary.

**Fix:** Make every repository read perform descriptor-anchored `get_size`/bounded streaming before allocation and reject anything larger than the exact caller-owned limit. Thread the same `LifecycleLimits` instance into all operation/clear codecs (or validate through repository-owned decode helpers) and apply `max_operation_field_bytes` consistently. Test with a smaller injected limit and assert the JSON parser and full-file read callback are never reached.

### CR-05: BLOCKER — Reconciliation resume tokens reuse a deterministic keystream and accept unbounded input

**File:** `src/cacheness/storage/reconciliation.py:718-787`

**Issue:** Resume-token “encryption” XORs every payload with the same deterministic SHA-256-derived mask because there is no nonce. Multiple tokens therefore expose XORs of their plaintexts; a token for a known/chosen cursor reveals keystream bytes that decrypt corresponding bytes in other tokens. This does not satisfy the phase’s opaque/non-disclosing cursor contract. The decoder also base64-decodes the entire caller-supplied token and allocates a same-sized mask before applying any size bound, creating another public memory/CPU denial-of-service path.

The current test checks only that one literal operation ID is not a substring of one encoded token; it does not test semantic confidentiality or oversized input.

**Fix:** Use a standard nonce-based AEAD construction (for example AES-GCM or ChaCha20-Poly1305) with a fresh random nonce and domain-separated key, or use a server-side opaque cursor identifier. Reject encoded length before base64 decoding and decoded length before authentication/decryption. Add round-trip, tamper, nonce-uniqueness, cross-token disclosure, and oversize-before-allocation tests.

### CR-06: BLOCKER — `exists()` and `list()` do not honor the public concurrent-read error contract

**Files:** `src/cacheness/storage/blob_store.py:558-598`, `src/cacheness/storage/blob_store.py:600-649`, `src/cacheness/storage/blob_store.py:1095-1121`, `tests/test_blob_store_concurrency.py:252-329`, `tests/test_blob_store_read_contract.py:632-696`

**Issue:** `get()` performs M1/snapshot/M2 validation and retries one proven generation change. `exists()` performs only M1 and then snapshots the old locator. An independent writer can commit and reclaim that locator between those steps, causing `exists()` to report `CacheBlobPayloadMissingError` for a key that has a healthy newer committed generation. `list()` first snapshots key names, then asserts that each later manifest lookup is non-`None`; an independent delete between those calls raises bare `AssertionError` rather than a typed lifecycle outcome. Neither method is protected by the per-instance key lock against independent instances.

The focused tests cover the race only for `get()` and cover `exists()`/`list()` only without an independent mutation.

**Fix:** Give `exists()` the same M1/snapshot/M2 generation loop as `get()` (without deserialization). For `list()`, replace the assertion with an explicit bounded policy: retry a stable page/snapshot, or raise `CacheBlobLifecycleConflictError` when a selected record disappears or changes. Add independent-store write/exists and delete/list race tests.

## Warnings

### WR-01: WARNING — Successful clear permanently retains target pages and progress checkpoints

**Files:** `src/cacheness/storage/lifecycle.py:379-434`, `src/cacheness/storage/operation_repository.py:135-188`, `src/cacheness/storage/operation_repository.py:265-383`

**Issue:** Clear retires only the 32-character main operation record after reaching terminal state. There is no repository method or lifecycle call that retires `clear-target-page-*` or `clear-target-checkpoint-*` files. Every successful clear therefore leaves two permanent control files per page containing keys, locators, exact manifest bytes, and completed-index history. The normal operation scanner deliberately skips those names, so neither startup recovery nor reconciliation ever removes them. This creates unbounded stale control state and retains sensitive metadata after the blobs were cleared.

**Fix:** Add exact conditional retirement for a completed checkpoint and its immutable page, execute it only after every target is terminal, and retire the main clear record last. Preserve resumability across interruption during this cleanup, and assert that a successful/reopened clear leaves no page/checkpoint artifacts.

---

_Reviewed: 2026-08-31T12:51:19Z_  
_Reviewer: the agent (gsd-code-reviewer)_  
_Depth: deep_
