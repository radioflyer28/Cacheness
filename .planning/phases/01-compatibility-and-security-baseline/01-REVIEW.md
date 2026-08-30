---
phase: 01-compatibility-and-security-baseline
reviewed: 2026-08-30T00:32:17Z
depth: standard
files_reviewed: 56
files_reviewed_list:
  - README.md
  - docs/SECURITY.md
  - src/cacheness/__init__.py
  - src/cacheness/config.py
  - src/cacheness/core.py
  - src/cacheness/decorators.py
  - src/cacheness/error_handling.py
  - src/cacheness/handlers.py
  - src/cacheness/interfaces.py
  - src/cacheness/metadata.py
  - src/cacheness/query_validation.py
  - src/cacheness/security.py
  - src/cacheness/sql_cache.py
  - src/cacheness/storage/backends/blob_backends.py
  - src/cacheness/storage/blob_store.py
  - src/cacheness/storage/guarded_handler_io.py
  - src/cacheness/storage/path_security.py
  - tests/fixtures/compat/README.md
  - tests/fixtures/compat/array-raw-v035-compress/payload.b2nd
  - tests/fixtures/compat/array-raw-v037-compress2/payload.b2nd
  - tests/fixtures/compat/decorator-key-v0313/metadata.json
  - tests/fixtures/compat/decorator-key-v0313/payload.npz
  - tests/fixtures/compat/decorator-key-v0313/provenance.json
  - tests/fixtures/compat/json-nested-v0314/metadata.json
  - tests/fixtures/compat/json-nested-v0314/payload.npz
  - tests/fixtures/compat/json-nested-v0314/provenance.json
  - tests/fixtures/compat/json-split-signed-v038/metadata.json
  - tests/fixtures/compat/json-split-signed-v038/payload.npz
  - tests/fixtures/compat/json-split-signed-v038/provenance.json
  - tests/fixtures/compat/json-split-unsigned-v037/metadata.json
  - tests/fixtures/compat/json-split-unsigned-v037/payload.npz
  - tests/fixtures/compat/json-split-unsigned-v037/provenance.json
  - tests/fixtures/compat/manifest.json
  - tests/fixtures/compat/sqlite-columns-v0314/metadata.sqlite3
  - tests/fixtures/compat/sqlite-columns-v0314/payload.npz
  - tests/fixtures/compat/sqlite-columns-v0314/provenance.json
  - tests/fixtures/compat/sqlite-metadata-json-v039/metadata.sqlite3
  - tests/fixtures/compat/sqlite-metadata-json-v039/payload.npz
  - tests/fixtures/compat/sqlite-metadata-json-v039/provenance.json
  - tests/fixtures/compat/validate_corpus.py
  - tests/test_blob_backend_registry.py
  - tests/test_cache_integrity.py
  - tests/test_config_validation.py
  - tests/test_core.py
  - tests/test_directory_sharding.py
  - tests/test_filesystem_containment.py
  - tests/test_handlers.py
  - tests/test_legacy_array_security.py
  - tests/test_phase1_quality_gates.py
  - tests/test_public_api_contract.py
  - tests/test_query_meta.py
  - tests/test_query_meta_security.py
  - tests/test_security_documentation.py
  - tests/test_sql_cache.py
  - tests/test_sql_cache_failure_contract.py
  - tests/test_stored_compatibility.py
findings:
  critical: 6
  warning: 0
  info: 0
  total: 6
status: issues_found
---

# Phase 1: Code Review Report

**Reviewed:** 2026-08-30T00:32:17Z
**Depth:** standard
**Files Reviewed:** 56
**Status:** issues_found

## Summary

The prior two fix iterations close their exact reported reproductions: failed overwrite digests and strict signing preserve the committed entry, finite-value validation excludes stored NaN/infinities, and pre-commit `BlobStore.clear()` failures restore the tested state. The final pass still found six release-blocking lifecycle and boundary failures. Three are direct regressions or incomplete variants of the new fixes: regular-file staging swaps are published, candidate payloads survive metadata-commit failure, and final clear tombstones are not durably recoverable. The remaining failures are deterministic state corruption in `BlobStore.put()` and `UnifiedCache.clear_all()`, plus an unhandled valid-integer query boundary.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: Descriptor-mode staging publishes a regular file swapped in after validation

**Classification:** BLOCKER

**File:** `src/cacheness/storage/guarded_handler_io.py:82-128,130-209`

**Issue:** `_staged_artifact()` validates an inode but returns only its resolved pathname. `_open_staged_artifact()` then opens that pathname later and checks only that the newly opened inode is a regular single-link file; descriptor mode never compares it with the inode validated earlier. Replacing the validated artifact with another ordinary file between those calls therefore succeeds. A direct reproduction replaced the validated `good` artifact with an ordinary file containing `swapped`; `GuardedHandlerIO.put()` published `b"swapped"`. The new tests cover a symlink leaf and symlink ancestor, but not an ordinary-file or ordinary-directory replacement. This defeats the claimed validation/open binding for handler-controlled publication.

**Fix:** Open the staged artifact during validation and carry that descriptor (or its recorded `st_dev`/`st_ino` identity) into publication. If reopening is unavoidable, compare the opened descriptor with the exact validated identity after the descriptor-relative walk and reject any mismatch. Add leaf regular-file and ancestor ordinary-directory replacement regressions in both descriptor and fallback modes.

### CR-02: `BlobStore.put()` corrupts an existing value when metadata commit fails

**Classification:** BLOCKER

**File:** `src/cacheness/storage/blob_store.py:159-188`

**Issue:** `BlobStore.put()` publishes an overwrite to the deterministic live `storage_id` before calling `backend.put_entry()`. If metadata publication raises, the old metadata remains but its locator now contains the replacement bytes. A direct reproduction wrote `{"v": "old"}`, forced `put_entry()` to fail during replacement, observed the exception and unchanged metadata, then `get("k")` returned `{"v": "new"}`. The operation reports failure while silently changing committed data, and a changed handler type/format can instead make the old entry unreadable.

**Fix:** Give `BlobStore.put()` a private-candidate protocol: serialize to a unique candidate, complete validation, then atomically publish metadata/locator ownership. Preserve the prior payload and metadata until commit succeeds, discard only the candidate on failure, and delete the prior payload only after the new record is authoritative. Add first-write and overwrite metadata-failure tests with differing handler formats.

### CR-03: `UnifiedCache.put()` leaks arbitrary candidate payloads when metadata commit fails

**Classification:** BLOCKER

**File:** `src/cacheness/core.py:1064-1163`

**Issue:** The candidate fix cleans up only the explicit invalid-digest and strict-signing branches. `metadata_backend.put_entry()` is outside any rollback handler, so a commit exception leaves the already-published candidate in managed storage with no metadata owner. A direct reproduction forced `put_entry()` to raise and found a complete `*-candidate-*.pkl` payload afterward. These candidates may contain sensitive or executable serialized application data, are not discoverable through normal cache cleanup, and accumulate on retries. The same gap applies to exceptions after guarded publication but before the two explicit cleanup branches.

**Fix:** Track candidate ownership around the entire pre-commit region. In one `try/finally`, delete the candidate on every path until metadata commit has definitely succeeded; treat a false/failed delete as an explicit reconciliation error. Add injected failures for snapshot opening, digesting, signing, and metadata publication, asserting no candidate residue and exact preservation of an overwritten entry.

### CR-04: `UnifiedCache.clear_all()` still performs irreversible prefix deletion

**Classification:** BLOCKER

**File:** `src/cacheness/core.py:1466-1482`

**Issue:** `clear_all()` deletes payloads sequentially and clears metadata only afterward. If any later delete raises, earlier payloads are gone while all metadata records remain. A two-entry reproduction failed the second deletion and left one payload missing with both metadata rows still committed. This is the same deterministic partial-data-loss pattern that the second iteration fixed only in `BlobStore.clear()`.

**Fix:** Route `UnifiedCache.clear_all()` through a transactional/reconciliation protocol: stage every payload into durable same-root tombstones, clear metadata only after staging succeeds, restore all payloads on any pre-commit failure, and durably journal post-commit finalization. Inject failures at every payload position, during metadata clear, and during finalization.

### CR-05: Finalization-failed clear tombstones have no durable recovery identity

**Classification:** BLOCKER

**File:** `src/cacheness/storage/blob_store.py:401-415,419-490`

**Issue:** After metadata is successfully cleared, a tombstone-delete failure raises and deliberately leaves `clear-tombstone-<uuid>` files. The only mapping from those random names to their original payload locators is the local `staged_payloads` list, which is discarded when `clear()` returns. There is no persisted journal, encoded original identity, startup reconciliation, or public recovery operation. The new test labels these files “recoverable” but asserts only that they remain; after close/restart they are anonymous retained copies. Thus `clear()` can remove all records, return failure, and indefinitely retain the supposedly deleted payload bytes—an erasure/confidentiality and deterministic-reconciliation failure.

**Fix:** Persist an atomic clear journal mapping each tombstone to its original locator and operation state before deleting live payloads. On startup or an explicit recovery call, deterministically roll forward committed clears (remove tombstones) or roll back uncommitted clears. Test close/reopen after every final-delete failure and prove the journal drives a terminal state.

### CR-06: Valid large Python integer filters collapse into an untyped query miss

**Classification:** BLOCKER

**File:** `src/cacheness/query_validation.py:58-67`; `src/cacheness/core.py:642-683,723-727`

**Issue:** Numeric validation rejects only non-finite floats. Python integers outside SQLite's signed 64-bit bind range are accepted, then passed directly as bind values. SQLite raises `OverflowError: Python int too large to convert to SQLite INTEGER`; the broad query catch logs it and returns `None`. A direct reproduction stored `score=10**100` and queried `score=10**99`; instead of returning the matching entry or a typed validation error, `query_meta()` returned `None`. These are valid Python values under the documented raw numeric threshold contract, and `None` is indistinguishable from unsupported backend/configuration.

**Fix:** Define and enforce a numeric domain before opening the session. Either preserve arbitrary integer ordering with a decimal-safe representation/comparison, or reject values outside the supported backend range with `CacheQueryValidationError(reason=invalid_query_value)`. Never swallow numeric-domain failures into `None`. Add values at and beyond `-(2**63)`/`2**63-1`, very large positive/negative integers, and mixed int/float boundary tests.

---

_Reviewed: 2026-08-30T00:32:17Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
