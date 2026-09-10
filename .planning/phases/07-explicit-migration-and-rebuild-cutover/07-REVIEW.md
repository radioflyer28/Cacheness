---
phase: 07-explicit-migration-and-rebuild-cutover
reviewed: 2026-09-10T13:05:52Z
depth: standard
files_reviewed: 29
files_reviewed_list:
  - src/cacheness/storage/migration.py
  - src/cacheness/storage/migration_authority.py
  - src/cacheness/storage/migration_evidence.py
  - src/cacheness/storage/memory_lifecycle_authority.py
  - src/cacheness/storage/sqlite_lifecycle_authority.py
  - src/cacheness/storage/backends/postgresql_lifecycle_authority.py
  - src/cacheness/storage/backends/s3_backend.py
  - src/cacheness/storage/manifest.py
  - src/cacheness/storage/blob_store.py
  - src/cacheness/interfaces.py
  - src/cacheness/handlers.py
  - src/cacheness/storage/__init__.py
  - src/cacheness/error_handling.py
  - tests/test_migration_cutover.py
  - tests/test_stored_compatibility.py
  - tests/test_migration_plan_contract.py
  - tests/test_migration_inspection.py
  - tests/test_lifecycle_authority_contract.py
  - tests/contracts/test_postgresql_lifecycle_authority.py
  - tests/test_migration_run_evidence.py
  - tests/test_projection_sql_atomicity.py
  - tests/test_migration_remote_contract.py
  - tests/test_rebuild_workflow.py
  - tests/test_handler_registration.py
  - tests/test_migration_public_contract.py
  - tools/verify_phase7_contracts.py
  - tests/test_phase7_contract_verifier.py
  - docs/BACKEND_SELECTION.md
  - docs/STORAGE_MIGRATION.md
findings:
  critical: 6
  warning: 2
  info: 0
  total: 8
status: issues_found
---

# Phase 7: Code Review Report

**Reviewed:** 2026-09-10T13:05:52Z
**Depth:** standard
**Files Reviewed:** 29
**Status:** issues_found

## Narrative Findings (AI reviewer)

## Summary

The Phase 7 implementation has six shipping blockers in migration/rebuild recovery, compatibility enforcement, metadata confidentiality, PostgreSQL worker fencing, and the acceptance verifier. These are finite contract defects, not requests for stronger cross-topology ACID or scheduling guarantees. The known clear/delete snapshot race accepted under ADR 0001 is not repeated as a finding, and none of the fixes below requires another process-local lock, queue, lease, or lifecycle authority.

Two additional warnings affect evidence durability and custom-handler validation. Live PostgreSQL/S3/Windows/performance qualification remains correctly deferred to Phase 8.

## Critical Issues

### CR-01 [BLOCKER]: Migration staging can deterministically strand untracked candidates and cannot abort a partially staged run

**Files:** `src/cacheness/storage/migration_authority.py:21`, `src/cacheness/storage/migration_authority.py:184`, `src/cacheness/storage/migration.py:2333`, `src/cacheness/storage/migration.py:2442`, `src/cacheness/storage/migration.py:2911`

**Issue:** Inventory is paged, but the candidate receipt rejects a whole-store candidate containing more than 256 entries. `stage()` performs every external candidate write before it computes that digest and before it records any candidate receipt. A 257-entry store therefore writes all 257 candidate payloads, raises `ValueError`, and leaves evidence in `STAGING`. `abort()` explicitly rejects `STAGING`, so those effects cannot be retired through the public workflow. The same loss of attribution occurs on any exception after an earlier candidate write but before the final `STAGED` checkpoint. In addition, if `abort()` deletes some candidates and a later deletion fails, it records no progress; retry first requires every candidate to still exist and therefore fails on the already deleted ones. This violates MIGR-05 and D-19/D-21's resumable, attributable cleanup contract.

**Fix:** Keep the authority as the sole visibility owner, but checkpoint authenticated, bounded candidate-batch descriptors in maintenance evidence after each external batch. Derive the final whole-store digest incrementally (without a total-entry cap), resume by re-verifying exact recorded outputs, and let abort treat an exact recorded candidate as either matching or already absent while retaining retryable debt for the remainder. Add deterministic tests for 257+ entries, interruption after the first batch, and failure after the first abort deletion.

### CR-02 [BLOCKER]: Rebuild evidence cannot resume or clean a crash/verification failure

**File:** `src/cacheness/storage/migration.py:2042`, `src/cacheness/storage/migration.py:2091`, `src/cacheness/storage/migration.py:2189`, `src/cacheness/storage/migration.py:2235`, `src/cacheness/storage/migration.py:3141`

**Issue:** `stage_rebuild()` holds destination receipts only in a local list and checkpoints just one aggregate digest after the entire rebuild. A crash in `REBUILDING` leaves authoritative destination entries but no exact receipt set in evidence. Re-entering `stage_rebuild()` encounters those keys as pre-existing and aborts, while `resume()` has no branch for any `REBUILDING`/`REBUILD_*` state. Worse, verification failures call `_abort_rebuild_after_failure(..., ())`, so the cleanup routine receives no receipts and marks evidence `ABORTED` while leaving all staged destination entries live. This contradicts Plan 07-09's resumable rebuild result and makes an interrupted rebuild neither resumable nor deterministically cleanable.

**Fix:** Persist exact `BlobReceipt` identity (key, generation, locator, expectation) for each bounded rebuild batch in authenticated evidence. On resume, verify recorded receipts against destination authority state, continue only missing planned keys, and on failure delete only entries still matching those exact receipts while recording debt for failures. Extend `resume()` to the rebuild states and add crash tests before/after each entry and during verification.

### CR-03 [BLOCKER]: Directed compatibility targets and handler transformations are ignored by execution

**Files:** `src/cacheness/storage/migration.py:433`, `src/cacheness/storage/migration.py:1790`, `src/cacheness/storage/migration.py:2168`, `src/cacheness/storage/migration.py:2420`, `src/cacheness/handlers.py:1476`

**Issue:** `MigrationCompatibilityEdge.supports()` compares only the source version; its `destination` is never consulted. Inspection consequently marks an entry migratable from any edge with a matching source, and `stage()` copies the original payload while changing only the locator in the original manifest. The handler-owned `resolve_payload_transformation()`/`transform_payload()` seam has no migration or rebuild caller; rebuild simply deserializes the old value and performs an ordinary destination `put_entry()`. A future source-to-target edge can therefore be advertised while producing source-version bytes/manifests rather than the declared destination contract, or while silently bypassing the exact transform that D-10 says must own the conversion.

**Fix:** Bind plan classification to both the exact source and the actual destination contract. For identity-compatible entries, retain verified byte copy. For changed payload contracts, resolve exactly one registered handler transformation edge and use its guarded result to build a destination-version manifest. If no exact executable edge exists, classify the entry rebuild-only/refused. Add an integration test whose source and destination format versions differ and assert both the transform call and destination manifest identity.

### CR-04 [BLOCKER]: Canonical machine plans serialize arbitrary catalog metadata and full manifests despite the no-credentials claim

**Files:** `src/cacheness/storage/migration.py:1032`, `docs/STORAGE_MIGRATION.md:61`

**Issue:** Every plan entry emits `catalog_values` verbatim and embeds the complete signed manifest as base64. Manifests themselves contain catalog values, user metadata, and handler metadata. Because those application-defined mappings may contain passwords, tokens, connection strings, or provider paths, `plan.to_canonical_bytes()` directly serializes the classes of material the runbook says plans never contain. The Phase 7 secret-output checks do not inspect this serialization path. This is an information-disclosure risk when machine plans are stored, attached to tickets, or transferred for operator approval.

**Fix:** Make the persisted/shareable plan contain only bounded non-secret identifiers, counts, classifications, and authenticated digests. Re-read and authenticate the manifest/catalog values at the execution boundary to preserve D-11, with the plan binding their digest rather than embedding their contents. If any metadata must remain in an operator artifact, define an explicit sensitive-field policy and protected artifact format instead of claiming unconditional redaction.

### CR-05 [BLOCKER]: A fresh PostgreSQL-backed BlobStore can initialize through `activated_offline`

**Files:** `src/cacheness/storage/backends/postgresql_lifecycle_authority.py:944`, `src/cacheness/storage/blob_store.py:332`

**Issue:** `publication_state()` returns `IDLE` without consulting PostgreSQL whenever the authority instance has not yet loaded `store_identity`. The `BlobStore.initialize()` decorator performs the worker fence before initialization, sees that synthetic `IDLE`, then validates/opens the existing database but never re-runs the fence. A newly constructed worker can therefore report successful initialization against an authority whose persisted state is `activated_offline`, contrary to RQ-03 and the runbook's promise that initialization is blocked until rollback or finalize. Subsequent data operations happen to fence, but the explicit startup boundary has already returned success and may allow the application to announce readiness.

**Fix:** After authority initialization/open validation loads persisted identity, call the same `_require_canonical_store()` fence again before materializing payload resources or setting `_initialized`. Add a deterministic PostgreSQL adapter test that constructs a fresh authority over persisted `activated_offline` state and requires `BlobStore.initialize()` to raise `CacheBlobMigrationOfflineDecisionRequiredError`.

### CR-06 [BLOCKER]: The fixed verifier can report decision/threat coverage after the relevant test is removed

**File:** `tools/verify_phase7_contracts.py:113`

**Issue:** Requirement, decision, threat, and assumption maps point only to whole test filenames. `validate_fixed_manifest()` checks that each filename exists and belongs to the fixed file set, then pytest runs the entire files. It never binds an item to an exact test function or verifies that the mapped behavior still exists. For example, every D-10-specific assertion can be deleted from `tests/test_handler_registration.py`; the mapping remains non-empty, unrelated tests in that file pass, and the verifier still prints D-10/MIGR-06 as PASS. This directly contradicts Plan 07-11 and `07-VALIDATION.md`, which claim named executable evidence and omission detection for every decision/threat.

**Fix:** Map each requirement/decision/threat to exact `path::test_name` selectors, parse the test modules to confirm each selector exists, reject duplicate/empty selectors, and pass those exact nodes to pytest. Add verifier self-tests that remove or rename one mapped function while leaving its file and unrelated tests intact.

## Warnings

### WR-01 [WARNING]: Evidence replacement is not made durable at the directory boundary

**File:** `src/cacheness/storage/migration_evidence.py:696`

**Issue:** `_atomic_write()` fsyncs the temporary file and renames it, but never fsyncs the parent directory. On filesystems where rename durability requires a directory sync, a power loss can discard the newly created evidence name or restore the previous checkpoint. The canonical source remains safe, but the promised resumable maintenance evidence can be lost at an acknowledged transition.

**Fix:** On supported POSIX filesystems, fsync an opened parent-directory descriptor after `os.replace`; provide a documented platform fallback where directory fsync is unavailable. Add a fault-injection test around the post-rename durability step and keep failure fail-closed.

### WR-02 [WARNING]: A handler can advertise transformations while inheriting the always-failing base implementation

**Files:** `src/cacheness/interfaces.py:197`, `src/cacheness/handlers.py:1674`

**Issue:** Registry validation checks only that `transform_payload` is callable. Any `CacheHandler` subclass inherits the callable base method, which always raises `CacheFormatError`, so it may declare non-empty transformation edges and still pass registration. The registry then resolves the edge as supported even though execution cannot succeed.

**Fix:** When edges are non-empty, require the concrete handler type to override `CacheHandler.transform_payload` (or replace the default with an abstract method/capability object) and add a registration test for a subclass that declares an edge without implementing the transform.

---

_Reviewed: 2026-09-10T13:05:52Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
