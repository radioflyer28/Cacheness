---
phase: 04-metadata-composition-and-topology-contracts
reviewed: 2026-09-08T04:29:37Z
depth: standard
files_reviewed: 60
files_reviewed_list:
  - AGENTS.md
  - CONTEXT.md
  - docs/CATALOG_AND_TOPOLOGY.md
  - docs/STORAGE_INITIALIZATION.md
  - src/cacheness/__init__.py
  - src/cacheness/config.py
  - src/cacheness/core.py
  - src/cacheness/decorators.py
  - src/cacheness/error_handling.py
  - src/cacheness/metadata.py
  - src/cacheness/storage/__init__.py
  - src/cacheness/storage/backends/__init__.py
  - src/cacheness/storage/backends/blob_backends.py
  - src/cacheness/storage/backends/postgresql_backend.py
  - src/cacheness/storage/blob_store.py
  - src/cacheness/storage/catalog.py
  - src/cacheness/storage/composition.py
  - src/cacheness/storage/lifecycle.py
  - src/cacheness/storage/lifecycle_authority.py
  - src/cacheness/storage/manifest.py
  - src/cacheness/storage/memory_lifecycle_authority.py
  - src/cacheness/storage/projections.py
  - src/cacheness/storage/read_contract.py
  - src/cacheness/storage/sqlite_lifecycle_authority.py
  - tests/_lifecycle_test_support.py
  - tests/fixtures/phase4_ruff_baseline.json
  - tests/test_blob_manifest.py
  - tests/test_blob_manifest_backends.py
  - tests/test_blob_store_atomic_lifecycle.py
  - tests/test_blob_store_close_contract.py
  - tests/test_blob_store_composition.py
  - tests/test_blob_store_concurrency.py
  - tests/test_blob_store_integrity.py
  - tests/test_blob_store_legacy_contract.py
  - tests/test_blob_store_read_contract.py
  - tests/test_blob_store_reconciliation.py
  - tests/test_cached_custom_metadata.py
  - tests/test_catalog_projection.py
  - tests/test_catalog_query_contract.py
  - tests/test_catalog_schema.py
  - tests/test_clear_recovery.py
  - tests/test_config_validation.py
  - tests/test_custom_metadata.py
  - tests/test_filesystem_containment.py
  - tests/test_manifest_repository_cas.py
  - tests/test_metadata.py
  - tests/test_metadata_backend_registry.py
  - tests/test_metadata_role_contract.py
  - tests/test_phase3_local_workflows.py
  - tests/test_phase3_scheduler_retirement.py
  - tests/test_phase3_windows_contract.py
  - tests/test_postgresql_backend.py
  - tests/test_projection_mutation_contract.py
  - tests/test_projection_sql_atomicity.py
  - tests/test_public_api_contract.py
  - tests/test_sqlite_bootstrap_concurrency.py
  - tests/test_topology_capabilities.py
  - tests/test_unified_cache_adversarial_lifecycle.py
  - tests/test_unified_cache_lifecycle_authority.py
  - tools/verify_phase4_ruff_delta.py
findings:
  critical: 6
  warning: 2
  info: 0
  total: 8
status: issues_found
---

# Phase 4: Code Review Report

**Reviewed:** 2026-09-08T04:29:37Z
**Depth:** standard
**Files Reviewed:** 60
**Status:** issues_found

## Summary

The clean-cutover direction is sound and the focused catalog/topology matrix passes, but the current implementation does not yet deliver the Phase 4 contract end to end. The largest gap is that `BlobStore` cannot write declared catalog values at all: its public write/update methods only populate untyped user metadata, while catalog query tests seed descriptors through private authority primitives. The selected filesystem payload participant is also ignored by the lifecycle engine, and application-registered named participants cannot reach `BlobStore` through the advertised composition path.

Projection failure handling, input bounds, and composition validation have additional fail-closed defects. These findings do not recommend locks, queues, readiness registries, cross-resource ACID, compatibility shims, or another lifecycle authority; the fixes stay inside the single `BlobStore`/authority boundary required by ADR 0001.

The reviewed Phase 4 test scope currently has 6 failures (with 5 capability/platform skips). Three failures use a retired private signing helper and three retain pre-cutover facade assumptions. The repository-wide collection additionally stops on tests outside this review scope that still import removed metadata-authority symbols, plus optional pandas-dependent SQL-cache modules; this agrees with `04-VALIDATION.md` and must not be “fixed” by restoring compatibility exports.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: [BLOCKER] Declared catalog metadata cannot be written or updated through BlobStore

**File:** `/Users/akriz/code/cacheness/src/cacheness/storage/lifecycle.py:240-257`

**Issue:** Every normal put hard-codes the default catalog identity and writes `catalog_values={}`/`catalog_presence=()`. Correspondingly, `BlobStore.put_entry()` accepts only `metadata`, and `update_metadata()` only patches `user_metadata` (`blob_store.py:353-393`). Therefore a direct user cannot store, validate, or update values for a `CatalogSchema`, and a subsequent `query_catalog(..., schema=application_schema)` returns no matches. The passing end-to-end query test bypasses `BlobStore` and manually constructs, signs, verifies, and promotes manifests through private authority primitives (`tests/test_catalog_query_contract.py:237-273`). This leaves BACK-07 and D-01 through D-04 unimplemented on the supported public path.

**Fix:** Add one explicit catalog input to the clean `BlobStore` write/update surface (for example `catalog_values` plus `catalog_schema`), validate it with `CatalogSchema.validate_mapping()` before handler staging, and bind the schema ID/revision/fingerprint, stored values, and exact presence set into the canonical manifest. Add a same-generation catalog-update operation through the existing authority transaction seam. Replace the private-authority test seeding with public put/update/query round trips; do not add a second metadata backend or compatibility adapter.

### CR-02: [BLOCKER] The selected filesystem payload participant is not used by the lifecycle engine

**File:** `/Users/akriz/code/cacheness/src/cacheness/storage/blob_store.py:566-574`

**Issue:** `_materialize_authority_store()` uses the selected participant only when it is `InMemoryBlobBackend`. Every other payload participant is discarded and replaced with `GuardedHandlerIO(self.cache_dir)`. A topology can select a `FilesystemBlobBackend(base_dir=A)` while `BlobStore(cache_dir=B)` writes every payload under B and leaves A empty. This violates exact participant selection (BACK-03/D-10) and means the advertised payload role is a capability label rather than the engine supplying storage operations.

**Fix:** Define the narrow immutable-generation/handler-I/O primitive required by `AuthorityLifecycleEngine` and have the selected payload participant supply it. For filesystem storage, build guarded I/O from that participant's already validated root; for memory, retain the existing memory adapter. Reject participants that cannot supply the primitive during composition. Keep promotion and recovery sequencing in `BlobStore`/the lifecycle authority rather than moving it into payload adapters.

### CR-03: [BLOCKER] Named application registrations cannot be composed into BlobStore

**File:** `/Users/akriz/code/cacheness/src/cacheness/storage/composition.py:521-539`

**Issue:** `StoreTopology.resolve()` creates a fresh built-in-only `RoleRegistry` whenever no registry argument is supplied, and `BlobStore.__init__()` always calls `topology.resolve()` without any way to supply an application registry (`blob_store.py:162`). Thus an application can register a name in a `RoleRegistry`, but it cannot construct a `BlobStore` using that name. The still-public `register_blob_backend()` API writes to a separate `_blob_backend_registry` (`blob_backends.py:450-512`) which `StoreTopology` never consults, making that advertised registration path doubly disconnected. BACK-03's registered-name selection guarantee is not available at the high-level composition root.

**Fix:** Make the registry an explicit part of the one clean composition root (for example, an immutable `registry` field on `StoreTopology` or a `registry=` argument on `BlobStore`) and resolve all named roles through it. Consolidate or remove the disconnected legacy blob registry and its package exports as part of the approved pre-production cutover; do not maintain two synchronized registries or add a compatibility shim.

### CR-04: [BLOCKER] Unexpected projection exceptions escape after a successful authority commit without the receipt

**File:** `/Users/akriz/code/cacheness/src/cacheness/storage/projections.py:49-55`

**Issue:** `best_effort()` and `refresh()` catch only a hand-picked tuple of exception classes (`projections.py:316-364`). A valid external sink can raise a custom `Exception`, `LookupError`, `AttributeError`, or driver-specific exception. That exception escapes `BlobStore.put_entry()` after the blob has already committed, so the caller receives an apparent failed put with no `BlobReceipt`; explicitly requested refresh likewise fails to produce the required committed-partial error. This violates D-15 and can induce unsafe blind retries even though canonical data is present. A custom `Exception` raised from `save_projection_checkpoint()` reproduces the defect while `get_entry_info(key)` confirms the entry committed.

**Fix:** At the external projection boundary, catch `Exception` (not `BaseException`) and translate every sink failure into a named dirty `ProjectionOutcome` for best-effort delivery or `CacheBlobCommittedPartialError` carrying the exact receipt for requested refresh. Continue allowing `KeyboardInterrupt`, `SystemExit`, and other `BaseException` subclasses to propagate. Add a custom-exception regression test.

### CR-05: [BLOCKER] Opaque catalog cursors are decoded without an encoded-size bound

**File:** `/Users/akriz/code/cacheness/src/cacheness/storage/catalog.py:523-530`

**Issue:** Public page validation checks only that a cursor is a non-empty string (`catalog.py:400-402`), then `CatalogCursor.inspect()` base64-decodes the entire caller-controlled string before applying any size bound. An arbitrarily large cursor can force correspondingly large allocations and JSON parsing work. This contradicts the documented bounded portable query contract and leaves an avoidable denial-of-service boundary in every authority implementation.

**Fix:** Define a maximum encoded cursor byte length derived from the finite cursor schema, reject longer strings before base64 decoding, and bound decoded bytes before JSON parsing. Also bound the cursor's string fields (including schema/store/query identities) at construction and inspection, then add over-limit tests that verify authority dispatch is never reached.

### CR-06: [BLOCKER] Composition accepts invalid authorities and can leak store-owned participants on role-validation failure

**File:** `/Users/akriz/code/cacheness/src/cacheness/storage/composition.py:619-648`

**Issue:** `_validate_participant_role()` only rejects two known cross-role concrete types; it does not require an authority to satisfy the runtime-checkable `LifecycleAuthority` protocol or require a payload participant to supply the engine's payload primitive. `StoreTopology(object(), object()).resolve()` therefore succeeds and reports a topology that cannot operate, contrary to D-11's construction-time failure rule. For named factories, `RoleRegistry.construct()` validates the returned object before `_resolve_ref()` adds it to the ownership ledger (`composition.py:434-443`), so a newly created store-owned resource that fails role validation is never closed. A transfer-owned injected invalid participant has the same gap because it is validated before being appended.

**Fix:** Validate each participant structurally against a role-specific protocol during resolution. Add a just-created or transfer-owned participant to a local ownership guard before validation, and close it exactly once if validation or later composition fails. Reject duplicate store-owned references or deduplicate the close ledger by identity so one object cannot be closed multiple times.

## Warnings

### WR-01: [WARNING] Rebuild capability checks use the aggregate topology instead of the selected projection's capability

**File:** `/Users/akriz/code/cacheness/src/cacheness/storage/blob_store.py:246-255`

**Issue:** Every `ProjectionController` receives `self.capabilities`, whose projection booleans are composed with `all(...)` across every configured projection (`composition.py:698-723`). If one sink supports offline rebuild and another does not, the aggregate reports false and `rebuild_projection()` rejects rebuilding the capable sink. D-16 requires each adapter to report its own online/offline mode.

**Fix:** Pass `ParticipantCapabilities.from_participant(projection, BackendRole.PROJECTION.value)` to each controller. Keep the aggregate topology report for callers asking what the whole topology guarantees, but use the named sink's local capability for a named rebuild operation. Add a two-projection test with different rebuild modes.

### WR-02: [WARNING] Retained Phase 4 regression tests are stale after the atomic cutover

**File:** `/Users/akriz/code/cacheness/tests/test_blob_store_integrity.py:816-826`

**Issue:** Three integrity tests call the removed private `_manifest_key()` helper instead of the current authority-key seam. In `tests/test_unified_cache_lifecycle_authority.py`, the distinct-key test requires a removed `_lock` before exercising the behavior (line 75), invalidation still expects a conflict to escape although policy deliberately converts an exact-generation conflict to a non-deletion result (lines 215-219), and the relative-root test asserts an authored public path is absolute rather than verifying the guarded resolved path and round trip (lines 327-332). Running all reviewed Phase 4 test modules yields 6 failures. These failures hide the intended security/concurrency assertions and contradict Plan 04-06/07's claim that retained regressions were fully migrated.

**Fix:** Update the signing helper to use `_authority_manifest_key()` (or a purpose-built test fixture), remove the obsolete `_lock` representation assertion while retaining the distinct-key overlap behavior, assert that invalidation preserves the replacement generation without requiring a particular policy-level exception, and inspect the guarded resolved root rather than `cache_dir`. Do not restore private aliases, facade locks, or old path representations.

---

_Reviewed: 2026-09-08T04:29:37Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
