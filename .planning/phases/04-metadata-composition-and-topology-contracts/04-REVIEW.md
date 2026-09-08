---
phase: 04-metadata-composition-and-topology-contracts
reviewed: 2026-09-08T07:48:57Z
depth: standard
files_reviewed: 33
files_reviewed_list:
  - docs/CATALOG_AND_TOPOLOGY.md
  - examples/custom_metadata_demo.py
  - metadata.py
  - src/cacheness/__init__.py
  - src/cacheness/storage/__init__.py
  - src/cacheness/storage/backends/__init__.py
  - src/cacheness/storage/backends/blob_backends.py
  - src/cacheness/storage/backends/s3_backend.py
  - src/cacheness/storage/blob_store.py
  - src/cacheness/storage/catalog.py
  - src/cacheness/storage/composition.py
  - src/cacheness/storage/guarded_handler_io.py
  - src/cacheness/storage/lifecycle.py
  - src/cacheness/storage/memory_lifecycle_authority.py
  - src/cacheness/storage/projections.py
  - tests/fixtures/phase4_ruff_baseline.json
  - tests/test_blob_backend_registry.py
  - tests/test_blob_store_composition.py
  - tests/test_blob_store_integrity.py
  - tests/test_cached_query_meta.py
  - tests/test_catalog_projection.py
  - tests/test_catalog_query_contract.py
  - tests/test_core.py
  - tests/test_filesystem_containment.py
  - tests/test_lifecycle_authority_contract.py
  - tests/test_phase3_gap_acceptance.py
  - tests/test_public_api_contract.py
  - tests/test_s3_blob_backend.py
  - tests/test_sqlite_metadata_bootstrap_atomicity.py
  - tests/test_stored_compatibility.py
  - tests/test_topology_capabilities.py
  - tests/test_unified_cache_lifecycle_authority.py
  - tools/verify_phase4_cutover.py
findings:
  critical: 1
  warning: 2
  info: 0
  total: 3
status: issues_found
---

# Phase 4: Post-Fix Code Review Report

**Reviewed:** 2026-09-08T07:48:57Z
**Depth:** standard
**Files Reviewed:** 33
**Status:** issues_found

## Summary

The gap closure fixes the original six blockers and two warnings without adding a lock, queue, readiness registry, second authority, compatibility shim, or universal topology guarantee. Public catalog values now flow through `BlobStore`; the selected payload participant supplies generation I/O; application registrations reach the one topology registry; projection failures preserve committed receipts; cursor work is bounded; structural role/ownership checks fail closed; named rebuild uses per-sink capability; and the previously stale regression modules pass.

One release-blocking composition inconsistency remains: the built-in registry advertises JSON and PostgreSQL projection names whose constructed participants cannot satisfy the projection protocol that `StoreTopology.resolve()` now enforces. Two review-quality issues also remain in the cutover verifier and workspace hygiene.

Focused verification passed: the Phase 4 41-module matrix reports 588 passed and 6 capability/platform skips on Python 3.13; the direct prior-finding regression selection passes; the Ruff delta passes; and `git diff --check` is clean. The separately documented pandas SQL-cache collection diagnostic remains explicitly non-green and deferred to Phase 8.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: [BLOCKER] Built-in projection registrations cannot be resolved into a topology

**File:** `/Users/akriz/code/cacheness/src/cacheness/storage/composition.py:392-402`

**Issue:** `RoleRegistry` registers `json` and `postgresql` as built-in projection participants, but the factories return `JsonProjection` and `PostgresBackend` placeholders that do not implement `apply_projection_batch()`, `save_projection_checkpoint()`, or `load_projection_checkpoint()`. The structural check at lines 666-672 therefore rejects the registry's own built-ins. A direct `StoreTopology` resolution using the advertised `json` name reproducibly raises `CompositionValidationError: A projection participant must satisfy ProjectionSink`. This contradicts the single composition-root contract and the module documentation that JSON is available as an explicitly composed projection. PostgreSQL remains Phase 5 work, but registering an unusable participant in Phase 4 still presents a false construction path.

**Fix:** Either implement the narrow derived `ProjectionSink` contract for JSON now, or remove unqualified JSON/PostgreSQL registrations until the phase that supplies their delivery contract. Keep `resolve_metadata_role()` for explicit role classification if needed. Do not weaken structural validation, add fallback selection, or let either projection become lifecycle authority. Add a test that resolves every built-in registration expected to be constructible, and a separate typed unsupported-registration test for deferred families.

## Warnings

### WR-01: [WARNING] The executable-consumer audit misses common retired public API uses

**File:** `/Users/akriz/code/cacheness/tools/verify_phase4_cutover.py:215-270`

**Issue:** `_RetiredConsumerVisitor` detects retired symbols imported from backend submodules, but it does not flag `from cacheness import register_blob_backend`, `import cacheness as c; c.register_blob_backend(...)`, or `from cacheness.storage.backends import blob_backends as b; b.register_blob_backend(...)`. The last case is caused in part by an unreachable `elif`: the broader `module in {"cacheness.storage.backends", ...}` branch consumes the import before the later `module == "cacheness.storage.backends"` branch can bind `blob_backends`. Direct AST probes return no findings for all three examples, so the release audit can claim a clean executable tree while an uncollected example/tool still uses the retired surface.

**Fix:** Track the root `cacheness` module and direct root imports, handle submodule aliases within the first backend branch, and resolve aliases consistently before checking attributes. Add table-driven verifier tests for direct imports, aliased root imports, submodule imports, bound aliases, and string-only negative assertions.

### WR-02: [WARNING] Accidental root-level `metadata.py` remains as an importable duplicate

**File:** `/Users/akriz/code/cacheness/metadata.py:1`

**Issue:** The untracked repository-root module is a near-copy of `src/cacheness/metadata.py`. Its filesystem birth time is 2026-09-08 00:08 EDT, before the Phase 4 gap cycle began and four minutes before commit `1351b9b` added the intended package module. No test or example writes this path, and its production-module content identifies it as an accidental Plan 04-08 draft rather than test pollution. Because the repository root is normally on `sys.path`, `import metadata` can now resolve this stray file and obscure mistakes; it can also be committed accidentally.

**Fix:** After confirming no human work depends on it, remove or archive the untracked root file and retain only `src/cacheness/metadata.py`. No source compatibility alias is needed.

## Prior Finding Re-Test

| Prior finding | Current verdict | Evidence |
|---|---|---|
| CR-01 public catalog write/update absent | Resolved | Public put/update/query/reopen tests pass; catalog validation precedes lifecycle dispatch. |
| CR-02 selected filesystem payload ignored | Resolved | Selected-root A/B composition regression passes; payloads use participant-provided handler I/O. |
| CR-03 application registrations unreachable | Resolved | High-level custom named payload/authority/projection composition passes; legacy blob registry exports are absent. |
| CR-04 projection exceptions lose receipt | Resolved | Arbitrary `Exception` becomes dirty/committed-partial evidence; `BaseException` passthrough tests pass. |
| CR-05 cursor input unbounded | Resolved | Encoded, decoded, field, shape, signature, and no-authority-dispatch boundary tests pass. |
| CR-06 invalid participant/resource unwind | Resolved | Structural role rejection and identity-deduplicated reverse close tests pass. |
| WR-01 aggregate rebuild capability | Resolved | Mixed-capability named projection rebuild test passes using per-sink capability. |
| WR-02 stale regression tests | Resolved | Integrity, lifecycle-authority, core, Phase 3 acceptance, and SQLite bootstrap modules pass on current seams. |

---

_Reviewed: 2026-09-08T07:48:57Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
