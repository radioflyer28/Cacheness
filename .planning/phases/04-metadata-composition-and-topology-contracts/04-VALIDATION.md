---
phase: 04
slug: metadata-composition-and-topology-contracts
status: evidence-recorded
nyquist_compliant: true
created: 2026-09-07
updated: 2026-09-08
---

# Phase 04 — Validation Strategy and Release Evidence

Phase 4 qualifies the direct BlobStore/catalog cutover. It does not claim
live PostgreSQL or S3 lifecycle behavior, native Windows behavior, cross-resource
ACID, or universal same-key progress. Those topology claims remain governed by
the accepted storage-guarantees ADR and their later qualification phases.

## Test Infrastructure

| Property | Value |
| --- | --- |
| Framework | pytest `>=8.4.1` |
| Lockfile | `uv.lock` with `uv run --frozen` |
| Consumer audit | `tools/verify_phase4_cutover.py --audit` |
| Owned release matrix | `tools/verify_phase4_cutover.py --matrix` |
| Collection diagnostic | `tools/verify_phase4_cutover.py --diagnostic` |
| Ruff delta | `tools/verify_phase4_ruff_delta.py` |

The release matrix takes its test paths only from the first bounded section
below. The deferred section is intentionally disjoint: it can only classify a
full-tree collection result and can never be passed to the matrix runner.

## Phase 4-Owned Matrix Paths

<!-- phase4-owned-matrix:start -->
- tests/test_blob_backend_registry.py
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
- tests/test_cached_query_meta.py
- tests/test_catalog_projection.py
- tests/test_catalog_query_contract.py
- tests/test_catalog_schema.py
- tests/test_clear_recovery.py
- tests/test_config_validation.py
- tests/test_core.py
- tests/test_custom_metadata.py
- tests/test_filesystem_containment.py
- tests/test_lifecycle_authority_contract.py
- tests/test_manifest_repository_cas.py
- tests/test_metadata.py
- tests/test_metadata_backend_registry.py
- tests/test_metadata_role_contract.py
- tests/test_phase3_gap_acceptance.py
- tests/test_phase3_local_workflows.py
- tests/test_phase3_scheduler_retirement.py
- tests/test_phase3_windows_contract.py
- tests/test_postgresql_backend.py
- tests/test_projection_mutation_contract.py
- tests/test_projection_sql_atomicity.py
- tests/test_public_api_contract.py
- tests/test_s3_blob_backend.py
- tests/test_sqlite_bootstrap_concurrency.py
- tests/test_sqlite_metadata_bootstrap_atomicity.py
- tests/test_stored_compatibility.py
- tests/test_topology_capabilities.py
- tests/test_unified_cache_adversarial_lifecycle.py
- tests/test_unified_cache_lifecycle_authority.py
<!-- phase4-owned-matrix:end -->

## Deferred Diagnostic SQL-Cache Paths

<!-- phase4-deferred-sql-cache:start -->
- tests/test_sql_cache.py
- tests/test_sql_cache_documentation.py
- tests/test_sql_cache_failure_contract.py
<!-- phase4-deferred-sql-cache:end -->

The three SQL-cache paths are pandas-dependent in the locked base environment.
They are neither deselected nor reported as passing Phase 4 evidence. Phase 8
owns installation/import qualification for advertised optional dependency groups.

## Required Checks

```text
uv run --frozen --python 3.11 python tools/verify_phase4_cutover.py --all
uv run --frozen --python 3.13 python tools/verify_phase4_cutover.py --all
uv run --frozen --python 3.11 python tools/verify_phase4_ruff_delta.py
uv run --frozen --python 3.13 python tools/verify_phase4_ruff_delta.py
```

`--all` parses the two marker-bounded lists, AST-audits every executable Python
consumer, runs the owned list verbatim through pytest, and then records a
separate collection diagnostic. A classified deferred diagnostic is an honest
non-green full-tree result, not a full-suite success.

## 04-13 Release Evidence — 2026-09-08

The exact commands above were run with CPython 3.11 and CPython 3.13. The
verifier output records the interpreter version, the exact matrix path count,
pytest pass/skip counts, and whether the full-tree collection result is the
bounded deferred pandas diagnostic. This document is updated only from those
commands; no `-k`, `--ignore`, `--deselect`, synthetic xfail, or test deletion
is part of the owned-matrix command.

| Interpreter | Audit | Owned matrix | Full-tree collection diagnostic | Ruff delta |
| --- | --- | --- | --- | --- |
| CPython 3.11.16 | ✅ pass | ✅ 588 passed, 6 skipped in 13.39s across 41 modules | Classified deferred diagnostic: exactly `test_sql_cache.py`, `test_sql_cache_documentation.py`, and `test_sql_cache_failure_contract.py` fail collection only because pandas is absent; explicitly not green | ✅ pass |
| CPython 3.13.15 | ✅ pass | ✅ 588 passed, 6 skipped in 9.78s across 41 modules | Classified deferred diagnostic: exactly `test_sql_cache.py`, `test_sql_cache_documentation.py`, and `test_sql_cache_failure_contract.py` fail collection only because pandas is absent; explicitly not green | ✅ pass |

The executable-tree audit scanned tracked-or-present Python consumers under
`src`, `tests`, `tools`, `examples`, and `benchmarks`, plus `verify_platform.py`.
It found no executable import, bound alias, or package attribute use of a
retired metadata authority or blob selector surface. String-only absence
assertions and unrelated `Base` classes remain outside that executable-use rule.

### Qualification inventory maintenance

The frozen Ruff inventory was extended only with the ten Python paths declared
by Plans 04-09 through 04-13 that were absent from the original Plan 04-01
through 04-08 inventory. They were individually confirmed Ruff-clean before
being added as clean `new_paths`; no existing diagnostic fingerprint was
refreshed, removed, or forgiven.

## Explicit Non-Claims

- **Live PostgreSQL:** not executed; no PostgreSQL lifecycle-authority claim.
- **S3:** mocked factory coverage only; no S3 topology-delivery claim.
- **Windows:** not executed; no native Windows qualification claim.
- **Full tree with pandas/SQL-cache:** a classified deferred collection diagnostic
  is not a green full-suite result.
- **Atomicity and progress:** no claim exceeds the selected topology's declared
  boundary under [ADR 0001](../../../docs/adr/0001-topology-specific-storage-guarantees.md).

## Validation Sign-Off

- [x] The owned and deferred sets are marker-bounded, normalized, exact, and disjoint.
- [x] The owned set retains catalog, configuration, local-workflow, RoleRegistry,
  mocked-S3, integrity, lifecycle-authority, core, Phase 3 acceptance, SQLite
  bootstrap, query metadata, and stored-format coverage.
- [x] The executable consumer audit is independent of normal pytest collection.
- [x] The deferred SQL-cache paths are unavailable to `PHASE4_MATRIX`.
- [ ] The optional pandas/SQL-cache dependency group is qualified (Phase 8).
