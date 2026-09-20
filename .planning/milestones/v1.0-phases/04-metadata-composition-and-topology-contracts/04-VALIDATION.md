---
phase: 04
slug: metadata-composition-and-topology-contracts
status: validated
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
- tests/test_phase4_cutover_verifier.py
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

## 04-14 Release Evidence — 2026-09-08

The exact required commands were rerun after the final projection and
consumer-audit fixes. The owned matrix now contains 42 modules, including the
table-driven AST source fixtures. No selection, ignore, deselect, xfail, or
test deletion was used.

| Interpreter | Audit and AST fixtures | Owned matrix | Full-tree collection diagnostic | Ruff delta |
| --- | --- | --- | --- | --- |
| CPython 3.11.16 | ✅ repository audit pass; 15 adversarial source fixtures pass | ✅ 607 passed, 6 skipped in 13.55s across 42 modules | Classified deferred diagnostic: exactly `test_sql_cache.py`, `test_sql_cache_documentation.py`, and `test_sql_cache_failure_contract.py` fail collection only because pandas is absent; explicitly not green | ✅ pass |
| CPython 3.13.15 | ✅ repository audit pass; 15 adversarial source fixtures pass | ✅ 607 passed, 6 skipped in 12.46s across 42 modules | Classified deferred diagnostic: exactly `test_sql_cache.py`, `test_sql_cache_documentation.py`, and `test_sql_cache_failure_contract.py` fail collection only because pandas is absent; explicitly not green | ✅ pass |

`JsonProjection` is the only constructible built-in projection in Phase 4. It
implements bounded `ProjectionSink` apply/checkpoint delivery and reports only
`projection_refresh`; its replay-safe JSON state remains derived-only.
`resolve_metadata_role("postgresql")` continues to classify PostgreSQL as a
derived projection, while named `RoleRegistry` resolution now fails typed until
Phase 5 qualifies a real sink. Neither projection family can authorize
canonical membership, reads, deletes, cleanup, repair, or query completeness.

The untracked repository-root `metadata.py` was a confirmed stale 42-line
near-copy of the earlier JSON placeholder and was deliberately removed. It was
not migration tooling and was not moved, imported, or retained as a
compatibility module; `src/cacheness/metadata.py` is the sole package
implementation. Explicit store/schema format detection and the Phase 7 offline
migration/rebuild boundary remain unchanged.

The frozen Ruff inventory now records
`tests/test_phase4_cutover_verifier.py` as a clean new path and records the
deliberately removed root `metadata.py` within the declared Phase 4 scope. No
existing diagnostic fingerprint was refreshed, removed, or forgiven.

## Nyquist Per-Task Verification Map

The audit treats all 29 tasks from Plans 04-01 through 04-14 as delivered
behavior, including the Wave 0 and cutover-staging tasks whose historical
commands intentionally expected red tests. Their final-state evidence is the
green owned matrix, consumer audit, and Ruff delta below.

Command keys:

- `M311`: `uv run --frozen --python 3.11 python tools/verify_phase4_cutover.py --all`
- `M313`: `uv run --frozen --python 3.13 python tools/verify_phase4_cutover.py --all`
- `R311`: `uv run --frozen --python 3.11 python tools/verify_phase4_ruff_delta.py`
- `R313`: `uv run --frozen --python 3.13 python tools/verify_phase4_ruff_delta.py`

| Task ID | Requirement(s) | Final behavioral evidence | Automated command | Status |
| --- | --- | --- | --- | --- |
| 04-01-01 | BACK-02, BACK-03, BACK-06, BACK-07 | Frozen, scope-aware Ruff regression oracle | `R311`, `R313` | green |
| 04-01-02 | BACK-02, BACK-03, BACK-06, BACK-07 | Catalog, query, composition, topology, role, and projection contract suites | `M311`, `M313` | green |
| 04-02-01 | BACK-02 | Native schema, exact scalar validation, stored presence, predicates, cursors, and bounds | `M311`, `M313` | green |
| 04-02-02 | BACK-07 | Format 2, independent version dimensions, non-mutating rejection, and frozen receipt | `M311`, `M313` | green |
| 04-03-01 | BACK-03 | One role-aware registry path, exact injection, and pre-I/O role rejection | `M311`, `M313` | green |
| 04-03-02 | BACK-02, BACK-06, BACK-07 | Ownership/capability enforcement and same-process memory tracer | `M311`, `M313` | green |
| 04-04-01 | BACK-02, BACK-07 | Explicit idempotent SQLite initialization and unchanged unsupported layouts | `M311`, `M313` | green |
| 04-04-02 | BACK-02, BACK-07 | Signed canonical descriptor transaction and persistent reopen | `M311`, `M313` | green |
| 04-04-03 | BACK-06, BACK-07 | Complete bounded revision-bound memory/SQLite catalog queries | `M311`, `M313` | green |
| 04-05-01 | BACK-02, BACK-06, BACK-07 | Derived-only bounded projection pull and checkpoint-after-apply | `M311`, `M313` | green |
| 04-05-02 | BACK-02, BACK-06, BACK-07 | Committed receipt preservation and capability-qualified refresh/rebuild | `M311`, `M313` | green |
| 04-06-01 | BACK-02, BACK-03, BACK-07 | StoreTopology lifecycle, manifest, close, read, and recovery regressions | `M311`, `M313` | green |
| 04-06-02 | BACK-02, BACK-03, BACK-06, BACK-07 | Integrity, containment, concurrency, and portable Windows logic regressions | `M311`, `M313` | green |
| 04-07-01 | BACK-03, BACK-07 | Mixed configuration/UnifiedCache consumers retain policy behavior on current storage seams | `M311`, `M313` | green |
| 04-07-02 | BACK-02, BACK-03, BACK-06, BACK-07 | Public, optional-backend, and projection consumers use current role contracts | `M311`, `M313` | green |
| 04-08-01 | BACK-02, BACK-03, BACK-06, BACK-07 | Atomic public cutover, one BlobStore lifecycle facade, and retired-surface absence | `M311`, `M313` | green |
| 04-08-02 | BACK-02, BACK-03, BACK-06, BACK-07 | Supported-interpreter owned release matrix and honest non-claims | `M311`, `M313`, `R311`, `R313` | green |
| 04-09-01 | BACK-07 | Public catalog put/update/query/reopen with exact presence and stale conflict | `M311`, `M313` | green |
| 04-09-02 | BACK-03, BACK-06 | Selected payload participant supplies actual immutable-generation I/O | `M311`, `M313` | green |
| 04-10-01 | BACK-02, BACK-03, BACK-06 | Application registrations reach BlobStore through topology-owned RoleRegistry | `M311`, `M313` | green |
| 04-10-02 | BACK-02, BACK-03, BACK-06 | Structural role validation and identity-deduplicated owned-resource unwind | `M311`, `M313` | green |
| 04-11-01 | BACK-02, BACK-06, BACK-07 | Ordinary projection failures preserve exact receipts; named rebuild uses per-sink capability | `M311`, `M313` | green |
| 04-11-02 | BACK-02, BACK-06, BACK-07 | Encoded, decoded, field, and signature cursor bounds precede authority dispatch | `M311`, `M313` | green |
| 04-12-01 | BACK-02, BACK-03, BACK-06, BACK-07 | Stable RoleRegistry and mocked S3 factory consumers retain meaningful behavior | `M311`, `M313` | green |
| 04-12-02 | BACK-02, BACK-03, BACK-06, BACK-07 | Stable lifecycle, integrity, initialization, and cache regressions use current seams | `M311`, `M313` | green |
| 04-13-01 | BACK-02, BACK-03, BACK-06, BACK-07 | Query, stored-format, and catalog example consumers use public current-format seams | `M311`, `M313` | green |
| 04-13-02 | BACK-02, BACK-03, BACK-06, BACK-07 | Disjoint owned/deferred sets, executable-tree audit, and two-interpreter matrix | `M311`, `M313`, `R311`, `R313` | green |
| 04-14-01 | BACK-02, BACK-03, BACK-06, BACK-07 | Replay-safe JSON ProjectionSink and typed PostgreSQL construction deferral | `M311`, `M313` | green |
| 04-14-02 | BACK-02, BACK-03, BACK-06, BACK-07 | Alias-aware retired-consumer AST audit with adversarial positive/negative fixtures | `M311`, `M313` | green |
| 04-14-03 | BACK-02, BACK-03, BACK-06, BACK-07 | Root draft absence plus refreshed matrix/audit/Ruff evidence | `M311`, `M313`, `R311`, `R313` | green |

## Requirement Verification Map

| Requirement | Observable behavior | Principal test files | Status |
| --- | --- | --- | --- |
| BACK-02 | Authority and projection roles are explicit; JSON is derived-only and PostgreSQL is honestly deferred | `tests/test_metadata_role_contract.py`, `tests/test_catalog_projection.py`, `tests/test_postgresql_backend.py` | green |
| BACK-03 | Exact injected and named application participants remain selected through one registry/composition root | `tests/test_blob_store_composition.py`, `tests/test_blob_backend_registry.py`, `tests/test_s3_blob_backend.py` | green |
| BACK-06 | Capabilities describe active participants; impossible minima and invalid roles fail before I/O | `tests/test_topology_capabilities.py`, `tests/test_blob_store_composition.py` | green |
| BACK-07 | Public schema/value put, update, query, cursor, projection-failure, and reopen behavior is executable | `tests/test_catalog_schema.py`, `tests/test_catalog_query_contract.py`, `tests/test_catalog_projection.py`, `tests/test_stored_compatibility.py` | green |

## Manual-Only Items

None within the Phase 4 acceptance boundary. Native Windows and live
PostgreSQL/S3 qualification, migration execution, and pandas/SQL-cache
installation qualification are explicitly owned by later phases and are not
Phase 4 validation gaps. The three deferred SQL-cache paths remain a non-green
diagnostic rather than manual evidence.

## Validation Audit 2026-09-08

| Metric | Count |
| --- | ---: |
| Plan tasks audited | 29 |
| Phase requirements audited | 4 |
| Genuine validation gaps found | 0 |
| Gaps resolved with new tests | 0 |
| Escalated | 0 |

Fresh serial execution produced the following current truth:

- CPython 3.11.16: consumer audit passed; owned matrix 607 passed and 6
  skipped in 13.52s across 42 modules; only the exact three-path absent-pandas
  collection diagnostic remained non-green.
- CPython 3.13.15: consumer audit passed; owned matrix 607 passed and 6
  skipped in 12.48s across 42 modules; only the exact three-path absent-pandas
  collection diagnostic remained non-green.
- The Phase 4 Ruff delta passed independently on both interpreters.
- An initial concurrent interpreter attempt was discarded because both `uv`
  invocations shared `.venv`; all evidence above comes from serial runs that
  reported the intended interpreter from inside the matrix runner.

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
