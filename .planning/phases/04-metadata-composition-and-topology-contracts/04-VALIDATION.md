---
phase: 04
slug: metadata-composition-and-topology-contracts
status: evidence-recorded
nyquist_compliant: false
wave_0_complete: true
created: 2026-09-07
---

# Phase 04 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest `>=8.4.1` |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run --frozen pytest -q -o log_cli=false tests/test_catalog_schema.py tests/test_catalog_query_contract.py tests/test_blob_store_composition.py` |
| **Full suite command** | `uv run --frozen pytest -q -o log_cli=false` |
| **Estimated runtime** | Quick suite under 30 seconds; full-suite baseline must be measured during execution |

---

## Sampling Rate

- **After every task commit:** Run the task's targeted test module plus the closest existing authority/read-contract regression module.
- **After every plan wave:** Run `uv run --frozen pytest -q -o log_cli=false` and `uv run ruff check src tests`.
- **Before `$gsd-verify-work`:** Full suite and the Phase 4 contract matrix must be green, with any pre-existing lint baseline distinguished from Phase 4 changes.
- **Max feedback latency:** 30 seconds for per-task targeted checks.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 04-W0-01 | TBD | 0 | BACK-07 | T-04-01 | Exact type validation, bounded schema inputs, and typed rejection of unsupported layouts | unit | `uv run --frozen pytest -q tests/test_catalog_schema.py` | ❌ W0 | ⬜ pending |
| 04-W0-02 | TBD | 0 | BACK-07 | T-04-02, T-04-03 | Bound typed predicates; authenticated revision-bound cursors | contract | `uv run --frozen pytest -q tests/test_catalog_query_contract.py` | ❌ W0 | ⬜ pending |
| 04-W0-03 | TBD | 0 | BACK-02, BACK-03 | T-04-04 | Projections never become authority; one selector path preserves exact injected resources and explicit ownership | contract | `uv run --frozen pytest -q tests/test_metadata_role_contract.py tests/test_blob_store_composition.py` | ❌ W0 | ⬜ pending |
| 04-W0-04 | TBD | 0 | BACK-06 | T-04-05 | Invalid capability claims fail before payload mutation | contract | `uv run --frozen pytest -q tests/test_topology_capabilities.py` | ❌ W0 | ⬜ pending |
| 04-W0-05 | TBD | 0 | BACK-07 | T-04-06 | Projection failure preserves canonical receipt and isolated rebuild state | fault/integration | `uv run --frozen pytest -q tests/test_catalog_projection.py` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_catalog_schema.py` — schema validation, opaque undeclared fields, current-layout reopen, missing/default semantics, explicit evolution boundaries, and non-mutating migration-required rejection of unsupported layouts.
- [ ] `tests/test_catalog_query_contract.py` — portable predicates, bounds, deterministic key/generation ordering, cursor authentication, mismatch, and stale-revision behavior across memory and SQLite.
- [ ] `tests/test_metadata_role_contract.py` — authority/projection role matrix for JSON, memory, SQLite, and PostgreSQL-facing metadata implementations.
- [ ] `tests/test_blob_store_composition.py` — one-root injection, registered-name parity, ambiguity rejection, close ownership, initialization-failure cleanup, and absence of superseded parallel selectors.
- [ ] `tests/test_topology_capabilities.py` — participant/composed capability reporting and construction-time rejection of impossible guarantees.
- [ ] `tests/test_catalog_projection.py` — duplicate/interrupted pull, checkpointing, committed-partial outcomes, and isolated rebuild publication.
- [ ] Shared fixtures for schemas, bounded pages, fault injection, registered roles, and caller-owned resources.

---

## Manual-Only Verifications

All core Phase 4 behaviors have automated verification. Historical compatibility tests may be rewritten or retired when they assert a removed pre-production API; explicit version detection and future migration-tool seams remain tested. A live PostgreSQL service is not used to claim PostgreSQL authority qualification in this phase; that service-backed topology evidence belongs to Phase 5.

---

## Validation Sign-Off

- [ ] All planned tasks have an automated verify command or an explicit Wave 0 dependency.
- [ ] Sampling continuity: no three consecutive tasks lack automated verification.
- [ ] Wave 0 covers all missing test references.
- [ ] Commands contain no watch-mode flags.
- [ ] Targeted feedback latency is below 30 seconds.
- [ ] Python 3.11 and 3.13 public catalog-value/cursor serialization checks are recorded.
- [ ] `nyquist_compliant: true` is set in frontmatter after validation.

**Approval:** release evidence recorded; the unexcluded complete suite is not
green until retired compatibility tests are removed or rewritten and the SQL
cache tests receive their optional dataframe dependency group.

---

## 04-08 Release Evidence — 2026-09-08

### Canonical contract and policy checks

| Command | Result | Evidence |
| --- | --- | --- |
| `uv run --frozen python -c "import cacheness; from cacheness.storage import BlobStore, StoreTopology, BlobReceipt"` | ✅ pass | Package import and the clean direct-storage composition surface are importable. |
| `uv run --frozen pytest -q tests/test_catalog_schema.py tests/test_catalog_query_contract.py tests/test_blob_store_composition.py tests/test_topology_capabilities.py tests/test_metadata_role_contract.py tests/test_catalog_projection.py tests/test_public_api_contract.py tests/test_config_validation.py tests/test_phase3_local_workflows.py -x` | ✅ pass | 185 passed in 1.07s. Covers format-2/schema/cursor/topology/projection/cutover and local lifecycle regressions. |
| `uv run --frozen pytest -q tests/test_decorators.py -x` | ✅ pass | 20 passed, 1 skipped (pandas unavailable); decorator policy continues to use the internal BlobStore engine. |
| `uv run --frozen python tools/verify_phase4_ruff_delta.py` | ✅ pass | No Phase 4 Ruff delta. |

### Supported-interpreter matrix

The commands below were run exactly as written, without deselection or an
exclusion flag. They both stopped during collection before executing a complete
suite; this is a release-evidence result, not a green-suite claim.

| Interpreter | Command | Result | Honest limitation |
| --- | --- | --- | --- |
| CPython 3.13.15 | `uv run --frozen --python 3.13 pytest -q -o log_cli=false` | ❌ collection stopped | Seven modules still import removed `cacheness.metadata` authority types (`CacheEntry`, `CachedMetadataBackend`, `JsonBackend`, `SqliteBackend`, and `Base`). Three SQL-cache modules also require pandas, which is not in the locked default environment. |
| CPython 3.11.16 | `uv run --frozen --python 3.11 pytest -q -o log_cli=false` | ❌ collection stopped | The same seven collection errors and the same absent-pandas SQL-cache collection errors occurred. |
| CPython 3.13.15 | `uv run --frozen --python 3.13 python tools/verify_phase4_ruff_delta.py` | ✅ pass | Phase 4 Ruff delta passed. |
| CPython 3.11.16 | `uv run --frozen --python 3.11 python tools/verify_phase4_ruff_delta.py` | ✅ pass | Phase 4 Ruff delta passed. |

The metadata-import collection errors are expected consequences of the explicit
pre-production removal of the former metadata authority. They must be retired
or rewritten against `BlobStore`, not repaired with compatibility exports. The
pandas failures are an environment/optional-dependency qualification gap, not
evidence that PostgreSQL, S3, or a remote service was tested.

### Documentation and guarantee audit

`docs/CATALOG_AND_TOPOLOGY.md` and `docs/STORAGE_INITIALIZATION.md` were
checked against the exported final surface. The initialization guide was
corrected in this task to remove the dictionary list filter, `BlobEntryInfo`,
legacy-signature, and custom-ORM runtime guidance.

| Contract | Audit result |
| --- | --- |
| Format and version dimensions | Store format remains **2**. Store epoch, manifest schema/version, payload format/version, and SQLite `user_version` remain independent values; no normal open rewrites an old layout. |
| Stored presence/defaults | Declared catalog fields distinguish absent, stored null, and materialized defaults. Additive reads do not rewrite the descriptor; incompatible changes require offline migration. |
| SQLite maintenance boundary | The current local authority is format-2 SQLite. An incomplete, foreign, or obsolete root fails with migration/rebuild-required evidence before mutation. Future schema migration/rebuild work is offline maintenance, not an initialization side effect. |
| JSON and PostgreSQL | Both are derived projection roles only in Phase 4. Neither can authorize lifecycle promotion, cleanup, recovery, or portable-query completeness. |
| Query/index acceleration | Portable catalog pages use authenticated canonical scans and revision-bound cursors. Derived indexes are not an implied acceleration guarantee or alternate authority. |
| Topology-qualified guarantees | Memory is same-process and non-durable. Filesystem plus SQLite is a local-host topology with conflict/timeout outcomes rather than a universal-success claim. No cross-resource ACID claim is made. |

### Explicit non-claims

- **Live PostgreSQL:** not executed; no PostgreSQL lifecycle-authority claim.
- **S3:** not executed; no S3 topology-delivery claim.
- **Windows:** not executed; no native Windows qualification claim.
- **External-service API coverage:** not applicable. Phase 4 added no external
  service API or SDK contract; mocked/portable tests are not used to infer one.

### Follow-up required before a green complete-suite release gate

1. Delete or rewrite the retained compatibility test modules so they test the
   supported BlobStore composition instead of importing retired metadata
   authority symbols.
2. Qualify the SQL-cache test subset with its optional pandas dependency group,
   separately from the BlobStore cutover evidence.
