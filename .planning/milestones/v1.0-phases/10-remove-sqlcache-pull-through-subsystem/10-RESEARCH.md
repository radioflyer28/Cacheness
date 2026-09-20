# Phase 10: Remove SqlCache Pull-Through Subsystem - Research

**Researched:** 2026-09-17
**Domain:** Pre-production Python API/subsystem removal, dependency pruning, and wheel-contract verification
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

### Removed import and data behavior
- **D-01:** Delete `cacheness.sql_cache`, its top-level exports, aliases, and every compatibility hook. Old imports fail naturally with `ModuleNotFoundError` or `ImportError`; do not retain a tombstone module, `__getattr__` shim, or tailored runtime exception. — **Reversibility:** costly — undoing this after the first supported release would reintroduce a separate public product and maintenance contract.
- **D-02:** Existing canonical guidance explains alternatives by use case: `UnifiedCache` for object/function caching and `BlobStore` for object persistence. It must explicitly state that Cacheness has no in-package replacement for range-aware SQL pull-through.
- **D-03:** Caller-owned development databases and tables are left untouched and unsupported. Phase 10 must not discover, delete, mutate, export, or migrate them, and must not retain implementation solely to provide cleanup tooling.
- **D-04:** Phase 10 does not change the package version. Version selection and publication remain release-process work.

### Dependencies and extras
- **D-05:** Delete the `sql` optional extra rather than repurposing it or retaining an empty compatibility extra.
- **D-06:** Remove every DuckDB product dependency and claim: manifest and lockfile entries, runtime branches, tests, examples, docs, packaging matrices, and CI references. DuckDB is not retained as a development-only dependency.
- **D-07:** Do not redesign the remaining SQLAlchemy/PostgreSQL dependency organization. Remove only entries proven exclusive to `SqlCache`; preserve dependencies required by lifecycle authorities, catalog metadata, format handlers, and supported backends.
- **D-08:** Prune every additional manifest/lock dependency whose last supported use disappears with the deleted subsystem. Retain anything still used by `BlobStore`, `UnifiedCache`, handlers, authorities, or development tooling.

### Documentation and history
- **D-09:** Delete dedicated SqlCache documents, examples, and tests outright. Do not archive them inside the repository or leave unsupported files with warning banners.
- **D-10:** Preserve truthful historical planning artifacts, completed phase records, and dated audits. Update a historical document only if it falsely presents `SqlCache` as part of the current post-Phase-10 product; do not rewrite history merely to remove the name.
- **D-11:** Surgically remove SqlCache-specific material from mixed current-facing documents while retaining useful content about pandas/dataframes, handlers, or supported storage/cache behavior.
- **D-12:** Record the cutover as a concise note in existing canonical API/migration guidance and the docs index. Do not create a standalone long-lived SqlCache removal guide.

### Regression and package contracts
- **D-13:** Tests must prove complete absence: no top-level export, package attribute, source module, importable `cacheness.sql_cache`, compatibility hook/tombstone, or wheel member.
- **D-14:** Scan current-facing source, tests, examples, package metadata, README, and documentation for dangling references. Use an explicit narrow allowlist for the concise removal note and truthful historical planning/dated-audit references; do not impose repository-wide historical erasure.
- **D-15:** Rewrite useful older public-surface, suite-isolation, lint, and packaging contracts to enforce the new boundary. Do not delete or exclude a mixed test merely because it previously asserted SqlCache behavior.
- **D-16:** Final acceptance includes a freshly built and isolated wheel: supported `BlobStore`/`UnifiedCache` imports and representative local round trips pass, SqlCache and DuckDB are absent, and installed dependency metadata matches the pruned manifest.

### the agent's Discretion
- Exact task slicing and ordering of source deletion, contract inversion, documentation cleanup, and dependency lock refresh.
- Exact wording and location within the existing canonical API/migration pages, provided the note remains concise and follows D-02/D-12.
- The explicit reference allowlist structure and test helpers, provided historical artifacts remain preserved and all current-facing surfaces are covered.

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CACH-07 | The unrelated table/range-oriented `SqlCache` pull-through subsystem is removed before the first supported release, including its public exports, dedicated implementation, tests, examples, documentation, and dependencies that become unused; BlobStore, UnifiedCache, SQL-backed lifecycle authorities, catalog metadata, and dataframe format handlers remain intact. | Exact deletion/surgical-edit inventories, orphan-dependency proof, historical-reference policy, contract inversion order, and isolated-wheel validation below. [VERIFIED: `.planning/REQUIREMENTS.md:31-39`] |
</phase_requirements>

## Summary

Phase 10 is a bounded product deletion, not a refactor of the storage engine. The implementation is concentrated in one 2,074-line module, but the real planning risk is the network of tests and verification manifests that still treat its three dedicated test modules as live contract nodes. Deleting only `sql_cache.py`, its tests, docs, and examples will make Phase 4, Phase 6, Phase 07.1, public-surface, suite-isolation, documentation, and wheel-contract tests fail before they can prove the new boundary. [VERIFIED: `src/cacheness/sql_cache.py:90-246`; `src/cacheness/sql_cache.py:2073-2074`; `tools/verify_phase4_cutover.py:28-37`; `tools/verify_phase6_contracts.py:27-68`; `tools/verify_phase071_contracts.py:375-393`]

The dependency cut is equally precise. The manifest's current installable optional extras are quoted verbatim as `recommended`, `dataframes`, `tensorflow`, `s3`, `postgresql`, and `cloud`; there is no installable `sql` extra. The current `sql` surface is instead a uv dependency group whose exact members are `"duckdb-engine>=0.16.0"` and `"sqlalchemy>=2.0.0"`. Remove that group and remove `duckdb-engine` from both `recommended` lists. Regenerating the lock should then remove the `duckdb-engine` and `duckdb` packages; SQLAlchemy, pandas, PyArrow, and psycopg remain owned by supported PostgreSQL/projection or dataframe-handler surfaces. [VERIFIED: `pyproject.toml:18-76`; `uv.lock:178-260`; `uv.lock:632-673`; `src/cacheness/handlers.py:30-45`; `src/cacheness/storage/backends/postgresql_backend.py:10-17`; `src/cacheness/storage/backends/postgresql_lifecycle_authority.py:73-80`]

**Primary recommendation:** Invert the reusable contracts and fixed manifests first, perform one coordinated runtime/API/dependency deletion second, clean only current-facing guidance third, then close with the existing Phase 8 fresh-wheel harness extended to inspect wheel membership and installed metadata. Do not add a replacement, tombstone, cleanup path, lifecycle change, or new coordination mechanism. [VERIFIED: `.planning/phases/10-remove-sqlcache-pull-through-subsystem/10-CONTEXT.md`; `docs/adr/0001-topology-specific-storage-guarantees.md:139-180`]

## Project Constraints (from AGENTS.md)

- This is a pre-production cutover: development APIs/layouts may be replaced rather than supported through compatibility adapters. [VERIFIED: `AGENTS.md:13-17`]
- Preserve explicit schema/format versions and offline migration/rebuild tooling, but do not make SqlCache tables part of that tooling. [VERIFIED: `AGENTS.md:15-17`; `.planning/phases/10-remove-sqlcache-pull-through-subsystem/10-CONTEXT.md` D-03]
- `BlobStore` remains the storage-lifecycle owner and `UnifiedCache` remains its cache-policy layer; Phase 10 removes the unrelated product. [VERIFIED: `AGENTS.md:17-18`; `docs/adr/0001-topology-specific-storage-guarantees.md:36-44`]
- Preserve Python `>=3.11`; the manifest's exact requirement is `requires-python = ">=3.11"`, while the checked uv environment used for this research is Python 3.13.15. [VERIFIED: `pyproject.toml:1-16`; observed `uv run --frozen python` output]
- Keep package re-exports explicit in `__init__.py`, name tests `test_<subject>.py`, use four-space PEP 8 style and the configured 88-character Ruff target, and run scoped Ruff without claiming the existing repository baseline is clean. [VERIFIED: `AGENTS.md:120-143`; `src/cacheness/__init__.py:12-49`]
- Storage safety remains fail-closed and topology-specific. This phase must preserve the one-authority, immutable-generation, reconciliation, and policy-above-storage contracts rather than alter or re-test them as part of the deletion. [VERIFIED: `AGENTS.md:19-23`; `docs/adr/0001-topology-specific-storage-guarantees.md:83-100`; `docs/adr/0001-topology-specific-storage-guarantees.md:139-164`]

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Remove range-aware SQL pull-through | Public API / package | Test and docs contracts | Delete the separate product and its exposure; do not route this behavior into `UnifiedCache`. [VERIFIED: `.planning/ROADMAP.md:668-679`] |
| Preserve object persistence | Storage | Format handlers | `BlobStore` remains the lifecycle owner and handlers remain its serialization strategies. [VERIFIED: `docs/adr/0001-topology-specific-storage-guarantees.md:36-44`; `src/cacheness/handlers.py:327-557`] |
| Preserve cache policy | Cache facade | Storage | `UnifiedCache` continues to depend on `BlobStore`; it does not become a query/range cache. [VERIFIED: `docs/adr/0001-topology-specific-storage-guarantees.md:42-44`; `docs/API_REFERENCE.md:97-100`] |
| Prune dependency metadata | Packaging | Lockfile | `pyproject.toml` is authoritative; `uv.lock` must converge without unrelated upgrades. [VERIFIED: `pyproject.toml:18-76`; `uv.lock:166-288`] |
| Prove shipped absence | Wheel qualification | Public API tests | Source deletion alone cannot prove a built artifact excludes the module or dependency metadata. The existing harness already builds exactly one wheel and probes it outside the checkout. [VERIFIED: `tools/run_phase8_packaging.py:217-238`; `tools/run_phase8_packaging.py:253-375`] |

## Standard Stack

### Core

| Tool / library | Version | Purpose | Why this is the standard here |
|----------------|---------|---------|-------------------------------|
| Python | `>=3.11`; checked uv runtime 3.13.15 | Runtime and negative-import tests | This is the project's declared runtime boundary. [VERIFIED: `pyproject.toml:1-16`; observed environment] |
| uv | 0.12.12 | Lock refresh, frozen test execution, wheel build, isolated install | The repository's build and qualification tools invoke uv directly. [VERIFIED: `tools/run_phase8_packaging.py:217-238`; `tools/run_phase8_packaging.py:333-375`; observed environment] |
| pytest | locked 8.4.1 | Public API, source scanner, docs, verifier, and wheel contracts | Existing tests and dependency group already own it. [VERIFIED: `pyproject.toml:68-73`; observed environment] |
| `uv_build` | `>=0.9.0,<1.0.0` | Build the wheel to inspect and install | It is the configured PEP 517 build backend. [VERIFIED: `pyproject.toml:78-80`] |
| Python stdlib `zipfile`, `importlib`, `importlib.metadata`, `tomllib` | runtime | Inspect wheel membership, natural import failure, installed requirements/extras, and TOML manifests | These checks need no new dependency and compose with the existing source-free probe. [VERIFIED: `tools/run_phase8_packaging.py:10-18`; recommendation based on D-13/D-16] |

### Supporting contracts

| Existing asset | Purpose in Phase 10 | Required change |
|----------------|---------------------|-----------------|
| `tools/run_phase8_packaging.py` | Build one wheel, reject checkout imports, test BlobStore and UnifiedCache round trips | Remove SqlCache from `BASE_PUBLIC_EXPORTS`; add its names to the retired top-level set; inspect the wheel and installed metadata for module/DuckDB absence. [VERIFIED: `tools/run_phase8_packaging.py:47-68`; `tools/run_phase8_packaging.py:173-182`; `tools/run_phase8_packaging.py:253-330`] |
| `tests/packaging/test_wheel_matrix.py` | Integration proof for the real wheel artifact | Extend the existing real build/probe test instead of creating a second packaging harness. [VERIFIED: `tests/packaging/test_wheel_matrix.py:43-82`; `tests/packaging/test_wheel_matrix.py:195-216`] |
| Phase 9 docs contracts | Explicit documentation inventory and ownership | Change blanket no-mention assertions only at the three canonical removal-note owners. [VERIFIED: `tests/test_phase9_documentation.py:71-89`; `tests/test_phase9_documentation.py:91-188`] |

### Package legitimacy audit

Not applicable. Phase 10 installs no new package; it removes `duckdb-engine` and the transitively locked `duckdb`. [VERIFIED: `pyproject.toml:18-76`; `uv.lock:632-673`]

## Exact Change Inventory

### Delete outright

| Category | Paths | Evidence and disposition |
|----------|-------|--------------------------|
| Runtime | `src/cacheness/sql_cache.py` | The file owns `SQLCacheError`, result/failure types, three dedicated errors, `MissingDependencyError`, `SqlCacheAdapter`, `SqlCache`, all builders, and the exact alias `SQLAlchemyDataAdapter = SqlCacheAdapter`; delete the file as one unit. [VERIFIED: `src/cacheness/sql_cache.py:90-246`; `src/cacheness/sql_cache.py:2073-2074`] |
| Dedicated tests | `tests/test_sql_cache.py`, `tests/test_sql_cache_documentation.py`, `tests/test_sql_cache_failure_contract.py` | All three collect only for this removed subsystem and are exactly the old deferred diagnostic set. Delete, then remove every fixed-manifest reference. [VERIFIED: `tools/verify_phase4_cutover.py:28-37`; repository file inspection] |
| Dedicated docs | `docs/SQL_CACHE.md`, `docs/CUSTOM_GAP_DETECTION.md`, `docs/ARBITRARY_TIME_INCREMENTS.md` | These pages exclusively document builders, range/gap behavior, and DuckDB/SQL backends; do not retain banners or archives. [VERIFIED: `docs/SQL_CACHE.md:1-50`; `docs/CUSTOM_GAP_DETECTION.md:1-84`; `docs/ARBITRARY_TIME_INCREMENTS.md:1-21`] |
| Dedicated/mixed obsolete examples | `examples/beginner_sql_cache.py`, `examples/simple_stock_cache.py`, `examples/stock_cache_example.py`, `examples/database_backend_comparison.py`, `examples/intelligent_storage_demo.py`, `examples/simple_backend_demo.py` | Every file imports or constructs SqlCache; the mixed files' valid general themes are already covered by Phase 9's canonical example surface, so retaining partial rewrites would create a fifth unqualified journey. [VERIFIED: repository reference scan; `tests/test_phase9_quality_workflow.py:67-76`] |

### Surgical edits required

| Surface | Required edit | Why it cannot be deleted wholesale |
|---------|---------------|------------------------------------|
| `src/cacheness/__init__.py` | Rewrite the module docstring; remove the exact import `from .sql_cache import SqlCache, SqlCacheAdapter` and exact `__all__` values `"SqlCache"`, `"SqlCacheAdapter"`; keep version `"0.3.14"`. [VERIFIED: `src/cacheness/__init__.py:1-49`] | It is the supported BlobStore/UnifiedCache barrel. |
| `src/cacheness/error_handling.py` | Remove the four now-orphaned exact `CacheReason` values `"sql_cache_fetch_failed"`, `"sql_cache_gap_detection_failed"`, `"sql_cache_upsert_failed"`, and `"missing_optional_dependency"`; update the literal reason-set test. [VERIFIED: `src/cacheness/error_handling.py:19-45`; repository symbol scan] | The rest of the error hierarchy remains shared. |
| `pyproject.toml` / `uv.lock` | Remove `duckdb-engine` from `[project.optional-dependencies].recommended` and `[dependency-groups].recommended`; delete `[dependency-groups].sql`; run plain `uv lock` and verify lock convergence. [VERIFIED: `pyproject.toml:18-62`; `uv.lock:192-200`; `uv.lock:218-231`; `uv.lock:236-288`] | SQLAlchemy, pandas, PyArrow, and psycopg still support retained surfaces. |
| `docs/PANDAS_API_AUDIT.md` | Remove only the SqlCache/test_sql_cache ownership rows; retain pandas handler/Parquet material. [VERIFIED: `docs/PANDAS_API_AUDIT.md:13`; `docs/PANDAS_API_AUDIT.md:163-167`] | This is a dataframe-handler audit, not a dedicated SqlCache page. |
| `docs/API_REFERENCE.md`, `docs/STORAGE_MIGRATION.md`, `docs/README.md` | Add one concise bounded cutover note: supported alternatives by use case, no range-aware SQL pull-through replacement, and caller tables are untouched/unsupported. [VERIFIED: `docs/API_REFERENCE.md:1-42`; `docs/STORAGE_MIGRATION.md:1-36`; `docs/README.md:1-41`; CONTEXT D-02/D-03/D-12] | They are canonical current guidance and index owners. |
| `docs/CROSS_PLATFORM_GUIDE.md` | Remove the stale current diagnostic claim that names SQL-cache; keep the general isolated-suite explanation. [VERIFIED: `docs/CROSS_PLATFORM_GUIDE.md:135-148`] | The guide remains relevant to supported platforms. |
| `AGENTS.md` and `.planning/codebase/{ARCHITECTURE,CONCERNS,CONVENTIONS,INTEGRATIONS,STACK,STRUCTURE,TESTING}.md` | Regenerate/surgically update current codebase guidance so it describes the post-removal tree and retained SQL/PostgreSQL/dataframe uses. [VERIFIED: repository current-map reference scan] | These are current-facing agent maps, unlike completed phase records and dated audits. |
| `tests/test_public_api_contract.py` | Invert the optional-surface test into natural absence assertions; remove the four orphan reason values from the exact set; replace the API-reference blanket ban with a bounded-note assertion. [VERIFIED: `tests/test_public_api_contract.py:132-233`; `tests/test_public_api_contract.py:281-322`] | It protects multiple supported public/error/doc contracts. |
| `tests/test_phase6_public_api_contract.py` | Remove SqlCache imports/names from the canonical tuple; invert the separate-supported-surface assertion into absence while retaining BlobStore/UnifiedCache ownership assertions. [VERIFIED: `tests/test_phase6_public_api_contract.py:20-63`; `tests/test_phase6_public_api_contract.py:233-241`] | It remains the canonical cache-policy public contract. |
| `tests/test_phase6_suite_isolation.py` | Remove deleted nodes; retain a two-module both-orders isolation check for `test_public_api_contract.py` and `test_phase6_public_api_contract.py`. [VERIFIED: `tests/test_phase6_suite_isolation.py:150-193`] | D-15 explicitly preserves useful isolation coverage. |
| `tests/test_phase1_quality_gates.py` | Remove deleted file constants/manifests and replace the SqlCache-class no-print assertion with a general AST-helper sentinel if needed. [VERIFIED: `tests/test_phase1_quality_gates.py:27-50`; `tests/test_phase1_quality_gates.py:235-243`] | Other quality gates remain useful. |
| `tests/test_full_suite_environment.py` | Rewrite the exact bare-collection diagnostic that currently says pandas is missing for the three SQL-cache modules. [VERIFIED: `tests/test_full_suite_environment.py:15-36`] | The frozen full-suite command and other environment contracts remain current. |
| `tools/verify_phase4_cutover.py` + its tests | Stop parsing or validating the historical deferred SqlCache list as live paths; preserve the historical Phase 4 validation document unchanged; a full collection failure is no longer excused through those paths. [VERIFIED: `tools/verify_phase4_cutover.py:20-37`; `tools/verify_phase4_cutover.py:113-189`; `tools/verify_phase4_cutover.py:391-420`] | The Phase 4 owned matrix still protects retained lifecycle behavior. |
| `tools/verify_phase6_contracts.py`, `tests/test_phase6_contract_verifier.py` | Replace old `tests/test_sql_cache.py` membership and `"CACH-07 SqlCache regression"` with the Phase 10 negative contract, or remove CACH-07 from Phase 6 ownership and have the Phase 10 verifier own it. Prefer the latter to preserve truthful phase ownership. [VERIFIED: `tools/verify_phase6_contracts.py:27-90`; `tests/test_phase6_contract_verifier.py:42-63`] | The verifier still owns CACH-01 through CACH-06 and retained regressions. |
| `tools/verify_phase071_contracts.py` | Replace the old positive SqlCache selector with the new negative Phase 10 public-removal selector. [VERIFIED: `tools/verify_phase071_contracts.py:375-393`] | The all-node verifier otherwise points at a deleted test function. |
| `tools/run_phase8_packaging.py`, `tests/packaging/test_wheel_matrix.py` | Extend the established artifact, retired-export, ZIP-member, installed-metadata, and local-round-trip proof. [VERIFIED: `tools/run_phase8_packaging.py:47-68`; `tools/run_phase8_packaging.py:173-182`; `tests/packaging/test_wheel_matrix.py:43-82`] | This is already the source-free wheel authority. |
| `tests/test_phase9_documentation.py` | Permit and require removal wording only in docs index/API/migration owners; continue banning it in README and task guides that are not owners. [VERIFIED: `tests/test_phase9_documentation.py:28-89`; `tests/test_phase9_documentation.py:91-188`; `tests/test_phase9_documentation.py:300-322`] | It protects the supported documentation architecture. |

### Preserve unchanged or preserve semantically

- Preserve `docs/phase3-architecture-audit-2026-09-06.md`: its sentence that SqlCache “remains separate” is truthful dated architecture history, not current API guidance. [VERIFIED: `docs/phase3-architecture-audit-2026-09-06.md:101`; CONTEXT D-10]
- Preserve completed `.planning/phases/**`, dates, decisions, and evidence as historical records. Current ROADMAP/STATE/REQUIREMENTS may receive normal GSD completion bookkeeping, but not a name-erasure rewrite. [VERIFIED: CONTEXT D-10/D-14; `.planning/ROADMAP.md:668-701`]
- Preserve exact installable optional extras `"recommended"`, `"dataframes"`, `"tensorflow"`, `"s3"`, `"postgresql"`, and `"cloud"`; removing the uv dependency group named `sql` must not accidentally change this tuple. [VERIFIED: `pyproject.toml:18-45`; `uv.lock:260`]
- Preserve pandas/Polars/PyArrow handlers and PostgreSQL authority/projection paths. Pandas is still imported by handlers, psycopg by the lifecycle authority, and SQLAlchemy availability by the PostgreSQL projection participant. [VERIFIED: `src/cacheness/handlers.py:30-45`; `src/cacheness/handlers.py:327-557`; `src/cacheness/storage/backends/postgresql_lifecycle_authority.py:73-80`; `src/cacheness/storage/backends/postgresql_backend.py:10-17`]
- Preserve package version exact value `"0.3.14"` in both project metadata and `cacheness.__version__`. [VERIFIED: `pyproject.toml:1-4`; `src/cacheness/__init__.py:27-29`; CONTEXT D-04]

## Dependency and Lockfile Proof

The manifest currently contains `duckdb-engine` in exactly three declarations: installable `recommended`, uv group `recommended`, and uv group `sql`. The group `sql` is not present in the exact `provides-extras = ["recommended", "dataframes", "tensorflow", "s3", "postgresql", "cloud"]` list, so the implementation must not invent or preserve an empty installable extra while satisfying D-05. [VERIFIED: `pyproject.toml:18-62`; `uv.lock:236-260`]

The frozen inverse tree is unambiguous: `duckdb-engine v0.17.0` has only `cacheness v0.3.14 (extra: recommended)` as a project parent, and `duckdb v1.3.2` has only `duckdb-engine v0.17.0` as a parent. SQLAlchemy `v2.0.43` still has direct project parents from exact extras `cloud`, `postgresql`, and `recommended`. Therefore the lockfile packages proven orphaned are `duckdb-engine` and `duckdb`, not `sqlalchemy` or `packaging`. [VERIFIED: observed `uv tree --frozen --invert` outputs; `uv.lock:632-673`]

Use `uv lock` after the manifest edit, without an upgrade flag, then run `uv lock --check`. Review the diff for: removal of the three cacheness-to-duckdb-engine edges, removal of the uv `sql` group, removal of both package records, and no unrelated package version churn. [VERIFIED: `uv.lock:192-242`; `uv.lock:274-288`; `uv.lock:632-673`; recommendation]

## Architecture Patterns

### System architecture after removal

```text
Application object/function request
            |
            +--> UnifiedCache (keying, TTL, admission, eviction)
            |          |
            |          v
            +------> BlobStore (single lifecycle owner)
                         |
             +-----------+------------+
             v                        v
      lifecycle authority       immutable payload participant
      memory/SQLite/Postgres     memory/filesystem/S3 via obstore
             |
             v
      authoritative catalog

Dataframe/array/object value --> store-local FormatHandler --> native payload

SQL range/query pull-through: no Cacheness component after Phase 10
```

This preserves ADR 0001's exact boundary: `BlobStore` owns lifecycle and `UnifiedCache` adds policy above it. Phase 10 adds no state machine, lock, queue, migration path, projection, or second authority. [VERIFIED: `docs/adr/0001-topology-specific-storage-guarantees.md:36-44`; `docs/adr/0001-topology-specific-storage-guarantees.md:139-180`]

### Pattern 1: Contract inversion before physical deletion

**What:** First change mixed/fixed contract inventories from “SqlCache is supported” to “the range-query product is absent,” while keeping all retained assertions. Then delete files and metadata in one coordinated cut. [VERIFIED: CONTEXT D-13/D-15; current positive contracts listed in Exact Change Inventory]

**Why:** Several verifier modules normalize their literal test paths at import time and require each file to exist. If dedicated tests are deleted before those inventories are changed, pytest collection fails inside the verifier rather than reporting the intended product boundary. [VERIFIED: `tools/verify_phase4_cutover.py:113-189`; `tools/verify_phase6_contracts.py:27-68`]

### Pattern 2: One explicit current-surface scanner

**What:** Add a small test helper with named scan roots and a literal allowlist. Source/package/examples/manifest should have zero positive references; canonical docs may contain only the concise cutover note; tests/tools may contain only negative assertions/fixtures; dated audits and completed planning are outside the current-surface scan. [VERIFIED: CONTEXT D-10/D-12/D-14; recommendation]

**Why:** A repository-wide `rg` cannot distinguish a dangling product claim from truthful history, while a broad allowlist can hide new regressions. Keep the allowlist path-and-purpose specific. [VERIFIED: current dated-audit and current-map inventories; CONTEXT D-14]

### Pattern 3: Artifact-first absence proof

**What:** Build exactly one wheel, inspect its ZIP members, install that exact artifact in an isolated/no-project environment, then assert imports, package attributes, installed requirements/extras, and supported local round trips. [VERIFIED: `tools/run_phase8_packaging.py:217-238`; `tools/run_phase8_packaging.py:253-375`; CONTEXT D-16]

**Why:** A checkout can be correct while a stale build configuration still ships a deleted module; an import can fail while stale dependency metadata remains. D-13/D-16 require both structural and behavioral proof. [VERIFIED: CONTEXT D-13/D-16]

### Recommended implementation order

1. **Wave 0 — invert reusable contracts:** add the Phase 10 negative contract and repair Phase 4/6/07.1 fixed manifests, Phase 1 lint sentinel, suite isolation, full-suite-environment description, docs tests, and Phase 8 packaging inventories. [VERIFIED: the fixed references in Exact Change Inventory]
2. **Wave 1 — coordinated product cut:** delete `sql_cache.py`, its three dedicated tests, three dedicated docs, and six examples; remove exports, orphan reasons, DuckDB declarations, and the uv `sql` group; refresh the lockfile. [VERIFIED: deletion and dependency inventories above]
3. **Wave 2 — current guidance:** add the three concise canonical notes; clean pandas/cross-platform current docs; update AGENTS/current codebase maps; preserve dated audits and completed phase artifacts. [VERIFIED: CONTEXT D-09 through D-14]
4. **Wave 3 — closure:** run focused tests, scoped Ruff, lock check, explicit current-surface scan, frozen non-live full suite, and the fresh isolated wheel proof. [VERIFIED: `.planning/config.json:20-49`; existing qualification commands in `tests/test_full_suite_environment.py:15-24`]

## Don't Hand-Roll

| Problem | Don't build | Use instead | Why |
|---------|-------------|-------------|-----|
| Compatibility after removal | Tombstone module, package `__getattr__`, alias, custom exception | Natural Python `ImportError` / `ModuleNotFoundError` | Compatibility code would preserve the product surface D-01 removes. [VERIFIED: CONTEXT D-01/D-13] |
| Caller data cleanup | Database discovery, migration, export, drop-table command | Concise migration note saying tables are untouched/unsupported | Caller databases are explicitly outside scope and may be destructive to inspect or mutate. [VERIFIED: CONTEXT D-03/D-12] |
| SQL pull-through replacement | New UnifiedCache adapter or BlobStore query layer | Truthful “no in-package replacement” guidance | Neither retained abstraction implements range-gap query caching. [VERIFIED: CONTEXT D-02; `.planning/ROADMAP.md:668-679`] |
| Wheel verification | A second build script or source-tree-only import check | Extend `tools/run_phase8_packaging.py` and its tests | The existing harness binds one artifact to isolated round trips. [VERIFIED: `tools/run_phase8_packaging.py:217-238`; `tests/packaging/test_wheel_matrix.py:43-62`] |
| Dependency removal | Manual lockfile surgery | Edit `pyproject.toml`, then plain `uv lock`, `uv lock --check`, and inspect the diff/tree | The lock is derived state and contains multiple dependency tables/edges. [VERIFIED: `uv.lock:166-288`; `uv.lock:632-673`] |
| Reference erasure | Repository-wide replace/delete | Explicit current-surface inventory plus narrow allowlist | Historical evidence must remain truthful. [VERIFIED: CONTEXT D-10/D-14] |
| Storage lifecycle work | New locks, recovery states, or authority coordination | Leave BlobStore/UnifiedCache/authority code unchanged and run existing regressions | ADR 0001's stop conditions explicitly reject coordination accretion. [VERIFIED: `docs/adr/0001-topology-specific-storage-guarantees.md:172-180`] |

## Runtime State Inventory

| Category | Items found | Action required |
|----------|-------------|-----------------|
| Stored data | Two tracked `.sqlite3` files are compatibility fixtures for BlobStore metadata layouts, not SqlCache assets. Untracked local `cache_metadata.db`/authority databases also exist in ignored cache paths. Caller-owned SqlCache tables may exist outside the repository by definition. [VERIFIED: `git ls-files` and bounded repository `find`; CONTEXT D-03] | Preserve all databases. No discovery, data migration, export, mutation, or deletion task. Document only that caller tables are untouched/unsupported. |
| Live service config | No GitHub workflow or service configuration directly names SqlCache or DuckDB; the current CI uses the Phase 8/9 harnesses. [VERIFIED: bounded `.github` reference scan; `.github/workflows/quality.yml:40-52`] | No external service mutation. Update only tests/tools invoked by existing CI. |
| OS-registered state | None found or implied: this is an in-process Python library subsystem with no CLI entry, daemon, launchd/systemd registration, or project script. [VERIFIED: `pyproject.toml:1-90`; repository file inventory] | None. |
| Secrets / environment variables | No SqlCache/DuckDB-specific environment-variable reader or secret name was found in source/current workflow files. [VERIFIED: bounded source/workflow reference scan] | None; preserve unrelated PostgreSQL/S3 qualification secrets and settings. |
| Build artifacts / installed packages | Ignored bytecode files `src/cacheness/__pycache__/sql_cache.cpython-311.pyc` and `sql_cache.cpython-313.pyc` exist locally; an older external installation may also retain the prior wheel. [VERIFIED: bounded repository `find`] | Do not add cleanup code or mutate user environments. Fresh isolated wheel inspection is the release proof; developers reinstall the new artifact normally. |

## Common Pitfalls

### Pitfall 1: Deleting tests before fixed manifests
**What goes wrong:** Phase 4/6 verifier imports fail on missing literal paths. **Why:** validation paths are normalized and existence-checked at module import. **Avoid:** invert/refactor manifests before or atomically with deletion. **Warning sign:** collection errors in verifier tests rather than a Phase 10 assertion. [VERIFIED: `tools/verify_phase4_cutover.py:113-189`; `tools/verify_phase6_contracts.py:27-68`]

### Pitfall 2: Treating `sql` as an installable extra
**What goes wrong:** a planner may edit the wrong TOML table or add an empty compatibility extra. **Why:** context uses “optional extra,” but the actual exact `provides-extras` list has no `sql`; it is a uv dependency group. **Avoid:** delete `[dependency-groups].sql` and do not change the six installable extras. [VERIFIED: `pyproject.toml:18-76`; `uv.lock:260`]

### Pitfall 3: Over-pruning SQL/dataframe dependencies
**What goes wrong:** PostgreSQL or Parquet handlers stop installing. **Why:** SqlCache used SQLAlchemy/pandas, but it was not their only supported owner. **Avoid:** remove only `duckdb-engine` and transitive `duckdb`; preserve SQLAlchemy, pandas, PyArrow, and psycopg declarations. [VERIFIED: `src/cacheness/handlers.py:30-45`; `src/cacheness/handlers.py:327-557`; `src/cacheness/storage/backends/postgresql_backend.py:10-42`; `src/cacheness/storage/backends/postgresql_lifecycle_authority.py:73-80`]

### Pitfall 4: Blanket documentation erasure
**What goes wrong:** truthful dated evidence is rewritten, or canonical removal notes are rejected by old “no mention” tests. **Avoid:** distinguish dedicated docs (delete), mixed current docs (surgical edit), canonical removal-note owners (narrow allow), and dated/planning history (preserve). [VERIFIED: CONTEXT D-09 through D-14; `tests/test_phase9_documentation.py:71-188`]

### Pitfall 5: Claiming UnifiedCache replaces SqlCache
**What goes wrong:** users infer support for gap detection, query-range completeness, or table upserts that no retained component implements. **Avoid:** say only that UnifiedCache covers object/function caching and BlobStore object persistence, followed by the explicit non-replacement sentence. [VERIFIED: CONTEXT D-02; `.planning/ROADMAP.md:668-679`]

### Pitfall 6: Proving checkout absence but shipping stale metadata
**What goes wrong:** a wheel can omit an export yet retain `cacheness/sql_cache.py` or a `Requires-Dist: duckdb-engine` edge. **Avoid:** inspect ZIP members and installed distribution metadata from the exact freshly built artifact. [VERIFIED: CONTEXT D-13/D-16; `tools/run_phase8_packaging.py:217-238`]

### Pitfall 7: Accidental lifecycle scope expansion
**What goes wrong:** removal work is used to redesign PostgreSQL, catalog, handler, or concurrency behavior. **Avoid:** no changes to BlobStore, UnifiedCache, lifecycle authority, topology, or persistence contracts; existing regressions are sufficient. [VERIFIED: `.planning/ROADMAP.md:673-679`; `docs/adr/0001-topology-specific-storage-guarantees.md:139-180`]

## Code Examples

The removed public names are quoted verbatim as `"SqlCache"` and `"SqlCacheAdapter"` in the current `__all__`; the removed module is exactly `cacheness.sql_cache`. [VERIFIED: `src/cacheness/__init__.py:32-49`]

### Natural absence contract

```python
import importlib.util

import cacheness
import pytest

assert "SqlCache" not in cacheness.__all__
assert "SqlCacheAdapter" not in cacheness.__all__
assert not hasattr(cacheness, "SqlCache")
assert not hasattr(cacheness, "SqlCacheAdapter")
assert importlib.util.find_spec("cacheness.sql_cache") is None

with pytest.raises(ImportError):
    exec("from cacheness import SqlCache", {})
with pytest.raises(ModuleNotFoundError):
    importlib.import_module("cacheness.sql_cache")
```

Use an isolated subprocess for the definitive package proof so an already imported checkout module cannot remain in `sys.modules`. [VERIFIED: `tools/run_phase8_packaging.py:241-284`; recommendation]

### Wheel member and installed metadata proof

```python
from importlib import metadata
from zipfile import ZipFile

with ZipFile(wheel_path) as wheel:
    members = set(wheel.namelist())
assert "cacheness/sql_cache.py" not in members

distribution = metadata.distribution("cacheness")
requirements = tuple(distribution.requires or ())
assert all("duckdb" not in requirement.casefold() for requirement in requirements)
assert "sql" not in {extra.casefold() for extra in distribution.metadata.get_all("Provides-Extra", [])}
```

Run the metadata portion inside the source-free environment created by the existing base probe; run ZIP inspection against the same `WheelArtifact`. [VERIFIED: `tools/run_phase8_packaging.py:189-238`; `tools/run_phase8_packaging.py:333-375`; recommendation]

## Assumptions Log

All implementation claims were verified from the repository, frozen dependency graph, or locked Phase 10 decisions. No `[ASSUMED]` claim remains.

## Open Questions

None. The only terminology mismatch is resolved by the source of truth: D-05 calls `sql` an optional extra, while the manifest proves it is currently a uv dependency group and not an installable project extra. The implementation should delete that group and must not introduce an installable `sql` extra. [VERIFIED: CONTEXT D-05; `pyproject.toml:18-76`; `uv.lock:260`]

## Environment Availability

| Dependency | Required by | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| Python | tests/build | yes | system 3.12.1; uv runtime 3.13.15 | Run the existing CI matrix for supported 3.11/3.12/3.13 evidence. [VERIFIED: observed environment; `pyproject.toml:9`] |
| uv | lock/build/isolated probes | yes | 0.12.12 | None needed. [VERIFIED: observed environment] |
| pytest | contracts | yes in dev resolution | 8.4.1 | None needed. [VERIFIED: `pyproject.toml:68-73`; observed environment] |
| Live PostgreSQL/S3 | retained non-live regressions | not required | — | Use existing non-live/local qualification; this phase makes no live-service claim. [VERIFIED: `.github/workflows/quality.yml:40-52`; `.planning/ROADMAP.md:673-679`] |

The sandboxed research shell required a writable temporary `UV_CACHE_DIR`; this is an agent-environment constraint, not a project requirement. [VERIFIED: observed environment]

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.4.1 [VERIFIED: `pyproject.toml:68-73`; observed environment] |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` [VERIFIED: `pyproject.toml:82-90`] |
| Focused command | `uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false tests/test_phase10_sqlcache_removal.py tests/test_public_api_contract.py tests/test_phase6_public_api_contract.py tests/test_phase6_suite_isolation.py tests/test_phase4_cutover_verifier.py tests/test_phase6_contract_verifier.py tests/test_phase071_contract_verifier.py tests/test_phase9_documentation.py tests/packaging/test_wheel_matrix.py` [RECOMMENDATION] |
| Full non-live suite | `uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false -m 'not (live_postgresql or live_aws_s3 or live_remote)'` [VERIFIED: `tests/qualification/phase8_coverage_baseline.json:2`] |
| Lock gate | `uv lock --check` plus reviewed `uv tree --frozen --invert --package sqlalchemy` and absence checks for DuckDB packages [RECOMMENDATION] |
| Lint gate | `uv run --isolated --group dev --frozen ruff check <changed-python-files>` [RECOMMENDATION consistent with `AGENTS.md:135-143`] |

### CACH-07 behavior-to-test map

| Behavior | Test type | Automated proof | File exists? |
|----------|-----------|-----------------|--------------|
| No source module, top-level export/attribute, alias, tombstone, or compatibility hook | unit/source contract | New `tests/test_phase10_sqlcache_removal.py`; natural failures plus source inspection | ❌ Wave 0 |
| Dedicated tests/docs/examples deleted; current-facing references are bounded | source/documentation contract | New explicit scanner in Phase 10 test plus updated Phase 9 docs contracts | ❌ Wave 0 / existing tests updated |
| Orphan reasons removed | unit/exact-set contract | Updated `tests/test_public_api_contract.py` exact `CacheReason` set | ✅ update |
| `sql` uv group and DuckDB manifest/lock edges absent | manifest/lock contract | Parse TOML/lock and run `uv lock --check`; assert retained extras/deps | ❌ Wave 0 helper or packaging test |
| BlobStore/UnifiedCache, PostgreSQL authority/projection, and dataframe handlers survive | regression | Existing local round trips, handler tests, PostgreSQL non-live tests, full suite | ✅ existing |
| Wheel contains no module and installed metadata contains no DuckDB/`sql` surface | packaging integration | Extended Phase 8 `WheelArtifact` inspection + isolated base probe | ✅ update |
| Caller databases untouched | source/documentation contract | Scanner forbids cleanup/migration code; migration note states unsupported/untouched | ❌ Wave 0 / docs test update |
| Package version unchanged at exact `"0.3.14"` | manifest/public contract | Assert pyproject and `cacheness.__version__` remain equal | ✅ extend existing contracts [VERIFIED: `pyproject.toml:3`; `src/cacheness/__init__.py:27`] |

### Goal-backward acceptance sequence

1. **Structural absence:** assert the source file and all dedicated paths are gone; top-level `__all__`/attributes and import spec are absent; no `__getattr__` or alias supplies the names. [VERIFIED: CONTEXT D-01/D-09/D-13]
2. **Metadata absence:** parse pyproject and lock, prove no `sql` dependency group and no DuckDB edge/package, and prove exact remaining optional extras/dependencies. [VERIFIED: current exact values in `pyproject.toml:18-76`; `uv.lock:178-288`]
3. **Reference correctness:** scan current surfaces with a literal allowlist for three canonical notes and negative tests; preserve dated audits/planning. [VERIFIED: CONTEXT D-10/D-12/D-14]
4. **Retained behavior:** run public/cache/store/handler/PostgreSQL non-live regression nodes, then the frozen non-live full suite. [VERIFIED: `.planning/ROADMAP.md:673-679`; existing test inventory]
5. **Artifact truth:** build one fresh wheel, inspect members, install it source-free, inspect distribution metadata, run BlobStore and UnifiedCache local round trips, and prove natural old-import failures. [VERIFIED: CONTEXT D-16; `tools/run_phase8_packaging.py:217-375`]

### Sampling rate

- **Per task commit:** targeted files for the edited contract plus `uv lock --check` when packaging changes. [RECOMMENDATION]
- **Per wave merge:** focused Phase 10/public/verifier/docs/packaging command above. [RECOMMENDATION]
- **Phase gate:** frozen non-live full suite, scoped Ruff, explicit reference scan, and freshly built isolated wheel all green. [VERIFIED: `.planning/config.json:20-49`; CONTEXT D-13 through D-16]

### Wave 0 gaps

- [ ] `tests/test_phase10_sqlcache_removal.py` — negative source/API/reference/manifest boundary for CACH-07.
- [ ] Extend `tools/run_phase8_packaging.py` and `tests/packaging/test_wheel_matrix.py` — wheel-member and installed-metadata absence proof.
- [ ] Refactor fixed verifier manifests before deleting the three dedicated test modules.
- [ ] Update Phase 9 docs assertions to name exact cutover-note owners instead of blanket bans.

No new test framework, package, service, or fixture is required. [VERIFIED: existing pytest and packaging infrastructure]

## Security Domain

### Applicable ASVS categories

| ASVS category | Applies | Standard control |
|---------------|---------|------------------|
| V2 Authentication | no | No authentication surface changes. [VERIFIED: phase boundary] |
| V3 Session Management | no | No session surface changes. [VERIFIED: phase boundary] |
| V4 Access Control | no | No access-control surface changes. [VERIFIED: phase boundary] |
| V5 Validation, Sanitization and Encoding | yes | Parse TOML/ZIP/distribution metadata with stdlib APIs; keep test-path/reference inventories literal and normalized; never execute documentation text. [VERIFIED: existing path normalization in `tools/verify_phase4_cutover.py:109-128`; recommendation] |
| V6 Stored Cryptography | no new work | Preserve existing signing/integrity implementation unchanged. [VERIFIED: `.planning/ROADMAP.md:673-679`; AGENTS security constraint] |

### Threat patterns and mitigations

| Pattern | STRIDE | Mitigation |
|---------|--------|------------|
| Stale wheel ships deleted module or dependency edge | Tampering / supply-chain | Inspect exact wheel members and installed metadata from the artifact whose digest the harness records. [VERIFIED: `tools/run_phase8_packaging.py:189-238`; CONTEXT D-16] |
| Shim makes old symbol appear “removed” while keeping behavior | Spoofing | Assert no attribute/export/import spec/tombstone and require natural import exceptions. [VERIFIED: CONTEXT D-01/D-13] |
| Broad scrub destroys historical evidence | Repudiation | Explicitly exclude completed planning and dated audits from current-surface absence enforcement. [VERIFIED: CONTEXT D-10/D-14] |
| Over-pruning breaks PostgreSQL/dataframe support | Denial of service | Exact dependency ownership proof plus retained-surface regressions. [VERIFIED: retained source imports cited above] |
| Cleanup code mutates caller-owned tables | Tampering / data loss | Add no cleanup/migration path and document that tables remain untouched/unsupported. [VERIFIED: CONTEXT D-03] |

## Sources

### Primary (HIGH confidence)

- `.planning/phases/10-remove-sqlcache-pull-through-subsystem/10-CONTEXT.md` — locked D-01 through D-16 and scope.
- `.planning/REQUIREMENTS.md:31-39` and `.planning/ROADMAP.md:668-701` — CACH-07, goal, and acceptance boundary.
- `docs/adr/0001-topology-specific-storage-guarantees.md:32-180` — storage ownership and stop conditions.
- `src/cacheness/sql_cache.py`, package exports, error reasons, manifest, lockfile, tests, docs, examples, tools, workflows, and current codebase maps — exact live-tree inventory.
- Observed `uv tree --frozen --invert`, interpreter/tool versions, bounded filesystem inventory, and pytest collection diagnostic — dependency/runtime verification.

No web or third-party documentation was needed: this phase introduces no external API or package and all decisive facts are repository-local or locked user decisions. [VERIFIED: phase boundary and package audit]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — read from the manifest, lock, existing harness, and observed tool versions.
- Architecture: HIGH — locked by ADR 0001, Phase 10 context, roadmap, and current source.
- Exact inventory: HIGH — comprehensive bounded reference scans plus direct source reads.
- Dependency pruning: HIGH — manifest/lock reads and inverse dependency-tree proof.
- Validation architecture: HIGH — extends existing Phase 8/9 contracts rather than inventing a parallel harness.

**Research date:** 2026-09-17
**Valid until:** implementation of Phase 10 or any intervening manifest/public-surface change.
