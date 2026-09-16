# Phase 9: Adoption and Release Surface Closure - Research

**Researched:** 2026-09-16
**Domain:** Python public API cutover, executable documentation, packaging, and release-surface qualification
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Product story and documentation structure
- **D-01:** The README is a concise gateway rather than a comprehensive manual. It briefly establishes the BlobStore-first architecture and provides two quick starts: direct storage and caching through `UnifiedCache`.
- **D-02:** Linked documentation is organized first around user tasks—store objects, cache function results, add a file format, and operate or migrate a store—with focused component references for `BlobStore`, `UnifiedCache`, handlers, and related APIs.
- **D-03:** Documents whose primary purpose is the removed pre-cutover API are deleted from supported documentation rather than retained in a historical docs archive. Git history and planning artifacts preserve historical evidence.
- **D-04:** The README contains a short, prominent qualification-status box. One linked guarantees page owns the detailed topology, platform, payload-bound, and performance claim matrix.
- **D-05:** The supported product story contains `BlobStore` as the storage foundation and `UnifiedCache` as cache policy over that foundation. Phase 9 does not promote `SqlCache`; its separately approved removal belongs to Phase 10.

#### Installation and release posture
- **D-06:** Until an immutable release is published, the primary installation path is a checked-out repository managed with `uv`. Local wheel build/install instructions are secondary.
- **D-07:** Describe the current state as a **local-ready development version**: the checked-out revision is qualified for local use, has not been published, and may change before the first supported release. Do not call it an alpha release candidate or imply a frozen v1.0 API.
- **D-08:** PostgreSQL and S3 appear only in guarantees/reference material and remain visibly `NOT_QUALIFIED`. They do not appear in quick starts, normal recommendations, or promoted examples.
- **D-09:** Documentation starts with the minimal installation and introduces capability-specific extras only in the task guide that needs them. Do not recommend an install-everything or vague broad bundle as the default.

#### Executable examples
- **D-10:** Maintain four canonical example journeys: in-memory `BlobStore`; durable filesystem-plus-SQLite `BlobStore` with catalog metadata; `UnifiedCache` plus decorator use; and store-local MCAP-style custom format registration.
- **D-11:** Audit the existing example inventory aggressively. Delete obsolete, redundant, or unsupported scripts unless they demonstrate a distinct current workflow and can be made executable. Do not create an unsupported-example archive.
- **D-12:** CI executes the exact published example files unchanged. Do not maintain test-only copies of documentation snippets or a second equivalent example implementation.
- **D-13:** Canonical examples are self-verifying and disposable: they use isolated temporary/private storage, assert expected results, print a short success indication, and leave no files behind.

#### Format-handler public surface
- **D-14:** Phase 9 includes one practical MCAP-style tutorial plus the minimum extension contract: store-local registration, stable data/payload identity, version declaration, safe suffixes, contained path I/O, and successful round trips. Reusable conformance tooling and broader long-term extension policy remain future developer-kit work.
- **D-15:** Rename the public `CacheHandler` protocol directly to `FormatHandler` throughout implementation, built-in subclasses, public imports, annotations, errors/messages, tests, documentation, and examples. Do not retain a compatibility alias. — **Reversibility:** costly — Undoing this cutover would touch the same public imports, subclass declarations, tests, and documentation again; the project is intentionally making the correction before a supported release.
- **D-16:** Preserve existing persisted `data_type`, `payload_format`, and payload-format-version identities. Values such as `array`, `object`, `pandas_dataframe`, `npz`, and `parquet` already describe the data/native format and do not encode the misleading Python protocol name. The rename must not create a migration or rebuild requirement.
- **D-17:** Keep the already accepted per-store registration shape, `store.handlers.register_handler(...)`; the rename clarifies the protocol without creating a new registry API.

Source: copied verbatim from the locked implementation decisions. [VERIFIED: `.planning/phases/09-adoption-and-release-surface-closure/09-CONTEXT.md:22-48`]

### the agent's Discretion
- Choose exact task-guide filenames, navigation order, and concise prose while preserving the README/task/reference hierarchy above.
- Decide which obsolete documents and examples contain unique current material worth folding into a canonical guide before deletion.
- Choose the bounded test harness that executes exact canonical example files in isolated temporary environments.
- Choose internal transitional edit order for the alias-free `FormatHandler` cutover, provided no commit or completed plan leaves persisted handler identities changed.

Source: copied verbatim. [VERIFIED: `.planning/phases/09-adoption-and-release-surface-closure/09-CONTEXT.md:50-54`]

### Deferred Ideas (OUT OF SCOPE)
- Add a separate Phase 10 before milestone completion to remove `SqlCache` directly and without a compatibility alias. That phase owns removal of public exports, implementation, dedicated tests, examples, documentation, and dependency surface that becomes unused. Phase 9 should not spend effort promoting or comprehensively rewriting `SqlCache` documentation.
- A future handler developer-kit phase owns reusable conformance tooling, multiple third-party format examples, and a broader long-term handler compatibility policy beyond Phase 9's minimum tutorial contract.
- Real PostgreSQL/Amazon-S3 qualification and immutable publication remain `SEED-007`; controlled-Linux performance remains `SEED-006`; native Windows qualification remains Phase 999.1.

Source: copied verbatim. [VERIFIED: `.planning/phases/09-adoption-and-release-surface-closure/09-CONTEXT.md:125-130`]
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CACH-06 | “The milestone publishes one coherent cache import, constructor, configuration, decorator, and result surface over `BlobStore`.” The implementation is complete; Phase 9 must replace primary documentation and examples that still publish removed APIs. | The live public surface, alias-free handler rename inventory, exact four-example CI contract, task-first documentation map, minimal-install silence gate, and evidence-refresh boundaries below close only the publication/adoption gap. [VERIFIED: `.planning/REQUIREMENTS.md:31-39`] |
</phase_requirements>

## Summary

Phase 9 should be planned as a bounded public-surface cutover, not a storage change. The implementation already exposes `BlobStore(topology, ...)`, `UnifiedCache(config, *, store=...)`, the explicit `cached(cache=...)` decorator, typed cache outcomes, store-local handler registration, and exact versioned manifest identities. The remaining gap is that primary documentation, package prose, and most examples still teach removed cache-first constructors or promote the unrelated SQL pull-through subsystem. [VERIFIED: `src/cacheness/storage/blob_store.py:166-212`; `src/cacheness/core.py:124-188`; `src/cacheness/decorators.py:52-105`; `.planning/v1.0-v1.0-MILESTONE-AUDIT.md:163-175,219-248`]

The highest-risk edit is the alias-free `CacheHandler` to `FormatHandler` rename. It is safe only if the Python symbol, `CacheHandlerError` terminology, built-in bases, active annotations, barrels, tests, wheel export inventory, active agent guidance, and supported docs change together while persisted values remain byte-for-byte unchanged. The manifest is built with `handler_type=handler.data_type`, `payload_format=<handler/result value>`, and a separately stored `payload_format_version`; it does not persist the Python base-class name. [VERIFIED: `src/cacheness/storage/lifecycle.py:492-555`; `src/cacheness/storage/manifest.py:120-170,173-210,262-287`]

Plan the phase in four dependent slices: (1) one atomic active-tree rename plus base-import silence and wheel gates; (2) exactly four disposable examples and their exact-file CI harness; (3) the README/task/reference rewrite and package identity; and (4) bounded planning-evidence cleanup plus the missing Narwhals seed. Do not add a lock, queue, retry coordinator, storage adapter, compatibility alias, format migration, new dependency, or remote-service test. ADR 0001 explicitly places lifecycle authority in `BlobStore` and says planning must stop before adding another coordination mechanism. [VERIFIED: `docs/adr/0001-topology-specific-storage-guarantees.md:30-45,139-170,172-192`; `.planning/ROADMAP.md:615-627`]

**Primary recommendation:** Cut the Python name atomically while freezing persisted identities, then make the four executable example files the sole onboarding source of truth and rewrite documentation around them.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Direct object persistence | Database / Storage (`BlobStore`) | Format handler | `BlobStore` owns lifecycle and calls the selected handler only for native serialization/deserialization. [VERIFIED: `src/cacheness/storage/blob_store.py:166-185,246-265`; `docs/adr/0001-topology-specific-storage-guarantees.md:36-44`] |
| Cache policy and decorators | API / Backend (`UnifiedCache`) | Database / Storage (`BlobStore`) | `UnifiedCache` owns keying, TTL, eviction, invalidation, and outcomes while delegating entry lifecycle to one store; `cached` requires an explicit cache. [VERIFIED: `src/cacheness/core.py:124-188,893-951,1001-1046`; `src/cacheness/decorators.py:52-105`] |
| Native format extension | API / Backend (`FormatHandler` + `HandlerRegistry`) | Database / Storage guarded staging | The handler declares stable data/format/version identities and path-based put/get methods; guarded I/O chooses and contains private paths. [VERIFIED current seam: `src/cacheness/interfaces.py:38-112,151-210,243-263`; `src/cacheness/storage/guarded_handler_io.py:166-220,270-339`] |
| Canonical catalog and topology claims | Database / Storage authority | Documentation/release surface | Only the authority determines committed membership; docs must describe the qualified topology rather than infer support from constructibility. [VERIFIED: `docs/CATALOG_AND_TOPOLOGY.md:1-31,66-87`; `docs/adr/0001-topology-specific-storage-guarantees.md:70-81`] |
| Installation, examples, qualification status | Release/adoption surface | CI | README, task guides, examples, package metadata, and exact-file tests publish the usable contract; they must not strengthen storage guarantees. [VERIFIED: `.planning/phases/09-adoption-and-release-surface-closure/09-CONTEXT.md:25-48`; `.planning/ROADMAP.md:620-627`] |

## Project Constraints (from AGENTS.md)

- The project is explicitly pre-production, so development-only APIs may be replaced without compatibility adapters, but explicit schema/format versions and offline migration/rebuild tooling must remain. [VERIFIED: `AGENTS.md:13-17`]
- `BlobStore` owns storage lifecycle, `UnifiedCache` owns cache policy above it, and Phase 10—not Phase 9—removes `SqlCache`. [VERIFIED: `AGENTS.md:16-18`]
- Trusted application payloads do not relax safe parsing, path containment, or fail-closed integrity boundaries. [VERIFIED: `AGENTS.md:19-21`]
- Python support is `>=3.11`; verification cannot rely only on the current Python 3.13 development interpreter. [VERIFIED: `AGENTS.md:22-23`; `pyproject.toml:1-16`]
- Keep public barrels explicit, use package-relative internal imports, guard optional dependencies, preserve causes when translating failures, and use targeted Ruff rather than assuming the legacy tree is globally clean. [VERIFIED: `AGENTS.md:119-149,151-165,183-189`]
- Any lifecycle, concurrency, recovery, topology, timeout, or `BlobStore`/`UnifiedCache` composition change must be checked against ADR 0001; Phase 9 should not make such a change. [VERIFIED: `AGENTS.md:27-36`]

## Standard Stack

### Core

| Library/tool | Version | Purpose | Why Standard Here |
|---------|---------|---------|--------------|
| Python | `>=3.11` | Library, examples, subprocess probes | This is the declared package floor, and local 3.11, 3.12, 3.13, and 3.14 interpreters were available during research. [VERIFIED: `pyproject.toml:1-16`; local environment probe 2026-09-16] |
| `uv` | local `0.12.12` | Checkout environment, frozen dependency sync, local wheel build | Locked project and CI commands already use `uv`; Phase 9 should document checkout-first use rather than introduce another installer workflow. [VERIFIED: `.github/workflows/quality.yml:19-47,210-255`; local environment probe 2026-09-16] |
| pytest | locked `8.4.1` | Exact-file example harness and public-surface regressions | Existing configuration discovers `tests/test_*.py`, defines live-service exclusions, and already has reusable subprocess/example tests. [VERIFIED: `uv.lock:2142-2152`; `pyproject.toml:82-102`; `tests/test_phase6_examples.py:1-130`] |
| Ruff | locked `0.12.9` | Targeted lint on changed Python files | The repository target is Python 3.11 and line length 88, but legacy global debt means Phase 9 should lint its changed scope. [VERIFIED: `uv.lock:2590-2594`; `pyproject.toml:132-147`; `AGENTS.md:133-144`] |
| obstore | locked `0.11.1` | Existing payload participant used by the examples | Phase 9 documents and exercises the existing participant; it neither changes nor replaces it. [VERIFIED: `pyproject.toml:9-16`; `uv.lock:1444-1452`; `src/cacheness/storage/obstore_generation_io.py:65-113`] |

### Supporting

| Existing component | Purpose | When to Use |
|-------------------|---------|-------------|
| `TemporaryDirectory` | Disposable example roots | Every canonical example must create and clean its own private storage. Existing cache examples already follow this pattern. [VERIFIED: `examples/simple_object_caching.py:7-35,75-83`; `examples/api_request_caching.py:7-52,135-142`] |
| `tools/run_phase8_packaging.py` | Source-free wheel and explicit export probes | Extend its literal storage export inventory from `CacheHandler` to `FormatHandler` and add old-name absence/import-silence checks; do not build a second wheel runner. [VERIFIED: `tools/run_phase8_packaging.py:47-178,213-234,249-361`] |
| `docs/RELEASE_QUALIFICATION.md` | Existing evidence/nonclaim owner | Retain this path and turn it into the one detailed guarantees/qualification page, because existing verifiers and workflow tests depend on it. [VERIFIED: `docs/RELEASE_QUALIFICATION.md:1-38,58-92,127-152`; `tools/verify_phase8_contracts.py:31-44`; `tests/qualification/test_phase8_quality_workflow.py:17`] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Direct alias-free rename | Keep `CacheHandler = FormatHandler` | Rejected by D-15; an alias would prolong the misleading cache-only concept and widen the first supported surface. [VERIFIED: `.planning/phases/09-adoption-and-release-surface-closure/09-CONTEXT.md:44-48`] |
| Exact executable example files | Duplicate snippets in tests/docs | Rejected by D-12 because copies drift; subprocess tests should run the published files unchanged. [VERIFIED: `.planning/phases/09-adoption-and-release-surface-closure/09-CONTEXT.md:38-42`] |
| Existing pytest/subprocess harness | Add a documentation framework or example runner dependency | Unnecessary; the current harness already provides isolation, network blocking, repeat execution, and marker checks. [VERIFIED: `tests/test_phase6_examples.py:18-102`] |
| Seed Narwhals | Install and integrate Narwhals now | Out of scope; Phase 8 recorded the investigation as future handler/extensibility work and no seed currently exists. [VERIFIED: `.planning/phases/08-production-gates-and-performance-stabilization/08-CONTEXT.md:134`; `.planning/v1.0-v1.0-MILESTONE-AUDIT.md:227-248`; `.planning/seeds/` inventory probe 2026-09-16] |

**Installation:** no package installation or lockfile change is required for Phase 9. Keep the current checkout workflow and existing dependencies. [VERIFIED: locked D-06/D-09; `pyproject.toml:9-45`]

## Package Legitimacy Audit

Not applicable: this phase should add no external package. The Narwhals work is captured as a seed, not installed. Therefore no package-legitimacy gate is required for execution. [VERIFIED: `.planning/phases/09-adoption-and-release-surface-closure/09-CONTEXT.md:32-36,125-130`; `.planning/ROADMAP.md:620-627`]

## Live Repository Findings

### Public rename inventory

The active, non-historical tree contains 15 files with `CacheHandler`, `CacheHandlerError`, or “cache handler” terminology: `src/cacheness/{interfaces.py,handlers.py,error_handling.py}`, both storage barrels, four active test modules, the packaging runner, `README.md`, three supported docs, and `AGENTS.md`. Historical `.planning/`, spike evidence, and abandoned `.claude/worktrees/` also contain the old term, but they are evidence rather than the live API. [VERIFIED: scoped `rg` inventory executed 2026-09-16]

Rename in one atomic active-tree change:

| Surface | Required action |
|---------|-----------------|
| `src/cacheness/interfaces.py` | Rename `CacheHandler` to `FormatHandler`; rename public `CacheHandlerError` to `FormatHandlerError`; update specialized bases, factory/registry annotations, guarded-I/O annotation, docstrings, and logged message. Preserve `PAYLOAD_FORMAT_VERSION = 1` and all identity methods. [VERIFIED: `src/cacheness/interfaces.py:151-210,243-267,340-388,415-470`] |
| `src/cacheness/handlers.py` | Rename imports, every built-in base, return/parameter annotations, default-transform identity comparison, validation text, and examples. Do not rename `HandlerRegistry` or change registration semantics. [VERIFIED: `src/cacheness/handlers.py:1386-1520,1522-1702`] |
| `src/cacheness/error_handling.py` | Rename the second cross-cutting `CacheHandlerError` class too; otherwise two contradictory error vocabularies remain. [VERIFIED: `src/cacheness/error_handling.py:111-112`] |
| `src/cacheness/storage/__init__.py` and `src/cacheness/storage/handlers/__init__.py` | Export only `FormatHandler`/`FormatHandlerError`; do not export old aliases. Keep the current storage barrel as the public import location rather than adding a new top-level `cacheness.FormatHandler` surface. [VERIFIED: `src/cacheness/storage/__init__.py:26-50,240-270`; scoped source inventory 2026-09-16] |
| Tests and packaging | Rename imports/assertions, assert new imports work, and assert `CacheHandler`/`CacheHandlerError` are absent from public barrels and source-free wheels. Change the literal wheel export at `BASE_PUBLIC_EXPORTS`, not a runtime-derived list. [VERIFIED: `tools/run_phase8_packaging.py:47-178`; `tests/test_interfaces.py` and `tests/test_handler_registration.py` scoped inventory 2026-09-16] |
| Active guidance | Update README, supported docs, AGENTS, and the auto-loaded `spike-findings-cacheness` skill's current instructions/reference prose. Do not rewrite old phase plans/summaries merely to make a repository-wide grep empty. [VERIFIED: `.planning/phases/09-adoption-and-release-surface-closure/09-CONTEXT.md:58-85`; scoped `rg` inventory 2026-09-16] |

`CacheWriteError`, `CacheReadError`, `CacheFormatError`, and `CacheValidationError` can remain: none encodes the misleading protocol name. Only their base becomes `FormatHandlerError`. This is the smallest direct interpretation of D-15's “errors/messages” requirement. [VERIFIED current hierarchy: `src/cacheness/interfaces.py:340-380`; locked rename: `.planning/phases/09-adoption-and-release-surface-closure/09-CONTEXT.md:44-48`]

### Persistence invariant

The exact stored fields are quoted verbatim as `"handler_type"`, `"payload_format"`, and `"payload_format_version"`; `StoreVersionDimensions` also keeps `"store_epoch"`, `"manifest_schema_version"`, `"sqlite_user_version"`, and `"store_format_version"` independently. Manifest construction assigns `handler_type=handler.data_type`. No stored field contains the Python identifier `CacheHandler`. [VERIFIED: `src/cacheness/storage/manifest.py:120-170,173-210,262-287`; `src/cacheness/storage/lifecycle.py:492-555`]

The rename gate must therefore compare before/after built-in identities and reopen a durable store without migration. Preserve verbatim existing identities including `"array"`, `"object"`, `"pandas_dataframe"`, `"npz"`, and `"parquet"`; changing any of them is a Phase 7-format migration concern and is forbidden here. [VERIFIED locked values: `.planning/phases/09-adoption-and-release-surface-closure/09-CONTEXT.md:44-48`; current examples: `src/cacheness/handlers.py:235-241,504-512,569-585,753-782,936-953,1247-1248`]

### Documentation and example audit

The current README is 812 lines and still begins with `from cacheness import cacheness`, constructs the removed `cacheness()` facade, uses `@cached(ttl_hours=24)`, promotes `SqlCache`, and imports `CacheHandler` from a top-level barrel that does not export it. [VERIFIED: `README.md:1-99,126-156,206-251,732-759`; `src/cacheness/__init__.py:12-49`]

The primary stale documents are `docs/BLOB_STORE.md`, `docs/API_REFERENCE.md`, `docs/BACKEND_SELECTION.md`, `docs/CONFIGURATION.md`, `docs/SECURITY.md`, and `docs/PLUGIN_DEVELOPMENT.md`; they contain removed constructors, global registration, or obsolete topology claims. Existing current material worth folding rather than losing includes the trusted-payload section in `docs/SECURITY.md`, catalog/topology contracts, explicit initialization, migration tooling, and Phase 8 nonclaims. [VERIFIED: `.planning/v1.0-v1.0-MILESTONE-AUDIT.md:219-248`; `tests/test_security_documentation.py:20-75`; `docs/STORAGE_MIGRATION.md:1-30,98-121`; `docs/RELEASE_QUALIFICATION.md:1-38,127-152`]

Use this disposition during planning:

| Current material | Phase 9 disposition |
|------------------|---------------------|
| `README.md` | Replace with concise gateway, checkout-first install, two quick starts, qualification box, and task links. [VERIFIED locked D-01/D-04/D-06: `09-CONTEXT.md:25-34`] |
| `docs/README.md` | Rewrite as task-first navigation: store objects; cache function results; add a format; operate/migrate; references. [VERIFIED locked D-02: `09-CONTEXT.md:25-30`] |
| `docs/BLOB_STORE.md` | Rewrite as the direct-storage task/reference guide using explicit `StoreTopology`, `initialize()`, result/receipt, catalog, and close semantics. [VERIFIED public API: `src/cacheness/storage/blob_store.py:166-212,370-417,487-559,616-692,851-891`] |
| `docs/CACHE_POLICY.md` | Retain and narrow to the current `UnifiedCache`/decorator/result contract; remove `SqlCache` promotion and keep exact outcomes `"hit"`, `"absent"`, `"expired"`, `"corrupt"`, `"conflict"`, and `"backend_error"`. [VERIFIED: `src/cacheness/cache_policy.py:22-30`; `tests/test_phase6_examples.py:105-130`] |
| `docs/PLUGIN_DEVELOPMENT.md` | Replace with the one MCAP-style `FormatHandler` tutorial and minimum safe extension contract. [VERIFIED locked D-14/D-17: `09-CONTEXT.md:44-48`] |
| `docs/API_REFERENCE.md` | Narrow to the actual BlobStore, UnifiedCache, result, catalog, topology, migration, and format-handler public imports. Do not comprehensively rewrite SqlCache. [VERIFIED: `src/cacheness/__init__.py:12-49`; `src/cacheness/storage/__init__.py:49-270`; locked D-05/deferred Phase 10: `09-CONTEXT.md:25-30,125-130`] |
| `docs/CONFIGURATION.md`, `docs/BACKEND_SELECTION.md` | Fold current setup into task/reference guides; delete if their primary content remains removed pre-cutover API. Keep explicit local topology and remote NOT_QUALIFIED boundaries. [VERIFIED locked D-03/D-08/D-09: `09-CONTEXT.md:25-36`] |
| `docs/SECURITY.md` | Rewrite stale examples but preserve its tested `## Trusted Payload and Executable Serializer Boundary`, object-array predicates, and fail-closed language. [VERIFIED: `tests/test_security_documentation.py:20-75`] |
| `docs/RELEASE_QUALIFICATION.md` | Keep path; make it the single detailed guarantees/qualification matrix and update stale “Phase 8 will qualify remote” statements elsewhere to SEED-007/NOT_QUALIFIED. [VERIFIED: `docs/RELEASE_QUALIFICATION.md:1-38,94-152`; `docs/CATALOG_AND_TOPOLOGY.md:33-64`] |
| Pre-cutover design/audit/feature guides whose examples depend on removed APIs | Fold unique accurate material, then delete; do not create a supported-docs archive. Planning artifacts and Git remain the history. [VERIFIED locked D-03: `09-CONTEXT.md:25-30`] |
| Dedicated SqlCache docs/examples | Remove from navigation and promotion but leave their physical Phase 10 removal to CACH-07. This resolves the apparent conflict between the Phase 9 cleanup goal and the explicit Phase 10 ownership boundary. [VERIFIED: `.planning/ROADMAP.md:635-646`; `09-CONTEXT.md:125-130`] |

The example directory currently contains 19 Python scripts. Existing examples include removed decorators/constructors, runtime network/package behavior, unqualified S3 use, and SqlCache-only workflows. The tested Phase 6 allowlist currently runs only `simple_object_caching.py`, `configurable_serialization_demo.py`, and `api_request_caching.py`; the durable catalog demo is close to the desired second journey but needs explicit initialization and self-verifying assertions. [VERIFIED: example inventory probe 2026-09-16; `tests/test_phase6_examples.py:54-102`; `examples/custom_metadata_demo.py:25-77`]

Publish exactly these four canonical journeys, preferably with purpose-revealing names:

1. `memory_blob_store.py` — explicit memory/memory topology; initialize, put, get, assert, close.
2. `durable_catalog_store.py` — temporary filesystem payload + SQLite authority; initialize, catalog put/query/update, close/reopen, assert durability.
3. `unified_cache.py` — caller-selected store/topology, typed lookup, `@cached(cache=cache)`, cached `None` or call-count proof, bounded clear.
4. `custom_mcap_format.py` — `FormatHandler`, stable MCAP data/payload/version identity, `.mcap` suffix, path-based I/O, `store.handlers.register_handler(..., priority=0)`, round trip.

The registered topology names are verbatim `"memory"`, `"filesystem"`, and `"sqlite"`; filesystem uses option `"base_dir"`, while SQLite uses `"root"`. [VERIFIED: `src/cacheness/storage/composition.py:573-615,747-759`]

Retire the three old Phase 6 example names after their unique behavior is folded into the four files. Delete obsolete non-SqlCache scripts; keep any Phase-10-owned SqlCache scripts unlinked and explicitly noncanonical until Phase 10 deletes them. The Phase 9 harness should maintain a literal four-file allowlist, run each file unchanged twice from a fresh directory with sockets disabled, assert its success marker, and verify the example leaves no storage residue. [VERIFIED pattern: `tests/test_phase6_examples.py:18-102`; locked D-10-D-13: `09-CONTEXT.md:38-42`]

### Optional dependency and package surface

The minimal install currently emits `“⚠️  Neither Polars nor Pandas available - DataFrame caching disabled”` on a plain `import cacheness`; the warning is produced at module import time. An isolated minimal-install probe reproduced the message on stderr. Remove the import-time availability logs entirely and keep missing-feature feedback at the point where a dataframe handler is actually requested. [VERIFIED: `src/cacheness/handlers.py:30-45,222-232`; isolated `uv run --isolated --no-project --with . python -c 'import cacheness'` probe 2026-09-16]

The project advertises the exact optional groups `"recommended"`, `"dataframes"`, `"tensorflow"`, `"s3"`, `"postgresql"`, and `"cloud"`, and the wheel matrix freezes that inventory. Phase 9 should change guidance, not opportunistically redesign extras: base install first; `[dataframes]` only in dataframe tasks; remote groups only in the NOT_QUALIFIED reference. Phase 10 may later remove dependency surface made unused by SqlCache. [VERIFIED: `pyproject.toml:18-45`; `tools/run_phase8_packaging.py:26-35,364-375`; `.planning/ROADMAP.md:635-646`]

Update the stale package description `“High-performance disk caching library using Blosc compression...”` to the BlobStore-first product identity, without claiming publication or remote qualification. [VERIFIED: `pyproject.toml:1-5`; locked D-05/D-07: `09-CONTEXT.md:25-35`]

### Evidence metadata and missing seed

Phase 3's checked-in verification still says `status: gaps_found`, `score: 3/8`, while the milestone audit says Plans 03-21 through 03-25 and subsequent Phase 07.1/08 suites closed those runtime gaps and directs a metadata/evidence refresh rather than another race-fix cycle. Phase 9 may update the report only to claims directly supported by those existing artifacts and named passing tests; it must not rerun an automated fix loop or edit lifecycle production code. [VERIFIED: `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-VERIFICATION.md:1-70`; `.planning/v1.0-v1.0-MILESTONE-AUDIT.md:182-202`]

Phase 8 verification is passed, but `08-VALIDATION.md` frontmatter still says verbatim `status: planned` and `wave_0_complete: false`. Existing Phase 8 verification and local-readiness evidence support refreshing those metadata fields without inventing live, Linux-performance, Windows, publication, PostgreSQL, or S3 evidence. The local record says `"status":"LOCAL_READY"`, includes four local PASS classes, and retains nonclaims `"controlled_performance"`, `"linux_matrix"`, `"windows"`, `"live_services"`, and `"immutable_publication"`. [VERIFIED: `.planning/phases/08-production-gates-and-performance-stabilization/08-VALIDATION.md:1-15`; `.planning/phases/08-production-gates-and-performance-stabilization/08-VERIFICATION.md:1-39`; `.planning/phases/08-production-gates-and-performance-stabilization/08-LOCAL-READINESS.json:1`]

No Narwhals seed exists. Create exactly one next-numbered dormant seed for investigating Narwhals as a dataframe-handler compatibility layer across pandas, PyArrow, and Polars while retaining Parquet as handler-owned format. The seed should point to the Phase 8 decision, current dataframe handlers, `[dataframes]` extra, and future handler developer-kit work; it must not add Narwhals to `pyproject.toml` or the lockfile. [VERIFIED: `.planning/phases/08-production-gates-and-performance-stabilization/08-CONTEXT.md:134`; `.planning/v1.0-v1.0-MILESTONE-AUDIT.md:227-248`; `.planning/seeds/SEED-001...SEED-007` inventory probe 2026-09-16]

## Architecture Patterns

### System Architecture Diagram

```text
Application value / function call
          |
          +---------------- direct storage ----------------+
          |                                                |
          v                                                v
      BlobStore <---- StoreTopology ---- qualified local participant pair
          |                        memory+memory OR sqlite+filesystem
          |
          +--> HandlerRegistry --> FormatHandler --> private suffix-preserving stage
          |                              |                       |
          |                              | stable identities     v
          |                              +--------------> existing obstore participant
          |
          +--> one LifecycleAuthority --> signed/versioned canonical manifest
          |
          +---------------- cache policy ------------------+
                                                           v
                                                     UnifiedCache
                                                           |
                                                     cached(cache=...)

The same public paths feed four exact examples --> pytest subprocess harness --> CI
README/task guides link those examples and one guarantees page; they do not define
another storage lifecycle or stronger qualification class.
```

This is the existing architectural flow; Phase 9 changes names, examples, and publication surfaces only. [VERIFIED: `src/cacheness/storage/blob_store.py:166-212,246-265`; `src/cacheness/core.py:124-188`; `src/cacheness/decorators.py:52-105`; `docs/adr/0001-topology-specific-storage-guarantees.md:30-45`]

### Recommended Project Structure

```text
README.md                         # concise gateway + two quick starts
docs/
├── README.md                    # task-first navigation
├── BLOB_STORE.md                # store objects / direct API
├── CACHE_POLICY.md              # cache function results / result model
├── PLUGIN_DEVELOPMENT.md        # add an MCAP-style format
├── STORAGE_INITIALIZATION.md    # operate initialized stores
├── STORAGE_MIGRATION.md         # explicit offline migration/rebuild
├── SECURITY.md                  # trusted payload and integrity boundary
├── API_REFERENCE.md             # focused current symbol reference
└── RELEASE_QUALIFICATION.md     # sole detailed guarantees/status matrix
examples/
├── memory_blob_store.py
├── durable_catalog_store.py
├── unified_cache.py
└── custom_mcap_format.py
tests/
└── test_phase9_examples.py      # launches those exact four files
```

The filenames are agent discretion except for retaining the already-wired qualification path; this structure is the recommended concrete choice. [VERIFIED discretion: `.planning/phases/09-adoption-and-release-surface-closure/09-CONTEXT.md:50-54`; qualification wiring: `tools/verify_phase8_contracts.py:31-44`]

### Pattern 1: Atomic symbol cutover with persistence lock

**What:** Rename every active Python type/import/message and test in one dependency wave, while adding explicit tests that old public names are absent and persisted handler identities are unchanged.

**When to use:** For the D-15 one-way public cutover before any docs/examples adopt the new name.

**Required gate:** a filesystem+SQLite entry written before the in-test module reload/reopen remains readable with the same `handler_type`, `payload_format`, and `payload_format_version`; no migration service runs. [VERIFIED persistence seam: `src/cacheness/storage/lifecycle.py:492-555`; reopen pattern: `tests/test_stored_compatibility.py:102-130`]

### Pattern 2: Published file as executable truth

**What:** Put complete examples in `examples/`, then execute those exact paths from a fresh subprocess. The test owns network blocking and process isolation but not an alternate implementation.

**When to use:** For all four canonical journeys and CI.

**Required gate:** literal allowlist equality, repeatable stdout markers, zero sockets, zero residual storage, and a workflow step that invokes the same pytest test. [VERIFIED reusable pattern: `tests/test_phase6_examples.py:18-102`; locked exact-file decision: `09-CONTEXT.md:38-42`]

### Pattern 3: Single claim owner

**What:** Keep one detailed guarantees matrix in `docs/RELEASE_QUALIFICATION.md`; README and task guides link it and repeat only the small warning necessary for their task.

**When to use:** For topology, platform, transfer ceiling, controlled performance, and publication status.

**Required gate:** PostgreSQL/S3 remain `NOT_QUALIFIED`, controlled Linux remains deferred, Windows remains unqualified, publication remains `NOT_PUBLISHED`, and local readiness never becomes a remote/release claim. [VERIFIED: `docs/RELEASE_QUALIFICATION.md:1-38,40-56,58-92`; `08-LOCAL-READINESS.json:1`]

### Pattern 4: Request-bound optional behavior

**What:** Importing the base wheel is silent about absent optional dataframe packages; guidance mentions `[dataframes]` only in the dataframe task, and a missing requested capability yields focused feedback at the request boundary.

**When to use:** For pandas/polars/pyarrow handlers.

**Required gate:** source-free minimal wheel import has empty package-generated stderr and retains generic/NumPy round trips; dataframe extras retain their independent Parquet probe. [VERIFIED current runner: `tools/run_phase8_packaging.py:249-361,383-425`]

### Anti-Patterns to Avoid

- **Compatibility alias:** D-15 explicitly forbids it; test old-name absence instead. [VERIFIED: `09-CONTEXT.md:44-48`]
- **Changing `data_type` during class rename:** this creates an unsupported stored-format transition and violates D-16. [VERIFIED: `09-CONTEXT.md:44-48`; `src/cacheness/storage/manifest.py:173-210`]
- **Another lifecycle or example abstraction:** examples should call current public APIs directly; no helper should hide topology or initialization. [VERIFIED: `docs/adr/0001-topology-specific-storage-guarantees.md:139-170`; locked D-12: `09-CONTEXT.md:38-42`]
- **Deleting SqlCache implementation/docs under Phase 9 cleanup:** stop promoting it, but Phase 10 owns removal. [VERIFIED: `.planning/ROADMAP.md:635-646`; `09-CONTEXT.md:125-130`]
- **Treating constructibility as qualification:** S3/PostgreSQL may exist in registries/extras while remaining NOT_QUALIFIED. [VERIFIED: `src/cacheness/storage/composition.py:616-643`; `docs/RELEASE_QUALIFICATION.md:94-106`]
- **Refreshing evidence by assertion:** update only metadata supported by checked-in reports/tests; do not infer a new exact commit, remote run, or performance result. [VERIFIED: `.planning/v1.0-v1.0-MILESTONE-AUDIT.md:182-202,227-248`]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Example verification | A second example implementation or Markdown-snippet evaluator | pytest subprocess execution of the exact files | Existing isolation/network/repeat pattern is sufficient and D-12 forbids copies. [VERIFIED: `tests/test_phase6_examples.py:18-102`; `09-CONTEXT.md:38-42`] |
| Minimal-install verification | A new installer matrix | Existing source-free wheel runner | It already builds one wheel, strips inherited Python paths, and probes literal exports/extras. [VERIFIED: `tools/run_phase8_packaging.py:213-361,364-375`] |
| Handler path safety | Tutorial-side managed-path logic | Existing private `GuardedHandlerIO` boundary | It validates suffixes, containment, regular files, descriptor identity, and private snapshot lifetime. [VERIFIED: `src/cacheness/storage/guarded_handler_io.py:166-220,270-339`] |
| MCAP integration | A new MCAP dependency or backend | A tiny MCAP-style byte-record tutorial handler | Phase 9 proves the extension contract, not the MCAP ecosystem. [VERIFIED locked scope: `09-CONTEXT.md:44-48,125-130`] |
| Format compatibility | A global converter or implicit version upgrade | Existing handler-declared exact contract and offline migration/rebuild tooling | Unknown contracts fail closed and directed transformations stay handler-owned. [VERIFIED: `src/cacheness/interfaces.py:159-210`; `src/cacheness/handlers.py:1446-1520`; `.planning/REQUIREMENTS.md:52-60`] |
| Docs claim synchronization | Copy the full matrix into every guide | One guarantees page plus focused links | D-04 assigns one owner, preventing drift. [VERIFIED: `09-CONTEXT.md:25-30`] |
| Reliability fix during docs work | New lock/queue/retry/timeout | Existing ADR-declared outcomes and current engine | Phase 9 has no lifecycle defect to solve and ADR 0001 defines stop conditions. [VERIFIED: `docs/adr/0001-topology-specific-storage-guarantees.md:172-192`] |

**Key insight:** this phase succeeds by deleting contradictory surface area and binding prose to executable public examples; additional architecture would increase risk without addressing CACH-06.

## Runtime State Inventory

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | Current manifests persist `"handler_type"`, `"payload_format"`, and `"payload_format_version"`, populated from handler identities—not the `CacheHandler` class name. [VERIFIED: `src/cacheness/storage/lifecycle.py:492-555`; `src/cacheness/storage/manifest.py:173-210,262-287`] | **No data migration.** Add a durable reopen regression that proves the rename leaves these values unchanged; preserve all schema/version/migration tools. |
| Live service config | No external service configuration needs renaming. Repository CI contains Phase 8 jobs but no `CacheHandler` reference; Phase 9 only adds the exact-example test step. [VERIFIED: `.github/workflows/quality.yml:18-103,128-263`; scoped workflow `rg` 2026-09-16] | Code edit to workflow only; no external dashboard/API mutation. PostgreSQL/S3 remain uninvoked. |
| OS-registered state | None found: Cacheness is an in-process Python library with no systemd/launchd/Task Scheduler registration in this phase. [VERIFIED: `AGENTS.md:105-109`; repository OS-registration search 2026-09-16] | None. |
| Secrets/env vars | No secret or environment variable name contains `CacheHandler`; examples must not require credentials. Live service secrets remain out of scope and protected. [VERIFIED: scoped `rg` 2026-09-16; `.github/workflows/live_qualification.yml` inventory; locked D-08: `09-CONTEXT.md:32-36`] | None; retain network-blocked example execution. |
| Build artifacts / installed packages | Previously built wheels or bytecode can still expose the old symbol, while the packaging runner creates a fresh wheel under its supplied output directory and hashes it. [VERIFIED: `tools/run_phase8_packaging.py:204-234`] | Rebuild a fresh source-free wheel; assert `FormatHandler` present and old names absent. Do not edit generated artifacts. |

## Common Pitfalls

### Pitfall 1: Renaming the class but not both error hierarchies
**What goes wrong:** `CacheHandler` disappears while `CacheHandlerError` remains in `interfaces.py`, `error_handling.py`, tests, or barrels.
**Why it happens:** The repository currently has two classes with that base name. [VERIFIED: `src/cacheness/interfaces.py:340-380`; `src/cacheness/error_handling.py:111-112`]
**How to avoid:** Include both in the atomic rename inventory; keep narrower `CacheWriteError`/`CacheReadError` subclasses.
**Warning signs:** active-tree grep finds `CacheHandler` or “Cache handler error” after the cutover.

### Pitfall 2: Accidentally converting a source rename into a format migration
**What goes wrong:** built-in `data_type`, native format, or version values change, so existing manifests become unsupported.
**Why it happens:** `handler_type` sounds like a Python type but actually stores `handler.data_type`. [VERIFIED: `src/cacheness/storage/lifecycle.py:492-555`]
**How to avoid:** freeze exact identity tuples and run a filesystem+SQLite close/reopen test.
**Warning signs:** migration/rebuild exceptions appear after only a Python symbol rename.

### Pitfall 3: Leaving import-time optional warnings
**What goes wrong:** minimal `import cacheness` warns that dataframe support is absent even when no dataframe was requested.
**Why it happens:** `handlers.py` logs availability during module import. [VERIFIED: `src/cacheness/handlers.py:30-45,222-232`]
**How to avoid:** remove availability logging and make the source-free base probe reject package-generated output.
**Warning signs:** isolated base import stderr contains “Neither Polars nor Pandas”.

### Pitfall 4: Docs rewrite breaks inherited contract gates
**What goes wrong:** concise docs omit security/transport terms that Phase 07.1 tests deliberately protect, or rename the wired qualification path.
**Why it happens:** existing verifiers scan README, API, plugin, security, and release-guide content. [VERIFIED: `tools/verify_phase071_contracts.py:665-719`; `tests/test_public_api_contract.py:215-247`; `tests/test_security_documentation.py:20-75`]
**How to avoid:** preserve the actual contract or deliberately update the corresponding test to point to the single canonical owner while keeping the semantic assertion.
**Warning signs:** Phase 07.1 fixed verifier fails on missing terms despite correct code.

### Pitfall 5: “Exactly four examples” deletes Phase-10-owned SqlCache assets early
**What goes wrong:** Phase 9 crosses the explicit CACH-07 removal boundary.
**Why it happens:** D-11's aggressive cleanup can be read as physical removal of every script.
**How to avoid:** make exactly four examples canonical and CI-executed; delete obsolete non-SqlCache scripts now, leave unlinked SqlCache-dedicated deletion to Phase 10. [VERIFIED: `09-CONTEXT.md:38-42,125-130`; `.planning/ROADMAP.md:635-646`]
**Warning signs:** Phase 9 touches `src/cacheness/sql_cache.py`, removes its exports, or drops SQL-cache-only dependencies.

### Pitfall 6: Evidence refresh invents a current qualification
**What goes wrong:** stale frontmatter is changed to “passed” without naming the existing evidence, or local readiness is presented as remote/performance/publication qualification.
**Why it happens:** Phase 8 has PASS verification beside explicit deferred/nonclaim fields. [VERIFIED: `08-VERIFICATION.md:1-39`; `08-LOCAL-READINESS.json:1`]
**How to avoid:** cite the existing report and preserve every deferred/nonclaim value; use documentation-only metadata changes.
**Warning signs:** BACK-05/QUAL-06 become complete, publication becomes published, or a new evidence SHA appears without a run.

## Code Examples

These skeletons use only values and signatures opened in the current source. The exact registered topology values are `"memory"`, `"filesystem"`, and `"sqlite"`; the MCAP test identities are `"mcap"`, `"mcap-v1"`, `"mcap-v2"`, versions `1`/`2`, and suffix `".mcap"`. [VERIFIED: `src/cacheness/storage/composition.py:573-615`; `tests/test_handler_registration.py:66-96`; `tests/contracts/test_obstore_generation_io.py:301-330`]

### Direct memory store

```python
from tempfile import TemporaryDirectory

from cacheness.storage import BackendRef, BlobStore, StoreTopology

with TemporaryDirectory() as root:
    store = BlobStore(
        StoreTopology(
            payload=BackendRef(name="memory"),
            authority=BackendRef(name="memory"),
        ),
        cache_dir=root,
    )
    try:
        store.initialize()
        key = store.put({"answer": 42}, key="example")
        assert store.get(key) == {"answer": 42}
    finally:
        store.close()
```

Source pattern: [VERIFIED: `tools/run_phase8_packaging.py:279-315`; `src/cacheness/storage/blob_store.py:370-404,541-559,617-619,851-891`]

### Format handler minimum contract

```python
from pathlib import Path

from cacheness.storage import FormatHandler


class McapFormatHandler(FormatHandler):
    @property
    def data_type(self) -> str:
        return "mcap"

    @property
    def payload_format(self) -> str:
        return "mcap-v2"

    @property
    def payload_format_version(self) -> int:
        return 2

    def can_handle(self, data, config=None) -> bool:
        return isinstance(data, McapRecord)

    def get_file_extension(self, config) -> str:
        return ".mcap"

    def put(self, data, file_path: Path, config):
        actual_path = file_path.with_suffix(".mcap")
        actual_path.write_bytes(data.payload)
        return {
            "actual_path": str(actual_path),
            "file_size": actual_path.stat().st_size,
            "payload_format": self.payload_format,
            "payload_format_version": self.payload_format_version,
        }

    def get(self, file_path: Path, metadata):
        return McapRecord(file_path.read_bytes())


store.handlers.register_handler(McapFormatHandler(), priority=0)
```

Source pattern after applying the locked rename: current path-based methods and result keys are defined in the existing interface, while store-local registration and `.mcap` containment behavior already pass contract tests. [VERIFIED: `src/cacheness/interfaces.py:38-112,151-210`; `src/cacheness/handlers.py:1522-1572`; `tests/contracts/test_obstore_generation_io.py:301-330`]

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Cache-first `cacheness()` facade and implicit decorator configuration | Explicit `BlobStore`, `UnifiedCache(config, store=...)`, and `cached(cache=...)` | Delivered by Phases 3/6 before this research | Phase 9 must publish the current constructor/result surface rather than maintain docs compatibility. [VERIFIED: `src/cacheness/__init__.py:1-49`; `src/cacheness/core.py:124-188`; `src/cacheness/decorators.py:52-105`] |
| Backend-specific/manual payload mechanics | Unified obstore payload participant behind guarded handler paths | Phase 07.1 | Examples and handlers never receive obstore objects or managed locators. [VERIFIED: `src/cacheness/storage/obstore_generation_io.py:65-113`; `src/cacheness/storage/guarded_handler_io.py:166-220`] |
| Global/cache-named format extension guidance | Store-local `FormatHandler` registration | Phase 9 cutover | Clarifies generic storage role without changing registry shape or persisted identities. [VERIFIED locked D-14-D-17: `09-CONTEXT.md:44-48`] |
| Broad narrative/example inventory | Four exact disposable examples plus task-first guides | Phase 9 | CI and docs share one executable truth. [VERIFIED locked D-01-D-13: `09-CONTEXT.md:25-42`] |
| Implied remote/release readiness | Local-ready development version plus explicit NOT_QUALIFIED/NOT_PUBLISHED classes | Phase 8/9 | Local use is supported from a checkout without claiming real S3/PostgreSQL, controlled performance, Windows, or publication. [VERIFIED: `docs/RELEASE_QUALIFICATION.md:1-38,58-92`; `08-LOCAL-READINESS.json:1`] |

**Deprecated/outdated:**
- `from cacheness import cacheness`, implicit `@cached(ttl_hours=...)`, and global handler registration are removed pre-cutover APIs and should disappear from supported docs/examples. [VERIFIED: `.planning/v1.0-v1.0-MILESTONE-AUDIT.md:219-225`; current public exports `src/cacheness/__init__.py:32-49`]
- `CacheHandler`/`CacheHandlerError` become absent public names, not aliases. [VERIFIED locked D-15: `09-CONTEXT.md:44-48`]
- Phase 8 language that remote qualification “will” complete in Phase 8 is stale; SEED-007 owns it and the current state is NOT_QUALIFIED/NOT_PUBLISHED. [VERIFIED: `docs/CATALOG_AND_TOPOLOGY.md:41-50`; `docs/RELEASE_QUALIFICATION.md:94-106`]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| — | None. Recommendations are derived from locked Phase 9 decisions, opened source, tests, workflows, and checked-in evidence. | — | — |

## Open Questions

None blocking. The two apparent scope ambiguities are resolved by existing decisions:

1. `FormatHandler` should remain public from the storage-oriented barrels where `CacheHandler` is currently public; Phase 9 should not broaden the top-level `cacheness` barrel merely because stale docs incorrectly imported from it. [VERIFIED: `src/cacheness/__init__.py:12-49`; `src/cacheness/storage/__init__.py:26-50,257-261`]
2. SqlCache-dedicated implementation/assets remain physically owned by Phase 10; Phase 9 removes them from the promoted story and canonical example allowlist. [VERIFIED: `.planning/ROADMAP.md:635-646`; `09-CONTEXT.md:125-130`]

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `uv` | checkout install, tests, wheel build | ✓ | `0.12.12` | none needed [VERIFIED: local environment probe 2026-09-16] |
| Python 3.11 | supported-floor validation | ✓ | `3.11.16` | CI stable matrix [VERIFIED: local environment probe 2026-09-16; `.github/workflows/quality.yml:19-47`] |
| Python 3.12 | supported-version / packaging row | ✓ | `3.12.1` | CI stable matrix [VERIFIED: local environment probe 2026-09-16; `.github/workflows/quality.yml:49-78`] |
| Python 3.13 | primary local development/CI row | ✓ | `3.13.15` | none needed [VERIFIED: local environment probe 2026-09-16; `.github/workflows/quality.yml:43-47`] |
| Python 3.14 | supported boundary | ✓ | `3.14.7` | CI stable matrix [VERIFIED: local environment probe 2026-09-16; `.github/workflows/quality.yml:19-47,80-103`] |
| PostgreSQL / S3 | not required by Phase 9 | intentionally not probed | — | keep NOT_QUALIFIED reference only [VERIFIED locked D-08: `09-CONTEXT.md:32-36`] |

**Missing dependencies with no fallback:** none for Phase 9.

**Missing dependencies with fallback:** none; remote services are out of scope rather than missing prerequisites.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest `8.4.1` [VERIFIED: `uv.lock:2142-2152`] |
| Config file | `pyproject.toml` with strict markers and `tests/test_*.py` discovery [VERIFIED: `pyproject.toml:82-102`] |
| Quick run command | `uv run pytest -q -o log_cli=false tests/test_phase9_examples.py tests/test_interfaces.py tests/test_handler_registration.py tests/test_guarded_handler_io.py tests/test_public_api_contract.py tests/test_security_documentation.py -x` |
| Full non-live suite command | `uv run pytest -q -o log_cli=false -m "not (live_postgresql or live_aws_s3 or live_remote)" -x` [VERIFIED marker names: `pyproject.toml:94-102`] |

The current reusable baseline passed during research: `uv run pytest -q -o log_cli=false tests/test_phase6_examples.py tests/test_handler_registration.py tests/test_guarded_handler_io.py` (14 tests). [VERIFIED: local pytest execution 2026-09-16]

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CACH-06 | New `FormatHandler` public imports work and old names/aliases are absent | unit + isolated wheel | `uv run pytest -q tests/test_interfaces.py tests/test_public_api_contract.py tests/packaging/test_wheel_matrix.py -x` | Existing files need Phase 9 edits |
| CACH-06 | Persisted handler/data/payload/version identities survive rename and durable reopen | integration | `uv run pytest -q tests/test_stored_compatibility.py tests/test_handler_registration.py -x` | Existing files; add one rename-specific assertion |
| CACH-06 | Exactly four canonical files run unchanged, twice, network-free, self-verify, and clean up | subprocess integration | `uv run pytest -q tests/test_phase9_examples.py -x` | ❌ Wave 0 rename/replacement of current Phase 6 harness |
| CACH-06 | Minimal source-free wheel import is quiet without dataframe extras; dataframe extra still round-trips | packaging | `uv run pytest -q tests/packaging/test_wheel_matrix.py -x` | Existing file needs silence/rename assertions |
| CACH-06 | README/task/reference/security/guarantee surfaces contain current APIs and bounded claims only | documentation contract | `uv run pytest -q tests/test_public_api_contract.py tests/test_security_documentation.py tests/test_phase9_documentation.py -x` | ❌ Wave 0 new focused adoption-doc test |
| CACH-06 | Exact canonical examples run in CI | workflow contract | `uv run pytest -q tests/test_phase9_quality_workflow.py -x` | ❌ Wave 0 new workflow assertion or focused extension of existing workflow test |
| CACH-06 | Phase 3/8 metadata refresh uses existing evidence and preserves nonclaims | artifact contract | `uv run pytest -q tests/test_phase9_evidence_metadata.py -x` | ❌ Wave 0 bounded artifact test |

### Sampling Rate

- **Per rename task commit:** handler/interface/public API/stored compatibility focused tests plus targeted Ruff.
- **Per example/docs task commit:** `tests/test_phase9_examples.py` plus focused documentation/security tests.
- **Per wave merge:** packaging wheel matrix and the existing Phase 07.1 fixed verifier because docs and export literals are part of inherited contract gates. [VERIFIED: `tools/verify_phase071_contracts.py:665-719,722-801`]
- **Phase gate:** full non-live suite green, targeted Ruff green, exact four examples green, source-free wheel matrix green, then run the existing clean-source local-ready verifier from the final committed revision and write any fresh output outside tracked evidence unless the workflow explicitly intends a new evidence artifact. [VERIFIED clean-source behavior: `tools/run_phase8_local_gates.py:107-165,225-257`; existing local-ready command: `docs/RELEASE_QUALIFICATION.md:58-92`]

### Wave 0 Gaps

- [ ] `tests/test_phase9_examples.py` — replace/rename Phase 6 harness with literal four-file execution and residue checks.
- [ ] `tests/test_phase9_documentation.py` — current imports, task navigation, one guarantees owner, install posture, bounded claims, and no promoted old API/SqlCache.
- [ ] `tests/test_phase9_quality_workflow.py` — prove CI invokes the exact example harness on a stable local row.
- [ ] `tests/test_phase9_evidence_metadata.py` — constrain Phase 3/8 metadata refresh to checked-in evidence/nonclaims.
- [ ] Extend `tests/packaging/test_wheel_matrix.py` — new/old public names and silent minimal import.
- [ ] Extend `tests/test_stored_compatibility.py` — identity-preserving durable reopen across the source rename.

### Recommended Phase-Gate Commands

```bash
uv run pytest -q -o log_cli=false \
  tests/test_phase9_examples.py \
  tests/test_phase9_documentation.py \
  tests/test_phase9_quality_workflow.py \
  tests/test_phase9_evidence_metadata.py \
  tests/test_interfaces.py \
  tests/test_error_handling.py \
  tests/test_handlers.py \
  tests/test_handler_registration.py \
  tests/test_guarded_handler_io.py \
  tests/test_stored_compatibility.py \
  tests/test_public_api_contract.py \
  tests/test_security_documentation.py \
  tests/packaging/test_wheel_matrix.py -x

uv run ruff check \
  src/cacheness/interfaces.py src/cacheness/handlers.py \
  src/cacheness/error_handling.py src/cacheness/storage/__init__.py \
  src/cacheness/storage/handlers/__init__.py \
  examples tests/test_phase9_examples.py tests/test_phase9_documentation.py \
  tools/run_phase8_packaging.py

uv run pytest -q -o log_cli=false \
  -m "not (live_postgresql or live_aws_s3 or live_remote)" -x
```

These commands use existing declared tools/markers; the final plan may split them by task but should not weaken them. [VERIFIED: `pyproject.toml:68-102,132-147`]

## Security Domain

Security enforcement is enabled at ASVS level 1. [VERIFIED: `.planning/config.json:20-49`]

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | The library has no authentication boundary in Phase 9; examples are local/in-process. [VERIFIED: `AGENTS.md:105-109`] |
| V3 Session Management | no | No web/session runtime exists. [VERIFIED: `AGENTS.md:105-109`] |
| V4 Access Control | limited | Remote IAM/bucket ownership is documentation-only and NOT_QUALIFIED; examples must not use credentials or remote services. [VERIFIED: `docs/RELEASE_QUALIFICATION.md:94-106`; locked D-08] |
| V5 Input Validation | yes | Keep exact manifest/catalog parsing, handler suffix validation, path containment, and regular-file checks; tutorial code may only receive private paths. [VERIFIED: `src/cacheness/storage/manifest.py:198-220,303-321`; `src/cacheness/storage/guarded_handler_io.py:270-339`] |
| V6 Cryptography | yes, unchanged | Preserve existing SHA-256 payload digest and HMAC manifest semantics; Phase 9 does not change algorithms or key management. [VERIFIED: `src/cacheness/storage/manifest.py:173-220`; `.planning/v1.0-v1.0-MILESTONE-AUDIT.md:250-263`] |

### Known Threat Patterns for This Phase

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Custom handler escapes its private staging path or returns a symlink/directory | Tampering | Existing guarded staging rejects escapes, symlinks, non-regular files, and inode substitution; reuse the contract tests. [VERIFIED: `src/cacheness/storage/guarded_handler_io.py:270-339`; `tests/test_guarded_handler_io.py:13-78`] |
| Tutorial implies pickle/dill is safe for hostile input | Elevation of privilege | Preserve the tested trusted-application-payload statement and explicit arbitrary-code warning. [VERIFIED: `tests/test_security_documentation.py:20-33,66-75`] |
| Docs imply ETag/transport evidence replaces canonical digest | Tampering | Keep SHA-256 canonical; transport evidence is corroborating only. [VERIFIED: `src/cacheness/storage/blob_store.py:419-485`; `tools/verify_phase071_contracts.py:665-719`] |
| Example writes credentials or shared state | Information disclosure / Tampering | Use temporary local roots, memory/local qualified topologies, blocked sockets, and no environment-derived credentials. [VERIFIED pattern: `tests/test_phase6_examples.py:18-51`; locked D-08/D-13] |
| Alias or implicit migration accepts an unintended format | Tampering | No compatibility alias; exact format/version resolution and explicit offline migration/rebuild remain. [VERIFIED: `src/cacheness/handlers.py:1446-1520`; `.planning/REQUIREMENTS.md:52-60`] |

## Sources

### Primary (HIGH confidence)

- Phase 9 locked context and roadmap — scope, examples, docs, rename, nonclaims. [VERIFIED: `.planning/phases/09-adoption-and-release-surface-closure/09-CONTEXT.md`; `.planning/ROADMAP.md:615-646`]
- CACH-06 and migration/security requirements. [VERIFIED: `.planning/REQUIREMENTS.md:31-69`]
- Milestone audit — stale adoption paths, evidence refresh limits, smallest closure. [VERIFIED: `.planning/v1.0-v1.0-MILESTONE-AUDIT.md:145-175,182-248,250-270`]
- ADR 0001 — one authority, topology-specific guarantees, stop conditions. [VERIFIED: `docs/adr/0001-topology-specific-storage-guarantees.md:30-68,83-115,139-209`]
- Live Python source and tests cited inline — public signatures, persistence fields, handler registry, guarded staging, packaging, and example harness.
- Local environment and test probes on 2026-09-16 — Python/uv availability, 14 focused tests passing, and reproduced minimal-import dataframe warning.

### Secondary (MEDIUM confidence)

- None. No external web/documentation claims were required; research-plan provider configuration exposed no enabled external research providers. [VERIFIED: `.planning/config.json:1-12`; research-plan seam returned zero items]

### Tertiary (LOW confidence)

- None.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — existing manifest/lock/workflow and environment probes.
- Architecture: HIGH — live source plus accepted ADR.
- Rename scope: HIGH — scoped active-tree inventory plus direct source inspection.
- Documentation/examples: HIGH — complete inventories, opened canonical files, locked decisions, and reusable tests.
- Pitfalls: HIGH — each maps to a current failing/stale surface or explicit phase boundary.

**Research date:** 2026-09-16
**Valid until:** 2026-10-16 (stable codebase-specific planning guidance; refresh if Phase 10 starts or public barrels change first)
