# Cacheness

## What This Is

Cacheness is a Python storage and caching library for arbitrary objects, arrays, dataframes, and function results. Its object-storage path uses a reliable `BlobStore` foundation that `UnifiedCache` consumes as its policy layer. Built-in filesystem, memory, and S3 payload mechanics now share one guarded obstore participant beneath the single lifecycle authority. Catalog customization, cache-policy coverage, and explicit offline migration/rebuild tooling are implemented; final production qualification remains in Phase 8. `SqlCache` remains separate.

The intended audience is Python applications that need local or remote persistence with predictable cache semantics across filesystem, memory, S3, JSON, SQLite, and PostgreSQL backends.

The primary product is a reliable blob store with application-customizable catalog metadata. Cache instances use `BlobStore` as their engine, not merely its serializers or low-level utilities. Cache and non-cache instances may own separate namespaces/catalogs; no instance must simultaneously serve both roles. See `CONTEXT.md` for vocabulary and ADR 0001 for topology-specific guarantees.

## Core Value

Applications can store and retrieve data reliably through one backend-neutral lifecycle, with caching policy layered above storage without compromising integrity or cleanup correctness.

## Requirements

### Validated

- ✓ Users can cache and retrieve arbitrary Python objects through `UnifiedCache` and the `cacheness` public alias — existing
- ✓ Users can cache NumPy arrays and pandas/polars dataframes through type-specific handlers and compressed formats — existing
- ✓ Users can cache function results through decorator APIs with deterministic parameter-based keys — existing
- ✓ Users can persist metadata with JSON, memory, or SQLite implementations and access an optional PostgreSQL implementation — existing
- ✓ Users can access filesystem, memory, and optional S3 blob backend implementations through backend registries — existing
- ✓ Users can use `BlobStore` directly for object storage without TTL or eviction semantics — existing
- ✓ Users can use `SqlCache` separately for SQLAlchemy-backed pull-through dataframe caching — existing
- ✓ Users can configure TTL, compression, integrity hashing, HMAC entry signing, handler selection, and cache statistics — existing
- ✓ Initialized local SQLite/filesystem storage preserves complete generations through exact publication, attributable cleanup debt and resumable recovery — Phase 3 direct qualification at `5282dca`
- ✓ Canonical local cache operations consume BlobStore entry snapshots/receipts; optional projection failure cannot revoke a valid blob, and requested external metadata failures report committed partial outcomes — Phase 3
- ✓ Direct application mapping metadata supports inspection/filtering/update, and separate object/cache namespaces preserve independent retention policy — Phase 3; richer customization remains Phase 4
- ✓ Application-defined catalog metadata is validated, queried, and updated through direct `BlobStore` operations without another lifecycle backend — Phase 4
- ✓ Remaining cache policy, decorator, and statistics behavior runs above the canonical `BlobStore` engine — Phase 6
- ✓ The pre-production `BlobStore`/cache surface has explicit versioned migration and confirmed rebuild tooling; ordinary opens never perform implicit upgrades — Phases 4, 6, and 7
- ✓ Persisted metadata and paths use safe parsing, containment, signing, and fail-closed integrity boundaries — Phases 1 through 5
- ✓ Built-in filesystem, memory, and S3 payload mechanics use one guarded obstore participant while `BlobStore` retains sole lifecycle authority — Phase 07.1; live-service release qualification remains Phase 8
- ✓ Opaque S3 ETag/version evidence can be cataloged and compared without replacing canonical SHA-256/size verification — Phase 07.1

### Active

- [ ] Release-qualify `BlobStore` lifecycle behavior across every advertised built-in backend and supported environment
- [ ] Publish complete generations through one catalog transaction; coordinate external payload effects with attributable intent/debt and deterministic reconciliation when resources are available, not cross-resource ACID or instantaneous orphan-free cleanup
- [ ] Guarantee same-key integrity and defined success/conflict/retryable-failure outcomes within each explicitly supported backend topology
- [ ] Correct package dependency and optional-feature detection so a minimal supported installation imports reliably
- [ ] Establish CI, backend contract tests, lint policy, coverage thresholds, and supported-Python/backend matrices
- [ ] Establish performance benchmarks and final regression budgets while prioritizing correctness during migration

### Out of Scope

- Redesigning or merging `SqlCache` into the object/blob storage architecture — it remains a separate subsystem for this cycle
- Building a general cache-policy plugin framework now — the layered design must leave a clean seam for that later evolution
- Supporting hostile or untrusted pickle/dill payloads — application data is trusted; boundaries, metadata, paths, and parsers are still hardened
- Defending a live local store against its owning OS principal deleting or rebinding Cacheness lifecycle-control objects — control objects are validated and fail closed when tampering is detected, but the store owner is part of the trusted deployment boundary
- Sharing one Windows local store across different users, services, or interactive sessions in this milestone — initial Windows lifecycle coordination is scoped to one OS user and session
- Runtime compatibility adapters for pre-production APIs or development-only stored layouts — current callers/data may break during the cleanup; unsupported stores fail with explicit migration/rebuild-required evidence
- Adding new storage or metadata backend families — this cycle makes every already-advertised built-in backend work end to end
- Requiring one live store to serve durable-storage and cache retention roles simultaneously — shared implementation with separate instances/namespaces meets the product goal
- Universal success during concurrent first creation or online schema migration — initialization before shared workers and maintenance-only schema changes were approved and implemented in Phase 3

## Context

The codebase began as a disk cache and expanded into direct blob storage, backend registries, remote S3/PostgreSQL support, custom metadata, and a separate SQL pull-through cache. The expansion left three adjacent products (`UnifiedCache`, `BlobStore`, and `SqlCache`) plus blob backend implementations that are not composed into one lifecycle.

At project initialization, payload writes and metadata writes were separate, without rollback; cleanup and composition were incomplete. Phase 3 now has a SQLite lifecycle authority, immutable native generations, exact publication, and intent/debt recovery. The direct implementation removed the cache's projection-repair/deferred-cleanup orchestration and added supported same-generation entry snapshots and receipts. See [the implementation ledger](../docs/phase3-direct-implementation-2026-09-06.md) and [initialization/failure guide](../docs/STORAGE_INITIALIZATION.md), not the earlier audit alone, for the delivered baseline.

Phases 4 through 7 delivered injected catalog composition, cache-policy reuse of `BlobStore`, and explicit stopped-worker migration/rebuild tooling. Phase 07.1 replaced the separate filesystem/memory/S3 payload mechanics with one guarded obstore participant while preserving the single lifecycle authority and path-based custom-handler seam. Phase 8 owns the remaining release qualification: minimal packaging, supported Python/platform matrices, real PostgreSQL and AWS S3 evidence, coverage policy, and measured performance. Deterministic adapters or standalone backend implementations are not substitutes for those live release claims.

The security model assumes trusted application payloads and a trusted owner for each local store, so pickle/dill remain available and Cacheness lifecycle-control objects must not be deleted or rebound by that owner while the store is live. Persisted metadata and paths are still untrusted inputs: the implementation must eliminate `eval`, contain filesystem paths, bind structured query paths safely, and fail closed when signing, integrity verification, or control-object identity checks fail. Initial Windows local-store coordination supports processes running as one OS user in one interactive or service session; cross-user, cross-service, and cross-session sharing requires a future explicit authority and ACL contract.

Historical baseline (2026-08-29): 777 collected tests, 749 passing, 26 skipped and 2 YAML path-round-trip failures; statement coverage 66% and 137 repository-wide Ruff findings. These are not current measurements. Phase 3 qualification at `5282dca` passed 1,453 tests with 26 skips on Python 3.11, 127 focused tests on Python 3.13, named gates, scoped Ruff, the Phase 3 lint delta and the unchanged lifecycle benchmark. Full coverage, packaging, supported-Python/service CI and release performance acceptance remain Phase 8. Native Windows remains unqualified.

## Constraints

- **Pre-production cutover**: Cacheness is not yet in production, so the milestone may replace current public APIs and development-only stored layouts instead of carrying runtime compatibility adapters. Preserve only deliberately reaffirmed contracts; incompatible stores fail explicitly.
- **Migration**: Keep versioned schema/format identification plus explicit offline migration and rebuild tooling for future releases; dropping current backward compatibility does not authorize implicit upgrade, silent deletion, or removal of migration infrastructure.
- **Architecture**: `BlobStore` owns storage lifecycle; `UnifiedCache` depends on it and owns cache policy; `SqlCache` remains separate
- **Backends**: Cover all advertised backend families through explicit supported combinations and capability tiers; a backend-neutral interface does not imply identical durability/progress or all Cartesian pairings
- **Security**: Treat application payloads as trusted while enforcing safe parsing, path containment, and fail-closed integrity boundaries
- **Local-store trust**: Treat the OS principal that owns a local store and its lifecycle-control namespace as trusted not to delete or rebind live control objects; detect observable substitution and fail closed
- **Windows sharing**: Support local-store coordination within one Windows OS user/session in this milestone; do not claim cross-user, cross-service, or cross-session authority
- **Reliability**: The selected transactional catalog owns descriptor, authoritative user metadata, intent, and cleanup debt; external payload effects use deterministic reconciliation, not cross-resource ACID
- **Concurrency**: Same-key operations must not corrupt payloads or produce metadata/payload disagreement
- **Startup and maintenance**: Initialize before shared workers; schema migration/cutover requires stopped workers and explicit maintenance, not an online startup protocol
- **Derived state**: Optional projections never gate canonical reads/cleanup or revoke commits; explicitly requested external metadata failures retain typed committed-partial outcomes, including concurrent cache close after the engine commit
- **Performance**: Correctness comes first during migration; final acceptance includes measured budgets against checked-in benchmarks
- **Runtime**: Maintain Python 3.11+ support and verify supported versions rather than relying only on the current Python 3.13 environment

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Start with a layered `BlobStore` core and `UnifiedCache` policy layer | Provides the cleanest incremental repair while preserving a future path to pluggable cache policies | ✓ Local engine integration qualified in Phase 3; remaining policy/backend coverage stays scoped downstream |
| Defer the policy-plugin framework | Avoids over-engineering before lifecycle contracts and backend behavior are reliable | — Pending |
| Keep `SqlCache` separate | Its row/table pull-through model has different query and lifecycle semantics | — Pending |
| Use a pre-production compatibility reset while retaining migration tooling | No production deployment depends on the current API/layout, so a clean composition/catalog contract is cheaper and safer than maintaining parallel legacy paths; future releases still need explicit migrations | ✓ User approved, 2026-09-07 |
| Target trusted application payloads | Retains useful pickle/dill capabilities while focusing security work on boundaries the library can enforce | — Pending |
| Trust the local store owner for lifecycle-control availability | Portable per-key coordination cannot remain immutable against the same principal deleting every authority object without an external coordinator or store-wide serialization | ✓ Phase 3 contract clarification |
| Scope initial Windows local-store sharing to one user/session | Preserves advertised Windows use for ordinary applications without claiming an unverified cross-principal authority or ACL model | ✓ Phase 3 contract clarification |
| Include every advertised built-in backend | Backend unification is not credible if S3 or PostgreSQL remain direct-use-only or untested | — Pending |
| Prioritize correctness with measured performance guardrails | Allows safe architectural migration while preventing an unbounded final performance regression | — Pending |
| Cache instances consume BlobStore; separate cache/non-cache namespaces are sufficient | Reuse the complete engine without imposing shared retention policy or a dual-role store | ✓ User clarification, 2026-09-06 |
| Customize catalog metadata without replacing the lifecycle implementation | Cataloging blobs is a primary product use case, not a cache-only extension | ✓ User goal; BACK-07 acceptance added for Phase 4 |
| Prove the local cache/storage composition before multiplying adapters | A thin early integration exposes duplicate authority responsibilities while changes remain local | ✓ Delivered in Phase 3; complete policy coverage remains Phase 6 |
| Initialize before shared workers; keep migration explicit | Removes concurrent lazy bootstrap as a required availability protocol | ✓ User approved and implemented in Phase 3; offline migration tooling remains Phase 7 |
| Keep optional exports separate from canonical commit | A second catalog cannot become a synchronous authority or gate cleanup | ✓ Phase 3 warnings/committed-partial outcomes; preserve across future adapters and cache policies |
| Require explicit stopped-worker migration and rebuild | Ordinary constructors must validate rather than silently mutate; maintenance uses bounded authenticated evidence and exact authority-owned recovery | ✓ Phase 7 |
| Support current plus immediately previous released layouts | Bounds migration support while allowing older released stores to advance through declared steps; pre-production layouts need not be fabricated | ✓ Phase 7 |
| Keep handler-owned transforms and explicit rebuild separate from lifecycle authority | Format handlers own serialization changes while the catalog remains the sole visibility and recovery authority | ✓ Phase 7 |
| Use obstore for built-in payload object mechanics only | Removes duplicate filesystem/S3 mechanics while preserving `BlobStore`/`AuthorityLifecycleEngine` as the sole owner of intent, visibility, reconciliation, and cleanup debt | ✓ Phase 07.1 |
| Treat ETag/version as opaque signed transport evidence | Enables server-side corroboration without pretending ETag is the canonical blob digest or extending database ACID across the object store | ✓ Phase 07.1 |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `$gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `$gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-09-13 after Phase 07.1 completion*
