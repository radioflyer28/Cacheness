# Cacheness

## What This Is

Cacheness is a Python storage and caching library for arbitrary objects, arrays, dataframes, and function results. It currently exposes overlapping cache, blob-storage, backend-registry, and SQL pull-through systems; this project will converge the object-storage path around a reliable `BlobStore` foundation that `UnifiedCache` uses as its policy layer.

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

### Active

- [ ] Make `BlobStore` the canonical owner of payload and metadata lifecycle across every advertised built-in backend
- [ ] Refactor `UnifiedCache` to compose `BlobStore` and own only cache policy: keying, TTL, eviction, invalidation, and statistics
- [ ] Expose application-defined catalog metadata through direct `BlobStore` validation, query, and update operations without requiring a new lifecycle backend
- [ ] Publish complete generations through one catalog transaction; coordinate external payload effects with attributable intent/debt and deterministic reconciliation when resources are available, not cross-resource ACID or instantaneous orphan-free cleanup
- [ ] Guarantee same-key integrity and defined success/conflict/retryable-failure outcomes within each explicitly supported backend topology
- [ ] Preserve public APIs while providing an explicit migration or rebuild path for incompatible stored entries
- [ ] Remove unsafe parsing, enforce filesystem containment, and make required signing and integrity verification fail closed
- [ ] Correct package dependency and optional-feature detection so a minimal supported installation imports reliably
- [ ] Establish CI, backend contract tests, lint policy, coverage thresholds, and supported-Python/backend matrices
- [ ] Establish performance benchmarks and final regression budgets while prioritizing correctness during migration

### Out of Scope

- Redesigning or merging `SqlCache` into the object/blob storage architecture — it remains a separate subsystem for this cycle
- Building a general cache-policy plugin framework now — the layered design must leave a clean seam for that later evolution
- Supporting hostile or untrusted pickle/dill payloads — application data is trusted; boundaries, metadata, paths, and parsers are still hardened
- Defending a live local store against its owning OS principal deleting or rebinding Cacheness lifecycle-control objects — control objects are validated and fail closed when tampering is detected, but the store owner is part of the trusted deployment boundary
- Sharing one Windows local store across different users, services, or interactive sessions in this milestone — initial Windows lifecycle coordination is scoped to one OS user and session
- Breaking public APIs without compatibility adapters — stored data may require an explicit migration or documented rebuild
- Adding new storage or metadata backend families — this cycle makes every already-advertised built-in backend work end to end
- Requiring one live store to serve durable-storage and cache retention roles simultaneously — shared implementation with separate instances/namespaces meets the product goal
- Universal success during concurrent first creation or online schema migration — the proposed Phase 3 initialization-before-workers contract has an explicit implementation approval checkpoint

## Context

The codebase began as a disk cache and expanded into direct blob storage, backend registries, remote S3/PostgreSQL support, custom metadata, and a separate SQL pull-through cache. The expansion left three adjacent products (`UnifiedCache`, `BlobStore`, and `SqlCache`) plus blob backend implementations that are not composed into one lifecycle.

At project initialization, payload writes and metadata writes were separate, without rollback; cleanup and composition were incomplete. Phase 3 now has an SQLite lifecycle authority, immutable native generations, exact publication, and intent/debt recovery. The remaining architectural problem is the cache's synchronous compatibility catalog coordination around that engine. The 2026-09-06 architecture audit and gap plans 03-21 through 03-25 address that transition; these plans are not yet implementation evidence.

Backend extensibility is partially disconnected: custom metadata backend registration is not consulted by `UnifiedCache`, injected metadata backend instances are overwritten by config selection, and configured blob backends do not route `UnifiedCache` or `BlobStore` writes through filesystem, memory, or S3 implementations.

The security model assumes trusted application payloads and a trusted owner for each local store, so pickle/dill remain available and Cacheness lifecycle-control objects must not be deleted or rebound by that owner while the store is live. Persisted metadata and paths are still untrusted inputs: the implementation must eliminate `eval`, contain filesystem paths, bind structured query paths safely, and fail closed when signing, integrity verification, or control-object identity checks fail. Initial Windows local-store coordination supports processes running as one OS user in one interactive or service session; cross-user, cross-service, and cross-session sharing requires a future explicit authority and ACL contract.

The independent baseline on 2026-08-29 is 777 collected tests with 749 passing, 26 skipped, and 2 YAML path-round-trip failures. Statement coverage is 66%, with `BlobStore` at 19%, PostgreSQL metadata at 30%, and `SqlCache` at 49%. Ruff reports 137 repository-wide findings, and no CI workflow enforces tests, linting, coverage, minimal-install imports, or optional backend matrices.

## Constraints

- **Compatibility**: Preserve supported public APIs; allow stored-data migration or rebuild only through an explicit, documented path
- **Architecture**: `BlobStore` owns storage lifecycle; `UnifiedCache` depends on it and owns cache policy; `SqlCache` remains separate
- **Backends**: Cover all advertised backend families through explicit supported combinations and capability tiers; a backend-neutral interface does not imply identical durability/progress or all Cartesian pairings
- **Security**: Treat application payloads as trusted while enforcing safe parsing, path containment, and fail-closed integrity boundaries
- **Local-store trust**: Treat the OS principal that owns a local store and its lifecycle-control namespace as trusted not to delete or rebind live control objects; detect observable substitution and fail closed
- **Windows sharing**: Support local-store coordination within one Windows OS user/session in this milestone; do not claim cross-user, cross-service, or cross-session authority
- **Reliability**: The selected transactional catalog owns descriptor, authoritative user metadata, intent, and cleanup debt; external payload effects use deterministic reconciliation, not cross-resource ACID
- **Concurrency**: Same-key operations must not corrupt payloads or produce metadata/payload disagreement
- **Performance**: Correctness comes first during migration; final acceptance includes measured budgets against checked-in benchmarks
- **Runtime**: Maintain Python 3.11+ support and verify supported versions rather than relying only on the current Python 3.13 environment

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Start with a layered `BlobStore` core and `UnifiedCache` policy layer | Provides the cleanest incremental repair while preserving a future path to pluggable cache policies | — Pending |
| Defer the policy-plugin framework | Avoids over-engineering before lifecycle contracts and backend behavior are reliable | — Pending |
| Keep `SqlCache` separate | Its row/table pull-through model has different query and lifecycle semantics | — Pending |
| Preserve public APIs but permit stored-data migration | Protects callers without forcing the new architecture to preserve flawed on-disk representations forever | — Pending |
| Target trusted application payloads | Retains useful pickle/dill capabilities while focusing security work on boundaries the library can enforce | — Pending |
| Trust the local store owner for lifecycle-control availability | Portable per-key coordination cannot remain immutable against the same principal deleting every authority object without an external coordinator or store-wide serialization | ✓ Phase 3 contract clarification |
| Scope initial Windows local-store sharing to one user/session | Preserves advertised Windows use for ordinary applications without claiming an unverified cross-principal authority or ACL model | ✓ Phase 3 contract clarification |
| Include every advertised built-in backend | Backend unification is not credible if S3 or PostgreSQL remain direct-use-only or untested | — Pending |
| Prioritize correctness with measured performance guardrails | Allows safe architectural migration while preventing an unbounded final performance regression | — Pending |
| Cache instances consume BlobStore; separate cache/non-cache namespaces are sufficient | Reuse the complete engine without imposing shared retention policy or a dual-role store | ✓ User clarification, 2026-09-06 |
| Customize catalog metadata without replacing the lifecycle implementation | Cataloging blobs is a primary product use case, not a cache-only extension | ✓ User goal; BACK-07 acceptance added for Phase 4 |
| Prove the local cache/storage composition before multiplying adapters | A thin early integration exposes duplicate authority responsibilities while changes remain local | Planned in 03-23/24; complete policy coverage remains Phase 6 |
| Initialize before shared workers; keep migration explicit | Removes concurrent lazy bootstrap as a required availability protocol | Proposed in 03-22; approval required before compatibility changes |

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
*Last updated: 2026-09-06 after architecture-audit gap replanning; implementation remains pending*
