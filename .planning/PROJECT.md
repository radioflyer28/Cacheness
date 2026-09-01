# Cacheness

## What This Is

Cacheness is a Python storage and caching library for arbitrary objects, arrays, dataframes, and function results. It currently exposes overlapping cache, blob-storage, backend-registry, and SQL pull-through systems; this project will converge the object-storage path around a reliable `BlobStore` foundation that `UnifiedCache` uses as its policy layer.

The intended audience is Python applications that need local or remote persistence with predictable cache semantics across filesystem, memory, S3, JSON, SQLite, and PostgreSQL backends.

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
- [ ] Guarantee atomic writes, rollback or reconciliation after partial failure, correct deletion, and orphan-free cleanup
- [ ] Guarantee safe same-key concurrency and deterministic lifecycle behavior across filesystem, memory, S3, JSON, SQLite, and PostgreSQL
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

## Context

The codebase began as a disk cache and expanded into direct blob storage, backend registries, remote S3/PostgreSQL support, custom metadata, and a separate SQL pull-through cache. The expansion left three adjacent products (`UnifiedCache`, `BlobStore`, and `SqlCache`) plus blob backend implementations that are not composed into one lifecycle.

Current payload writes occur through type handlers while metadata is written separately. There is no transaction or rollback spanning both operations. Invalidation, expiration, size cleanup, and `BlobStore.clear()` can remove metadata without removing payloads; size enforcement calls a backend method that is not implemented. `UnifiedCache` allocates a lock but does not use it around payload-plus-metadata operations.

Backend extensibility is partially disconnected: custom metadata backend registration is not consulted by `UnifiedCache`, injected metadata backend instances are overwritten by config selection, and configured blob backends do not route `UnifiedCache` or `BlobStore` writes through filesystem, memory, or S3 implementations.

The security model assumes trusted application payloads and a trusted owner for each local store, so pickle/dill remain available and Cacheness lifecycle-control objects must not be deleted or rebound by that owner while the store is live. Persisted metadata and paths are still untrusted inputs: the implementation must eliminate `eval`, contain filesystem paths, bind structured query paths safely, and fail closed when signing, integrity verification, or control-object identity checks fail. Initial Windows local-store coordination supports processes running as one OS user in one interactive or service session; cross-user, cross-service, and cross-session sharing requires a future explicit authority and ACL contract.

The independent baseline on 2026-08-29 is 777 collected tests with 749 passing, 26 skipped, and 2 YAML path-round-trip failures. Statement coverage is 66%, with `BlobStore` at 19%, PostgreSQL metadata at 30%, and `SqlCache` at 49%. Ruff reports 137 repository-wide findings, and no CI workflow enforces tests, linting, coverage, minimal-install imports, or optional backend matrices.

## Constraints

- **Compatibility**: Preserve supported public APIs; allow stored-data migration or rebuild only through an explicit, documented path
- **Architecture**: `BlobStore` owns storage lifecycle; `UnifiedCache` depends on it and owns cache policy; `SqlCache` remains separate
- **Backends**: The unified lifecycle must cover filesystem, memory, S3, JSON, SQLite, and PostgreSQL implementations
- **Security**: Treat application payloads as trusted while enforcing safe parsing, path containment, and fail-closed integrity boundaries
- **Local-store trust**: Treat the OS principal that owns a local store and its lifecycle-control namespace as trusted not to delete or rebind live control objects; detect observable substitution and fail closed
- **Windows sharing**: Support local-store coordination within one Windows OS user/session in this milestone; do not claim cross-user, cross-service, or cross-session authority
- **Reliability**: Payload and metadata operations must have atomic commit, rollback, or deterministic reconciliation semantics
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
*Last updated: 2026-08-29 after initialization*
