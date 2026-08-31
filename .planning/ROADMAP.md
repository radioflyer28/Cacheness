# Roadmap: Cacheness

## Overview

Cacheness will move from overlapping cache and storage paths to one production-grade object lifecycle in eight horizontal phases. The roadmap freezes compatibility and hardens exposed boundaries first, defines the canonical storage contract before any backend is allowed to shape it, proves atomicity and recovery in the lifecycle core, brings every metadata and payload backend behind that contract, then rewires `UnifiedCache` as policy over `BlobStore`. Explicit migration and release-quality gates complete the cutover without absorbing the separate `SqlCache` subsystem or prematurely introducing a cache-policy plugin framework.

## Phases

- [x] **Phase 1: Compatibility and Security Baseline** - Freeze supported behavior and close the currently exposed path, parser, query, and trust-boundary gaps.
- [x] **Phase 2: Canonical Storage and Integrity Contract** - Give direct `BlobStore` callers one versioned manifest, committed-read model, and fail-closed integrity contract. (completed 2026-08-30)
- [ ] **Phase 3: Atomic Lifecycle and Recovery Engine** - Make writes, replacements, deletions, races, and crash residue converge without corrupting the last valid generation.
- [ ] **Phase 4: Metadata Composition and Topology Contracts** - Unify all metadata backends behind one injectable CAS contract with honest capability validation.
- [ ] **Phase 5: Payload Backends and Full Matrix Parity** - Put filesystem, memory, and AWS S3 payloads through the complete lifecycle across every supported metadata pairing.
- [ ] **Phase 6: UnifiedCache Policy Composition** - Route cache behavior through `BlobStore` while preserving public APIs, cache semantics, and the `SqlCache` boundary.
- [ ] **Phase 7: Explicit Migration and Rebuild Cutover** - Give existing stores an inspectable, resumable same-backend migration or confirmed rebuild path.
- [ ] **Phase 8: Production Gates and Performance Stabilization** - Make packaging, service integration, fault testing, coverage, and measured scalability release requirements.

## Phase Details

### Phase 1: Compatibility and Security Baseline

**Goal**: Users have a frozen compatibility baseline and safe boundary behavior before lifecycle ownership changes.
**Depends on**: Nothing (first phase)
**Requirements**: MIGR-01, CACH-07, SECU-01, SECU-02, SECU-06, SECU-07
**Success Criteria** (what must be TRUE):

  1. Users can run the supported public imports, constructors, configuration names, registries, aliases, decorators, exceptions, and representative result behaviors against an executable compatibility baseline.
  2. Existing `SqlCache` imports and representative pull-through workflows continue to operate independently of the object-storage refactor.
  3. Filesystem operations reject traversal, absolute-path, drive, UNC, and symlink escapes for reads, writes, deletes, and listings.
  4. Structured metadata and query fields are parsed and constructed without metadata-controlled `eval` or interpolated backend query fragments.
  5. Users can identify the trusted-application-payload boundary and the risks and required configuration for unsafe serializers from project documentation.

**Plans**: 15/15 plans complete; review, security audit, and validation approved

Plans:
**Wave 1**

- [x] 01-01-PLAN.md — Freeze corrected public/error/configuration compatibility contracts.

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 01-02-PLAN.md — Enforce low-level filesystem containment and no-follow operations.
- [x] 01-06-PLAN.md — Make SqlCache strict-by-default and explicit about partial results.
- [x] 01-08-PLAN.md — Generate and verify both legacy raw-array compatibility variants.

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 01-03-PLAN.md — Carry containment through guarded high-level handler I/O and safe physical names.
- [x] 01-09-PLAN.md — Add unsigned and signed split-map JSON compatibility fixtures.

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 01-04-PLAN.md — Replace unsafe legacy array parsing and ordinary pickle-enabled array loading.
- [x] 01-05-PLAN.md — Validate and bind metadata query fields without changing documented semantics.
- [x] 01-10-PLAN.md — Add legacy SQLite and pre-unified decorator-key fixtures.

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 01-11-PLAN.md — Complete the corpus with current JSON and SQLite controls.

**Wave 6** *(blocked on Wave 5 completion)*

- [x] 01-12-PLAN.md — Implement exact production metadata/signature/decorator compatibility adapters.

**Wave 7** *(blocked on Wave 6 completion)*

- [x] 01-07-PLAN.md — Publish the serializer trust boundary and seal phase quality gates.

**Wave 8** *(gap closure; blocked on Wave 7 completion)*

- [x] 01-13-PLAN.md — Reopen validation and close staged-publication and numeric-query boundary gaps.

**Wave 9** *(gap closure; blocked on Wave 8 completion)*

- [x] 01-14-PLAN.md — Build truthful durable metadata and bounded BlobStore clear recovery.

**Wave 10** *(gap closure; blocked on Wave 9 completion)*

- [x] 01-15-PLAN.md — Close candidate and UnifiedCache clear gaps and stage renewed evidence.

### Phase 2: Canonical Storage and Integrity Contract

**Goal**: Direct `BlobStore` users interact with one versioned, backend-neutral record and deterministic fail-closed read contract.
**Depends on**: Phase 1
**Requirements**: STOR-01, STOR-02, STOR-08, SECU-03, SECU-04, SECU-05, SECU-08, MIGR-02, MIGR-07
**Success Criteria** (what must be TRUE):

  1. Every stored object exposes one canonical versioned manifest with independently versioned metadata and payload formats, regardless of backend.
  2. Normal reads expose only committed generations and distinguish missing, corrupt, conflict, and backend failures through typed `BlobStore` results or exceptions.
  3. Manifest authenticity and payload integrity are verified before deserialization, and required signing fails closed when its key, signature, permissions, or configuration is invalid.
  4. Signatures bind the critical locator, handler/type, format, digest, size, and lifecycle-generation fields rather than an incomplete metadata subset.
  5. Unknown future metadata or payload versions fail explicitly, while `UnifiedCache` has a defined seam for translating typed integrity failures into separately recorded cache misses later.

**Plans**: TBD

- [x] 02-01-PLAN.md
- [x] 02-02-PLAN.md
- [x] 02-03-PLAN.md
- [x] 02-04-PLAN.md
- [x] 02-05-PLAN.md
- [x] 02-06-PLAN.md
- [x] 02-07-PLAN.md

### Phase 3: Atomic Lifecycle and Recovery Engine

**Goal**: Object lifecycle operations preserve an old or new complete generation and leave every incomplete outcome recoverable.
**Depends on**: Phase 2
**Requirements**: STOR-03, STOR-04, STOR-05, STOR-06, STOR-07
**Success Criteria** (what must be TRUE):

  1. A write or overwrite exposes either the previous complete generation or the new complete generation, never mixed payload and metadata state.
  2. Failures at serialization, payload publication, metadata commit, or old-generation cleanup preserve the last valid generation and leave all residue detectable.
  3. Repeated overwrite, delete, clear, and close operations converge safely while cleaning both payload and metadata state.
  4. Operators can dry-run and resume reconciliation to repair, quarantine, or report inconsistent entries without guessing their provenance.
  5. Forced same-key write, delete, and read races have deterministic outcomes without globally serializing operations on distinct keys.

**Plans**: TBD
**Wave 1**

- [x] 03-01-PLAN.md

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 03-02-PLAN.md

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 03-03-PLAN.md

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 03-04-PLAN.md

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 03-05-PLAN.md

**Wave 6** *(blocked on Wave 5 completion)*

- [x] 03-06-PLAN.md

**Wave 7** *(blocked on Wave 6 completion)*

- [x] 03-07-PLAN.md

**Wave 8** *(blocked on Wave 7 completion)*

- [x] 03-08-PLAN.md

**Wave 9** *(blocked on Wave 8 completion)*

- [ ] 03-09-PLAN.md

**Wave 10** *(blocked on Wave 9 completion)*

- [ ] 03-10-PLAN.md

### Phase 4: Metadata Composition and Topology Contracts

**Goal**: Users can select any advertised metadata backend through one composition root and receive only guarantees that the chosen topology can provide.
**Depends on**: Phase 3
**Requirements**: BACK-02, BACK-03, BACK-06
**Success Criteria** (what must be TRUE):

  1. JSON, memory, SQLite, and PostgreSQL round-trip the same canonical entries and expose the same revision, conflict, tombstone, listing, and lifecycle semantics.
  2. A caller-injected backend instance remains the selected instance, and a registered backend name resolves through the same construction path used by direct and composed storage.
  3. Users can inspect durability, process/host sharing, compare-and-swap, streaming, and listing capabilities for the active backend pair.
  4. Invalid topology claims, such as durable multi-host storage backed by process-local payloads or unsupported coordination, fail during configuration rather than during a write.

**Plans**: TBD

### Phase 5: Payload Backends and Full Matrix Parity

**Goal**: Every advertised payload backend works end to end with every supported metadata backend under the canonical lifecycle.
**Depends on**: Phase 4
**Requirements**: BACK-01, BACK-04, BACK-05
**Success Criteria** (what must be TRUE):

  1. Filesystem, memory, and S3 payload backends pass the same immutable-generation, read, delete, list, integrity, resource-cleanup, and reconciliation contract.
  2. Every allowed combination in the three-payload by four-metadata matrix completes storage lifecycle workflows with no backend-specific behavior leaking to callers.
  3. PostgreSQL concurrency and transaction behavior is verified against a real PostgreSQL service, including conflicts and cleanup after partial failure.
  4. AWS S3 is the authoritative remote-object test target for conditional operations, checksums, streaming, pagination, retries, and cleanup; compatible services are claimed only where explicitly verified.

**Plans**: TBD

### Phase 6: UnifiedCache Policy Composition

**Goal**: Cache users retain their public workflows while all payload-plus-metadata lifecycle work is delegated to `BlobStore`.
**Depends on**: Phase 5
**Requirements**: CACH-01, CACH-02, CACH-03, CACH-04, CACH-05, CACH-06
**Success Criteria** (what must be TRUE):

  1. Existing cache imports, constructors, aliases, configuration names, decorators, and result behavior remain callable through compatibility adapters.
  2. Cache puts, reads, invalidations, and clears reach payload and metadata only through `BlobStore`; TTL, eviction, keying, statistics, and decorators remain policy owned by `UnifiedCache`.
  3. TTL expiry, size eviction, predicate invalidation, decorator clearing, single-key deletion, and global clear remove complete stored entries through one lifecycle primitive.
  4. Cached `None` values are returned as hits rather than recomputed as misses, and decorator clear reports the entries it actually removed.
  5. Statistics preserve compatible aggregate counters while separately identifying absent, expired, corrupt, conflict, and backend-error outcomes.

**Plans**: TBD

### Phase 7: Explicit Migration and Rebuild Cutover

**Goal**: Existing users can adopt the canonical format without silent mutation or losing the only valid copy of stored data.
**Depends on**: Phase 6
**Requirements**: MIGR-03, MIGR-04, MIGR-05, MIGR-06
**Success Criteria** (what must be TRUE):

  1. Users can inspect a store without mutation and receive human-readable and machine-readable migration plans with entry counts, bytes, incompatibilities, and intended actions.
  2. Supported same-backend metadata and payload format migrations use copy-verify-switch semantics and retire the prior copy only after verified publication.
  3. An interrupted migration can resume idempotently from its journal without losing or replacing the only valid generation.
  4. Incompatible formats and cross-backend moves offer an explicit, scoped, confirmed rebuild path rather than implicit deletion or a promise of universal physical migration.

**Plans**: TBD

### Phase 8: Production Gates and Performance Stabilization

**Goal**: Users can rely on reproducible release evidence across supported installations, Python versions, backends, failures, and operational scale.
**Depends on**: Phase 7
**Requirements**: QUAL-01, QUAL-02, QUAL-03, QUAL-04, QUAL-05, QUAL-06, QUAL-07
**Success Criteria** (what must be TRUE):

  1. A clean minimal wheel imports every guaranteed public symbol and completes a memory-backed round trip, while each advertised extra installs and imports independently.
  2. Required CI passes across supported Python versions, backend contracts, lint and coverage policy, packaging, PostgreSQL, and authoritative AWS S3 integration.
  3. Deterministic fault, race, and crash-recovery suites verify every lifecycle commit boundary, and lifecycle/cache-policy modules meet the project’s statement and branch coverage gates.
  4. Checked-in benchmarks establish reviewed correctness-aware latency and throughput budgets after lifecycle behavior stabilizes.
  5. Inventory, reconciliation, statistics, clear, and aggregate operations demonstrate bounded memory and backend-call behavior without accidental N+1 access patterns.

**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Compatibility and Security Baseline | 15/15 | Complete | 2026-08-30 |
| 2. Canonical Storage and Integrity Contract | 7/7 | Complete    | 2026-08-30 |
| 3. Atomic Lifecycle and Recovery Engine | 8/10 | In Progress|  |
| 4. Metadata Composition and Topology Contracts | 0/TBD | Not started | - |
| 5. Payload Backends and Full Matrix Parity | 0/TBD | Not started | - |
| 6. UnifiedCache Policy Composition | 0/TBD | Not started | - |
| 7. Explicit Migration and Rebuild Cutover | 0/TBD | Not started | - |
| 8. Production Gates and Performance Stabilization | 0/TBD | Not started | - |
