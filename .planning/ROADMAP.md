# Roadmap: Cacheness

## Overview

Cacheness will move from overlapping cache and storage paths to one storage engine in eight phases. After compatibility/security and canonical storage contracts, Phase 3 proves one initialized SQLite/filesystem lifecycle and a thin cache-over-BlobStore slice before backend expansion. Phase 4 makes catalog customization first-class and narrows transactional adapters; Phase 5 qualifies explicitly supported payload/catalog pairings. Phase 6 completes cache policy and compatibility coverage. Migration and release gates complete the cutover without absorbing `SqlCache` or requiring a dual-role live store. ADR 0001 governs topology-specific integrity, recovery, progress and performance; a common interface does not promise cross-resource ACID or identical availability.

## Phases

- [x] **Phase 1: Compatibility and Security Baseline** - Freeze supported behavior and close the currently exposed path, parser, query, and trust-boundary gaps.
- [x] **Phase 2: Canonical Storage and Integrity Contract** - Give direct `BlobStore` callers one versioned manifest, committed-read model, and fail-closed integrity contract. (completed 2026-08-30)
- [x] **Phase 3: Atomic Lifecycle and Recovery Engine** - Make writes, replacements, deletions, races, and crash residue converge without corrupting the last valid generation.
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

**Goal**: Within the initialized local SQLite/filesystem topology, preserve complete generations with attributable recovery evidence and prove that cache policy consumes the same BlobStore engine without a second catalog authority.
**Depends on**: Phase 2
**Requirements**: STOR-03, STOR-04, STOR-05, STOR-06, STOR-07
**Success Criteria** (what must be TRUE):

  1. A write or overwrite exposes either the previous complete generation or the new complete generation, never mixed payload and metadata state.
  2. Failures at serialization, payload publication, metadata commit, or old-generation cleanup preserve the last valid generation and leave all residue detectable.
  3. Repeated overwrite, delete, clear, and close operations converge safely while cleaning both payload and metadata state.
  4. Operators can dry-run and resume reconciliation to repair, quarantine, or report inconsistent entries without guessing their provenance.
  5. Initialized same-key write/delete/read contention preserves integrity with success, exact conflict or typed retryable timeout; distinct-key payload work overlaps while short SQLite write transactions may serialize.
  6. Direct application metadata round-trips through BlobStore, and a separate cache instance uses its supported same-generation read/write interface for a local put/get/TTL slice. Corrupt or unavailable derived projections cannot revoke valid blobs.
  7. Startup and post-commit compatibility changes have recorded approval; incomplete initialization fails closed unchanged, with no concurrent-first-create availability promise or new filesystem publication protocol.

**Plans**: 24/24 canonical plans complete (03-19 superseded). Plans 03-21 through 03-25 were implemented and qualified directly by the primary agent at the user's request, bypassing GSD execute/review/checker. The exact qualified tree is `5282dca`; see `03-25-SUMMARY.md` and `docs/phase3-direct-implementation-2026-09-06.md`. The older GSD verification report is historical, not a fresh independent verdict or a reason to repeat closed gaps.

Plans:
**Wave 1**

- [x] 03-01-PLAN.md — Characterize public contracts, release history, locator, and deletion checkpoint

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 03-02-PLAN.md — Prove the authority-backed SQLite BlobStore write/read/reopen tracer and in-memory parity

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 03-03-PLAN.md — Complete hardened SQLite ownership, schema, transactions, deadlines, and rollback

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 03-04-PLAN.md — Move put, overwrite, metadata, delete, close, and same-key races onto the authority

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 03-05-PLAN.md — Implement exact clear snapshots and bounded high-water reconciliation

**Wave 6** *(blocked on Wave 5 completion)*

- [x] 03-06-PLAN.md — Make JSON a revision-checked projection and seal downstream capability contracts

**Wave 7** *(blocked on Wave 6 completion)*

- [x] 03-07-PLAN.md — Delete the abandoned scheduler core and prove it cannot be reached

**Wave 8** *(blocked on Wave 7 completion)*

- [x] 03-08-PLAN.md — Trim retained helpers while preserving payload safety and one-authority composition

**Wave 9** *(blocked on Wave 8 completion)*

- [x] 03-09-PLAN.md — Close deterministic crash, race, Python-version, and platform-runner gates without claiming native Windows qualification

**Wave 10** *(blocked on Wave 9 completion)*

- [x] 03-12-PLAN.md — Add a digest-bound contract distinguishing repository-runtime evidence from Python 3.11 native-qualification attestation

**Wave 11** *(blocked on Wave 10 completion)*

- [x] 03-11-PLAN.md — Capture and attest the current host's UNAVAILABLE/NOT_QUALIFIED Windows evidence after the additive command contract closes

**Wave 12** *(blocked on Wave 11 completion)*

- [x] 03-10-PLAN.md — Measure lifecycle budgets and close final Phase 3 acceptance after 03-11

**Wave 13** *(blocked on Wave 12 completion)*

- [x] 03-13-PLAN.md — Isolate the complete repository suite, remove nested environment mutation, and close Plan 03-10 only on green evidence

**Wave 14** *(blocked on Wave 13 completion)*

- [x] 03-14-PLAN.md — Make facade projections generation-conditional, committed-only, and coherent under deterministic races

**Wave 15** *(blocked on Wave 14 completion)*

- [x] 03-15-PLAN.md — Admit facade lifecycle work and move projection/link ownership behind exact promotion

**Wave 16** *(blocked on Wave 15 completion)*

- [x] 03-16-PLAN.md — Make SQL projection CAS atomic and restore cached custom-metadata/signing parity

**Wave 17** *(blocked on Wave 16 completion)*

- [x] 03-17-PLAN.md — Harden fresh-root bootstrap and run complete adversarial acceptance

**Wave 18** *(blocked on Wave 17 completion)*

- [x] 03-18-PLAN.md — Retire failed coordination state, enforce fork/deadline safety, and restore cached query-meta capability

**Wave 19** *(blocked on Wave 18 completion)*

- 03-19-PLAN.md — **Superseded by ADR 0001**; retained as interrupted implementation history and excluded from canonical counts
- [x] 03-20-PLAN.md — Preserve real Plan 03-19 integrity fixes, cover partial exclusive-publication crashes, remove deadline-driven authority complexity, then qualify committed topology-specific recovery/bounded outcomes from a detached worktree

**Wave 20** *(blocked on Wave 19 completion)*

- [x] 03-21-PLAN.md — Close bounded memory abort/debt and strict projection-parser gaps; replaces the unexecuted private-bootstrap draft

**Wave 21** *(blocked on Wave 20 completion)*

- [x] 03-22-PLAN.md — Approve and implement initialization before shared-worker use; classify SQLite operational failures precisely

**Wave 22** *(blocked on Wave 21 completion)*

- [x] 03-23-PLAN.md — Expose supported same-generation BlobStore entry snapshots and committed receipts with engine-owned cleanup

**Wave 23** *(blocked on Wave 22 completion)*

- [x] 03-24-PLAN.md — Approve derived-state failure semantics, route canonical cache policy through BlobStore, and remove duplicate facade sequencing

**Wave 24** *(blocked on Wave 23 completion)*

- [x] 03-25-PLAN.md — Qualify finite public workflows and named failure classes at an exact committed tree; no automatic repair loop

Cross-cutting constraints: one authoritative catalog; immutable native payloads outside transactions; exact-generation deletion; optional projections never revoke valid data; initialization and failure-contract checkpoints; integrity/recovery separate from progress/performance; original dirty fixture evidence preserved; Windows remains UNAVAILABLE/NOT_QUALIFIED.

The current gap design is `03-GAP-REPLAN.md`. The broader adapter protocol is transitional: Phase 3 deepens the caller interface, while Phase 4 must narrow transactional adapters before implementing additional lifecycle backends. Complete CACH-01/02/03/06 and BACK-07 acceptance stays in its owner phase.

### Phase 4: Metadata Composition and Topology Contracts

**Goal**: Users can customize BlobStore catalog metadata without implementing lifecycle sequencing, select advertised metadata backends through one composition root, and receive only guarantees supported by the topology.
**Depends on**: Phase 3
**Requirements**: BACK-02, BACK-03, BACK-06, BACK-07
**Success Criteria** (what must be TRUE):

  1. JSON, memory, SQLite, and PostgreSQL represent the canonical entries through explicitly declared authority/projection roles and supported capability tiers. Centralize lifecycle sequencing and narrow transactional catalog operations before multiplying adapters; do not copy the current broad state machine into every backend.
  2. A caller-injected backend instance remains the selected instance, and a registered backend name resolves through the same construction path used by direct and composed storage.
  3. Users can inspect durability, process/host sharing, compare-and-swap, streaming, and listing capabilities for the active backend pair.
  4. Invalid topology claims, such as durable multi-host storage backed by process-local payloads or unsupported coordination, fail during configuration rather than during a write.
  5. Direct BlobStore users can define application metadata, validate/query supported fields, update attributes and reopen without implementing a lifecycle backend. Authoritative attributes commit with the descriptor; external ORM links/indexes are explicitly derived unless participating in that same transaction. Specify the schema/extension interface and compatibility impact during Phase 4 discussion; no unchosen schema framework is implied.

**Plans**: TBD

### Phase 5: Payload Backends and Full Matrix Parity

**Goal**: Every advertised payload backend works end to end with every supported metadata backend under the canonical lifecycle.
**Depends on**: Phase 4
**Requirements**: BACK-01, BACK-04, BACK-05
**Success Criteria** (what must be TRUE):

  1. Filesystem, memory, and S3 payload backends pass the same immutable-generation, read, delete, list, integrity, resource-cleanup, and reconciliation contract.
  2. Every explicitly supported advertised payload/catalog pairing completes the common workflows at its declared durability/sharing/progress tier. Publish unsupported combinations and reject impossible claims; backend-neutral calls do not hide topology-specific outcomes.
  3. PostgreSQL concurrency and transaction behavior is verified against a real PostgreSQL service, including conflicts and cleanup after partial failure.
  4. AWS S3 is the authoritative remote-object test target for conditional operations, checksums, streaming, pagination, retries, and cleanup; compatible services are claimed only where explicitly verified.

**Plans**: TBD

### Phase 6: UnifiedCache Policy Composition

**Goal**: Cache users retain their public workflows while all payload-plus-metadata lifecycle work is delegated to `BlobStore`.
**Depends on**: Phase 5
**Requirements**: CACH-01, CACH-02, CACH-03, CACH-04, CACH-05, CACH-06

**Scope carried forward**: Reuse the Phase 3 canonical local read/write/TTL seam. Complete all policy paths, decorators, cached-None behavior, statistics and supported topology coverage; do not recreate the removed facade storage lifecycle or require cache/non-cache namespace sharing.
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
| 3. Atomic Lifecycle and Recovery Engine | 24/24 | Complete — direct qualification | 2026-09-06 |
| 4. Metadata Composition and Topology Contracts | 0/TBD | Not started | - |
| 5. Payload Backends and Full Matrix Parity | 0/TBD | Not started | - |
| 6. UnifiedCache Policy Composition | 0/TBD | Not started | - |
| 7. Explicit Migration and Rebuild Cutover | 0/TBD | Not started | - |
| 8. Production Gates and Performance Stabilization | 0/TBD | Not started | - |

## Backlog

### Phase 999.1: Qualify native Windows lifecycle authority (BACKLOG)

**Goal:** Run the Phase 3 native-Windows release qualification in an eligible Python 3.11 NTFS environment and attach the machine-readable evidence.
**Requirements:** TBD
**Plans:** 0 plans

Plans:

- [ ] Verify protected-DACL provisioning, same-session SQLite contention, and different-token denial; promote with $gsd-review-backlog when ready.
