# Roadmap: Cacheness

## Overview

Cacheness will move from overlapping cache and storage paths to one storage engine in eight phases. After compatibility/security characterization and canonical storage contracts, Phase 3 proves one initialized SQLite/filesystem lifecycle and a thin cache-over-BlobStore slice before backend expansion. Phase 4 makes catalog customization first-class and replaces overlapping backend selection with one composition contract; Phase 5 qualifies explicitly supported payload/catalog pairings. Phase 6 publishes one coherent cache policy surface rather than preserving pre-production aliases. Migration and release gates establish future version-to-version tooling without absorbing `SqlCache` or requiring a dual-role live store. ADR 0001 governs topology-specific integrity, recovery, progress and performance; a common interface does not promise cross-resource ACID or identical availability.

## Phases

- [x] **Phase 1: Compatibility and Security Baseline** - Freeze supported behavior and close the currently exposed path, parser, query, and trust-boundary gaps.
- [x] **Phase 2: Canonical Storage and Integrity Contract** - Give direct `BlobStore` callers one versioned manifest, committed-read model, and fail-closed integrity contract. (completed 2026-08-30)
- [x] **Phase 3: Atomic Lifecycle and Recovery Engine** - Make writes, replacements, deletions, races, and crash residue converge without corrupting the last valid generation.
- [x] **Phase 4: Metadata Composition and Topology Contracts** - Unify all metadata backends behind one injectable CAS contract with honest capability validation. (completed 2026-09-08)
- [ ] **Phase 5: Payload Backends and Supported Topology Qualification** - Qualify explicitly supported filesystem, memory, and AWS S3 pairings at their declared capability tiers.
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

**Plans**: 7/7 plans complete

Plans:

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

The delivered foundation is documented in [the Phase 3 implementation ledger](../docs/phase3-direct-implementation-2026-09-06.md) and [storage initialization guide](../docs/STORAGE_INITIALIZATION.md); `03-GAP-REPLAN.md` is planning history. Future phases reuse `initialize`, `put_entry`, `get_entry_info`, `open_entry`, and exact conditional deletion, preserving their semantics rather than reopening prepare/promote/cleanup in callers. The broader adapter protocol remains transitional. Complete CACH-01/02/03/06 and BACK-07 acceptance stays in its owner phase.

For Phases 4–8, every lifecycle change must name its supported topology, transaction boundary, finite integrity/recovery cases and typed progress outcomes. Initialize before shared workers; maintenance requires stopped workers. Optional projection failure cannot revoke a committed blob; explicitly requested external metadata may report a committed partial outcome, including after cache close. Do not expand storage admission across a second catalog to hide that outcome. Apply ADR 0001 stop conditions before adding coordination or strengthening a guarantee. Separate cache/non-cache namespaces remain sufficient; `SqlCache` stays separate.

### Phase 4: Metadata Composition and Topology Contracts

**Goal**: Users can customize BlobStore catalog metadata without implementing lifecycle sequencing, select advertised metadata backends through one composition root, and receive only guarantees supported by the topology.
**Depends on**: Phase 3
**Requirements**: BACK-02, BACK-03, BACK-06, BACK-07
**Starting point**: Phase 3 already provides authenticated mapping metadata and the same-generation engine interface. This phase adds field validation/query customization and narrows transactional adapters; it does not reimplement the storage lifecycle. The Phase 4 discussion selected a native schema/composition contract, then the user approved a pre-production compatibility reset: legacy selection APIs and development-only catalog layouts may be replaced rather than adapted, while explicit version/migration-required evidence remains.
**Success Criteria** (what must be TRUE):

  1. JSON, memory, SQLite, and PostgreSQL have explicit authority/projection roles and supported capability tiers. Demonstrate the narrowed transactional adapter with the existing local engine before adding further authority implementations; adapters supply catalog transactions, not duplicate lifecycle sequencing. JSON projections do not become an independent transactional authority.
  2. A caller-injected backend instance remains the selected instance, and a registered backend name resolves through the same construction path. Retire overlapping legacy selectors instead of preserving parallel compatibility paths.
  3. Users can inspect durability, process/host sharing, compare-and-swap, streaming, and listing capabilities for the active backend pair.
  4. Invalid topology claims, such as durable multi-host storage backed by process-local payloads or unsupported coordination, fail during configuration rather than during a write.
  5. Direct BlobStore users can define application metadata, validate/query supported fields, update attributes and reopen current-layout stores without implementing a lifecycle backend. Authoritative attributes commit with the descriptor; external ORM links/indexes are explicitly derived. Pre-Phase-4 development layouts need not reopen through runtime shims and instead fail with explicit migration/rebuild-required evidence; no unchosen schema framework is implied.
  6. Catalog and derived-index APIs state their consistency and failure behavior. Preserve committed receipts, exact expectations, non-destructive corruption handling and explicit partial outcomes; any index reconstruction is an explicit derived operation, not a prerequisite for canonical reads or cleanup.

**Plans**: 14/14 plans complete; final code review, 18/18 goal verification,
Nyquist validation, and security verification passed on 2026-09-08

Plans:
**Wave 0**

- [x] 04-01-PLAN.md — Freeze Wave 0 catalog, composition, role, projection, format-rejection, and Ruff-delta contracts.

**Wave 1** *(blocked on Wave 0 completion)*

- [x] 04-02-PLAN.md — Define the native catalog/query vocabulary, independent version dimensions, format 2, and BlobReceipt.

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 04-03-PLAN.md — Implement one role-aware StoreTopology resolver with explicit ownership, capabilities, and a memory tracer.

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 04-04-PLAN.md — Integrate signed catalog state and bounded revision-bound scans with memory and initialized SQLite authorities.

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 04-05-PLAN.md — Add bounded derived projections, checkpoints, committed-partial receipts, refresh, and isolated rebuild contracts.

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 04-06-PLAN.md — Rewrite the first bounded set of lifecycle and security consumers onto StoreTopology and BlobReceipt.

**Wave 6** *(blocked on Wave 5 completion)*

- [x] 04-07-PLAN.md — Prepare remaining mixed-scope and legacy-symbol tests without deleting unrelated coverage.

**Wave 7** *(blocked on Wave 6 completion)*

- [x] 04-08-PLAN.md — Atomically remove obsolete source/public surfaces and run the Python 3.11/3.13 release matrix.

**Wave 8** *(gap closure; blocked on Wave 7 completion)*

- [x] 04-09-PLAN.md — Close the public catalog and selected-payload data-flow gaps through the existing BlobStore/authority lifecycle.

**Wave 9** *(blocked on Wave 8 completion)*

- [x] 04-10-PLAN.md — Close application registration, structural validation, and ownership unwind in the single composition root.

**Wave 10** *(blocked on Wave 9 completion)*

- [x] 04-11-PLAN.md — Harden post-commit projection outcomes and bounded public cursor inputs.

**Wave 11** *(blocked on Wave 10 completion)*

- [x] 04-12-PLAN.md — Rewrite remaining registry, lifecycle, integrity, initialization, and cache consumers against the final public seams.

**Wave 12** *(blocked on Wave 11 completion)*

- [x] 04-13-PLAN.md — Finish consumer cutover and produce honest, reproducible Python 3.11/3.13 release evidence.

**Wave 13** *(final gap closure; blocked on Wave 12 completion)*

- [x] 04-14-PLAN.md — Make advertised projections constructible, harden the retired-API audit, and remove the accidental root draft.

### Phase 5: Payload Backends and Supported Topology Qualification

**Goal**: Each advertised payload backend works through the shared engine in explicitly supported payload/catalog pairings, with verified topology-specific guarantees rather than Cartesian or identical-availability parity.
**Depends on**: Phase 4
**Requirements**: BACK-01, BACK-04, BACK-05
**Success Criteria** (what must be TRUE):

  1. Filesystem, memory, and S3 payload adapters reuse the engine and pass the applicable immutable-generation, verified-read, delete, list, integrity, resource-cleanup and reconciliation cases at their declared durability tier. Each advertised family has an explicit supported pairing; memory is not assigned crash-durability claims.
  2. Every explicitly supported advertised payload/catalog pairing completes the common workflows at its declared durability/sharing/progress tier. Publish unsupported combinations and reject impossible claims; backend-neutral calls do not hide topology-specific outcomes.
  3. PostgreSQL concurrency and transaction behavior is verified against a real PostgreSQL service, including conflicts and cleanup after partial failure.
  4. AWS S3 is the authoritative remote-object test target for conditional operations, checksums, streaming, pagination, retries, and cleanup; compatible services are claimed only where explicitly verified.
  5. Shared-store tests initialize before workers and account for success, exact conflict and declared typed retryable outcomes separately from integrity failures. PostgreSQL transactions do not encompass S3/filesystem effects; partial external effects remain attributable recovery work, not a reason for another coordinator.

**Plans**: 4/10 plans executed

Plans:
**Wave 1**

- [x] 05-01-PLAN.md — Freeze the exact three-profile support boundary and reject every unqualified/edge pairing before I/O.

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 05-02-PLAN.md — Establish reusable local payload and deterministic lifecycle-fault contracts.
- [x] 05-03-PLAN.md — Replace direct S3 CRUD with bounded immutable-generation I/O for the shared engine.
- [x] 05-04-PLAN.md — Implement explicit versioned PostgreSQL initialization and core exact-CAS transitions.

**Wave 3** *(blocked on Wave 2 completion)*

- [ ] 05-05-PLAN.md — Complete bounded PostgreSQL authority workflows and typed progress contracts.

**Wave 4** *(blocked on Wave 3 completion)*

- [ ] 05-06-PLAN.md — Compose all three profiles through the one lifecycle engine and bounded reconciliation evidence.
- [ ] 05-07-PLAN.md — Build the sanitized, non-skippable real-service qualification harness.

**Wave 5** *(blocked on Wave 4 completion)*

- [ ] 05-08-PLAN.md — Define real PostgreSQL, real AWS S3, and independent-client integration suites.

**Wave 6** *(blocked on Wave 5 completion)*

- [ ] 05-09-PLAN.md — Publish exact API/topology coverage and a reproducible local architecture gate.

**Wave 7** *(blocked on Wave 6 completion)*

- [ ] 05-10-PLAN.md — Run the non-substitutable real-service qualification and record final support evidence.

### Phase 6: UnifiedCache Policy Composition

**Goal**: Cache users receive one coherent policy API while all payload-plus-metadata lifecycle work is delegated to `BlobStore`.
**Depends on**: Phase 5
**Requirements**: CACH-01, CACH-02, CACH-03, CACH-04, CACH-05, CACH-06

**Scope carried forward**: Phase 3 delivered canonical local reads/writes, TTL and authenticated inventory/deletion through BlobStore, with engine-owned cleanup and declared derived-state failure outcomes. Extend and qualify remaining eviction/predicate/decorator paths, cached-None behavior, statistics and supported topology coverage. Do not redo the local integration, recreate removed repair hooks, or require cache/non-cache namespace sharing.
**Success Criteria** (what must be TRUE):

  1. One documented cache import, constructor, configuration, decorator, and result surface remains; redundant pre-production aliases and constructors are removed rather than retained through compatibility adapters.
  2. Cache puts, reads, invalidations and clears reach payloads and authoritative catalog state only through the supported BlobStore interface; TTL, eviction, keying, statistics and decorators remain policy owned by UnifiedCache. Separate derived metadata/statistics adapters cannot authorize canonical repair or deletion.
  3. TTL expiry, size eviction, predicate invalidation, decorator clearing, single-key deletion, and global clear remove complete stored entries through one lifecycle primitive.
  4. Cached `None` values are returned as hits rather than recomputed as misses, and decorator clear reports the entries it actually removed.
  5. Statistics use one documented aggregate/result model that separately identifies absent, expired, corrupt, conflict, and backend-error outcomes; legacy counter shapes are not a compatibility requirement.
  6. Preserve optional-export warnings and typed committed-partial errors for explicitly requested external metadata, including post-engine cache close. Canonical corruption is non-destructive; strict direct projection observations still reject malformed evidence. Cached-None policy uses explicit entry presence rather than recreating storage reads.

**Plans**: TBD

### Phase 7: Explicit Migration and Rebuild Cutover

**Goal**: With workers stopped, users can explicitly migrate or rebuild supported versioned stores without silent mutation or losing the only valid copy of stored data; the tooling establishes future release migration discipline even though current pre-production layouts may be unsupported.
**Depends on**: Phase 6
**Requirements**: MIGR-03, MIGR-04, MIGR-05, MIGR-06
**Success Criteria** (what must be TRUE):

  1. Users can inspect a store without mutation and receive human-readable and machine-readable migration plans with entry counts, bytes, incompatibilities, and intended actions.
  2. Supported same-backend metadata and payload format migrations use offline copy-verify-switch semantics and retire the prior copy only after verified publication. Workers remain stopped for mutation/cutover; ordinary constructors and initialize do not perform implicit upgrades.
  3. An interrupted migration can resume idempotently from its explicit maintenance evidence without losing the only valid generation. Preserve signing material and unknown/incomplete catalogs; do not adopt an unexplained empty database or add online writer coordination.
  4. Incompatible formats and cross-backend moves offer an explicit, scoped, confirmed rebuild path rather than implicit deletion or a promise of universal physical migration.
  5. Define the supported source-version window explicitly; it may exclude pre-production development layouts. Distinguish catalog/payload migration from rebuilding derived indexes; a missing or stale index cannot require rewriting valid payloads or become a second cutover authority.

**Plans**: TBD

### Phase 8: Production Gates and Performance Stabilization

**Goal**: Users can rely on reproducible release evidence across supported installations, Python versions, backends, failures, and operational scale.
**Depends on**: Phase 7
**Requirements**: QUAL-01, QUAL-02, QUAL-03, QUAL-04, QUAL-05, QUAL-06, QUAL-07
**Success Criteria** (what must be TRUE):

  1. A clean minimal wheel imports every guaranteed public symbol and completes a memory-backed round trip, while each advertised extra installs and imports independently.
  2. Required CI passes across supported Python versions, backend contracts, lint and coverage policy, packaging, PostgreSQL, and authoritative AWS S3 integration.
  3. Carry forward Phase 3's finite integrity/recovery regressions and add named cases for new supported topologies and commit boundaries; lifecycle/cache-policy modules meet the project's statement and branch coverage gates. Shared-worker fixtures initialize first and distinguish valid typed contention outcomes from corruption. This is not a proof of every interleaving or an automatic repeated race-repair loop.
  4. Checked-in benchmarks establish reviewed latency and throughput budgets for named workloads/environments after lifecycle behavior stabilizes, recording distributions separately from correctness gates. A regression cannot silently strengthen runtime deadlines or public failure semantics.
  5. Inventory, reconciliation, statistics, clear, and aggregate operations demonstrate bounded memory and backend-call behavior without accidental N+1 access patterns.
  6. Qualify exact commits in isolated environments and retain failed-run evidence plus migration fixtures for deliberately supported source versions. Historical compatibility tests may remain as evidence but are not release blockers for removed pre-production APIs/layouts. Do not count unavailable services/platforms as passes; Windows remains UNAVAILABLE/NOT_QUALIFIED until Phase 999.1 supplies native evidence. Optimization must preserve one authority and may not introduce a projection-repair prerequisite for canonical operations.

**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Compatibility and Security Baseline | 15/15 | Complete | 2026-08-30 |
| 2. Canonical Storage and Integrity Contract | 7/7 | Complete    | 2026-08-30 |
| 3. Atomic Lifecycle and Recovery Engine | 24/24 | Complete — direct qualification | 2026-09-06 |
| 4. Metadata Composition and Topology Contracts | 14/14 | Complete    | 2026-09-08 |
| 5. Payload Backends and Supported Topology Qualification | 4/10 | In Progress|  |
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
