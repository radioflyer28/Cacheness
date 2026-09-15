# Roadmap: Cacheness

## Overview

Cacheness will move from overlapping cache and storage paths to one storage engine in eight phases. After compatibility/security characterization and canonical storage contracts, Phase 3 proves one initialized SQLite/filesystem lifecycle and a thin cache-over-BlobStore slice before backend expansion. Phase 4 makes catalog customization first-class and replaces overlapping backend selection with one composition contract; Phase 5 qualifies explicitly supported payload/catalog pairings. Phase 6 publishes one coherent cache policy surface rather than preserving pre-production aliases. Migration and release gates establish future version-to-version tooling without absorbing `SqlCache` or requiring a dual-role live store. ADR 0001 governs topology-specific integrity, recovery, progress and performance; a common interface does not promise cross-resource ACID or identical availability.

## Phases

- [x] **Phase 1: Compatibility and Security Baseline** - Freeze supported behavior and close the currently exposed path, parser, query, and trust-boundary gaps.
- [x] **Phase 2: Canonical Storage and Integrity Contract** - Give direct `BlobStore` callers one versioned manifest, committed-read model, and fail-closed integrity contract. (completed 2026-08-30)
- [x] **Phase 3: Atomic Lifecycle and Recovery Engine** - Make writes, replacements, deletions, races, and crash residue converge without corrupting the last valid generation.
- [x] **Phase 4: Metadata Composition and Topology Contracts** - Unify all metadata backends behind one injectable CAS contract with honest capability validation. (completed 2026-09-08)
- [x] **Phase 5: Payload Backends and Supported Topology Qualification** - Qualify explicitly supported filesystem, memory, and AWS S3 pairings at their declared capability tiers. (completed 2026-09-08)
- [x] **Phase 6: UnifiedCache Policy Composition** - Route cache behavior through `BlobStore` while preserving public APIs, cache semantics, and the `SqlCache` boundary. (completed 2026-09-09)
- [x] **Phase 7: Explicit Migration and Rebuild Cutover** - Give existing stores an inspectable, resumable same-backend migration or confirmed rebuild path. (completed 2026-09-11)
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
**Requirements**: BACK-01, BACK-04
**Success Criteria** (what must be TRUE):

  1. Filesystem, memory, and S3 payload adapters reuse the engine and pass the applicable deterministic immutable-generation, verified-read, delete, list, integrity, resource-cleanup and reconciliation contracts at their declared durability tier. Memory is not assigned crash-durability claims.
  2. Memory/memory and SQLite/filesystem complete their qualified workflows. PostgreSQL/Amazon-S3 composes through the same engine and passes deterministic participant/topology contracts, but remains explicitly unqualified for release until Phase 8 closes BACK-05 with real-service evidence.
  3. The frozen real PostgreSQL, real Amazon S3, and independent-client suites plus their sanitized runner are executable and fail closed: missing services produce non-passing UNAVAILABLE evidence rather than a skip, emulator substitution, or support claim.
  4. Publish unsupported combinations and reject impossible claims before I/O; backend-neutral calls do not hide topology-specific outcomes. Compatible S3 services remain unclaimed without their own evidence.
  5. Shared-store tests initialize before workers and account for success, exact conflict and declared typed retryable outcomes separately from integrity failures. PostgreSQL transactions do not encompass S3/filesystem effects; partial external effects remain attributable recovery work, not a reason for another coordinator.

**Plans**: 9/9 canonical plans complete; 05-10 superseded by the Phase 8
BACK-05 real-service qualification plan

Plans:
**Wave 1**

- [x] 05-01-PLAN.md — Freeze the exact three-profile support boundary and reject every unqualified/edge pairing before I/O.

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 05-02-PLAN.md — Establish reusable local payload and deterministic lifecycle-fault contracts.
- [x] 05-03-PLAN.md — Replace direct S3 CRUD with bounded immutable-generation I/O for the shared engine.
- [x] 05-04-PLAN.md — Implement explicit versioned PostgreSQL initialization and core exact-CAS transitions.

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 05-05-PLAN.md — Complete bounded PostgreSQL authority workflows and typed progress contracts.

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 05-06-PLAN.md — Compose all three profiles through the one lifecycle engine and bounded reconciliation evidence.
- [x] 05-07-PLAN.md — Build the sanitized, non-skippable real-service qualification harness.

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 05-08-PLAN.md — Define real PostgreSQL, real AWS S3, and independent-client integration suites.

**Wave 6** *(blocked on Wave 5 completion)*

- [x] 05-09-PLAN.md — Publish exact API/topology coverage and a reproducible local architecture gate.

**Wave 7** *(blocked on Wave 6 completion)*

- [~] 05-10-PLAN.md — Superseded: non-substitutable real-service qualification moved intact to Phase 8; current UNAVAILABLE evidence is not a pass.

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

**Plans**: 11/11 plans executed

- [x] 06-01-PLAN.md
- [x] 06-02-PLAN.md
- [x] 06-03-PLAN.md
- [x] 06-04-PLAN.md
- [x] 06-05-PLAN.md
- [x] 06-06-PLAN.md
- [x] 06-07-PLAN.md
- [x] 06-08-PLAN.md
- [x] 06-09-PLAN.md — Migrate retained format, containment, signing, array-security, and public-contract tests to canonical cache construction/results.
- [x] 06-10-PLAN.md — Move retired metadata-query and key-parameter tests to bounded BlobStore catalog contracts.
- [x] 06-11-PLAN.md — Isolate the complete non-live local suite and extend the fixed Phase 6 verifier/evidence ledger.

**Wave 1**

- [x] 06-01: Presence-bearing lookup and immutable statistics tracer

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 06-02: Bounded exact removal and invalidation reports

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 06-03: Bounded resumable size enforcement

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 06-04: Explicit function-cache decorator policy

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 06-05: BlobStore topology composition and ownership

**Wave 6** *(blocked on Wave 5 completion)*

- [x] 06-06: Canonical public API and configuration cutover

**Wave 7** *(blocked on Wave 6 completion)*

- [x] 06-07: Canonical cache-policy documentation and examples

**Wave 8** *(blocked on Wave 7 completion)*

- [x] 06-08: Fixed Phase 6 contract verifier and validation ledger

**Wave 9** *(gap closure; blocked on Wave 8 completion)*

- [x] 06-09: Canonical format/security/public test migration
- [x] 06-10: Canonical catalog/query test migration

**Wave 10** *(blocked on both Wave 9 plans)*

- [x] 06-11: Deterministic non-live suite isolation and fixed verification

### Phase 7: Explicit Migration and Rebuild Cutover

**Goal**: With workers stopped, users can explicitly migrate or rebuild supported versioned stores without silent mutation or losing the only valid copy of stored data; the tooling establishes future release migration discipline even though current pre-production layouts may be unsupported.
**Depends on**: Phase 6
**Requirements**: MIGR-03, MIGR-04, MIGR-05, MIGR-06
**Success Criteria** (what must be TRUE):

  1. Users can inspect a store without mutation and receive human-readable and machine-readable migration plans with entry counts, bytes, incompatibilities, and intended actions.
  2. Supported same-backend metadata and payload format migrations use offline copy-verify-switch semantics and retire the prior copy only after verified publication. Workers remain stopped for mutation/cutover; ordinary constructors and initialize do not perform implicit upgrades.
  3. An interrupted bounded migration can resume idempotently for effects durably attributed in authority-owned evidence without losing the only valid generation. Each run enforces explicit entry, byte, and evidence limits; larger stores split into independently recoverable maintenance runs. Preserve signing material and unknown/incomplete catalogs; do not adopt unexplained state or add online writer coordination.
  4. Incompatible formats and cross-backend moves offer an explicit, scoped, confirmed rebuild path rather than implicit deletion or a promise of universal physical migration.
  5. Define the supported source-version window explicitly; it may exclude pre-production development layouts. Distinguish catalog/payload migration from rebuilding derived indexes; a missing or stale index cannot require rewriting valid payloads or become a second cutover authority.

**Accepted recovery/progress limit (checker override, 2026-09-10):** A crash after immutable payload publication but before its authority checkpoint may leave an invisible, unattributed orphan. It is never visible, never automatically adopted, and guaranteed exact cleanup is outside Phase 7. Integrity and authority visibility remain guaranteed while perfect orphan reclamation is not; resume/abort guarantees apply only to effects durably attributed in authority-owned evidence. This is an ADR 0001 topology-specific recovery/progress limit, not failed atomicity. Phase 7 adds no pre-publication maintenance intent, extra journal/lifecycle state, coordination mechanism, or production obstore adoption.

**Plans**: 24/24 plans executed. Plan 07-24 closes the sole remaining exact-verifier binding gap before independent re-verification.

Plans:

**Wave 0**

- [x] 07-01-PLAN.md — Prove one explicit memory inspect-plan-stage-verify-activate tracer and preserve ordinary no-upgrade behavior.

**Wave 1** *(blocked on Wave 0 completion)*

- [x] 07-02-PLAN.md — Confirm the one-way release-window, rebuild-exclusion, and finalize/purge decisions.

**Wave 2** *(blocked on the decision gates)*

- [x] 07-03-PLAN.md — Publish the independent compatibility matrix, release window, and canonical human/machine plan model.

**Wave 3** *(blocked on compatibility contracts)*

- [x] 07-04-PLAN.md — Add bounded entry-complete authority inventory for memory, SQLite, and deterministic PostgreSQL.
- [x] 07-05-PLAN.md — Authenticate, contain, and explicitly resume maintenance evidence without candidate adoption.

**Wave 4** *(blocked on inventory and evidence)*

- [x] 07-06-PLAN.md — Complete local whole-store stage/verify/activate and interruption recovery with derived projections separate.

**Wave 5** *(blocked on local cutover)*

- [x] 07-07-PLAN.md — Extend deterministic PostgreSQL/S3 maintenance contracts without claiming live qualification.
- [x] 07-08-PLAN.md — Implement offline rollback, finalize, abort, and separately confirmed idempotent purge.

**Wave 6** *(blocked on retirement semantics)*

- [x] 07-09-PLAN.md — Deliver include-all, exact-confirmation rebuild through registered handlers and destination BlobStore.

**Wave 7** *(blocked on remote contracts and rebuild)*

- [x] 07-10-PLAN.md — Publish the single maintenance API, operator runbook, and reasoned external-API coverage declaration.

**Wave 8** *(blocked on the complete public workflow)*

- [x] 07-11-PLAN.md — Run the fixed Phase 7 contract, architecture, security, Nyquist, full-suite, and Ruff gates.

**Wave 9** *(gap closure; blocked on verified Plan 07-11 baseline)*

- [x] 07-12-PLAN.md — Add bounded durable migration-batch attribution, uncapped complete-candidate aggregation, and resumable STAGING abort/debt.
- [x] 07-13-PLAN.md — Re-run the canonical PostgreSQL worker fence after persisted identity/state load and before readiness.

**Wave 10** *(blocked on both Wave 9 repairs)*

- [x] 07-14-PLAN.md — Add exact canonical operation replay for rebuild response-loss recovery without another authority.

**Wave 11** *(blocked on canonical operation replay)*

- [x] 07-15-PLAN.md — Persist rebuild intents/receipts and resume or clean every rebuild interruption from exact run-owned evidence.

**Wave 12** *(blocked on durable rebuild recovery)*

- [x] 07-16-PLAN.md — Enforce exact destination compatibility and execute only concrete handler-owned guarded transformations.

**Wave 13** *(blocked on the destination contract)*

- [x] 07-17-PLAN.md — Replace raw plan metadata with digest-bound confidential records and authenticated execution-time rereads.

**Wave 14** *(blocked on all behavioral blocker repairs)*

- [x] 07-18-PLAN.md — Bind every fixed Phase 7 claim and gap threat to exact executable test selectors.

**Wave 15** *(blocked on exact selector verification)*

- [x] 07-19-PLAN.md — Run quick/all fixed verification and record truthful blocker, warning, spike-boundary, and Phase 8 evidence.

**Wave 16** *(gap closure; blocked on the verified Plan 07-19 baseline)*

- [x] 07-20-PLAN.md — Correct same-version format transforms and make typed S3 abort failures resumable through exact existing cleanup debt.

**Wave 17** *(blocked on migration target and abort repairs)*

- [x] 07-21-PLAN.md — Settle rebuild cleanup debt from authenticated exact receipts before entering terminal ABORTED.

**Wave 18** *(blocked on all three behavioral repairs)*

- [x] 07-22-PLAN.md — Bind the repairs to exact fixed-verifier selectors and record truthful final validation.

**Wave 19** *(gap closure; blocked on the verified Plan 07-22 baseline)*

- [x] 07-23-PLAN.md — Fence authentic rebuild cleanup debt before direct progression or terminal-state effects and bind the invariant to the fixed verifier.

**Wave 20** *(gap closure; blocked on the verified Plan 07-23 implementation)*

- [x] 07-24-PLAN.md — Bind the forward-fence/resume regression to the existing Plan 21 terminal-state threat and make the fixed verifier fail closed on either missing selector.

### Phase 07.1: Obstore Payload Participant Unification (INSERTED)

**Goal:** Users get one obstore-backed payload participant for filesystem, memory, and S3 while BlobStore's lifecycle authority remains the sole owner of intent, visibility, reconciliation, and cleanup debt.
**Requirements**: [STOR-01, STOR-02, STOR-03, STOR-04, STOR-05, STOR-06, STOR-07, STOR-08, BACK-01, BACK-02, BACK-03, BACK-04, BACK-06, BACK-07, CACH-01, CACH-02, CACH-03, SECU-01, SECU-03, SECU-04, SECU-05, SECU-07, SECU-08, MIGR-01, MIGR-02, MIGR-03, MIGR-04, MIGR-05, MIGR-06, MIGR-07, QUAL-01, QUAL-04, QUAL-07]
**Depends on:** Phase 7
**Plans:** 11/11 plans complete

Plans:
**Wave 1**

- [x] 07.1-01-PLAN.md — Human package-legitimacy gate for exact obstore 0.11.1.

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 07.1-02-PLAN.md — Pin obstore and prove the complete consumed SDK surface before production edits.

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 07.1-03-PLAN.md — Build the shared local/memory adapter and preserve guarded path handlers.

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 07.1-04-PLAN.md — Qualify mocked-S3 direct publication, recovery, bounds, and D-16 configuration policy.

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 07.1-05-PLAN.md — Authenticate immutable-payload-bound transport evidence in the existing verification transition.

**Wave 6** *(blocked on Wave 5 completion)*

- [x] 07.1-06-PLAN.md — Persist transport evidence in explicit SQLite/PostgreSQL authority schema versions.

**Wave 7** *(blocked on Wave 6 completion)*

- [x] 07.1-07-PLAN.md — Expose a read-only authoritative-generation transport comparison API.
- [x] 07.1-08-PLAN.md — Route migration, rebuild, and reconciliation payload effects through the participant.

**Wave 8** *(blocked on Wave 7 completion)*

- [x] 07.1-09-PLAN.md — Atomically cut all built-in topology factories over to obstore.

**Wave 9** *(blocked on Wave 8 completion)*

- [x] 07.1-10-PLAN.md — Delete custom/boto3/multipart mechanics and update retained contract suites.

**Wave 10** *(blocked on Wave 9 completion)*

- [x] 07.1-11-PLAN.md — Close packaging, documentation, fixed verification, and Phase 8 non-claim boundaries.

### Phase 8: Production Gates and Performance Stabilization

**Goal**: Users can rely on reproducible local-readiness evidence for Cacheness installations, lifecycle failures, cache policy, and operational scale without inheriting unproved remote-service, platform, performance, or publication claims.
**Depends on**: Phase 07.1
**Requirements**: QUAL-01, QUAL-02, QUAL-03, QUAL-04, QUAL-05, QUAL-07
**Success Criteria** (what must be TRUE):

  1. A clean minimal wheel imports every guaranteed public symbol and completes a memory-backed round trip, while each advertised extra installs and imports independently.
  2. Local deterministic, package, coverage, Ruff, and structural gates pass through fixed commands. The checked workflow definitions retain the supported-Python and protected-live machinery, but only evidence actually produced on an eligible host/service can qualify that platform or topology.
  3. Carry forward Phase 3's finite integrity/recovery regressions and add named cases for new supported topologies and commit boundaries; lifecycle/cache-policy modules meet the project's statement and branch coverage gates. Shared-worker fixtures initialize first and distinguish valid typed contention outcomes from corruption. This is not a proof of every interleaving or an automatic repeated race-repair loop.
  4. The checked-in benchmark harness, representative workloads, SHA-256/XXH3 comparison, and runner preflight remain reproducible future qualification capability. Controlled-Linux budgets and QUAL-06 are explicitly `DEFERRED`/`NOT_QUALIFIED` under `SEED-006` and do not block this milestone; macOS measurements remain diagnostic and make no Linux-equivalence or cross-platform budget claim.
  5. Inventory, reconciliation, statistics, clear, and aggregate operations demonstrate bounded memory and backend-call behavior without accidental N+1 access patterns.
  6. A fixed local-readiness report is bound to one exact commit and distinguishes passing local evidence from closed nonclaims. `BACK-05` remains `DEFERRED`/`NOT_QUALIFIED` and immutable publication remains `DEFERRED`/`NOT_PUBLISHED` under SEED-007; QUAL-06 remains deferred under SEED-006; Windows remains `UNAVAILABLE`/`NOT_QUALIFIED` until Phase 999.1. No mock, local substitute, configuration preflight, or stale artifact can become remote/platform/release evidence. Optimization must preserve one authority and may not introduce a projection-repair prerequisite for canonical operations.

**Plans**: 15/17 canonical plans executed; 08-11 and 08-12 superseded by D-24/SEED-007

Plans:

- [x] 08-01-PLAN.md — Establish strict evidence classes and the deterministic exact-commit tracer.
- [x] 08-02-PLAN.md — Qualify the base wheel and every optional group in isolated environments.
- [x] 08-03-PLAN.md — Encode supported Python/platform evidence and truthful Windows nonclaims.
- [x] 08-04-PLAN.md — Close named lifecycle and cache-policy coverage gaps.
- [x] 08-05-PLAN.md — Record branch-aware coverage floors and direct Ruff ratchets.
- [x] 08-06-PLAN.md — Prove structural backend-call and peak-memory bounds at scale.
- [x] 08-07-PLAN.md — Build layer-separated format/hash benchmarks and controlled workflow.
- [x] 08-08-PLAN.md — Freeze protected real PostgreSQL/Amazon-S3 qualification machinery.
- [x] 08-09-PLAN.md — Compose deterministic quality gates into the supported CI matrix.
- [x] 08-10-PLAN.md — Build the fixed verifier, exact-SHA workflow collector, release aggregator, and publication verifier.
- [x] 08-13-PLAN.md — Add the read-only exact-commit controlled-runner preflight required before baseline capture.
- [x] 08-14-PLAN.md — Make controlled performance an explicit deferred nonclaim in release collection, aggregation, verification, and documentation.
- [x] 08-15-PLAN.md — Add a configuration-only protected-live preflight and bind it into the fixed verifier.
- [~] 08-11-PLAN.md — Superseded: exact-SHA real PostgreSQL/Amazon-S3 collection moved intact to SEED-007; `BACK-05` remains NOT_QUALIFIED.
- [~] 08-12-PLAN.md — Superseded: immutable GitHub release publication moved intact to SEED-007; publication remains NOT_PUBLISHED.
- [x] 08-17-PLAN.md — Correct the inherited exact-snapshot clear/delete test to accept the ADR-defined typed contention outcome while preserving every safety and recovery assertion.
- [x] 08-18-PLAN.md — Replace the stale concurrent-first-creation assertion with the approved explicit-initialization-before-shared-workers SQLite contract.
- [ ] 08-19-PLAN.md — Restore the frozen statement/branch ratchet with deterministic SQLite validation and fail-closed schema/identity coverage after the approved Plan 08-18 test correction.
- [ ] 08-16-PLAN.md — Close deterministic local readiness and encode the remote/performance/platform/publication nonclaims after 08-19 restores the coverage gate.

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Compatibility and Security Baseline | 15/15 | Complete | 2026-08-30 |
| 2. Canonical Storage and Integrity Contract | 7/7 | Complete    | 2026-08-30 |
| 3. Atomic Lifecycle and Recovery Engine | 24/24 | Complete — direct qualification | 2026-09-06 |
| 4. Metadata Composition and Topology Contracts | 14/14 | Complete    | 2026-09-08 |
| 5. Payload Backends and Supported Topology Qualification | 9/9 | Complete    | 2026-09-08 |
| 6. UnifiedCache Policy Composition | 11/11 | Complete    | 2026-09-09 |
| 7. Explicit Migration and Rebuild Cutover | 24/24 | Complete    | 2026-09-11 |
| 07.1. Obstore Payload Participant Unification | 11/11 | Complete | 2026-09-13 |
| 8. Production Gates and Performance Stabilization | 15/17 | In Progress|  |

## Backlog

### Phase 999.1: Qualify native Windows lifecycle authority (BACKLOG)

**Goal:** Run the Phase 3 native-Windows release qualification in an eligible Python 3.11 NTFS environment and attach the machine-readable evidence.
**Requirements:** TBD
**Plans:** 0 plans

Plans:

- [ ] Verify protected-DACL provisioning, same-session SQLite contention, and different-token denial; promote with $gsd-review-backlog when ready.

### Phase 999.2: Formalize custom payload handler contract and developer kit (BACKLOG)

**Goal:** Give third-party format authors one canonical, BlobStore-first extension contract with accurate per-store registration documentation, stable payload identity/version guidance, representative native-format examples, and reusable conformance tests.
**Requirements:** TBD
**Plans:** 0 plans

Plans:

- [ ] Replace stale global cache-handler examples with `store.handlers.register_handler(...)`, document lifecycle ownership and migration compatibility responsibilities, and provide a contract-test kit; promote with $gsd-review-backlog when ready.
