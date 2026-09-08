# Phase 5: Payload Backends and Supported Topology Qualification - Context

**Gathered:** 2026-09-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Qualify each advertised payload family through the shared `BlobStore` engine in
a small, explicit set of coherent backend pairings. Phase 5 makes memory,
filesystem, and Amazon S3 payload participants satisfy the applicable immutable
generation, verified-read, deletion, listing, cleanup, and reconciliation
contracts at their declared capability tier. It adds and qualifies the
PostgreSQL authority needed by the remote multi-host reference topology.

This phase does not build a Cartesian product of every payload/catalog
combination, duplicate the lifecycle state machine inside backend adapters,
complete `UnifiedCache` policy composition, execute stored-format migration, or
claim that mocks/emulators qualify real PostgreSQL or AWS behavior. ADR 0001's
integrity, recovery, progress, performance, and ACID boundaries remain binding.

</domain>

<decisions>
## Implementation Decisions

### Supported Topology Inventory

- **D-01:** V1 qualifies three reference topologies: memory authority with
  memory payloads for one-process ephemeral use; SQLite authority with
  filesystem payloads for initialized single-host persistence; and PostgreSQL
  authority with Amazon S3 payloads for correctly configured multi-host use.
  This covers every advertised authority and payload family without implying
  Cartesian parity. — **Reversibility:** costly — Applications and release
  documentation will rely on these named support boundaries; adding a pairing
  later is easy, but changing an existing pairing's guarantees requires a new
  qualification record.
- **D-02:** Other combinations, including PostgreSQL/filesystem,
  SQLite/S3, memory/persistent mixtures, and S3 with a process-local authority,
  are unsupported unless they receive a separate explicit qualification.
  Composition rejects them rather than warning, silently downgrading, or
  inferring support from constructibility.
- **D-03:** Registration and direct construction prove only that a participant
  can be instantiated. A pairing becomes supported only when its complete
  topology contract and service evidence pass.
- **D-04:** JSON remains an optional derived projection for supported stores.
  It never becomes a lifecycle authority, query-completeness source, or fallback
  authority. PostgreSQL may have separate authority and projection adapters;
  each object has one declared role.

### Live-Service Qualification

- **D-05:** PostgreSQL and Amazon S3 support claims require integration tests
  against real services. Moto, fakes, and local compatible services remain
  valuable contract tests, but cannot satisfy BACK-05 or create a production
  qualification claim.
- **D-06:** If a live endpoint or credentials are unavailable, the run records
  `UNAVAILABLE`/`NOT_QUALIFIED` and leaves the relevant acceptance gate open for
  an exact rerun. It must not substitute an emulator, convert the case into a
  passing skip, or weaken the supported-topology statement.
- **D-07:** Live tests use externally supplied standard driver/SDK credentials,
  unique test-owned PostgreSQL schema/table names and S3 prefixes, bounded
  idempotent cleanup, and no repository-persisted secrets. Destructive cleanup
  is limited to resources created by the individual qualification run.
- **D-08:** Amazon S3 is the authoritative remote-object target. An
  S3-compatible service is supported only after its own named qualification;
  endpoint compatibility is not inferred from boto3 API similarity.

### S3 Generation and Integrity Semantics

- **D-09:** Each S3 payload generation is a uniquely named immutable object.
  Creation is conditional where required to prevent replacement of an existing
  generation; PostgreSQL promotion, not object presence or listing, is the
  visibility point. Bucket versioning or object lock is not required for the V1
  contract. — **Reversibility:** one-way — Published locators and recovery
  evidence depend on immutable generation identity; moving to mutable stable
  keys would invalidate the lifecycle and migration contract.
- **D-10:** The canonical manifest's SHA-256 digest and byte size establish
  payload integrity. S3 ETags and transport checksums may be retained as
  diagnostics or transport validation but never substitute for the signed
  canonical digest.
- **D-11:** Handler-owned payload bytes are streamed from their private snapshot
  through bounded multipart upload where supported. Reads produce one contained
  private snapshot, verify its size and digest, and only then invoke the handler.
  The public engine does not expose an unverified network body to deserializers
  or require whole-object memory buffering.
- **D-12:** S3 listing and inventory are continuation-token based and bounded by
  page, object, byte, and work limits. Reconciliation consumes those bounded
  primitives; it does not materialize an unbounded bucket listing.
- **D-13:** Failed post-promotion deletion remains attributable cleanup debt in
  the PostgreSQL authority and is retried or reconciled idempotently. A failed
  S3 delete never causes the committed generation to be reported as rolled back.

### PostgreSQL Authority and Contention

- **D-14:** The remote reference topology uses a real PostgreSQL lifecycle
  authority. Its transactions own canonical descriptors, authoritative catalog
  fields, generation expectations, operation intent, and cleanup debt. The
  Phase 4 placeholder remains unregistered until the qualified authority exists;
  a projection adapter cannot stand in for it. — **Reversibility:** costly — The
  database schema and public capability contract become persisted release
  surfaces governed by Phase 7 migrations.
- **D-15:** PostgreSQL concurrency relies on transactions, unique constraints,
  and expected-generation conditional writes. Do not add an application-side
  distributed queue, process-local correctness gate, filesystem authority,
  global advisory-lock protocol, or a second lifecycle coordinator.
- **D-16:** Exact generation conflicts remain distinct from database progress
  failures. Serialization failures, deadlocks, lock timeouts, and connection
  timeouts become typed retryable outcomes with preserved driver cause,
  operation stage, and bounded context; callers are not promised that every
  contender succeeds.
- **D-17:** PostgreSQL initialization is explicit, idempotent, and completed
  before shared workers start. Ordinary construction/open does not silently
  migrate schemas. Incompatible schema changes remain stopped-worker Phase 7
  maintenance.
- **D-18:** A PostgreSQL transaction cannot include S3 effects. Immutable object
  creation precedes authority promotion; durable intent and cleanup debt make
  interruption attributable and deterministically reconcilable without a
  cross-resource ACID claim.

### Qualification Evidence and Claims

- **D-19:** Publish one exact capability matrix naming each supported pairing,
  coordination scope, durability/atomicity boundary, bounded progress outcomes,
  required service configuration, projection availability, and unsupported
  combinations. Backend-neutral method names never imply equal guarantees.
- **D-20:** Contract tests classify assertions as integrity, recovery, progress,
  or performance. Typed retryable contention is a valid progress outcome;
  safety tests still require no corrupt or mixed committed generation.
- **D-21:** Performance measurements remain separate statistical evidence and
  never become universal runtime deadlines or success requirements.
- **D-22:** Phase 5 qualifies backend behavior and the three reference
  topologies. Phase 8 owns the full supported-Python/platform/install/service
  matrix and final performance budgets, but it reuses rather than replaces the
  real-service suites created here.

### the agent's Discretion

- Exact class/module names for the PostgreSQL authority and S3 generation I/O
  adapter, provided each implements the existing narrow role contract and not a
  parallel lifecycle engine.
- PostgreSQL table/index layout, isolation level, and bounded retry defaults,
  provided exact CAS and typed outcomes meet D-15 through D-18.
- Exact test-run identifiers, environment-variable names, AWS region, and
  PostgreSQL namespace strategy, provided credentials remain external and
  cleanup is strictly test-owned.
- Multipart thresholds, chunk sizes, and page sizes within explicit bounded
  limits and without changing handler-owned payload formats.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Governing Product and Guarantee Contracts

- `docs/adr/0001-topology-specific-storage-guarantees.md` — Mandatory topology,
  integrity, recovery, progress, ACID, and race-loop stop conditions.
- `.planning/PROJECT.md` — Defines the BlobStore-first product, advertised
  backend scope, compatibility reset, and milestone constraints.
- `.planning/ROADMAP.md` § Phase 5 — Defines BACK-01/BACK-04/BACK-05 acceptance
  and the supported-topology qualification boundary.
- `.planning/REQUIREMENTS.md` — Authoritative Phase 5 requirements and explicit
  exclusions from universal Cartesian parity.
- `CONTEXT.md` — Canonical vocabulary for lifecycle authority, payload store,
  catalog, projection, and cache policy.

### Delivered Storage and Composition Contracts

- `.planning/phases/04-metadata-composition-and-topology-contracts/04-CONTEXT.md`
  — Locks one typed composition root, exact participant selection, capability
  minima, derived projections, and the pre-production compatibility reset.
- `.planning/phases/04-metadata-composition-and-topology-contracts/04-VALIDATION.md`
  — Records the final owned matrix and explicit PostgreSQL/S3 non-claims that
  Phase 5 must close without overstating other environments.
- `.planning/phases/04-metadata-composition-and-topology-contracts/04-SECURITY.md`
  — Verifies the role, cursor, containment, projection, and release-evidence
  threat boundaries inherited by backend expansion.
- `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md` — Locks
  immutable external effects, one transactional authority, attributable debt,
  explicit initialization, and qualified contention outcomes.
- `.planning/phases/02-canonical-storage-and-integrity-contract/02-CONTEXT.md` —
  Locks canonical manifests, handler-owned native formats, signed digests,
  committed-only reads, and typed integrity outcomes.
- `docs/STORAGE_INITIALIZATION.md` — Defines explicit initialization and
  maintenance boundaries for shared-worker topologies.
- `docs/CATALOG_AND_TOPOLOGY.md` — Documents current catalog roles,
  capabilities, projections, and unsupported topology handling.
- `docs/SECURITY.md` — Defines trusted application payloads and fail-closed
  parser, path, integrity, and credential boundaries.

### Codebase Maps

- `.planning/codebase/STACK.md` — Records boto3, SQLAlchemy, psycopg, Python,
  test, and optional-dependency constraints.
- `.planning/codebase/INTEGRATIONS.md` — Identifies AWS/PostgreSQL integration
  seams, credentials, current mocks, and earlier disconnected wiring.
- `.planning/codebase/ARCHITECTURE.md` — Maps lifecycle, composition, handler,
  payload, and cache boundaries; historical details must be checked against the
  Phase 4 qualified source.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `src/cacheness/storage/composition.py`: `RoleRegistry`, `StoreTopology`,
  structural role protocols, ownership tracking, `StoreCapabilities`, and
  pre-I/O capability validation are the one composition seam to extend.
- `src/cacheness/storage/blob_store.py`: already materializes generation I/O
  from the selected payload participant and delegates lifecycle sequencing to
  one engine.
- `src/cacheness/storage/lifecycle.py`: owns immutable creation, promotion,
  cleanup debt, reads, deletes, and reconciliation ordering; backend work must
  supply primitives to it rather than copy this sequence.
- `src/cacheness/storage/lifecycle_authority.py`: existing authority types and
  expectations define the transactional boundary a PostgreSQL implementation
  must satisfy and should be narrowed where the phase plan can do so safely.
- `src/cacheness/storage/backends/blob_backends.py`: filesystem and memory
  participants already expose `materialize_handler_io()` and semantic
  capabilities through the selected-payload path.
- `src/cacheness/storage/backends/s3_backend.py`: contains direct boto3 byte,
  streaming, HEAD, ETag, listing, and delete operations plus a locally tested
  role-registration seam, but does not yet implement qualified generation I/O.

### Established Patterns

- SQLite/filesystem and memory/memory are already the reference local
  topologies; Phase 5 should preserve their behavior while extracting a common
  contract matrix.
- Payloads are native handler files. Lifecycle code coordinates immutable
  generations and signed descriptors but never wraps payload bytes in a custom
  Cacheness container.
- Authority promotion is the visibility point. External payload presence and
  listings are evidence, never authority.
- Injected participants remain caller-owned unless ownership is explicitly
  transferred; construction failure unwinds only store-owned resources once.

### Integration Points

- Adapt `S3BlobBackend` to the `PayloadGenerationIOProvider` role and route
  exact generation I/O through the existing `BlobStore`/lifecycle engine.
- Introduce the PostgreSQL authority behind the authority role in
  `StoreTopology` and register it only after structural and live-service
  qualification.
- Extend capability composition with the exact remote sharing, durability,
  streaming, conditional-create, listing, and progress semantics proved by the
  active pair.
- Build bounded contract suites reusable across the three reference topologies,
  with separate live AWS/PostgreSQL fixtures and explicit unavailable evidence.

</code_context>

<specifics>
## Specific Ideas

- The smallest credible V1 support matrix is deliberately preferred over
  preserving every historical backend combination.
- Real-service evidence is a release truthfulness boundary: unavailable access
  blocks the claim but does not justify another coordination mechanism.
- The remote topology should look like the local engine from `BlobStore`'s
  perspective: different primitives and progress outcomes, same lifecycle
  owner and immutable-generation rules.

</specifics>

<deferred>
## Deferred Ideas

- Qualifying PostgreSQL/filesystem, SQLite/S3, other persistent/ephemeral
  mixtures, or additional S3-compatible services — later releases, each with
  its own explicit capability and service evidence.
- Bucket-versioning, object-lock, cross-region replication, and CDN policies —
  deployment features outside the V1 lifecycle contract.
- Complete cache policy, decorator, invalidation, and statistics composition —
  Phase 6.
- Stored-schema migration/rebuild execution — Phase 7.
- Full Python/platform/optional-install CI and final performance budgets —
  Phase 8.

</deferred>

---

*Phase: 5-payload-backends-and-supported-topology-qualification*
*Context gathered: 2026-09-08*
