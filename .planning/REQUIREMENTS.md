# Requirements: Cacheness

**Defined:** 2026-08-29
**Core Value:** Applications can store and retrieve data reliably through one backend-neutral lifecycle, with caching policy layered above storage without compromising integrity or cleanup correctness.

## v1 Requirements

### Storage Lifecycle

Phase 3 completion below is scoped to the qualified initialized local SQLite/filesystem topology and the declared single-process memory-authority tests. It is not qualification of every backend pairing or native Windows. Future pairings extend these invariants through BACK-04; they do not reopen completed local integration by default.

- [x] **STOR-01**: Every stored entry uses one versioned canonical manifest across all supported backends.
- [x] **STOR-02**: Normal reads expose only committed entry generations.
- [x] **STOR-03**: A write exposes either the previous complete generation or the new complete generation, never partial payload or metadata state.
- [x] **STOR-04**: A failed write preserves the last valid generation and leaves any residue detectable and recoverable; post-commit cleanup failure preserves the newly committed generation and reports debt rather than implying rollback.
- [x] **STOR-05**: Overwrite, delete, clear, and close converge idempotently; incomplete external cleanup remains attributable debt and is reclaimed when required resources are available without deleting a current generation.
- [x] **STOR-06**: Operators can run dry-run and resumable reconciliation that detects inconsistent state and safely repairs, quarantines, or reports it.
- [x] **STOR-07**: Within a declared initialized topology, same-key contention preserves integrity through backend generation checks and yields success, conflict, or typed retryable failure; payload work on distinct keys is not globally serialized (short SQLite write transactions may serialize).
- [x] **STOR-08**: Direct `BlobStore` operations distinguish missing, corrupt, conflict, and backend failures through typed results or exceptions.

### Backend Unification

- [ ] **BACK-01**: Filesystem, memory, and S3 payload adapters supply storage operations to the shared BlobStore engine and satisfy applicable immutable-generation and recovery contracts at their declared capability tier; they do not duplicate lifecycle sequencing.
- [x] **BACK-02**: JSON, memory, SQLite, and PostgreSQL metadata implementations have explicit authority/projection roles through one composition contract. Narrow transactional catalog adapters before expansion; a derived JSON view does not become an independent lifecycle authority.
- [x] **BACK-03**: Caller-injected and registered backend implementations remain selected rather than being silently replaced by configuration defaults.
- [ ] **BACK-04**: Every explicitly supported pairing of advertised payload and metadata backends passes the lifecycle contract at its declared capability tier; unsupported Cartesian combinations are documented and rejected rather than silently downgraded.
- [ ] **BACK-05**: PostgreSQL and AWS S3 behavior is verified with real-service integration coverage; compatible S3 services are supported only where explicitly verified.
- [x] **BACK-06**: Backends expose durability, process/host sharing, compare-and-swap, streaming, and listing capabilities, and configurations cannot claim guarantees their topology cannot provide.
- [x] **BACK-07**: Direct `BlobStore` users can store, validate, query, and update application-defined catalog metadata without implementing a lifecycle backend; supported fields/operators and transactional limits are explicit. Extend the existing mapping/entry interface rather than replacing the engine. Authoritative metadata commits with the blob descriptor; external indexes or ORM links are explicitly derived unless they join that same transaction, with consistency and partial-failure behavior stated.

### Cache Composition

- [ ] **CACH-01**: `UnifiedCache` delegates every payload-plus-authoritative-catalog lifecycle operation to the supported BlobStore entry interface; separate derived adapters never authorize canonical repair or deletion. Complete remaining paths/topologies over the delivered Phase 3 local integration.
- [ ] **CACH-02**: `UnifiedCache` exclusively owns cache keying, TTL, eviction, invalidation, statistics, and decorator-facing policy.
- [ ] **CACH-03**: Every invalidation path, including TTL, eviction, predicate, decorator, single-key, and global clear, removes the complete stored entry through the storage lifecycle.
- [ ] **CACH-04**: A cached `None` value remains distinguishable from a cache miss.
- [ ] **CACH-05**: Cache statistics distinguish absent, expired, corrupt, conflict, and backend-error outcomes through one documented aggregate/result model; legacy counter shapes need not be preserved.
- [ ] **CACH-06**: The milestone publishes one coherent cache import, constructor, configuration, decorator, and result surface over `BlobStore`. Pre-production aliases and overlapping constructors may be removed instead of routed through compatibility adapters; explicit initialization, optional-export warnings, and requested-external-metadata committed-partial errors remain required semantic contracts.
- [x] **CACH-07**: `SqlCache` remains a separate subsystem and retains representative import and behavioral regression coverage.

### Security and Integrity

- [x] **SECU-01**: Filesystem reads, writes, deletes, and listings reject traversal, absolute-path, drive, UNC, and symlink escapes from the configured root.
- [x] **SECU-02**: Structured metadata uses typed safe parsers, and metadata-controlled `eval` is eliminated.
- [x] **SECU-03**: Configured manifest authenticity and payload integrity are verified before deserialization.
- [x] **SECU-04**: Required signing fails closed when keys, signatures, permissions, or signer configuration are missing or invalid.
- [x] **SECU-05**: Signed manifests bind critical locator, handler/type, format, and lifecycle-generation fields.
- [x] **SECU-06**: Metadata query fields are validated and safely constructed rather than interpolated into backend queries.
- [x] **SECU-07**: Documentation states the trusted-application-payload boundary and the risks and configuration requirements of unsafe serializers.
- [x] **SECU-08**: `BlobStore` raises a typed integrity exception for corrupt or invalidly signed entries, while `UnifiedCache` may translate it into a separately recorded non-destructive cache miss. Derived metadata corruption does not revoke a valid canonical generation; direct observations of malformed projection fields still fail closed.

### Migration and Compatibility

- [x] **MIGR-01**: The pre-refactor public surface was characterized before ownership changes. This corpus is historical evidence, not a requirement to retain every pre-production API after the 2026-09-07 compatibility reset.
- [x] **MIGR-02**: Stored metadata schemas and payload formats have explicit, independently versioned identifiers.
- [ ] **MIGR-03**: Migration begins with a non-mutating inventory and produces both human-readable and machine-readable plans.
- [ ] **MIGR-04**: Supported same-backend format and schema migrations use explicit offline resumable copy-verify-switch semantics with workers stopped; ordinary opens and initialize do not silently upgrade schemas.
- [ ] **MIGR-05**: Interrupted offline migrations resume safely from explicit maintenance evidence without losing the only valid copy of an entry or adopting unexplained incomplete catalogs. Preserve signing material; derived-index reconstruction is not another canonical cutover authority.
- [ ] **MIGR-06**: Incompatible formats and cross-backend moves have an explicit, confirmed rebuild path rather than implicit deletion or universal physical migration.
- [x] **MIGR-07**: Unknown future formats fail explicitly rather than being guessed, rewritten, or silently deleted.

### Delivery Confidence

- [ ] **QUAL-01**: A clean minimal wheel installation imports every guaranteed public symbol and completes a memory-backed round trip.
- [ ] **QUAL-02**: Each advertised optional dependency group installs and imports independently.
- [ ] **QUAL-03**: CI covers supported Python versions, backend contracts, lint policy, coverage, packaging, PostgreSQL, and AWS S3 integration.
- [ ] **QUAL-04**: Carry forward the finite Phase 3 integrity/recovery regressions and cover named commit boundaries for each new supported topology with deterministic fault/crash tests. Shared-worker fixtures initialize first; success/conflict/typed retryable outcomes are distinguished from corruption. No universal scheduling guarantee or automatic repeated race-fix loop is required.
- [ ] **QUAL-05**: Lifecycle and cache-policy code meets targeted statement and branch coverage thresholds established by the project.
- [ ] **QUAL-06**: Checked-in benchmarks establish final performance budgets and distributions for named workloads/environments after lifecycle behavior stabilizes. Benchmark thresholds do not become runtime deadlines or strengthen public progress/atomicity promises.
- [ ] **QUAL-07**: Supported inventory and aggregate operations avoid unbounded memory use and accidental N+1 backend calls.

## v2 Requirements

Deferred to future releases and not included in the current roadmap.

### Extensibility and Scale

- **EXTN-01**: Cache policies can be supplied through a general plugin contract layered over `BlobStore`.
- **EXTN-02**: Additional payload and metadata backend families can be added through supported extension contracts.
- **EXTN-03**: Applications can use native asynchronous storage and cache APIs.
- **EXTN-04**: Deployments can opt into deduplication and reference-counted payload ownership.
- **EXTN-05**: Multi-host caches can opt into distributed invalidation and coherence mechanisms.

## Out of Scope

| Feature | Reason |
|---------|--------|
| `SqlCache` redesign or merger into `BlobStore` | Its table-oriented pull-through lifecycle is separate; this release protects it with regression coverage. |
| Safe deserialization of hostile pickle or dill payloads | These formats execute code by design; v1 documents and enforces a trusted-application-payload boundary. |
| Runtime compatibility with pre-production APIs and development-only stored layouts | No production deployment depends on them. Unsupported layouts fail explicitly; migration/rebuild tooling remains for future versioned releases. |
| Universal physical migration between backend combinations | V1 supports same-backend format/schema migration and an explicit rebuild path for incompatible or cross-backend data. |
| New backend families | Reliability across every already-advertised built-in backend is the current priority. |

## Traceability

Phase 3 delivered an early local composition proof for CACH-01/02/03/06 and existing mapping metadata for BACK-07 at qualified commit `5282dca`; see [the direct implementation ledger](../docs/phase3-direct-implementation-2026-09-06.md). It does not mark those full requirements complete: Phase 4 owns catalog customization and Phase 6 owns the coherent cache-policy API. Cache instances must use BlobStore as their engine; sharing a live namespace with non-cache stores is not required. On 2026-09-07 the user approved a pre-production compatibility reset: historical characterization remains evidence, but current development-only APIs/layouts need not retain runtime adapters. Explicit schema/format identification and future migration/rebuild tooling remain required. Phase 3 completion is direct primary-agent qualification under the user's process override, not a fresh independent GSD verifier verdict.

On 2026-09-08 the user approved moving `BACK-05`'s non-substitutable real
PostgreSQL/Amazon-S3 qualification gate from Phase 5 to Phase 8. Phase 5 retains
the candidate implementation, deterministic contracts, frozen real-service
suites, fail-closed runner, and truthful `UNAVAILABLE` evidence, but does not
claim release qualification. Phase 8 must run the gate unchanged before marking
`BACK-05` complete or advertising PostgreSQL/Amazon-S3 as a supported release
topology.

| Requirement | Phase | Status |
|-------------|-------|--------|
| STOR-01 | Phase 2 | Complete |
| STOR-02 | Phase 2 | Complete |
| STOR-03 | Phase 3 | Complete — qualified local scope |
| STOR-04 | Phase 3 | Complete — qualified local scope |
| STOR-05 | Phase 3 | Complete — qualified local scope |
| STOR-06 | Phase 3 | Complete — qualified local scope |
| STOR-07 | Phase 3 | Complete — qualified local scope |
| STOR-08 | Phase 2 | Complete |
| BACK-01 | Phase 5 | Pending |
| BACK-02 | Phase 4 | Complete |
| BACK-03 | Phase 4 | Complete |
| BACK-04 | Phase 5 | Pending |
| BACK-05 | Phase 8 | Pending — real-service gate moved intact |
| BACK-06 | Phase 4 | Complete |
| BACK-07 | Phase 4 | Complete |
| CACH-01 | Phase 6 | Pending |
| CACH-02 | Phase 6 | Pending |
| CACH-03 | Phase 6 | Pending |
| CACH-04 | Phase 6 | Pending |
| CACH-05 | Phase 6 | Pending |
| CACH-06 | Phase 6 | Pending |
| CACH-07 | Phase 1 | Complete |
| SECU-01 | Phase 1 | Complete |
| SECU-02 | Phase 1 | Complete |
| SECU-03 | Phase 2 | Complete |
| SECU-04 | Phase 2 | Complete |
| SECU-05 | Phase 2 | Complete |
| SECU-06 | Phase 1 | Complete |
| SECU-07 | Phase 1 | Complete |
| SECU-08 | Phase 2 | Complete |
| MIGR-01 | Phase 1 | Complete |
| MIGR-02 | Phase 2 | Complete |
| MIGR-03 | Phase 7 | Pending |
| MIGR-04 | Phase 7 | Pending |
| MIGR-05 | Phase 7 | Pending |
| MIGR-06 | Phase 7 | Pending |
| MIGR-07 | Phase 2 | Complete |
| QUAL-01 | Phase 8 | Pending |
| QUAL-02 | Phase 8 | Pending |
| QUAL-03 | Phase 8 | Pending |
| QUAL-04 | Phase 8 | Pending |
| QUAL-05 | Phase 8 | Pending |
| QUAL-06 | Phase 8 | Pending |
| QUAL-07 | Phase 8 | Pending |

**Coverage:**

- v1 requirements: 44 total
- Mapped to phases: 44
- Unmapped: 0 ✓

---
*Requirements defined: 2026-08-29*
*Last updated: 2026-09-08 after moving BACK-05 real-service release qualification to Phase 8*
