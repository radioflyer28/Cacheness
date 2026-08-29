# Requirements: Cacheness

**Defined:** 2026-08-29
**Core Value:** Applications can store and retrieve data reliably through one backend-neutral lifecycle, with caching policy layered above storage without compromising integrity or cleanup correctness.

## v1 Requirements

### Storage Lifecycle

- [ ] **STOR-01**: Every stored entry uses one versioned canonical manifest across all supported backends.
- [ ] **STOR-02**: Normal reads expose only committed entry generations.
- [ ] **STOR-03**: A write exposes either the previous complete generation or the new complete generation, never partial payload or metadata state.
- [ ] **STOR-04**: A failed write preserves the last valid generation and leaves any residue detectable and recoverable.
- [ ] **STOR-05**: Overwrite, delete, clear, and close operations are idempotent and clean up both payload and metadata state.
- [ ] **STOR-06**: Operators can run dry-run and resumable reconciliation that detects inconsistent state and safely repairs, quarantines, or reports it.
- [ ] **STOR-07**: Same-key races have deterministic outcomes through per-key coordination and backend generation checks without globally serializing distinct keys.
- [ ] **STOR-08**: Direct `BlobStore` operations distinguish missing, corrupt, conflict, and backend failures through typed results or exceptions.

### Backend Unification

- [ ] **BACK-01**: Filesystem, memory, and S3 payload backends implement one payload lifecycle contract.
- [ ] **BACK-02**: JSON, memory, SQLite, and PostgreSQL metadata backends implement one metadata lifecycle contract.
- [ ] **BACK-03**: Caller-injected and registered backend implementations remain selected rather than being silently replaced by configuration defaults.
- [ ] **BACK-04**: Every supported combination in the three-payload by four-metadata backend matrix passes the common lifecycle contract.
- [ ] **BACK-05**: PostgreSQL and AWS S3 behavior is verified with real-service integration coverage; compatible S3 services are supported only where explicitly verified.
- [ ] **BACK-06**: Backends expose durability, process/host sharing, compare-and-swap, streaming, and listing capabilities, and configurations cannot claim guarantees their topology cannot provide.

### Cache Composition

- [ ] **CACH-01**: `UnifiedCache` delegates every payload-plus-metadata lifecycle operation to `BlobStore`.
- [ ] **CACH-02**: `UnifiedCache` exclusively owns cache keying, TTL, eviction, invalidation, statistics, and decorator-facing policy.
- [ ] **CACH-03**: Every invalidation path, including TTL, eviction, predicate, decorator, single-key, and global clear, removes the complete stored entry through the storage lifecycle.
- [ ] **CACH-04**: A cached `None` value remains distinguishable from a cache miss.
- [ ] **CACH-05**: Cache statistics distinguish absent, expired, corrupt, conflict, and backend-error outcomes while preserving compatible aggregate counters.
- [ ] **CACH-06**: Existing supported public cache imports, constructors, aliases, configuration names, decorators, and result behavior remain callable through compatibility adapters.
- [x] **CACH-07**: `SqlCache` remains a separate subsystem and retains representative import and behavioral regression coverage.

### Security and Integrity

- [x] **SECU-01**: Filesystem reads, writes, deletes, and listings reject traversal, absolute-path, drive, UNC, and symlink escapes from the configured root.
- [x] **SECU-02**: Structured metadata uses typed safe parsers, and metadata-controlled `eval` is eliminated.
- [ ] **SECU-03**: Configured manifest authenticity and payload integrity are verified before deserialization.
- [ ] **SECU-04**: Required signing fails closed when keys, signatures, permissions, or signer configuration are missing or invalid.
- [ ] **SECU-05**: Signed manifests bind critical locator, handler/type, format, and lifecycle-generation fields.
- [ ] **SECU-06**: Metadata query fields are validated and safely constructed rather than interpolated into backend queries.
- [ ] **SECU-07**: Documentation states the trusted-application-payload boundary and the risks and configuration requirements of unsafe serializers.
- [ ] **SECU-08**: `BlobStore` raises a typed integrity exception for corrupt or invalidly signed entries, while `UnifiedCache` may translate it into a separately recorded cache miss.

### Migration and Compatibility

- [x] **MIGR-01**: Public imports, constructors, configuration names, registries, decorators, aliases, exceptions, and result behavior have characterization tests before ownership changes.
- [ ] **MIGR-02**: Stored metadata schemas and payload formats have explicit, independently versioned identifiers.
- [ ] **MIGR-03**: Migration begins with a non-mutating inventory and produces both human-readable and machine-readable plans.
- [ ] **MIGR-04**: Same-backend format and schema migrations use resumable copy-verify-switch semantics.
- [ ] **MIGR-05**: Interrupted migrations resume safely without losing the only valid copy of an entry.
- [ ] **MIGR-06**: Incompatible formats and cross-backend moves have an explicit, confirmed rebuild path rather than implicit deletion or universal physical migration.
- [ ] **MIGR-07**: Unknown future formats fail explicitly rather than being guessed, rewritten, or silently deleted.

### Delivery Confidence

- [ ] **QUAL-01**: A clean minimal wheel installation imports every guaranteed public symbol and completes a memory-backed round trip.
- [ ] **QUAL-02**: Each advertised optional dependency group installs and imports independently.
- [ ] **QUAL-03**: CI covers supported Python versions, backend contracts, lint policy, coverage, packaging, PostgreSQL, and AWS S3 integration.
- [ ] **QUAL-04**: Deterministic fault-injection, race, and crash-recovery tests cover lifecycle commit boundaries.
- [ ] **QUAL-05**: Lifecycle and cache-policy code meets targeted statement and branch coverage thresholds established by the project.
- [ ] **QUAL-06**: Checked-in benchmarks establish correctness-aware final performance budgets after lifecycle behavior stabilizes.
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
| Universal physical migration between backend combinations | V1 supports same-backend format/schema migration and an explicit rebuild path for incompatible or cross-backend data. |
| New backend families | Reliability across every already-advertised built-in backend is the current priority. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| STOR-01 | Phase 2 | Pending |
| STOR-02 | Phase 2 | Pending |
| STOR-03 | Phase 3 | Pending |
| STOR-04 | Phase 3 | Pending |
| STOR-05 | Phase 3 | Pending |
| STOR-06 | Phase 3 | Pending |
| STOR-07 | Phase 3 | Pending |
| STOR-08 | Phase 2 | Pending |
| BACK-01 | Phase 5 | Pending |
| BACK-02 | Phase 4 | Pending |
| BACK-03 | Phase 4 | Pending |
| BACK-04 | Phase 5 | Pending |
| BACK-05 | Phase 5 | Pending |
| BACK-06 | Phase 4 | Pending |
| CACH-01 | Phase 6 | Pending |
| CACH-02 | Phase 6 | Pending |
| CACH-03 | Phase 6 | Pending |
| CACH-04 | Phase 6 | Pending |
| CACH-05 | Phase 6 | Pending |
| CACH-06 | Phase 6 | Pending |
| CACH-07 | Phase 1 | Complete |
| SECU-01 | Phase 1 | Complete |
| SECU-02 | Phase 1 | Complete |
| SECU-03 | Phase 2 | Pending |
| SECU-04 | Phase 2 | Pending |
| SECU-05 | Phase 2 | Pending |
| SECU-06 | Phase 1 | Pending |
| SECU-07 | Phase 1 | Pending |
| SECU-08 | Phase 2 | Pending |
| MIGR-01 | Phase 1 | Complete |
| MIGR-02 | Phase 2 | Pending |
| MIGR-03 | Phase 7 | Pending |
| MIGR-04 | Phase 7 | Pending |
| MIGR-05 | Phase 7 | Pending |
| MIGR-06 | Phase 7 | Pending |
| MIGR-07 | Phase 2 | Pending |
| QUAL-01 | Phase 8 | Pending |
| QUAL-02 | Phase 8 | Pending |
| QUAL-03 | Phase 8 | Pending |
| QUAL-04 | Phase 8 | Pending |
| QUAL-05 | Phase 8 | Pending |
| QUAL-06 | Phase 8 | Pending |
| QUAL-07 | Phase 8 | Pending |

**Coverage:**

- v1 requirements: 43 total
- Mapped to phases: 43
- Unmapped: 0 ✓

---
*Requirements defined: 2026-08-29*
*Last updated: 2026-08-29 after roadmap creation*
