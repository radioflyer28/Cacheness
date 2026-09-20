# Phase 4: Metadata Composition and Topology Contracts - Context

**Gathered:** 2026-09-07; compatibility scope revised during planning 2026-09-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Make `BlobStore` catalog metadata customizable without asking applications or
metadata adapters to implement storage lifecycle sequencing. Phase 4 defines a
small optional schema over the existing application metadata mapping, a portable
query/index contract, one typed composition root for payload/authority/projection
selection, and truthful capabilities for the active topology. Authoritative
application attributes commit with the blob descriptor. External ORM models and
indexes remain derived, rebuildable projections.

This phase narrows the catalog transaction seam around the lifecycle authority;
it does not introduce another lifecycle coordinator, expand the payload-backend
matrix, complete `UnifiedCache` policy migration, or strengthen guarantees beyond
the topology-specific contract in ADR 0001.

</domain>

<decisions>
## Implementation Decisions

### Catalog Schema

- **D-01:** An application metadata mapping remains the default surface. Callers may optionally attach a declared catalog schema; a schema is not required merely to store and reopen current-layout metadata. This preserves the useful data model, not the pre-Phase-4 implementation or constructor compatibility surface.
- **D-02:** When a schema is present, undeclared metadata fields are preserved and round-trip as opaque values. Only declared fields receive validation and portable query/index guarantees. They must not be silently dropped or retroactively rejected.
- **D-03:** Phase 4 provides a small Cacheness-native declarative schema. Field definitions cover name, supported value type, required/default behavior, validation, and query/index intent. The native contract may expose an adapter seam for external model libraries later, but Pydantic, dataclasses, SQLAlchemy models, or another framework do not define the canonical Phase 4 API. — **Reversibility:** costly — Applications and backend adapters will persist and consume these declarations, so replacing the public schema vocabulary would require compatibility adapters and schema migration.
- **D-04:** Within the new supported catalog format, additive schema evolution is readable using explicit missing/default semantics; incompatible future changes require explicit offline migration. Pre-Phase-4 development layouts are not required to reopen through runtime compatibility code and may return typed migration/rebuild-required evidence. Reads never rewrite metadata as an incidental schema upgrade. — **Reversibility:** costly — Future published schemas will depend on this explicit version/migration boundary even though the current pre-production cutover is allowed to break old layouts.

### Portable Query and Index Contract

- **D-05:** The initial portable query language supports typed equality, comparison/range, membership, and existence predicates over declared queryable fields. Multiple predicates compose with AND. Nested OR/NOT expressions and backend-native query strings are outside the portable contract.
- **D-06:** Portable pagination uses a stable opaque cursor with deterministic canonical ordering by entry key and generation identity plus bounded page sizes. Arbitrary field sorting is capability-specific and is not implied by the portable query contract. Offset pagination is not the resumability contract.
- **D-07:** Queryability and index intent are explicit parts of each declared field. Backends do not silently index every field or choose indexes as an undocumented semantic behavior. Secondary indexes are rebuildable acceleration state; they never become canonical read, delete, cleanup, or repair authority.
- **D-08:** A successful portable query returns results complete for the canonical catalog snapshot promised by that backend. It may use an index maintained in the same catalog transaction or a canonical scan. A stale or independently updated external index cannot silently answer a portable query with incomplete results. — **Reversibility:** costly — Callers may rely on catalog queries for inventory and administrative workflows, so weakening completeness later would invalidate published behavior.

### Backend Composition and Capabilities

- **D-09:** One typed store configuration is the only primary composition root. It selects the payload backend, catalog authority, optional projections, and minimum required capabilities as one validated topology. Overlapping pre-production constructors, backend overloads, duplicate factories, and legacy configuration names may be removed rather than preserved as adapters. — **Reversibility:** costly — This becomes the shared construction contract for direct `BlobStore` use and later `UnifiedCache` composition.
- **D-10:** A caller-injected backend instance remains the exact selected instance. Registered backend names resolve through the same construction path as built-ins. Supplying both an instance and a name for one role is an error; no selector silently wins and options are not merged into caller-owned instances.
- **D-11:** Every composed store exposes the actual semantic capabilities of its active payload/authority/projection pairing. Callers may request minimum guarantees; unmet requirements and inherently invalid pairings fail during construction. Callers are not required to choose a named topology tier, and Cacheness does not infer a stronger promise for them to depend upon implicitly.
- **D-12:** Resource ownership is explicit. Backends created by the composition root are store-owned and closed with the store. Injected instances are caller-owned by default and remain open, with an explicit option to transfer ownership. Hidden reference counting or resource sharing is not introduced. — **Reversibility:** costly — Close behavior is observable and shared injected resources depend on it.

### Derived Projections and External ORM Integration

- **D-13:** Phase 4 external ORM models and external indexes are derived-only integrations. Authoritative custom fields remain in the native catalog transaction with the blob descriptor. An external model cannot replace the catalog authority or participate in lifecycle sequencing in this phase.
- **D-14:** Derived projections synchronize by idempotently pulling bounded pages of committed canonical catalog state from a checkpoint. Notifications may prompt a refresh but are not the correctness source. Phase 4 does not require a background worker, synchronous-callback-only delivery, or exactly-once event protocol.
- **D-15:** Projection failure cannot roll back or revoke an already committed blob. The committed receipt exposes projection status. Ordinary best-effort refresh may warn; when a caller explicitly requests projection refresh, failure produces a typed committed-partial outcome that retains the canonical receipt and attribution of remaining derived work. — **Reversibility:** costly — This is the public failure boundary that prevents applications from treating a committed generation as absent.
- **D-16:** Projection rebuild is an explicit, capability-qualified operation. It builds isolated derived state from canonical catalog state and publishes that state only when complete. Each adapter reports whether it can catch up online or requires offline maintenance; Phase 4 promises neither mode universally. Normal reads and writes do not perform hidden projection repair.

### Pre-Production Compatibility Reset

- **D-17:** Cacheness has no production deployment to preserve. Phase 4 should delete superseded selection, metadata-backend, and catalog-layout paths when the new composition/catalog contract replaces them, rather than maintaining dual old/new behavior. Historical characterization tests remain evidence but may be retired or rewritten when they assert removed APIs. — **Source:** explicit user direction during Phase 4 planning, 2026-09-07. — **Reversibility:** one-way — Reintroducing the removed pre-production surface later would create a new compatibility contract and duplicate composition paths.
- **D-18:** Dropping current backward compatibility does not drop migration infrastructure. Persisted formats and schemas remain explicitly versioned; unsupported layouts fail without mutation; Phase 7 still delivers non-mutating inventory, offline migration, resumable copy-verify-switch, and confirmed rebuild tooling for future released versions. — **Source:** explicit user direction during Phase 4 planning, 2026-09-07.

### the agent's Discretion

- Exact public class, method, and module names for schema fields, query predicates,
  cursors, composition specifications, capability reports, and projection reports.
- The finite portable scalar/type set, missing-value representation, and validation
  error types, provided validation does not silently coerce values into a different
  catalog meaning.
- Default and maximum query page sizes and internal cursor encoding, provided the
  cursor is opaque, bounded, deterministic, and rejects incompatible versions.
- Internal schema fingerprint/version encoding and projection checkpoint storage,
  subject to authenticated canonical catalog state and the ADR stop conditions.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Governing Architecture and Product Scope

- `docs/adr/0001-topology-specific-storage-guarantees.md` — Mandatory integrity, recovery, topology, progress, and stop-condition guardrails; prevents projection/catalog work from becoming another coordinator.
- `.planning/PROJECT.md` — Defines BlobStore lifecycle ownership, UnifiedCache policy composition, backend coverage, compatibility, and reliability constraints.
- `.planning/ROADMAP.md` § Phase 4 — Defines this phase's goal, starting point, success criteria, and boundaries with Phases 3, 5, and 6.
- `.planning/REQUIREMENTS.md` — Authoritative Phase 4 requirements BACK-02, BACK-03, BACK-06, and BACK-07.
- `CONTEXT.md` — Canonical product vocabulary distinguishing blob storage, catalog authority, projections, and cache policy.

### Prior Locked Contracts

- `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md` — Locks one transactional lifecycle authority, native immutable payloads, JSON as projection, topology failures at construction, caller-owned injected resources, and Phase 4's catalog boundary.
- `.planning/phases/02-canonical-storage-and-integrity-contract/02-CONTEXT.md` — Locks authenticated committed descriptor state, native handler payload formats, typed integrity outcomes, and read ordering.
- `.planning/phases/01-compatibility-and-security-baseline/01-CONTEXT.md` — Locks public compatibility, containment, evidence preservation, and trusted-payload boundaries.
- `docs/phase3-direct-implementation-2026-09-06.md` — Records the delivered Phase 3 engine surface that Phase 4 extends rather than replaces.
- `docs/STORAGE_INITIALIZATION.md` — Defines explicit initialization and maintenance boundaries for supported shared-worker topologies.

### Codebase Maps

- `.planning/codebase/ARCHITECTURE.md` — Maps current lifecycle ownership, metadata composition seams, and the distinction between BlobStore and UnifiedCache.
- `.planning/codebase/INTEGRATIONS.md` — Documents SQLite, PostgreSQL, S3, and optional dependency constraints relevant to backend composition.
- `.planning/codebase/STACK.md` — Records supported runtimes, SQLAlchemy/driver dependencies, and current installation constraints.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `src/cacheness/storage/lifecycle_authority.py`: `LifecycleAuthority`,
  `AuthorityCapabilities`, `EntrySnapshot`, and generation expectations provide the
  existing authority boundary. Phase 4 should add a narrow catalog transaction/query
  surface rather than copy its complete lifecycle transition protocol into metadata
  adapters.
- `src/cacheness/storage/blob_store.py`: `put_entry`, `get_entry_info`, `open_entry`,
  `get_metadata`, `update_metadata`, and `list` identify the useful same-generation
  BlobStore catalog seam. Preserve their lifecycle meaning where the clean API retains
  them, but do not carry obsolete constructor/selector shapes solely for compatibility.
- `src/cacheness/storage/backends/__init__.py`: metadata and blob registries expose
  named construction, but duplicate ABC/factory paths may be consolidated or removed.
  The new role registry/composition root is the only supported construction path.
- `src/cacheness/config.py`: `LifecycleAuthorityTopology` and `CacheConfig` already
  express minimum local-authority requirements. Generalize semantic capability
  reporting to the active backend pair without relying on backend-name allowlists.

### Established Patterns

- The SQLite lifecycle authority is canonical for the qualified local persistent
  topology; the memory authority is process-local and ephemeral.
- Application metadata already round-trips inside authenticated entry descriptors.
  Phase 4 adds optional declaration, validation, and querying without replacing the
  mapping or adding a payload wrapper.
- JSON export is a compatibility projection after authority commit. Its failure is
  non-authoritative and cannot revoke a valid generation.
- Store initialization precedes shared workers, and authority schema migration is an
  explicit maintenance operation rather than an implicit concurrent-first-use race.

### Integration Points

- Replace `BlobStore._select_projection_backend()` and
  `BlobStore._create_lifecycle_authority()` role overloading with the one typed,
  ownership-aware composition path; delete superseded entry points instead of adapting
  them when they duplicate the new root.
- Narrow the current broad `MetadataBackend`/`LifecycleAuthority` overlap: catalog
  adapters supply atomic descriptor/application-metadata transactions and portable
  queries; they do not reproduce prepare/promote/cleanup lifecycle sequencing.
- Extend entry descriptors and authority schemas to retain schema identity, declared
  catalog values, and transactional query-index state without changing handler-owned
  payload bytes.
- Replace `BlobStore.list(..., metadata_filter=dict)` with the typed, cursor-based
  portable query surface; an equality convenience need only remain if it fits the clean
  API without an alternate query implementation.
- Replace legacy custom-metadata and SQLAlchemy session hooks with the derived projection
  adapter instead of exposing backend sessions as canonical BlobStore transactions.

</code_context>

<specifics>
## Specific Ideas

- All four recommended design directions were selected without modification: a
  conservative native schema, a deliberately bounded portable query language, an
  explicit composition root, and rebuildable pull-based projections.
- The intended product remains a reliable blob store whose engine later supports
  caching; a store need not serve cache and non-cache roles in one live namespace.

</specifics>

<deferred>
## Deferred Ideas

- A transactional external ORM extension that joins the native authority database
  transaction is not part of Phase 4. Reconsider only with a concrete use case that
  cannot be represented by native authoritative fields plus a derived projection.
- Full PostgreSQL authority and filesystem/memory/S3 payload-pair qualification belongs
  to Phase 5.
- Complete `UnifiedCache` policy delegation, statistics, invalidation, and compatibility
  acceptance belongs to Phase 6.
- Stored-format and schema migration execution belongs to Phase 7. The tooling remains
  required for future released versions, but the supported source-version window may
  exclude pre-Phase-4 development formats.

</deferred>

---

*Phase: 4-metadata-composition-and-topology-contracts*
*Context gathered: 2026-09-07*
