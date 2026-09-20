# Phase 7: Explicit Migration and Rebuild Cutover - Context

**Gathered:** 2026-09-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 7 delivers an explicit, offline operator workflow to inspect, plan, stage,
verify, activate, resume, roll back, finalize, and purge supported versioned
BlobStore migrations. It also provides an explicit scoped rebuild path for
unsupported layouts, incompatible payload contracts, and cross-backend moves.
Ordinary construction and initialization never migrate, rebuild, adopt, or
delete a store.

The current canonical post-refactor BlobStore layout is the first supported
release baseline. Phase 7 does not promise physical migration from older
pre-refactor or transitional Cacheness layouts into the upcoming release; those
sources may be inspected and classified for explicit rebuild. The durable
migration promise begins with the upcoming release as a source for future
releases.

</domain>

<decisions>
## Implementation Decisions

### Supported version window
- **D-01:** The current canonical post-refactor BlobStore layout becomes the first supported release baseline. Older pre-refactor and transitional layouts are inspectable but rebuild-only; Phase 7 must not create compatibility readers merely to migrate those development layouts.
- **D-02:** Each future release directly supports its current layout and the immediately previous released layout. Older released stores advance through successive declared migration steps rather than receiving indefinite direct migration support. — **Reversibility:** one-way — Once published, applications may depend on the immediately-previous-release migration promise; narrowing it would break the release migration contract.
- **D-03:** Migration eligibility is defined by an explicit compatibility matrix across independently versioned persisted contracts: overall store layout, topology-specific lifecycle-authority schema, signed manifest schema, handler-owned payload contract, and application catalog schema where transformation is required. A top-level store version never implies compatibility for every component.
- **D-04:** Inspection remains available outside the supported source window and reports the source as rebuild-only. Mutation through the migration path refuses, and there is no force override for unsupported migration.
- **D-05:** Phase 7 primarily establishes migration from the upcoming release to later releases, not migration from historical Cacheness releases into the upcoming release. If Phase 7 does not itself require a persisted layout change, it must not manufacture one merely to demonstrate migration.

### Plan and incompatibility behavior
- **D-06:** Non-mutating inspection produces aggregate entry counts and bytes plus an entry-complete classification. Every canonical entry is classified as migratable, rebuildable, blocked, or unverifiable with a stable reason.
- **D-07:** A supported same-backend migration is a whole-store operation. If any canonical entry cannot be copied and verified, candidate publication is blocked; migration never silently publishes a partial replacement.
- **D-08:** Incompatible entries and unsupported or cross-backend sources use a separately generated rebuild plan. Rebuild excludes nothing by default. Any exclusion requires a new plan naming exact keys or explicitly defined categories, counts, bytes, and reasons, followed by confirmation. — **Reversibility:** one-way — Confirmed exclusions can intentionally omit the only reconstructable form of data from the rebuilt store.
- **D-09:** Every mutating action revalidates the inspected source identity and revision fingerprint. A changed source makes the plan stale and requires reinspection; the tool never refreshes or overrides a stale plan automatically.
- **D-10:** Native payload formats remain owned by their handlers and libraries. Phase 7 does not invent a universal NPZ, Blosc2, Parquet, pickle, or dill converter. A payload transformation is supported only through an explicitly declared handler version edge; otherwise the entry follows the confirmed rebuild path.
- **D-11:** Unknown but authenticated catalog attributes are preserved rather than discarded. Derived indexes and projections are rebuilt separately after canonical cutover and never determine payload migration eligibility or publication.

### Cutover and retirement
- **D-12:** Migration uses distinct offline actions: stage/copy, verify, then separately and explicitly activate the verified candidate. Activation is never inferred merely because a candidate exists.
- **D-13:** Rollback to the prior verified store is allowed only while workers remain stopped after activation. Once writers restart, rollback eligibility ends because reverting could discard newly acknowledged writes.
- **D-14:** The prior valid store is retained until explicit operator action. No timer, constructor, background cleanup, cache eviction, or ordinary reconciliation may delete migration recovery material.
- **D-15:** `finalize` records operator acceptance of the activated store and permanently ends rollback eligibility. Physical deletion is a later, separately confirmed, idempotent purge. — **Reversibility:** one-way — Finalization deliberately removes the supported path back to the prior store, and purge may destroy its only remaining physical copy.
- **D-16:** Cleanup remains an external post-publication effect. A purge failure is reported as retryable cleanup work and cannot retroactively redefine a successful canonical activation.

### Resume and operator evidence
- **D-17:** Every maintenance run uses an operator-supplied work directory separate from both source and destination stores. Its evidence coordinates only the offline maintenance workflow; ordinary BlobStore operations never consult it and it is not a second lifecycle authority.
- **D-18:** Canonical JSON is the authoritative plan and run-evidence representation. The human-readable report is rendered from the same structured model, including stable reason codes, source and destination identities, counts, bytes, intended actions, and state.
- **D-19:** Resume requires an explicit run ID and evidence path. It revalidates already completed outputs and continues idempotently only from authenticated or otherwise verifiable recorded steps; it does not search for a latest run or infer progress from incidental blob presence.
- **D-20:** Missing, corrupt, or mismatched maintenance evidence fails closed with recovery diagnostics. The tool does not adopt an unexplained candidate or incomplete catalog and requires reinspection or a narrowly defined evidence-recovery action.
- **D-21:** `abort` may remove only an unactivated candidate proven to be owned by that run. It preserves the source and audit evidence. Once activation occurs, abort refuses and directs the operator to the offline rollback or finalize workflow.
- **D-22:** Existing signing identity/material is preserved through the configured key-provider boundary and is never serialized into plans, reports, or run logs. If required signing material cannot be obtained or verified, migration blocks rather than re-signing implicitly.

### the agent's Discretion
None of the discussed behavior was delegated. Planning retains ordinary implementation discretion over public symbol names, internal module boundaries, bounded batch sizes, and topology-specific publication primitives, provided every decision and ADR guardrail above remains observable and testable.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Scope and acceptance
- `.planning/PROJECT.md` — Defines the pre-production compatibility reset, BlobStore ownership, explicit future migration discipline, and backend/security/reliability constraints.
- `.planning/ROADMAP.md` — Defines the Phase 7 boundary and success criteria for offline copy-verify-switch migration and explicit rebuild.
- `.planning/REQUIREMENTS.md` — Defines MIGR-03 through MIGR-06 and the completed version-identification prerequisites.

### Lifecycle and topology guardrails
- `docs/adr/0001-topology-specific-storage-guarantees.md` — Mandatory source of truth for one lifecycle authority, topology-specific guarantees, external-effect reconciliation, and the stop conditions against adding coordination machinery.
- `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md` — Establishes immutable generations, authority promotion, deterministic recovery, cleanup debt, and the bounded-outcome vocabulary Phase 7 must reuse.
- `.planning/phases/05-payload-backends-and-supported-topology-qualification/05-CONTEXT.md` — Establishes qualified topology boundaries, explicit initialization, remote-authority constraints, and the Phase 8 deferral for live PostgreSQL/S3 qualification.

### Persisted contracts and composition
- `.planning/phases/02-canonical-storage-and-integrity-contract/02-CONTEXT.md` — Establishes signed canonical manifests, independently versioned persisted contracts, and fail-closed integrity behavior.
- `.planning/phases/04-metadata-composition-and-topology-contracts/04-CONTEXT.md` — Establishes customizable catalog schemas, preservation of undeclared fields, one composition root, and derived-only projection rebuild semantics.
- `.planning/phases/06-unifiedcache-policy-composition/06-CONTEXT.md` — Establishes that UnifiedCache is policy over one BlobStore and may not create a parallel lifecycle or migration path.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/cacheness/storage/blob_store.py`: exposes explicit initialization, canonical manifest authentication, legacy evidence inspection, canonical catalog paging, and isolated projection rebuild entry points.
- `src/cacheness/storage/catalog.py`: defines `STORE_FORMAT_VERSION`, catalog schemas and fingerprints, canonical pagination, and validation suitable for bounded inventory.
- `src/cacheness/storage/manifest.py`: defines current store, manifest, SQLite-schema, and payload version fields plus signed canonical manifest verification.
- `src/cacheness/storage/sqlite_lifecycle_authority.py`: provides the local transactional authority, explicit schema validation, canonical scans, immutable-generation lifecycle state, and current SQLite schema identifier.
- `src/cacheness/storage/backends/postgresql_lifecycle_authority.py`: provides the remote transactional authority and an independently versioned PostgreSQL authority schema/capability marker.
- `src/cacheness/storage/projections.py`: already isolates derived projection rebuild and publication from canonical lifecycle authority.
- `src/cacheness/handlers.py`: exposes handler-owned payload-format declarations and contract resolution; it is the only appropriate seam for declared payload transformations.

### Established Patterns
- Authority promotion is the sole visibility point; filesystem paths, object listings, migration evidence, and projections are never lifecycle authority.
- Payloads are immutable generations created before promotion. Destructive cleanup happens after promotion and remains retryable.
- Unsupported persisted versions and unsafe or unauthenticated records fail closed; ordinary initialization validates but does not upgrade.
- Catalog scans and reconciliation are bounded and cursor-driven. Migration inventory and copying should retain bounded work without turning a performance budget into a correctness deadline.
- Projection refresh/rebuild publishes from isolated destinations and is derived-only. Phase 7 should invoke or schedule it after canonical migration rather than include it in canonical cutover.

### Integration Points
- Add a dedicated offline maintenance boundary above `BlobStore` and its lifecycle authority; do not widen `UnifiedCache` or ordinary constructors with migration behavior.
- Inventory reads topology identity, authority/catalog versions, canonical manifests, payload contracts, byte counts, and signing availability without mutation.
- Same-backend migration delegates topology-specific transactional publication to the single selected authority while treating payload copy and purge as verifiable external effects.
- Rebuild reads only authenticated entries through declared source handlers, writes through the destination BlobStore lifecycle, and carries exact catalog values and signing identity subject to the confirmed plan.
- PostgreSQL/S3 live-service qualification remains Phase 8 work. Phase 7 defines and tests the contract deterministically without upgrading unavailable environments into support claims.

</code_context>

<specifics>
## Specific Ideas

- The operator flow should read conceptually as `inspect -> plan -> stage -> verify -> activate`, followed while still offline by either `rollback` or `finalize`; `purge` is a later explicit action.
- Human and machine plans must be two renderings of one model, not separately maintained outputs.
- The migration system must distinguish Cacheness contract versions from native library formats. NPZ, Blosc2, Parquet, pickle, and dill are handler payload contracts, not a custom Cacheness container format.
- Historical layouts may still yield useful non-mutating inventory and rebuild diagnostics without gaining a compatibility or physical-migration promise.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 7-explicit-migration-and-rebuild-cutover*
*Context gathered: 2026-09-09*
