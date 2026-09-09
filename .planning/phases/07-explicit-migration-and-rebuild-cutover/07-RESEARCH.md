# Phase 7: Explicit Migration and Rebuild Cutover - Research

**Researched:** 2026-09-09
**Domain:** Offline, version-aware BlobStore migration and explicit rebuild
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

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

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| MIGR-03 | Migration begins with a non-mutating inventory and produces both human-readable and machine-readable plans. | Use a raw, bounded authority scan rather than schema-filtered catalog queries; authenticate every manifest; classify every canonical entry; serialize one canonical plan model to JSON and render the human report from it. [VERIFIED: `.planning/REQUIREMENTS.md:56`] |
| MIGR-04 | Supported same-backend format and schema migrations use explicit offline resumable copy-verify-switch semantics with workers stopped; ordinary opens and initialize do not silently upgrade schemas. | Add a dedicated offline maintenance coordinator and a narrow topology-specific authority publication capability. Stage immutable external payloads first, verify the whole candidate, and activate only inside the single authority transaction. [VERIFIED: `.planning/REQUIREMENTS.md:57`; `docs/adr/0001-topology-specific-storage-guarantees.md:36-44,141-161`] |
| MIGR-05 | Interrupted offline migrations resume safely from explicit maintenance evidence without losing the only valid copy of an entry or adopting unexplained incomplete catalogs. Preserve signing material; derived-index reconstruction is not another canonical cutover authority. | Bind authenticated run evidence to run ID, source/destination identities, authority revision, plan digest, and completed-step outputs; revalidate before resume; retain the prior verified authority state until finalize; rebuild projections afterward. [VERIFIED: `.planning/REQUIREMENTS.md:58`; `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:44-56`] |
| MIGR-06 | Incompatible formats and cross-backend moves have an explicit, confirmed rebuild path rather than implicit deletion or universal physical migration. | Separate rebuild plans from physical migration. Rebuild reads through the registered source handler and writes through the destination BlobStore, with no default exclusions and no universal native-format converter. [VERIFIED: `.planning/REQUIREMENTS.md:59`; `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:38-40`] |
</phase_requirements>

## Summary

Phase 7 should introduce a dedicated **offline maintenance boundary above BlobStore**, not migration behavior in `BlobStore.__init__`, `initialize()`, `UnifiedCache`, reconciliation, or backend adapters. The phase is about establishing the first release baseline and the machinery for later one-step release migrations. Historical pre-refactor and transitional layouts remain inspectable but rebuild-only. This is a locked scope, not a missing converter. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:9-21,29-33,104-109`]

The most important architectural gap is whole-store authority publication. Existing lifecycle operations publish per entry, while Phase 7 requires a candidate that is completely copied and verified before one store-wide activation. The single selected lifecycle authority must own that activation, rollback eligibility, and finalization. External payload copies happen before activation; later purge is retryable cleanup, so no plan may describe the database plus filesystem/S3 work as one ACID transaction. [VERIFIED: `src/cacheness/storage/lifecycle_authority.py:28-190,284-383`; `docs/adr/0001-topology-specific-storage-guarantees.md:36-40,63-65,88-100,141-155`]

Inspection also needs a new topology-neutral read seam. Current catalog paging silently skips entries whose catalog schema differs from the requested schema, SQLite's `list_entries()` materializes every entry, and PostgreSQL's `list_entries()` is only a capped diagnostic call that fails above the cap. None can satisfy the locked entry-complete, bounded inventory contract as written. [VERIFIED: `src/cacheness/storage/catalog.py:738-838`; `src/cacheness/storage/sqlite_lifecycle_authority.py:1469-1487`; `src/cacheness/storage/backends/postgresql_lifecycle_authority.py:961-984`]

**Primary recommendation:** Plan a small migration domain model plus offline coordinator, add narrow authority-owned maintenance primitives for identity/raw scan/activate/rollback/finalize, and implement migration and rebuild as two explicit workflows sharing authenticated inspection and evidence.

## Project Constraints (from AGENTS.md)

- `BlobStore` owns storage lifecycle, `UnifiedCache` owns policy above it, and `SqlCache` stays separate. [VERIFIED: `AGENTS.md:15-18`]
- Preserve explicit schema/format versions and offline migration/rebuild tooling; unsupported current layouts must fail rather than upgrade implicitly. [VERIFIED: `AGENTS.md:15-17`]
- Cover the existing memory, filesystem, SQLite, PostgreSQL, and S3 lifecycle compositions without claiming equal guarantees for each topology. [VERIFIED: `AGENTS.md:18`; `docs/adr/0001-topology-specific-storage-guarantees.md:70-81`]
- Treat application payloads as trusted, while enforcing safe parsing, path containment, authenticated metadata, and fail-closed integrity boundaries. [VERIFIED: `AGENTS.md:19`]
- Same-key activity may not corrupt a committed generation or produce metadata/payload disagreement; correctness must not depend on a process-local lock. [VERIFIED: `AGENTS.md:20-21`; `docs/adr/0001-topology-specific-storage-guarantees.md:88-100,148-150`]
- Maintain Python `">=3.11"` support; the package manifest and lockfile quote that exact range. [VERIFIED: `pyproject.toml:9`; `uv.lock:3`]
- Use snake_case modules and tests named `test_<subject>.py`; expose supported public convenience imports through package `__init__.py` barrels. [VERIFIED: `AGENTS.md:119-131,183-189`]
- New public APIs need typed boundaries and project-style docstrings; translate narrow operational errors with their causes preserved rather than adding broad exception suppression. [VERIFIED: `AGENTS.md:135-158,175-181`]
- Run `uv run ruff check src tests` and avoid increasing the existing lint baseline. [VERIFIED: `AGENTS.md:139-143`]
- The ADR checklist is mandatory: name topology, guarantee category, single authority and transaction boundary, external-blob interruption behavior, typed bounded outcomes, separated tests, and stop-condition review. [VERIFIED: `docs/adr/0001-topology-specific-storage-guarantees.md:194-209`]

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Inspect and classify | Offline maintenance service | Lifecycle authority + handler registry | The coordinator assembles evidence, while canonical identity/revision/entries come from the authority and payload eligibility comes from registered handlers. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:104-108`] |
| Stage/copy/verify | Offline maintenance service | Payload backend | Payloads are immutable external effects created before authority publication and must be verified without treating their presence as visibility. [VERIFIED: `docs/adr/0001-topology-specific-storage-guarantees.md:36-40,88-97`] |
| Activate/rollback/finalize | Lifecycle authority | Offline maintenance service | These operations change canonical store selection and therefore belong inside the one transactional authority; work-directory evidence can request but never authorize them. [VERIFIED: `docs/adr/0001-topology-specific-storage-guarantees.md:141-155`; `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:44-55`] |
| Explicit rebuild | Offline maintenance service | Source and destination BlobStore instances | Rebuild is application-level read/rewrite through declared handlers, not a claim of universal physical migration. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:38-40,108`] |
| Projection reconstruction | Projection subsystem | Offline coordinator | Projections publish from isolated derived state after canonical cutover and never affect payload eligibility. [VERIFIED: `src/cacheness/storage/projections.py:1-7,176-200,360-412`] |
| Cache-policy behavior | UnifiedCache | BlobStore | UnifiedCache must remain a policy consumer and must not create a parallel migration path. [VERIFIED: `AGENTS.md:17`; `docs/adr/0001-topology-specific-storage-guarantees.md:162-164`] |

## Standard Stack

### Core

| Library / module | Version | Purpose | Why Standard |
|------------------|---------|---------|--------------|
| Python standard library (`dataclasses`, `enum`, `json`, `hashlib`, `hmac`, `pathlib`) | Python `">=3.11"` | Immutable plan/evidence models, canonical JSON, digests, authenticated evidence, safe path handling | Already-supported runtime; this phase needs no new runtime dependency. [VERIFIED: `pyproject.toml:9-15`] |
| Existing lifecycle authority implementations | Current tree: SQLite schema `"7"`, PostgreSQL schema `"3"`; Phase 7 release baseline: SQLite schema `"8"`, PostgreSQL schema/capability `"4"` | Canonical identity/revision, raw inventory, and topology-specific publication transactions | Whole-store candidate/retained-prior selection cannot be represented by the current entry-only schemas, so Phase 7 deliberately evolves the authority schemas to SQLite `SQLITE_USER_VERSION = 8` and PostgreSQL `POSTGRESQL_AUTHORITY_SCHEMA_VERSION = 4` / `postgresql-lifecycle-authority-v4`. These become the first release baseline; development schemas 7/3 receive no compatibility promise. [VERIFIED current values: `src/cacheness/storage/sqlite_lifecycle_authority.py:62-79`; `src/cacheness/storage/backends/postgresql_lifecycle_authority.py:62-69`; RESOLVED design: RQ-01 below] |
| Existing BlobStore + HandlerRegistry | Store format `"2"`; handler base payload contract `"1"` | Authenticated payload access and custom-handler contract resolution | Exact values are `STORE_FORMAT_VERSION = 2` and `CacheHandler.PAYLOAD_FORMAT_VERSION = 1`; custom handlers registered on the store are part of the source contract. [VERIFIED: `src/cacheness/storage/catalog.py:30-43`; `src/cacheness/interfaces.py:112-148`; `src/cacheness/handlers.py:1250-1473`] |
| pytest | `"8.4.1"` locked | Unit, contract, interruption, and topology tests | Existing test framework and strict marker configuration. [VERIFIED: `uv.lock:1608-1609`; `pyproject.toml:84-104`] |

### Supporting

| Library / module | Version | Purpose | When to Use |
|------------------|---------|---------|-------------|
| `src/cacheness/storage/reconciliation.py` evidence patterns | In-repo | Bounded authenticated continuation model and dual JSON/human rendering precedent | Reuse its envelope/digest/idempotence patterns, but do not make reconciliation evidence migration authority. [VERIFIED: `src/cacheness/storage/reconciliation.py:114-162,165-318,627-690`] |
| `src/cacheness/storage/projections.py` | In-repo | Derived-only isolated rebuild and publication | Invoke after canonical activation when a projection is missing, dirty, or stale. Exact existing status values are `"current"`, `"dirty"`, and `"partial"`. [VERIFIED: `src/cacheness/storage/projections.py:41-47,176-200,360-412`] |
| SQLite application metadata | `application_id = 0x43414348`; current tree `user_version = 7`, Phase 7 release baseline `user_version = 8` | Local authority identity and schema compatibility | Inspect without mutation; the explicit RQ-01 checkpoint changes the fresh-schema baseline, while ordinary open never rewrites version 7. [VERIFIED current tree: `src/cacheness/storage/sqlite_lifecycle_authority.py:62-79,835-928`; RESOLVED: RQ-01] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Authority-owned whole-store publication | Filesystem rename, “current” pointer file, work-directory state | Rejected: paths and sidecars would become a second lifecycle authority, and filesystem replacement is not portable across filesystems/backends. [VERIFIED: `docs/adr/0001-topology-specific-storage-guarantees.md:92-97,141-155`; CITED: https://docs.python.org/3.11/library/os.html#os.replace] |
| Raw bounded canonical scan | Existing catalog query path | Rejected: the current page builder continues past schema-mismatched entries, so it cannot prove entry completeness. [VERIFIED: `src/cacheness/storage/catalog.py:780-805`] |
| Handler-declared directed transformation | Universal NPZ/Blosc2/Parquet/pickle/dill converter | Rejected by locked decision D-10; native formats remain handler/library contracts. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:40`] |
| Separate explicit rebuild | Treat cross-backend copy as same-backend migration | Rejected by MIGR-06 and D-08 because rebuild has different confirmation and exclusion semantics. [VERIFIED: `.planning/REQUIREMENTS.md:59`; `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:38`] |

**Installation:** No new package installation is recommended. Use the existing locked environment.

## Architecture Patterns

### System Architecture Diagram

```text
Operator supplies source + destination + separate work directory
                              |
                              v
                 Non-mutating inspection
          +-------------------+-------------------+
          | authority identity/revision/raw scan  |
          | signed manifests + handler contracts  |
          +-------------------+-------------------+
                              |
                     canonical plan model
                      /                \
             JSON evidence          human report
                      \                /
                              |
             +----------------+----------------+
             | compatibility decision          |
             +----------------+----------------+
                    | supported same backend
                    v
        stage immutable candidate payloads/catalog
                    -> verify every entry
                    -> explicit authority activation
                    -> [offline rollback | finalize]
                    -> separately confirmed purge
                    |
                    +--> rebuild derived projections

             incompatible / unsupported / cross-backend
                    v
          explicit confirmed rebuild plan
          source handler read -> destination BlobStore write
          -> destination verification -> explicit acceptance
```

The diagram preserves one lifecycle authority: neither the work directory, blob listing, candidate presence, nor projection state can publish a generation. [VERIFIED: `docs/adr/0001-topology-specific-storage-guarantees.md:88-98,141-155`; `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:51-55`]

### Recommended Project Structure

```text
src/cacheness/storage/
├── migration.py              # immutable plans, classification, workflow coordinator
├── migration_evidence.py     # canonical JSON, authentication, resume validation
├── migration_authority.py    # narrow maintenance capability/protocol
├── lifecycle_authority.py    # existing entry lifecycle; minimal shared value objects only
├── sqlite_lifecycle_authority.py
└── backends/postgresql_lifecycle_authority.py

tests/
├── test_migration_inspection.py
├── test_migration_plan_contract.py
├── test_migration_run_evidence.py
├── test_migration_cutover.py
└── test_rebuild_workflow.py
```

This boundary keeps migration orchestration out of `UnifiedCache` and avoids further widening the already-transitional general `LifecycleAuthority` protocol. [VERIFIED: `docs/adr/0001-topology-specific-storage-guarantees.md:144-170`; `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:104-109`]

### Pattern 1: Inspect Once, Revalidate Before Every Mutation

**What:** Inspection captures source identity, topology-specific authority schema/capability, authority revision, store/manifest/payload/catalog dimensions, signing availability, and a digest of the complete plan. Every mutating action re-reads identity/revision and rejects a stale plan. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:31,36,39,51-54`]

**When to use:** Always. Inspection is non-mutating even for unsupported historical layouts; unsupported migration eligibility never permits mutation. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:29-33,36`]

**Implementation direction:** Add a topology-neutral `AuthorityIdentitySnapshot` and a raw keyset page that returns all canonical entries at one bound revision. Do not reuse schema-filtered `CatalogPage` as the migration inventory. The current authenticated cursor already demonstrates how to bind store identity, schema/query fingerprints, authority revision, and last key/generation. [VERIFIED: `src/cacheness/storage/catalog.py:484-654,738-838`]

### Pattern 2: Compatibility Is a Matrix, Not One Integer

**What:** Classify independently across overall store layout, lifecycle-authority schema/capability, signed manifest schema, handler payload contract, and catalog schema transformation. A supported migration edge must name source and destination for each changed dimension. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:29-33`]

**Critical seam:** `StoreVersionDimensions` currently quotes the fields `"store_epoch"`, `"manifest_schema_version"`, `"sqlite_user_version"`, `"payload_format_version"`, and `"store_format_version"`. A PostgreSQL store nevertheless has its own exact persisted identifiers `POSTGRESQL_AUTHORITY_SCHEMA_VERSION = 3` and `POSTGRESQL_AUTHORITY_CAPABILITY = "postgresql-lifecycle-authority-v3"`. Planning must replace the SQLite-specific compatibility assumption with topology-neutral authority identity; do not merely add another ambiguous top-level version. [VERIFIED: `src/cacheness/storage/manifest.py:120-170`; `src/cacheness/storage/backends/postgresql_lifecycle_authority.py:62-69`]

### Pattern 3: Copy, Verify, Then Authority-Atomic Activate

**What:** Stage immutable candidate payloads and a candidate catalog without changing visibility. Verify entry count, bytes, keys/generations, manifest authenticity, payload digest/size, handler eligibility, and candidate completeness. Only then invoke one authority transaction that switches the canonical whole-store generation and retains the prior verified selection. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:36-48`; `docs/adr/0001-topology-specific-storage-guarantees.md:36-40,88-97,141-155`]

**Topology rule:** SQLite publication uses one SQLite transaction; PostgreSQL publication uses one PostgreSQL transaction. External filesystem/S3 copies and purges remain before/after effects with durable evidence and idempotent retry, never part of that ACID scope. SQLite allows one simultaneous write transaction, and PostgreSQL rollback affects the current database transaction; neither fact makes external blobs transactional. [CITED: https://www.sqlite.org/lang_transaction.html; CITED: https://www.postgresql.org/docs/current/sql-rollback.html; VERIFIED: `docs/adr/0001-topology-specific-storage-guarantees.md:63-65,72-77`]

### Pattern 4: Authenticated, Explicit Resume Evidence

**What:** Canonical JSON evidence is the sole maintenance continuation representation. Sign or otherwise authenticate an envelope containing run ID, plan digest, source and destination identities, source revision fingerprint, workflow state, completed batch identities/digests, candidate ownership, and prior/active authority receipts. Never include key bytes. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:51-56`]

**When to use:** Every resumable stage, copy, verify, activation, rollback, finalize, abort, and purge action. A missing/corrupt/mismatched envelope fails closed and never triggers “find latest” or candidate adoption. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:51-56`]

**Reuse:** The reconciliation token already authenticates run ID, authority revision, cursors, and high-water state using an HMAC domain. Reuse its bounded parsing and authenticity pattern, not its reconciliation action semantics. [VERIFIED: `src/cacheness/storage/reconciliation.py:165-318,627-690`]

### Pattern 5: Handlers Own Copy/Transform/Rebuild Eligibility

**What:** For exact-compatible payload contracts, physical copy can preserve bytes after manifest/digest verification. A format-changing migration requires a handler-declared directed version edge. Otherwise, rebuild loads through the exact registered source handler and writes through the destination store. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:40,108`; `src/cacheness/interfaces.py:120-148`; `src/cacheness/handlers.py:1437-1473`]

**Custom-handler requirement:** `store.handlers` is part of the migration input. Strengthen registration or inspection so a custom handler cannot be accepted for writes without declaring a readable payload identity and any supported transformation edges. Current duck-typed registration checks the older write/read surface, while later payload resolution expects payload-contract behavior; that mismatch can make a custom payload writable but not migration-readable. [VERIFIED: `src/cacheness/handlers.py:1250-1473`]

### Anti-Patterns to Avoid

- **Migration in ordinary open/initialize:** Existing initialization is deliberate and validates current schema; it must not become an implicit upgrade path. [VERIFIED: `src/cacheness/storage/blob_store.py:332-359`; `.planning/REQUIREMENTS.md:57`]
- **Candidate-path-as-authority:** A rename, marker, symlink, object listing, or “candidate exists” check cannot activate a store. [VERIFIED: `docs/adr/0001-topology-specific-storage-guarantees.md:92-97,141-155`]
- **Per-entry activation during whole-store migration:** It would expose a partial replacement and violate D-07. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:37`]
- **Using work evidence as runtime state:** Ordinary BlobStore operations must never consult it; it coordinates only the stopped-worker workflow. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:51`]
- **Scanning only the requested catalog schema:** Current catalog paging skips different-schema manifests and therefore cannot support entry-complete classification. [VERIFIED: `src/cacheness/storage/catalog.py:780-805`]
- **Re-signing with newly invented identity:** If the configured key provider cannot supply/verify the existing key, block. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:56`; `src/cacheness/storage/blob_store.py:754-786`]
- **Deleting the prior store during activation or finalize:** Finalize ends rollback eligibility; purge is separately confirmed, idempotent physical cleanup. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:46-48`]
- **Calling projection state canonical:** Projection status is derived and rebuildable; it cannot gate or publish payload migration. [VERIFIED: `src/cacheness/storage/projections.py:1-7,41-47,176-200`]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Payload conversion | Generic NPZ/Blosc2/Parquet/pickle/dill transcoder | Handler-declared directed edge, else rebuild | Native container meaning and library compatibility belong to the handler. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:40`] |
| Store activation | Pointer file, symlink, directory swap, “latest candidate” discovery | One topology-specific lifecycle-authority transaction | Visibility belongs to one authority, and `os.replace` may fail across filesystems. [VERIFIED: `docs/adr/0001-topology-specific-storage-guarantees.md:92-97,141-155`; CITED: https://docs.python.org/3.11/library/os.html#os.replace] |
| Cross-resource transaction | Custom two-phase commit spanning DB and filesystem/S3 | Immutable copy + durable evidence + authority commit + idempotent cleanup | The ADR explicitly limits ACID to one transactional resource. [VERIFIED: `docs/adr/0001-topology-specific-storage-guarantees.md:36-40,63-65,102-115`] |
| Migration runner coordination | New lock server, online lease, scheduler, or worker protocol | Stopped-worker precondition plus stale-plan revision checks | Online writer coordination is out of scope and another coordination system triggers ADR stop conditions. [VERIFIED: `.planning/ROADMAP.md:423-432`; `docs/adr/0001-topology-specific-storage-guarantees.md:172-192`] |
| Human report model | A second independently generated plan | Render from canonical JSON model | D-18 requires two renderings of one authoritative structured model. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:52`] |
| Projection migration | Copying derived indexes as canonical data | Existing isolated projection rebuild | Projection publication is already derived-only and isolated. [VERIFIED: `src/cacheness/storage/projections.py:176-200,360-412`] |
| Evidence secret storage | Serializing HMAC/signing keys into JSON | Existing `ManifestKeyProvider` boundary | Existing key reads are non-mutating and key bytes must not appear in plans/logs. [VERIFIED: `src/cacheness/storage/integrity.py:374-467`; `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:56`] |

**Key insight:** The hard problem is not copying bytes; it is proving complete eligibility, preserving one authority, and making every interrupted state attributable without ever discarding the last valid generation.

## Runtime State Inventory

| Category | Items Found | Action Required |
|----------|-------------|-----------------|
| Stored data | Read-only probes found an ignored local current authority at `cache/.cacheness/blobstore/.cacheness/lifecycle-authority-v2.sqlite3` with 20 entries and `user_version = 7`, plus a transitional `cache/.cacheness/unified-cache-v1/.cacheness/lifecycle-authority-v1.sqlite3` with 20 entries and `user_version = 1`. The source declares the current path exactly as `Path(".cacheness") / "lifecycle-authority-v2.sqlite3"`. [VERIFIED: read-only SQLite probes on 2026-09-09; `src/cacheness/storage/sqlite_lifecycle_authority.py:62-79`] | Do not mutate these development stores during implementation tests. Treat the v1/transitional store as inspectable rebuild-only and use `tmp_path` fixtures for mutation/interruption tests. No historical data migration task is required. |
| Live service config | No live PostgreSQL or S3 service was used during research; the current roadmap defers real PostgreSQL/S3 qualification to Phase 8. [VERIFIED: `.planning/ROADMAP.md:436-445`; local environment audit 2026-09-09] | Implement deterministic authority/backend contracts now; do not turn unavailable live services into support claims. Preserve explicit DSN/bucket configuration boundaries in tests. |
| OS-registered state | None found: the library runs in the host Python process and the repository has no migration daemon, launchd unit, systemd unit, or task-scheduler registration. [VERIFIED: repository `rg` audit on 2026-09-09; `AGENTS.md:57`] | None. Do not add an online worker/coordinator service for this phase. |
| Secrets/env vars | Both observed local stores contain a 32-byte `blob_manifest_hmac_key.bin`; the default key location is constructed at `cache_dir / "blob_manifest_hmac_key.bin"`. Test-only environment names include `"CACHENESS_TEST_MANIFEST_KEY_B64"`, PostgreSQL DSN, and S3 bucket/region configuration. [VERIFIED: read-only filesystem probes on 2026-09-09; `src/cacheness/storage/blob_store.py:228-231`; repository `rg` audit on 2026-09-09] | Read key material only through the configured provider; never copy it into plan/report/run JSON. Tests must inject providers and assert redaction. Existing local keys remain untouched. |
| Build artifacts / installed packages | `.pytest_cache`, `.ruff_cache`, and Python `__pycache__` directories exist; they carry no lifecycle authority and are not migration inputs. [VERIFIED: read-only filesystem audit on 2026-09-09] | None for migration. Test cleanup may use temporary directories only; do not treat cache artifacts as candidate evidence. |

The canonical runtime-state answer is therefore: updating repository files is insufficient because existing ignored store roots and signing keys remain on disk, but Phase 7 must inspect rather than upgrade them; there is no live-service or OS registration to migrate in this workspace. [VERIFIED: read-only filesystem/SQLite/environment audits on 2026-09-09]

## Common Pitfalls

### Pitfall 1: The Current Layout Inspector Does Not Recognize a Real Current Store

**What goes wrong:** `inspect_store_layout()` calls a store current only when it finds `.cacheness/store-format.json` or `store-format.json`; otherwise any artifact makes the root `"foreign-or-incomplete"`. The implementation currently has no production write path for that marker. [VERIFIED: `src/cacheness/storage/manifest.py:426-454`; repository `rg` audit on 2026-09-09]

**Why it happens:** SQLite authority identity became canonical, while the earlier file-marker inspector remained disconnected from initialization. [VERIFIED: `src/cacheness/storage/sqlite_lifecycle_authority.py:735-928`; `src/cacheness/storage/manifest.py:426-454`]

**How to avoid:** Build inspection from topology composition plus authority-owned persisted identity/schema. Keep exact historical fixture recognition in the legacy inspector and classify all other pre-baseline roots rebuild-only. Do not add a marker file as a second authority.

**Warning signs:** A newly initialized store is classified `"foreign-or-incomplete"`, or tests create marker files manually instead of inspecting an initialized authority. [VERIFIED: `src/cacheness/storage/manifest.py:430-453`; `tests/test_catalog_schema.py:154-166`]

### Pitfall 2: Entry-Complete Inventory Built on a Filtering Query

**What goes wrong:** Entries using a different catalog schema are silently skipped, so aggregate counts/bytes and migration eligibility become incomplete. [VERIFIED: `src/cacheness/storage/catalog.py:780-805`]

**Why it happens:** The catalog API is a user query surface, not an administrative raw inventory surface. [VERIFIED: `src/cacheness/storage/catalog.py:738-838`]

**How to avoid:** Add raw, bounded keyset enumeration at a fixed authority revision, then authenticate and classify every entry in the maintenance layer.

**Warning signs:** Inventory totals depend on the selected application catalog schema or `list_entries()` allocates the whole SQLite catalog. [VERIFIED: `src/cacheness/storage/sqlite_lifecycle_authority.py:1469-1487`]

### Pitfall 3: One Top-Level Version Hides Topology Differences

**What goes wrong:** A PostgreSQL store can be judged by the manifest field named `"sqlite_user_version"`, even though the PostgreSQL authority has its own schema version and capability. [VERIFIED: `src/cacheness/storage/manifest.py:120-170`; `src/cacheness/storage/backends/postgresql_lifecycle_authority.py:62-69`]

**How to avoid:** Represent authority kind/capability/schema as its own compatibility dimension and require an explicit edge for every changed dimension.

**Warning signs:** Compatibility code compares only `STORE_FORMAT_VERSION = 2` or `CURRENT_SQLITE_USER_VERSION = 7`. [VERIFIED: `src/cacheness/storage/catalog.py:30-43`; `src/cacheness/storage/manifest.py:73-75`]

### Pitfall 4: Resume Infers Progress from Candidate Bytes

**What goes wrong:** A crash-created blob is adopted without proof that it belongs to the run or matches the plan, making incidental presence a second authority. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:53-55`; `docs/adr/0001-topology-specific-storage-guarantees.md:92-97`]

**How to avoid:** Require explicit run ID/evidence path, authenticate evidence, revalidate each recorded result, and continue only idempotent steps. Missing evidence yields diagnostics, not adoption.

**Warning signs:** Code searches for “latest” run, scans a candidate directory to reconstruct state, or resumes without checking source revision and plan digest.

### Pitfall 5: Activation Accidentally Deletes Rollback Material

**What goes wrong:** Eager cleanup leaves no prior verified store after activation, or a purge failure is reported as failed activation. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:44-48`]

**How to avoid:** Retain the prior selection inside authority state and keep the activated authority in `activated_offline`, which rejects every ordinary worker open/read/query/mutation entry point. While workers remain stopped, the operator either rolls back or finalizes; finalize commits `active` and seals rollback before workers restart. Purge is later, confirmed, and retryable.

**Warning signs:** Candidate activation and blob deletion appear in the same database success path, or finalize physically deletes data.

### Pitfall 6: Custom Handlers Are Treated as Second-Class

**What goes wrong:** A custom handler accepted for writes may lack a complete declared payload-read contract, or migration code may bypass it and attempt a built-in native converter. [VERIFIED: `src/cacheness/handlers.py:1250-1473`; `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:40`]

**How to avoid:** Inventory against `source_store.handlers`; make readable payload identity and directed transformation edges explicit; route incompatible entries to rebuild/block with stable reasons.

**Warning signs:** `isinstance` branches over built-in handler classes, format names are hard-coded in the coordinator, or a source store's custom registry is ignored.

### Pitfall 7: Documentation Preserves a False Automatic-Migration Promise

**What goes wrong:** `docs/BACKEND_SELECTION.md` currently says migrations are “seamless” and “automatically” convert schemas, which contradicts current implementation and the Phase 7 contract. [VERIFIED: `docs/BACKEND_SELECTION.md:256-275`; `.planning/REQUIREMENTS.md:57`]

**How to avoid:** Include documentation corrections and negative tests proving constructors/initialize do not mutate incompatible stores.

## Code Examples

Verified design patterns and implementation-ready skeletons:

### Entry-Complete Classification Type

The locked exact classifications are `"migratable"`, `"rebuildable"`, `"blocked"`, and `"unverifiable"`. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:36`]

```python
# Source: .planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:36
from dataclasses import dataclass
from typing import Literal

EntryDisposition = Literal[
    "migratable",
    "rebuildable",
    "blocked",
    "unverifiable",
]


@dataclass(frozen=True)
class MigrationEntryAssessment:
    key: str
    generation: str
    byte_size: int
    disposition: EntryDisposition
    reason_code: str
```

### Narrow Authority Maintenance Capability

```python
# Source pattern: docs/adr/0001-topology-specific-storage-guarantees.md:141-170
from typing import Protocol


class MigrationAuthority(Protocol):
    def identity_snapshot(self) -> "AuthorityIdentitySnapshot": ...

    def inventory_page(
        self,
        cursor: str | None,
        *,
        revision: int,
        limit: int,
    ) -> "AuthorityInventoryPage": ...

    def activate_verified_candidate(
        self,
        receipt: "VerifiedCandidateReceipt",
        *,
        expected_source: "AuthorityIdentitySnapshot",
    ) -> "ActivationReceipt": ...

    def rollback_activation(self, receipt: "ActivationReceipt") -> "RollbackReceipt": ...

    def finalize_activation(self, receipt: "ActivationReceipt") -> "FinalizeReceipt": ...
```

The protocol should expose transactional primitives, not duplicate orchestration for each backend. [VERIFIED: `docs/adr/0001-topology-specific-storage-guarantees.md:144-147,166-170`]

### Preserve Unknown Authenticated Catalog Attributes

```python
# Source pattern: src/cacheness/storage/catalog.py:284-317
source_values = source_schema.read_mapping(authenticated_manifest.catalog_values)
destination_values = dict(authenticated_manifest.catalog_values)
destination_values.update(transformed_declared_values)
```

`CatalogSchema.read_mapping()` materializes additive defaults without rewriting stored values, and validation accepts bounded undeclared values when policy permits. Migration must preserve the authenticated original mapping and transform only explicitly declared fields. [VERIFIED: `src/cacheness/storage/catalog.py:284-317`; `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:41`]

## State of the Art

| Old Approach | Current Phase 7 Approach | When Changed | Impact |
|--------------|--------------------------|--------------|--------|
| Historical compatibility readers and fixture-specific recognition | Upcoming release is first migration baseline; historical layouts are inspectable rebuild-only | Phase 7 decisions, 2026-09-09 | Planner must not build converters for development layouts. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:16-21,29-33`] |
| Layout marker plus SQLite-specific version bundle | Topology-neutral authority identity plus independent compatibility matrix | Required by D-03 and current PostgreSQL seam | Prevents a top-level version from implying false compatibility. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:31`; `src/cacheness/storage/manifest.py:120-170`; `src/cacheness/storage/backends/postgresql_lifecycle_authority.py:62-69`] |
| Per-entry lifecycle promotion | Verified whole-store candidate activation | Required by D-07/D-12 | Prevents partial migrated publication. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:37,44`] |
| Catalog query paging | Raw, bounded, revision-bound maintenance inventory | Required by D-06 | Includes every canonical entry, including schema-incompatible entries. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:36`; `src/cacheness/storage/catalog.py:780-805`] |
| Ad hoc continuation or incidental candidate discovery | Explicit authenticated run ID/evidence path | Required by D-17 through D-21 | Makes interruption recovery deterministic and fail-closed. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:51-55`] |

**Deprecated/outdated:**

- `docs/BACKEND_SELECTION.md` automatic/seamless migration language is outdated and must be replaced with explicit offline migration/rebuild guidance. [VERIFIED: `docs/BACKEND_SELECTION.md:256-275`; `.planning/REQUIREMENTS.md:57-59`]
- `inspect_store_layout()` as the sole current-store detector is outdated because current initialization does not create the marker it requires. [VERIFIED: `src/cacheness/storage/manifest.py:426-454`; repository `rg` audit on 2026-09-09]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The suggested filenames (`migration.py`, `migration_evidence.py`, `migration_authority.py`) are the best internal split. [ASSUMED] | Recommended Project Structure | Low: module boundaries are explicitly delegated to the planner; behavior remains unchanged. |
| A2 | HMAC with the existing signing identity is the preferred authentication mechanism for maintenance evidence rather than a different verifiable envelope. [ASSUMED] | Authenticated Resume Evidence | Medium: D-20 allows “authenticated or otherwise verifiable”; planner may choose another mechanism without exposing key material. |
| A3 | A same-format migration may copy verified immutable bytes without deserializing them. [ASSUMED] | Handler Eligibility | Medium: a future handler may require semantic validation even when its payload identity is unchanged; the handler contract should decide. |

## Resolved Design Decisions

1. **RQ-01 — Persisted authority schema evolution is required.** The current authority schemas cannot atomically represent a verified whole-store candidate, the activated-but-still-offline rollback window, and the retained prior selection. Phase 7 therefore changes SQLite `SQLITE_USER_VERSION` from `7` to `8`, PostgreSQL `POSTGRESQL_AUTHORITY_SCHEMA_VERSION` from `3` to `4`, and the PostgreSQL capability from `postgresql-lifecycle-authority-v3` to `postgresql-lifecycle-authority-v4`. SQLite `authority_state` and PostgreSQL `authority_meta` add the exact columns `migration_run_id`, `migration_plan_digest`, `migration_candidate_digest`, `migration_source_revision`, nullable `migration_activated_revision`, `migration_state`, `migration_active_selection`, and `migration_rollback_eligible`. Both schemas also add `migration_store_entries(run_id, selection, key, generation, locator, manifest, manifest_digest, lineage, entry_revision)` with primary key `(run_id, selection, key)`, where `selection` is exactly `candidate` or `prior`. `migration_state` is exactly `idle`, `candidate`, `activated_offline`, `active`, or `rolled_back`; `migration_active_selection` is exactly `source`, `candidate`, or `prior`. In `idle`, migration identity/digests/revisions are null, selection is source, and rollback is false. Recording a completely verified candidate stores run/plan/candidate/source values and commits `candidate/source/false`. Activation saves current entries as `prior`, replaces canonical `entries` from `candidate`, stores `migration_activated_revision`, and commits `activated_offline/candidate/true`. Rollback restores `prior` and commits `rolled_back/prior/false`; finalize commits `active/candidate/false` without deleting prior bytes. Every transition and canonical-entry replacement is one transaction in the selected authority. External payload creation, evidence checkpointing, and purge remain attributable effects outside it and recover through exact receipts, never cross-resource ACID. [VERIFIED need: `src/cacheness/storage/lifecycle_authority.py:28-190,284-383`; `docs/adr/0001-topology-specific-storage-guarantees.md:36-40,88-100,141-155`]

   A blocking one-way decision checkpoint precedes publication of this schema/semantic contract. Ordinary `open()`/`initialize()` continues to reject schema 7/3 without DDL or mutation. Because the repository is pre-release, schema 8/4 is the first release baseline: no production migration edge or compatibility promise is created for superseded development schemas 7/3, and tests use fresh 8/4 authorities or test-injected future edges per D-01/D-05.

2. **RQ-02 — The Phase 7 operator entry point is the Python library API only.** `OfflineMigrationService` and its immutable plan/evidence/receipt models are exported from `cacheness.storage`; canonical JSON and the human report render from the same model. Phase 7 adds no CLI and no `[project.scripts]` entry. A later CLI may be a stateless adapter over this API but may not own paths, run discovery, evidence, or lifecycle state. [VERIFIED existing packaging: `pyproject.toml:1-82`; locked model rule: D-18]

3. **RQ-03 — Stopped-worker evidence is explicit acknowledgement plus authority revision checks, not proof of global quiescence.** Every mutating maintenance call requires a `StoppedWorkerAcknowledgement` bound to the run ID, plan digest, source store identity, and inspected source revision; the acknowledgement is stored in canonical run evidence and rechecked together with the live source revision before mutation. It is an operator assertion and never a lease or runtime authority. After activation, the authority is `activated_offline`: every ordinary worker open/read/query/mutation entry point fails with a typed offline-decision-required outcome, while narrow maintenance receipt/status methods remain available. Workers must remain stopped until the operator either rolls back or calls `finalize`. `finalize` changes the authority to `active` and permanently seals rollback before workers restart. There is no first-post-activation-write marker, worker registry, online detector, lock, lease, queue, or global-quiescence claim. [VERIFIED scope: D-09, D-13, D-17; ADR stop conditions]

4. **RQ-04 — Directed transformation edges live on the registered handler contract.** Add frozen `PayloadTransformationEdge(source_format, source_version, target_format, target_version)` and two `CacheHandler` methods: `payload_transformation_edges() -> tuple[PayloadTransformationEdge, ...]` (default empty) and `transform_payload(snapshot: GuardedReadSnapshot, edge: PayloadTransformationEdge, *, destination_io: GuardedHandlerIO, key: str, config: CacheConfig) -> GuardedWriteResult` (default rejects). `HandlerRegistry.resolve_payload_transformation(handler_type, source_format, source_version, target_format, target_version)` returns the store-local handler plus one exact declared edge or raises the typed unsupported-contract error. Exact-contract verified byte copy remains the default and needs no edge. A changed contract may transform only through that handler-declared directed edge; a missing/ambiguous edge classifies the entry rebuildable or blocked. Phase 7 does not rename `CacheHandler`/`HandlerRegistry`, create a global transform registry, or centralize native-format conversion. [VERIFIED current seam: `src/cacheness/interfaces.py:112-148`; `src/cacheness/handlers.py:1437-1473`; locked ownership: D-10]

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| `uv` | Project commands/tests | ✓ | `0.12.9` | — [VERIFIED: environment probe 2026-09-09] |
| CPython | Library/tests | ✓ | `3.11.16` via `uv run --no-sync`; project range is `">=3.11"` | Phase 8 owns full supported-version matrix. [VERIFIED: environment probe 2026-09-09; `pyproject.toml:9`] |
| SQLite CLI/library | Local store inspection and deterministic local contracts | ✓ | CLI `3.51.0` | Python `sqlite3` module. [VERIFIED: environment probe 2026-09-09] |
| pytest | Validation | ✓ | locked `8.4.1` | — [VERIFIED: `uv.lock:1608-1609`] |
| PostgreSQL client/service | Remote contract integration | ✗ client in audited shell; no live service configured | — | Deterministic contract tests in Phase 7; real gate remains Phase 8. [VERIFIED: environment probe 2026-09-09; `.planning/ROADMAP.md:436-445`] |
| `psycopg` | PostgreSQL implementation | ✗ in the no-sync Python 3.11 environment | — | Optional dependency path; skip live test locally, keep deterministic contract tests. [VERIFIED: environment probe 2026-09-09; `src/cacheness/storage/backends/postgresql_lifecycle_authority.py:55-59`] |
| AWS CLI | Optional operator diagnostics only | ✓ | installed; version not relied on | Not required by implementation. [VERIFIED: environment probe 2026-09-09] |

**Missing dependencies with no fallback:** None for Phase 7 deterministic implementation and contract testing. [VERIFIED: environment audit 2026-09-09]

**Missing dependencies with fallback:** Live PostgreSQL/`psycopg` is unavailable locally; use deterministic contract tests and retain the Phase 8 real-service gate. [VERIFIED: `.planning/ROADMAP.md:436-445`]

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest `"8.4.1"` locked [VERIFIED: `uv.lock:1608-1609`] |
| Config file | `pyproject.toml` with `testpaths = ["tests"]`, `python_files = ["test_*.py"]`, and strict markers [VERIFIED: `pyproject.toml:84-104`] |
| Quick run command | `uv run pytest -q tests/test_migration_inspection.py tests/test_migration_plan_contract.py tests/test_migration_run_evidence.py -o log_cli=false` |
| Full suite command | `uv run pytest -q -o log_cli=false` |

The focused existing compatibility/catalog/projection suite passed during research. [VERIFIED: `uv run --no-sync pytest -q tests/test_stored_compatibility.py tests/test_blob_store_legacy_contract.py tests/test_catalog_schema.py tests/test_projection_sql_atomicity.py -o log_cli=false`, 2026-09-09]

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MIGR-03 | Non-mutating, entry-complete bounded inventory; stable dispositions/reasons; one JSON model and human rendering | unit + contract | `uv run pytest -q tests/test_migration_inspection.py tests/test_migration_plan_contract.py -o log_cli=false` | ❌ Wave 0 |
| MIGR-04 | Same-backend stage/verify/explicit activate; stale revision rejection; constructors/initialize never migrate | fault-injection + contract | `uv run pytest -q tests/test_migration_cutover.py tests/test_stored_compatibility.py -o log_cli=false` | ❌ Wave 0 for cutover; existing compatibility tests ✓ |
| MIGR-05 | Resume from explicit authenticated evidence at every interruption point; retain prior; rollback/finalize/purge; projection remains derived | fault-injection + integration | `uv run pytest -q tests/test_migration_run_evidence.py tests/test_migration_cutover.py tests/test_projection_sql_atomicity.py -o log_cli=false` | ❌ Wave 0 for migration files; projection tests ✓ |
| MIGR-06 | Cross-backend/incompatible rebuild; no default exclusions; exact confirmed exclusion plan; custom handler round trip | integration + contract | `uv run pytest -q tests/test_rebuild_workflow.py tests/test_handler_registration.py -o log_cli=false` | ❌ Wave 0 for rebuild file; handler tests ✓ |

### Required Test Dimensions

- **Integrity/safety:** no partial activation, no unauthenticated plan/evidence/manifest use, source and prior verified store remain intact through every pre-activation failure. [VERIFIED: `docs/adr/0001-topology-specific-storage-guarantees.md:48-68,88-100`]
- **Recovery:** interrupt after every durable evidence/payload/authority step; resume idempotently; abort only owned unactivated candidates; cleanup failure becomes retryable work. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:44-55`]
- **Progress:** one inventory/copy/resume call performs bounded work or returns typed continuation; do not require every PostgreSQL contender to succeed. Exact existing retryable PostgreSQL outcomes are `"serialization"`, `"deadlock"`, `"lock_timeout"`, and `"statement_timeout"`. [VERIFIED: `src/cacheness/storage/backends/postgresql_lifecycle_authority.py:74-80`; `docs/adr/0001-topology-specific-storage-guarantees.md:129-137`]
- **Performance:** keep benchmarks separate; Phase 7 correctness tests must not encode universal deadlines. [VERIFIED: `docs/adr/0001-topology-specific-storage-guarantees.md:117-137`]
- **Topologies:** memory/memory contract, SQLite/filesystem deterministic integration, PostgreSQL/S3 deterministic contract only until Phase 8 live qualification. [VERIFIED: `src/cacheness/storage/composition.py:127-190`; `.planning/ROADMAP.md:436-445`]

### Sampling Rate

- **Per task commit:** the affected new test module plus nearest existing contract module.
- **Per wave merge:** `uv run pytest -q tests/test_migration_*.py tests/test_rebuild_workflow.py tests/test_stored_compatibility.py tests/test_handler_registration.py -o log_cli=false`
- **Phase gate:** full suite green plus `uv run ruff check src tests`; verify no incompatible-store fixture changed byte-for-byte during negative tests.

### Wave 0 Gaps

- [ ] `tests/test_migration_inspection.py` — raw bounded inventory, current identity, historical rebuild-only classification, no mutation.
- [ ] `tests/test_migration_plan_contract.py` — canonical JSON/human rendering, stable reasons, compatibility matrix, exact totals.
- [ ] `tests/test_migration_run_evidence.py` — authentication, redaction, run/source/destination mismatch, corruption, explicit resume.
- [ ] `tests/test_migration_cutover.py` — stage/verify/activate/rollback/finalize/purge and every interruption point.
- [ ] `tests/test_rebuild_workflow.py` — cross-backend/custom-handler rebuild and exclusion confirmation.
- [ ] Extend `tests/test_lifecycle_authority_contract.py` and `tests/contracts/test_postgresql_lifecycle_authority.py` for the narrow maintenance capability.
- [ ] Extend `tests/test_stored_compatibility.py` so an actually initialized current store is inspectable without a manually created format marker.

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no for users; yes for evidence provenance | Authenticate plan/run envelopes with configured signing identity or an equally verifiable mechanism; no user-auth system is in scope. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:51-56`] |
| V3 Session Management | no | Offline stateless operator actions require explicit run ID/evidence path; there is no session. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:53`] |
| V4 Access Control | yes at authority boundary | Only the selected authority transaction may activate/rollback/finalize; work evidence and blob presence cannot authorize lifecycle state. [VERIFIED: `docs/adr/0001-topology-specific-storage-guarantees.md:92-97,141-155`] |
| V5 Input Validation | yes | Bounded canonical JSON parsing, duplicate-key rejection, exact path containment, enum/reason validation, size/depth limits, identity/revision checks. Existing manifest parsing quotes limits such as `MAX_MANIFEST_BYTES = 1_048_576`, `MAX_NESTING_DEPTH = 16`, and `MAX_TOTAL_NODES = 16_384`. [VERIFIED: `src/cacheness/storage/manifest.py:31-42,385-423`] |
| V6 Cryptography | yes | Reuse provider-supplied key material and standard HMAC/SHA-256 implementation; never serialize secrets or silently generate a replacement identity. Exact manifest algorithms are `"hmac-sha256"` and `"sha256"`. [VERIFIED: `src/cacheness/storage/manifest.py:31-34,329-345`; `src/cacheness/storage/integrity.py:374-467`] |

### Known Threat Patterns for This Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Forged or edited plan/evidence | Tampering | Canonical encoding, authenticated envelope, bounded parser, plan digest, explicit run/source/destination binding. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:51-56`; `src/cacheness/storage/reconciliation.py:627-690`] |
| Path traversal or workdir overlaps source/destination | Tampering / Elevation | Resolve and prove separate contained roots before mutation; reject symlink/path escape; never accept payload locator as an arbitrary destination path. [VERIFIED: `AGENTS.md:19`; `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:51`] |
| Source changes after inspection | Tampering | Revalidate persisted identity and authority revision before every mutation; stale plan requires reinspection. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:39`] |
| Secret disclosure in reports/logs | Information Disclosure | Evidence stores identity/fingerprint only; key bytes stay behind `ManifestKeyProvider`; redaction tests inspect JSON and human output. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:56`; `src/cacheness/storage/integrity.py:374-467`] |
| Unsafe deserialization before authenticity checks | Tampering / Elevation | Authenticate manifest and verify payload digest/size before handler deserialization; payloads remain trusted-application data, not hostile input. [VERIFIED: `AGENTS.md:19`; `src/cacheness/storage/blob_store.py:817-836`] |
| Exclusion ambiguity destroys sole reconstructable data | Repudiation / Tampering | No exclusions by default; regenerated plan names exact keys/categories, counts, bytes, reasons, and confirmation. [VERIFIED: `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md:38`] |
| Resource exhaustion from unbounded inventory/evidence | Denial of Service | Raw keyset pages, bounded batches, capped JSON fields/depth/bytes, explicit continuation. [VERIFIED: `src/cacheness/storage/catalog.py:30-50,738-838`; `src/cacheness/storage/manifest.py:35-41`] |

## Sources

### Primary (HIGH confidence)

- `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md` — locked version window, compatibility, workflow, rollback/finalize/purge, evidence, signing, and custom-handler decisions.
- `.planning/REQUIREMENTS.md` and `.planning/ROADMAP.md` — MIGR-03 through MIGR-06 and phase acceptance boundary.
- `docs/adr/0001-topology-specific-storage-guarantees.md` — one authority, topology tiers, external-effect recovery, and stop conditions.
- `src/cacheness/storage/catalog.py` — store version, authenticated cursor, schema behavior, and current paging limitation.
- `src/cacheness/storage/manifest.py` — persisted manifest dimensions, signing, and disconnected layout-marker inspection.
- `src/cacheness/storage/lifecycle_authority.py`, `sqlite_lifecycle_authority.py`, and `backends/postgresql_lifecycle_authority.py` — current identity/version/paging/publication seams.
- `src/cacheness/interfaces.py` and `handlers.py` — payload identities, registered-handler resolution, and custom-handler gap.
- `src/cacheness/storage/reconciliation.py`, `projections.py`, and `integrity.py` — reusable evidence, derived rebuild, and key-provider patterns.

### Secondary (MEDIUM confidence)

- https://docs.python.org/3.11/library/os.html#os.replace — replacement semantics and cross-filesystem limitation.
- https://www.sqlite.org/lang_transaction.html — SQLite transaction and writer behavior.
- https://www.sqlite.org/pragma.html#pragma_user_version — application-owned SQLite schema version field.
- https://www.sqlite.org/backup.html — separate database snapshot/backup semantics; useful for candidates, not authority publication.
- https://www.postgresql.org/docs/current/sql-rollback.html — transaction rollback semantics.
- https://www.postgresql.org/docs/current/ddl-alter.html — PostgreSQL schema alteration transaction context.

### Tertiary (LOW confidence)

- None. Unverified design choices are isolated in the Assumptions Log.

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — no new packages; versions and seams were read from current source and lockfile.
- Architecture: HIGH — derived from locked decisions, ADR authority rules, and line-level current seam inspection.
- Pitfalls: HIGH — reproduced from current code paths and repository-wide searches; the focused existing tests passed.
- Runtime state: MEDIUM — local filesystem/SQLite state was inspected read-only, but ignored development artifacts can change after this research date.

**Research date:** 2026-09-09
**Valid until:** 2026-10-09 for stable project contracts; re-run seam audit after any lifecycle-authority or manifest change.
