# Phase 3 architecture audit

Date: 2026-09-06  
Code reviewed: `f5d406a` plus the uncommitted draft `03-21-PLAN.md`  
Status: audit and recommendations; not an approved replacement design

## Conclusion

The project is feasible with SQLite for a local, single-host store. Phase 3 has made valuable progress, but its architecture still carries the overlapping lifecycle responsibilities that produced the repair loops. ADR 0001 corrected the guarantee vocabulary and removed some unnecessary machinery; it has not yet been translated into a sufficiently small implementation boundary or an updated roadmap.

Do not execute draft Plan 03-21 unchanged. Preserve the real integrity fixes and tests already built, but revise the architecture and acceptance contract before expanding the bootstrap protocol or multiplying backend implementations.

The user's clarified product goal is a reliable blob store with customizable catalog metadata, whose core components also support caching. A store need not simultaneously serve durable storage and cache roles. Sharing implementation is sufficient; sharing a live namespace, catalog projection, or retention policy is not required.

## Audit scope and evidence limits

This audit inspected PROJECT, REQUIREMENTS, ROADMAP, Phase 3 context, selected plans/summaries and historical review/fix reports, the accepted ADR, the latest verification/review, and current storage/cache implementation. It is an architecture and scope audit, not another exhaustive bug hunt. No production code or existing planning artifacts were changed. The pending revision agent was already interrupted.

The previous execution's clean Python 3.11 suite failure and the verifier's reproductions are cited as prior evidence, not represented as tests rerun in this audit. A passing subsequent focused test does not invalidate a previously observed intermittent failure. Nor does one passing full suite establish that all interleavings are correct.

## Findings

### 1. Critical: the supposedly derived catalog still participates in lifecycle correctness

`UnifiedCache._init_lifecycle_state()` creates an internal BlobStore under `.cacheness/unified-cache-v1`, while the cache separately constructs its metadata backend. `put()` reads the projection, invokes private BlobStore admission/put methods, publishes the projection after authority promotion, settles cleanup, and finally writes custom metadata. `_authority_snapshot_entry()` repeatedly compares, repairs, and rechecks the two representations before serving a read.

Evidence: `src/cacheness/core.py:279`, `:1006`, `:1286`, `:1369`, `:1524`.

These are independently sequenced operations across a canonical database and a compatibility catalog. Although only one is called authoritative, both influence success, reads, signing, and cleanup. The latest CR-03 demonstrates the consequence: corruption in a projection can lead cache signature policy to retire a valid authority generation. This is the ADR's second-authority problem in behavior, even without two formally authoritative databases.

Recommendation: commit the blob descriptor and its authoritative user catalog metadata together in the selected transactional catalog. Put optional search/export projections outside the storage success boundary. A projection may be unavailable or stale; it must not authorize deletion of the authoritative entry. Keep legacy compatibility translation at an adapter or migration boundary. Where relational custom metadata must commit with a blob, its extension must use that same catalog transaction; an arbitrary external ORM session cannot silently receive that guarantee.

### 2. High: the backend interface exports too much of the state machine

`LifecycleAuthority` exposes prepare, verify, promote, abort, tombstone retirement, clear membership and checkpoints, reconciliation snapshots/pages/checkpoints, and projection revision/backup operations. SQLite and memory each implement substantial lifecycle behavior. The engine, reconciler, BlobStore, and cache facade coordinate pieces of those operations.

Evidence: `src/cacheness/storage/lifecycle_authority.py:280`, `src/cacheness/storage/lifecycle.py:63`, `src/cacheness/storage/reconciliation.py`, and both authority implementations.

The memory abort and debt-cursor defects are ordinary data-structure mistakes, not SQLite races. Their architectural significance is that a second implementation reproduces the same recovery semantics with different internal representations. Adding PostgreSQL and other variants to this interface would multiply that proof burden.

Recommendation: centralize lifecycle sequencing behind one storage service. Keep transaction boundaries explicit and let catalog adapters supply a small set of transactional record operations and conditional updates. Avoid both extremes: a method for every lifecycle stage on every backend, and a generic key/value interface that forces the service to reinvent atomic multi-record updates. The core needs a transaction capable of publishing an entry and recording its cleanup obligation together.

### 3. High: lazy bootstrap has become a separate distributed protocol

The canonical SQLite leaf is exclusively created and closed before schema/application identity is committed. Concurrent readers see an object that looks like a database but has no initialized identity. `_connection()` mixes materialization, connection setup, schema migration, validation, and normal operation.

Evidence: `src/cacheness/storage/sqlite_lifecycle_authority.py:486`, `:690`, `:789`, `:1048`; Phase 3 CR-04; `03-02-SUMMARY.md` explicitly records first-mutation creation.

The checker correctly rejected the proposed zero-application-ID/zero-version/empty-schema test as proof of a live initializer. An abandoned or independently created empty database can look identical. The earlier review's suggested fix was therefore insufficiently grounded, and the planner amplified it into an executable requirement.

My subsequent instruction to prepare a private database and atomically install it was also premature. It creates candidate naming, containment, no-clobber installation, durability, loser cleanup, and crash-residue obligations. It might work, but it is another filesystem protocol and should not be the default response to this finding.

Recommendation: evaluate an explicit create/initialize boundary, completed before application worker processes begin using the store. Ordinary open validates an existing initialized catalog; incompatible or incomplete stores receive a typed non-destructive error. A single-process convenience constructor may initialize a fresh store synchronously. Concurrent first creation, online schema migration, and uninterrupted availability during initialization need not be part of the initial shared-store guarantee.

This is a proposed contract change, not a claim that current lazy-initialization tests can simply be deleted. Trace its compatibility impact and update those tests and docs deliberately. SQLite transactions still protect catalog initialization; orchestration of worker startup belongs to the application/deployment contract. Do not rename or replace a database that other connections may have open.

### 4. High: metadata customization is not a first-class roadmap deliverable

The roadmap devotes substantial detail to backend parity, recovery machinery, and compatibility. REQUIREMENTS has no dedicated acceptance requirement for a direct BlobStore user to define, validate, query, and update a catalog schema. BlobStore offers mapping metadata and filtering, while richer custom-schema/session APIs live on UnifiedCache.

Evidence: `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md:197`, `src/cacheness/storage/blob_store.py:368`, `src/cacheness/core.py:454`, `:529`, `:673`, `src/cacheness/custom_metadata.py`.

This does not mean BlobStore lacks metadata. It means the central customization use case is less precisely specified than the recovery implementation supporting it.

Recommendation: define the catalog contract explicitly: immutable blob identity/locator/digest, separately revisable user attributes, query/index support for declared attributes, and transactional extension points for richer relational metadata where supported. Distinguish customizing catalog fields from replacing the catalog backend. A user should not need to implement the entire lifecycle protocol to catalog blobs differently.

Metadata-only updates currently create a new manifest generation and rehash the payload (`storage/lifecycle.py:327`). Consider separate payload-generation and catalog-revision concepts, so changing a label does not require reading the entire blob. This is a design opportunity requiring signing/compatibility review, not a demonstrated corruption defect.

### 5. High: phase ordering delays the integration that reveals the architecture's flaws

The roadmap explicitly uses eight horizontal phases. Metadata composition is Phase 4, payload parity Phase 5, and cache composition Phase 6. Yet Phase 3 already changed cache publication, signing, queries, custom links, and projection CAS to keep the system functioning.

Evidence: `.planning/ROADMAP.md:5`, Phases 4–6; `03-15`, `03-16`, `03-18`, and `03-20` plans/summaries.

The work has crossed those boundaries while completion is still judged against a foundational phase. This encourages repairs to a transitional composition whose intended simplification remains several phases away.

Recommendation: revise the dependency order around one complete local slice: initialized SQLite catalog + filesystem payloads + customizable metadata + recovery, then a cache instance consuming the same storage service through its supported API. Prove that composition before multiplying backends. Preserve public entry points with thin adapters rather than forcing the new service to maintain every old internal representation during each operation.

### 6. High: the planning system still rewards satisfying every accumulated constraint

The records show two distinct patterns. Historical file-scheduler fixes addressed receipts, inventory events, anchors, platform control objects, and clear admission. The SQLite attempt then accumulated writer admission, fork resets, bootstrap coordination, and exact timeout stages. Plan 20 removed much of the second set, but the next review again prescribed a bootstrap mechanism before validating the necessary product guarantee.

Evidence: `03-REVIEW-FIX.iter2.md`, `.iter10.md`, `.iter20.md`, `.iter27.md`; `03-18-SUMMARY.md`; `03-20-SUMMARY.md`; current review and draft Plan 21.

This is evidence of process and boundary problems. It is not enough to attribute the outcome to a particular model, and this audit does not claim every historical fix introduced a new race. Many fixes closed real integrity defects. The failure is the automatic progression from a newly found interleaving to a new mandatory mechanism, with no effective opportunity to simplify or narrow the contract.

Recommendation: before a finding becomes a plan task, record the violated user-visible invariant, supported topology, smallest deterministic reproduction, and whether it is corruption, recoverable residue, transient failure, unsupported operation, or performance. Review the proposed fix separately from the finding. At an ADR stop condition, compare simplification and contract narrowing before admitting another mechanism. A previously locked implementation choice is not permanently binding when a later accepted ADR supersedes it.

### 7. Medium: the remaining source requirements still invite overclaims

PROJECT asks for “orphan-free cleanup.” BACK-04 describes a three-by-four backend matrix. The roadmap asks for common lifecycle behavior without backend-specific behavior leaking to callers. Those phrases can be read more strongly than the accepted topology-specific ADR permits.

Evidence: `.planning/PROJECT.md:30`, `.planning/REQUIREMENTS.md` BACK-04/BACK-06, `.planning/ROADMAP.md` Phases 4/5.

Recommendation: say that incomplete external effects leave attributable cleanup obligations, which converge when the required resources are available. Publish supported backend/topology combinations and their outcomes. “Backend-neutral API” does not imply identical durability or progress guarantees. Retain advertised compatibility through explicit support tiers or migration/deprecation decisions; do not silently drop a backend in the name of simplification.

## Product shape to aim for

Both an object-store instance and a cache instance use the same storage service and serializers. Each instance can own a separate namespace and catalog. The object-store facade has no implicit TTL or eviction. The cache facade adds key derivation, TTL, eviction, and statistics, and invokes the same storage operations.

The catalog is the authority for blob identity, generation, user metadata, intent, and cleanup obligations. The payload backend owns immutable byte generations. Type handlers continue to produce their native formats. Optional secondary indexes or exports are derived. SqlCache remains separate.

No “dual-role” flag is required just to reuse code. A shared physical store with per-entry retention could be a future feature, but it should not be assumed or imposed on this refactor. The current code already creates an internal cache BlobStore; the evidence points more strongly to duplicate catalog/compatibility responsibilities than to an existing explicit dual-role implementation.

## Disposition of current findings and Plan 21

| Item | Audit disposition |
|---|---|
| CR-01: memory abort leaves dangling reconciliation entry | Valid bounded recovery defect; fix stable identity/terminal-state handling with a public regression. |
| CR-02: mutable memory debt positions invalidate resumes | Valid bounded recovery defect; stable IDs and idempotent retirement are appropriate. Do not turn this into a new scheduling subsystem. |
| CR-03: malformed projection metadata can delete valid canonical state | Highest-priority integrity issue. Strict decoding is necessary; removing derived-catalog authority over deletion is the architectural correction. |
| CR-04: first-use reader sees incompatible empty authority | Valid error-classification/initialization defect. Reconsider concurrent lazy creation before choosing a publication protocol. |
| WR-01: operational SQLite failures mapped to migration | Valid bounded typing fix. Preserve cause and distinguish operational failure from incompatible stored data. |
| Draft 03-21 | Keep as unapproved planning evidence. Revise after the initialization and catalog-boundary decisions; do not execute the current bootstrap proposal. |

## Recommended next work

1. Update the project contract with the user's separate-instance clarification and concrete customizable-catalog acceptance criteria. Reconcile the roadmap and requirements with ADR 0001.
2. Choose and document the initial initialization/maintenance contract. My recommendation is initialization before concurrent worker use; schema migration is explicit maintenance.
3. Specify one catalog transaction boundary for descriptor and authoritative user metadata. Define a supported read result that returns the validated payload and its metadata from one generation, so cache code does not reconstruct that relationship through a second catalog.
4. Replan the remainder of Phase 3 around the local storage slice and the concrete existing integrity/recovery defects. Bring a thin cache composition proof forward before backend multiplication. Do not restart or discard all completed work.
5. Verify public workflows and a finite failure model: first creation under the declared contract; put/reopen; metadata query/update; overwrite conflict; delete; interruption around publication; cleanup retry; corrupt metadata; cache expiry. Add new schedules when a concrete invariant justifies them. Keep performance and platform qualification separate.

Completion should mean the supported workflows satisfy the declared invariants with clear failure outcomes and sufficient recovery evidence. It cannot mean that all conceivable races have been eliminated.

## Work worth retaining

Native serialization without a custom payload wrapper; immutable generations; authenticated/bounded manifests; contained filesystem I/O; exact conditional publication; short SQLite transactions; durable intent and cleanup debt; typed conflict/timeout outcomes; compatibility fixtures; deterministic interruption tests; and honest Windows qualification status are valuable foundations.

SQLite supports multiple readers and one writer, and `BEGIN IMMEDIATE` may return SQLITE_BUSY. These are normal design constraints, not evidence that the requested blob store requires PostgreSQL. See [SQLite transactions](https://www.sqlite.org/lang_transaction.html). PostgreSQL may be appropriate for a shared multi-host catalog, but it does not remove the external-payload recovery boundary.

SQLite also documents hazards around database-file replacement, open-file identity, and journal handling. That is a reason to keep proposed bootstrap file publication small and justified, not proof that every closed-candidate installation is unsafe. See [How to corrupt an SQLite database](https://www.sqlite.org/howtocorrupt.html).
