# Architecture Patterns

**Domain:** Backend-neutral Python object storage with a cache-policy layer
**Project:** Cacheness layered storage refactor
**Researched:** 2026-08-29
**Overall confidence:** HIGH for codebase integration; MEDIUM for the exact migration and cross-process locking implementation choices

## Recommended Architecture

Make `BlobStore` the only public object-lifecycle coordinator. `UnifiedCache` keeps its public facade but becomes a client of `BlobStore`: it derives cache keys, decides freshness, selects eviction victims, performs invalidation, and records cache statistics. It must not serialize payloads, resolve physical paths, call metadata backends, or delete backend objects directly. `SqlCache` remains an independent subsystem.

```text
Public API / compatibility adapters                         SqlCache (unchanged)
  cacheness alias, decorators, get_cache                           │
                    │                                               └─ SQLAlchemy table + fetch adapter
                    ▼
        UnifiedCache — cache policy only (modified)
        keying · TTL · eviction · invalidation · stats
                    │
                    │ BlobStore contract only
                    ▼
        BlobStore — lifecycle owner (modified)
        put/get/stat/list/touch/delete/clear/reconcile
          │            │                 │
          ▼            ▼                 ▼
  Handler codec   Payload backend   Metadata repository
  adapter         filesystem        JSON
  (new)           memory            memory
                  S3                SQLite
                                    PostgreSQL
          │            │                 │
          └────────────┴─────────────────┘
                canonical StoredEntry + BlobRef (new)

        Explicit migration service + legacy reader (new)
                    │
          old actual_path/nested records → versioned records
```

The core consistency model is deliberately not a distributed transaction. Payload backends and metadata backends do not share a transaction manager. Instead, payloads are written under immutable, generation-specific physical identifiers; a conditional metadata write publishes one generation as the current logical key. Metadata publication is the commit point. Failed publication leaves an unreferenced payload that is deleted immediately when possible and deterministically collected by reconciliation otherwise.

### Component Boundaries

| Status | Component | Responsibility | Must Not Own | Likely Location |
|---|---|---|---|---|
| **Modified** | `UnifiedCache` | Cache-key derivation, TTL decisions, eviction ordering, invalidation intent, hit/miss statistics, public miss behavior | Serialization, payload references, integrity verification, payload deletion, metadata backend selection | `src/cacheness/core.py`, with policy helpers extracted only as needed |
| **Modified** | `BlobStore` | Canonical payload-plus-metadata lifecycle; public direct-storage API; typed error-to-compatibility-result mapping | TTL, LRU/LFU policy, cache hit/miss meaning, function argument keying | `src/cacheness/storage/blob_store.py` |
| **New** | Lifecycle value objects | One normalized entry shape, opaque payload reference, revisions, receipts, delete/reconciliation reports | Backend I/O or policy | `src/cacheness/storage/models.py` |
| **New** | Handler codec adapter | Convert existing path-oriented handlers to/from a lifecycle-owned staged artifact; keep format metadata stable | Final payload location or metadata publication | `src/cacheness/storage/codecs.py` |
| **Modified** | `CacheHandler` implementations | Type detection and format-specific serialization/deserialization | Choosing final filesystem/S3 locations | `src/cacheness/interfaces.py`, `src/cacheness/handlers.py` |
| **Modified** | `BlobBackend` contract and implementations | Store/read/delete/list immutable byte objects and return/consume opaque backend references | Logical keys, cache TTL, metadata records | `src/cacheness/storage/backends/blob_backends.py`, `s3_backend.py` |
| **Modified** | Metadata backend contract and implementations | Atomic entry compare-and-swap, tombstones, listing, touch/counter primitives, schema version | Payload I/O or cache eviction decisions | Consolidate the duplicate contracts in `metadata.py` and `storage/backends/base.py` |
| **New** | Reconciler | Repair crash windows: missing payloads, unreferenced generations, tombstones, abandoned staging artifacts | Selecting cache eviction victims | `src/cacheness/storage/reconciliation.py` |
| **New** | Migration service + legacy adapter | Inspect, dry-run, migrate, verify, resume, and optionally rebuild old stored entries | Silent startup conversion or indefinite dual-write support | `src/cacheness/storage/migration.py` |
| **Modified** | Backend/config factories | Honor injected instances and registries; validate backend capability/durability combinations | Reimplementing backend selection in coordinators | `src/cacheness/config.py`, `storage/backends/__init__.py` |
| **Modified** | Custom metadata facade | Preserve registered-model APIs as an auxiliary, non-authoritative metadata extension routed through BlobStore's metadata unit of work | Payload references, cache visibility, or independent lifecycle deletion | `src/cacheness/custom_metadata.py` |
| **Modified** | Security/integrity services | Sign and verify the canonical entry; fail closed when configured; hash serialized bytes | Cache miss policy | `src/cacheness/security.py`, `file_hashing.py` |
| **Unchanged boundary** | `SqlCache` | Range-aware SQL pull-through dataframe caching | Object/blob lifecycle integration | `src/cacheness/sql_cache.py` |

### Canonical Contracts

Use typed internal records even if compatibility APIs continue to accept and return dictionaries.

```python
@dataclass(frozen=True)
class BlobRef:
    backend: str               # "filesystem", "memory", "s3"
    locator: str               # opaque to callers; interpreted only by that backend
    generation: str            # immutable physical-object generation
    size_bytes: int
    digest_algorithm: str      # e.g. "sha256"; never infer integrity from an S3 ETag
    digest: str
    backend_version: str | None = None   # S3 version ID or backend CAS token
    etag: str | None = None              # optimization/concurrency token, not content hash

@dataclass(frozen=True)
class StoredEntry:
    schema_version: int
    key: str
    revision: int              # logical metadata revision used by CAS
    state: Literal["active", "tombstone"]
    blob: BlobRef | None
    data_type: str
    storage_format: str
    codec_version: int
    codec_metadata: Mapping[str, JsonValue]
    user_metadata: Mapping[str, JsonValue]
    policy_metadata: Mapping[str, JsonValue]  # opaque to BlobStore
    created_at: datetime
    signature_version: int | None
    signature: str | None
```

`actual_path` must disappear from the canonical model. Filesystem paths, `memory://` identifiers, and `s3://` URIs are backend locators inside `BlobRef`; only the selected payload backend may interpret them. A record must also carry an independent cryptographic digest because S3 ETags are not universally content hashes.

Required internal operations:

| Contract | Semantics |
|---|---|
| `payload.write(generation, stream) -> BlobRef` | Writes a new immutable generation. A successful return means later read sees complete bytes, never a partial object. |
| `payload.read(ref) -> stream` | Reads the exact generation in `ref`; never reconstructs a path from the logical key. |
| `payload.delete(ref) -> bool` | Idempotently deletes only the exact referenced generation. |
| `payload.iter_refs(namespace)` | Supports bounded reconciliation inside a store-owned namespace. |
| `metadata.get(key) -> StoredEntry | None` | Returns a normalized entry on every backend. |
| `metadata.publish(key, entry, expected_revision) -> revision` | Atomic compare-and-swap; `None` means create-only. Raises a typed conflict on a race. |
| `metadata.touch(key, accessed_at)` | Updates policy-owned access state without changing the payload generation. |
| `metadata.iter_entries(state=...)` | Stable, paginated inventory for cleanup, migration, and policy selection. |
| `BlobStore.delete_if_revision(key, revision)` | Tombstones only the generation the caller evaluated, preventing eviction from deleting a concurrent replacement. |

Registry construction must be the single injection seam. Constructors accept instances first, registered names second, and built-in defaults last. An explicitly supplied backend is never overwritten by config selection. Capability flags should describe atomic publication, streaming, conditional writes, listing, durability, and process/distributed coordination; configuration validates combinations rather than scattering backend-name branches through `BlobStore`.

The existing registered custom-metadata API is an auxiliary index, not the authority for whether a payload exists. Route its create/query/delete operations through `BlobStore` so `UnifiedCache` still composes only the storage facade. When the custom rows and entry record share SQLite/PostgreSQL, publish them in the same metadata transaction. A backend that cannot provide the requested atomic extension must reject that configuration explicitly rather than commit an entry and silently swallow the extension failure. Hit/miss counters are different: they are policy telemetry owned by `UnifiedCache`, updated through a narrow counter/touch gateway, and a counter failure must not roll back an otherwise valid storage commit.

### Data Flow

#### Put / Replace

1. `UnifiedCache` derives the logical cache key and policy metadata, or a direct `BlobStore` caller supplies a key.
2. `BlobStore` validates the key and selects a handler. The handler codec adapter serializes into a unique lifecycle-owned staging artifact; handlers never choose the final payload location.
3. `BlobStore` computes the payload digest and size over the exact serialized bytes. Required signer initialization and canonical-record validation happen before publication; required security cannot silently degrade.
4. The payload backend writes those bytes under a unique generation such as `<namespace>/<logical-key>/<uuid>`. Filesystem uses a unique temporary sibling plus same-filesystem atomic replace; memory publishes under a lock; S3 writes a unique object key.
5. `BlobStore` builds and signs a canonical `StoredEntry`, then calls `metadata.publish(..., expected_revision=observed_revision)`. Backend-supported registered custom metadata participates in that metadata unit of work. This conditional metadata publication is the logical commit point.
6. On success, the new generation is visible. The previous generation becomes eligible for deferred deletion. On conflict or metadata failure, the new generation is not visible and is deleted best-effort; reconciliation removes it if compensation fails.
7. `UnifiedCache` applies size policy only after a successful commit. Any selected victim is deleted with its observed revision.

#### Get

1. `UnifiedCache` evaluates freshness from an entry view obtained through `BlobStore.stat`; a direct caller simply invokes `BlobStore.get`.
2. `BlobStore` loads one normalized active record, verifies its signature before following the locator, and reads the exact `BlobRef` through the selected payload backend.
3. `BlobStore` verifies size and digest before deserialization. Required signing or integrity failures raise typed internal errors and quarantine/tombstone the exact revision; compatibility facades may map them to the existing `None`/miss behavior.
4. If the referenced generation disappears during a concurrent replacement, `BlobStore` re-reads metadata once. A changed revision retries the new reference; an unchanged revision is a real missing/corrupt payload and is reconciled.
5. The handler codec reconstructs the object. `UnifiedCache` then records hit/access policy state; BlobStore does not interpret hit/miss semantics.

#### Delete / Expire / Evict / Clear

1. `UnifiedCache` decides *why* an entry should go; `BlobStore` decides *how* payload and metadata are removed.
2. `BlobStore.delete_if_revision` conditionally publishes a tombstone at the next revision. The tombstone commit makes the logical key unavailable before physical deletion.
3. It deletes the exact `BlobRef`. Success permits tombstone purge after a bounded retention window; failure leaves cleanup intent durable for reconciliation.
4. `clear` is an idempotent series of per-key tombstone/delete operations and returns a structured report. Do not promise global atomic clear across S3 and a database.

### Commit, Rollback, and Reconciliation Semantics

| Failure window | Visible state | Required response |
|---|---|---|
| Serialization/staging fails | Old entry remains visible | Remove staging artifact; publish nothing. |
| Payload write fails | Old entry remains visible | Delete partial backend object if one exists; publish nothing. |
| Payload succeeds, metadata publish fails | Old entry remains visible; new generation is orphaned | Delete new generation best-effort and record/log a typed cleanup failure; reconciler later removes it. |
| Two same-key publishes race | Exactly one CAS wins | Losing writer deletes its generation; caller receives deterministic conflict/retry behavior. |
| New metadata commits, old payload deletion fails | New entry is visible; old generation is orphaned | Keep success result, enqueue/reconcile old generation; never roll metadata back to the old value. |
| Tombstone commits, payload deletion fails | Key remains logically deleted | Retain tombstone and retry deletion; never resurrect from payload presence. |
| Active metadata points to missing/corrupt payload | Entry is not a valid hit | Fail closed, tombstone/quarantine exact revision, record miss/integrity event, reconcile. |
| Process crashes at any step | Determined by metadata record | Startup/explicit reconciliation applies the same rules idempotently. |

Reconciliation compares two inventories inside a store-owned namespace:

- active record + present valid payload: keep;
- active record + missing/invalid payload: tombstone or quarantine and report;
- tombstone + present payload: retry exact deletion, then age out tombstone;
- payload generation not referenced by any active record and older than the grace period: delete;
- abandoned handler staging artifact older than its deadline: delete;
- malformed, unknown-version, or wrong-backend reference: quarantine and require migration/operator action.

The grace period prevents a just-written payload from being collected before its metadata publish. Reads that began on an old generation are allowed to finish; a reader that loses that race retries the current revision once. Per-key in-process locks are useful contention optimizations, but CAS in the metadata repository is the correctness boundary across instances.

Backend-specific implementations of the same contract:

| Backend | Publication / coordination semantics |
|---|---|
| Filesystem payload | Unique temp file in the destination directory, flush/fsync as configured, atomic `os.replace`, containment check on every locator operation. Never use a fixed `<key>.tmp`. |
| Memory payload | Lock-protected immutable generation map; reference is process-scoped and explicitly ephemeral. |
| S3 payload | Unique object key per generation; capture version ID/ETag; use conditional operations where overwriting/deleting a known generation is required. S3 single-key operations are atomic and strongly consistent, but metadata CAS still selects the logical winner. |
| JSON metadata | Reload-under-lock, compare revision, write a complete versioned document to a unique temp file, atomic replace. Add an inter-process store lock if multi-process use remains supported. |
| Memory metadata | Lock-protected CAS; process-local guarantee only. |
| SQLite metadata | One explicit transaction containing revision check and insert/update; indexed revision/state fields. SQLite serializes writers, so keep transactions short. |
| PostgreSQL metadata | Row-level CAS (`UPDATE ... WHERE revision = :expected`) or transaction-scoped advisory lock keyed by store/key; commit/rollback in one explicit transaction. Prefer CAS for normal writes and advisory locks only for compound maintenance. |

### Migration Seam

Public API compatibility and stored-data compatibility are separate promises. Preserve constructors, aliases, decorator behavior, and dictionary-shaped public metadata through adapters, while making stored-data conversion explicit.

1. Add `schema_version` to every canonical entry and a store-level format version. Fresh stores start at the current version.
2. Implement a read-only `LegacyEntryAdapter` for the current top-level/nested `actual_path` shapes. It may serve legacy reads during a documented transition, but it must not silently rewrite entries.
3. Provide `inspect_store()` and `plan_migration()` before mutation. The plan reports counts, bytes, unknown handlers, missing files, integrity failures, backend compatibility, and whether rebuild is safer.
4. `migrate_store()` is dry-run by default, idempotent, resumable, and journaled. For each entry: read legacy metadata → read/serialize or copy payload → write new generation → verify digest and decode → CAS-publish versioned entry → retire old payload only after commit.
5. Use versioned relational schema migrations for SQLite/PostgreSQL; use atomic snapshot/backup replacement for JSON. A library-owned migration journal spans relational, JSON, and payload conversion.
6. Offer an explicit rebuild command/path for caches whose contents are reproducible or whose old handler cannot be migrated. Never guess at unknown formats.
7. Avoid dual writes. They multiply failure states and make rollback ambiguous. A bounded dual-read period plus explicit one-way migration is sufficient.

### Future Cache-Policy Seam

Do not add a policy registry, dynamic loading, entry points, or user-selectable policy classes in this milestone. Instead, leave the seam by keeping all policy decisions in `UnifiedCache` and making them depend only on backend-neutral entry views and these `BlobStore` operations: `stat`, `list`, `touch`, `delete_if_revision`, and storage usage summaries.

TTL evaluation, victim ordering, and size thresholds should be isolated pure functions or small private helpers that accept entry snapshots and configuration. Later, those helpers can satisfy a `CachePolicy` protocol without changing `BlobStore` or backend contracts. The milestone is successful when a future policy implementation would replace decision logic in `UnifiedCache`, not when a plugin framework exists.

## Patterns to Follow

### Pattern 1: Immutable Payload, Mutable Pointer

**What:** Each write creates a new immutable physical generation. A versioned metadata record points the logical key to exactly one generation.
**When:** Always, including local filesystem and memory; using one model across all payload backends removes same-key overwrite races.
**Example:**

```python
old = metadata.get(key)
ref = payload.write(new_generation(key), staged.stream)
try:
    committed = metadata.publish(key, build_entry(ref), expected_revision=old.revision if old else None)
except RevisionConflict:
    payload.delete(ref)
    raise
```

### Pattern 2: Metadata as the Visibility Commit Point

**What:** A payload is not live merely because it exists. Only an active, signed metadata record makes it addressable.
**When:** Put, replace, migration, and recovery.
**Example:** Reconciliation may freely remove old unreferenced generations after a grace period because payload presence cannot resurrect a key.

### Pattern 3: Revision-Qualified Destructive Operations

**What:** Cleanup carries the revision it evaluated and tombstones through CAS.
**When:** TTL expiration, size eviction, invalidation, integrity quarantine, and migration retirement.
**Example:** An eviction scan that saw revision 4 cannot delete revision 5 written concurrently.

### Pattern 4: Compatibility at the Edges

**What:** Preserve old call signatures and metadata dictionaries in facades, normalize immediately into internal typed records, and emit one canonical shape from repositories.
**When:** Throughout the refactor.
**Example:** Keep `BlobStore(backend="sqlite")` as a deprecated alias for its historical metadata-backend meaning while adding unambiguous `metadata_backend=` and `blob_backend=` arguments.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Treating Filesystem Paths as Storage References

**What:** Passing `Path(actual_path)` through `BlobStore` and handlers.
**Why bad:** It cannot represent memory/S3 safely, leaks backend rules upward, breaks containment, and caused the existing nested-metadata deletion bugs.
**Instead:** Persist `BlobRef`; stage locally only inside the codec adapter.

### Anti-Pattern 2: Metadata-Only Cleanup

**What:** Letting `cleanup_expired`, `clear_all`, or `remove_entry` erase metadata directly.
**Why bad:** It leaks payloads and destroys the information needed to clean them later.
**Instead:** Policy selects a revision; BlobStore tombstones and deletes the exact generation.

### Anti-Pattern 3: In-Place Same-Key Payload Replacement

**What:** Two writers target the same final file or fixed `.tmp` name.
**Why bad:** Readers can see mismatched metadata/payload and writers can corrupt one another.
**Instead:** Unique immutable generations plus metadata CAS.

### Anti-Pattern 4: Pretending Cross-Backend Atomicity

**What:** Reporting a rollback as complete merely because a compensating delete was attempted.
**Why bad:** S3/network failures can leave objects behind and no transaction spans payload plus metadata.
**Instead:** Define the commit point, make compensation idempotent, persist tombstones where possible, and reconcile.

### Anti-Pattern 5: Backend Branches in `UnifiedCache`

**What:** TTL or eviction code checks for filesystem, S3, SQLite, or PostgreSQL names.
**Why bad:** Cache policy becomes coupled to storage topology and blocks later policy replacement.
**Instead:** Depend on typed capability-neutral `BlobStore` operations.

### Anti-Pattern 6: Silent Migration on Read or Startup

**What:** Automatically rewrite legacy records when first encountered.
**Why bad:** A routine cache read becomes destructive, failures are difficult to resume, and rollback is unclear.
**Instead:** Bounded legacy reads plus an explicit dry-run-capable migrator or rebuild.

## Dependency-Aware Build Order

| Order | Deliverable | Depends On | Exit Gate |
|---|---|---|---|
| 1 | Characterization fixtures and backend contract test harness | Existing behavior | Legacy API fixtures, fault-injection matrix, same-key race tests, and known stored-entry samples checked in. |
| 2 | Canonical models, typed errors, and one metadata contract | 1 | All metadata backends normalize to one entry shape; duplicate ABCs no longer diverge. |
| 3 | Handler codec/staging adapter | 2 | Every built-in handler round-trips through bytes/streams without owning final locations; current handler APIs remain adapted. |
| 4 | Payload backend contract hardening | 2–3 | Filesystem, memory, and S3 pass the same immutable-generation, containment, list, integrity, and concurrency tests. |
| 5 | Metadata CAS and schema versioning | 2 | JSON, memory, SQLite, and PostgreSQL pass create/replace/conflict/tombstone/touch tests; injected/registered backends are honored. |
| 6 | BlobStore lifecycle + reconciliation | 3–5 | Faults at every crash window leave either a valid committed entry or a reconciliable orphan; delete/clear never lose cleanup intent. |
| 7 | Explicit migration/rebuild tooling | 6 | Dry-run, resume, verification, rollback/backup, legacy reads, and cross-backend migration are covered. |
| 8 | Rewire `UnifiedCache` as policy layer | 6–7 | `UnifiedCache` has no handler-path or metadata-backend calls; TTL, eviction, invalidation, and stats use only BlobStore contracts. |
| 9 | Public compatibility and decorator integration | 8 | Existing constructors, alias, direct keys, decorators, `None` values, and management methods meet compatibility tests. |
| 10 | Production quality matrix and performance budgets | All | Python 3.11+ CI, minimal install, optional S3/PostgreSQL jobs, lint/coverage gates, security tests, reconciliation soak tests, and checked-in benchmarks pass. |

Security work is not deferred to the end: safe parsing, signer fail-closed behavior, and canonical integrity validation enter with steps 3–6. Step 10 turns those contracts into release gates and measures regression budgets after correctness stabilizes.

## Scalability Considerations

| Concern | At 100 users/entries | At 10K entries | At 1M entries |
|---|---|---|---|
| Metadata inventory | JSON/memory acceptable for single-process use | Prefer SQLite; paginate listing and aggregate sizes in SQL | PostgreSQL with indexed state/revision/access fields and cursor pagination |
| Payload layout | Filesystem or memory | Filesystem sharding or S3 | S3 with namespace prefixes, paginated inventory, lifecycle metrics |
| Same-key contention | In-process keyed lock + CAS | CAS remains authoritative | Backoff/jitter on conflicts; PostgreSQL row CAS; no global lock |
| Reconciliation | Full scan | Paginated incremental scan with grace period | Checkpointed partitions/prefixes; rate limits and bounded work per run |
| Eviction | Full list acceptable | Indexed oldest-access/size query | Policy requests pages/candidates; revision-qualified deletes |
| Migration | One-shot with backup | Resumable journal | Partitioned, rate-limited copy/verify/publish with progress metrics |

JSON metadata remains a portability backend, not the recommended high-write production backend. Memory payload or metadata is explicitly process-scoped. Durable metadata paired with ephemeral payloads must be rejected or require an explicit ephemeral/reconciliation mode; otherwise restarts create guaranteed dangling records. PostgreSQL metadata plus S3 payload is the intended distributed combination, while SQLite plus filesystem is the default durable local combination.

## Sources

### Codebase Evidence — HIGH confidence

- `.planning/PROJECT.md` — target layering, compatibility, backend, reliability, and scope constraints.
- `.planning/codebase/ARCHITECTURE.md` — current call paths and disconnected backend registries.
- `.planning/codebase/CONCERNS.md` — lifecycle leaks, concurrency gaps, injection bugs, integrity boundaries, and test gaps.
- `src/cacheness/core.py` — current direct handler writes, metadata-only invalidation/expiration, signing, and broken size cleanup.
- `src/cacheness/storage/blob_store.py` — duplicated lifecycle and nested `actual_path` inconsistency.
- `src/cacheness/storage/backends/blob_backends.py`, `s3_backend.py` — current raw blob contracts and backend capabilities.
- `src/cacheness/metadata.py`, `storage/backends/postgresql_backend.py` — current entry shapes and transaction behavior.
- `src/cacheness/interfaces.py`, `handlers.py` — path-oriented handler contract that requires the staging adapter.

### External Platform Semantics — MEDIUM confidence via research seam, official sources cross-checked

- [Python `os.replace` documentation](https://docs.python.org/3.11/library/os.html#os.replace) — successful same-filesystem replace is atomic on POSIX.
- [Amazon S3 consistency model](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html#ConsistencyModel) — strong read-after-write consistency and atomic updates to one key.
- [Amazon S3 conditional writes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html) — `If-None-Match` and ETag-based `If-Match` concurrency controls.
- [SQLite isolation](https://www.sqlite.org/isolation.html) — serialized writers and journal/WAL isolation behavior.
- [PostgreSQL explicit/advisory locking](https://www.postgresql.org/docs/current/explicit-locking.html#ADVISORY-LOCKS) — application-defined transaction-scoped coordination.
- [SQLAlchemy session transaction framing](https://docs.sqlalchemy.org/en/20/orm/session_basics.html#framing-out-a-begin-commit-rollback-block) — explicit begin/commit/rollback context boundaries.
- [Alembic migration tutorial](https://alembic.sqlalchemy.org/en/latest/tutorial.html) — versioned relational change scripts and migration environment.

## Open Decisions for Phase Planning

- Select the cross-platform inter-process lock implementation for JSON metadata and filesystem maintenance; validate it on the supported OS matrix.
- Decide tombstone retention and orphan grace defaults from fault-injection/soak results rather than intuition.
- Decide whether the first migration supports physical payload transfer across all backend pairs or limits v1 to in-place format migration plus documented rebuild.
- Define public behavior for direct `BlobStore.get` on integrity failure: typed exception is safer internally, but compatibility may require an opt-in strict mode before changing the default.
