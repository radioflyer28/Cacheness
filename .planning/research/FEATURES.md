# Feature Landscape

**Domain:** Production-grade Python blob storage foundation with a composed cache-policy layer
**Project:** Cacheness lifecycle refactor milestone
**Researched:** 2026-08-29
**Overall confidence:** MEDIUM — recommendations combine direct codebase evidence with current official documentation; final performance budgets require project-owned baseline measurements.

## Scope Interpretation

This milestone is not a new cache product. It is a reliability refactor with a strict ownership boundary:

- `BlobStore` owns serialization coordination, payload storage, metadata persistence, integrity, deletion, rollback, and reconciliation.
- `UnifiedCache` composes `BlobStore` and owns cache keying, TTL, eviction, invalidation, statistics, and decorator-facing cache behavior.
- `SqlCache` remains a separate pull-through table cache and is protected by regression tests, not absorbed into the blob lifecycle.
- Existing public APIs remain callable through compatibility adapters. Stored representations may change only through an explicit, observable migration or rebuild workflow.

## Table Stakes

Features users expect. Missing = the library cannot credibly advertise reliable storage or production cache semantics.

### Canonical Storage Contract and Atomic Lifecycle

| Feature | Why Expected | Complexity | Dependencies | Notes |
|---------|--------------|------------|--------------|-------|
| Versioned canonical entry manifest | Every backend must agree on key, payload locator, handler/type, format version, size, checksum, timestamps, signing state, and lifecycle generation | High | Public API inventory; metadata normalization | Use one typed internal shape. Backend-specific details belong in a namespaced field, never at inconsistent nesting levels. Persist both manifest schema version and payload format version. |
| Explicit entry state machine | Recovery is impossible if staged, committed, replacing, deleting, and quarantined entries are indistinguishable | High | Canonical manifest | Define valid transitions and commit point. Only `committed` entries are readable through normal APIs. Reject or reconcile impossible states. |
| Staged payload write and atomic publication | A successful read must observe the complete old generation or complete new generation, never a partial payload | High | Blob backend staging primitives; unique temporary identifiers | Filesystem: write unique temp in the destination filesystem, flush/close, then atomic replace. S3: upload an immutable generation object and publish its locator conditionally. Memory: replace one value under a per-key lock. |
| Metadata as the visibility commit point | Payload and metadata cannot share one transaction across filesystem/S3 and SQL/JSON, so the library must define when a new generation becomes visible | High | Entry state machine; staged payload | Publish metadata only after payload write and integrity calculation succeed. Readers follow only the committed generation referenced by metadata. Do not overwrite the old payload before the new manifest commits. |
| Compensating rollback for failed writes | Metadata failure after payload success must not leak an untracked object; payload failure must not mutate live metadata | High | Staging; idempotent delete; fault injection | On pre-commit failure, delete the staged generation. On uncertain remote outcomes, record or discover the candidate during reconciliation. Preserve the previous committed generation until replacement succeeds. |
| Correct overwrite semantics | Same-key replacement is a normal operation and must not delete the last known-good value early | High | Generation identifiers; conditional metadata update | Use copy-on-write generations. After commit, retire the prior generation; if retirement fails, leave a reclaimable orphan rather than breaking the new value. |
| Idempotent `delete`, `clear`, and invalidation primitives | Cleanup retries and crash recovery require repeated calls to converge without corrupting unrelated entries | High | Manifest lookup; backend delete contract | Delete payload and metadata as one lifecycle operation with a documented missing-key result. `clear` inventories entries before removing metadata and reports partial failures. Cache invalidation delegates to this primitive. |
| Read integrity and authenticity verification | A payload must be verified before unsafe or expensive deserialization when integrity/signing is required | High | Canonical checksum/signature fields; streaming checksum API | Verify manifest signature first, then payload checksum, then deserialize. Required verification fails closed with a typed corrupt/untrusted result; it must never silently downgrade to unsigned operation. |
| Deterministic missing/corrupt behavior | Callers need to distinguish an ordinary cache miss from an operational storage failure when using `BlobStore` directly | Medium | Typed error model | `BlobStore` should expose typed not-found/corrupt/conflict/backend errors. `UnifiedCache` may translate configured classes of failure into cache misses while recording reasoned statistics. |
| Resource lifecycle ownership | Database pools, file handles, multipart uploads, temporary files, and executor resources must close predictably | Medium | Backend interface | Support `close()` and context managers; make close idempotent. Avoid relying on `__del__` for meaningful cleanup. Abort incomplete remote uploads where possible. |

### Rollback, Reconciliation, and Operational Repair

| Feature | Why Expected | Complexity | Dependencies | Notes |
|---------|--------------|------------|--------------|-------|
| Dry-run reconciliation inventory | Cross-resource atomicity is unavailable, so production recovery needs a supported scanner rather than ad hoc filesystem scripts | High | Canonical manifest; list/iterate capability on both sides | Classify healthy entries, staged leftovers, payload-only orphans, metadata-only entries, checksum mismatches, superseded generations, and incomplete deletions. Dry-run is the default. |
| Deterministic repair policy | An inventory without a safe action leaves operators to guess | High | Reconciliation inventory; quarantine/delete primitives | Repair or quarantine when provenance is sufficient; otherwise remove metadata for rebuildable cache entries or preserve and report ambiguous artifacts. Never guess a handler or attach an arbitrary payload to metadata. |
| Resumable, idempotent reconciliation | Large S3/PostgreSQL stores cannot restart from zero after every interruption | High | Stable pagination; checkpoint format | Persist cursor/checkpoint and operation outcomes; repeated runs converge. Bound memory and API requests. Report counts, bytes, failures, and remaining work. |
| Failure injection at every lifecycle seam | Rollback behavior is not credible if only happy paths are tested | High | Internal operation boundaries; test doubles | Inject failures before/during/after payload write, checksum, metadata insert/update, old-generation cleanup, delete, list, and close. Include ambiguous remote success and retry cases. |
| Crash-recovery acceptance tests | Exceptions are easier than process death; temp files and half-published state must still reconcile | High | Subprocess test harness; durable test backends | Terminate a writer at defined state transitions, restart, then prove the previous or new committed generation is readable and all other state is reportable/reclaimable. |

### Concurrency and Consistency

| Feature | Why Expected | Complexity | Dependencies | Notes |
|---------|--------------|------------|--------------|-------|
| Per-key writer coordination | Concurrent writes to one key must not interleave payload bytes or commit mismatched metadata | High | Generations; backend conditional operations | Use in-process per-key locks plus backend-level compare-and-swap/version checks for multi-process or multi-host cases. A global write lock is not acceptable. |
| Optimistic generation checks | PostgreSQL/S3 deployments cannot depend on a Python thread lock | High | Manifest generation/version; conditional metadata update | A writer commits only if the observed generation still matches. Return a typed conflict or retry according to a bounded policy; never silently clobber a newer generation. |
| Read-during-overwrite consistency | Readers must see old or new committed data, not a half-state or transient miss | High | Immutable generations; metadata commit point | Keep old generations readable until new metadata commits. Garbage collection must tolerate readers that already resolved the prior locator. |
| Delete-versus-write ordering | A delete racing with a put must have a documented winner and leave no untracked live generation | High | Per-key sequencing; generation checks | Choose linearizable per-key semantics where supported. Tests must cover put/put, put/delete, delete/put, get/overwrite, and clear/write races. |
| Bounded retries and deadlock handling | Remote conflicts, PostgreSQL deadlocks, SQLite busy errors, and S3 conditional conflicts are normal operational events | Medium | Typed transient errors; retry policy | Retry only idempotent stages with jitter/backoff and limits. Surface exhausted conflicts. Acquire multiple locks in a stable order. |
| Explicit topology capabilities | A local memory payload plus persistent metadata is not durable or multi-host merely because the metadata backend is PostgreSQL | Medium | Backend capability descriptors | Expose/test capabilities such as durable, process-shared, host-shared, supports-CAS, supports-streaming, and supports-listing. Reject configurations that cannot meet a requested durability profile. |

### Backend Parity

| Feature | Why Expected | Complexity | Dependencies | Notes |
|---------|--------------|------------|--------------|-------|
| One payload backend protocol | Filesystem, memory, and S3 must support the lifecycle actually used by `BlobStore` | High | Backend-neutral locator and staging model | Contract includes stage/write or stream, read/stream, stat/checksum metadata, publish/conditional write as applicable, delete, exists, list/paginate, and close. Avoid leaking local paths into portable manifests. |
| One metadata backend protocol | JSON, memory, SQLite, and PostgreSQL must expose the same entry, generation, query, stats, and cleanup semantics | High | Canonical manifest; transaction/locking strategy | Backend methods must not reshape entries. Injection and registry construction must preserve the caller-selected instance/name. |
| Parametrized backend contract suite | Class unit tests do not prove interchangeable behavior | High | Protocols; reusable fixtures | Run identical lifecycle, error, pagination, stats, concurrency, and reconciliation expectations against every implementation. Pytest fixture parametrization is designed for this pattern. |
| Full supported pair matrix | Advertising two backend axes implies their compatible combinations work end to end | High | Payload and metadata contracts | Exercise all 3 payloads × 4 metadata stores where the combination is allowed. Capability-based skips require an explicit documented reason; no accidental skips. |
| Real-service integration jobs | Mocks cannot validate PostgreSQL locking, S3 conditional requests, multipart cleanup, authentication, or pagination | High | CI service credentials/containers; isolated test resources | Keep moto/unit tests fast, but gate releases on PostgreSQL and an S3-compatible or AWS integration job. Use unique namespaces and guaranteed cleanup. |
| Backend-neutral statistics and sizes | Eviction and observability cannot depend on a SQLite-only method or an S3-unaware file size | Medium | Canonical manifest; aggregate API | Define count/bytes/type/access aggregates consistently. Prefer database aggregation over materializing all rows. |

### Cache-Policy Layer

| Feature | Why Expected | Complexity | Dependencies | Notes |
|---------|--------------|------------|--------------|-------|
| `UnifiedCache` composes `BlobStore` | There must be one payload-plus-metadata lifecycle, not two almost-identical coordinators | High | Production-ready `BlobStore`; compatibility adapter | `UnifiedCache` delegates put/get/delete/exists/clear and never directly unlinks payloads or writes entry manifests. Keep the public constructor and alias behavior stable. |
| TTL as policy metadata and read decision | Expiration is cache policy, not object-storage behavior | Medium | BlobStore custom/system metadata; clock abstraction | `UnifiedCache` calculates expiry and translates expired reads to misses, then requests lifecycle deletion. `BlobStore` remains usable without TTL semantics. |
| Correct eviction with whole-entry cleanup | Size limits must reclaim payload bytes, not just metadata rows | High | Accurate sizes; lifecycle delete; access ordering | Define deterministic LRU (or the existing documented order), perform bounded eviction, and report failures. Protect the just-written entry from immediate inconsistent removal. |
| Invalidation through storage lifecycle | All invalidation paths must delete payload and metadata consistently | Medium | BlobStore idempotent delete | Single-key, predicate/custom-metadata, TTL, size, decorator clear, and global clear paths converge on the same primitive. |
| Reasoned hit/miss/error statistics | A production cache needs trustworthy observability without counting corruption or backend outages as ordinary misses invisibly | Medium | Typed BlobStore results; metadata aggregate API | Track hit, absent, expired, corrupt/signature failure, conflict, and backend error separately while preserving compatible aggregate counters. |
| Correct cached-`None` and decorator clear behavior | `None` is a valid Python function result and attached cache controls must perform the operation they report | Medium | Explicit found/not-found result; key index/namespace | Stop using `None` as the internal miss sentinel. Make `cache_clear()` invalidate its decorator namespace or documented entries and return an accurate count. |
| `SqlCache` regression isolation | The milestone must not destabilize a separate product while moving shared imports/configuration | Medium | Public API tests; packaging matrix | Keep its modules, builders, and semantics separate. Test imports and representative existing workflows, but do not route it through `BlobStore`. |

### Migration and Compatibility

| Feature | Why Expected | Complexity | Dependencies | Notes |
|---------|--------------|------------|--------------|-------|
| Public API compatibility inventory | Refactors fail when only obvious constructors are preserved | High | Current docs/tests/export scan | Cover imports, aliases, constructor arguments, config names, registries, decorators, result shapes, exceptions, and context-manager behavior. Add contract tests before moving ownership. |
| Compatibility adapters with deprecation policy | Internals can change without forcing immediate caller rewrites | Medium | Compatibility inventory; canonical APIs | Preserve supported calls and issue actionable deprecations only where necessary. Do not maintain two independent lifecycle implementations behind old and new names. |
| Explicit stored-data format/schema versions | Readers and migrators must detect old data rather than infer layouts from missing fields | High | Canonical manifest | Version JSON documents, SQL schemas, and payload/handler format independently. Refuse unknown future versions with a typed error. |
| Dry-run migration plan | Users need to know entry counts, bytes, incompatible handlers, destination capacity, and destructive actions before mutation | High | Inventory/reconciliation; version detection | Produce a machine-readable and human-readable plan. Distinguish metadata-only schema upgrade, payload rewrite/copy, and safe rebuild/drop. |
| Resumable copy-verify-switch migration | In-place rewriting risks destroying the only valid copy | High | New lifecycle; checksums; checkpoints | Copy to a new generation/location, verify read and checksum, conditionally switch metadata, then retire source. Make repeated invocation idempotent and safe after interruption. |
| Documented rebuild path | Cache data is often disposable; a rebuild can be safer and faster than universal conversion | Low | Version detection; clear namespace | Require explicit confirmation/scope, report what will be removed, and preserve public API configuration. Never silently purge incompatible data during normal initialization. |
| Rolling-read compatibility window | A deployed fleet may briefly contain old and new library versions | High | Versioned reader; deployment guidance | If practical, new code reads the immediately previous format while writing only the new format. Otherwise require a stop-the-world migration and state that constraint clearly. |

### Security and Integrity Boundaries

| Feature | Why Expected | Complexity | Dependencies | Notes |
|---------|--------------|------------|--------------|-------|
| Filesystem containment at the backend boundary | Sanitizing only high-level keys leaves direct backend calls and persisted locators vulnerable | High | Resolved base path; platform tests | Resolve symlinks and `..`, reject absolute/drive/UNC escapes, then prove the final path is under the configured root for read, write, delete, and list. Test Linux, macOS, and Windows path rules. |
| Safe structured metadata parsing | Metadata-controlled `eval` or object-array loading broadens trusted-payload assumptions into trusted metadata assumptions | Medium | Handler format revisions | Replace `eval` with strict typed parsing. Default NumPy loads to `allow_pickle=False`; permit trusted object payloads only through an explicit handler with integrity checks and documentation. |
| Fail-closed required signing | A bad/missing key must not silently disable a requested authenticity boundary | Medium | Signer initialization; typed configuration errors | Validate key availability and permissions at startup. If signing is required, reject unsigned or invalid manifests before resolving payload paths. Use constant-time signature comparison. |
| Integrity algorithm and scope contract | A checksum is useful only if the same bytes and metadata are covered consistently | Medium | Canonical serialization; streaming IO | Specify checksum algorithm/version and whether compression bytes or logical content are hashed. Bind critical locator/type/format/generation fields into the signed manifest. |
| Safe metadata query construction | Bound values do not protect interpolated JSON-path keys | Medium | Structured filter schema; backend adapters | Validate field names and build SQL expressions through SQLAlchemy/backend APIs. Reject unsupported nested paths rather than interpolating raw text. |
| Trusted-payload boundary documentation | Pickle/dill cannot be made safe for hostile bytes by this refactor | Low | Public docs; security configuration | State that application payloads are trusted, while metadata, paths, remote responses, configuration, and on-disk artifacts still receive containment and integrity enforcement. |

### Packaging, CI, Testing, and Release Quality

| Feature | Why Expected | Complexity | Dependencies | Notes |
|---------|--------------|------------|--------------|-------|
| Minimal wheel build/install/import smoke test | Source-tree tests in a fully populated environment hide missing core dependencies and omitted package files | Medium | Correct `pyproject.toml`; build tooling | Build sdist and wheel, install the wheel into a clean environment with only mandatory dependencies, import every guaranteed public symbol, and run a basic memory round trip. |
| Independent optional-extra smoke tests | Feature flags are credible only if each advertised extra installs and imports by itself | Medium | Truthful optional dependency groups | Test S3, PostgreSQL, YAML, dataframe/array, and other advertised extras individually; test the no-extra case. Feature availability checks must inspect the real dependency. |
| Supported Python and OS CI matrix | `requires-python >=3.11` is a claim about more than the developer machine | Medium | Reproducible lock/install strategy | Gate Python 3.11 through the newest supported version on Linux. Add filesystem/package smoke jobs on Windows and macOS, especially for atomic replacement and path containment. |
| Fast PR gate plus full backend gate | Developers need fast feedback without making production backends optional forever | Medium | Test markers; CI workflows | PR gate: unit, local contracts, moto, Ruff, coverage, wheel smoke. Protected/release gate: PostgreSQL, real S3-compatible/AWS tests, migration, crash recovery, and benchmarks. |
| Clean baseline before ratcheting | A gate that starts red is ignored | Medium | Fix two YAML failures; lint plan | Make the current suite pass, eliminate repository Ruff violations or adopt a temporary explicit baseline, then reject new violations. Do not silently exclude lifecycle modules. |
| Coverage focused on critical behavior | A global percentage can hide an untested `BlobStore` | Medium | Branch coverage; test ownership | Milestone target: at least 90% statements and 85% branches for new/refactored lifecycle and policy modules; raise repository statement coverage from 66% to at least 75% without excluding `SqlCache` from reporting. Ratchet upward after the milestone. |
| Property/state-machine lifecycle tests | Handwritten examples rarely cover long put/overwrite/delete/clear/reconcile sequences | High | Stable model semantics; Hypothesis optional dev dependency | Compare every backend pair with an in-memory reference model. Invariants: committed metadata resolves to exactly one valid payload; no readable uncommitted payload; reconciliation is idempotent; size/count aggregates match inventory. |
| Concurrency stress and deterministic race tests | A same-key test that passes once does not establish ordering | High | Hooks/barriers around state transitions | Use barriers to force important interleavings, then add repeated thread/process stress. For PostgreSQL/S3, use separate clients. Validate results and absence of orphans, not merely lack of exceptions. |
| Migration compatibility fixtures | Stored-data compatibility must be tested against real previous layouts | Medium | Versioned golden fixtures; migration API | Check in small, non-secret artifacts from supported old versions. Test read, dry-run, migrate, resume-after-failure, and rebuild. Never regenerate fixtures with the code under test. |

### Performance Acceptance

| Feature | Why Expected | Complexity | Dependencies | Notes |
|---------|--------------|------------|--------------|-------|
| Checked-in benchmark scenarios and metadata | “No regression” is meaningless without fixed workloads and environment context | Medium | Correct implementation; benchmark tool | Cover hit, miss, put, overwrite, delete, clear, TTL cleanup, size eviction, reconciliation, and stats at 1/100/10,000 entries with representative 1 KiB, 1 MiB, and streaming-large payloads. Record Python, OS, dependency, backend, payload, and concurrency. |
| Baseline-relative local performance budget | Atomicity will add work, but it should not allow an unbounded regression | Medium | Stable dedicated runner; saved baseline | Final gate recommendation: no individual memory/filesystem/SQLite benchmark regresses more than 20% in median latency, and the geometric-mean regression stays within 10%, versus a checked-in pre-release baseline. Rebaseline only through reviewed changes with rationale. |
| Tail latency and contention budget | Mean-only benchmarks hide lock convoys and cleanup pauses | High | Concurrent benchmark harness | Measure p50/p95/p99 and throughput at 1, 8, and 32 workers for distinct-key and same-key loads. Require distinct-key throughput to scale without a global lock; document intentional same-key serialization. |
| Remote performance trend gate | S3/PostgreSQL latency is noisy, but large regressions and excessive API calls must still be caught | High | Stable test account/service; request instrumentation | Track latency plus request/query count and bytes. Use a wider initial alert threshold (25%) and manual confirmation before blocking; strictly gate accidental N+1/list-all behavior with operation-count assertions. |
| Memory and inventory bounds | Reconciliation, clear, and stats must not load an unbounded store into RAM | Medium | Pagination/streaming contracts | Demonstrate bounded memory at 10,000+ entries and stable batch sizes. JSON remains a small/local backend; document its scale envelope rather than disguising whole-document costs. |
| Correctness gate precedes performance gate | Fast corrupt writes are not an acceptable optimization | Low | All lifecycle tests green | Tune and freeze budgets only after lifecycle, migration, security, and backend contracts pass. Any optimization must rerun failure-injection and concurrency suites. |

## Differentiators

Features that set the library apart. Not strictly required for basic correctness, but valuable once table stakes are stable.

| Feature | Value Proposition | Complexity | Dependencies | Notes |
|---------|-------------------|------------|--------------|-------|
| First-class `audit()` / `reconcile()` API with explainable plans | Most Python caches make orphan repair an operator script; built-in dry-run and machine-readable repair reports make remote/local deployments operable | High | Canonical manifest; inventory; typed repair actions | Promote only after reconciliation proves idempotent across all backend pairs. |
| Backend conformance kit for third-party implementations | Extension authors can run the exact lifecycle contract before registration | Medium | Stable backend protocols and pytest fixtures | Publish fixtures/reference models separately from internal tests and include capability declaration validation. |
| Generation-aware optimistic concurrency API | Advanced callers can perform put-if-absent or replace-if-version without introducing another locking system | Medium | Conditional commit and generation tokens | Expose optional preconditions while preserving simple `put`. Particularly useful for S3/PostgreSQL. |
| Migration planner with copy/verify/switch and progress reports | Makes stored-layout evolution safer than “delete the cache and hope” for expensive artifacts | High | Versioned manifests; reconciliation; checkpoints | Support explicit rebuild as a cheaper alternative; do not promise arbitrary downgrade. |
| Durability profiles validated against backend capabilities | Users can request process-local, host-durable, or multi-host-shared semantics and receive early configuration errors | Medium | Capability descriptors; topology tests | Prevents misleading combinations such as persistent remote metadata with ephemeral memory payloads. |
| Structured lifecycle telemetry | Entry-state counts, repair outcomes, conflict rates, checksum failures, and backend timings shorten incident diagnosis | Medium | Typed events/errors; statistics model | Keep hooks dependency-neutral; logging/metrics adapters can be added without turning policy into a plugin framework. |
| Streaming integrity-preserving transfers | Large artifacts can be written/read without full memory buffering while retaining checksums and atomic publication | High | Streaming handler/backend protocols; staged commit | Valuable for S3 and large arrays/dataframes; should not delay small-object correctness unless existing APIs already require it. |

## Anti-Features

Features to explicitly NOT build in this milestone.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Distributed transaction abstraction across S3/filesystem and SQL metadata | These resources do not share one atomic transaction; pretending otherwise hides ambiguous failures | Use immutable staged generations, a metadata commit point, compensating cleanup, conditional writes, and reconciliation. |
| General cache-policy plugin framework | Lifecycle and backend contracts are not stable enough; this would multiply integration states before correctness exists | Keep a clean `UnifiedCache`→`BlobStore` seam and implement the existing TTL/eviction/invalidation policy directly. |
| Merge or rewrite `SqlCache` into `BlobStore` | Its range/query/gap/upsert lifecycle is a distinct product and would expand risk dramatically | Keep it separate and protect public imports plus representative behavior with regression tests. |
| New backend families | More adapters dilute work needed to make the six advertised backend implementations reliable | Finish filesystem, memory, S3, JSON, SQLite, and PostgreSQL with full contracts first. |
| Silent automatic destructive migration on import/startup | Unexpected network cost, downtime, data loss, and irreversible partial upgrades are unacceptable | Detect versions, report incompatibility, and require an explicit dry-run plus migrate or rebuild command/API. |
| Byte-for-byte preservation of flawed stored layouts | It would force new internals to retain inconsistent metadata nesting and local path assumptions forever | Preserve APIs; provide a bounded compatibility reader and explicit copy/verify/switch migration. |
| Global lock around all cache operations | It is simple but destroys distinct-key concurrency and cannot coordinate multiple processes/hosts | Use per-key locks locally and backend generation/CAS semantics for shared deployments. |
| “Exactly once” retries | Network responses can be lost after a remote operation succeeds; claiming exactly-once is misleading | Design idempotent operations, use request/generation identifiers, inspect state after ambiguity, and reconcile. |
| Best-effort signing fallback | Continuing unsigned after a configured signing failure violates the operator’s stated security boundary | Fail construction or reads closed when signing is required; offer an explicit non-strict mode only when configured. |
| Hostile pickle/dill sandbox | Python’s object serializers can execute code and cannot be made safe here without changing the supported data model | Keep payloads explicitly trusted, authenticate artifacts, and offer safe handlers/formats for users who need untrusted input. |
| Transparent distributed cache coherence or replication | S3/PostgreSQL availability does not create coherent in-memory caches across hosts, and this is not required for lifecycle unification | Document consistency/topology capabilities and rely on conditional metadata generations. |
| Async API expansion | It doubles surface area and test matrices without solving the current correctness failures | Stabilize the synchronous lifecycle; later add async adapters only from measured demand. |
| Deduplication/reference counting as a release requirement | Shared payload ownership greatly complicates deletion, migration, and reconciliation | Use one immutable generation per committed entry initially. Consider deduplication after reference invariants are proven. |

## Feature Dependencies

```text
Public API compatibility inventory
    └── Canonical versioned manifest + typed errors
            ├── Payload backend protocol (filesystem / memory / S3)
            ├── Metadata backend protocol (JSON / memory / SQLite / PostgreSQL)
            ├── Security boundary (path containment / parsing / signing / checksum)
            └── Entry state machine + generation tokens
                    └── Staged write + metadata visibility commit
                            ├── Rollback / idempotent delete
                            ├── Same-key concurrency / conditional commit
                            ├── Read-during-overwrite semantics
                            └── Reconciliation / crash recovery
                                    └── Production BlobStore lifecycle
                                            ├── UnifiedCache composition
                                            │       ├── TTL
                                            │       ├── invalidation
                                            │       ├── size eviction
                                            │       └── statistics / decorators
                                            └── Stored-data migration
                                                    ├── dry-run inventory
                                                    ├── copy / verify / switch
                                                    └── resumable checkpoints

Backend protocols + lifecycle invariants
    └── Parametrized contract and state-machine tests
            ├── Full 3 × 4 backend pair matrix
            ├── failure injection / race / crash tests
            └── CI, packaging, coverage, and service gates
                    └── checked-in benchmarks and final regression budgets
```

Ordering matters. Implementing `UnifiedCache` composition before `BlobStore` has reliable delete/rollback semantics simply moves current leaks behind a new facade. Running migrations before manifests and reconciliation are versioned makes failures unrecoverable. Freezing performance budgets before lifecycle correctness stabilizes rewards preserving defective shortcuts.

## MVP Recommendation

For this milestone, prioritize:

1. **Lock the compatibility and entry contracts.** Add public API characterization tests, one canonical versioned manifest, typed lifecycle results/errors, backend capabilities, and explicit state transitions.
2. **Make `BlobStore` the only lifecycle owner.** Connect all payload and metadata backends; implement staged writes, conditional commit, integrity verification, idempotent delete/clear, overwrite generations, rollback, close, and correct existence/listing.
3. **Prove recovery and concurrency.** Add per-key coordination, backend CAS/generation checks, failure injection, deterministic race tests, audit/reconciliation, and crash-restart tests.
4. **Compose `UnifiedCache` as policy.** Route TTL, invalidation, size eviction, stats, cached-`None`, and decorator clear through `BlobStore`; remove duplicate payload manipulation. Keep `SqlCache` separate.
5. **Ship an explicit migration/rebuild workflow and hard security boundary.** Version old layouts, dry-run inventory, copy/verify/switch with checkpoints, safe rebuild, path containment, safe parsers, bound queries, and fail-closed required signing.
6. **Turn quality into release gates.** Clean baseline tests/lint, minimal wheel and extras smoke tests, supported Python/OS matrix, full backend contracts, PostgreSQL/S3 jobs, targeted coverage thresholds, then checked-in correctness-aware benchmarks.

Defer: general policy plugins, new backends, deduplication, async APIs, hostile-pickle support, distributed coherence, and `SqlCache` redesign. They increase state-space without improving the milestone’s core value: one dependable payload-plus-metadata lifecycle.

## Acceptance Summary by Area

| Area | Release Acceptance | Confidence |
|------|--------------------|------------|
| Atomic lifecycle | Every supported backend pair returns old or new committed generation; injected partial failures leave previous data readable and all residue removable by idempotent reconciliation | MEDIUM |
| Concurrency | Deterministic same-key race suite passes for threads and shared backends; distinct keys are not serialized by a global lock | MEDIUM |
| Backend parity | Common contract passes for filesystem/memory/S3 payloads and JSON/memory/SQLite/PostgreSQL metadata, with real PostgreSQL and S3 service coverage | MEDIUM |
| Migration | Dry-run, migrate, interruption/resume, verify, and explicit rebuild pass against checked-in old-layout fixtures | MEDIUM |
| Security | No `eval` on metadata; contained filesystem operations; bound query construction; required signing and integrity fail closed before deserialization | HIGH for boundary requirements; MEDIUM for implementation specifics |
| Packaging | Clean minimal wheel imports and round-trips; each advertised extra installs/imports independently; supported Python matrix is green | HIGH |
| Testing/coverage | Current failures resolved; lint gate green/baselined; lifecycle/policy ≥90% statements and ≥85% branches; repository statements ≥75% | LOW for exact numeric targets until effort is measured |
| Performance | Stable-runner baseline stored; local individual median regression ≤20%, geometric mean ≤10%; remote trend alert ≤25%; no unbounded inventory memory or N+1 calls | LOW until benchmarks establish variance and baseline |

## Sources

### Project Evidence

- `.planning/PROJECT.md` — milestone scope, active requirements, constraints, and measured baseline. **Confidence: HIGH**
- `.planning/codebase/ARCHITECTURE.md` — traced ownership boundaries and current two-step payload/metadata flows. **Confidence: HIGH**
- `.planning/codebase/CONCERNS.md` — independently reproduced lifecycle, packaging, security, concurrency, and cleanup defects. **Confidence: HIGH**
- `.planning/codebase/TESTING.md` — executed test/lint/coverage baseline and test gaps. **Confidence: HIGH**

### Current Official Documentation

- [Amazon S3 consistency model](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html) — strong per-key visibility/atomicity, but no atomic cross-key updates and no built-in writer lock. **Confidence: MEDIUM (official source, cross-checked)**
- [Amazon S3 conditional writes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html) and [PutObject API](https://docs.aws.amazon.com/AmazonS3/latest/API/API_PutObject.html) — `If-Match`/`If-None-Match`, 409/412 conflicts, and retry implications. **Confidence: MEDIUM (official sources, cross-checked)**
- [Amazon S3 object integrity](https://docs.aws.amazon.com/hands-on/latest/amazon-s3-with-additional-checksums/amazon-s3-with-additional-checksums.html) — upload/download and at-rest checksum capabilities. **Confidence: MEDIUM (official source)**
- [Python `os.replace`](https://docs.python.org/3.12/library/os.html#os.replace) — successful same-filesystem replacement is atomic under POSIX and provides cross-platform overwrite semantics. **Confidence: MEDIUM (official source)**
- [SQLite isolation](https://www.sqlite.org/isolation.html), [transactional guarantees](https://sqlite.org/transactional.html), and [WAL](https://www.sqlite.org/wal.html) — serializable transactions, single-writer behavior, atomic commit, and WAL topology limits. **Confidence: MEDIUM (official sources, cross-checked)**
- [PostgreSQL explicit locking](https://www.postgresql.org/docs/current/explicit-locking.html) — row locking, deadlock behavior, transaction-scoped release, and stable lock ordering. **Confidence: MEDIUM (official source)**
- [Python `pathlib`](https://docs.python.org/3/library/pathlib.html) — resolve symlinks/`..` before ancestry checks; lexical checks alone are insufficient. **Confidence: MEDIUM (official source)**
- [Python pickle security warning](https://docs.python.org/3/library/pickle.html) — unpickle only trusted bytes and use HMAC when tamper detection is required. **Confidence: MEDIUM (official source)**
- [Alembic migration tutorial](https://alembic.sqlalchemy.org/en/latest/tutorial.html) — versioned relational change scripts, current revision, and ordered upgrade/downgrade model. **Confidence: MEDIUM (official source)**
- [PyPA `pyproject.toml` specification](https://packaging.python.org/en/latest/specifications/pyproject-toml/) and [packaging flow](https://packaging.python.org/en/latest/flow/) — required dependencies, optional extras, wheel build/install semantics. **Confidence: MEDIUM (official sources, cross-checked)**
- [GitHub Actions Python build/test matrix](https://docs.github.com/en/actions/tutorials/build-and-test-code/python) — supported-version matrix mechanics. **Confidence: MEDIUM (official source)**
- [pytest parametrization](https://docs.pytest.org/en/stable/how-to/parametrize.html) and [parametrized fixtures](https://docs.pytest.org/en/stable/how-to/fixtures.html#parametrizing-fixtures) — one behavior suite across implementations/configurations. **Confidence: MEDIUM (official sources, cross-checked)**
- [Hypothesis stateful testing](https://hypothesis.readthedocs.io/en/latest/reference/api.html#stateful-tests) — model-based sequences for lifecycle invariants. **Confidence: MEDIUM (official source)**
- [Ruff CI integration](https://docs.astral.sh/ruff/integrations/) — pass/fail lint enforcement in CI. **Confidence: MEDIUM (official source)**
- [pytest-benchmark comparison](https://pytest-benchmark.readthedocs.io/en/v5.0.0/comparing.html) — saved-run comparison and absolute/percentage regression gates. **Confidence: MEDIUM (official source)**

## Research Gaps

- Exact performance thresholds are recommendations, not observed Cacheness results. Establish variance on a stable runner before making them blocking.
- The supported rolling-upgrade window depends on historical manifest/payload samples and how widely older releases are deployed.
- Real S3 acceptance may use AWS or an S3-compatible service; conditional-write and checksum parity must be verified against the chosen target rather than assumed from moto.
- Full Windows/macOS lifecycle semantics need implementation-time probes, especially open-file replacement, directory fsync expectations, symlinks, drive/UNC paths, and process locks.
- The intended support lifetime for existing stored formats is not yet defined; choose a concrete window during migration-phase planning.
