# Domain Pitfalls

**Domain:** Refactoring a Python cache-first library into a backend-neutral blob-store foundation with cache policy layered above it
**Project:** Cacheness
**Researched:** 2026-08-29
**Overall confidence:** MEDIUM — current code paths were inspected directly and critical distributed-storage claims were cross-checked against current official documentation; production S3/PostgreSQL behavior still needs phase-specific integration validation.

## Critical Pitfalls

Mistakes that cause rewrites, data loss, silent corruption, or an unsafe cutover.

### Pitfall 1: Treating Payload and Metadata as One Transaction

**What goes wrong:** A payload write succeeds and metadata publication fails, leaving an orphan; or metadata commits and the payload is missing, incomplete, or points at an older generation. A delete can fail in the opposite direction. No filesystem, memory, or S3 blob operation participates in the JSON/SQLite/PostgreSQL metadata transaction.

**Why it happens:** The old cache-first flow performs two writes in sequence and calls the operation successful if the second step returns. Refactoring the calls into `BlobStore` without defining an explicit visibility/commit protocol preserves the same split-brain failure under a cleaner API.

**Consequences:** Reads become nondeterministic misses or return stale data, cleanup leaks storage, retries overwrite evidence needed for recovery, and operators cannot tell whether metadata or payload is authoritative.

**Warning signs:**

- `put()` writes directly to the final logical key before a metadata generation/version exists.
- Recovery behavior is described only as “roll back the payload” even though rollback can also fail.
- Metadata contains a user key but no immutable payload locator, generation, checksum, lifecycle state, or schema version.
- Tests assert an exception but do not inventory both stores after failure.
- Orphan counts, dangling-reference counts, or last reconciliation time are not observable.

**Prevention:** Make metadata publication the visibility point. Stage an immutable generation-specific payload, calculate and verify an explicit checksum and size, then conditionally publish metadata that references that exact generation. If publication fails, leave or best-effort-delete the unreferenced generation and let an idempotent reconciler collect it after a safety grace period. Reads use only committed metadata; they never discover staged blobs by listing. Make `put`, `delete`, and reconciliation retry-safe and specify the outcome for failure after every lifecycle step.

**Address in build order:** Phase 1, canonical entry schema and lifecycle state machine, before implementing any backend pair. Prove the protocol first with memory/fault-injection doubles, then implement local backends in Phase 2 and remote backends in Phase 3.

### Pitfall 2: Overwriting the Same Physical Payload During Concurrent Writes

**What goes wrong:** Two writers for one key interleave bytes or replace the same temporary file; metadata from writer A points to bytes from writer B; a slower writer wins after a newer write; a delete removes a newly published payload after reading stale metadata.

**Why it happens:** Per-instance Python locks do not coordinate processes or hosts, and Cacheness currently creates a lock without guarding the full payload-plus-metadata lifecycle. The filesystem backend also derives one `.tmp` name per final path. PostgreSQL/SQLite metadata locks cannot protect an object written outside their transaction.

**Consequences:** Same-key races produce valid-looking but semantically wrong entries. Hash checks may not help when metadata and payload from the same losing writer are paired incorrectly, and lock-based fixes can deadlock or destroy throughput.

**Warning signs:**

- Payload locators are derived only from the public cache key rather than key plus immutable generation/content digest.
- A fixed `<key>.tmp` staging path is used.
- Concurrency tests use many distinct keys but never force two writes, delete-versus-put, or cleanup-versus-get on one key.
- Correctness depends on `threading.Lock`, including for PostgreSQL/S3 deployments.
- Metadata updates are unconditional last-write-wins assignments with no generation comparison.

**Prevention:** Write each attempt to a unique immutable locator. Publish with a compare-and-swap generation/version check or a backend-specific same-key critical section, and define the intended winner rule. On PostgreSQL use a row lock, unique-key upsert with version predicate, or transaction-level advisory lock plus bounded retry; on SQLite use a short write transaction and bounded `BUSY` retry; on JSON use an inter-process lock or document single-process scope. Delete the exact generation observed, never “whatever is currently at this key.” Establish one global lock ordering for operations that touch multiple keys.

**Address in build order:** Define generation/CAS semantics in Phase 1; implement and stress same-key thread/process behavior with filesystem/memory and JSON/SQLite in Phase 2; validate multi-host contention with PostgreSQL/S3 in Phase 3.

### Pitfall 3: Implementing Delete, TTL, Eviction, and Clear as Metadata-Only Operations

**What goes wrong:** Invalidation and expiration make entries invisible but leak payloads; `clear()` erases the inventory before it can delete objects; an old cleanup pass deletes a new generation; or eager payload deletion races with a reader that already resolved metadata.

**Why it happens:** Cache policy historically owns cleanup and metadata is cheap to enumerate. Once `BlobStore` owns storage lifecycle, all removal paths must converge on the same generation-aware primitive rather than each policy feature calling a metadata backend directly.

**Consequences:** Unbounded disk/S3 cost, dangling metadata, read failures, unsafe repair scripts, and irrecoverable loss of the only payload inventory. Cacheness already exhibits metadata-only invalidation/cleanup and `BlobStore.clear()` behavior.

**Warning signs:**

- `UnifiedCache` calls `metadata_backend.remove_entry()` or `cleanup_expired()` directly.
- `BlobStore.clear()` delegates to metadata `clear_all()` before enumerating exact payload locators.
- Eviction selects by key but deletion does not include the observed generation.
- Tests verify that `list()` is empty without checking filesystem/S3 inventory.
- Cleanup has no tombstone/lease/grace-period semantics and no retry queue.

**Prevention:** Put one idempotent generation-aware delete primitive in `BlobStore`. Mark/tombstone or CAS-remove visibility first, then delete the recorded immutable payload; retain enough tombstone/retry information to finish after a crash. TTL, eviction, invalidation, corruption handling, and clear must call this primitive. Reconciliation should distinguish safe unreferenced generations from newly staged writes using lifecycle state and age.

**Address in build order:** Phase 1 contract and state model, Phase 2 local delete/recovery implementation, then Phase 4 when `UnifiedCache` policy paths are redirected exclusively through `BlobStore`.

### Pitfall 4: Mistaking S3 Per-Object Consistency for Cross-Store Atomicity

**What goes wrong:** The implementation either adds obsolete polling because it assumes S3 LIST is eventually consistent, or—more dangerously—assumes strong S3 consistency makes an S3 object plus PostgreSQL/SQLite metadata atomic. Concurrent writes overwrite one key without preconditions, delete markers/versioning alter existence behavior, and retryable `409`/`412` outcomes are treated as generic failures.

**Why it happens:** S3 semantics are reduced to “remote filesystem.” S3 now gives strong read-after-write consistency and atomic single-key updates, but conditional operations, versioning, multipart uploads, permissions, and a separate metadata store still create a distributed lifecycle.

**Consequences:** Lost updates, stale generation publication, leaked multipart uploads or objects, incorrect retry behavior, and production-only bugs hidden by an in-memory mock.

**Warning signs:**

- Normal writes use stable logical S3 keys and unconditional `PutObject`.
- The code retries all S3 errors identically or treats `412 Precondition Failed` as transient.
- Object version ID/ETag/checksum are discarded after upload.
- Correctness depends on bucket listing immediately after write rather than the returned locator.
- S3 behavior is tested only with moto and not with a real versioned/unversioned bucket matrix.

**Prevention:** Use immutable generation keys and publish the returned locator. Where mutable object keys are unavoidable, use `If-None-Match` for create-if-absent or `If-Match` against the observed ETag and handle `409`/`412` explicitly. Keep bucket versioning behavior visible in the locator/model. Use direct GET/HEAD for known objects, not LIST as the commit oracle. Abort/expire incomplete multipart uploads operationally. Treat upload success and metadata commit as separate steps with reconciliation.

**Address in build order:** Phase 3, after the lifecycle contract and local reference implementation are stable. Do not make S3 the first backend used to discover the state machine.

### Pitfall 5: Using ETag as a Universal Integrity Hash

**What goes wrong:** Integrity passes or fails incorrectly because an S3 ETag is assumed to be an MD5 of the full payload. Multipart uploads and some server-side encryption modes produce ETags that are not full-object MD5 digests.

**Why it happens:** S3 exposes ETag conveniently through PUT/HEAD, and existing code describes it as a content hash. That is only conditionally true.

**Consequences:** False corruption reports, inability to verify migrated objects, or a security boundary that authenticates the wrong value.

**Warning signs:**

- Metadata has `etag` but no algorithm-qualified payload checksum.
- Multipart and SSE-KMS cases are absent from contract tests.
- Local XXH3 values and S3 ETags are compared as though they share an algorithm.
- Integrity validation happens only after handler deserialization.

**Prevention:** Define an algorithm-qualified digest in the canonical entry schema and compute it over the exact serialized payload bytes. For S3, provide/check a supported full-object checksum during upload or store the library-computed digest in authenticated metadata. Retain ETag/version ID for conditional requests and object identity, not as the portable integrity contract. Verify before unsafe deserialization.

**Address in build order:** Phase 1 schema and integrity contract; implement local checksum flow in Phase 2 and S3 checksum/ETag separation in Phase 3.

### Pitfall 6: Assuming Relational Transactions Include External Blobs

**What goes wrong:** A PostgreSQL or SQLite transaction is held open while uploading a large payload, yet rollback still cannot undo the external object. Long transactions amplify lock contention; serializable failures retry uploads; database commit succeeds but process death prevents cleanup.

**Why it happens:** ACID metadata backends create a false sense that the whole `BlobStore.put()` is transactional. They protect only rows/pages in that database.

**Consequences:** Lock queues, deadlocks, `SQLITE_BUSY`, PostgreSQL serialization failures, connection exhaustion, duplicate large uploads, and persistent split-brain state.

**Warning signs:**

- Upload/serialization occurs inside a database session/transaction block.
- A retry decorator repeats the entire operation without an idempotency token or immutable locator.
- PostgreSQL uses `SERIALIZABLE` but has no `40001` whole-transaction retry test.
- SQLite uses default DEFERRED transactions for read-then-write publication under contention.
- Database sessions remain “idle in transaction” while file/network I/O runs.

**Prevention:** Serialize and upload before opening the short metadata publication transaction. Within the transaction, lock/CAS only the logical key row and publish the verified locator. Make retry reuse or safely abandon the immutable generation. Use bounded retry with jitter for transient database/S3 errors, never for validation or precondition failures. Keep cleanup/reconciliation outside the user request transaction.

**Address in build order:** Phase 1 transaction boundary; SQLite implementation in Phase 2; PostgreSQL locking, isolation, retry, and deadlock tests in Phase 3.

### Pitfall 7: Breaking Stored Data While Preserving Only Method Signatures

**What goes wrong:** Public constructors and methods still import, but old entries become misses, point at wrong locations, deserialize under the wrong handler, or are silently rewritten in a format older processes cannot read. Key derivation or path normalization changes can make every existing entry undiscoverable.

**Why it happens:** API compatibility is confused with persistence compatibility. Current metadata shapes already disagree about nested versus top-level `actual_path`, and the new backend-neutral locator/schema will necessarily differ from legacy direct paths.

**Consequences:** Surprise cold-cache events, data loss for users treating `BlobStore` as durable artifact storage, failed rollback, and unbounded duplicate payloads.

**Warning signs:**

- No explicit entry schema version, key-algorithm version, handler/format version, or backend locator type.
- New code guesses legacy shape from field presence or file extension.
- Migration happens implicitly during ordinary reads with no dry run, report, checkpoint, or failure ledger.
- Tests create “legacy” fixtures using current code rather than frozen golden artifacts from released versions.
- Rollback is claimed after new writes stop being readable by the previous release.

**Prevention:** Freeze representative released metadata/payload fixtures first. Add explicit schema, key, handler, and locator versions. Use expand–migrate–contract: dual-read old/new, write the new canonical form, provide an explicit idempotent inventory/migrate/rebuild command with dry-run and resumable checkpoints, verify counts/checksums, then remove legacy reads only in a later compatibility boundary. Do not silently treat a legacy miss as successful migration.

**Address in build order:** Capture golden compatibility fixtures in Phase 0; define version fields in Phase 1; build explicit migration/rebuild tooling before cutover in Phase 5; contract legacy support only after production validation.

### Pitfall 8: Moving Integrity Checks Behind Unsafe Deserialization

**What goes wrong:** The refactor reads remote/local bytes into pickle, dill, NPZ object arrays, or an unsafe metadata parser before authenticating the entry and verifying payload integrity. Legacy compatibility can silently allow unsigned entries even when signing is configured as required.

**Why it happens:** Handler APIs currently combine reading and deserialization, and cache behavior treats corruption as a miss. Backend neutrality can obscure where the trust boundary belongs.

**Consequences:** Tampered metadata can redirect reads; crafted payloads can execute code before rejection; security configuration degrades silently; corruption cleanup may delete only metadata and preserve the hostile blob.

**Warning signs:**

- `handler.get()` runs before digest/signature verification.
- Signer initialization catches errors and continues with `signer=None`.
- `allow_unsigned_entries` is used as a permanent compatibility default rather than a time-bounded migration option.
- The signature omits payload digest, backend locator, handler/format version, or generation.
- Corruption tests mutate bytes but do not exercise metadata tampering, missing keys, key rotation, and downgrade paths.

**Prevention:** Treat application payloads as trusted only after metadata/path boundaries and tamper checks pass. Authenticate the canonical metadata envelope—including generation, locator, checksum algorithm/value, size, and handler version—then fetch and verify bytes before deserialization. Replace `eval`-style parsing, default NPZ to `allow_pickle=False` where compatible, and fail construction/read closed when signing is required but unavailable. Define key rotation and legacy-signature migration explicitly.

**Address in build order:** Phase 0 security baseline and malicious-fixture tests; finalize the authenticated envelope in Phase 1; enforce it in every backend contract in Phases 2–3 before composing `UnifiedCache`.

### Pitfall 9: Enforcing Path Safety Only in `BlobStore` Key Sanitization

**What goes wrong:** Direct backend callers or tampered legacy metadata supply an absolute path, traversal component, symlink escape, or foreign S3 bucket/key, bypassing higher-level key sanitization. Delete is especially dangerous because it acts on persisted locators.

**Why it happens:** Validation is placed at the public key boundary instead of every backend locator boundary. The current filesystem backend transforms `..` but accepts caller-provided read/delete paths directly; the S3 parser warns on a bucket mismatch and continues.

**Consequences:** Reads or deletes outside the configured storage root, cross-tenant object access, corruption of unrelated files, and security regressions introduced by migration tooling.

**Warning signs:**

- Filesystem `read_blob`/`delete_blob` call `Path(locator)` without resolved containment checks.
- Symlink behavior is undocumented or untested.
- An `s3://other-bucket/key` locator logs a warning but still acts on the parsed key in the configured bucket.
- Migration accepts arbitrary absolute `actual_path` values without an allowlisted source root.
- Error messages/logs expose credentials or signed URLs.

**Prevention:** Use a structured backend-owned locator rather than arbitrary strings. Resolve filesystem candidates and require containment under the configured root for read, write, delete, and migration; define symlink policy and use least-privilege permissions. Validate S3 scheme, bucket, prefix, and expected account/endpoint; reject mismatches. Make migration source roots explicit. Keep credentials out of metadata and logs, relying on the SDK credential chain and temporary roles.

**Address in build order:** Phase 0 for immediate path/parser fixes; Phase 1 locator contract; backend-specific enforcement in Phases 2–3; adversarial migration tests in Phase 5.

### Pitfall 10: Letting `UnifiedCache` Continue Owning Storage Side Effects

**What goes wrong:** `UnifiedCache` nominally composes `BlobStore` but still writes handler files, edits storage metadata, verifies/deletes payloads, or invokes metadata cleanup directly. Two lifecycle implementations survive and drift.

**Why it happens:** Moving all behavior at once is risky, so compatibility code accumulates as permanent bypasses. Policy concepts (TTL, hit/miss, LRU) also leak downward into `BlobStore`, while storage concepts leak upward.

**Consequences:** Backend selection remains cosmetic, cleanup differs by entry path, security fixes must be duplicated, and future cache policies cannot reuse the storage foundation.

**Warning signs:**

- `UnifiedCache` imports concrete blob or metadata backend classes.
- It stores `actual_path`, calls handlers directly for persistence, or removes metadata directly.
- `BlobStore` contains TTL, eviction ordering, or cache hit/miss counters.
- Backend contract tests pass but `UnifiedCache(blob_backend="s3")` still creates local payload files.
- `SqlCache` starts depending on `BlobStore` despite being explicitly out of scope.

**Prevention:** Establish a narrow storage API with complete lifecycle results/errors and make it the only payload/metadata owner. `UnifiedCache` should map keys, decide TTL/eviction/invalidation, maintain cache statistics, and invoke generation-aware `BlobStore` operations. Add architectural tests/spies proving no direct handler/backend writes occur. Keep `SqlCache` imports, schema, tests, and release behavior isolated.

**Address in build order:** Phase 1 boundaries; local/remote `BlobStore` completeness in Phases 2–3; only then compose and remove storage bypasses from `UnifiedCache` in Phase 4.

## Moderate Pitfalls

### Pitfall 11: Promising One Concurrency Model Across JSON, Memory, SQLite, and PostgreSQL

**What goes wrong:** Thread safety is advertised as process/distributed safety, or the lowest backend silently violates the canonical contract. In-memory locks protect one instance only; JSON whole-document rewrites lose updates across processes; SQLite has one writer and WAL requires a same-host database; PostgreSQL coordinates hosts but needs explicit lock/retry policy.

**Warning signs:** A single “thread-safe” flag covers all backends; JSON is accepted for multi-process writers; SQLite files are placed on NFS; PostgreSQL operations rely on a per-process lock; no capability metadata documents deployment scope.

**Prevention:** Keep lifecycle correctness invariant but document backend capability envelopes. Use process-wide locking/atomic replace for supported JSON modes or explicitly constrain it to one writer. Keep SQLite local, configure bounded busy timeout/WAL deliberately, and test one-writer contention. Use database-enforced generation/CAS or transaction-level locks for PostgreSQL. Reject unsupported configurations instead of degrading silently.

**Address in build order:** Phase 1 capability contract; Phase 2 local backend limits; Phase 3 distributed guarantees and configuration validation.

### Pitfall 12: Designing Only the Happy-Path Backend Interface

**What goes wrong:** A minimal `write/read/delete/exists` protocol cannot express staging, conditional publish, immutable generation, checksum, idempotency, not-found versus forbidden, or retryability. High-level code starts using `hasattr`, backend type checks, or swallowed exceptions.

**Warning signs:** Boolean return values collapse absent, forbidden, conflict, and transport error; `get_size()` downloads the whole object by default on a hot path; S3-only methods leak into `UnifiedCache`; exceptions are logged and converted to false/empty lists.

**Prevention:** Define typed results/errors and the semantics required by the lifecycle, not just common method names. Separate capabilities that are optional optimizations from correctness requirements. Contract-test every advertised payload/metadata combination and fail configuration when a required capability is missing.

**Address in build order:** Phase 1 before backend implementation; prohibit backend-specific branching above `BlobStore` in Phase 4.

### Pitfall 13: Relying on JSON Metadata as a Production Hot Path

**What goes wrong:** Every access-time/stat update rewrites the entire metadata document. Multiple processes lose updates, large caches stall on serialization and fsync, and a crash/permission error can strand the replacement file.

**Warning signs:** Put/get latency grows linearly with entry count; metadata bytes written greatly exceed payload bytes; `.tmp` files accumulate; access-time updates dominate profiles; a process-local lock is the only concurrency control.

**Prevention:** Position JSON as simple/small/single-writer unless a real inter-process protocol is built. Batch noncritical access/stat updates, use unique atomic replacement files with durability rules, and recommend SQLite/PostgreSQL for larger or concurrent deployments. Benchmark at realistic entry counts, not just large payload sizes.

**Address in build order:** Phase 2 contract and scale limits; final performance budgets in Phase 6.

### Pitfall 14: Building Reconciliation from Expensive Full Scans Only

**What goes wrong:** Startup cleanup lists every S3 object and every metadata row, delaying startup and increasing API/database cost. Concurrent staging objects are mistaken for orphans. Repeated list/head/delete calls are unbounded.

**Warning signs:** `cleanup_on_init` performs a global scan; no prefix/generation/age checkpoints exist; reconciliation has no page limit or resume cursor; S3 LIST is used per request; deletion rate is unbounded.

**Prevention:** Maintain lifecycle state/timestamps, use generation-specific prefixes, reconcile incrementally with durable cursors and a safety grace period, and make it an explicit bounded maintenance operation. Prefer metadata-driven dangling-reference checks and separately scheduled object inventory for orphan discovery. Emit counts and timing.

**Address in build order:** Define recoverable states in Phase 1; implement bounded local reconciliation in Phase 2 and paginated/rate-limited S3 reconciliation in Phase 3; benchmark in Phase 6.

### Pitfall 15: Testing Mocks Instead of Failure Semantics

**What goes wrong:** Unit tests prove method calls while missing real SQLite locks, PostgreSQL isolation/retries, S3 conditional/versioning/checksum behavior, process crashes, and filesystem durability/path semantics. A large passing suite gives false confidence because `BlobStore` and production backends remain lightly covered.

**Warning signs:** Moto is the only S3 job; PostgreSQL tests skip by default with no required service CI job; backend tests do not share one parameterized contract; fault injection stops at “backend raises”; no child-process kill/restart test inventories both stores.

**Prevention:** Build a parameterized backend-pair contract suite and a deterministic failure matrix for every step: serialize, stage, verify, publish, post-commit response loss, tombstone, delete, and reconciliation. Add same-key thread and process races. Keep fast doubles, but require containerized PostgreSQL and a real S3-compatible/AWS semantics job for release. Assert payload inventory, metadata inventory, generation, checksum, and observable metrics after each test.

**Address in build order:** Start the contract/fault harness in Phase 0–1 and make each backend phase pass it before advancing; make production service jobs mandatory in Phase 6.

### Pitfall 16: Optimizing Before the Lifecycle Is Stable

**What goes wrong:** Write batching, async cleanup, shared mutable keys, cached metadata, and zero-copy streaming make partial failure and ordering harder to reason about. Conversely, the completed refactor can regress badly through whole-object buffering, per-hit metadata commits, full-table statistics, and repeated hashing.

**Warning signs:** Benchmarks compare only warm single-thread memory hits; default streaming implementations read entire objects; every cache hit writes access time synchronously; size enforcement scans/materializes every row; directory/file parameters are fully rehashed on every lookup; performance changes lack a checked-in baseline.

**Prevention:** Prioritize a correct synchronous state machine, while recording baseline workloads early. After semantics freeze, benchmark p50/p95 hit and put latency, throughput under same/distinct-key contention, peak memory for large streams, metadata write amplification, S3 request counts, cleanup duration, and object-count scaling. Optimize behind unchanged contract tests and set explicit regression budgets by backend class.

**Address in build order:** Record baseline scenarios in Phase 0, avoid premature concurrency tricks in Phases 1–4, then optimize and set release budgets in Phase 6.

### Pitfall 17: Losing Registry and Dependency-Injection Compatibility

**What goes wrong:** The canonical path accepts only literal built-in names, overwrites injected backend instances, imports optional dependencies eagerly, or constructs backend pairs differently between direct `BlobStore` and `UnifiedCache` use.

**Warning signs:** Registered custom backends pass registry unit tests but cannot be selected end to end; injected object identity changes; minimal wheel import requires NumPy/S3/PostgreSQL extras; direct and composed construction produce different configuration validation.

**Prevention:** Make one factory/composition root authoritative, preserve explicitly injected instances, resolve registered names consistently, validate pair capabilities there, and lazy-load extras only when selected. Add minimal-install and each-extra-in-isolation CI jobs plus identity-preservation tests.

**Address in build order:** Phase 1 composition root; backend construction in Phases 2–3; public compatibility verification in Phase 4; packaging matrix gate in Phase 6.

## Minor Pitfalls

### Pitfall 18: Letting Statistics Change Storage Correctness

**What goes wrong:** A failed hit/access-time/statistics update turns a successful payload read into a miss or blocks the hot path; statistics counters drift from actual entries and drive incorrect eviction.

**Warning signs:** Reads update several metadata fields synchronously; statistics and entry publication share one large transaction; denormalized counters are not updated on every lifecycle path.

**Prevention:** Treat statistics as policy telemetry with explicit best-effort or transactional semantics, never as the source of payload truth. Eviction inventory must come from canonical entry records, and aggregate queries should run in the database rather than materializing all rows.

**Address in build order:** Phase 4 policy composition; performance/correctness validation in Phase 6.

### Pitfall 19: Using Finalizers as the Resource-Ownership Model

**What goes wrong:** SQLAlchemy engines or streams close during interpreter teardown when modules/exceptions are partially destroyed, causing ignored errors and leaked resources.

**Warning signs:** Correct cleanup depends on `__del__`; tests pass but emit teardown exceptions; returned S3 streaming bodies are not context-managed.

**Prevention:** Make `close()` and context-manager ownership explicit and idempotent; keep finalizers minimal and exception-safe; test repeated close and interpreter/subprocess shutdown across supported Python versions.

**Address in build order:** Phase 2 for local resources, Phase 3 for remote streams/pools, Python-version gate in Phase 6.

### Pitfall 20: Retaining Short Hashes Without Collision Verification

**What goes wrong:** Truncated 64-bit cache/content identifiers eventually collide, and a new lifecycle may overwrite or alias unrelated entries while appearing internally consistent.

**Warning signs:** The locator is a 16-hex digest with no stored full digest/input fingerprint; content-addressable mode falls back to `repr`; collisions are resolved by overwrite.

**Prevention:** Store an algorithm-qualified full digest or verify full identity on collision; version the key algorithm and preserve legacy lookup through the migration layer. Never use `repr` as a durable content identity fallback.

**Address in build order:** Phase 1 key/entry schema; migration mapping in Phase 5.

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation / Exit Evidence |
|-------------|----------------|----------------------------|
| **Phase 0 — Characterization and security baseline** | Refactor begins without frozen legacy behavior, malicious fixtures, or performance baseline | Golden artifacts from released layouts; current public API contract tests; path/parser/signature fail-closed tests; baseline workloads; explicit statement that `SqlCache` is unchanged |
| **Phase 1 — Canonical BlobStore contract and entry model** | A prettier API preserves dual writes and ambiguous entry shapes | Versioned canonical envelope; immutable generation locator; lifecycle state machine; typed errors; checksum/signature scope; capability model; deterministic failure matrix passing with doubles |
| **Phase 2 — Local lifecycle (filesystem/memory + JSON/SQLite)** | “Atomic rename” is mistaken for full crash durability; JSON/SQLite locks are over-promised | Unique same-directory staging, atomic replace and documented durability; path containment; same-key thread/process stress; SQLite busy/WAL tests; crash/restart reconciliation; JSON deployment limits |
| **Phase 3 — Remote lifecycle (S3 + PostgreSQL)** | S3 consistency/ETag and PostgreSQL ACID are mistaken for a distributed transaction | Immutable S3 generations; explicit checksums; conditional operations; versioning cases; short PostgreSQL CAS/lock transactions; retry/deadlock tests; real-service CI; bounded reconciliation |
| **Phase 4 — UnifiedCache composition** | Policy layer keeps direct storage escape hatches or pushes TTL into BlobStore | Spies/architecture tests prove all payload and metadata lifecycle calls route through `BlobStore`; TTL/eviction/invalidation/statistics remain policy-only; every removal uses exact-generation delete |
| **Phase 5 — Stored-data migration and compatibility** | Public API passes while persisted entries disappear or rollback becomes impossible | Dry-run inventory; explicit schema/key/backend versions; idempotent resumable migrate/rebuild; golden legacy fixtures; count/checksum report; mixed old/new read window; documented rollback boundary |
| **Phase 6 — Production quality and performance gates** | Mocks, skips, and microbenchmarks hide production failure modes | Required Python matrix, minimal install, optional extras, PostgreSQL service and S3 semantics jobs, coverage floor focused on lifecycle, lint gate, fault/concurrency suite, checked-in backend-specific regression budgets |

## Build-Order Rules

1. Do not compose `UnifiedCache` over `BlobStore` until `BlobStore` alone passes the lifecycle, partial-failure, delete, and reconciliation contracts for the target backend pair.
2. Do not implement S3/PostgreSQL semantics before the canonical generation/visibility protocol is proven with deterministic doubles and local backends.
3. Add schema/key/locator version fields before any new-format data is written; build migration tooling before making new writes the default.
4. Add tests with each lifecycle capability, not as a final hardening phase. Phase 6 makes them release gates and adds real-service/performance breadth.
5. Keep `SqlCache` on its independent test and dependency boundary throughout; shared utilities may be hardened, but its row/range lifecycle must not be forced into the blob model.

## Sources

All web-derived claims below are **MEDIUM confidence** per the research confidence seam after verification against multiple current official sources. Codebase-specific findings were additionally checked against `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/CONCERNS.md`, `.planning/codebase/TESTING.md`, and the referenced implementation files.

- [Amazon S3 data consistency model](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html) — strong consistency and atomic single-key updates
- [Amazon S3 conditional writes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html) — `If-None-Match`, `If-Match`, and race outcomes
- [Amazon S3 object integrity checks](https://docs.aws.amazon.com/AmazonS3/latest/userguide/checking-object-integrity-upload.html) — multipart/encryption ETag limits and full-object checksums
- [AWS transactional outbox pattern](https://docs.aws.amazon.com/en_en/prescriptive-guidance/latest/cloud-design-patterns/transactional-outbox.html) — dual-write inconsistency, idempotency, and eventual completion
- [Python `os` documentation](https://docs.python.org/3.12/library/os.html) — atomic replace/rename and `fsync`
- [PostgreSQL transaction isolation](https://www.postgresql.org/docs/current/transaction-iso.html) — anomalies, serializable semantics, and required retries
- [PostgreSQL explicit and advisory locking](https://www.postgresql.org/docs/current/explicit-locking.html) — row locks, lock ordering, and transaction-level advisory locks
- [SQLite write-ahead logging](https://www.sqlite.org/wal.html) — one-writer concurrency, checkpoints, and same-host constraint
- [SQLite transactions](https://www.sqlite.org/lang_transaction.html) — DEFERRED/IMMEDIATE behavior and `SQLITE_BUSY`
- [SQLite busy timeout](https://sqlite.org/c3ref/busy_timeout.html) — bounded contention waiting
- [SQLite over a network](https://www.sqlite.org/useovernet.html) — network-filesystem limitations and client/server alternative
- [OpenStack Glance zero-downtime migration guidance](https://docs.openstack.org/glance/latest/contributor/database_migrations.html) — expand, migrate, contract ordering
- [Python pickle security warning](https://docs.python.org/3.12/library/pickle.html) — trusted-input requirement and HMAC recommendation
- [AWS credential provider guidance](https://docs.aws.amazon.com/sdkref/latest/guide/standardized-credentials.html) — provider chains and renewable credentials
- [pytest parametrization](https://docs.pytest.org/en/stable/how-to/parametrize.html) and [monkeypatch](https://docs.pytest.org/en/stable/how-to/monkeypatch.html) — backend matrices and controlled fault injection

---

*Pitfalls research: 2026-08-29*
