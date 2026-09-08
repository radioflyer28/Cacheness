# Phase 5: Payload Backends and Supported Topology Qualification - Research

**Researched:** 2026-09-08
**Domain:** BlobStore payload adapters, PostgreSQL lifecycle authority, AWS S3 immutable generations, topology qualification
**Confidence:** HIGH for architecture and official service semantics; MEDIUM for live qualification readiness because no live credentials/endpoints are presently discoverable

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

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

### Deferred Ideas (OUT OF SCOPE)

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
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BACK-01 | Filesystem, memory, and S3 payload adapters supply storage operations to the shared BlobStore engine and satisfy applicable immutable-generation and recovery contracts at their declared capability tier; they do not duplicate lifecycle sequencing. | Preserve `PayloadGenerationIOProvider` and the five-method guarded I/O seam; implement S3 primitives below it and run one parameterized payload contract. [VERIFIED: `.planning/REQUIREMENTS.md:23`] |
| BACK-04 | Every explicitly supported pairing passes the lifecycle contract at its tier; unsupported Cartesian combinations are documented and rejected. | Add a support-profile check separate from participant registration and publish the exact three-row qualification matrix. [VERIFIED: `.planning/REQUIREMENTS.md:26`] |
| BACK-05 | PostgreSQL and AWS S3 behavior is verified with real-service integration coverage. | Separate fakes/Moto contract evidence from live PostgreSQL plus AWS S3 qualification evidence; absence remains an open gate. [VERIFIED: `.planning/REQUIREMENTS.md:27`] |
</phase_requirements>

## Summary

Phase 5 should extend the existing engine at exactly two primitive seams: an S3 implementation of generation I/O and a PostgreSQL implementation of `LifecycleAuthority`. The lifecycle sequence already is “prepare, publish, independently verify, promote, settle debt”; copying that sequence into either adapter would recreate the coordination problem prohibited by ADR 0001. [VERIFIED: `src/cacheness/storage/lifecycle.py:186-360`; VERIFIED: `src/cacheness/storage/composition.py:67-78`]

The remote topology is a deliberately non-ACID composition. PostgreSQL transactions decide visibility and retain intent/debt; S3 contains immutable candidate generations. Exact-generation CAS protects safety, while deadlocks, serialization failures, lock timeouts, connection timeouts, S3 conditional conflicts, and ambiguous network completion are progress outcomes that must be classified, not hidden by an unbounded retry mechanism. [CITED: https://www.postgresql.org/docs/current/mvcc-serialization-failure-handling.html] [CITED: https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html]

The current S3 class is not usable as the payload participant: it performs overwrite-capable `PutObject`, buffers reads, treats ETag as integrity, materializes all listing pages, suppresses listing failures, and lacks `materialize_handler_io()`. The PostgreSQL class is explicitly only an unregistered projection placeholder. Both should be replaced at their role boundaries under the pre-production cutover, not wrapped in compatibility adapters. [VERIFIED: `src/cacheness/storage/backends/s3_backend.py:170-204,369-408,438-464`; VERIFIED: `src/cacheness/storage/backends/postgresql_backend.py:1-45`]

**Primary recommendation:** implement the S3 and PostgreSQL primitives behind the existing engine, introduce an explicit three-profile support registry independent of construction, and make live-service evidence a hard Phase 5 gate rather than turning unavailable services into passing skips.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Lifecycle sequencing and recovery | API / Backend (`BlobStore` engine) | Database / Storage | The existing engine owns ordering; adapters expose only primitives. [VERIFIED: `src/cacheness/storage/lifecycle.py:186-360`] |
| Canonical descriptor, expectation, operation intent, cleanup debt | Database / Storage (PostgreSQL authority) | — | These must change transactionally at the visibility boundary. [VERIFIED: `src/cacheness/storage/lifecycle_authority.py:284-383`] |
| Immutable payload generation, verified snapshot, delete, bounded inventory | Database / Storage (S3 payload) | API / Backend | S3 owns object mechanics; the engine decides whether they are visible or reclaimable. [VERIFIED: `src/cacheness/storage/composition.py:67-78`] |
| Topology qualification and rejection | API / Backend (composition root) | Test/release evidence | `StoreTopology.resolve()` already validates roles and capabilities before use; Phase 5 adds pairing support status. [VERIFIED: `src/cacheness/storage/composition.py:498-575`] |
| Optional JSON projection | Database / Storage (derived sink) | — | It consumes committed batches and never receives lifecycle mutation authority. [VERIFIED: `src/cacheness/storage/composition.py:45-64`] |

## Project Constraints (from AGENTS.md)

- Preserve the pre-production cutover: development-only APIs and layouts may be replaced; keep explicit versions plus offline migration/rebuild tooling, and fail explicitly on unsupported layouts. [VERIFIED: `AGENTS.md:13-14`]
- `BlobStore` owns storage lifecycle, `UnifiedCache` owns policy above it, and `SqlCache` stays separate. [VERIFIED: `AGENTS.md:15`]
- The unified lifecycle covers filesystem, memory, S3, JSON, SQLite, and PostgreSQL, with safe parsing, path containment, fail-closed integrity, attributable rollback/reconciliation, and same-key non-corruption. [VERIFIED: `AGENTS.md:16-19`]
- Maintain Python 3.11+ and verify supported versions; correctness precedes performance budgets. [VERIFIED: `AGENTS.md:20-21`]
- Read ADR 0001 before lifecycle/topology work; do not convert performance targets into unsupported atomicity or availability promises. [VERIFIED: `AGENTS.md:25-32`]
- Use domain exceptions with preserved causes, narrow exception catches, explicit optional-dependency failures, and parameterized/safely constructed queries. [VERIFIED: `AGENTS.md:193-205`]

## Standard Stack

### Core

| Library / component | Version | Purpose | Why standard here |
|---------------------|---------|---------|-------------------|
| Python | `>=3.11`; local probes found 3.11.16 and 3.13.15 | Library/runtime | Project contract; Phase 8, not Phase 5, expands the full runtime matrix. [VERIFIED: `pyproject.toml`; VERIFIED: local runtime probe] |
| Existing `AuthorityLifecycleEngine` | repository source | Sole cross-resource lifecycle coordinator | Already supplies prepare/publish/verify/promote/debt ordering. [VERIFIED: `src/cacheness/storage/lifecycle.py:186-360`] |
| psycopg | locked `3.3.2` | Direct PostgreSQL transaction and error boundary | Its transaction contexts and typed SQLSTATE exception classes map cleanly to the authority protocol without adding an ORM lifecycle layer. [VERIFIED: `uv.lock`; CITED: https://www.psycopg.org/psycopg3/docs/basic/transactions.html] |
| boto3 / botocore | locked and locally installed `1.42.36` | AWS S3 requests, credential chain, streaming bodies | Use the AWS SDK for signing, retries, credentials, and service error typing. [VERIFIED: `uv.lock`; VERIFIED: local package probe; CITED: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html] |
| pytest | locked and locally installed `8.4.1` | Contract, fault, and live qualification suites | Existing project test framework. [VERIFIED: `uv.lock`; VERIFIED: local package probe] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Moto | locked `5.1.20` | Fast S3 API contract tests | Contract/fault tests only; never BACK-05 qualification. [VERIFIED: `uv.lock`; VERIFIED: `tests/test_s3_blob_backend.py`] |
| SQLAlchemy | locked `2.0.43` | Existing SQL and projection integrations | Keep for existing subsystems; do not require it inside the new narrow authority if direct psycopg is sufficient. [VERIFIED: `uv.lock`; VERIFIED: `src/cacheness/storage/backends/postgresql_backend.py:13-42`] |
| `hashlib.sha256` | Python stdlib | Canonical payload verification | Signed manifest digest and size remain authoritative across all payload tiers. [VERIFIED: `src/cacheness/storage/lifecycle.py:219-255,312-330`] |

**Installation:** no new package is recommended. Use the existing project extras/lock; the present `.venv` lacks psycopg and SQLAlchemy even though they are locked, so the live PostgreSQL job must synchronize the appropriate existing extra before running. [VERIFIED: local import probe; VERIFIED: `uv.lock`]

## Package Legitimacy Audit

Not applicable: Phase 5 should add no dependency. It uses packages already declared and locked by the project. [VERIFIED: `pyproject.toml`; VERIFIED: `uv.lock`]

## Architecture Patterns

### System Architecture Diagram

```text
application value
      |
      v
BlobStore -> handler private stage -> signed manifest + expected generation
      |
      v
AuthorityLifecycleEngine (the only sequence owner)
      |
      +-- prepare intent/CAS --------------------> PostgreSQL transaction
      |
      +-- publish unique generation ------------> AWS S3 conditional object create
      |                                                |
      |<-- contained private snapshot + HEAD/body -----+
      |
      +-- verify SHA-256 + size
      |
      +-- record proof + promote CAS ------------> PostgreSQL transaction (visibility)
      |
      +-- delete superseded generation ----------> AWS S3
             failure => PostgreSQL cleanup debt, not rollback of committed entry

Reads: PostgreSQL committed descriptor -> contained S3 snapshot -> verify -> handler
Listing: bounded S3 evidence pages; never authoritative visibility
Projection: committed authority page -> optional JSON sink
```

### Recommended Project Structure

```text
src/cacheness/storage/
├── lifecycle.py                         # unchanged sole sequence owner
├── lifecycle_authority.py               # semantic protocol/types only
├── composition.py                       # roles + explicit support profiles/matrix
└── backends/
    ├── blob_backends.py                 # memory/filesystem participants
    ├── s3_backend.py                    # S3 participant + guarded generation I/O
    └── postgresql_lifecycle_authority.py # direct authority implementation
tests/
├── contracts/                           # reusable tier-aware authority/payload cases
├── integration/                         # live PostgreSQL/AWS suites
└── qualification/                       # evidence writer + exact rerun gate
```

### Pattern 1: Primitive Adapter, One Lifecycle Owner

`PayloadGenerationIOProvider` deliberately exposes only `materialize_handler_io()`, and `BlobStore` validates the returned object has exactly the operational surface `stage`, `publish_generation`, `open_snapshot`, `delete_or_prove_absent`, and `close`. S3 should satisfy this seam; it must not call authority transitions itself. [VERIFIED: `src/cacheness/storage/composition.py:67-78`; VERIFIED: `src/cacheness/storage/blob_store.py:690-708`]

The PostgreSQL class should implement the existing semantic authority transitions, not expose sessions/tables to the engine. The protocol includes expectations, prepared mutations, verification, promotion, cleanup debt, clear/reconciliation paging, catalog paging, projection revision, and close. [VERIFIED: `src/cacheness/storage/lifecycle_authority.py:284-383`]

### Pattern 2: Exact Supported-Topology Profiles

Participant registration and capability composition are insufficient to prove a pairing. The current composition code can construct registered roles and computes the weakest scope/capability, but contains no allow-list of qualified pair identities. [VERIFIED: `src/cacheness/storage/composition.py:347-397,498-575,724-778`]

Add a small declarative qualification registry at composition time with exactly these built-in profiles: memory/memory, SQLite/filesystem, PostgreSQL/AWS-S3. A registered or injected participant may still be structurally constructible, but public “supported topology” construction must reject any pair without a qualification record. The record should include authority identity, payload identity, capability declaration, allowed progress outcomes, required service conditions, projection availability, and evidence artifact identifier. [VERIFIED: Phase 5 decisions D-01 through D-03 and D-19]

Do not implement the check with exact concrete-class tests: that would defeat application-defined participants and conflate type identity with qualification. Use explicit role/participant identity attached to the registration or injected reference. [VERIFIED: `src/cacheness/storage/composition.py:498-575`]

### Pattern 3: PostgreSQL Exact CAS at READ COMMITTED

Use short transactions, unique constraints, and predicates on expected lineage/generation. `INSERT ... ON CONFLICT` provides an atomic insert-or-update outcome under concurrency; a conditional `UPDATE ... WHERE expected fields ... RETURNING ...` makes zero returned rows an exact lifecycle conflict. [CITED: https://www.postgresql.org/docs/current/sql-insert.html]

READ COMMITTED is the PostgreSQL default and is adequate when every state transition is expressed as one exact conditional write or an explicitly locked short transaction. This avoids treating SERIALIZABLE as magic cross-resource atomicity. If SERIALIZABLE is used for a compound transition, retry must encompass the entire transaction. [CITED: https://www.postgresql.org/docs/current/transaction-iso.html] [CITED: https://www.postgresql.org/docs/current/mvcc-serialization-failure-handling.html]

Set transaction-local `lock_timeout` and `statement_timeout`; map SQLSTATE `40001`, `40P01`, `55P03`, `57014`, and connection-level operational failures into a typed retryable lifecycle outcome with the original exception as cause and the operation stage in bounded context. Do not retry indefinitely inside the authority. [CITED: https://www.postgresql.org/docs/current/runtime-config-client.html] [CITED: https://www.psycopg.org/psycopg3/docs/api/errors.html]

Create connections after fork and avoid sharing cursors across threads. Psycopg documents that connections serialize query execution, cursors are not thread-safe, and connections are not process-safe. Prefer a connection factory/pool with one transaction-scoped lease rather than a single global connection. [CITED: https://www.psycopg.org/psycopg3/docs/advanced/async.html]

### Pattern 4: Conditional, Immutable S3 Publication

Use a unique generation key under a contained store prefix and `IfNoneMatch="*"` on both single-request `PutObject` and `CompleteMultipartUpload`. AWS documents `412 Precondition Failed` for an existing key and possible `409 Conflict` around concurrent deletes; a multipart `409` requires a newly initiated multipart upload. [CITED: https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html]

The locally installed botocore S3 service model includes `IfNoneMatch` on both operations, but `S3Transfer.ALLOWED_UPLOAD_ARGS` does not. Therefore the current `upload_fileobj()` abstraction cannot carry the required conditional complete semantics; implement bounded multipart using the low-level boto3 client operations, always attempting `AbortMultipartUpload` for an observed pre-completion failure. [VERIFIED: local boto3 1.42.36 service-model and transfer introspection]

On an ambiguous network completion, inspect the exact immutable key. If absent, surface retryable backend failure; if present, stream it to a private snapshot and compare signed manifest SHA-256 and size; exact match classifies publication as successful, mismatch is a conflict/integrity failure. This closes the engine's current `candidate_persisted` uncertainty without treating list results as visibility. [VERIFIED: `src/cacheness/storage/lifecycle.py:305-343`; VERIFIED: ADR 0001]

Incomplete multipart uploads are not visible objects. The adapter should abort a known upload ID on observed failure; bounded recovery may page `ListMultipartUploads` under the exact managed prefix and abort only attributable stale sessions. AWS also recommends an `AbortIncompleteMultipartUpload` lifecycle rule to limit cost after process death, but that deployment rule is resource hygiene, not authority or correctness. [CITED: https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html]

### Pattern 5: Verified Private Snapshot

`open_snapshot()` must HEAD/preflight the expected content length, stream the body in bounded chunks into a mode-0600 private temporary file, enforce the configured byte/work limit while copying, close the network body, and return only the contained file. The engine independently hashes that file before promotion or deserialization. [VERIFIED: `src/cacheness/storage/lifecycle.py:312-330`; VERIFIED: existing guarded-I/O contract]

ETag must be diagnostic only: AWS states multipart ETags and encrypted-object ETags are not a universal full-object MD5. Transport checksums may detect transfer corruption but do not replace the signed canonical SHA-256 and byte count. [CITED: https://docs.aws.amazon.com/AmazonS3/latest/userguide/checking-object-integrity-upload.html]

### Pattern 6: Bounded Evidence Listing

Expose one page operation with prefix, continuation token, maximum objects, maximum represented bytes, and work cap. S3 `ListObjectsV2` returns at most 1,000 objects per request and provides `NextContinuationToken`; never retain every page in one Python list. [CITED: https://docs.aws.amazon.com/AmazonS3/latest/API/API_ListObjectsV2.html]

Payload inventory is evidence only. Normal reads and promotion follow PostgreSQL; reconciliation may consume bounded pages to find attributable orphan candidates or incomplete uploads, but absence from a list never revokes a committed descriptor. [VERIFIED: Phase 5 decisions D-09 and D-12]

### Pattern 7: Explicit Schema Version, No Runtime Compatibility Layer

Create the PostgreSQL schema only during explicit `initialize()`. Store a backend schema version and store identity in PostgreSQL; subsequent opens validate exact compatibility and raise the existing migration-required family on older, newer, or malformed state. Do not silently run DDL or translate the Phase 4 placeholder at ordinary construction. [VERIFIED: Phase 5 decisions D-14 and D-17; VERIFIED: `.planning/REQUIREMENTS.md:90-96`]

Keep payload format and canonical manifest version checks. Phase 5 need not execute migrations or generalize every existing version field; Phase 7 owns stopped-worker schema/format migration and rebuild. [VERIFIED: `.planning/REQUIREMENTS.md:54-59`; VERIFIED: Phase 5 deferred ideas]

### Anti-Patterns to Avoid

- **Cartesian support inference:** capabilities show what participants can do, not whether a pair was qualified. Reject unqualified combinations. [VERIFIED: D-02/D-03]
- **S3 as visibility or catalog:** object existence/listing never authorizes a read, rollback, repair, or delete. [VERIFIED: D-09]
- **High-level transfer with lost precondition:** `upload_fileobj()` cannot express the required `IfNoneMatch` complete in the installed SDK transfer layer. [VERIFIED: local SDK introspection]
- **ETag integrity:** ETag is not the canonical digest. [CITED: https://docs.aws.amazon.com/AmazonS3/latest/userguide/checking-object-integrity-upload.html]
- **Catch-and-return-miss:** network, delete, list, parser, and integrity failures must remain typed; the current S3 `list_keys()` returns an indistinguishable empty list on `ClientError`. [VERIFIED: `src/cacheness/storage/backends/s3_backend.py:438-464`]
- **Global advisory locks or queues:** they create another coordination surface without making S3 transactional. [VERIFIED: D-15/D-18]
- **Retry-until-success:** contention is allowed to return a typed retryable outcome. [VERIFIED: D-16/D-20]
- **Local signing key in a multi-host topology:** the current non-memory default is a local file, so remote workers would not share manifest verification state. Require an explicitly injected shared signing-key provider for PostgreSQL/S3 qualification. [VERIFIED: `src/cacheness/storage/blob_store.py:220-227`]
- **Emulator-qualified production claim:** Moto and compatible endpoints do not satisfy BACK-05. [VERIFIED: D-05/D-08]

## Exact Capability and Support Matrix

The planner should make one source of truth generate runtime reporting, docs, and parameterized expectations. The quoted capability vocabulary is: `"durable"`, `"process_scope"`, `"host_scope"`, `"transaction_scope"`, `"exact_cas"`, `"immutable_generations"`, `"streaming"`, `"listing"`, `"portable_query"`, `"canonical_scan"`, `"index_acceleration"`, `"projection_refresh"`, `"projection_rebuild"`, `"online_rebuild"`, `"offline_rebuild"`. [VERIFIED: `src/cacheness/storage/composition.py:91-109`]

| Qualified profile | Authority / payload declarations | Atomicity and visibility boundary | Allowed progress outcomes | Required configuration | Projection |
|-------------------|----------------------------------|-----------------------------------|---------------------------|------------------------|------------|
| memory / memory | non-durable; process/process; authority transaction; exact CAS; immutable generation; no streaming/listing; portable query + canonical scan | One process; in-memory authority promotion | `success`, `conflict` | One process | optional JSON, derived only |
| SQLite / filesystem | durable; host/host; authority transaction; exact CAS; immutable generation; streaming/listing; portable query + canonical scan | Initialized single host; SQLite promotion is visibility, filesystem effect is outside its transaction | `success`, `conflict`, `retryable_timeout`, recoverable cleanup | Shared contained local root, explicit initialize | optional JSON, derived only |
| PostgreSQL / AWS S3 | durable; multi-host/multi-host; authority transaction; exact CAS; immutable generation; streaming/listing; portable query + canonical scan | PostgreSQL promotion is visibility; no PostgreSQL/S3 transaction | `success`, `conflict`, typed retryable progress failure, committed-with-recoverable-cleanup | real PostgreSQL; real AWS S3; external shared signer; contained bucket prefix; least-privilege credentials; explicit initialize | optional JSON only where its host-local scope is declared; never authority |

The first two rows extend existing declarations. The current source quotes only `{"success", "conflict"}` for authority kind `"memory"` and `{"success", "conflict", "retryable_timeout"}` for `"sqlite-local"`; PostgreSQL needs a new declared branch rather than falling through `CompositionValidationError`. [VERIFIED: `src/cacheness/storage/composition.py:800-808`]

The SQLite registry currently declares `"portable_query": False` while the delivered authority supplies catalog paging. Resolve that inconsistency before freezing the Phase 5 matrix; BACK-07 requires direct query capability, so the registry should report the actually implemented portable scan behavior. [VERIFIED: `src/cacheness/storage/composition.py:377-390`; VERIFIED: `.planning/REQUIREMENTS.md:29`]

Every cross-pair not named above is unsupported, including PostgreSQL/filesystem, SQLite/S3, memory/persistent mixtures, S3/process-local authority, and every S3-compatible service without its own record. [VERIFIED: D-02/D-08]

## PostgreSQL Authority Design

Use a dedicated schema with tables corresponding to existing semantic state, not the historical generic metadata backend: store identity/version, entry lineage, committed entries, prepared mutations/verification, authority revision, cleanup debt, clear runs/targets, reconciliation runs/actions, and projection revisions. This mirrors the already proven SQLite authority while letting SQL constraints and transaction isolation implement the same protocol. [VERIFIED: `src/cacheness/storage/lifecycle_authority.py:284-383`; VERIFIED: existing SQLite authority schema inspected this session]

Required invariants:

- one committed row per logical key and one monotonically checked expectation/lineage; exact expected values are present in every mutation predicate; [VERIFIED: `src/cacheness/storage/lifecycle_authority.py:292-308`]
- one durable operation identity for prepare/verify/promote/abort classification; rerunning an operation is idempotent or returns a typed conflict; [VERIFIED: `src/cacheness/storage/lifecycle_authority.py:298-308`]
- canonical manifest bytes and authoritative catalog values commit together; derived projections never participate; [VERIFIED: D-14]
- old-generation and aborted-candidate cleanup debt is inserted in the same PostgreSQL transaction that establishes the corresponding authoritative state; [VERIFIED: D-13/D-18]
- catalog/clear/reconciliation operations page by stable cursor and configured work caps; no `list_entries()` materialization should be used on an unbounded remote catalog; [VERIFIED: `src/cacheness/storage/lifecycle_authority.py:310-375`]
- initialization takes a bounded database lock or uniqueness race only inside explicit initialization, validates the schema version, and never migrates under workers. [VERIFIED: D-17]

Use a dynamic test schema only through `psycopg.sql.Identifier`; bind all values. Never interpolate a qualification run ID into SQL text. [CITED: https://www.psycopg.org/psycopg3/docs/api/sql.html]

## S3 Payload Design

The S3 participant should be configured with an existing boto3 client/session or standard SDK credential resolution, an exact bucket, region, managed prefix, optional expected bucket owner, bounded transfer settings, and private local staging directory. Do not log credential material. Boto3 documents its credential provider chain and recommends against hard-coded credentials. [CITED: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html]

Locator parsing must accept only normalized relative generation locators from the engine and map them below the configured prefix. A locator naming a different bucket, escaping the prefix, containing traversal-like components, or exceeding configured text bounds must fail closed; the current warning-and-accept behavior is not suitable for a security boundary. [VERIFIED: current S3 path parser inspected this session; VERIFIED: AGENTS.md security constraint]

Deletion must distinguish “confirmed absent,” “deleted,” “retryable/ambiguous backend failure,” and permission/configuration failure. `DeleteObject` success alone is not proof of absence when access/versioning semantics differ; for V1’s unversioned/general-purpose contract, follow with exact-key HEAD only where needed by `delete_or_prove_absent`, bounded by configured time/work. Never turn `ClientError` into `False` or an empty list. [VERIFIED: existing guarded-I/O semantic method name at `src/cacheness/storage/blob_store.py:694-700`; ASSUMED]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| AWS authentication/request signing | Custom HMAC/V4 client | boto3/botocore credential chain and S3 client | Handles rotating credentials, roles, signing, endpoints, and modeled errors. [CITED: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html] |
| PostgreSQL SQL composition | String-interpolated identifiers/values | psycopg parameters plus `psycopg.sql.Identifier` | Values and identifiers have different safe-composition rules. [CITED: https://www.psycopg.org/psycopg3/docs/api/sql.html] |
| Cross-resource transaction | Queue, global lock, advisory-lock protocol, filesystem lease | Existing intent/publish/verify/promote/debt engine | PostgreSQL cannot atomically include S3; adding a lock does not change that. [VERIFIED: D-15/D-18] |
| Payload content format | Cacheness envelope/header | Existing handler-native staged file | Payload format belongs to NumPy/Blosc2/Parquet/pickle/etc.; lifecycle signs a separate descriptor. [VERIFIED: Phase 5 code context] |
| S3 integrity hash | ETag-as-MD5 | Canonical manifest SHA-256 + byte size; SDK checksum only supplemental | ETag semantics vary. [CITED: https://docs.aws.amazon.com/AmazonS3/latest/userguide/checking-object-integrity-upload.html] |
| Universal retry scheduler | Endless internal contention loop | Typed conflict/retryable outcome, optionally one explicitly bounded whole-transaction policy | PostgreSQL requires whole-transaction retry; caller success is not guaranteed. [CITED: https://www.postgresql.org/docs/current/mvcc-serialization-failure-handling.html] |
| Backend migration in open/constructor | Compatibility adapters or implicit DDL | Exact version detection and explicit Phase 7 migration/rebuild | Preserves future migration tooling without retaining development-only runtime layouts. [VERIFIED: `.planning/REQUIREMENTS.md:90-96`] |

## Runtime State Inventory

| Category | Items Found | Action Required |
|----------|-------------|-----------------|
| Stored data | No live PostgreSQL/S3 store was discoverable. Current repository formats remain explicitly versioned; existing development layouts are not a compatibility target. [VERIFIED: environment-name and repository scan; VERIFIED: `.planning/REQUIREMENTS.md:90-96`] | Add PostgreSQL authority schema version/store identity and reject mismatches. Do not migrate data in Phase 5. |
| Live service config | No repository CI/compose live-service harness and no current environment variable **names** matching AWS, S3, PostgreSQL, or Cacheness were found. Values were not inspected. [VERIFIED: name-only environment scan and repository file scan] | Create externally configured live fixtures and emit `UNAVAILABLE`/`NOT_QUALIFIED` evidence until endpoints exist. |
| OS-registered state | None found or required; this library has no installed service registration in scope. [VERIFIED: repository scan] | None. |
| Secrets/env vars | Existing tests set only mock AWS names: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SECURITY_TOKEN`, `AWS_SESSION_TOKEN`, `AWS_DEFAULT_REGION`. [VERIFIED: `tests/test_s3_blob_backend.py`] | Live tests should use standard SDK credentials plus non-secret bucket/prefix settings; never persist or print values. Add a separately injected shared manifest signing provider for multi-host tests. |
| Build artifacts / installed packages | Docker 29.7.2 is available; `psql` and `pg_isready` are absent. The local `.venv` has boto3/botocore but not psycopg/SQLAlchemy despite the latter being locked. [VERIFIED: local tool/import probes] | Sync existing PostgreSQL extras for live tests. A real PostgreSQL Docker container may prove database behavior, but it does not replace a genuinely multi-host deployment test or real AWS S3. |

## Common Pitfalls

### Pitfall 1: Closing an S3 Ambiguity as “Not Persisted”

**What goes wrong:** the request succeeds server-side but the response is lost; the engine records `candidate_persisted=False`, aborts authority intent, and leaves an untracked immutable object. [VERIFIED: `src/cacheness/storage/lifecycle.py:305-343`]

**Avoidance:** the S3 publish primitive must classify ambiguity by exact-key contained read and canonical digest/size. Tests must inject failures before service acceptance, after acceptance/before response, during multipart, and after candidate verification. [VERIFIED: ADR 0001]

### Pitfall 2: Treating 412 and 409 as the Same Retry

**What goes wrong:** a `412` means the immutable name already exists, while a multipart `409` requires re-initiating the multipart upload. Blindly retrying completion may loop or overwrite intent classification. [CITED: https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html]

**Avoidance:** map existing-key `412` to exact-key verification/conflict and multipart `409` to abort/new upload only within an explicitly bounded attempt; preserve AWS error and stage.

### Pitfall 3: Sharing Connections Across Forks

**What goes wrong:** processes operate on the same PostgreSQL connection/socket; psycopg says connections are not process-safe. [CITED: https://www.psycopg.org/psycopg3/docs/advanced/async.html]

**Avoidance:** construct the connection factory/pool after fork or lazily per process; use transaction-scoped leases and close only store-owned resources.

### Pitfall 4: Capability Drift

**What goes wrong:** registry declarations, instance declarations, docs, and tests disagree. The current SQLite registry says portable query is false while BACK-07 is delivered, and current allowed progress outcomes have no PostgreSQL branch. [VERIFIED: `src/cacheness/storage/composition.py:377-390,800-808`; VERIFIED: `.planning/REQUIREMENTS.md:29`]

**Avoidance:** generate tests/docs from one exact profile table and assert registry and runtime reports match it.

### Pitfall 5: “Real Service” Test That Passes by Skipping

**What goes wrong:** the normal suite is green without credentials, and the release claim silently becomes stronger than the evidence. [VERIFIED: D-05/D-06]

**Avoidance:** contract tests may skip optional live fixtures in ordinary developer runs, but the qualification command must write explicit unavailable evidence and exit non-success until both real-service suites pass.

### Pitfall 6: Unbounded Remote Catalog/Inventory Work

**What goes wrong:** current S3 listing accumulates all pages and suppresses service errors, producing memory spikes and false empty inventories. [VERIFIED: `src/cacheness/storage/backends/s3_backend.py:438-464`]

**Avoidance:** every remote page accepts and enforces object, byte, page, and work caps; test over-limit pages and cursor resumption.

### Pitfall 7: Local Signing State in a Multi-Host Store

**What goes wrong:** each host generates/reads a different local manifest HMAC key, so valid committed manifests fail on another host. [VERIFIED: `src/cacheness/storage/blob_store.py:220-227`]

**Avoidance:** the PostgreSQL/S3 supported profile requires an external shared `ManifestSigningKeyProvider`; live tests exercise write on client A and read on separately constructed client B.

## Code Examples

### PostgreSQL transaction boundary and typed progress failures

```python
# Sources:
# https://www.psycopg.org/psycopg3/docs/basic/transactions.html
# https://www.psycopg.org/psycopg3/docs/api/errors.html
with connection_factory() as conn:
    with conn.transaction():
        conn.execute(
            "SELECT set_config('lock_timeout', %s, true)",
            (lock_timeout,),
        )
        promoted = conn.execute(
            """
            UPDATE cacheness_entries
               SET generation = %s, manifest = %s
             WHERE key = %s AND lineage = %s AND generation IS NOT DISTINCT FROM %s
         RETURNING lineage, generation, manifest
            """,
            (candidate_generation, manifest_bytes, key, expected_lineage,
             expected_generation),
        ).fetchone()
        if promoted is None:
            raise CacheBlobLifecycleConflictError("Expected generation changed")
```

Use a fixed table name inside a controlled schema or construct dynamic schema/table identifiers with `psycopg.sql.Identifier`; never bind an identifier as a value. The code is a planning skeleton, not a prescribed final table layout. [CITED: https://www.psycopg.org/psycopg3/docs/api/sql.html]

### S3 immutable single-request creation

```python
# Source: https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html
client.put_object(
    Bucket=bucket,
    Key=managed_generation_key,
    Body=private_snapshot,
    ContentLength=byte_size,
    IfNoneMatch="*",
)
```

For multipart, use `create_multipart_upload`, bounded `upload_part` calls, then `complete_multipart_upload(..., IfNoneMatch="*")`; the high-level transfer allow-list in boto3 1.42.36 does not include `IfNoneMatch`. [VERIFIED: local boto3 service-model/transfer introspection]

### Bounded inventory page

```python
# Source: https://docs.aws.amazon.com/AmazonS3/latest/API/API_ListObjectsV2.html
response = client.list_objects_v2(
    Bucket=bucket,
    Prefix=managed_prefix,
    ContinuationToken=cursor,
    MaxKeys=page_limit,
)
items = tuple(response.get("Contents", ()))
next_cursor = response.get("NextContinuationToken")
```

The production primitive must additionally enforce represented-byte and total-work budgets and must surface service/parser errors rather than return an empty page. [VERIFIED: D-12]

## State of the Art

| Old approach in this repository | Phase 5 approach | Impact |
|---------------------------------|------------------|--------|
| S3 direct byte CRUD, overwrite-capable put, buffered reads | Guarded immutable generation I/O under BlobStore | S3 becomes a primitive participant, not a parallel store. [VERIFIED: `src/cacheness/storage/backends/s3_backend.py:170-210`] |
| ETag verification | Signed SHA-256 and byte-size verification; ETag diagnostic only | Correct for multipart and encryption modes. [CITED: https://docs.aws.amazon.com/AmazonS3/latest/userguide/checking-object-integrity-upload.html] |
| Unbounded paginator accumulation | One continuation-token page under multiple bounds | Supports deterministic recovery without bucket-sized memory. [CITED: https://docs.aws.amazon.com/AmazonS3/latest/API/API_ListObjectsV2.html] |
| PostgreSQL projection placeholder | PostgreSQL lifecycle authority plus separately role-declared projection if desired | Remote visibility and debt have one transactional authority. [VERIFIED: `src/cacheness/storage/backends/postgresql_backend.py:1-45`] |
| Constructible pair implies possible use | Explicit qualified profile distinct from registration | Unsupported Cartesian combinations fail before I/O. [VERIFIED: D-02/D-03] |
| Runtime compatibility adapters | Clean pre-production cutover with exact version rejection | Reduces phase complexity while retaining future migration/rebuild seams. [VERIFIED: `.planning/REQUIREMENTS.md:90-96`] |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | A post-delete HEAD is necessary for every successful `DeleteObject` in the V1 `delete_or_prove_absent` implementation. | S3 Payload Design | Adds a request and latency; decide from exact absence contract and AWS bucket/versioning policy during planning. |
| A2 | A Docker-hosted real PostgreSQL instance is acceptable for the database half of BACK-05, while still not proving multi-host deployment behavior. | Runtime State Inventory | If qualification requires an externally managed PostgreSQL service, the database gate also remains unavailable. |

## Open Questions

1. **Where will the shared manifest signing key come from for live multi-host qualification?**
   - What we know: the current persistent default is a local file and cannot qualify cross-host reads. [VERIFIED: `src/cacheness/storage/blob_store.py:220-227`]
   - Recommendation: require an injected provider in the PostgreSQL/S3 profile; the live harness supplies it externally and never stores the key in evidence.

2. **Are real AWS and PostgreSQL endpoints available to the executor?**
   - What we know: no relevant environment variable names or repository harness were found; no values were inspected. [VERIFIED: name-only environment scan]
   - Recommendation: plan the implementation and contract suites, but make final support registration/acceptance conditional on a recorded real-service pass. Do not mark Phase 5 complete if evidence is `UNAVAILABLE`/`NOT_QUALIFIED`.

3. **How should incomplete multipart sessions be reclaimed after process death?**
   - What we know: the engine intent contains the candidate object locator but not an upload ID; AWS exposes bounded multipart listing and recommends an abort-incomplete lifecycle rule. [VERIFIED: lifecycle authority/source inspection; CITED: https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html]
   - Recommendation: the adapter aborts known IDs immediately; reconciliation pages multipart uploads under the exact managed prefix and uses age/run attribution. Require the bucket lifecycle rule as cost hygiene, not correctness.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| Python | local tests | yes | 3.11.16 in `.venv`; 3.13.15 executable | Phase 8 expands matrix. [VERIFIED: local probe] |
| pytest | all tests | yes | 8.4.1 | none needed. [VERIFIED: local probe] |
| boto3/botocore | S3 adapter/contracts | yes | 1.42.36 | Moto for contract only. [VERIFIED: local probe] |
| psycopg | PostgreSQL authority/live tests | not in active `.venv` | locked 3.3.2 | synchronize existing extra; no substitute qualifies PostgreSQL. [VERIFIED: local probe; VERIFIED: `uv.lock`] |
| SQLAlchemy | existing projection/SQL integrations | not in active `.venv` | locked 2.0.43 | not required for direct authority design. [VERIFIED: local probe; VERIFIED: `uv.lock`] |
| PostgreSQL client/service | live qualification | no CLI/service configuration discovered | unknown | Docker 29.7.2 can run real PostgreSQL if image/network available; otherwise `UNAVAILABLE`. [VERIFIED: local probe] |
| AWS S3 credentials/bucket | live qualification | no relevant environment variable names discovered | unknown | none; Moto/compatible services remain contract-only. [VERIFIED: name-only environment scan] |

**Missing dependencies with no qualification fallback:** externally configured real AWS S3; a real PostgreSQL service and installed psycopg runtime. [VERIFIED: local probe]

**Available contract fallbacks:** Moto/fakes for S3 API and deterministic fault tests; Docker may host actual PostgreSQL locally, subject to image availability. Neither changes the locked AWS live-service gate. [VERIFIED: D-05/D-08]

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.4.1 [VERIFIED: local probe] |
| Config file | `pyproject.toml` [VERIFIED: repository scan] |
| Quick run command | `.venv/bin/pytest -q tests/contracts/test_payload_generation_io.py tests/test_supported_topologies.py -x` [ASSUMED] |
| Focused authority command | `.venv/bin/pytest -q tests/contracts/test_lifecycle_authority.py -x` [ASSUMED] |
| Full local suite command | `.venv/bin/pytest -q -o log_cli=false` [VERIFIED: AGENTS.md stack instructions] |
| Qualification command | a checked-in Phase 5 runner that executes both real-service suites and writes sanitized evidence; exact command name is planner discretion [ASSUMED] |

### Test Taxonomy

| Class | Environment | May satisfy |
|-------|-------------|-------------|
| Pure participant contract | memory/filesystem, fake authority/client | method semantics, containment, bounds, error mapping; BACK-01 portions |
| Emulator/mock contract | Moto S3 | request construction, multipart branches, pagination, common modeled errors; never BACK-05 [VERIFIED: D-05] |
| Deterministic fault injection | fake boundary around existing engine | interruption classification at every prepare/publish/verify/promote/delete boundary; BACK-01/BACK-04 safety/recovery |
| Real PostgreSQL integration | actual server | transactions, constraints, SQLSTATE/timeouts, cross-connection contention, explicit schema initialization; BACK-05 database half |
| Real AWS S3 integration | AWS endpoint | conditional create, multipart completion, HEAD/GET/delete/list pagination, IAM/credential behavior; BACK-05 AWS half |
| Combined remote qualification | separate BlobStore clients over same PostgreSQL/S3 namespace | supported multi-host contract, shared signer, exact CAS, cleanup debt; BACK-04/BACK-05 |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BACK-01 | all three payloads satisfy applicable immutable publish, contained verified read, exact delete/absence, bounded listing, and recovery without lifecycle duplication | parameterized contract + deterministic faults | `.venv/bin/pytest -q tests/contracts/test_payload_generation_io.py tests/test_payload_faults.py -x` | ❌ Wave 0 [ASSUMED] |
| BACK-04 | exactly three supported profiles run common lifecycle scenarios; every cross-pair is rejected before I/O; reports match the matrix | unit + integration | `.venv/bin/pytest -q tests/test_supported_topologies.py tests/contracts/test_topology_lifecycle.py -x` | ❌ Wave 0 [ASSUMED] |
| BACK-05 | real PostgreSQL and AWS S3 single/multipart, contention, recovery, pagination, cleanup, and cross-client read pass with sanitized evidence | live integration/qualification | checked-in qualification runner with explicit endpoint configuration | ❌ Wave 0; credentials unavailable [ASSUMED] |

### Contract Scenarios Required

- **Integrity:** staged bytes equal committed bytes; published read is contained; size/digest verified before handler; ETag never authorizes success; prefix/bucket mismatch fails closed. [VERIFIED: D-09 through D-11]
- **Recovery:** failure before/after S3 acceptance, before/after PostgreSQL prepare/verification/promotion, delete failure, ambiguous commit response, process recreation, and reconciliation replay leave either the old committed generation or one fully verified new generation. [VERIFIED: ADR 0001]
- **Progress:** exact CAS conflicts are distinct from retryable timeout/deadlock/serialization/connection outcomes; no test requires all contenders to succeed. [VERIFIED: D-16/D-20]
- **Bounds:** multipart chunk/part count, download byte cap, object page, represented bytes, work cap, catalog page, and cleanup batch are asserted explicitly. [VERIFIED: D-11/D-12]
- **Performance:** measurements emit distributions separately and cannot fail a safety test based on a universal deadline. [VERIFIED: D-21]

### Live Evidence Contract

Each run records git revision, UTC timestamp, Python/SDK versions, PostgreSQL server version, AWS region and service identity, bucket configuration relevant to the contract, unique redacted run namespace, test list, result per guarantee class, cleanup result, and overall `QUALIFIED`, `UNAVAILABLE`, or `NOT_QUALIFIED`. The locked unavailable strings are `UNAVAILABLE` and `NOT_QUALIFIED`; the proposed success spelling is an implementation choice. [VERIFIED: D-06; ASSUMED]

Evidence must contain no connection URL, host credentials, access-key fragments, session tokens, signing keys, object contents, or general bucket inventory. Cleanup targets only the exact run schema and exact S3 prefix and is idempotent. [VERIFIED: D-07]

### Sampling Rate

- **Per task commit:** quick participant/topology contract under 30 seconds where practical. [ASSUMED]
- **Per wave merge:** full focused Phase 5 contract/fault suite plus existing local regression suite. [ASSUMED]
- **Phase gate:** full local suite green, both real-service suites pass, combined remote qualification passes, and sanitized evidence reports no cleanup leak. An unavailable live dependency leaves the gate open. [VERIFIED: D-05/D-06]
- **Phase 8:** reuse these suites across supported Python/platform/install/service cells; do not redefine backend guarantees. [VERIFIED: D-22]

### Wave 0 Gaps

- [ ] Reusable payload generation-I/O contract for memory, filesystem, and S3.
- [ ] Reusable lifecycle-authority contract for memory, SQLite, and PostgreSQL, tier-aware rather than equality-assuming.
- [ ] Exact supported-topology matrix and negative cross-pair tests.
- [ ] S3 deterministic fake for response loss after server acceptance, multipart failures, pagination, delete ambiguity, and malformed metadata.
- [ ] PostgreSQL real-service fixture with unique schema and bounded cleanup.
- [ ] AWS S3 real-service fixture with unique managed prefix and bounded cleanup.
- [ ] Combined remote multi-client suite and external shared signer fixture.
- [ ] Sanitized evidence writer whose unavailable state is non-passing.
- [ ] Pytest live markers documented in `pyproject.toml`; ordinary local suite may deselect them, qualification runner may not.

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no (library has no user authentication) | Deployment supplies AWS/PostgreSQL identities. [VERIFIED: project architecture] |
| V3 Session Management | no | No application sessions. [VERIFIED: project architecture] |
| V4 Access Control | yes | Least-privilege PostgreSQL role/schema and S3 bucket/prefix IAM; expected-bucket-owner where configured. [CITED: https://docs.aws.amazon.com/AmazonS3/latest/userguide/security_iam_service-with-iam.html] |
| V5 Validation | yes | Bound/validate locators, identifiers, cursors, manifests, lengths, page/work limits, and service responses before use. [VERIFIED: AGENTS.md security constraint] |
| V6 Cryptography | yes | Existing HMAC/SHA-256; TLS and request signing from drivers/SDK; no custom crypto. [VERIFIED: `src/cacheness/storage/lifecycle.py:219-255`] |
| V7 Error/Logging | yes | Typed errors with causes/stages; redact credentials and signed material. [VERIFIED: AGENTS.md error/logging constraints] |
| V8 Data Protection | yes | Mode-0600 private snapshots, explicit cleanup, no payloads or secrets in evidence. [VERIFIED: D-07/D-11] |
| V12 File/Resource Upload | yes | Bounded multipart parts, bytes, snapshots, pages, and work; abort incomplete uploads. [VERIFIED: D-11/D-12] |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Locator escapes bucket/prefix or swaps bucket | Tampering | Normalize relative locator; exact bucket/prefix containment; reject rather than warn. [VERIFIED: AGENTS.md security boundary] |
| SQL identifier/value injection | Tampering | Fixed schema objects; psycopg parameters and `sql.Identifier`. [CITED: https://www.psycopg.org/psycopg3/docs/api/sql.html] |
| Existing S3 generation overwritten | Tampering | Unique generation key plus `IfNoneMatch="*"`; optional bucket-policy enforcement. [CITED: https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes-enforce.html] |
| Unverified network bytes reach pickle/dill | Elevation of privilege | Contained private snapshot, signed manifest, SHA-256/size before handler. Payload remains trusted-application-only. [VERIFIED: D-10/D-11] |
| Credentials leak in logs/evidence | Information disclosure | Standard providers, no value inspection/logging, evidence allow-list. [CITED: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html] |
| Unbounded list/download/multipart | Denial of service | Page/object/byte/work/chunk caps; close streaming bodies; fail on excess. [VERIFIED: D-11/D-12] |
| Stale client deletes current generation | Tampering | PostgreSQL expected-generation CAS plus engine re-read before cleanup; payload listing never authorizes delete. [VERIFIED: `src/cacheness/storage/lifecycle.py:150-173`] |
| Destructive live cleanup crosses namespace | Tampering/DoS | Unique run schema/prefix; validate ownership marker; exact-target idempotent cleanup only. [VERIFIED: D-07] |

## Sources

### Primary (HIGH confidence)

- Repository source opened this session: `composition.py`, `blob_store.py`, `lifecycle.py`, `lifecycle_authority.py`, `s3_backend.py`, `postgresql_backend.py`, existing authority/payload implementations and tests.
- [AWS S3 conditional writes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html) — `If-None-Match`, 409/412, multipart retry/cleanup behavior.
- [AWS S3 conditional-write enforcement](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes-enforce.html) — bucket-policy enforcement.
- [AWS object-integrity checking](https://docs.aws.amazon.com/AmazonS3/latest/userguide/checking-object-integrity-upload.html) — checksum and ETag limitations.
- [AWS ListObjectsV2 API](https://docs.aws.amazon.com/AmazonS3/latest/API/API_ListObjectsV2.html) — continuation-token and per-request bounds.
- [Boto3 credentials guide](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html) — provider chain and no hard-coded credentials.
- [PostgreSQL INSERT](https://www.postgresql.org/docs/current/sql-insert.html) — atomic `ON CONFLICT` semantics.
- [PostgreSQL transaction isolation](https://www.postgresql.org/docs/current/transaction-iso.html) and [serialization retry guidance](https://www.postgresql.org/docs/current/mvcc-serialization-failure-handling.html).
- [PostgreSQL client timeouts](https://www.postgresql.org/docs/current/runtime-config-client.html).
- [Psycopg transactions](https://www.psycopg.org/psycopg3/docs/basic/transactions.html), [concurrency](https://www.psycopg.org/psycopg3/docs/advanced/async.html), [errors](https://www.psycopg.org/psycopg3/docs/api/errors.html), and [SQL composition](https://www.psycopg.org/psycopg3/docs/api/sql.html).

### Secondary (MEDIUM confidence)

- None used for architectural claims.

### Tertiary (LOW confidence)

- The two explicitly logged assumptions above require planner/executor confirmation.

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — existing lock/runtime plus official boto3/psycopg documentation.
- Architecture: HIGH — derived from opened repository protocols, lifecycle engine, ADR, and locked phase decisions.
- S3 semantics: HIGH — AWS official service/API documentation plus installed SDK-model introspection.
- PostgreSQL semantics: HIGH — PostgreSQL and psycopg official documentation.
- Live readiness: MEDIUM — name-only environment and local-tool probes found no service configuration, but credentials may be provided later by the executor.
- Qualification commands/file names: LOW — deliberately left to planner discretion and tagged assumed.

**Research date:** 2026-09-08
**Valid until:** 2026-10-08 for architecture; recheck AWS SDK/service documentation and live environment at execution time.
