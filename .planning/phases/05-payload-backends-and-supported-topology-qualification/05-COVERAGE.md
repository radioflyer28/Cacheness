# Phase 5 API Coverage: boto3/Amazon S3 and psycopg/PostgreSQL

**Status:** Planning baseline; implementation and live evidence pending
**Scope:** Only capabilities needed by the declared S3 payload-participant and
PostgreSQL lifecycle-authority roles. `INTEGRATE` is the default. Every excluded
service/driver feature is named and reasoned rather than silently omitted.

## Amazon S3 / boto3 capability matrix

| capability | decision | reason |
|---|---|---|
| Standard boto3 credential-provider chain | INTEGRATE | External credentials only; no secret values in constructors, logs, or evidence. |
| Injected boto3 S3 client | INTEGRATE | Deterministic contract tests and caller-owned service configuration without exposing credentials. |
| Region plus `ExpectedBucketOwner` | INTEGRATE | Bind requests/evidence to the intended AWS account and region when configured. |
| `PutObject` + `IfNoneMatch="*"` | INTEGRATE | Conditional immutable creation for single-request generations. |
| `CreateMultipartUpload` | INTEGRATE | Begin bounded uploads for handler snapshots above the configured threshold. |
| `UploadPart` | INTEGRATE | Stream bounded parts without whole-object memory buffering. |
| `CompleteMultipartUpload` + `IfNoneMatch="*"` | INTEGRATE | Conditional immutable completion; high-level transfer helpers cannot carry this precondition. |
| `AbortMultipartUpload` | INTEGRATE | Best-effort exact-upload cleanup after known pre-completion failure. |
| `ListMultipartUploads` | INTEGRATE | Bounded, exact-prefix recovery evidence for attributable stale uploads. |
| `HeadObject` | INTEGRATE | Exact-key absence/size/preflight and ambiguous-response classification; never visibility authority. |
| `GetObject` / `StreamingBody` | INTEGRATE | Bounded streaming to one contained private snapshot, with response closure. |
| `DeleteObject` | INTEGRATE | Exact-generation cleanup, followed by exact absence proof when the semantic contract requires it. |
| `ListObjectsV2` / continuation token | INTEGRATE | One bounded evidence page for reconciliation; never catalog membership or visibility. |
| SDK transport checksums | INTEGRATE | Supplemental transport diagnostics only; signed manifest SHA-256 and byte size remain authoritative. |
| ETag response metadata | INTEGRATE | Diagnostic recording only; multipart/encryption semantics prevent canonical integrity use. |
| Botocore modeled service/credential/timeout errors | INTEGRATE | Preserve cause/stage and distinguish conflict, retryable ambiguity, permissions, and configuration. |
| Bucket versioning | OPT-OUT | D-09 does not require it; immutable unique generation keys establish the lifecycle contract. |
| S3 Object Lock/WORM | OPT-OUT | Deployment retention feature, not needed for conditional unique-key publication. |
| Cross-region replication | OPT-OUT | Deployment durability/availability feature outside the qualified topology contract. |
| Bucket creation/deletion or policy management | OPT-OUT | Qualification uses an externally supplied least-privilege bucket and never mutates shared infrastructure. |
| Bucket lifecycle-rule management | OPT-OUT | Abort-incomplete rules are recommended cost hygiene, not correctness or authority; tests do not install policies. |
| Event notifications/SQS/SNS/Lambda | OPT-OUT | No event-driven coherence or second coordination channel is permitted. |
| Presigned URLs | OPT-OUT | The engine reads/writes through the participant and must verify a private snapshot before handler access. |
| `CopyObject` / mutable stable-key replacement | OPT-OUT | Replacement creates a new immutable generation; copy-based publication would widen ambiguity/overwrite semantics. |
| S3 Select, inventory reports, batch operations | OPT-OUT | They are broad/unbounded derived services and cannot become lifecycle authority. |
| Custom S3-compatible endpoint support | OPT-OUT | D-08 qualifies Amazon S3 only; another service requires its own named live qualification. |

## PostgreSQL / psycopg capability matrix

| capability | decision | reason |
|---|---|---|
| External DSN / injected connection factory or pool | INTEGRATE | One transaction-scoped lease per operation; connections are created safely after fork. |
| `Connection.transaction()` | INTEGRATE | Short database-only authority transitions with rollback on pre-commit failures. |
| Bound SQL values | INTEGRATE | Prevent value injection and preserve query plan/type behavior. |
| `psycopg.sql.Identifier` | INTEGRATE | Safely compose the explicit test/store schema identifier. |
| `INSERT ... ON CONFLICT` | INTEGRATE | Operation idempotency and unique-key preparation under concurrency. |
| Conditional `UPDATE ... RETURNING` | INTEGRATE | Exact expected lineage/generation/revision/digest CAS; zero rows is conflict. |
| Keyset queries with `LIMIT` | INTEGRATE | Bounded catalog, cleanup-debt, clear, and reconciliation pages. |
| Transaction-local `lock_timeout` | INTEGRATE | Bounded lock wait mapped to a typed retryable progress result. |
| Transaction-local `statement_timeout` | INTEGRATE | Bounded operation policy without a universal success promise. |
| SQLSTATE exception classes (`40001`, `40P01`, `55P03`, `57014`) | INTEGRATE | Distinguish serialization, deadlock, lock, and statement progress failures from exact CAS conflicts. |
| Connection operational/timeout errors | INTEGRATE | Typed retryable backend outcome with preserved cause and redacted bounded context. |
| Server/version identity query | INTEGRATE | Sanitized live evidence and exact compatibility checks. |
| Explicit DDL in `initialize()` | INTEGRATE | Idempotent current-schema creation before workers; ordinary open is validation-only. |
| Advisory locks | OPT-OUT | D-15 prohibits a global advisory-lock protocol; exact constraints/CAS are the correctness boundary. |
| Two-phase commit/prepared transactions | OPT-OUT | PostgreSQL still cannot enlist S3; 2PC would not create the claimed cross-resource transaction. |
| `LISTEN`/`NOTIFY` | OPT-OUT | Distributed invalidation/coherence is deferred and would add a coordination channel. |
| Async psycopg API | OPT-OUT | Native async storage APIs are deferred; the Phase 5 public engine is synchronous. |
| Logical/physical replication APIs | OPT-OUT | Deployment HA/replication is outside participant semantics and qualification scope. |
| SQLAlchemy ORM lifecycle layer | OPT-OUT | The narrow authority uses direct semantic transactions; the ORM projection/SqlCache systems remain separate. |
| Implicit schema migration on connect/open | OPT-OUT | D-17 requires exact read-only version detection and stopped-worker Phase 7 migration. |
| Server-side retry-until-success procedure | OPT-OUT | D-16 permits typed retryable progress outcomes; retries, if any, are bounded whole transactions. |
| Cross-resource foreign-data wrapper or filesystem authority | OPT-OUT | S3 remains an external immutable effect coordinated by intent/debt, not another authority. |

## Coverage gates

- Plan 05-03 implements every S3 `INTEGRATE` row and tests service errors/bounds
  with deterministic fakes/Moto; those tests do not satisfy BACK-05.
- Plans 05-04 and 05-05 implement every psycopg `INTEGRATE` row and the complete
  semantic `LifecycleAuthority` protocol.
- Plans 05-07 and 05-08 bind these capabilities to fixed live suites.
- Plan 05-10 alone may change the remote profile from live-pending to qualified,
  and only after real PostgreSQL plus Amazon S3 evidence is `QUALIFIED`.

## Explicit non-capabilities

This phase does not provide a transaction spanning PostgreSQL and S3, universal
contender success, bucket/service administration, S3-compatible-service parity,
native async APIs, distributed cache coherence, or runtime schema migration.
Those exclusions do not weaken immutable publication, signed integrity,
deterministic intent/debt recovery, exact CAS, contained reads, or bounded work.
