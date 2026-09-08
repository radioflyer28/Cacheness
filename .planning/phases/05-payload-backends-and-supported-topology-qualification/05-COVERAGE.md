# Phase 5 API Coverage: boto3/Amazon S3 and psycopg/PostgreSQL

**Scope:** Capabilities required by the declared Amazon S3 payload-participant
and PostgreSQL lifecycle-authority roles. `INTEGRATE` is the default. Each
explicit `OPT-OUT` records why the API cannot be silently implied by the three
supported profiles. This is an immutable capability contract, not a service-run
report.

<!-- phase5-api-coverage:start -->
| role | capability | decision | reason |
| --- | --- | --- | --- |
| Amazon S3 / boto3 | Standard boto3 credential-provider chain | INTEGRATE | External credentials only; no secret values in constructors, logs, or evidence. |
| Amazon S3 / boto3 | Injected boto3 S3 client | INTEGRATE | Supports deterministic contract tests and caller-owned service configuration without credential exposure. |
| Amazon S3 / boto3 | Region plus ExpectedBucketOwner | INTEGRATE | Binds configured requests to the intended AWS account and region. |
| Amazon S3 / boto3 | PutObject + IfNoneMatch="*" | INTEGRATE | Conditionally creates an immutable single-request generation. |
| Amazon S3 / boto3 | CreateMultipartUpload | INTEGRATE | Begins bounded upload of snapshots above the configured threshold. |
| Amazon S3 / boto3 | UploadPart | INTEGRATE | Streams bounded parts without whole-object memory buffering. |
| Amazon S3 / boto3 | CompleteMultipartUpload + IfNoneMatch="*" | INTEGRATE | Conditionally completes immutable multipart publication. |
| Amazon S3 / boto3 | AbortMultipartUpload | INTEGRATE | Performs bounded cleanup for a known incomplete upload. |
| Amazon S3 / boto3 | ListMultipartUploads | INTEGRATE | Produces bounded exact-prefix evidence for attributable stale uploads. |
| Amazon S3 / boto3 | HeadObject | INTEGRATE | Supports exact-key preflight and ambiguity classification, never visibility authority. |
| Amazon S3 / boto3 | GetObject / StreamingBody | INTEGRATE | Streams one contained private snapshot and closes the response before handler access. |
| Amazon S3 / boto3 | DeleteObject | INTEGRATE | Deletes an exact generation, then supports an exact absence proof when required. |
| Amazon S3 / boto3 | ListObjectsV2 / continuation token | INTEGRATE | Provides one bounded reconciliation evidence page, never catalog membership. |
| Amazon S3 / boto3 | SDK transport checksums | INTEGRATE | Supplies supplemental transport diagnostics; signed manifest SHA-256 and size stay authoritative. |
| Amazon S3 / boto3 | ETag response metadata | INTEGRATE | Records diagnostics only because multipart and encryption prevent canonical integrity use. |
| Amazon S3 / boto3 | Botocore modeled service/credential/timeout errors | INTEGRATE | Preserves cause and stage while classifying conflict, retryable ambiguity, permission, and configuration failures. |
| Amazon S3 / boto3 | Bucket versioning | OPT-OUT | Unique immutable generation keys provide the V1 lifecycle boundary without it. |
| Amazon S3 / boto3 | S3 Object Lock/WORM | OPT-OUT | A deployment retention feature is not required for conditional unique-key publication. |
| Amazon S3 / boto3 | Cross-region replication | OPT-OUT | Deployment durability and availability are outside the qualified topology contract. |
| Amazon S3 / boto3 | Bucket creation/deletion or policy management | OPT-OUT | The store uses an externally supplied least-privilege bucket and never mutates shared infrastructure. |
| Amazon S3 / boto3 | Bucket lifecycle-rule management | OPT-OUT | Abort-incomplete rules are cost hygiene, not authority or correctness. |
| Amazon S3 / boto3 | Event notifications/SQS/SNS/Lambda | OPT-OUT | Event coherence would introduce a second coordination channel. |
| Amazon S3 / boto3 | Presigned URLs | OPT-OUT | The participant must verify a private snapshot before handler access. |
| Amazon S3 / boto3 | CopyObject / mutable stable-key replacement | OPT-OUT | Replacement makes a new immutable generation; stable-key copy broadens overwrite ambiguity. |
| Amazon S3 / boto3 | S3 Select, inventory reports, batch operations | OPT-OUT | Broad derived services cannot become lifecycle authority. |
| Amazon S3 / boto3 | Custom S3-compatible endpoint support | OPT-OUT | Only Amazon S3 is named by the remote profile; another service needs separate qualification. |
| PostgreSQL / psycopg | External DSN / injected connection factory or pool | INTEGRATE | Uses one transaction-scoped lease per operation and safe post-fork creation. |
| PostgreSQL / psycopg | Connection.transaction() | INTEGRATE | Delimits short database-only authority transitions with pre-commit rollback. |
| PostgreSQL / psycopg | Bound SQL values | INTEGRATE | Prevents value injection and preserves driver typing and query planning. |
| PostgreSQL / psycopg | psycopg.sql.Identifier | INTEGRATE | Safely composes the explicit test/store schema identifier. |
| PostgreSQL / psycopg | INSERT ... ON CONFLICT | INTEGRATE | Makes operation preparation idempotent under unique constraints. |
| PostgreSQL / psycopg | Conditional UPDATE ... RETURNING | INTEGRATE | Enforces exact lineage/generation/revision/digest CAS; zero rows is conflict. |
| PostgreSQL / psycopg | Keyset queries with LIMIT | INTEGRATE | Bounds catalog, cleanup debt, clear, and reconciliation pages. |
| PostgreSQL / psycopg | Transaction-local lock_timeout | INTEGRATE | Bounds lock waits and maps them to a retryable progress result. |
| PostgreSQL / psycopg | Transaction-local statement_timeout | INTEGRATE | Bounds operations without a universal success promise. |
| PostgreSQL / psycopg | SQLSTATE exception classes (40001, 40P01, 55P03, 57014) | INTEGRATE | Separates serialization, deadlock, lock, and statement progress failures from CAS conflict. |
| PostgreSQL / psycopg | Connection operational/timeout errors | INTEGRATE | Returns a typed retryable backend outcome with preserved cause and redacted bounded context. |
| PostgreSQL / psycopg | Server/version identity query | INTEGRATE | Supports exact compatibility checks and sanitized service evidence. |
| PostgreSQL / psycopg | Explicit DDL in initialize() | INTEGRATE | Creates current schema idempotently before workers; ordinary open validates only. |
| PostgreSQL / psycopg | Advisory locks | OPT-OUT | A global advisory-lock protocol is prohibited; constraints and CAS are the correctness boundary. |
| PostgreSQL / psycopg | Two-phase commit/prepared transactions | OPT-OUT | PostgreSQL cannot enlist S3, so this cannot make a cross-resource transaction. |
| PostgreSQL / psycopg | LISTEN/NOTIFY | OPT-OUT | Distributed invalidation would add a coordination channel outside this scope. |
| PostgreSQL / psycopg | Async psycopg API | OPT-OUT | Native async storage APIs are deferred; this lifecycle engine is synchronous. |
| PostgreSQL / psycopg | Logical/physical replication APIs | OPT-OUT | Deployment replication is outside participant semantics and qualification scope. |
| PostgreSQL / psycopg | SQLAlchemy ORM lifecycle layer | OPT-OUT | The authority uses direct semantic transactions; ORM projection and SqlCache remain separate. |
| PostgreSQL / psycopg | Implicit schema migration on connect/open | OPT-OUT | Version detection is read-only and Phase 7 owns stopped-worker migration/rebuild. |
| PostgreSQL / psycopg | Server-side retry-until-success procedure | OPT-OUT | Declared typed retryable outcomes remain valid; retries are bounded whole transactions. |
| PostgreSQL / psycopg | Cross-resource foreign-data wrapper or filesystem authority | OPT-OUT | S3 is an immutable external effect reconciled by intent/debt, not another authority. |
<!-- phase5-api-coverage:end -->

## Qualification and performance boundaries

Contract and fake-service coverage proves the listed primitive semantics. The
remote profile additionally requires the exact real PostgreSQL/Amazon S3
evidence requirement in the topology matrix. The evidence artifact is the only
place a service-run result is recorded. Performance measurements are a separate
distribution and Phase 8 owns final budgets and matrix expansion; neither a
benchmark nor a local contract test changes atomicity or progress guarantees.

## Explicit non-capabilities

The supported profiles provide no transaction spanning PostgreSQL and S3,
universal contender success, bucket/service administration, compatible-service
parity, native async APIs, distributed cache coherence, or runtime schema
migration. These exclusions do not weaken immutable publication, signed
integrity, deterministic intent/debt recovery, exact CAS, contained reads, or
bounded work.
