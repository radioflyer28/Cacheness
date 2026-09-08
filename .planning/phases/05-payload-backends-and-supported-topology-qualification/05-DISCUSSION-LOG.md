# Phase 5: Payload Backends and Supported Topology Qualification - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution
> agents. Decisions are captured in CONTEXT.md; this log preserves alternatives.

**Date:** 2026-09-08
**Phase:** 5-payload-backends-and-supported-topology-qualification
**Mode:** Auto; the user previously authorized recommended answers for the
milestone continuation loop.
**Areas discussed:** Supported topology inventory, live-service qualification,
S3 generation semantics, PostgreSQL authority semantics, evidence and failure
claims

## Supported Topology Inventory

| Option | Description | Selected |
| --- | --- | --- |
| Minimal coherent matrix | Memory/memory, SQLite/filesystem, and PostgreSQL/AWS S3; reject other combinations unless separately qualified | ✓ |
| Every plausible pairing | Support all combinations that seem technically usable | |
| Cartesian parity | Require every payload backend with every authority | |

**Choice:** Minimal coherent matrix (recommended default).

JSON remains projection-only. Registration or direct construction is not
support evidence, and unqualified combinations fail at composition rather than
silently downgrading.

## Live-Service Qualification

| Option | Description | Selected |
| --- | --- | --- |
| Require real services | PostgreSQL and AWS runs establish support; mocks remain contract tests | ✓ |
| Accept emulators | Let local compatible services qualify the production claims | |
| Accept mocks | Treat unit-level SDK behavior as sufficient | |

**Choice:** Require real services (recommended default).

Unavailable credentials/endpoints produce explicit not-qualified evidence.
Tests use external credentials and unique test-owned namespaces with bounded
cleanup; secrets never enter the repository.

## S3 Generation Semantics

| Option | Description | Selected |
| --- | --- | --- |
| Immutable generation objects | Unique keys, conditional creation, manifest SHA-256, bounded streaming/paging, cleanup debt | ✓ |
| Mutable stable objects | Overwrite a fixed object key and use its presence as current state | |
| Bucket-version authority | Require S3 versioning and treat object versions as lifecycle authority | |

**Choice:** Immutable generation objects (recommended default).

ETag is diagnostic rather than the security digest. Reads verify one private
snapshot before handler invocation, and failed deletion remains authority-owned
cleanup debt.

## PostgreSQL Authority Semantics

| Option | Description | Selected |
| --- | --- | --- |
| Transactional row CAS | PostgreSQL owns canonical rows, intent, and debt; transactions and conditional writes coordinate callers | ✓ |
| Global advisory locks | Serialize operations through a separate database lock protocol | |
| Application queue | Coordinate hosts through a Cacheness-managed distributed queue | |

**Choice:** Transactional row CAS (recommended default).

Initialization is explicit; ordinary opens do not migrate. Conflicts remain
distinct from typed retryable database outcomes. PostgreSQL does not extend its
ACID boundary over S3.

## Evidence and Failure Claims

| Option | Description | Selected |
| --- | --- | --- |
| Exact capability matrix | Name pairings, scopes, outcomes, service requirements, and non-claims; separate safety/recovery/progress/performance evidence | ✓ |
| Generic supported label | Present all backends behind one undifferentiated guarantee | |
| Class-presence support | Treat implementation/registration as sufficient evidence | |

**Choice:** Exact capability matrix (recommended default).

Phase 5 owns backend-focused real-service qualification. Phase 8 reuses these
tests for the complete Python/platform/install matrix and final performance
budgets.

## the agent's Discretion

- Exact adapter names, PostgreSQL schema details, isolation settings, bounded
  retry defaults, test namespace formats, and S3 multipart/page sizes within
  the locked behavioral boundaries.

## Deferred Ideas

- Additional payload/authority pairings and named S3-compatible services.
- S3 bucket versioning, object lock, cross-region replication, and CDN policy.
- Cache policy composition (Phase 6), migration execution (Phase 7), and full
  release/platform/performance qualification (Phase 8).
