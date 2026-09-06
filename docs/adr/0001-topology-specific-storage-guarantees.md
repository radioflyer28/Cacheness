# ADR 0001: Match storage guarantees to backend topology

**Status:** Accepted

## Context

Cacheness stores payload bytes outside the metadata database. Depending on
configuration, those bytes may live in memory, on a local filesystem, or in an
object store such as S3. Metadata may likewise live in memory, JSON, SQLite, or
PostgreSQL.

SQLite gives us a strong transactional authority for local metadata, but its
transaction cannot include filesystem renames, file deletion, or S3 requests.
PostgreSQL improves multi-host coordination, but it cannot make those external
blob operations part of its transaction either. Full ACID behavior therefore
ends at the boundary of the selected transactional resource.

Phase 3 exposed a recurring failure mode in our design process: a safety goal
such as "same-key operations must not corrupt data" gradually became an
availability goal such as "every contender must succeed within a benchmark-
derived deadline." Each failure then prompted another lock, queue, timeout
stage, or compatibility coordination mechanism. Those patches can close an
individual race while increasing the number of mechanisms participating in one
logical lifecycle, creating still more seams to coordinate.

This ADR defines the guarantee model that future phases must use. It is the
source of truth when requirements, tests, or reviews appear to demand stronger
concurrency or atomicity guarantees.

## Decision

Cacheness will provide **topology-specific guarantees**. It will not claim the
same atomicity, availability, or concurrency properties for every combination
of metadata and blob backends.

For durable configurations, one metadata authority owns the lifecycle state
machine. Payloads are written as immutable generations before that authority
publishes them. Operations spanning the authority and an external blob store
are crash-consistent through durable intent, idempotent completion, and
deterministic reconciliation; they are not described as one ACID transaction.

`BlobStore` owns this storage lifecycle. `UnifiedCache` depends on `BlobStore`
and adds cache policy such as TTL, admission, and eviction. Process-local
coordination may improve performance, but correctness must not depend on it.

## Guarantee vocabulary

Every concurrency or reliability requirement must be classified before it is
implemented or tested:

- **Integrity (safety):** an operation never exposes a mixed generation,
  corrupts a committed payload, escapes an allowed path, or makes metadata
  identify bytes that were never validly published.
- **Recovery:** after interruption, durable evidence is sufficient to finish,
  roll back, or reconcile an incomplete lifecycle transition
  deterministically.
- **Progress (availability):** an operation completes within a stated bound.
  Under supported contention, a documented, typed, retryable timeout may be a
  correct bounded outcome.
- **Performance:** latency or throughput measured under a named workload and
  environment. A performance budget detects regressions; it does not silently
  strengthen the correctness contract.
- **ACID scope:** atomicity applies only within one transactional resource.
  Coordination with filesystem or object-store effects uses the recovery
  model above.

Tests and plans must use these terms explicitly rather than the unqualified
words "atomic," "race-free," or "reliable."

## Topology capability matrix

| Topology | Supported coordination scope | Durability and atomicity | Declared progress behavior |
| --- | --- | --- | --- |
| Memory authority + memory blobs | One process | Atomic only within the process; no crash durability | Process scheduling and configured bounds apply |
| SQLite authority + local filesystem blobs | One host, multiple processes | SQLite transaction is authoritative; immutable blobs plus reconciliation provide crash consistency, not cross-resource ACID | Contention may return a typed, retryable timeout |
| PostgreSQL authority + filesystem or S3 blobs | Multiple hosts where the blob backend is shared and correctly configured | PostgreSQL coordinates lifecycle state; external blob effects still require intent and reconciliation | Serialization failures, deadlocks, or timeouts may be typed, retryable outcomes |
| JSON metadata + filesystem blobs | Explicitly limited local topology | No database transaction; support requires a documented lock/recovery scheme and must not inherit SQLite claims | Strong multiprocess or multihost progress is not implied |

Backend construction must reject or clearly label configurations whose
topology cannot satisfy the requested capability tier. A convenient backend
name is not evidence of a stronger guarantee.

## Non-negotiable safety and recovery invariants

All supported topologies must meet the invariants applicable to their declared
durability tier:

1. A reader observes either a complete previously committed generation or a
   complete newly committed generation, never a mixture.
2. Payload bytes are immutable once eligible for publication. Replacement
   creates a new generation rather than modifying visible bytes in place.
3. Authority promotion is the visibility point. Filesystem presence, a staging
   filename, or an object listing is not authority.
4. A failed post-commit cleanup becomes durable cleanup debt. It is retried or
   reconciled; it is not silently forgotten.
5. Recovery derives action from durable state and verifiable blob identity. It
   does not guess lifecycle state from timing or incidental path presence.
6. Corrupt metadata, unsafe paths, and integrity failures fail closed.
7. Same-key contention may delay or reject an operation, but it must not
   corrupt a committed generation or create metadata/payload disagreement.

## Guarantees we intentionally do not make

Cacheness does not promise:

- one ACID transaction across SQLite or PostgreSQL and filesystem or S3;
- wait-free or starvation-free operation under arbitrary process scheduling;
- that every concurrent contender succeeds within a fixed small deadline;
- identical concurrency guarantees across memory, JSON, SQLite, and PostgreSQL;
- that PostgreSQL removes the need for immutable blobs and reconciliation;
- that a benchmark threshold is a runtime correctness deadline.

These are boundary statements, not permission to corrupt or lose acknowledged
data. Integrity and deterministic recovery remain mandatory within the
declared topology.

## Runtime deadlines and benchmarks

Runtime deadlines are operational policy. They must be configurable, measured
from a clearly stated boundary, and end in a stable typed exception containing
enough context for retry and diagnosis. A timeout can be a valid result of
contention; the system must remain safe after it.

Benchmarks are statistical acceptance tools tied to a workload, platform, and
sample method. They should report distributions and regressions. They must not
be copied into lifecycle code as universal deadlines or used to require every
operation in a concurrent test to succeed.

Concurrency tests therefore separate:

- safety assertions: no corruption, split-brain publication, or lost cleanup
  obligation;
- recovery assertions: interrupted transitions converge deterministically;
- progress assertions: each operation either completes or returns one of the
  topology's declared bounded outcomes;
- performance assertions: aggregate latency/throughput in dedicated benchmark
  suites, outside correctness tests.

## Design rules for future phases

1. **One lifecycle authority.** Lifecycle state, generation selection, intent,
   and cleanup debt have one canonical owner. Filesystem paths and compatibility
   metadata are not additional authorities.
2. **Deep storage seam.** The lifecycle state machine belongs in one deep
   module. Backend adapters should expose the smallest transactional primitives
   needed by that state machine, not duplicate the lifecycle as a wide method-
   for-method interface.
3. **No correctness in process-local gates.** FIFO admission, mutexes, fork
   hooks, and in-process registries may reduce contention. Correctness must
   survive their absence, process death, and independent processes.
4. **Derived compatibility views.** Compatibility metadata is either updated
   in the same authority transaction or treated as rebuildable projection. A
   synchronous second source of truth is not allowed.
5. **Immutable external effects.** Blob creation precedes promotion; destructive
   cleanup follows promotion and is retryable.
6. **Capabilities at composition time.** Validate requested guarantees against
   metadata backend, blob backend, and deployment topology when constructing the
   store.
7. **Explicit initialization.** Schema/bootstrap work should happen at a
   deliberate initialization boundary where practical, not emerge as a hidden
   first-operation race.
8. **Policy stays above storage.** `UnifiedCache` expresses TTL, eviction, and
   admission through `BlobStore`; it does not recreate a parallel payload and
   metadata lifecycle.

The current broad `LifecycleAuthority` protocol and split among the lifecycle
engine, backend adapters, compatibility projections, and cache facade are
transitional architecture. Before expanding backend parity, prefer reducing
that interface and centralizing the state machine over adding another method or
coordination layer.

## Stop conditions for planning and review

Stop and revisit this ADR before implementing a proposed fix when any of these
conditions occurs:

- passing a test requires adding another lock, queue, sidecar, lease, or source
  of lifecycle truth;
- a test treats a documented typed contention timeout as corruption or requires
  every contender to succeed;
- a requirement says "atomic" across more than one transactional resource
  without defining intent and reconciliation;
- a guarantee cannot be stated for the selected backend topology;
- compatibility behavior requires two independently authoritative writes;
- a latency target changes public failure semantics instead of remaining a
  benchmark or configurable policy;
- a new backend adapter must reproduce lifecycle sequencing rather than supply
  transactional primitives.

The appropriate response is to narrow or tier the guarantee, simplify the
authority boundary, or change the topology—not to keep layering coordination
mechanisms until one stress test happens to pass.

## Required phase checklist

Before planning or approving a phase that changes storage lifecycle,
concurrency, recovery, backend composition, or timeout behavior:

1. Name the supported topology or topology tier.
2. Classify every relevant requirement as integrity, recovery, progress, or
   performance.
3. Identify the single lifecycle authority and its transaction boundary.
4. Describe external blob steps and their interruption/reconciliation behavior.
5. List typed bounded outcomes that callers must handle.
6. Verify tests keep safety, progress, and performance assertions separate.
7. Check the stop conditions above before adding coordination machinery.

A plan is not ready until these answers are explicit in its context, spec, or
plan document.

## Considered alternatives

### Claim full ACID for SQLite plus filesystem storage

Rejected because neither SQLite transactions nor filesystem operations can
atomically commit the other resource. Locking can serialize participants but
cannot create a shared transaction or eliminate crash windows.

### Require PostgreSQL for reliable operation

Rejected as the universal answer. PostgreSQL is the appropriate authority for
multi-host coordination, but external blobs still sit outside its transaction.
It strengthens a topology; it does not eliminate lifecycle design.

### Make process-local FIFO admission the primary concurrency mechanism

Rejected as a correctness boundary because it disappears on crash or fork and
cannot coordinate unrelated processes or hosts. It may remain an optional
optimization if benchmarks justify its complexity.

### Coordinate primarily through filesystem locks and paths

Rejected for database-backed configurations. It recreates multiple authorities
and platform-specific failure modes. Filesystem state is an external effect;
the lifecycle authority records what it means.

## Consequences

- SQLite remains a valid, reliable choice for a clearly scoped single-host
  topology.
- PostgreSQL becomes the path to stronger multi-host coordination, not a claim
  of cross-resource ACID.
- Some operations may return typed retryable contention outcomes even when all
  safety invariants hold.
- Backend documentation and configuration validation must expose capability
  differences rather than implying universal parity.
- Future race findings are evaluated against declared invariants before they
  trigger implementation work.
- Later architecture work should deepen and narrow the lifecycle authority seam
  before multiplying adapters or compatibility paths.
