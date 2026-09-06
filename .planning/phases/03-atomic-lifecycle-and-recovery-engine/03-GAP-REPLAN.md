# Phase 3 audit-grounded gap replan

Date: 2026-09-06. Baseline: `f5d406a`. Mode: `--gaps --inline`.
Planning and semantic review were performed by the primary agent, not independent
subagents. Implementation and fresh phase verification remain pending.

## Scope and precedence

ADR 0001 governs this plan set. The user requires a reliable BlobStore with
customizable catalog metadata and cache instances that consume that engine.
Separate cache/non-cache instances and namespaces meet that requirement.
Plans 01–18 and 20 remain completed implementation history; 19 stays superseded.
The unexecuted private-database-install draft of 21 is replaced, not approved.
Current `03-VERIFICATION.md` remains `gaps_found` and must not be overwritten by
planning success. Historical review files remain untouched.

| Plan / wave | Bounded deliverable | Dependency |
|---|---|---|
| 21 / 20 | Memory recovery IDs; strict projection decoding | 20 |
| 22 / 21 | Explicit initialization and SQLite failure classification | 21; compatibility checkpoint |
| 23 / 22 | Same-generation BlobStore read/write result interface | 22 |
| 24 / 23 | Canonical cache slice consumes that interface; projection failure is non-destructive | 23; failure-behavior checkpoint |
| 25 / 24 | Finite end-to-end acceptance and exact-commit qualification | 24 |

No production changes are part of this planning operation. Execution must not
interpret these five plans as five invitations to restart the architecture.

## Finding versus remedy

| Evidence | Violated invariant / class | Smallest required proof | Chosen remedy |
|---|---|---|---|
| CR-01 | Aborted memory work remains recoverable/pageable; recovery | Abort before publication, then public dry-run/apply | Stable terminal/retired row handling, no new scheduler |
| CR-02 | Resume never skips captured cleanup; recovery + integrity | Three debts, page/action size 1, intervening retirement | Monotonic IDs/keyset cursors, idempotent retirement |
| CR-03 | Corrupt projection cannot delete valid authority; integrity | Malformed SQLite parameters under strict signing | Raw strict decoder, then remove projection dependence from canonical cache reads/deletion |
| CR-04 | Initialization is confused with normal shared-store operation; progress + integrity | Initialize once, close, spawn independent workers, reopen | Explicit initialization-before-workers contract, subject to checkpoint; reject incomplete evidence unchanged |
| WR-01 | Operational failure falsely demands migration; recovery + progress | Primary and extended SQLite result-code matrix | Typed code-based translation with original cause |
| Audit 1/5 | Cache sequences the engine's lifecycle; integrity + recovery | Put/read/TTL and projection outage through cache using BlobStore | Supported result seam and engine-owned cleanup; preserve legacy adapter separately |
| Audit 4 | Primary catalog customization goal lacks acceptance | Direct put/query/update/reopen with application metadata | Existing mapping proof in 23/25; BACK-07 in Phase 4 |

## Supported topology and transactional scope

- Durable Phase 3: initialized stdlib SQLite authority + local filesystem blobs,
  one host, multiple independent processes. Memory authority: one process only;
  its fixture payload configuration is named, not falsely called crash-durable.
- The authority transaction commits descriptor, authenticated user metadata,
  intent/debt and progress records. Native payload I/O remains outside it.
- Before promotion, failure preserves the old generation and attributable intent;
  after promotion, the new generation remains authoritative and cleanup debt
  persists. Optional projection completion never gates old-payload reclamation.
- Success, exact conflict, typed retryable contention timeout, typed operational
  backend failure, fail-closed corruption/incompatibility, and recoverable
  post-commit cleanup are distinct outcomes. Do not require all contenders to win.
- Native Windows stays UNAVAILABLE/NOT_QUALIFIED (999.1); PostgreSQL/multihost and
  S3 composition are later work. Existing public compatibility is not silently
  withdrawn. Benchmark results are performance evidence only.

## Planned initialization contract (approval at 22)

Add an idempotent `BlobStore.initialize()` for an absent/empty supported root or
an already valid current catalog. It finishes schema creation/identity validation
and the storage-owned prerequisites needed by independent workers, then returns.
Applications call it once before starting workers; workers use normal constructors
to open the initialized store. A cache initializer delegates to its internal
BlobStore and completes its own optional metadata setup before sharing the instance.

Preserve zero-mutation inspection of absent stores. Preserve ordinary single-process
first-write convenience by routing it through the same initializer, not another
bootstrap implementation. That convenience is not a concurrent-create guarantee.
Normal opens/mutations validate existing schema; they do not silently migrate it.
Previously accepted internal schema upgrades require an explicit maintenance
invocation or typed migration-required outcome with a documented path (Phase 7
owns the general migration tool). No initialized database is renamed/replaced.

Initialization may leave incomplete contained evidence after a crash. The next
open must reject it unchanged with an actionable typed outcome; it must not adopt
an empty application-ID-zero leaf as proof of library ownership or a live writer.
No automatic destructive repair is allowed. Two concurrent first initializers
have no success/progress promise, but must still fail closed without clobbering
one another or exposing a partial entry. No private-candidate installation,
hardlink publication, readiness sidecar, FIFO, lease, or cross-process lock is
introduced to turn unsupported startup into supported availability.

## Planned small interface (23/24)

Keep `put -> key`, `get -> value or None`, `get_metadata`, `list`, and conditional
`delete` compatible. Add an additive supported seam for the cache:

- `open_entry(key)` returns a context-managed, verified entry snapshot or explicit
  absence. It exposes immutable authenticated metadata and an opaque exact-entry
  expectation; `read()` deserializes the already verified private snapshot.
  Snapshot acquisition completes generation revalidation and digest verification
  before yielding. TTL/signature policy can reject before deserialization. Closing
  releases the snapshot and instance admission; no DB transaction spans its life.
  A stored None is an entry with `read() is None`, not the absence result.
- `put_entry(...)` returns an immutable committed receipt (key, exact expectation,
  committed metadata) after the engine owns publication and cleanup. Recoverable
  cleanup errors carry that committed identity, never masquerade as a failed
  pre-commit write. The receipt exposes no prepare/promote/settle steps, raw DB
  connection, admission gate, or authority row implementation.
- Existing `delete(key, expected=...)` consumes the opaque expectation. Callers
  do not reread a different generation to authorize deleting the observed one.

Use the existing verified manifest and `LifecyclePutResult` internally; no new
stored schema or generic transaction framework is needed. Cache signing metadata
already stored in the manifest remains authenticated. Any retained pre-signing
compatibility transform must be pure, storage-owned invocation with no catalog
writes or ability to change identity/locator/digest. Do not add a generic policy
plugin framework. Avoid re-opening/re-hashing payloads in the cache facade.

## Derived state and compatibility (approval at 24)

Canonical reads obtain TTL/key diagnostics/signing data from the authenticated
entry, not from the compatibility projection. They neither repair nor delete a
corrupt projection. Direct projection/query APIs still report corrupt evidence
with key-attributed typed errors. Normal reads may serve the valid canonical
value while the corrupt projection remains available for diagnosis.

Normal put acknowledges authoritative storage independently of optional derived
index publication: retain its key result, emit a structured warning for a failed
optional export, and leave generation-tagged reconstruction possible. If the
caller explicitly requested custom ORM metadata and its separate transaction
fails, report a typed post-commit partial outcome with key/generation and cause;
never roll back/delete the blob or silently claim the custom link committed.
Existing typed recoverable cleanup outcomes remain distinct. Failure policy is
approved before changing characterized public behavior in Plan 24.

The Plan 24 checkpoint also covers concurrent cache close after the canonical
operation completes: remaining optional export or explicitly requested external
links may report the same declared diagnostic/partial outcome. Do not extend
storage admission across a second catalog to guarantee that derived work finishes.
BlobStore's own in-flight operation, snapshot lifetime and owned-resource close
guarantees remain intact; a stronger facade close promise requires a new decision,
not an automatically added cross-catalog barrier.

Canonical TTL/eviction/predicate deletion must use authenticated metadata and
exact expectations; stale/corrupt projected policy inputs cannot select a victim.
Legacy recognized formats retain their read-only compatibility path, signing
checks and errors. No new write is dual-written to a legacy payload lifecycle.
Generation-conditional projection/link updates may remain as derived adapter
operations; they cannot borrow a newer generation, trigger canonical repair, or
become a new durable coordination protocol. Pure aggregate projection reads may
omit stale rows, but must not hide malformed rows they actually decode.

## Finite failure model and stop rule

Acceptance covers: initialize/reopen; direct metadata round trip; canonical cache
put/get/TTL with separate store isolation; same-key overwrite/delete; named
pre/post-promotion interruptions; partial-stream and fsync publication interruption;
memory abort and paged debt; repeated cleanup/apply; corrupt manifest/projection;
operational errors and contention; legacy characterization; full repository gate.
Use deterministic barriers/events and spawned independent processes where sharing
is claimed. Test joins have harness timeouts, not universal service deadlines.

For a newly discovered issue, record the actual user-visible invariant, topology,
reproduction, and failure class BEFORE proposing a fix. At an ADR stop condition,
stop execution and report the smallest simplification/contract decision needed.
Do not add another mechanism or start an automatic repeated fix/review loop. A
failed qualification keeps gaps_found; one passing rerun never erases a failure.

## Spec-less assumptions and scope fences

No Phase 3 SPEC exists. The fallback probe leaves STOR-03/04/06/07 unclassified
and asks idempotency/concurrency questions for STOR-05. The finding table and
finite model above supply explicit planning assumptions, not new universal
guarantees. Running apply twice must converge; interrupted/parallel durable work
is governed by SQLite CAS and debt, memory by its declared one-process scope.

Defer richer schema/index design, payload-generation versus catalog-revision
separation, general adapter transaction narrowing, full backend pairings, all
cache policy migration and service qualification to Phases 4–8. Phase 4 must
narrow the current broad adapter protocol BEFORE copying it into more backends.
Those deferred items are not hidden Phase 3 completion gates.

## Protected state and evidence

Use only explicit owned paths for commits. Preserve dirty codebase maps, config,
milestone.lock, .claude, historical review/fix artifacts, cache/.cacheness, and the
original sqlite-columns-v0314 WAL/SHM fixtures, including lstat identity/timestamps
and SHA-256. Run compatibility/full tests against disposable copies in an isolated
checkout; never test against those original sidecars. Plan 25 records the exact
committed code, commands, outcomes, and before/after fingerprints. Planning checks
are not test execution and do not change qualification status.
