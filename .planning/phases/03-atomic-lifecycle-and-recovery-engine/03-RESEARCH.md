# Phase 3: Atomic Lifecycle and Recovery Engine - Research

**Researched:** 2026-09-04
**Domain:** Transactional storage lifecycle authority, crash recovery, and concurrency
**Confidence:** HIGH for the in-repository replacement boundary; MEDIUM for the SQLite design because official documentation was read but the configured research provider fell back from Context7 to web search

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

DATA_76D97E72_START

### Immutable Generations and Publication

- **D-01:** Payload generations are immutable after candidate creation. A write serializes to a new generation-specific locator, verifies the candidate, and conditionally publishes one committed canonical manifest that names it. Payload bytes at an already published generation locator are never overwritten in place. — **Reversibility:** one-way — In-place generations would invalidate the same-snapshot and recovery guarantees and require another stored-layout migration.
- **D-02:** Publication uses an expected-generation compare-and-swap contract. Creating an absent key expects absence; overwriting expects the authenticated committed generation read by the operation. A failed comparison returns a typed conflict and never revokes the winner.
- **D-03:** Readers observe only committed manifests from Phase 2. Prepared intent and cleanup records are recovery evidence, not alternate normal-read authorities. A read may take one bounded retry only when it detects a generation changed during acquisition; it must not guess between generations.
- **D-04:** The authority point is the successful conditional manifest publication. Before it, the previous committed generation remains authoritative; after it, the new committed generation remains authoritative even if retirement of the old payload fails.

### Failure Residue and Operation Evidence

- **D-05:** Every multi-step mutation creates bounded, versioned operation evidence before the first externally persistent side effect. It records operation ID, logical key, expected generation, candidate/new generation, relevant locators, intended transition, and progress without copying payload contents.
- **D-06:** Operation evidence and generated locators carry enough authenticated or strictly validated provenance to distinguish library-owned residue from unrelated files/objects. Recovery never infers ownership from a loose filename pattern or handler payload contents.
- **D-07:** Failures before authority publication preserve the previous generation and leave the candidate plus intent detectable for rollback/reconciliation. Failures after authority publication preserve the new generation and leave old-generation reclamation as explicit cleanup debt. — **Reversibility:** costly — External operators and later migration tooling will depend on this pre/post-authority classification.
- **D-08:** Serialization failure before a candidate exists leaves no persistent lifecycle record. Once operation evidence or a candidate is persisted, cleanup failure is surfaced as a typed recoverable outcome rather than hidden behind the original operation result.

### Delete, Clear, and Close Convergence

- **D-09:** Delete conditionally publishes a signed tombstone for the expected generation before reclaiming payload bytes and retiring the tombstone. Repeated delete is successful for an already absent key, resumes reclamation for the same tombstone, and conflicts rather than deleting a newer generation.
- **D-10:** Clear is a resumable store-wide operation built from an authenticated bounded snapshot of keys and per-entry expected generations. It may use a store-wide admission barrier while establishing/committing that clear, but ordinary operations on distinct keys are not globally serialized. Keys created after the clear snapshot are not accidentally deleted.
- **D-11:** Cleanup is idempotent: missing already-reclaimed payloads or operation records count as completed steps when provenance and generation still match; a locator now owned by a different generation is a conflict and is never removed.
- **D-12:** `close()` is ownership-aware and idempotent. It stops new operations on that instance, waits for or deterministically cancels its in-flight work, flushes owned durable state, and releases owned resources. It does not clear stored user data and does not close caller-injected backends.

### Reconciliation Policy

- **D-13:** Reconciliation is dry-run by default and returns both a stable machine-readable report and a human-readable summary. Findings include authoritative generation, operation provenance, residue type, proposed action, reason, and whether the action is safe, blocked, or requires confirmation.
- **D-14:** Apply mode is resumable and checkpoints each completed action. Re-running after interruption converges without repeating destructive work or losing the only valid copy.
- **D-15:** Automatic repair is limited to outcomes proven by authenticated manifests, validated operation records, generation checks, and contained/backend-owned locators. Ambiguous, malformed, unsupported-version, or provenance-free evidence is quarantined only when a safe backend-native quarantine is possible; otherwise it is reported untouched.
- **D-16:** Reconciliation never deserializes trusted application payloads merely to decide ownership or lifecycle state, never guesses future schemas, and never silently deletes the last valid generation.

### Same-Key Coordination

- **D-17:** In-process work uses a bounded per-key coordination mechanism, not one global mutex. Coordination entries are retired when unused so unbounded key cardinality does not become a memory leak.
- **D-18:** Backend generation compare-and-swap is the correctness boundary across processes/instances; local locks are an optimization and ordering aid, not the cross-process guarantee. A topology that cannot truthfully provide the required conditional publication fails with a typed unsupported-capability outcome rather than downgrading.
- **D-19:** Same-key write/write races have one conditional-publication winner; losers receive a typed conflict and clean or report only their own candidate residue. Write/delete races obey the same expected-generation rule. Reads return a complete committed generation or a typed lifecycle/conflict outcome, never mixed bytes and metadata.
- **D-20:** Operations on distinct keys proceed independently except for an explicit store-wide clear/reconciliation admission barrier. Lock acquisition order for multi-key work is deterministic to avoid deadlocks.

### Local Authority Trust and Windows Scope

- **D-21:** The OS principal that owns a local store is inside the trusted deployment boundary for lifecycle-control availability. Cacheness validates control-object type, containment, identity, and authenticated contents and fails closed when observable substitution occurs, but it does not promise continued operation or immutable per-key authority if that same principal deliberately deletes or rebinds every lifecycle authority object while the store is live. Ordinary Cacheness processes never perform such rebinding. — **Reversibility:** costly — Defending against a hostile store owner would require an external coordinator, privileged mandatory controls, or store-wide serialization and therefore changes the architecture or deployment model.
- **D-22:** In this milestone, Windows local-store coordination is supported only among processes running as one OS user in one interactive or service session. Cross-user, cross-service, and cross-session access to the same local store is unsupported and must fail or be prevented by deployment ACLs; supporting it later requires an explicit global authority namespace, ACL/security-descriptor contract, and native Windows validation. Advertised Windows compatibility otherwise remains in force. — **Reversibility:** reversible — A later milestone may broaden the topology after implementing and validating that authority contract.

### Transactional Authority Replan

- **D-23:** The file-native lifecycle scheduler built from operation receipts, inventory events, heads, anchors, cursors, pending-control records, and staged control files is rejected. The replacement removes that protocol rather than wrapping or incrementally repairing it. D-05 through D-20 remain behavioral requirements only where they do not prescribe that discarded mechanism. — **Reversibility:** costly — Reintroducing file-native transactional coordination would require new proof that it is smaller and more reliable than the transactional authority.
- **D-24:** `BlobStore` depends on one deep lifecycle-authority module whose interface exposes complete entry-state transactions, not individual receipt/checkpoint/storage primitives. The authority atomically owns canonical manifest state, mutation intent, cleanup debt, clear membership/progress, and reconciliation checkpoints. Callers never coordinate those records themselves.
- **D-25:** Payload generations remain immutable native handler output outside the authority transaction. A durable authority intent is committed before the first persistent payload side effect; a later authority transaction conditionally promotes the verified generation and records any cleanup debt. Recovery queries indexed intent/debt rows and never discovers protocol state by scanning filenames.
- **D-26:** The local persistent reference adapter uses Python's standard-library SQLite engine as the transactional authority, with explicit durability configuration and schema/version migration. JSON remains a supported metadata projection and compatibility representation, but it is not an independent cross-process transaction authority. Configuration that requests durable multi-process guarantees without a capable authority fails with a typed unsupported-capability outcome rather than silently downgrading.
- **D-27:** The in-memory authority adapter provides deterministic same-process behavior for tests and ephemeral stores. PostgreSQL and S3-capable authority adapters use their native transaction or conditional-write facilities through the same lifecycle-authority interface in Phases 4 and 5; Phase 3 must not recreate the discarded filesystem scheduler inside those adapters.
- **D-28:** Distinct-key serialization, payload verification, and cleanup proceed concurrently. A backend may serialize only the short authority commit itself (SQLite has one writer), and no authority transaction or global/store/family lease may span handler serialization, payload upload/fsync, payload verification, or reclamation.
- **D-29:** Clear stores its exact target generations and progress as transactional authority rows. Reconciliation pages indexed authority rows using stable database cursors and explicit work/byte/action limits; it does not maintain a second file inventory or infer provenance.
- **D-30:** The implementation must delete or retire the abandoned file-native scheduler modules and tests rather than carry both lifecycle engines. Replacement tests exercise behavior through the lifecycle-authority interface and `BlobStore`; low-level receipt/inventory tests are historical evidence, not the new test surface.
- **D-31:** Windows local persistence uses the same SQLite transactional authority and the D-22 one-user/session trust scope. No custom Win32/POSIX lock-file authority, inode/handle identity registry, extended-attribute reservation, or platform-specific receipt protocol is part of the replacement.

DATA_76D97E72_END

### the agent's Discretion

DATA_412BC23C_START

- Exact class/module names for the lifecycle authority, transaction records, and reconciliation reports.
- Whether local per-key coordination uses lock striping or dynamically retained locks, provided unrelated keys remain concurrent and retention is bounded.
- Exact bounded database busy retry/backoff values and authority-row encoding, provided failure outcomes remain deterministic and testable.
- Whether safe quarantine is implemented as a contained rename, backend namespace move, or immutable report-only disposition for a backend that cannot move atomically.

DATA_412BC23C_END

### Deferred Ideas (OUT OF SCOPE)

DATA_78D12C18_START

- Concrete capability declarations and full metadata-backend conditional publication across JSON, memory, SQLite, and PostgreSQL — Phase 4.
- Full filesystem, memory, and S3 payload lifecycle implementation and matrix verification — Phase 5.
- `UnifiedCache` delegation, TTL/eviction/invalidation policy, and miss/statistics translation — Phase 6.
- Stored-format inventory and copy-verify-switch migration execution — Phase 7; Phase 3 reconciliation handles lifecycle inconsistency, not format migration.

DATA_78D12C18_END
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| STOR-03 | A write exposes either the previous complete generation or the new complete generation, never partial payload or metadata state. | The prepare/publish/verify/promote sequence makes one SQLite transaction the only visibility switch while payloads remain immutable. |
| STOR-04 | A failed write preserves the last valid generation and leaves any residue detectable and recoverable. | A committed mutation-intent row identifies the exact candidate locator before publication; cleanup debt makes post-promotion residue queryable. |
| STOR-05 | Overwrite, delete, clear, and close operations are idempotent and clean up both payload and metadata state. | Tombstone promotion, exact-generation clear targets, idempotent debt retirement, and ownership-aware close are modeled as authority transitions. |
| STOR-06 | Operators can run dry-run and resumable reconciliation that detects inconsistent state and safely repairs, quarantines, or reports it. | Indexed mutation/debt/clear tables, captured high-water marks, keyset cursors, and transactional checkpoints replace filename discovery. |
| STOR-07 | Same-key races have deterministic outcomes through per-key coordination and backend generation checks without globally serializing distinct keys. | Expected-generation plus expected-manifest-digest CAS selects one winner; only short SQLite writer transactions serialize and payload work overlaps. |
</phase_requirements>

## Summary

Phase 3 should replace the rejected filesystem scheduler with one deep `LifecycleAuthority` module. The module owns the only transactional source of truth for committed manifests, prepared mutations, cleanup debt, clear targets/progress, and reconciliation checkpoints. `BlobStore` asks it to perform complete state transitions; neither `BlobStore` nor the lifecycle orchestrator manipulates receipt fragments, inventory heads, anchors, staged control files, or raw authority tables. This boundary directly implements D-23 through D-31 and keeps payload formats native. [VERIFIED: `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:80-99`]

The persistent local reference adapter should use Python's standard-library `sqlite3` with explicit transactions, rollback-journal `DELETE` mode, `synchronous=EXTRA`, a bounded busy deadline, and local-filesystem-only topology. SQLite permits multiple readers but only one simultaneous writer; `BEGIN IMMEDIATE` starts a write transaction immediately and can report `SQLITE_BUSY`. [CITED: https://www.sqlite.org/lang_transaction.html] The short writer transaction is acceptable commit serialization under D-28: it contains only comparisons and row changes, never handler serialization, payload publication/fsync, verification, deletion, projection export, or callbacks.

The essential simplification is that recovery no longer asks “which files imply unfinished protocol state?” It asks indexed authority tables for exact unfinished operations and debts. The authority records a durable intent before the first managed payload publication, then payload I/O proceeds outside any authority transaction, and a later conditional transaction promotes the verified generation and records old-generation cleanup debt atomically. JSON metadata becomes a rebuildable projection rather than a second cross-process authority.

**Primary recommendation:** Implement a small transactional state machine behind `LifecycleAuthority`; ship `SqliteLifecycleAuthority` and `InMemoryLifecycleAuthority`; delete the file-native scheduling protocol rather than adapting it.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|--------------|----------------|-----------|
| Canonical entry visibility and expected-generation CAS | Database / Storage authority | API / Backend orchestration | Visibility changes only through an atomic authority transaction. |
| Native payload serialization, publication, verification, reclamation | Database / Storage payload tier | Handler strategies | Payload work is deliberately outside authority transactions. |
| Mutation intent and cleanup debt | Database / Storage authority | API / Backend orchestration | Recovery evidence must commit atomically with lifecycle state. |
| Clear target membership and progress | Database / Storage authority | API / Backend orchestration | A transaction captures exact generations; workers execute bounded target rows. |
| Reconciliation reporting/checkpointing | API / Backend orchestration | Database / Storage authority | The orchestrator renders reports; the authority supplies stable pages and checkpoints. |
| JSON metadata compatibility | Projection/export boundary | Database / Storage authority | JSON mirrors committed state but never arbitrates concurrent publication. |
| In-process admission and per-key ordering | API / Backend process | Database / Storage authority | Local coordination is an optimization; authority CAS remains correctness boundary. |

## Standard Stack

### Core

| Library/runtime | Version | Purpose | Why standard here |
|-----------------|---------|---------|-------------------|
| Python `sqlite3` | Python 3.11+ standard library; runtime-linked SQLite | Local persistent lifecycle authority | Locked by D-26; no new package or dependency-resolution risk. |
| SQLite rollback journal | Runtime SQLite | Atomic local authority transactions | Works on the supported stdlib runtimes without depending on a sufficiently patched WAL implementation. |
| Existing canonical manifest/signing code | Repository implementation | Authenticated committed entry state | Phase 2 already defines committed-only visibility and native payload integrity. |
| Existing `GuardedHandlerIO` payload primitives | Repository implementation | Stage, immutable publish, verify, reclaim | Keeps native handler bytes outside the authority database. |
| `threading` synchronization | Python standard library | In-memory adapter and bounded local key coordination | Same-process ordering only; never the cross-process correctness boundary. |

### Supporting

| Library/runtime | Version | Purpose | When to use |
|-----------------|---------|---------|-------------|
| Python `multiprocessing` / subprocesses | Python 3.11+ | Crash and cross-process CAS tests | Spawn fresh processes with independently opened authorities. |
| pytest | Repository-locked test framework | Contract, fault-injection, race, and crash tests | All authority adapters and `BlobStore` lifecycle behavior. |

No external package installation is required, so the package-legitimacy gate does not apply. The project declares Python `>=3.11`, and the repository pins Python `3.13`. [VERIFIED: `pyproject.toml:9`; quote: `requires-python = ">=3.11"`; VERIFIED: `.python-version:1`; quote: `3.13`]

### Journal-mode decision

Use `PRAGMA journal_mode=DELETE` with `PRAGMA synchronous=EXTRA` for the Phase 3 local authority. Do **not** make WAL the default.

- SQLite WAL improves reader/writer overlap but still permits only one writer at a time and requires every process to be on the same host; it does not work over a network filesystem. [CITED: https://www.sqlite.org/wal.html]
- SQLite now documents a rare WAL-reset corruption bug affecting versions `3.7.0` through `3.51.2` under concurrent writes/checkpoints, fixed in `3.51.3` with specified backports. [CITED: https://www.sqlite.org/wal.html]
- The currently checked system and project Python interpreters link SQLite versions `3.43.1` and `3.47.1`, respectively. [VERIFIED: runtime probe performed 2026-09-04; commands: `python3 -c 'import sqlite3; print(sqlite3.sqlite_version)'` and `.venv/bin/python -c 'import sqlite3; print(sqlite3.sqlite_version)'`; quoted results: `3.43.1`, `3.47.1`]
- In rollback-journal mode, `synchronous=EXTRA` adds a directory sync after unlinking the rollback journal, strengthening the durability of the most recent commit beyond `FULL` on relevant filesystems. [CITED: https://www.sqlite.org/pragma.html#pragma_synchronous]

WAL can be reconsidered only as an explicit later capability after the runtime SQLite version is proven to contain the fix and the checkpoint/crash matrix is added. That is not needed to satisfy D-28 because the long-running payload work, rather than the short row commit, is the concurrency-sensitive portion.

### Required SQLite connection policy

Each authority method should open or borrow a connection owned by the current process and thread, set and verify connection pragmas, execute one bounded operation, and close/return the connection. The simplest Phase 3 reference is a method-scoped connection with `check_same_thread=True` and `isolation_level=None`, followed by explicit SQL `BEGIN IMMEDIATE` for state-changing transactions.

Recommended connection contract:

| Concern | Required behavior |
|---------|-------------------|
| Transaction control | Set `isolation_level=None`; issue explicit `BEGIN IMMEDIATE`, `COMMIT`, and `ROLLBACK`. Python documents that `None` leaves the underlying SQLite library in autocommit mode, so code controls transactions explicitly. [CITED: https://docs.python.org/3.11/library/sqlite3.html#transaction-control] |
| Writer acquisition | Acquire the writer at transaction start with `BEGIN IMMEDIATE`, so contention is reported before application reads data it intends to update. SQLite documents that it may fail with `SQLITE_BUSY`. [CITED: https://www.sqlite.org/lang_transaction.html] |
| Busy/deadline | Carry one absolute authority deadline into connection `timeout`/`PRAGMA busy_timeout`; cap any retry at remaining time and translate final `BUSY`/`LOCKED` to a typed lifecycle timeout. Python's default connection timeout is five seconds, but Phase 3 should make the bounded policy explicit. [CITED: https://docs.python.org/3.11/library/sqlite3.html#sqlite3.connect] |
| Durability | Require returned `journal_mode` to be `delete` and returned `synchronous` to be the requested strong setting; fail typed if the database cannot honor them. |
| Integrity | Set `foreign_keys=ON` outside a transaction, `trusted_schema=OFF`, and run `PRAGMA integrity_check` plus `PRAGMA foreign_key_check` in validation/reopen diagnostics. SQLite recommends explicitly setting foreign-key enforcement and encourages disabling trusted schema. [CITED: https://www.sqlite.org/pragma.html#pragma_foreign_keys] [CITED: https://www.sqlite.org/pragma.html#pragma_trusted_schema] |
| Schema identity | Assign an application-specific `application_id` and maintain ordered schema migrations with `user_version`. Both are application-controlled header fields. Configure/verify journal mode before migration, then use one bounded `BEGIN EXCLUSIVE` migration transaction so another process cannot observe a partial schema. [CITED: https://www.sqlite.org/pragma.html#pragma_application_id] [CITED: https://www.sqlite.org/pragma.html#pragma_user_version] [CITED: https://www.sqlite.org/lang_transaction.html] |
| Close | Explicitly close every connection; Python's connection context manager commits or rolls back but does not close the connection. [CITED: https://docs.python.org/3.11/library/sqlite3.html#how-to-use-the-connection-context-manager] |
| Fork | Capture creating PID and reject use after fork; the child must construct/open a new authority. SQLite warns not to carry a connection across `fork()` or close the inherited connection from the child. [CITED: https://www.sqlite.org/howtocorrupt.html#_carrying_an_open_database_connection_across_a_fork_] |
| Filesystem | Support only a local filesystem for this adapter. SQLite documents unreliable locking/synchronization risks over network filesystems. [CITED: https://www.sqlite.org/useovernet.html] |

Do not claim portable automatic detection of every network or userspace filesystem. Require configuration/topology capability validation and fail closed when a durable multi-process local authority cannot be established. [ASSUMED]

## Architecture Patterns

### System Architecture Diagram

```text
BlobStore put/delete/clear/reconcile
                |
                v
       LifecycleEngine orchestration
                |
                +-------- private handler staging (no managed-store side effect)
                |
                v
       LifecycleAuthority deep module
       +-------------------------------+
       | committed canonical entries   |
       | mutation intent                |
       | cleanup debt                   |
       | clear targets/progress         |
       | reconciliation checkpoints    |
       +-------------------------------+
          |                    |
          | short transaction  | stable indexed pages
          v                    v
  SQLite local adapter     In-memory adapter
          |
          | committed intent / later promotion
          v
 GuardedHandlerIO / native payload backend
          |
          +--> immutable generation publish
          +--> exact digest/size verification
          +--> idempotent old-payload cleanup

Committed authority revision
          |
          v
 JSON projection exporter (derived compatibility snapshot; never CAS authority)

Later phases only:
  PostgreSQL authority adapter     S3-native conditional authority adapter
```

### Recommended Project Structure

```text
src/cacheness/storage/
├── lifecycle_authority.py          # deep interface, domain records, complete transitions
├── sqlite_lifecycle_authority.py   # local persistent reference adapter
├── memory_lifecycle_authority.py   # deterministic ephemeral adapter
├── lifecycle.py                    # thin payload-I/O orchestration
├── reconciliation.py               # report rendering and bounded orchestration
├── guarded_handler_io.py           # retained native immutable payload operations
├── manifest.py                     # retained Phase 2 canonical manifest
├── coordination.py                 # only bounded in-process key/instance coordination
└── path_security.py                # only containment and durable payload primitives
```

Names are recommendations under the agent's discretion, not existing public API. [ASSUMED]

### Pattern 1: One deep lifecycle-authority seam

The authority interface should expose complete state transitions, not raw table CRUD, locks, receipts, or checkpoint fragments. A concrete adapter must make each method atomic according to its contract.

Proposed interface responsibilities:

| Method family | Complete responsibility |
|---------------|-------------------------|
| `read_entry(key)` | Return the one committed/tombstoned canonical entry snapshot and its authority revision. |
| `prepare_mutation(spec)` | In one transaction validate expectations and persist bounded authenticated intent, exact candidate/prior locators, and operation identity before managed payload publication. |
| `record_candidate_verified(op_id, proof)` | Persist exact digest/size verification without changing normal-read visibility. |
| `promote_mutation(op_id, manifest)` | Revalidate expected generation and exact prior manifest digest; atomically replace entry state, mark operation promoted, create cleanup debt, mark projections dirty, and advance authority revision. |
| `abort_mutation(op_id, residue)` | Atomically retain or create exact candidate cleanup debt; never erase the prior committed entry. |
| `snapshot_clear()` | In one transaction create a clear run and exact target rows from the committed entry snapshot. |
| `page_clear_targets(clear_id, after_id, limits)` | Keyset-page immutable target rows under explicit work/action/byte limits. |
| `record_cleanup_result(...)` | Revalidate bindings and atomically retire debt or record bounded failure information. |
| `start/page/checkpoint_reconciliation(...)` | Capture high-water bounds, return stable indexed work, and checkpoint applied actions. |
| `close()` | Release adapter-owned resources idempotently without deleting stored state. |

Equivalent method grouping is acceptable, but callers must not be able to compose a partially atomic authority transition themselves.

### Pattern 2: Normalized authority state

Use normalized bounded rows. Exact schema names and encodings are discretionary, but the following ownership must remain inside one database and one transaction boundary:

| Logical table | Required contents | Required indexes/invariants |
|---------------|-------------------|-----------------------------|
| Authority metadata | Store identity, application/schema version, monotonically increasing authority revision | Singleton row; unknown newer version fails closed. |
| Key state / entries | Logical key, monotonically advancing per-key state version, lifecycle state, generation, exact canonical manifest bytes when committed/tombstoned, manifest digest, revision, timestamps | Primary key on logical key; decoded indexed fields must agree with authenticated manifest. A compact absent lineage marker prevents create/delete/create ABA after tombstone retirement. |
| Mutations | Monotonic work ID, unique operation ID, key, operation kind, expected generation/digest, candidate generation/locator, prior locator, bounded phase/proof/error fields | Unique operation ID; indexes on phase/work ID and key; check constraints on kinds/phases. |
| Cleanup debt | Monotonic debt ID, operation/key/generation/locator/role, state, bounded attempts/error | Index on state plus debt ID; unique logical debt identity where possible. |
| Clear runs | Clear ID, state, snapshot revision, progress counters | Index on active state. |
| Clear targets | Monotonic target ID, clear ID, key, expected generation and manifest digest, state/error | Unique clear/key; index on clear/state/target ID. |
| Reconciliation runs | Run ID, mode, immutable high-water IDs, cursor IDs, action counts/state | Resume by run ID; cursors never point into filename order. |
| Projection state | Projection name, exported authority revision, dirty bit, bounded last error | Dirty is set in the same transaction as entry change. |

The proposed row/table names and state labels are [ASSUMED]; the ownership and atomicity are locked by D-24 and D-29. Store canonical bytes once rather than maintaining independently editable manifest columns; indexed columns are corroborating query fields, not a second canonical record. Preserve a monotonic per-key state token even after a tombstone stops being externally visible. Otherwise an operation prepared against an ancient absence could promote after an intervening create/delete cycle (the ABA problem). The compact lineage row is bounded to one row per key rather than one row per historical operation.

### Pattern 3: Put/overwrite state machine

```text
PRIVATE_STAGE
  handler serializes in an OS-private temporary area
  failure => no authority row, no managed payload
        |
        v
PREPARED (short authority transaction)
  validate expected per-key state token/generation/digest
  commit exact op ID and candidate locator bound to op ID + generation
        |
        v
CANDIDATE_PUBLISHED (outside authority transaction)
  create immutable generation exclusively and durably
        |
        v
CANDIDATE_VERIFIED (payload verification outside transaction;
                    proof recorded in short transaction)
        |
        v
PROMOTED (short authority transaction)
  exact expected-generation + prior-manifest-digest CAS
  replace committed manifest
  create prior-payload cleanup debt
  mark JSON projection dirty
        |
        v
CLEANUP (outside authority transaction)
  revalidate debt binding, idempotently remove prior payload
        |
        v
RETIRED (short authority transaction)
  retire cleanup debt and terminal operation evidence
```

The existing handler staging path uses an OS temporary directory and deliberately avoids managed-store effects before publication. [VERIFIED: `src/cacheness/storage/guarded_handler_io.py:149-170`; quote: `tempfile.TemporaryDirectory(prefix="cacheness-handler-")`] Therefore D-08 and D-25 are compatible: private serialization may precede intent, but the first **managed persistent payload publication** must occur only after the PREPARED transaction commits. The existing immutable publication primitive uses exclusive creation and durability operations. [VERIFIED: `src/cacheness/storage/path_security.py:1984-2046`; quote: `create_stream_durable_exclusive`]

The promotion CAS must compare the expected per-key state token, generation, and digest of exact canonical prior bytes. Generation alone is insufficient if malformed/substituted authority content reuses a generation identifier, and bare “absence” is insufficient across create/delete/create ABA. Candidate locators must be bound to the operation ID and proposed generation and installed with exclusive creation; a collision is never overwritten.

### Pattern 4: Failure classification and convergence

| Failure/crash point | Authoritative entry | Indexed evidence | Deterministic recovery |
|---------------------|---------------------|------------------|------------------------|
| During private serialization | Previous/absent | None | Return serialization failure; OS temporary cleanup is outside store lifecycle. |
| After PREPARED, before candidate creation | Previous/absent | Prepared mutation | Confirm candidate absent; retire/abort operation. |
| During immutable candidate creation | Previous/absent | Prepared mutation with exact locator | Remove or quarantine the exact partial candidate after provenance/containment checks. |
| After candidate publication, before verification | Previous/absent | Prepared mutation and candidate | Verify then resume, or convert to candidate cleanup debt. |
| After verification, before promotion | Previous/absent | Verified mutation and candidate | Retry exact CAS or clean only that candidate on conflict. |
| During promotion commit | Old or new complete | SQLite journal plus mutation row | Reopen and inspect entry/operation; classify committed versus rolled back, never guess from exception text. |
| After promotion, before old cleanup | New complete | Cleanup-debt row created atomically with promotion | Delete the exact old locator after revalidation. |
| After physical cleanup, before debt retirement | New complete | Cleanup debt whose locator is already missing | Missing is successful idempotent cleanup; retire debt. |

An uncertain commit outcome must be resolved by reopening the authority and reading the operation/entry state. Do not retry promotion blindly.

### Pattern 5: Same-key CAS and distinct-key concurrency

SQLite permits only one simultaneous write transaction. [CITED: https://www.sqlite.org/lang_transaction.html] That constraint does not violate D-28. A short `BEGIN IMMEDIATE` transaction that performs indexed reads, comparisons, and row writes is the explicitly allowed **authority commit serialization**. The implementation violates D-28 if it holds that transaction—or any global/store/family lease—while running any of these:

- handler serialization or private staging;
- candidate publication, upload, or filesystem synchronization;
- digest/size verification;
- payload deletion or quarantine;
- JSON projection generation/publication;
- user callbacks, log handlers, or retry sleep.

Same-key ordering has two layers:

1. A bounded dynamically retained per-key lock may order threads in one process and reduce wasted candidates. The existing `KeyCoordinatorRegistry` is a separable local mechanism, whereas cross-process correctness comes from authority CAS. [VERIFIED: `src/cacheness/storage/coordination.py:639-778`; quote: `class KeyCoordinatorRegistry`]
2. Every process commits through expected-generation plus expected-prior-manifest-digest CAS. Two operations prepared from the same prior state can serialize and verify concurrently, but exactly one promotion transaction wins; the loser atomically records cleanup debt for only its own candidate and returns a typed conflict.

Do not hold the local key lock while blocked on SQLite writer acquisition if doing so can prevent another operation that owns the state needed for progress. Establish one documented acquisition order and test it.

**Measurable D-28 proof:** deterministic test hooks pause operation A for key A during payload publication, verification, and cleanup. Operation B for key B must reach committed success before A is released. Instrumentation must show `max_concurrent_payload_work >= 2`, `max_concurrent_authority_write_transactions == 1`, and `open_authority_write_transactions == 0` at every payload-I/O pause point. A wall-clock speedup alone is not proof.

### Pattern 6: Delete and cleanup debt

Delete uses the same state machine rather than a special scheduler:

1. Read and authenticate the current committed manifest.
2. Prepare an exact-generation delete intent.
3. In one short transaction, revalidate the generation/digest, publish the signed tombstone as the canonical entry, mark the operation promoted, create payload cleanup debt, mark projections dirty, and advance revision.
4. Outside the transaction, re-read/revalidate the debt's exact key, generation, and locator binding, then delete idempotently.
5. In a short transaction, retire debt and convert the tombstone to a compact absent lineage marker only if it is still the same tombstone. Do not erase the per-key state token.

A repeated delete of an externally absent key preserves compatible success/false behavior. A repeated delete that finds its own tombstone resumes cleanup. A delete that finds a newer generation conflicts and never removes the newer payload. If two workers claim the same cleanup debt, duplicate physical delete is safe only after each revalidates that the locator is not current/live; a missing locator counts as success.

### Pattern 7: Transactional clear snapshot

Clear must not use a file inventory or hold a global barrier across the entire deletion pass.

Recommended sequence:

1. One short authority transaction creates the clear run and executes an indexed `INSERT ... SELECT` equivalent that records every currently committed key with its exact generation and manifest digest.
2. Commit. Keys created or promoted after that snapshot are not members of the clear.
3. Workers keyset-page target rows and invoke the same exact-generation delete transition for each target.
4. A target whose generation/digest changed is marked conflict/skipped; it is never deleted speculatively.
5. Each successful target and the aggregate counters are checkpointed transactionally. Completion is set only when no bounded target state remains.

This may briefly serialize the snapshot transaction with other authority writers, but it does not need the discarded `StoreAdmissionBarrier` or an external key inventory. A large store may make one `INSERT ... SELECT` transaction expensive; the acceptance benchmark must measure snapshot duration and writer wait. If it exceeds the explicit budget, the follow-up design must use a revisioned snapshot strategy that remains exact—never a weak filename scan. [ASSUMED]

### Pattern 8: Stable bounded reconciliation

Reconciliation works only from indexed authority rows and exact locators already recorded there:

- At run start, capture immutable high-water IDs for mutations and cleanup debt in a short transaction. Persist them with the run for apply mode.
- Page with monotonic keyset predicates such as “work ID greater than cursor and less than or equal to high-water,” ordered by the indexed ID. Do not use mutable-state offset pagination.
- Enforce independent limits for inspected rows, proposed/applied actions, and payload bytes. Stop before a limit is exceeded and return a stable resume handle.
- Dry-run produces findings without creating destructive action checkpoints. Apply mode persists completed action/checkpoint state in the authority transaction that records the action outcome.
- Before every destructive action, reload the exact row and current entry and revalidate operation provenance, generation, manifest digest, containment, and locator non-ownership by a live generation.
- Work created after the captured high-water belongs to the next run. A row already retired when resumed counts as completed; it must not cause cursor rewind or livelock.
- Never deserialize a payload to establish lifecycle ownership and never enumerate filenames to discover operation state.

Suggested report fields follow D-13: stable finding ID, authority/run revision, operation ID, key, expected/authoritative generation, exact residue role/locator, proposed action, reason code, disposition (`safe`, `blocked`, or `confirmation_required`), and applied/checkpoint state. Exact reason values are public-contract decisions and must be characterized before renaming; proposed values are [ASSUMED].

### Pattern 9: JSON as a projection, not an authority

All direct `BlobStore` reads, writes, metadata reads, listing, clear, and recovery must use the lifecycle authority. JSON remains a supported compatibility representation, but it cannot independently decide cross-process publication.

Projection algorithm:

1. The promoting transaction sets the JSON projection row dirty and advances the authority revision.
2. An exporter creates a mode-restricted process-private temporary SQLite destination and uses SQLite's online backup API to capture a consistent authority snapshot without copying live database files directly. It closes the live source connection and reads revision `R` from the snapshot. [CITED: https://docs.python.org/3.11/library/sqlite3.html#sqlite3.Connection.backup]
3. It keyset-pages committed rows from the private snapshot, streams and atomically replaces the JSON projection, and removes the private snapshot. Rendering therefore holds no transaction on the live authority and does not require all manifests in memory.
4. A short authority transaction marks the projection clean only if the current authority revision still equals `R`. If not, it remains dirty so a newer export will rebuild it.

This final revision check is essential: without it, an older exporter could overwrite a newer JSON snapshot and falsely mark it current. Projection write failure does not roll back an already committed lifecycle mutation; it leaves a dirty/error marker and a rebuild action. A missing/corrupt projection is rebuilt from authority. A missing/corrupt authority is **not** reconstructed automatically from JSON because doing so would promote a derived representation into an unproven transaction source. The private backup is projection staging, not lifecycle evidence; it must use restrictive permissions and contain no payload bytes.

Do not incrementally patch JSON as part of the authority transaction, and do not resurrect cross-process JSON lock/CAS files. If a public API returns JSON-shaped metadata, preserve its shape through an adapter over authority rows and characterization tests.

### Pattern 10: In-memory adapter semantics

`InMemoryLifecycleAuthority` should implement the exact same complete-transition contract with a private lock around each authority method and immutable/copy-on-read records. It should provide:

- same-process deterministic CAS and revision behavior;
- identical operation/debt/clear/reconciliation states and typed outcomes;
- atomic state changes visible to threads in that Python process;
- no durable crash-recovery or cross-process claim;
- explicit capability rejection when durable or multiprocess guarantees are requested;
- idempotent `close()` that marks the instance closed but does not imply persistence.

Separate in-memory authority instances do not implicitly share state; callers that want multiple `BlobStore` instances to coordinate in memory must inject the same authority instance. [ASSUMED] `BlobStore.close()` closes only an authority it owns; a caller-injected authority remains caller-owned, consistent with D-12. Tests must run the common interface contract against both adapters. Adapter-specific tests may verify SQLite durability/reopen and memory ephemerality, but the lifecycle state-machine tests should not branch on storage internals.

### Pattern 11: Future authority adapters without pre-implementation

Phase 3 defines semantic capabilities, not concrete Phase 4/5 backend plumbing:

| Future adapter | Required semantic mapping | Deferred work |
|----------------|---------------------------|---------------|
| PostgreSQL | Transactions and row-level/unique-constraint CAS implement complete authority methods; indexed sequence IDs page work. | Connection/config registry, real-service matrix, isolation/deadlock tuning, migrations, and operational tests belong to Phases 4/5. |
| S3-native | Conditional object operations must implement equivalent complete-state publication, or a capable metadata authority must be paired with S3 payloads. | ETag/versioning semantics, listing consistency claims, multipart cleanup, compatible-service scope, and real AWS tests belong to Phases 4/5. |

The interface must describe invariants and outcomes (`prepared`, `promoted`, conflict, cleanup debt, clear snapshot, stable page), not SQLite concepts such as SQL text, row IDs as public identities, busy pragmas, or filesystem paths. Conversely, Phase 3 must not add receipt files inside a future adapter to simulate transactions.

### Anti-Patterns to Avoid

- **Dual authorities:** Keeping canonical JSON and SQLite independently writable recreates split-brain metadata.
- **CRUD-shaped authority interface:** Exposing `put_receipt`, `append_inventory`, or raw table methods lets callers rebuild partial transactions.
- **Transaction across payload I/O:** This serializes unrelated keys, increases busy failures, and makes external I/O part of a database lock lifetime.
- **Filename-driven recovery:** A directory scan cannot prove operation ownership, stable bounded progress, or whether an unindexed temporary is safe to delete.
- **Mutable offset pagination:** Updates/deletes between pages cause skip/replay; use high-water plus keyset cursors.
- **Blind retry after commit error:** The commit may already have succeeded; reopen and classify by operation/entry state.
- **WAL by assumption:** The bundled SQLite version and topology are runtime properties; current checked runtimes fall in the documented WAL-reset affected range.
- **Inherited post-fork authority:** Connections and locks belong to their creating process.
- **Tests that name receipt files:** These ossify the rejected mechanism rather than verify lifecycle behavior.

## Don't Hand-Roll

| Problem | Don't build | Use instead | Why |
|---------|-------------|-------------|-----|
| Cross-process atomic commit | Receipt/head/anchor/pending files and custom lock authority | SQLite transactions | SQLite already provides journaled atomic transactions and crash rollback. [CITED: https://sqlite.org/atomiccommit.html] |
| Database journaling | Custom replay log or manipulation of SQLite journal sidecars | SQLite's rollback journal | Touching/copying live journal files can corrupt or misclassify database state. [CITED: https://www.sqlite.org/howtocorrupt.html] |
| Compare-and-swap | File identity/inode/handle registries | Conditional row update inside one authority transaction | CAS belongs with the canonical entry and operation/debt changes. |
| Stable work enumeration | File inventories, anchors, cursors, receipts | Indexed rows, monotonic IDs, captured high-water, keyset pagination | Database indexes give stable bounded traversal without a second history protocol. |
| Migration transaction | Ad hoc file-copy switching of a live DB | Ordered schema migration transaction plus supported backup/quiesce procedure | The authority schema must never appear partially migrated. |
| Payload framing | Cacheness wrapper/header around handler output | Native NumPy/Blosc2/Parquet/pickle/dill formats plus separate manifest | Locked project intent keeps handlers responsible for payload format. |

**Key insight:** SQLite is not being used as a cache of the filesystem protocol; it replaces that protocol as the single local transactional authority. Payloads remain outside the database, so this is not turning payload storage into a filesystem database.

## Runtime State Inventory

This is a replacement/refactor phase, so repository edits alone are insufficient. The canonical question is: after file-native code is deleted, what runtime state still contains or depends on that protocol?

| Category | Items found | Action required |
|----------|-------------|-----------------|
| Stored data | Existing stores may contain canonical Phase 2 manifests/metadata plus internal file-native operation, inventory, anchor, receipt, pending-control, clear, and reconciliation artifacts. Exact released prevalence is not established in this research. [ASSUMED] | Do not silently import or delete. Detect exact known legacy control roots/sentinels and return a typed migration/rebuild-required outcome. Phase 7 owns non-mutating inventory and copy-verify-switch. For explicitly disposable development stores, document a rebuild path. |
| Live service config | No external service configuration is required by the local SQLite adapter. PostgreSQL/S3 service state remains deferred. | None for Phase 3; planner must avoid introducing external configuration. |
| OS-registered state | No systemd, launchd, Task Scheduler, or service registration is part of the current Python library architecture. [VERIFIED: `AGENTS.md:90-95`; quote: `the library has no web server, worker runtime, container definition, or hosting manifest`] | None. Local database and payload roots remain process-opened resources. |
| Secrets/env vars | Existing manifest-signing key/config remains security-critical. No repository environment-variable parser exists. [VERIFIED: `AGENTS.md:80-85`; quote: `no environment-variable parser is implemented in src/cacheness/`] | Reuse the existing signer/key path; do not invent an authority secret namespace. Schema rows containing recovery evidence must either authenticate bounded provenance or be strictly derived/corroborated from signed manifests. |
| Build artifacts / installed packages | Installed wheels may retain the abandoned modules after source deletion until reinstalled; `__pycache__` may retain bytecode but is not an authority. [ASSUMED] | Rebuild/reinstall the package for validation. Do not scan or execute stale artifacts as recovery state. |

### Authority schema migration versus stored-format migration

Keep two migration classes separate:

- **Authority database schema migration (Phase 3):** configure/verify journal mode, validate `application_id` and `user_version`, reject a wrong application ID or unknown newer version, apply ordered migrations in one bounded `BEGIN EXCLUSIVE` transaction, reopen, and run integrity/foreign-key checks. Concurrent initializers obey the same absolute busy deadline and then re-read the winning schema. Never expose a partially migrated schema. Exact application ID and schema version are implementation constants and remain [ASSUMED] until defined in source.
- **Existing cache/store adoption (Phase 7):** inventory canonical manifests and legacy control state without mutation, produce a machine/human plan, then copy-verify-switch or explicit rebuild. Do not infer a correct SQLite authority state by scanning arbitrary filenames during normal Phase 3 recovery.

Do not copy a live database file as a backup. Quiesce/close it or use a supported SQLite backup mechanism, and never manually copy the database without any live rollback journal or WAL sidecars. [CITED: https://www.sqlite.org/howtocorrupt.html]

## Project Constraints (from AGENTS.md)

- Preserve supported public APIs; stored-data migration/rebuild requires an explicit documented path.
- `BlobStore` owns storage lifecycle; `UnifiedCache` later consumes it as cache-policy layer; `SqlCache` remains separate.
- The eventual unified lifecycle covers filesystem, memory, S3, JSON, SQLite, and PostgreSQL backends, while this phase implements only the local SQLite and in-memory authority adapters.
- Treat application payloads as trusted while enforcing safe parsing, path containment, and fail-closed integrity boundaries.
- Payload and metadata operations require atomic commit, rollback, or deterministic reconciliation.
- Same-key operations must not corrupt payloads or create payload/metadata disagreement.
- Correctness precedes performance; acceptance still needs checked-in measured budgets.
- Maintain Python 3.11+ and verify supported versions rather than testing only Python 3.13.
- Use domain exceptions, preserve causes with `raise ... from e`, and avoid broad exceptions that turn corruption/backend failure into a cache miss.
- Keep new functions focused and do not grow the already-large orchestration modules.
- Keep optional dependency/runtime checks aligned with packaging extras; no new external dependency is required here.
- Run targeted pytest and Ruff checks without treating the repository-wide existing Ruff baseline as a clean invariant.

These constraints are extracted from the supplied `AGENTS.md` project/stack/conventions sections. [VERIFIED: `AGENTS.md:13-22`, `AGENTS.md:119-175`; quoted directives include `Preserve supported public APIs`, `BlobStore owns storage lifecycle`, `Correctness comes first during migration`, and `Maintain Python 3.11+ support`]

## Deletion and Replacement Map

The current scheduler is distributed across eight large modules; deletion should be driven by responsibility, not by mechanically preserving all module names.

| Current mechanism | Evidence on current main | Replacement action |
|-------------------|--------------------------|--------------------|
| `operation_repository.py` / `FileOperationRecordRepository` | The file is 4,014 lines and defines `class FileOperationRecordRepository`. [VERIFIED: `src/cacheness/storage/operation_repository.py:299-4014`] | Delete the file-native repository: receipts, inventory events/heads/tails, anchors, compaction, pending controls, clear target pages, and file reconciliation cursors. Replace with normalized rows behind `LifecycleAuthority`; do not retain a compatibility wrapper around this repository. |
| `operation_record.py` | The file is 1,323 lines. [VERIFIED: `src/cacheness/storage/operation_record.py:1-1323`] | Delete bespoke signed receipt/page/checkpoint schemas. Reintroduce only small private domain dataclasses/enums needed by the authority contract, with bounded safe row decoders. |
| `coordination.py` | Contains `StoreAdmissionBarrier`, `KeyCoordinatorRegistry`, and `InstanceAdmission`. [VERIFIED: `src/cacheness/storage/coordination.py:354-837`; quote: `class StoreAdmissionBarrier`, `class KeyCoordinatorRegistry`, `class InstanceAdmission`] | Delete interprocess file locks and `StoreAdmissionBarrier`. Retain/extract bounded in-process `KeyCoordinatorRegistry` and `InstanceAdmission` behavior for per-key ordering and close ownership. |
| `clear_recovery.py` | Contains `ClearRecoveryCoordinator` and `LegacyClearEvidenceAdapter`. [VERIFIED: `src/cacheness/storage/clear_recovery.py:88-913`; quote: `class ClearRecoveryCoordinator`, `class LegacyClearEvidenceAdapter`] | Delete bounded JSON journals and lock/admission orchestration. Preserve a read-only legacy detector only if release/migration evidence proves it is required; otherwise report exact legacy sentinel presence as migration-required. |
| Scheduler portions of `path_security.py` | The module also contains retained payload primitives `create_stream_durable_exclusive` and `delete_durable`. [VERIFIED: `src/cacheness/storage/path_security.py:1984-2048`; quote: `def create_stream_durable_exclusive`, `def delete_durable`] | Delete platform lock-file/inode/handle/xattr reservation, pending-control promotion, and scheduler control staging. Retain containment, safe open/read, immutable exclusive payload publication, fsync, and durable deletion. |
| Scheduler portions of `manifest_repository.py` | Current lifecycle separates manifest repository from file operation repository. [VERIFIED: `src/cacheness/storage/lifecycle.py:36-65`; quote: `FileOperationRecordRepository`] | Move canonical committed manifest ownership into the authority transaction. Keep only compatibility projection/import adapters required by public behavior; do not run JSON/SQLite manifest inventory alongside authority rows. |
| `reconciliation.py` | Current file is 1,854 lines. [VERIFIED: `src/cacheness/storage/reconciliation.py:1-1854`] | Replace scheduler/sidecar/pending-control multiplexer with a thin report/orchestration layer over authority pages. |
| `lifecycle.py` | `LifecycleEngine` currently constructs `FileOperationRecordRepository`. [VERIFIED: `src/cacheness/storage/lifecycle.py:47-65`; quote: `class LifecycleEngine`, `self.operation_repository = FileOperationRecordRepository`] | Rewrite as a thin state-machine coordinator that calls complete authority operations around payload I/O. No raw SQL or scheduler file manipulation. |
| `blob_store.py` composition | Imports `StoreAdmissionBarrier` and acquires it for the root. [VERIFIED: `src/cacheness/storage/blob_store.py:63-90`, `src/cacheness/storage/blob_store.py:280-292`; quote: `StoreAdmissionBarrier.acquire`] | Inject/select a lifecycle authority; remove store-root barrier composition; preserve instance admission, local key coordination, handler I/O, and caller-owned resource rules. |

### Test retirement map

Delete or rewrite tests whose subject is receipt/inventory/anchor/pending-control/lock-file structure. Current examples include direct imports of `FileOperationRecordRepository`, private `_append_inventory_event`, `StoreAdmissionBarrier`, and scheduler receipt pause hooks. [VERIFIED: `tests/test_manifest_repository_cas.py:38-75`, `tests/test_blob_store_concurrency.py:19-20`, `tests/test_blob_store_concurrency.py:464-562`, `tests/test_blob_store_close_contract.py:25-27`]

Preserve behavior cases while changing their fixture seam:

| Historical test concern | New test surface |
|-------------------------|------------------|
| Old-or-new put visibility | `BlobStore` plus common authority contract and payload fault injection. |
| Same-key writer/delete races | Expected-generation CAS results through public lifecycle operations. |
| Distinct-key progress | Deterministic payload-stage hooks and authority transaction counters. |
| Clear snapshot excludes later writes | Transactional clear target membership through `BlobStore.clear`. |
| Close drains/blocks instance work | Instance admission and authority resource ownership, not repository lock handles. |
| Recovery boundedness/resume | Authority high-water/keyset pages and stable reports, not receipt filenames. |
| Windows behavior | Same SQLite adapter and same-process/session scope; no Win32 fallback protocol expectations. |

## Common Pitfalls

### Pitfall 1: Recreating the scheduler inside SQLite

**What goes wrong:** Tables mirror event heads, receipt chains, anchors, and pending files instead of representing domain state.

**Why it happens:** A mechanical port preserves implementation artifacts rather than invariants.

**How to avoid:** Design complete authority methods and normalized current/work/debt rows first. There is no event-replay requirement.

**Warning signs:** `append_*`, `head`, `tail`, `receipt`, `pending_control`, or file-identity concepts appear in the authority API.

### Pitfall 2: Holding a writer transaction across payload I/O

**What goes wrong:** A slow serializer, fsync, upload, verification, or delete blocks all authority writers and defeats distinct-key concurrency.

**How to avoid:** Commit intent, close transaction, perform payload work, then reopen for short verification/promotion/debt transitions.

**Warning signs:** Any payload backend/handler call appears syntactically inside a transaction context.

### Pitfall 3: Ambiguous transaction outcome

**What goes wrong:** A commit raises after state may have become durable, and blind retry creates conflict or duplicate cleanup.

**How to avoid:** Every operation has a unique ID; reopen and classify the operation plus canonical entry.

**Warning signs:** Retry catches `OperationalError` around `commit()` without a state read.

### Pitfall 4: Projection false-clean race

**What goes wrong:** Exporter A captures revision 5, exporter B writes revision 6, then A overwrites JSON and marks it clean.

**How to avoid:** Mark clean only by compare-and-set on the captured authority revision; otherwise remain dirty.

**Warning signs:** Projection success updates no expected revision.

### Pitfall 5: Unsafe cleanup from stale evidence

**What goes wrong:** Reconciliation deletes a locator now referenced by a newer committed generation.

**How to avoid:** Revalidate exact debt/op/key/generation/manifest digest and current locator non-ownership immediately before deletion.

**Warning signs:** Cleanup accepts only a pathname or checks only that a file exists.

### Pitfall 6: Unstable reconciliation pagination

**What goes wrong:** Mutable filters plus offset skip/repeat rows; newly inserted work prevents convergence.

**How to avoid:** Capture high-water IDs and keyset-page immutable monotonic IDs; persist apply checkpoints.

**Warning signs:** `OFFSET`, lexical filename cursors, or a moving “all pending” scan.

### Pitfall 7: Treating timeout as correctness

**What goes wrong:** Different machines produce nondeterministic races and false test confidence.

**How to avoid:** Inject barriers/events at transaction and payload boundaries; timeout only guards a hung test.

### Pitfall 8: Assuming local topology from a path

**What goes wrong:** SQLite authority is placed on NFS/SMB/FUSE with locking/durability semantics the adapter does not support.

**How to avoid:** State local-only capability explicitly and fail closed for declared remote topology; document that automatic detection is incomplete. [CITED: https://www.sqlite.org/useovernet.html]

## Instrumentation and Operational Evidence

Instrumentation should describe lifecycle semantics, not SQLite internals alone.

| Signal | Required dimensions | Purpose |
|--------|---------------------|---------|
| Mutation transition count | operation kind, from/to phase, result/reason; never raw key/payload | Detect stuck prepared/verified/promoted operations. |
| Authority transaction duration | adapter, method family, success/busy/rollback | Prove commits remain short and diagnose writer contention. |
| Authority busy/deadline count | method family, elapsed bucket | Separate expected bounded contention from systemic stalls. |
| Payload stage/publish/verify/cleanup duration | operation kind, backend family, result | Confirm payload work dominates outside authority transactions and overlaps across keys. |
| Cleanup debt age/count | residue role, state, retry bucket | Detect convergence failures before storage leaks grow. |
| Reconciliation progress | mode, scanned/actions/bytes, cursor/high-water age, disposition | Show bounded progress and resumability. |
| Clear progress | target count, completed/conflict/blocked, age | Identify exact incomplete clear runs. |
| Projection lag | projection name, authority revision minus exported revision, dirty age | Surface JSON compatibility lag without treating it as authority failure. |

For validation builds, add injectable observers around transaction begin/end and payload-stage begin/end. Production metrics may expose aggregate counters, but tests need deterministic event hooks. Never include arbitrary logical keys, locators, manifest bytes, exception reprs with credentials, or payload-derived metadata in logs by default.

Recommended initial budgets are [ASSUMED] until measured on checked-in benchmarks: authority write transactions should be millisecond-scale and exclude all payload I/O; every reconciliation call must obey configured action/row/byte limits; busy waits must never exceed the caller's absolute deadline. The planner should create a benchmark-baseline task rather than lock unmeasured numeric latency promises.

## Security Domain

Security enforcement is enabled in `.planning/config.json`; Phase 3 therefore requires explicit input, file, integrity, and logging controls. [VERIFIED: `.planning/config.json:20-49`; quote: `"security_enforcement": true`, `"security_asvs_level": 1`]

### Applicable ASVS Categories

| ASVS category | Applies | Standard control |
|---------------|---------|------------------|
| V2 Authentication | No user authentication surface | Store signing/authenticity keys remain configuration credentials, not user authentication. |
| V3 Session Management | No web session surface | Process/connection ownership and post-fork rejection are lifecycle controls, not sessions. |
| V4 Access Control | Yes at local resource boundary | Contained private store root, owner/session deployment scope, fail-closed topology and resource-type checks. |
| V5 Validation, Sanitization and Encoding | Yes | Bound SQL parameters, strict bounded row decoders, enum/version/length validation, canonical manifest verification before trusting locators. |
| V6 Stored Cryptography | Yes | Reuse established HMAC/canonical manifest signing; never invent custom cryptography. |
| V7 Error Handling and Logging | Yes | Typed conflict/integrity/backend/timeout outcomes, cause preservation, bounded sanitized logs. |
| V8 Data Protection | Yes | Never log payload or secrets; database and journals live under protected contained root. |
| V10 Malicious Code | Trusted application payload boundary | Reconciliation never deserializes payload to infer lifecycle ownership. |
| V12 Files and Resources | Yes | Existing path containment, exclusive immutable generation creation, no symlink escape, exact-locator delete. |
| V14 Configuration | Yes | Verify pragmas/capabilities, reject unsupported durable multiprocess configurations, version schema. |

### Known threat and failure patterns

| Pattern | STRIDE / reliability class | Required mitigation |
|---------|----------------------------|---------------------|
| SQL injection via key/error/report fields | Tampering | Parameter binding only; never interpolate identifiers or values from metadata. |
| Oversized/malformed authority rows | Denial of service / tampering | Column/check constraints plus bounded decode before allocation; unknown versions fail closed. |
| Manifest/row disagreement | Tampering | Authenticate canonical manifest first; corroborate indexed generation/state/locator/digest columns exactly. |
| Locator traversal/symlink substitution | Tampering/elevation | Retain Phase 1 contained file operations and exact ownership revalidation immediately before deletion. |
| Deleting a newly reused locator | Integrity/destruction | Immutable generation-specific locators and current-entry non-ownership check; no loose glob cleanup. |
| Database replacement or wrong store | Spoofing/tampering | Contained regular-file checks, store identity, SQLite `application_id`, schema version, and deployment ACL assumptions. |
| SQLite sidecar manipulation | Tampering/corruption | Never open, scan, move, or delete SQLite journal files as Cacheness protocol artifacts. |
| Network-filesystem lock failure | Tampering/corruption | Local-filesystem-only capability; fail typed for unsupported declared topology. [CITED: https://www.sqlite.org/useovernet.html] |
| WAL-reset bug in affected SQLite | Corruption | Phase 3 default rollback `DELETE`, not WAL. [CITED: https://www.sqlite.org/wal.html] |
| Post-fork inherited connection/lock | Corruption/deadlock | PID ownership check; child constructs a fresh store/authority. [CITED: https://www.sqlite.org/howtocorrupt.html#_carrying_an_open_database_connection_across_a_fork_] |
| Cleanup/reconciliation payload deserialization | Code execution | Decide ownership only from authenticated authority/manifest evidence; trusted payload is read only by normal handler path. |
| Sensitive errors in operation rows | Information disclosure | Store bounded reason codes and sanitized text; no credentials, signed URLs, or arbitrary exception repr. |

### Connection and database resource hardening

- Place the authority database in the already-contained private store root; validate parent and target resource type before opening.
- Treat database, rollback journal, and temporary SQLite-created files as one SQLite-owned resource family. Cacheness never directly reconciles its sidecars.
- Configure pragmas on every new connection and verify values returned by pragmas where SQLite can silently choose another mode.
- Set `trusted_schema=OFF` and do not register application-defined SQL functions/collations needed by schema objects.
- Bound key, locator, manifest, digest, error, and report fields before database write and after database read.
- Never dynamically construct table/column names from untrusted metadata. Ordered migrations use constants embedded in code.
- On integrity-check failure, stop lifecycle mutation and preserve the entire authority family for diagnosis; do not rebuild automatically from JSON.

## Code Examples

The following are proposed patterns, not existing source. Names, enum values, table names, application ID, and version constants are [ASSUMED] until the planner defines them and implementation tests lock them.

### Explicit short write transaction

```python
# Proposed pattern. API names/constants are [ASSUMED].
def _write_transaction(self, deadline, body):
    connection = self._open_connection(deadline)
    try:
        connection.execute("BEGIN IMMEDIATE")
        result = body(connection)  # indexed reads/comparisons/row writes only
        connection.execute("COMMIT")
        return result
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
```

Python documents `isolation_level=None` for explicit transaction control. [CITED: https://docs.python.org/3.11/library/sqlite3.html#transaction-control] Payload I/O must never be passed as `body`.

### Conditional promotion as one authority operation

```python
# Proposed interface shape; names/outcomes are [ASSUMED].
promotion = authority.promote_mutation(
    operation_id=prepared.operation_id,
    expected_generation=prepared.expected_generation,
    expected_manifest_digest=prepared.expected_manifest_digest,
    verified_manifest=new_manifest,
)
if promotion.conflict:
    raise CacheConflictError(promotion.reason)
```

The implementation transaction must load the prepared operation, compare the exact current entry, replace the canonical manifest, create prior-payload cleanup debt, advance authority revision, and mark the projection dirty before commit.

### Stable reconciliation page

```sql
-- Proposed SQLite adapter query; table/column/state names are [ASSUMED].
SELECT debt_id, operation_id, key, generation, locator, role
FROM cleanup_debt
WHERE state = ? AND debt_id > ? AND debt_id <= ?
ORDER BY debt_id
LIMIT ?
```

The resume cursor is the last processed monotonic ID, and the run's high-water is immutable. An index beginning with `(state, debt_id)` supports the filter, subject to query-plan verification.

## State of the Art

| Superseded approach | Recommended approach | Reason for change | Impact |
|---------------------|----------------------|-------------------|--------|
| File-native receipts, event inventories, heads/tails, anchors, pending controls, staged control files | One transactional lifecycle authority | D-23 rejects the scheduler after repeated non-convergence. | Delete protocol and internal tests; retain behavior contracts. |
| Cross-process lock/file identity authority | SQLite transaction and row CAS | Platform-specific identity and crash gaps multiplied state transitions. | Same local adapter on Windows/macOS/Linux under declared trust topology. |
| Filename scans/cursors for recovery | Indexed rows with high-water/keyset cursor | Stable bounded progress must not infer ownership. | Reconciliation becomes query/report orchestration. |
| JSON conditional publication as local process authority | SQLite canonical authority plus JSON projection | Independent JSON CAS cannot atomically combine entry, intent, debt, clear, and checkpoint state. | Projection failures become rebuildable lag, not split-brain lifecycle state. |
| Whole-clear admission protocol plus file target inventory | One transactional target snapshot and per-target exact CAS | Exact membership/progress belong in the authority. | New keys after snapshot survive without a long global lease. |

The final archived iteration reported that crash residue from `.cas.tmp` and `.pending.*.tmp` control staging could be unindexed while reconciliation reported clean, demonstrating that additional file protocol layers were not converging on D-25/D-29. [VERIFIED: `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.file-native-final.md:64-84`, `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.file-native-final.md:106-150`; quote: `reconciliation can consequently report a clean store while leaving managed control residue behind`]

## Environment Availability

| Dependency | Required by | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| CPython system | SQLite design/runtime probe | Yes | Python 3.12.1; SQLite 3.43.1; `sqlite3.threadsafety == 3` [VERIFIED: runtime probe 2026-09-04; quoted result: `Python 3.12.1`, `3.43.1 3`] | Project venv for tests. |
| Project virtualenv | Phase tests | Yes | Python 3.13.3; SQLite 3.47.1 [VERIFIED: runtime probe 2026-09-04; quoted result: `Python 3.13.3`, `3.47.1 3`] | Supported-version CI in Phase 8; Phase 3 should add local matrix where available. |
| pytest | Validation | Yes | 8.4.1 in project venv [VERIFIED: runtime probe 2026-09-04; quoted result: `pytest 8.4.1`] | None required. |
| `uv` | Repository workflow | Installed, but sandbox cache access was denied in this research session [VERIFIED: command probe 2026-09-04; quoted error: `Failed to initialize cache`] | Repository lock/tooling present | Direct `.venv/bin/python` and `.venv/bin/pytest` for local probes; normal project execution may use approved `uv run`. |
| Local writable filesystem | SQLite authority and filesystem payloads | Yes for repository/test temp roots | Platform-provided | In-memory authority for ephemeral same-process tests only. |
| PostgreSQL/S3 | Future adapters | Not required in Phase 3 | Deferred | None; do not pre-implement. |

Both inspected runtime SQLite versions fall in the official WAL advisory's affected range, reinforcing rollback-journal mode for Phase 3. [CITED: https://www.sqlite.org/wal.html]

**Missing dependencies with no fallback:** None for Phase 3.

**Missing dependencies with fallback:** `uv` cache access in the sandbox; the existing project venv remains usable for read-only version probes and targeted tests.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.4.1 |
| Config file | `pyproject.toml` (`[tool.pytest.ini_options]`) [VERIFIED: `pyproject.toml:83`] |
| Quick run command | `.venv/bin/pytest -q tests/test_lifecycle_authority_contract.py tests/test_sqlite_lifecycle_authority.py -x` |
| Full phase command | `.venv/bin/pytest -q tests/test_lifecycle_authority_contract.py tests/test_sqlite_lifecycle_authority.py tests/test_blob_store_atomic_lifecycle.py tests/test_blob_store_reconciliation.py tests/test_blob_store_concurrency.py tests/test_blob_store_close_contract.py -o log_cli=false` |
| Repository gate | `uv run pytest -q -o log_cli=false` plus `uv run ruff check src tests` under normal approved project tooling |

### Phase Requirements → Test Map

| Req ID | Behavior | Test type | Automated command | File exists? |
|--------|----------|-----------|-------------------|--------------|
| STOR-03 | Every fault point shows old or new complete generation only | Interface contract + subprocess crash integration | `.venv/bin/pytest -q tests/test_lifecycle_authority_contract.py tests/test_blob_store_atomic_lifecycle.py -x` | Contract file ❌ Wave 0; BlobStore file exists but requires rewrite |
| STOR-04 | Prepared/candidate/verified/promoted residue is exact and recoverable | Fault injection + reopen/crash | `.venv/bin/pytest -q tests/test_sqlite_lifecycle_authority.py tests/test_blob_store_reconciliation.py -x` | SQLite file ❌ Wave 0; reconciliation file exists but requires rewrite |
| STOR-05 | Overwrite/delete/clear/close resume idempotently | Interface + BlobStore behavior | `.venv/bin/pytest -q tests/test_blob_store_atomic_lifecycle.py tests/test_blob_store_close_contract.py -x` | Existing files require scheduler-decoupled rewrite |
| STOR-06 | Dry-run stable, apply resumable, bounded, destructive action revalidated | Interface + report integration | `.venv/bin/pytest -q tests/test_blob_store_reconciliation.py -x` | Exists but requires replacement internals |
| STOR-07 | Same-key one winner; distinct-key payload work overlaps; no transaction spans I/O | Deterministic thread/process concurrency | `.venv/bin/pytest -q tests/test_blob_store_concurrency.py -x` | Exists but receipt/barrier tests require replacement |

### Common authority-adapter contract

Run the same parametrized contract against memory and SQLite:

- absent/create CAS, overwrite CAS, delete/tombstone CAS, and conflict outcomes;
- prepared intent does not affect normal read;
- promotion atomically changes entry, mutation, cleanup debt, projection dirty state, and revision;
- rollback leaves all participating tables unchanged;
- operation IDs and calls are idempotent after uncertain outcomes;
- clear snapshot captures exact target generations and excludes later commits;
- high-water/keyset pages neither skip nor repeat eligible starting rows;
- apply checkpoints resume after interruption;
- malformed/oversized/unknown-version state fails typed and unchanged;
- close is idempotent and caller-owned resources are not closed.

### Deterministic crash matrix

Use spawned subprocesses that open their own authority/store, signal a parent at a named hook, and are terminated at these boundaries:

1. before intent commit;
2. after intent commit and before payload creation;
3. during candidate creation;
4. after candidate fsync and before verification;
5. after verification and before promotion;
6. inside promotion before commit;
7. immediately after promotion commit;
8. during old-payload cleanup;
9. after physical cleanup and before debt retirement;
10. during clear target processing and reconciliation checkpointing.

After each crash: reopen, run `PRAGMA integrity_check` and `PRAGMA foreign_key_check`, inspect normal read, run dry-run twice for stable report equality, apply with an injected interruption, resume, and assert convergence plus no deletion of the only valid generation.

### Deterministic concurrency tests

- Two same-key writes prepared from the same generation: release both promotion barriers; exactly one commits and one receives typed conflict; loser cleans only its candidate.
- ABA regression: prepare an absent-key create, execute an intervening create/delete to an absent lineage state, then prove the stale prepared operation cannot promote against the later absence.
- Write versus delete from the same generation: exactly one tombstone/new-generation promotion wins.
- Reader paused after manifest snapshot while writer promotes: reader returns one complete snapshot or the one bounded retry outcome, never mixed state.
- Distinct keys: pause A at each payload boundary; B must finish before A release. Assert `max_concurrent_payload_work >= 2`, `max_concurrent_authority_write_transactions == 1`, and no authority transaction open at pause.
- SQLite busy deadline: hold a writer transaction from a test-only connection; operation times out within bounded tolerance and returns the typed authority timeout with preserved cause.
- Cross-process: use `multiprocessing` spawn, not a fork-inherited store. Each worker constructs its own authority.
- Clear race: snapshot exact targets, then overwrite/create; clear never deletes post-snapshot generation/key.

Avoid timing-only races and sleeps as the synchronization mechanism. Events/barriers and fault hooks decide ordering; timeouts only fail hung tests.

### SQLite adapter-specific tests

- Create/reopen schema; verify application ID, user version, `journal_mode=delete`, `synchronous=extra`, foreign keys on, trusted schema off.
- Reject unknown newer schema and wrong-store/application identity without mutation.
- Roll back a multi-table promotion fault at each statement; assert no partial entry/debt/revision/projection state.
- Confirm database recovery after process termination during write transaction.
- Reject post-fork inherited authority by PID; fresh child authority succeeds where topology is supported.
- Verify `close()` releases owned connections and is idempotent.
- Exercise Windows/macOS/Linux in CI on supported Python versions; no platform-specific receipt/lock behavior is expected.

### Sampling Rate

- **Per task commit:** common authority contract plus the directly changed behavior file.
- **Per wave merge:** full phase command.
- **Phase gate:** full repository suite, targeted supported-Python/platform matrix, Ruff delta clean, and crash/concurrency suite green before `$gsd-verify-work`.

### Wave 0 Gaps

- [ ] `tests/test_lifecycle_authority_contract.py` — parametrized semantic contract for memory and SQLite.
- [ ] `tests/test_sqlite_lifecycle_authority.py` — pragmas, schema, migration, busy, recovery, close/fork.
- [ ] Shared deterministic authority/payload fault hooks and spawned-process helpers in `tests/conftest.py` or a focused private helper.
- [ ] Replace scheduler-internal cases in `tests/test_manifest_repository_cas.py`, `tests/test_blob_store_atomic_lifecycle.py`, `tests/test_blob_store_reconciliation.py`, `tests/test_blob_store_concurrency.py`, `tests/test_clear_recovery.py`, and `tests/test_blob_store_close_contract.py` with interface/behavior assertions.
- [ ] Add benchmark probes for authority transaction duration, clear snapshot duration, busy wait, and distinct-key payload overlap; establish budgets from measured baselines.

## Assumptions Log

| # | Claim | Section | Risk if wrong |
|---|-------|---------|---------------|
| A1 | Portable automatic network/userspace-filesystem detection is incomplete; explicit topology/capability declaration is needed. | SQLite connection policy | Unsupported topology might be accepted or a valid local topology rejected. |
| A2 | Proposed module, method, table, column, state, report-reason, application-ID, and schema-version names are not public locked values yet. | Architecture / code examples | Prematurely locking names could break compatibility or force needless migrations. |
| A3 | One-transaction `INSERT ... SELECT` clear snapshot will meet the measured store-size/writer-wait budget. | Transactional clear | Large clears could hold the sole SQLite writer too long; benchmark before locking budget. |
| A4 | Existing deployed stores may contain file-native iteration artifacts; their released prevalence is not established. | Runtime inventory / migration | Removing recognition without an explicit compatibility decision could strand a real user store. |
| A5 | Installed wheels/bytecode can retain removed Python modules until reinstall, but they are not lifecycle authority. | Runtime inventory | Test environment could accidentally import stale code and mask source deletion. |
| A6 | Initial numeric transaction-duration and busy-wait budgets must be derived from checked-in benchmarks. | Instrumentation | Arbitrary targets could be either meaningless or prohibitively strict. |
| A7 | Separate in-memory authority instances do not share state; multi-store coordination requires injection of the same instance. | In-memory adapter | A hidden global registry would change ownership, isolation, and close semantics. |

The planner should turn A3, A4, and public compatibility around report/error values into explicit Wave 0 characterization or benchmark tasks. Other assumed names remain implementation discretion and should be locked by the first authority-contract tests.

## Open Questions

1. **Was any file-native Phase 3 scheduler format released or used outside disposable development worktrees?**
   - What we know: current main contains the scheduler modules, and the final iteration-28 review identifies unrecoverable unindexed control residue. [VERIFIED: `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.file-native-final.md:64-84`]
   - What's unclear: whether user stores exist whose only recovery evidence is that internal protocol.
   - Recommendation: inventory package release history and supported stored formats in Wave 0. If never released, document dev-store rebuild and delete it. If released, add only an exact read-only detector that returns migration-required; do not resume the protocol.

2. **What exact public result/error/reconciliation fields are already compatibility-locked?**
   - What we know: STOR-08 and earlier phases protect typed outcomes, and D-13 requires stable machine/human reconciliation reports.
   - What's unclear: which scheduler-era reason strings were public versus internal.
   - Recommendation: characterize public calls and serialization shapes before naming new authority enum/reason values; map deprecated values at the boundary rather than leaking SQLite errors.

3. **What clear-snapshot writer-hold budget is acceptable at expected maximum local store size?**
   - What we know: exact clear membership must be transactional, and SQLite has one writer. [CITED: https://www.sqlite.org/lang_transaction.html]
   - What's unclear: target cardinality and acceptable p95/p99 writer wait.
   - Recommendation: implement the single-transaction snapshot first, benchmark realistic cardinalities, and set a checked-in threshold. Do not add a chunked weak snapshot preemptively.

4. **Where should the authority database live relative to existing metadata paths?**
   - What we know: it must be under a contained local root and must not be confused with JSON projection or payload files.
   - What's unclear: the compatibility-safe filename/config mapping.
   - Recommendation: select one reserved contained locator during planning, characterize collisions with released paths, and fail typed if an incompatible object occupies it. The proposed filename is intentionally not specified here.

## Sources

### Primary repository evidence (HIGH confidence)

- `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md` — locked D-01 through D-31, discretion, and deferrals.
- `.planning/REQUIREMENTS.md` — STOR-03 through STOR-07 exact requirement text.
- `.planning/phases/02-canonical-storage-and-integrity-contract/02-CONTEXT.md` — canonical manifest and committed-only read contract.
- `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-RESEARCH.file-native.md` — superseded architecture/history only.
- `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.file-native-final.md` and iteration archives — non-convergence/failure evidence.
- `src/cacheness/storage/lifecycle.py`, `operation_repository.py`, `operation_record.py`, `coordination.py`, `clear_recovery.py`, `path_security.py`, `manifest_repository.py`, `reconciliation.py`, and `blob_store.py` — current main integration and deletion seams.
- Targeted test files named in the deletion/validation sections — behavior worth preserving and scheduler coupling to remove.

### Official documentation (MEDIUM confidence in this research)

- [SQLite Transaction](https://www.sqlite.org/lang_transaction.html) — explicit transactions, one writer, `BEGIN IMMEDIATE`, busy behavior.
- [SQLite Write-Ahead Logging](https://www.sqlite.org/wal.html) — concurrency, same-host requirement, persistent mode, current WAL-reset bug advisory.
- [SQLite PRAGMA Statements](https://www.sqlite.org/pragma.html) — synchronous modes, busy timeout, foreign keys, trusted schema, integrity checks, application ID, user version.
- [SQLite Atomic Commit](https://sqlite.org/atomiccommit.html) — rollback-journal atomicity model and filesystem assumptions.
- [SQLite How To Corrupt](https://www.sqlite.org/howtocorrupt.html) — journal-family handling, network filesystems, and post-fork connections.
- [SQLite Over a Network](https://www.sqlite.org/useovernet.html) — local/server-process topology guidance and network locking/sync risks.
- [Python 3.11 `sqlite3`](https://docs.python.org/3.11/library/sqlite3.html) — connection timeout/thread behavior, explicit transaction control, connection context-manager and close behavior.

Context7 was selected by the research seam but was unavailable in this environment, so official documentation was read through web-search fallback. The seam classified `websearch` LOW; the document uses `[CITED]` rather than `[VERIFIED]` for those external facts and keeps design choices clearly identified as recommendations.

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH for “stdlib SQLite, no new dependency” because D-26 locks it; MEDIUM for exact durability configuration because it is derived from official SQLite documentation and current runtime probes.
- Architecture: HIGH because it follows locked D-23 through D-31 and the current-main integration/deletion audit.
- Failure model: HIGH for the old scheduler's failure evidence; MEDIUM for the proposed normalized state machine until implementation fault tests pass.
- Concurrency: HIGH for the contract and proof shape; MEDIUM for performance until benchmarked across supported platforms.
- Migration: MEDIUM because deployed file-native-format prevalence is unresolved and Phase 7 owns execution.
- Security: HIGH for retained Phase 1/2 boundaries; MEDIUM for SQLite resource hardening until platform matrix validation.

**Research date:** 2026-09-04
**Valid until:** 2026-10-04 for stable SQLite transaction fundamentals; recheck the WAL advisory and bundled runtime versions immediately before considering WAL.

## RESEARCH COMPLETE

Recommended plan decomposition:

1. **Wave 0 — contract and compatibility characterization:** lock public outcomes/report shapes, determine whether scheduler formats were released, create common authority test fixtures, deterministic fault hooks, and benchmark probes.
2. **Authority core — deep interface plus in-memory reference:** define bounded domain records and complete state transitions; prove CAS, atomic mutation/debt/clear/checkpoint ownership against the common contract.
3. **SQLite authority — durable local reference:** implement schema identity/migrations, rollback `DELETE` + `synchronous=EXTRA`, explicit short `BEGIN IMMEDIATE` transactions, bounded busy deadlines, integrity checks, local-only capability, and close/fork ownership.
4. **Lifecycle orchestration — native payload state machine:** rewire put/overwrite/delete around durable intent, immutable publish, verification, conditional promotion, cleanup debt, and retirement without a transaction across payload I/O.
5. **Clear, reconciliation, and JSON projection:** implement transactional clear targets, indexed high-water/keyset recovery, resumable apply checkpoints, stable reports, and revision-checked JSON rebuild/export.
6. **Delete the abandoned scheduler:** remove `operation_repository.py`, scheduler record/journal/lock/inventory mechanisms, platform receipt authority, and internal tests; retain only containment, immutable payload I/O, bounded local key/instance coordination, and any proven read-only migration detector.
7. **Cross-platform acceptance:** run deterministic crash/race tests on supported Python versions and Windows/macOS/Linux, prove D-28 overlap counters, establish transaction/clear/busy budgets, and gate on full regression plus lint delta.

This decomposition satisfies STOR-03 through STOR-07 without recreating a filesystem database: SQLite stores only small transactional lifecycle authority rows, while payloads remain immutable native handler outputs in their payload backend.
