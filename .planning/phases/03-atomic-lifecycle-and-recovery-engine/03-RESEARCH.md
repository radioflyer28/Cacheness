# Phase 3: Atomic Lifecycle and Recovery Engine - Research

**Researched:** 2026-08-30
**Domain:** backend-neutral atomic object lifecycle, crash recovery, reconciliation, and same-key concurrency
**Confidence:** HIGH for in-repository architecture and locked behavior; MEDIUM for external standards mapping

<user_constraints>
## User Constraints (from CONTEXT.md)

<!-- DATA_Q7M4Z2KP_START -->
### Locked Decisions

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

### the agent's Discretion

- Exact class/module names for lifecycle operations, journals, reconciliation reports, and coordination registries.
- Whether local per-key coordination uses lock striping or dynamically retained locks, provided unrelated keys remain concurrent and retention is bounded.
- Exact bounded retry/backoff values and operation-record encoding, provided failure outcomes remain deterministic and testable.
- Whether safe quarantine is implemented as a contained rename, backend namespace move, or immutable report-only disposition for a backend that cannot move atomically.

### Deferred Ideas (OUT OF SCOPE)

- Concrete capability declarations and full metadata-backend conditional publication across JSON, memory, SQLite, and PostgreSQL — Phase 4.
- Full filesystem, memory, and S3 payload lifecycle implementation and matrix verification — Phase 5.
- `UnifiedCache` delegation, TTL/eviction/invalidation policy, and miss/statistics translation — Phase 6.
- Stored-format inventory and copy-verify-switch migration execution — Phase 7; Phase 3 reconciliation handles lifecycle inconsistency, not format migration.
<!-- DATA_Q7M4Z2KP_END -->
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| STOR-03 | A write exposes either the previous complete generation or the new complete generation, never partial payload or metadata state. | Split private serialization from persistent candidate publication; use immutable generation locators and one expected-generation manifest CAS as the authority point. [VERIFIED: `.planning/REQUIREMENTS.md:8-12`; `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:18-31`] |
| STOR-04 | A failed write preserves the last valid generation and leaves any residue detectable and recoverable. | Persist bounded authenticated operation evidence before candidate publication, classify failures by whether manifest CAS succeeded, and retain cleanup debt until idempotent reclamation finishes. [VERIFIED: `.planning/REQUIREMENTS.md:10`; `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:33-41`] |
| STOR-05 | Overwrite, delete, clear, and close operations are idempotent and clean up both payload and metadata state. | Route mutations through one lifecycle engine; delete uses a CAS tombstone, clear uses a bounded generation snapshot, and close uses instance admission plus ownership-aware resource release. [VERIFIED: `.planning/REQUIREMENTS.md:11`; `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:43-50`] |
| STOR-06 | Operators can run dry-run and resumable reconciliation that detects inconsistent state and safely repairs, quarantines, or reports it. | Reconcile only authenticated manifests, validated operation records, and contained owned locators; revalidate immediately before apply and checkpoint every completed action. [VERIFIED: `.planning/REQUIREMENTS.md:12`; `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:52-61`] |
| STOR-07 | Same-key races have deterministic outcomes through per-key coordination and backend generation checks without globally serializing distinct keys. | Use dynamically retained per-key locks for local ordering, a short store-wide barrier only for clear/reconciliation admission, and backend CAS for the actual winner. [VERIFIED: `.planning/REQUIREMENTS.md:13`; `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:63-72`] |
</phase_requirements>

## Summary

Phase 3 should introduce one lifecycle engine beneath direct `BlobStore` mutations, not add transaction logic independently to `put`, `delete`, and `clear`. The engine's invariant is simple: immutable payload bytes are prepared first, then one signed canonical manifest is conditionally published; that successful CAS is the only authority transition. Everything before CAS is rollback/reconciliation work for an uncommitted candidate, and everything after CAS is forward-only cleanup debt for a committed winner. This is the only design that composes filesystem payloads with separately persisted metadata without pretending they share a distributed transaction. [VERIFIED: `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:18-41`; `.planning/ROADMAP.md:104-119`]

The most important implementation consequence is that `GuardedHandlerIO.put()` must be split. It currently serializes in a private stage and publishes into managed storage in one method, while the locked contract requires serialization failure to leave no record and requires durable operation evidence before the first managed-store side effect. The planned seam must therefore be: private handler serialization → durable operation record → exclusive immutable generation publication → candidate verification → manifest CAS → old-generation/tombstone reclamation → operation-record retirement. Native handlers still own NPZ, Parquet, pickle, dill, and Blosc2 payload bytes; no lifecycle header or wrapper belongs in the payload. [VERIFIED: `src/cacheness/storage/guarded_handler_io.py:301-334`; `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:33-41,109-111`; `.planning/phases/02-canonical-storage-and-integrity-contract/02-VERIFICATION.md:35-38`]

The existing Phase 1 clear coordinator is a pattern library, not the finished engine. Its strongest reusable ideas are durable prepared/committed evidence, strict bounded parsing, topology binding, deterministic admission, and pre/post-authority recovery. Its limitations are equally important: it is clear-only, snapshots whole backend state, uses one global admission lock for normal mutations, recognizes candidate ownership partly by filename grammar, and cannot produce a dry-run/resumable operator report. Absorb those ideas into the new operation/reconciliation model, then leave full S3/PostgreSQL and backend capability truthfulness to Phases 4-5. [VERIFIED: `src/cacheness/storage/clear_recovery.py:1-6,34-70,198-225,336-470,564-642`; `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:93-111,119-124`]

**Primary recommendation:** Build a small explicit lifecycle state machine around immutable generation locators, authenticated bounded operation records, and exact expected-generation CAS; make dry-run reconciliation and deterministic fault/race tests first-class consumers of that state machine. [VERIFIED: `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:18-72`]

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Lifecycle orchestration for write/overwrite/delete/clear/close | API / Backend (`BlobStore`) | Database / Storage | `BlobStore` owns payload-plus-metadata lifecycle; storage adapters perform only atomic primitives. [VERIFIED: `AGENTS.md:13-21`; `src/cacheness/storage/blob_store.py:178-206`] |
| Conditional authority publication | Database / Storage (manifest repository) | API / Backend | Backend-local CAS is the cross-instance correctness boundary; the engine supplies an authenticated expectation and signed replacement. [VERIFIED: `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:21,66`] |
| Native payload serialization and snapshots | Database / Storage (`GuardedHandlerIO` + handlers) | API / Backend | Handlers own formats; guarded I/O owns containment, private staging, exclusive generation publication, and one-snapshot reads. [VERIFIED: `src/cacheness/storage/guarded_handler_io.py:1-7,301-367`; `.planning/phases/02-canonical-storage-and-integrity-contract/02-VERIFICATION.md:35-38`] |
| Operation evidence and recovery decisions | API / Backend (lifecycle/reconciliation engine) | Database / Storage (operation repository) | The engine interprets authenticated manifests and operation progress; repositories persist exact bounded records without guessing intent. [VERIFIED: `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:33-41,52-61`] |
| In-process same-key ordering | API / Backend (coordination registry) | — | Local locks order work within an instance but cannot replace backend CAS across processes. [VERIFIED: `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:63-72`] |
| Store-wide clear/reconciliation admission | API / Backend | Database / Storage | A bounded snapshot/barrier protects aggregate setup while per-entry expected generations prevent deletion of later writes. [VERIFIED: `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:45,72`] |
| Cache policy, TTL, eviction, statistics | API / Backend (`UnifiedCache`) | — | Explicitly deferred to Phase 6; Phase 3 must not duplicate or rewire policy. [VERIFIED: `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:119-123`] |
| SQL pull-through cache | API / Backend (`SqlCache`) | Database / Storage | It remains a separate subsystem and is outside this lifecycle. [VERIFIED: `.planning/PROJECT.md:40-44`; `AGENTS.md:13-21`] |

## Project Constraints (from AGENTS.md)

- Preserve supported public APIs; stored data may change only through an explicit documented migration or rebuild path. [VERIFIED: `AGENTS.md:13-21`]
- Keep ownership layered: `BlobStore` owns storage lifecycle, `UnifiedCache` owns cache policy, and `SqlCache` remains separate. [VERIFIED: `AGENTS.md:13-21`]
- Design the lifecycle contract so filesystem, memory, S3, JSON, SQLite, and PostgreSQL can eventually implement it, while this phase does not claim the deferred backend matrix. [VERIFIED: `AGENTS.md:13-21`; `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:119-124`]
- Treat application payloads as trusted, but treat manifests, operation records, locators, and control metadata as untrusted; enforce safe parsing, containment, and fail-closed integrity. [VERIFIED: `AGENTS.md:13-21`; `docs/SECURITY.md:7-29`]
- Give payload/metadata operations atomic commit, rollback, or deterministic reconciliation semantics, and prevent same-key corruption without globally serializing unrelated keys. [VERIFIED: `AGENTS.md:13-21`]
- Correctness precedes performance during migration; final performance budgets come from checked-in benchmarks, not estimates in this phase. [VERIFIED: `AGENTS.md:13-21`]
- Maintain Python `>=3.11`; the checked development environment is Python 3.13, and supported versions must be verified rather than inferred from that one interpreter. The manifest says verbatim: `requires-python = ">=3.11"`. [VERIFIED: `pyproject.toml:1-13`; `AGENTS.md:29-41`]
- Use `snake_case.py`, `snake_case` callables, PascalCase classes/exceptions, focused abstract/protocol interfaces, four-space indentation, module/public docstrings, and Google-style `Args`/`Returns`/`Raises` where appropriate. [VERIFIED: `AGENTS.md:99-153`]
- Keep public re-exports in package `__init__.py`; use package-relative imports internally; guard optional dependencies explicitly. [VERIFIED: `AGENTS.md:99-165,188-199`]
- Raise domain-specific errors, preserve original causes with `raise ... from exc`, catch narrow operational failures, and make partial-success/recovery policy explicit. [VERIFIED: `AGENTS.md:166-176`]
- Use structured contextual logging at appropriate levels; tests that assert logs use `caplog`. [VERIFIED: `AGENTS.md:178-185`]
- Run targeted Ruff over changed source/tests and do not add to the repository's pre-existing lint baseline. [VERIFIED: `AGENTS.md:127-153`]
- Use pytest test names `test_<subject>.py`, specific `pytest.raises`, temporary local filesystem/SQLite fixtures, and explicit optional/integration markers. [VERIFIED: `AGENTS.md:99-105,166-176`; `pyproject.toml:82-111`]
- Do not edit protected `.planning/codebase/*.md` files or `.planning/milestone.lock`; preserve unrelated dirty-worktree changes. [VERIFIED: orchestrator assignment; `git status --short` showed pre-existing `.planning/codebase/*.md` modifications during this research]

## Standard Stack

### Core

| Library / Module | Version | Purpose | Why Standard |
|------------------|---------|---------|--------------|
| Python standard library (`dataclasses`, `enum`, `threading`, `contextlib`, `hashlib`, `hmac`, `uuid`, `pathlib`) | Python `>=3.11`; uv environment 3.13.3 | Immutable records, typed states, per-key coordination, deterministic contexts, cryptographic provenance, opaque IDs | Already the project runtime and sufficient for the lifecycle engine; no new runtime dependency is needed. [VERIFIED: `pyproject.toml:1-13`; environment probe 2026-08-30] |
| Existing `BlobManifestV1` + integrity helpers | manifest schema 1 / payload contract 1 | Sole signed committed authority and payload digest/size verification | Phase 2 verified exact canonical bytes, complete signed fields, and committed-only reads. [VERIFIED: `src/cacheness/storage/manifest.py:238-395`; `.planning/phases/02-canonical-storage-and-integrity-contract/02-VERIFICATION.md:27-64`] |
| Existing `ManifestRepository` adapters, extended with CAS | current local JSON, memory, SQLite reference adapters | Exact raw records plus atomic expected-generation publication | The current protocol has only `get_raw`, unconditional `put_raw`, `remove`, and listing; CAS is the missing authority primitive. Verbatim methods: `get_raw`, `put_raw`, `remove`, `list_keys`, `list_backend_entries`. [VERIFIED: `src/cacheness/storage/manifest_repository.py:38-60`] |
| Existing `GuardedHandlerIO` / `ManagedFileOps`, split at the stage-publication seam | current repository implementation | Private native serialization, contained exclusive generation publication, private snapshots, durable operation files | It already prevents handler-controlled paths from reaching managed storage and provides durable/exclusive control-file primitives. [VERIFIED: `src/cacheness/storage/guarded_handler_io.py:1-7,95-121,301-367`; `src/cacheness/storage/path_security.py:615-672`] |
| pytest | 8.4.1 in uv environment | Fault injection, crash/reopen, and deterministic race contracts | Existing project framework and test conventions. The dependency group says verbatim: `"pytest>=8.4.1"`. [VERIFIED: `pyproject.toml:68-73`; environment probe 2026-08-30] |

### Supporting

| Library / Module | Version | Purpose | When to Use |
|------------------|---------|---------|-------------|
| `ClearRecoveryCoordinator` patterns | journal version 1 | Reuse prepared/committed failure classification, strict bounds, topology binding, durable evidence, and reopen convergence | Mine patterns and tests; replace clear-only ownership rather than nesting a second journal beneath the new engine. Its current constants are verbatim: `MAX_JOURNAL_BYTES = 64 * 1024 * 1024`, `MAX_JOURNAL_ENTRIES = 100_000`, `MAX_JOURNAL_FIELD_BYTES = 8192`. [VERIFIED: `src/cacheness/storage/clear_recovery.py:1-6,34-70`] |
| `threading.Lock`, `Condition`, `Event`, `Barrier` | Python standard library | Per-key exclusion, in-flight close coordination, and deterministic interleaving tests | Use `Lock`/`Condition` in production state; use `Event`/`Barrier` in tests with timeouts. Lock waiter order is unspecified, so correctness must not depend on fairness. [CITED: https://docs.python.org/3/library/threading.html] |
| Existing typed BlobStore error hierarchy | current public contract | Conflict, backend, integrity, version, migration, and recoverable-cleanup outcomes | Extend stable lower-snake-case reasons rather than returning booleans for ambiguity or generic `CacheStorageError`. Current exact reasons include `"blob_lifecycle_conflict"`, `"blob_backend_failure"`, and `"blob_migration_required"`. [VERIFIED: `src/cacheness/error_handling.py:19-54,155-299`] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Dynamically retained per-key lock entries | Fixed lock striping | Striping is fixed-memory but serializes unrelated colliding keys, weakening D-20. Refcounted per-key entries preserve distinct-key concurrency and retire at zero users. [VERIFIED: `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:63-72`] |
| Immutable generations + manifest CAS | In-place payload replacement or rollback of the winning manifest | In-place mutation breaks complete-generation reads; rolling the manifest back after authority can revoke a valid concurrent winner. Both contradict locked D-01/D-04. [VERIFIED: `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:18-31`] |
| Project-owned operation records | Inferring orphans from filename patterns | The workspace currently contains many legitimate payloads whose names include `-candidate-`; filename matching cannot prove ownership or lifecycle status. D-06 explicitly forbids it. [VERIFIED: read-only workspace inventory, 76 such files on 2026-08-30; `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:35`] |
| Standard-library synchronization | New lock/journal package | No external package supplies backend CAS or cross-resource recovery; adding one would not remove the need for the project-owned state machine. [VERIFIED: current repository contracts and locked D-18; `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:66`] |

**Installation:** None. This phase should add no external package and therefore needs no package-legitimacy audit. [VERIFIED: the recommended stack above is stdlib plus existing locked dependencies]

## Runtime State Inventory

| Category | Items Found | Action Required |
|----------|-------------|-----------------|
| Stored data | Two ignored SQLite databases and a repository `cache/` directory exist. The cache database contains 29 `cache_entries`, has no `cacheness_manifest_records_v1` table, and the directory contains 76 files with `-candidate-` in their names. [VERIFIED: read-only filesystem and SQLite `mode=ro&immutable=1` inventory, 2026-08-30] | Treat all as pre-canonical/compatibility evidence outside Phase 3. Do not migrate, rename, deserialize, or delete it. Phase 7 owns stored-format inventory/migration; Phase 6 owns `UnifiedCache` delegation. [VERIFIED: `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:121-124`] |
| Live service config | No live service configuration is required by the Phase 3 local reference path. PostgreSQL/S3 integration exists in source but full service capability plumbing is deferred. [VERIFIED: `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:119-124`; `AGENTS.md:91-95`] | No API/UI patch. Do not claim remote CAS or reconciliation support in this phase. |
| OS-registered state | None: this is an in-process Python library with no web server, worker, container, or hosting manifest. [VERIFIED: `AGENTS.md:91-95`] | None. |
| Secrets/env vars | No `.env` file or live HMAC key was found in the audited repository root. Canonical `BlobStore` constructs its key provider at verbatim path `self.cache_dir / "blob_manifest_hmac_key.bin"`; PostgreSQL test configuration uses `CACHENESS_TEST_POSTGRES_URL`. [VERIFIED: `src/cacheness/storage/blob_store.py:282-287`; `tests/test_postgresql_backend.py:56`; read-only filesystem inventory 2026-08-30] | Do not rename or rotate keys. New operation evidence should authenticate with a domain-separated projection under the existing canonical store key boundary; never log key bytes. [VERIFIED: `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:35,58-61`] |
| Build artifacts / installed packages | `.venv` exists; no repository-local `*.egg-info` directory was found. [VERIFIED: read-only filesystem inventory, 2026-08-30] | Source/test changes require no artifact migration. Run tests through `uv run`; packaging remains Phase 8. [VERIFIED: `.planning/REQUIREMENTS.md:66-73`] |

**Runtime-state conclusion:** Updating every source file would still leave the observed compatibility databases and 76 candidate-named payloads unchanged. That is correct: Phase 3 reconciliation must report provenance-free evidence untouched, not absorb it as lifecycle residue. [VERIFIED: `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:57-61,123-124`; runtime inventory above]

## Architecture Patterns

### System Architecture Diagram

```text
Direct BlobStore call
        |
        v
Instance admission (reject after close begins)
        |
        +------ clear/reconcile ------> bounded store-wide admission barrier
        |
        v
Per-key coordinator (local ordering only)
        |
        v
Private native handler stage --serialization failure--> no durable evidence
        |
        v
Authenticated/versioned operation record (durable, bounded)
        |
        v
Exclusive immutable generation publication --> verify digest/size
        |
        v
Manifest repository CAS(expected generation / absence)
        |                         |
        | conflict                | success = AUTHORITY POINT
        v                         v
clean/report own candidate     committed/tombstoned manifest
                                  |
                                  v
                         idempotent old-payload reclamation
                                  |
                                  v
                         checkpoint + retire operation record

Read path:
committed signed manifest -> private immutable snapshot -> reread generation
      | same                                      | changed/missing
      v                                           v
digest/size -> native handler             one bounded reacquisition retry
```

This flow keeps the signed manifest as the only normal-read authority; operation records never become an alternate value source. [VERIFIED: `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:22-23,36-40,53-61`]

### Recommended Project Structure

```text
src/cacheness/storage/
├── blob_store.py                 # compatible direct facade; delegates lifecycle
├── lifecycle.py                  # mutation state machine and authority classification [ASSUMED]
├── operation_record.py           # bounded authenticated operation model/codec [ASSUMED]
├── operation_repository.py       # durable record CRUD/checkpoint seam [ASSUMED]
├── reconciliation.py             # dry-run/apply reports and bounded resume [ASSUMED]
├── coordination.py               # per-key registry + instance/store admission [ASSUMED]
├── manifest_repository.py        # exact raw persistence + conditional publication
├── guarded_handler_io.py         # split native stage from exclusive generation publish
├── manifest.py                   # existing signed authority model
└── clear_recovery.py             # compatibility shim or absorbed predecessor, not nested engine

tests/
├── test_manifest_repository_cas.py
├── test_blob_store_atomic_lifecycle.py
├── test_blob_store_reconciliation.py
├── test_blob_store_concurrency.py
└── test_blob_store_close_contract.py
```

Names marked `[ASSUMED]` are recommendations only; CONTEXT delegates exact module/type names. The responsibility split is the prescriptive part. [VERIFIED: `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:74-80`]

### Pattern 1: Split Private Serialization from Persistent Publication

**What:** Refactor guarded handler output into a context-owned staged artifact. Handler serialization completes in the private temporary directory first. Only after it succeeds does the lifecycle engine allocate the generation/operation identifiers, persist durable evidence, and publish the staged stream exclusively at its immutable generation locator. [VERIFIED: locked D-01/D-05/D-08 in `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:18-21,33-41`; current combined seam at `src/cacheness/storage/guarded_handler_io.py:301-334`]

**Why:** The current method performs handler serialization at line 312 and managed publication at lines 320-325 inside one call. There is no legal point for D-05 evidence between those steps. [VERIFIED: `src/cacheness/storage/guarded_handler_io.py:301-334`]

**Planning requirements:**

1. The staged-artifact context must retain its validated file descriptor identity until publication, preserving the existing path-race defense. [VERIFIED: `src/cacheness/storage/guarded_handler_io.py:52-86,134-211,215-300`]
2. The generation identifier must be allocated before the managed locator, and the locator provenance must bind the logical-key physical ID, operation ID, and generation rather than using an unrelated candidate UUID. The current code creates `candidate_id = f"{storage_id}-candidate-{uuid.uuid4().hex}"` and separately creates `generation=uuid.uuid4().hex`. [VERIFIED: `src/cacheness/storage/blob_store.py:379-385,417-422`]
3. Publication must be create-exclusive, streamed, contained, and durably acknowledged; the existing general stream writer atomically replaces a locator, while only the bytes helper is create-exclusive. Add the missing exclusive stream primitive instead of buffering payloads into memory. [VERIFIED: `src/cacheness/storage/path_security.py:553-586,615-665`]
4. Candidate verification must hash/size the managed candidate before manifest construction and must not deserialize it. [VERIFIED: `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:18,58-61`; existing digest seam `src/cacheness/storage/blob_store.py:393-432`]

### Pattern 2: Exact Expectation CAS at the Manifest Repository

**What:** Extend `ManifestRepository` with one conditional publication primitive that receives the logical key, an authenticated expectation, the signed replacement record, and compatibility projection. The expectation represents absence for create or the exact authenticated committed generation for overwrite/delete. [VERIFIED: locked D-02/D-18 in `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:21,66`]

**Recommended strengthening:** Carry both `expected_generation` and a digest/revision of the exact canonical record observed by the engine. Generation enforces the public lifecycle contract; exact-record comparison prevents an authenticated same-generation metadata update from being silently lost. This extra token is an implementation recommendation, not a new public stored field. [ASSUMED]

**Repository rule:** The repository must not independently trust/decode an unauthenticated manifest merely to discover a generation. The engine authenticates the current record first; the adapter atomically compares the supplied opaque expectation with current storage and writes the replacement. [VERIFIED: Phase 2 read ordering at `src/cacheness/storage/blob_store.py:1006-1076`; raw repository boundary at `src/cacheness/storage/manifest_repository.py:38-60`]

**Local reference adapters:**

- In-memory: compare and replace under the backend object's lock. [VERIFIED: current adapter is process-local at `src/cacheness/storage/manifest_repository.py:166-170`]
- SQLite: perform compatibility projection and canonical BLOB conditional update in one transaction; current `put_raw` already writes both in one transaction but unconditionally upserts the BLOB. [VERIFIED: `src/cacheness/storage/manifest_repository.py:236-307`]
- JSON: hold only a short document-publication lock across refresh → compare → durable replace, not across serialization or payload work. Cross-process truthfulness requires an OS-backed compare/publication boundary; otherwise construction fails typed rather than downgrading. [VERIFIED: locked D-18; current JSON adapter's unconditional backend call at `src/cacheness/storage/manifest_repository.py:106-125`]

Full capability declarations and PostgreSQL implementation remain Phase 4. Full S3 payload semantics remain Phase 5. [VERIFIED: `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:119-124`]

### Pattern 3: One Operation Record, Pre/Post-Authority Recovery

**What:** Use one bounded versioned record per mutation. It contains no payload bytes and binds operation ID, logical key, operation kind, expected generation/absence, candidate/tombstone generation, old/new locators, intended transition, progress checkpoint, and topology/store provenance. Authenticate its canonical projection or enforce equally strict owner/version/topology/locator validation. [VERIFIED: locked D-05/D-06 at `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:33-36`]

**Recovery decision table:**

| Observed authority | Candidate / old payload | Safe conclusion | Required action |
|--------------------|-------------------------|-----------------|-----------------|
| Expected old generation (or absence) still authoritative | Candidate present and operation provenance valid | CAS never committed | Delete/quarantine only this operation's candidate, then retire evidence; failure remains typed cleanup debt. [VERIFIED: D-04/D-07/D-11] |
| New generation authoritative and matches operation record | Old payload or tombstone remains | CAS committed | Never roll manifest back; reclaim proven superseded bytes, checkpoint, retire evidence. [VERIFIED: D-04/D-07/D-11] |
| Different generation authoritative | Any residue | Another operation won | Do not delete anything unless the residue is exclusively proven to belong to the losing operation and is not the current locator; report conflict. [VERIFIED: D-02/D-11/D-19] |
| Manifest/evidence malformed, unauthenticated, future-version, or provenance-free | Any | Ownership/authority cannot be proven | Report untouched; quarantine only through a separately safe backend-native move. [VERIFIED: D-15/D-16] |

The current `_cleanup_uncommitted_candidate` and `_cleanup_prior_payload` perform immediate deletion with no durable operation identity. When cleanup itself fails, they raise a generic storage error but leave no machine-readable resumption record. Replace these helpers with lifecycle-engine checkpoint/reclamation calls. [VERIFIED: `src/cacheness/storage/blob_store.py:911-956`]

### Pattern 4: Tombstone-First Delete

**What:** Authenticate the committed manifest, persist delete evidence, then CAS a signed `tombstoned` manifest against the expected generation. The tombstone retains enough signed payload identity to prove which old locator may be reclaimed. Only after payload reclamation succeeds does the engine conditionally remove the tombstone and retire operation evidence. [VERIFIED: locked D-09/D-11; existing exact state string `"tombstoned"` at `src/cacheness/storage/manifest.py:304-310`]

The current delete order is the inverse: it deletes the payload first and then removes the manifest. A failure between lines 614 and 615 leaves committed metadata naming a missing payload. [VERIFIED: `src/cacheness/storage/blob_store.py:594-618`]

Repeated delete behavior must distinguish three cases without destructive guessing: true absence is a compatible successful no-op; the same validated tombstone resumes cleanup; a newer committed generation conflicts. Preserve the legacy boolean adapter's documented meaning where possible while treating “already absent” as lifecycle success rather than an error. The exact already-absent return boolean needs a compatibility decision before planning locks the public assertion. [VERIFIED: locked D-09; current return contract `True if deleted, False if not found` at `src/cacheness/storage/blob_store.py:594-611`]

### Pattern 5: Bounded Generation-Snapshot Clear

**What:** Replace whole-backend snapshot/restore with an authenticated, bounded list of `(key, expected generation)` targets and a durable per-entry checkpoint. Establish the snapshot under the explicit store-wide admission barrier, then delete each target through the same tombstone/CAS lifecycle. A key created after the snapshot is absent from the target set; a key overwritten after snapshot conflicts on expected generation and is not deleted. [VERIFIED: locked D-10/D-20; `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:45,72`]

The Phase 1 clear journal is valuable for fault classification but not the target data model: it stores a complete `metadata_snapshot` and requires snapshot keys to equal mapping keys. Its limits are 100,000 entries and 64 MiB, but it is still all-at-once and cannot resume/report individual operator actions. [VERIFIED: `src/cacheness/storage/clear_recovery.py:34-52,448-470,564-642`]

Use stable page ordering/cursors and fixed maximum page/report/record sizes. Do not select numeric defaults in planning from intuition; derive tombstone retention, orphan grace, page size, and checkpoint frequency from fault/crash tests and measured local call counts. [VERIFIED: `.planning/STATE.md:143-147`; `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:76-80`]

### Pattern 6: Read Acquisition with One Generation Retry

**What:** Read and authenticate committed manifest M1, attempt one private immutable snapshot, then re-read/authenticate the manifest before deserialization. If snapshot acquisition fails and the generation changed, or if M2 differs from M1, discard the snapshot and retry acquisition once. If the same generation still names missing/tampered bytes, preserve the existing typed integrity failure. Once a complete private snapshot is acquired and the generation recheck passes, later cleanup cannot create mixed bytes because the handler reads only the private snapshot. [VERIFIED: locked D-03/D-19; existing snapshot contract `src/cacheness/storage/guarded_handler_io.py:336-367`; existing integrity order `src/cacheness/storage/blob_store.py:473-522`]

Do not retry malformed manifests, invalid signatures, unsupported versions, backend failures, or same-generation integrity failures. Retry is only for observed generation movement during acquisition. [VERIFIED: locked D-03 and Phase 2 typed failure boundary `.planning/phases/02-canonical-storage-and-integrity-contract/02-VERIFICATION.md:39-64`]

### Pattern 7: Refcounted Per-Key Coordination and Instance Close Admission

**What:** Use a small registry guard protecting a dictionary of key → `{lock, users}`. Increment `users` while holding the guard before waiting on the key lock; on release, decrement under the guard and remove only the same entry when zero. This keeps memory proportional to active/waiting keys and avoids lock striping collisions. [ASSUMED] Behavior is required by D-17/D-20; this implementation detail is not locked.

Production correctness must not rely on lock fairness: Python documents that selection among blocked `Lock.acquire()` waiters is undefined. Multi-key work sorts one stable key representation before acquisition. [CITED: https://docs.python.org/3/library/threading.html] [VERIFIED: deterministic ordering required by `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:72`]

Instance admission should reject new operations once close begins, count in-flight operations with a condition, wait deterministically for those already admitted, flush/reconcile only owned durable state, close `GuardedHandlerIO`, and close the backend only when `_owns_backend` is true. The current `close()` unconditionally closes both guarded I/O and the backend and has no idempotence/admission state. [VERIFIED: `src/cacheness/storage/blob_store.py:217-232,266-280,860-870`; locked D-12]

### Pattern 8: Dry-Run-First Reconciliation

**What:** Separate analysis from mutation. A dry run consumes bounded manifest pages, bounded operation-record pages, and only backend-provided owned-locator inventory. It emits stable machine records plus a human summary. Apply consumes those records but re-authenticates/revalidates the current manifest, operation record, generation, and locator immediately before each action; stale findings become conflicts, not stale deletes. [VERIFIED: locked D-13 through D-16]

**Required finding data (verbatim from D-13):** `authoritative generation`, `operation provenance`, `residue type`, `proposed action`, `reason`, and whether the action is `safe`, `blocked`, or `requires confirmation`. [VERIFIED: `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:52-55`]

Phase 3 must not implement a universal payload scan by walking arbitrary filenames. It may reconcile authenticated manifests and its own operation namespace on current local reference storage. If an adapter cannot enumerate backend-owned locators safely and boundedly, orphan inventory is reported as unsupported/blocked until the Phase 4/5 capability work. [VERIFIED: D-06/D-15/D-18 and deferred Phase 4/5 scope]

### Anti-Patterns to Avoid

- **One global mutex around every operation:** It violates distinct-key concurrency and can hide missing backend CAS in tests. Use per-key ordering plus a short explicit aggregate barrier. [VERIFIED: D-17/D-18/D-20]
- **Deleting the old payload before CAS or deleting the new manifest after cleanup failure:** Either loses the last valid generation or revokes the winner. [VERIFIED: D-04/D-07]
- **Treating a candidate filename as ownership proof:** Current authoritative/compatibility payloads already contain `-candidate-`; loose matching is unsafe. [VERIFIED: D-06; runtime inventory]
- **Catching `BaseException` and performing speculative destructive cleanup:** `BaseException` models process-loss boundaries in existing crash tests. Persist evidence before the boundary and let reopen reconciliation decide. [VERIFIED: `src/cacheness/storage/clear_recovery.py:346-404`; `tests/test_clear_recovery.py:27-31,234-326`]
- **Deserializing during reconciliation:** Pickle/dill may execute code and are trusted only at the normal authenticated read boundary. Lifecycle ownership comes from signed metadata, not payload interpretation. [VERIFIED: D-16; `docs/SECURITY.md:12-23`]
- **Restoring an old backend snapshot after uncertain publication:** Once CAS may have succeeded, rollback can erase a valid winner. Unknown authority must poison/block until evidence can decide. [VERIFIED: D-04; analogous clear classification `src/cacheness/storage/clear_recovery.py:362-404,428-446`]
- **Full-store list materialization or N+1 discovery without bounds:** STOR-06 and QUAL-07 require bounded/resumable work; page and call-count assertions belong in validation. [VERIFIED: `.planning/REQUIREMENTS.md:12,73`]
- **Adding a Cacheness payload wrapper/header:** Native handlers remain the payload-format owners. Lifecycle data belongs only in manifests/operation records. [VERIFIED: `.planning/phases/02-canonical-storage-and-integrity-contract/02-VERIFICATION.md:35-38,147-151`]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Payload serialization/container format | Lifecycle envelope around NPZ/Parquet/pickle/dill/Blosc2 | Existing handler + guarded private stage | A new wrapper creates another stored format and migration obligation. [VERIFIED: Phase 2 prohibition] |
| Cryptographic signing/hash comparison | New MAC/hash scheme | Existing HMAC-SHA256 key provider, canonical signing bytes, SHA-256/size helpers with domain-separated operation-record projection | Phase 2 already verified key provenance and fail-closed integrity. [VERIFIED: `.planning/phases/02-canonical-storage-and-integrity-contract/02-VERIFICATION.md:47-64`] |
| Path validation or direct `Path.unlink` recovery | Filename sanitization and loose root checks | `ManagedFileOps` + `resolve_managed_locator` + exclusive/durable contained primitives | Existing code guards symlink/race/root-replacement boundaries. [VERIFIED: `src/cacheness/storage/path_security.py:220-314,421-476,615-672`] |
| Distributed lock pretending to span payload and metadata stores | Python mutex/file lock as the correctness boundary | Immutable payload generation + backend manifest CAS + reconciliation | Local locks do not coordinate processes/hosts or two resource managers. [VERIFIED: D-18] |
| Retry framework | Generic retry of entire mutation | Explicit bounded retry only on idempotent steps and the single permitted read reacquisition | Retrying serialization/CAS/cleanup indiscriminately can duplicate residue or revoke a winner. [VERIFIED: D-03/D-11/D-19] |
| Reconciliation ownership inference | Glob/extension/payload sniffing | Authenticated manifests + validated operation records + backend-owned inventory | Filename/payload content does not prove library ownership. [VERIFIED: D-06/D-15/D-16] |

**Key insight:** the hard problem is not atomic file replacement; it is preserving a provable authority decision across two independent resources. The signed manifest CAS plus durable operation evidence is the transaction protocol. [VERIFIED: D-04 through D-07]

## Common Pitfalls

### Pitfall 1: Journaling Too Early or Too Late

**What goes wrong:** Journaling before handler serialization leaves durable records for operations that never produced a candidate; journaling after managed candidate publication leaves unowned residue on process loss. [VERIFIED: locked D-05/D-08]

**Why it happens:** The current `GuardedHandlerIO.put()` combines serialization and publication, so there is no intermediate hook. [VERIFIED: `src/cacheness/storage/guarded_handler_io.py:301-334`]

**How to avoid:** Split the API and place durable operation-record creation after private serialization succeeds but before the exclusive managed publish begins. [VERIFIED: D-05/D-08]

**Warning signs:** An injected serialization fault creates an operation record, or a `BaseException` during candidate publication leaves a file with no validated operation record.

### Pitfall 2: CAS That Is Only “Check Then Put”

**What goes wrong:** Two instances both read generation G, both pass a non-atomic check, and the later unconditional `put_raw` silently clobbers the first winner. [VERIFIED: current repository `put_raw` is unconditional at `src/cacheness/storage/manifest_repository.py:44-51,106-125,236-252`]

**Why it happens:** A local per-key lock makes single-instance tests pass but does not coordinate other processes/instances. [VERIFIED: D-18]

**How to avoid:** Make compare and publication one backend-local atomic primitive and test with independent store/repository instances. [VERIFIED: D-02/D-18/D-19]

**Warning signs:** Repository code invokes `get_raw()` and then `put_raw()` as separate unlocked/transactionally separate calls; race tests use only one `BlobStore` object.

### Pitfall 3: Cleanup Revokes or Damages the Winner

**What goes wrong:** A losing writer deletes a locator now named by the winning manifest, or a post-commit error handler restores the old manifest. [VERIFIED: D-02/D-04/D-11/D-19]

**Why it happens:** Cleanup is keyed only by logical key/filename rather than operation ID, generation, and exact locator provenance. [VERIFIED: D-06]

**How to avoid:** Re-read/authenticate authority immediately before destructive cleanup; require both operation ownership and a generation mismatch from the current locator. Treat changed ownership as conflict. [VERIFIED: D-11/D-15]

**Warning signs:** Cleanup accepts just `key`; a conflict handler calls repository `remove(key)`; tests assert only the returned exception and not winner readability/residue ownership.

### Pitfall 4: Tombstone Removal Is Not Conditional

**What goes wrong:** Delete publishes a tombstone, another writer creates a new generation, then delete's final `remove(key)` removes that new generation. [VERIFIED: D-09/D-19]

**Why it happens:** The current repository exposes unconditional `remove`. [VERIFIED: `src/cacheness/storage/manifest_repository.py:53-54,127-132,309-319`]

**How to avoid:** Tombstone retirement is itself conditional on the exact tombstone generation/revision. A new generation produces conflict and survives. [VERIFIED: D-09/D-11]

**Warning signs:** Delete finalization calls unconditional manifest removal; no write-after-tombstone forced interleaving test exists.

### Pitfall 5: Read Retry Hides Corruption

**What goes wrong:** A same-generation missing/tampered payload is retried and eventually translated into a miss, weakening Phase 2's typed integrity contract. [VERIFIED: Phase 2 observable truths 6-10 in `.planning/phases/02-canonical-storage-and-integrity-contract/02-VERIFICATION.md:39-64`]

**Why it happens:** Retry is keyed to any `FileNotFoundError` rather than a proven generation change. [VERIFIED: D-03]

**How to avoid:** Re-read the manifest; retry once only if authenticated generation changed. Otherwise re-raise the original typed integrity/backend failure. [VERIFIED: D-03]

**Warning signs:** A loop count greater than two acquisition attempts; retry catches malformed/auth/signature/version errors; tests do not assert exact snapshot/manifest-read counts.

### Pitfall 6: Reconciliation Is “Dry Run” in Name Only

**What goes wrong:** Analysis updates access counters, creates a signing key, quarantines a file, cleans a temp, or invokes a handler. [VERIFIED: D-13/D-16 and Phase 2 non-mutating read truth]

**Why it happens:** Reuse of high-level read APIs carries side effects or deserialization. [VERIFIED: Phase 2 explicitly separated exact raw records and guarded reads; `.planning/phases/02-canonical-storage-and-integrity-contract/02-VERIFICATION.md:65-68`]

**How to avoid:** Dry run consumes raw repository/evidence inventory through dedicated non-mutating readers, authenticates records, and hashes only when needed; mutation callbacks are absent/spied to fail. [VERIFIED: D-13/D-16]

**Warning signs:** Dry-run tests merely inspect final report without comparing full pre/post bytes, mtimes, key material, and backend call logs.

### Pitfall 7: Close Transfers Ownership

**What goes wrong:** A `BlobStore` closes a caller-injected backend, accepts operations after root descriptor closure, double-closes a descriptor, or clears data. [VERIFIED: D-12; current unconditional close `src/cacheness/storage/blob_store.py:860-870`]

**Why it happens:** Close is treated as two direct `.close()` calls instead of an operation-admission state transition. [VERIFIED: current implementation]

**How to avoid:** Track owned resources, reject new operations before draining, wait on in-flight count, make resource release exactly-once, and never invoke `clear()`. Existing constructor-failure tests already distinguish owned and injected backends. [VERIFIED: `tests/test_blob_store_read_contract.py:90-288`]

**Warning signs:** `close()` lacks a guard/state; direct APIs do not share an admission context; injected-backend close spy fires.

### Pitfall 8: Aggregate Work Is Unbounded

**What goes wrong:** Clear/reconcile materializes every manifest/operation/payload or restarts from zero after an interruption. [VERIFIED: STOR-06/QUAL-07]

**Why it happens:** Existing `list_keys()` returns a full list, and the clear journal snapshots the full backend. [VERIFIED: `src/cacheness/storage/manifest_repository.py:56-60,134-156,321-341`; `src/cacheness/storage/clear_recovery.py:448-470`]

**How to avoid:** Introduce stable bounded page/cursor interfaces and per-action checkpoints. Keep compatibility list APIs as adapters over bounded internals only where safe. [VERIFIED: D-10/D-14]

**Warning signs:** `list(...)` around all keys/records in reconciliation; no max-page/report bound; crash resume repeats already completed deletes; call count scales worse than a documented constant per item.

## Code Examples

The following are planning skeletons, not locked public names. Exact type/module names are delegated. [ASSUMED] Behavior and ordering are locked by CONTEXT. [VERIFIED: `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:18-72`]

### Conditional Publication Contract

```python
from dataclasses import dataclass
from typing import Mapping, Protocol


@dataclass(frozen=True)
class ManifestExpectation:  # suggested internal name
    generation: str | None       # None means the caller authenticated absence
    record_digest: str | None    # strengthens against same-generation lost update


class ManifestRepository(Protocol):
    def publish_if_expected(
        self,
        key: str,
        expected: ManifestExpectation,
        record: bytes,
        *,
        entry_data: Mapping[str, object],
    ) -> None:
        """Publish atomically or raise the typed lifecycle conflict."""
```

This keeps raw storage opaque: the lifecycle engine authenticates the expected manifest; the repository performs one atomic compare/publication and never treats a failed compare as permission to overwrite. [VERIFIED: D-02/D-18; raw repository boundary `src/cacheness/storage/manifest_repository.py:38-60`]

### Write Authority Skeleton

```python
# Pseudocode: names are discretionary; order is mandatory.
with instance_admission.operation(), key_coordinator.hold(key):
    with guarded_io.stage(handler, value, config) as staged:
        current = load_authenticated_committed_or_absent(key)
        operation = operation_records.create_before_persistent_payload(
            key=key,
            expected_generation=None if current is None else current.generation,
            candidate_generation=new_generation(),
            candidate_locator=derive_owned_locator(key),
        )
        candidate = guarded_io.publish_exclusive(staged, operation.candidate_locator)
        verify_digest_and_size(candidate)
        manifests.publish_if_expected(key, operation.expectation, signed_manifest(candidate))
        operation_records.checkpoint_authority_published(operation)
        reclaim_superseded_payload_if_still_owned(current, operation)
        operation_records.retire(operation)
```

The private stage precedes durable evidence; every managed persistent side effect follows it. Manifest publication is the single authority point. [VERIFIED: D-01/D-04/D-05/D-08]

### Read Reacquisition Skeleton

```python
for acquisition_attempt in range(2):  # one initial attempt, one permitted retry
    first = load_authenticated_committed(key)
    try:
        snapshot = guarded_io.acquire_snapshot(first.locator)
    except FileNotFoundError:
        second = load_authenticated_committed(key)
        if second.generation != first.generation and acquisition_attempt == 0:
            continue
        raise_typed_missing_for_same_generation()

    second = load_authenticated_committed(key)
    if second.generation != first.generation:
        snapshot.close()
        if acquisition_attempt == 0:
            continue
        raise_typed_lifecycle_conflict()
    return verify_then_deserialize(snapshot, first)
```

The numeric bound `range(2)` directly implements locked D-03's “one bounded retry”; it is not a general retry policy. [VERIFIED: `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:22`]

### Deterministic Race Test Pattern

```python
entered = threading.Event()
release = threading.Event()

def pause_before_cas(*_args: object) -> None:
    entered.set()
    assert release.wait(timeout=5)

# Start the first operation, wait until the exact seam is reached, run the
# contender, then release. Never use sleep() as the ordering oracle.
```

Python `Event` is a flag-based signal and `Barrier` coordinates a fixed number of threads; both accept bounded waits suitable for deterministic tests. [CITED: https://docs.python.org/3/library/threading.html]

## State of the Art

| Old Approach in Repository | Required Current Approach | Change Boundary | Impact |
|----------------------------|---------------------------|-----------------|--------|
| Candidate UUID locator unrelated to signed generation, unconditional manifest `put_raw`, immediate best-effort cleanup | Generation-specific immutable locator, durable operation record, expected-generation CAS, explicit cleanup debt | Phase 3 | Establishes old-or-new completeness and deterministic conflicts. [VERIFIED: current `src/cacheness/storage/blob_store.py:364-466`; locked D-01-D-08] |
| Delete payload then remove metadata | CAS signed tombstone, reclaim payload, conditional tombstone retirement | Phase 3 | Prevents committed manifests naming deleted payloads and prevents late delete from removing a new write. [VERIFIED: current `src/cacheness/storage/blob_store.py:594-618`; locked D-09] |
| Clear-only whole-store snapshot journal with prepared/committed recovery | Bounded authenticated generation snapshot using the common per-entry lifecycle plus resumable checkpoints/reports | Phase 3 | Makes clear safe with later writes and gives operators dry-run/resume. [VERIFIED: current `src/cacheness/storage/clear_recovery.py:336-470`; locked D-10,D-13-D-16] |
| Global clear admission serializes ordinary lifecycle mutations | Refcounted per-key coordination; store-wide barrier only for aggregate operations | Phase 3 | Unrelated keys proceed independently. [VERIFIED: current `src/cacheness/storage/clear_recovery.py:198-215`; locked D-17-D-20] |
| Unconditional close of guarded I/O and backend | Instance admission/drain plus ownership-aware exactly-once release | Phase 3 | Injected backends remain caller-owned and repeated close converges. [VERIFIED: current `src/cacheness/storage/blob_store.py:860-870`; locked D-12] |

**Deprecated/outdated within the Phase 3 path:**

- Direct calls to manifest `put_raw`/`remove` from multi-step mutations are insufficient; use conditional authority operations. [VERIFIED: D-02/D-09]
- `_cleanup_uncommitted_candidate` and `_cleanup_prior_payload` cannot be the recovery system because they retain no resumable provenance. [VERIFIED: `src/cacheness/storage/blob_store.py:931-956`; D-05-D-08]
- Filename grammar such as `_CANDIDATE_PREFIX` cannot establish ownership for generalized reconciliation. [VERIFIED: `src/cacheness/storage/clear_recovery.py:62-65,678-708`; D-06]
- Whole-backend metadata snapshot restoration must not be used after an uncertain authority transition. [VERIFIED: `src/cacheness/storage/clear_recovery.py:362-404`; D-04]

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.4.1 [VERIFIED: `pyproject.toml:68-73`; uv environment probe 2026-08-30] |
| Config file | `pyproject.toml` [VERIFIED: `pyproject.toml:82-111`] |
| Discovery | `tests/test_*.py`, `Test*`, `test_*`; strict markers enabled. [VERIFIED: `pyproject.toml:82-99`] |
| Quick run command | `uv run pytest -q -o log_cli=false tests/test_manifest_repository_cas.py tests/test_blob_store_atomic_lifecycle.py -x` [ASSUMED] New filenames are proposed. |
| Full phase command | `uv run pytest -q -o log_cli=false tests/test_manifest_repository_cas.py tests/test_blob_store_atomic_lifecycle.py tests/test_blob_store_reconciliation.py tests/test_blob_store_concurrency.py tests/test_blob_store_close_contract.py tests/test_blob_manifest.py tests/test_blob_manifest_backends.py tests/test_blob_store_read_contract.py tests/test_blob_store_integrity.py tests/test_clear_recovery.py tests/test_filesystem_containment.py -x` [ASSUMED] New filenames are proposed; existing filenames are verified. |
| Full suite command | `uv run pytest -q -o log_cli=false` [VERIFIED: repository pytest configuration and prior verification command] |
| Targeted lint | `uv run ruff check <phase-created-or-modified-python-files>` [VERIFIED: `AGENTS.md:127-153`] |

Focused environment proof during research ran seven parametrized cases covering prepared reopen rollback, committed reopen roll-forward, clear admission, SQLite atomic raw publication, and the canonical tracer; all passed. [VERIFIED: `uv run pytest` probe, 2026-08-30]

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| STOR-03 | Every injected write boundary exposes old complete or new complete generation; read acquisition never mixes manifest/payload | unit + integration + crash/reopen | `uv run pytest -q -o log_cli=false tests/test_blob_store_atomic_lifecycle.py -k 'write or read_acquisition' -x` | ❌ Wave 0 [ASSUMED] filename |
| STOR-04 | Pre-authority failure preserves old authority and leaves recoverable owned residue; post-authority failure preserves new authority and cleanup debt | fault matrix + reopen | `uv run pytest -q -o log_cli=false tests/test_blob_store_atomic_lifecycle.py -k 'failure or recovery or reopen' -x` | ❌ Wave 0 [ASSUMED] filename |
| STOR-05 | Overwrite/delete/clear/close converge on repetition; tombstone and operation retirement are conditional and ownership-aware | contract + integration | `uv run pytest -q -o log_cli=false tests/test_blob_store_atomic_lifecycle.py tests/test_blob_store_close_contract.py -k 'idempotent or delete or clear or close' -x` | ❌ Wave 0 [ASSUMED] filenames |
| STOR-06 | Dry run is byte-for-byte non-mutating; apply is bounded/checkpointed/resumable; ambiguous evidence is untouched or safely quarantined | unit + adversarial + crash/reopen | `uv run pytest -q -o log_cli=false tests/test_blob_store_reconciliation.py -x` | ❌ Wave 0 [ASSUMED] filename |
| STOR-07 | Forced write/write, write/delete, and read/write same-key races have deterministic winners; distinct keys overlap; registry retires entries | deterministic threads + independent instances + stress | `uv run pytest -q -o log_cli=false tests/test_blob_store_concurrency.py -x` | ❌ Wave 0 [ASSUMED] filename |

### Required Fault-Injection Matrix

Each boundary must be tested with ordinary `Exception` and process-loss-style `BaseException` where meaningful. `BaseException` tests must reopen a fresh store rather than relying on the interrupted live object's cleanup. This matches the established `_SimulatedClearInterruption` pattern. [VERIFIED: `tests/test_clear_recovery.py:27-31,234-326`; `src/cacheness/storage/clear_recovery.py:346-404`]

| Operation | Inject Immediately Before / During | Required Observable Result |
|-----------|------------------------------------|----------------------------|
| write/overwrite | handler serialization | No operation record, no managed candidate, old/absence unchanged. [VERIFIED: D-08] |
| write/overwrite | operation-record exclusive create | No managed candidate, old/absence unchanged, typed backend/recoverable failure. [VERIFIED: D-05/D-08] |
| write/overwrite | candidate stream creation/write/fsync/directory acknowledgement | Old/absence remains authority; partial/complete candidate is tied to detectable operation evidence; reopen dry run reports it. [VERIFIED: D-05-D-08] |
| write/overwrite | candidate digest/size verification | Old remains authority; candidate never published in a manifest; evidence remains recoverable. [VERIFIED: D-01/D-07] |
| write/overwrite | CAS compare failure | One winner remains readable; loser gets typed conflict and may clean/report only its own candidate. [VERIFIED: D-02/D-19] |
| write/overwrite | CAS post-write acknowledgement ambiguity | Reopen authenticates actual manifest; if new generation is authority, only roll forward; never speculative rollback. [VERIFIED: D-04/D-07] |
| write/overwrite | authority checkpoint update | New manifest remains authority; prepared-looking record is classified by manifest match and resumed forward. [VERIFIED: D-04/D-14] |
| write/overwrite | old payload reclamation / operation retirement | New remains readable; cleanup debt is typed and second recovery converges. [VERIFIED: D-07/D-11/D-14] |
| delete | tombstone CAS | Old remains until tombstone wins; compare conflict cannot delete a newer write. [VERIFIED: D-09/D-19] |
| delete | payload reclamation | Signed tombstone remains and repeated delete/reconcile resumes. [VERIFIED: D-09/D-11] |
| delete | conditional tombstone retirement | Absence or newer generation survives; late finalizer cannot remove newer record. [VERIFIED: D-09/D-11] |
| clear | bounded snapshot page/checkpoint, each per-key CAS/reclaim, interruption between pages | Completed entries remain completed; later keys/new generations are not accidentally deleted; resume continues from stable checkpoint. [VERIFIED: D-10/D-14] |
| close | transition to rejecting new work, waiting for in-flight, record flush, owned resource close | New calls fail typed; admitted call completes deterministically; repeated close is no-op/same outcome; injected backend remains open. [VERIFIED: D-12] |

### Required Race Matrix

Use `threading.Event`/`Barrier` hooks at exact lifecycle seams with timeouts; do not use `sleep()` as the interleaving oracle. Python documents that lock waiter choice is undefined, so assertions target the CAS/state invariant, not which thread happens to win. [CITED: https://docs.python.org/3/library/threading.html]

| Race | Forced Interleaving | Assertion |
|------|---------------------|-----------|
| write vs write, absent | both read absence; both candidates verified; release CAS together | Exactly one committed generation; one typed conflict; winner readable; loser cleanup touches only loser locator. [VERIFIED: D-19] |
| write vs write, overwrite | both read generation G; first wins G→N1; second attempts G→N2 | N1 remains authority; loser cannot reclaim G until it proves ownership rules and cannot delete N1. [VERIFIED: D-02/D-11/D-19] |
| write vs delete | both read G; alternate which CAS wins | Winner determines committed new generation or tombstone; loser conflicts; no missing payload under committed manifest. [VERIFIED: D-19] |
| read vs overwrite | reader reads G, writer commits N and starts G reclamation | Reader returns complete G snapshot or retries once and returns N/typed conflict; never mixed. [VERIFIED: D-03/D-19] |
| read vs delete | reader reads G, delete tombstones/reclaims | Reader returns complete G snapshot or one bounded lifecycle outcome; handler never sees partial bytes. [VERIFIED: D-03/D-19] |
| distinct-key operations | block key A at pre-CAS, complete key B | B completes before A releases; proves no global ordinary-operation mutex. [VERIFIED: D-20] |
| clear vs post-snapshot create/overwrite | establish clear snapshot, then admit later write where protocol permits | New key absent from snapshot survives; changed generation conflicts rather than being deleted. [VERIFIED: D-10] |
| close vs in-flight operation | pause admitted op, begin close, attempt second op | close waits/cancels deterministically; second op rejected; no resource use after close. [VERIFIED: D-12] |

### Reconciliation Validation

Dry-run tests must compare complete pre/post observable state: exact manifest bytes, operation-record bytes, payload bytes or hashes, mtimes where stable, signing-key bytes, directory membership, backend counters, and mutation-call spies. The report must be deterministic for the same snapshot and serializable to the documented machine format. [VERIFIED: D-13/D-16; established zero-mutation audit `.planning/phases/02-canonical-storage-and-integrity-contract/02-VERIFICATION.md:65-68`]

Apply/resume tests must interrupt after every action checkpoint, reopen, re-run, and assert convergence without repeated destructive calls. Mutate authority between dry run and apply and assert that revalidation converts the finding to blocked/conflict. Include malformed, oversized, duplicate, unknown-version, wrong-store, wrong-key, locator-escape, symlink, and provenance-free evidence. [VERIFIED: D-14-D-16; bounded clear adversarial precedent `tests/test_clear_recovery.py:944-1106,1538-2008`]

### Sampling Rate

- **Per task commit:** Run the smallest new test module for the touched seam plus its nearest Phase 2/clear regression file. [VERIFIED: existing test organization]
- **Per wave merge:** Run the full phase command above. [ASSUMED] New filenames are proposed.
- **Before phase verification:** Run full phase command, full suite, targeted Ruff, and the independent compatibility corpus validator `uv run python tests/fixtures/compat/validate_corpus.py --expected-through sqlite-columns-v0314`. [VERIFIED: Phase 2 verification command `.planning/phases/02-canonical-storage-and-integrity-contract/02-VERIFICATION.md:95-101`]
- **Phase gate:** Full suite green relative to the checked baseline; every fault/race/reopen test active with no unconditional skip. [VERIFIED: project delivery constraints]

### Wave 0 Gaps

- [ ] `tests/test_manifest_repository_cas.py` — exact create-if-absent, replace-if-generation, conditional tombstone retirement, independent-instance conflict, SQLite rollback, JSON cross-instance refresh/locking. [ASSUMED] Filename is proposed.
- [ ] `tests/test_blob_store_atomic_lifecycle.py` — complete write/delete/clear fault matrix and reopen recovery. [ASSUMED] Filename is proposed.
- [ ] `tests/test_blob_store_reconciliation.py` — dry-run immutability, stable reports, bounded pages, apply revalidation, checkpoints, quarantine/report policy. [ASSUMED] Filename is proposed.
- [ ] `tests/test_blob_store_concurrency.py` — forced same-key and distinct-key race matrix, one-retry read acquisition, registry retirement. [ASSUMED] Filename is proposed.
- [ ] `tests/test_blob_store_close_contract.py` — operation admission/drain, exactly-once owned closes, injected backend retention, no data clear. [ASSUMED] Filename is proposed.
- [ ] Shared fault hooks/fixtures at lifecycle boundaries; prefer explicit test-only callbacks over monkeypatching implementation-private line order after the design stabilizes. [ASSUMED]
- [ ] Add test-only bounded inventory/call counters so QUAL-07 behavior is asserted rather than inferred. [VERIFIED: `.planning/REQUIREMENTS.md:73`]

## Security Domain

Security enforcement is enabled at ASVS level 1 in `.planning/config.json`. [VERIFIED: `.planning/config.json` `workflow.security_enforcement=true`, `security_asvs_level=1`]

OWASP ASVS 5.0.0 is the latest stable release identified by the official project; its chapter numbering differs from ASVS 4.x, so references below use version-qualified functional categories. [CITED: https://github.com/OWASP/ASVS]

### Applicable ASVS 5.0 Categories

| ASVS Category | Applies | Standard Control for Phase 3 |
|---------------|---------|------------------------------|
| V1 Encoding and Sanitization | yes | Canonical bounded structured encoding; decode once; reject duplicate/unknown/oversized fields before use. Existing manifest codec is the model. [CITED: https://github.com/OWASP/ASVS/blob/master/5.0/docs_en/OWASP_Application_Security_Verification_Standard_5.0.0_en.json] [VERIFIED: `src/cacheness/storage/manifest.py:80-228`] |
| V2 Validation and Business Logic | yes | Validate state transition, expected generation, checkpoint monotonicity, operation/store ownership, and destructive-action preconditions; reject invalid evidence rather than normalize it. [CITED: https://github.com/OWASP/ASVS/blob/master/5.0/docs_en/OWASP_Application_Security_Verification_Standard_5.0.0_en.flat.json] |
| V5 File Handling | yes | Internally derive generation locators; contain and no-follow every open/write/delete/quarantine; enforce size/count limits before parsing or copying. [CITED: https://github.com/OWASP/ASVS/blob/master/5.0/docs_en/OWASP_Application_Security_Verification_Standard_5.0.0_en.flat.json] [VERIFIED: `src/cacheness/storage/path_security.py:220-314`] |
| V11 Cryptography | yes | Reuse HMAC-SHA256/SHA-256 through the verified key provider; domain-separate operation evidence; fail closed on missing/invalid key or signature. [CITED: https://github.com/OWASP/ASVS/blob/master/5.0/en/0x20-V11-Cryptography.md] [VERIFIED: Phase 2 verification truths 8-10] |
| V15 Secure Coding and Architecture | yes | Bound memory/record/page/key-lock retention, expose unsupported topology instead of false guarantees, and keep authority/recovery responsibilities separated. [CITED: https://github.com/OWASP/ASVS/blob/master/5.0/docs_en/OWASP_Application_Security_Verification_Standard_5.0.0_en.json] |
| V16 Security Logging and Error Handling | yes | Stable typed reasons, operation/generation context without payload/key material leakage, and fail-closed ambiguous publication. [CITED: https://github.com/OWASP/ASVS/blob/master/5.0/en/0x25-V16-Security-Logging-and-Error-Handling.md] |
| Authentication / Session Management / Web Access Control | no | `BlobStore` is an in-process library and this phase adds no user/session/web authorization boundary. Do not invent one. [VERIFIED: `AGENTS.md:42-45,91-95`] |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Forged operation record claims ownership of an unrelated locator | Spoofing / Tampering | Authenticate or strictly validate owner/version/store topology; derive contained locators; require manifest/generation corroboration before mutation. [VERIFIED: D-06/D-15] |
| Losing writer deletes winning generation | Tampering / Denial of Service | Exact expected-generation CAS, operation-specific locators, and revalidation immediately before cleanup. [VERIFIED: D-02/D-11/D-19] |
| Malformed/oversized journal or reconciliation inventory exhausts memory | Denial of Service | Enforce encoded-byte, item, nesting, field, page, and total-run bounds before allocation/deserialization; stable resumable cursors. [VERIFIED: D-05/D-10/D-14; existing bounds `src/cacheness/storage/clear_recovery.py:34-36`] |
| Reconciliation invokes pickle/dill to identify residue | Elevation of Privilege | Never deserialize for ownership/lifecycle decisions; payloads are opaque and trusted only after normal authenticated read checks. [VERIFIED: D-16; `docs/SECURITY.md:12-23`] |
| Dry-run mutates/quarantines evidence | Tampering / Repudiation | Dedicated non-mutating analysis path; exact pre/post byte/call audit; explicit apply mode and checkpoints. [VERIFIED: D-13/D-14] |
| Cleanup/report logs payload bytes, signing material, or unsafe raw paths | Information Disclosure | Record identifiers, generations, bounded reason codes, and redacted/relative locators only; never payload contents or key bytes. [VERIFIED: D-05; ASVS V16 guidance] |
| Unknown future schema is interpreted as current lifecycle evidence | Tampering | Exact version dispatch; unsupported-version report untouched; no guessed repair. [VERIFIED: D-15/D-16; Phase 2 MIGR-07] |
| Symlink/root replacement redirects deletion/quarantine | Tampering / Elevation of Privilege | Reuse `ManagedFileOps` descriptor/no-follow/root-identity checks for every destructive and quarantine operation. [VERIFIED: `src/cacheness/storage/path_security.py:220-314,421-476`] |
| Unbounded per-key lock registry from attacker-chosen keys | Denial of Service | Refcount entries for active/waiting users and retire at zero; assert registry size returns to baseline after high-cardinality tests. [VERIFIED: D-17] |

### Security-Specific Prohibitions

- Do not use operation evidence as an alternate read authority. [VERIFIED: D-03]
- Do not parse handler payloads or filename suffixes to infer ownership. [VERIFIED: D-06/D-16]
- Do not quarantine when the backend cannot perform a contained/backend-owned safe move; report untouched. [VERIFIED: D-15]
- Do not log secret material, payload contents, or unbounded attacker-controlled metadata. [VERIFIED: D-05; ASVS V16]
- Do not weaken Phase 2 authentication/locator/digest ordering to make recovery easier. [VERIFIED: Phase 2 verification truths 5-10]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Suggested module/type names (`lifecycle.py`, `operation_record.py`, `operation_repository.py`, `reconciliation.py`, `coordination.py`, `ManifestExpectation`) are implementation placeholders. | Recommended Project Structure; Code Examples | Low: CONTEXT explicitly delegates names; planner may rename without changing responsibilities. |
| A2 | An exact canonical-record digest/revision should accompany expected generation internally to prevent same-generation metadata lost updates. | Pattern 2 | Medium: if metadata updates are changed to allocate a new generation, the extra digest may be redundant; if omitted while same-generation patches remain, lost updates are possible. |
| A3 | A dynamically retained refcounted per-key registry is preferable to striping. | Pattern 7 | Low/medium: behavior is locked, implementation is discretionary; a carefully designed stripe scheme might be accepted but can serialize distinct keys. |
| A4 | New test filenames and their exact CLI selectors will follow the five-file Wave 0 layout shown. | Validation Architecture | Low: planner may consolidate files; requirement-to-test coverage and commands must be updated together. |
| A5 | Shared lifecycle fault hooks will be exposed as explicit test-only callbacks/fixtures after seam design stabilizes. | Wave 0 Gaps | Low: monkeypatch seams can work, but stable hooks make race/fault tests less coupled to line order. |
| A6 | `delete()` preserves `False` for true absence while treating that result as an idempotent non-error. | Open Questions — RESOLVED | Resolved by Plan 03-04 and D-09 compatibility coverage. |
| A7 | `LifecycleLimits` makes tombstone/grace/page/close policy explicit, finite, validated, and test-derived. | Open Questions — RESOLVED | Resolved by Plans 03-03 and 03-09 with deterministic boundary tests. |

No retention/grace/page-size/retry-delay numeric default is assumed here. CONTEXT delegates bounded values, and STATE explicitly says tombstone retention and orphan grace must be derived from fault/crash testing. [VERIFIED: `.planning/STATE.md:143-147`; `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:76-80`]

## Open Questions — RESOLVED

1. **[RESOLVED] What should `delete()` return when the key is already absent?**
   - What we know: D-09 calls repeated delete “successful”; the current public docstring says `True if deleted, False if not found`. [VERIFIED: CONTEXT D-09; `src/cacheness/storage/blob_store.py:594-611`]
   - Resolution: Preserve `False` for true absence as the compatible boolean meaning “nothing removed in this call,” while treating it as an idempotent non-error. Plan 03-04 locks this behavior and tests absent, resumed-tombstone, deleted, and conflict outcomes separately. [RESOLVED: Plan 03-04; D-09]
   - Recommendation adopted: Preserve `False` for true absence as the legacy boolean meaning “nothing removed in this call,” while documenting/testing that it is an idempotent non-error. Use a richer internal result for absent/resumed/deleted/conflict. [RESOLVED]

2. **[RESOLVED] How much conditional-publication implementation belongs in Phase 3 versus Phase 4?**
   - What we know: Phase 3 requires backend CAS as its correctness boundary, while Phase 4 owns full JSON/memory/SQLite/PostgreSQL parity and capability declarations. [VERIFIED: D-18; deferred scope]
   - Resolution: Phase 3 implements and independently tests truthful exact-record CAS for the exact local repositories admitted by Phase 2—JSON, memory, and SQLite. Registered/custom/PostgreSQL capability composition remains Phase 4 scope. [RESOLVED: Plans 03-01 and 03-02; D-02/D-18]
   - Recommendation: Implement truthful CAS for the exact local repositories already admitted by Phase 2 (JSON, memory, SQLite) and define the backend-neutral protocol now; defer registered/custom/PostgreSQL capability composition to Phase 4. Otherwise Phase 3 cannot validate independent-instance races on real direct `BlobStore` paths. [VERIFIED: current admitted set `src/cacheness/storage/manifest_repository.py:351-366`; deferred Phase 4]

3. **[RESOLVED] What payload inventory can Phase 3 reconciliation truthfully claim?**
   - What we know: D-06 forbids filename inference; full payload-backend listing/capability parity is Phase 5. [VERIFIED: D-06; deferred scope]
   - Resolution: Phase 3 reconciles stable manifest pages plus authenticated operation-record pages. A payload inventory that cannot prove ownership is represented as a blocked/unsupported finding and remains untouched; Phase 3 never scans compatibility payload names or claims Phase 5 inventory capability. [RESOLVED: Plans 03-05 and 03-07; D-13-D-16]
   - Recommendation: Reconcile manifests plus authenticated operation records in Phase 3 and add a narrow bounded local owned-locator inventory only if ownership can be proven without filename guessing. Otherwise emit a blocked/unsupported inventory finding and let Phase 5 fill the payload contract. Never scan the repository's existing 76 candidate-named compatibility payloads as Phase 3 residue. [VERIFIED: D-13-D-16; runtime inventory]

4. **[RESOLVED] What are the default tombstone retention, orphan grace, reconciliation page size, and close wait policy?**
   - What we know: Values must be bounded/deterministic; STATE requires retention/grace to come from fault/crash testing. [VERIFIED: discretion; `.planning/STATE.md:143-147`]
   - Resolution: `LifecycleLimits` makes every limit finite, explicit, validated, and caller-configurable where operational policy is required. Defaults are selected and documented from deterministic page/call-count, frozen-clock, fault, and close-deadline tests; retention is semantic (until exact terminal cleanup), not an age-only evidence purge. [RESOLVED: Plan 03-03 and Plan 03-09]
   - Recommendation adopted: Make limits injectable in tests, choose documented project defaults from deterministic fault/call-count/deadline evidence, and expose policy through the frozen `LifecycleLimits` boundary. [RESOLVED]

5. **[RESOLVED] What might this research have missed?**
   - Resolution: Plans require independent-instance CAS tests, exact-record expectations for same-generation metadata patches, and one lifecycle/recovery engine with the Phase 1 clear coordinator retained only as an exact legacy-evidence adapter. These requirements close the three residual risks below. [RESOLVED: Plans 03-02, 03-04, 03-05, and 03-06; D-02/D-04/D-18]
   - The largest residual risk is a backend-local CAS implementation that is atomic inside one Python object but not across independent instances/processes. The plan checker should require independent-instance tests and refuse topology claims that cannot pass them. [VERIFIED: D-18]
   - A second risk is treating metadata-only same-generation updates as irrelevant to CAS. Plan 03-02 adopts exact-record expectations for same-generation metadata patches. [RESOLVED]
   - A third risk is retaining the Phase 1 clear coordinator alongside the new engine so two journals can both claim one operation. The plan should choose one authority/recovery engine and provide only a compatibility adapter for old clear evidence. [VERIFIED: current clear-only coordinator plus Phase 3 ownership boundary]

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| CPython | library/runtime tests | ✓ | system `python3` 3.12.1; uv environment 3.13.3 | Verify 3.11+ matrix later in Phase 8; Phase 3 tests run in uv environment. [VERIFIED: environment probe; `pyproject.toml:9`] |
| uv | dependency/test runner | ✓ | 0.12.6 | Direct `.venv/bin/python -m pytest` only for diagnosis; checked commands should use uv. [VERIFIED: environment probe; `AGENTS.md:35-41`] |
| pytest | validation | ✓ | 8.4.1 | None needed. [VERIFIED: environment probe; `pyproject.toml:68-73`] |
| Ruff | changed-file lint | ✓ through uv lock/environment | locked 0.12.9 per project instructions | Run only targeted changed files during phase; repository baseline is not clean. [VERIFIED: `AGENTS.md:46-51,127-153`] |
| POSIX advisory locking | current JSON/local cross-process admission reference | ✓ on this macOS host | stdlib `fcntl.flock` | A topology without the required primitive must fail typed; do not downgrade to a process-only lock. [VERIFIED: `src/cacheness/storage/clear_recovery.py:74-83,149-195`; D-18] |
| SQLite | local CAS reference adapter | ✓ through SQLAlchemy/uv environment | SQLAlchemy 2.0.43 locked; SQLite runtime supplied by Python | In-memory repository for unit tests, but not a substitute for SQLite transaction tests. [VERIFIED: `AGENTS.md:44-65`] |
| PostgreSQL service | deferred backend CAS/matrix | not required for Phase 3 | — | Phase 4/5 real-service coverage. [VERIFIED: deferred scope] |
| AWS S3 | deferred payload lifecycle/matrix | not required for Phase 3 | — | Phase 5 real AWS behavior. [VERIFIED: deferred scope] |

**Missing dependencies with no fallback:** None for Phase 3's local reference implementation. [VERIFIED: environment probe and phase boundary]

**Missing dependencies with fallback/deferred owner:** PostgreSQL and AWS S3 are intentionally deferred, not Phase 3 blockers. [VERIFIED: `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md:119-124`]

## Sources

### Primary (HIGH confidence)

- `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md` — locked authority, evidence, reconciliation, coordination, and scope decisions.
- `.planning/REQUIREMENTS.md` and `.planning/ROADMAP.md` — STOR-03 through STOR-07 and observable success criteria.
- `.planning/phases/02-canonical-storage-and-integrity-contract/02-VERIFICATION.md` — verified canonical authority/read ordering, native payload ownership, and typed error boundary.
- `src/cacheness/storage/blob_store.py` — current direct lifecycle, failure cleanup, delete/clear/close, and read seams.
- `src/cacheness/storage/manifest.py` — exact canonical fields and verbatim lifecycle-state values `"prepared"`, `"committed"`, `"replacing"`, `"tombstoned"`, `"conflicted"`. [VERIFIED: `src/cacheness/storage/manifest.py:238-310`]
- `src/cacheness/storage/manifest_repository.py` — current exact raw adapter protocol and local persistence transactions.
- `src/cacheness/storage/guarded_handler_io.py` and `src/cacheness/storage/path_security.py` — validated stage identity, one private snapshot, atomic/durable/exclusive contained I/O.
- `src/cacheness/storage/clear_recovery.py` and `tests/test_clear_recovery.py` — prepared/committed recovery precedent, strict bounds, fault/crash/admission tests.
- `src/cacheness/error_handling.py` and `src/cacheness/storage/read_contract.py` — public typed reasons/categories.
- `AGENTS.md`, `pyproject.toml`, `.planning/config.json`, `docs/SECURITY.md` — project constraints, runtime/test configuration, security enforcement, trusted-payload boundary.
- Read-only runtime inventory and focused pytest probe performed 2026-08-30.

### Secondary (MEDIUM confidence)

- [Python `threading` documentation](https://docs.python.org/3/library/threading.html) — lock semantics/fairness, conditions, events, barriers, and bounded waits.
- [OWASP ASVS official project](https://github.com/OWASP/ASVS) — latest stable 5.0.0 and version-qualified references.
- [OWASP ASVS 5.0 machine-readable requirements](https://github.com/OWASP/ASVS/blob/master/5.0/docs_en/OWASP_Application_Security_Verification_Standard_5.0.0_en.flat.json) — current chapter/category mapping.
- [OWASP ASVS 5.0 Cryptography](https://github.com/OWASP/ASVS/blob/master/5.0/en/0x20-V11-Cryptography.md) — fail-secure cryptographic controls.
- [OWASP ASVS 5.0 Logging and Error Handling](https://github.com/OWASP/ASVS/blob/master/5.0/en/0x25-V16-Security-Logging-and-Error-Handling.md) — fail-closed error handling and safe logging.

### Tertiary (LOW confidence)

- None used as factual authority. Assumptions are isolated in the Assumptions Log.

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — no new dependency; versions and contracts were read from `pyproject.toml`, uv environment, and current source.
- Architecture: HIGH — derived from locked decisions, current source-of-truth seams, and passing Phase 2 verification.
- Lifecycle failure/recovery pattern: HIGH — authority rules are locked and analogous clear fault/reopen behavior is already extensively tested.
- Reconciliation API names/report encoding: MEDIUM — behavior/fields are locked; exact internal names and codec remain delegated.
- Cross-process JSON/SQLite CAS details: MEDIUM — the contract is locked and current storage primitives are known, but Phase 3 must prove exact adapter implementations with independent-instance tests.
- Security mapping: MEDIUM — mapped to official ASVS 5.0 sources via the research seam; ASVS is web-application-oriented while Cacheness is an in-process library.

**Research date:** 2026-08-30
**Valid until:** 2026-09-29 (30 days; codebase architecture is stable enough for planning, but re-check after any Phase 3 precursor changes)
