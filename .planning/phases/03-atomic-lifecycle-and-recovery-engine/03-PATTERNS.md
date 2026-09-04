# Phase 3: Atomic Lifecycle and Recovery Engine — Transactional Authority Pattern Map

**Mapped:** 2026-09-04  
**Baseline:** `14a9fc7` (current `main`)  
**Files classified:** 29 proposed create/modify/delete targets  
**Analogs found:** 27 / 29 (the two new authority modules have no exact analogue)

This map is for the transactional-authority replan. It supersedes the historical
`03-PATTERNS.file-native.md` and the file-native scheduler plans. Those artifacts
are useful evidence about failure modes, but are not implementation instructions.
The planner must not preserve the old receipt/event/head/anchor/cursor protocol by
wrapping it in a new facade.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `src/cacheness/storage/lifecycle_authority.py` | service/model contract | CRUD, request-response, event state | `manifest_repository.py`, `manifest.py` | role-match; new seam |
| `src/cacheness/storage/sqlite_lifecycle_authority.py` | persistence adapter | CRUD, batch, transactional | `metadata.py:SqliteBackend`, `manifest_repository.py` | role/data-flow match; implementation must be stdlib sqlite3 |
| `src/cacheness/storage/memory_lifecycle_authority.py` | persistence adapter | CRUD, event state | `metadata.py:InMemoryBackend` | exact role match |
| `src/cacheness/storage/lifecycle.py` | service/orchestrator | streaming/file-I/O/request-response | current lifecycle engine + `guarded_handler_io.py` | role match; remove scheduler protocol |
| `src/cacheness/storage/reconciliation.py` | service/report model | batch, request-response | existing report classes + metadata paging | role match; rewrite source of truth |
| `src/cacheness/storage/blob_store.py` | facade/coordinator | request-response, file-I/O | current BlobStore | exact role; compose authority |
| `src/cacheness/storage/manifest_repository.py` | compatibility adapter | CRUD/CAS | `ManifestExpectation` and protocol in same file | partial; shrink to authority projection/import |
| `src/cacheness/storage/guarded_handler_io.py` | utility/payload boundary | streaming/file-I/O | current guarded handler IO | exact; retain native payload primitives |
| `src/cacheness/storage/path_security.py` | utility/security boundary | file-I/O | `ManagedFileOps` and path helpers | exact for retained primitives |
| `src/cacheness/storage/coordination.py` | utility/concurrency | request-response/event | `KeyCoordinatorRegistry`, `InstanceAdmission` | exact for bounded locks; delete file barrier |
| `src/cacheness/storage/clear_recovery.py` | compatibility detector | read-only file-I/O | `LegacyClearEvidenceAdapter` | delete scheduler; retain detector only if needed |
| `src/cacheness/storage/operation_record.py` | model | scheduler event state | none after redesign | delete/reabsorb only small private value objects |
| `src/cacheness/storage/operation_repository.py` | persistence/service | scheduler event state | none after redesign | delete |
| `src/cacheness/error_handling.py` | error model | request-response | existing `CacheReason`/typed errors | role match |
| `src/cacheness/config.py` | config model | request-response | `LifecycleLimits` | exact |
| `src/cacheness/storage/__init__.py` | package barrel | request-response | current storage barrel | exact |
| `src/cacheness/__init__.py` | package barrel | request-response | current public barrel | exact |
| `tests/_lifecycle_test_support.py` | test utility | setup/fixture | `tests/test_clear_recovery.py` helpers | role match |
| `tests/test_lifecycle_authority_contract.py` | contract test | CRUD, event state | `tests/test_blob_store_read_contract.py` | role/data-flow match |
| `tests/test_sqlite_lifecycle_authority.py` | adapter/integration test | CRUD, transaction, concurrency | `tests/test_sqlite_concurrency.py` | role/data-flow match |
| `tests/test_blob_store_atomic_lifecycle.py` | integration test | file-I/O/request-response | current BlobStore lifecycle tests | exact role |
| `tests/test_blob_store_reconciliation.py` | integration test | batch/reconciliation | `tests/test_blob_store_concurrency.py` | role match |
| `tests/test_blob_store_concurrency.py` | concurrency test | event/request-response | existing same-key tests | exact role; assertions move to public behavior |
| `tests/test_blob_store_close_contract.py` | lifecycle test | event-driven/close | current close contract tests | exact role |
| `tests/test_manifest_repository_cas.py` | compatibility test | CRUD/CAS | existing manifest repository tests | partial; remove scheduler cases |
| `tests/test_clear_recovery.py` | legacy compatibility test | read-only file-I/O | `LegacyClearEvidenceAdapter` | delete scheduler cases, retain only explicit legacy evidence |
| `tests/test_blob_store_read_contract.py` | integration test | request-response/file-I/O | existing read contract | exact |
| `tests/test_config_validation.py` | config test | request-response | existing `LifecycleLimits` tests | exact |
| `benchmarks/lifecycle_authority_benchmark.py` | benchmark | batch/transaction | `benchmarks/threshold_benchmark.py` | role match |

## Pattern Assignments

### `lifecycle_authority.py` (service/model, transactional CRUD)

This is the new deep seam. Define one complete authority interface rather than
exposing separate receipt, inventory, head, anchor, cursor, and pending-control
repositories. It must express intent, promotion/retirement, cleanup debt, clear
snapshots, reconciliation pages, and projection revision as coherent transitions.
The authority owns normalized state; it does not perform serialization, payload
I/O, fsync, verification, or cleanup inside a transaction.

Use the immutable manifest value object as the data-shape precedent. In
`src/cacheness/storage/manifest.py:238-395`, `BlobManifestV1` is frozen, has
explicit fields, and supplies `to_mapping()`, `canonical_bytes()`, and
`from_mapping()`. Preserve that style for immutable authority snapshots and
records. The raw manifest parser at `manifest.py:80-228` is bounded and
fail-closed; authority-facing record decoding should have the same explicit
bounds.

Use `ManifestExpectation` at
`src/cacheness/storage/manifest_repository.py:66-110` as the exact-evidence/CAS
precedent: generation plus SHA-256 of the exact bytes, with a `.matches()`
predicate. Authority transitions should accept expected lineage/generation and
return a deterministic conflict rather than silently overwriting another
operation.

Suggested public families (names may be refined by the planner):

```python
class LifecycleAuthority(Protocol):
    def prepare(self, intent: PrepareIntent) -> PreparedRecord: ...
    def promote(self, token: str, evidence: PromotionEvidence) -> EntryRecord: ...
    def retire(self, key: str, expected: EntryExpectation) -> RetiredRecord: ...
    def record_cleanup_debt(self, debt: CleanupDebt) -> None: ...
    def begin_clear(self, request: ClearRequest) -> ClearSnapshot: ...
    def page_clear(self, snapshot: ClearSnapshot, cursor: ClearCursor) -> ClearPage: ...
    def begin_reconciliation(self, request: ReconciliationRequest) -> ReconciliationRun: ...
    def page_reconciliation(self, run: ReconciliationRun, cursor: RowCursor) -> ReconciliationPage: ...
    def project_json(self, expected_revision: int | None = None) -> ProjectionResult: ...
```

Keep the interface testable with a memory adapter. Do not expose private
`receipt()`/`inventory()` methods as the testing contract.

### `sqlite_lifecycle_authority.py` (stdlib sqlite3 adapter)

The closest useful source is the stdlib connection/query style in
`tests/test_sqlite_concurrency.py:64-118` and the read-only sqlite inspection
connection in `src/cacheness/metadata.py:1441-1475`. The current
`SqliteBackend` (`metadata.py:1345-2087`) is useful for schema and entry
semantics, but do not copy its SQLAlchemy session boundary, WAL mode, or
`synchronous=NORMAL` settings: the new local authority explicitly uses stdlib
`sqlite3`, rollback journal `DELETE`, `synchronous=EXTRA`, foreign keys,
`trusted_schema=OFF`, `application_id`, and `user_version` migration checks.

The old conditional transaction in
`manifest_repository.py:2136-2186` is the control-flow precedent:
`BEGIN IMMEDIATE`, rollback on failure, commit on success. Adapt it to one
short transaction containing only normalized row mutations and authority
metadata. Set a bounded busy deadline and translate lock exhaustion into a
typed domain error. Never hold a transaction while handler serialization,
payload publish, fsync, manifest verification, or cleanup runs.

```python
connection.execute("BEGIN IMMEDIATE")
try:
    # Read expected lineage and mutate authority rows only.
    result = mutate_authority_rows(connection, request)
except Exception:
    connection.rollback()
    raise
else:
    connection.commit()
    return result
```

Use normalized tables for authority metadata, entries (including lineage
token), mutations, cleanup debt, clear runs/targets, reconciliation runs, and
projection state. Add indexes for key, generation, run status, debt status, and
keyset pagination. Schema migration must be explicit and monotonic through
`user_version`; fail closed on an unknown application id/version.

### `memory_lifecycle_authority.py` (in-memory adapter)

`src/cacheness/metadata.py:543-765` is the closest implementation. It uses an
`RLock`, a private dictionary, explicit entry operations, and deep copies during
clear (`metadata.py:746-760`). Reuse the private-state/copy-on-read discipline,
but implement the same public authority transitions and conflict semantics as
SQLite. The lock protects each short transition; it must not be used around
payload I/O. Keep close idempotent as in `InMemoryBackend.close()` at
`metadata.py:762-765`.

### `lifecycle.py` (orchestrator rewrite)

The current payload ordering is valuable. In `src/cacheness/storage/guarded_handler_io.py:133-185`,
`stage()` creates a private temporary output and `publish_generation()` performs
exclusive immutable publication. The existing lifecycle sequence in
`src/cacheness/storage/lifecycle.py:1442-1638` also shows handler stage,
publish, verify, and manifest construction. Retain that native format path but
remove every operation-repository checkpoint, receipt, event, head, anchor,
cursor, or pending-control call.

Implement this bounded sequence:

1. Authority `prepare`: durable intent before the first managed persistent
   payload effect.
2. Handler writes to a private stage (no transaction held).
3. Exclusive immutable candidate publish and fsync.
4. Reopen/read the candidate and verify exact size/digest/manifest/handler.
5. Authority `promote`: commit entry, manifest metadata, dirty projection, and
   cleanup debt in one short transaction.
6. Perform obsolete payload deletion outside the transaction; record or clear
   cleanup debt through idempotent authority transitions.
7. Authority `retire` and close operation state only after deterministic
   reconciliation of failures.

Failure paths must be explicit: if preparation exists but publish fails, record
the terminal failed intent; if publish succeeds but promotion fails, leave
durable cleanup debt and let reconciliation act; never claim committed state
from a payload-only observation.

### `blob_store.py` (facade composition)

Replace current initialization at `src/cacheness/storage/blob_store.py:243-353`:
it presently creates `StoreAdmissionBarrier`, old manifest repository,
`LifecycleEngine`, `Reconciler`, and clear recovery. Construct one
`LifecycleAuthority` (SQLite by default, memory by explicit backend), pass it to
the lifecycle/reconciliation services, and retain only bounded key coordination
and instance admission.

Keep public `put`, `delete`, `clear`, and `reconcile` entry points at
`blob_store.py:405-437`, `584-600`, and `736-777`, but make them delegate to the
authority-backed lifecycle. Preserve authenticated read ordering from
`blob_store.py:1193-1281` and tests in `tests/test_blob_store_read_contract.py`:
committed authority lookup, bounded manifest parse, authenticity, critical
fields, snapshot, digest/size, then handler read. A missing direct key remains
`None`; typed integrity/backend errors remain distinct.

Retain `_delete_or_prove_absent()` (`blob_store.py:1094-1116`) as a useful
idempotent payload cleanup primitive, but its outcome must update authority
cleanup debt rather than a scheduler receipt. Close must drain instance-owned
operations without holding authority transactions, following the event-driven
tests in `tests/test_blob_store_close_contract.py:45-190`.

### `manifest_repository.py` (compatibility projection/import only)

Keep `ManifestExpectation` and the narrow protocol shape as compatibility
helpers. Canonical ownership of committed manifests, generations, and CAS
belongs to the authority. Remove the old SQLAlchemy/file-native inventory,
manifest sidecars, and scheduler tables. If old persisted data must be read,
implement a one-way compatibility projection/import with explicit migration
evidence; it must not become a second source of truth.

### `guarded_handler_io.py` (retain payload primitives)

Reuse the imports and safety boundary at lines 1-29, private staging at
`133-170`, immutable exclusive generation publication at `172-185`, and
no-follow snapshot reads at `394-425`. Native handlers continue to own NumPy,
Blosc2, Parquet, pickle, and other format headers. Do not add a custom payload
header. The authority stores lifecycle metadata beside the native immutable
payload; it does not reinterpret handler bytes.

### `path_security.py` (trim scheduler machinery, retain safety)

Retain `validate_blob_id()`/`resolve_managed_locator()` at
`src/cacheness/storage/path_security.py:642-795`, plus the safe open/read,
exclusive create, fsync directory, durable delete, and `sha256_and_size`
primitives at `1748-1815`, `1984-2129`, and `2239-2266`. These are the payload
security boundary: containment, no symlink traversal, immutable creation, and
exact evidence.

Delete scheduler-only lock/identity/xattr/pending-promotion machinery (for
example `ensure_fixed_lock_file()` around `1359+`, lock identity/registry
helpers, and `promote_durable_pending_control()` around `1837+`). SQLite is the
single local coordination authority; do not recreate a file lock protocol.

### `coordination.py` (bounded process-local coordination)

Delete `StoreAdmissionBarrier` (`coordination.py:354-638`) and all interprocess
file-lock behavior. Retain `KeyCoordinatorRegistry` (`639-693`) for bounded
same-key serialization and `InstanceAdmission` (`696-827`) for ownership,
close-drain, and idempotent release. These locks coordinate in-process work;
they are not a durability or transaction mechanism and must not span payload
I/O longer than the existing operation needs.

### `reconciliation.py` (authority-indexed rewrite)

Reuse the stable public report value objects at
`src/cacheness/storage/reconciliation.py:55-166`:
`ReconciliationStatus`, `ReconciliationAction`, `ReconciliationFinding`, and
`ReconciliationReport.to_dict()/human_summary()`. Replace `_Reconciler`'s
manifest/operation/sidecar/pending-file inventory scan (`306-501`) with pages
from authority-indexed rows and high-water/keyset cursors. A dry run must return
a stable report. Apply mode must persist a resumable run/checkpoint, honor row,
action, byte, and time budgets, revalidate exact evidence immediately before a
destructive delete, and make cleanup-debt resolution idempotent.

### `clear_recovery.py` (delete scheduler; optional legacy detector)

Delete `ClearRecoveryCoordinator` and its JSON journal/admission protocol
(`clear_recovery.py:88-427`). At most, retain the read-only
`LegacyClearEvidenceAdapter` (`875-912`) for an explicit compatibility release
path. New clear uses authority transactions: capture exact generations in a
clear snapshot, page targets by indexed keyset, and retire only when expected
lineage still matches. No JSON journal is a second authority.

### Delete/reabsorb `operation_repository.py` and `operation_record.py`

`operation_repository.py` (about 4,014 lines) and `operation_record.py` (about
1,323 lines) encode the superseded file-native protocol: receipts, inventory,
heads/tails, anchors, pending controls, clear pages, and reconciliation cursors.
Delete them rather than wrapping them. Reabsorb only genuinely reusable small
private value objects into the authority module; do not preserve their public
repository methods or tables.

### Errors, config, and barrels

Extend existing typed reasons in `src/cacheness/error_handling.py:19-63` and
constructors at `272-412`; preserve causes with `raise ... from e` and use
narrow operational exception catches. Add only authority-specific reasons if
existing blob lifecycle/backend/cleanup/reconciliation reasons cannot express
the failure.

Use `LifecycleLimits` in `src/cacheness/config.py:352-400` for caller-owned
bounds (transaction/operation bytes, page/action counts, reconciliation and
close budgets). Do not make payload byte limits a policy for native handler
headers. Export authority adapters and retained public lifecycle types through
`src/cacheness/storage/__init__.py:31-143` and root `src/cacheness/__init__.py:32-109`
only when they are stable API commitments.

## Shared Patterns

### One transactional authority

Every durable lifecycle state transition goes through the same authority
interface and normalized store. A transaction is short and contains row reads,
CAS checks, and row writes only. Payload writes, verification, fsync, and
cleanup are outside transactions and communicate through durable intent,
immutable evidence, and cleanup debt.

### Native payload and retained security boundary

Handlers own their native formats; `guarded_handler_io.py` and
`path_security.py` own staging, containment, no-follow reads, exclusive
immutable publication, digest/size, fsync, and deletion. Authority records
reference payload evidence; they do not add a wrapper/header or parse foreign
format internals.

### Deterministic CAS and idempotency

Use generation/lineage plus exact digest evidence (the
`ManifestExpectation.matches()` pattern). Repeated promotion, retirement,
cleanup-debt resolution, clear-page application, and reconciliation actions must
produce the same result or a typed conflict. Never infer committed state by
scanning payloads.

### JSON is a projection

Reuse the locking/atomic replacement ideas in `metadata.py:1018-1125` and the
copy discipline in `metadata.py:1137-1187`, but write JSON only as a revisioned
projection of authority rows. Use revision-CAS, temp-file + fsync + atomic
replace, and deterministic rebuild; JSON must not be used for lifecycle intent
or recovery truth.

### Close and bounded coordination

Reuse `InstanceAdmission` and close-contract event tests. Close drains owned
operations and makes later calls fail clearly, but never waits while holding a
SQLite transaction or a process-wide file lock. Same-key coordination may
serialize concurrent payload work; distinct keys should overlap subject to
backend limits.

## Test and Benchmark Assignments

Tests should exercise the public authority contract and BlobStore behavior, not
private receipt/inventory methods.

- `test_lifecycle_authority_contract.py`: run the same transition suite against
  memory and SQLite adapters; cover prepare/promote/retire, CAS conflicts,
  cleanup debt, clear snapshots, reconciliation pages, projection revisions,
  close, and idempotent repeats.
- `test_sqlite_lifecycle_authority.py`: inspect with stdlib sqlite3; assert
  schema/version/pragmas, rollback on injected failure, busy deadline, restart
  durability, concurrent same-key conflict, and distinct-key overlap. Do not
  assert the legacy WAL mode.
- BlobStore atomic lifecycle/reconciliation/read/close/concurrency tests:
  preserve native handler and read-order cases, but assert committed public
  outcomes and authority-backed cleanup/reconciliation. Remove scheduler
  receipt/head/anchor/inventory assertions.
- `test_manifest_repository_cas.py` and `test_clear_recovery.py`: retain only
  compatibility projection/legacy-evidence tests; delete file-native scheduler
  cases.
- `tests/_lifecycle_test_support.py`: provide deterministic injected handler,
  fault, clock, and authority fixtures without a broad `conftest.py` change.
- `benchmarks/lifecycle_authority_benchmark.py`: follow
  `benchmarks/threshold_benchmark.py:28-148`; measure authority transaction
  duration, clear snapshot/page duration, busy-wait behavior, reconciliation
  page/action throughput, and same-key versus distinct-key overlap. Check in
  budgets and run acceptance on supported Python 3.11+ as practical.

## No Analog Found

| File | Reason |
|---|---|
| `src/cacheness/storage/lifecycle_authority.py` | No existing deep, normalized transactional lifecycle interface; this is the intended architectural seam. |
| `src/cacheness/storage/sqlite_lifecycle_authority.py` | Existing SQLite metadata is SQLAlchemy/WAL-oriented and is deliberately not a safe implementation template for the new stdlib sqlite3 authority. |

## Historical Evidence Excluded from Current Patterns

The superseded `.file-native.md` plans/patterns, operation repository/record,
clear-recovery journal, file lock identity protocol, sidecar manifest inventory,
and pending promotion controls were scanned only to identify race and recovery
failure modes. They must not be assigned as analogs for new plans. The retained
patterns are the native handler payload boundary, guarded filesystem safety,
manifest parsing/authenticity, typed errors, bounded key coordination, memory
backend copying, JSON atomic projection mechanics, and existing public BlobStore
read/close semantics.

## Metadata

**Analog search scope:** `src/cacheness/storage/`, `src/cacheness/metadata.py`,
`src/cacheness/config.py`, `src/cacheness/error_handling.py`, package barrels,
`tests/`, and `benchmarks/`.  
**Baseline verified:** `git rev-parse --short HEAD` = `14a9fc7`.  
**Pattern extraction date:** 2026-09-04.
