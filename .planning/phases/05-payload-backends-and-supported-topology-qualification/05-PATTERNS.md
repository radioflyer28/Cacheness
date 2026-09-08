# Phase 5: Payload Backends and Supported Topology Qualification - Pattern Map

**Mapped:** 2026-09-08
**Files analyzed:** 18 planned source, test, configuration, and documentation files
**Analogs found:** 18 / 18 (all have a strong role or contract analog)

This map is for the pre-production cutover. Runtime compatibility adapters and
historical constructor/selector shims are not patterns to copy. Explicit
schema, manifest, and payload-format version checks remain required so Phase 7
can perform stopped-worker migration or rebuild.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `src/cacheness/storage/backends/s3_backend.py` | component / payload adapter | streaming file-I/O | `src/cacheness/storage/backends/blob_backends.py` (`FilesystemBlobBackend`, `InMemoryHandlerIO`) and `src/cacheness/storage/guarded_handler_io.py` | exact seam, different transport |
| `src/cacheness/storage/backends/postgresql_lifecycle_authority.py` | service / authority adapter | transactional CRUD + paged reads | `src/cacheness/storage/sqlite_lifecycle_authority.py` and `src/cacheness/storage/memory_lifecycle_authority.py` | exact semantic contract |
| `src/cacheness/storage/composition.py` | composition/config component | request-response / validation | existing `RoleRegistry` and `StoreTopology` | exact extension point |
| `src/cacheness/storage/backends/__init__.py` | package barrel / registration | construction | existing conditional S3 export | role-match |
| `tests/contracts/test_payload_generation_io.py` | contract test | streaming/file-I/O | `tests/test_blob_store_read_contract.py`, `tests/test_blob_store_integrity.py` | exact contract style |
| `tests/contracts/test_lifecycle_authority.py` | contract test | transactional CRUD | `tests/test_lifecycle_authority_contract.py` | exact semantic cases |
| `tests/contracts/test_topology_lifecycle.py` | contract test | request-response lifecycle | `tests/test_blob_store_atomic_lifecycle.py` | exact lifecycle assertions |
| `tests/test_payload_faults.py` | fault-injection test | event-driven/file-I/O | `tests/_lifecycle_test_support.py` and `tests/test_blob_store_reconciliation.py` | exact deterministic-fault style |
| `tests/test_supported_topologies.py` | matrix/validation test | request-response | `tests/test_topology_capabilities.py` | exact matrix style |
| `tests/integration/test_postgresql_authority.py` | live integration test | transactional CRUD / contention | `tests/test_sqlite_lifecycle_authority.py` | role-match, live service |
| `tests/integration/test_s3_generation.py` | live integration test | streaming object I/O | `tests/test_s3_blob_backend.py` | role-match, real AWS only |
| `tests/integration/test_remote_topology.py` | live integration test | multi-client lifecycle | `tests/test_blob_store_concurrency.py` plus authority contracts | role-match |
| `tests/qualification/conftest.py` | fixture/provider | external service setup/cleanup | `tests/test_s3_blob_backend.py` fixtures and `tests/conftest.py` | role-match |
| `tests/qualification/test_live_evidence.py` | qualification/evidence test | batch/report transform | `tests/test_phase3_release_evidence.py` | role-match |
| `tests/qualification/run_remote_qualification.py` | runner/utility | batch/event-driven | `tests/_lifecycle_test_support.py` subprocess helpers | partial analog |
| `pyproject.toml` | config | test discovery | existing `[tool.pytest.ini_options]` markers | exact extension point |
| `docs/CATALOG_AND_TOPOLOGY.md` | documentation | transform/report | existing role and projection matrix text | exact documentation seam |
| `docs/STORAGE_INITIALIZATION.md` | documentation | request-response guidance | existing SQLite initialization and failure table | exact documentation seam |

## Pattern Assignments

### `src/cacheness/storage/backends/s3_backend.py` (payload adapter, streaming file-I/O)

**Analogs:** `src/cacheness/storage/backends/blob_backends.py` lines
175-270 and 348-425; `src/cacheness/storage/guarded_handler_io.py` lines
148-185 and 411-445.

The class may retain boto3 construction/credential-chain conventions from the
current file (lines 38-147), but the current direct CRUD behavior is not a
pattern to preserve: it overwrites keys (lines 170-204), buffers reads (lines
206-229), treats ETag as integrity (lines 369-408), accumulates all listing
pages and turns errors into `[]` (lines 438-464), and accepts a mismatched
bucket with only a warning (lines 466-491).

**Imports and optional dependency pattern** (current S3 lines 30-47): keep
lazy/guarded boto3 imports and a clear install-oriented `ImportError`. Use
typed botocore errors at the adapter boundary; do not log credential values or
convert service errors to misses.

**Participant and guarded-I/O pattern** (`blob_backends.py:264-266`):

```python
topology_capabilities = {
    "durable": True,
    "process_scope": "multi_host",
    "host_scope": "multi_host",
    "immutable_generations": True,
    "streaming": True,
    "listing": True,
}

def materialize_handler_io(self) -> GuardedHandlerIO:
    return S3HandlerIO(self, ...)
```

The exact class name is planner discretion, but it must implement only the
five methods validated by `BlobStore._materialize_authority_store()`
(`src/cacheness/storage/blob_store.py:690-708`): `stage`,
`publish_generation`, `open_snapshot`, `delete_or_prove_absent`, and `close`.
`publish_generation` must use unique generation locators and conditional
single/multipart creation. `open_snapshot` must HEAD and stream to one
mode-0600 private file, enforce byte/work limits, close the response body, and
return a `GuardedReadSnapshot` only after the copy. Verify SHA-256 and size in
the existing engine before handler access; ETag is diagnostic only.

**Containment and error pattern:** copy the fail-closed boundary in
`guarded_handler_io.py:423-442` and `path_security` helpers. Parse only a
normalized relative locator below the configured bucket/prefix. Reject bucket
swaps, prefix escapes, traversal-like components, oversized locators, malformed
service responses, and permission failures. Distinguish confirmed absence,
conditional conflict, retryable/ambiguous service failure, and configuration
failure with domain errors and preserved causes.

**Listing pattern:** expose one bounded continuation-token page, not a
bucket-sized list. The page must enforce object/byte/page/work bounds and
surface list errors. Listing is evidence for reconciliation; it never
authorizes a read, promotion, rollback, or delete.

**Forbidden duplication:** this adapter must not call `LifecycleAuthority`,
`prepare_mutation`, `promote_mutation`, `abort_mutation`, or implement a second
put/get/reconcile sequence. The engine in
`src/cacheness/storage/lifecycle.py:186-360` remains the only ordering owner.

---

### `src/cacheness/storage/backends/postgresql_lifecycle_authority.py` (authority adapter, transactional CRUD)

**Analogs:** `src/cacheness/storage/sqlite_lifecycle_authority.py` lines 1-32,
90-148, 498-519, 607-637, 944-1057, 1101-1175, 1212-1377, and
1489-1588; `src/cacheness/storage/memory_lifecycle_authority.py` lines 41-109,
148-239, and 241-269.

**Contract/import pattern:** import the semantic dataclasses from
`lifecycle_authority.py:28-75,79-190` and implement the complete
`LifecycleAuthority` protocol at `lifecycle_authority.py:284-383`. Do not
expose SQL sessions/tables to `BlobStore`; callers receive `EntrySnapshot`,
`PreparedMutation`, `PromotionResult`, `ReconciliationPage`, and other
semantic types only.

**Initialization/version pattern:** follow SQLite's explicit `initialize()`
and read-only validation boundary (`sqlite_lifecycle_authority.py:498-519`
and `906-942`). Use a dedicated schema/store-identity record, exact current
schema version, idempotent explicit initialization, and no DDL or migration in
ordinary construction/open. Older, newer, malformed, or foreign state raises
`CacheBlobMigrationRequiredError`, preserving future Phase 7 offline migration
and rebuild tooling.

**Connection/transaction pattern:** use a connection factory or pool with one
transaction-scoped connection per operation, constructed safely after fork.
Adapt SQLite's bounded `_transaction()` shape (`1101-1175`): start one short
transaction, run the semantic callback, commit, rollback every pre-commit
failure, and translate driver errors with operation/stage context. Use
parameterized values and `psycopg.sql.Identifier` for dynamic schema
identifiers; never interpolate run IDs or values into SQL. Apply transaction-
local lock/statement timeouts.

**Exact CAS pattern:** copy the SQLite expectation and mutation flow
(`1212-1266`, `1268-1298`, `1304-1382`): operation identity is idempotent;
expected lineage/revision/generation/digest are checked exactly; promotion
uses a conditional write/`RETURNING`; zero rows is
`CacheBlobLifecycleConflictError`; canonical manifest, entry, revision,
mutation state, and cleanup debt commit together in the authority transaction.
Use PostgreSQL `INSERT ... ON CONFLICT` for unique-key preparation/upsert where
appropriate, not an ORM lifecycle layer.

**Progress/error pattern:** map SQLSTATE serialization failures, deadlocks,
lock/statement/connection timeouts to a typed retryable lifecycle outcome with
the original exception, operation stage, and bounded context. Do not retry
indefinitely and do not require every contender to succeed. Preserve the
existing domain hierarchy (`src/cacheness/error_handling.py:346-434`) and
`raise ... from error` conventions.

**Paging/recovery pattern:** use the SQLite keyset catalog page
(`1489-1588`) and reconciliation snapshot/page methods as semantic models,
but query PostgreSQL in bounded pages. Store operation intent and cleanup debt
in the authority. Never inspect S3 listing to decide canonical membership or
revocation.

**Forbidden duplication:** PostgreSQL owns transactional authority state only;
it must not publish/delete S3 objects, stage handlers, verify payload files, or
reproduce `AuthorityLifecycleEngine` ordering. It is not a projection adapter.

---

### `src/cacheness/storage/composition.py` (composition/config component, validation)

**Analog:** existing `RoleRegistry` and `StoreTopology`.

Use the immutable dataclass and validation conventions at
`composition.py:91-192` and `498-575`. Extend builtin registration near
`347-397` and capability declarations near `724-778`, then add a small
declarative qualified-profile allow-list. Profile identity must be an explicit
role/participant identity, not an exact concrete-class check.

The three qualified profile records are memory/memory, SQLite/filesystem, and
PostgreSQL/AWS-S3. Registration or direct construction alone remains
constructibility, not support. Reject every unqualified cross-pair before
payload/authority I/O with `CompositionValidationError` or
`CapabilityRequirementError`; do not warn, silently downgrade, or infer a
Cartesian matrix. Add the PostgreSQL branch to
`allowed_progress_outcomes()` (`800-808`) and correct SQLite's truthful
portable-query declaration.

Preserve ownership/unwind behavior: resolve records owned participants,
validates capabilities before named factories perform I/O, and closes owned
participants in reverse order (`525-567`, `700-704`). Do not add a queue,
distributed lock, readiness registry, second authority, or lifecycle method
sequence here.

---

### `src/cacheness/storage/backends/__init__.py` (package barrel)

**Analog:** current conditional S3 export at `backends/__init__.py:8-41`.

Re-export the qualified PostgreSQL authority only when its dependency is
available, while keeping the package import usable without psycopg. Do not
register PostgreSQL as a supported profile merely because it imports; the
composition qualification record and live evidence gate remain separate.

---

### `tests/contracts/test_payload_generation_io.py` (contract test, streaming/file-I/O)

**Analogs:** `tests/test_blob_store_read_contract.py` and
`tests/test_blob_store_integrity.py`; private staging assertions in
`tests/test_blob_store_composition.py:434-487`.

Use pytest fixtures and parameterization over memory, filesystem, and S3/fake
participants. Assert integrity and bounds separately from progress and
performance. Required cases include immutable publication, contained private
snapshot, SHA-256/size verification before handler access, exact deletion or
absence proof, bounded listing, malformed locator rejection, and service-error
propagation. Do not assert universal contender success or use sleep-based
ordering.

---

### `tests/contracts/test_lifecycle_authority.py` (authority contract, transactional CRUD)

**Analog:** `tests/test_lifecycle_authority_contract.py:205-368` and
`tests/test_sqlite_lifecycle_authority.py`.

Parameterize the semantic authority contract for memory, SQLite, and the
PostgreSQL implementation. Copy the transition vocabulary from
`test_lifecycle_authority_contract.py:319-368`: create, overwrite, exact stale
expectation conflict, idempotent operation replay, verification, promotion,
debt, deletion, bounded catalog/reconciliation paging, explicit close, and
schema-version rejection. Keep only current-format behavior; remove tests
whose sole purpose is preserving old constructors, old selectors, or old
layouts. Progress tests accept `success`, exact `conflict`, or declared
retryable outcomes.

---

### `tests/contracts/test_topology_lifecycle.py` (contract test, lifecycle request-response)

**Analog:** `tests/test_blob_store_atomic_lifecycle.py` and the public facade in
`src/cacheness/storage/blob_store.py:360-424,519-653`.

Drive all supported profiles through the same `BlobStore` lifecycle and assert
the same safety/recovery vocabulary: old or new complete generation only,
authority promotion as visibility, durable cleanup debt after post-commit
delete failure, and deterministic reconciliation. The test must not reach
adapter tables or build a second lifecycle sequence.

---

### `tests/test_payload_faults.py` (fault test, event-driven/file-I/O)

**Analogs:** `tests/_lifecycle_test_support.py:25-70` and
`tests/test_blob_store_reconciliation.py`.

Use `BoundaryHooks`/`InjectedLifecycleFault` to inject failures before/after
stage, publication, verification, promotion, delete, and ambiguous S3
completion. Follow `test_lifecycle_authority_contract.py:418-483` for checking
rollback and exact-operation recovery. Never use timing races, arbitrary
polling, or a new production coordination seam merely to make a fault test
pass.

---

### `tests/test_supported_topologies.py` (matrix test, validation)

**Analog:** `tests/test_topology_capabilities.py:62-161`.

Make the single support-profile table drive positive profile construction,
capability/progress reporting, and negative cross-pair rejection. Use explicit
participant identities and prove rejection occurs before staging/I/O. Include
JSON only as a derived projection and assert it cannot authorize lifecycle
operations. Do not test a participant's importability as support evidence.

---

### Live integration and qualification files (integration/fixture/runner)

**Analogs:** `tests/test_s3_blob_backend.py:7-101` for optional dependency
fixtures and `tests/test_phase3_release_evidence.py` for sanitized evidence.

`tests/integration/test_postgresql_authority.py` should use a real PostgreSQL
server, distinct test-owned schema/table namespace, external credentials, and
bounded idempotent cleanup. Exercise transactions, constraints, CAS,
SQLSTATE/timeout mapping, cross-connection contention, explicit initialization,
and schema mismatch behavior.

`tests/integration/test_s3_generation.py` should use real Amazon S3, not Moto
or an inferred compatible endpoint, with a unique managed prefix. Exercise
conditional single/multipart generation creation, ambiguous completion
classification, bounded snapshot/list/delete, IAM/credential boundary, and
manifest digest/size verification. Moto remains a contract/fault test only.

`tests/integration/test_remote_topology.py` should construct two independent
clients over the same PostgreSQL/S3 namespace and externally supplied shared
manifest signer, then test cross-client read, exact CAS conflict, recovery,
and cleanup debt.

Fixtures and runner should record only sanitized version/region/service/result
metadata. On missing endpoint or credentials, the qualification command must
write `UNAVAILABLE` or `NOT_QUALIFIED` and exit non-success; it must not turn
the live gate into a passing skip. Cleanup may touch only the exact run schema
and S3 prefix, and must not emit connection URLs, secrets, keys, tokens,
payload contents, or general bucket inventory.

---

### `pyproject.toml` (test configuration)

**Analog:** existing marker declarations at `pyproject.toml:84-101`.

Add strict-marker declarations for the live qualification marker(s), keeping
ordinary local runs able to deselect them while the qualification runner always
executes them. Do not add a dependency solely for this phase; use locked
psycopg/boto3/Moto extras and document unavailable live services explicitly.

---

### `docs/CATALOG_AND_TOPOLOGY.md` and `docs/STORAGE_INITIALIZATION.md`
 (documentation, transform/report)

**Analogs:** `docs/CATALOG_AND_TOPOLOGY.md:3-44,90-117` and
`docs/STORAGE_INITIALIZATION.md:1-8,30-48,80-110`.

Update the exact three-row support matrix, coordination scope, durability and
atomicity boundary, declared progress outcomes, required service conditions,
projection role, and unsupported combinations. State that PostgreSQL promotion
is visibility but S3 effects are outside the PostgreSQL transaction; immutable
generations plus intent/debt provide crash consistency. Keep explicit
initialization and stopped-worker version maintenance instructions. Do not
describe backend method-name parity as guarantee parity, and do not claim live
AWS/PostgreSQL qualification until sanitized evidence exists.

## Shared Patterns

### One lifecycle owner

**Sources:** `src/cacheness/storage/lifecycle.py:72-101,150-184,186-360` and
`src/cacheness/storage/blob_store.py:139-145,690-708`.

The engine sequences stage → prepare intent → publish immutable candidate →
verify private snapshot → record proof → promote authority → settle cleanup
debt. Payload adapters supply mechanics; authorities supply semantic
transactions. Neither adapter may duplicate that sequence.

### Native payload formats and separate signed descriptor

**Sources:** `src/cacheness/storage/guarded_handler_io.py:148-185` and
`src/cacheness/storage/lifecycle.py:214-267,292-333`.

Handlers continue to own NumPy/Blosc2/Parquet/pickle/etc. file formats. The
engine computes SHA-256 and size and signs a `BlobManifest`; no custom Cacheness
header/container is introduced. Payload/manifest version identifiers remain
explicit and unsupported versions fail closed for future Phase 7 tooling.

### Typed failures and preserved causes

**Sources:** `src/cacheness/error_handling.py:238-369,398-447` and
`src/cacheness/storage/sqlite_lifecycle_authority.py:607-637`.

Use domain-specific integrity, backend, conflict, migration-required, cleanup,
and lifecycle-timeout errors. Catch narrow driver/SDK exceptions, attach only
bounded operation/stage context, and use `raise ... from error`. A retryable
contention outcome is valid progress evidence; it is not corruption.

### Deterministic test evidence

**Sources:** `tests/_lifecycle_test_support.py:29-70` and
`tests/test_lifecycle_authority_contract.py:158-197`.

Use named boundary hooks, fault injection, subprocess/process recreation, and
state snapshots. Separate integrity, recovery, progress, and performance
assertions. Never introduce sleeps, universal deadlines, or a production lock/
queue because a stress test expects every contender to succeed.

### Explicit ownership and initialization

**Sources:** `src/cacheness/storage/composition.py:525-567` and
`src/cacheness/storage/sqlite_lifecycle_authority.py:498-519,906-1057`.

Resolve and validate the topology before I/O, close only store-owned
participants, initialize shared durable authority explicitly before workers,
and validate existing schemas read-only. Preserve current-format detection and
future offline migration/rebuild boundaries; no implicit migration.

## No Analog Found

None. The PostgreSQL authority and real-service qualification harness are new
implementations, but their semantic, transaction, fixture, and evidence
patterns have the SQLite, memory, S3, lifecycle, and Phase 3 analogs above.

## Metadata

**Analog search scope:** `src/cacheness/storage/`,
`src/cacheness/storage/backends/`, `tests/`, `docs/`, and `pyproject.toml`
**Files scanned:** 15 source/support files, 7 test/support files, 3 docs/config
files
**Pattern extraction date:** 2026-09-08
