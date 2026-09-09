# Phase 7: Explicit Migration and Rebuild Cutover - Pattern Map

**Mapped:** 2026-09-09  
**Files analyzed:** 18 new/modified files inferred from CONTEXT.md, RESEARCH.md, and the Wave 0 gap list  
**Analogs found:** 18 / 18 (role/data-flow analogs; no exact migration implementation exists)

This map treats Phase 7 as a new offline maintenance boundary above `BlobStore`.
The work directory and its JSON evidence coordinate the stopped-worker workflow
only; they must never become lifecycle authority. Authority promotion remains the
only visibility point. Do not add migration behavior to ordinary constructors,
`BlobStore.initialize()`, `UnifiedCache`, reconciliation, or payload listings.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `src/cacheness/storage/migration.py` | service, model | batch, file-I/O, request-response | `src/cacheness/storage/reconciliation.py` (`_AuthorityReconciler`) and `src/cacheness/storage/projections.py` (`ProjectionController`) | role-match |
| `src/cacheness/storage/migration_evidence.py` | utility, model | file-I/O, transform | `src/cacheness/storage/reconciliation.py` resume envelope plus `src/cacheness/storage/manifest.py` canonical encoding | role-match |
| `src/cacheness/storage/migration_authority.py` | protocol, model | request-response / CRUD | `src/cacheness/storage/lifecycle_authority.py` | role-match |
| `src/cacheness/storage/blob_store.py` | service/composition seam | request-response, file-I/O | `BlobStore.initialize`, `_authenticated_authority_manifest`, `_materialize_authority_store` | role-match |
| `src/cacheness/storage/lifecycle_authority.py` | protocol, value objects | request-response / CRUD | existing `LifecycleAuthority` contracts | exact role |
| `src/cacheness/storage/sqlite_lifecycle_authority.py` | authority adapter | CRUD, batch | `catalog_page`, `_transaction`, `_initialize_schema` | exact role |
| `src/cacheness/storage/backends/postgresql_lifecycle_authority.py` | authority adapter | CRUD, batch | `catalog_page`, `_transaction`, `initialize` | exact role |
| `src/cacheness/handlers.py` | registry/contract | transform, file-I/O | `resolve_payload_contract`, `register_handler` | exact role |
| `src/cacheness/storage/__init__.py` | public API barrel | request-response | existing storage exports | exact role |
| `tests/test_migration_inspection.py` | test | batch, request-response | `tests/test_stored_compatibility.py`, `tests/test_blob_store_reconciliation.py` | role-match |
| `tests/test_migration_plan_contract.py` | contract test | transform | `tests/test_blob_store_reconciliation.py`, `tests/test_lifecycle_authority_contract.py` | role-match |
| `tests/test_migration_run_evidence.py` | fault/contract test | file-I/O, request-response | `tests/test_blob_store_reconciliation.py`, `tests/test_phase3_windows_qualification_attestation.py` | role-match |
| `tests/test_migration_cutover.py` | integration/fault test | CRUD, file-I/O | `tests/test_blob_store_atomic_lifecycle.py`, `tests/test_phase3_local_workflows.py` | role-match |
| `tests/test_rebuild_workflow.py` | integration/contract test | transform, file-I/O | `tests/test_projection_sql_atomicity.py`, `tests/test_handler_registration.py` | role-match |
| `tests/test_lifecycle_authority_contract.py` | authority contract extension | CRUD, request-response | existing authority transition/fault tests | exact role |
| `tests/contracts/test_postgresql_lifecycle_authority.py` | topology contract extension | CRUD, request-response | existing DB-API fake/SQLSTATE contract | exact role |
| `tests/test_stored_compatibility.py` | regression extension | request-response, file-I/O | existing no-implicit-upgrade tests | exact role |
| `docs/BACKEND_SELECTION.md` | documentation | transform/reporting | existing backend/migration section | exact role |

## Pattern Assignments

### `src/cacheness/storage/migration.py` (service/model, batch + file-I/O + request-response)

**Analog:** `src/cacheness/storage/reconciliation.py` (`_AuthorityReconciler`,
lines 165-318, 627-690) and `src/cacheness/storage/projections.py`
(`ProjectionController`, lines 203-305, 360-412).

Use immutable dataclasses/enums for inventory, compatibility matrix, entry
dispositions, plan, run state, receipts, and bounded batch results. Keep
inventory non-mutating and revision-bound. The coordinator owns workflow
sequencing and evidence calls, while the selected authority owns activation,
rollback, and finalize. A same-backend candidate is copied and completely
verified before a single authority activation; never publish per entry.

**Imports and domain model pattern** (`reconciliation.py:7-24`,
`projections.py:9-25`):

```python
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from cacheness.config import LifecycleLimits
from cacheness.error_handling import CacheStorageError
from .lifecycle_authority import ReconciliationSnapshot, ReconciliationWork
```

**Bounded loop / explicit continuation** (`reconciliation.py:185-236`):

```python
deadline = time.monotonic() + self.lifecycle_limits.authority_busy_timeout_seconds
row_budget = self.lifecycle_limits.operation_page_size
while inspected_rows < row_budget and time.monotonic() < deadline:
    page = self.authority.page_reconciliation_work(
        snapshot, mutation_cursor=mutation_cursor, debt_cursor=debt_cursor
    )
    if not page.works:
        break
    # Advance only authority-indexed work; return an authenticated continuation
    # when the bounded budget is exhausted.
```

Migration inventory should use a new raw, bounded keyset authority page at one
captured revision, not `BlobStore.query_catalog()`/`CatalogPage`: the latter
intentionally skips schema-mismatched entries (`catalog.py:780-805`). Every
entry must be authenticated and classified as exactly `migratable`,
`rebuildable`, `blocked`, or `unverifiable`, with stable reasons and aggregate
counts/bytes.

**Derived work remains post-commit** (`blob_store.py:274-301`):

```python
def _run_post_commit_projections(self, receipt: BlobReceipt) -> BlobReceipt:
    outcomes: dict[str, ProjectionOutcome] = {}
    for controller in self._projection_controllers:
        attempt = controller.best_effort(receipt)
        if attempt.outcome is not None:
            outcomes[controller.projection_name] = attempt.outcome
    return receipt.with_projection_outcomes(outcomes)
```

Apply the same ordering to migration: canonical activation completes first;
projection rebuild is separate derived work and cannot gate eligibility or
publication.

**Error translation:** use narrow domain errors and `raise ... from error`, as
`ProjectionController.rebuild()` does (`projections.py:360-412`). A stale
source revision, unsupported dimension, missing evidence, or unverifiable
manifest blocks the action; do not convert it to a cache miss.

### `src/cacheness/storage/migration_evidence.py` (utility/model, file-I/O + transform)

**Analog:** reconciliation resume token (`reconciliation.py:627-690`) plus
canonical manifest encoding/authentication (`manifest.py:289-345,563-625`).

Canonical JSON is the sole plan/evidence representation. Human reports must be
rendered from that model, never generated independently. Bind evidence to
explicit run ID, plan digest, source/destination identities, authority revision,
workflow state, candidate ownership, and completed batch digests. Store key
identity/fingerprint only; never serialize signing key bytes, raw DSNs, or secret
paths.

**Authenticated bounded resume pattern** (`reconciliation.py:649-690`):

```python
encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
signature = hmac.new(
    self.store._authority_manifest_key(initialize_new_store=False),
    self._TOKEN_DOMAIN + encoded,
    hashlib.sha256,
).hexdigest()
return base64.urlsafe_b64encode(encoded + b"." + signature.encode("ascii")).decode("ascii")
```

Reuse the shape—domain-separated HMAC, canonical ordering, bounded parsing,
constant-time compare—but define a migration-specific version/domain and
validate all fields before use. Resume requires an operator-supplied run ID and
evidence path; never search for a latest run or infer state from candidate blob
presence. Missing/corrupt/mismatched evidence fails closed.

**Canonical JSON and duplicate-key rejection** (`manifest.py:348-367,
477-486,563-582`):

```python
record = json.loads(
    raw.decode("utf-8"),
    object_pairs_hook=_reject_duplicate_keys,
    parse_int=_parse_canonical_int,
    parse_float=_reject_float,
    parse_constant=_reject_constant,
)
_validate_canonical_value(record, depth=1, nodes=[0])
encoded = json.dumps(
    record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
).encode("utf-8")
```

Apply size/depth/node bounds and reject duplicate keys, non-finite numbers,
unsafe paths, invalid enums, and malformed digests. Render human text from the
validated model and test that secrets and raw locator material do not appear.

### `src/cacheness/storage/migration_authority.py` (protocol/model, request-response / CRUD)

**Analog:** `LifecycleAuthority` protocol (`lifecycle_authority.py:285-403`).
Do not widen the general protocol with orchestration. Add a narrow semantic
maintenance capability/protocol, implemented by each qualified authority, for
identity snapshot, raw revision-bound inventory, verified whole-store activation,
rollback, and finalize. It should expose transactional primitives, not SQL
tables or backend-specific workflow methods.

**Existing value-object validation pattern** (`lifecycle_authority.py:28-79`):

```python
@dataclass(frozen=True)
class EntryExpectation:
    lineage: int | None
    revision: int | None
    generation: str | None = None
    manifest_digest: str | None = None

    def __post_init__(self) -> None:
        # Bound identifiers and require the expected SHA-256 representation.
        ...
```

**Narrow protocol shape** (ADR/research contract):

```python
class MigrationAuthority(Protocol):
    def identity_snapshot(self) -> AuthorityIdentitySnapshot: ...
    def inventory_page(
        self, cursor: str | None, *, revision: int, limit: int
    ) -> AuthorityInventoryPage: ...
    def activate_verified_candidate(
        self, receipt: VerifiedCandidateReceipt, *,
        expected_source: AuthorityIdentitySnapshot
    ) -> ActivationReceipt: ...
    def rollback_activation(self, receipt: ActivationReceipt) -> RollbackReceipt: ...
    def finalize_activation(self, receipt: ActivationReceipt) -> FinalizeReceipt: ...
```

The snapshot must contain topology/authority kind, capability, independent
authority schema, store identity, and revision. Do not compare only
`STORE_FORMAT_VERSION` or the SQLite field in `StoreVersionDimensions`.

### `src/cacheness/storage/blob_store.py` (service/composition seam, request-response + file-I/O)

**Analog:** `BlobStore.initialize()` (`blob_store.py:333-362`),
`_authenticated_authority_manifest()` (`blob_store.py:817-836`), and
`_materialize_authority_store()` (`blob_store.py:838-860`).

If a seam is needed, expose only read/authentication and payload-participant
operations needed by the offline service. Keep initialization explicitly
current-layout-only and keep handler I/O behind the guarded participant.

```python
@_ordinary_admitted
def initialize(self) -> None:
    """Initialize once before workers start; never upgrade existing schemas."""
    if self._initialized:
        self.lifecycle_authority.preflight_mutation()
        return
    initializer = getattr(self.lifecycle_authority, "initialize", None)
    if callable(initializer):
        initializer()
    self.lifecycle_authority.preflight_mutation()
```

Do not add `migrate`, `rebuild`, adoption, candidate selection, or cleanup to
this ordinary path. Offline actions belong in the new maintenance boundary.

### `src/cacheness/storage/lifecycle_authority.py` (protocol/value objects)

**Analog:** existing authority values and protocol (`lifecycle_authority.py:120-170,285-383`).
Preserve immutable frozen values, bounded text/digests, exact expectations, and
authority-owned state. The maintenance capability must not expose filesystem
paths as visibility state.

```python
@dataclass(frozen=True)
class VerificationProof:
    digest: str
    byte_size: int
    manifest: bytes = b""

    def __post_init__(self) -> None:
        if len(self.digest) != 64 or any(c not in "0123456789abcdef" for c in self.digest):
            raise ValueError("digest must be a lowercase SHA-256 hexadecimal value")
```

Keep candidate/activation/rollback/finalize receipts immutable and corroborated
by source identity/revision. The resolved publication states are `candidate`,
`activated_offline`, `active`, and `rolled_back`; every ordinary worker
open/read/query/mutation entry point fails with the typed offline-decision
outcome while activated-offline, and finalize seals rollback before workers
restart. Narrow maintenance status/receipt methods remain available. These
states belong to this authority—not evidence JSON, a pointer file, a
first-write hook, or a projection.

### `src/cacheness/storage/sqlite_lifecycle_authority.py` (authority adapter, CRUD + batch)

**Analog:** schema bootstrap/validation (`sqlite_lifecycle_authority.py:735-905`),
bounded catalog scan (`1469-1588`), and transaction wrapper (`1101-1150`).

**Fresh schema versus existing schema:**

```python
def initialize(self) -> None:
    """Create/validate this catalog before starting shared workers.

    Existing incomplete or obsolete catalogs are never adopted or upgraded.
    """
    started_at = self._now()
    with self._connection(mutation=True, deadline=self._deadline(None), started_at=started_at):
        pass
```

`_validate_schema()` rejects wrong application ID/version and missing columns;
retain this fail-closed behavior. Only explicit migration may use a new
schema step, and it must be the smallest change required for retained
candidate/activation state.

**Raw keyset paging pattern** (`sqlite_lifecycle_authority.py:1521-1588`):

```python
connection.execute("BEGIN")
revision = connection.execute(
    "SELECT revision FROM authority_state WHERE singleton = 1"
).fetchone()[0]
rows = connection.execute(
    "SELECT key, generation, locator, manifest, manifest_digest, lineage, revision "
    "FROM entries ORDER BY key, generation LIMIT ?", (work_cap + 1,)
).fetchall()
```

Copy the transaction/keyset shape for an administrative raw inventory page,
but do not pass it through schema-filtering `page_from_canonical_scan()`.
Authenticate/classify every snapshot in `migration.py`, preserving unknown
authenticated catalog fields.

**Atomic transaction/error shape** (`sqlite_lifecycle_authority.py:1101-1150`):

```python
try:
    result = callback(connection)
    self._reach_transaction_boundary("authority.transaction.before_commit")
    self._execute_for_stage(connection, "COMMIT", ...)
    self._reach_transaction_boundary("authority.transaction.committed")
    return result
except BaseException:
    if connection.in_transaction:
        connection.execute("ROLLBACK")
    raise
```

Activation/rollback/finalize must be one SQLite authority transaction. Payload
copy and purge are external effects with evidence and retryable cleanup; never
claim SQLite+filesystem ACID.

### `src/cacheness/storage/backends/postgresql_lifecycle_authority.py` (authority adapter, CRUD + batch)

**Analog:** independent authority identity/capability (`postgresql_lifecycle_authority.py:62-69`),
explicit `initialize()` (`419-441`), bounded `catalog_page()` (`986-1075`),
and SQLSTATE handling (`74-80`, `1074-1077`).

```python
POSTGRESQL_AUTHORITY_SCHEMA_VERSION = 3  # current tree before Phase 7
POSTGRESQL_AUTHORITY_CAPABILITY = "postgresql-lifecycle-authority-v3"

def open(self) -> "PostgresqlLifecycleAuthority":
    """Validate an existing authority without creating or upgrading it."""
    self._read_only("schema_validate", self._validate_schema)
    return self
```

Use PostgreSQL's own schema/capability in the compatibility matrix; never infer
it from `sqlite_user_version`. Resolved RQ-01 changes the first release baseline
to schema 4 / `postgresql-lifecycle-authority-v4` and adds the exact authority
publication fields plus candidate/prior rows; schema 3 gains no production
migration edge. Each maintenance primitive runs in one PostgreSQL transaction
and preserves typed retryable outcomes with original causes. Live PostgreSQL/S3
qualification remains Phase 8.

### `src/cacheness/handlers.py` (registry/contract, transform + file-I/O)

**Analog:** `CacheHandler` payload identity (`interfaces.py:112-148`) and
registry resolution (`handlers.py:1437-1473`).

```python
def resolve_payload_contract(
    self, handler_type: str, payload_format: str, payload_format_version: int
) -> CacheHandler:
    """Resolve a declared native handler contract without opening payload bytes."""
    handler = self.get_handler_by_type(handler_type)
    supports_contract = getattr(handler, "supports_payload_contract", None)
    if not callable(supports_contract) or not supports_contract(
        payload_format, payload_format_version
    ):
        raise CacheManifestUnsupportedVersionError(
            "Canonical manifest declares an unsupported native payload contract"
        )
    return handler
```

Resolve against the source store's registered handler set. Exact matching may
permit verified byte copy; format changes require a handler-owned directed edge.
Otherwise rebuild through source handler read and destination `BlobStore` write.
Do not hard-code NPZ, Blosc2, Parquet, pickle, or dill conversion. Preserve
custom handler priority/duplicate-name behavior from `register_handler()`
(`handlers.py:1475-1525`) when adding readable-contract checks.

### `src/cacheness/storage/__init__.py` (public API barrel, request-response)

**Analog:** deliberate re-export barrel (`storage/__init__.py:49-76,114-181`).
Export the supported maintenance symbols through the Python library API and
keep optional PostgreSQL/S3 imports guarded. Resolved RQ-02 adds no CLI or
`[project.scripts]` entry in Phase 7. Any later CLI is a stateless adapter over
the same model and must not implement a second plan/evidence model.

### `tests/test_migration_inspection.py` (test, batch + request-response)

**Analog:** `tests/test_stored_compatibility.py:53-105,108-144` and
`tests/test_blob_store_reconciliation.py:110-163`.

```python
before = _tree_bytes(root)
with pytest.raises(CacheBlobMigrationRequiredError):
    store.initialize()
assert _tree_bytes(root) == before
```

Test non-mutating empty/current/historical inspection, topology-neutral
identity, bounded raw inventory, exact totals, every disposition and stable
reason, and that schema-mismatched entries are not omitted. Snapshot bytes
before/after unsupported inspection to prove no journal, marker, payload, or
key mutation.

### `tests/test_migration_plan_contract.py` (contract test, transform)

**Analog:** `tests/test_blob_store_reconciliation.py:110-160` and
`tests/test_lifecycle_authority_contract.py:81-124`.

Use exact dictionary assertions for canonical plan fields, disposition/reason
enums, compatibility dimensions, source/destination identities, counts/bytes,
intended action, and state. Assert human rendering equals a rendering of the
same model and is deterministic. Include unknown authenticated catalog
attributes and explicit rebuild exclusions (keys/categories/counts/bytes/reasons).

### `tests/test_migration_run_evidence.py` (fault/contract test, file-I/O + request-response)

**Analog:** reconciliation resume tests (`tests/test_blob_store_reconciliation.py:165-205`)
and fixed evidence binding tests (`tests/test_phase3_windows_qualification_attestation.py:186-287`).

```python
first = store.reconcile(apply=True)
assert first.resume_token is not None
second = store.reconcile(apply=True, resume_token=first.resume_token)
assert second.resume_token is None
```

Adapt this to explicit evidence path/run ID, authenticated JSON, source and
destination identity/revision/plan digest binding, corruption/mismatch failure,
redaction of key bytes, idempotent resume, and rejection of “latest run” or
incidental candidate adoption. Inject the key provider and assert neither JSON
nor human output contains key material.

### `tests/test_migration_cutover.py` (integration/fault test, CRUD + file-I/O)

**Analog:** `tests/test_blob_store_atomic_lifecycle.py` and
`tests/test_lifecycle_authority_contract.py:419-484`.

```python
before = authority.snapshot_state()
hooks.arm_fault("promote.after_entry")
with pytest.raises(InjectedLifecycleFault):
    authority.promote_mutation(replacement)
assert authority.read_entry("atomic-key") == previous
assert authority.snapshot_state() == before
```

Test stopped-worker acknowledgement; stage/copy, whole-candidate verify, and
explicit activate; interruption after each evidence/payload/authority boundary;
stale revision rejection; no partial publication; source/prior retention;
`activated_offline` refusal of ordinary worker open/read/query/mutation; rollback while workers remain
stopped; finalize sealing rollback before restart; and separate, confirmed,
idempotent purge with retryable cleanup failure. Separate safety, recovery,
progress, and performance assertions and cover memory/memory plus
SQLite/filesystem deterministic tiers.

### `tests/test_rebuild_workflow.py` (integration/contract test, transform + file-I/O)

**Analog:** isolated derived rebuild (`tests/test_projection_sql_atomicity.py:72-104`)
and custom handler behavior (`tests/test_handler_registration.py:60-95`).

```python
with pytest.raises(ProjectionRebuildError):
    controller.rebuild()
assert sink.published is None
assert sink.discarded is not None
```

Test cross-backend/incompatible-contract rebuild through declared source
handlers and destination `BlobStore`, no default exclusions, and explicit
confirmed exclusion plans. Verify source before handler read, preserve unknown
catalog fields, custom-handler round trips, isolated discard on failure, and
that projection/candidate presence cannot publish canonical state.

### Authority contract tests

**Files:** `tests/test_lifecycle_authority_contract.py` and
`tests/contracts/test_postgresql_lifecycle_authority.py`.

**Analog:** local transition/fault contract (`test_lifecycle_authority_contract.py:320-369,419-484`)
and PostgreSQL DB-API/SQLSTATE fake contract (`tests/contracts/test_lifecycle_authority.py:25-121,166-240`).

Extend only for identity snapshot, bounded raw inventory, verified activation,
rollback/finalize state, and idempotence. Assert SQLite rollback restores all
authority rows on injected faults. Assert PostgreSQL SQLSTATE outcomes remain
typed and causal; do not require all contenders to succeed or claim live
PostgreSQL/S3 qualification.

### `tests/test_stored_compatibility.py` (regression extension, request-response + file-I/O)

**Analog:** current reopen/no-upgrade tests (`test_stored_compatibility.py:53-82,92-144`).
Prove inspection recognizes authority-owned current identity without a manually
created `store-format.json` marker. Preserve byte-for-byte negative tests for
legacy, foreign, corrupt, or unsupported layouts. Constructors and `initialize()`
continue to reject rather than upgrade.

### `docs/BACKEND_SELECTION.md` (documentation/reporting)

**Analog:** existing migration section (`docs/BACKEND_SELECTION.md:256-290`).
Replace “seamless”/automatic migration claims with explicit offline
`inspect -> plan -> stage -> verify -> activate`, rebuild-only classification,
stopped-worker and topology-specific guarantees, retained rollback material,
separate finalize/purge, and Phase 8 status for live PostgreSQL/S3. Do not imply
that constructors, pointer files, object listings, or projections activate stores.

## Shared Patterns

### One Lifecycle Authority

**Sources:** `docs/adr/0001-topology-specific-storage-guarantees.md:36-40,88-100,141-170`;
`src/cacheness/storage/lifecycle_authority.py:285-383`.

Payloads are immutable external effects created before authority promotion. The
authority transaction is the sole visibility point; work-directory evidence,
filesystem paths, staging names, S3 listings, candidate existence, and derived
projections are never authority. Do not add an online lock server, lease,
queue, scheduler, or second lifecycle database.

### Independent Version Compatibility

**Sources:** `src/cacheness/storage/manifest.py:120-170`;
`src/cacheness/storage/catalog.py:30-43`;
`src/cacheness/storage/backends/postgresql_lifecycle_authority.py:62-69`.

Compare independently: store epoch/layout, topology authority kind/capability/
schema, signed manifest schema, handler payload identity/version, and catalog
schema/transformation. `sqlite_user_version` is not a PostgreSQL authority
version; `STORE_FORMAT_VERSION` does not imply all other dimensions.

### Authenticated Canonical Evidence

**Sources:** `src/cacheness/storage/reconciliation.py:627-690`;
`src/cacheness/storage/manifest.py:289-345,585-631`;
`src/cacheness/storage/integrity.py:374-467`.

Use bounded canonical JSON, domain-separated HMAC or another verifiable envelope,
explicit run ID/evidence path, plan digest, source revision, and output digests.
Read existing signing material through `ManifestSigningKeyProvider`/
`ManifestKeyProvider`; never serialize key bytes or silently mint a replacement
identity. Verify manifest authenticity and payload digest/size before handler
deserialization.

### Bounded, Revision-Bound Work

**Sources:** `src/cacheness/storage/catalog.py:484-654,738-838`;
`src/cacheness/storage/projections.py:49-117,268-305`;
`src/cacheness/storage/reconciliation.py:185-281`.

Use keyset cursors, fixed authority revision/high-water marks, bounded rows,
bytes, and time budgets, returning explicit continuation rather than partial
success. A stale cursor/plan is a typed retryable or stale-plan outcome, never
silently refreshed. Do not allocate an entire remote catalog in `list_entries()`.

### Preserve Unknown Catalog Attributes

**Source:** `src/cacheness/storage/catalog.py:284-317`.

```python
source_values = source_schema.read_mapping(authenticated_manifest.catalog_values)
destination_values = dict(authenticated_manifest.catalog_values)
destination_values.update(transformed_declared_values)
```

Preserve the authenticated original mapping and transform only explicitly
declared fields. Derived indexes/projections are rebuilt separately and never
determine canonical migration eligibility.

### Topology-Specific Outcomes and Error Causality

**Sources:** `src/cacheness/storage/sqlite_lifecycle_authority.py:171-210,1101-1150`;
`src/cacheness/storage/backends/postgresql_lifecycle_authority.py:74-80`;
`src/cacheness/error_handling.py:212-235,307-447`.

Keep SQLite busy deadlines and PostgreSQL serialization/deadlock/lock-timeout/
statement-timeout as typed bounded outcomes with original causes. Use domain
errors for migration/rebuild, manifest/payload integrity, backend/conflict/
timeout, and retryable cleanup; avoid broad exception-to-miss conversion.

### Derived Projection Rebuild

**Source:** `src/cacheness/storage/projections.py:176-209,268-305,360-412`.

Projection checkpoints, isolated candidates, apply-after-checkpoint, and
post-publication partial outcomes are reusable. Projection state is derived,
not canonical. Rebuild it only after canonical activation and retain the
canonical receipt regardless of derived failure.

## No Exact Analog Found

No current module implements explicit whole-store migration/rebuild. The three
new migration modules therefore combine role-matched patterns rather than copy
an existing migration authority:

| File | Closest evidence | Planner warning |
|---|---|---|
| `src/cacheness/storage/migration.py` | `_AuthorityReconciler` + `ProjectionController` | Reconciliation is authority-residue cleanup, not migration state; do not reuse its action semantics or make evidence lifecycle authority. |
| `src/cacheness/storage/migration_evidence.py` | reconciliation token + manifest canonical JSON | Reuse bounded authentication only; use a migration-specific domain/version and explicit evidence path. |
| `src/cacheness/storage/migration_authority.py` | `LifecycleAuthority` | Add narrow transactional primitives; do not widen every adapter with duplicated orchestration. |

## Metadata

**Analog search scope:** `src/cacheness/storage/`, `src/cacheness/handlers.py`,
`src/cacheness/interfaces.py`, `src/cacheness/error_handling.py`, migration,
authority, compatibility, projection, reconciliation, and handler tests, plus
`docs/BACKEND_SELECTION.md` and the mandatory lifecycle ADR.  
**Files scanned:** 18 primary analog files plus targeted existing tests/docs.  
**Pattern extraction date:** 2026-09-09
