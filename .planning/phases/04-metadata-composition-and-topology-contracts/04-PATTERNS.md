# Phase 4: Metadata Composition and Topology Contracts - Pattern Map

**Mapped:** 2026-09-07  
**Files analyzed:** 20 new/modified source and test files inferred from CONTEXT.md and RESEARCH.md  
**Analogs found:** 20 / 20 (three new public surfaces have role-match analogs, not exact vocabulary matches)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `src/cacheness/storage/catalog.py` | model/utility | transform + request-response | `src/cacheness/storage/lifecycle_authority.py`, `src/cacheness/storage/manifest.py` | role-match |
| `src/cacheness/storage/composition.py` | config/provider | request-response | `src/cacheness/storage/blob_store.py`, `src/cacheness/storage/backends/__init__.py` | role-match |
| `src/cacheness/storage/projections.py` | service/provider | batch pull + event-driven hint | `src/cacheness/storage/manifest_repository.py` | role-match |
| `src/cacheness/storage/lifecycle_authority.py` | service/protocol | CRUD + request-response | itself (current authority contract) | exact |
| `src/cacheness/storage/lifecycle.py` | service/coordinator | CRUD + request-response | itself (current sole lifecycle engine) | exact |
| `src/cacheness/storage/blob_store.py` | facade/service | CRUD + request-response | itself (current public facade) | exact |
| `src/cacheness/storage/memory_lifecycle_authority.py` | authority/model | CRUD + request-response | current in-memory authority | exact |
| `src/cacheness/storage/sqlite_lifecycle_authority.py` | authority/model | CRUD + file-I/O transaction | current SQLite authority | exact |
| `src/cacheness/config.py` | config | transform/validation | `LifecycleAuthorityTopology`, `LifecycleLimits` | role-match |
| `src/cacheness/error_handling.py` | utility | request-response failure translation | current `CacheReason`/error hierarchy | exact |
| `src/cacheness/storage/__init__.py` | provider/barrel | request-response | current storage barrel | exact |
| `tests/test_catalog_schema.py` | test | unit/contract | `tests/test_blob_manifest.py`, `tests/test_config_validation.py` | role-match |
| `tests/test_catalog_query_contract.py` | test | unit/contract + request-response | `tests/test_query_meta_security.py`, `tests/test_lifecycle_authority_contract.py` | role-match |
| `tests/test_metadata_role_contract.py` | test | contract/integration | `tests/test_projection_mutation_contract.py` | role-match |
| `tests/test_blob_store_composition.py` | test | unit/contract | `tests/test_blob_store_close_contract.py`, `tests/test_blob_store_read_contract.py` | role-match |
| `tests/test_topology_capabilities.py` | test | unit/contract | `tests/test_blob_store_read_contract.py`, `tests/test_lifecycle_authority_contract.py` | role-match |
| `tests/test_catalog_projection.py` | test | integration/fault + batch pull | `tests/test_projection_mutation_contract.py`, `tests/test_projection_sql_atomicity.py` | role-match |
| `tests/test_lifecycle_authority_contract.py` | test | contract/fault | current common memory/SQLite transition suite | exact |
| `tests/test_blob_store_read_contract.py` | test | compatibility/request-response | current authority-vs-projection tests | exact |
| `tests/test_blob_store_close_contract.py` | test | concurrency/resource lifecycle | current admission/ownership tests | exact |

## Pattern Assignments

### `src/cacheness/storage/catalog.py` (model/utility, transform + request-response)

**Analog:** `src/cacheness/storage/lifecycle_authority.py` (lines 26-75, 267-305) plus `src/cacheness/storage/manifest.py` (lines 60-163, 238-280).

Use frozen, validated value objects and a runtime-checkable protocol. `EntryExpectation`/`EntrySnapshot` validate exact types and corroborating digests at construction; `AuthorityCapabilities` is a frozen semantic report; `LifecycleAuthority` exposes methods rather than backend tables. Copy this shape for schema/field/predicate/page/cursor/result values. Do not put SQLAlchemy/Pydantic models in the canonical API.

**Value-object validation pattern** (`lifecycle_authority.py:26-75`):

```python
@dataclass(frozen=True)
class EntrySnapshot:
    key: str
    generation: str
    locator: str
    manifest: bytes
    expectation: EntryExpectation

    def __post_init__(self) -> None:
        for field_name in ("key", "generation", "locator"):
            _bounded_text(getattr(self, field_name), field_name)
        if not isinstance(self.manifest, bytes) or len(self.manifest) > _MAX_MANIFEST_BYTES:
            raise ValueError("manifest must be bounded bytes")
        if self.expectation.generation != self.generation:
            raise ValueError("entry expectation generation must corroborate the entry")
```

**Bounded canonical-value pattern** (`manifest.py:116-163`):

```python
if value is None or isinstance(value, bool):
    return
if isinstance(value, int):
    if not MIN_SIGNED_64 <= value <= MAX_SIGNED_64:
        raise CacheManifestIntegrityError("Manifest integer exceeds the signed-64 range")
    return
if isinstance(value, str):
    if len(value.encode("utf-8")) > MAX_STRING_UTF8_BYTES:
        raise CacheManifestIntegrityError("Manifest string byte limit exceeded")
    return
```

Reuse these exact finite scalar/bounds conventions, distinguish `bool` from `int`, retain an explicit missing sentinel, and reject coercion. Reuse `query_validation.py:18-57` for bounded identifier validation and `CacheQueryValidationError`/`CacheReason` for invalid fields. Cursor encoding must be bounded canonical JSON authenticated with the existing manifest key provider; never accept raw SQL or an unsigned tuple.

### `src/cacheness/storage/composition.py` (config/provider, request-response)

**Analog:** `src/cacheness/storage/blob_store.py:117-276`, `src/cacheness/storage/backends/__init__.py:99-235`, and `src/cacheness/config.py:352-391`.

Create one typed role-resolution path. Preserve the constructor and registry APIs as adapters, but do not copy the current overloaded `_select_projection_backend()`/`_create_lifecycle_authority()` precedence. A role reference must accept exactly one of name or instance; options apply only to a name. Resolve registered names and built-ins uniformly, then validate actual participant capabilities before initialization or payload staging.

**Current selector seam to replace** (`blob_store.py:192-245`):

```python
self.backend = self._select_projection_backend(backend)
selected = lifecycle_authority or self._create_lifecycle_authority(backend)
self._validate_authority_capabilities(getattr(selected, "capabilities", None))
self.lifecycle_authority = selected
```

The new root must preserve the injected object identity (`is`), reject name+instance/options+instance ambiguity, and keep JSON as a derived projection. Do not infer guarantees from the backend string.

**Capability validation shape** (`blob_store.py:246-276`):

```python
required = {
    "durable": topology.durable,
    "multiprocess": topology.multiprocess,
    "exact_cas": topology.exact_cas,
    "indexed_paging": topology.indexed_paging,
    "projection": topology.projection,
    "transactional": True,
}
unavailable = [name for name, requested in required.items()
               if requested and not getattr(capabilities, name)]
if unavailable:
    raise CacheBlobBackendError(
        "Lifecycle authority cannot satisfy requested topology capabilities",
        context={"unsupported_capabilities": unavailable},
        reason=CacheReason.BLOB_BACKEND_CAPABILITY_UNSUPPORTED,
    )
```

Generalize this to the composed payload + authority + projection report, with explicit sharing scope, durability, CAS, transaction, streaming/listing, query, and projection rebuild/catch-up semantics. `LifecycleAuthorityTopology` (`config.py:352-391`) is the pattern for frozen config validation and fail-fast construction errors.

**Ownership/failed-initialization pattern** (`blob_store.py:135-154, 286-301`):

```python
self._owns_lifecycle_authority = lifecycle_authority is None
self._owns_backend = False
try:
    self._initialize(...)
except BaseException:
    self._close_failed_initialization_resources()
    raise
```

Constructed resources are store-owned; injected resources remain caller-owned unless explicitly transferred. Preserve close-once bookkeeping and never close caller-owned resources on construction failure.

### `src/cacheness/storage/projections.py` (service/provider, batch pull + event-driven hint)

**Analog:** `src/cacheness/storage/manifest_repository.py:33-71, 82-141, 149-180`.

Keep projection APIs narrow: consume bounded semantic catalog pages/checkpoints, never a lifecycle transaction handle, private SQLite tables, or a payload listing. Notifications are hints only. Use isolated candidate state and publish only after complete canonical snapshot validation.

**Derived projection rendering** (`manifest_repository.py:33-71`):

```python
class JsonProjectionExporter:
    """Rebuild JSON metadata from a revision-bound authority snapshot."""

    def __init__(self, authority: LifecycleAuthority, projection_path: Path | str, *, page_size: int = 128):
        if type(page_size) is not int or page_size <= 0:
            raise ValueError("projection page_size must be a positive integer")
        self.authority = authority
        self.projection_path = Path(projection_path)
        self.page_size = page_size
```

Reuse the bounded keyset pull, temporary candidate, `os.fsync`, `os.replace`, and exact revision check in `manifest_repository.py:82-175`. On ordinary best-effort refresh, warn and retain derived debt. On explicit refresh, return/raise a typed committed-partial report that carries the already committed `BlobEntryInfo`; never turn projection failure into rollback/absence.

### `src/cacheness/storage/lifecycle_authority.py` (service/protocol, CRUD + request-response)

**Analog:** the existing file itself, especially `AuthorityCapabilities` and `LifecycleAuthority` at lines 267-366.

Extend the protocol narrowly with schema identity, same-transaction declared catalog values, and bounded revision-bound page operations. Keep `AuthorityLifecycleEngine` as the only prepare/verify/promote/delete/cleanup coordinator. `PromotionResult` (lines 180-186) is the pattern for returning the exact committed snapshot plus cleanup debt; add catalog/projection status compatibly rather than replacing `BlobEntryInfo`.

Do not add a second catalog state machine or expose SQLite connections. New authority methods should preserve `EntrySnapshot` authentication/corroboration and be implemented by both memory and SQLite authorities.

### `src/cacheness/storage/lifecycle.py` (service/coordinator, CRUD + request-response)

**Analog:** current sole engine, `lifecycle.py:180-325, 328-382, 445-555`.

Preserve the sequencing: handler stage -> immutable payload publication -> digest/size verification -> signed manifest -> authority `record_verification` -> one `promote_mutation` -> cleanup debt settlement. Authoritative catalog fields must be attached before promotion, not in a post-commit hook.

**Core lifecycle pattern** (`lifecycle.py:180-219, 285-325`):

```python
handler = self.store.handlers.get_handler(data)
self.authority.preflight_mutation()
self.store.initialize()
previous = self.authority.read_entry(key)
with guarded_io.stage(handler, data, self.store.config) as staged:
    prepared = self.authority.prepare_mutation(MutationSpec.create(...))
    try:
        published = guarded_io.publish_generation(staged, locator)
        ...
        self.authority.record_verification(prepared, VerificationProof(...))
        promoted = self.authority.promote_mutation(prepared)
    except BaseException as error:
        self._abort(prepared, candidate_persisted=candidate_persisted)
        raise
```

For metadata updates, retain immutable-field rejection and generation replacement (`lifecycle.py:328-382`), but validate declared schema values before staging. For legacy `list`, preserve equality filtering and implement it through the canonical authority path; typed query is additive and bounded.

### `src/cacheness/storage/blob_store.py` (facade/service, CRUD + request-response)

**Analog:** current public facade, `blob_store.py:313-419, 454-484`.

Keep `_ordinary_admitted` as the public admission boundary, `initialize()` as explicit startup/schema validation, `put_entry()` returning an exact `BlobEntryInfo`, and `list(..., metadata_filter=...)` compatibility. Add the typed catalog query beside these methods rather than replacing them.

**Public lifecycle boundary** (`blob_store.py:339-419`):

```python
@_ordinary_admitted
def put_entry(self, data: Any, key=None, metadata=None) -> BlobEntryInfo:
    result = self._put_with_result_admitted(data, key=key, metadata=metadata)
    return replace(
        self.lifecycle.entry_info(result.promoted),
        previous_locator=None if result.previous is None else result.previous.locator,
    )

@_ordinary_admitted
def list(self, prefix=None, metadata_filter=None) -> List[str]:
    return self.lifecycle.list(prefix, metadata_filter)
```

Validate schema/query input before handler staging or authority dispatch. Keep JSON export best-effort after authority commit; projection status belongs in a compatible receipt/report, not a second write authority.

**Close/ownership boundary** (`blob_store.py:454-484`):

```python
should_release = self._instance_admission.begin_close()
if not should_release:
    return
try:
    self._release_owned_resources()
finally:
    self._instance_admission.finish_close(closed=closed)
```

Extend the same path for composed resources and explicit ownership transfer; preserve bounded drain/timeout behavior.

### `src/cacheness/storage/memory_lifecycle_authority.py` (authority/model, CRUD + request-response)

**Analog:** current implementation, `memory_lifecycle_authority.py:41-80, 126-209, 241-245, 487-500`.

Every mutation, catalog value/index update, revision increment, and projection-dirty update belongs inside `_transition()`’s one `RLock` critical section. Copy-on-read snapshots (`lines 89-102`) and idempotent promoted mutation results (`lines 156-164`) for parity with SQLite. Preserve truthful `durable=False`, `multiprocess=False` capabilities and closed-state checks.

```python
def _transition(self, callback):
    self._require_open()
    with self._lock:
        self.open_write_transactions += 1
        try:
            return callback()
        finally:
            self.open_write_transactions -= 1
```

### `src/cacheness/storage/sqlite_lifecycle_authority.py` (authority/model, CRUD + file-I/O transaction)

**Analog:** current implementation, `sqlite_lifecycle_authority.py:1030-1105, 1222-1298, 1385-1403, 1965-1969`.

Add typed catalog rows/index state to the existing authority schema/transaction only when the explicit migration boundary permits it. Do not add a competing lock, queue, or adapter transaction. Preserve the method-scoped connection, `BEGIN IMMEDIATE`, rollback, uncertain-commit classifier, and transaction boundary hooks.

**Atomic transaction pattern** (`sqlite_lifecycle_authority.py:1030-1105`):

```python
with self._connection(mutation=True, deadline=absolute_deadline, started_at=started_at) as connection:
    connection.execute("BEGIN IMMEDIATE")
    try:
        result = callback(connection)
        self._reach_transaction_boundary("authority.transaction.before_commit")
        self._execute_for_stage(connection, "COMMIT", ...)
        self._reach_transaction_boundary("authority.transaction.committed")
        return result
    except BaseException as error:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        ...
```

**Promotion placement** (`sqlite_lifecycle_authority.py:1222-1298`):

```python
connection.execute("INSERT INTO entries(...) VALUES (...) ON CONFLICT(key) DO UPDATE ...", ...)
self._reach_transaction_boundary("promote.after_entry")
connection.execute("UPDATE mutations SET state = 'promoted' ...")
...
connection.execute("UPDATE authority_state SET revision = ?, projection_dirty = 1 ...")
self._reach_transaction_boundary("promote.after_projection")
```

Insert declared catalog rows/indexes between descriptor and final revision update in this same callback, with fault hooks proving rollback of all rows together. `list_entries()` is only an existing compatibility scan; new pages must be bounded, ordered by `(key, generation)`, and revision-bound.

### `src/cacheness/config.py` (config, transform/validation)

**Analog:** `LifecycleAuthorityTopology` and `LifecycleLimits` (`config.py:352-430`).

Use frozen dataclasses for the new store composition/capability requirements, validate booleans/bounds in `__post_init__`, and preserve existing `CacheConfig` flat arguments as compatibility adapters. Do not add backend-name allowlists as evidence of capability; accept role references and validate the active instances in the composition root.

### `src/cacheness/error_handling.py` (utility, request-response failure translation)

**Analog:** `CacheReason` and domain hierarchy (`error_handling.py:19-90`).

Add stable typed reasons for invalid schema/query/cursor, stale snapshot, committed-partial projection, and composition ambiguity/capability failure under the existing `CacheError`/`CacheStorageError` hierarchy. Preserve context and causes (`raise ... from error`); do not return incomplete pages or silently downgrade failures.

```python
class CacheReason(str, Enum):
    INVALID_QUERY_FIELD = "invalid_query_field"
    INVALID_QUERY_VALUE = "invalid_query_value"
    BLOB_BACKEND_CAPABILITY_UNSUPPORTED = "blob_backend_capability_unsupported"
    BLOB_MIGRATION_REQUIRED = "blob_migration_required"
```

### `src/cacheness/storage/__init__.py` (provider/barrel, request-response)

**Analog:** current conditional/re-export barrel (`storage/__init__.py:31-87, 98-146`).

Re-export the native schema/query/composition/report types deliberately and append them to `__all__`. Keep optional imports guarded and retain compatibility exports (`BlobStore`, `BlobEntryInfo`, `BlobManifestV1`, existing errors/backends). Do not import optional ORM dependencies as a requirement for the native catalog API.

### `tests/test_catalog_schema.py` (test, unit/contract)

**Analog:** `tests/test_blob_manifest.py:41-61, 81-125` and `tests/test_config_validation.py:35-45`.

Build complete fixtures through a helper, parameterize exact bounds and invalid values, assert typed errors and stable context, and test reordered canonical schema identity. Cover undeclared-field preservation, missing vs null/default, new-write/update validation, and explicit migration-required evidence without rewriting on read.

### `tests/test_catalog_query_contract.py` (test, unit/contract + request-response)

**Analog:** `tests/test_query_meta_security.py:20-74` and `tests/test_lifecycle_authority_contract.py:310-359, 409-474`.

Test validation before backend access using a spy, every finite operator/type, AND semantics, deterministic `(key, generation)` order, bounded page sizes/membership, signed/tampered/mismatched cursors, and stale revision restart. Parameterize memory and SQLite to prove semantic parity; inject faults at each catalog/index promotion boundary.

### `tests/test_metadata_role_contract.py` (test, contract/integration)

**Analog:** `tests/test_projection_mutation_contract.py:53-87, 178-195, 264-275`.

Use fixtures for JSON, memory, SQLite, and mocked PostgreSQL projection roles. Assert a derived projection can never authorize lifecycle/query completeness, and that all role implementations expose the same explicit status/capability contract. Keep external ORM models derived-only.

### `tests/test_blob_store_composition.py` (test, unit/contract)

**Analog:** `tests/test_blob_store_close_contract.py:19-23, 66-113` and registry fixtures in `tests/test_metadata_backend_registry.py:46-60`.

Use caller-owned spies and registered fake roles. Assert injected identity, name/instance ambiguity rejection, options not merged into instances, close exactly once for constructed/transfer-owned resources, caller-owned resources remain open, and failed construction cleans only resources the store owns.

### `tests/test_topology_capabilities.py` (test, unit/contract)

**Analog:** `tests/test_blob_store_read_contract.py:1230-1262` and `tests/test_lifecycle_authority_contract.py:362-374`.

Assert memory requires explicit ephemeral topology, SQLite/local pairing reports only its actual durability/sharing/CAS scope, and impossible minimum capabilities fail before root creation or payload staging. Test composition reports from all active participants, not backend names.

### `tests/test_catalog_projection.py` (test, integration/fault + batch pull)

**Analog:** `tests/test_projection_mutation_contract.py:118-158` and `tests/test_projection_sql_atomicity.py:120-148, 200-237`.

Test duplicate pages/idempotent convergence, monotonic checkpoints, failure after canonical commit yielding committed-partial attribution, and isolated rebuild publication preserving old projection on interruption. Use independent adapters/processes where concurrency semantics matter; notifications must not be the correctness source.

### Existing contract test extensions

Keep `tests/test_lifecycle_authority_contract.py`’s parameterized memory/SQLite setup and rollback/uncertain-commit assertions (`lines 310-359, 409-474`) as the shared authority contract. Preserve `tests/test_blob_store_read_contract.py`’s no-schema shape, authority-over-corrupt-projection, and explicit memory topology checks (`lines 1194-1280`). Preserve `tests/test_blob_store_close_contract.py`’s bounded admission/timeout/ownership behavior (`lines 30-113`).

## Shared Patterns

### Authority is the sole visibility switch

**Sources:** `src/cacheness/storage/lifecycle.py:285-325`, `src/cacheness/storage/sqlite_lifecycle_authority.py:1222-1298`, `src/cacheness/storage/memory_lifecycle_authority.py:166-209`.

Authoritative descriptor, declared catalog values, and semantic index maintenance must join `promote_mutation()` / `_transition()`. Payload I/O remains outside the database transaction. External JSON/ORM/index projections run only after commit and cannot authorize reads, deletes, cleanup, repair, or query completeness.

### Integrity and safe parsing

**Sources:** `src/cacheness/storage/manifest.py:116-180, 238-280`, `src/cacheness/storage/integrity.py`, `src/cacheness/query_validation.py:18-82`.

Reuse canonical JSON bounds, SHA-256/HMAC signing, store identity key material, strict scalar types, bounded identifiers, and bound SQL parameters. Cursors bind version/store/schema/query/revision/last identity and fail closed when malformed or stale.

### Explicit capability and ownership reporting

**Sources:** `src/cacheness/storage/lifecycle_authority.py:267-276`, `src/cacheness/config.py:352-391`, `src/cacheness/storage/blob_store.py:135-154, 454-484`.

Capabilities describe the active pairing and scope; construction rejects unmet requirements before mutation. Created resources are owned and closed by the store; injected resources remain open by default. No hidden reference counting or selector precedence.

### Projection pull/rebuild semantics

**Sources:** `src/cacheness/storage/manifest_repository.py:82-175`, `tests/test_projection_mutation_contract.py`, `tests/test_projection_sql_atomicity.py`.

Pull bounded committed pages from a checkpoint, apply idempotently, checkpoint only after apply, and publish isolated rebuild state atomically. Best-effort refresh warns; explicit refresh returns a typed committed-partial result. A stale external projection must never produce a successful incomplete canonical query.

### Domain errors and compatibility

**Sources:** `src/cacheness/error_handling.py:19-90`, `src/cacheness/storage/read_contract.py:30-42`, `src/cacheness/storage/__init__.py:98-146`.

Use stable `CacheReason` values, preserve causes and context, retain `BlobEntryInfo` fields and no-schema metadata/list behavior, and add new typed APIs/re-exports alongside existing compatibility surfaces.

## No Exact Analog Found

The following new vocabulary has no exact existing implementation; planner should combine the role analogs above with the locked research contract:

| Surface | Existing partial analog | Gap to design explicitly |
|---|---|---|
| Native catalog schema/typed predicates/cursors | `lifecycle_authority.py` value objects + `manifest.py` validation | Schema evolution, missing/default semantics, finite AST, authenticated revision cursor |
| One payload/authority/projection composition root | `BlobStore` selectors + backend registries | Exact instance selection, ownership transfer, participant capability composition |
| Pull/checkpoint/isolated projection adapter | `JsonProjectionExporter` | Generic bounded semantic pages, committed-partial receipt, rebuild publication |

## Metadata

**Analog search scope:** `src/cacheness/storage/`, `src/cacheness/metadata.py`, `src/cacheness/config.py`, `src/cacheness/error_handling.py`, `src/cacheness/query_validation.py`, and relevant authority/projection/config/compatibility tests.  
**Files scanned:** 31 targeted source/test files and all phase context/research/ADR instructions.  
**Pattern extraction date:** 2026-09-07

**Do not copy:** the duplicate nominal `MetadataBackend` ABCs (`src/cacheness/metadata.py:269-382` vs `src/cacheness/storage/backends/base.py:12-159`), overloaded BlobStore selector precedence, broad exception-to-miss behavior in legacy cache paths, or projection CAS as canonical authority. These are documented migration hazards in RESEARCH.md and must be replaced/adapted under the Phase 4 contracts.
