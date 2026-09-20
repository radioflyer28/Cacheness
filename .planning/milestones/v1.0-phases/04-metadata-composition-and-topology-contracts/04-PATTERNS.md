# Phase 4: Metadata Composition and Topology Contracts - Pattern Map

**Mapped:** 2026-09-07 (after the pre-production compatibility reset)
**Files analyzed:** 24 new/modified/consolidated source and test files inferred from CONTEXT.md, RESEARCH.md, and VALIDATION.md
**Analogs found:** 24 / 24 (new catalog/composition/projection vocabulary has role analogs, not an existing exact implementation)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `src/cacheness/storage/catalog.py` | model/utility | transform + request-response | `lifecycle_authority.py`, `manifest.py` | role-match |
| `src/cacheness/storage/composition.py` | config/provider | request-response | `blob_store.py`, backend registries | role-match |
| `src/cacheness/storage/projections.py` | service/provider | batch pull + event-driven hint | `manifest_repository.py` | role-match |
| `src/cacheness/storage/lifecycle_authority.py` | protocol/service | CRUD + request-response | current authority contract | exact |
| `src/cacheness/storage/lifecycle.py` | coordinator/service | CRUD + request-response | current sole lifecycle engine | exact |
| `src/cacheness/storage/blob_store.py` | facade/service | CRUD + request-response | current BlobStore facade | exact |
| `src/cacheness/storage/memory_lifecycle_authority.py` | authority/model | CRUD + request-response | current memory authority | exact |
| `src/cacheness/storage/sqlite_lifecycle_authority.py` | authority/model | CRUD + file-I/O transaction | current SQLite authority | exact |
| `src/cacheness/config.py` | config | transform/validation | `LifecycleAuthorityTopology`, `LifecycleLimits` | role-match |
| `src/cacheness/error_handling.py` | utility | failure translation | `CacheReason` hierarchy | exact |
| `src/cacheness/storage/__init__.py` | provider/barrel | request-response | current storage barrel | exact |
| `src/cacheness/metadata.py` | backend/projection | CRUD | current metadata backends | consolidation target |
| `src/cacheness/storage/backends/base.py` | protocol/registry | request-response | duplicate metadata ABC | delete/consolidate |
| `src/cacheness/storage/backends/__init__.py` | registry/provider | request-response | current named registries | consolidate into role registry |
| `src/cacheness/storage/manifest_repository.py` | projection service | batch/file-I/O | `JsonProjectionExporter` | replace with `projections.py` |
| `src/cacheness/storage/read_contract.py` | result model | request-response | `BlobEntryInfo` | replace with new commit result |
| `src/cacheness/custom_metadata.py` | projection integration | event/batch | current ORM link hooks | replace with derived adapter |
| `src/cacheness/core.py` | policy facade | request-response | current custom-metadata hooks | remove obsolete storage hooks; Phase 6 owns policy |
| `tests/test_catalog_schema.py` | test | unit/contract | `test_blob_manifest.py`, `test_config_validation.py` | role-match |
| `tests/test_catalog_query_contract.py` | test | unit/contract | `test_query_meta_security.py`, authority contract | role-match |
| `tests/test_metadata_role_contract.py` | test | contract/integration | projection mutation tests | role-match |
| `tests/test_blob_store_composition.py` | test | unit/contract | close/read/registry tests | role-match |
| `tests/test_topology_capabilities.py` | test | unit/contract | topology/read tests | role-match |
| `tests/test_catalog_projection.py` | test | integration/fault + batch pull | projection race/atomicity tests | role-match |

Historical characterization modules such as `tests/test_metadata_backend_registry.py`, `tests/test_blob_backend_registry.py`, `tests/test_blob_store_legacy_contract.py`, and shape-focused portions of `tests/test_blob_store_read_contract.py`/`tests/test_public_api_contract.py` are evidence for behavior to reassess. Rewrite or retire assertions for removed APIs; do not add shims to make them pass.

## Pattern Assignments

### `src/cacheness/storage/catalog.py` (model/utility, transform + request-response)

**Analog:** `src/cacheness/storage/lifecycle_authority.py:26-75, 267-305` and `src/cacheness/storage/manifest.py:60-163, 238-280`.

Use frozen, validated value objects and a narrow runtime-checkable protocol. `EntryExpectation`/`EntrySnapshot` validate exact types and corroborating digests; `AuthorityCapabilities` is a frozen semantic report; authority methods hide backend tables. Apply this structure to schema/field/predicate/page/cursor/result values.

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

Reuse `manifest.py:116-163` for the finite scalar/bounds vocabulary, including the explicit `bool` before `int` check, signed-64 limits, string byte limits, nesting, and collection bounds. Keep missing distinct from stored null, reject coercion, and preserve undeclared metadata as opaque authenticated data. Reuse `query_validation.py:18-57` for bounded identifiers and typed `CacheQueryValidationError` reasons. Cursor material must be bounded canonical JSON authenticated with the existing store key; never raw SQL or an unsigned tuple.

The new catalog/manifest format needs a distinguishable explicit version. The current development manifest v1 is implementation evidence only; do not design around its exact top-level shape or reopen it through compatibility code.

### `src/cacheness/storage/composition.py` (config/provider, request-response)

**Analog:** `src/cacheness/storage/blob_store.py:117-276`, `src/cacheness/storage/backends/__init__.py:99-235`, and `src/cacheness/config.py:352-391`.

Implement one typed composition root for payload, catalog authority, projections, ownership, and minimum capabilities. A role reference accepts exactly one of name or instance; options apply only to named construction. Resolve built-ins and registered names through the same role registry and validate actual participant capabilities before initialization or payload staging.

The existing selector is a seam to delete, not copy:

```python
self.backend = self._select_projection_backend(backend)
selected = lifecycle_authority or self._create_lifecycle_authority(backend)
self._validate_authority_capabilities(getattr(selected, "capabilities", None))
```

Remove the overloaded `backend=` argument and private `_select_projection_backend()`/`_create_lifecycle_authority()` paths after the root owns supported construction. Do not retain old factories, constructor aliases, or duplicate registries merely for historical calls. Preserve only the semantic rule that an injected instance remains the selected object.

Use `blob_store.py:246-276` as the starting shape for fail-fast capability rejection, generalized to payload + authority + projection scope (durability, process/host sharing, CAS, transaction, immutable generations, streaming/listing, portable query, index state, and rebuild/catch-up mode). `LifecycleAuthorityTopology` (`config.py:352-391`) is the pattern for frozen configuration validation.

Use explicit ownership bookkeeping from `blob_store.py:135-154, 286-301`: constructed resources are store-owned, injected resources caller-owned by default, transfer is explicit, and failed construction closes only owned resources. Keep the bounded close/admission policy from `blob_store.py:454-484`.

### `src/cacheness/storage/projections.py` (service/provider, batch pull + event-driven hint)

**Analog:** `src/cacheness/storage/manifest_repository.py:33-71, 82-141, 149-180`.

Replace the SQLite-private-backup exporter with a generic adapter that consumes bounded semantic catalog pages and checkpoints. Projections receive no lifecycle transaction handle and never authorize canonical reads, deletes, cleanup, recovery, or query completeness. Notifications only prompt a pull.

```python
class JsonProjectionExporter:
    def __init__(self, authority: LifecycleAuthority, projection_path: Path | str, *, page_size: int = 128):
        if type(page_size) is not int or page_size <= 0:
            raise ValueError("projection page_size must be a positive integer")
        self.authority = authority
        self.projection_path = Path(projection_path)
        self.page_size = page_size
```

Reuse the bounded keyset pull, temporary candidate, `os.fsync`, `os.replace`, and exact revision check in `manifest_repository.py:82-175`, but source pages through the new authority query surface. Apply pages idempotently and checkpoint after apply. Best-effort refresh records debt/warns; explicit refresh returns a typed committed-partial result containing the new commit result and remaining work. Rebuild isolated derived state and publish only after complete validation; preserve old published state on interruption.

### `src/cacheness/storage/lifecycle_authority.py` (protocol/service, CRUD + request-response)

**Analog:** current file, especially `AuthorityCapabilities` and `LifecycleAuthority` at lines 267-366.

Narrow the protocol to semantic lifecycle primitives plus schema/catalog transaction and bounded page operations. Keep `AuthorityLifecycleEngine` as the only prepare/verify/promote/delete/cleanup coordinator. Add schema identity, declared values, and semantic index updates to the promotion input/result; do not expose SQLite connections or create a catalog adapter state machine.

`PromotionResult` at lines 180-186 is useful evidence for an immutable exact-generation result plus cleanup debt, but the new API may replace `BlobEntryInfo` with a cleaner commit result carrying exact generation/expectation and projection status. Do not preserve the old class or field shape solely for compatibility.

### `src/cacheness/storage/lifecycle.py` (coordinator/service, CRUD + request-response)

**Analog:** current engine, `lifecycle.py:180-325, 328-382, 445-555`.

Retain the sequencing pattern: handler stage -> immutable payload publication -> digest/size verification -> signed descriptor -> authority verification -> one promotion -> cleanup-debt settlement. Authoritative catalog values must be included in that promotion transaction, not written by a post-commit hook.

```python
handler = self.store.handlers.get_handler(data)
self.authority.preflight_mutation()
self.store.initialize()
previous = self.authority.read_entry(key)
with guarded_io.stage(handler, data, self.store.config) as staged:
    prepared = self.authority.prepare_mutation(MutationSpec.create(...))
    try:
        published = guarded_io.publish_generation(staged, locator)
        self.authority.record_verification(prepared, VerificationProof(...))
        promoted = self.authority.promote_mutation(prepared)
    except BaseException:
        self._abort(prepared, candidate_persisted=candidate_persisted)
        raise
```

Remove the dictionary `list(prefix, metadata_filter)` branch at `lifecycle.py:543-555`; expose only the bounded typed query. Retain metadata update generation/conflict semantics from `lifecycle.py:328-382`, but validate declared fields before payload/authority side effects.

### `src/cacheness/storage/blob_store.py` (facade/service, CRUD + request-response)

**Analog:** current facade, `blob_store.py:313-419, 454-484`.

Refactor BlobStore to accept the single composition specification and expose the new catalog query/commit-result surface. Keep lifecycle meaning—explicit initialization, exact generation observation, immutable payloads, cleanup debt, and committed-partial projection outcomes—but do not preserve the overloaded constructor, `backend=` selector, old dictionary metadata filter, or exact `BlobEntryInfo` shape.

The current public flow is sequencing evidence:

```python
@_ordinary_admitted
def put_entry(self, data: Any, key=None, metadata=None) -> BlobEntryInfo:
    result = self._put_with_result_admitted(data, key=key, metadata=metadata)
    return replace(self.lifecycle.entry_info(result.promoted), ...)

@_ordinary_admitted
def list(self, prefix=None, metadata_filter=None):
    return self.lifecycle.list(prefix, metadata_filter)
```

Translate it into the clean typed API: schema validation before staging, bounded `query_catalog(...)`, one immutable commit result, and explicit projection status. Historical list-filter/receipt compatibility is not a requirement. Unsupported pre-Phase-4 layouts must be detected before opening/mutating the store and return typed migration/rebuild-required evidence.

### `src/cacheness/storage/memory_lifecycle_authority.py` (authority/model, CRUD + request-response)

**Analog:** current implementation, `memory_lifecycle_authority.py:41-80, 126-209, 241-245, 487-500`.

Put catalog values, revision changes, indexes, and projection-dirty state inside the existing `_transition()` `RLock` critical section. Copy-on-read snapshots and idempotent promoted-operation results provide semantic parity with SQLite. Keep truthful process-local/ephemeral capability reporting; do not make memory satisfy durable or multi-process requirements.

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

Use the existing method-scoped connection, `BEGIN IMMEDIATE`, rollback, uncertain-commit classifier, and transaction hooks. Add the new supported catalog schema/index layout as an explicitly versioned authority format. Detect the development layout before mutation; do not canonical-scan it, auto-create side tables, silently delete it, or attach a runtime adapter.

```python
with self._connection(mutation=True, deadline=absolute_deadline, started_at=started_at) as connection:
    connection.execute("BEGIN IMMEDIATE")
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

Insert declared catalog rows/index state in the existing `promote_mutation()` callback (`lines 1222-1298`) before revision/dirty-state commit. Fault boundaries must prove descriptor, catalog, index, lineage, debt, and revision rollback together. New pages are bounded, revision-bound, and ordered by `(key, generation)`.

### `src/cacheness/config.py` (config, transform/validation)

**Analog:** `LifecycleAuthorityTopology` and `LifecycleLimits` (`config.py:352-430`).

Use frozen dataclasses for the new composition specification and minimum capability requirements. Validate booleans, bounds, role ambiguity, and unsupported pairings in `__post_init__`/composition construction. Remove legacy flat backend-selection fields when the typed root replaces them; do not add a second compatibility constructor path.

### `src/cacheness/error_handling.py` (utility, request-response failure translation)

**Analog:** `CacheReason` and domain hierarchy (`error_handling.py:19-90`).

Add typed reasons/errors for schema/query/cursor validation, stale snapshots, composition ambiguity/capability failure, committed-partial projection, and unsupported migration layouts. Preserve context and causes (`raise ... from error`); never return an incomplete page or silently downgrade an unsupported format.

### `src/cacheness/storage/__init__.py` (provider/barrel, request-response)

**Analog:** current re-export barrel (`storage/__init__.py:31-87, 98-146`).

Export only the clean native schema/query/composition/commit-result surface plus selected backends/projections. Remove exports for deleted factories, duplicate metadata ABCs, old receipt contracts, and legacy session hooks. Optional imports remain guarded; native catalog functionality must not require ORM packages.

## Consolidation and Deletion Targets

These are intentional cutover targets, not analogs to preserve:

| Existing target | Clean replacement | Action |
|---|---|---|
| `BlobStore._select_projection_backend()` / `_create_lifecycle_authority()` and overloaded `backend=` | `storage/composition.py` typed role specs/root | Delete after migration to one root |
| `metadata.py` `MetadataBackend` factory path and duplicate ABC in `storage/backends/base.py` | One structural role registry/protocol set | Consolidate/delete superseded ABC/factory code |
| `storage/backends/__init__.py` split metadata/blob registries | Role-aware registry used by composition root | Remove parallel selection semantics; retain only code that the root calls |
| `create_metadata_backend()` and legacy backend-name construction | Composition root named-role resolution | Delete callable factory if no longer used |
| `lifecycle.py:list(..., metadata_filter=dict)` | Typed bounded equality/range/membership/existence query | Delete list-filter branch |
| `manifest_repository.py` SQLite-private backup exporter | `projections.py` semantic page/checkpoint adapter | Replace/delete old exporter |
| `custom_metadata.py` / `core.py` runtime ORM session/custom-metadata hooks | Derived projection adapter | Remove canonical-write/session hooks where Phase 4 owns the seam; Phase 6 handles remaining cache policy |
| `read_contract.py:BlobEntryInfo` exact shape | New immutable commit result/receipt model | Replace; retain only semantic exact-generation evidence |
| Development SQLite/manifest v1 layouts | New distinguishable versioned catalog/manifest format | Reject before mutation with typed migration/rebuild-required evidence; no runtime adapter |

When deleting code, update imports and `__all__` deliberately. Do not leave dead aliases or callable shims solely because historical tests refer to them.

## Test Pattern Assignments

### `tests/test_catalog_schema.py`

**Analogs:** `tests/test_blob_manifest.py:41-61, 81-125` and `tests/test_config_validation.py:35-45`.

Construct complete schema fixtures through helpers, parameterize exact bounds and invalid values, and assert typed errors/context. Cover opaque undeclared fields, missing vs null/default, new-format write/update validation, additive supported evolution, and non-mutating rejection of the development layout. Assert the new explicit version rather than the old manifest field set.

### `tests/test_catalog_query_contract.py`

**Analogs:** `tests/test_query_meta_security.py:20-74` and `tests/test_lifecycle_authority_contract.py:310-359, 409-474`.

Use a backend-access spy to prove validation precedes dispatch. Cover every finite operator/type, AND semantics, bounded membership/page size, deterministic `(key, generation)` ordering, cursor authentication/version/store/schema/query mismatch, and stale-revision restart. Parameterize memory and SQLite; inject faults around every catalog/index promotion boundary.

### `tests/test_metadata_role_contract.py`

**Analog:** `tests/test_projection_mutation_contract.py:53-87, 178-195, 264-275`.

Exercise JSON, memory, SQLite, and PostgreSQL-facing implementations as explicit authority or projection roles. Assert projections cannot authorize lifecycle/query completeness and that the clean role registry resolves all registered implementations through the same composition path. Remove tests that require old factory names or metadata ABC inheritance.

### `tests/test_blob_store_composition.py`

**Analogs:** `tests/test_blob_store_close_contract.py:19-23, 66-113` and registry fixtures in `tests/test_metadata_backend_registry.py:46-60`.

Use injected spies and registered fake roles to assert exact instance identity, name/instance ambiguity rejection, options isolation, ownership transfer, close-once behavior, initialization-failure cleanup, and absence of superseded selectors/factories/constructor overloads. These are negative API/source tests, not compatibility tests.

### `tests/test_topology_capabilities.py`

**Analogs:** `tests/test_blob_store_read_contract.py:1230-1262` and `tests/test_lifecycle_authority_contract.py:362-374`.

Assert memory is explicitly ephemeral, SQLite/local reports only its actual scope, and impossible minimum capabilities fail before root creation or payload staging. Compute reports from all active participants, never backend names.

### `tests/test_catalog_projection.py`

**Analogs:** `tests/test_projection_mutation_contract.py:118-158` and `tests/test_projection_sql_atomicity.py:120-148, 200-237`.

Test duplicate/interrupted pull, idempotent convergence, monotonic checkpoints, post-commit failure with committed-partial attribution, and isolated rebuild publication. Use independent adapters/processes where concurrency semantics matter; notifications are not correctness evidence.

Historical tests should be rewritten or retired when they assert `list(metadata_filter=...)`, old constructors/factories/registries, exact `BlobEntryInfo` fields, runtime session hooks, or reopening the development layout. Keep tests for semantic invariants: complete generations, exact expectations, projection non-authority, typed partial results, and non-mutating unsupported-layout rejection.

## Shared Patterns

### One authority visibility switch

**Sources:** `lifecycle.py:285-325`, `sqlite_lifecycle_authority.py:1222-1298`, `memory_lifecycle_authority.py:166-209`.

Authoritative descriptor, schema identity, declared catalog values, semantic indexes, lineage, cleanup debt, and revision update together in authority promotion. Payload I/O remains external and reconciled. No metadata adapter repeats lifecycle sequencing.

### Integrity and safe parsing

**Sources:** `manifest.py:116-180, 238-280`, `integrity.py`, `query_validation.py:18-82`.

Reuse canonical bounds, SHA-256/HMAC, store identity key material, strict scalar types, bounded identifiers, and bound SQL parameters. Cursors bind version/store/schema/query/revision/last identity and fail closed when malformed or stale.

### Semantic capabilities and explicit ownership

**Sources:** `lifecycle_authority.py:267-276`, `config.py:352-391`, `blob_store.py:135-154, 454-484`.

Capability reports describe the active payload/authority/projection pairing and scope. Construction rejects unmet guarantees before mutation. Created resources are owned by the store; injected resources remain caller-owned unless transfer is explicit.

### Derived projection pull/rebuild

**Sources:** `manifest_repository.py:82-175`, `tests/test_projection_mutation_contract.py`, `tests/test_projection_sql_atomicity.py`.

Pull bounded committed pages, apply idempotently, checkpoint after apply, and publish isolated rebuild state only after complete validation. Best-effort failure warns/records debt; explicit failure returns a committed-partial result. A stale projection never yields successful incomplete canonical results.

### Versioning and migration boundary

**Sources:** `manifest.py:22-53, 388-395`, `sqlite_lifecycle_authority.py:53-66, 802-854`, `error_handling.py:41-60`.

The new manifest/catalog format has explicit distinguishable versions. Unsupported development/future layouts are classified before mutation and fail with typed migration/rebuild-required evidence. Reads do not rewrite or auto-create schema. Preserve Phase 7 seams for non-mutating inventory, stopped-worker offline migration, resumable copy-verify-switch, signing-material preservation, and confirmed rebuild; Phase 7 need not support the pre-Phase-4 source layout.

## No Exact Analog Found

No existing implementation has the exact clean public vocabulary for:

| New surface | Closest pattern | Design gap |
|---|---|---|
| Native schema, typed predicates, revision cursor | frozen authority values + manifest/query validation | schema fingerprint/evolution, missing/default semantics, finite AST, authenticated cursor |
| Single composition root | BlobStore selectors + split registries | deletion of overloads, structural role registry, composed participant capabilities |
| Generic projection pull/checkpoint/rebuild | JSON exporter | semantic page source, committed-partial result, isolated publication |

## Metadata

**Analog search scope:** `src/cacheness/storage/`, `src/cacheness/metadata.py`, `src/cacheness/config.py`, `src/cacheness/error_handling.py`, `src/cacheness/query_validation.py`, `src/cacheness/core.py`, `src/cacheness/custom_metadata.py`, and authority/projection/configuration tests.
**Files scanned:** targeted current source and test analogs plus all revised phase context/research/validation/project/requirements/roadmap/AGENTS/ADR instructions.
**Pattern extraction date:** 2026-09-07

**Do not copy:** overloaded selector precedence, duplicate nominal metadata ABCs, legacy factory/list-filter/session APIs, exact `BlobEntryInfo` field requirements, SQLite-private projection authority, broad legacy exception-to-miss behavior, or runtime compatibility for pre-Phase-4 layouts.
