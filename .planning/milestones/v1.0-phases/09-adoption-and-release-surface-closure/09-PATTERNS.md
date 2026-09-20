# Phase 9: Adoption and Release Surface Closure - Pattern Map

**Mapped:** 2026-09-16  
**Files analyzed:** 35 implementation, test, example, documentation, tooling, and planning surfaces  
**Analogs found:** 35 / 35 (direct analogs for code/tests; role-match analogs for docs and planning evidence)

Phase 9 is a public-surface and documentation cutover. It must not introduce a
new storage authority, backend adapter, lock, queue, retry coordinator, or
compatibility alias. Phase 10 owns removal of `SqlCache`; Phase 9 only removes
its promotion from the supported product story.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `src/cacheness/interfaces.py` | protocol/model/error contract | request-response / transform | `src/cacheness/interfaces.py` existing `CacheHandler` contract | exact self-replacement |
| `src/cacheness/handlers.py` | component/registry | transform + file I/O | `src/cacheness/handlers.py` existing built-ins and `HandlerRegistry` | exact self-replacement |
| `src/cacheness/error_handling.py` | error utility | request-response | `src/cacheness/interfaces.py` handler error hierarchy | role-match |
| `src/cacheness/storage/__init__.py` | public barrel | request-response | existing explicit `__all__` barrel | exact |
| `src/cacheness/storage/handlers/__init__.py` | compatibility/public barrel | request-response | existing re-export barrel | exact |
| `tests/test_interfaces.py` | contract test | request-response | `tests/test_public_api_contract.py` | role + contract match |
| `tests/test_handler_registration.py` | registry/extension test | transform + file I/O | existing custom-handler tests in same file | exact self-replacement |
| `tests/test_handlers.py` | built-in handler test | transform + file I/O | existing handler round-trip tests in same file | exact self-replacement |
| `tests/test_guarded_handler_io.py` | security/IO test | streaming + file I/O | existing retained-descriptor tests in same file | exact self-replacement |
| `tests/test_stored_compatibility.py` | persistence regression test | CRUD + file I/O | existing current-format reopen test in same file | exact self-replacement |
| `tests/test_public_api_contract.py` | public API test | request-response | existing retired-symbol and typed-result assertions | exact |
| `tests/test_phase9_examples.py` | executable-example harness | subprocess + file I/O | `tests/test_phase6_examples.py` | exact role/data-flow match |
| `tests/test_security_documentation.py` | documentation contract test | batch/text validation | existing section/phrase assertions | exact |
| `tests/test_sql_cache_documentation.py` | documentation regression | batch/text validation | existing stale SQL docs checks (likely remove/narrow) | role-match; Phase 10 owns implementation removal |
| `tools/run_phase8_packaging.py` | packaging verifier | subprocess + batch | same runner's wheel probes | exact self-replacement |
| `pyproject.toml` | package config | build/config | existing metadata and extras tables | exact self-replacement |
| `README.md` | gateway documentation | request-response (reader journey) | `docs/README.md` and current README | role-match |
| `docs/README.md` | navigation documentation | request-response (reader journey) | current navigation page | exact self-replacement |
| `docs/BLOB_STORE.md` | task/reference guide | CRUD + file I/O | `examples/custom_metadata_demo.py` and `docs/STORAGE_MIGRATION.md` | role + API match |
| `docs/CACHE_POLICY.md` | task/reference guide | request-response + policy | `examples/simple_object_caching.py`, `src/cacheness/decorators.py` | role + API match |
| `docs/PLUGIN_DEVELOPMENT.md` | extension tutorial | transform + file I/O | `tests/test_handler_registration.py`, `tests/test_guarded_handler_io.py` | role + safety match |
| `docs/API_REFERENCE.md` | API reference | request-response | current public barrels and `tests/test_public_api_contract.py` | role-match |
| `docs/SECURITY.md` | security guide | validation + file I/O | `tests/test_security_documentation.py` | exact content contract |
| `docs/RELEASE_QUALIFICATION.md` | qualification matrix | batch/evidence | current qualification matrix | exact self-replacement |
| `docs/STORAGE_INITIALIZATION.md` | operational guide | request-response + lifecycle | `src/cacheness/storage/blob_store.py:370-404` and migration guide | role + API match |
| `docs/STORAGE_MIGRATION.md` | operational guide | batch + migration | current explicit offline maintenance guide | exact self-replacement |
| `docs/BACKEND_SELECTION.md` | obsolete/merged guide | request-response | `docs/RELEASE_QUALIFICATION.md` and `docs/BLOB_STORE.md` | role-match; delete or fold |
| `docs/CONFIGURATION.md` | obsolete/merged guide | request-response | `docs/CACHE_POLICY.md` and current config classes | role-match; delete or fold |
| `examples/memory_blob_store.py` | canonical example | CRUD | `examples/simple_object_caching.py` + public wheel probe | role + API match |
| `examples/durable_catalog_store.py` | canonical example | CRUD + file I/O | `examples/custom_metadata_demo.py` + `tests/test_stored_compatibility.py` | exact workflow match |
| `examples/unified_cache.py` | canonical example | request-response + policy | `examples/simple_object_caching.py` and `src/cacheness/decorators.py` | exact workflow match |
| `examples/custom_mcap_format.py` | canonical extension example | transform + file I/O | `tests/test_handler_registration.py` and `tests/test_guarded_handler_io.py` | exact safety seam match |
| `examples/*` obsolete scripts | deleted/merged examples | mixed | `tests/test_phase6_examples.py` allowlist | role-match; fold unique material, then delete |
| `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-VERIFICATION.md` | evidence metadata | batch/evidence | Phase 3 gap tests and milestone audit | role-match; refresh only supported fields |
| `.planning/phases/08-production-gates-and-performance-stabilization/08-VALIDATION.md` | validation metadata | batch/evidence | `08-VERIFICATION.md` and `08-LOCAL-READINESS.json` | exact evidence match |
| `.planning/seeds/SEED-008-narwhals...md` (next available number) | backlog/seed | request-response | existing `.planning/seeds/SEED-001`–`SEED-007` | exact seed format match |
| `.codex/skills/spike-findings-cacheness/SKILL.md` | agent guidance | request-response | existing skill guidance and ADR guardrail | role-match |
| `AGENTS.md` | project guidance | request-response | existing generated GSD guidance sections | exact self-replacement |

Deletion candidates are not an archive: obsolete pre-cutover docs and examples
should either have accurate unique material folded into the retained task guides
or be removed. Dedicated `SqlCache` source/docs/tests/examples remain Phase 10
scope; Phase 9 may unlink or stop promoting them but must not partially remove
the subsystem.

## Pattern Assignments

### `src/cacheness/interfaces.py` — alias-free `FormatHandler` cutover

**Analog:** existing `CacheHandler` interface in the same file, lines 151-210;
error and registry contracts, lines 339-470.

**Protocol and identity pattern** (lines 151-187):

```python
class CacheHandler(CacheabilityChecker, CacheWriter, CacheReader, FormatProvider):
    PAYLOAD_FORMAT_VERSION = 1

    @property
    def payload_format(self) -> str:
        return self.data_type

    @property
    def payload_format_version(self) -> int:
        return self.PAYLOAD_FORMAT_VERSION

    def payload_identity(self) -> tuple[str, int]:
        return self.payload_format, self.payload_format_version

    def supports_payload_contract(
        self, payload_format: str, payload_format_version: int
    ) -> bool:
        return (
            payload_format == self.payload_format
            and payload_format_version == self.payload_format_version
        )
```

Rename only the Python protocol and error vocabulary. Preserve
`PAYLOAD_FORMAT_VERSION`, `data_type`, `payload_format`, and version behavior.
The persisted identity belongs to the handler/native format, not its Python
base-class name.

**Error pattern** (lines 339-380):

```python
class CacheHandlerError(Exception):
    def __init__(self, message: str, handler_type: Optional[str] = None,
                 data_type: Optional[str] = None):
        self.handler_type = handler_type
        self.data_type = data_type
        super().__init__(message)
        logger.error(
            f"Cache handler error: {message} "
            f"(handler={handler_type}, data_type={data_type})"
        )

class CacheWriteError(CacheHandlerError):
    pass
```

Use the same constructor/context and subclass hierarchy under
`FormatHandlerError`; retain `CacheWriteError`, `CacheReadError`,
`CacheFormatError`, and `CacheValidationError` because those names describe
failure kinds, not the old protocol.

**Registry typing pattern** (lines 415-450): update annotations and docstrings
from `CacheHandler` to `FormatHandler`; retain `register_handler`,
`get_handler`, and `get_handler_by_type` signatures and semantics.

### `src/cacheness/handlers.py` — built-ins and per-store registry

**Analog:** built-in handler classes and `HandlerRegistry` in the same file,
especially lines 1251-1320 and 1522-1680.

**Registration pattern** (lines 1522-1572):

```python
def register_handler(
    self, handler: CacheHandler,
    priority: Optional[int] = None,
    name: Optional[str] = None,
) -> None:
    self._validate_handler(handler)
    handler_name = name or handler.data_type
    existing_names = [h.data_type for h in self.handlers]
    if handler_name in existing_names:
        raise ValueError(f"Handler '{handler_name}' already registered...")
    if priority is not None:
        if priority < 0:
            priority = 0
        if priority >= len(self.handlers):
            self.handlers.append(handler)
        else:
            self.handlers.insert(priority, handler)
    else:
        self.handlers.append(handler)
```

Keep this as the store-local registration seam. Do not rename `HandlerRegistry`,
change priorities, add a module-global registration authority, or change
persisted `data_type` values.

**Validation pattern** (lines 1632-1680): retain explicit required method and
property checks, exact non-wildcard `payload_format`, positive integer format
version, tuple transformation edges, and the concrete transform guard. Only
change the explanatory protocol name and the comparison from
`CacheHandler.transform_payload` to `FormatHandler.transform_payload`.

**Built-in identity pattern:** built-ins keep their existing
`data_type`/`payload_format` values (`array`, `object`, `pandas_dataframe`,
`npz`, `parquet`, etc.). A class rename must not alter manifest fields.

### `src/cacheness/storage/__init__.py` and `src/cacheness/storage/handlers/__init__.py`

**Analog:** current explicit re-export and `__all__` barrels. In
`storage/__init__.py`, imports begin at lines 30-35 and handler exports are
listed at lines 257-261. In `storage/handlers/__init__.py`, imports are at
lines 28-49 and `__all__` at lines 70-88.

```python
from .handlers import (
    CacheHandler,
    HandlerRegistry,
    ArrayHandler,
    ObjectHandler,
)

__all__ = [
    "CacheHandler",
    "HandlerRegistry",
    "ArrayHandler",
    "ObjectHandler",
]
```

Replace the old symbol with `FormatHandler` and `FormatHandlerError` in both
barrels, and assert the old names are absent. Keep direct storage imports under
`cacheness.storage`; do not create a new top-level handler export merely to
support the rename.

### `tests/test_phase9_examples.py` — exact published-file harness

**Analog:** `tests/test_phase6_examples.py`, lines 13-102.

**Child-process isolation pattern** (lines 18-51):

```python
def _blocked_network_module(temporary_directory: Path) -> Path:
    guard_directory = temporary_directory / "network_guard"
    guard_directory.mkdir()
    (guard_directory / "sitecustomize.py").write_text(
        "import socket\n"
        "def _blocked(*_args, **_kwargs):\n"
        "    raise AssertionError('network access is prohibited in examples')\n"
        "socket.create_connection = _blocked\n"
        "socket.socket.connect = _blocked\n",
        encoding="utf-8",
    )
    return guard_directory

return subprocess.run(
    [sys.executable, str(EXAMPLES / name)],
    cwd=temporary_directory,
    env=environment,
    capture_output=True,
    text=True,
    check=False,
)
```

**Allowlist/repeatability pattern** (lines 54-102): parametrize a literal
four-file allowlist with expected stdout markers; run each exact file twice in
fresh directories; assert return code, markers, and identical output. Extend
the existing pattern with cleanup/residue assertions and ensure no test-only
copy of an example is maintained.

### Canonical examples

#### `examples/memory_blob_store.py`

**Analog:** `examples/simple_object_caching.py`, lines 22-29 and 34-79, plus
the isolated wheel probe in `tools/run_phase8_packaging.py`, lines 279-315.

Use explicit `StoreTopology` with `BackendRef(name="memory")` for both payload
and authority. Call `store.initialize()`, perform a small `put_entry`/`get`
round trip, assert the value, print one deterministic success marker, and close
in `finally` or a context manager. Keep `TemporaryDirectory` even when the
memory topology does not need a persistent root so all examples share the same
disposable contract.

#### `examples/durable_catalog_store.py`

**Analog:** `examples/custom_metadata_demo.py`, lines 25-73.

```python
return StoreTopology(
    payload=BackendRef(name="filesystem", options={"base_dir": root}),
    authority=BackendRef(name="sqlite", options={"root": root}),
)

with TemporaryDirectory(prefix="cacheness-catalog-") as temporary:
    root = Path(temporary)
    with BlobStore(local_topology(root), cache_dir=root) as store:
        receipt = store.put_entry(
            {"weights": [1, 2, 3]}, key="experiment-001",
            catalog_schema=schema,
            catalog_values={"experiment": "baseline", "accuracy": 95},
        )
        page = store.query_catalog(CatalogQuery(...), schema=schema)
        updated = store.update_catalog(..., expected=receipt.expectation)
```

Retain the explicit schema/query/update workflow, add explicit initialization
where the current API requires it, and close/reopen to prove durability. Do not
use old `BlobStore(cache_dir=..., backend=...)` construction.

#### `examples/unified_cache.py`

**Analog:** `examples/simple_object_caching.py`, lines 31-79, and
`src/cacheness/decorators.py`, lines 52-105.

```python
cache = UnifiedCache(config, store=memory_topology())
try:
    cache.initialize()
    written = cache.put(...)
    lookup = cache.lookup(cache_key=written.receipt.key)
    assert lookup.outcome is CacheOutcome.HIT
finally:
    cache.close()
```

The decorator example must use the explicit `@cached(cache=cache)` shape and
prove a cache hit/call-count result. It must show `UnifiedCache` as policy over
the store; do not recreate lifecycle operations in the example.

#### `examples/custom_mcap_format.py`

**Analog:** `_MappingHandler` in `tests/test_handler_registration.py`, lines
15-55, `_McapHandler` lines 66-96, and the safe staging contract in
`tests/test_guarded_handler_io.py`, lines 13-42.

```python
class _MappingHandler:
    @property
    def data_type(self) -> str:
        return "test_mapping"

    @property
    def payload_format(self) -> str:
        return "test-mapping"

    @property
    def payload_format_version(self) -> int:
        return 1

    def get_file_extension(self, config) -> str:
        return ".mapping"

store.handlers.register_handler(McapHandler(), priority=0)
```

Adapt this to an MCAP-like `.mcap` suffix with stable data/payload/version
identities and path-based `put`/`get`. The tutorial must demonstrate that the
handler receives a private staging path, returns a contained regular artifact,
and is selected by the store-local registry. It must not teach a handler about
obstore locators or expose a lifecycle authority.

### `src/cacheness/storage/guarded_handler_io.py` and handler safety tests

**Analog:** `tests/test_guarded_handler_io.py` lines 27-79 and implementation
`src/cacheness/storage/guarded_handler_io.py` lines 166-237, 240-339.

Keep the current private-stage → validated artifact → immutable publication
boundary. The test pattern asserts the handler path is outside the managed
root, preserves the native suffix, reads through a retained descriptor, and
rejects symlinks, directories, oversized suffixes, and path races. Phase 9 only
documents/tests this existing contract; it must not add a second I/O or
lifecycle mechanism.

### `tests/test_stored_compatibility.py` — persisted identity regression

**Analog:** lines 102-131.

```python
with BlobStore(_local_topology(root), cache_dir=root) as store:
    store.initialize()
    receipt = store.put_entry(...)

with BlobStore(_local_topology(root), cache_dir=root) as reopened:
    assert reopened.get(receipt.key) == {"answer": 42}
```

Use this close/reopen shape to capture handler/payload identity values before
the Python rename and verify they are identical afterward. No migration or
implicit upgrade belongs in this test.

### `tools/run_phase8_packaging.py` — source-free wheel/public export gate

**Analog:** same file, lines 47-178 and 249-361.

`BASE_PUBLIC_EXPORTS` is deliberately literal (lines 47-170), so update the
handler export entry directly to `FormatHandler`, not by deriving it from
runtime `__all__`. The base probe imports the wheel outside the checkout,
asserts each literal export, and exercises BlobStore/NumPy/UnifiedCache (lines
264-315). Extend the probe with old-name absence and preserve the existing
source-free environment and bounded subprocess error handling. Do not add a
second packaging runner.

### `pyproject.toml` — package identity and extras

**Analog:** current project metadata lines 1-16 and optional groups lines 18-45.

Change only the description to the BlobStore-first product identity and keep
the existing capability-specific extras inventory unless a later Phase 10
removal makes a dependency unused. The pattern is explicit `requires-python`,
base dependencies, and named optional groups. Do not add Narwhals, change
`uv.lock`, or introduce a broad install-everything recommendation.

### Documentation files

**Gateway analogs:** current `docs/README.md` lines 5-48 and
`docs/RELEASE_QUALIFICATION.md` lines 1-38. `docs/README.md` currently provides
category links but promotes stale constructors and `SqlCache`; retain the
navigation role while changing the order to task-first. `RELEASE_QUALIFICATION`
already centralizes evidence classes, blocking states, and nonclaims; make it
the sole detailed guarantees/qualification owner.

**README.md:** concise gateway pattern: current checkout/`uv` installation,
status box, BlobStore quick start, `UnifiedCache` quick start, and links to
task guides. It should not duplicate the full guarantee matrix or promote
PostgreSQL/S3/SqlCache.

**`docs/BLOB_STORE.md`:** copy direct-storage API shape from
`examples/custom_metadata_demo.py` and `BlobStore.initialize()` lines 370-404.
Document explicit topology, initialization, receipt/result, catalog, and close
semantics.

**`docs/CACHE_POLICY.md`:** copy typed outcome language and explicit decorator
shape from `src/cacheness/decorators.py:52-105` and the existing outcome doc
assertions in `tests/test_phase6_examples.py:105-130`. Keep policy above storage;
do not present SqlCache as part of this journey.

**`docs/PLUGIN_DEVELOPMENT.md`:** replace global-registration prose with the
MCAP tutorial pattern above. Include stable identities, version, suffix,
contained path I/O, and round-trip assertions; defer reusable conformance kit.

**`docs/API_REFERENCE.md`:** enumerate only actual current public imports from
the two barrels and the cache/storage modules. Use `tests/test_public_api_contract.py`
as the executable authority for symbol presence/absence.

**`docs/SECURITY.md`:** preserve the tested section
`Trusted Payload and Executable Serializer Boundary` and the fail-closed/native
format statements. The existing assertions in `tests/test_security_documentation.py`
are the contract; update stale quick starts without weakening trusted-payload,
path-containment, signature, or object-array warnings.

**`docs/STORAGE_INITIALIZATION.md` and `docs/STORAGE_MIGRATION.md`:** use the
explicit operator-boundary style already present in the migration guide lines
5-30 and 34-65: stopped workers, explicit roots/work directories, typed
failures, no hidden migration, and no second authority.

**`docs/BACKEND_SELECTION.md` and `docs/CONFIGURATION.md`:** fold any accurate
current setup into task/reference guides or delete if their primary material
is obsolete. Do not preserve stale backend recommendations merely for link
compatibility.

### Evidence metadata and seed

Use existing planning evidence as the analog, not a new runtime probe. Refresh
Phase 3/8 frontmatter/status only where named existing verification artifacts
support the value. Do not invent remote, Windows, controlled-Linux,
publication, PostgreSQL, or S3 evidence.

Create one next-numbered dormant seed in the existing seed format for Narwhals
investigation across pandas, PyArrow, and Polars while retaining handler-owned
Parquet. The seed should link Phase 8's decision, current dataframe handlers,
the `[dataframes]` extra, and future developer-kit work; it must not touch
`pyproject.toml` or `uv.lock`.

## Shared Patterns

### Single lifecycle authority

**Source:** `docs/adr/0001-topology-specific-storage-guarantees.md:30-45,
139-170` and `src/cacheness/storage/blob_store.py:166-172,246-272`.

Phase 9 documentation/examples may explain the lifecycle but must call the
existing public `BlobStore`/`UnifiedCache` surfaces directly. Filesystem paths,
compatibility metadata, examples, and docs are not authorities. `UnifiedCache`
remains policy over one store.

### Explicit initialization and disposable resources

**Source:** `src/cacheness/storage/blob_store.py:370-404` and
`examples/simple_object_caching.py:34-79`.

Examples create isolated temporary roots, initialize deliberately, close in a
`finally`/context manager, assert observable results, and print deterministic
success markers. No singleton/global factory or shared repository cache is
allowed in canonical examples.

### Narrow, typed failures

**Source:** `src/cacheness/error_handling.py:93-124`,
`tests/test_public_api_contract.py:108-170`, and packaging runner error handling
at `tools/run_phase8_packaging.py:327-361`.

Preserve domain-specific exceptions and causes; public docs should describe
typed outcomes and bounded nonclaims instead of broad “race-free” or universal
ACID language.

### Optional capability at request boundary

**Source:** `examples/configurable_serialization_demo.py:21-24,59-71` and
`tools/run_phase8_packaging.py:383-425`.

Base import should be silent when dataframe extras are absent. Mention
`[dataframes]` only in the dataframe task; a missing optional capability should
surface when requested, not as an import-time warning. Retain Parquet handlers
and existing NumPy NPZ/Blosc behavior.

### Qualification/nonclaim vocabulary

**Source:** `docs/RELEASE_QUALIFICATION.md:17-38,40-56,58-92`.

Use `PASS`, `UNAVAILABLE`, `NOT_QUALIFIED`, `NOT_PUBLISHED`, and `DEFERRED`
with the existing evidence-class meaning. Local readiness does not imply
remote service, Windows, controlled Linux, or immutable publication support.

## No Analog / Planner Notes

No Phase 9 file lacks a usable analog. New canonical filenames and the Narwhals
seed have role-match analogs rather than identical predecessors. The planner
should keep the four new examples and their exact-file harness together, and
the `FormatHandler` rename plus barrel/tests/persistence gate together, because
partial completion would leave either the published examples or public import
surface inconsistent.

## Metadata

**Analog search scope:** `src/cacheness/`, `tests/`, `examples/`, `docs/`,
`tools/`, `.planning/phases/`, `.planning/seeds/`, and `.codex/skills/`  
**Files scanned:** targeted live source, tests, examples, docs, packaging,
ADR, and planning evidence  
**Pattern extraction date:** 2026-09-16
