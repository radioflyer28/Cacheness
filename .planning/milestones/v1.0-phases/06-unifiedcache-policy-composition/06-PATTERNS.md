# Phase 6: UnifiedCache Policy Composition - Pattern Map

**Mapped:** 2026-09-08  
**Files analyzed:** 17 likely new/modified implementation, public-surface, example, and contract-test files  
**Analogs found:** 17 / 17 (role matches; several are direct extensions of the current seam)

This map is for the pre-production policy cutover. Preserve `BlobStore` and
`AuthorityLifecycleEngine` as the only payload/authoritative-catalog lifecycle
authority. Do not add a cache-side lock, queue, coordinator, projection
authority, repair hook, or compatibility wrapper.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `src/cacheness/cache_policy.py` | model/utility/service | transform + observer | `src/cacheness/storage/read_contract.py`, `src/cacheness/storage/projections.py` | role-match |
| `src/cacheness/core.py` | policy facade/controller | request-response + CRUD | current `src/cacheness/core.py`, `src/cacheness/storage/blob_store.py` | exact seam |
| `src/cacheness/decorators.py` | decorator/hook adapter | request-response | current `src/cacheness/decorators.py` | exact seam (stale behavior) |
| `src/cacheness/config.py` | config/model/validation | transform | current nested config classes and `validate_config()` | exact seam (compatibility-heavy) |
| `src/cacheness/__init__.py` | public API/barrel | request-response | `src/cacheness/storage/__init__.py` | role-match |
| `src/cacheness/storage/blob_store.py` | storage facade/service | CRUD + paginated catalog | `src/cacheness/storage/lifecycle.py` | exact lifecycle primitive |
| `src/cacheness/storage/catalog.py` | model/utility | bounded query/transform | current `CatalogQuery`/`CatalogPage` | exact contract |
| `tests/test_phase6_lookup_contract.py` | unit/contract test | request-response | `tests/test_blob_store_read_contract.py` | role-match |
| `tests/test_phase6_removal_contract.py` | adversarial contract test | CRUD + paginated catalog | `tests/test_unified_cache_lifecycle_authority.py`, `tests/test_catalog_query_contract.py` | role/data-flow match |
| `tests/test_phase6_statistics.py` | unit test | transform/observer | `tests/test_core.py`, `tests/test_projection_mutation_contract.py` | role-match |
| `tests/test_phase6_decorator_contract.py` | unit/public contract test | request-response | `tests/test_decorators.py`, `tests/test_cached_custom_metadata.py` | role/data-flow match |
| `tests/test_phase6_public_api_contract.py` | public contract test | request-response | `tests/test_public_api_contract.py` | exact test shape |
| `tests/test_phase6_policy_contract.py` | unit/architecture test | CRUD + policy transform | `tests/test_unified_cache_lifecycle_authority.py`, `tests/test_config_validation.py` | role/data-flow match |
| `tests/contracts/test_phase6_topology_policy.py` | topology contract test | CRUD + bounded catalog | `tests/contracts/test_topology_lifecycle.py` | exact topology shape |
| `examples/api_request_caching.py` | example/documentation | request-response | existing example | role-match; update surface |
| `examples/simple_object_caching.py` | example/documentation | request-response | existing example | role-match; decorator semantics |
| `examples/configurable_serialization_demo.py` | example/documentation | transform + request-response | existing example | role-match; constructor/import surface |

`cache_policy.py` is a recommended new focused module from RESEARCH.md, not an
existing contract. It should borrow immutable-result patterns below without
becoming a second storage sequencer.

## Pattern Assignments

### `src/cacheness/cache_policy.py` (model/utility/service, transform + observer)

**Analogs:** `src/cacheness/storage/read_contract.py`,
`src/cacheness/storage/projections.py`, and `src/cacheness/storage/catalog.py`.

Use frozen, validated value objects for the shared outcome, lookup result,
statistics snapshot, and removal report. `read_contract.py:30-114` uses a
frozen receipt plus a presence-bearing `BlobEntry`; `read_contract.py:119-157`
uses a string-valued enum and a pure classifier. `projections.py:41-173` shows
validated frozen status/checkpoint/batch/outcome records.

**Imports and immutable value pattern** (`read_contract.py:3-15, 30-44, 119-127`):

```python
from enum import Enum
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class BlobReceipt:
    operation_id: str
    # ... immutable identity and expectation fields

class CacheReadFailureCategory(str, Enum):
    INTEGRITY = "integrity"
    LIFECYCLE_CONFLICT = "lifecycle_conflict"
    BACKEND_FAILURE = "backend_failure"
```

Implement the Phase 6 outcome enum with the locked values `hit`, `absent`,
`expired`, `corrupt`, `conflict`, and `backend_error`. Make the lookup result
carry `value` independently of presence so `value=None` remains a hit. Preserve
the original `cause` for conflict/backend failures. Statistics must be frozen,
contain one count for each outcome, and derive totals/rates from those counts;
do not list the catalog to compute statistics and do not make counters an
authority.

**Catalog/report shape** (`projections.py:49-60, 119-173`):

```python
@dataclass(frozen=True)
class ProjectionOutcome:
    name: str
    status: ProjectionStatus
    checkpoint: ProjectionCheckpoint | None = None
    error_type: str | None = None
```

Follow this validation style for removal reports: bounded attempted/removed/
conflicted-or-retryable/failed counts, plus an opaque continuation cursor and
an explicit completion flag where work is incomplete. Reports contain counts,
keys/identities, and typed causes only; never payload bytes, credentials, or
signing material.

**Do not copy:** `read_contract.py` deliberately leaves direct absence as
`None`; the cache policy must translate the single `BlobEntry` observation into
the new presence-bearing result exactly once.

### `src/cacheness/core.py` (policy facade/controller, request-response + CRUD)

**Analog:** current `src/cacheness/core.py`; storage sequencing analog is
`src/cacheness/storage/blob_store.py`.

**Composition/imports** (`core.py:20-31`):

```python
from .config import CacheConfig, _DEFAULT_TTL, create_cache_config
from .error_handling import (
    CacheBlobIntegrityError,
    CacheBlobLifecycleConflictError,
    CacheBlobStoreClosedError,
    CacheUnsafePathError,
)
from .handlers import HandlerRegistry
from .serialization import create_unified_cache_key
from .storage.blob_store import BlobStore
from .storage.composition import BackendRef, StoreTopology
```

Keep the facade's explicit `CacheConfig` and one private/composed
`BlobStore` (`core.py:65-125`). Construction should validate/select the
supported `StoreTopology`; initialization and close remain deliberate
boundaries. If a store is injected, follow the documented ownership rule and
do not create a second lifecycle owner.

**Keying and current stale policy paths** (`core.py:127-130, 157-182`):

```python
def _create_cache_key(self, params: Mapping[str, Any]) -> str:
    return create_unified_cache_key(dict(params), self.config)

def _authority_snapshot_entry(self, cache_key: str):
    snapshot = self._cache_blob_store.get_entry_info(cache_key)
    return (snapshot, self._cache_entry(snapshot)) if snapshot is not None else (None, None)
```

Retain deterministic key generation and the authenticated snapshot conversion,
but stop policy from re-reading to discover presence. A direct lookup should
call `BlobStore.open_entry()` once, inspect its snapshot metadata, classify TTL,
then call `snapshot.read()` only for a non-expired entry. Use the expectation on
that same snapshot for expiry cleanup. `CacheBlobIntegrityError` can become the
non-destructive `corrupt` outcome; `CacheBlobLifecycleConflictError` and
backend failures retain typed causes instead of becoming misses.

**Current lifecycle paths to replace** (`core.py:232-279, 291-344, 346-409`):

```python
for cache_key in self._cache_blob_store.list():
    snapshot, entry = self._authority_snapshot_entry(cache_key)
    # ... classify and self._retire_exact_authority_snapshot(...)

self._cache_blob_store.put_entry(data, key=cache_key, metadata=metadata)

with self._cache_blob_store.open_entry(cache_key) as snapshot:
    if snapshot is None:
        return None
    # ... expiry and snapshot.read()

snapshot = self._cache_blob_store.get_entry_info(cache_key)
if snapshot is not None:
    self._retire_exact_authority_snapshot(cache_key, snapshot)
```

Replace every unbounded local `list()` loop with bounded `query_catalog()` pages
and an explicit work budget. Every candidate must retain its authenticated
snapshot/expectation through `BlobStore.delete(key, expected=...)`. Convert
delete results and typed contention into one removal report. Size policy may
select deterministically from authenticated descriptor facts, but a partial
bounded traversal must report continuation/incomplete work; no `_policy_stats`
or projection may authorize deletion.

Keep `put()`'s ordering (`core.py:281-302`): canonical BlobStore commit first,
policy maintenance second. A post-commit stats/eviction failure cannot roll
back or revoke the committed receipt. `close()` (`core.py:411-417`) should
close the owned BlobStore and preserve committed-partial semantics.

### `src/cacheness/decorators.py` (decorator adapter, request-response)

**Analog:** current `src/cacheness/decorators.py` (`_generate_cache_key()` and
the `cached` wrapper, lines 38-223). It contains the deterministic namespace
logic to retain and the ownership/`None` seams to remove.

**Key namespace pattern** (`decorators.py:60-76`):

```python
func_name = getattr(func, "__qualname__", getattr(func, "__name__", "unknown"))
func_module = getattr(func, "__module__", "unknown")
func_id = f"{func_module}.{func_name}"
normalized_params = _normalize_function_args(func, args, kwargs)
enhanced_params = {**normalized_params, "__function__": func_id}
return create_unified_cache_key(enhanced_params, config)
```

Retain function identity and normalized arguments as the cache namespace. Store
the function identity as authoritative catalog policy metadata when writing so
`cache_clear()` can query it without reversing an opaque hash.

**Required wrapper shape** (`decorators.py:144-199`, stale behavior marked):

```python
@functools.wraps(func)
def wrapper(*args, **kwargs):
    result = cache.lookup_function(func, args, kwargs)
    if result.outcome is CacheOutcome.HIT:
        return result.value       # includes a legitimate None
    # recompute only for explicitly allowed miss outcomes
    value = func(*args, **kwargs)
    cache.store_function_result(func, args, kwargs, value)
    return value
```

Bind the sole public `cached` decorator to an explicit `UnifiedCache` instance;
remove implicit `UnifiedCache()` creation, weakref/atexit ownership, alternate
decorators, and context/factory paths. Preserve `functools.wraps` and make
`cache_clear()` call the facade's bounded function-prefix/predicate invalidator
and return its structured report. Suppression must be explicit and record the
actual outcome before fallback; never catch broad failures and relabel them as
absence.

### `src/cacheness/config.py` (config/model/validation, transform)

**Analog:** nested dataclasses and validation already in `config.py`.

**Nested vocabulary** (`config.py:36-78`):

```python
@dataclass
class CacheStorageConfig:
    cache_dir: str = "./cache"
    max_cache_size_mb: Optional[int] = 2000
    cleanup_on_init: bool = True

@dataclass
class CacheMetadataConfig:
    default_ttl_hours: float = 24
    enable_cache_stats: bool = True
```

Use the same dataclass composition style but make cache-policy fields visibly
separate from storage topology/handler/security fields. Remove flat legacy
aliases and destructive `cleanup_on_init`-during-construction behavior per the
locked cutover; initialization/maintenance must be explicit.

**Current compatibility constructor to simplify** (`config.py:434-527`):

```python
class CacheConfig:
    storage: CacheStorageConfig = field(default_factory=CacheStorageConfig)
    metadata: CacheMetadataConfig = field(default_factory=CacheMetadataConfig)
    blob: CacheBlobConfig = field(default_factory=CacheBlobConfig)
    # ... many flat legacy parameters
```

Retain nested defaults and `LifecycleLimits`; route topology capability
validation through `StoreTopology`/BlobStore preflight before policy calls.
Avoid adding another backend registry or a cache-specific topology branch.

**Fail-before-I/O validation** (`config.py:847-910, 1013-1033`):

```python
errors = validate_config(config)
if errors:
    error_messages = [str(error) for error in errors]
    raise ValueError(
        f"Invalid configuration ({len(errors)} errors):\n  - "
        + "\n  - ".join(error_messages)
    )
```

Keep structured field errors, positive finite TTL/size checks, and clear typed
optional-feature failures. Unsupported topology or optional dependency must
fail before policy operations begin; do not preserve misleading availability
flags merely for compatibility.

### `src/cacheness/__init__.py` (public API/barrel, request-response)

**Analogs:** current package barrel (`__init__.py:32-48, 178-244`) and the
storage barrel (`storage/__init__.py:49-69, 114-175`).

**Current stale surface** (`__init__.py:32-33`):

```python
from .core import CacheConfig, UnifiedCache as cacheness, get_cache
from .decorators import cached
```

Publish `UnifiedCache`, `CacheConfig`, and the one explicit-instance `cached`
surface. Remove the `cacheness` class alias, `get_cache`/`reset_cache`, and
alternate decorator/constructor names rather than forwarding wrappers. Keep
`SqlCache` independently importable and do not merge its surface.

**Optional export pattern** (`__init__.py:68-103, 226-244`):

```python
try:
    from .sql_cache import SqlCache, SqlCacheAdapter
except ImportError:
    _has_sql_cache = False
else:
    _has_sql_cache = True
```

Retain honest optional imports and install-oriented typed errors. Update
`__all__` as a single cutover, including policy result/report models if they
are public. Use the storage barrel's deliberate re-export style; do not expose
legacy metadata/blob backend registries as new cache authorities.

### `src/cacheness/storage/blob_store.py` (storage facade/service, CRUD + paginated catalog)

**Analog:** `src/cacheness/storage/lifecycle.py`, with current public facade
methods at `blob_store.py:349-430, 535-631, 669-715`.

**Single-snapshot read boundary** (`blob_store.py:349-363`; engine details in
`lifecycle.py:608-678`):

```python
@contextmanager
def open_entry(self, key: str):
    self._require_canonical_store()
    with self._instance_admission.operation():
        with self.lifecycle.open_entry(key) as entry:
            yield entry

@_ordinary_admitted
def get_entry_info(self, key: str) -> BlobEntry | None:
    return self.lifecycle.get_entry_info(key)
```

Keep `open_entry()` as the policy seam: the lifecycle engine authenticates the
manifest, verifies the payload before handler deserialization, retries its
bounded generation observation internally, and returns a `BlobEntry` carrying
`expectation`. Policy must not add a second read or verifier.

**Bounded catalog query** (`blob_store.py:547-631`):

```python
validate_catalog_page_request(
    query, schema=schema, cursor=effective_cursor,
    limit=effective_limit, work_cap=effective_work_cap,
)
return self.lifecycle_authority.catalog_page(
    query, effective_cursor, schema=schema, limit=effective_limit,
    work_cap=effective_work_cap, signing_key=signing_key,
    manifest_loader=self._authenticated_authority_manifest,
)
```

Extend this existing facade only as narrowly as needed to expose authenticated
`created_at`/`byte_size` facts and the exact expectation in the catalog view.
Do not expose raw authority rows, fabricate expectations in policy, or add a
second index. Keep `CatalogQuery` validation and opaque authority cursors.

**Exact deletion and lifecycle ownership** (`blob_store.py:535-540, 669-715`):

```python
@_ordinary_admitted
def delete(self, key: str, *, expected: EntryExpectation | None = None) -> bool:
    return self.lifecycle.delete(key=key, expected=expected)

def clear(self) -> int:
    with self._instance_admission.clear_operation() as release_snapshot:
        token = self.lifecycle.begin_clear()
        release_snapshot()
        return self.lifecycle.complete_clear(token)
```

All cache invalidation paths call `delete(expected=...)` or bounded catalog
operations; none deletes payload files, metadata rows, or S3 objects directly.
Use existing `clear()` only for global authority-owned clear; translate its
typed conflict/retryable outcomes into policy removal reports without adding a
cache coordinator.

### `src/cacheness/storage/catalog.py` (model/utility, bounded query/transform)

**Analog:** `CatalogField`, `CatalogQuery`, `CatalogEntry`, and `CatalogPage`
(`catalog.py:183-329, 657-704`).

**Immutable page pattern** (`catalog.py:657-704`):

```python
@dataclass(frozen=True, slots=True)
class CatalogEntry:
    key: str
    generation: str
    values: Mapping[str, Any]

@dataclass(frozen=True, slots=True)
class CatalogPage:
    entries: tuple[CatalogEntry, ...]
    revision: int
    cursor: str | None
    exhausted: bool
```

Preserve schema/query validation, finite operators, authenticated descriptor
decoding, keyset ordering, bounded page size, and opaque signed cursors. If
`CatalogEntry` is extended for policy eviction, expose only authenticated
descriptor facts plus exact lineage/expectation; do not change page semantics
or introduce an age/size projection authority. Malformed predicates must fail
before the first query/delete.

### `tests/test_phase6_lookup_contract.py` (unit/contract, request-response)

**Analog:** `tests/test_blob_store_read_contract.py:43-78`.

Use the existing fixture style: construct a local BlobStore topology, put a
`None` payload, assert missing versus `exists=True`, and close in `finally`.
Add call-count spies around `open_entry()` to prove one lookup observation for
hit, absent, expired, and corruption. Assert the shared outcome enum/result,
preserved typed causes, non-destructive corruption, and cached-`None` return.

### `tests/test_phase6_removal_contract.py` (adversarial, CRUD + paginated catalog)

**Analogs:** `tests/test_unified_cache_lifecycle_authority.py:24-331` and
`tests/test_catalog_query_contract.py:74-160, 225-240`.

Reuse `tmp_path`, independent cache/store instances, `Event`/`Thread` race
scheduling, and `pytest.raises` for typed conflicts. Cover replacement-after-
selection (generation B survives exact-delete conflict), malformed predicate
no-I/O, bounded page continuation without duplicate materialization, TTL,
size, key, predicate, decorator, and global clear through one report. A
retryable/contended result is reported as conflicted/retryable, not absence.

### `tests/test_phase6_statistics.py` (unit, transform/observer)

**Analogs:** `tests/test_core.py:336-348` (legacy dict assertions) and
`tests/test_projection_mutation_contract.py:66-93` (immutable derived-result
semantics).

Replace legacy `cache_hits`/`cache_misses` assertions with frozen aggregate
fields for hit, absent, expired, corrupt, conflict, and backend-error. Assert
derived totals/rates and immutability. Spy that stats do not list/query the
catalog and that stats/projection failure cannot alter a canonical receipt.

### `tests/test_phase6_decorator_contract.py` (unit/public, request-response)

**Analogs:** `tests/test_decorators.py:144-223, 438-461` and
`tests/test_cached_custom_metadata.py:66-93`.

Retain invocation-count and `functools.wraps` assertions. Strengthen the
existing `returns_none()` case with a count of exactly one across two calls;
assert the second call consumes a hit with `value is None`. Pass an explicit
cache instance, assert no implicit global/owned cache, test explicit suppression
only for declared outcomes, and assert `cache_clear()` returns the actual
structured report while isolating one function's namespace.

### `tests/test_phase6_public_api_contract.py` (public contract, request-response)

**Analog:** `tests/test_public_api_contract.py:17-211`.

Keep the executable `__all__`/star-import and subprocess-blocked-optional-
dependency pattern. Add positive checks for canonical `UnifiedCache`,
`CacheConfig`, `cached`, lookup/result/report names and explicit init/close;
add negative checks that old aliases/factories/decorators are absent. Preserve
`SqlCache` import/regression coverage and typed optional/committed-partial
error evidence.

### `tests/test_phase6_policy_contract.py` (unit/architecture, CRUD + transform)

**Analogs:** `tests/test_unified_cache_lifecycle_authority.py:52-188` and
`tests/test_config_validation.py:59-170`.

Use lifecycle hooks to prove policy never bypasses BlobStore or deletes a
replacement generation. Assert TTL uses the same snapshot, size policy is
deterministic/bounded, reports incomplete work with an opaque cursor, and
derived statistics do not authorize eviction. Validate unsupported topology,
flat legacy config removal, and explicit initialization before policy work.

### `tests/contracts/test_phase6_topology_policy.py` (topology contract, CRUD + bounded catalog)

**Analog:** `tests/contracts/test_topology_lifecycle.py:1-235`.

Reuse its parametrized local `memory` and `sqlite-filesystem` profiles, shared
`AuthorityLifecycleEngine` assertion, explicit `BlobStore` close, and
deterministic remote doubles. Add policy put/get/None/expiry/removal checks;
remote tests must forbid unbounded `list()` and assert typed retryable outcomes
without claiming live PostgreSQL/S3 qualification.

### Examples: `examples/api_request_caching.py`, `examples/simple_object_caching.py`, `examples/configurable_serialization_demo.py`

**Analogs:** each current file itself; representative import/decorator shapes
are `api_request_caching.py:8-29`, `simple_object_caching.py:13-48`, and
`configurable_serialization_demo.py:11-65`.

Update examples to the canonical `from cacheness import UnifiedCache,
CacheConfig, cached` surface and explicit cache binding/lifecycle. Remove
`cacheness` alias, flat constructor settings, `cached.for_api`, and implicit
decorator cache creation. Keep the examples' useful TTL/function namespace
demonstrations and close explicitly where a cache instance is created.

## Shared Patterns

### One lifecycle authority and exact expectations

**Sources:** `src/cacheness/storage/blob_store.py:535-631`,
`src/cacheness/storage/lifecycle.py:608-678, 708-822`,
`src/cacheness/storage/lifecycle_authority.py:28-76`.

Policy selects candidates and classifies outcomes. BlobStore and its engine own
manifest authenticity, immutable generations, promotion, exact deletion,
cleanup debt, and typed topology progress. Carry `EntryExpectation` from the
authenticated snapshot/page into every delete; never resolve by key again.

### Fail-closed validation and typed errors

**Sources:** `src/cacheness/storage/catalog.py:246-442`,
`src/cacheness/storage/read_contract.py:119-157`,
`src/cacheness/error_handling.py:116-156, 450-484`.

Validate complete queries and configured work limits before authority I/O.
Translate storage failures narrowly with `raise ... from error`; preserve
conflict/backend causes. Corruption may be a non-destructive cache miss, but
must not be silently relabeled absent or destructively cleaned. A post-commit
derived failure preserves `CacheBlobCommittedPartialError.receipt`,
`remaining_cursor`, and projection evidence.

### Explicit admission/initialization/close boundaries

**Sources:** `src/cacheness/storage/blob_store.py:349-430, 669-715`,
`tests/test_unified_cache_adversarial_lifecycle.py:39-153`.

Use the existing BlobStore admission and close semantics. Initialization is
explicit before sharing a cache with workers; close drains/returns typed
outcomes and cannot revoke an earlier canonical commit. Process-local admission
may improve ordering but is not a correctness authority; do not add another
lock or queue at the policy layer.

### Bounded, resumable catalog work

**Sources:** `src/cacheness/storage/catalog.py:30-58, 657-704`,
`src/cacheness/storage/blob_store.py:547-631`,
`tests/contracts/test_topology_lifecycle.py:193-232`.

Every selection operation specifies page size and total work cap, follows the
authority-owned opaque cursor, and returns incomplete/continuation state rather
than draining an unbounded inventory. Use portable typed predicates, never
raw SQL/lambdas/eval, filesystem walks, S3 listings, offsets, or a second index.

### Derived state is observational only

**Sources:** `src/cacheness/storage/projections.py:1-6, 144-173`,
`tests/test_projection_mutation_contract.py:66-93`.

Statistics, projections, and removal summaries observe canonical outcomes.
Failures or loss in derived state cannot authorize deletion, change a receipt,
or create a repair prerequisite. Keep result/report models immutable and
bounded.

## No Analog Found

No file is completely without a role/data-flow analog. The exact
`cache_policy.py` module and its class names are new design decisions; use the
frozen value-object and pure-classifier patterns from `read_contract.py`,
`projections.py`, and `catalog.py`, with semantic names from CONTEXT.md rather
than copying a legacy stats dictionary.

## Metadata

**Analog search scope:** `src/cacheness/`, `src/cacheness/storage/`,
`tests/`, `tests/contracts/`, `examples/`, and Phase 6 context/research/ADR.  
**Files scanned:** 17 primary analog files plus focused catalog/lifecycle/error
sections.  
**Pattern extraction date:** 2026-09-08
