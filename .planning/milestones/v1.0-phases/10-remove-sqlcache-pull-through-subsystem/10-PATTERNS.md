# Phase 10: Remove SqlCache Pull-Through Subsystem - Pattern Map

**Mapped:** 2026-09-17  
**Files analyzed:** 43 planned deletions/edits/additions  
**Analogs found:** 30 / 43 (the remaining 13 are intentionally deleted artifacts)

Phase 10 is a pre-production product-surface deletion. The implementation should
copy the repository's explicit public inventories, closed test manifests,
source-free wheel probe, and task-oriented documentation patterns. It must not
copy the old SqlCache implementation into a replacement layer or introduce a
new lifecycle coordinator.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `src/cacheness/sql_cache.py` | component/service (delete) | CRUD/batch SQL | `src/cacheness/storage/blob_store.py` | no analog; delete whole subsystem |
| `tests/test_sql_cache.py` | test (delete) | CRUD/batch | `tests/test_handlers.py` | no analog; dedicated contract removed |
| `tests/test_sql_cache_documentation.py` | test (delete) | request-response/documentation | `tests/test_phase9_documentation.py` | no analog; dedicated contract removed |
| `tests/test_sql_cache_failure_contract.py` | test (delete) | request-response/error contract | `tests/test_public_api_contract.py` | no analog; dedicated contract removed |
| `docs/SQL_CACHE.md` | documentation (delete) | request-response | `docs/BLOB_STORE.md` | no analog; obsolete product guide |
| `docs/CUSTOM_GAP_DETECTION.md` | documentation (delete) | CRUD/batch | `docs/STORAGE_MIGRATION.md` | no analog; obsolete feature guide |
| `docs/ARBITRARY_TIME_INCREMENTS.md` | documentation (delete) | CRUD/batch | `docs/BLOB_STORE.md` | no analog; obsolete feature guide |
| `examples/beginner_sql_cache.py` | example (delete) | request-response | `examples/README.md` | no analog; replace with canonical local journeys already listed |
| `examples/simple_stock_cache.py` | example (delete) | request-response | `examples/unified_cache.py` | no analog; obsolete SQL-cache journey |
| `examples/stock_cache_example.py` | example (delete) | request-response | `examples/unified_cache.py` | no analog; obsolete SQL-cache journey |
| `examples/database_backend_comparison.py` | example (delete) | request-response | `examples/durable_catalog_store.py` | no analog; mixed obsolete journey |
| `examples/intelligent_storage_demo.py` | example (delete) | request-response | `examples/durable_catalog_store.py` | no analog; mixed obsolete journey |
| `examples/simple_backend_demo.py` | example (delete) | request-response | `examples/memory_blob_store.py` | no analog; mixed obsolete journey |
| `src/cacheness/__init__.py` | package barrel/config | request-response | `src/cacheness/storage/__init__.py` | exact role match |
| `src/cacheness/error_handling.py` | utility/error model | request-response | same file's `CacheReason` enum | exact role match |
| `pyproject.toml` | config/packaging | build/package metadata | `tools/run_phase8_packaging.py` | role match |
| `uv.lock` | config/lockfile | build/package metadata | `pyproject.toml` | role match |
| `docs/PANDAS_API_AUDIT.md` | documentation/audit | batch/transform | `docs/API_REFERENCE.md` | role match |
| `docs/API_REFERENCE.md` | documentation/API guide | request-response | `docs/BLOB_STORE.md` | role + flow match |
| `docs/STORAGE_MIGRATION.md` | documentation/runbook | batch/transform | `docs/API_REFERENCE.md` | role match |
| `docs/README.md` | documentation/index | request-response/navigation | `examples/README.md` | role + flow match |
| `docs/CROSS_PLATFORM_GUIDE.md` | documentation/qualification | request-response/batch | `tests/test_full_suite_environment.py` | role match |
| `AGENTS.md` | agent/config guidance | request-response | `.planning/codebase/ARCHITECTURE.md` | role match |
| `.planning/codebase/ARCHITECTURE.md` | architecture map | request-response/transform | `docs/BLOB_STORE.md` | role match |
| `.planning/codebase/CONCERNS.md` | architecture/risk map | request-response | `AGENTS.md` | role match |
| `.planning/codebase/CONVENTIONS.md` | convention map | request-response | `AGENTS.md` | role match |
| `.planning/codebase/INTEGRATIONS.md` | integration map | request-response/batch | `docs/RELEASE_QUALIFICATION.md` | role match |
| `.planning/codebase/STACK.md` | stack map/config | build/package metadata | `pyproject.toml` | role + flow match |
| `.planning/codebase/STRUCTURE.md` | structure map | request-response | `.planning/codebase/ARCHITECTURE.md` | role match |
| `.planning/codebase/TESTING.md` | test map | batch/request-response | `tests/test_phase9_quality_workflow.py` | role + flow match |
| `tests/test_public_api_contract.py` | test/contract | request-response | `tests/test_phase6_public_api_contract.py` | exact role match |
| `tests/test_phase6_public_api_contract.py` | test/contract | request-response | `tests/test_public_api_contract.py` | exact role match |
| `tests/test_phase6_suite_isolation.py` | test/integration harness | event-driven/process | `tests/test_phase071_contract_verifier.py` | role match |
| `tests/test_phase1_quality_gates.py` | test/AST quality gate | transform/static analysis | `tests/test_phase071_contract_verifier.py` | role + flow match |
| `tests/test_full_suite_environment.py` | test/environment contract | request-response/process | `tests/test_phase9_quality_workflow.py` | role match |
| `tools/verify_phase4_cutover.py` | verifier/utility | batch/process | `tools/verify_phase6_contracts.py` | exact role match |
| `tests/test_phase4_cutover_verifier.py` | test/verifier contract | batch/process | `tests/test_phase071_contract_verifier.py` | exact role match |
| `tools/verify_phase6_contracts.py` | verifier/utility | batch/process | `tools/verify_phase071_contracts.py` | exact role match |
| `tests/test_phase6_contract_verifier.py` | test/verifier contract | batch/process | `tests/test_phase071_contract_verifier.py` | exact role match |
| `tools/verify_phase071_contracts.py` | verifier/utility | batch/process | `tools/verify_phase6_contracts.py` | exact role match |
| `tools/run_phase8_packaging.py` | packaging harness/utility | batch/process | `tools/phase8_evidence.py` | exact role + flow match |
| `tests/packaging/test_wheel_matrix.py` | test/packaging integration | batch/process | `tools/run_phase8_packaging.py` | exact role match |
| `tests/test_phase9_documentation.py` | test/documentation contract | request-response/static scan | `tests/test_public_api_contract.py` | exact role + flow match |
| `tests/test_phase10_sqlcache_removal.py` | new test/absence contract | static scan/request-response | `tests/test_phase9_documentation.py` | role + flow match |

## Pattern Assignments

### Package public surface: `src/cacheness/__init__.py`

**Analog:** `src/cacheness/storage/__init__.py` (lines 1-24, 165-180).

The package barrel uses a descriptive module docstring and explicit imports/
`__all__`, rather than dynamic discovery. Remove only the SqlCache import and
names, retain the version and supported storage/cache exports, and rewrite the
docstring to describe the post-cutover surface.

```python
"""Storage Layer
===============

Direct object storage is composed from one ``StoreTopology`` and owned by
``BlobStore``.  Catalog authority is not selected independently from the
payload lifecycle.
"""

from .blob_store import BlobStore
from .composition import BackendRef, BackendRole, RoleRegistry, StoreTopology

__all__ = [
    "BlobStore",
    "BackendRef",
    "BackendRole",
    "RoleRegistry",
    "StoreTopology",
]
```

Do not add `__getattr__`, aliases, or a tombstone module. The natural absence
contract is exercised in subprocesses so `sys.modules` cannot mask deletion.

### Error vocabulary: `src/cacheness/error_handling.py`

**Analog:** the local `CacheReason` enum (lines 19-55).

Retain stable `str, Enum` values for shared storage/catalog/migration failures;
remove only values proven to have no remaining callers. Update the exact-set
contract at `tests/test_public_api_contract.py:129-191` in the same change.

```python
class CacheReason(str, Enum):
    """Stable machine-readable reasons for public cache boundary failures."""

    PATH_TRAVERSAL = "path_traversal"
    MANIFEST_INVALID = "manifest_invalid"
    BLOB_LIFECYCLE_CONFLICT = "blob_lifecycle_conflict"
    CATALOG_VALIDATION_FAILED = "catalog_validation_failed"
```

The three SqlCache-specific values and `missing_optional_dependency` are not a
reason to change the shared exception hierarchy. Preserve typed inheritance and
`raise ... from error` handling in the remainder of the utility.

### Manifest and lock edits: `pyproject.toml`, `uv.lock`

**Analogs:** `tools/run_phase8_packaging.py:378-389` and
`tests/packaging/test_wheel_matrix.py:43-62`.

`pyproject.toml` is authoritative. Remove `duckdb-engine` from the two
recommended lists and remove the `[dependency-groups].sql` table; preserve the
six installable extras and remaining SQLAlchemy/pandas/PyArrow/psycopg uses.
Regenerate `uv.lock` with plain `uv lock`, then use `uv lock --check` and inspect
the inverse tree. Do not hand-edit lock package records or upgrade unrelated
dependencies.

```python
def optional_groups_from_pyproject(path: Path) -> tuple[str, ...]:
    """Read and freeze the reviewed optional-dependency inventory exactly."""
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    optional = document["project"]["optional-dependencies"]
    if not isinstance(optional, dict) or tuple(optional) != OPTIONAL_GROUPS:
        raise PackagingQualificationError("optional group inventory does not match review")
    return tuple(optional)
```

### Canonical documentation: `docs/API_REFERENCE.md`, `docs/STORAGE_MIGRATION.md`, `docs/README.md`

**Analogs:** `docs/API_REFERENCE.md:1-6`, `docs/API_REFERENCE.md:44-76`,
`docs/STORAGE_MIGRATION.md:3-20`, and `docs/README.md:3-26`.

Keep guidance task-oriented and explicit about ownership:

```markdown
`BlobStore` owns storage lifecycle; `UnifiedCache` adds cache policy above a
store selected by the application.
```

Add a concise cutover note in these existing owners, not a new guide. Explain
that `UnifiedCache` is for object/function caching, `BlobStore` is for direct
object persistence, there is no in-package range-aware SQL pull-through
replacement, and caller-owned SQL tables are untouched/unsupported. Do not
claim that a retained component reproduces gap detection or table upserts.
Keep the migration guide's explicit offline/versioned tooling intact.

### Mixed documentation and current maps

**Analogs:** `docs/BLOB_STORE.md:1-16` for the current storage narrative and
`examples/README.md:7-20` for a closed canonical journey index.

- `docs/PANDAS_API_AUDIT.md`: remove only the `sql_cache.py` and SqlCache test
  ownership rows at lines 13 and 163-167; retain dataframe/Parquet API claims.
- `docs/CROSS_PLATFORM_GUIDE.md`: preserve the isolated-suite explanation at
  lines 132-140, but remove the stale SQL-cache product claim in lines 142-150.
- `AGENTS.md` and the seven `.planning/codebase/*.md` maps: update current
  architecture, stack, integration, structure, conventions, concerns, and
  testing claims to remove live SqlCache/DuckDB ownership. Preserve truthful
  completed phase records and dated audit history.

Use the same vocabulary as the current architecture: BlobStore is the lifecycle
owner, UnifiedCache is policy above it, handlers remain native-format strategies,
and SQLAlchemy/PostgreSQL survives where retained authorities/projections need it.

### Negative public and absence contract: `tests/test_phase10_sqlcache_removal.py`

**Analog:** `tests/test_public_api_contract.py:31-49, 129-184, 194-233` and
`tests/test_phase071_contract_verifier.py:48-75`.

Use literal, fail-closed assertions. Test source paths, package attributes,
natural import failures, current-surface references, TOML/lock metadata, and
the exact retained version (`0.3.14`). Use an isolated subprocess for import
absence, as the existing public test does for optional-import behavior.

```python
assert "SqlCache" not in cacheness.__all__
assert not hasattr(cacheness, "SqlCache")
assert importlib.util.find_spec("cacheness.sql_cache") is None

with pytest.raises(ModuleNotFoundError):
    importlib.import_module("cacheness.sql_cache")
```

The current-surface scanner must name roots and use a narrow path-purpose
allowlist. Allow only the three canonical cutover notes and negative assertions;
exclude truthful dated audits/completed planning artifacts rather than applying
a repository-wide string ban.

### Public contract inversion: `tests/test_public_api_contract.py`

**Analog:** `tests/test_phase6_public_api_contract.py:46-80`.

Continue freezing the canonical tuple and retained role ownership. Replace the
positive optional SqlCache subprocess at lines 194-233 with a natural-absence
subprocess, remove the four orphan reason values from the exact set, and make
the API-reference check assert the bounded cutover note rather than blanket
rejecting a permitted canonical mention.

```python
for retired in ("SqlCache", "SqlCacheAdapter"):
    assert retired not in cacheness.__all__
    assert not hasattr(cacheness, retired)
```

### Phase 6 public/suite contracts

**Analogs:** `tests/test_public_api_contract.py:31-49` and
`tests/test_phase6_suite_isolation.py:150-193`.

- `tests/test_phase6_public_api_contract.py`: remove SqlCache imports and names
  from `CANONICAL_PUBLIC_NAMES`; invert its separate-surface assertion while
  retaining BlobStore/UnifiedCache module ownership assertions.
- `tests/test_phase6_suite_isolation.py`: retain both relative orders for the
  two surviving public-contract modules. Do not preserve a deleted test node in
  `ORDER_ISOLATION_NODE_ORDERS`.

```python
for first, second in (
    ("tests/test_public_api_contract.py", "tests/test_phase6_public_api_contract.py"),
):
    relative_orders = {
        nodes.index(first) < nodes.index(second)
        for nodes in ORDER_ISOLATION_NODE_ORDERS
    }
    assert relative_orders == {False, True}
```

### Static quality and environment contracts

**Analogs:** `tests/test_phase1_quality_gates.py:56-65, 220-243` and
`tests/test_full_suite_environment.py:15-36`.

- `tests/test_phase1_quality_gates.py`: remove the deleted path constant and
  wave manifest entry. If the no-print sentinel remains valuable, apply it to a
  retained AST helper, following `_module_tree`/`_method_node`, not to a
  nonexistent SqlCache class.
- `tests/test_full_suite_environment.py`: preserve exact frozen-suite command
  assertions, but rewrite the bare-collection diagnostic so it no longer names
  deleted SQL-cache modules. Keep the test-isolation/cascade explanation only
  for still-supported optional dependencies.

### Closed verifier manifests

**Analogs:** `tools/verify_phase6_contracts.py:27-68`,
`tools/verify_phase4_cutover.py:113-189`, and
`tests/test_phase071_contract_verifier.py:48-75`.

All verifier inventories are literal and existence-checked; they must be
repaired before or atomically with deleting dedicated tests.

- `tools/verify_phase4_cutover.py` and `tests/test_phase4_cutover_verifier.py`:
  remove the historical deferred SqlCache paths from live parsing/validation;
  preserve the historical `04-VALIDATION.md` document unchanged and retain the
  Phase 4 owned lifecycle matrix.
- `tools/verify_phase6_contracts.py` and `tests/test_phase6_contract_verifier.py`:
  remove `tests/test_sql_cache.py` and the `CACH-07 SqlCache regression` node;
  let Phase 10 own CACH-07 while Phase 6 continues to own CACH-01 through
  CACH-06 and retained lifecycle regressions.
- `tools/verify_phase071_contracts.py`: replace its positive SqlCache selector
  with the Phase 10 negative selector; leave Phase 07.1's own fixed plan,
  decision, threat, and non-claim inventories intact.

```python
def _normalise_test_path(candidate: str) -> str:
    """Validate a literal repository-relative test module path."""
    path = PurePosixPath(candidate)
    if candidate != path.as_posix() or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"matrix path is not normalized: {candidate}")
    if not (REPOSITORY_ROOT / path).is_file():
        raise ValueError(f"matrix path does not exist: {candidate}")
    return path.as_posix()
```

Do not replace a closed manifest with discovery from a branch diff or arbitrary
repository scraping.

### Fresh wheel proof: `tools/run_phase8_packaging.py` and
`tests/packaging/test_wheel_matrix.py`

**Analog:** `tools/run_phase8_packaging.py:217-238, 241-330` and
`tests/packaging/test_wheel_matrix.py:43-82`.

Extend the existing one-artifact source-free probe. Remove SqlCache from
`BASE_PUBLIC_EXPORTS`, add it to the retired/absent checks, inspect ZIP members
for absence of `cacheness/sql_cache.py`, and inspect installed distribution
metadata for absent DuckDB requirements and absent `sql` extra. Keep the
existing local BlobStore/UnifiedCache round trips and no-import-noise checks.

```python
artifact = runner.build_wheel(tmp_path / "dist")
result = runner.run_base_probe(artifact, workspace=tmp_path / "probe")
assert artifact.path.is_file()
assert len(artifact.sha256) == 64
assert result.probes == (
    "public_exports",
    "blobstore_generic",
    "blobstore_numpy_pickle",
    "blobstore_numpy_npz",
    "unified_cache_generic",
)
```

The wheel path and digest must remain bound through build, ZIP inspection,
isolated installation, and probe. A source-tree import check alone is
insufficient.

### Documentation contract: `tests/test_phase9_documentation.py`

**Analog:** `tests/test_phase9_documentation.py:16-25, 71-89, 285-298`.

Retain helper-based document section extraction, exact navigation assertions,
and relative-link resolution. Change only ownership rules: require the concise
cutover note in API/migration/index owners, continue rejecting stale product
claims in README/non-owner guides, and ensure deleted guide links fail the
link-resolution check.

```python
def _section(source: str, heading: str) -> str:
    """Return a level-two Markdown section without its following peer."""
    start = source.index(heading)
    remainder = source[start + len(heading) :]
    next_heading = remainder.find("\n## ")
    return remainder if next_heading == -1 else remainder[:next_heading]
```

## Shared Patterns

### Explicit inventories, not discovery

Public exports, optional groups, verifier nodes, canonical examples, and wheel
probe names are all literal tuples/dicts in current code. Update their fixed
inventories deliberately and test omissions/renames as failures. Never derive
Phase 10 scope from `git diff`, recursive string replacement, or current
`__all__` at runtime.

### Natural absence and fail-closed boundaries

Removal means physical deletion and ordinary Python failure. Do not add a
tombstone, compatibility alias, `__getattr__`, tailored exception, database
cleanup, or replacement query adapter. Preserve narrow exception handling and
causal chaining in shared code.

### Canonical ownership and historical truth

Current docs must describe only the supported BlobStore/UnifiedCache surface;
dedicated obsolete pages/examples/tests are deleted. Mixed current docs are
surgically edited. Dated audits and completed planning records remain truthful
history and are not subject to blanket erasure.

### Artifact-first package qualification

The acceptance chain is: source/static absence -> TOML/lock absence -> current
reference scan -> retained local regression -> exact fresh wheel member and
installed-metadata inspection. The wheel probe must run outside the checkout and
must retain the existing representative native-format round trips.

### No lifecycle scope expansion

Do not modify BlobStore, UnifiedCache, lifecycle authorities, payload
participants, handler protocols, topology guarantees, or caller databases.
Phase 10 removes an unrelated product and leaves ADR 0001's one-authority
boundary intact. SQLAlchemy/PostgreSQL and dataframe handlers remain where
their supported owners still require them.

## No Analog Found

These files are intentionally removed rather than reimplemented:

| File | Role | Data Flow | Reason |
|---|---|---|---|
| `src/cacheness/sql_cache.py` | component/service | CRUD/batch SQL | Separate range-aware product is out of scope; no replacement is planned. |
| `tests/test_sql_cache.py` | test | CRUD/batch | Dedicated product contract is deleted. |
| `tests/test_sql_cache_documentation.py` | test | documentation | Dedicated product documentation contract is deleted. |
| `tests/test_sql_cache_failure_contract.py` | test | request-response | Dedicated failure contract is deleted with its product. |
| `docs/SQL_CACHE.md` | documentation | request-response | Obsolete product guide. |
| `docs/CUSTOM_GAP_DETECTION.md` | documentation | CRUD/batch | Obsolete gap-detection guide. |
| `docs/ARBITRARY_TIME_INCREMENTS.md` | documentation | CRUD/batch | Obsolete query feature guide. |
| `examples/beginner_sql_cache.py` | example | request-response | Obsolete SQL-cache journey. |
| `examples/simple_stock_cache.py` | example | request-response | Obsolete SQL-cache journey. |
| `examples/stock_cache_example.py` | example | request-response | Obsolete SQL-cache journey. |
| `examples/database_backend_comparison.py` | example | request-response | Mixed obsolete journey already superseded by canonical examples. |
| `examples/intelligent_storage_demo.py` | example | request-response | Mixed obsolete journey already superseded by canonical examples. |
| `examples/simple_backend_demo.py` | example | request-response | Mixed obsolete journey already superseded by canonical examples. |

## Metadata

**Analog search scope:** `src/cacheness/`, `tests/`, `tools/`, `docs/`,
`examples/`, `AGENTS.md`, `.planning/codebase/`, `pyproject.toml`, and
`uv.lock`.  
**Files scanned:** 43 planned files plus 12 strong analogs.  
**Pattern extraction date:** 2026-09-17

