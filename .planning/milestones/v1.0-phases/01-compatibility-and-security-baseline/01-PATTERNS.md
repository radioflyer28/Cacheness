# Phase 1: Compatibility and Security Baseline - Pattern Map

**Mapped:** 2026-08-29  
**Files analyzed:** 16 new/modified paths (including the proposed compatibility fixture directory)  
**Analogs found:** 15 / 16 (the shared path-security utility is a new boundary)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `src/cacheness/storage/path_security.py` | utility | file-I/O | `src/cacheness/error_handling.py:185-225` (`validate_file_path`) | partial (path policy is new) |
| `src/cacheness/error_handling.py` | utility | request-response | `src/cacheness/error_handling.py:18-112` | exact (extend existing hierarchy) |
| `src/cacheness/__init__.py` | provider/barrel | request-response | `src/cacheness/__init__.py:32-94,207-291` | exact |
| `src/cacheness/config.py` | config | transform | `src/cacheness/config.py:20-42,1032-1144` | exact |
| `src/cacheness/handlers.py` | component/serializer | file-I/O | `src/cacheness/handlers.py:426-615` | exact |
| `src/cacheness/core.py` | service/coordinator | CRUD + request-response | `src/cacheness/core.py:535-651,924-1053` | exact |
| `src/cacheness/sql_cache.py` | service | batch/request-response | `src/cacheness/sql_cache.py:96-139,486-529,600-626,710-756` | exact |
| `src/cacheness/storage/blob_store.py` | service | file-I/O + CRUD | `src/cacheness/storage/blob_store.py:128-242,285-365` | exact |
| `src/cacheness/storage/backends/blob_backends.py` | backend | file-I/O/streaming | `src/cacheness/storage/backends/blob_backends.py:203-324` | exact |
| `tests/test_public_api_contract.py` | test | request-response/transform | `tests/test_config_validation.py:472-508`, `tests/test_backend_compatibility.py` | role-match |
| `tests/test_stored_compatibility.py` | test/fixture integration | file-I/O + transform | `tests/test_config_validation.py:431-466,560-588`, `tests/test_cross_system_compatibility.py` | role-match |
| `tests/test_filesystem_containment.py` | test | file-I/O/streaming | `tests/test_directory_sharding.py:66-115,265-315`, `tests/test_blob_backend_registry.py:633-645` | role-match |
| `tests/test_legacy_array_security.py` | test | file-I/O/transform | `tests/test_handlers.py:553-593`, `src/cacheness/handlers.py:426-615` | role-match |
| `tests/test_query_meta_security.py` | test | request-response/CRUD | `tests/test_query_meta.py:19-140` | exact behavior, security extension |
| `tests/test_sql_cache_failure_contract.py` | test | batch/request-response | `tests/test_sql_cache.py:21-117,169-205,224-300` | exact behavior, failure extension |
| `tests/fixtures/compat/` (immutable artifacts + provenance) | fixture | file-I/O/transform | `tests/test_config_validation.py:431-466` | no existing fixture corpus; use research procedure |
| `docs/SECURITY.md` (and README security section if kept as the public landing page) | documentation | request-response | `docs/SECURITY.md:1-27,142-176,249-265` | role-match |

`CONTEXT.md`/`RESEARCH.md` explicitly recommend the first seven new implementation/test paths and a checked-in compatibility fixture set. Existing files in the table are the integration points named by the research: exports, typed errors, authored paths, handlers, query construction, `SqlCache`, persisted blob locators, and security documentation. Do not introduce a manifest, migration runner, or new raw-array container in this phase.

## Pattern Assignments

### `src/cacheness/storage/path_security.py` (utility, file-I/O)

**Analog:** `src/cacheness/error_handling.py:185-225`, with the containment policy derived from `src/cacheness/storage/backends/blob_backends.py:219-324`.

This is a new shared boundary; there is no existing safe containment helper. Keep it small and standard-library-only. The existing helper shows the project’s `Path` normalization and domain-error style, but it creates directories and only checks an optional existence flag, so it must not be copied as-is for hostile persisted locators.

**Existing path/error pattern** (`src/cacheness/error_handling.py:199-225`):

```python
path = Path(file_path)
if not path.name:
    raise CacheStorageError(
        "Invalid file path: empty filename", {"file_path": str(file_path)}
    )
...
except OSError as e:
    raise CacheStorageError(
        f"File system error: {e}",
        {"file_path": str(file_path), "must_exist": must_exist},
    ) from e
```

**Backend pattern to replace** (`src/cacheness/storage/backends/blob_backends.py:306-324`):

```python
safe_id = blob_id.replace("..", "__").replace("/", os.sep).replace("\\", os.sep)
if self.shard_chars > 0 and len(safe_id) >= self.shard_chars:
    shard_dir = safe_id[:self.shard_chars]
    return self.base_dir / shard_dir / safe_id
return self.base_dir / safe_id
```

Planner should assign one guard API that (1) resolves the configured root itself, (2) rejects opaque IDs outside the restricted grammar plus POSIX absolute, drive, UNC, rooted-backslash, mixed-separator, and traversal forms, (3) rejects existing symlink components below the root, and (4) proves the final path is contained immediately before access. Preserve the authored config string; runtime resolution belongs here. Raise a typed unsafe-path error with a stable reason code and do not sanitize-and-continue.

### `src/cacheness/error_handling.py` (utility, request-response)

**Analog:** the current module itself (`src/cacheness/error_handling.py:18-112,119-152`). Extend the hierarchy rather than exposing `ValueError`, `OSError`, or SQLAlchemy exceptions at public boundaries. Candidate categories from the phase are unsafe path, query validation, format/security, optional dependency, and strict partial-fetch errors; exact class/reason names are still a planner choice and become contractual once selected.

**Imports and domain base** (`lines 8-15,18-29`):

```python
import functools
import logging
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, Union
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class CacheError(Exception):
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.context = context or {}
        super().__init__(message)
        ...
```

**Cause/context translation** (`lines 84-112`):

```python
try:
    return func(*args, **kwargs)
except CacheError:
    raise
except Exception as e:
    error_context = (context or {}).copy()
    error_context.update({
        "function": func.__name__,
        "args_count": len(args),
        "kwargs_keys": list(kwargs.keys()),
        "original_error": str(e),
        "original_error_type": type(e).__name__,
    })
    cache_error = error_type(f"Error in {func.__name__}: {e}", error_context)
    raise cache_error from e
```

Use `context` for stable machine-readable `reason`/`code`, operation, and locator/field details. Narrow operational catches at new boundaries; do not route security validation through the decorator’s fallback/miss behavior.

### `src/cacheness/__init__.py` (provider/barrel, request-response)

**Analog:** current export and optional-import blocks (`src/cacheness/__init__.py:32-94,207-291`). Keep the public inventory centralized and retain compatibility aliases.

**Import/optional dependency pattern** (`lines 32-36,56-82`):

```python
from .core import CacheConfig, UnifiedCache as cacheness, get_cache
from .decorators import cached
from .handlers import ArrayHandler, HandlerRegistry, ObjectHandler
from .metadata import JsonBackend, create_metadata_backend

try:
    from .config import load_config_from_yaml, save_config_to_yaml
    _has_yaml_config = True
except ImportError:
    _has_yaml_config = False

try:
    from .sql_cache import SqlCache, SqlCacheAdapter
    _has_sql_cache = True
    SQLAlchemyPullThroughCache = SqlCache
    SQLAlchemySqlCacheAdapter = SqlCacheAdapter
except ImportError:
    _has_sql_cache = False
```

**Inventory pattern** (`lines 207-244,246-291`):

```python
__all__ = ["cacheness", "CacheConfig", ... "cached", "__version__"]
if _has_yaml_config:
    __all__.extend(["load_config_from_yaml", "save_config_to_yaml"])
if _has_sql_cache:
    __all__.extend([
        "SqlCache", "SqlCacheAdapter",
        "SQLAlchemyPullThroughCache", "SQLAlchemyDataAdapter",
    ])
```

Characterization must exercise `from cacheness import *`, every guaranteed historical symbol in both full and dependency-blocked subprocesses, and signatures/aliases. Fix the current name mismatch (`SQLAlchemySqlCacheAdapter` defined, `SQLAlchemyDataAdapter` listed) without freezing the broken star import. Optional symbols should remain importable; dependency errors move to construction/use and should be actionable.

### `src/cacheness/config.py` (config, transform)

**Analog:** dataclass/config serialization in `src/cacheness/config.py:20-42,1032-1144`, plus `tests/test_config_validation.py:431-466,610-658`.

**Dataclass and current path behavior** (`lines 20-42`):

```python
@dataclass
class CacheStorageConfig:
    cache_dir: str = "./cache"
    ...
    def __post_init__(self):
        if self.max_cache_size_mb is not None and self.max_cache_size_mb <= 0:
            raise ValueError("max_cache_size_mb must be positive")
        if not Path(self.cache_dir).is_absolute() and self.cache_dir != "./cache":
            self.cache_dir = str(Path.cwd() / self.cache_dir)
```

**Serialization pattern** (`lines 1087-1102,1130-1144`):

```python
data = {
    "storage": asdict(config.storage),
    "metadata": asdict(config.metadata),
    "blob": asdict(config.blob),
    "compression": asdict(config.compression),
    "serialization": asdict(config.serialization),
    "handlers": asdict(config.handlers),
    "security": asdict(config.security),
}
with path.open("w", encoding="utf-8") as f:
    json.dump(data, f, indent=indent, default=str)
```

Preserve every user-authored relative `cache_dir` in `CacheStorageConfig`, JSON, and YAML. Add a separate runtime `Path.resolve(strict=False)` at filesystem operations; do not resolve in `__post_init__`. Retain the nested dataclass shape and `asdict` round-trip. Keep lazy `import yaml` and actionable `ImportError` wording from `lines 1059-1071,1122-1128`.

### `src/cacheness/handlers.py` (component/serializer, file-I/O)

**Analog:** `ArrayHandler` itself (`src/cacheness/handlers.py:426-615`), with pytest handler/fallback style from `tests/test_handlers.py:553-593`.

**Imports and optional dependency pattern** (`lines 9-22,24-69`):

```python
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional
from .interfaces import (
    CacheHandler, CacheHandlerError, CacheWriteError,
    CacheReadError, CacheFormatError,
)
from .error_handling import cache_operation_context
try:
    import blosc2
    BLOSC2_AVAILABLE = True
except ImportError:
    blosc2 = None
    BLOSC2_AVAILABLE = False
```

**Declared format and legacy writer** (`lines 448-490,542-553`):

```python
if config.compression.use_blosc2_arrays and BLOSC2_AVAILABLE:
    blosc2_path = file_path.with_suffix("").with_suffix(".b2nd")
    self._write_blosc2_array(data, blosc2_path, config)
    return {"storage_format": "blosc2", "file_size": blosc2_path.stat().st_size,
            "actual_path": str(blosc2_path), ...}
...
shape_bytes = str(data.shape).encode("utf-8")
dtype_bytes = str(data.dtype).encode("utf-8")
f.write(len(shape_bytes).to_bytes(4, "little"))
f.write(shape_bytes)
f.write(len(dtype_bytes).to_bytes(4, "little"))
f.write(dtype_bytes)
f.write(compressed_data)
```

**Current unsafe reader/dispatch to harden** (`lines 555-604`):

```python
shape_len = int.from_bytes(f.read(4), "little")
shape_str = f.read(shape_len).decode("utf-8")
dtype_len = int.from_bytes(f.read(4), "little")
dtype_str = f.read(dtype_len).decode("utf-8")
decompressed = blosc2.decompress2(compressed_data)
shape = eval(shape_str)
dtype = np.dtype(dtype_str)
return np.frombuffer(decompressed, dtype=dtype).reshape(shape)
...
data = np.load(npz_path, allow_pickle=True)
```

Replace only the legacy read path: parse the fixed tuple grammar with bounded manual checks; validate framing, rank/dimensions, non-object dtype, checked element count, and `dtype.itemsize * count == len(decompressed)` before reconstruction. A declared `blosc2` artifact has exactly one reader—malformed input raises typed format/security error and never falls through to NPZ. Ordinary NPZ uses `with np.load(..., allow_pickle=False)` and rejects object arrays unless an explicit trusted-object path is selected. Retain native Blosc2/NPZ containers and do not add a new raw-array format or write the legacy header.

### `src/cacheness/core.py` (service/coordinator, CRUD + request-response)

**Analog:** existing `UnifiedCache.query_meta` and `get` (`src/cacheness/core.py:535-651,924-1053`). Keep cache policy and metadata ownership in `UnifiedCache`; do not move this work into `BlobStore`.

**Initialization and optional backend style** (`lines 68-103,109-146`):

```python
self.config = config or CacheConfig()
self.cache_dir = Path(self.config.storage.cache_dir)
self.cache_dir.mkdir(exist_ok=True, parents=True)
self._lock = threading.Lock()
self.handlers = HandlerRegistry(self.config)
self._init_metadata_backend(metadata_backend)
```

**Current query shape to preserve while securing** (`lines 577-623`):

```python
from sqlalchemy import text
with self.metadata_backend.SessionLocal() as session:
    where_conditions = []
    params = {}
    for key, value in filters.items():
        param_name = f"param_{len(params)}"
        if isinstance(value, (int, float)):
            where_conditions.append(
                f"(JSON_EXTRACT(cache_key_params, '$.{key}') = :{param_name} OR "
                f"CAST(JSON_EXTRACT(cache_key_params, '$.{key}') AS REAL) >= :{param_name})"
            )
        else:
            where_conditions.append(
                f"JSON_EXTRACT(cache_key_params, '$.{key}') = :{param_name}"
            )
        params[param_name] = value
```

Validate every field path before `SessionLocal()` is called. Construct JSON paths through SQLAlchemy expressions/bound parameters (values and paths both out of SQL text), retaining raw numeric int/float >= semantics and exact string semantics. Reject the whole query with the typed query-validation error; do not return `None` or catch it in the generic block at `lines 649-651`.

**Persisted-locator/error boundary** (`lines 958-1053`):

```python
metadata = entry.get("metadata", {})
actual_path = metadata.get("actual_path")
file_path = Path(actual_path) if actual_path else base_file_path
...
except FileNotFoundError:
    self.metadata_backend.remove_entry(cache_key)
    self.metadata_backend.increment_misses()
    return None
except Exception as e:
    self.metadata_backend.remove_entry(cache_key)
    self.metadata_backend.increment_misses()
    return None
```

Validate `actual_path` before integrity checks, handler reads, or generic exception translation. Unsafe-path errors must bypass cleanup/miss conversion and leave metadata and payload untouched. Preserve existing ordinary missing/corrupt-entry behavior for non-security failures.

### `src/cacheness/sql_cache.py` (service, batch/request-response)

**Analog:** adapter contract and constructor (`src/cacheness/sql_cache.py:52-139,301-379`) plus existing integration tests (`tests/test_sql_cache.py:21-117,169-205,224-300`). Keep `SqlCache` architecturally separate from `UnifiedCache` and `BlobStore`.

**Adapter contract** (`lines 96-139`):

```python
class SqlCacheAdapter(ABC):
    @abstractmethod
    def get_table_definition(self) -> 'Table': ...
    @abstractmethod
    def fetch_data(self, **kwargs) -> 'pd.DataFrame': ...
    @abstractmethod
    def parse_query_params(self, **kwargs) -> Dict[str, Any]: ...
```

**Current fetch loop to preserve and change** (`lines 502-529`):

```python
parsed_params = self.data_adapter.parse_query_params(**query_params)
with self.Session() as session:
    cached_data = self._get_cached_data(session, parsed_params)
    missing_ranges = self._find_missing_data(parsed_params, cached_data)
    if missing_ranges:
        for missing_params in missing_ranges:
            try:
                fresh_data = self.data_adapter.fetch_data(**missing_params)
                if not fresh_data.empty:
                    self._store_in_cache(session, fresh_data)
            except Exception as e:
                print(f"Warning: Failed to fetch data for {missing_params}: {e}")
        session.commit()
    return self._get_cached_data(session, parsed_params)
```

Strict mode must rollback and raise a typed, cause-preserving fetch error identifying every failed range; it must never return incomplete data as complete. An explicit best-effort mode may continue, but returns partial data through a caller-inspectable failure channel and logs every failed range structurally. Parse/query and caller adapter failures raise unless an explicit fallback policy allows otherwise.

**Equivalent internal fallback** (`lines 620-626`):

```python
try:
    self._upsert_records(session, records)
except Exception as e:
    print(f"Upsert failed, using fallback method: {e}")
    self._fallback_upsert(session, records)
```

Retain bulk-to-row fallback because it is semantically equivalent, but replace `print` with `logger.warning(..., extra=...)`; if both bulk and row operations fail, raise with cause. Custom gap detector failure at `lines 736-743` is caller-controlled and must raise by default, not silently switch to built-in detection.

### `src/cacheness/storage/blob_store.py` (service, file-I/O + CRUD)

**Analog:** `BlobStore.put/get/delete/exists/list` (`src/cacheness/storage/blob_store.py:128-242,285-365`). It currently demonstrates the metadata/payload split that Phase 1 hardens without redesigning lifecycle ownership.

**Put metadata pattern** (`lines 146-180`):

```python
if self.content_addressable:
    blob_key = self._compute_content_hash(data)
elif key:
    blob_key = self._sanitize_key(key)
else:
    blob_key = self._generate_unique_key()
handler = self.handlers.get_handler(data)
base_path = self.cache_dir / blob_key
result = handler.put(data, base_path, self.config)
custom_metadata["actual_path"] = str(result.get("actual_path", base_path))
entry_data = {"cache_key": blob_key, "data_type": handler.data_type,
              "file_size": result.get("file_size", 0), ...}
self.backend.put_entry(blob_key, entry_data)
```

**Current inconsistent locator paths** (`lines 196-217,295-326`):

```python
actual_path_str = entry.get("actual_path") or entry.get("metadata", {}).get("actual_path")
actual_path = Path(actual_path_str) if actual_path_str else self.cache_dir / key
if not actual_path.exists():
    return None
...
actual_path = Path(entry.get("actual_path", self.cache_dir / key))
if actual_path.exists():
    actual_path.unlink()
```

Use the shared guard for key and every persisted `actual_path` before get, metadata-derived access, delete, exists, list-derived access, and clear. For an unsafe locator, raise and preserve metadata/outside bytes; do not return `None`/`False`, remove entries, or follow common-extension guesses. Keep nested metadata flattening and handler dispatch for safe entries.

### `src/cacheness/storage/backends/blob_backends.py` (backend, file-I/O/streaming)

**Analog:** `FilesystemBlobBackend` (`src/cacheness/storage/backends/blob_backends.py:203-324`) and its abstract contract (`lines 56-196`). The backend is the lowest filesystem boundary and should own opaque ID validation, sharding, root resolution, and all operation guards.

**Atomic write pattern** (`lines 232-248`):

```python
blob_path = self._get_blob_path(blob_id)
blob_path.parent.mkdir(parents=True, exist_ok=True)
temp_path = blob_path.with_suffix(blob_path.suffix + ".tmp")
try:
    temp_path.write_bytes(data)
    temp_path.replace(blob_path)
except Exception:
    if temp_path.exists():
        temp_path.unlink()
    raise
return str(blob_path)
```

Apply the guard before parent creation and before each `read_blob`, `delete_blob`, `exists`, `read_blob_stream`, and `get_size` operation (`lines 250-304`). Resolve `base_dir` once as the allowed root (root symlink allowed), reject symlink components below it immediately before access, and return paths compatible with existing sharding tests. Opaque IDs must be restricted; user-facing logical keys are encoded/hashed before this backend. Preserve atomic temp-file replacement and cleanup, but narrow new exception translation to domain errors where appropriate.

### `tests/test_public_api_contract.py` (test, request-response/transform)

**Analog:** `tests/test_config_validation.py:472-508` and registry import tests in `tests/test_blob_backend_registry.py:12-47`. Use pytest modules/classes, direct assertions, and subprocess/import blocking for optional dependencies.

```python
class TestModuleLevelConfigAPI:
    def test_config_classes_exported(self):
        assert hasattr(cacheness, "CacheConfig")
        assert hasattr(cacheness, "CacheStorageConfig")
    def test_loading_functions_exported(self):
        assert hasattr(cacheness, "load_config_from_dict")
        assert hasattr(cacheness, "load_config_from_json")
```

Build a matrix from `__all__`, documented imports/examples, constructors/signatures, aliases, registries, decorators, exception classes/reasons, and result shapes. Include `from cacheness import *`, `SQLAlchemyDataAdapter`, YAML/SQLAlchemy/pandas absence, and actionability of deferred dependency errors. Tests must assert corrected intended behavior, not the known broken star import or unsafe sanitization.

### `tests/test_stored_compatibility.py` (test/fixture integration, file-I/O + transform)

**Analog:** config save/reload (`tests/test_config_validation.py:431-466,560-588`) and existing compatibility suite files (for example `tests/test_cross_system_compatibility.py`).

```python
original = CacheConfig(cache_dir="./cache", metadata_backend="sqlite", blob_backend="memory")
config_file = temp_dir / "saved_config.json"
save_config_to_json(original, config_file)
loaded = load_config_from_json(config_file)
assert loaded.storage.cache_dir == original.storage.cache_dir
```

Generate candidate artifacts in isolated historical checkouts for identifiable `0.3.7`, `0.3.9`, `0.3.13`, and `0.3.14` layouts, then check in only readable/safely identifiable artifacts plus provenance and expected-read metadata under `tests/fixtures/compat/`. Tests consume immutable fixtures and current readers; never install/network-fetch old releases during pytest. Cover config JSON/YAML relative-path round trips and representative current/earlier stored entries. Unsafe or unreadable formats are documented exclusions, not silently accepted.

### `tests/test_filesystem_containment.py` (test, file-I/O/streaming)

**Analog:** temp-dir and sharding tests (`tests/test_directory_sharding.py:66-115,265-315`) plus large-blob read/write (`tests/test_blob_backend_registry.py:633-645`).

```python
@pytest.fixture
def temp_dir(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

def test_shard_chars_two_creates_subdirectory(self, temp_dir):
    backend = FilesystemBlobBackend(temp_dir, shard_chars=2)
    blob_path = backend.write_blob("abc123def456", b"test data")
    path = Path(blob_path)
    assert path.parent.name == "ab"
    assert path.parent.parent == temp_dir
```

Replace the current `test_path_traversal_safe_with_sharding` expectation (`lines 274-288`, which expects sanitization) with `pytest.raises` for traversal, POSIX absolute, drive, UNC, rooted backslash, and mixed separator corpora. Parameterize all read/write/delete/exists/list/stream/size entry points, test allowed root symlink versus rejected ancestor/leaf managed symlinks, and snapshot metadata plus outside bytes to prove rejected persisted locators mutate nothing.

### `tests/test_legacy_array_security.py` (test, file-I/O/transform)

**Analog:** handler fallback tests (`tests/test_handlers.py:553-593`) and the current `ArrayHandler` format tests (`src/cacheness/handlers.py:519-604`). Use temporary paths, NumPy arrays, `pytest.raises`, and optional `skipif` only when the dependency itself is unavailable.

```python
def test_compression_fallback(self, config):
    with tempfile.TemporaryDirectory() as temp_dir:
        handler = ObjectHandler()
        file_path = Path(temp_dir) / "test"
        metadata = handler.put({"large": list(range(1000))}, file_path, config)
        loaded_data = handler.get(Path(metadata["actual_path"]), metadata)
        assert loaded_data == {"large": list(range(1000))}
```

Create valid scalar/1-D/n-D legacy `.b2nd` fixtures and a negative corpus: truncated length words, oversized headers, invalid UTF-8, non-tuple grammar, negative/excessive dimensions, object dtype, decompression failure, byte mismatch, and malformed declared Blosc2 with a valid NPZ sidecar. Assert typed format/security errors, no `eval`/`literal_eval`, `allow_pickle=False`, and no declared-format sidecar fallback. Test ordinary arrays and explicit trusted-object behavior separately.

### `tests/test_query_meta_security.py` (test, request-response/CRUD)

**Analog:** `tests/test_query_meta.py:19-140`.

```python
@pytest.fixture
def temp_cache():
    temp_dir = tempfile.mkdtemp()
    config = CacheConfig(
        cache_dir=str(Path(temp_dir) / "cache"),
        metadata_backend="sqlite",
        store_cache_key_params=True,
    )
    cache = UnifiedCache(config)
    yield cache
    cache.close()
```

Retain the existing setup and result assertions, but add raw `int`/`float` tests for >= (the current test passes serialized strings at `lines 103-122` and therefore does not exercise numeric behavior), exact raw strings, safe nested fields, hostile field syntax, and all-fields-validate-first call-order spies proving no `SessionLocal`/DB access on invalid input. Assert typed query-validation errors identify the field and are never converted to `None`.

### `tests/test_sql_cache_failure_contract.py` (test, batch/request-response)

**Analog:** existing SQLite adapter fixtures (`tests/test_sql_cache.py:21-57`) and success/lifecycle tests (`lines 59-117,169-205,224-300`).

```python
class TestSqlCacheAdapter(SqlCacheAdapter):
    def parse_query_params(self, **kwargs):
        return kwargs
    def fetch_data(self, **kwargs):
        return pd.DataFrame(self.test_data)
```

Use deterministic failing adapters/gap detectors and SQLite sessions. Cover one/multiple missing ranges, first/middle/last failure, empty fetches, custom gap-detector exceptions, strict default rollback/typed error with failed ranges, explicit best-effort partial result with every failure reported/logged, and bulk-upsert failure followed by row fallback success/failure. Preserve independent import and normal success coverage from `tests/test_sql_cache.py`.

### `tests/fixtures/compat/` (fixture, file-I/O/transform)

**Analog:** no existing checked-in cross-release artifact corpus. Follow the research procedure: isolated historical checkout generation, immutable copied bytes, and provenance/expected-read metadata. Do not make this directory depend on package installation, network, migration code, or a new container format. If a historical artifact cannot be safely identified/read, record the exclusion instead of adding a permissive reader.

### `docs/SECURITY.md` / README security section (documentation, request-response)

**Analog:** current signing/security guide (`docs/SECURITY.md:1-27,142-176,249-265`) and existing README serializer warnings (`README.md:260-357`).

```markdown
## Security Considerations

**Protected Against:**
- Cache metadata tampering ...

**Not Protected Against:**
- Direct file system access to cached data files
```

Add a clearly scoped trusted-application-payload boundary: pickle and dill deserialization can execute code; integrity/HMAC verifies authenticity but does not make hostile executable serialization safe; ordinary NumPy loading disables pickle; object-dtype arrays require explicit trusted configuration; unsafe serializers require the documented configuration and threat model. Keep examples and links consistent with actual defaults. A documentation assertion test may use the project’s existing pytest style, but prose review remains the final check.

## Shared Patterns

### Typed errors and causes

**Source:** `src/cacheness/error_handling.py:18-29,84-112,221-225`  
**Apply to:** path utility, handlers, core, `SqlCache`, `BlobStore`, and public API tests.

```python
raise CacheStorageError(
    f"File system error: {e}",
    {"file_path": str(file_path), "must_exist": must_exist},
) from e
```

Keep stable machine-readable reason codes in error context (or the selected exception API), preserve `raise ... from e`, and use `pytest.raises(SpecificType, match=...)` where message fragments are stable. Security/validation outcomes must not be collapsed to a miss, `False`, `None`, or generic `Exception`.

### Temporary filesystem and integration fixtures

**Source:** `tests/test_query_meta.py:19-39`, `tests/test_config_validation.py:39-49`, `tests/test_directory_sharding.py:69-73`  
**Apply to:** all new filesystem, config, array, compatibility, and SQL tests.

```python
with tempfile.TemporaryDirectory() as tmpdir:
    yield Path(tmpdir)
```

Use real local filesystem/SQLite integrations and mocks/spies only at optional-service/session boundaries. Snapshot bytes and metadata before hostile operations. Keep optional dependency tests isolated and skip based on the dependency itself.

### Optional imports and actionable dependency failures

**Source:** `src/cacheness/__init__.py:56-82`, `src/cacheness/handlers.py:24-69`, `src/cacheness/sql_cache.py:52-93`.  
**Apply to:** exports, handlers, and `SqlCache` tests/implementation.

```python
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
```

Keep historical top-level names importable even when extras are absent; fail at use/construction with install-oriented typed errors. Avoid eager optional imports that make the baseline import matrix environment-dependent.

### Atomic payload writes and metadata ordering

**Source:** `src/cacheness/storage/backends/blob_backends.py:232-248`; metadata recording in `src/cacheness/storage/blob_store.py:161-180` and `src/cacheness/core.py:811-922` (research reference).  
**Apply to:** filesystem/blob lifecycle edits and compatibility tests.

Write payloads through a temp file and atomic replace, then record metadata. Before any read/delete/exists/list access to metadata-controlled paths, validate containment. On unsafe locators, do not clean up metadata or payload; migration/reconciliation is explicitly out of scope for Phase 1.

### Declared-format dispatch

**Source:** `src/cacheness/handlers.py:579-604`.  
**Apply to:** `ArrayHandler` and legacy array tests.

The persisted `storage_format` chooses exactly one reader. A malformed declared artifact is a typed failure, not permission to guess another extension. NPZ access is context-managed and `allow_pickle=False`; legacy Blosc2 reads are bounded and read-only.

### Strict-by-default failure, explicit fallback only

**Source:** `src/cacheness/sql_cache.py:502-529,620-626,736-743`.  
**Apply to:** `SqlCache` and its failure-contract tests.

Caller adapter, parser, and gap-detector failures are raised with cause by default. Equivalent internal bulk-to-row fallback may warn structurally and proceed; if fallback fails, raise. Explicit best effort must report every failed range and partiality in a stable result channel.

## No Analog Found

| File | Role | Data Flow | Reason |
|---|---|---|---|
| `src/cacheness/storage/path_security.py` | utility | file-I/O | No existing helper rejects cross-platform path forms and managed symlinks; `validate_file_path` only validates a filename and may create directories. |
| `tests/fixtures/compat/` | fixture | file-I/O/transform | No immutable cross-release artifact corpus exists; research requires generating and vetting it from identifiable commits. |

## Metadata

**Analog search scope:** `src/cacheness/`, `src/cacheness/storage/`, `src/cacheness/storage/backends/`, `tests/`, `docs/`, `README.md`  
**Files scanned:** 16 primary analogs plus related registry/handler/config/security tests  
**Pattern extraction date:** 2026-08-29
