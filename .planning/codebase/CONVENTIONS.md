# Coding Conventions

**Analysis Date:** 2026-04-02

## Code Style

**Formatting:**
- Tool: **Ruff** (configured in `pyproject.toml`)
- Line length: **88** characters
- Target version: **Python 3.12** (`target-version = "py312"`)
- Run: `uv run ruff format .` (auto-format), `uv run ruff check --fix .` (auto-fix lint)

**Type Checking:**
- Tool: **ty** (`uv run ty check`)
- Configured as dev dependency in `pyproject.toml`

**Package Manager:**
- **uv** exclusively — never raw `pip` or `python`
- Commands: `uv run pytest`, `uv run python`, `uv sync`, `uv add <pkg>`

## Naming Patterns

**Files:**
- Snake_case for all Python modules: `blob_store.py`, `error_handling.py`, `size_utils.py`
- Test files prefixed with `test_`: `test_core.py`, `test_blob_store.py`
- No abbreviations in module names (except established ones like `config`, `utils`)

**Classes:**
- PascalCase: `UnifiedCache`, `BlobStore`, `HandlerRegistry`, `CacheEntrySigner`
- ABC/interface classes: `MetadataBackend`, `CacheHandler`, `BlobBackend`
- Exception classes: `CacheError`, `CacheStorageError`, `CacheConfigurationError`
- TypedDicts: `SignableFields`, `EntrySummary`, `BlobReadContext`
- Dataclasses: `CacheConfig`, `CacheStorageConfig`, `NamespaceInfo`
- Internal helper classes: `_PutCleanup` (leading underscore for private)

**Functions/Methods:**
- snake_case: `put_entry()`, `get_handler()`, `verify_integrity()`
- Private methods: leading underscore `_init_metadata_backend()`, `_cleanup_expired()`
- Module-level helpers: leading underscore `_normalize_function_args()`, `_serialize_path_object()`
- Boolean helpers: `is_pickleable()`, `is_dill_serializable()`

**Variables:**
- snake_case: `cache_dir`, `blob_path`, `entry_data`
- Module-level constants: UPPER_SNAKE_CASE — `DEFAULT_NAMESPACE`, `BLOSC2_AVAILABLE`, `NAMESPACE_ID_PATTERN`
- Availability flags: `POLARS_AVAILABLE`, `PANDAS_AVAILABLE`, `TENSORFLOW_AVAILABLE`
- Private module-level state: `_custom_metadata_registry`, `_decorator_cache_instances`

**Constants/Sentinels:**
- Sentinel objects: `_DEFAULT_TTL = object()` (to distinguish None from unspecified)
- Regex patterns: `NAMESPACE_ID_PATTERN = re.compile(r"^[a-z0-9_]{1,48}$")`

## Import Organization

**Order (enforced by Ruff):**
1. Standard library (`import os`, `import threading`, `from pathlib import Path`)
2. Third-party packages (`import xxhash`, `import numpy as np`)
3. Local/project imports (`from .core import UnifiedCache`, `from .interfaces import ...`)

**Relative imports within package:**
- Always use relative imports within `src/cacheness/`: `from .config import CacheConfig`
- Cross-subpackage: `from ..config import CacheConfig` (in `storage/` subpackage)

**Optional dependency pattern:**
- Use try/except blocks with availability flags at module level:
```python
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    pl = None  # type: ignore
    POLARS_AVAILABLE = False
```
- Used extensively in `src/cacheness/handlers.py`, `src/cacheness/serialization.py`

**Lazy imports for heavy dependencies:**
- TensorFlow uses lazy import pattern via `_lazy_import_tensorflow()` in `src/cacheness/handlers.py`
- Avoids slow startup times and system-level issues

**noqa usage:**
- `# noqa: F401` for re-exports in `__init__.py` files
- Import `# noqa: E402` collapsed to single-line to suppress properly (documented gotcha)

## Error Handling

**Exception hierarchy** (defined in `src/cacheness/error_handling.py`):
```
CacheError (base)
├── CacheConfigurationError
├── CacheStorageError
├── CacheSerializationError
├── CacheHandlerError
├── CacheIntegrityError
└── CacheMetadataError
```

**Base exception includes context dict:**
```python
class CacheError(Exception):
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.context = context or {}
```

**Decorator-based error handling:**
- `@with_error_handling(error_type=CacheStorageError)` — converts generic exceptions to typed cache errors
- Re-raises `CacheError` subclasses as-is; wraps other exceptions

**Context manager:**
- `cache_operation_context(operation, **context)` — standardized logging + timing for operations
- Used in `src/cacheness/core.py` for put/get operations

**Rollback pattern:**
- `_PutCleanup` class in `src/cacheness/core.py` — tracks resources (local blob, remote S3 object) during `put()`, rolls back on failure
- Commit/rollback semantics: `cleanup.commit()` disarms, `cleanup.rollback()` deletes orphans

**Error propagation principles:**
- `get()` is destructive on errors — auto-deletes entries that fail to load (except transient IO)
- CacheError subclasses propagate through; generic exceptions are wrapped
- Silent suppression only in cleanup/shutdown paths (`pass` in `except` blocks)

## Logging

**Framework:** Python `logging` module

**Pattern:** Module-level logger in every source file:
```python
logger = logging.getLogger(__name__)
```

**Log levels used:**
- `logger.debug()` — operation start/completion, timing, configuration details
- `logger.info()` — initialization success, backend selection, feature availability
- `logger.warning()` — fallback paths, failed cleanup, suppressed errors
- `logger.error()` — operation failures (via CacheError constructor)

**Emoji usage in logs:**
- ✅ for initialization success: `"✅ Unified cache initialized: ..."`
- 📊 for feature availability: `"📊 Both Polars and Pandas available"`
- 🗄️ for backend selection: `"🗄️  Using SQLite backend"`
- ⚠️ for warnings: `"⚠️  Neither Polars nor Pandas available"`
- ⚡ for performance info: `"⚡ Using in-memory SQLite backend"`

**Test logging config** (in `pyproject.toml`):
- `log_cli = true`, level `WARNING`
- Override with `--log-cli-level=INFO`

## Design Patterns

**Strategy Pattern:**
- Core architecture: `UnifiedCache` delegates format-specific operations to `CacheHandler` implementations
- `HandlerRegistry` selects the appropriate handler based on data type
- Handlers: `ArrayHandler`, `ObjectHandler`, `PolarsDataFrameHandler`, `PandasDataFrameHandler`, `BytesHandler`

**Registry Pattern:**
- `HandlerRegistry` in `src/cacheness/handlers.py` — handler selection by data type
- Metadata backend registry in `src/cacheness/storage/backends/__init__.py` — `register_metadata_backend()`, `get_metadata_backend()`
- Blob backend registry in `src/cacheness/storage/backends/blob_backends.py` — `register_blob_backend()`, `get_blob_backend()`
- Custom metadata model registry in `src/cacheness/custom_metadata.py` — `@custom_metadata_model()` decorator

**Factory Pattern:**
- `create_metadata_backend()` in `src/cacheness/metadata.py` — creates backend by type string
- `create_cache_config()` in `src/cacheness/config.py` — creates config from kwargs
- `get_blob_backend()` in `src/cacheness/storage/backends/blob_backends.py`

**Abstract Base Class (ABC):**
- `MetadataBackend` in `src/cacheness/metadata.py` — interface for JSON/SQLite/PostgreSQL backends
- `CacheHandler` in `src/cacheness/interfaces.py` — interface for type-specific handlers
- `BlobBackend` in `src/cacheness/storage/backends/blob_backends.py` — interface for blob storage

**Dataclass Configuration:**
- `CacheConfig` composed of sub-configs: `CacheStorageConfig`, `CacheMetadataConfig`, `CompressionConfig`, `SerializationConfig`, `HandlerConfig`, `SecurityConfig`, `HooksConfig`
- Each sub-config uses `@dataclass` with defaults and `__post_init__` validation

**Thread Safety:**
- `threading.RLock()` in `UnifiedCache` — re-entrant lock for put/get/delete atomicity
- Thread-safe metadata backends (SQLite WAL mode, JSON file locking)

**Cleanup/Resource Management:**
- `_PutCleanup` in `src/cacheness/core.py` — RAII-style cleanup for `put()` operations
- `atexit.register()` in `src/cacheness/decorators.py` — cleanup decorator-created caches
- `weakref.ref` tracking for decorator caches to avoid preventing garbage collection

## Type Annotations

**Approach:** Comprehensive type annotations throughout, targeting Python 3.12+

**Common patterns:**
- `Optional[X]` for nullable parameters/returns
- `Dict[str, Any]` for metadata dicts
- `Union[str, int]` for flexible inputs (e.g., size config)
- `Callable` for function parameters (decorators, hooks)
- `Tuple` for multi-return values
- `List[Dict[str, Any]]` for entry listings

**TypedDict usage** (in `src/cacheness/interfaces.py`):
- `SignableFields` — fields included in HMAC signatures
- `EntrySummary` — lightweight flat dict from `iter_entry_summaries()`
- `BlobReadContext` — metadata dict passed to handler `get()`
- `total=False` used for optional fields

**Type comments for import guards:**
```python
pl = None  # type: ignore
```

## Documentation / Docstrings

**Style:** Google-style docstrings with `Args:`, `Returns:`, `Raises:` sections

**Module-level docstrings:**
- Present in every source module
- Include feature lists, usage examples, and architectural notes
- Use reStructuredText-style header underlines (`===`, `---`)

**Class docstrings:**
- Describe purpose and usage pattern
- Note thread safety if applicable

**Method docstrings:**
- `Args:` with type and description for each parameter
- `Returns:` with type and description
- `Raises:` for expected exceptions

**Attribute docstrings:**
- `@dataclass` fields documented via class docstring or inline comments
- `NamespaceInfo` uses `Attributes:` section in class docstring

**Public API docstrings** in `src/cacheness/__init__.py`:
- Full module-level docstring with Quick Start example
- Key features listed

---

*Convention analysis: 2026-04-02*
