# Coding Conventions

**Analysis Date:** 2026-04-07

## Language & Runtime

**Primary:** Python 3.12+ (target-version in ruff config)
**Package Manager:** `uv` — never `pip` or `python` directly
**Build Backend:** `uv_build` via `pyproject.toml`

## Naming Patterns

**Files:**
- Source modules: `snake_case.py` — e.g. `blob_store.py`, `error_handling.py`, `write_intent.py`
- Private mixins: `_snake_case.py` — e.g. `_verification_mixin.py`, `_batch_mixin.py`, `_put_cleanup.py`
- Compatibility shims: `_compat.py` — one per package (`handlers/_compat.py`, `metadata/_compat.py`)
- Test files: `test_snake_case.py` — mirror source module names

**Classes:**
- PascalCase: `UnifiedCache`, `BlobStore`, `HandlerRegistry`, `WriteIntentJournal`
- Mixins: `*Mixin` suffix — `VerificationMixin`, `StorageModeMixin`, `BatchMixin`
- Dataclasses: PascalCase — `CacheConfig`, `CacheStorageConfig`, `RotationResult`, `HandlerResult`
- TypedDicts: PascalCase — `SignableFields`, `EntrySummary`, `BlobReadContext`
- Exceptions: `Cache*Error` — `CacheSecurityError`, `CacheBackendError`, `CacheConfigurationError`
- Backends: `*Backend` suffix — `JsonBackend`, `SqliteBackend`, `PostgresBackend`
- Handlers: `*Handler` suffix — `ArrayHandler`, `PandasDataFrameHandler`, `ObjectHandler`

**Functions:**
- snake_case: `create_metadata_backend()`, `validate_namespace_id()`, `derive_encryption_key()`
- Private helpers: `_snake_case` — `_normalize_function_args()`, `_intent_filename()`, `_make_cache()`
- Factory functions: `create_*` — `create_metadata_backend()`, `create_cache_config()`, `create_entry_cache()`

**Variables / Constants:**
- Module-level flags: `SCREAMING_SNAKE` — `BLOSC2_AVAILABLE`, `PANDAS_AVAILABLE`, `CRYPTOGRAPHY_AVAILABLE`
- Module-level defaults: `SCREAMING_SNAKE` — `DEFAULT_NAMESPACE`, `NAMESPACE_ID_PATTERN`
- Sentinel values: `_SCREAMING_SNAKE` — `_DEFAULT_TTL = object()`
- Logger: `logger = logging.getLogger(__name__)` — one per module, always at module top

## Code Style

**Formatting:**
- Ruff formatter — config in `pyproject.toml`
- Line length: 88 characters
- Target version: Python 3.12

**Linting:**
- Ruff linter with default rule set
- Ignored rules: `B008` (function calls in defaults), `C901` (complexity)
- Per-file ignores: `tests/**` → `F401`, `F841`, `E721`; `benchmarks/**` → `F841`; `examples/**` → `F401`, `E402`
- Type checking: `ty` (`uv run ty check`) — excludes `tests/`, `benchmarks/`, `examples/`

**Two-phase quality gates:**
```bash
# Phase 1 — auto-fix (never fails)
uv run ruff format $files; uv run ruff check --fix $files
# Phase 2 — validate (may fail)
uv run ruff check $files; uv run ty check $files
```

## Import Organization

**Order (enforced by ruff/isort):**
1. Standard library (`import os`, `from pathlib import Path`)
2. Third-party (`import numpy as np`, `import pytest`)
3. Local/project (`from .config import CacheConfig`, `from ._compat import ...`)

**Key patterns:**

**`_compat.py` as shared import hub:**
Each sub-package (`handlers/`, `metadata/`) has a `_compat.py` that centralizes imports from the parent package and optional dependency detection. All sub-modules import shared symbols from `_compat` rather than reaching up to parent packages directly.

```python
# src/cacheness/handlers/_compat.py
from ..interfaces import (  # noqa: F401
    CacheHandler, HandlerResult, BlobReadContext,
    CacheWriteError, CacheReadError,
)
from ..error_handling import cache_operation_context  # noqa: F401
from ..compress_pickle import (
    BLOSC_AVAILABLE,  # noqa: F401
    write_file as write_compressed_pickle,  # noqa: F401
    read_file as read_compressed_pickle,  # noqa: F401
)
```

**CRITICAL:** Re-exports in `_compat.py` MUST use `# noqa: F401` to prevent ruff from removing them as unused imports. Missing these annotations breaks all sub-module imports.

**Optional dependency guards:**
```python
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    pl = None  # type: ignore
    POLARS_AVAILABLE = False
```

**Lazy imports for heavy dependencies:**
```python
# TensorFlow — lazy loaded to avoid startup cost
TENSORFLOW_AVAILABLE = False
tf = None
_tensorflow_import_attempted = False

def _lazy_import_tensorflow():
    global tf, TENSORFLOW_AVAILABLE, _tensorflow_import_attempted
    if _tensorflow_import_attempted:
        return tf, TENSORFLOW_AVAILABLE
    _tensorflow_import_attempted = True
    try:
        import tensorflow as tf_module
        tf = tf_module
        TENSORFLOW_AVAILABLE = True
    except ImportError:
        tf = None
```

**`__init__.py` for backward compatibility:**
Package `__init__.py` files re-export all public names so external code using `from cacheness.handlers import ArrayHandler` continues to work after monolith-to-package splits.

**`TYPE_CHECKING` for circular imports:**
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .interfaces import RotationResult
```

## Error Handling

**Exception hierarchy:**
All custom exceptions inherit from `CacheError` (base), defined in `src/cacheness/error_handling.py`:

```
CacheError
├── CacheConfigurationError    — invalid configuration
├── CacheStorageError          — storage I/O failures
├── CacheSerializationError    — key serialization failures
├── CacheHandlerError          — handler put/get failures
├── CacheIntegrityError        — hash/signature verification failures
├── CacheMetadataError         — metadata backend failures
├── CacheSecurityError         — signing, verification, key management
└── CacheBackendError          — storage I/O, blob operations
```

`CacheError` accepts optional `context: Dict[str, Any]` and auto-logs on creation.

**`# intentionally broad` annotation:**
Every deliberate `except Exception` in the codebase is annotated with a comment explaining WHY it's broad. No unannotated bare `except Exception` remains anywhere in `src/cacheness/`. The annotation follows the pattern:

```python
except Exception as e:  # intentionally broad — <reason>
```

Common reasons:
- `# intentionally broad — atexit cleanup` (cleanup must not raise)
- `# intentionally broad — pickle may raise anything` (deserialization)
- `# intentionally broad — cache retrieval may fail any way` (decorator resilience)
- `# intentionally broad — best-effort re-sign` (key rotation fallback)
- `# intentionally broad — hooks must not crash the caller` (lifecycle callbacks)
- `# intentionally broad — re-raises after cleanup` (error conversion)

**Error handling decorators:**
```python
# src/cacheness/error_handling.py
@with_error_handling(error_type=CacheStorageError, context={...})
def some_storage_op():
    ...
```

**Context manager for operations:**
```python
with cache_operation_context("store_pandas_dataframe", shape=data.shape):
    ...  # auto-logs start/end/duration, raises CacheError on failure
```

**Config validation with fix suggestions:**
`CacheConfigurationError` in `src/cacheness/config.py` provides actionable messages:
```python
raise CacheConfigurationError(
    "Encryption is enabled with entry signing disabled. "
    "Encrypted data without integrity signing is unsafe. "
    "Set enable_entry_signing=True to ensure encrypted "
    "entries are signed."
)
```

## Architecture Patterns

### Mixin Decomposition

`UnifiedCache` in `src/cacheness/core.py` uses 11 mixins to keep the main class focused on coordination. Each mixin is a private `_*.py` module:

```python
class UnifiedCache(
    VerificationMixin,    # _verification_mixin.py — signing, integrity
    StatsMixin,           # _stats_mixin.py — hit/miss stats
    CustomMetadataMixin,  # _custom_metadata_mixin.py — linked tables
    StorageModeMixin,     # _storage_mode_mixin.py — no-eviction mode
    QueryMixin,           # _query_mixin.py — list/search operations
    ConvenienceMixin,     # _convenience_mixin.py — get_or_set, etc.
    BatchMixin,           # _batch_mixin.py — delete_by_prefix, put_batch
    FileOpsMixin,         # _file_ops_mixin.py — put_file/get_file
    GetVariantsMixin,     # _get_variants_mixin.py — get+return metadata
    UpdateMixin,          # _update_mixin.py — update_data, update_description
    InlineBlobMixin,      # _inline_blob_mixin.py — small blob inlining
):
```

Mixins access `self.config`, `self._lock`, `self.metadata_backend`, `self._blob_store` from UnifiedCache.

### Handler Priority System

Handlers implement `CacheHandler` ABC from `src/cacheness/interfaces.py`. Each handler declares a `priority: int` class attribute. Lower number = higher priority (checked first).

```python
# Priority order (lower = checked first):
PolarsSeriesHandler      # priority: 10
PandasSeriesHandler      # priority: 20
PolarsDataFrameHandler   # priority: 30
PandasDataFrameHandler   # priority: 40
ArrayHandler             # priority: 50  (NumPy arrays)
BytesHandler             # priority: 60  (bytes/bytearray/memoryview)
ObjectHandler            # priority: 100 (pickle fallback — always last)
```

`HandlerRegistry` (`src/cacheness/handlers/registry.py`) iterates handlers in priority order, calling `can_handle(data)`. First match wins. Custom handlers can be registered with explicit priority.

### Configuration Dataclasses

`CacheConfig` in `src/cacheness/config.py` is the root config, composing 8 focused sub-configs:

```python
CacheConfig
├── storage: CacheStorageConfig     — cache_dir, max_cache_size, atomic writes
├── metadata: CacheMetadataConfig   — backend type, TTL, memory cache
├── blob: CacheBlobConfig           — blob backend, sharding, inline size
├── compression: CompressionConfig  — parquet, pickle, blosc2 codecs
├── serialization: SerializationConfig — key hashing, depth limits
├── handlers: HandlerConfig         — enable/disable, priority order
├── security: SecurityConfig        — signing, encryption, key fallback
└── hooks: HooksConfig              — on_evict, on_integrity_failure callbacks
```

Each dataclass has `__post_init__` validation. `CacheConfig.__init__` also accepts flat kwargs for backward compatibility (e.g., `cache_dir=...` maps to `storage.cache_dir`).

### Crash Safety: WriteIntentJournal

`WriteIntentJournal` in `src/cacheness/write_intent.py` implements crash-safe blob writes:

1. `record_intent(cache_key, blob_path)` — writes `.intent` JSON file
2. Handler writes blob to disk
3. Metadata backend commits entry
4. `clear_intent(cache_key)` — removes `.intent` file

On cache init, `cleanup_stale_intents()` deletes orphaned blobs from crashed writes.

### TypedDict Contracts

Interface contracts for cross-module dict passing use `TypedDict` with `total=False`:
- `SignableFields` — fields included in HMAC signatures (`src/cacheness/interfaces.py`)
- `EntrySummary` — lightweight entry dict from `iter_entry_summaries()` (`src/cacheness/interfaces.py`)
- `BlobReadContext` — metadata dict passed to `handler.get()` (`src/cacheness/interfaces.py`)

### Result Dataclasses

Operations return well-defined dataclass results:
- `HandlerResult` — handler put() output (storage_format, file_size, actual_path, extra) (`src/cacheness/interfaces.py`)
- `RotationResult` — key rotation stats (total, re_signed, re_encrypted, failed, skipped) (`src/cacheness/interfaces.py`)
- `IntegrityReport` — integrity audit results (`src/cacheness/interfaces.py`)
- `EntryData` — full entry data + metadata (`src/cacheness/interfaces.py`)

### Decorator Pattern

`src/cacheness/decorators.py` provides `@cached` and `@cache_if` decorators. Key patterns:
- `weakref` tracking of cache instances for atexit cleanup
- `atexit.register` for cleanup on interpreter exit
- `functools.wraps` for metadata preservation
- Thread-safe via `threading.Lock`

### Thread Safety

`UnifiedCache` uses `threading.RLock` (`self._lock`) for all state-mutating operations. The RLock allows re-entrant calls (e.g., `delete_where` → `invalidate` → lock again).

### Namespace Isolation

Namespaces provide multi-tenant cache isolation:
- Validated by `validate_namespace_id()` — lowercase alphanumeric + underscore, 1-48 chars
- Per-namespace tables (SQLite/PG) or files (JSON)
- Per-namespace HKDF-derived signing/encryption keys

### Encryption at Rest

AES-256-GCM in `src/cacheness/encryption.py`:
- Encrypt-then-sign: blob encrypted, then HMAC signs ciphertext
- HKDF-SHA256 key derivation per namespace
- Two write paths: `BlobStore.put()` and `BlobStore._write_blob()` (both handle encryption)
- Metadata stores `encryption_algorithm` and `encryption_iv` fields

## Logging

**Framework:** Python `logging` module
**Pattern:** One logger per module at module level:
```python
logger = logging.getLogger(__name__)
```

**Levels used:**
- `logger.debug(...)` — config details, operation timing, flow tracing
- `logger.info(...)` — cache operations (put/get/delete), cleanup summaries
- `logger.warning(...)` — fallback behavior, deprecated features, suppressed errors
- `logger.error(...)` — operation failures (via `CacheError.__init__` auto-logging)

**f-string formatting** used throughout (not `%` or `.format()`).

## Comments & Documentation

**Module docstrings:** Triple-quoted at top of every module, describing purpose and key abstractions.

**Class docstrings:** Present on all public classes with usage examples in `HandlerRegistry`, `WriteIntentJournal`.

**Inline comments:** Explain *why*, not *what*. Key patterns:
- `# intentionally broad — <reason>` on `except Exception`
- `# noqa: F401` on re-exports in `_compat.py` and `__init__.py`
- `# noqa: E402` on imports after `pytest.importorskip()`

## Module Design

**Exports:** `__init__.py` re-exports public API via explicit imports. `__all__` used in `handlers/__init__.py`.

**Package splits:** Monoliths split via the pattern:
1. `_compat.py` — shared imports, optional deps, ORM models
2. `base.py` — ABCs and base classes
3. `*_backend.py` / `*_handler.py` — concrete implementations
4. `__init__.py` — re-exports + factory functions

**Sentinel values:** Use `object()` sentinels to distinguish "not provided" from `None`:
```python
_DEFAULT_TTL = object()
```

---

*Convention analysis: 2026-04-07*
