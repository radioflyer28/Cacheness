# GitHub Copilot Instructions for Cacheness

## Project Overview

**Cacheness** is a Python disk caching library with pluggable metadata backends and handler-based serialization. It provides persistent caching for expensive computations with support for multiple storage backends (JSON, SQLite, PostgreSQL) and type-aware serialization handlers (DataFrames, NumPy arrays, TensorFlow tensors, etc.).

## Package Manager: uv

**ALWAYS use `uv` for Python operations** — it's already available in PATH.

```bash
# Run tests
uv run pytest tests/ -x -q

# Run specific test file
uv run pytest tests/test_core.py -v

# Run benchmarks
uv run python benchmarks/management_ops_benchmark.py

# Install dependencies
uv sync

# Add new dependency
uv add package-name

# Run Python scripts
uv run python script.py
```

**Do NOT use:** `python`, `pip`, `python -m pytest` — use `uv run` instead.


## Issue Tracking

This project uses **beads-mcp** for issue tracking. Use the "beads" MCP server for all issue-related operations.


## Test Suite

- **Location:** `tests/` directory
- **Current baseline:** 696 passed, 70 skipped, 0 failures
- **Run command:** `uv run pytest tests/ -x -q`
- **Conventions:**
  - Use `-x` flag to stop on first failure
  - Use `-q` for quiet output, `-v` for verbose
  - Fixtures defined in `tests/conftest.py`

### Key Test Files
- `test_core.py` — Main cache operations
- `test_update_operations.py` — Management operations (update, touch, delete_where, batch ops)
- `test_handlers.py` — Serialization handlers
- `test_metadata.py` — Metadata backend tests
- `test_security.py` — HMAC signing and verification

## Architecture

### Storage Layer Separation (Phase 1 — Complete)

```
UnifiedCache (core.py)
    ↓
├── Handlers (handlers.py) — Type-specific serialization
│   ├── DataFrameHandler
│   ├── NumpyHandler
│   ├── TensorFlowHandler
│   └── PickleHandler (fallback)
│
├── Metadata Backends (metadata.py)
│   ├── JsonBackend
│   ├── SQLiteBackend
│   ├── PostgresBackend
│   └── MemoryCacheWrapper
│
└── Blob Storage (compress_pickle.py)
    └── Atomic file writes with .tmp + rename
```

### Key Components

1. **`src/cacheness/core.py`** — Main `UnifiedCache` class (1,689 lines)
   - Single coordination layer
   - Delegates to handlers and backends
   - **IMPORTANT:** Has a `_lock` field that is created but **never acquired** — not thread-safe

2. **`src/cacheness/handlers.py`** — Type-aware serialization registry
   - Handlers registered via `@register_handler` decorator
   - Priority system for handler selection
   - `can_handle(obj)` → `put(obj, path)` → `get(path)` pattern

3. **`src/cacheness/metadata.py`** — Metadata backend implementations
   - `MetadataBackend` abstract base class
   - Each backend implements: `put_entry()`, `get_entry()`, `list_entries()`, `delete_entry()`
   - **NEW:** `iter_entry_summaries()` for lightweight iteration (added for batch operations optimization)

4. **`src/cacheness/compress_pickle.py`** — Blob compression/decompression
   - **CRITICAL:** Always reads entire file with `f.read()` (fixed from 2GB buffer issue)
   - Supports blosc2, lz4, zstd, gzip

## Coding Conventions

### 1. Always Check Backend Capabilities
```python
# Backend-specific optimizations exist
if hasattr(self.metadata_backend, 'query_meta'):
    # SQLite fast path with JSON column search
    matching_keys = self.metadata_backend.query_meta(**kwargs)
else:
    # Generic path — iterate all entries
    for entry in self.metadata_backend.list_entries():
        # ... filter manually
```

### 2. Use `iter_entry_summaries()` for Batch Operations
```python
# ❌ BAD — Loads full ORM objects
for entry in self.metadata_backend.list_entries():
    if some_filter(entry):
        keys.append(entry['cache_key'])

# ✅ GOOD — Lightweight flat dicts, no ORM hydration
for entry in self.metadata_backend.iter_entry_summaries():
    if some_filter(entry):
        keys.append(entry['cache_key'])
```

### 3. Error Handling in `get()`
`get()` is **destructive** on errors — it auto-deletes entries that fail to load:
```python
try:
    obj = handler.get(actual_path)
except Exception:
    # WARNING: This deletes the metadata entry permanently
    self.metadata_backend.delete_entry(cache_key)
    return None
```

### 4. Named Parameters in `_create_cache_key()`
**Critical bug pattern:** Named parameters (`prefix`, `description`, `custom_metadata`, `ttl_seconds`) must be stripped before hashing:
```python
# ✅ CORRECT
kwargs_for_key = {k: v for k, v in kwargs.items() 
                  if k not in ('prefix', 'description', 'custom_metadata')}
cache_key = self._create_cache_key(**kwargs_for_key)
```

### 5. Backend Thread Safety
- **JSON:** Protected by `threading.Lock()`
- **SQLite:** `check_same_thread=False` + WAL mode + per-operation locks
- **PostgreSQL:** `ThreadPoolExecutor` around all operations
- **UnifiedCache itself:** NOT thread-safe (lock never acquired)

## Common Patterns

### Adding a New Handler
```python
@register_handler(priority=100)
class MyTypeHandler(Handler):
    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, MyType)
    
    def put(self, obj: MyType, file_path: str) -> Dict[str, Any]:
        # Serialize and return metadata
        return {"storage_format": "myformat"}
    
    def get(self, file_path: str) -> MyType:
        # Deserialize and return object
        return MyType(...)
```

### Adding a Management Operation
```python
def my_operation(self, **filter_kwargs) -> int:
    """New management operation."""
    count = 0
    # Use iter_entry_summaries for performance
    for entry in self.metadata_backend.iter_entry_summaries():
        if self._matches_filters(entry, filter_kwargs):
            # Perform operation
            count += 1
    return count
```

## Known Issues & Gotchas

1. **Race condition in `put()`:** Two threads calling `put()` with the same key can corrupt state (lock never used)
2. **Auto-deletion on read errors:** Transient I/O errors cause permanent metadata loss
3. **Orphaned blobs:** If process crashes between blob write and metadata write, disk space consumed but invisible
4. **SQLite "database is locked":** 30-second timeout can be exceeded under heavy concurrent writes
5. **JSON backend scales O(n²):** Each write re-serializes entire JSON file

## Benchmarks

Run benchmarks to validate performance:
```bash
uv run python benchmarks/management_ops_benchmark.py
uv run python benchmarks/handler_benchmark.py
uv run python benchmarks/serialization_benchmark.py
```

Expected baseline (200 entries):
- `touch_batch`: ~0.79ms (SQLite)
- `delete_where`: ~44ms (SQLite)
- `list_entries`: ~20ms (SQLite), ~0.40ms (JSON)

## Documentation

- **API Reference:** `docs/API_REFERENCE.md`
- **Performance:** `docs/PERFORMANCE.md`
- **Backend Selection:** `docs/BACKEND_SELECTION.md`
- **Development Planning:** `docs/DEVELOPMENT_PLANNING.md`

## When Modifying Core Logic

1. **Always run tests after changes:** `uv run pytest tests/ -x -q`
2. **Check for regressions:** Baseline is 696 passed, 70 skipped
3. **Run relevant benchmark:** Ensure no performance degradation
4. **Update documentation:** If changing public API

## Dependencies

All dependencies managed in `pyproject.toml`:
- **Core:** `sqlalchemy`, `blosc2`, `pandas`, `numpy`
- **Optional:** `tensorflow`, `torch`, `psycopg2-binary`
- **Dev:** `pytest`, `pytest-cov`, `hypothesis`

Install with: `uv sync`

---

**Last Updated:** February 2026  
**Test Baseline:** 696 passed, 70 skipped, 0 failures
