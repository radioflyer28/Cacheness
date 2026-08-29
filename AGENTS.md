<!-- GSD:project-start source:PROJECT.md -->

## Project

**Cacheness**

Cacheness is a Python storage and caching library for arbitrary objects, arrays, dataframes, and function results. It currently exposes overlapping cache, blob-storage, backend-registry, and SQL pull-through systems; this project will converge the object-storage path around a reliable `BlobStore` foundation that `UnifiedCache` uses as its policy layer.

The intended audience is Python applications that need local or remote persistence with predictable cache semantics across filesystem, memory, S3, JSON, SQLite, and PostgreSQL backends.

**Core Value:** Applications can store and retrieve data reliably through one backend-neutral lifecycle, with caching policy layered above storage without compromising integrity or cleanup correctness.

### Constraints

- **Compatibility**: Preserve supported public APIs; allow stored-data migration or rebuild only through an explicit, documented path
- **Architecture**: `BlobStore` owns storage lifecycle; `UnifiedCache` depends on it and owns cache policy; `SqlCache` remains separate
- **Backends**: The unified lifecycle must cover filesystem, memory, S3, JSON, SQLite, and PostgreSQL implementations
- **Security**: Treat application payloads as trusted while enforcing safe parsing, path containment, and fail-closed integrity boundaries
- **Reliability**: Payload and metadata operations must have atomic commit, rollback, or deterministic reconciliation semantics
- **Concurrency**: Same-key operations must not corrupt payloads or produce metadata/payload disagreement
- **Performance**: Correctness comes first during migration; final acceptance includes measured budgets against checked-in benchmarks
- **Runtime**: Maintain Python 3.11+ support and verify supported versions rather than relying only on the current Python 3.13 environment

<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->

## Technology Stack

## Languages

- Python 3.11+ (the project requires `>=3.11`; the repository pins Python 3.13 in `.python-version`) - library implementation, tests, examples, and benchmarks in `src/cacheness/`, `tests/`, `examples/`, and `benchmarks/`
- SQL (generated and dialect-specific SQLAlchemy expressions, plus SQLite pragmas) - relational metadata and pull-through cache operations in `src/cacheness/metadata.py`, `src/cacheness/sql_cache.py`, and `src/cacheness/storage/backends/postgresql_backend.py`

## Runtime

- CPython 3.13 for the checked-in development environment; supported runtime range is Python `>=3.11` (`.python-version`, `pyproject.toml`)
- `uv` - dependency resolution, environment management, and lockfile workflow (`uv.lock`)
- Lockfile: present (`uv.lock`, lock format revision 2)

## Frameworks

- No application framework - `cacheness` is a framework-independent Python caching library (`src/cacheness/__init__.py`)
- SQLAlchemy 2.0 - optional ORM/engine layer for SQLite, DuckDB, PostgreSQL, and custom metadata (`src/cacheness/metadata.py`, `src/cacheness/sql_cache.py`)
- pytest 8.4.1 (locked) - test discovery and execution (`pyproject.toml`, `tests/`)
- pytest-cov 6.2.1 (locked) - coverage reporting (`pyproject.toml`)
- moto 5.1.20 (locked in the development environment) - AWS S3 mocking for `tests/test_s3_blob_backend.py`
- `uv_build` (`>=0.9.0,<1.0.0`) - PEP 517 build backend (`pyproject.toml`)
- Ruff `0.12.9` (locked) - lint/format tooling; target Python version is `py312`, line length is 88 (`pyproject.toml`)
- `verify_platform.py` - repository-level compatibility smoke checks

## Key Dependencies

- `xxhash>=3.5.0` (locked 3.5.0) - XXH3 cache-key and file-content hashing in `src/cacheness/serialization.py`, `src/cacheness/file_hashing.py`, and `src/cacheness/core.py`
- `cachetools>=6.1.0` (locked 6.1.0) - optional TTL/LRU-style in-memory metadata layer in `src/cacheness/metadata.py`; the code falls back when it is unavailable
- `numpy>=2.0.0` (locked 2.3.2) - array handling and serialization; imported eagerly by `src/cacheness/compress_pickle.py` and `src/cacheness/handlers.py`
- `blosc2>=3.5.1` (locked 3.7.0) - high-performance array and pickle compression; guarded with a fallback in `src/cacheness/compress_pickle.py` and `src/cacheness/handlers.py`
- `pandas>=2.0.0,<4.0.0` (locked 2.3.1) - optional DataFrame/Series handlers and SQL pull-through results (`src/cacheness/handlers.py`, `src/cacheness/sql_cache.py`)
- `polars>=1.30.0` (locked 1.32.3) - optional DataFrame/Series Parquet handlers (`src/cacheness/handlers.py`)
- `pyarrow>=21.0.0` (locked 21.0.0) - Parquet engine used by pandas-oriented storage (`src/cacheness/handlers.py`)
- `orjson>=3.8.0` (locked 3.11.2) - optional fast JSON metadata serialization, falling back to stdlib `json` (`src/cacheness/json_utils.py`, `src/cacheness/metadata.py`)
- `PyYAML` (locked 6.0.3 in the development resolution, not a direct project dependency) - optional YAML configuration load/save, imported lazily (`src/cacheness/config.py`)
- `dill>=0.4.0` (locked 0.4.0) - optional fallback for objects standard pickle cannot serialize (`src/cacheness/compress_pickle.py`, `src/cacheness/handlers.py`)
- `sqlalchemy>=2.0.0` (locked 2.0.43) - optional metadata ORM and SQL cache engine
- `duckdb-engine>=0.16.0` (locked 0.17.0) - SQLAlchemy DuckDB dialect for analytical pull-through caches (`src/cacheness/sql_cache.py`)
- `psycopg[binary]>=3.1.0` (locked `psycopg` 3.3.2) - PostgreSQL driver for the PostgreSQL metadata backend and PostgreSQL SQL cache
- `boto3>=1.26.0` (locked 1.42.36) - optional S3-compatible blob backend (`src/cacheness/storage/backends/s3_backend.py`)
- `tensorflow>=2.0.0` (locked 2.20.0) - optional, lazily loaded TensorFlow tensor handler; disabled by default (`src/cacheness/handlers.py`, `src/cacheness/config.py`)

## Installability Reality Check

- `pyproject.toml` declares only `cachetools` and `xxhash` as mandatory dependencies, but `src/cacheness/handlers.py` and `src/cacheness/compress_pickle.py` import NumPy unconditionally. Because `src/cacheness/__init__.py` imports `core.py` and `handlers.py`, a nominal base installation without the `recommended` or `dataframes` extras cannot import `cacheness` unless NumPy happens to be supplied transitively.
- Treat NumPy as an undeclared runtime dependency in the current release. Either move it into `[project.dependencies]` or make the array/compression modules genuinely optional behind lazy imports.
- PyYAML is present in the current `uv.lock`, but is not a direct project or development dependency. The `try` block in `src/cacheness/__init__.py` imports YAML helper functions, not `yaml` itself, so `_has_yaml_config` can be true even when calling those helpers later raises `ImportError`.
- SQLAlchemy, pandas, S3, PostgreSQL, and TensorFlow availability is mostly determined by guarded imports in their implementation modules. Keep packaging extras and those runtime checks synchronized; the current manifest/runtime split is part of the compatibility surface.
- The checked environment resolves CPython 3.13, NumPy 2.3.2, SQLAlchemy 2.0.43, pytest 8.4.1, and Ruff 0.12.9 from `uv.lock`.
- `uv run pytest -q -o log_cli=false` collects 777 tests: 749 pass, 26 skip, and 2 fail.
- `uv run ruff check . --output-format concise` reports 137 findings repository-wide; no CI configuration currently enforces either gate.

## Configuration

- Runtime configuration is passed through `CacheConfig` and its sub-configurations in `src/cacheness/config.py`; no environment-variable parser is implemented in `src/cacheness/`
- JSON configuration can be loaded/saved with `load_config_from_json()` and `save_config_to_json()`; YAML support is lazy and requires separately installed PyYAML (`src/cacheness/config.py`)
- Main defaults are local `./cache` blobs, auto-selected SQLite metadata when SQLAlchemy works (JSON fallback otherwise), 24-hour TTL, LZ4 Parquet, Zstandard pickle compression, and HMAC entry signing (`src/cacheness/config.py`, `src/cacheness/core.py`)
- `.env`/credential files are not detected; generated cache databases and serialized artifacts are excluded by `.gitignore`
- `pyproject.toml` - package metadata, dependency extras, pytest settings, coverage settings, Ruff settings, and `uv_build` backend
- `.python-version` - development interpreter pin (`3.13`)
- `uv.lock` - resolved dependency graph and platform-specific wheels

## Platform Requirements

- Python 3.11 or newer, `uv`, and a writable local filesystem; optional features require their corresponding extras (`pyproject.toml`)
- SQL-backed features require SQLAlchemy; DuckDB requires `duckdb-engine`; PostgreSQL requires a reachable PostgreSQL server and `psycopg`; S3 requires AWS-compatible credentials/configuration and `boto3`
- Deploy as a Python package in the host process; the library has no web server, worker runtime, container definition, or hosting manifest
- Use a writable filesystem for default blobs and SQLite metadata, or provide PostgreSQL metadata/S3 blobs through the pluggable backend APIs (`src/cacheness/storage/backends/`)

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

## Naming Patterns

- Use lowercase `snake_case.py` for library modules, for example `src/cacheness/error_handling.py` and `src/cacheness/file_hashing.py`.
- Name tests `test_<subject>.py` under `tests/`, such as `tests/test_config_validation.py` and `tests/test_s3_blob_backend.py`.
- Keep package API re-exports in `__init__.py` files, especially `src/cacheness/__init__.py` and `src/cacheness/storage/backends/__init__.py`.
- Use lowercase `snake_case` for public and private functions and methods, with a leading underscore for implementation helpers such as `_normalize_function_args` in `src/cacheness/core.py` and `_hash_single_file` in `src/cacheness/file_hashing.py`.
- Use verbs for operations (`create_metadata_backend`, `validate_config`, `register_handler`) and `is_`/`has_`/`can_` predicates (`is_custom_metadata_available`, `can_handle`).
- Preserve `__dunder__` names for protocol methods and use `@property` for read-only identifiers such as handler `data_type`.
- Use descriptive lowercase `snake_case` names (`cache_dir`, `metadata_backend`, `error_context`).
- Use uppercase names for module constants and capability flags (`SQLALCHEMY_AVAILABLE`, `PSYCOPG_AVAILABLE`, `BLOSC2_AVAILABLE`, `DEFAULT_TTL`-style sentinels).
- Prefix intentionally private module state with `_`, for example `_default_registry` in `src/cacheness/__init__.py` and `_metadata_backend_registry` in `src/cacheness/storage/backends/__init__.py`.
- Use PascalCase for classes and exception types (`UnifiedCache`, `CacheConfig`, `CacheStorageError`, `PostgresBackend`).
- Model grouped configuration as `@dataclass` classes in `src/cacheness/config.py`.
- Use abstract base classes and focused interfaces in `src/cacheness/interfaces.py` and `src/cacheness/storage/backends/base.py`; concrete implementations inherit the relevant interface.
- Type hints mix Python 3.11+ built-in generics (`list`, `dict`) with `typing.Optional`, `List`, `Dict`, `Tuple`, and `Union`. Match the surrounding module when extending it and type public boundaries where practical.

## Code Style

- Use four-space indentation and conventional PEP 8 spacing. The configured target line length is 88 in `pyproject.toml` (`[tool.ruff]`), although existing source and tests contain longer lines and trailing whitespace.
- Start modules with a descriptive module docstring; multi-line public APIs generally use Google-style `Args`, `Returns`, and `Raises` sections. Examples include `src/cacheness/core.py`, `src/cacheness/error_handling.py`, and `src/cacheness/interfaces.py`.
- Use section banner comments (`# =============================================================================`) in larger modules and test files to separate registries, fixtures, and behavior groups; follow the pattern in `src/cacheness/handlers.py` and `tests/test_blob_backend_registry.py`.
- Use f-strings for structured messages, paths, and log records. Preserve comments explaining optional dependency behavior, compatibility aliases, and platform-specific workarounds.
- Run `uv run ruff check src tests`; Ruff is declared at `>=0.12.8` in the `dev` dependency group.
- `pyproject.toml` sets Ruff `target-version = "py312"` and ignores `B008` and `C901`. The intended lint groups are documented in comments (`E`, `W`, `F`, `I`, `B`, `C4`, `UP`), but the `lint.select` setting is commented out, so do not assume import sorting or all optional rule groups are enforced.
- The current source/test tree produces 123 findings under the active Ruff defaults; the complete repository produces 137 findings because examples, benchmarks, and `verify_platform.py` add another 14. Findings include unused imports/locals, redefinitions, late imports, bare `except`, and lambda assignment. New code should avoid adding to this baseline and should not use `# noqa` without a local reason.
- Ruff is configured but no CI workflow runs it, and the active default rules already fail. Treat the style guidance as a target, not as a verified invariant of existing files.
- `lint.select` is commented out in `pyproject.toml`; only Ruff's default rule set plus the two ignores is active. The comments listing `I`, `B`, `C4`, and `UP` do not enable those groups.
- Formatting is not configured separately (`ruff format`, Black, or Prettier equivalent); line length 88 informs Ruff rules but does not prove the tree is formatter-clean.

## Import Organization

- No configured import path aliases were detected. Use package-relative imports inside `src/cacheness` and `cacheness.<module>` imports in tests, as shown in `tests/test_core.py` and `tests/test_error_handling.py`.
- Import optional dependencies lazily or behind `try/except ImportError` when the feature is optional. Examples include `src/cacheness/__init__.py`, `src/cacheness/handlers.py`, and `src/cacheness/storage/backends/s3_backend.py`.

## Error Handling

- Raise the domain-specific hierarchy from `src/cacheness/error_handling.py` (`CacheError` and its configuration, storage, serialization, handler, integrity, and metadata subclasses) for cross-cutting cache failures.
- Handler-specific failures use `CacheHandlerError` and its `CacheWriteError`, `CacheReadError`, `CacheFormatError`, and `CacheValidationError` subclasses in `src/cacheness/interfaces.py`.
- Preserve the original cause with `raise ... from e` when translating `OSError`, import, serialization, or backend failures. `with_error_handling` in `src/cacheness/error_handling.py` adds function/argument context and either reraises or returns a configured fallback.
- Use `pytest.raises` with a specific exception and, where stable, `match=` in tests; see `tests/test_directory_sharding.py`, `tests/test_handler_registration.py`, and `tests/test_error_handling.py`.
- Handle optional features explicitly: capability detection is represented by flags such as `SQLALCHEMY_AVAILABLE`, and unavailable optional paths should be skipped or produce a clear install-oriented error.
- Do not copy the current broad-exception pattern into new boundaries. `src/cacheness/core.py`, `src/cacheness/handlers.py`, `src/cacheness/metadata.py`, and `src/cacheness/sql_cache.py` frequently catch `Exception`; several paths convert failures into misses, empty results, warnings, or partial data. New code should catch the narrow operational exception, preserve its cause, and make partial-success policy explicit.

## Logging

- Use `debug` for configuration and operation details, `info` for backend selection/lifecycle and successful cache operations, `warning` for fallbacks or suppressed failures, and `error` for domain failures.
- Include operation context in messages or with `extra=...`; `src/cacheness/error_handling.py` demonstrates both structured context and duration logging.
- Tests that assert logs should use `caplog.at_level(...)` and inspect `caplog.text`, as in `tests/test_error_handling.py` and `tests/test_interfaces.py`.
- Preserve the existing user-facing log style, including backend/lifecycle messages and the existing emoji status prefixes in `src/cacheness/core.py`, when modifying adjacent operations.

## Comments

- Add module/class/function docstrings for public APIs and explain non-obvious serialization, backend, concurrency, or compatibility decisions.
- Use short inline comments for algorithm steps and resource cleanup; larger phase/feature sections in `src/cacheness/config.py` and `tests/test_config_validation.py` use banner comments.
- Document why imports are lazy, why a fallback is selected, or why a test is skipped. Avoid comments that merely restate a simple line.
- Not applicable. This is a Python project; docstrings are the API documentation mechanism.
- Google-style docstrings are common for library interfaces, with examples in `src/cacheness/interfaces.py` and `src/cacheness/error_handling.py`. Tests generally use one-line docstrings on classes and test methods.

## Function Design

- Keep new functions focused around one cache, serialization, backend, or validation responsibility. Existing large modules (`src/cacheness/sql_cache.py`, `src/cacheness/metadata.py`, `src/cacheness/handlers.py`, and `src/cacheness/core.py`) contain long orchestration methods, so extract helpers rather than growing those methods further.
- Type public parameters when stable; use `Optional[...]` for optional configuration and `**kwargs` for backend-specific options or cache-key parameters, matching `src/cacheness/core.py` and `src/cacheness/storage/backends/blob_backends.py`.
- Pass grouped behavior through configuration objects (`CacheConfig` and its sub-configurations in `src/cacheness/config.py`) instead of adding unrelated flags to every handler method.
- Return concrete values that callers can inspect: handlers return metadata dictionaries, backends return storage paths/bytes/bools, and validators return lists of errors or raise in strict mode.
- Use `None` for an absent optional object and explicit booleans for predicates. Keep metadata keys stable because tests and signing code inspect them directly (for example, `tests/test_handlers.py` and `tests/test_cache_integrity.py`).

## Module Design

- Expose the supported convenience API through `__all__` in `src/cacheness/__init__.py`; optional exports are added only when their dependencies/imports are available.
- Keep registry functions and backend classes together with their registry implementation (`src/cacheness/storage/backends/__init__.py` and `src/cacheness/storage/backends/blob_backends.py`).
- When an export is conditional, test the dependency itself rather than merely importing a helper function. `src/cacheness/__init__.py` currently reports YAML helpers as available even though PyYAML is only imported when the helper is called.
- Package `__init__.py` files act as deliberate barrels for public convenience imports: `src/cacheness/__init__.py` and `src/cacheness/storage/__init__.py` re-export core classes, handlers, metadata backends, and storage APIs.
- Prefer direct module imports for internal implementation dependencies to avoid expanding the public surface or creating circular imports; optional imports in `src/cacheness/core.py` are kept inside methods for this reason.

<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

## System Overview

```text

```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Public API and optional exports | Re-export cache classes, configuration, handlers, registries, and optional integrations | `src/cacheness/__init__.py` |
| Unified cache coordinator | Own cache lifecycle, key/value operations, TTL, metadata, integrity, signing, custom metadata, and size enforcement | `src/cacheness/core.py` |
| Configuration model | Compose storage, metadata, blob, compression, serialization, handler, and security settings; validate combinations | `src/cacheness/config.py` |
| Function cache facade | Normalize function arguments, add function identity to keys, and call `UnifiedCache` through `@cached` | `src/cacheness/decorators.py` |
| Handler registry | Select a handler by `can_handle`, resolve persisted `data_type`, and manage priorities/registration | `src/cacheness/handlers.py` |
| Handler contracts | Define cacheability, write/read, format, and specialized handler interfaces | `src/cacheness/interfaces.py` |
| Format handlers | Persist/reconstruct pandas/polars objects, NumPy arrays, TensorFlow tensors, and arbitrary objects | `src/cacheness/handlers.py` |
| Cache-key serializer | Deterministically serialize parameters and hash them with XXH3_64 | `src/cacheness/serialization.py` |
| Metadata abstraction | Persist entry records and hit/miss statistics through interchangeable backends | `src/cacheness/metadata.py` |
| Backend registries | Register and construct metadata and blob backend classes | `src/cacheness/storage/backends/__init__.py`, `src/cacheness/storage/backends/blob_backends.py` |
| BlobStore | Store arbitrary handler-backed objects with explicit keys/content hashes and metadata | `src/cacheness/storage/blob_store.py` |
| SQL pull-through cache | Query cached rows, detect missing ranges, fetch gaps, upsert, and return DataFrames | `src/cacheness/sql_cache.py` |
| Custom metadata | Register SQLAlchemy metadata models and link them to cache entries | `src/cacheness/custom_metadata.py` |
| Cross-cutting utilities | Compression, file hashing, HMAC signing, JSON compatibility, and error wrappers | `src/cacheness/compress_pickle.py`, `src/cacheness/file_hashing.py`, `src/cacheness/security.py`, `src/cacheness/json_utils.py`, `src/cacheness/error_handling.py` |

## Pattern Overview

- `UnifiedCache` delegates data-format decisions to ordered `CacheHandler` strategies rather than branching on every type in the coordinator (`src/cacheness/handlers.py`).
- Metadata and blob backends are selected through factories/registries, while the primary `UnifiedCache` path writes handler-produced files and records their paths in metadata (`src/cacheness/core.py`).
- Optional dependencies are imported conditionally; handlers are enabled only when their libraries are available (`src/cacheness/handlers.py`, `src/cacheness/__init__.py`).
- SQL caching uses an adapter contract for schema, query parsing, and external fetches, allowing builder methods to generate simple adapters (`src/cacheness/sql_cache.py`).
- Compatibility re-exports preserve older import paths through `src/cacheness/storage/handlers/__init__.py` and `src/cacheness/storage/backends/__init__.py`.

## Lifecycle Boundaries

```text

```

- `UnifiedCache.put()` writes a handler payload first, computes metadata/signature information, and then writes the metadata entry (`src/cacheness/core.py:811-922`). There is no rollback if metadata persistence fails, so payload and metadata are not one atomic transaction.
- Reads perform the inverse lookup through metadata and then the recorded `actual_path`; integrity/signature failures can remove metadata without consistently removing the payload (`src/cacheness/core.py:924-1053`).
- `BlobStore` repeats a similar two-step payload/metadata lifecycle independently (`src/cacheness/storage/blob_store.py:128-241`). It merges nested metadata during reads, but deletion/existence do not use the same normalization.
- Handler instances are genuinely selected through `HandlerRegistry`.
- Metadata backend classes can be constructed through a registry, but `UnifiedCache` does not consult that registry. Even an explicitly injected backend instance is overwritten by the subsequent config-selection branch (`src/cacheness/core.py:109-212`).
- Blob backends have a registry and implementations, but no high-level coordinator injects them into handler persistence. Treat these as incomplete composition seams, not interchangeable production strategies.

## Layers

- Purpose: Expose the stable user-facing constructors, decorators, factories, and registries.
- Location: `src/cacheness/__init__.py`, `src/cacheness/decorators.py`.
- Contains: `cacheness` alias, `cached`, `get_cache`, configuration helpers, and optional integrations.
- Depends on: `core.py`, configuration, handlers, metadata, and optional storage/SQL modules.
- Used by: Applications, examples, and tests.
- Purpose: Implement key/value caching semantics and orchestrate handlers plus metadata.
- Location: `src/cacheness/core.py`.
- Contains: `UnifiedCache.put`, `UnifiedCache.get`, invalidation, cleanup, stats, custom metadata, and global-cache factories.
- Depends on: `CacheConfig`, `HandlerRegistry`, key serialization, metadata factory, file hashing, and signing.
- Used by: Public API and decorators.
- Purpose: Map Python values to storage formats and reconstruct them.
- Location: `src/cacheness/handlers.py`, `src/cacheness/interfaces.py`, `src/cacheness/compress_pickle.py`.
- Contains: Parquet handlers, NumPy Blosc2/NPZ handling, pickle/dill object handling, and format metadata.
- Depends on: Optional NumPy, pandas, polars, TensorFlow, Blosc2, dill, and compression helpers.
- Used by: `UnifiedCache` and `BlobStore`.
- Purpose: Store cache-entry descriptors, timestamps, statistics, and backend-specific fields.
- Location: `src/cacheness/metadata.py`, `src/cacheness/storage/backends/`.
- Contains: JSON, SQLite, in-memory, PostgreSQL, and optional memory-cache wrapper implementations.
- Depends on: JSON utilities; SQLAlchemy for relational backends; psycopg for PostgreSQL.
- Used by: `UnifiedCache`, `BlobStore`, and custom metadata support.
- Purpose: Provide reusable object/blob storage independent of cache TTL and eviction semantics.
- Location: `src/cacheness/storage/blob_store.py`, `src/cacheness/storage/backends/blob_backends.py`, `src/cacheness/storage/backends/s3_backend.py`.
- Contains: `BlobStore`, filesystem/in-memory blob contracts, and optional S3 implementation.
- Depends on: Handler registry, metadata backend, compression, and boto3 for S3.
- Used by: Consumers importing the storage API directly. `UnifiedCache` does not instantiate `BlobStore`.
- Purpose: Cache tabular query results in a SQL table and fetch only missing data.
- Location: `src/cacheness/sql_cache.py`.
- Contains: `SqlCache`, `SqlCacheAdapter`, backend factories, query condition builders, TTL columns, gap detection, and upserts.
- Depends on: SQLAlchemy, pandas, and a caller-provided fetch adapter/function.
- Used by: Applications needing range-aware or analytical tabular caching.

## Data Flow

### Primary Request Path

### Decorated Function Flow

### SQL Pull-Through Flow

- `UnifiedCache` holds a per-instance `HandlerRegistry`, metadata backend, optional signer, and a `threading.Lock`; the module also exposes a lazily initialized global cache (`src/cacheness/core.py:68-108`, `src/cacheness/core.py:1204-1238`).
- Persistent metadata is authoritative for cache entries; cache files contain payloads and metadata stores `actual_path`, type, format, timestamps, and size.
- `CachedMetadataBackend` can add a `cachetools` in-memory entry layer over JSON/SQLite metadata (`src/cacheness/metadata.py:253-457`).
- SQL cache state lives in caller-defined tables with `cached_at` and optional `expires_at` columns (`src/cacheness/sql_cache.py:301-383`).

## Key Abstractions

- Purpose: Contract for data detection, write/read operations, file format, and persisted type identifier.
- Examples: `ArrayHandler`, `PandasDataFrameHandler`, `PolarsDataFrameHandler`, `ObjectHandler` in `src/cacheness/handlers.py`.
- Pattern: Strategy objects selected in registry order; register new handlers through `HandlerRegistry.register_handler` or the module-level API.
- Purpose: Abstract entry CRUD, statistics, TTL cleanup, and lifecycle operations.
- Examples: `JsonBackend`, `SqliteBackend`, `InMemoryBackend` in `src/cacheness/metadata.py`; `PostgresBackend` in `src/cacheness/storage/backends/postgresql_backend.py`.
- Pattern: Factory/registry selection through `create_metadata_backend` and `get_metadata_backend`; custom backends implement the backend contract.
- Purpose: Abstract binary blob read/write/delete/stream operations independently of metadata semantics.
- Examples: `FilesystemBlobBackend`, `InMemoryBlobBackend`, `S3BlobBackend` in `src/cacheness/storage/backends/`.
- Pattern: Registry lookup via `get_blob_backend`; S3 is optional and must be available/registered before use.
- Purpose: Bind a SQL table schema and external data source to pull-through caching.
- Examples: User-defined adapters or generated `SimpleTimeseriesAdapter`, `SimpleLookupAdapter`, and `SimpleAnalyticsAdapter` in `src/cacheness/sql_cache.py`.
- Pattern: Adapter plus builder/factory methods; fetchers return pandas DataFrames matching the table columns.
- Purpose: Central configuration object with focused sub-configurations and backward-compatible flat arguments.
- Examples: `CacheStorageConfig`, `CacheMetadataConfig`, `CacheBlobConfig`, `CompressionConfig`, `SerializationConfig`, `HandlerConfig`, and `SecurityConfig` in `src/cacheness/config.py`.
- Pattern: Dataclass sub-configurations composed by an imperative compatibility-aware constructor and factory helpers.

## Entry Points

- Location: `src/cacheness/__init__.py`.
- Triggers: `import cacheness`.
- Responsibilities: Expose public aliases, optional dependency guards, registration APIs, and version metadata.
- Location: `src/cacheness/core.py:60` (`UnifiedCache`), `src/cacheness/core.py:1219` (`get_cache`).
- Triggers: Direct construction, `cacheness(...)`, `get_cache()`, or `cached()`.
- Responsibilities: Key/value cache lifecycle and persistence coordination.
- Location: `src/cacheness/decorators.py:80` (`cached`).
- Triggers: `@cached`, `@memoize`, `cache_function`, or `CacheContext`.
- Responsibilities: Function-aware keying and transparent fallback to original function execution.
- Location: `src/cacheness/storage/blob_store.py:56`.
- Triggers: `from cacheness.storage import BlobStore`.
- Responsibilities: Explicit-key/content-addressable object storage with metadata and handler-backed formats.
- Location: `src/cacheness/sql_cache.py:142` (`SqlCache`).
- Triggers: `SqlCache.with_*` or `SqlCache.for_*` builders.
- Responsibilities: Query-aware tabular cache and external fetch orchestration.

## Architectural Constraints

- **Threading:** `UnifiedCache` creates `self._lock` but does not acquire it in `put()`, `get()`, invalidation, or cleanup. JSON/in-memory metadata backends use their own locks, and SQLite/PostgreSQL rely on SQLAlchemy sessions/pools, so metadata operations have some local coordination while the payload-plus-metadata lifecycle is not serialized (`src/cacheness/core.py:82-103`, `src/cacheness/metadata.py`, `src/cacheness/storage/backends/postgresql_backend.py`).
- **Global state:** `_global_cache` in `src/cacheness/core.py`, the decorator weak-reference list in `src/cacheness/decorators.py`, and handler/backend registries in `src/cacheness/handlers.py` and `src/cacheness/storage/backends/` are module-level mutable state.
- **Circular imports:** Compatibility modules intentionally re-export parent implementations: `src/cacheness/storage/handlers/__init__.py` imports `cacheness.handlers`, while `src/cacheness/storage/backends/__init__.py` imports implementations from `cacheness.metadata`.
- **Optional dependencies:** pandas, polars, SQLAlchemy, Blosc2, dill, TensorFlow, boto3, and database drivers are declared as optional and mostly guarded at runtime. NumPy is the exception: it is declared only in optional groups but imported eagerly by package-import paths, making it an effective undeclared base requirement (`pyproject.toml`, `src/cacheness/handlers.py`, `src/cacheness/compress_pickle.py`, `src/cacheness/__init__.py`).
- **Filesystem payloads:** The primary `UnifiedCache` handler path writes directly under `CacheStorageConfig.cache_dir`; persisted metadata must retain `actual_path` because handler extensions can be dynamic (`src/cacheness/core.py`, `src/cacheness/handlers.py`).
- **SQL schema ownership:** `SqlCache` mutates the supplied SQLAlchemy `Table` by appending cache columns before creating it (`src/cacheness/sql_cache.py:459-483`); callers must provide compatible primary keys and column definitions.

## Anti-Patterns

### Bypassing the handler and metadata contracts

### Treating `BlobStore` as a TTL cache

### Mixing SQL pull-through and key/value APIs

### Importing implementation modules through unstable compatibility paths

## Error Handling

- `UnifiedCache.put` propagates I/O and handler failures after logging; `UnifiedCache.get` treats missing/corrupt/unreadable entries as misses, removes metadata, and returns `None` (`src/cacheness/core.py:811-1054`).
- Handler-level domain exceptions are defined in `src/cacheness/interfaces.py`; broader cache error classes and decorators live in `src/cacheness/error_handling.py`.
- `@cached` optionally suppresses key/retrieval/storage errors and executes the wrapped function according to `ignore_errors` (`src/cacheness/decorators.py:145-220`).
- `SqlCache.get_data` wraps failures in `SQLCacheError`; failures fetching one missing range are printed as warnings while other ranges continue (`src/cacheness/sql_cache.py:486-530`).
- Optional feature imports raise focused `ImportError`/`MissingDependencyError` messages when a requested backend or format is unavailable (`src/cacheness/core.py`, `src/cacheness/sql_cache.py`).

## Cross-Cutting Concerns

<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
