# Codebase Structure

**Analysis Date:** 2026-08-29

## Directory Layout

```text
cacheness/
├── src/cacheness/                 # Installable Python package
│   ├── __init__.py                # Public exports and optional feature guards
│   ├── core.py                    # UnifiedCache coordinator and global cache
│   ├── decorators.py              # @cached, memoize, cache context
│   ├── config.py                  # Composed cache configuration
│   ├── handlers.py                # Built-in handlers and handler registry
│   ├── interfaces.py              # Handler contracts and handler errors
│   ├── metadata.py                # Metadata models, factories, and backends
│   ├── serialization.py           # Cache-key serialization and hashing
│   ├── sql_cache.py               # SQL pull-through cache
│   ├── custom_metadata.py         # Registered SQLAlchemy metadata models
│   ├── storage/                    # Reusable low-level storage API
│   │   ├── blob_store.py           # BlobStore object API
│   │   ├── compression.py          # Compression re-exports
│   │   ├── security.py             # Security re-export
│   │   ├── backends/               # Metadata/blob backend contracts and registries
│   │   └── handlers/               # Handler compatibility re-exports
│   └── ...                         # Hashing, compression, JSON, security, errors
├── tests/                         # Pytest unit, integration, and optional-dependency tests
├── examples/                      # Runnable usage examples
├── benchmarks/                    # Performance and backend benchmark scripts
├── docs/                          # API, configuration, backend, and feature documentation
├── pyproject.toml                 # Packaging, dependencies, pytest, coverage, Ruff
├── uv.lock                        # Locked development/project dependencies
├── verify_platform.py             # Platform compatibility verification
└── .gitignore                     # Excludes generated cache payloads and local artifacts
```

## Directory Purposes

**`src/cacheness/`:**
- Purpose: Installable library implementation using the src layout.
- Contains: Public facades, cache coordination, handlers, backends, configuration, utilities, and SQL caching.
- Key files: `src/cacheness/__init__.py`, `src/cacheness/core.py`, `src/cacheness/handlers.py`, `src/cacheness/metadata.py`, `src/cacheness/sql_cache.py`.

**`src/cacheness/storage/`:**
- Purpose: Reusable lower-level storage surface separated from higher-level TTL/eviction behavior.
- Contains: `BlobStore`, compression/security exports, and backend/handler subpackages.
- Key files: `src/cacheness/storage/blob_store.py`, `src/cacheness/storage/__init__.py`.

**`src/cacheness/storage/backends/`:**
- Purpose: Pluggable metadata and blob backend contracts, implementations, and registries.
- Contains: `base.py`, `blob_backends.py`, `postgresql_backend.py`, optional `s3_backend.py`, and registry exports in `__init__.py`.
- Key files: `src/cacheness/storage/backends/__init__.py`, `src/cacheness/storage/backends/blob_backends.py`, `src/cacheness/storage/backends/postgresql_backend.py`.

**`src/cacheness/storage/handlers/`:**
- Purpose: Compatibility import path for type handlers and handler interfaces.
- Contains: `__init__.py`, which re-exports implementations from `src/cacheness/handlers.py` and contracts from `src/cacheness/interfaces.py`.
- Key files: `src/cacheness/storage/handlers/__init__.py`.

**`tests/`:**
- Purpose: Pytest coverage for core behavior, serialization, metadata, handlers, storage registries, SQL cache, concurrency, and optional integrations.
- Contains: `test_core.py`, `test_decorators.py`, `test_handlers.py`, `test_metadata.py`, `test_sql_cache.py`, backend-specific tests, and compatibility tests.
- Key files: `tests/test_integration.py`, `tests/test_backend_compatibility.py`, `tests/test_sqlite_concurrency.py`.

**`examples/`:**
- Purpose: User-facing recipes for key/value caching, decorators, configuration, dataframes, SQL, S3, metadata, and ML artifacts.
- Contains: Small scripts such as `examples/simple_function_caching.py`, `examples/beginner_sql_cache.py`, and `examples/s3_caching.py`.
- Key files: `examples/README.md`, `examples/simple_config_demo.py`.

**`benchmarks/`:**
- Purpose: Measure backend, serialization, threshold, and memory/performance behavior.
- Contains: Standalone Python benchmark scripts and checked-in analysis output.
- Key files: `benchmarks/comprehensive_backend_benchmark.py`, `benchmarks/test_performance_comparison.py`.

**`docs/`:**
- Purpose: Detailed reference and design guidance beyond the package docstrings.
- Contains: API reference, configuration, backend selection, SQL cache, custom metadata, security, compatibility, and performance documentation.
- Key files: `docs/API_REFERENCE.md`, `docs/CONFIGURATION.md`, `docs/BACKEND_SELECTION.md`, `docs/SQL_CACHE.md`.

## Key File Locations

**Entry Points:**
- `src/cacheness/__init__.py`: Public package namespace and optional exports.
- `src/cacheness/core.py`: `UnifiedCache`, `get_cache`, and `reset_cache`.
- `src/cacheness/decorators.py`: `cached`, `cache_function`, `memoize`, and `CacheContext`.
- `src/cacheness/storage/blob_store.py`: Low-level `BlobStore` API.
- `src/cacheness/sql_cache.py`: `SqlCache` and `SqlCacheAdapter`.

**Configuration:**
- `pyproject.toml`: Project metadata, Python requirement, dependencies, optional extras, pytest settings, coverage, and Ruff settings.
- `src/cacheness/config.py`: Runtime `CacheConfig` and focused sub-configurations.
- `src/cacheness/json_utils.py`: JSON serialization compatibility helpers used by metadata backends.
- `.gitignore`: Generated cache databases, payloads, metadata JSON, signing key path, and Python build artifacts.

**Core Logic:**
- `src/cacheness/core.py`: Key/value orchestration, lifecycle, TTL, integrity, signing, and eviction.
- `src/cacheness/handlers.py`: Format-specific persistence and ordered handler selection.
- `src/cacheness/serialization.py`: Parameter serialization and XXH3_64 cache-key generation.
- `src/cacheness/metadata.py`: Metadata contracts, JSON/SQLite/in-memory persistence, and factory selection.
- `src/cacheness/sql_cache.py`: SQL schema augmentation, query conditions, gap detection, upsert, and tabular retrieval.
- `src/cacheness/custom_metadata.py`: Decorator registry and cache-entry link-table model.

**Testing:**
- `tests/test_core.py`: UnifiedCache behavior and lifecycle.
- `tests/test_decorators.py`: Decorator keying and cache behavior.
- `tests/test_handlers.py`, `tests/test_serialization.py`: Type handlers and key serialization.
- `tests/test_metadata.py`, `tests/test_metadata_backend_registry.py`: Metadata persistence and registries.
- `tests/test_sql_cache.py`, `tests/test_sqlite_concurrency.py`: SQL pull-through and concurrency behavior.
- `tests/test_s3_blob_backend.py`, `tests/test_postgresql_backend.py`: Optional external backend coverage.

## Naming Conventions

**Files:**
- Use lowercase `snake_case.py` for implementation modules, e.g. `file_hashing.py`, `sql_cache.py`, and `blob_store.py`.
- Use `test_<subject>.py` for pytest modules, e.g. `tests/test_cache_integrity.py` and `tests/test_query_meta.py`.
- Use uppercase Markdown names for top-level documentation, e.g. `docs/API_REFERENCE.md` and `docs/CONFIGURATION.md`.
- Use suffixes that describe the implementation role: `_backend.py`, `_store.py`, `_cache.py`, `_utils.py`.

**Directories:**
- Use lowercase plural or domain names: `tests/`, `examples/`, `benchmarks/`, `docs/`, `storage/`, `backends/`, and `handlers/`.
- Keep installable code under `src/cacheness/`; do not place new package modules at repository root.

**Python symbols:**
- Classes use `PascalCase`, functions and variables use `snake_case`, and module-level feature flags use uppercase names such as `SQLALCHEMY_AVAILABLE` (`src/cacheness/metadata.py`).
- Backend and handler classes end with `Backend` or `Handler`; SQL cache adapters end with `Adapter` (`src/cacheness/handlers.py`, `src/cacheness/sql_cache.py`).
- Public compatibility aliases are declared in `src/cacheness/__init__.py`; preserve them when changing canonical implementations.

## Where to Add New Code

**New Feature:**
- Primary code: Put UnifiedCache behavior in `src/cacheness/core.py` only when it is cross-format coordination; put format behavior in `src/cacheness/handlers.py`; put standalone storage behavior in `src/cacheness/storage/`.
- Tests: Add focused coverage under `tests/test_<feature>.py`, with integration/backend-specific coverage in an existing relevant module when appropriate.
- Documentation: Add or update the nearest domain reference in `docs/` and a runnable recipe in `examples/` for user-facing APIs.

**New Component/Module:**
- Implementation: Add a module under `src/cacheness/` for top-level behavior or under `src/cacheness/storage/` for reusable storage infrastructure.
- New cache data type: Implement `CacheHandler` in `src/cacheness/handlers.py`, add it to `HandlerRegistry` priority/config handling, and expose it through `src/cacheness/storage/handlers/__init__.py` only if the compatibility surface requires it.
- New metadata backend: Implement the metadata contract, register it through `src/cacheness/storage/backends/`, and route construction through `create_metadata_backend`/registry APIs (`src/cacheness/metadata.py`, `src/cacheness/storage/backends/__init__.py`).
- New blob backend: Implement `BlobBackend` in `src/cacheness/storage/backends/`, register it in `blob_backends.py`, and add optional imports/exports without making the base install require its SDK.
- New SQL access pattern: Add a `SqlCache` builder in `src/cacheness/sql_cache.py` that creates a schema and an adapter, and cover range/filter/TTL behavior in `tests/test_sql_cache.py`.

**Utilities:**
- Shared helpers: Put cache-key logic in `src/cacheness/serialization.py`, file hashing in `src/cacheness/file_hashing.py`, compression in `src/cacheness/compress_pickle.py` or its storage re-export, error wrappers in `src/cacheness/error_handling.py`, and signing in `src/cacheness/security.py`.
- Avoid placing generic helpers in `core.py`; keep the coordinator focused on orchestration.

## Special Directories

**`cache/`:**
- Purpose: Default runtime payload/metadata location used by `UnifiedCache` and local examples.
- Generated: Yes; handlers may create `.pkl`, `.pkl.<codec>`, `.npz`, `.b2nd`, and `.parquet` payloads, while metadata backends create database/JSON files.
- Committed: No; generated cache artifacts are excluded by `.gitignore`.

**`.planning/codebase/`:**
- Purpose: Codebase mapping artifacts consumed by planning/execution workflows.
- Generated: Yes, by mapping agents.
- Committed: Intended as project planning documentation; write analysis documents here and do not mix them into `src/`.

**`__pycache__/`, `.pytest_cache/`, `.ruff_cache/`:**
- Purpose: Python, pytest, and Ruff generated caches.
- Generated: Yes.
- Committed: No; excluded or treated as local build/test artifacts.

**`.venv/`:**
- Purpose: Local Python virtual environment for development.
- Generated: Yes.
- Committed: No; excluded by `.gitignore`.

---

*Structure analysis: 2026-08-29*
