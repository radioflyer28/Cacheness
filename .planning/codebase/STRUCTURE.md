# Codebase Structure

**Analysis Date:** 2026-04-02

## Directory Layout

```
Cacheness/
├── src/
│   └── cacheness/              # Main package (16,600+ lines of Python)
│       ├── __init__.py         # Public API exports
│       ├── core.py             # UnifiedCache — central coordinator (~3,900 lines)
│       ├── config.py           # Hierarchical dataclass configuration (~750 lines)
│       ├── handlers.py         # Type-aware serialization handlers (~900 lines)
│       ├── metadata.py         # Metadata backends: JSON, SQLite, ABC (~2,100 lines)
│       ├── interfaces.py       # ABCs, TypedDicts, dataclass contracts (~450 lines)
│       ├── decorators.py       # @cached / @cache_if decorators (~300 lines)
│       ├── security.py         # HMAC-SHA256 entry signing (~300 lines)
│       ├── serialization.py    # Cache key generation (xxhash-based)
│       ├── compress_pickle.py  # blosc2/pickle compression I/O
│       ├── custom_metadata.py  # SQLAlchemy-based custom metadata models
│       ├── entry_list.py       # Rich list wrapper for entry results
│       ├── error_handling.py   # Exception hierarchy and context managers
│       ├── file_hashing.py     # Parallel file/directory hashing (xxhash)
│       ├── json_utils.py       # orjson-backed JSON helpers with fallback
│       ├── size_utils.py       # Size/duration parsing and formatting
│       └── storage/            # Low-level storage sub-package
│           ├── __init__.py     # Storage layer public API
│           ├── blob_store.py   # BlobStore — standalone storage engine
│           ├── paths.py        # Path normalization (relative, cross-platform)
│           ├── compression.py  # Re-exports from compress_pickle.py
│           ├── security.py     # Re-exports from parent security.py
│           ├── handlers/       # Re-exports from parent handlers.py
│           │   └── __init__.py
│           └── backends/       # Pluggable backend implementations
│               ├── __init__.py            # Backend registry and factories
│               ├── base.py                # MetadataBackend ABC re-export
│               ├── blob_backends.py       # BlobBackend ABC + Filesystem/InMemory
│               ├── s3_backend.py          # S3BlobBackend (boto3-based)
│               └── postgresql_backend.py  # PostgresBackend (SQLAlchemy+psycopg2)
├── tests/                      # Test suite (~60 test files, 1427+ tests)
│   ├── conftest.py             # Shared fixtures, temp dirs, backend parametrization
│   ├── test_core.py            # Core UnifiedCache tests
│   ├── test_blob_store.py      # BlobStore standalone tests
│   ├── test_handlers.py        # Handler dispatch and serialization
│   ├── test_metadata.py        # Metadata backend tests
│   ├── test_decorators.py      # @cached / @cache_if tests
│   ├── test_security.py        # Entry signing and verification
│   ├── test_storage_mode.py    # Storage mode (no TTL/eviction)
│   ├── test_custom_metadata.py # Custom metadata model tests
│   ├── test_backend_parity.py  # Cross-backend equivalence tests
│   └── ...                     # ~50 more specialized test files
├── docs/                       # Documentation (Markdown)
│   ├── README.md               # Comprehensive guide index
│   ├── API_REFERENCE.md        # Public API documentation
│   ├── ARCHITECTURE.md         # Architecture overview
│   ├── BACKEND_SELECTION.md    # Backend comparison and selection guide
│   ├── BLOB_STORE.md           # BlobStore standalone usage
│   ├── CONFIGURATION.md        # Configuration reference
│   ├── SECURITY.md             # Security model documentation
│   ├── TROUBLESHOOTING.md      # Common issues and fixes
│   └── ...                     # ~20 more topic-specific docs
├── examples/                   # Usage examples (~20 scripts)
│   ├── simple_function_caching.py
│   ├── simple_api_caching.py
│   ├── ml_model_versioning.py
│   ├── s3_caching.py
│   └── ...
├── benchmarks/                 # Performance benchmarks
│   ├── comprehensive_backend_benchmark.py
│   ├── compression_benchmark.py
│   ├── handler_benchmark.py
│   └── ...
├── config/                     # Deployment and test configurations
│   ├── local_sqlite_fs.yaml    # Example YAML config
│   ├── test_config.json        # Example JSON config
│   ├── test_config.yaml        # Example YAML config
│   ├── Dockerfile.garage       # S3-compatible test server (Garage)
│   ├── garage.toml             # Garage configuration
│   └── garage-init.sh          # Garage initialization script
├── scripts/                    # Development scripts
│   ├── quality-check.ps1       # Windows quality gate script
│   ├── quality-check.sh        # Unix quality gate script
│   ├── install-hooks.ps1       # Git hook installer (Windows)
│   ├── install-hooks.sh        # Git hook installer (Unix)
│   └── hooks/                  # Git hook scripts
├── cache/                      # Default cache directory (gitignored data)
├── worktrees/                  # Git worktree container for feature branches
├── pyproject.toml              # Project metadata, dependencies, tool config
├── docker-compose.yml          # Docker services (PostgreSQL, Garage S3)
├── Makefile                    # Build/test shortcuts
├── README.md                   # Project README
├── CHANGELOG.md                # Release changelog
├── LICENSE                     # License file
└── AGENTS.md                   # Redirects to .github/copilot-instructions.md
```

## Directory Purposes

**`src/cacheness/`:**
- Purpose: Core library package — all production code
- Contains: Python modules implementing the cache system
- Key files: `core.py` (coordinator), `handlers.py` (serialization), `metadata.py` (backends), `config.py` (configuration)

**`src/cacheness/storage/`:**
- Purpose: Low-level storage sub-package — separates storage concerns from caching semantics
- Contains: `BlobStore`, path helpers, compression wrappers, backend implementations
- Key files: `blob_store.py`, `backends/blob_backends.py`, `backends/s3_backend.py`, `backends/postgresql_backend.py`

**`src/cacheness/storage/backends/`:**
- Purpose: Pluggable backend implementations and registries
- Contains: Metadata backend ABCs and factories, blob backend ABCs and implementations
- Key files: `blob_backends.py` (BlobBackend ABC + filesystem/memory), `postgresql_backend.py`, `s3_backend.py`

**`src/cacheness/storage/handlers/`:**
- Purpose: Re-export layer for handler classes — provides `from cacheness.storage.handlers import ...` path
- Contains: Only `__init__.py` that re-exports from `src/cacheness/handlers.py`

**`tests/`:**
- Purpose: Comprehensive test suite — unit, integration, property-based, stress tests
- Contains: ~60 test files, shared fixtures in `conftest.py`
- Key files: `test_core.py`, `test_blob_store.py`, `test_handlers.py`, `test_metadata.py`, `test_backend_parity.py`

**`docs/`:**
- Purpose: Detailed documentation beyond README — guides, references, troubleshooting
- Contains: ~25 Markdown files covering API, architecture, backends, security, configuration

**`examples/`:**
- Purpose: Runnable usage examples for common scenarios
- Contains: ~20 Python scripts demonstrating decorator, direct API, storage mode, S3, ML, and custom metadata usage

**`benchmarks/`:**
- Purpose: Performance measurement scripts for backends, handlers, compression, serialization
- Contains: ~10 benchmark scripts with result analysis

**`config/`:**
- Purpose: Example and test configuration files, Docker infrastructure for integration tests
- Contains: YAML/JSON config examples, Dockerfile and config for Garage (S3-compatible server)

**`scripts/`:**
- Purpose: Development automation — quality checks, git hooks, environment setup
- Contains: PowerShell and bash scripts for linting (ruff), type checking (ty), hook installation

## Key File Locations

**Entry Points:**
- `src/cacheness/__init__.py`: Package public API — exports `cacheness` (UnifiedCache), decorators, handlers, backends, config
- `src/cacheness/core.py`: `UnifiedCache` class — the main cache coordinator
- `src/cacheness/decorators.py`: `@cached()` and `@cache_if()` function decorators
- `src/cacheness/storage/blob_store.py`: `BlobStore` — standalone storage engine

**Configuration:**
- `src/cacheness/config.py`: All `@dataclass` configs (`CacheConfig`, `CacheStorageConfig`, `CacheMetadataConfig`, `CacheBlobConfig`, `CompressionConfig`, `SerializationConfig`, `HandlerConfig`, `SecurityConfig`, `HooksConfig`)
- `pyproject.toml`: Project metadata, dependencies, ruff/ty config, pytest settings
- `config/local_sqlite_fs.yaml`: Example YAML configuration for file-based setup
- `config/test_config.json`: Example JSON configuration

**Core Logic:**
- `src/cacheness/core.py`: Cache orchestration — put/get/delete/query/verify/stats/eviction (~3,900 lines)
- `src/cacheness/handlers.py`: Handler implementations — Polars/Pandas DataFrame, NumPy array, pickle/dill object, bytes, TensorFlow tensor
- `src/cacheness/metadata.py`: `MetadataBackend` ABC, `JsonBackend`, `SqliteBackend` (SQLAlchemy ORM), namespace management, schema versioning
- `src/cacheness/storage/backends/postgresql_backend.py`: `PostgresBackend` for distributed caching
- `src/cacheness/storage/backends/blob_backends.py`: `BlobBackend` ABC, `FilesystemBlobBackend`, `InMemoryBlobBackend`, blob registry
- `src/cacheness/storage/backends/s3_backend.py`: `S3BlobBackend` for S3-compatible storage

**Interfaces and Contracts:**
- `src/cacheness/interfaces.py`: All shared ABCs (`CacheHandler`, `CacheWriter`, `CacheReader`) and typed contracts (`HandlerResult`, `WriteBlobResult`, `IntegrityReport`, `EntryData`, `EntrySummary`, `BlobReadContext`, `SignableFields`)

**Security:**
- `src/cacheness/security.py`: `CacheEntrySigner` — HMAC-SHA256 signing with versioned field lists, key management

**Utilities:**
- `src/cacheness/serialization.py`: `create_unified_cache_key()` — deterministic key generation via xxhash
- `src/cacheness/compress_pickle.py`: blosc2/pickle compression — `write_file()`, `read_file()`, codec detection
- `src/cacheness/file_hashing.py`: `hash_file_content()`, `hash_directory_parallel()` — xxhash-based integrity
- `src/cacheness/size_utils.py`: `parse_size()`, `parse_duration()`, `format_size()` — human-readable unit conversion
- `src/cacheness/json_utils.py`: `dumps()`, `loads()` — orjson with stdlib fallback
- `src/cacheness/error_handling.py`: `CacheError` hierarchy, `cache_operation_context()` context manager
- `src/cacheness/entry_list.py`: `EntryList` — list subclass with `.to_dataframe()`, `.filter()`, `.sort_by()`, `.keys()`

**Custom Metadata:**
- `src/cacheness/custom_metadata.py`: `@custom_metadata_model()` decorator, `CustomMetadataBase` mixin, namespace-aware dynamic table creation

**Testing:**
- `tests/conftest.py`: Shared pytest fixtures — temp cache dirs, backend parametrization, Docker group markers
- `tests/test_core.py`: Core `UnifiedCache` tests (put/get/delete/TTL/eviction)
- `tests/test_blob_store.py`: `BlobStore` standalone tests
- `tests/test_handlers.py`: Handler dispatch and type-specific serialization
- `tests/test_metadata.py`: Metadata backend CRUD operations
- `tests/test_backend_parity.py`: Cross-backend equivalence (JSON vs SQLite vs PostgreSQL)
- `tests/test_cache_integrity.py`: Integrity verification and repair
- `tests/test_security.py`: Entry signing, verification, key rotation
- `tests/test_storage_mode.py`: Storage mode (no TTL, no eviction)
- `tests/test_s3_blob_backend.py`: S3 blob backend integration tests

## Naming Conventions

**Files:**
- Source modules: `snake_case.py` (e.g., `blob_store.py`, `compress_pickle.py`, `size_utils.py`)
- Test files: `test_<module_or_feature>.py` (e.g., `test_core.py`, `test_blob_store.py`, `test_backend_parity.py`)
- Docs: `UPPER_SNAKE_CASE.md` (e.g., `API_REFERENCE.md`, `BACKEND_SELECTION.md`)
- Example scripts: `snake_case.py` with descriptive names (e.g., `simple_function_caching.py`, `ml_model_versioning.py`)

**Directories:**
- Package directories: `snake_case` (e.g., `cacheness/`, `storage/`, `backends/`, `handlers/`)
- Top-level directories: `lowercase` (e.g., `tests/`, `docs/`, `examples/`, `benchmarks/`, `config/`, `scripts/`)

**Classes:**
- `PascalCase` (e.g., `UnifiedCache`, `BlobStore`, `HandlerRegistry`, `CacheEntrySigner`, `S3BlobBackend`)
- Config dataclasses: `Cache<Concern>Config` (e.g., `CacheStorageConfig`, `CacheMetadataConfig`)
- Handler classes: `<Type>Handler` (e.g., `PolarsDataFrameHandler`, `ArrayHandler`, `ObjectHandler`)
- Backend classes: `<Type>Backend` (e.g., `JsonBackend`, `SqliteBackend`, `PostgresBackend`, `FilesystemBlobBackend`)

**Functions/Methods:**
- Public: `snake_case` (e.g., `put()`, `get()`, `list_entries()`, `verify_integrity()`)
- Private: `_snake_case` (e.g., `_init_blob_store()`, `_resolve_cache_key()`, `_is_expired()`)
- Factory functions: `create_<thing>()` or `get_<thing>()` (e.g., `create_metadata_backend()`, `get_blob_backend()`, `get_cache()`)

## Where to Add New Code

**New Handler (e.g., Arrow, Avro):**
- Implementation: Add handler class in `src/cacheness/handlers.py` implementing `CacheHandler` ABC from `src/cacheness/interfaces.py`
- Registration: Add to `HandlerRegistry.__init__()` in `src/cacheness/handlers.py` with appropriate priority
- Config toggle: Add `enable_<type>` flag to `HandlerConfig` in `src/cacheness/config.py`
- Tests: `tests/test_handlers.py` or a new `tests/test_<type>_handler.py`

**New Metadata Backend (e.g., Redis, DynamoDB):**
- Implementation: New file in `src/cacheness/storage/backends/` implementing `MetadataBackend` ABC from `src/cacheness/metadata.py`
- Registration: Register via `register_metadata_backend()` in `src/cacheness/storage/backends/__init__.py`
- Tests: `tests/test_<backend>_backend.py`

**New Blob Backend (e.g., GCS, Azure Blob):**
- Implementation: New file in `src/cacheness/storage/backends/` implementing `BlobBackend` ABC from `src/cacheness/storage/backends/blob_backends.py`
- Registration: Register via `register_blob_backend()` in `src/cacheness/storage/backends/blob_backends.py`
- Tests: `tests/test_<backend>_blob_backend.py`

**New Utility:**
- Shared helpers: `src/cacheness/` (new module or extend existing `size_utils.py`, `json_utils.py`, etc.)
- Storage-specific: `src/cacheness/storage/`

**New Feature on UnifiedCache:**
- Add method(s) to `UnifiedCache` in `src/cacheness/core.py`
- Export from `src/cacheness/__init__.py` if public
- Tests: `tests/test_core.py` or new `tests/test_<feature>.py`

## Special Directories

**`cache/`:**
- Purpose: Default cache data directory (created at runtime)
- Generated: Yes
- Committed: Partially — `cache/default/` structure is committed but data files are gitignored

**`worktrees/`:**
- Purpose: Git worktree container for isolated feature branch development
- Generated: Yes (by `git worktree add`)
- Committed: No (gitignored)

**`.planning/`:**
- Purpose: GSD planning artifacts — phase plans, codebase maps, session notes
- Generated: Yes (by GSD workflow)
- Committed: Yes

**`.github/`:**
- Purpose: GitHub-specific configs — Copilot instructions, agent definitions, GSD skills
- Generated: No
- Committed: Yes
- Key file: `.github/copilot-instructions.md` — canonical agent instructions

**`logs/`:**
- Purpose: Log output directory
- Generated: Yes
- Committed: Directory only

---

*Structure analysis: 2026-04-02*
