# Codebase Structure

**Analysis Date:** 2026-04-07

## Directory Layout

```
Cacheness/
├── src/
│   └── cacheness/                    # Main package (37 .py files)
│       ├── __init__.py               # Public API re-exports (274 lines)
│       ├── core.py                   # UnifiedCache class, put/get/invalidate (1217 lines)
│       ├── config.py                 # Dataclass configs, validation, JSON/YAML loaders (1253 lines)
│       ├── interfaces.py            # ABCs, TypedDicts, dataclasses — all contracts (505 lines)
│       ├── decorators.py            # @cached, @cache_if function decorators (568 lines)
│       ├── encryption.py            # AES-256-GCM encrypt/decrypt, key derivation (81 lines)
│       ├── security.py              # CacheEntrySigner, HMAC-SHA256, HKDF (470 lines)
│       ├── error_handling.py        # CacheError hierarchy, @cache_operation_context (302 lines)
│       ├── serialization.py         # Cache key generation, serialize_for_cache_key (361 lines)
│       ├── compress_pickle.py       # Blosc2/pickle I/O, compression utilities (760 lines)
│       ├── file_hashing.py          # xxhash file/directory hashing (192 lines)
│       ├── size_utils.py            # parse_size, format_size, parse_duration (260 lines)
│       ├── json_utils.py            # orjson wrapper with stdlib fallback (67 lines)
│       ├── entry_list.py            # EntryList rich list wrapper (148 lines)
│       ├── write_intent.py          # WriteIntentJournal for crash recovery (89 lines)
│       ├── custom_metadata.py       # SQLAlchemy custom metadata models (547 lines)
│       │
│       ├── _verification_mixin.py   # Integrity verification, signing (261 lines)
│       ├── _stats_mixin.py          # Hit/miss recording, get_stats (26 lines)
│       ├── _custom_metadata_mixin.py # Custom metadata store/retrieve/query (315 lines)
│       ├── _storage_mode_mixin.py   # Storage-mode passthrough put/get (170 lines)
│       ├── _query_mixin.py          # query_meta, delete_where (247 lines)
│       ├── _convenience_mixin.py    # put/get_with_meta, put/get_with_model (328 lines)
│       ├── _batch_mixin.py          # delete_by_prefix, put_batch, batch ops (264 lines)
│       ├── _file_ops_mixin.py       # put_file, get_file (182 lines)
│       ├── _get_variants_mixin.py   # get_with_metadata, get_metadata (169 lines)
│       ├── _update_mixin.py         # exists, update_data, touch (315 lines)
│       ├── _inline_blob_mixin.py    # Inline blob helpers (232 lines)
│       ├── _put_cleanup.py          # _PutCleanup rollback guard (58 lines)
│       │
│       ├── handlers/                 # Type-aware serialization package (11 files)
│       │   ├── __init__.py           # Re-exports all handlers (54 lines)
│       │   ├── _compat.py            # Shared imports, optional dep flags (82 lines)
│       │   ├── registry.py           # HandlerRegistry with priority selection (289 lines)
│       │   ├── pandas_dataframe.py   # PandasDataFrameHandler → Parquet (87 lines)
│       │   ├── pandas_series.py      # PandasSeriesHandler → Parquet (102 lines)
│       │   ├── polars_dataframe.py   # PolarsDataFrameHandler → Parquet (96 lines)
│       │   ├── polars_series.py      # PolarsSeriesHandler → Parquet (99 lines)
│       │   ├── numpy_array.py        # ArrayHandler → blosc2 .npz (244 lines)
│       │   ├── bytes_handler.py      # BytesHandler → raw bytes (88 lines)
│       │   ├── object_handler.py     # ObjectHandler → pickle/dill (343 lines)
│       │   └── tensorflow_tensor.py  # TensorFlowTensorHandler (139 lines)
│       │
│       ├── metadata/                 # Pluggable metadata backends package (5 files)
│       │   ├── __init__.py           # create_metadata_backend() factory (148 lines)
│       │   ├── _compat.py            # ORM models, namespace utils, migrations (268 lines)
│       │   ├── base.py               # MetadataBackend ABC, CachedMetadataBackend (578 lines)
│       │   ├── json_backend.py       # JsonBackend — JSON file storage (632 lines)
│       │   └── sqlite_backend.py     # SqliteBackend — SQLAlchemy ORM (1122 lines)
│       │
│       └── storage/                  # Low-level storage layer package
│           ├── __init__.py           # Re-exports BlobStore, backends, handlers (77 lines)
│           ├── blob_store.py         # BlobStore — standalone blob API (1199 lines)
│           ├── compression.py        # Re-exports from compress_pickle (49 lines)
│           ├── paths.py              # resolve_actual_path, to_relative_path (41 lines)
│           ├── security.py           # Re-exports CacheEntrySigner (23 lines)
│           ├── handlers/
│           │   └── __init__.py       # Re-exports from cacheness.handlers (86 lines)
│           └── backends/
│               ├── __init__.py       # Backend registries, re-exports (240 lines)
│               ├── base.py           # MetadataBackend re-export alias (25 lines)
│               ├── blob_backends.py  # BlobBackend ABC, Filesystem, InMemory (532 lines)
│               ├── s3_backend.py     # S3BlobBackend for S3/MinIO (542 lines)
│               └── postgresql_backend.py  # PostgresBackend for PostgreSQL (1410 lines)
│
├── tests/                            # Test suite (76 test files, ~1660 tests)
│   ├── conftest.py                   # Shared fixtures, tmp_path helpers
│   ├── test_core.py                  # Core put/get/invalidate tests (1086 lines)
│   ├── test_decorators.py            # @cached/@cache_if tests (869 lines)
│   ├── test_update_operations.py     # update_data, put_batch tests (730 lines)
│   ├── test_handlers.py             # Handler type detection, serialization (595 lines)
│   ├── test_metadata.py             # Metadata backend CRUD tests
│   ├── test_blob_store.py           # BlobStore standalone tests
│   ├── test_sqlite_schema_versioning.py  # SQLite migration tests (983 lines)
│   ├── test_pg_schema_versioning.py  # PostgreSQL migration tests (576 lines)
│   ├── test_json_schema_versioning.py # JSON schema tests
│   ├── test_backend_parity.py       # Cross-backend parametrized tests (629 lines)
│   ├── test_backend_compatibility.py # Backend compat tests
│   ├── test_encryption_at_rest.py   # AES-256-GCM encryption tests
│   ├── test_inline_blobs.py         # Inline blob tests
│   ├── test_config_validation.py    # Config validation tests (520 lines)
│   ├── test_security.py             # Signing/verification tests (implied)
│   ├── test_cache_integrity.py      # Integrity report tests
│   ├── test_cache_integrity_verification.py  # Signature verification
│   ├── test_concurrency_stress.py   # Thread safety stress tests
│   ├── test_sqlite_concurrency.py   # SQLite concurrent access (579 lines)
│   ├── test_thread_safety.py        # Thread safety tests
│   ├── test_key_rotation.py         # Key rotation tests
│   ├── test_key_rotation_api.py     # Key rotation API tests
│   ├── test_key_fallback_policy.py  # Key fallback policy tests
│   ├── test_hkdf_derivation.py      # HKDF key derivation tests
│   ├── test_namespace_signing.py    # Namespace signing tests
│   ├── test_s3_blob_backend.py      # S3 backend tests (645 lines)
│   ├── test_postgresql_backend.py   # PostgreSQL backend tests (517 lines)
│   ├── test_property_based.py       # Property-based tests (698 lines)
│   ├── test_custom_metadata.py      # Custom metadata tests (569 lines)
│   ├── test_custom_metadata_backends.py  # Custom metadata backend tests (626 lines)
│   ├── test_put_get_file.py         # File operations tests (538 lines)
│   ├── test_write_intent.py         # Write intent journal tests
│   ├── test_storage_mode.py         # Storage mode tests
│   ├── test_size_utils.py           # Size/duration parsing tests (514 lines)
│   ├── test_file_hashing.py         # File hashing tests (663 lines)
│   ├── test_query_meta.py           # query_meta tests (601 lines)
│   ├── test_convenience_helpers.py  # Convenience API tests
│   ├── test_entry_list.py           # EntryList tests
│   └── ... (40+ more test files)
│
├── config/                           # External configuration
│   ├── Dockerfile.garage             # Garage S3-compatible server for testing
│   ├── garage-init.sh                # Garage initialization script
│   ├── garage.toml                   # Garage S3 server config
│   ├── local_sqlite_fs.yaml         # Example: SQLite + filesystem config
│   ├── test_config.json             # Test configuration (JSON format)
│   ├── test_config.yaml             # Test configuration (YAML format)
│   └── README.md                     # Config file documentation
│
├── docs/                             # Documentation (26 files)
│   ├── README.md                     # Comprehensive documentation hub
│   ├── API_REFERENCE.md              # Public API reference
│   ├── ARCHITECTURE.md               # Architecture overview
│   ├── BACKEND_SELECTION.md          # Backend selection guide
│   ├── SECURITY.md                   # Security model documentation
│   ├── CONFIGURATION.md              # Configuration guide
│   ├── BLOB_STORE.md                 # BlobStore usage guide
│   ├── CUSTOM_METADATA.md            # Custom metadata guide
│   ├── PLUGIN_DEVELOPMENT.md         # Handler plugin guide
│   ├── PERFORMANCE.md                # Performance benchmarks
│   ├── TROUBLESHOOTING.md            # Common issues and fixes
│   ├── CROSS_PLATFORM_GUIDE.md       # Cross-platform compatibility
│   ├── WINDOWS_COMPATIBILITY.md      # Windows-specific notes
│   └── ... (13 more topic-specific docs)
│
├── examples/                         # Usage examples (20 files)
│   ├── simple_function_caching.py    # Basic function caching
│   ├── simple_api_caching.py         # API request caching
│   ├── ml_model_versioning.py        # ML model storage
│   ├── pipeline_artifact_storage.py  # Data pipeline checkpoints
│   ├── custom_metadata_demo.py       # Custom metadata usage
│   ├── s3_caching.py                 # S3 blob backend usage
│   └── ... (14 more examples)
│
├── benchmarks/                       # Performance benchmarks (11 files)
│   ├── comprehensive_backend_benchmark.py
│   ├── compression_benchmark.py
│   ├── handler_benchmark.py
│   ├── serialization_benchmark.py
│   └── ... (7 more benchmarks)
│
├── scripts/                          # Development scripts
│   ├── quality-check.ps1             # Windows quality gates (ruff+ty)
│   ├── quality-check.sh              # Unix quality gates
│   ├── install-hooks.ps1             # Install git hooks (Windows)
│   ├── install-hooks.sh              # Install git hooks (Unix)
│   ├── setup_local_env.py            # Local environment setup
│   └── hooks/
│       └── pre-commit                # Pre-commit quality check hook
│
├── pyproject.toml                    # Project config, dependencies, tool settings
├── docker-compose.yml                # PostgreSQL + MinIO for testing
├── Makefile                          # Build/test shortcuts
├── CHANGELOG.md                      # Version changelog
└── README.md                         # Project README
```

## Directory Purposes

**`src/cacheness/`:**
- Purpose: Main library package — all production code
- Contains: Core cache logic, handlers, metadata backends, storage layer, security, config
- Key files: `core.py` (1217 lines), `config.py` (1253 lines), `interfaces.py` (505 lines)

**`src/cacheness/handlers/`:**
- Purpose: Type-specific serialization handlers
- Contains: One file per data type handler + registry + shared compat imports
- Key files: `registry.py` (handler selection), `object_handler.py` (fallback handler)

**`src/cacheness/metadata/`:**
- Purpose: Pluggable metadata backend implementations
- Contains: ABC, JSON backend, SQLite backend, ORM models, factory
- Key files: `sqlite_backend.py` (1122 lines — largest metadata impl), `base.py` (578 lines — ABC)

**`src/cacheness/storage/`:**
- Purpose: Low-level blob storage infrastructure
- Contains: BlobStore, blob backends (filesystem/S3/in-memory), compression wrappers, path utilities
- Key files: `blob_store.py` (1199 lines), `backends/postgresql_backend.py` (1410 lines)

**`tests/`:**
- Purpose: Comprehensive test suite — unit, integration, property-based, stress tests
- Contains: 76 test files, ~1,660 tests (1558 passed + 102 skipped baseline)
- Key files: `test_core.py` (1086 lines), `conftest.py` (shared fixtures)

**`config/`:**
- Purpose: External configuration files and Docker configs for testing
- Contains: Example YAML/JSON configs, Garage S3 server config, Dockerfile
- Generated: No
- Committed: Yes

**`docs/`:**
- Purpose: Comprehensive project documentation
- Contains: 26 markdown files covering API, architecture, security, troubleshooting
- Key files: `README.md` (hub), `API_REFERENCE.md`, `SECURITY.md`

**`examples/`:**
- Purpose: Runnable usage examples for documentation
- Contains: 20 Python scripts demonstrating various features
- Key files: `simple_function_caching.py`, `ml_model_versioning.py`

## Key File Locations

**Entry Points:**
- `src/cacheness/__init__.py`: Library public API, `from cacheness import cacheness`
- `src/cacheness/core.py`: `UnifiedCache` class — main cache implementation
- `src/cacheness/decorators.py`: `@cached`, `@cache_if` function decorators
- `src/cacheness/storage/blob_store.py`: `BlobStore` — standalone blob API

**Configuration:**
- `src/cacheness/config.py`: All dataclass configs, validation, loaders (1253 lines)
- `pyproject.toml`: Package metadata, dependencies, tool config (ruff, pytest, ty)
- `config/local_sqlite_fs.yaml`: Example config file

**Core Logic:**
- `src/cacheness/core.py`: `put()`, `get()`, `invalidate()`, `rotate_key()` (1217 lines)
- `src/cacheness/storage/blob_store.py`: `_write_blob()`, `_read_blob()`, `put()`, `get()` (1199 lines)
- `src/cacheness/interfaces.py`: All type contracts (505 lines)

**Security:**
- `src/cacheness/security.py`: `CacheEntrySigner`, HMAC signing, HKDF derivation (470 lines)
- `src/cacheness/encryption.py`: `encrypt_blob()`, `decrypt_blob()`, AES-256-GCM (81 lines)

**Testing:**
- `tests/conftest.py`: Shared fixtures (`tmp_path`, backend factories, Docker checks)
- `tests/test_core.py`: Core functionality tests (1086 lines)
- `tests/test_backend_parity.py`: Cross-backend parametrized tests (629 lines)
- `tests/test_sqlite_schema_versioning.py`: Schema migration tests (983 lines)

## Naming Conventions

**Files:**
- `_*.py` prefix: Private mixins and internal modules (e.g. `_batch_mixin.py`, `_compat.py`)
- `*.py` without prefix: Public modules (e.g. `core.py`, `config.py`, `encryption.py`)
- `test_*.py`: Test files matching source modules (e.g. `test_core.py`, `test_handlers.py`)

**Directories:**
- Lowercase, underscores: `handlers/`, `metadata/`, `storage/`, `backends/`
- Each package has `__init__.py` re-exporting public names for backward compatibility
- Each package has `_compat.py` for shared internal imports

**Classes:**
- PascalCase: `UnifiedCache`, `BlobStore`, `HandlerRegistry`, `MetadataBackend`
- Handlers: `{Type}Handler` (e.g. `PandasDataFrameHandler`, `ArrayHandler`, `ObjectHandler`)
- Backends: `{Provider}Backend` (e.g. `JsonBackend`, `SqliteBackend`, `PostgresBackend`, `S3BlobBackend`)
- Mixins: `{Concern}Mixin` (e.g. `VerificationMixin`, `BatchMixin`, `InlineBlobMixin`)

**Functions:**
- `snake_case`: `create_metadata_backend()`, `validate_config()`, `encrypt_blob()`
- Private: `_init_*()` for initialization, `_resolve_*()` for path/key resolution
- Test: `test_{feature}_{scenario}` (e.g. `test_put_get_roundtrip`, `test_encryption_rotate_key`)

## Where to Add New Code

**New Handler:**
- Implementation: `src/cacheness/handlers/{type}_handler.py` (implement `CacheHandler` ABC from `interfaces.py`)
- Registration: Add to `src/cacheness/handlers/registry.py` — `_setup_default_handlers()` with priority
- Re-exports: Add to `src/cacheness/handlers/__init__.py`
- Tests: `tests/test_handlers.py` or `tests/test_{type}_handler.py`

**New Metadata Backend:**
- Implementation: `src/cacheness/storage/backends/{provider}_backend.py` (extend `MetadataBackend` from `metadata/base.py`)
- Registration: Add to `src/cacheness/metadata/__init__.py` — `create_metadata_backend()` factory
- Re-exports: Add to `src/cacheness/storage/backends/__init__.py`
- Tests: `tests/test_{provider}_backend.py` + add to `tests/test_backend_parity.py` parametrization

**New Blob Backend:**
- Implementation: `src/cacheness/storage/backends/{provider}_backend.py` (extend `BlobBackend` from `blob_backends.py`)
- Registration: Use `register_blob_backend()` in `blob_backends.py`
- Tests: `tests/test_{provider}_blob_backend.py`

**New Mixin:**
- Implementation: `src/cacheness/_{concern}_mixin.py`
- Integration: Add to `UnifiedCache` bases in `src/cacheness/core.py`
- Tests: Add to `tests/test_core.py` or create `tests/test_{concern}.py`

**New Config Section:**
- Implementation: Add `@dataclass` to `src/cacheness/config.py`
- Integration: Add field to `CacheConfig` dataclass, update `create_cache_config()`
- Tests: `tests/test_config_validation.py`

**Utilities:**
- Shared helpers: `src/cacheness/{utility_name}.py` (e.g. `size_utils.py`, `json_utils.py`)
- Tests: `tests/test_{utility_name}.py`

## Special Directories

**`cache/`:**
- Purpose: Default cache directory for development/testing
- Generated: Yes (at runtime)
- Committed: Partially (`.gitkeep` only)

**`worktrees/`:**
- Purpose: Git worktree container for feature branches
- Generated: Yes (by git worktree commands)
- Committed: No (gitignored)

**`.planning/`:**
- Purpose: GSD project management artifacts (roadmap, phases, codebase docs)
- Generated: Semi-automated by GSD skills
- Committed: Yes

**`logs/`:**
- Purpose: Runtime log files
- Generated: Yes
- Committed: No (gitignored)

---

*Structure analysis: 2026-04-07*
