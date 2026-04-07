# Technology Stack

**Analysis Date:** 2026-04-07

## Languages

**Primary:**
- Python 3.12+ — all source code, tests, benchmarks, examples (`src/cacheness/`, `tests/`, `benchmarks/`, `examples/`)

**Secondary:**
- Shell (POSIX sh) — Docker init scripts (`config/garage-init.sh`), git hooks (`scripts/hooks/pre-commit`)
- PowerShell — development scripts (`scripts/install-hooks.ps1`, `scripts/quality-check.ps1`)
- TOML — project config, tool config (`pyproject.toml`, `config/garage.toml`)
- YAML — test configs (`config/test_config.yaml`, `config/local_sqlite_fs.yaml`)
- JSON — test configs, namespace registry (`config/test_config.json`, `cacheness_namespaces.json`)

## Runtime

**Environment:**
- CPython 3.11+ (minimum), 3.12+ (target/recommended)
- No async runtime — all operations are synchronous with threading locks

**Package Manager:**
- `uv` — sole package manager. Never use `pip`, `python`, or `python -m` directly
- Build backend: `uv_build>=0.10.2,<0.11.0`
- Lockfile: `uv.lock` (committed, deterministic)
- Commands: `uv run pytest ...`, `uv sync --all-groups`, `uv add <pkg>`

## Frameworks

**Core (no web framework):**
- This is a library, not an application — no web framework, no CLI framework
- Pure Python with optional C-extension dependencies for performance

**Testing:**
- `pytest>=8.4.1` — test runner (`tests/`)
- `pytest-xdist>=3.8.0` — parallel test execution (`-n auto --dist loadgroup`)
- `pytest-cov>=6.2.1` — coverage collection
- `pytest-json-report>=1.5.0` — machine-readable test results (`.test-results.json`)
- `hypothesis>=6.151.5` — property-based testing
- `moto[s3]>=5.0.0` — AWS S3 mock for tests

**Linting & Type Checking:**
- `ruff>=0.12.8` — linting + formatting (replaces black, isort, flake8)
- `ty>=0.0.15` — type checking

**Build/Dev:**
- `uv_build>=0.10.2,<0.11.0` — PEP 517 build backend
- `beads-mcp>=0.55.4` — issue tracking (dev-only CLI)

## Key Dependencies

### Core (always installed)

| Package | Version | Purpose | Used in |
|---------|---------|---------|---------|
| `xxhash>=3.5.0` | Required | Fast non-cryptographic hashing for cache keys and file integrity (xxh3_64, xxh64) | `src/cacheness/serialization.py`, `src/cacheness/file_hashing.py`, `src/cacheness/write_intent.py` |
| `cachetools>=6.1.0` | Required | In-memory LRU/LFU/FIFO caches for metadata caching layer | `src/cacheness/metadata/_compat.py`, `src/cacheness/metadata/base.py` |

### Optional — Recommended

| Package | Version | Extra | Purpose | Used in |
|---------|---------|-------|---------|---------|
| `numpy>=2.0.0` | Optional | `recommended` | NumPy array handler (blosc2 compression) | `src/cacheness/handlers/numpy_array.py`, `src/cacheness/compress_pickle.py` |
| `blosc2>=3.5.1` | Optional | `recommended` | High-performance compression for arrays and pickled objects | `src/cacheness/compress_pickle.py`, `src/cacheness/handlers/numpy_array.py` |
| `pandas>=2.0.0,<4.0.0` | Optional | `recommended`, `dataframes` | DataFrame/Series handler (parquet serialization) | `src/cacheness/handlers/pandas_dataframe.py`, `src/cacheness/handlers/pandas_series.py` |
| `pyarrow>=21.0.0` | Optional | `recommended`, `dataframes` | Parquet engine for DataFrame serialization | `src/cacheness/handlers/pandas_dataframe.py` |
| `sqlalchemy>=2.0.0` | Optional | `recommended`, `postgresql` | ORM for SQLite and PostgreSQL metadata backends | `src/cacheness/metadata/sqlite_backend.py`, `src/cacheness/metadata/_compat.py` |
| `duckdb-engine>=0.16.0` | Optional | `recommended` | DuckDB SQLAlchemy engine (experimental) | Registered but not primary |
| `orjson>=3.8.0` | Optional | `recommended` | 2-5x faster JSON serialization (auto-detected) | `src/cacheness/json_utils.py` |
| `dill>=0.4.0` | Optional | `recommended` | Enhanced pickle for lambdas, closures, complex objects | `src/cacheness/handlers/object_handler.py`, `src/cacheness/compress_pickle.py` |

### Optional — Data Formats

| Package | Version | Extra | Purpose | Used in |
|---------|---------|-------|---------|---------|
| `polars>=1.30.0` | Optional | `dataframes` | Polars DataFrame/Series handler | `src/cacheness/handlers/polars_dataframe.py`, `src/cacheness/handlers/polars_series.py` |
| `tensorflow>=2.0.0` | Optional | `tensorflow` | TensorFlow tensor handler | `src/cacheness/handlers/tensorflow_tensor.py` |

### Optional — Cloud & Distributed

| Package | Version | Extra | Purpose | Used in |
|---------|---------|-------|---------|---------|
| `boto3>=1.26.0` | Optional | `s3`, `cloud` | S3-compatible blob backend (AWS, MinIO, Garage) | `src/cacheness/storage/backends/s3_backend.py` |
| `psycopg[binary]>=3.1.0` | Optional | `postgresql`, `cloud` | PostgreSQL adapter for metadata backend | `src/cacheness/storage/backends/postgresql_backend.py` |

### Optional — Security

| Package | Version | Extra | Purpose | Used in |
|---------|---------|-------|---------|---------|
| `cryptography>=41.0.0` | Optional | `encryption` | AES-256-GCM encryption at rest, HKDF key derivation | `src/cacheness/encryption.py` |

**Note:** `cryptography` is also in the `dev` dependency group so tests always have it available.

### Standard Library (notable usage)

| Module | Purpose | Used in |
|--------|---------|---------|
| `hmac`, `hashlib` | HMAC-SHA256 entry signing, HKDF key derivation | `src/cacheness/security.py` |
| `pickle` | Default object serialization | `src/cacheness/compress_pickle.py`, `src/cacheness/handlers/object_handler.py` |
| `threading` | Thread-safety via `threading.Lock` / `RLock` | `src/cacheness/core.py`, `src/cacheness/metadata/sqlite_backend.py` |
| `subprocess` | Windows key file permissions via `icacls` | `src/cacheness/security.py` |
| `secrets` | Cryptographic key generation | `src/cacheness/security.py` |
| `dataclasses` | All config dataclasses | `src/cacheness/config.py`, `src/cacheness/interfaces.py` |
| `typing_extensions` | `TypedDict` for typed metadata contracts | `src/cacheness/interfaces.py` |
| `concurrent.futures` | Parallel file hashing (`ProcessPoolExecutor`) | `src/cacheness/file_hashing.py` |
| `pathlib` | All path operations | Throughout |
| `logging` | Structured logging in every module | Throughout |

## Configuration System

**Dataclass-based configuration** in `src/cacheness/config.py`:

| Config Class | Responsibility |
|-------------|---------------|
| `CacheConfig` | Top-level composite config (holds all sub-configs) |
| `CacheStorageConfig` | Cache dir, max size, cleanup, temp dir, intent threshold |
| `CacheMetadataConfig` | Backend type, TTL, memory cache layer, stats |
| `CacheBlobConfig` | Blob backend type, atomic writes, sharding, inline blob threshold |
| `CompressionConfig` | Parquet codec, blosc2 levels, pickle codec, compression threshold |
| `SerializationConfig` | Path hashing, type validation, recursion limits |
| `HandlerConfig` | Handler priority, per-handler enable/disable flags |
| `SecurityConfig` | Signing, encryption, key fallback policy, HKDF derivation |
| `HooksConfig` | Lifecycle callbacks (`on_evict`, `on_integrity_failure`) |

**Config loading:**
- `create_cache_config(**kwargs)` — flat kwargs (backward-compatible)
- `load_config_from_json(path)` — JSON file (`config/test_config.json`)
- `load_config_from_yaml(path)` — YAML file, requires PyYAML (`config/test_config.yaml`)
- `load_config_from_dict(d)` — dictionary
- `validate_config(config)` / `validate_config_strict(config)` — validation

**Config validation (HARD-01):**
- `SecurityConfig.__post_init__()` validates bad config combos at construction time:
  - Encryption enabled without key file (and no in-memory key)
  - Encryption enabled with signing disabled
  - In-memory key with encryption (data unrecoverable after restart)

## Build & Package

**Package structure:** `src/cacheness/` layout (PEP 621 compliant)
- Build backend: `uv_build`
- No `setup.py` or `setup.cfg` — pure `pyproject.toml`
- Version: `0.6.0` (in `pyproject.toml` and `__init__.py`)

**Install extras:**
```bash
uv add cacheness[recommended]     # NumPy, blosc2, pandas, SQLAlchemy, orjson, dill
uv add cacheness[encryption]      # AES-256-GCM encryption
uv add cacheness[s3]              # S3 blob backend
uv add cacheness[postgresql]      # PostgreSQL metadata backend
uv add cacheness[cloud]           # S3 + PostgreSQL
uv add cacheness[dataframes]      # pandas + polars + pyarrow
uv add cacheness[tensorflow]      # TensorFlow tensor handler
```

**Quality gates (2-phase):**
```bash
# Phase 1 — auto-fix (never fails)
uv run ruff format . && uv run ruff check --fix .
# Phase 2 — validate (may fail)
uv run ruff check . && uv run ty check
```

**Test suite:**
```bash
uv run pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py
# Parse results:
(Get-Content .test-results.json | ConvertFrom-Json).summary
```

**Ruff config:** Target Python 3.12, line length 88, ignores B008 and C901. Per-file overrides for tests, benchmarks, examples.

## Platform Requirements

**Development:**
- Python 3.12+
- `uv` package manager
- Docker (for PostgreSQL + Garage S3 integration tests)
- Windows, macOS, or Linux

**Production:**
- Python 3.11+ (minimum)
- Filesystem access for blob storage (or S3/compatible)
- Optional: PostgreSQL for distributed metadata
- Optional: `cryptography` package for encryption at rest

**Windows-specific:**
- TensorFlow handler tests skipped (hang on Windows)
- Key file permissions set via `icacls` instead of `os.chmod` (`src/cacheness/security.py`)
- ASCII-only git commit messages (Unicode causes terminal hangs)

---

*Stack analysis: 2026-04-07*
