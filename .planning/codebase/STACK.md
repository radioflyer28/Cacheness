# Technology Stack

**Analysis Date:** 2026-08-29

## Languages

**Primary:**
- Python 3.11+ (the project requires `>=3.11`; the repository pins Python 3.13 in `.python-version`) - library implementation, tests, examples, and benchmarks in `src/cacheness/`, `tests/`, `examples/`, and `benchmarks/`

**Secondary:**
- SQL (generated and dialect-specific SQLAlchemy expressions, plus SQLite pragmas) - relational metadata and pull-through cache operations in `src/cacheness/metadata.py`, `src/cacheness/sql_cache.py`, and `src/cacheness/storage/backends/postgresql_backend.py`

## Runtime

**Environment:**
- CPython 3.13 for the checked-in development environment; supported runtime range is Python `>=3.11` (`.python-version`, `pyproject.toml`)

**Package Manager:**
- `uv` - dependency resolution, environment management, and lockfile workflow (`uv.lock`)
- Lockfile: present (`uv.lock`, lock format revision 2)

## Frameworks

**Core:**
- No application framework - `cacheness` is a framework-independent Python caching library (`src/cacheness/__init__.py`)
- SQLAlchemy 2.0 - optional ORM/engine layer for SQLite, DuckDB, PostgreSQL, and custom metadata (`src/cacheness/metadata.py`, `src/cacheness/sql_cache.py`)

**Testing:**
- pytest 8.4.1 (locked) - test discovery and execution (`pyproject.toml`, `tests/`)
- pytest-cov 6.2.1 (locked) - coverage reporting (`pyproject.toml`)
- moto 5.1.20 (locked in the development environment) - AWS S3 mocking for `tests/test_s3_blob_backend.py`

**Build/Dev:**
- `uv_build` (`>=0.9.0,<1.0.0`) - PEP 517 build backend (`pyproject.toml`)
- Ruff `0.12.9` (locked) - lint/format tooling; target Python version is `py312`, line length is 88 (`pyproject.toml`)
- `verify_platform.py` - repository-level compatibility smoke checks

## Key Dependencies

**Critical:**
- `xxhash>=3.5.0` (locked 3.5.0) - XXH3 cache-key and file-content hashing in `src/cacheness/serialization.py`, `src/cacheness/file_hashing.py`, and `src/cacheness/core.py`
- `cachetools>=6.1.0` (locked 6.1.0) - optional TTL/LRU-style in-memory metadata layer in `src/cacheness/metadata.py`; the code falls back when it is unavailable

**Infrastructure:**
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

The optional dependency groups are declared in `pyproject.toml`: `recommended`, `dataframes`, `tensorflow`, `s3`, `postgresql`, and `cloud`. Core runtime dependencies are only `cachetools` and `xxhash`, although NumPy is imported eagerly by core handler modules.

## Configuration

**Environment:**
- Runtime configuration is passed through `CacheConfig` and its sub-configurations in `src/cacheness/config.py`; no environment-variable parser is implemented in `src/cacheness/`
- JSON configuration can be loaded/saved with `load_config_from_json()` and `save_config_to_json()`; YAML support is lazy and requires separately installed PyYAML (`src/cacheness/config.py`)
- Main defaults are local `./cache` blobs, auto-selected SQLite metadata when SQLAlchemy works (JSON fallback otherwise), 24-hour TTL, LZ4 Parquet, Zstandard pickle compression, and HMAC entry signing (`src/cacheness/config.py`, `src/cacheness/core.py`)
- `.env`/credential files are not detected; generated cache databases and serialized artifacts are excluded by `.gitignore`

**Build:**
- `pyproject.toml` - package metadata, dependency extras, pytest settings, coverage settings, Ruff settings, and `uv_build` backend
- `.python-version` - development interpreter pin (`3.13`)
- `uv.lock` - resolved dependency graph and platform-specific wheels

## Platform Requirements

**Development:**
- Python 3.11 or newer, `uv`, and a writable local filesystem; optional features require their corresponding extras (`pyproject.toml`)
- SQL-backed features require SQLAlchemy; DuckDB requires `duckdb-engine`; PostgreSQL requires a reachable PostgreSQL server and `psycopg`; S3 requires AWS-compatible credentials/configuration and `boto3`

**Production:**
- Deploy as a Python package in the host process; the library has no web server, worker runtime, container definition, or hosting manifest
- Use a writable filesystem for default blobs and SQLite metadata, or provide PostgreSQL metadata/S3 blobs through the pluggable backend APIs (`src/cacheness/storage/backends/`)

---

*Stack analysis: 2026-08-29*
