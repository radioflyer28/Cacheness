# Technology Stack

**Analysis Date:** 2026-04-02

## Languages

**Primary:**
- Python >=3.11 (target version 3.12 for linting) — all source, tests, benchmarks, and scripts

**Secondary:**
- Shell (POSIX sh) — `config/garage-init.sh`, `scripts/install-hooks.sh`, `scripts/quality-check.sh`
- PowerShell — `scripts/install-hooks.ps1`, `scripts/quality-check.ps1`
- Dockerfile — `config/Dockerfile.garage`
- TOML/YAML/JSON — configuration files

## Runtime

**Environment:**
- CPython >=3.11 (linting targets 3.12)
- No `.python-version` file; version constraint defined in `pyproject.toml`

**Package Manager:**
- **uv** — sole package manager; lockfile managed via `uv.lock`
- Build backend: `uv_build>=0.10.2,<0.11.0` (declared in `pyproject.toml` `[build-system]`)
- Commands: `uv run`, `uv sync`, `uv add`

## Frameworks & Libraries

**Core (required):**
| Package | Version | Purpose |
|---------|---------|---------|
| `cachetools` | >=6.1.0 | In-memory cache primitives |
| `xxhash` | >=3.5.0 | Fast non-cryptographic hashing for cache keys & file integrity |

**Recommended (optional extras):**
| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | >=2.0.0 | NumPy array handler — blosc2 tensor storage |
| `blosc2` | >=3.5.1 | High-performance compression (lz4, zstd, blosclz, etc.) |
| `pandas` | >=2.0.0,<4.0.0 | DataFrame handler — Parquet serialization |
| `pyarrow` | >=21.0.0 | Parquet I/O engine for DataFrames |
| `sqlalchemy` | >=2.0.0 | ORM for SQLite and PostgreSQL metadata backends |
| `duckdb-engine` | >=0.16.0 | DuckDB SQLAlchemy dialect |
| `orjson` | >=3.8.0 | High-performance JSON serialization (2-5× stdlib) |
| `dill` | >=0.4.0 | Extended pickle for lambdas, closures, classes |

**DataFrame extras:**
| Package | Version | Purpose |
|---------|---------|---------|
| `polars` | >=1.30.0 | Polars DataFrame handler |

**S3 extras:**
| Package | Version | Purpose |
|---------|---------|---------|
| `boto3` | >=1.26.0 | S3-compatible blob storage backend |

**PostgreSQL extras:**
| Package | Version | Purpose |
|---------|---------|---------|
| `psycopg[binary]` | >=3.1.0 | PostgreSQL adapter (psycopg3) |
| `sqlalchemy` | >=2.0.0 | ORM layer for PostgreSQL backend |

**TensorFlow extras:**
| Package | Version | Purpose |
|---------|---------|---------|
| `tensorflow` | >=2.0.0 | TensorFlow tensor handler |

**Cloud bundle** (`cacheness[cloud]`):
- Combines `boto3`, `psycopg[binary]`, and `sqlalchemy`

## Dev Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | >=8.4.1 | Test runner |
| `pytest-cov` | >=6.2.1 | Coverage reporting |
| `pytest-xdist` | >=3.8.0 | Parallel test execution (`-n auto --dist loadgroup`) |
| `ruff` | >=0.12.8 | Linter and formatter (replaces flake8, isort, black) |
| `ty` | >=0.0.15 | Type checker |
| `moto[s3]` | >=5.0.0 | AWS S3 mocking for tests |
| `hypothesis` | >=6.151.5 | Property-based testing |
| `beads-mcp` | >=0.55.4 | Issue tracking CLI (`bd create`, `bd close`, etc.) |

## Build & Packaging

**Build system:**
- Backend: `uv_build` (declared in `pyproject.toml` `[build-system]`)
- Source layout: `src/cacheness/` (src-layout)
- Package name: `cacheness`, version `0.6.0`

**Optional dependency groups** (pip extras): `recommended`, `dataframes`, `tensorflow`, `s3`, `postgresql`, `cloud`

**Dependency groups** (uv groups, not pip-installable): `recommended`, `sql`, `dataframes`, `dev`, `tensorflow`, `s3`, `postgresql`, `cloud`

## Standard Library Usage

Key stdlib modules used across the codebase:
- `threading` — thread-safe cache operations (`src/cacheness/core.py`, `src/cacheness/metadata.py`)
- `hashlib`, `hmac`, `secrets` — HMAC-SHA256 cryptographic signing (`src/cacheness/security.py`)
- `pickle` — default object serialization (`src/cacheness/compress_pickle.py`)
- `logging` — structured logging throughout all modules
- `pathlib.Path` — all file path operations
- `dataclasses` — configuration and data structures (`src/cacheness/config.py`, `src/cacheness/interfaces.py`)
- `abc` — abstract base classes for backends and handlers
- `inspect` — function signature normalization for decorator cache keys (`src/cacheness/core.py`)
- `concurrent.futures` — parallel file hashing via `ProcessPoolExecutor` (`src/cacheness/file_hashing.py`)
- `typing` / `typing_extensions` — type annotations (`TypedDict` for signable fields)
- `uuid` — unique identifier generation
- `re` — regex-based namespace validation, size/duration parsing
- `contextlib` — context managers for error handling (`src/cacheness/error_handling.py`)
- `functools` — decorator composition (`src/cacheness/decorators.py`)
- `weakref` — weak references for decorator cache cleanup
- `atexit` — automatic cleanup of decorator-created caches
- `ast` — AST parsing in handler utilities

## Configuration

**Project configuration:**
- `pyproject.toml` — package metadata, dependencies, tool configs (ruff, pytest, coverage, ty)

**Cache configuration formats:**
- YAML: `config/test_config.yaml`, `config/local_sqlite_fs.yaml` — human-friendly config
- JSON: `config/test_config.json` — machine-parseable config
- Python dataclasses: `src/cacheness/config.py` — `CacheConfig`, `CacheStorageConfig`, `CacheMetadataConfig`, `CacheBlobConfig`, `CompressionConfig`, `SerializationConfig`, `HandlerConfig`, `HooksConfig`, `SecurityConfig`
- Programmatic: `load_config_from_yaml()`, `load_config_from_json()`, `load_config_from_dict()`, `create_cache_config()`

**Tool configuration (all in `pyproject.toml`):**
- `[tool.pytest.ini_options]` — pytest markers, parallel execution, logging
- `[tool.ruff]` — linting rules, per-file ignores, line length 88
- `[tool.ty.src]` — type checker exclusions
- `[tool.coverage.run]` / `[tool.coverage.report]` — coverage settings

**Infrastructure configuration:**
- `docker-compose.yml` — PostgreSQL and Garage containers
- `config/garage.toml` — Garage S3 server config
- `config/Dockerfile.garage` — multi-stage Dockerfile for Garage
- `Makefile` — development task runners

## Code Quality

**Linting & formatting:**
- Ruff (format + lint) — line length 88, Python 3.12 target
- Two-phase workflow: Phase 1 auto-fix (`ruff format` + `ruff check --fix`), Phase 2 validate (`ruff check` + `ty check`)
- Pre-commit hooks: `scripts/install-hooks.ps1` / `scripts/install-hooks.sh`

**Testing:**
- pytest with xdist parallelism (`-n auto --dist loadgroup`)
- Hypothesis for property-based tests
- moto for S3 mocking
- Docker-based integration tests (PostgreSQL, Garage S3)
- Baseline: 1427 passed, 102 skipped

---

*Stack analysis: 2026-04-02*
