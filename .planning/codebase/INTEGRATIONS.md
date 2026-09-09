# External Integrations

**Analysis Date:** 2026-08-29

**Independent Review:** 2026-08-29 — integration claims were checked against the actual construction and call paths, not only the registered classes

## APIs & External Services

**Object storage:**
- Amazon S3 and S3-compatible services (including MinIO) - optional remote blob storage with direct put/get/delete, HEAD checks, ETag inspection, and streaming uploads/downloads
  - SDK/Client: `boto3` (`src/cacheness/storage/backends/s3_backend.py`)
  - Auth: Explicit `access_key`/`secret_key` options or boto3's standard credential chain; the library does not read a project-specific secret variable
  - Registration: Import `S3BlobBackend` and register it with `register_blob_backend("s3", S3BlobBackend)`; filesystem and memory are the built-in blob backends (`src/cacheness/storage/backends/__init__.py`, `src/cacheness/storage/backends/blob_backends.py`)

**Caller-provided APIs/data sources:**
- HTTP APIs, market-data services, or other upstream sources - not called by the library core; `SqlCacheAdapter.fetch_data()` is the integration seam used to retrieve missing ranges (`src/cacheness/sql_cache.py`)
  - SDK/Client: Caller-selected client (examples use `requests` and `yfinance` in `examples/api_request_caching.py` and `examples/stock_cache_example.py`)
  - Auth: Caller-managed; no API client or API-key environment variable is defined by the package
- S3 file downloads in the example - demonstrated with direct `boto3` calls and ETag-based invalidation, separate from the reusable S3 blob backend (`examples/s3_caching.py`)

## Data Storage

**Databases:**
- SQLite - default persistent metadata backend when SQLAlchemy is available; stores cache entries/statistics and custom metadata in `cache_metadata.db` under the cache directory (`src/cacheness/core.py`, `src/cacheness/metadata.py`)
  - Connection: Local path configured by `CacheMetadataConfig.sqlite_db_file`; `sqlite:///:memory:` is supported for ephemeral SQL caches
  - Client: Python `sqlite3` through SQLAlchemy 2.0 ORM/engine
- DuckDB - SQL pull-through cache backend optimized for analytical/time-series workloads (`src/cacheness/sql_cache.py`)
  - Connection: `duckdb:///...` URL or a simple database path passed to `SqlCache.with_duckdb()`
  - Client: SQLAlchemy 2.0 plus `duckdb-engine`
- PostgreSQL - optional distributed metadata backend and SQL pull-through backend (`src/cacheness/storage/backends/postgresql_backend.py`, `src/cacheness/sql_cache.py`)
  - Connection: Caller-supplied SQLAlchemy `connection_url`/database URL, commonly `postgresql://...`; configuration is passed through `metadata_backend_options`
  - Client: SQLAlchemy 2.0 plus `psycopg` (psycopg2 is also detected for compatibility)
- MySQL - accepted as a generic SQLAlchemy URL by `SqlCache`, with generic insert fallback; no dedicated backend module or package extra is present (`src/cacheness/sql_cache.py`)

**File Storage:**
- Local filesystem - default blob backend, using cache files with Git-style sharding and atomic-write support (`src/cacheness/storage/backends/blob_backends.py`, `src/cacheness/storage/blob_store.py`)
- Amazon S3/MinIO - optional via `S3BlobBackend`; S3 object paths are returned as `s3://bucket/key` URIs and support 5 MB+ multipart-capable streaming through boto3 (`src/cacheness/storage/backends/s3_backend.py`)
- In-memory blobs - built-in ephemeral backend for tests and short-lived processes (`src/cacheness/storage/backends/blob_backends.py`)

**Caching:**
- In-process metadata cache - optional `cachetools.TTLCache` layer, controlled by `CacheMetadataConfig.enable_memory_cache`; no external Redis/Memcached integration is implemented (`src/cacheness/metadata.py`)

## Integration Wiring Status

| Integration surface | Implementation exists | Reached by `UnifiedCache` | Independent assessment |
|---|---:|---:|---|
| JSON / memory / SQLite metadata | Yes | Yes | Selected directly in `UnifiedCache._init_metadata_backend()` (`src/cacheness/core.py:109-212`) |
| PostgreSQL metadata | Yes | Yes, by the literal `postgresql` config value | Requires SQLAlchemy, a driver, and `metadata_backend_options.connection_url` |
| Registered custom metadata backend | Yes | No | `register_metadata_backend()` populates a registry, but `UnifiedCache` never queries it; unknown names fall into auto SQLite/JSON selection (`src/cacheness/storage/backends/__init__.py:127-266`, `src/cacheness/core.py:181-211`) |
| Injected metadata backend instance | API exists | No in normal configurations | A supplied instance is assigned and then immediately overwritten by config-based selection; a runtime probe with `InMemoryBackend` produced `SqliteBackend False sqlite` (`src/cacheness/core.py:114-212`) |
| Filesystem / memory blob registry | Yes | No | The blob registry is tested independently but is not used by `UnifiedCache` or `BlobStore` payload writes |
| S3 blob backend | Yes | No | `S3BlobBackend` is usable directly after explicit registration, but `blob_backend="s3"` does not route main-cache or `BlobStore` payloads to it |
| SQL pull-through cache | Yes | Separate subsystem | `SqlCache` talks to SQLAlchemy and caller adapters directly; it does not share `UnifiedCache` metadata or payload lifecycle |

This distinction is operationally important: the repository contains more integration implementations than the public high-level cache actually composes. When documenting a backend as supported, specify whether it is direct-use only, registry-constructible, or end-to-end wired into `UnifiedCache`.

## Authentication & Identity

**Auth Provider:**
- None - `cacheness` has no user identity, login, OAuth, or authorization subsystem
  - Implementation: S3 credentials are delegated to boto3 (explicit options or its standard chain); PostgreSQL credentials are embedded in the caller-provided SQLAlchemy URL or handled by the selected driver (`src/cacheness/storage/backends/s3_backend.py`, `src/cacheness/storage/backends/postgresql_backend.py`)

## Monitoring & Observability

**Error Tracking:**
- None detected - no Sentry, OpenTelemetry, hosted error tracker, or metrics exporter is configured

**Logs:**
- Standard-library `logging` with module loggers throughout `src/cacheness/`; pytest enables CLI INFO logging through `pyproject.toml`
- Cache statistics (hits, misses, size, hit rate) are stored by metadata backends and exposed through `UnifiedCache.get_stats()` (`src/cacheness/core.py`, `src/cacheness/metadata.py`)
- S3 operations log through the module logger and translate missing objects to `FileNotFoundError` (`src/cacheness/storage/backends/s3_backend.py`)

## CI/CD & Deployment

**Hosting:**
- None configured in the repository; this is a distributable Python package built by `uv_build` (`pyproject.toml`)

**CI Pipeline:**
- None detected (no `.github/workflows/`, GitLab CI file, Dockerfile, or deployment manifest)
- Cross-platform and packaging guidance is documentation-only in `docs/CROSS_PLATFORM_GUIDE.md`; local verification uses `python verify_platform.py`

## Environment Configuration

**Required env vars:**
- None required by the library itself; configuration is passed as Python objects/options
- `CACHENESS_TEST_POSTGRES_URL` is read only by PostgreSQL integration tests to locate an optional test server (`tests/test_postgresql_backend.py`)
- AWS credentials/region may be supplied through boto3's conventional `AWS_*` environment variables when using S3, but `cacheness` does not read them directly (`src/cacheness/storage/backends/s3_backend.py`)

**Secrets location:**
- Caller-managed process environment, boto3 credential chain/profile, or database connection URL; no project secret store or secret-management service is integrated
- Cache signing keys are generated/managed locally by `src/cacheness/security.py` according to `SecurityConfig`; the default key filename is `cache_signing_key.bin` and is excluded from version control by `.gitignore`

## Webhooks & Callbacks

**Incoming:**
- None - no HTTP server or webhook endpoint is implemented

**Outgoing:**
- None from library code - the package does not make general HTTP requests; upstream fetches are supplied by caller adapters/decorated functions (`src/cacheness/sql_cache.py`, `src/cacheness/decorators.py`)

---

*Integration audit: 2026-08-29*
