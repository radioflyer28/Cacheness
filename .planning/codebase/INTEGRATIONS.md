# External Integrations

**Analysis Date:** 2026-04-02

## Metadata Backends

Cacheness supports three pluggable metadata backends, selected at cache creation time. All implement `MetadataBackend` ABC defined in `src/cacheness/metadata.py`.

**JSON Backend (built-in, no dependencies):**
- Implementation: `src/cacheness/metadata.py` → `JsonBackend`
- Storage: single JSON file per cache directory
- Use case: <200 entries, development, NOT safe for concurrent access
- Thread safety: file-level locking via `threading.Lock`

**SQLite Backend (requires `sqlalchemy`):**
- Implementation: `src/cacheness/metadata.py` → `SqliteBackend`
- Storage: SQLite database file (e.g., `metadata.db`)
- Use case: 200+ entries, production, multi-process safe
- ORM: SQLAlchemy >=2.0 with declarative models
- Custom metadata: SQLAlchemy models via `src/cacheness/custom_metadata.py` (`@custom_metadata_model` decorator)
- Schema versioning: built-in migration system
- Config example: `config/local_sqlite_fs.yaml`

**PostgreSQL Backend (requires `psycopg`, `sqlalchemy`):**
- Implementation: `src/cacheness/storage/backends/postgresql_backend.py` → `PostgresBackend`
- Connection: SQLAlchemy `create_engine()` with connection pooling
- Use case: distributed teams, multi-server caching
- Features: SSL/TLS support, configurable pool size, automatic table creation with indexes
- Config example: `config/test_config.yaml` → `metadata.metadata_backend_options.connection_url`

**Backend registry:**
- Registration: `register_metadata_backend()` in `src/cacheness/storage/backends/__init__.py`
- Factory: `get_metadata_backend(name, **kwargs)` / `create_metadata_backend()`
- Listing: `list_metadata_backends()`

## Blob Storage Backends

Blob backends handle raw data storage, separate from metadata. All implement `BlobBackend` ABC defined in `src/cacheness/storage/backends/blob_backends.py`.

**Filesystem Backend (built-in, default):**
- Implementation: `src/cacheness/storage/backends/blob_backends.py` → `FilesystemBlobBackend`
- Storage: local filesystem with configurable directory sharding (`shard_chars`)
- Use case: single-machine caching, development

**In-Memory Backend (built-in):**
- Implementation: `src/cacheness/storage/backends/blob_backends.py` → `InMemoryBlobBackend`
- Use case: testing only

**S3-Compatible Backend (requires `boto3`):**
- Implementation: `src/cacheness/storage/backends/s3_backend.py` → `S3BlobBackend`
- Supports: Amazon S3, Garage, MinIO, any S3-compatible service
- Features: namespace isolation via key prefixes, integrity verification (`S3IntegrityError`), configurable endpoint/region/SSL
- Auth: `aws_access_key_id`, `aws_secret_access_key` passed via config or environment
- Config example: `config/test_config.yaml` → `blob.blob_backend_options`

**Backend registry:**
- Registration: `register_blob_backend()` in `src/cacheness/storage/backends/blob_backends.py`
- Factory: `get_blob_backend(name, **kwargs)`
- Listing: `list_blob_backends()`

## Docker / Container Setup

**Docker Compose** (`docker-compose.yml`) provides two services for local development and integration testing:

**PostgreSQL:**
- Image: `postgres:16-alpine`
- Container: `cacheness-postgres`
- Port: `5432:5432`
- Database: `cacheness_test`
- Healthcheck: `pg_isready -U cacheness`
- Volume: `postgres_data` (persistent)
- Network: `cacheness-network` (bridge)

**Garage (S3-compatible object storage):**
- Base image: `dxflrs/garage:v1.3.1` (multi-stage via `config/Dockerfile.garage`)
- Runtime image: `alpine:3.19` with openssl
- Container: `cacheness-garage`
- Ports: `3900` (S3 API), `3903` (Admin API)
- Init script: `config/garage-init.sh` — auto-configures layout, creates API keys, provisions buckets (`cache-bucket`, `test-bucket`)
- Config: `config/garage.toml` — SQLite metadata engine, single-node replication, region `us-east-1`
- Volumes: `garage_meta`, `garage_data` (persistent)
- Network: `cacheness-network` (bridge)

**Setup tooling:**
- `Makefile` — `make up`, `make down`, `make clean`, `make logs`
- `scripts/setup_local_env.py` — automated environment setup script
- `scripts/setup_local_env.bat` — Windows batch setup

## S3-Compatible Cloud Storage

**Integration points:**
- Backend: `src/cacheness/storage/backends/s3_backend.py`
- Client: `boto3` SDK
- Tested with: Garage (local dev via Docker), moto (unit tests via `mock_aws`)
- Config keys: `bucket`, `endpoint_url`, `aws_access_key_id`, `aws_secret_access_key`, `region_name`
- Features: content-addressable storage, namespace isolation via key prefixes, SHA-256 integrity checks

**Test infrastructure:**
- Unit tests: `moto[s3]` mock (no real S3 needed) — `tests/conftest.py` → `mock_aws` fixture
- Integration tests: Docker Garage container — `tests/conftest.py` → `get_s3_config()`
- Docker group: tests using real S3/PostgreSQL are grouped via `@pytest.mark.xdist_group("docker")`

## PostgreSQL Integration

**Integration points:**
- Backend: `src/cacheness/storage/backends/postgresql_backend.py`
- Client: `psycopg` (psycopg3) via SQLAlchemy engine
- Connection string format: `postgresql+psycopg://user:pass@host:port/db`

**Test infrastructure:**
- Integration tests: Docker PostgreSQL container
- Connection config: `tests/conftest.py` → `get_postgres_url()` reads from env vars with fallback defaults
- Environment variables: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`

## Data Serialization Integrations

**Type-aware handlers** in `src/cacheness/handlers.py` and `src/cacheness/storage/handlers/__init__.py`:

| Handler | Data Type | Serialization Format | Library |
|---------|-----------|---------------------|---------|
| `ArrayHandler` | NumPy arrays | blosc2 compressed tensor | `numpy`, `blosc2` |
| `PandasDataFrameHandler` | Pandas DataFrames | Parquet via PyArrow | `pandas`, `pyarrow` |
| `PandasSeriesHandler` | Pandas Series | Parquet via PyArrow | `pandas`, `pyarrow` |
| `PolarsDataFrameHandler` | Polars DataFrames | Parquet | `polars`, `pyarrow` |
| `BytesHandler` | bytes/bytearray/memoryview | Raw (no serialization) | stdlib |
| `ObjectHandler` | Generic Python objects | pickle/dill + blosc2 compression | `pickle`, `dill`, `blosc2` |
| `TensorHandler` | TensorFlow tensors | TF-native format | `tensorflow` |

**Handler registry:** `HandlerRegistry` in `src/cacheness/handlers.py` — auto-detects type and dispatches to appropriate handler.

## Compression Integrations

Compression support in `src/cacheness/compress_pickle.py` and `src/cacheness/storage/compression.py`:

| Codec | Library | Notes |
|-------|---------|-------|
| blosc2/blosclz | `blosc2` | Default, fastest for arrays |
| lz4 | `blosc2` | Very fast |
| lz4hc | `blosc2` | High compression variant |
| zstd | `blosc2` | Excellent ratio |
| zlib | `blosc2` | Standard |
| gzip | stdlib | Fallback |
| snappy | `blosc2` | Google's fast compressor |

## Security Integrations

**Cryptographic signing** (`src/cacheness/security.py`):
- Algorithm: HMAC-SHA256
- Purpose: cache entry integrity verification (tamper detection)
- Key management: auto-generated key files, key rotation support
- Signature versioning: version-based signed field lists for safe schema evolution
- Uses stdlib `hmac`, `hashlib`, `secrets`

**File integrity:**
- xxhash-based file hashing (`src/cacheness/file_hashing.py`)
- Parallel directory hashing via `ProcessPoolExecutor`

## Authentication & Identity

- No auth provider — Cacheness is a library, not a service
- S3 auth: AWS credential pairs passed via config
- PostgreSQL auth: connection string credentials
- Cache signing keys: local binary files (auto-generated)

## CI/CD & Deployment

**CI Pipeline:**
- No CI config files detected in the repository (no `.github/workflows/`, no `.gitlab-ci.yml`)
- Quality gates run locally via pre-commit hooks and manual scripts

**Deployment:**
- Distributed as a Python package via `uv_build`
- pip-installable: `pip install cacheness` or `pip install cacheness[cloud]`

## Monitoring & Observability

**Logging:**
- Framework: Python stdlib `logging`
- All modules create module-level loggers: `logger = logging.getLogger(__name__)`
- Test log level: WARNING (configurable via `--log-cli-level`)

**Error tracking:**
- Custom exception hierarchy in `src/cacheness/error_handling.py`: `CacheError` base with context dict
- Handler-specific errors: `CacheWriteError`, `CacheReadError`, `CacheFormatError` in `src/cacheness/interfaces.py`
- S3 integrity errors: `S3IntegrityError` in `src/cacheness/storage/backends/s3_backend.py`

**Metrics:**
- Built-in cache statistics: `cache.get_stats()` — hit/miss rates, entry counts, size
- Integrity reports: `IntegrityReport` dataclass in `src/cacheness/interfaces.py`

## Environment Configuration

**Required env vars:** None (all have sensible defaults)

**Optional env vars for integration testing:**
- `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB` — PostgreSQL connection
- `S3_ENDPOINT_URL`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET`, `S3_REGION` — S3 connection

**Config files:**
- `config/test_config.yaml` — PostgreSQL + S3 Garage (full integration)
- `config/test_config.json` — same as YAML in JSON format
- `config/local_sqlite_fs.yaml` — SQLite + filesystem (no Docker needed)
- `config/garage.toml` — Garage S3 server configuration
- `config/Dockerfile.garage` — multi-stage Dockerfile for Garage container

## Webhooks & Callbacks

**Incoming:** None

**Outgoing:** None

**Hook system:**
- `HooksConfig` in `src/cacheness/config.py` — internal lifecycle hooks (not external webhooks)
- Git hooks: `scripts/hooks/` — pre-commit quality checks, bd sync

---

*Integration audit: 2026-04-02*
