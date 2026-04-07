# External Integrations

**Analysis Date:** 2026-04-07

## APIs & External Services

**No external API calls at runtime.** Cacheness is a local-first caching library. External services (S3, PostgreSQL) are optional backends configured by the user.

## Data Storage

### Blob Storage Backends

All blob backends implement `BlobBackend` ABC from `src/cacheness/storage/backends/blob_backends.py`.

**Filesystem (default):**
- Implementation: `FilesystemBlobBackend` in `src/cacheness/storage/backends/blob_backends.py`
- Write path: `BlobStore._write_blob()` in `src/cacheness/storage/blob_store.py`
- Features: atomic writes (temp file + rename), git-style directory sharding (configurable `shard_chars`, default 2)
- Storage: local filesystem under `cache_dir` (default `./cache/default/`)

**In-Memory:**
- Implementation: `MemoryBlobBackend` in `src/cacheness/storage/backends/blob_backends.py`
- Purpose: testing and ephemeral caches
- Storage: Python dict in process memory

**S3-Compatible:**
- Implementation: `S3BlobBackend` in `src/cacheness/storage/backends/s3_backend.py`
- SDK: `boto3>=1.26.0` (optional dependency, `cacheness[s3]`)
- Supports: Amazon S3, MinIO, Garage (S3-compatible)
- Auth: `aws_access_key_id` + `aws_secret_access_key` via backend options (never hardcoded)
- Features: namespace-prefixed keys, integrity verification via S3 MD5 ETags
- Config option: `blob_backend_options.endpoint_url` for non-AWS endpoints
- Registration: `register_blob_backend("s3", S3BlobBackend)`

**Inline Blob Storage:**
- Implementation: `InlineBlobMixin` in `src/cacheness/_inline_blob_mixin.py`
- Purpose: store small blobs directly in metadata DB (avoids filesystem I/O)
- Config: `CacheBlobConfig.max_inline_size` (0 = disabled, recommended 4000 for SQLite, 2000 for PostgreSQL)
- Stored as: base64-encoded column in metadata entry
- Encryption: inline blobs are encrypted when content encryption is enabled

### Metadata Backends

All metadata backends implement `MetadataBackend` ABC from `src/cacheness/metadata/base.py`.

**JSON Backend (simple, dev-only):**
- Implementation: `JsonBackend` in `src/cacheness/metadata/json_backend.py`
- Storage: single JSON file per namespace (`cache_metadata.json` / `{namespace}_metadata.json`)
- Thread safety: `threading.Lock` per instance
- Limitations: NOT safe for multi-process concurrency, O(n) scans, no schema migration
- Best for: <200 entries, single-process use, development

**SQLite Backend (production default):**
- Implementation: `SqliteBackend` in `src/cacheness/metadata/sqlite_backend.py`
- ORM: SQLAlchemy 2.0+ with declarative models
- Database file: configurable `sqlite_db_file` (default `cache_metadata.db`)
- Thread safety: `threading.Lock` + SQLite WAL mode journal
- Schema versioning: migration system with version tracking per namespace
  - v1 → v2: partial index on `metadata_dict IS NOT NULL`
  - v2 → v3: `cacheness_version` column
  - v3 → v4: `encryption_algorithm`, `encryption_iv`, `cacheness_version` columns (idempotent via `PRAGMA table_info`)
- Migrations defined in: `src/cacheness/metadata/sqlite_backend.py` (`get_migrations()`)
- Features: `query_meta()` fast path, `keys_by_prefix()` via SQL LIKE, batch operations
- Best for: 200+ entries, production, multi-process

**PostgreSQL Backend (distributed):**
- Implementation: `PostgresBackend` in `src/cacheness/storage/backends/postgresql_backend.py`
- ORM: SQLAlchemy 2.0+ with psycopg3 adapter
- Connection: `postgresql+psycopg://` URL via `metadata_backend_options.connection_url`
- Adapter: `psycopg[binary]>=3.1.0` (optional dependency, `cacheness[postgresql]`)
- Thread safety: SQLAlchemy connection pooling
- Schema versioning: same v3→v4 migration as SQLite (encryption columns)
- Migrations defined in: `src/cacheness/storage/backends/postgresql_backend.py` (`get_migrations()`)
- Features: connection pooling, SSL/TLS support, optimized indexes
- Best for: distributed teams, multi-server deployments

**ORM Models** shared across SQLite and PostgreSQL (`src/cacheness/metadata/_compat.py`):
- `CacheEntryMixin` — per-namespace table with columns: `id`, `cache_key`, `file_path`, `data_type`, `file_hash`, `file_size_bytes`, `description`, `metadata_dict`, `signature`, `signed_fields_version`, `encryption_algorithm`, `encryption_iv`, `cacheness_version`, `inline_blob`, timestamps
- `CacheStats` — cache hit/miss statistics
- `CacheNamespace` — namespace registry with metadata
- Dynamic table creation via `_get_namespace_models(namespace_id)`

### Caching Layer (in-memory)

- Implementation: `CachedMetadataBackend` wrapper in `src/cacheness/metadata/base.py`
- Uses: `cachetools` (LRU, LFU, FIFO, or random replacement)
- Config: `CacheMetadataConfig.enable_memory_cache`, `memory_cache_type`, `memory_cache_maxsize`, `memory_cache_ttl_seconds`
- Purpose: reduces disk I/O for metadata lookups (sits between application and disk backend)

## Authentication & Security

### Entry Signing (HMAC-SHA256)

- Implementation: `CacheEntrySigner` in `src/cacheness/security.py`
- Algorithm: HMAC-SHA256 with version-based signed field lists
- Key file: configurable `signing_key_file` (default `cache_signing_key.bin`)
- Key generation: `secrets.token_bytes(32)` via `src/cacheness/security.py`
- Key derivation: HKDF-SHA256 per-namespace keys (`_hkdf_sha256()` in `src/cacheness/security.py`)
  - Info string: `b"cacheness-hmac-v1:" + namespace_id.encode()`
- Signed fields (v2): `cache_key`, `data_type`, `file_hash`, `file_size_bytes`, `file_path`, `description`
- Verification: on every `get()` call when `enable_entry_signing=True`
- Key fallback policy: `"raise"` | `"warn"` | `"fallback"` — configurable via `SecurityConfig.key_fallback_policy`
- Key rotation: `UnifiedCache.rotate_signing_key()` re-signs all entries with new key

### Encryption at Rest (AES-256-GCM)

- Implementation: `src/cacheness/encryption.py`
- Algorithm: AES-256-GCM authenticated encryption (12-byte random IV per blob)
- Key derivation: HKDF-SHA256 per-namespace encryption keys
  - Info string: `b"cacheness-aes-gcm-v1:" + namespace_id.encode()`
  - Master key: 32-byte key from `encryption_key_file` (reuses signing key by default)
- Dependency: `cryptography>=41.0.0` (optional, `cacheness[encryption]`)
- Integration points:
  - Blob writes: `BlobStore._write_blob()` encrypts between handler compression and backend write
  - Blob reads: `BlobStore._read_blob()` decrypts between backend read and handler decompression
  - Inline blobs: `InlineBlobMixin._try_inline_write()` encrypts before DB storage
- Metadata fields: `encryption_algorithm`, `encryption_iv` stored per entry (schema v4)
- Key rotation: `UnifiedCache.rotate_signing_key()` also re-encrypts all blobs with new key
- Config validation: prevents unsafe combos (encryption without signing, in-memory key with encryption)

### Key File Security

- Key file permissions:
  - Unix: `os.chmod(path, 0o600)` — owner-only read/write
  - Windows: `icacls` — removes inherited permissions, grants owner full control (`src/cacheness/security.py`)
- Key file location: inside cache directory by default
- In-memory keys: `SecurityConfig.use_in_memory_key=True` for testing (not compatible with encryption)

## File Integrity

**xxHash-based file hashing:**
- Algorithm: xxh3_64 (fast, non-cryptographic)
- Implementation: `src/cacheness/file_hashing.py`
- Used for: blob integrity verification, content-addressable storage
- Parallel hashing: `ProcessPoolExecutor` for large directories
- Hash stored in: `file_hash` column in metadata entry (encrypted ciphertext is hashed, not plaintext)

**Write Intent Journal:**
- Implementation: `WriteIntentJournal` in `src/cacheness/write_intent.py`
- Purpose: crash-safe blob writes — records intent before blob write, removes after metadata commit
- Stale intent cleanup: orphaned intents older than `stale_intent_threshold_seconds` (default 300s) are cleaned on cache init
- Storage: `.intents/` subdirectory under cache dir

**Integrity Verification:**
- Implementation: `VerificationMixin` in `src/cacheness/_verification_mixin.py`
- `verify_integrity()` detects: orphaned blobs, dangling metadata, hash mismatches, signature failures
- `verify_integrity(verify_signatures=True)` also checks HMAC signatures
- Returns: `IntegrityReport` dataclass (`src/cacheness/interfaces.py`)

## Monitoring & Observability

**Error Tracking:**
- No external error tracking service
- Custom exception hierarchy in `src/cacheness/error_handling.py`:
  `CacheError` → `CacheConfigurationError`, `CacheStorageError`, `CacheSerializationError`, `CacheHandlerError`, `CacheIntegrityError`, `CacheMetadataError`, `CacheSecurityError`, `CacheBackendError`

**Logs:**
- Python `logging` module throughout
- Logger names: `cacheness.core`, `cacheness.security`, `cacheness.encryption`, etc.
- Test config: `log_cli_level = "WARNING"` (override with `--log-cli-level=INFO`)

**Lifecycle Hooks:**
- `HooksConfig.on_evict(cache_key, reason)` — called on entry eviction
- `HooksConfig.on_integrity_failure(cache_key, failure_type, detail)` — called on hash/signature failures
- Hooks are synchronous, exceptions are swallowed

## CI/CD & Deployment

**Hosting:**
- Library published as Python package (no hosted deployment)

**CI Pipeline:**
- No CI/CD config in repository (local development only)
- Quality gates: `scripts/quality-check.ps1` / `scripts/quality-check.sh`
- Pre-commit hook: `scripts/hooks/pre-commit` (auto-runs ruff format + check)

**Docker (testing only):**
- `docker-compose.yml` at repo root
- PostgreSQL 16 Alpine — metadata backend testing (port 5432)
- Garage (S3-compatible) — blob backend testing (port 3900 S3 API, port 3903 admin)
- NOT for production — hardcoded test credentials

## Environment Configuration

**Required env vars:**
- None — all config is via Python objects or config files

**Optional env vars:**
- Standard AWS env vars (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`) — used by boto3 if S3 backend is configured without explicit credentials

**Config files:**
- `config/test_config.json` — JSON test config (PostgreSQL + Garage S3)
- `config/test_config.yaml` — YAML test config (same as JSON)
- `config/local_sqlite_fs.yaml` — Local SQLite + filesystem config
- `config/garage.toml` — Garage S3 server config
- `config/Dockerfile.garage` — Garage Docker image build

## Webhooks & Callbacks

**Incoming:** None — this is a library, not a service

**Outgoing:** None — no external API calls at runtime

**Internal callbacks:**
- `HooksConfig.on_evict` — eviction notification
- `HooksConfig.on_integrity_failure` — integrity failure notification
- Custom metadata models via `@custom_metadata_model` decorator (`src/cacheness/custom_metadata.py`)

---

*Integration audit: 2026-04-07*
