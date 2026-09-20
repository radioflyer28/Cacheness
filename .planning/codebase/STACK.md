<!-- refreshed: 2026-09-17 -->
# Technology Stack

**Analysis date:** 2026-09-17

## Runtime and tooling

- **Python:** `>=3.11`; this checkout pins Python 3.13 in `.python-version`.
- **Package workflow:** `uv`, `uv.lock`, and the `uv_build` backend.
- **Tests:** pytest with pytest-cov in the `dev` dependency group.
- **Static checks:** Ruff targeting Python 3.11 with an 88-character line
  length. Run scoped checks over touched Python files rather than treating the
  repository-wide historical baseline as a clean gate.

## Primary dependencies

| Dependency | Role |
|---|---|
| `obstore==0.11.1` | Immutable local, memory, and S3 payload object I/O beneath `BlobStore` ownership. |
| `cryptography` | Manifest/signing and integrity primitives. |
| `numpy` | Base array and serialization dependency. |
| `xxhash` | Fast cache-key and content hashing where configured. |
| `cachetools` | Optional in-process cache helpers. |
| `sqlalchemy` | SQLite metadata/projection work and optional PostgreSQL lifecycle/projection integration. |
| `psycopg` | Optional PostgreSQL authority driver. |
| `pandas`, `pyarrow`, `polars` | Optional dataframe/Parquet handler integrations. |
| `blosc2`, `dill`, `orjson` | Optional format and serialization handler integrations. |

## Published extras

The exact installable extras are:

`recommended`, `dataframes`, `s3`, `postgresql`, and `cloud`.

The wheel contract verifies this complete retained set along with the absence of
retired modules, requirements, extras, and handler identities. The `s3` group
has no additional package requirement because base `obstore` supplies the
participant. Do not add compatibility groups or remove SQLAlchemy, `psycopg`,
pandas, or PyArrow without tracing their retained authority, projection, and
handler owners.

## Installability and configuration

- Base installation declares NumPy, so package import does not rely on it being
  transitively present. Optional dependencies must remain capability-guarded at
  their feature boundaries.
- `CacheConfig` and topology objects provide runtime configuration; the library
  has no general project `.env` parser.
- Ordinary local use needs a writable filesystem. Optional PostgreSQL and S3
  paths require the corresponding caller-supplied service configuration.

## Qualification boundary

- The checked-in lock is authoritative for frozen local development and fresh
  wheel tests inspect both archive members and installed metadata.
- The non-live test command excludes only `live_postgresql`, `live_aws_s3`, and
  `live_remote`. It establishes local/controlled contract evidence, not a
  release claim for real remote services or controlled-Linux performance.

---

*Current stack map refreshed for the post-cut product boundary on 2026-09-19.*
