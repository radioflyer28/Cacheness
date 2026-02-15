# Architecture Document: libsql Metadata Backend for Cacheness

**Status:** Proposal (P3)
**Date:** 2026-02-15
**Author:** Copilot research session

## TL;DR

Adding a libsql (Turso) metadata backend enables **local-first SQLite caching with optional cloud sync** via embedded replicas — filling the gap between local-only SQLite and fully remote PostgreSQL. The implementation faces two key design decisions: (1) namespace isolation strategy (table-per-namespace vs database-per-namespace), and (2) SQLAlchemy vs raw DBAPI. This document analyzes trade-offs and recommends a path forward.

---

## 1. What libsql Brings to Cacheness

| Feature | Current SQLite | libsql (new) | PostgreSQL |
|---------|---------------|--------------|------------|
| Location | Local file | Local file + optional cloud sync | Remote server |
| Multi-machine | No | Yes (embedded replicas) | Yes |
| Latency | Microseconds | Microseconds reads, async writes | Milliseconds |
| Encryption at rest | No | Built-in (`encryption_key` param) | Server-managed |
| Offline capable | Yes | Yes (writes queue locally) | No |
| Setup complexity | Zero | Near-zero (local); moderate (Turso Cloud) | High |

**Embedded replicas** are the killer feature: a local `.db` file syncs with a remote Turso/sqld primary. Reads hit local disk (microsecond latency), writes propagate to the remote. This gives Cacheness distributed caching without PostgreSQL's complexity.

---

## 2. Current Namespace Implementation (Status Quo)

Defined in `src/cacheness/metadata.py`:

- **`MetadataBackend` ABC** (L324): 14 abstract methods + concrete namespace lifecycle defaults
- **EntityName pattern** (`_get_namespace_models()`, L228): `type()` dynamically creates ORM subclasses per namespace — `CacheEntry_production` → `cache_entries_production` table
- **Default namespace**: `"default"` maps to unsuffixed tables (`cache_entries`, `cache_stats`) for backward compat
- **Registry**: `CacheNamespace` table (`cacheness_namespaces`) tracks all namespaces
- **`SqliteBackend`** (L1672): SQLAlchemy engine + sessionmaker, mixed ORM/raw SQL queries, threading.Lock
- **`PostgresBackend`** (`storage/backends/postgresql_backend.py`): Separate module, separate `PostgresBase`, same EntityName pattern

Each `UnifiedCache` instance is bound to one namespace (immutable after init). The metadata backend is scoped to that namespace's tables.

---

## 3. Decision: Namespace Isolation Strategy

### Option A: Table-per-Namespace (Current Pattern)

All namespaces in a single `.db` file with separate tables per namespace.

| Aspect | Assessment |
|--------|-----------|
| Code reuse | Maximal — reuse EntityName pattern, `_get_namespace_models()`, `CacheNamespace` registry |
| Sync behavior | Entire DB syncs as one unit — all namespaces replicated together |
| Encryption | Single key for all namespaces |
| Drop namespace | `DROP TABLE` (cannot reclaim space without `VACUUM`) |
| Connection count | Single engine/connection pool |
| Schema migration | One migration pass per namespace in the same DB |
| Isolation quality | Logical only — corrupt DB affects all namespaces |

**Best for:** Single-user local caching, simple Turso Cloud setups where all namespaces share the same replication and encryption policy.

### Option B: Database-per-Namespace

Each namespace gets its own `.db` file (each with independent sync/encryption).

| Aspect | Assessment |
|--------|-----------|
| Code reuse | Moderate — need a connection manager, but each DB uses simple (default-namespace) tables |
| Sync behavior | Per-namespace — can sync `production` to Turso Cloud but keep `dev` local-only |
| Encryption | Per-namespace keys possible |
| Drop namespace | Delete the file — instant, full space reclaim |
| Connection count | One engine per namespace — more resources, but libsql connections are lightweight |
| Schema migration | Simpler — each DB is self-contained |
| Isolation quality | Physical — corrupt `dev.db` doesn't touch `production.db` |

**Best for:** Multi-tenant scenarios, per-namespace sync policies, per-namespace encryption, clean namespace lifecycle.

### Option C: Hybrid (Table-per-namespace default, opt-in DB-per-namespace)

Default to table-per-namespace for simplicity; allow `namespace_isolation="database"` config option for physical isolation.

| Aspect | Assessment |
|--------|-----------|
| Flexibility | Maximum — covers both use cases |
| Complexity | Highest — two code paths, more testing surface |
| Config surface | Larger — must document when to use which |

### Trade-off Analysis

The key differentiator is **sync granularity**. With embedded replicas:

- **Table-per-namespace**: All namespaces sync together. You can't sync `production` to Turso Cloud while keeping `experiments` local. This is the same limitation as regular SQLite — adequate for users who just want "SQLite but replicated."
- **Database-per-namespace**: Each namespace can have its own `sync_url` (or none). This is the model Turso was designed for ("database-per-tenant").

If the primary value proposition is *"local SQLite that optionally syncs"*, then **database-per-namespace** aligns naturally with libsql's architecture. But it's more work and doesn't match the existing SqliteBackend pattern.

If the goal is *"drop-in SQLite replacement with extra features"*, then **table-per-namespace** is the pragmatic choice — minimal code, proven pattern.

---

## 4. Decision: SQLAlchemy Dialect vs Raw DBAPI

### Option X: `sqlalchemy-libsql` Dialect

Turso's official SQLAlchemy dialect: `sqlite+libsql://`

| Factor | Assessment |
|--------|-----------|
| Code reuse | **Very high** — swap `create_engine()` URL, most SqliteBackend works as-is |
| Platform | **Linux/macOS only** — stated explicitly in README |
| Maturity | **Very low** — v0.1.0-pre, 18 stars, 2 contributors, 0 releases, still imports deprecated `libsql_experimental` (issue #9 open), SIGSEGV on Python 3.14 (issue #13) |
| Async support | Yes (`sqlite+aiolibsql://`) |
| Dependencies | `sqlalchemy-libsql` → `libsql-experimental` (deprecated) |
| Pragma support | Not validated — dialect overrides `on_connect()` and notes *"no support for create_function()"*; unclear if Cacheness's WAL/mmap pragmas work |
| Maintenance risk | High — 8 months since last commit, small team |

**Source analysis** (`sqlalchemy_libsql/libsql.py`): The dialect is ~70 lines subclassing `SQLiteDialect_pysqlite`, overriding `import_dbapi()`, `on_connect()`, and `create_connect_args()`. It's thin — which is good for simplicity but means SQLite-specific features like custom pragmas may not be validated.

### Option Y: Raw DBAPI (No SQLAlchemy)

Use `libsql.connect()` directly, write queries as parameterized SQL strings.

| Factor | Assessment |
|--------|-----------|
| Code reuse | **Moderate** — can reuse query logic from SqliteBackend's raw SQL paths (`put_entry`, `iter_entry_summaries`) but must rewrite ORM-based paths |
| Platform | **Cross-platform** — `libsql` SDK supports Windows, Linux, macOS |
| Maturity | **Good** — v0.1.11, 195 stars, 15 contributors, stable API |
| Async support | Not built-in, but connections are lightweight |
| Dependencies | `libsql>=0.1.11` only (no SQLAlchemy needed) |
| Pragma support | Full control — `conn.execute("PRAGMA ...")` works natively |
| Maintenance risk | Lower — Turso's primary Python SDK, actively maintained |

**API surface** — `libsql.connect()` returns a connection with: `execute(sql, params)`, `cursor()`, `commit()`, `sync()`, `close()`. Parameters can be tuples or lists. Context manager support for connections.

### Comparison: What Needs Rewriting

Current `SqliteBackend` uses two patterns:

| Pattern | Methods | Reuse with dialect? | Reuse with raw DBAPI? |
|---------|---------|--------------------|-----------------------|
| ORM (`select/update/delete` + session) | `get_entry`, `remove_entry`, `update_entry_metadata`, `list_entries`, `update_access_time`, `increment_*`, `cleanup_*` | Yes (identical) | Must rewrite as SQL |
| Raw SQL (`text()` + session) | `put_entry`, `iter_entry_summaries`, `create_namespace`, `drop_namespace` | Yes (near-identical) | Minor adaptation (`:name` → `?` params) |

**Effort estimate:**
- Dialect approach: ~200 LOC new backend class (mostly config/init), ~50 LOC adapted from SqliteBackend
- Raw DBAPI approach: ~600-800 LOC new backend class, all queries rewritten as raw SQL

### Pragmatic Assessment

The `sqlalchemy-libsql` dialect is **not production-ready** for Cacheness:
1. Still imports deprecated `libsql_experimental` (not `libsql`)
2. No Windows support (Cacheness's primary dev environment)
3. Segfaults on Python 3.14
4. 0 published releases

However, it demonstrates that SQLAlchemy + libsql is *architecturally possible*. If the dialect matures, a future migration from raw DBAPI to dialect could be done without changing the backend's API.

---

## 5. libsql-Specific Features Worth Leveraging

### 5a. Embedded Replicas

```python
conn = libsql.connect(
    "local_cache.db",              # local file for microsecond reads
    sync_url="libsql://...",       # remote primary
    auth_token="...",              # Turso token
    sync_interval=60,             # auto-sync every 60s
)
```

**Integration point:** `create_metadata_backend("libsql", sync_url=..., auth_token=..., sync_interval=...)` — if no `sync_url`, behaves as local-only SQLite.

### 5b. Encryption at Rest

```python
conn = libsql.connect("encrypted.db", encryption_key="secret")
```

**Complements** Cacheness's existing `cache_signing` feature. Signing ensures integrity; encryption ensures confidentiality. They're orthogonal.

### 5c. Offline Writes

libsql v0.1.11 added `offline=True` — writes go to local DB when remote is unreachable, sync when connectivity returns. Natural fit for laptop/edge caching scenarios.

---

## 6. Configuration Surface

Proposed config extension for `CacheConfig`:

```yaml
metadata:
  backend: libsql
  db_file: cache_metadata.db      # local file path
  sync_url: libsql://...          # optional — enables embedded replicas
  auth_token: ${TURSO_AUTH_TOKEN}  # optional — for Turso Cloud
  sync_interval: 60               # optional — auto-sync seconds
  encryption_key: ${CACHE_ENC_KEY} # optional — encryption at rest
  offline: false                  # optional — allow offline writes
```

**Backend type routing** in `create_metadata_backend()`:
- `"libsql"` → `LibsqlBackend(db_file=..., sync_url=..., ...)`
- `"auto"` cascade: SQLite → libsql → JSON (or: libsql → SQLite → JSON if libsql is detected)

---

## 7. File Structure

Following the PostgreSQL backend's precedent (separate module, lazy-imported):

```
src/cacheness/
├── metadata.py                          # unchanged (ABC, SQLite, JSON)
├── storage/backends/
│   ├── postgresql_backend.py            # existing
│   └── libsql_backend.py               # NEW
```

**Dependencies** (new `pyproject.toml` extras):
```toml
[project.optional-dependencies]
libsql = ["libsql>=0.1.11"]

[dependency-groups]
libsql = ["libsql>=0.1.11"]
```

No SQLAlchemy dependency for the libsql backend — it's a standalone raw DBAPI implementation.

---

## 8. Open Questions

1. **Namespace strategy**: Need decision on table-per-namespace vs database-per-namespace (or hybrid). This is the architectural fork that shapes everything.

2. **Sync semantics for cache writes**: When a `put_entry()` succeeds locally, should `sync()` be called immediately, deferred to `sync_interval`, or manual? Immediate sync guarantees remote durability but adds latency; deferred sync keeps writes fast but risks data loss on crash.

3. **Test infrastructure**: Raw libsql tests can run locally (no Turso account needed). Embedded replica tests need either a Turso Cloud account or a self-hosted `sqld` container. Should CI include a `sqld` docker service (like we have docker-compose for PostgreSQL)?

4. **Migration path**: Should `SqliteBackend` users be able to migrate their `.db` file to `LibsqlBackend` by just changing the backend type? (Likely yes — libsql is a superset of SQLite, so existing `.db` files should work.)

---

## 9. Recommendation Summary

| Decision | Recommended | Rationale |
|----------|------------|-----------|
| DBAPI approach | **Raw DBAPI** (`libsql>=0.1.11`) | Cross-platform, mature SDK, no dependency on half-baked SQLAlchemy dialect |
| Namespace strategy | **Table-per-namespace initially**, with architecture that doesn't preclude database-per-namespace later | Proven pattern, lower risk, matches existing backends. Database-per-namespace can be added as a config option if demand emerges |
| Sync mode | **Configurable** — default to `sync_interval` if `sync_url` is set, with manual `sync()` exposed | Balances latency and durability |
| File location | `src/cacheness/storage/backends/libsql_backend.py` | Follows PostgreSQL precedent |
