# Feature Landscape

**Domain:** Disk caching / persistent key-value store library (Python)
**Milestone:** v0.8.0 API & Robustness
**Researched:** 2026-04-02

## Table Stakes

Features users expect in a production-quality disk caching / key-value store library. Missing = product feels incomplete for real workloads.

### Management APIs

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| `get_metadata()` — retrieve entry metadata without deserializing blob | Every key-value store exposes metadata inspection. diskcache has `.peek()`, Redis has `TYPE`/`OBJECT`, shelve has `keys()`. Users need to inspect entries without loading multi-GB blobs into memory. | **Low** | Backend methods already exist (`get_entry()`, `BlobStore.get_metadata()`). This is a thin wrapper in `UnifiedCache` that resolves cache key and returns the metadata dict. |
| `touch()` — reset TTL without reloading data | Redis `EXPIRE`/`PERSIST`, diskcache `Cache.touch()`, memcached `touch`. Standard in every cache system with TTL. Without it, keeping hot entries alive requires a full `get()` + `put()` round-trip — wasteful for multi-GB blobs. | **Low** | Needs `update_entry_timestamp()` on all metadata backends. JSON: rewrite entry timestamp. SQLite: single `UPDATE` statement. PostgreSQL: single `UPDATE`. No blob I/O. |
| `delete_by_prefix()` — bulk delete entries matching a key prefix | Redis `SCAN` + `DEL` pattern, diskcache `Cache.evict()` with tag, S3 `DeleteObjects` with prefix. Essential for namespace cleanup (delete all `exp_v1/*` entries). Without it, users loop `list_entries()` + `invalidate()` — O(n) metadata rewrites on JSON backend. | **Medium** | SQLite/PG can use `LIKE 'prefix%'` for single-query delete. JSON must iterate. Should return count of deleted entries. Must delete blobs too (not just metadata). |
| `update_blob_data()` — replace blob at existing key | Every mutable store supports in-place update. Redis `SET` overwrites, diskcache `Cache.__setitem__` overwrites, shelve `__setitem__` overwrites. Without it, users must `invalidate()` + `put()` — a race window where the key doesn't exist. | **Medium** | Already designed in MISSING_MANAGEMENT_API.md and partially implemented as `update_data()` (exists in `core.py` with staging path pattern). Needs to re-sign the entry, update file_size/file_hash/created_at, handle old blob cleanup. |
| Batch `get_batch()` / `delete_batch()` | Redis `MGET`/`DEL` (variadic), diskcache doesn't have batch (users iterate). Expected when working with experiment sets — "load all runs for experiment X". | **Medium** | SQLite/PG can batch metadata lookups in one query (`WHERE cache_key IN (...)`). JSON must iterate. Blob reads are inherently per-file — no shortcut. Main benefit is reduced metadata round-trips and transactional deletes. |

### Concurrency Safety

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Thread-safe `put()`/`get()` | joblib.Memory is process-safe via filesystem atomicity. diskcache is fully thread/process-safe via SQLite transactions. shelve is explicitly NOT thread-safe (documented). Users expect either safety or a clear documented boundary. Cacheness has an `_lock` that is acquired in `put()`/`get()` but backend-level safety varies. | **Low** | `put()` and `get()` already acquire `self._lock` (RLock). SQLite backend uses WAL mode. The real gap is documentation clarity and testing under concurrent load. JSON backend is fundamentally single-writer — this must be documented as a limitation, not "fixed". |

### Orphaned Blob Prevention

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Crash-safe two-phase writes with automatic recovery | LevelDB uses a write-ahead log. SQLite uses WAL + rollback journal. S3 has multi-part upload with abort. Users storing important data expect that a power failure or `kill -9` doesn't leave the cache in an inconsistent state that requires manual intervention. | **Medium** | Current `_PutCleanup` handles in-process exceptions but not process kills. `verify_integrity(repair=True)` catches orphans but must be called manually. The gap is detectable: a startup check or periodic background scan. |

## Differentiators

Features that set Cacheness apart. Not baseline-expected, but valuable — especially for the ML/data science audience.

### HMAC Blob Signing

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| HMAC-SHA256 over blob content (not just metadata) | **Tamper detection for stored blobs.** Current signing covers metadata fields only — an attacker with filesystem access can replace blob bytes while metadata signatures remain valid. Docker Content Trust signs image layers. S3 uses `Content-MD5` for upload integrity. AWS SSE signs objects at rest. Extending HMAC to blob content closes the "blob swap" attack vector, which the CONCERNS.md audit flagged as a security gap. | **Medium** | Blob content hash (`file_hash` via xxh3_64) already exists and is verified on `get()`. The differentiator is adding an HMAC *signature* over the hash — so the hash itself can't be tampered with in metadata. Implementation: include `file_hash` in HMAC signed fields (already in v1/v2 field lists!). The real work is ensuring `file_hash` is always populated (currently optional) and the signature covers it reliably. May already be partially working via signature v2 field list including `file_hash`. |
| Signature coverage of inline blob data | Inline blobs store `blob_data` directly in metadata. The HMAC should cover a hash of this data to prevent metadata-level tampering of inline entries. | **Low** | Compute xxh3_64 of `blob_data`, store as `file_hash` in metadata (may already happen), ensure it's in the signed field set. |

### Management APIs (Advanced)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| `update_batch()` — bulk update blob data | Batch checkpoint replacement for ML pipelines. Not offered by diskcache or joblib. Reduces per-entry overhead (single metadata transaction for N updates). | **High** | Each blob requires separate serialization + file write. Metadata updates can be batched in SQLite/PG. Complex error handling: partial batch failure semantics (all-or-nothing vs best-effort). |
| `touch_batch()` — bulk TTL refresh | Keep a working set alive in one call. Redis supports `EXPIRE` pipelining. Useful for "refresh all entries for active experiment". | **Low** | Single `UPDATE ... WHERE cache_key IN (...)` on SQLite/PG. JSON iterates. No blob I/O. |
| `get_metadata_batch()` — bulk metadata retrieval | Inspect many entries without loading blobs. Useful for dashboards, monitoring, experiment comparison. | **Low** | Single metadata query, no blob I/O. Natural extension of `get_metadata()`. |

### Orphaned Blob Prevention (Advanced)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Write-ahead intent log for crash recovery | Before writing a blob, record the intended operation (cache_key, target_path) in a lightweight journal. On startup or periodic check, replay incomplete operations (delete orphaned blobs from incomplete puts). This is how LevelDB, InnoDB, and PostgreSQL handle crash recovery. Transforms `verify_integrity` from a full-scan O(n) operation into a targeted O(pending) check. | **Medium-High** | Requires a new file/table for the intent log. Must be atomic itself (single `fsync`'d append or SQLite `INSERT`). Adds write overhead (one extra write per `put()`). Worth it for large caches where `verify_integrity` full-scan is expensive. |
| Startup orphan detection | Automatically detect and warn about (or repair) orphans when a cache is opened, rather than requiring explicit `verify_integrity()` calls. | **Low-Medium** | Could scan for files in cache_dir not referenced by metadata. For large caches, this is slow at startup — should be optional (`auto_repair_on_open=True`). For small caches (<1000 entries), the cost is negligible. |

## Anti-Features

Features to explicitly NOT build in v0.8.0.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| File-level locking (flock/lockf) for multi-process safety | Adds platform-specific complexity (Windows vs Unix locking semantics), doesn't work on NFS/network filesystems, and the primary concurrency model is already handled by SQLite WAL for the recommended production backend. diskcache uses SQLite-level locking, not file locks. | Document that multi-process safety requires SQLite or PostgreSQL backend. JSON backend is single-process only. Thread safety is provided by `_lock`. |
| Async `put()`/`get()` (`asyncio` support) | Large feature that deserves its own milestone. Requires async blob I/O, async metadata queries, async handler serialization. Would need `AsyncUnifiedCache` class or wrapper. | Defer to separate milestone. Document as a future improvement. |
| Global distributed locking (e.g., Redis-based locks for PostgreSQL backend) | Over-engineering for the current user base. PostgreSQL's MVCC already handles concurrent access. Adding distributed locks adds a Redis dependency and operational complexity. | PostgreSQL backend already provides row-level isolation via MVCC. Document that concurrent writers to the same key will last-write-win. |
| Automatic background eviction / GC thread | Daemon threads complicate shutdown, make testing fragile, and surprise users. diskcache avoids background threads. | Keep eviction synchronous (on `put()` via `_enforce_size_limit()`). Provide `verify_integrity(repair=True)` for manual cleanup. Users who want periodic cleanup can use `schedule` or `APScheduler`. |
| Transaction rollback across blob + metadata layers | Would require a WAL or undo log spanning both layers, fundamentally changing the architecture. The current "orphaned blobs are harmless" invariant is simpler and proven. | Keep the current two-phase write ordering (blob-first, metadata-second). Improve crash detection with intent logging rather than trying to make cross-layer writes transactional. |
| `copy()`/`move()` entry operations | Low priority convenience features that compose from existing CRUD primitives. Adding them increases API surface and test burden for operations users rarely need. | Document the `get()` + `put()` pattern for copying. Defer to a future milestone if demand materializes. |
| Encryption at rest for blob content | Feature addition, not hardening. Requires key management, performance impact assessment, migration path for existing caches. | Defer to a separate milestone. Document as a security enhancement opportunity. |

## Feature Dependencies

```
get_metadata() → (standalone, no dependencies)
touch() → needs update_entry_timestamp() on all metadata backends
delete_by_prefix() → needs prefix-aware delete on all metadata backends + blob cleanup
update_blob_data() → needs _PutCleanup, blob signing, file_hash computation
                    → improved by HMAC blob signing (sign new blob on update)

get_batch() → needs get_metadata() pattern (batch key resolution)
delete_batch() → needs delete_by_prefix() pattern (batch metadata + blob cleanup)
touch_batch() → needs touch() (batch timestamp update)
update_batch() → needs update_blob_data() (batch version)

HMAC blob signing → depends on file_hash always being populated
                  → should be done BEFORE update_blob_data (so updates produce signed blobs)

Orphan intent log → standalone, can be added before or after management APIs
                  → improves update_blob_data() crash recovery

Thread safety documentation → standalone, should be done early (informs API design)

Concurrency testing → depends on thread safety documentation (tests verify documented guarantees)
```

**Recommended ordering by dependency chain:**

1. Thread safety documentation + concurrency tests (foundation — clarifies guarantees)
2. `get_metadata()` + `touch()` (simple, no dependencies)
3. HMAC blob signing (make `file_hash` always-on, verify it's in signed fields)
4. `delete_by_prefix()` + `delete_batch()` (bulk cleanup)
5. `update_blob_data()` (complex, benefits from HMAC signing being in place)
6. `get_batch()` + `touch_batch()` + `get_metadata_batch()` (batch versions of already-working singles)
7. Orphan prevention improvements (intent log or startup detection)

## MVP Recommendation

Prioritize:

1. **`get_metadata()`** — trivial to implement, high daily-use value, zero risk
2. **`touch()`** — essential cache operation, low complexity, no blob I/O
3. **`delete_by_prefix()`** — most-requested cleanup operation, medium complexity
4. **HMAC blob signing verification** — security gap flagged in audit, may already be partially working (file_hash is in v2 signed fields), needs validation and testing
5. **Concurrency safety documentation + tests** — `put()`/`get()` already acquire the lock; document and test the actual guarantees rather than adding new mechanisms
6. **`update_blob_data()`** — completes CRUD operations, staging-path pattern already designed

Defer: `update_batch()` (highest complexity, lowest frequency), intent-log orphan prevention (medium-high complexity, current `verify_integrity` is adequate for most users), `copy()`/`move()` (convenience, composable from CRUD).

## Sources

- [MISSING_MANAGEMENT_API.md](../../docs/MISSING_MANAGEMENT_API.md) — existing API design proposals with layer analysis (HIGH confidence)
- [TRANSACTION_GUARANTEES.md](../../docs/TRANSACTION_GUARANTEES.md) — current crash recovery and concurrency model (HIGH confidence)
- [COMPARISON_TO_EXISTING_SOLUTIONS.md](../../docs/COMPARISON_TO_EXISTING_SOLUTIONS.md) — landscape analysis vs diskcache, joblib, shelve (HIGH confidence)
- [CONCERNS.md](../codebase/CONCERNS.md) — codebase audit: security gaps, missing features (HIGH confidence)
- [SECURITY.md](../../docs/SECURITY.md) — current HMAC signing architecture (HIGH confidence)
- Source: `core.py` (put/get with _lock, _PutCleanup), `security.py` (CacheEntrySigner, signed field lists), `_verification_mixin.py` (file_hash verification, verify_integrity), `blob_store.py` (orphan detection, metadata-first delete ordering)
- diskcache — HIGH confidence: SQLite-based, fully thread/process-safe via SQLite locking, `Cache.touch()` for TTL refresh, `Cache.evict()` for tag-based cleanup, no batch API
- joblib.Memory — HIGH confidence: filesystem-based, process-safe via atomic file operations, no thread safety guarantees, no TTL, no management APIs beyond `clear()`
- shelve — HIGH confidence: dbm-based, explicitly NOT thread-safe (Python docs), no TTL, no management APIs beyond `keys()`/`del`
- Redis — HIGH confidence: thread-safe, full management API (`MGET`, `EXPIRE`, `SCAN`+`DEL`)
- LevelDB/RocksDB — MEDIUM confidence: write-ahead log for crash recovery, `WriteBatch` for atomic batch operations
- Docker Content Trust — MEDIUM confidence: signs image layers (analogous to blob content signing), uses Notary/TUF for key management
- S3 Content-MD5 — HIGH confidence: upload-time integrity verification, ETag for subsequent verification
