# Transaction Guarantees & ACID Analysis

This document describes Cacheness's transaction guarantees across all backend pairings. It covers per-operation atomicity, crash recovery, concurrency models, and the design rationale behind the two-layer (blob + metadata) architecture.

## Table of Contents

- [Quick Reference](#quick-reference)
- [Architecture Overview](#architecture-overview)
- [Per-Backend Guarantees](#per-backend-guarantees)
- [Per-Operation Analysis](#per-operation-analysis)
- [Combined Backend Pairings](#combined-backend-pairings)
- [Crash Scenarios](#crash-scenarios)
- [Concurrency Model](#concurrency-model)
- [Recovery Mechanisms](#recovery-mechanisms)
- [Design Rationale](#design-rationale)

## Quick Reference

What you can rely on when calling the Cacheness API:

| Operation | Guarantee | Caveat |
|-----------|-----------|--------|
| **`put()`** | Data is either fully stored or not stored at all. | On crash between blob write and metadata write, an orphaned blob may remain (harmless, cleaned by `verify_integrity`). Duplicate keys silently overwrite — old blob is cleaned up if the file path changes. |
| **`get()`** | Returns the stored value or `None`. | **Destructive on corruption:** if the blob is missing or corrupt, `get()` auto-deletes both the blob file and metadata entry, then returns `None`. Transient I/O errors are re-raised without deleting. |
| **`update_data()`** | Old value remains readable until the new value is fully written. | Uses a staging path — the old entry is valid until metadata swaps to the new blob. Old blob deletion is best-effort. |
| **`invalidate()`** | Entry is removed. | Uses metadata-first ordering. On crash between steps, an orphaned blob may remain (harmless, cleaned by `verify_integrity`). |
| **`verify_integrity()`** | Detects and optionally repairs all inconsistencies. | Scans all entries — not for hot paths. |

**Key behaviors to know:**
- **`get()` is destructive on errors** — it deletes both the blob file and metadata entry for corrupt/missing blobs (except transient I/O errors). This is intentional self-healing, not a bug.
- **Duplicate keys overwrite** — calling `put()` with a key that already exists silently replaces the entry. If the data type changes (different file extension), the old blob is cleaned up.
- **No cross-layer atomicity** — blob storage and metadata are two independent atomic operations. All operations are ordered so that crashes can only produce harmless orphaned blobs, never dangling metadata pointers.

### Storage Mode Differences

When `storage_mode=True`, Cacheness acts as a **persistent key-value store** rather than a cache. The transaction mechanics (blob-first writes, two-layer architecture) are identical, but the durability contract changes:

| Behavior | Cache Mode | Storage Mode |
|----------|------------|--------------|
| **`get()` on corrupt/missing blob** | **Deletes entry**, returns `None` | Returns `None`, **entry preserved** |
| **TTL expiration** | Entries auto-expire | Disabled — entries persist indefinitely |
| **Eviction** | LRU/size-based cleanup | Disabled — no automatic deletion |
| **Integrity/signature failure** | Entry deleted | Returns `None`, entry preserved |

**Why it matters:** In cache mode, the source of truth is the original computation — losing a cache entry just means recomputing. In storage mode, the cached data *is* the source of truth, so auto-deletion on errors would cause data loss. Storage mode preserves entries on all failures, leaving repair decisions to the caller or `verify_integrity`.

## Architecture Overview

Cacheness uses a **two-layer storage architecture**:

1. **Blob layer** — stores serialized data (filesystem or S3)
2. **Metadata layer** — stores keys, hashes, timestamps, TTL, custom metadata (JSON, SQLite, or PostgreSQL)

These two layers are **not wrapped in a single transaction**. Each operation performs two independent atomic steps, typically blob-first, metadata-second. This design is intentional — it trades strict cross-layer atomicity for simplicity, portability, and self-healing recovery.

```
┌─────────────────────────────────────────────┐
│                  core.py                     │
│         put() / get() / delete()             │
└────────────┬───────────────────┬─────────────┘
             │                   │
     ┌───────▼───────┐   ┌──────▼───────┐
     │  Blob Layer   │   │ Metadata Layer│
     │  (blob_store) │   │  (metadata)   │
     └───────┬───────┘   └──────┬────────┘
             │                   │
     ┌───────▼───────┐   ┌──────▼───────────────┐
     │ Filesystem /  │   │ JSON / SQLite /       │
     │ S3            │   │ PostgreSQL             │
     └───────────────┘   └──────────────────────┘
```

**Key invariant:** A blob without metadata is a harmless orphan (cleaned up by `verify_integrity`). Metadata pointing to a missing blob is a **dangling pointer** — the dangerous failure mode.

**All operations** are ordered so that crashes can only produce **orphaned blobs** (harmless), never **dangling pointers** (data loss). Write operations (`put`, `update_data`) write the blob first, then metadata. Delete operations (`invalidate`) remove metadata first, then delete the blob. Both orderings guarantee that the only possible crash artifact is an orphaned blob.

## Per-Backend Guarantees

### Blob Backends

| Property | Filesystem | S3 |
|----------|------------|-----|
| **Write atomicity** | Yes — temp file + `os.replace()` (atomic on all OSes) | Yes — `PutObject` is atomic |
| **Write integrity** | OS-level (no explicit fsync) | `ContentMD5` header verified server-side |
| **Read consistency** | Immediate | Strong (S3 strong consistency since Dec 2020) |
| **Durability** | After OS flush (no explicit fsync) | Durable after 200 OK |
| **Concurrent writes** | Last-write-wins (atomic rename) | Last-write-wins |
| **Delete atomicity** | Yes — `os.remove()` is atomic | Yes — `DeleteObject` is atomic |

### Metadata Backends

| Property | JSON | SQLite | PostgreSQL |
|----------|------|--------|------------|
| **Transaction support** | None (full file rewrite) | Implicit per-statement | Full ACID |
| **Write atomicity** | File-level — temp + `os.replace()` | Row-level — WAL mode | Row-level — MVCC |
| **Durability** | After OS flush (no fsync) | `synchronous=NORMAL` — WAL synced at checkpoint | Configurable (default: fsync on commit) |
| **Concurrency** | **Not safe** for multiple processes | WAL mode + `busy_timeout=30s` + `threading.Lock` | Full MVCC, connection pooling |
| **Rollback** | None (crash during write = corrupt) | Automatic WAL rollback | Explicit `session.rollback()` |
| **Scale limit** | <200 entries recommended | Millions of entries | Unlimited (distributed) |

> **Important:** JSON backend performs a **full file rewrite** on every mutation. It is not safe for concurrent access from multiple processes. Use SQLite or PostgreSQL for production workloads.

## Per-Operation Analysis

### `put()` — Store New Entry

```
1. Serialize data → blob bytes (in-memory)
2. Write blob to storage              ← Blob layer (atomic)
3. Store metadata (key, hash, path)   ← Metadata layer (atomic)
```

**Failure window:** Between steps 2 and 3, a crash leaves an orphaned blob with no metadata pointing to it. This is harmless — `verify_integrity(repair=True)` cleans it up.

**Rollback:** `_PutCleanup` context manager wraps steps 2-3. If step 3 raises an exception, it makes a best-effort attempt to delete the orphaned blob. This is cooperative cleanup, not a true transaction rollback.

### `get()` — Retrieve Entry

```
1. Look up metadata by key            ← Metadata layer (read)
2. Read blob from storage             ← Blob layer (read)
3. Deserialize blob → Python object   (in-memory)
```

**Self-healing (destructive on errors):** If step 2 fails (blob missing or corrupt), `get()` **automatically deletes both the blob file and metadata entry**, then returns `None`. This means a read can mutate state — it is the primary recovery mechanism for dangling pointers, but callers should be aware that a failed `get()` permanently removes the entry. Transient I/O errors are the exception — they are re-raised without deleting.

### `update_data()` — Modify Existing Entry

This is the most complex operation, using a **write-then-swap** pattern:

```
1. Serialize new data → blob bytes              (in-memory)
2. Write blob to STAGING path (_stg{uuid})      ← Blob layer (atomic)
3. Update metadata: point to staging path       ← Metadata layer (atomic)
4. Delete OLD blob                              ← Blob layer (best-effort)
5. If step 3-4 fail: delete staging blob        (rollback attempt)
```

**Key property:** The old entry remains fully valid until step 3 completes. If a crash occurs during step 2, the staging blob is an orphan (harmless). If a crash occurs during step 3, either the old or new metadata is in place — never a partial state.

**Caveat:** Step 4 (old blob deletion) is best-effort. A crash here leaves a harmless orphan.

### `invalidate()` / `delete()` — Remove Entry

```
1. Look up metadata by key            ← Metadata layer (read)
2. Resolve blob path from entry       (in-memory)
3. Remove metadata entry               ← Metadata layer (atomic)
4. Delete blob from storage            ← Blob layer (best-effort)
```

**Metadata-first ordering:** Metadata is removed before the blob is deleted. A crash between steps 3 and 4 leaves an orphaned blob (harmless — cleaned by `verify_integrity`), never a dangling pointer. Blob deletion is best-effort — if it fails, a warning is logged and the orphan remains for `verify_integrity` to clean up.

### `verify_integrity()` — Audit & Repair

```
For each entry:
1. Check metadata exists and is well-formed
2. Check blob exists at recorded path
3. Verify blob hash matches stored hash
4. (S3) Use ETag for fast integrity check — skip download if match
5. Optionally repair: delete entries with missing/corrupt blobs
```

This is the **catch-all recovery mechanism**. It detects and repairs all inconsistencies that can arise from crashes in other operations.

## Combined Backend Pairings

The effective guarantees depend on which blob + metadata backends are paired:

### Filesystem + JSON (Development)

| Guarantee | Level |
|-----------|-------|
| **Atomicity** | Per-layer only. No cross-layer transaction. |
| **Consistency** | Eventual — orphans cleaned by `verify_integrity` |
| **Isolation** | **None** — not safe for concurrent processes |
| **Durability** | After OS flush (no explicit fsync in either layer) |
| **Recovery** | `get()` self-healing + `verify_integrity(repair=True)` |

**Use for:** Local development, small caches, single-process scripts.

### Filesystem + SQLite (Production — Local)

| Guarantee | Level |
|-----------|-------|
| **Atomicity** | Per-layer only. No cross-layer transaction. |
| **Consistency** | Eventual — orphans cleaned by `verify_integrity` |
| **Isolation** | Multi-process safe (WAL + busy_timeout + threading.Lock) |
| **Durability** | SQLite: WAL checkpointed. Filesystem: after OS flush. |
| **Recovery** | `get()` self-healing + `verify_integrity(repair=True)` |

**Use for:** Production local caching, multi-process workloads, caches with 200+ entries.

### S3 + PostgreSQL (Production — Distributed)

| Guarantee | Level |
|-----------|-------|
| **Atomicity** | Per-layer only. No cross-layer transaction. |
| **Consistency** | Eventual — orphans cleaned by `verify_integrity` |
| **Isolation** | Full MVCC (PostgreSQL). S3 is last-write-wins. |
| **Durability** | S3: 11 nines. PostgreSQL: configurable (default: fsync). |
| **Recovery** | `get()` self-healing + `verify_integrity(repair=True)` + ETag checks |

**Use for:** Distributed systems, multi-node access, high-availability requirements.

## Crash Scenarios

| Scenario | State After Crash | Recovery |
|----------|-------------------|----------|
| Crash during `put()` step 2 (blob write) | Partial/no blob, no metadata | No action needed — incomplete temp file |
| Crash during `put()` step 3 (metadata write) | Orphaned blob, no metadata | `verify_integrity(repair=True)` cleans orphan |
| Crash during `get()` step 2 (blob read) | No state change | Retry `get()` |
| Crash during `update_data()` step 2 (staging write) | Orphaned staging blob | `verify_integrity(repair=True)` cleans orphan |
| Crash during `update_data()` step 3 (metadata swap) | Old entry still valid OR new entry in place | Either state is consistent |
| Crash during `update_data()` step 4 (old blob delete) | Old blob orphaned, new entry valid | `verify_integrity(repair=True)` cleans orphan |
| Crash during `delete()` step 3 (metadata remove) | Orphaned blob, no metadata | `verify_integrity(repair=True)` cleans orphan |
| Crash during `delete()` step 4 (blob delete) | Both gone or orphaned blob | Clean state or `verify_integrity` cleans orphan |
| JSON backend: crash during file rewrite | **Corrupt metadata file** | Manual recovery required |
| SQLite backend: crash during write | WAL rollback restores last consistent state | Automatic |
| PostgreSQL: crash during write | Transaction rollback | Automatic |

## Concurrency Model

### JSON Backend
- **Thread safety:** None built-in
- **Process safety:** **Not safe** — concurrent writes corrupt the file
- **Recommendation:** Single-process, single-thread only

### SQLite Backend
- **Thread safety:** `threading.Lock` serializes in-process access
- **Process safety:** WAL mode allows concurrent readers with one writer. `busy_timeout=30s` retries on lock contention.
- **Connection management:** One connection per `SqliteMetadataBackend` instance
- **Recommendation:** Safe for multi-process, multi-thread workloads

### PostgreSQL Backend
- **Thread safety:** SQLAlchemy session management
- **Process safety:** Full MVCC — multiple readers and writers
- **Connection management:** Connection pooling via SQLAlchemy engine
- **Recommendation:** Safe for distributed, multi-node workloads

### Blob Layer Concurrency
- **Filesystem:** `os.replace()` is atomic — concurrent writes are last-write-wins, no corruption
- **S3:** `PutObject` is atomic — concurrent writes are last-write-wins, no corruption

## Recovery Mechanisms

Cacheness uses a **defense-in-depth** approach to handle inconsistencies:

### Layer 1: Cooperative Cleanup (`_PutCleanup`)
During `put()`, if metadata storage fails after the blob is written, the `_PutCleanup` context manager attempts to delete the orphaned blob. This is best-effort — if the cleanup itself fails, the orphan remains for Layer 3.

### Layer 2: Self-Healing Reads (`get()`)
When `get()` encounters a missing or corrupt blob, it automatically deletes the stale metadata entry and returns `None`. This handles the most common inconsistency (dangling pointer) transparently. Transient I/O errors are **not** treated as corruption — they are re-raised.

### Layer 3: Audit & Repair (`verify_integrity`)
The `verify_integrity(repair=True)` method performs a full audit of all entries:
- Detects orphaned blobs (blob without metadata)
- Detects dangling pointers (metadata without blob)
- Verifies blob hashes match stored hashes
- Uses S3 ETags for efficient remote integrity checks
- Optionally removes all inconsistent entries

This is the catch-all mechanism — it can recover from any state that Layers 1 and 2 missed.

## Design Rationale

### Why No Cross-Layer Transaction?

A true cross-layer transaction would require:
1. A distributed transaction coordinator (2PC) between blob storage and metadata
2. Both backends to support the same transaction protocol
3. Significant complexity and performance overhead

This is impractical when the blob layer might be S3 and the metadata layer might be PostgreSQL on a different host. Instead, Cacheness uses the **blob-first** ordering combined with **self-healing reads** to achieve eventual consistency with minimal complexity.

### Why Blob-First?

Writing the blob before metadata ensures that the only possible inconsistency is an **orphaned blob** (harmless), never a **dangling pointer** (data loss). Orphaned blobs are cleaned up lazily by `verify_integrity`.

### Why No fsync?

Explicit `fsync` after every write significantly impacts performance. Cacheness trades durability for speed — data is durable after the OS flushes its buffers, which is acceptable for a cache (the source of truth is the original computation). For scenarios requiring stronger durability, S3 + PostgreSQL provides persistence guarantees from the underlying infrastructure.

### When to Run `verify_integrity`

- **After unclean shutdown** — to clean up any partial operations
- **Periodically in long-running services** — as a consistency health check
- **Before critical reads** — when cache correctness is essential
- **Never in hot paths** — it scans all entries and is not designed for per-request use
