# Technology Stack — v0.8.0 Additions

**Project:** Cacheness v0.8.0 API & Robustness
**Researched:** 2026-04-02
**Scope:** Stack changes needed for management APIs, concurrency safety, HMAC blob signing, orphaned blob prevention

## Executive Summary

All four v0.8.0 features can be implemented using **Python standard library only** — no new third-party dependencies required. The existing stack (xxhash, hmac, threading, pathlib) provides everything needed. This is the ideal outcome for a hardening milestone.

## Recommended Stack Additions

### No New Dependencies Required

| Feature | Technology | Already Available | Rationale |
|---------|-----------|-------------------|-----------|
| Management APIs | Python stdlib | Yes — `pathlib`, `typing` | Pure method additions to `UnifiedCache`; no external libraries needed |
| Concurrency safety | `threading.RLock` (stdlib) | Yes — already in core.py L173 | Extend existing lock to `put()`/`get()` paths; configurable via CacheConfig |
| HMAC blob signing | `hmac` + `hashlib` (stdlib) | Yes — already in security.py | Extend `CacheEntrySigner` to sign blob content hash; uses same HMAC-SHA256 |
| Orphaned blob prevention | `pathlib` + `os` (stdlib) | Yes — `_PutCleanup` exists | Write-ahead intent log or metadata-first ordering; no external deps |

### Existing Dependencies Leveraged

| Library | Version | Current Use | v0.8.0 Use |
|---------|---------|-------------|------------|
| `xxhash` | ≥3.5.0 | File content hashing (`file_hash` field) | Blob HMAC: hash blob content for signing; orphan detection: verify blob existence |
| `hmac` + `hashlib` | stdlib | HMAC-SHA256 metadata entry signatures | Extend to include blob content hash in signed fields |
| `threading` | stdlib | `RLock` in core.py (management ops only) | Extend lock scope to `put()`/`get()` when `thread_safe=True` |
| `SQLAlchemy` | ≥2.0.0 | SQLite/PostgreSQL metadata backends | Batch operations via bulk inserts/updates; `delete_by_prefix` via SQL LIKE |
| `cachetools` | ≥6.1.0 | In-memory metadata cache layer | No changes needed |

## Feature-Specific Stack Analysis

### 1. Management APIs

**Stack impact: Zero new dependencies.**

| API | Implementation Approach | Stack Used |
|-----|------------------------|------------|
| `update_blob_data()` | Write new blob → update metadata entry → delete old blob | Existing handler pipeline + metadata backends |
| `delete_by_prefix()` | Backend-specific: SQL `LIKE` for SQLite/PG, dict iteration for JSON | SQLAlchemy (existing) for SQL backends |
| `touch()` | Update `accessed_at` timestamp in metadata | Already partially implemented (core.py L2435) |
| `get_metadata()` | Read metadata entry without loading blob | Metadata backend `.get_meta()` — already exists at backend level |
| Batch operations | Loop with optional transaction wrapping | SQLAlchemy sessions (existing) for SQL; dict ops for JSON |

**Key integration point:** `update_blob_data()` must reuse the existing `_write_blob()` → `_store_metadata()` pipeline from `put()`. Extract shared helper methods rather than duplicating logic.

**Batch operations consideration:** For `get_batch()`/`delete_batch()`, SQLite and PostgreSQL can use `IN (...)` clauses for efficiency. The JSON backend will use sequential dict operations. No new dependency needed — SQLAlchemy handles the SQL generation.

### 2. Concurrency Safety

**Stack impact: Zero new dependencies. Stdlib `threading` already imported.**

**Current state:**
- `self._lock = threading.RLock()` exists at core.py L173
- Lock acquired in ~12 management methods (list, delete, clear, etc.)
- Lock **NOT** acquired in `put()` and `get()` — the hot paths
- Backend-level locks: JSON uses `threading.Lock()`, SQLite uses WAL mode

**Recommended approach — configurable thread-safe mode:**

```python
# In CacheConfig or a new ConcurrencyConfig dataclass
thread_safe: bool = False  # Default off for backward compat + zero overhead
```

When `thread_safe=True`, wrap `put()` and `get()` in `with self._lock:`. The `RLock` (reentrant) already handles nested calls (e.g., `put()` → `_write_blob()` → sign entry).

**Why NOT `filelock` or `portalocker`:**
- These provide **cross-process** file locking, which is a different problem
- SQLite already handles multi-process via WAL mode + `PRAGMA busy_timeout`
- JSON backend is explicitly documented as "not safe for concurrency" — file locks won't fix its fundamental O(n²) issue
- Adding cross-process locking would be a significant scope increase for v0.8.0
- Thread-level safety (within a single process) is the stated goal

**Why NOT `asyncio` locks:**
- Async support is explicitly out of scope (separate milestone)
- `threading.RLock` is the correct primitive for synchronous thread safety

**Why `RLock` over `Lock`:**
- Already chosen in existing code (L173)
- Correct: `put()` calls `_sign_entry_if_enabled()` which calls `signer.sign_entry()` — reentrant lock prevents deadlock if internal methods also acquire the lock

**Performance note:** When `thread_safe=False` (default), add zero overhead. Don't use a context manager — use a simple `if self._thread_safe: self._lock.acquire()` pattern or a no-op context manager to avoid runtime cost when disabled.

### 3. HMAC Blob Signing

**Stack impact: Zero new dependencies. Extends existing `hmac` + `hashlib` + `xxhash` usage.**

**Current signing flow:**
1. `CacheEntrySigner.sign_entry()` creates HMAC-SHA256 over metadata fields (security.py)
2. `SignableFields` TypedDict defines the signed field set (interfaces.py)
3. `file_hash` (xxhash digest of blob content) is already a signed field in v2
4. Signing key is stored at `{cache_dir}/.cache_signing_key`

**Current gap:** The `file_hash` in metadata IS signed, but an attacker with filesystem access could:
1. Replace the blob file with malicious content
2. Update the `file_hash` in metadata to match the new blob
3. Re-sign the metadata (if they have the signing key) — but the signing key is on the same filesystem

**Recommended approach — include blob content hash in HMAC payload:**

The `file_hash` field is already included in `SIGNED_FIELDS_BY_VERSION[2]`. The real protection requires:

1. **Verify blob hash on read** — in `get()`, compute xxhash of the blob file and compare to stored `file_hash` before deserialization. This is partially done when `verify_cache_integrity=True` but not on every `get()`.
2. **Add `blob_hmac` field** — a separate HMAC-SHA256 of the raw blob content using the signing key. Stored in metadata alongside `entry_signature`. Verified on `get()` before deserialization.

```python
# New field in metadata entry (computed during put())
blob_hmac = hmac.new(secret_key, blob_bytes, hashlib.sha256).hexdigest()
```

**Why HMAC over just xxhash verification:**
- xxhash is fast but NOT cryptographic — collision-prone against adversaries
- HMAC-SHA256 provides authentication: proves the blob was written by someone with the signing key
- Consistent with existing metadata signing approach

**Performance consideration:** Reading the entire blob to compute HMAC on every `get()` adds I/O. Make it configurable:
- `verify_blob_hmac: bool = True` (in `SecurityConfig` or `CacheConfig`)
- For large blobs (>100MB), consider chunked HMAC computation (HMAC naturally supports incremental `update()`)

**Streaming HMAC for large files:**
```python
h = hmac.new(secret_key, digestmod=hashlib.sha256)
with open(blob_path, "rb") as f:
    while chunk := f.read(8192):
        h.update(chunk)
blob_hmac = h.hexdigest()
```

This uses zero additional memory beyond the 8KB read buffer. `hmac` stdlib module handles this natively.

**Schema version bump:** Add `blob_hmac` to `SignableFields` TypedDict and bump `CURRENT_SIGNATURE_VERSION` to 3. Existing v2 entries without `blob_hmac` continue to verify via backward-compatible version branching (already implemented in `parse_versioned_signature()`).

### 4. Orphaned Blob Prevention

**Stack impact: Zero new dependencies.**

**Current orphan creation pattern:**
```
put() flow:
  1. Write blob to filesystem       ← crash here = orphaned blob
  2. Store metadata entry            ← crash here = orphaned blob
  3. cleanup.commit()                ← after this, entry is consistent
```

`_PutCleanup` (core.py L85) handles normal exception rollback but NOT process kills.

**Approach A — Write-Ahead Intent Log (recommended):**

Use a simple intent file in the cache directory:

```python
# Before writing blob:
intent_path = cache_dir / ".put_intents" / f"{cache_key}.intent"
intent_path.write_text(json.dumps({"key": cache_key, "ts": now}))

# After metadata confirmed:
intent_path.unlink()

# On startup (cleanup_on_init):
# Scan .put_intents/ for stale intents → delete corresponding blobs
```

**Why a file-based intent log:**
- Survives process crashes (unlike in-memory tracking)
- No new dependencies — `pathlib` + `json` (stdlib)
- Atomic file creation is reliable on both Windows and Unix
- Intent files are tiny (<200 bytes) — negligible disk impact
- Compatible with all blob backends (filesystem, S3, in-memory)

**Why NOT SQLite WAL or a separate recovery database:**
- Adds complexity for a simple problem
- JSON backend users don't have SQLite available
- Intent files are simpler and more reliable

**Approach B — Metadata-First Write Order:**

Write metadata first (with a `status: "pending"` field), then write blob, then update metadata to `status: "committed"`. On startup, entries with `status: "pending"` are cleaned up.

**Tradeoff:** Approach B avoids the intent file but changes the write order. The current blob-first order is intentional — it ensures blobs exist before metadata references them. Reversing this creates "dangling metadata" entries (metadata pointing to non-existent blobs) instead of orphaned blobs. Dangling metadata is arguably worse because `get()` would return errors instead of silently wasting disk space.

**Recommendation:** Approach A (intent log). It preserves the current blob-first write order, is lightweight, and handles the crash window cleanly.

**S3 blob backend consideration:** For S3, orphaned objects after crash are cleaned up the same way — the intent file records the S3 URI, and cleanup deletes the remote object. The `_PutCleanup.set_remote()` pattern already tracks remote resources.

## What NOT to Add

| Dependency | Why Not |
|-----------|---------|
| `filelock` / `portalocker` | Cross-process locking is out of scope; SQLite WAL handles multi-process; thread safety is the goal |
| `asyncio` locks | Async support is a separate milestone |
| `aiofiles` | No async file I/O needed |
| `atomicwrites` | `shutil.move` + `NamedTemporaryFile` already provides atomic writes |
| `tenacity` / retry libraries | Retry logic is simple enough to inline (3-5 lines) |
| `cryptography` package | `hmac` + `hashlib` from stdlib is sufficient for HMAC-SHA256 |
| Any database migration tool | Schema changes (adding `blob_hmac` column) handled by existing SQLAlchemy + Alembic-style migrations in metadata backends |

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Thread safety | `threading.RLock` (existing) | `threading.Lock` | RLock already chosen; supports reentrant calls from put→sign |
| Thread safety | Configurable `thread_safe=True` | Always-on locking | Backward compat + zero overhead when disabled |
| Blob HMAC | `hmac.new()` + `hashlib.sha256` (stdlib) | `cryptography.hazmat.primitives.hmac` | Stdlib is sufficient; `cryptography` adds ~30MB dependency |
| Blob integrity | HMAC-SHA256 of blob content | xxhash verification only | xxhash isn't cryptographic — doesn't prove authorship |
| Orphan prevention | File-based intent log | Metadata-first write order | Preserves current blob-first order; avoids dangling metadata |
| Orphan prevention | File-based intent log | SQLite-based WAL | Simpler; works with JSON backend too |
| Batch ops | Sequential with optional SQL transactions | `concurrent.futures` parallelism | I/O bound operations; thread pool adds complexity for marginal gain |

## Configuration Surface

New config fields needed (all in existing `CacheConfig` dataclass tree):

```python
@dataclass
class CacheConfig:
    # Existing fields...
    thread_safe: bool = False           # Enable thread-safe put()/get()

@dataclass
class SecurityConfig:
    # Existing fields...
    sign_blob_content: bool = True      # Compute HMAC over blob content
    verify_blob_hmac: bool = True       # Verify blob HMAC on get()
```

## Installation

```bash
# No new dependencies to install
# Existing install commands unchanged:
uv add cacheness
uv add cacheness[recommended]
```

## Sources

- Python stdlib `hmac` module: HIGH confidence — used in existing `security.py`
- Python stdlib `threading` module: HIGH confidence — `RLock` already in `core.py`
- Existing codebase: `_PutCleanup` pattern (core.py L85-140), `CacheEntrySigner` (security.py), `SignableFields` (interfaces.py)
- `CONCERNS.md` codebase audit: orphaned blob pattern, `_lock` inconsistency, signing gaps
- xxhash non-cryptographic nature: HIGH confidence — documented in xxhash README as "extremely fast non-cryptographic hash"

## Confidence Assessment

| Area | Confidence | Reason |
|------|------------|--------|
| Management APIs | HIGH | Pure API additions; no new tech needed; patterns exist in codebase |
| Concurrency safety | HIGH | `threading.RLock` already in use; configurable extension is straightforward |
| HMAC blob signing | HIGH | `hmac` + `hashlib` stdlib; extends existing `CacheEntrySigner` pattern |
| Orphaned blob prevention | MEDIUM | Intent-log approach is sound but needs careful testing for Windows file semantics and S3 failure modes |
| No new deps needed | HIGH | All four features use stdlib or existing dependencies |
