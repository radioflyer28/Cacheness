# Architecture Patterns — v0.8.0 API & Robustness Integration

**Domain:** Python disk caching library — management APIs, concurrency, signing, crash safety
**Researched:** 2026-04-02
**Overall confidence:** HIGH (all analysis based on direct codebase inspection of current `dev` branch)

## Executive Summary

The v0.8.0 milestone adds four capabilities to an already well-decomposed codebase: management APIs, concurrency safety, HMAC blob signing, and orphaned blob prevention. Each feature integrates at clearly defined points in the existing architecture without requiring structural refactors. The key insight is that **these features are behavioral additions, not structural changes** — unlike v0.7.0 which reorganized files, v0.8.0 extends existing classes and adds new methods/fields.

The four features have a strict dependency chain:

1. **Orphaned blob prevention** (independent — `_PutCleanup` + intent journal)
2. **Concurrency safety** (independent — `_lock` in `put()`/`get()`, already partially in place)
3. **HMAC blob signing** (depends on blob hashing infra, already exists via `_calculate_file_hash`)
4. **Management APIs** (depends on all three above for safe batch operations)

Build order should follow this chain: orphan prevention and concurrency can run in parallel, then blob signing, then management APIs last.

## Current Architecture (Post v0.7.0)

### Layer Diagram

```
┌─────────────────────────────────────────────────────┐
│                    Public API                       │
│  UnifiedCache (core.py ~2500 lines)                 │
│  ├─ VerificationMixin   (_verification_mixin.py)    │
│  ├─ StatsMixin          (_stats_mixin.py)           │
│  ├─ CustomMetadataMixin (_custom_metadata_mixin.py) │
│  └─ StorageModeMixin    (_storage_mode_mixin.py)    │
├─────────────────────────────────────────────────────┤
│               Delegation Layer                      │
│  BlobStore (storage/blob_store.py)                  │
│  ├─ _write_blob(), _read_blob()                     │
│  ├─ verify_integrity()                              │
│  └─ delete(), clear()                               │
├─────────────┬───────────────────────────────────────┤
│  Metadata   │          Blob Backends                │
│  Backends   │  ┌─ FilesystemBlobBackend             │
│ ┌─ JSON     │  ├─ S3BlobBackend                     │
│ ├─ SQLite   │  └─ MemoryBlobBackend                 │
│ └─ Postgres │                                       │
├─────────────┴───────────────────────────────────────┤
│  Cross-cutting: CacheEntrySigner (security.py)      │
│  Cross-cutting: HandlerRegistry (handlers/)         │
│  Cross-cutting: _PutCleanup (core.py)               │
│  Cross-cutting: CacheConfig (config.py)             │
└─────────────────────────────────────────────────────┘
```

### Key Shared Resources

| Resource | Owner | Shared With | Notes |
|----------|-------|-------------|-------|
| `_lock` (RLock) | `UnifiedCache.__init__` | `BlobStore._lock` (same object) | Reentrant; acquired in `put()`, `get()`, and ~18 other methods |
| `signer` | `UnifiedCache.__init__` | `BlobStore.signer` (same object) | `CacheEntrySigner` or `None` |
| `handlers` | `UnifiedCache.__init__` | `BlobStore.handlers` (same object) | `HandlerRegistry` with priority-based selection |
| `metadata_backend` | `UnifiedCache.__init__` | `BlobStore.backend` (same object) | JSON/SQLite/PostgreSQL backend |
| `config` | `UnifiedCache.__init__` | `BlobStore.config` (same object) | `CacheConfig` dataclass |

---

## Feature 1: Management APIs

### Integration Points

**Where they go:** Most management APIs already exist in `core.py` directly on `UnifiedCache`. The v0.8.0 additions fit naturally as peer methods:

| API | Status | Location | Notes |
|-----|--------|----------|-------|
| `update_data()` | ✅ EXISTS | `core.py:2219` | Replaces blob at existing key |
| `touch()` | ✅ EXISTS | `core.py:2435` | Resets `created_at` to extend TTL |
| `get_metadata()` | ✅ EXISTS | `core.py:1545` | Returns entry metadata without loading blob |
| `delete_where()` | ✅ EXISTS | `core.py:2521` | Filter-based deletion |
| `delete_matching()` | ✅ EXISTS | `core.py:2582` | Kwargs-based deletion |
| `delete_batch()` | ✅ EXISTS | `core.py:2678` | List-of-kwargs batch delete |
| `touch_batch()` | ✅ EXISTS | `core.py:2717` | Filter-based batch touch |
| `update_blob_data()` | ❌ MISSING | → `core.py` | Alias/enhancement of `update_data()` |
| `delete_by_prefix()` | ❌ MISSING | → `core.py` | Prefix-based deletion |
| Batch `put()` | ❌ MISSING | → `core.py` | Store multiple items in one call |
| Batch `get()` | ❌ MISSING | → `core.py` | Retrieve multiple items in one call |

### Architectural Decision: Mixin vs. Inline

**Recommendation: Add new management APIs directly to `core.py`** rather than creating a new mixin.

Rationale:
- The existing management methods (`update_data`, `touch`, `delete_where`, `delete_batch`, `touch_batch`) are all inline in `core.py`
- These methods are tightly coupled to `put()`/`get()` internals (`_resolve_cache_key`, `_build_metadata_dict`, `_PutCleanup`)
- Extracting to a mixin would require exposing many private methods via `self`
- The v0.7.0 mixin extractions worked because those concerns were self-contained (verification, stats, custom metadata, storage mode). Management ops are not — they're variants of the put/get/delete core loop.

### Batch Operations Pattern

Batch `put()` and `get()` should follow the existing `delete_batch()` pattern: iterate over a list of operations, delegating each to the single-item method. The lock is per-operation (already RLock/reentrant), not per-batch.

```python
def put_batch(self, items: List[Dict[str, Any]]) -> List[str]:
    """Store multiple items. Each dict has 'data' + key params."""
    keys = []
    for item in items:
        data = item.pop("data")
        key = self.put(data, **item)  # Reuses existing put()
        keys.append(key)
    return keys
```

### `delete_by_prefix()` Integration

Requires scanning metadata entries. Two paths:

1. **SQLite/PostgreSQL fast path:** `WHERE cache_key LIKE '{prefix}%'` — add `delete_by_prefix()` to `MetadataBackend` ABC
2. **Generic fallback:** `iter_entry_summaries()` + Python-side prefix filtering + per-entry `invalidate()`

This follows the same pattern as `query_meta()` which already has SQLite fast path + generic fallback.

### New Components

None. All management APIs are methods on existing `UnifiedCache`.

### Modified Components

| Component | Change | Scope |
|-----------|--------|-------|
| `core.py` | Add `update_blob_data()`, `delete_by_prefix()`, `put_batch()`, `get_batch()` | ~100-150 lines |
| `metadata/base.py` | Add `delete_by_prefix()` to ABC (optional override) | ~10 lines |
| `metadata/sqlite_backend.py` | `delete_by_prefix()` SQL fast path | ~20 lines |
| `metadata/json_backend.py` | `delete_by_prefix()` Python fallback | ~15 lines |

---

## Feature 2: Concurrency Safety

### Current State

The `_lock` (threading.RLock) **is** acquired in `put()` and `get()` — see `core.py:1168` (`with self._lock:` at the top of `put()`) and `core.py:1306` (`with self._lock:` at the top of `get()`). The CONCERNS.md note about `_lock` not being acquired in put/get was from a pre-v0.7.0 audit and was addressed during v0.7.0.

Current locking status:
- `put()`: ✅ Holds `_lock` for entire operation (blob write + metadata write)
- `get()`: ✅ Holds `_lock` for entire operation (metadata read + blob read)
- `update_data()`: ✅ Holds `_lock`
- `touch()`: ✅ Holds `_lock`
- `delete_where()`: ✅ Holds `_lock`
- `invalidate()`: ✅ Holds `_lock`
- `BlobStore.put()`: ✅ Holds `_lock`
- `BlobStore.get()`: ✅ Holds `_lock`

**The lock is a coarse-grained mutex** — the entire operation executes under a single RLock. This is correct for single-process thread safety but means:
1. No concurrent reads (get() blocks other get() calls)
2. No concurrent writes (put() blocks other put() calls)
3. The RLock allows re-entrant calls (e.g., `delete_where` → `invalidate`)

### What Needs to Change

**Minimal changes needed.** The lock is already in place. The v0.8.0 scope should focus on:

1. **Document the threading model** — the lock provides serialized access, not fine-grained concurrency. This is appropriate for a disk-caching library.

2. **Ensure new management APIs acquire the lock** — any new methods (`delete_by_prefix`, `put_batch`, `get_batch`) must wrap in `with self._lock:`

3. **Consider read-write lock upgrade (optional)** — replace `threading.RLock()` with a `ReadWriteLock` to allow concurrent `get()` calls while serializing `put()` calls. This is a performance optimization, not a correctness fix.

### Read-Write Lock Architecture (Optional Enhancement)

```python
class _ReadWriteLock:
    """Simple readers-writer lock using threading primitives."""
    def __init__(self):
        self._readers = 0
        self._readers_lock = threading.Lock()
        self._writer_lock = threading.Lock()
    
    @contextmanager
    def read(self):
        with self._readers_lock:
            self._readers += 1
            if self._readers == 1:
                self._writer_lock.acquire()
        try:
            yield
        finally:
            with self._readers_lock:
                self._readers -= 1
                if self._readers == 0:
                    self._writer_lock.release()
    
    @contextmanager
    def write(self):
        self._writer_lock.acquire()
        try:
            yield
        finally:
            self._writer_lock.release()
```

**Where it would plug in:**
- `get()`, `get_metadata()`, `exists()`, `list_entries()` → `with self._lock.read():`
- `put()`, `update_data()`, `touch()`, `invalidate()`, `clear()` → `with self._lock.write():`

**Risk:** Breaking re-entrant calls. The current RLock allows `delete_where` → `invalidate` because RLock is reentrant. A ReadWriteLock would deadlock if a write-locked method calls another write-locked method. This requires auditing all call chains.

**Recommendation:** Keep the RLock for v0.8.0 correctness. Document the threading model. Consider ReadWriteLock for a future performance milestone if profiling shows lock contention.

### New Components

None (unless ReadWriteLock is pursued).

### Modified Components

| Component | Change | Scope |
|-----------|--------|-------|
| `core.py` | Ensure new methods acquire `_lock` | Already pattern-established |
| `docs/` | Document threading model, concurrency boundaries | New doc section |

---

## Feature 3: HMAC Blob Signing

### Current Signing Architecture

`CacheEntrySigner` (security.py) currently signs **metadata fields only**:

```
Signed fields (v2): cache_key, data_type, file_size, file_hash, object_type,
                     storage_format, serializer, compression_codec, created_at
```

The `file_hash` field (XXH3_64 of blob content) is already included in the signature. This means:
- If someone modifies the blob → file_hash changes → signature becomes invalid
- **BUT** this only works when `verify_cache_integrity=True` (default: True) and the `file_hash` was computed at write time

**The gap:** The file_hash is a non-cryptographic hash (XXH3_64). While fast, it's not collision-resistant against adversarial tampering. An attacker with filesystem access could craft a blob that produces the same XXH3_64 hash.

### Integration Approach: Extend Existing Signing

**Recommendation: Add an HMAC-SHA256 of blob content as a new signed field, not a replacement of file_hash.**

1. **New field:** `blob_hmac` — HMAC-SHA256 of the blob's raw bytes, using the same signing key as entry signatures
2. **Stored in:** metadata dict (alongside `file_hash`, `entry_signature`)
3. **Computed during:** `put()` and `update_data()`, after blob is written
4. **Verified during:** `_verify_entry()`, before blob is deserialized

### CacheEntrySigner Changes

```python
# New signature version (v3) includes blob_hmac in signed fields
SIGNED_FIELDS_BY_VERSION = {
    1: [...],  # legacy
    2: [...],  # current
    3: [       # v0.8.0 — adds blob_hmac
        "cache_key", "data_type", "file_size", "file_hash",
        "blob_hmac",  # NEW: cryptographic blob content hash
        "object_type", "storage_format", "serializer",
        "compression_codec", "created_at",
    ],
}
CURRENT_SIGNATURE_VERSION = 3
```

Add new methods for computing/verifying the blob HMAC:

```python
def compute_blob_hmac(self, blob_content: bytes) -> str:
    """Compute HMAC-SHA256 of blob content using the signing key."""
    return hmac.new(
        self.secret_key, blob_content, hashlib.sha256
    ).hexdigest()

def verify_blob_hmac(self, blob_content: bytes, stored_hmac: str) -> bool:
    """Verify HMAC-SHA256 of blob content."""
    expected = self.compute_blob_hmac(blob_content)
    return hmac.compare_digest(expected, stored_hmac)
```

### Blob Backend Abstraction Concern

**Problem:** To compute HMAC over blob content, we need to read the blob bytes. For local files, this is trivial (`Path.read_bytes()`). For S3 blobs, this requires downloading the entire object.

**Solution:** Compute the HMAC **at write time** when the data is already in memory (handler just serialized it). Don't re-read the blob for HMAC computation.

Integration in `put()` flow:
1. Handler serializes data → writes to local file
2. `_calculate_file_hash()` computes XXH3_64 (fast integrity check)
3. **NEW:** If signing enabled, compute `blob_hmac` from the same bytes
4. Blob backend persists the file (S3 upload, filesystem rename)
5. Metadata written with `file_hash` + `blob_hmac` + `entry_signature`

Integration in `get()` flow:
1. Metadata read, signature verified (includes `blob_hmac` in v3)
2. Blob loaded from backend
3. **NEW:** If `blob_hmac` present, verify it against loaded bytes
4. Handler deserializes data

**Where to compute HMAC at write time:**

In `BlobStore._write_blob()`: after `handler.put()` writes to local staging path, read the bytes for HMAC before `blob_backend.write_blob_from_path()` moves the file. This is the only point where the serialized bytes are on the local filesystem regardless of the final backend.

```python
# In _write_blob(), after handler.put():
blob_hmac = None
if self.signer and compute_hash:
    handler_path = Path(result.actual_path)
    blob_bytes = handler_path.read_bytes()
    blob_hmac = self.signer.compute_blob_hmac(blob_bytes)
```

**Where to verify HMAC at read time:**

In `_verify_entry()` (VerificationMixin), after hash verification and before signature verification:

```python
# After file_hash check, before signature check:
if self.signer and metadata.get("blob_hmac"):
    blob_bytes = self._blob_store.blob_backend.read_blob(str(file_path))
    if not self.signer.verify_blob_hmac(blob_bytes, metadata["blob_hmac"]):
        # Handle failure (delete or warn based on config)
```

**Performance concern:** HMAC verification reads the entire blob. For large blobs (100MB+ DataFrames), this adds latency. Consider making blob HMAC verification configurable via `SecurityConfig.verify_blob_hmac: bool = True`.

### Backward Compatibility

- Entries signed with v1/v2 have no `blob_hmac` field → continue working (verification skips HMAC check if field absent)
- New entries are signed with v3 → includes `blob_hmac` in signature
- `verify_entry()` already handles version-based field selection via `parse_versioned_signature()`

### Inline Blob Consideration

When `is_inline=True`, the blob data is stored in the metadata database (`blob_data` column). The HMAC should be computed from the `blob_data` bytes. The `_verify_entry()` code already handles inline blobs for XXH3_64 verification — the same pattern applies for HMAC.

### New Components

None. All changes are extensions to existing `CacheEntrySigner` and `VerificationMixin`.

### Modified Components

| Component | Change | Scope |
|-----------|--------|-------|
| `security.py` | Add v3 signed fields, `compute_blob_hmac()`, `verify_blob_hmac()` | ~30 lines |
| `config.py` | Add `SecurityConfig.verify_blob_hmac: bool = True` | ~3 lines |
| `_verification_mixin.py` | Add HMAC verification step in `_verify_entry()` | ~20 lines |
| `storage/blob_store.py` | Compute `blob_hmac` in `_write_blob()`, return in `WriteBlobResult` | ~10 lines |
| `core.py` | Store `blob_hmac` in metadata dict during `put()`, `update_data()` | ~10 lines |
| `interfaces.py` | Add `blob_hmac` to `SignableFields` TypedDict | ~1 line |

---

## Feature 4: Orphaned Blob Prevention

### Current State

**`_PutCleanup` (core.py:82-137):** A lightweight context manager that tracks blob paths written during `put()`. On exception, it deletes the orphaned blob (local file + S3 object). On success, `commit()` disarms it. This handles application-level exceptions cleanly.

**Gap:** Process kills (SIGKILL, power failure, OOM killer) between blob write and metadata write leave orphaned blobs. `_PutCleanup.rollback()` never executes. The only recovery is `verify_integrity(repair=True)`.

### Intent Journal Architecture

**Recommendation: Write-ahead intent journal** — before writing a blob, record the intent in a lightweight journal file. After metadata is committed, remove the journal entry. On startup, replay incomplete journal entries to clean up orphans.

```
Write flow:
  1. Write intent → journal  (blob_id, timestamp)     ← crash here = journal entry, no blob
  2. Handler serializes → local file                   ← crash here = journal + orphan blob
  3. Blob backend persists (S3 upload, rename)         ← crash here = journal + orphan blob
  4. Metadata write                                    ← crash here = journal + orphan blob
  5. Remove intent from journal                        ← crash here = committed entry + stale journal

Recovery (on init):
  1. Read journal entries older than threshold (e.g., 60 seconds)
  2. For each stale entry:
     a. If blob exists but no metadata → delete blob (orphan)
     b. If metadata exists → remove journal entry (write succeeded)
     c. If neither exists → remove journal entry (blob write never happened)
```

### Journal Implementation

**Location:** `{cache_dir}/.write_journal` (single append-only file)

**Format:** One line per intent, newline-delimited:
```
{blob_id}\t{timestamp_epoch}\t{namespace}\n
```

**Why not SQLite for journal?** The journal must survive even when the SQLite metadata database is corrupted. A flat file with append-only writes is more crash-resilient than SQLite (no WAL, no page management). It's also simpler and avoids circular dependency with the metadata backend.

### Integration with `_PutCleanup`

Extend `_PutCleanup` to manage journal entries:

```python
class _PutCleanup:
    def __init__(self, journal: Optional[WriteJournal] = None):
        self._journal = journal
        self._journal_entry_id: Optional[str] = None
        # ... existing fields ...
    
    def record_intent(self, blob_id: str, namespace: str):
        """Write intent to journal before blob creation."""
        if self._journal:
            self._journal_entry_id = self._journal.record(blob_id, namespace)
    
    def commit(self):
        """Disarm cleanup and remove journal entry."""
        self._committed = True
        if self._journal and self._journal_entry_id:
            self._journal.remove(self._journal_entry_id)
    
    def rollback(self):
        """Clean up blob + remove journal entry."""
        # ... existing rollback logic ...
        if self._journal and self._journal_entry_id:
            self._journal.remove(self._journal_entry_id)
```

### Startup Recovery

Add recovery to `UnifiedCache.__init__()`, after metadata backend and blob store initialization but before `cleanup_on_init`:

```python
def __init__(self, ...):
    # ... existing init ...
    self._write_journal = WriteJournal(self.cache_dir / ".write_journal")
    self._write_journal.recover(self._blob_store, self.metadata_backend)
    # ... cleanup_on_init ...
```

### New Components

| Component | Purpose |
|-----------|---------|
| `write_journal.py` | `WriteJournal` class: `record()`, `remove()`, `recover()` |

**Location:** `src/cacheness/write_journal.py` (sibling to `core.py`, not inside `storage/` since it's a cache-layer concern, not a storage-layer concern).

### Modified Components

| Component | Change | Scope |
|-----------|--------|-------|
| `core.py` | `_PutCleanup` gains journal awareness; `__init__` creates journal + runs recovery | ~30 lines |
| `config.py` | Add `CacheStorageConfig.enable_write_journal: bool = True` | ~3 lines |

---

## Component Boundaries

### New vs. Modified Components Summary

| Component | Status | Feature |
|-----------|--------|---------|
| `write_journal.py` | **NEW** | Orphan prevention |
| `core.py` | MODIFIED | Management APIs, concurrency, orphan prevention |
| `security.py` | MODIFIED | Blob HMAC signing |
| `_verification_mixin.py` | MODIFIED | Blob HMAC verification |
| `storage/blob_store.py` | MODIFIED | Blob HMAC computation at write time |
| `config.py` | MODIFIED | New config flags |
| `interfaces.py` | MODIFIED | `blob_hmac` in `SignableFields` |
| `metadata/base.py` | MODIFIED | `delete_by_prefix()` in ABC |
| `metadata/sqlite_backend.py` | MODIFIED | `delete_by_prefix()` SQL fast path |
| `metadata/json_backend.py` | MODIFIED | `delete_by_prefix()` fallback |

### Data Flow Changes

**`put()` flow — additions in bold:**

```
put(data, key, ...) 
  → _lock.acquire()
  → _resolve_cache_key()
  → **journal.record_intent(blob_id)**      ← NEW
  → _PutCleanup()
  → handler.put() → local file
  → **signer.compute_blob_hmac(bytes)**      ← NEW  
  → blob_backend.write_blob_from_path()
  → _calculate_file_hash()
  → _build_metadata_dict()
  → **metadata["blob_hmac"] = hmac**         ← NEW
  → _sign_entry_if_enabled()
  → metadata_backend.put_entry()
  → cleanup.commit()
  → **journal.remove(entry_id)**             ← NEW
  → _lock.release()
```

**`get()` flow — additions in bold:**

```
get(key, ...)
  → _lock.acquire()
  → metadata_backend.get_entry()
  → _is_expired()
  → _verify_entry()
      → file_hash check (existing)
      → **blob_hmac check**                   ← NEW
      → entry_signature check (existing)
  → _read_blob()
  → update_access_time()
  → _lock.release()
```

---

## Suggested Build Order

Based on dependency analysis and risk assessment:

### Phase 1: Orphaned Blob Prevention (independent)
- Create `write_journal.py`
- Extend `_PutCleanup` with journal awareness
- Add recovery logic in `__init__`
- Config flag: `enable_write_journal`
- **Risk:** Low — additive. Journal writes are non-blocking to the main flow. Recovery is best-effort.
- **Tests:** Simulate crash between blob write and metadata write. Verify recovery on reinit.

### Phase 2: Concurrency Safety (independent, can parallel with Phase 1)
- Audit all new methods for `_lock` usage (should already be covered)
- Document threading model
- Add concurrency stress tests
- **Risk:** Very low — the lock is already in place. This is primarily documentation + test coverage.
- **Tests:** Thread pool exercising concurrent put/get/delete. Verify no data races.

### Phase 3: HMAC Blob Signing (after Phases 1-2 provide stable base)
- Add `compute_blob_hmac()` / `verify_blob_hmac()` to `CacheEntrySigner`
- Add signature v3 with `blob_hmac` field
- Compute HMAC in `_write_blob()` 
- Verify HMAC in `_verify_entry()`
- Config flag: `verify_blob_hmac`
- **Risk:** Medium — backward compatibility with v1/v2 entries. Must NOT break verification of existing unsigned or v2-signed entries.
- **Tests:** Sign with v3, verify v3. Verify v2 entries still load. Tamper blob → verify fails.

### Phase 4: Management APIs (after all above are stable)
- `update_blob_data()` — thin wrapper/alias for `update_data()` with clearer naming
- `delete_by_prefix()` — with SQLite fast path
- `put_batch()` / `get_batch()` — iteration over single-item methods
- **Risk:** Low — these are convenience wrappers over existing operations.
- **Tests:** Standard CRUD tests + backend parity tests across JSON/SQLite/PostgreSQL.

## Patterns to Follow

### Pattern 1: Backend Fast Path + Generic Fallback
**What:** SQLite/PostgreSQL-specific SQL optimization with Python-side fallback for JSON backend.
**When:** Any new metadata query or bulk operation.
**Existing example:** `query_meta()` in `core.py:420-490`

### Pattern 2: Config-Gated Behavior
**What:** New behaviors behind config flags with sensible defaults.
**When:** Any behavioral change that affects performance or compatibility.
**Existing example:** `SecurityConfig.enable_entry_signing`, `CacheMetadataConfig.verify_cache_integrity`

### Pattern 3: _PutCleanup for Write Atomicity
**What:** Track resources during writes, roll back on failure.
**When:** Any new write operation.
**Existing example:** `_PutCleanup` in `core.py:82-137`, used in `put()` and `update_data()`

### Pattern 4: Mixin for Self-Contained Concerns
**What:** Extract to mixin when concern has clear boundaries and limited coupling to core.
**When:** Concern has 5+ methods, doesn't need access to put/get internals.
**Existing example:** `VerificationMixin`, `StatsMixin`, `CustomMetadataMixin`, `StorageModeMixin`
**Not applicable for:** Management APIs (too coupled to put/get internals)

## Anti-Patterns to Avoid

### Anti-Pattern 1: Holding Lock During I/O
**What:** Acquiring `_lock` and then performing network I/O (S3 upload/download).
**Why bad:** A slow S3 upload blocks all other cache operations.
**Mitigation:** For v0.8.0, accept this (coarse-grained locking). For future, consider lock-per-key or async I/O.

### Anti-Pattern 2: HMAC Verification Requiring Full Blob Download
**What:** Verifying `blob_hmac` on every `get()` for S3 blobs downloads the entire object just to hash it.
**Why bad:** Doubles latency for remote blobs; blob is then downloaded again for deserialization.
**Mitigation:** For remote blobs with S3 ETag verification, skip `blob_hmac` check (ETag already confirms integrity). Only verify `blob_hmac` for local filesystem blobs.

### Anti-Pattern 3: Journal Becoming a Performance Bottleneck
**What:** Synchronous fsync on every journal write.
**Why bad:** Adds disk I/O latency to every `put()` call.
**Mitigation:** Use append-only writes without fsync. The journal is best-effort — if it's lost in a crash, `verify_integrity(repair=True)` is the fallback. The journal reduces orphans; it doesn't need to eliminate them absolutely.

## Scalability Considerations

| Concern | Current (v0.7.0) | v0.8.0 | Future |
|---------|-------------------|--------|--------|
| Concurrent access | Serialized (RLock) | Serialized (RLock) + documented | ReadWriteLock or lock-per-key |
| Blob integrity | XXH3_64 (fast, non-crypto) | XXH3_64 + HMAC-SHA256 (crypto) | Same |
| Crash recovery | `verify_integrity(repair=True)` | Intent journal + verify_integrity | WAL-based recovery |
| Batch operations | Per-item iteration | Per-item iteration (documented) | Bulk metadata ops |

## Sources

- Direct codebase inspection of `src/cacheness/` on `dev` branch (2026-04-02, post-v0.7.0)
- `core.py` (~2500 lines), `security.py` (~450 lines), `storage/blob_store.py` (~1080 lines), `storage/backends/blob_backends.py` (~500 lines)
- `_verification_mixin.py`, `_storage_mode_mixin.py` (mixin pattern reference)
- `config.py` (SecurityConfig, CacheStorageConfig, CacheMetadataConfig)
- `.planning/codebase/CONCERNS.md` (known issues from v0.7.0 audit)
- `.planning/PROJECT.md` (v0.8.0 milestone requirements)
