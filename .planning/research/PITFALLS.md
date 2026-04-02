# Domain Pitfalls — Cacheness v0.8.0 API & Robustness

**Domain:** Adding management APIs, concurrency safety, HMAC blob signing, and orphaned blob prevention to an existing Python disk caching library
**Researched:** 2026-04-02
**Overall confidence:** HIGH (pitfalls derived from actual codebase analysis of core.py, security.py, blob_store.py, json_backend.py, sqlite_backend.py, and the v0.7.0 CONCERNS.md audit)

---

## Critical Pitfalls

Mistakes that cause data loss, security bypass, or widespread test regressions. Any of these requires a rewrite or emergency patch if shipped.

---

### P1: HMAC Blob Signing — Signing Before Write Completion Creates a TOCTOU Gap

**What goes wrong:** The natural instinct is to compute the HMAC over blob bytes during `put()`, right after serialization. But if blob signing happens *before* the blob is persisted to disk (or S3), a crash between HMAC computation and blob write leaves a metadata entry with a valid signature pointing to a blob that never landed. Worse: if the write partially succeeds (truncated file), the HMAC will verify against the *intended* bytes, not the *actual* bytes on disk.

**Why it happens:** The current `put()` flow is: serialize → write blob → write metadata. Adding HMAC naturally slots in after serialize but the bytes in memory may not match what's on disk (compression, temp-file rename, S3 multipart upload with eventual consistency).

**Consequences:**
- Silent data corruption: `get()` reads truncated/corrupted blob, HMAC passes because it was computed from the in-memory serialization, not from what's on disk.
- False sense of security: users trust HMAC verification but are only verifying serialization output, not storage fidelity.

**Prevention:**
1. **Compute HMAC from the persisted blob, not the in-memory bytes.** After `_write_blob()` or `blob_backend.write_blob_from_path()` succeeds, read the file back (or use the xxhash already computed by `_write_blob`) and sign *that*.
2. **For inline blobs**, sign the `blob_data` bytes directly — they *are* the persisted form.
3. **For S3 blobs**, the ETag serves as a partial integrity check. Consider storing both the blob HMAC (computed pre-upload) and the S3 ETag (received post-upload) and requiring both to verify.
4. **Add a `blob_signature` field** to metadata separate from `entry_signature`. Don't mix blob signing into the existing signature versioning — it's a different concern.

**Detection:** Test with fault injection: patch `blob_backend.write_blob_from_path` to truncate the output, verify that HMAC catches the mismatch.

**Phase:** Blob HMAC signing phase. Must be designed into the signing architecture, not bolted on after.

---

### P2: Concurrency — JSON Backend `_save_to_disk()` Is Globally Unsafe Under Concurrent Writers

**What goes wrong:** `JsonBackend._save_to_disk()` does read-modify-write of the entire JSON file on every `put_entry()`. Even though it uses `threading.RLock()` for in-process thread safety, **multiple processes** (or multiple `UnifiedCache` instances in the same process pointing to the same `cache_dir`) will race: Process A reads metadata, Process B reads metadata, Process A writes, Process B writes — Process A's entry is lost.

**Why it happens:** The JSON backend has no file-level locking. The RLock only protects within a single Python process. The `shutil.move` atomic-rename pattern prevents *corruption* (partial writes) but not *lost updates*.

**Consequences:**
- Silent metadata loss: entries vanish without errors or warnings.
- Intermittent failures: only manifests under concurrent load, hard to reproduce.

**Prevention:**
1. **Document JSON backend as single-process only.** The CONCERNS.md already notes "JSON <200 entries, NOT safe for concurrency" — this should be a loud warning in docs and possibly a runtime warning if concurrent access is detected.
2. **Do NOT add concurrency safety to the JSON backend.** Adding `fcntl.flock()`/`msvcrt.locking()` would be cross-platform painful and still has edge cases. The correct answer is: use SQLite for concurrent access.
3. **For the concurrency safety phase**, focus on SQLite (WAL mode already enables concurrent reads) and PostgreSQL (inherently concurrent). JSON gets a "not supported" doc label and maybe a runtime warning.
4. **Management APIs (`delete_by_prefix`, batch ops) must not assume safe concurrent access on JSON.** If a batch delete iterates entries and calls `_save_to_disk()` per entry, the O(n²) penalty is bad enough — concurrent modifications make it catastrophic.

**Detection:** Test with `multiprocessing.Pool` using JSON backend; assert all written entries are still present after concurrent writes.

**Phase:** Concurrency safety phase. Decision: exclude JSON from concurrency guarantees.

---

### P3: Orphaned Blob Prevention — WAL/Journal Approach Adds Init Latency and Windows Complexity

**What goes wrong:** The intuitive approach to preventing orphaned blobs is a write-ahead log (WAL): before writing a blob, log the intent; after metadata is committed, mark the WAL entry as complete; on init, replay incomplete WAL entries and clean up orphaned blobs. But this adds:
- **Startup latency**: scanning the WAL on every `__init__` call (the `cleanup_on_init` pattern already exists for TTL cleanup, but WAL replay adds disk I/O per incomplete entry).
- **Windows file locking**: WAL files need to be opened and held during writes. Windows exclusive file locking means a crash can leave the WAL file locked, preventing the *next* process from opening it.
- **S3 eventual consistency**: WAL says "blob was written to S3" but S3 may not have propagated the write yet. Deleting based on WAL staleness may delete a blob that's still propagating.

**Why it happens:** The current `_PutCleanup` class handles in-process exception rollback beautifully, but it's powerless against `kill -9`/power loss. The gap between blob-write and metadata-write is the vulnerability window.

**Consequences:**
- Orphaned blobs accumulate silently, consuming disk space.
- WAL replay on init slows down cache startup, especially with many incomplete entries.
- On Windows, locked WAL files can prevent cache initialization entirely.

**Prevention:**
1. **Reverse the write order for non-inline blobs**: write metadata first (with a "pending" flag), then write the blob, then clear the "pending" flag. On init, entries with "pending" flag and no blob → delete metadata. This is *metadata-first* instead of *blob-first*. Metadata backends (SQLite, PG) are transactional, so the metadata write is atomic.
2. **Alternative: extend `verify_integrity(repair=True)` to run as part of init when `cleanup_on_init=True`.** This already detects orphaned blobs. The only missing piece is making it fast enough for init (currently scans all blobs). A heuristic: only scan for blobs modified in the last N seconds.
3. **Do NOT implement file-based WAL.** Use the metadata backend itself as the write-ahead log (it's already a database). A "pending" column or table is simpler, cross-platform, and uses existing transactional guarantees.
4. **For S3**: blobs uploaded to S3 get an ETag back. Store the ETag in metadata as confirmation of write completion. On init, entries without ETag → verify via HEAD request or mark for re-upload.

**Detection:** Test with `os.kill(os.getpid(), signal.SIGKILL)` in a subprocess during `put()` to simulate crash. Verify that the next `__init__` cleans up correctly.

**Phase:** Orphaned blob prevention phase. Design decision needed before implementation.

---

### P4: Management APIs — `delete_by_prefix()` + Signing Creates an O(n) Re-verification Trap

**What goes wrong:** When implementing `delete_by_prefix()`, the obvious approach is: query entries matching the prefix, then delete each one. But if HMAC signing is enabled, each `remove_entry()` call must *not* re-sign remaining entries — only the *deleted* entries' signatures need to be invalidated. The trap: if the implementation naively calls `put_entry()` (which triggers signing) during the batch operation for any reason (e.g., updating stats, touching a counter), every remaining entry gets re-signed, turning an O(k) delete into O(n) where n is total entries.

**Why it happens:** The existing `put()` flow always signs on write. Management APIs that modify metadata (touch, update, batch delete) may inadvertently trigger the signing path.

**Consequences:**
- `delete_by_prefix("experiment_")` on a cache with 10,000 entries takes minutes instead of milliseconds.
- CPU spike during bulk operations.

**Prevention:**
1. **Implement management APIs at the metadata backend level**, not through repeated `put()`/`get()` calls. `delete_by_prefix()` should be a single SQL `DELETE WHERE cache_key LIKE ?` operation on SQLite/PostgreSQL, with a Python-side loop only for JSON.
2. **Signing should only happen on `put_entry()`, not on `remove_entry()` or `update_entry_metadata()`.** Verify this invariant with a test.
3. **Batch operations should hold the lock for the entire batch**, not per-item. This prevents interleaving with concurrent writers and avoids lock acquisition overhead.
4. **For JSON backend**: batch deletes should modify the in-memory dict and call `_save_to_disk()` once at the end, not per entry.

**Detection:** Benchmark `delete_by_prefix()` with 1,000 matching entries and 9,000 non-matching entries. Should complete in <100ms on SQLite.

**Phase:** Management APIs phase. Architecture decision: backend-level vs. cache-level implementation.

---

## Moderate Pitfalls

Mistakes that cause subtle bugs, performance degradation, or maintenance burden, but don't require immediate rewrites.

---

### P5: Concurrency — RLock Is Reentrant but NOT Multiprocess-Safe

**What goes wrong:** `UnifiedCache._lock` is a `threading.RLock()`. This protects against concurrent access from *threads* in the same process. But Python `multiprocessing` creates separate address spaces — the RLock is not shared. Two processes using the same `cache_dir` will have independent locks that don't coordinate.

**Why it happens:** The current code already correctly uses RLock (not Lock) to allow re-entrant calls like `delete_where → invalidate`. But the docs and concurrency phase may create a false expectation that "thread-safe mode" means "multiprocess-safe".

**Consequences:**
- Users who read "concurrency safe" may assume multi-process safety and hit silent data races.
- SQLite's WAL mode provides *database-level* concurrency but blob file operations (write, delete) are still uncoordinated across processes.

**Prevention:**
1. **Clearly separate "thread-safe" from "process-safe" in documentation.** Thread-safe = multiple threads in one process. Process-safe = multiple processes sharing a cache_dir.
2. **SQLite backend is inherently process-safe** for metadata operations (WAL + busy_timeout). Document this.
3. **Blob file operations are process-safe by accident** (each blob has a unique filename, overwrite is atomic via rename). Document this as "safe but not guaranteed."
4. **PostgreSQL is fully process-safe** by design. Document this.
5. **JSON is neither thread-safe between processes NOR safe for concurrent writers.** Hard-document this.

**Phase:** Concurrency safety phase. Documentation-first, then runtime guards.

---

### P6: HMAC Blob Signing — Performance Impact on Large Blobs (>100MB)

**What goes wrong:** HMAC-SHA256 over a 100MB blob takes ~200-300ms on modern hardware. For a 1GB blob (e.g., a large DataFrame or NumPy array), this is 2-3 seconds. If blob HMAC is computed on both `put()` and `get()`, every cache roundtrip for large blobs doubles in latency.

**Why it happens:** HMAC-SHA256 processes at ~500MB/s on a single core. The current xxhash-based `file_hash` is ~10x faster (~5GB/s). Users who enable blob signing for integrity may not expect the performance regression.

**Consequences:**
- Cache `put()` and `get()` become significantly slower for large blobs.
- Users disable signing entirely rather than accepting the tradeoff.

**Prevention:**
1. **Use xxhash for blob integrity, HMAC for blob authentication.** The existing `file_hash` (xxhash) already detects accidental corruption. HMAC-SHA256 adds *authentication* (proves the blob wasn't tampered with). Consider: is authentication actually needed for blob content? If the metadata HMAC already covers the `file_hash`, then tampering with the blob changes the hash, which invalidates the metadata signature. **Blob HMAC may be unnecessary if metadata HMAC covers `file_hash`.**
2. **If blob HMAC is still wanted**: compute it lazily on `get()` only when `verify_hashes=True` (already the default for `verify_integrity()`). Skip it during normal `get()` unless explicitly requested.
3. **Streaming HMAC**: compute HMAC while writing/reading the blob, not as a separate pass. This adds zero latency for blob I/O that's already happening.
4. **Configurable blob signing threshold**: only sign blobs below a size threshold (e.g., 10MB). Large blobs rely on `file_hash` + metadata HMAC for integrity.

**Phase:** Blob HMAC signing phase. Performance budget must be defined before implementation.

---

### P7: Management APIs — Backend Behavioral Differences in Batch Operations

**What goes wrong:** `delete_by_prefix()` has fundamentally different performance characteristics across backends:
- **JSON**: O(n) scan + O(n) rewrite. With 10,000 entries, this means reading and rewriting a multi-MB JSON file.
- **SQLite**: O(k) with index, where k is matching entries. `DELETE FROM cache_entries WHERE cache_key LIKE 'prefix%'` uses the index.
- **PostgreSQL**: O(k) with index, plus network round-trip latency.

If the management API is implemented at the cache layer (iterating `list_entries()` and calling `remove_entry()` per entry), **all backends degrade to O(n²)** because JSON rewrites the full file per deletion, and SQLite/PG commit per deletion.

**Why it happens:** The existing codebase already has this pattern: `cleanup_by_size()` and `_cleanup_expired()` iterate entries and delete individually. Adding more bulk operations amplifies the problem.

**Consequences:**
- `delete_by_prefix()` with 1,000 matches in a 10,000-entry JSON cache takes minutes (1,000 full-file rewrites).
- SQLite performs better but still wastes 1,000 transactions instead of one.
- Users file bug reports about "slow delete" that only reproduce with large caches.

**Prevention:**
1. **Add backend-level bulk operations.** `MetadataBackend.remove_entries(keys: List[str])` that each backend implements optimally:
   - JSON: one pass to remove keys, one `_save_to_disk()` call.
   - SQLite: `DELETE FROM ... WHERE cache_key IN (?)` in one transaction.
   - PostgreSQL: same, in one transaction.
2. **Add `MetadataBackend.remove_by_prefix(prefix: str)` as a first-class operation** with SQL `LIKE` for SQLite/PG and dict comprehension for JSON.
3. **`delete_by_prefix()` at the cache layer should delegate to the backend method**, not iterate.
4. **Test performance with 5,000+ entries** across all three backends. Set thresholds: JSON < 1s, SQLite < 100ms, PG < 200ms.

**Phase:** Management APIs phase. Backend interface must be extended before implementing cache-layer APIs.

---

### P8: Orphaned Blob Prevention — `verify_integrity(repair=True)` Deletes Blobs During Active Writes

**What goes wrong:** If `verify_integrity(repair=True)` runs while another thread/process is doing `put()`, the sequence can be:
1. Thread A starts `put()`: writes blob file `abc123.pkl.lz4`
2. Thread B runs `verify_integrity()`: scans blobs, finds `abc123.pkl.lz4` with no metadata entry → marks as orphaned
3. Thread A writes metadata for `abc123`
4. Thread B deletes `abc123.pkl.lz4` as "orphaned"
5. Thread A's `put()` returns success, but the blob is gone

**Why it happens:** The blob-first write order means there's a window where a blob exists without metadata. `verify_integrity()` can't distinguish "in-progress write" from "orphaned blob."

**Consequences:**
- Data loss: `get()` returns `None` or raises because the blob file was deleted.
- Intermittent: only happens when `verify_integrity(repair=True)` coincides with active `put()` operations.

**Prevention:**
1. **`verify_integrity()` currently holds `self._lock`**, which prevents concurrent `put()` *within the same process*. This is already correct for single-process use.
2. **For multi-process**: orphaned blob detection should have a grace period. Only consider blobs orphaned if they've existed for longer than a threshold (e.g., 60 seconds). A blob written 2 seconds ago is likely in-progress.
3. **Add a `min_age_seconds` parameter to `verify_integrity()`** defaulting to 60. Only blobs older than this are considered orphaned.
4. **For the metadata-first approach (P3 prevention)**: this problem disappears because metadata is written first; the blob is always "expected."

**Phase:** Orphaned blob prevention phase. Must coordinate with the write-order decision from P3.

---

### P9: HMAC Blob Signing — Signature Version Migration for Existing Caches

**What goes wrong:** The current signer has `SIGNED_FIELDS_BY_VERSION` with v1 and v2. Adding blob signing means either:
(a) Adding `blob_signature` as a new field in v3's signed fields (but blob signature is a *separate* concern from metadata fields), or
(b) Creating a separate `blob_signature` field that's independent of metadata signature versioning.

If you choose (a), upgrading existing caches requires re-signing all entries with v3, which needs the blob content to be read for each entry — an O(n × blob_size) migration that could take hours for large caches.

**Why it happens:** The signature versioning system is designed for adding/removing metadata fields, not for adding entirely new signing dimensions (blob content).

**Consequences:**
- Cache migration takes hours for large caches with big blobs.
- If migration is skipped, old entries can't be verified with v3 signatures.
- Mixed v2 and v3 entries in the same cache create confusion about what's actually verified.

**Prevention:**
1. **Keep blob HMAC separate from metadata HMAC.** Store it as `blob_hmac` alongside the existing `entry_signature`, not as part of the versioned signature scheme.
2. **Blob HMAC is opt-in for existing entries.** New entries get `blob_hmac` automatically when blob signing is enabled. Old entries have `blob_hmac=None`, which means "not signed" — not "failed verification."
3. **Provide a `resign_blobs()` utility** that iterates all entries, reads their blobs, and computes `blob_hmac` for entries that lack one. This is an explicit migration action, not automatic.
4. **In `_verify_entry()`, treat `blob_hmac=None` as "skip blob verification"**, not as a verification failure. This maintains backward compatibility.

**Phase:** Blob HMAC signing phase. Architecture must be settled before any implementation.

---

### P10: Concurrency — SQLite Backend `threading.Lock()` vs `threading.RLock()` Mismatch

**What goes wrong:** `SqliteBackend` uses `threading.Lock()` (non-reentrant), but `UnifiedCache` uses `threading.RLock()` (reentrant). If a management API (e.g., `delete_by_prefix()`) calls `remove_entry()` which calls a session operation, and the outer caller already holds the backend lock (e.g., via `list_entries()` → `remove_entry()`), the non-reentrant `Lock` will **deadlock** because the thread already holds it.

**Why it happens:** The SQLite backend was designed for independent operations (each method grabs the lock, does its thing, releases). Management APIs that compose multiple backend operations in a single logical unit break this assumption.

**Consequences:**
- Complete hang of the cache on the first use of a batch management API.
- Hard to diagnose: no error message, just a frozen process.

**Prevention:**
1. **Audit every backend method that acquires `self._lock`.** If any management API path calls two lock-acquiring methods in sequence, it will deadlock with `Lock()` but not with `RLock()`.
2. **Change `SqliteBackend._lock` to `threading.RLock()`** for consistency with `UnifiedCache._lock` and `JsonBackend._lock`. This is a safe change — RLock is strictly more permissive than Lock.
3. **Alternatively**, implement batch backend methods (`remove_entries()`, `remove_by_prefix()`) that acquire the lock once and do all work inside, avoiding the composition problem.
4. **Add a deadlock-detection test**: call every management API from within a `with backend._lock:` block and verify it doesn't hang (use `threading.Timer` to kill the test after 5 seconds).

**Phase:** Concurrency safety phase AND Management APIs phase (shared concern). Fix the Lock→RLock before implementing batch operations.

---

## Minor Pitfalls

Individually non-critical, but accumulation of these creates maintenance burden.

---

### P11: Management APIs — `touch()` Semantics Vary by Backend

**What goes wrong:** `touch()` should update `accessed_at` without modifying the blob or other metadata. But:
- JSON backend: updating `accessed_at` requires rewriting the entire JSON file (`_save_to_disk()`).
- SQLite backend: single `UPDATE cache_entries SET accessed_at = ? WHERE cache_key = ?` — fast.
- PostgreSQL: same as SQLite, but with network latency.

If `touch()` is called frequently (e.g., LRU-style access tracking), JSON backend becomes a bottleneck.

**Prevention:** Implement `touch()` at the backend level. For JSON, batch `touch()` calls and only flush periodically (or on close). Consider `touch_batch(keys: List[str])` that updates multiple entries with one `_save_to_disk()` call.

**Phase:** Management APIs phase.

---

### P12: `get_metadata()` May Leak Inline Blob Bytes

**What goes wrong:** A `get_metadata()` API should return metadata without deserializing the blob. But for inline blobs, `blob_data` bytes are stored *in* the metadata row. Naively returning the full entry dict would include the blob bytes, which: (a) wastes memory when the caller only wants metadata, (b) leaks raw blob bytes to the caller.

**Prevention:** `get_metadata()` must explicitly strip `blob_data` from the returned dict. Add a test that verifies `get_metadata()` never returns a `blob_data` key.

**Phase:** Management APIs phase.

---

### P13: Orphaned Blob Prevention — S3-Compatible Store Consistency Caveats

**What goes wrong:** After uploading a blob to S3, the `put()` method stores the S3 key and ETag in metadata. Standard AWS S3 has strong read-after-write consistency since 2020. However, **S3-compatible stores** (MinIO, Garage, Ceph RGW) may not provide the same guarantees, and Cacheness supports these via the S3 blob backend.

**Prevention:** Document that orphaned blob prevention assumes strong read-after-write consistency. For S3-compatible stores without this guarantee, recommend a longer grace period in `verify_integrity()` before deleting orphans.

**Phase:** Orphaned blob prevention phase. Documentation item.

---

### P14: Management APIs — `update_blob_data()` Must Re-sign Both Metadata and Blob

**What goes wrong:** `update_blob_data()` replaces the blob content for an existing key. This must update: `file_size`, `file_hash`, `actual_path` (if extension changed), `data_type` (if type changed), `entry_signature` (re-sign metadata), and `blob_hmac` (if blob signing is enabled). Missing any of these creates an inconsistent entry.

**Prevention:**
1. Implement `update_blob_data()` by delegating to the existing `put()` with the same `cache_key`. The `put()` path already handles all of these concerns (including stale blob cleanup via `_cleanup_stale_blob()`).
2. Add a test that updates blob data and then runs `verify_integrity()` — must report zero issues.
3. Add a test that updates blob data with signing enabled and verifies both metadata and blob signatures.

**Phase:** Management APIs phase. Depends on blob HMAC signing phase completing first (or implement without blob HMAC and add it later).

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation | Severity |
|-------------|---------------|------------|----------|
| Management APIs | Backend O(n²) for bulk ops on JSON | Implement backend-level bulk methods (P7) | HIGH |
| Management APIs | Lock deadlock in batch operations | RLock for SQLite backend (P10) | HIGH |
| Management APIs | `delete_by_prefix()` re-signs remaining entries | Sign only on `put_entry()`, not `remove_entry()` (P4) | MEDIUM |
| Management APIs | `get_metadata()` leaks blob_data bytes | Strip `blob_data` from returned dict (P12) | LOW |
| Management APIs | `update_blob_data()` incomplete metadata update | Delegate to `put()` with same key (P14) | MEDIUM |
| Concurrency | JSON backend data loss under concurrent writers | Document as single-process only (P2) | CRITICAL |
| Concurrency | RLock vs Lock mismatch in SQLite backend | Change to RLock (P10) | HIGH |
| Concurrency | Multi-process != multi-thread confusion | Clear documentation (P5) | MEDIUM |
| HMAC blob signing | TOCTOU gap: signing in-memory bytes, not persisted | Sign after write, or leverage file_hash in metadata HMAC (P1) | CRITICAL |
| HMAC blob signing | Performance regression on large blobs | Use xxhash for integrity, HMAC for auth; streaming HMAC (P6) | MEDIUM |
| HMAC blob signing | Signature migration for existing caches | Separate blob_hmac from entry_signature (P9) | HIGH |
| Orphaned blob prevention | WAL adds Windows complexity + init latency | Use metadata "pending" flag instead of file WAL (P3) | HIGH |
| Orphaned blob prevention | `verify_integrity(repair=True)` deletes in-progress blobs | Add `min_age_seconds` grace period (P8) | HIGH |
| Orphaned blob prevention | S3-compatible stores lack strong consistency | Document assumption + longer grace period (P13) | LOW |

## Integration Pitfalls

These pitfalls only emerge when multiple v0.8.0 features interact.

### HMAC + Batch Operations
If batch `delete_by_prefix()` is implemented before blob HMAC, the batch delete won't clean up `blob_hmac` values. This is harmless (orphaned metadata fields) but wasteful. Implement management APIs with awareness that a `blob_hmac` field may exist.

### Concurrency + Orphaned Blob Prevention
The "pending" flag approach (P3) for orphaned blob prevention requires transactional metadata writes. SQLite/PG support this natively. JSON does not — if using the "pending" flag with JSON backend, a crash between writing the "pending" flag and clearing it leaves a permanently-pending entry. Prevention: only support pending-flag approach on SQLite/PG backends; JSON continues to rely on `verify_integrity()`.

### Management APIs + Concurrency
`touch_batch()` and `delete_batch()` must hold the lock for the entire batch to prevent interleaving. But holding the lock for a long-running batch operation (e.g., deleting 10,000 entries) blocks all other `put()`/`get()` calls. Consider: release the lock between batches of 100 items to allow interleaving of critical operations.

---

## Sources

- Codebase analysis: `src/cacheness/core.py` (put/get flow, `_PutCleanup`, locking), `src/cacheness/security.py` (CacheEntrySigner, signature versioning), `src/cacheness/storage/blob_store.py` (verify_integrity, blob write/delete ordering), `src/cacheness/metadata/json_backend.py` (thread safety, save_to_disk), `src/cacheness/metadata/sqlite_backend.py` (WAL mode, threading.Lock)
- Prior audit: `.planning/codebase/CONCERNS.md` (v0.7.0 codebase analysis)
- Project context: `.planning/PROJECT.md` (v0.8.0 milestone requirements)
- Confidence: HIGH — all pitfalls derived from actual code paths and architectural analysis of the existing codebase
