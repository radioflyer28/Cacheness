# Cacheness Code Review — Findings & Recommendations

**Scope:** Cache reliability (primary emphasis), usability, bugs, and security.
Focus order per request: SQLite/JSON metadata backends + local filesystem blob store first; PostgreSQL/S3 second.

**Method:** Full read of `core.py`, `storage/blob_store.py`, `storage/backends/blob_backends.py`, `metadata/json_backend.py`, `metadata/sqlite_backend.py`, `write_intent.py`, `_put_cleanup.py`, `_verification_mixin.py`, `_inline_blob_mixin.py`, `serialization.py`, `storage/paths.py`, plus targeted review of `security.py`, `encryption.py`, `metadata/base.py`, `s3_backend.py`, `postgresql_backend.py`. Cross-checked against `.planning/codebase/CONCERNS.md` — findings already documented there are listed separately at the end and not re-reported as new.

**Severity legend:** 🔴 HIGH = data loss / silent corruption / broken guarantee · 🟠 MED = degraded reliability or security under realistic conditions · 🟡 LOW = hardening, polish, consistency.

---

## Executive Summary

The architecture is sound: layered metadata/blob separation, atomic temp-file+rename writes, WAL-mode SQLite, a write-intent journal for crash recovery, HMAC signing with HKDF key derivation, and a rollback guard for failed puts. These are the right building blocks.

However, the review found **several places where the reliability machinery doesn't actually deliver what it promises**:

1. **`clear_all()` does not delete blob files** — the glob looks in the wrong directory (R1).
2. **Crash recovery (write-intent journal) resolves blob paths against the process CWD**, so it almost never cleans the orphan it was built to clean — and could delete an unrelated file (R2).
3. **JSON backend silently loses metadata on save failure** — `put()` reports success while nothing was persisted (R3).
4. **Per-entry TTL columns (`ttl_seconds`, `expires_at`) are stored but never honored** anywhere — expiry is governed solely by caller/config TTL (R5).
5. **Cache keys are not stable across processes** when the `hash()` fallback is hit — a persistent disk cache that silently misses across runs (U1).

None of these are exotic edge cases; they affect mainline flows (`clear_all`, crash recovery, JSON persistence, TTL, key derivation). The good news: all have contained, low-risk fixes.

**Operating modes:** Cacheness runs in *cache mode* (TTL, eviction, stats) or *storage mode* (durable key-value store; all cache behaviors disabled). Findings were re-checked against both modes — see §1b for the storage-mode impact analysis. The short version: TTL/eviction findings (R5, R6, R13) are neutralized in storage mode by config, but the data-loss findings (R2, R3, R7) hit storage mode *harder*, because there the contract is durability, not best-effort caching.

---

## 1. Reliability — Local (SQLite/JSON metadata + filesystem blobs)

### 🔴 R1. `clear_all()` / `BlobStore.clear()` leaves every blob file on disk

- [src/cacheness/storage/blob_store.py](../src/cacheness/storage/blob_store.py#L1321) `_clear_blob_files()` globs only `cache_dir/*.{ext}` — **non-recursive, root level only**.
- But blobs are stored under `cache_dir/{namespace}/` (see `_get_cache_file_path()` in [src/cacheness/core.py](../src/cacheness/core.py#L686): "All namespaces (including default) store blob files under `cache_dir/{namespace}/`").
- `UnifiedCache.clear_all()` → `BlobStore.clear()` → `_clear_blob_files()` therefore deletes **metadata only**; all blob files survive as orphans. [core.py](../src/cacheness/core.py#L1155) and `clear_all_namespaces()` ([core.py](../src/cacheness/core.py#L1201)) are both affected.
- Likely a regression from the move to per-namespace blob directories.
- Secondary issue: the hardcoded extension whitelist (`pkl/npz/b2nd/b2tr/parquet` + pickle codecs) would miss custom-handler extensions even if the path were right.

**Fix:** delegate clearing to the blob backend (`blob_backend.delete_namespace_blobs()` / `list_blobs()` + delete), or glob recursively under each namespace dir. Add a regression test asserting the blob *files* are gone after `clear_all()`, not just the metadata.

### 🔴 R2. Write-intent crash recovery resolves blob paths against the CWD

- `record_intent(cache_key, result.actual_path)` is called with the **relative** stored path (e.g. `default/abc123.pkl`) — [core.py](../src/cacheness/core.py#L884).
- `cleanup_stale_intents()` then does `bp = Path(blob_path); bp.exists()` — [src/cacheness/write_intent.py](../src/cacheness/write_intent.py#L93). A relative path here resolves against the **process CWD**, not `cache_dir`.
- Consequences: (a) the orphaned blob is almost never found/deleted, so the journal's entire purpose silently fails; (b) pathologically, a file named `./default/abc123.pkl` relative to the CWD **could be deleted by mistake**.
- Additional gap: cleanup deletes the blob **without checking whether the metadata entry exists**. If a crash lands between `put_entry()` commit and `clear_intent()`, the next init deletes a *valid, referenced* blob → dangling metadata → destructive `get()` later removes the entry (data loss). The window is small but real.

**Fix:** resolve intent blob paths against `cache_dir` (store absolute or resolve at cleanup), and skip deletion when `metadata_backend.get_entry(cache_key)` exists and points at that blob.

### 🔴 R3. JSON backend: save failures are swallowed — `put()` lies about success

- `JsonBackend._save_to_disk()` catches **all** exceptions, logs, and returns ([src/cacheness/metadata/json_backend.py](../src/cacheness/metadata/json_backend.py)) — "best-effort". It also silently no-ops when the parent directory is missing.
- Every mutator (`put_entry`, `remove_entry`, `update_access_time`, …) calls it and reports success regardless. Disk-full, permission error, or directory removal → in-memory state diverges from disk; everything written since the failure is lost on process exit, **with no error surfaced to the caller**.

**Fix:** let `put_entry`/`remove_entry` propagate persistence failures (or at minimum set a "dirty/failed" flag surfaced via stats and retried on next write). Best-effort is defensible for `update_access_time`, not for entry writes.

### 🔴 R4. JSON backend: corrupt file → all metadata silently discarded

- `_load_from_disk()` on JSON parse failure logs a warning and "starts fresh" — the corrupt file is **not** preserved, every existing entry's metadata is gone, and all blobs become orphans.

**Fix:** rename the corrupt file to `*.corrupt-<timestamp>` before starting fresh, and log at ERROR. Consider attempting recovery from the temp file if present.

### 🔴 R5. Per-entry TTL (`ttl_seconds` / `expires_at`) is stored but never honored

- The v3 schema migration adds `ttl_seconds`/`expires_at` columns and `put_entry()` carefully computes `expires_at` ([sqlite_backend.py](../src/cacheness/metadata/sqlite_backend.py#L792)).
- But **nothing reads them**: `_is_expired()` uses only the caller/config TTL ([core.py](../src/cacheness/core.py#L703)), and `cleanup_expired()` in *all three* backends deletes by a single global cutoff on `created_at` ([sqlite_backend.py](../src/cacheness/metadata/sqlite_backend.py#L1161), [json_backend.py](../src/cacheness/metadata/json_backend.py#L371), [postgresql_backend.py](../src/cacheness/storage/backends/postgresql_backend.py#L1418)).
- Net effect: entries that conceptually have their own TTL are expired by whatever TTL the cleanup caller passes — premature eviction or zombie entries, depending on direction.

**Fix:** pick one semantics. Either (a) honor `expires_at` in `_is_expired()` and make `cleanup_expired()` use `expires_at < now` when set, or (b) drop the columns. Half-implemented TTL is worse than either.

### 🔴 R6. Init-time auto-cleanup orphans blob files

- `_cleanup_expired()` (run from `__init__` when `auto_cleanup_expired=True`) calls only `metadata_backend.cleanup_expired()` — [core.py](../src/cacheness/core.py#L763). Metadata rows are deleted; **blob files are not**.
- The *public* `cleanup_expired()` ([core.py](../src/cacheness/core.py#L1218)) does delete blob files first — the init path predates it and was never updated.
- Same gap on the read path: `get()` on an expired entry records a miss and returns `None` but leaves both metadata and blob in place. With default config (no TTL passed at init), expired blobs accumulate until someone runs `verify_integrity(repair=True)`.

**Fix:** make `_cleanup_expired()` call the public `cleanup_expired()`. Consider deleting (or at least flagging) expired entries encountered during `get()`.

### 🔴 R7. Overwriting an existing key can destroy it on failed put

- Re-putting the same key writes the new blob to the **same path** (path is derived from the key). If the subsequent metadata write fails, `_PutCleanup.rollback()` unlinks that path — which is now also the *old* entry's blob. Old metadata still points there → dangling entry → destructively removed on next `get()`.
- A put that fails should leave the previous value intact; today it can atomically destroy it.

**Fix:** write the new blob to a unique temp name and rename over the old one only after metadata commit, or snapshot the old blob before overwrite and restore on rollback.

### 🟠 R8. Write intent recorded *after* the blob is written

- In `put()`, `record_intent()` runs after `_write_blob()` completes ([core.py](../src/cacheness/core.py#L884)). A crash during handler serialization/encryption/rename leaves an orphan blob the journal never knew about. The journal only covers the blob-complete→metadata-commit window.
- Combined with R2 this means crash-orphans are effectively only cleaned by `verify_integrity(repair=True)`.

**Fix:** record intent (with the *planned* path) before invoking the handler.

### 🟠 R9. `INSERT OR REPLACE` resets `access_count` on overwrite

- [sqlite_backend.py](../src/cacheness/metadata/sqlite_backend.py#L808) — re-putting a hot key resets its `access_count` to 0 (default), so LRU/size eviction (`cleanup_by_size`, ordered by `accessed_at`) treats refreshed-but-hot entries as cold. Same pattern in PostgreSQL `_upsert_entry`.

**Fix:** `ON CONFLICT ... DO UPDATE` preserving `access_count` (and optionally `created_at` if TTL-on-first-write is desired).

### 🟠 R10. `update_entry_metadata()` resets `created_at` to now

- JSON, SQLite, and PostgreSQL ([postgresql_backend.py](../src/cacheness/storage/backends/postgresql_backend.py#L1211)) all stamp `created_at = now` on metadata update. This silently extends TTL, leaves a stale `expires_at` inconsistent with the new `created_at`, and — since `created_at` is a signed field — can invalidate existing signatures.

**Fix:** preserve `created_at` on metadata-only updates; only `update_data()` (new content) should reset it, and it should recompute `expires_at` and re-sign.

### 🟠 R11. Fixed temp-file name → cross-process write race

- `FilesystemBlobBackend.write_blob()` writes to `blob_path + ".tmp"` — a **deterministic** name. Two processes writing the same key concurrently (legitimate under the multi-process SQLite story) interleave on the same temp file → corrupted blob that then gets atomically renamed into place. The xxhash will catch it at read time, but the entry is destroyed.

**Fix:** `tempfile.mkstemp(dir=...)` style unique temp names (as `json_backend._save_to_disk` already does).

### 🟠 R12. `list_blobs()` extension whitelist blinds integrity checking

- `FilesystemBlobBackend.list_blobs()` enumerates only hardcoded extensions. Blobs written by custom handlers (or the `.bin` inline fallback ext) are invisible to `verify_integrity()` → orphans never detected, and `repair` can't clean them. Same root cause as the R1 whitelist issue.

**Fix:** enumerate everything under namespace dirs except reserved files (`.intents/`, metadata DB/JSON, temp files), or have handlers register their extensions.

### 🟠 R13. Eviction skips remote blobs — S3 storage leak

- `_enforce_size_limit()` deletes evicted entries' local files but explicitly skips paths containing `://`. Evicted S3 blobs are never deleted from the bucket — unbounded remote growth for size-limited caches.

**Fix:** route deletion through `blob_backend.delete_blob(uri)` instead of `Path.unlink`, which handles both cases.

### 🟡 R14. Lock discipline inconsistencies (SQLite)

- Writers take `self._lock` + session; readers (`get_entry`, `get_stats`) take no lock. Probably safe via SQLite's own locking + WAL, but the mixed discipline is fragile and makes reasoning hard. Also: stats row init (`id=1`) is unguarded across processes (benign IntegrityError race).

### 🟡 R15. SQLite PRAGMA nits

- `PRAGMA page_size=32768` on every connect is a no-op after DB creation (page size is fixed at creation; would need `VACUUM`).
- `PRAGMA optimize` runs on **connect**; SQLite docs recommend running it on **close**.

### 🟡 R16. No fsync anywhere (durability vs. crash-consistency)

- Atomic rename protects against partial writes, but neither JSON saves, blob writes, nor intent files fsync — on power loss, "committed" entries can vanish or be empty. SQLite (`synchronous=NORMAL` + WAL) is the only durable component. Acceptable for a cache; **not obviously acceptable for storage mode**, whose contract is durability — at minimum the limitation must be documented in `TRANSACTION_GUARANTEES.md`; consider an opt-in `fsync_on_write` for storage mode.

---

## 1b. Storage-mode impact analysis

Storage mode (`storage_mode=True`) disables TTL, size limits, stats, `cleanup_on_init`, and `auto_cleanup_expired` ([config.py](../src/cacheness/config.py#L780)); `_storage_mode_put`/`_storage_mode_get` ([_storage_mode_mixin.py](../src/cacheness/_storage_mode_mixin.py)) bypass cache concerns, and `_verify_entry(storage_mode=True)` **never deletes** entries. Re-checking each finding against this mode:

**Neutralized by storage mode (no action needed for this mode):**
- R5/R6 (TTL semantics, expired-entry cleanup) — TTL is forced off; `_is_expired()` is never called by `_storage_mode_get`.
- R13 (eviction skips remote blobs) — size limits forced off.
- R9 (`access_count` reset) — stats are off; only affects cache-mode LRU.
- Destructive `get()` / `delete_invalid_signatures` — correctly guarded by the `storage_mode` flag in `_verify_entry()`.

**Amplified by storage mode (same bug, higher stakes — the contract is durability):**
- 🔴 R3/R4 (JSON silent save failure / corrupt-file discard) — a *storage* system that silently fails to persist or silently discards all metadata is a contract violation, not a cache inefficiency. Storage mode + JSON backend is the worst combination.
- 🔴 R7 (failed overwrite destroys old entry) — **`_storage_mode_put` has the identical rollback hazard** ([_storage_mode_mixin.py](../src/cacheness/_storage_mode_mixin.py#L60)): old blob overwritten in place, `cleanup.rollback()` unlinks it on metadata failure. In storage mode that is unrecoverable durable-data loss. The fix (R7/TASK-11) must cover both `put()` and `_storage_mode_put()`.
- 🔴 R2 (write-intent cleanup) — the missing metadata-existence check matters most here: a stale intent whose metadata commit actually succeeded points at *durable data*.
- R10 (`created_at` reset on metadata update) — storage-mode users rely on `created_at` as provenance; silent resets corrupt the record (and can invalidate signatures).
- R16 (no fsync) — see above.

**New mode-specific finding:**
- 🟠 **R17. Write-intent lifecycle is incoherent in storage mode.** `_storage_mode_put` records intents ([_storage_mode_mixin.py](../src/cacheness/_storage_mode_mixin.py#L75)), but `cleanup_on_init=False` in storage mode means `_cleanup_stale_intents()` **never runs** — crash-leftover intent files accumulate forever and orphaned blobs are never recovered. Worse: if the same directory is later opened in *cache mode* (mixed-mode usage), stale-intent cleanup runs there and — pre-R2-fix — could delete a valid storage-mode blob. **Fix:** run stale-intent cleanup in storage mode too, but in a *conservative* variant: with the R2 metadata-existence check, deleting a truly orphaned blob (no metadata) is safe and consistent with storage-mode semantics (it was never durably committed). Alternatively, skip intent recording in storage mode and document that orphans require `verify_integrity(repair=True)`.

**Open design question (flagged, needs owner decision):** public destructive methods (`cleanup_expired()`, `cleanup_size()`/`_enforce_size_limit`, `clear_all()`) are still callable on a storage-mode instance. `cleanup_expired()` no-ops only because TTL is None — passing an explicit `ttl_seconds` **will delete durable entries**. Should storage mode hard-refuse (raise) cache-eviction APIs, or is explicit-call-means-consent acceptable? Recommend at least a loud warning log.

---

## 2. Correctness & Usability

### 🔴 U1. Cache keys are not stable across processes (hash()/str() fallbacks)

- [src/cacheness/serialization.py](../src/cacheness/serialization.py#L365): fallback 5 uses `hash(obj)`. Python's `hash()` for str/bytes (and anything containing them — e.g. **tuples longer than `max_tuple_recursive_length=10`**, frozensets) is randomized per process via `PYTHONHASHSEED`. Objects relying on default `object.__hash__` hash by **memory address**.
- Fallback 6 uses `str(obj)` — default `repr` embeds the memory address (`<Foo object at 0x...>`).
- Either fallback ⇒ a *persistent disk cache* whose keys change every run: silent 100% miss rate for affected parameters **plus** unbounded growth of unreachable entries. This is the worst failure mode for this library and it is completely silent.

**Fix:** never use `hash()`/raw `str()` for persistent keys. For large tuples, hash the recursively-serialized elements with xxhash instead of `hash()`. For objects, fall back to `__dict__`/pickle-based digests, and **log a warning** when an unstable fallback is the only option.

### 🔴 U2. SQLite/PostgreSQL silently drop user metadata that JSON preserves

- `SqliteBackend.put_entry()` pops known fields out of the nested `metadata` dict and **discards whatever remains** ([sqlite_backend.py](../src/cacheness/metadata/sqlite_backend.py#L732-L852)); PostgreSQL behaves the same. `JsonBackend` stores the whole nested dict.
- Impact: `BlobStore.put(..., metadata={"experiment": "x42"})` works on JSON, silently loses the keys on SQLite/PG — then `list_keys(metadata_filter=...)`/`get_metadata()` silently never match. A classic "works in dev (JSON), breaks in prod (SQLite)" parity trap.

**Fix:** serialize leftover keys into the existing `metadata_dict` JSON column (or reject unknown keys loudly). Add a backend-parity test for custom metadata round-trip via `BlobStore`.

### 🟠 U3. `BlobStore._sanitize_key()` can silently collide distinct keys

- Strips all non-`[A-Za-z0-9._-]` chars and truncates to 64. `"user:1|v2"` and `"user1v2"`, or two 70-char keys sharing a 64-char prefix, map to the same storage identity — last write silently wins.

**Fix:** when sanitization changes the key, append a short hash of the original (e.g. `f"{safe[:48]}_{xxh3(key)[:16]}"`).

### 🟠 U4. Two divergent signing schemes between `UnifiedCache` and `BlobStore`

- `UnifiedCache` signs via `_extract_signable_fields()` (normalizes `created_at`, fixed field superset); `BlobStore.put()` signs a flattened `{**entry_data, **custom_metadata}`. They share the same signer and metadata backend. An entry written by one API and read by the other risks spurious signature failures; the duplication invites drift.

**Fix:** make `BlobStore` use `_extract_signable_fields()` (move it to a shared module).

### 🟠 U5. JSON backend: every `get()` rewrites the entire metadata file — twice

- `get()` → `update_access_time()` → full-file save, then `record_hit()` → `increment_hits()` → full-file save. Reads cause 2× O(n) writes. This goes beyond the documented O(n²) *write* scaling — **read-heavy** workloads also degrade and churn the disk.

**Fix:** batch/debounce stats + access-time persistence (dirty flag, flush on close/interval), or make access-time updates in-memory-only for JSON with a documented caveat.

### 🟠 U6. `BlobStore.put()` mutates the caller's `metadata` dict

- `custom_metadata = metadata or {}` then in-place `update()` with `actual_path`, `file_hash`, signature, etc. Callers reusing a metadata dict across puts get cross-contamination.

**Fix:** `dict(metadata or {})`.

### 🟡 U7. Version metadata is stale

- [pyproject.toml](../pyproject.toml#L3) says `version = "0.6.0"`; the changelog is at v0.11.x. `cacheness_version` stamped into entries is therefore wrong, which matters for future migration logic.

### 🟡 U8. Packaging/import ergonomics

- `from cacheness import UnifiedCache` not working (exported as `cacheness`) is a recurring foot-gun (already noted in repo instructions). Consider re-exporting `UnifiedCache` from the package root.

### 🟡 U9. Double metadata read per `get()`

- `get()` fetches the entry, then `_is_expired()` fetches it again. Trivial fix (pass the entry in); halves backend reads on the hot path — meaningful for JSON and PG.

---

## 3. Security

### 🟠 S1. Unsigned-entry and version downgrade paths neutralize signing

- `_verify_entry()` only verifies **if a signature is present**; with the default `allow_unsigned_entries=True` ([config.py](../src/cacheness/config.py#L400)), an attacker who can edit the metadata store simply *deletes* the signature field and verification is skipped entirely — then the pickle handler deserializes their payload.
- Additionally, `verify_entry()` ([security.py](../src/cacheness/security.py#L375)) accepts v1 (bare hex), v2 (master key), and v3 (HKDF) signatures based on the *stored* prefix — rewriting `v3:` to `v2:` downgrades the scheme with no minimum-version enforcement.
- Mitigating context: anyone who can edit metadata can often edit blobs too, and `file_hash` is also signed — but the hash check is skipped when `file_hash` is absent, so the layers collapse together.

**Fix:** document loudly that signing only has teeth with `allow_unsigned_entries=False`; consider flipping that default for caches that have a signing key (key exists ⇒ entries must be signed). Add `minimum_signature_version` config; refuse v1/v2 when the cache was created at v3.

### 🟠 S2. Decrypted plaintext written to disk during reads

- For encrypted blobs, `BlobStore.get()`/`_read_blob()` decrypt to a **temp file on disk** so file-based handlers can read it; `_read_inline_blob()`'s slow path does the same in the system temp dir. Cleanup is in `finally`, but a crash mid-read leaks plaintext — partially defeating encryption-at-rest. (Encryption itself looks correct: AES-256-GCM, random nonce, HKDF per-namespace keys, encrypt-then-hash-then-sign.)

**Fix:** prefer the handler `get_bytes()` path for encrypted blobs (decrypt in memory); when a temp file is unavoidable, create it `0600` in the cache dir and register a startup sweep for stale `*.tmp` decrypt files.

### 🟠 S3. Encrypted-blob reads bypass the blob backend abstraction

- The decrypt path uses `actual_path.read_bytes()` directly rather than `blob_backend.read_blob()`. For `memory://` and `s3://` URIs this is broken outright (correctness), and it means backend-level controls are skipped.

### 🟠 S4. `rotate_key()` is not crash-safe

- The new key file is written **before** entries are re-signed/re-encrypted, and blob re-encryption uses in-place `write_bytes` (non-atomic). A crash mid-rotation leaves a mix of old-key signatures (now unverifiable) and possibly truncated blobs, with no rollback or resume marker.

**Fix:** two-phase rotation — keep both keys until all entries are re-signed (verify with old, sign with new), write re-encrypted blobs via temp+rename, delete the old key last.

### 🟡 S5. Key-file permission failures only warn

- [security.py](../src/cacheness/security.py#L265): if `chmod`/`icacls` fails, the HMAC key may sit world-readable with only a log line. Should at least be a prominent warning surfaced via stats/hook, arguably an exception under a strict mode.

### 🟡 S6. Blob path sanitization is fragile but currently safe

- `_get_blob_path()` neutralizes `..` but joins `base_dir / blob_id` where an **absolute** `blob_id` would escape the base dir entirely (`Path` join semantics). Blob IDs are internally generated today (hex keys), so not exploitable now — but one future caller passing user input through changes that. Add an explicit `is_absolute()` rejection.

---

## 4. Remote backends (PostgreSQL / S3) — secondary scope

Shared-pattern findings (R5 per-entry TTL, R10 `created_at` reset, R9 access-count reset, U2 metadata drop) apply to PostgreSQL as well — fix them together.

| Sev | Finding | Where |
|---|---|---|
| 🟠 | `cleanup_expired()` global-cutoff only; per-entry TTL ignored (= R5) | [postgresql_backend.py](../src/cacheness/storage/backends/postgresql_backend.py#L1418) |
| 🟠 | `update_entry_metadata()` resets `created_at` (= R10) | [postgresql_backend.py](../src/cacheness/storage/backends/postgresql_backend.py#L1211) |
| 🟠 | S3 ETag check-then-act dedup race — concurrent writers silently overwrite | [s3_backend.py](../src/cacheness/storage/backends/s3_backend.py#L270) |
| 🟠 | `delete_objects(Quiet=True)` hides per-object delete failures in namespace cleanup | [s3_backend.py](../src/cacheness/storage/backends/s3_backend.py#L633) |
| 🟠 | Evicted S3 blobs never deleted (= R13) | core `_enforce_size_limit()` |
| 🟡 | f-string table names — validated by `validate_namespace_id()`, but centralize quoting so safety doesn't rest on every call site remembering to validate | postgresql_backend.py throughout |
| 🟡 | `_parse_blob_path()` accepts a mismatched bucket URI with only a warning, then operates on `self.bucket` | [s3_backend.py](../src/cacheness/storage/backends/s3_backend.py#L612) |
| 🟡 | `CachedMetadataBackend` can serve stale reads vs. external (other-process) writes for up to its TTL — document the single-writer assumption | [metadata/base.py](../src/cacheness/metadata/base.py#L518) |

---

## 5. What's done well

Credit where due — these patterns are solid and should be preserved:

- **Atomic write discipline** (temp + rename) in the JSON backend and filesystem blob backend.
- **SQLite pragmas**: WAL, `busy_timeout=30s`, `synchronous=NORMAL` — the right trade for a cache.
- **Idempotent schema migrations** (v1→v4) guarded by `PRAGMA table_info` / `IF NOT EXISTS`, in both SQLite and PostgreSQL.
- **`_PutCleanup` rollback guard** with remote-blob awareness — the right shape, just needs R7 fixed.
- **Crypto fundamentals**: `hmac.compare_digest`, HKDF-SHA256 domain separation per namespace, AES-256-GCM with random nonces, hash-over-ciphertext consistency between file and inline paths.
- **S3 ETag fast path** in integrity verification avoids full downloads.
- **`iter_entry_summaries()`** as a lightweight batch-iteration API.
- Configurable destructive-get (`delete_on_error`) and storage-mode "never delete" semantics in `_verify_entry()`.
- Broad test suite (1400+ tests) with backend-parity and fault-injection coverage.

---

## 6. Recommended fix order

**Wave 1 — silent data loss / broken guarantees (small, contained fixes):**
1. R1 `_clear_blob_files()` namespace recursion (+ regression test).
2. R2 write-intent path resolution + metadata-existence check before blob delete.
3. R3/R4 JSON backend: propagate save failures; preserve corrupt file on load.
4. U1 key stability: eliminate `hash()`/`str()` fallbacks for persistent keys (warn at minimum).

**Wave 2 — consistency of the TTL/eviction story:**
5. R5 decide and implement per-entry TTL semantics end-to-end (or remove the columns).
6. R6 init cleanup → public `cleanup_expired()`; optionally purge expired entries on `get()`.
7. R9/R10 preserve `access_count`/`created_at` across overwrites and metadata updates.
8. R13 delete remote blobs on eviction.

**Wave 3 — multi-process & parity hardening:**
9. R11 unique temp names for blob writes.
10. U2 persist leftover user metadata on SQLite/PG (+ parity test).
11. R7 non-destructive overwrite (temp blob + rename after commit).
12. R12 backend-driven blob enumeration for clear/verify.

**Wave 4 — security posture:**
13. S1 unsigned-entry/downgrade hardening (defaults + min signature version).
14. S2/S3 in-memory decrypt path; route encrypted reads through the blob backend.
15. S4 two-phase `rotate_key()`.

---

## 7. Known issues acknowledged (already in `.planning/codebase/CONCERNS.md` — not re-reported)

- JSON backend O(n²) write scaling and concurrency unsafety (this review adds the *read*-path amplification, U5, and silent-failure dimension, R3).
- Pickle deserialization risk (this review ties it to the unsigned-downgrade path, S1).
- Orphaned blobs after hard crash / intent files not fsynced (this review adds that the recovery itself is broken, R2).
- Destructive `get()` on errors (configurable; by design).
- Cache-key strip-list fragility for new `put()` control params.
- No SQLite metadata-at-rest encryption; `postgresql_backend.py` size; TensorFlow handler disabled.
