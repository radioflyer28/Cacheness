# Code Review Remediation — Execution Tasks

Companion to [CODE_REVIEW_FINDINGS.md](CODE_REVIEW_FINDINGS.md). Each task below is self-contained and written for direct execution by an agent. Do tasks in order within a wave; waves in order. One task = one beads issue = one atomic commit.

## Global rules (apply to EVERY task)

- Follow the Mandatory Workflow in `.github/copilot-instructions.md` (worktree, beads issue, quality gates).
- Package manager is **uv only**: `uv run pytest ...`, `uv run python ...`.
- Test command suffix always includes `--ignore=tests/test_tensorflow_handler.py`.
- In tests, import as `from cacheness.core import UnifiedCache` (NOT `from cacheness import UnifiedCache`).
- After each task: run the task's listed Tier-1 tests, then quality gates (`uv run ruff format <files>; uv run ruff check --fix <files>; uv run ruff check <files>; uv run ty check <files>`).
- Full suite once before push: `uv run pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py` (baseline: 1427 passed, 102 skipped).
- Do not change public API signatures unless the task says so.
- If a task's "Verify first" step fails (behavior differs from described), STOP and report instead of improvising.
- **Storage-mode invariant:** Cacheness has two operating modes — cache mode and storage mode (`storage_mode=True`: TTL/eviction/stats disabled, never auto-delete entries; see `_storage_mode_mixin.py` and the storage-mode block in `config.py` ~line 780). Any task that touches put/get/delete/cleanup/rollback paths MUST NOT introduce deletion or expiry of entries under storage mode, and MUST include `tests/test_storage_mode.py` in its Tier-1 run. When in doubt whether a destructive behavior applies in storage mode: it does not.

---

## WAVE 1 — Silent data loss

### TASK-1: Fix `clear_all()` not deleting blob files (Finding R1)

**Files:** `src/cacheness/storage/blob_store.py` (function `_clear_blob_files`, ~line 1321)

**Verify first:**
```
uv run python -c "
from cacheness.core import UnifiedCache
from cacheness.config import CacheConfig
import tempfile, pathlib
d = tempfile.mkdtemp()
c = UnifiedCache(CacheConfig(cache_dir=d))
c.put({'a': 1}, key1='x')
c.clear_all()
leftover = [p for p in pathlib.Path(d).rglob('*.pkl*')]
print('LEFTOVER BLOBS:', leftover)  # BUG if non-empty
"
```

**Change:** In `_clear_blob_files()`, the glob patterns use `self.cache_dir / f"*.{ext}"` — root-only. Blobs live in `cache_dir/{namespace}/` (and the backend may shard further). Replace the glob approach: enumerate blobs via `self.blob_backend.list_blobs(namespace)` for every namespace directory (or `rglob` under `self.cache_dir` excluding the metadata DB/JSON files and the `.intents/` directory), and delete each. Keep the method's no-lock contract (caller holds lock). Do NOT delete: `*.db`, `*.db-wal`, `*.db-shm`, `cache_metadata.json*`, anything under `.intents/`, `.cache_signing_key*`.

**Also fix:** `UnifiedCache.clear_all_namespaces()` (`src/cacheness/core.py` ~line 1201) calls `_clear_blob_files()` directly — confirm it now clears namespace dirs too.

**Acceptance:**
- New test in `tests/test_core.py`: put 3 entries (default + a custom namespace), `clear_all()`, assert **no blob files remain** under the cache dir (rglob), and metadata DB / signing key files still exist.
- Tier 1: `uv run pytest tests/test_blob_store.py tests/test_core.py tests/test_storage_mode.py -x -q --ignore=tests/test_tensorflow_handler.py`

### TASK-2: Fix write-intent journal path resolution + safety check (Finding R2)

**Files:** `src/cacheness/write_intent.py`, `src/cacheness/core.py` (~line 884 `record_intent` call site, `_cleanup_stale_intents`)

**Bug 1:** `record_intent(cache_key, result.actual_path)` receives a relative path like `default/abc.pkl`. `cleanup_stale_intents()` does `bp = Path(blob_path)` → resolves against process CWD, not the cache dir.

**Change 1:** Give `WriteIntentJournal.__init__` the cache_dir (it already receives it — it stores `cache_dir / ".intents"`; also keep `self._cache_dir = cache_dir`). In `cleanup_stale_intents()`, resolve: `bp = Path(blob_path); bp = bp if bp.is_absolute() else self._cache_dir / bp`.

**Bug 2:** Cleanup deletes the blob without checking whether the metadata commit succeeded (crash between `put_entry()` and `clear_intent()` ⇒ valid blob deleted).

**Change 2:** Add an optional `entry_exists: Callable[[str], bool]` parameter to `cleanup_stale_intents()`. In `UnifiedCache._cleanup_stale_intents()` pass `lambda k: self.metadata_backend.get_entry(k) is not None`. In the journal: if `entry_exists(cache_key)` is True, delete ONLY the intent file, NOT the blob.

**Bug 3 (storage mode — Finding R17):** `_storage_mode_put` records intents (`_storage_mode_mixin.py` ~line 75), but storage mode forces `cleanup_on_init=False` (config.py ~line 787), so stale-intent cleanup NEVER runs in storage mode — crash-leftover intents accumulate forever, and a later cache-mode open of the same directory could (pre-fix) delete valid durable blobs.

**Change 3:** In `UnifiedCache.__init__`, run `self._cleanup_stale_intents()` UNCONDITIONALLY (move it out of the `if self.config.storage.cleanup_on_init:` block, ~core.py line 153). This is safe in storage mode ONLY because of Change 2: with the `entry_exists` check, only blobs with no committed metadata (never durably stored) are deleted. Keep `_cleanup_expired()` inside the `cleanup_on_init` guard.

**Acceptance:**
- New tests in `tests/test_atomic_writes.py` (or wherever write-intent tests live — search `grep -r "cleanup_stale_intents" tests/`):
  1. Create an intent with a relative blob path + the actual blob in `cache_dir/default/`; set intent `created_at` to `time.time() - 400`; run cleanup from a DIFFERENT cwd; assert blob deleted.
  2. Same setup but metadata entry exists for the key; assert blob NOT deleted, intent file removed.
  3. Storage mode: init a `storage_mode=True` cache over a dir containing a stale intent whose metadata entry EXISTS; assert blob and entry survive, intent file removed. Repeat with no metadata entry; assert orphan blob removed.
- Tier 1: `uv run pytest tests/test_atomic_writes.py tests/test_core.py tests/test_storage_mode.py -x -q --ignore=tests/test_tensorflow_handler.py`

### TASK-3: JSON backend — stop swallowing save failures; preserve corrupt files (Findings R3, R4)

**Files:** `src/cacheness/metadata/json_backend.py` (`_save_to_disk`, `_load_from_disk`, `put_entry`, `remove_entry`)

**Change 1 (R3):** `_save_to_disk()` currently catches all exceptions and returns. Refactor: add parameter `raise_on_error: bool = False`. When True, re-raise after logging. Call with `raise_on_error=True` from `put_entry()` and `remove_entry()` (data-critical). Keep best-effort (False) for `update_access_time`, `increment_hits`, `increment_misses` (stats-only). Also: when the parent directory is missing, attempt `mkdir(parents=True, exist_ok=True)` before giving up.

**Change 2 (R4):** In `_load_from_disk()`, when JSON parsing fails: before starting fresh, rename the corrupt file to `f"{path}.corrupt-{int(time.time())}"` (use `Path.replace`, wrap in try/except OSError), and log at `logger.error` (not warning) including the backup path.

**Acceptance:**
- Test: make cache dir read-only or mock `shutil.move`/`os.replace` to raise `OSError`; assert `put_entry()` raises; assert `update_access_time()` does NOT raise.
- Test: write garbage to the metadata JSON path, init backend; assert a `*.corrupt-*` file exists and backend starts empty.
- Tier 1: `uv run pytest tests/test_metadata.py tests/test_json_schema_versioning.py -x -q --ignore=tests/test_tensorflow_handler.py`
- NOTE: some existing tests may assert the old swallow behavior — update them to the new contract, and say so in the commit message.

### TASK-4: Stabilize cache keys — remove unstable hash()/str() fallbacks (Finding U1)

**Files:** `src/cacheness/serialization.py` (`_serialize_with_config`, fallbacks 5 and 6, ~lines 363-378; large-tuple branch ~line 330)

**Bug:** `hash(obj)` is PYTHONHASHSEED-randomized for anything containing strings (e.g. tuples longer than `max_tuple_recursive_length`), and id-based for default objects. `str(obj)` default repr embeds memory addresses. Both yield keys that differ between processes — silent permanent cache misses.

**Change:**
1. Large tuples (`len > max_tuple_length`): instead of falling through to `hash()`, serialize each element recursively and hash the joined string with xxhash: `f"tuple_hashed:{len(obj)}:{xxhash.xxh3_64(','.join(items).encode()).hexdigest()[:16]}"`. This is deterministic.
2. Fallback 5 (`enable_hashable`): only use `hash(obj)` for types whose hash is process-stable: `int`, `float`, `bool`, `complex`, `decimal.Decimal`, `datetime`/`date`/`time`, `Enum` members (hash of name instead), and `None`. For everything else, skip to the next fallback. Simplest implementation: check `isinstance(obj, (int, float, complex))` etc.; for other hashables emit `logger.warning` once (module-level `warnings` or a logged-once set) and fall through.
3. Fallback 6 (`enable_string`): if `str(obj)` matches the default-repr pattern (`" object at 0x"` in the string), replace the address part: use `f"{type(obj).__name__}:unstable_repr"` and log a warning that the key is low-quality.

**Compatibility note:** this CHANGES cache keys for affected objects — existing cached entries for those keys become unreachable (they were already unreachable across processes, which is the bug). State this in the commit message and CHANGELOG.

**Acceptance:**
- Test: build a 15-element tuple of strings, compute `create_unified_cache_key({'x': t})` twice in **subprocesses** (`subprocess.run([sys.executable, '-c', ...])` with different `PYTHONHASHSEED` env values); assert equal.
- Existing decorator/key tests must still pass.
- Tier 1: `uv run pytest tests/test_serialization.py tests/test_decorators.py tests/test_core.py -x -q --ignore=tests/test_tensorflow_handler.py` (adjust filenames to what exists — check with `ls tests/`).

---

## WAVE 2 — TTL & eviction consistency

### TASK-5: Honor per-entry TTL (`expires_at`) end-to-end (Finding R5)

**Files:** `src/cacheness/core.py` (`_is_expired` ~line 703), `src/cacheness/metadata/sqlite_backend.py` (`cleanup_expired` ~line 1161), `src/cacheness/metadata/json_backend.py` (`cleanup_expired` ~line 371), `src/cacheness/storage/backends/postgresql_backend.py` (`cleanup_expired` ~line 1418)

**Decision (approved direction):** honor stored `expires_at` when present; caller/config TTL applies only when the entry has no stored expiry.

**Change 1:** In `_is_expired()`: after fetching the entry, if `entry.get("expires_at")` is set, parse it (str → `datetime.fromisoformat`, assume UTC if naive) and return `now > expires_at`, ignoring the ttl param. Otherwise keep existing logic. **Storage-mode guard:** `_storage_mode_get` never calls `_is_expired()` — keep it that way; storage-mode entries must never expire even if they carry an `expires_at` (e.g., written earlier by a cache-mode instance).

**Change 2:** In each backend's `cleanup_expired(ttl_seconds)`: delete entries where (`expires_at` is set AND `expires_at < now`) OR (`expires_at` is NULL AND `created_at < cutoff`). SQLite/PG: single DELETE with OR condition. JSON: same logic in the Python loop.

**Acceptance:**
- Parity test across JSON + SQLite: store entry with `ttl_seconds` in entry_data such that `expires_at` is in the past but `created_at` is recent; `cleanup_expired(ttl_seconds=99999)` must delete it. Reverse case: `expires_at` in the future, `created_at` old → must survive.
- Storage-mode test: `storage_mode=True` cache reading an entry whose stored `expires_at` is in the past → `get()` still returns the data.
- Tier 1: `uv run pytest tests/test_metadata.py tests/test_core.py tests/test_backend_parity.py tests/test_storage_mode.py -x -q --ignore=tests/test_tensorflow_handler.py`

### TASK-6: Init-time cleanup must delete blob files (Finding R6)

**Files:** `src/cacheness/core.py` (`_cleanup_expired` ~line 763)

**Change:** Replace the body of `_cleanup_expired()` so it calls the public `self.cleanup_expired(ttl_seconds)` (which already deletes blob files and invokes `on_evict` hooks) instead of `self.metadata_backend.cleanup_expired(ttl_seconds)`. Beware: the public method takes `self._lock`; `_cleanup_expired` is called from `__init__` — confirm the lock is already constructed at that point and not held (it's an RLock shared with BlobStore, re-entry is safe).

**Acceptance:**
- Test: create cache with `default_ttl_seconds=1` disabled, put entry, manually backdate `created_at` in metadata, re-init cache with `auto_cleanup_expired=True` and a TTL; assert metadata gone AND blob file gone.
- Tier 1: `uv run pytest tests/test_core.py -x -q --ignore=tests/test_tensorflow_handler.py`

### TASK-7: Preserve `access_count` and `created_at` correctly (Findings R9, R10)

**Files:** `src/cacheness/metadata/sqlite_backend.py` (`put_entry` ~line 732, `update_entry_metadata` ~line 869), `src/cacheness/metadata/json_backend.py` (`update_entry_metadata`), `src/cacheness/storage/backends/postgresql_backend.py` (`_upsert_entry` ~line 1000, `update_entry_metadata` ~line 1211)

**Change 1 (R9):** SQLite `put_entry` uses `INSERT OR REPLACE` which zeroes `access_count`. Convert to `INSERT ... ON CONFLICT(cache_key) DO UPDATE SET ...` listing every column EXCEPT `access_count` (keep `access_count = access_count` semantics — i.e. omit it from the update set so the existing value survives; new inserts still use the `:access_count` param). PG `_upsert_entry`: apply the same rule.

**Change 2 (R10):** All three backends' `update_entry_metadata` set `created_at = now` — remove that line. Metadata-only updates must NOT touch `created_at` or `expires_at`. (Content updates go through `UnifiedCache.update_data()`, which re-puts.)

**Risk note:** `created_at` is a signed field. Check `_update_mixin.py` — if `update_metadata()` re-signs entries assuming the reset timestamp, update it to re-sign with the PRESERVED `created_at`. Search: `grep -n "created_at" src/cacheness/_update_mixin.py`.

**Acceptance:**
- Test: put entry, `get()` it 3× (access_count=3), put same key again → `access_count` still ≥ 3 (or document chosen semantics).
- Test: `update_metadata(key, description="new")` → `created_at` unchanged; entry still verifies (signature valid) on `get()`.
- Tier 1: `uv run pytest tests/test_metadata.py tests/test_update_operations.py tests/test_backend_parity.py tests/test_security.py -x -q --ignore=tests/test_tensorflow_handler.py`

### TASK-8: Delete remote blobs on eviction (Finding R13)

**Files:** `src/cacheness/core.py` (`_enforce_size_limit`)

**Change:** The loop deleting evicted entries' blob files skips any `actual_path` containing `://`. Replace direct `Path.unlink` with: if `"://" in actual_path` → `self._blob_store.blob_backend.delete_blob(actual_path)` wrapped in try/except with `logger.warning` on failure; else keep the existing local unlink.

**Acceptance:**
- Test with `InMemoryBlobBackend` (uses `memory://` URIs): fill past size limit, trigger eviction, assert evicted blob removed from the in-memory backend store.
- Tier 1: `uv run pytest tests/test_core.py tests/test_blob_store.py -x -q --ignore=tests/test_tensorflow_handler.py`

---

## WAVE 3 — Multi-process & parity hardening

### TASK-9: Unique temp names for blob writes (Finding R11)

**Files:** `src/cacheness/storage/backends/blob_backends.py` (`FilesystemBlobBackend.write_blob`)

**Change:** Temp path is currently deterministic (`<final>.tmp`). Use `tempfile.mkstemp(dir=blob_path.parent, suffix=".tmp")`, write via the returned fd, then `os.replace(tmp, blob_path)`. Clean up the temp file in an except branch.

**Acceptance:** existing blob tests pass; add a test that two sequential writes to the same blob_id succeed and content is the last write.
Tier 1: `uv run pytest tests/test_blob_store.py tests/test_blob_namespace.py -x -q --ignore=tests/test_tensorflow_handler.py`

### TASK-10: Persist leftover user metadata on SQLite/PG (Finding U2)

**Files:** `src/cacheness/metadata/sqlite_backend.py` (`put_entry` — after all the `.pop()` calls, ~line 775), `src/cacheness/storage/backends/postgresql_backend.py` (same pattern in `_upsert_entry`), read paths (`get_entry`/`list_entries` — wherever `metadata_dict` is deserialized)

**Bug:** After popping known fields from the nested `metadata` dict, leftover user keys (from `BlobStore.put(metadata={...})`) are silently discarded. JSON backend preserves them → parity break; `metadata_filter` never matches on SQLite/PG.

**Change:** If `metadata` (the leftovers dict) is non-empty after pops AND `metadata_dict_value` is None, serialize leftovers: `metadata_dict_value = json_dumps(metadata)`. If `metadata_dict_value` already exists (set by core), merge leftovers into it (core's value wins on conflicts). On the read path, ensure deserialized `metadata_dict` keys are exposed in the entry's nested `metadata` dict the same way JsonBackend exposes them — inspect how `BlobStore.get_metadata()`/`list_keys(metadata_filter=...)` reads them and match that.

**Acceptance:**
- Parity test: `BlobStore.put(key, data, metadata={"experiment": "x42"})` then `get_metadata(key)["experiment"] == "x42"` and `list_keys(metadata_filter={"experiment": "x42"}) == [key]` — run against JSON AND SQLite backends.
- Tier 1: `uv run pytest tests/test_blob_store.py tests/test_metadata.py tests/test_backend_parity.py -x -q --ignore=tests/test_tensorflow_handler.py`

### TASK-11: Non-destructive overwrite on failed put (Finding R7)

**Files:** `src/cacheness/core.py` (`put`), `src/cacheness/_storage_mode_mixin.py` (`_storage_mode_put` — **has the identical bug**, ~line 60), `src/cacheness/_put_cleanup.py`

**Bug:** Re-putting a key writes the new blob over the old entry's path; if metadata write fails, rollback unlinks it → old entry destroyed. This affects BOTH `put()` (cache mode) and `_storage_mode_put()` (storage mode) — in storage mode it is unrecoverable durable-data loss, so the fix is mandatory there too.

**Change (minimal-risk variant):** Before `_write_blob`, if an existing entry's `actual_path` resolves to the SAME path the new write will target, rename the old blob to `<path>.prev` first. On success (`cleanup.commit()` path): delete `<path>.prev`. On rollback: restore `<path>.prev` → original name (add `prev_blob_path` field + restore logic to `_PutCleanup.rollback()`). Skip all of this for remote URIs and inline entries. Implement once (helper or inside `_PutCleanup`) and apply in BOTH `put()` and `_storage_mode_put()`.

**Acceptance:**
- Test: put key (value A), monkeypatch `metadata_backend.put_entry` to raise on next call, put same key (value B) → exception; then `get(key)` returns A and blob file exists.
- Same test repeated with `storage_mode=True`.
- Tier 1: `uv run pytest tests/test_core.py tests/test_fault_injection.py tests/test_cache_integrity.py tests/test_storage_mode.py -x -q --ignore=tests/test_tensorflow_handler.py`

### TASK-12: Backend-driven blob enumeration for integrity checks (Finding R12)

**Files:** `src/cacheness/storage/backends/blob_backends.py` (`FilesystemBlobBackend.list_blobs`)

**Change:** Replace the hardcoded extension whitelist with directory enumeration: list all files under the namespace dir (recursive if sharding enabled), excluding `*.tmp`, `.intents/`, metadata files (same exclusion list as TASK-1 — extract a shared module-level constant `RESERVED_FILE_PATTERNS` if convenient).

**Acceptance:** integrity test: write a blob with a fake extension `.custom` into the namespace dir; `verify_integrity()` reports it as orphaned.
Tier 1: `uv run pytest tests/test_blob_store.py tests/test_cache_integrity_verification.py -x -q --ignore=tests/test_tensorflow_handler.py`

---

## WAVE 4 — Security posture

### TASK-13: Harden signature verification (Finding S1)

**Files:** `src/cacheness/config.py` (SecurityConfig ~line 393), `src/cacheness/security.py` (`verify_entry` ~line 375), `src/cacheness/_verification_mixin.py`

**Change 1:** Add `minimum_signature_version: int = 1` to SecurityConfig. In `verify_entry()`, reject signatures whose parsed version < minimum (return False). Document recommended `=3` for new deployments.
**Change 2:** Document in `docs/SECURITY.md`: with `allow_unsigned_entries=True` (default), an attacker with metadata write access can strip signatures; recommend `allow_unsigned_entries=False` for any threat model where the metadata store is attacker-writable. Do NOT change the default in this task (breaking change — file a separate backlog issue proposing it for the next major).

**Acceptance:** test: sign entry at v3, set `minimum_signature_version=3`, rewrite signature prefix to `v2:` → verification fails.
Tier 1: `uv run pytest tests/test_security.py -x -q --ignore=tests/test_tensorflow_handler.py`

### TASK-14: In-memory decryption path (Findings S2, S3)

**Files:** `src/cacheness/storage/blob_store.py` (`get`, `_read_blob` — encrypted branches)

**Change:**
1. Read ciphertext via `self.blob_backend.read_blob(actual_path_str)` instead of `actual_path.read_bytes()` (fixes memory:// and s3:// correctness).
2. After decrypting, FIRST try `handler.get_bytes(plaintext, metadata)` (zero-disk). Only when it raises `NotImplementedError`, fall back to the temp-file path — and create that temp file with `tempfile.mkstemp` in the cache dir, `os.chmod(path, 0o600)` on POSIX, and keep the existing `finally` unlink.

**Acceptance:**
- Test: encrypted cache with InMemoryBlobBackend round-trips (currently expected to fail before fix — confirm in Verify-first step).
- Test: encrypted pickle entry round-trips and no `*.tmp` file remains after `get()`.
- Tier 1: `uv run pytest tests/test_blob_store.py tests/test_security.py tests/test_core.py -x -q --ignore=tests/test_tensorflow_handler.py` (plus any `test_encryption*.py` if present).

### TASK-15: Two-phase key rotation (Finding S4)

**Files:** `src/cacheness/core.py` (`rotate_key`), `src/cacheness/storage/blob_store.py` (`rotate_key`), `src/cacheness/security.py` (key file write)

**Change:** Restructure both `rotate_key()` implementations:
1. Write new key to `<keyfile>.new` (not over the old file).
2. Construct a second signer/encryptor from the new key; iterate entries: verify with OLD, re-sign with NEW; re-encrypt blobs by writing to `<blob>.rotating` then `os.replace`.
3. After ALL entries succeed: `os.replace(<keyfile>.new, <keyfile>)`.
4. On startup, if `<keyfile>.new` exists, log an error telling the user a rotation was interrupted (resume logic out of scope — just don't lose the old key).

**Acceptance:** test: monkeypatch to raise mid-rotation after N entries; assert original key file unchanged and all entries still verify with the old key.
Tier 1: `uv run pytest tests/test_security.py tests/test_core.py -x -q --ignore=tests/test_tensorflow_handler.py`

---

## Small independent fixes (any time, low risk)

| ID | Change | File | Acceptance |
|---|---|---|---|
| TASK-16 | `BlobStore.put()`: `custom_metadata = dict(metadata or {})` (stop mutating caller's dict) | blob_store.py | test: caller dict unchanged after put |
| TASK-17 | `_sanitize_key()`: when sanitization alters the key, return `f"{safe[:48]}_{xxhash.xxh3_64(key.encode()).hexdigest()[:16]}"` | blob_store.py | test: `"a:b"` and `"a_b"` get distinct storage keys; round-trip works |
| TASK-18 | `get()`: pass already-fetched entry into `_is_expired()` to avoid double backend read | core.py | existing tests pass; no behavior change |
| TASK-19 | Bump `version` in pyproject.toml to match CHANGELOG (currently 0.6.0 vs 0.11.x) | pyproject.toml | `uv run python -c "import cacheness; print(cacheness.__version__)"` matches |
| TASK-20 | `_get_blob_path()`: reject absolute `blob_id` (`if Path(safe_id).is_absolute(): raise ValueError`) | blob_backends.py | test: absolute blob_id raises |
| TASK-21 | SQLite: run `PRAGMA optimize` in `close()` instead of on connect; drop ineffective per-connect `page_size` (or document why kept) | sqlite_backend.py | existing tests pass |
| TASK-22 | S3 `delete_namespace_blobs`: drop `Quiet=True`, log per-object failures, return (deleted, failed) counts | s3_backend.py | unit test with mocked client |
| TASK-23 | Re-export `UnifiedCache` from package root (`from cacheness import UnifiedCache` works) — check `__init__.py` naming conflict first | src/cacheness/__init__.py | import test |

## Explicitly OUT of scope (needs human decision first)

- Storage-mode API hardening: should `cleanup_expired(ttl_seconds=...)`, size-limit eviction, and other cache-eviction APIs raise (or loudly warn) when called on a `storage_mode=True` instance? Today they can delete durable entries if invoked explicitly. Needs an owner decision on strictness (Finding §1b open question).
- Flipping `allow_unsigned_entries` default (breaking change — backlog).
- JSON backend write-batching/debouncing (U5 — perf design decision).
- Unifying the two signing schemes (U4 — touches signature compatibility of existing caches; needs migration plan).
- fsync policy (R16 — documentation-only acceptable; durability change needs perf discussion).
- Recording write intent before blob write (R8 — interacts with TASK-2; design together).
