# Phase 29: ttl-eviction-consistency - Pattern Map

**Mapped:** 2026-06-13
**Files analyzed:** 13
**Analogs found:** 13 / 13

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/cacheness/core.py` (`_is_expired`) | service | request-response | `src/cacheness/core.py` `_is_expired` | exact |
| `src/cacheness/core.py` (`_cleanup_expired`, `cleanup_expired`) | service | batch + file-I/O | `src/cacheness/core.py` `cleanup_expired` | exact |
| `src/cacheness/core.py` (`_enforce_size_limit`) | service | batch + file-I/O | `src/cacheness/core.py` `_enforce_size_limit` | exact |
| `src/cacheness/metadata/json_backend.py` | model | CRUD + batch | `JsonBackend.put_entry/update_entry_metadata/cleanup_expired` | exact |
| `src/cacheness/metadata/sqlite_backend.py` | model | CRUD + batch | `SqliteBackend.put_entry/update_entry_metadata/cleanup_expired` | exact |
| `src/cacheness/storage/backends/postgresql_backend.py` | model | CRUD + batch | `PostgresBackend._upsert_entry/update_entry_metadata/cleanup_expired` | exact |
| `src/cacheness/_update_mixin.py` | service | transform + CRUD | `update_data` re-signing flow | exact |
| `tests/test_core.py` | test | request-response + batch + file-I/O | existing cleanup and expiry tests | exact |
| `tests/test_metadata.py` | test | CRUD + batch | backend metadata CRUD tests | role-match |
| `tests/test_backend_parity.py` | test | CRUD + batch | `test_cleanup_expired_parity` | exact |
| `tests/test_update_operations.py` | test | CRUD + transform | existing update workflow tests | exact |
| `tests/test_storage_mode.py` | test | request-response | storage-mode expiry/eviction no-op tests | exact |
| `tests/test_blob_store.py` / `tests/test_s3_blob_backend.py` | test | file-I/O | in-memory and S3 delete tests | role-match |

## Pattern Assignments

### `src/cacheness/core.py` (`_is_expired`, service, request-response)

**Analog:** `src/cacheness/core.py`

**Imports pattern**: keep using module-level `datetime`, `timedelta`, `timezone`, `Path`, and `logger`; avoid local dependencies for TTL parsing unless a small local helper removes duplication.

**Current TTL fallback pattern** (lines 705-750):
```python
def _is_expired(self, cache_key: str, ttl_seconds=_DEFAULT_TTL) -> bool:
    entry = self.metadata_backend.get_entry(cache_key)
    if not entry:
        return True

    if ttl_seconds is None:
        return False
    elif ttl_seconds is _DEFAULT_TTL:
        ttl_seconds = self.config.metadata.default_ttl_seconds
        if ttl_seconds is None:
            return False

    creation_time_str = entry["created_at"]
    if isinstance(creation_time_str, str):
        creation_time = datetime.fromisoformat(creation_time_str)
    else:
        creation_time = creation_time_str
    if creation_time.tzinfo is None:
        creation_time = creation_time.replace(tzinfo=timezone.utc)

    expiry_time = creation_time + timedelta(seconds=ttl_seconds)
    current_time = datetime.now(timezone.utc)
    return current_time > expiry_time
```

**Storage-mode bypass pattern** (lines 1017-1023):
```python
if self.config.storage_mode:
    return self._storage_mode_get(cache_key)

entry = self.metadata_backend.get_entry(cache_key)
if not entry or self._is_expired(cache_key, resolved_ttl):
    self._record_miss()
    return None
```

**Apply for TASK-5:** Insert stored `expires_at` precedence before the `ttl_seconds is None` early return. Parse strings with `datetime.fromisoformat`; if naive, `replace(tzinfo=timezone.utc)`. When `expires_at` exists, ignore caller/config TTL.

---

### `src/cacheness/core.py` (`cleanup_expired`, service, batch + file-I/O)

**Analog:** `src/cacheness/core.py`

**Public cleanup owns blob deletion and hooks** (lines 1225-1298):
```python
def cleanup_expired(self, ttl_seconds: Optional[float] = None) -> int:
    with self._lock:
        if ttl_seconds is None:
            ttl_seconds = self.config.metadata.default_ttl_seconds

        if not ttl_seconds:
            logger.debug("cleanup_expired: no TTL configured, nothing to do")
            return 0

        expired_entries = []
        for entry in self.metadata_backend.iter_entry_summaries():
            created_at = entry.get("created_at")
            ...
            if created_timestamp < cutoff_time:
                expired_entries.append(entry)

        for entry in expired_entries:
            entry_key = entry.get("cache_key", "unknown")
            self._invoke_hook("on_evict", entry_key, "expired")
            actual_path = entry.get("actual_path")
            if actual_path and "://" not in actual_path:
                blob_file = self._resolve_actual_path(actual_path)
                if isinstance(blob_file, Path) and blob_file.exists():
                    blob_file.unlink()

        removed_count = self.metadata_backend.cleanup_expired(ttl_seconds)
```

**Init cleanup currently bypasses public path** (lines 765-773):
```python
def _cleanup_expired(self):
    ttl_seconds = self.config.metadata.default_ttl_seconds
    if ttl_seconds is None:
        return
    removed_count = self.metadata_backend.cleanup_expired(ttl_seconds)
```

**Apply for TASK-6:** Change `_cleanup_expired()` to call `self.cleanup_expired(ttl_seconds)` so init cleanup uses the public hook/blob cleanup path. Keep the `ttl_seconds is None` no-op for storage mode.

---

### `src/cacheness/core.py` (`_enforce_size_limit`, service, batch + file-I/O)

**Analog:** `src/cacheness/core.py` plus blob backend delete implementations.

**Current local-only deletion pattern** (lines 1091-1125):
```python
result = self.metadata_backend.cleanup_by_size(target_size_bytes)
removed_entries = result.get("removed_entries", [])

for entry in removed_entries:
    entry_key = entry.get("cache_key", "unknown")
    self._invoke_hook("on_evict", entry_key, "size_limit")
    actual_path = entry.get("actual_path")
    if actual_path and "://" not in actual_path:
        blob_file = self._resolve_actual_path(actual_path)
        if isinstance(blob_file, Path) and blob_file.exists():
            try:
                blob_file.unlink()
                blobs_deleted += 1
            except OSError as exc:
                logger.warning(
                    f"Failed to delete blob file {actual_path} during size enforcement: {exc}"
                )
```

**Remote backend deletion contract** (lines `blob_backends.py` 451-457, `s3_backend.py` 354-377):
```python
def delete_blob(self, blob_path: str) -> bool:
    if blob_path in self._storage:
        del self._storage[blob_path]
        logger.debug(f"Deleted blob: {blob_path}")
        return True
    return False
```

```python
def delete_blob(self, blob_path: str) -> bool:
    s3_key = self._parse_blob_path(blob_path)
    try:
        self._client.delete_object(Bucket=self.bucket, Key=s3_key)
        logger.debug(f"Deleted blob from s3://{self.bucket}/{s3_key}")
        return True
    except ClientError as e:
        logger.error(f"Failed to delete blob from S3: {e}")
        return False
```

**Apply for TASK-8:** Preserve local unlink handling, add `if actual_path and "://" in actual_path:` branch calling `self._blob_store.blob_backend.delete_blob(actual_path)`. Catch broad backend failures and log `logger.warning` without crashing eviction.

---

### `src/cacheness/metadata/json_backend.py` (model, CRUD + batch)

**Analog:** `JsonBackend.put_entry`, `update_entry_metadata`, `cleanup_expired`.

**TTL field storage pattern** (lines 139-178):
```python
entry = {
    "description": entry_data.get("description", ""),
    "data_type": entry_data.get("data_type", "unknown"),
    "created_at": entry_data.get("created_at", now),
    "accessed_at": entry_data.get("accessed_at", now),
    "file_size": entry_data.get("file_size", 0),
    "access_count": entry_data.get("access_count", 0),
    "is_inline": entry_data.get("is_inline", 0),
    "metadata": metadata,
}
ttl_val = entry_data.get("ttl_seconds")
if ttl_val is not None:
    entry["ttl_seconds"] = ttl_val
    created_dt = datetime.fromisoformat(entry["created_at"])
    if created_dt.tzinfo is None:
        created_dt = created_dt.replace(tzinfo=timezone.utc)
    entry["expires_at"] = (created_dt + timedelta(seconds=float(ttl_val))).isoformat()
```

**Metadata update anti-pattern to remove** (lines 191-214):
```python
entry = entries.get(cache_key)
if not entry:
    return False

# Update derived metadata (file_size, content_hash, created_at)
now = datetime.now(timezone.utc)
entry["created_at"] = now.isoformat()  # Reset timestamp
```

**Cleanup fallback pattern** (lines 390-407):
```python
cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=ttl_seconds)
for cache_key, entry in entries.items():
    try:
        creation_time_str = entry.get("created_at")
        if creation_time_str:
            creation_time = datetime.fromisoformat(creation_time_str)
            if creation_time < cutoff_time:
                expired_keys.append(cache_key)
    except (ValueError, TypeError):
        expired_keys.append(cache_key)
```

**Apply for TASK-5/TASK-7:** Add `expires_at` branch to cleanup before `created_at` fallback. Preserve `created_at`, `ttl_seconds`, and `expires_at` in metadata-only updates unless explicitly present in `updates`.

---

### `src/cacheness/metadata/sqlite_backend.py` (model, CRUD + batch)

**Analog:** `SqliteBackend.put_entry`, `update_entry_metadata`, `cleanup_expired`.

**TTL parsing/storage pattern** (lines 778-844):
```python
created_at = entry_data.get("created_at")
if isinstance(created_at, str):
    created_at = datetime.fromisoformat(created_at)
elif created_at is None:
    created_at = datetime.now(timezone.utc)

ttl_seconds_val = entry_data.get("ttl_seconds")
expires_at = entry_data.get("expires_at")
if expires_at is None and ttl_seconds_val is not None:
    created_dt = created_at
    if created_dt.tzinfo is None:
        created_dt = created_dt.replace(tzinfo=timezone.utc)
    expires_at = created_dt + timedelta(seconds=float(ttl_seconds_val))
elif isinstance(expires_at, str):
    expires_at = datetime.fromisoformat(expires_at)
```

**Overwrite anti-pattern to replace** (lines 805-824):
```python
session.execute(
    text(f"""
        INSERT OR REPLACE INTO "{tbl}"
        (... created_at, accessed_at, access_count, ttl_seconds, expires_at, ...)
        VALUES (... :created_at, :accessed_at, :access_count, :ttl_seconds, :expires_at, ...)
    """),
    {"access_count": entry_data.get("access_count", 0), ...},
)
```

**Access count increment pattern to preserve** (lines 1122-1133):
```python
session.execute(
    update(self._CacheEntry)
    .where(self._CacheEntry.cache_key == cache_key)
    .values(
        accessed_at=datetime.now(timezone.utc),
        access_count=self._CacheEntry.access_count + 1,
    )
)
```

**Cleanup predicate to extend** (lines 1161-1175):
```python
cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=ttl_seconds)
result = session.execute(
    delete(self._CacheEntry).where(self._CacheEntry.created_at < cutoff_time)
)
```

**Apply for TASK-5/TASK-7:** Use SQLite conflict update semantics instead of `INSERT OR REPLACE`; omit `access_count` from update set. Extend cleanup to delete where `expires_at < now` when non-null, or `expires_at IS NULL AND created_at < cutoff`.

---

### `src/cacheness/storage/backends/postgresql_backend.py` (model, CRUD + batch)

**Analog:** `PostgresBackend._upsert_entry`, `update_entry_metadata`, `cleanup_expired`.

**Upsert parsing pattern** (lines 975-1028):
```python
created_at = entry_data.get("created_at")
if isinstance(created_at, str):
    created_at = datetime.fromisoformat(created_at)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    else:
        created_at = created_at.astimezone(timezone.utc)

ttl_seconds_val = entry_data.get("ttl_seconds")
expires_at = entry_data.get("expires_at")
if expires_at is None and ttl_seconds_val is not None:
    expires_at = created_at + timedelta(seconds=float(ttl_seconds_val))
elif isinstance(expires_at, str):
    expires_at = datetime.fromisoformat(expires_at)
```

**Access-count overwrite anti-pattern** (lines 1035-1058):
```python
if existing:
    session.execute(
        update(self._PgCacheEntry)
        .where(self._PgCacheEntry.cache_key == cache_key)
        .values(
            created_at=created_at,
            accessed_at=accessed_at,
            ...
            access_count=access_count_val,
            ttl_seconds=ttl_seconds_val,
            expires_at=expires_at,
        )
    )
```

**Metadata update anti-pattern** (lines 1187-1217):
```python
entry = session.execute(...).scalar_one_or_none()
if not entry:
    return False

# Update derived metadata fields
now = datetime.now(timezone.utc)
entry.created_at = now  # Reset timestamp
```

**Cleanup fallback pattern** (lines 1418-1445):
```python
if ttl_seconds <= 0:
    return 0

cutoff = datetime.now(timezone.utc) - __import__("datetime").timedelta(
    seconds=ttl_seconds
)
count = session.execute(
    select(func.count()).select_from(self._PgCacheEntry).where(
        self._PgCacheEntry.created_at < cutoff
    )
).scalar() or 0
session.execute(delete(self._PgCacheEntry).where(self._PgCacheEntry.created_at < cutoff))
```

**Apply for TASK-5/TASK-7:** Existing-row update should not set `access_count`. Metadata-only update should not reset `created_at` or TTL fields. Cleanup should use the same stored-expiry-or-created-at-fallback predicate as JSON/SQLite while preserving the existing `ttl_seconds <= 0` no-op policy unless stored-expiry cleanup is explicitly required by the plan.

---

### `src/cacheness/_update_mixin.py` and signing (service, transform + CRUD)

**Analog:** `update_data` re-signing and `VerificationMixin._extract_signable_fields`.

**Re-sign after metadata/data update** (lines 218-249):
```python
if self.signer:
    try:
        updated_entry = self.metadata_backend.get_entry(cache_key)
        if updated_entry:
            metadata = updated_entry.get("metadata", {})
            ...
            complete_entry_data = self._extract_signable_fields(
                cache_key=cache_key,
                entry_data=updated_entry,
                metadata=metadata,
            )
            new_signature = self.signer.sign_entry(complete_entry_data)
            metadata["entry_signature"] = new_signature
            updated_entry["metadata"] = metadata
            self.metadata_backend.put_entry(cache_key, updated_entry)
```

**Created-at normalization for signatures** (lines 38-60):
```python
created_at = entry_data.get("created_at")
if isinstance(created_at, datetime):
    created_at = created_at.replace(tzinfo=None).isoformat()
elif isinstance(created_at, str):
    try:
        dt = datetime.fromisoformat(created_at)
        created_at = dt.replace(tzinfo=None).isoformat()
    except (ValueError, TypeError):
        pass

signable_data = {
    "cache_key": cache_key,
    "data_type": entry_data.get("data_type"),
    "file_size": entry_data.get("file_size", 0),
    "created_at": created_at,
    "actual_path": metadata.get("actual_path", ""),
```

**Apply for TASK-7:** Preserve `created_at` across metadata-only updates so signatures do not drift unexpectedly. If metadata-only updates alter signed fields such as `file_size`, `file_hash`, `actual_path`, or `data_type`, keep using `_extract_signable_fields()` and signer flow.

---

### Tests (`tests/test_core.py`, `tests/test_backend_parity.py`, `tests/test_storage_mode.py`, `tests/test_update_operations.py`, blob tests)

**Core cleanup integration pattern** (lines `tests/test_core.py` 685-846):
```python
cache.put({"value": 1}, key="entry_1")
...
time.sleep(0.15)
removed = cache.cleanup_expired()
assert removed == 3
assert not cache.exists(key="entry_1")
```

```python
cache.put(np.array([1, 2, 3]), key="array_test")
entry = cache.metadata_backend.get_entry(cache_key)
metadata = entry.get("metadata", {})
actual_path = metadata.get("actual_path") or entry.get("actual_path")
blob_path = cache._resolve_actual_path(actual_path)
assert blob_path.exists()
removed = cache.cleanup_expired()
assert removed == 1
assert not blob_path.exists()
```

**Expiry unit pattern** (lines `tests/test_core.py` 1067-1096):
```python
with patch("cacheness.core.datetime") as mock_datetime:
    current_utc = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    mock_datetime.now.return_value = current_utc
    mock_datetime.fromisoformat = datetime.fromisoformat
    assert cache._is_expired(cache_key, ttl_seconds=86400) is False
```

**Backend parity cleanup pattern** (lines `tests/test_backend_parity.py` 270-305):
```python
old_time = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
recent_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
sqlite_backend.put_entry("old_key", old_entry)
sqlite_backend.put_entry("recent_key", recent_entry)
removed_count = sqlite_backend.cleanup_expired(ttl_seconds=86400)
assert sqlite_backend.get_entry("old_key") is None
assert sqlite_backend.get_entry("recent_key") is not None
```

**Storage-mode guard pattern** (lines `tests/test_storage_mode.py` 106-140):
```python
storage_cache.put("hello", cache_key="k1")
assert storage_cache.get(cache_key="k1") == "hello"

storage_cache.put("data", cache_key="k2")
storage_cache._cleanup_expired()
assert storage_cache.exists(cache_key="k2")

storage_cache.put("data", cache_key="big")
storage_cache._enforce_size_limit()
assert storage_cache.exists(cache_key="big")
```

**Update workflow pattern** (lines `tests/test_update_operations.py` 105-128, 220-247, 327-351):
```python
meta_before = memory_cache.get_metadata(test="metadata_update")
created_before = meta_before.get("created_at")
...
success = memory_cache.update_data(large_data, test="metadata_update")
assert success is True
meta_after = memory_cache.get_metadata(test="metadata_update")
```

```python
success = backend.update_entry_metadata(
    cache_key="test_key",
    updates={
        "file_size": 400,
        "content_hash": "sqlite_hash",
        "storage_format": "pickle",
        "data_type": "dict",
        "serializer": "pickle",
    },
)
assert success is True
entry = backend.get_entry("test_key")
assert entry["file_size"] == 400
```

```python
success = memory_cache.touch(test="touch_test")
assert success is True
meta_after = memory_cache.get_metadata(test="touch_test")
assert created_after != created_before
```

**Apply for test additions:** Add verify-first regressions beside these existing tests. Use `from cacheness.core import UnifiedCache` for any new direct class import per project instructions, and include `--ignore=tests/test_tensorflow_handler.py` in pytest commands.

## Shared Patterns

### Stored Expiry Parsing

**Source:** `src/cacheness/metadata/sqlite_backend.py` lines 791-803 and `src/cacheness/storage/backends/postgresql_backend.py` lines 1015-1026  
**Apply to:** `_is_expired()`, public cleanup scan, JSON/SQLite/PostgreSQL cleanup predicates.

```python
expires_at = entry_data.get("expires_at")
if expires_at is None and ttl_seconds_val is not None:
    expires_at = created_at + timedelta(seconds=float(ttl_seconds_val))
elif isinstance(expires_at, str):
    expires_at = datetime.fromisoformat(expires_at)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    else:
        expires_at = expires_at.astimezone(timezone.utc)
```

### Public Cleanup Boundary

**Source:** `src/cacheness/core.py` lines 1225-1298  
**Apply to:** init cleanup and any cleanup path that must remove blobs or invoke hooks.

```python
for entry in expired_entries:
    entry_key = entry.get("cache_key", "unknown")
    self._invoke_hook("on_evict", entry_key, "expired")
    actual_path = entry.get("actual_path")
    if actual_path and "://" not in actual_path:
        blob_file = self._resolve_actual_path(actual_path)
        if isinstance(blob_file, Path) and blob_file.exists():
            blob_file.unlink()

removed_count = self.metadata_backend.cleanup_expired(ttl_seconds)
```

### Metadata-Only Preservation

**Source:** `src/cacheness/_verification_mixin.py` lines 38-60 and backend update methods  
**Apply to:** JSON, SQLite, PostgreSQL `update_entry_metadata()`.

```python
created_at = entry_data.get("created_at")
if isinstance(created_at, datetime):
    created_at = created_at.replace(tzinfo=None).isoformat()
elif isinstance(created_at, str):
    dt = datetime.fromisoformat(created_at)
    created_at = dt.replace(tzinfo=None).isoformat()
```

Preserve `created_at`, `ttl_seconds`, `expires_at`, and existing `access_count` unless the operation is a content update, `touch()`, or an explicit TTL update path.

### Remote Blob Deletion

**Source:** `src/cacheness/storage/backends/blob_backends.py` lines 437-457 and `src/cacheness/storage/backends/s3_backend.py` lines 354-377  
**Apply to:** `_enforce_size_limit()` URI branch.

```python
blob_path = f"memory://{blob_id}"
self._storage[blob_path] = data
...
def delete_blob(self, blob_path: str) -> bool:
    if blob_path in self._storage:
        del self._storage[blob_path]
        return True
    return False
```

## No Analog Found

No Phase 29 file lacks a close analog. All work is modifying existing code paths and tests.

## Metadata

**Analog search scope:** `src/cacheness/core.py`, `src/cacheness/metadata/`, `src/cacheness/storage/backends/`, `src/cacheness/_update_mixin.py`, `src/cacheness/_verification_mixin.py`, `tests/`  
**Files scanned:** 18  
**Pattern extraction date:** 2026-06-13
