# Phase 30: Multi-Process & Backend Parity - Pattern Map

**Mapped:** 2026-06-14
**Files analyzed:** 13
**Analogs found:** 13 / 13

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/cacheness/storage/backends/blob_backends.py` | storage backend | file-I/O | same file: `FilesystemBlobBackend.write_blob`, `write_blob_stream`, `list_blobs` | exact |
| `src/cacheness/storage/blob_store.py` | service | CRUD + file-I/O + transform | same file: `put`, `list`, `verify_integrity` | exact |
| `src/cacheness/metadata/sqlite_backend.py` | metadata backend | CRUD + transform | same file: `put_entry`, `get_entry`, `iter_entry_summaries` | exact |
| `src/cacheness/storage/backends/postgresql_backend.py` | metadata backend | CRUD + transform | same file: `put_entry`, `_upsert_entry`, `_entry_to_dict`, `iter_entry_summaries` | exact |
| `src/cacheness/core.py` | service/orchestrator | request-response + file-I/O rollback | same file: `UnifiedCache.put` | exact |
| `src/cacheness/_storage_mode_mixin.py` | service/orchestrator | request-response + file-I/O rollback | same file: `_storage_mode_put` | exact |
| `src/cacheness/_put_cleanup.py` | utility | file-I/O rollback | same file + `tests/test_s3_orphan_cleanup.py` | exact |
| `tests/test_blob_namespace.py` | test | file-I/O | same file: namespace write/read tests | exact |
| `tests/test_blob_store.py` | test | CRUD + integrity | same file: `verify_integrity` tests | exact |
| `tests/test_backend_parity.py` | test | CRUD + parity | same file: parametrized backend parity tests | exact |
| `tests/test_fault_injection.py` | test | fault-injection + rollback | same file: metadata failure cleanup tests | exact |
| `tests/test_storage_mode.py` | test | storage-mode request-response | same file: storage-mode fixture and put/get tests | exact |
| `tests/test_cache_integrity_verification.py` | test | integrity enumeration + repair | same file: orphan/dangling tests | exact |

## Pattern Assignments

### `src/cacheness/storage/backends/blob_backends.py` (storage backend, file-I/O)

**Analog:** `src/cacheness/storage/backends/blob_backends.py`

**Imports pattern:** module already imports `os`, `Path`, `BinaryIO`, and logging. TASK-9 should add stdlib `tempfile` near the existing stdlib imports, not inside `write_blob`, unless keeping import scope local is preferred for minimal churn.

**Filesystem namespace/path pattern** (lines 261-280):
```python
def __init__(
    self,
    base_dir: Union[str, Path],
    shard_chars: int = 2,
    namespace: str = "default",
):
    root = Path(base_dir)
    self.base_dir = root / namespace
    self.shard_chars = shard_chars
    self._namespace = namespace
    self.base_dir.mkdir(parents=True, exist_ok=True)
```

**Current atomic write pattern to replace** (lines 286-299):
```python
blob_path = self._get_blob_path(blob_id)
blob_path.parent.mkdir(parents=True, exist_ok=True)

# Write atomically using temp file
temp_path = blob_path.with_suffix(blob_path.suffix + ".tmp")
try:
    temp_path.write_bytes(data)
    temp_path.replace(blob_path)
except Exception:  # intentionally broad — cleanup temp file on any failure
    if temp_path.exists():
        temp_path.unlink()
    raise
```

**Streaming write pattern to keep consistent with `write_blob`** (lines 344-362):
```python
temp_path = blob_path.with_suffix(blob_path.suffix + ".tmp")
try:
    with open(temp_path, "wb") as f:
        # Read in chunks for memory efficiency
        while True:
            chunk = stream.read(8192)
            if not chunk:
                break
            f.write(chunk)
    temp_path.replace(blob_path)
except Exception:  # intentionally broad — cleanup temp file on any failure
    if temp_path.exists():
        temp_path.unlink()
    raise
```

**Enumeration pattern to replace for TASK-12** (lines 380-396):
```python
def list_blobs(self) -> List[str]:
    """List all blob files in the storage directory."""
    import glob as _glob

    blob_extensions = ["pkl", "npz", "b2nd", "b2tr", "parquet"]
    pickle_codecs = ["lz4", "zstd", "gzip", "zst", "gz", "bz2", "xz"]

    results: set[str] = set()
    for ext in blob_extensions:
        for f in _glob.glob(str(self.base_dir / "**" / f"*.{ext}"), recursive=True):
            results.add(os.path.normpath(f))
    return sorted(results)
```

Apply TASK-12 by enumerating files recursively under `self.base_dir`, returning `os.path.normpath(str(path))`, and excluding reserved artifacts such as `.intents`, metadata DB/JSON files, signing keys, directories, and `*.tmp`.

### `src/cacheness/metadata/sqlite_backend.py` (metadata backend, CRUD + transform)

**Analog:** `src/cacheness/metadata/sqlite_backend.py`

**Imports pattern** (line 23):
```python
from ..json_utils import dumps as json_dumps, loads as json_loads
```

**Read path metadata shape** (lines 662-731):
```python
metadata: Dict[str, Any] = {}
if row.actual_path is not None:
    metadata["actual_path"] = row.actual_path
if row.file_hash is not None:
    metadata["file_hash"] = row.file_hash
if row.metadata_dict is not None:
    metadata["metadata_dict"] = row.metadata_dict
return {
    "description": row.description,
    "data_type": row.data_type,
    "file_size": row.file_size,
    "access_count": row.access_count or 0,
    "metadata": metadata,
}
```

For TASK-10, deserialize `row.metadata_dict` with `json_loads` when it is a JSON string, preserve `metadata["metadata_dict"]`, and also expose user fields into the nested `metadata` dict so `BlobStore.get_metadata(key)["metadata"]["experiment"]` and list/filter paths can see parity with JSON.

**Known-field extraction pattern where leftovers currently get dropped** (lines 737-779):
```python
metadata = entry_data.get("metadata", {}).copy()
object_type = metadata.pop("object_type", None)
storage_format = metadata.pop("storage_format", None)
serializer = metadata.pop("serializer", None)
compression_codec = metadata.pop("compression_codec", None)
actual_path = metadata.pop("actual_path", None)
file_hash = metadata.pop("file_hash", None)
entry_signature = metadata.pop("entry_signature", None)
s3_etag = metadata.pop("s3_etag", None)
cache_key_params = metadata.pop("cache_key_params", None)
metadata_dict_value = metadata.pop("metadata_dict", None)
inline_ext = metadata.pop("inline_ext", None)
encryption_algorithm = metadata.pop("encryption_algorithm", None)
encryption_iv = metadata.pop("encryption_iv", None)
cacheness_version = metadata.pop("cacheness_version", None)
metadata.pop("data_type", None)
```

Insert the leftover merge immediately after the known-field pops. If `metadata` still contains user keys, merge them into `metadata_dict_value`; when both sources define the same key, keep the existing/core `metadata_dict_value` key.

**Upsert pattern** (lines 807-860):
```python
session.execute(
    text(f"""
        INSERT INTO "{tbl}"
        (..., cache_key_params, metadata_dict, ...)
        VALUES (..., :cache_key_params, :metadata_dict, ...)
        ON CONFLICT(cache_key) DO UPDATE SET
            cache_key_params = excluded.cache_key_params,
            metadata_dict = excluded.metadata_dict,
            actual_path = excluded.actual_path
    """),
    {
        "cache_key": cache_key,
        "cache_key_params": cache_key_params,
        "metadata_dict": metadata_dict_value,
        "actual_path": actual_path,
    },
)
```

**Summary/filter flattening pattern** (lines 967-1013):
```python
if row[10] is not None:
    flat["actual_path"] = row[10]
if row[13] is not None:
    flat["metadata_dict"] = row[13]
```

If TASK-10 updates public `BlobStore.list(metadata_filter=...)`, this summary shape is the source for efficient filtering.

### `src/cacheness/storage/backends/postgresql_backend.py` (metadata backend, CRUD + transform)

**Analog:** `src/cacheness/storage/backends/postgresql_backend.py`

**JSONB helper pattern** (lines 115-127):
```python
def _ensure_jsonb_value(value):
    """Convert legacy JSON strings to JSONB-compatible Python objects."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            parsed = json_loads(value)
            return parsed if isinstance(parsed, (dict, list)) else value
        except Exception:
            return value
    return value
```

**Session rollback/error pattern** (lines 941-951):
```python
with self._lock:
    with self.SessionLocal() as session:
        try:
            self._upsert_entry(session, cache_key, entry_data)
            session.commit()
        except Exception as e:  # intentionally broad — re-raises after rollback
            session.rollback()
            logger.error(f"Failed to put entry {cache_key}: {e}")
            raise
```

**Known-field extraction and JSONB conversion** (lines 953-1015):
```python
metadata = entry_data.get("metadata", {}).copy()
object_type = metadata.pop("object_type", None)
storage_format = metadata.pop("storage_format", None)
serializer = metadata.pop("serializer", None)
compression_codec = metadata.pop("compression_codec", None)
actual_path = metadata.pop("actual_path", None)
file_hash = metadata.pop("file_hash", None)
entry_signature = metadata.pop("entry_signature", None)
s3_etag = metadata.pop("s3_etag", None)
cache_key_params = metadata.pop("cache_key_params", None)
metadata_dict_value = metadata.pop("metadata_dict", None)
...
jsonb_params = _ensure_jsonb_value(cache_key_params)
jsonb_metadata = _ensure_jsonb_value(metadata_dict_value)
```

Apply the same leftover-user-metadata merge here before `_ensure_jsonb_value(metadata_dict_value)`.

**Update/insert metadata_dict pattern** (lines 1037-1087):
```python
if existing:
    session.execute(
        update(self._PgCacheEntry)
        .where(self._PgCacheEntry.cache_key == cache_key)
        .values(
            actual_path=actual_path,
            cache_key_params=jsonb_params,
            metadata_dict=jsonb_metadata,
        )
    )
else:
    entry = self._PgCacheEntry(
        cache_key=cache_key,
        actual_path=actual_path,
        cache_key_params=jsonb_params,
        metadata_dict=jsonb_metadata,
        access_count=access_count_val,
    )
```

**Read/summary pattern needing user metadata exposure** (lines 1130-1165 and 1289-1331):
```python
if entry.actual_path:
    metadata["actual_path"] = entry.actual_path
if metadata:
    result["metadata"] = metadata

if row[13] is not None:
    flat["metadata_dict"] = row[13]
```

Merge `entry.metadata_dict` into nested `metadata` in `_entry_to_dict`, preserving technical fields on conflicts unless the implementation explicitly documents user fields winning. Keep `iter_entry_summaries()` flat for filtering.

### `src/cacheness/core.py` (service/orchestrator, request-response + file-I/O rollback)

**Analog:** `src/cacheness/core.py`

**Put orchestration and old-path capture** (lines 845-862):
```python
with self._lock:
    cache_key = self._resolve_hash_key_alias(cache_key, hash_key)
    cache_key = self._resolve_cache_key(cache_key, on, kwargs)

    if self.config.storage_mode:
        return self._storage_mode_put(data, cache_key, description)

    base_file_path = self._get_cache_file_path(cache_key)
    cleanup = _PutCleanup()
    old_blob_path: Optional[str] = None
    existing = self.metadata_backend.get_entry(cache_key)
    if existing:
        old_meta = existing.get("metadata", {})
        old_blob_path = old_meta.get("actual_path")
```

TASK-11 should use this existing `old_blob_path` and the planned target path to snapshot only same-local-path overwrites before `_write_blob`.

**Write intent and blob write pattern** (lines 882-910):
```python
planned_blob_path = base_file_path.with_suffix(
    handler.get_file_extension(self.config)
)
self._write_journal.record_intent(
    cache_key, str(planned_blob_path.relative_to(self.cache_dir))
)
wb = self._blob_store._write_blob(
    data,
    base_file_path,
    compute_hash=self.config.metadata.verify_cache_integrity,
)
handler, result, file_hash = wb.handler, wb.result, wb.file_hash

actual_path_str = result.actual_path
if "://" not in actual_path_str:
    cleanup.blob_path = self._resolve_actual_path(actual_path_str)
if "://" in actual_path_str:
    cleanup.set_remote(self._blob_store.blob_backend, actual_path_str)
metadata_dict = self._build_metadata_dict(result, file_hash)
```

**Commit and stale cleanup pattern** (lines 968-987):
```python
self.metadata_backend.put_entry(cache_key, entry_data)
self._write_journal.clear_intent(cache_key)
self._cleanup_stale_blob(cache_key, old_blob_path, result.actual_path)
...
cleanup.commit()
return cache_key
```

Delete `<path>.prev` on `cleanup.commit()` after metadata commit; avoid `_cleanup_stale_blob` deleting the snapshot or the restored previous blob.

**Rollback pattern** (lines 989-1000):
```python
except (OSError, IOError) as e:
    cleanup.rollback()
    self._write_journal.clear_intent(cache_key)
    logger.error(f"Failed to cache {data_type} (I/O error): {e}")
    raise
except Exception as e:
    cleanup.rollback()
    self._write_journal.clear_intent(cache_key)
    logger.error(f"Failed to cache {data_type}: {type(e).__name__}: {e}")
    raise
```

Extend `_PutCleanup.rollback()` so this existing call restores a previous local blob snapshot.

### `src/cacheness/_storage_mode_mixin.py` (service/orchestrator, request-response + file-I/O rollback)

**Analog:** `src/cacheness/_storage_mode_mixin.py`

**Storage-mode old-path and cleanup pattern** (lines 19-31):
```python
from .core import _PutCleanup

base_file_path = self._get_cache_file_path(cache_key)
cleanup = _PutCleanup()
old_blob_path: Optional[str] = None
existing = self.metadata_backend.get_entry(cache_key)
if existing:
    old_meta = existing.get("metadata", {})
    old_blob_path = old_meta.get("actual_path")
```

**Storage-mode write and rollback pattern** (lines 60-110):
```python
planned_blob_path = base_file_path.with_suffix(
    handler.get_file_extension(self.config)
)
self._write_journal.record_intent(
    cache_key, str(planned_blob_path.relative_to(self.cache_dir))
)
wb = self._blob_store._write_blob(data, base_file_path, compute_hash=True)
handler, result, file_hash = wb.handler, wb.result, wb.file_hash

actual_path_str = result.actual_path
if "://" not in actual_path_str:
    cleanup.blob_path = self._resolve_actual_path(actual_path_str)
if "://" in actual_path_str:
    cleanup.set_remote(self._blob_store.blob_backend, actual_path_str)
...
self.metadata_backend.put_entry(cache_key, entry_data)
self._write_journal.clear_intent(cache_key)
self._cleanup_stale_blob(cache_key, old_blob_path, result.actual_path)
cleanup.commit()
return cache_key
...
except Exception as e:
    cleanup.rollback()
    self._write_journal.clear_intent(cache_key)
    raise
```

TASK-11 must mirror cache-mode snapshot behavior here. Skip previous-blob snapshot/restore for inline entries and remote URIs.

### `src/cacheness/_put_cleanup.py` (utility, file-I/O rollback)

**Analog:** `src/cacheness/_put_cleanup.py` plus `tests/test_s3_orphan_cleanup.py`

**Current resource tracking pattern** (lines 10-38):
```python
class _PutCleanup:
    """Tracks resources written during ``put()`` so they can be rolled back."""

    __slots__ = ("_committed", "blob_path", "_blob_backend", "_blob_uri")

    def __init__(self) -> None:
        self._committed = False
        self.blob_path: Optional[Path] = None
        self._blob_backend: Any = None
        self._blob_uri: Optional[str] = None
```

Add slots/state for previous blob snapshot, for example original path and `.prev` path. Keep the API explicit, such as `snapshot_previous_blob(path: Path)`, so both `put()` paths call the same logic.

**Commit/rollback pattern** (lines 45-73):
```python
def commit(self) -> None:
    """Disarm — a subsequent ``rollback()`` will be a no-op."""
    self._committed = True

def rollback(self) -> None:
    """Delete every tracked resource.  Safe to call multiple times."""
    if self._committed:
        return
    if self.blob_path is not None:
        try:
            if self.blob_path.exists():
                self.blob_path.unlink()
        except OSError:
            pass
    if self._blob_backend is not None and self._blob_uri is not None:
        try:
            self._blob_backend.delete_blob(self._blob_uri)
        except Exception:
            logger.warning(...)
    self._committed = True
```

TASK-11 should make `commit()` remove the `.prev` snapshot and make `rollback()` restore `.prev` to the original path after removing the newly written local blob. Cleanup must remain best-effort and idempotent.

**Existing cleanup tests to extend** (`tests/test_s3_orphan_cleanup.py`, lines 80-155):
```python
cleanup = _PutCleanup()
cleanup.blob_path = blob
cleanup.commit()
cleanup.rollback()
assert blob.exists()

cleanup = _PutCleanup()
cleanup.blob_path = blob
cleanup.rollback()
assert not blob.exists()
```

Add tests for commit deleting the `.prev` snapshot, rollback restoring `.prev`, missing `.prev` no-op behavior, and double rollback safety.

### `src/cacheness/storage/blob_store.py` (service, CRUD + file-I/O + transform)

**Analog:** `src/cacheness/storage/blob_store.py`

**BlobStore public metadata write shape** (lines 450-538):
```python
def put(self, data: Any, key: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
    ...
    custom_metadata = metadata or {}
    custom_metadata["actual_path"] = self._to_relative_path(final_path)
    custom_metadata["storage_format"] = result.storage_format
    custom_metadata["compression_codec"] = self.compression
    if file_hash:
        custom_metadata["file_hash"] = file_hash
    entry_data = {
        "cache_key": blob_key,
        "data_type": handler.data_type,
        "file_size": result.file_size,
        "file_hash": file_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": custom_metadata,
    }
```

This is the public API path for TASK-10 parity tests: user fields arrive as leftover nested metadata.

**Public metadata read shape** (lines 669-680):
```python
def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
    with self._lock:
        return self.backend.get_entry(key)
```

Tests should check `store.get_metadata(key)["metadata"]["experiment"] == "x42"` unless the existing test expectation intentionally flattens through another helper.

**List/filter pattern** (lines 911-949):
```python
entries = self.backend.list_entries()
for entry in entries:
    key = entry.get("cache_key", "")
    if metadata_filter:
        match = True
        for field, value in metadata_filter.items():
            if entry.get(field) != value:
                match = False
                break
        if not match:
            continue
    keys.append(key)
```

If user metadata stays nested in backend entries, TASK-10 should either flatten metadata_dict/user fields in backend `list_entries()` or teach this filter to check nested `metadata`/`metadata_dict` consistently.

**Integrity inventory pattern** (lines 981-1034):
```python
blob_files: set[str] = set(self.blob_backend.list_blobs())
entry_paths: dict[str, dict] = {}
for entry in self.backend.iter_entry_summaries():
    actual_path = entry.get("actual_path")
    if actual_path:
        norm_path = self._resolve_actual_path(actual_path)
        entry_paths[norm_path] = {
            "cache_key": entry.get("cache_key", ""),
            "file_size": entry.get("file_size"),
            "file_hash": entry.get("file_hash"),
            "s3_etag": entry.get("s3_etag"),
        }
known_paths = set(entry_paths.keys())
orphaned_blobs = sorted(blob_files - known_paths)
```

TASK-12 belongs in the backend because `verify_integrity()` already delegates blob inventory to `blob_backend.list_blobs()`.

### `tests/test_blob_namespace.py` (test, file-I/O)

**Analog:** `tests/test_blob_namespace.py`

**Namespace and write/read test pattern** (lines 59-73):
```python
backend = FilesystemBlobBackend(tmp_path / "blobs", shard_chars=0)
path = backend.write_blob("test_blob", b"hello default")
assert Path(path).parent == tmp_path / "blobs" / "default"
assert backend.read_blob(path) == b"hello default"

backend = FilesystemBlobBackend(tmp_path / "blobs", shard_chars=0, namespace="staging")
path = backend.write_blob("test_blob", b"hello staging")
assert Path(path).parent == tmp_path / "blobs" / "staging"
assert backend.read_blob(path) == b"hello staging"
```

TASK-9 should add repeated same-blob overwrite coverage here: write `blob_id` with A, write the same `blob_id` with B, assert final path reads B and no `*.tmp` files remain under the namespace directory.

**Namespace isolation pattern** (lines 75-110):
```python
default_path = default_backend.write_blob("shared_id", b"default_data")
custom_path = custom_backend.write_blob("shared_id", b"analytics_data")
assert default_path != custom_path
assert default_backend.read_blob(default_path) == b"default_data"
assert custom_backend.read_blob(custom_path) == b"analytics_data"
```

### `tests/test_backend_parity.py` (test, CRUD + parity)

**Analog:** `tests/test_backend_parity.py`

**Imports and backend helpers** (lines 21-36, 74-105):
```python
from cacheness.config import CacheConfig, CacheMetadataConfig, CacheStorageConfig, CompressionConfig, SecurityConfig  # noqa: E402  # fmt: skip
from cacheness.core import UnifiedCache as cacheness  # noqa: E402
from cacheness.storage.blob_store import BlobStore  # noqa: E402

def _get_pg_url():
    return os.environ.get("CACHENESS_TEST_POSTGRES_URL")

def _make_encrypted_blobstore_for_backend(tmp_path, backend, **overrides):
    metadata_cfg = CacheMetadataConfig(metadata_backend=backend)
    if backend == "postgresql":
        metadata_cfg = CacheMetadataConfig(
            metadata_backend="postgresql",
            metadata_backend_options={"connection_url": _get_pg_url()},
        )
    ...
    if backend == "postgresql":
        backend_arg = PostgresBackend(connection_url=_get_pg_url())
    return BlobStore(cache_dir=tmp_path, backend=backend_arg, enable_signing=True, ...)
```

**Existing SQLite backend parity test style** (lines 140-165):
```python
entry_data = {
    "cache_key": "test_key_123",
    "description": "Test entry",
    "data_type": "array",
    "file_size": 2048,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "metadata": {
        "s3_etag": "test_etag_123",
        "actual_path": "/test/path.npz",
        "object_type": "<class 'numpy.ndarray'>",
        "storage_format": "numpy",
    },
}
sqlite_backend.put_entry("test_key_123", entry_data)
retrieved = sqlite_backend.get_entry("test_key_123")
assert retrieved is not None
```

**PostgreSQL skip gate and parametrization pattern** (lines 587-610):
```python
def _skip_if_pg_unavailable(backend):
    """Skip test if backend is postgresql and PG is not available."""
    if backend == "postgresql":
        if not _HAS_PG or not _get_pg_url():
            pytest.skip("PostgreSQL not available")

@pytest.mark.xdist_group("docker")
class TestEncryptionBackendParity_BlobStore:
    @pytest.mark.parametrize("backend", ["json", "sqlite", "postgresql"])
    def test_encrypted_put_get_roundtrip(self, tmp_path, backend):
        _skip_if_pg_unavailable(backend)
        store = _make_encrypted_blobstore_for_backend(tmp_path, backend)
```

TASK-10 should add a BlobStore public parity test following this pattern, with JSON and SQLite always covered and PostgreSQL skip-gated.

### `tests/test_fault_injection.py` (test, fault-injection + rollback)

**Analog:** `tests/test_fault_injection.py`

**Metadata failure injection pattern** (lines 60-77):
```python
cache = _make_cache(tmp_path)
data = {"key": "value", "numbers": [1, 2, 3]}

with patch.object(
    cache.metadata_backend,
    "put_entry",
    side_effect=RuntimeError("Simulated metadata write failure"),
):
    with pytest.raises(RuntimeError, match="Simulated metadata write failure"):
        cache.put(data, test_key="orphan_test")

blob_files = list(tmp_path.rglob("*.pkl*"))
assert blob_files == []
```

TASK-11 should reuse `patch.object(cache.metadata_backend, "put_entry", side_effect=...)`, but first commit value A, then fail value B for the same key, and assert `cache.get(...)` returns A and the old blob path exists.

### `tests/test_storage_mode.py` (test, storage-mode request-response)

**Analog:** `tests/test_storage_mode.py`

**Storage-mode fixture** (lines 27-32):
```python
config = CacheConfig(
    cache_dir=str(tmp_path / "store"),
    storage_mode=True,
)
return UnifiedCache(config=config)
```

**Storage-mode put/get assertion style** (lines 107-116):
```python
storage_cache.put("hello", cache_key="k1")
assert storage_cache.get(cache_key="k1") == "hello"

storage_cache.put("data", cache_key="k2")
storage_cache._cleanup_expired()
assert storage_cache.exists(cache_key="k2")
```

TASK-11 should add the storage-mode same-key failure regression here or in `tests/test_fault_injection.py`, using this fixture shape.

### `tests/test_cache_integrity_verification.py` (test, integrity enumeration + repair)

**Analog:** `tests/test_cache_integrity_verification.py`

**Orphan creation/detection pattern** (lines 87-101):
```python
cache = _make_cache(tmp_path)
cache.put("real data", test_key="real")

ns_dir = tmp_path / "default"
ns_dir.mkdir(exist_ok=True)
orphan = ns_dir / "orphaned_file.pkl"
orphan.write_bytes(b"fake pickle data")

report = cache.verify_integrity()

assert len(report["orphaned_blobs"]) == 1
assert os.path.normpath(str(orphan)) in report["orphaned_blobs"]
assert report["dangling_entries"] == []
```

TASK-12 should copy this test shape with `orphan.custom` or `orphan.bin` to prove extension-agnostic enumeration.

**Repair pattern** (lines 140-151):
```python
orphan = ns_dir / "orphaned.pkl"
orphan.write_bytes(b"garbage")

report = cache.verify_integrity(repair=True)

assert report["repaired"]["orphans_deleted"] == 1
assert not orphan.exists()
```

Add repair coverage only if it falls out naturally from the new enumeration behavior.

## Shared Patterns

### JSON Serialization
**Source:** `src/cacheness/json_utils.py` lines 20-49 and 58-82
**Apply to:** SQLite and PostgreSQL metadata parity work
```python
def dumps(obj: Any, sort_keys: bool = False, default: Any = None) -> str:
    return orjson.dumps(obj, option=option, default=default).decode("utf-8")

def loads(s: Union[str, bytes]) -> Any:
    return orjson.loads(s)
```

Use `json_dumps`/`json_loads` aliases already imported by SQLite and PostgreSQL. Do not introduce raw `json.dumps` in backend logic.

### Rollback Cleanup
**Source:** `src/cacheness/_put_cleanup.py` lines 45-73
**Apply to:** `core.py`, `_storage_mode_mixin.py`, `_put_cleanup.py`, fault-injection tests
```python
cleanup.commit()
...
except Exception:
    cleanup.rollback()
    raise
```

Keep rollback best-effort and idempotent. Add previous-blob snapshot/restore to `_PutCleanup` so cache mode and storage mode share behavior.

### Backend Inventory
**Source:** `src/cacheness/storage/blob_store.py` lines 1012-1034
**Apply to:** `FilesystemBlobBackend.list_blobs`, integrity tests
```python
blob_files: set[str] = set(self.blob_backend.list_blobs())
...
orphaned_blobs = sorted(blob_files - known_paths)
```

Integrity checks already depend on backend inventory, so TASK-12 should avoid new integrity-side special cases.

### PostgreSQL Availability
**Source:** `tests/test_backend_parity.py` lines 587-610
**Apply to:** TASK-10 PostgreSQL tests
```python
def _skip_if_pg_unavailable(backend):
    if backend == "postgresql":
        if not _HAS_PG or not _get_pg_url():
            pytest.skip("PostgreSQL not available")
```

Keep PostgreSQL tests under existing skip gates and `@pytest.mark.xdist_group("docker")`.

### Verification Commands
**Source:** `.github/copilot-instructions.md` and phase context
**Apply to:** all TASK-9 through TASK-12 plans
```powershell
uv run pytest tests/test_blob_store.py tests/test_blob_namespace.py -x -q --ignore=tests/test_tensorflow_handler.py
uv run pytest tests/test_blob_store.py tests/test_metadata.py tests/test_backend_parity.py -x -q --ignore=tests/test_tensorflow_handler.py
uv run pytest tests/test_core.py tests/test_fault_injection.py tests/test_cache_integrity.py tests/test_storage_mode.py -x -q --ignore=tests/test_tensorflow_handler.py
uv run pytest tests/test_blob_store.py tests/test_cache_integrity_verification.py -x -q --ignore=tests/test_tensorflow_handler.py
```

For changed Python files, also plan `uv run ruff format`, `uv run ruff check --fix`, `uv run ruff check`, and `uv run ty check` on touched files.

## No Analog Found

No files are without analogs. Every planned source and test file has an exact same-file pattern or a direct existing test analog.

## Metadata

**Analog search scope:** `src/cacheness/storage/backends`, `src/cacheness/storage`, `src/cacheness/metadata`, `src/cacheness/core.py`, `src/cacheness/_storage_mode_mixin.py`, `src/cacheness/_put_cleanup.py`, and focused tests named in the phase prompt.
**Files scanned:** 13 focused source/test files plus `AGENTS.md`, `.github/copilot-instructions.md`, `30-CONTEXT.md`, `30-RESEARCH.md`, and `src/cacheness/json_utils.py`.
**Pattern extraction date:** 2026-06-14
