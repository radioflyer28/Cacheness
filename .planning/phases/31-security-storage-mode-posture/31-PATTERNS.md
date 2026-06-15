# Phase 31: Security & Storage-Mode Posture - Pattern Map

**Mapped:** 2026-06-15
**Files analyzed:** 29
**Analogs found:** 21 / 21

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `src/cacheness/config.py` | config | request-response | `SecurityConfig` in `src/cacheness/config.py` | exact |
| `src/cacheness/security.py` | service | transform | `CacheEntrySigner.verify_entry()` and key helpers | exact |
| `src/cacheness/_verification_mixin.py` or new shared helper | utility | transform | `_extract_signable_fields()` | exact |
| `src/cacheness/storage/blob_store.py` | service | file-I/O | `_write_blob()`, `_read_blob()`, `rotate_key()` | exact |
| `src/cacheness/core.py` | controller/service | CRUD + file-I/O | `put()`, `rotate_key()`, cleanup APIs | exact |
| `src/cacheness/_storage_mode_mixin.py` | service | CRUD + file-I/O | `_storage_mode_put()` / `_storage_mode_get()` | exact |
| `src/cacheness/write_intent.py` | utility | file-I/O | `WriteIntentJournal` | exact |
| `src/cacheness/storage/backends/blob_backends.py` | service | file-I/O | `FilesystemBlobBackend.write_blob()` | exact |
| `src/cacheness/metadata/json_backend.py` | model/backend | file-I/O | `_save_to_disk()` / `_load_from_disk()` | exact |
| `docs/SECURITY.md` | docs | request-response | signing and encryption sections | exact |
| `docs/TRANSACTION_GUARANTEES.md` | docs | file-I/O | durability and storage-mode sections | exact |
| `tests/test_cache_signing.py` | test | transform | cache signing fixtures | role-match |
| `tests/test_blob_store.py` | test | file-I/O | BlobStore signing/integrity tests | exact |
| `tests/test_key_rotation.py` | test | transform + file-I/O | legacy key rotation tests | exact |
| `tests/test_key_rotation_api.py` | test | transform + file-I/O | rotate_key API tests | exact |
| `tests/test_storage_mode.py` | test | CRUD | storage-mode invariant tests | exact |
| `tests/test_atomic_writes.py` | test | file-I/O | atomic rename tests | exact |
| `tests/test_write_intent.py` | test | file-I/O | stale intent cleanup tests | exact |
| `tests/test_handler_bytes_protocol.py` | test | transform | `get_bytes()` protocol tests | exact |
| `tests/test_encryption_at_rest.py` | test | file-I/O | encrypted round-trip and rotation tests | exact |

## Pattern Assignments

### Signature Downgrade and Unsigned-Entry Risk

**Primary targets:** `src/cacheness/config.py`, `src/cacheness/security.py`, `src/cacheness/_verification_mixin.py`, `docs/SECURITY.md`, signing tests.

**Config pattern:** add new security options directly to `SecurityConfig`, near related signing compatibility fields. Current fields are in `src/cacheness/config.py:389`, with `allow_unsigned_entries` at `src/cacheness/config.py:400` and HKDF at `src/cacheness/config.py:412`.

```python
class SecurityConfig:
    enable_entry_signing: bool = True
    signing_key_file: str = "cache_signing_key.bin"
    use_in_memory_key: bool = False
    allow_unsigned_entries: bool = True
    delete_invalid_signatures: bool = True
    use_hkdf_derivation: bool = True
```

**Verification pattern:** version parsing is centralized in `CacheEntrySigner.parse_versioned_signature()` and used by `verify_entry()` at `src/cacheness/security.py:335` and `src/cacheness/security.py:352`. Add minimum-version rejection immediately after parsing and before payload construction.

```python
version, hex_sig = self.parse_versioned_signature(stored_signature)
payload = self._create_signature_payload(entry_data, version)
key = self.derived_key if version >= 3 else self.master_key
is_valid = hmac.compare_digest(expected_signature, hex_sig)
```

**Caller pattern:** `_verify_entry()` handles unsigned entries and storage-mode preservation at `src/cacheness/_verification_mixin.py:78` and `src/cacheness/_verification_mixin.py:176`. Keep storage mode non-destructive: reject by returning `False`, but do not delete.

```python
elif not self.config.security.allow_unsigned_entries:
    self._invoke_hook(..., "unsigned_rejected", ...)
    if storage_mode:
        logger.warning(...)
    else:
        self._blob_store.delete(cache_key)
    return False
```

**Tests to copy:** use `tests/test_cache_signing.py:24` and `tests/test_key_rotation.py:73` for strict security config setup. Add the downgrade regression beside existing signing tests: sign a v3 entry, mutate `entry_signature` from `v3:` to `v2:`, set `minimum_signature_version=3`, assert `get()` or direct verification fails.

**Pitfalls:** do not flip `allow_unsigned_entries` default. Do not let SEC-01 reject old signatures unless users opt into `minimum_signature_version=3`.

### Shared Canonical Signing Fields

**Primary targets:** `src/cacheness/_verification_mixin.py`, likely a new shared helper module, `src/cacheness/storage/blob_store.py`.

**Analog:** `VerificationMixin._extract_signable_fields()` at `src/cacheness/_verification_mixin.py:16` is the canonical field normalizer. Move or wrap this logic so `UnifiedCache` and `BlobStore` use the same superset.

```python
created_at = entry_data.get("created_at")
if isinstance(created_at, datetime):
    created_at = created_at.replace(tzinfo=None).isoformat()

signable_data = {
    "cache_key": cache_key,
    "data_type": entry_data.get("data_type"),
    "file_size": entry_data.get("file_size", 0),
    "created_at": created_at,
    "actual_path": metadata.get("actual_path", ""),
    "file_hash": metadata.get("file_hash"),
    "object_type": metadata.get("object_type"),
    "storage_format": metadata.get("storage_format"),
    "serializer": metadata.get("serializer"),
    "compression_codec": metadata.get("compression_codec"),
}
```

**Current mismatch to replace:** `BlobStore.rotate_key()` signs `{**full_entry, **nested_meta, "cache_key": cache_key}` at `src/cacheness/storage/blob_store.py:358`; `BlobStore.put()` also uses flattened metadata. Preserve compatibility for those old shapes, but new writes should use the shared helper.

**Tests to copy:** `tests/test_blob_store.py:188` checks stored signatures; `tests/test_cache_signing.py` covers cache signing; SEC-04 should add parity between equivalent `UnifiedCache` and `BlobStore` metadata.

**Pitfalls:** `created_at` normalization is part of the signature contract. Do not sign mutable access fields or custom metadata accidentally.

### Encrypted Reads Through Blob Backends

**Primary target:** `src/cacheness/storage/blob_store.py`.

**Unsafe analog to replace:** encrypted read branches currently use local path reads and plaintext temp files. See `src/cacheness/storage/blob_store.py:625` and `src/cacheness/storage/blob_store.py:1303`.

```python
ciphertext = path.read_bytes()
plaintext = decrypt_blob(ciphertext, self._encryption_key, iv)
tmp_fd = tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=path.suffix)
...
return handler.get(read_path, handler_metadata)
```

**Backend abstraction pattern:** route bytes through `BlobBackend.read_blob()` from `src/cacheness/storage/backends/blob_backends.py:90`. This keeps `memory://` and S3 viable.

```python
def read_blob(self, blob_path: str) -> bytes:
    """Read blob data from storage."""
```

**Handler byte protocol pattern:** tests in `tests/test_handler_bytes_protocol.py:46`, `tests/test_handler_bytes_protocol.py:84`, and `tests/test_handler_bytes_protocol.py:118` establish that default `get_bytes()` raises `NotImplementedError`, while bytes/object handlers can deserialize in memory. Encrypted reads should try `handler.get_bytes(plaintext, metadata)` first and only fall back on `NotImplementedError`.

**Temp fallback pattern:** if fallback is required, use `tempfile.mkstemp` in the cache directory and unlink in `finally`. Follow the unique-temp discipline from `FilesystemBlobBackend.write_blob()` at `src/cacheness/storage/backends/blob_backends.py:311`.

```python
fd, temp_name = tempfile.mkstemp(dir=blob_path.parent, suffix=".tmp")
with os.fdopen(fd, "wb") as f:
    f.write(data)
os.replace(temp_path, blob_path)
```

**Tests to copy:** `tests/test_encryption_at_rest.py:139`, `tests/test_encryption_at_rest.py:213`, and `tests/test_backend_parity.py:624` cover encrypted BlobStore/cache round trips. Add an `InMemoryBlobBackend` encrypted round trip and assert no stale decrypt temp files.

**Pitfalls:** hash verification is over ciphertext. Do not compute or compare `file_hash` over decrypted plaintext for file-backed encrypted blobs.

### Two-Phase Key Rotation

**Primary targets:** `src/cacheness/core.py`, `src/cacheness/storage/blob_store.py`, `src/cacheness/security.py`.

**Current unsafe pattern:** both rotation implementations overwrite the active key first, then re-sign entries. See `src/cacheness/core.py:415` and `src/cacheness/storage/blob_store.py:333`.

```python
dest = Path(self.signer.key_file_path)
dest.write_bytes(new_key_bytes)
new_signer = create_cache_signer(...)
```

**Current unsafe re-encryption pattern:** encrypted blobs are rewritten in place at `src/cacheness/core.py:511` and `src/cacheness/core.py:516`, mirrored in BlobStore at `src/cacheness/storage/blob_store.py:406` and `src/cacheness/storage/blob_store.py:409`.

```python
ciphertext = blob_path.read_bytes()
plaintext = decrypt_blob(ciphertext, old_enc_key, old_iv)
new_ciphertext, new_iv, _ = encrypt_blob(plaintext, new_enc_key)
blob_path.write_bytes(new_ciphertext)
```

**Key file helper pattern:** key generation writes 32 bytes then sets restrictive permissions at `src/cacheness/security.py:184` and `src/cacheness/security.py:226`. Reuse `_set_key_file_permissions()` for `<keyfile>.new`.

```python
self.key_file_path.write_bytes(key)
self._set_key_file_permissions(self.key_file_path)
```

**Required replacement pattern:** write `<keyfile>.new`, build a new signer from those bytes, verify each entry with the old signer before mutation, write re-encrypted blobs to `<blob>.rotating`, `os.replace()` only after success, then replace the active key file last. On startup, detect leftover `<keyfile>.new` and log an error.

**Tests to copy:** `tests/test_key_rotation_api.py:47`, `tests/test_key_rotation_api.py:199`, `tests/test_encryption_at_rest.py:297`, and `tests/test_key_rotation.py:73`. Add fault injection that raises mid-rotation and confirms the original key file still verifies old entries.

**Pitfalls:** `RotationResult.failed` currently tracks re-sign failures but re-encrypt failures append without incrementing `failed` in BlobStore. Preserve or fix accounting deliberately in tests.

### Storage-Mode Destructive API Warning Policy

**Primary targets:** `src/cacheness/core.py`, `src/cacheness/_storage_mode_mixin.py`, docs/tests.

**Storage-mode invariant pattern:** storage-mode reads call `_verify_entry(..., storage_mode=True)` and never delete on failure. See `src/cacheness/_storage_mode_mixin.py:121`.

```python
if not self._verify_entry(cache_key, entry, metadata, file_path, storage_mode=True):
    return None
```

**Config guard pattern:** storage mode disables cache behaviors in config; relevant docs/tests are in `tests/test_storage_mode.py:56` and `docs/TRANSACTION_GUARANTEES.md:34`.

**Destructive APIs to guard/warn:** `clear_all()` at `src/cacheness/core.py:1195`, `clear_all_namespaces()` at `src/cacheness/core.py:1215`, `cleanup_expired()` at `src/cacheness/core.py:1258`, and size enforcement at `src/cacheness/core.py` around `_enforce_size_limit`. Phase decision is warning-first, not hard raise.

**Tests to copy:** `tests/test_storage_mode.py:113` for cleanup no-op expectations, `tests/test_core.py:1474` for size enforcement disabled, and storage-mode corruption preservation tests in `tests/test_transaction_ordering.py:149`.

**Pitfalls:** warning should trigger for explicit destructive public APIs in storage mode, but default storage-mode operation must not reintroduce TTL, eviction, invalid-signature deletion, or corrupt-entry deletion.

### Storage-Mode Durability and Opt-In Fsync

**Primary targets:** `src/cacheness/config.py`, `src/cacheness/storage/backends/blob_backends.py`, `src/cacheness/metadata/json_backend.py`, `src/cacheness/write_intent.py`, `docs/TRANSACTION_GUARANTEES.md`.

**Atomic write pattern:** local blob writes use unique temp files plus `os.replace()` at `src/cacheness/storage/backends/blob_backends.py:311` and streams at `src/cacheness/storage/backends/blob_backends.py:371`.

**JSON atomic save pattern:** `JsonBackend._save_to_disk()` uses `tempfile.mkstemp`, writes JSON, then `shutil.move()` at `src/cacheness/metadata/json_backend.py:91`. Data-critical `put_entry()` and `remove_entry()` call `_save_to_disk(raise_on_error=True)` at `src/cacheness/metadata/json_backend.py:139`.

**Intent file pattern:** `WriteIntentJournal.record_intent()` writes a small JSON file in `.intents` at `src/cacheness/write_intent.py:45`.

**Docs pattern:** `docs/TRANSACTION_GUARANTEES.md:84`, `docs/TRANSACTION_GUARANTEES.md:96`, and `docs/TRANSACTION_GUARANTEES.md:281` already distinguish atomic rename from explicit fsync. Update this style, but add storage-mode-specific durability caveats and the opt-in `fsync_on_write` behavior.

**Implementation convention:** add `fsync_on_write: bool = False` near storage/blob durability config, default false. When enabled and the code owns a local file descriptor, flush and `os.fsync(fd)` before rename; after rename, fsync the parent directory where feasible. For metadata JSON and intents, do not swallow data-critical fsync failures if `raise_on_error=True`.

**Tests to copy:** `tests/test_atomic_writes.py:25` for atomic rename behavior and `tests/test_write_intent.py:32` for intent file lifecycle. New tests should monkeypatch `os.fsync` or a helper hook; do not claim power-loss simulation.

**Pitfalls:** S3/PostgreSQL durability comes from backend infrastructure. Do not force fsync by default or silently degrade cache-mode performance.

### Pre-Blob Write Intent Coverage

**Primary targets:** `src/cacheness/core.py`, `src/cacheness/_storage_mode_mixin.py`, `src/cacheness/write_intent.py`.

**Cache-mode pattern:** `UnifiedCache.put()` records the planned relative blob path before `_write_blob()` at `src/cacheness/core.py:893`.

```python
planned_blob_path = base_file_path.with_suffix(handler.get_file_extension(self.config))
self._write_journal.record_intent(
    cache_key, str(planned_blob_path.relative_to(self.cache_dir))
)
wb = self._blob_store._write_blob(data, base_file_path, compute_hash=True)
```

**Storage-mode pattern:** `_storage_mode_put()` mirrors the same ordering at `src/cacheness/_storage_mode_mixin.py:71`.

**Cleanup pattern:** `WriteIntentJournal.cleanup_stale_intents()` resolves relative paths against `cache_dir`, checks committed metadata via `entry_exists`, and tolerates missing blobs. See `src/cacheness/write_intent.py:70`.

```python
if entry_exists(cache_key):
    intent_path.unlink(missing_ok=True)
    continue
bp = Path(blob_path)
if not bp.is_absolute():
    bp = self._cache_dir / bp
if bp.exists():
    bp.unlink()
```

**Tests to copy:** `tests/test_write_intent.py:91`, `tests/test_write_intent.py:108`, `tests/test_write_intent.py:202`, `tests/test_write_intent.py:245`, `tests/test_write_intent.py:263`, and `tests/test_write_intent.py:278`.

**Pitfalls:** inline writes should still skip intents. Stale cleanup must not delete a blob if committed metadata exists for the key.

## Shared Patterns

### Backend Abstraction

**Source:** `src/cacheness/storage/backends/blob_backends.py`

Use `blob_backend.write_blob_from_path()`, `read_blob()`, `delete_blob()`, `exists()`, and `list_blobs()` rather than direct `Path` I/O whenever the stored path may be `memory://`, S3, or another backend URI.

### Error Handling

**Source:** existing cache code.

Operational cleanup is best-effort and logs warnings; data-critical writes should raise. JSON shows this split with `_save_to_disk(raise_on_error=True)` for `put_entry()`/`remove_entry()` and best-effort for stats/access updates.

### Storage Mode

**Source:** `src/cacheness/_storage_mode_mixin.py`, `docs/TRANSACTION_GUARANTEES.md`.

Storage mode preserves entries on read/verification failures and disables implicit cache behavior. Any explicit destructive API added or changed in Phase 31 should warn loudly but should not hard-raise by default.

### Test Commands

Use `uv` only. On Windows include the TensorFlow ignore flag:

```powershell
uv run --python 3.12 pytest tests/test_cache_signing.py tests/test_key_rotation.py tests/test_key_rotation_api.py tests/test_encryption_at_rest.py -x -q --ignore=tests/test_tensorflow_handler.py
```

`tests/test_security.py` appears in the source review's suggested commands but is absent in the current tree; map those checks to `tests/test_cache_signing.py`, `tests/test_key_rotation.py`, `tests/test_key_rotation_api.py`, `tests/test_encryption_at_rest.py`, and targeted new tests unless the planner intentionally creates a new security test file.

## No Analog Found

| File | Role | Data Flow | Reason |
|---|---|---|---|
| `src/cacheness/signing_fields.py` or equivalent new helper | utility | transform | No standalone helper exists; copy `_extract_signable_fields()` from `VerificationMixin`. |
| `minimum_signature_version` tests | test | transform | Existing signing tests cover signatures but not downgrade rejection; create adjacent tests. |
| `fsync_on_write` tests | test | file-I/O | Atomic write tests exist, but no fsync policy exists yet; monkeypatch a helper or `os.fsync`. |

## Metadata

**Analog search scope:** `src/cacheness`, `tests`, `docs`.
**Files scanned:** source, docs, and targeted Phase 31 test files listed in context.
**Pattern extraction date:** 2026-06-15.

## PATTERN MAP COMPLETE
