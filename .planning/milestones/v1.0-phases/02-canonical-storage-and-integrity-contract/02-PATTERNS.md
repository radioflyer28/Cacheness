# Phase 2: Canonical Storage and Integrity Contract - Pattern Map

**Mapped:** 2026-08-30  
**Files analyzed:** 17 new/modified paths  
**Analogs found:** 17 / 17 (the manifest wire model and raw-record adapter are new boundaries, but have close typed/config/backend analogs)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| src/cacheness/storage/manifest.py | model/utility | transform + request-response | src/cacheness/config.py:20-42,1032-1144; src/cacheness/json_utils.py:18-66 | role-match; new wire contract |
| src/cacheness/storage/manifest_repository.py | service/backend | CRUD + file I/O | src/cacheness/metadata.py:233-296,543-762,768-1337,1340-1682 | role-match; new raw-byte seam |
| src/cacheness/storage/integrity.py | utility/service | streaming file I/O + transform | src/cacheness/security.py:75-191; src/cacheness/storage/guarded_handler_io.py:301-367 | role-match |
| src/cacheness/storage/read_contract.py | utility/service | request-response + transform | src/cacheness/error_handling.py:18-152; src/cacheness/storage/blob_store.py:278-319 | role-match |
| src/cacheness/storage/blob_store.py | service/coordinator | file I/O + CRUD | src/cacheness/storage/blob_store.py:128-371 | exact extension point |
| src/cacheness/storage/__init__.py | provider/barrel | request-response | src/cacheness/storage/__init__.py:25-84 | exact |
| src/cacheness/error_handling.py | utility/error model | request-response | src/cacheness/error_handling.py:18-152 | exact extension |
| src/cacheness/security.py | utility/provider | transform + file I/O | src/cacheness/security.py:75-191 | exact legacy signer to isolate/harden |
| src/cacheness/interfaces.py | model/interface | file I/O | src/cacheness/interfaces.py:35-173 | exact extension |
| src/cacheness/handlers.py | component/serializer | file I/O + transform | src/cacheness/handlers.py:546-738,882-1123 | exact extension |
| tests/test_blob_manifest.py | test | transform + request-response | tests/test_interfaces.py:34-176; tests/test_config_validation.py | role-match |
| tests/test_blob_manifest_backends.py | test/integration | CRUD + file I/O | tests/test_metadata.py; tests/test_clear_recovery.py:1-190,1280-1375 | role-match |
| tests/test_blob_store_read_contract.py | test/integration | request-response + file I/O | tests/test_stored_compatibility.py:533-680 | exact behavior analog |
| tests/test_blob_store_integrity.py | test/integration | streaming file I/O | tests/test_cache_integrity.py:75-240,390-552 | exact behavior analog |
| tests/test_blob_store_legacy_contract.py | test/compatibility | file I/O + transform | tests/test_stored_compatibility.py:166-370,480-680 | exact behavior analog |
| tests/test_blob_store_translation_seam.py | test/utility | request-response | tests/test_error_handling.py; tests/test_cache_integrity.py:145-240 | role-match |

The Phase 2 research explicitly recommends the first seven implementation/test areas. interfaces.py, handlers.py, and security.py are included because it also requires independent handler format versions, non-deserializing support checks, and strict canonical signing. They are integration points, not permission to redesign the legacy cache path. Keep UnifiedCache, SqlCache, PostgreSQL, S3, generalized CAS/recovery, and migration execution out of this phase.

## Pattern Assignments

### src/cacheness/storage/manifest.py (model/utility, transform + request-response)

Analog: src/cacheness/config.py:20-42 for dataclass validation and src/cacheness/json_utils.py:18-66 for fallback conventions. No existing canonical wire model exists; freeze the selected field names and encoding with golden bytes.

Existing model pattern (src/cacheness/config.py:20-42):

    @dataclass
    class CacheStorageConfig:
        cache_dir: str = "./cache"
        max_cache_size_mb: Optional[int] = 2000

        def __post_init__(self):
            if self.max_cache_size_mb is not None and self.max_cache_size_mb <= 0:
                raise ValueError("max_cache_size_mb must be positive")

Use @dataclass(frozen=True) for BlobManifestV1. Validate all required structural fields in a focused validator; separate immutable structural/handler fields from mutable user_metadata. Do not resolve locators or invoke handlers in the model.

The existing JSON utility can select orjson, but canonical signatures must not depend on whichever backend is installed. Use restricted stdlib JSON: sorted keys, compact separators, ensure_ascii=False, allow_nan=False, UTF-8 bytes, plus explicit raw-byte/depth/string/collection bounds. Sign every canonical field except the signature value. Unknown schema or payload versions fail before locator-driven I/O.

### src/cacheness/storage/manifest_repository.py (service/backend, CRUD + file I/O)

Analog: the abstract metadata contract in src/cacheness/metadata.py:233-296 and locking/persistence implementations in InMemoryBackend (:543-762), JsonBackend (:768-1337), and SqliteBackend (:1340-1682). This is a raw-byte repository, not an extension of backend-shaped put_entry().

Existing contract pattern (src/cacheness/metadata.py:233-272):

    class MetadataBackend(ABC):
        @abstractmethod
        def get_entry(self, cache_key: str) -> Optional[Dict[str, Any]]:
            pass

        @abstractmethod
        def put_entry(self, cache_key: str, entry_data: Dict[str, Any]):
            pass

        @abstractmethod
        def remove_entry(self, cache_key: str):
            pass

        @abstractmethod
        def list_entries(self) -> List[Dict[str, Any]]:
            pass

Define get_raw(key) -> bytes | None, put_raw(key, bytes), remove(key), and list_keys(). JSON and memory adapters preserve exact canonical bytes (or a provably reversible representation). SQLite creates a dedicated table keyed by logical key and stores bytes as a BLOB. Do not pass manifests through SqliteBackend.put_entry(): its fixed-column extraction at src/cacheness/metadata.py:1595-1680 drops generation and payload-format version.

Preserve per-backend locks, copy-on-write persistence, and raise ... from exc. Repository methods do not decode, authenticate, update access bookkeeping, or delete evidence. Unsupported backend identities raise a typed configuration/backend error instead of claiming PostgreSQL or arbitrary injected-backend parity.

### src/cacheness/storage/integrity.py (utility/service, streaming file I/O + transform)

Analog: src/cacheness/security.py:75-191 and src/cacheness/storage/guarded_handler_io.py:301-367. Reuse standard-library primitives and the existing snapshot owner; do not copy the legacy configurable field subset or string-concatenation wire format.

Legacy signer shape to isolate (src/cacheness/security.py:162-191):

    def _create_signature_payload(self, entry_data: Dict[str, Any]) -> str:
        values = []
        for field in sorted(self.signed_fields):
            value = entry_data.get(field)
            if value is None:
                value = ""
            elif isinstance(value, datetime):
                value = value.isoformat()
            else:
                value = str(value)
            values.append(f"{field}:{value}")
        return "|".join(values)

    def sign_entry(self, entry_data: Dict[str, Any]) -> str:
        payload = self._create_signature_payload(entry_data)
        return hmac.new(self.secret_key, payload.encode("utf-8"), hashlib.sha256).hexdigest()

Canonical integrity signs deterministic canonical bytes with HMAC-SHA256 and verifies with hmac.compare_digest; the v1 projection is fixed and complete. Required signing rejects missing/invalid/unsafe key material, missing signatures, and unsupported algorithms without auto-generating a replacement key or downgrading unsigned.

Same-snapshot pattern (src/cacheness/storage/guarded_handler_io.py:337-367):

    with self._private_stage() as stage_root:
        snapshot_path = stage_root / f"snapshot{suffix}"
        with snapshot_path.open("xb") as destination:
            self.file_ops.copy_to_stream(managed_locator, destination)
            destination.flush()
            os.fsync(destination.fileno())
        snapshot_metadata = dict(metadata)
        snapshot_metadata["actual_path"] = str(snapshot_path)
        yield GuardedReadSnapshot(snapshot_path, snapshot_metadata)

Hash and count bytes from this private snapshot and invoke handler.get() on that same snapshot before the context exits. Never hash a managed path, close it, and ask the handler to reopen it.

### src/cacheness/storage/read_contract.py (utility/service, request-response + transform)

Analog: CacheReason and domain error construction in src/cacheness/error_handling.py:18-152 plus current BlobStore.get() at src/cacheness/storage/blob_store.py:278-319.

Stable reason pattern (src/cacheness/error_handling.py:19-39,90-112):

    class CacheReason(str, Enum):
        PATH_RACE = "path_race"
        INVALID_IDENTIFIER = "invalid_identifier"

    def _context_with_reason(context, reason):
        error_context = dict(context or {})
        error_context["reason"] = reason.value
        return error_context

Add BlobStore-specific typed errors/reasons for integrity, unsupported version, lifecycle conflict, backend failure, and migration required while retaining compatible CacheStorageError/CacheIntegrityError bases. The ordered pipeline is bounded decode/version dispatch, authenticate, validate key/state/format/locator/size, one snapshot, SHA-256/size verification, then handler deserialization. classify_cache_read_failure(error) is a pure future UnifiedCache seam and must not import or mutate UnifiedCache.

### src/cacheness/storage/blob_store.py (service/coordinator, file I/O + CRUD)

Analog: the existing class itself, especially candidate publication and guarded reads (src/cacheness/storage/blob_store.py:128-242,278-371).

Candidate write pattern (src/cacheness/storage/blob_store.py:210-252):

    candidate_id = f"{storage_id}-candidate-{uuid.uuid4().hex}"
    candidate_locator = None
    metadata_committed = False
    try:
        result = self.guarded_handler_io.put(handler, data, candidate_id, self.config)
        candidate_locator = resolve_managed_locator(
            self.guarded_handler_io.root, result["actual_path"],
            operation="candidate_publish",
        )
        self.backend.put_entry(blob_key, entry_data)
        metadata_committed = True
    except BaseException as exc:
        if candidate_locator is not None and not metadata_committed:
            self._cleanup_uncommitted_candidate(candidate_locator, exc)
        raise

Retain this Phase 1 candidate/clear boundary, but create and persist one signed BlobManifestV1 after payload publication and SHA-256 calculation. Do not add a Cacheness payload header/container. Replace current entry -> actual_path -> handler.get with the read-contract pipeline. Preserve None only for true absence; propagate typed corruption/conflict/backend failures instead of misses or read-side deletion.

### src/cacheness/storage/__init__.py (provider/barrel, request-response)

Analog: current barrel and conditional export inventory (src/cacheness/storage/__init__.py:25-84):

    from .backends import MetadataBackend, JsonBackend, create_metadata_backend
    from .blob_store import BlobStore
    __all__ = ["BlobStore", "MetadataBackend", "JsonBackend", "create_metadata_backend", ...]

Add only supported manifest/repository/integrity/error symbols. Preserve existing imports and conditional SQLite exports. Test __all__ and direct imports; do not expose private codec helpers or promise Phase 4/5 backend parity.

### src/cacheness/error_handling.py (utility/error model, request-response)

Analog: existing hierarchy (src/cacheness/error_handling.py:43-152):

    class CacheStorageError(CacheError):
        """Raised when cache storage operations fail."""

    class CacheIntegrityError(CacheError):
        """Raised when cache integrity verification fails."""

    class CacheUnsafePathError(CacheStorageError):
        def __init__(self, message, context=None, *, reason: CacheReason):
            super().__init__(message, _context_with_reason(context, reason))

Extend with BlobStore-specific subclasses retaining existing bases and a machine-readable reason in context. Preserve causes with raise ... from exc and use narrow catches. Do not let generic decorators turn canonical failures into fallback values.

### src/cacheness/security.py (utility/provider, transform + file I/O)

Analog: CacheEntrySigner initialization/verification (src/cacheness/security.py:75-160,193-241). This is a compatibility signer, not the v1 wire contract.

    if self.key_file_path.exists():
        key = self.key_file_path.read_bytes()
        if len(key) != 32:
            logger.warning("Invalid key length ... generating new key")
            return self._generate_new_key()

Canonical required signing must not inherit auto-regeneration or in-memory fallback. Factor a strict provider or explicitly keep CacheEntrySigner legacy-only: no replacement key for an existing store, no downgrade to unsigned, no user-configurable critical-field subset, and no read-time mutation.

### src/cacheness/interfaces.py (model/interface, file I/O)

Analog: handler and guarded-result protocols (src/cacheness/interfaces.py:35-173):

    class GuardedWriteResult(TypedDict, total=False):
        storage_format: str
        file_size: int
        actual_path: str
        metadata: Dict[str, Any]

    @dataclass(frozen=True)
    class GuardedReadSnapshot:
        path: Path
        metadata: Dict[str, Any]

Extend the write result with explicit handler-owned payload_format/payload_format_version and add a non-deserializing support check if needed. Keep put(data, Path, config) and get(Path, metadata) unchanged, so NumPy, Blosc2, Parquet, pickle, and dill remain native handler owners.

### src/cacheness/handlers.py (component/serializer, file I/O + transform)

Analog: ArrayHandler (src/cacheness/handlers.py:546-738) and ObjectHandler (:882-1123).

Native write identity (src/cacheness/handlers.py:573-634):

    return {
        "storage_format": "npz",
        "file_size": npz_path.stat().st_size,
        "actual_path": str(npz_path),
        "metadata": {"shape": data.shape, "dtype": str(data.dtype), "storage_format": "npz"},
    }

Object format identity (src/cacheness/handlers.py:973-1072):

    storage_format = f"compressed_{serializer_name}"
    return {
        "storage_format": storage_format,
        "file_size": pickle_path.stat().st_size,
        "actual_path": str(pickle_path),
        "metadata": metadata,
    }

Add stable format-version identities and a support check resolving (handler_type, payload_format, payload_format_version) without opening/deserializing payload bytes. Keep native containers, ordinary NPZ allow_pickle=False, and the bounded read-only legacy Blosc2 path.

## Test Pattern Assignments

### tests/test_blob_manifest.py

Analog: tests/test_interfaces.py:34-176 and config validation/round-trip tests. Use focused pytest functions, parametrized malformed values, and pytest.raises(..., match=...). Cover frozen model construction, exact required fields, canonical byte stability/golden bytes, bounded parse rejection, unknown manifest/payload versions, and complete signed-field projection. Assert no payload or handler is touched during codec/version rejection.

### tests/test_blob_manifest_backends.py

Analog: metadata backend tests and backend matrix/reopen fixtures in tests/test_clear_recovery.py:1-190,1280-1375. Parametrize json, memory, and sqlite repositories; round-trip identical canonical bytes, reopen JSON/SQLite, and assert all fields survive exactly. Test absent key, overwrite, remove, list_keys, backend-error translation, and SQLite dedicated-table isolation from legacy cache_entries. Keep PostgreSQL/S3 out.

### tests/test_blob_store_read_contract.py

Analog: one-snapshot event-order tests in tests/test_stored_compatibility.py:533-680:

    events = []
    # snapshot/digest/authenticate/handler spies append to events
    assert events == ["snapshot", "digest", "authenticate", "handler"]
    assert events.count("snapshot") == 1

Adapt order to bounded decode/version, authentication, critical validation, one snapshot, digest/size, handler. Assert handler never runs for invalid signatures, unknown versions, non-committed state, unsafe locators, digest mismatch, or size mismatch; only true absence returns None.

### tests/test_blob_store_integrity.py

Analog: tests/test_cache_integrity.py:75-240,390-552 and filesystem containment tests. Use temporary roots, deepcopy of prior evidence, monkeypatched boundary failures, and exact exception/reason assertions. Cover SHA-256/byte-size verification, tampering, missing payload as corruption, required-key/signature failures, critical locator/format/lifecycle tampering, same-snapshot verification, and no read-side cleanup/rewrite.

### tests/test_blob_store_legacy_contract.py

Analog: immutable compatibility corpus tests in tests/test_stored_compatibility.py:166-370,480-680. Copy fixtures into tmp_path, hash source/copy before and after, use pytest.warns(DeprecationWarning), and assert legacy adapters are read-only. Verify exact legacy identities, typed migration-required/unsupported outcomes, malformed signatures, and no deletion, rewriting, key rotation, or access-time persistence.

### tests/test_blob_store_translation_seam.py

Analog: typed-error/reason assertions in tests/test_error_handling.py and strict signing cases in tests/test_cache_integrity.py:145-240. Test the classifier exhaustively: absence is not integrity error; BlobStore integrity/version/conflict/backend errors retain stable reasons; only the intended integrity category is eligible for future cache-miss translation. Assert no UnifiedCache construction/mutation and no counter/metadata changes.

## Shared Patterns

### Fail-closed errors and causes

Sources: src/cacheness/error_handling.py:43-152 and src/cacheness/storage/path_security.py:186-220.

    try:
        ...
    except OSError as exc:
        raise CacheStorageError("...", context={"operation": operation}) from exc

Use narrow operational catches, stable CacheReason values, and untouched evidence on failure. Never convert corruption/backend failure to None.

### Containment and one private snapshot

Source: src/cacheness/storage/guarded_handler_io.py:337-367 plus src/cacheness/storage/path_security.py:186-220.

Resolve/validate the authenticated locator immediately before snapshot open, reject symlink/path-race/outside-root forms, hash and deserialize the one private snapshot, and keep managed paths out of handlers.

### Native handler ownership

Sources: src/cacheness/handlers.py:573-634,705-738,973-1072 and src/cacheness/interfaces.py:35-53.

Manifest format/version fields describe handler-owned containers. Cacheness must not prepend a header or duplicate NumPy, Blosc2, PyArrow, pickle, or dill parsing.

### Exact legacy, read-only compatibility

Source: tests/test_stored_compatibility.py:166-370,533-680 and src/cacheness/metadata.py:793-937,1457-1544.

Recognize only exact historical layouts, attach explicit in-memory identity, preserve source bytes/mtimes, and make mutation fail with a typed read-only legacy error. Migration execution belongs to Phase 7.

## No Analog Found

| File | Role | Data Flow | Reason |
|---|---|---|---|
| src/cacheness/storage/manifest.py | model/utility | transform | No existing immutable canonical manifest or deterministic signed-byte codec; use config/dataclass and strict stdlib JSON patterns above. |
| src/cacheness/storage/manifest_repository.py | service/backend | CRUD + file I/O | Existing metadata backends expose shaped dictionaries, not exact raw canonical bytes; use their locking/persistence mechanics without their schema projection. |

## Metadata

**Analog search scope:** src/cacheness/storage/, src/cacheness/{error_handling,security,interfaces,handlers,metadata,config,json_utils}.py, and tests/test_{cache_integrity,stored_compatibility,clear_recovery,interfaces,metadata,error_handling}.py.  
**Files scanned:** 16 implementation/test analogs plus Phase 2 context and research.  
**Pattern extraction date:** 2026-08-30

