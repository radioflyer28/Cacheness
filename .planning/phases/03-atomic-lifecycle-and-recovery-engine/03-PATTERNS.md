# Phase 3: Atomic Lifecycle and Recovery Engine - Pattern Map

**Mapped:** 2026-08-30  
**Files analyzed:** 22 proposed new/modified files  
**Analogs found:** 22 / 22 (role-match or stronger)

This map follows the proposed Phase 3 split in `03-RESEARCH.md`. Exact names for
the five new modules remain discretionary; the responsibilities, ordering, and
authority boundary below are the locked parts. Native handler payload bytes stay
owned by the existing handlers (`npz`, Parquet, pickle, dill, Blosc2); no
lifecycle wrapper or header should be introduced.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `src/cacheness/storage/lifecycle.py` | service/state machine | request-response + event-driven recovery | `src/cacheness/storage/blob_store.py`, `src/cacheness/storage/clear_recovery.py` | role + lifecycle match |
| `src/cacheness/storage/operation_record.py` | model/codec | transform + file I/O | `src/cacheness/storage/manifest.py`, `src/cacheness/storage/clear_recovery.py` | exact bounded-record match |
| `src/cacheness/storage/operation_repository.py` | repository/service | CRUD + file I/O | `src/cacheness/storage/manifest_repository.py`, `src/cacheness/storage/path_security.py` | role match |
| `src/cacheness/storage/reconciliation.py` | service/report model | batch + file I/O + request-response | `src/cacheness/storage/clear_recovery.py`, `tests/test_clear_recovery.py` | role/data-flow match |
| `src/cacheness/storage/coordination.py` | middleware/utility | request-response + event-driven synchronization | `src/cacheness/storage/clear_recovery.py`, `src/cacheness/storage/path_security.py` | synchronization match |
| `src/cacheness/storage/manifest_repository.py` | repository/backend adapter | CRUD + request-response | itself; `tests/test_blob_manifest_backends.py` | exact |
| `src/cacheness/storage/guarded_handler_io.py` | service/I/O adapter | file I/O + streaming | itself; `src/cacheness/storage/path_security.py` | exact |
| `src/cacheness/storage/path_security.py` | utility/I/O boundary | streaming + file I/O | itself (`ManagedFileOps`) | exact |
| `src/cacheness/storage/blob_store.py` | facade/coordinator | CRUD + streaming + event-driven recovery | itself; `tests/test_blob_store_read_contract.py` | exact public lifecycle match |
| `src/cacheness/storage/clear_recovery.py` | recovery service (compatibility predecessor) | batch + file I/O + event-driven recovery | itself; `tests/test_clear_recovery.py` | exact predecessor, to absorb/shim |
| `src/cacheness/error_handling.py` | model/error taxonomy | request-response | existing BlobStore error subclasses | exact taxonomy match |
| `src/cacheness/storage/__init__.py` | provider/barrel | request-response imports | itself | package-convention match |
| `src/cacheness/config.py` | config/model | request-response + transform | itself (`CacheStorageConfig`, `CacheMetadataConfig`, `CacheConfig`) | exact grouped-config match |
| `src/cacheness/__init__.py` | provider/barrel | request-response imports | itself (conditional exports and `__all__`) | exact package-convention match |
| `tests/test_manifest_repository_cas.py` | test | CRUD + concurrency/fault injection | `tests/test_blob_manifest_backends.py` | exact repository-test match |
| `tests/test_blob_store_atomic_lifecycle.py` | test | CRUD + streaming + crash/reopen | `tests/test_clear_recovery.py`, `tests/test_blob_store_read_contract.py` | lifecycle/fault match |
| `tests/test_blob_store_reconciliation.py` | test | batch + file I/O + report/apply | `tests/test_clear_recovery.py` | recovery-adversarial match |
| `tests/test_blob_store_concurrency.py` | test | event-driven synchronization | `tests/test_clear_recovery.py`, `tests/test_sqlite_concurrency.py` | deterministic concurrency match |
| `tests/test_blob_store_close_contract.py` | test | request-response + event-driven drain | `tests/test_blob_store_read_contract.py` | ownership/cleanup match |
| `tests/test_config_validation.py` | test | request-response + transform | itself (existing config validation suite) | exact validation-test match |
| `tests/test_blob_store_read_contract.py` | test | request-response + file I/O + event-driven fault injection | itself (existing read/ownership contract suite) | exact read-contract match |
| `tests/test_clear_recovery.py` | test | batch + file I/O + event-driven recovery | itself (existing predecessor recovery suite) | exact recovery-compatibility match |

## Pattern Assignments

### `src/cacheness/storage/lifecycle.py` (service/state machine, request-response + event-driven recovery)

**Analogs:** `src/cacheness/storage/blob_store.py` and
`src/cacheness/storage/clear_recovery.py`.

Use one engine beneath `BlobStore.put`, overwrite, delete, clear, and close.
The successful expected-generation manifest publication is the only authority
transition. Keep payload preparation, evidence, publication, reclamation, and
retirement as explicit stages so a failure can be classified as pre-authority
or post-authority.

**Facade/admission pattern** (`blob_store.py:119-175`):

```python
def _clear_coordinated(method: Callable) -> Callable:
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        coordinator = self._clear_recovery
        if coordinator is None:
            return method(self, *args, **kwargs)
        try:
            with coordinator.mutation_admission():
                return method(self, *args, **kwargs)
        except CacheStorageError as exc:
            _raise_translated_recovery_failure(coordinator, exc)
    return wrapped
```

Preserve the idea of a shared admission wrapper, but move ordinary operations
to per-key coordination and reserve the store-wide barrier for clear and
reconciliation. Typed failures must pass through unchanged when already
classified (`blob_store.py:153-175`).

**Write stage ordering** (`blob_store.py:364-466`):

```python
existing_manifest = self._load_authenticated_manifest(
    blob_key, operation="overwrite", require_locator=True
)
handler = self.handlers.get_handler(data)
candidate_id = f"{storage_id}-candidate-{uuid.uuid4().hex}"
result = self.guarded_handler_io.put(handler, data, candidate_id, self.config)
candidate_locator = resolve_managed_locator(
    self.guarded_handler_io.root, result["actual_path"], operation="candidate_publish"
)
digest, byte_size = sha256_and_size(candidate_locator)
manifest = BlobManifestV1(..., state="committed", locator=str(candidate_locator),
                          digest=digest, byte_size=byte_size, ...)
signed_manifest = manifest.with_signature(
    sign_hmac_sha256(manifest.signing_bytes(), self._manifest_key(...))
)
self.manifest_repository.put_raw(blob_key, signed_manifest.canonical_bytes(),
                                 entry_data=entry_data)
```

Refactor this order into: private native serialization; create bounded
operation evidence; exclusive streamed generation publication; digest/size
verification; signed manifest CAS; checkpoint authority publication; reclaim
only the operation's superseded locator; retire evidence. Serialization failure
must occur before operation evidence, while all managed-store effects follow
evidence creation. Do not use the old unrelated candidate UUID and separately
generated generation as ownership proof.

**Pre/post-authority recovery pattern** (`clear_recovery.py:336-446`):

```python
journal = self._new_prepared_journal(mappings)
self._validate_journal(journal)
self._create_journal(journal)
prepared_journal = deepcopy(journal)
try:
    for mapping in journal["mappings"]:
        self._stage_mapping(mapping)
    cleared_count = self._clear_backend()
except BaseException as exc:
    if isinstance(exc, Exception):
        self._rollback_prepared(journal)
    raise exc

journal["state"] = "committed"
try:
    self._replace_journal(journal)
except BaseException as exc:
    publication_state = self._publication_state_after_failure(
        prepared_journal, journal
    )
    if publication_state == "prepared":
        self._rollback_prepared(prepared_journal)
        raise
    self._poison()
    raise self._admission_error(..., lifecycle_conflict=True) from exc
self._roll_forward_committed(journal, wrap_errors=False)
```

Generalize this evidence/state classification to per-mutation operation
records. A `BaseException` represents process loss: do not speculate with
destructive cleanup; reopen a fresh store and reconcile. If authority may have
published, never restore the old manifest or revoke the winner.

**Delete/clear rules:** Use a signed tombstone CAS before payload reclamation,
then conditional tombstone retirement. Clear takes a bounded authenticated
snapshot of `(key, expected_generation)` and routes each target through the
same per-entry engine; later writes are excluded by generation CAS. Keep
ordinary distinct-key operations concurrent.

**Error pattern:** Translate narrow backend/serialization/path failures into
the domain taxonomy with `raise ... from exc`. Existing broad cleanup helpers
`_cleanup_uncommitted_candidate` and `_cleanup_prior_payload`
(`blob_store.py:911-956`) are not sufficient because they lack operation
identity, checkpoints, or resumable debt.

---

### `src/cacheness/storage/operation_record.py` (model/codec, transform + file I/O)

**Analogs:** `BlobManifestV1` in `src/cacheness/storage/manifest.py` and the
strict clear journal in `src/cacheness/storage/clear_recovery.py`.

Model operation evidence as an immutable/versioned dataclass or equivalent
record. It should contain operation ID, logical key, operation kind, expected
generation/absence, candidate/new generation, old/new locators, intended
transition, progress checkpoint, and store/topology provenance. It must not
copy payload bytes or infer ownership from handler contents or loose filenames.

**Bounded canonical codec pattern** (`manifest.py:22-33,80-185`):

```python
MANIFEST_SCHEMA_VERSION = 1
MAX_MANIFEST_BYTES = 1_048_576
MAX_NESTING_DEPTH = 16
MAX_COLLECTION_ITEMS = 4_096
MAX_TOTAL_NODES = 16_384

def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CacheManifestIntegrityError(
                f"Duplicate canonical manifest key: {key}"
            )
        result[key] = value
    return result

def _canonical_encode(record):
    _validate_canonical_value(record, depth=1, nodes=[0])
    text = json.dumps(record, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)
    encoded = text.encode("utf-8")
    if len(encoded) > MAX_MANIFEST_BYTES:
        raise CacheManifestIntegrityError(..., reason=CacheReason.MANIFEST_BOUNDS)
    return encoded
```

Copy the decode-once, duplicate-key rejection, exact-field-set, UTF-8,
collection/depth/node/encoded-byte limits from `manifest.py:188-228` and
`manifest.py:389-395`. Operation records need their own version/owner/domain
projection and bounded field limits; use the clear predecessor's public bounds
as an explicit baseline (`clear_recovery.py:34-41`):

```python
MAX_JOURNAL_BYTES = 64 * 1024 * 1024
MAX_JOURNAL_ENTRIES = 100_000
MAX_JOURNAL_FIELD_BYTES = 8192
```

Authenticate the operation projection with the existing HMAC key provider and
domain separation, or enforce equally strict owner/version/topology/locator
validation. Never log key bytes. Preserve unknown-version and malformed
evidence as typed, fail-closed findings.

**Immutable value pattern** (`manifest.py:238-339`):

```python
@dataclass(frozen=True)
class BlobManifestV1:
    schema_version: int
    key: str
    generation: str
    state: str
    locator: str
    ...

    def __post_init__(self):
        ...  # validate types, states, bounds, and metadata
        object.__setattr__(self, "handler_metadata",
                           _freeze_json_value(self.handler_metadata))
```

Use frozen records and validated transitions so checkpoints produce new
records rather than mutating an in-flight record. Keep lifecycle states aligned
with the existing manifest values (`prepared`, `committed`, `replacing`,
`tombstoned`, `conflicted`) at `manifest.py:304-313`.

---

### `src/cacheness/storage/operation_repository.py` (repository/service, CRUD + file I/O)

**Analogs:** `ManifestRepository` and its local adapters in
`src/cacheness/storage/manifest_repository.py`, plus durable file primitives in
`src/cacheness/storage/path_security.py`.

Keep the repository opaque: it stores exact operation-record bytes and
checkpoints; lifecycle code authenticates and interprets them. Use an explicit
bounded listing/page cursor rather than materializing an unbounded full-store
inventory.

**Narrow protocol/import pattern** (`manifest_repository.py:8-35,38-60`):

```python
from typing import Any, Mapping, Optional, Protocol
from cacheness.error_handling import CacheBlobBackendError, CacheError

class ManifestRepository(Protocol):
    def get_raw(self, key: str) -> Optional[bytes]: ...
    def put_raw(self, key: str, record: bytes, *,
                entry_data: Optional[Mapping[str, Any]] = None) -> None: ...
    def remove(self, key: str) -> None: ...
    def list_keys(self) -> list[str]: ...
    def list_backend_entries(self) -> list[dict[str, Any]]: ...
```

Mirror this style for operation records and expose create/get/checkpoint/
retire/list-page primitives with exact bytes. Translate only narrow operational
errors and preserve causes (`manifest_repository.py:63-73`):

```python
def _backend_failure(operation, backend, exc):
    return CacheBlobBackendError(
        f"Canonical manifest repository {operation} failed",
        context={"operation": operation, "backend": type(backend).__name__},
    )
```

**Local locking/transaction patterns:**

- In-memory/JSON adapters hold their backend lock around read/update and use
  copies for persisted projections (`manifest_repository.py:76-125`;
  `metadata.py:1152-1182`). Add expected-record/generation compare under the
  same lock, not a separate check then put.
- SQLite keeps compatibility projection and canonical bytes in one transaction
  (`manifest_repository.py:236-307`), under the backend lock. Add CAS to the
  same transaction and roll back both projections on any write failure.
- JSON cross-instance operations must refresh from disk before a short
  compare/publication critical section. Do not hold the lock across handler
  serialization or payload copying.

Use `ManagedFileOps.create_bytes_durable_exclusive` and
`write_bytes_durable` for operation evidence; they perform no-follow contained
validation, fsync, and directory acknowledgement
(`path_security.py:615-665`). Missing evidence is a completed idempotent
retirement step only when operation ownership/generation is still proven.

---

### `src/cacheness/storage/reconciliation.py` (service/report model, batch + file I/O + request-response)

**Analogs:** `ClearRecoveryCoordinator` and adversarial tests in
`tests/test_clear_recovery.py`.

Separate a non-mutating dry-run analyzer from apply/resume. Dry-run may read
bounded raw manifest/operation pages and backend-owned inventory, authenticate
records, and hash only what is needed. It must not invoke handlers, update hit
counters, create keys, quarantine, delete, or change mtimes. Report records
must include authoritative generation, operation provenance, residue type,
proposed action, reason, and status (`safe`, `blocked`, or
`requires_confirmation`).

**Strict evidence parsing pattern** (`clear_recovery.py:543-562,564-642`):

```python
def _read_journal(self):
    journal_size = self.file_ops.get_size(self.journal_path)
    if journal_size < 0 or journal_size > MAX_JOURNAL_BYTES:
        self._invalid_journal()
    raw_journal = self.file_ops.read_bytes(self.journal_path)
    if len(raw_journal) > MAX_JOURNAL_BYTES:
        self._invalid_journal()
    document = json_loads(raw_journal)
    if not isinstance(document, dict):
        self._invalid_journal()
    return document

def _validate_journal(self, journal, ...):
    if not isinstance(journal, dict) or set(journal) != _JOURNAL_FIELDS:
        self._invalid_journal()
    ...  # exact version/owner/state/topology/count checks
```

Apply must re-authenticate/revalidate immediately before each destructive
action. If authority changed since dry-run, convert the finding to blocked or
conflict. Checkpoint every completed action in durable operation evidence;
reopen and rerun must not repeat successful deletion or lose the valid
generation. Reuse `_publication_state_after_failure`
(`clear_recovery.py:428-446`) to avoid rollback after an uncertain authority
publication.

**Adversarial test pattern:** The clear tests build current-format journals,
mutate one field, write untrusted evidence, and assert rejection before any
callback (`tests/test_clear_recovery.py:944-975`). Oversize checks precede JSON
deserialization and count parser calls (`tests/test_clear_recovery.py:1538-1582`).
Use these for malformed, duplicate, unsupported-version, wrong-store/key,
locator escape, symlink, oversized, and provenance-free evidence. Quarantine
only via safe backend-native move; otherwise report untouched.

---

### `src/cacheness/storage/coordination.py` (middleware/utility, request-response + event-driven synchronization)

**Analogs:** clear admission in `clear_recovery.py:69-245` and root-safe
operations in `path_security.py:220-314`.

Implement a refcounted dynamically retained `key -> {lock, users}` registry.
Increment users while holding the registry guard before waiting for a key lock;
decrement under the guard and remove the exact entry at zero. This bounds
retention without serializing unrelated keys. Sort stable key representations
before multi-key acquisition.

**Existing process/root admission pattern** (`clear_recovery.py:149-195`):

```python
with _PROCESS_LOCKS_GUARD:
    if _PROCESS_LOCK_OWNERS.get(identity) == owner_thread:
        raise self._admission_error(..., lifecycle_conflict=True)
if not process_lock.acquire(blocking=blocking):
    raise self._admission_error(..., lifecycle_conflict=True)
try:
    with _PROCESS_LOCKS_GUARD:
        _PROCESS_LOCK_OWNERS[identity] = owner_thread
    lock_descriptor = self._acquire_advisory_lock(blocking=blocking)
    yield
finally:
    ...
    with _PROCESS_LOCKS_GUARD:
        _PROCESS_LOCK_OWNERS.pop(identity, None)
    process_lock.release()
```

Preserve cleanup in `finally`, deterministic same-root conflict detection, and
the store-wide barrier only for clear/reconciliation. A local lock is an
ordering optimization, never a substitute for backend generation CAS. Close
admission should use a condition/in-flight counter: reject new operations once
closing starts, drain admitted operations, then release only owned resources.

**Deterministic race seam:** Existing managed I/O exposes a test callback after
first validation and before final guarded action (`path_security.py:220-236,
285-314`). Use `threading.Event`/`Barrier` hooks at exact CAS or publication
boundaries. Never use `sleep()` to establish ordering.

---

### `src/cacheness/storage/manifest_repository.py` (repository/backend adapter, CRUD + request-response)

**Analog:** itself (`manifest_repository.py:38-60,76-164,180-348`) and
`tests/test_blob_manifest_backends.py`.

Retain exact raw canonical bytes and the current local backend identity checks.
Extend the protocol with an atomic `publish_if_expected`/conditional tombstone
retirement operation carrying an opaque expected generation plus exact-record
digest/revision. It must compare and publish in one backend-local critical
section/transaction and return the typed lifecycle conflict without overwriting
the winner.

**Raw adapter pattern** (`manifest_repository.py:82-125`):

```python
entry = self.backend.get_entry(key)
if entry is None:
    return None
metadata = entry.get("metadata") if isinstance(entry, Mapping) else None
if not isinstance(metadata, Mapping) or _RAW_MANIFEST_FIELD not in metadata:
    raise CacheBlobMigrationRequiredError(...)
encoded = metadata[_RAW_MANIFEST_FIELD]
try:
    return base64.b64decode(encoded, validate=True)
except (ValueError, UnicodeEncodeError) as exc:
    raise _backend_failure("get_raw", self.backend, exc) from exc
```

Do not decode an unauthenticated current manifest in the repository merely to
discover the expected generation; lifecycle code authenticates first.

**SQLite atomic projection pattern** (`manifest_repository.py:243-307`):

```python
with self.backend._lock, self.backend.engine.begin() as connection:
    if entry_data is not None:
        self._write_compatibility_projection(connection, key, entry_data)
    self._write_raw_row(connection, key, record)
```

Insert compare predicates into this same transaction; preserve rollback of both
`cache_entries` and `cacheness_manifest_records_v1`. For JSON, refresh the
document before compare and atomically replace a same-directory candidate; for
memory, compare/replace under the backend lock. Keep Phase 4 remote capability
claims out of this phase.

**Tests:** Copy fixture/parametrization and cause assertions from
`tests/test_blob_manifest_backends.py:52-89,172-195,229-270`. Add create-if-
absent, replace-if-generation, conditional tombstone retirement, independent
instances, and SQLite rollback cases.

---

### `src/cacheness/storage/guarded_handler_io.py` (service/I/O adapter, file I/O + streaming)

**Analog:** itself (`guarded_handler_io.py:95-121,301-367`) with
`ManagedFileOps` as the lower-level analog.

Split `put` into private serialization/staged-artifact validation and a caller-
provided generation-specific exclusive publication method. Retain the validated
descriptor identity until publication, preserving the existing path-race
defense.

**Private stage and safe artifact pattern** (`guarded_handler_io.py:110-121,
134-211`):

```python
with tempfile.TemporaryDirectory(prefix="cacheness-handler-") as temporary:
    stage_root = Path(temporary)
    stage_root.chmod(0o700)
    ...
raw_result = handler.put(data, stage_base, config)
artifact = self._staged_artifact(stage_root, stage_base, raw_result)
```

Retain absolute/relative stage containment, reject `..`, symlinks, non-regular
files, multiple links, and identity changes (`guarded_handler_io.py:139-211`).
Candidate publication must stream without loading the payload into memory.

**Snapshot pattern** (`guarded_handler_io.py:336-367`):

```python
managed_locator = resolve_managed_locator(self.root, locator, operation="snapshot")
with self._private_stage() as stage_root:
    snapshot_path = stage_root / f"snapshot{suffix}"
    with snapshot_path.open("xb") as destination:
        snapshot_path.chmod(0o600)
        self.file_ops.copy_to_stream(managed_locator, destination)
        destination.flush()
        os.fsync(destination.fileno())
    snapshot_metadata = dict(metadata)
    snapshot_metadata["actual_path"] = str(snapshot_path)
    yield GuardedReadSnapshot(snapshot_path, snapshot_metadata)
```

Readers must receive only this one private snapshot and finish digest,
signature, and handler deserialization before context exit. Add one bounded
reacquisition at the BlobStore/lifecycle layer only when a second authenticated
manifest proves generation movement.

---

### `src/cacheness/storage/path_security.py` (utility/I/O boundary, streaming + file I/O)

**Analog:** `ManagedFileOps` itself (`path_security.py:220-314,421-476,
539-586,615-750`).

Add an exclusive streamed write primitive parallel to
`create_bytes_durable_exclusive`, preserving containment, no-follow checks,
atomic temp-to-final publication, fsync, and directory acknowledgement. Do not
make lifecycle code call `Path.unlink`, `open`, or `os.rename` directly.

**Contained operation preparation** (`path_security.py:285-314`):

```python
def _prepare_locator(self, locator, *, operation, allow_missing_leaf=False):
    self._assert_root_identity()
    resolved = resolve_managed_locator(
        self.root, locator, operation=operation,
        allow_missing_leaf=allow_missing_leaf,
    )
    self._run_hook(operation, resolved)
    self._assert_root_identity()
    if not self._descriptor_mode:
        resolved = resolve_managed_locator(...)
    return resolved
```

The exclusive stream variant should use `O_EXCL|O_NOFOLLOW` in descriptor mode
and the fallback's same-directory exclusive/temp strategy. Reuse the existing
short-write loop, descriptor cleanup, and directory fsync behavior
(`path_security.py:437-476,615-665`). Missing deletes remain idempotent only
after safe locator validation (`path_security.py:714-750`); ownership/generation
checks belong to lifecycle code.

---

### `src/cacheness/storage/blob_store.py` (facade/coordinator, CRUD + streaming + event-driven recovery)

**Analog:** itself and `tests/test_blob_store_read_contract.py`.

Keep public `BlobStore` API compatibility, but route direct mutations through
the lifecycle engine. Preserve content-addressable key calculation,
`HandlerRegistry`, `CacheConfig`, manifest signing, and compatibility metadata
projection. Remove direct unconditional `put_raw`/`remove` from multi-step
mutations in favor of CAS/tombstone/reclamation calls.

**Constructor ownership pattern** (`blob_store.py:217-318`):

```python
self.guarded_handler_io = GuardedHandlerIO(self.cache_dir)
self._owns_backend = False
try:
    self._initialize_after_guarded_io(...)
except BaseException:
    self._close_failed_initialization_resources()
    raise
```

Retain this constructor-failure cleanup, then add lifecycle admission state.
`close()` must reject new operations, drain or deterministically cancel in-flight
work, flush owned evidence, close `GuardedHandlerIO`, and close the backend only
when `_owns_backend` is true. It must be idempotent and must never call
`clear()`.

**Read ordering pattern** (`blob_store.py:472-522,1006-1076`):

```python
authenticated = self._load_authenticated_manifest(
    key, operation="get", require_payload_contract=True, require_locator=True
)
if authenticated is None:
    return None
manifest, handler, actual_path = authenticated
with self.guarded_handler_io.open_snapshot(actual_path, handler_metadata) as snapshot:
    digest, byte_size = sha256_and_size(snapshot.path)
    if digest != manifest.digest or byte_size != manifest.byte_size:
        raise CacheBlobPayloadTamperedError(...)
    data = handler.get(snapshot.path, snapshot.metadata)
```

Preserve committed-only authentication, digest/size-before-handler ordering,
and typed failures. Add a second authenticated manifest read and retry exactly
once only if acquisition observed a changed generation; never retry malformed,
unauthenticated, unsupported, same-generation missing, or tampered payloads.

**Existing typed translation:** `_load_authenticated_manifest`
(`blob_store.py:1006-1076`) decodes, maps unsupported/malformed records to
public subclasses, verifies HMAC, validates key/state, then resolves locators.
Use it as the shared authority loader for lifecycle and reconciliation. Do not
deserialize payloads to decide ownership.

---

### `src/cacheness/storage/clear_recovery.py` (recovery service, compatibility predecessor, batch + file I/O)

**Analog:** itself and `tests/test_clear_recovery.py`.

Treat this as a pattern source or compatibility shim, not a second nested
journal beneath the new lifecycle engine. Preserve existing reopen behavior for
old clear evidence where compatibility requires it, but make new lifecycle
operations use the common operation-record/reconciliation model.

**Admission and topology binding** (`clear_recovery.py:86-147,245-315`):

```python
if not _advisory_lock_available(self.file_ops.root):
    raise self._admission_error("Reliable advisory locking is unavailable")
...
topology = {
    "root": str(self.file_ops.root),
    "root_device": root_stat.st_dev,
    "root_inode": root_stat.st_ino,
    "backend": self.kind,
}
```

Keep fail-before-mutation topology refusal, root identity, exact backend
identity, and typed conflict/backend contexts. Replace clear-only global
mutation serialization with the Phase 3 per-key registry plus aggregate
barrier.

**Durable/reopen pattern** (`clear_recovery.py:789-841`):

```python
original = self._journal_locator(mapping["original"])
tombstone = self._journal_locator(mapping["tombstone"])
with self.file_ops.open_read(original) as source:
    self.file_ops.write_stream_to_locator(tombstone, source)
self.file_ops._fsync_containing_directory(tombstone)
if not self.file_ops.delete_durable(original):
    raise FileNotFoundError(...)
```

Use the same idempotent missing-as-complete rule only when validated
provenance/generation still matches. Never restore a whole backend snapshot
after uncertain authority publication; the old `_snapshot_backend`/
`_restore_backend` approach is clear-only and all-at-once.

---

### `src/cacheness/error_handling.py` (model/error taxonomy, request-response)

**Analog:** existing BlobStore taxonomy (`error_handling.py:19-54,155-299`).

Add stable lower-snake-case reasons/types for recoverable cleanup, unsupported
conditional publication, operation-record integrity/version failures, and
reconciliation conflicts only if required by the new public boundary. Preserve
existing reasons (`blob_lifecycle_conflict`, `blob_backend_failure`,
`blob_migration_required`) and subclass relationships.

**Typed reason/context pattern** (`error_handling.py:107-126,263-299`):

```python
def _context_with_reason(context, reason):
    error_context = dict(context or {})
    error_context["reason"] = reason.value
    return error_context

class CacheBlobLifecycleConflictError(CacheStorageError):
    def __init__(self, message, context=None,
                 *, reason=CacheReason.BLOB_LIFECYCLE_CONFLICT):
        super().__init__(message, _context_with_reason(context, reason))
```

Every translated operational exception preserves its cause with
`raise ... from exc`. Avoid broad `except Exception` at new authority
boundaries; make cleanup-debt and partial-success policy explicit in context.

---

### `src/cacheness/storage/__init__.py` (provider/barrel, request-response)

**Analog:** existing storage barrel (`storage/__init__.py:31-73,84-119`).

Re-export only stable public lifecycle types, reports, and errors after their
modules are complete. Keep direct internal imports package-relative and guard
optional dependencies. Follow the existing deliberate `__all__` pattern and
conditional SQLite export; do not export implementation-only record codecs or
coordination internals unless the public contract requires them.

---

### `tests/test_manifest_repository_cas.py` (test, CRUD + concurrency/fault injection)

**Analog:** `tests/test_blob_manifest_backends.py`.

Copy its parametrized local repository fixture and exact-byte assertions
(`test_blob_manifest_backends.py:52-89`) and typed-cause style
(`test_blob_manifest_backends.py:172-195`). Extend with:

- create-if-absent and replace-if-exact-generation success;
- stale generation/record conflicts for independent repository instances;
- conditional tombstone retirement that cannot remove a newer manifest;
- SQLite transaction rollback proving compatibility projection and raw row both
  remain old on failure;
- JSON cross-instance refresh/locking and in-memory lock atomicity.

Use `pytest.raises(CacheBlobLifecycleConflictError)` and assert the winner's
raw bytes remain readable; never assert only that an exception occurred.

---

### `tests/test_blob_store_atomic_lifecycle.py` (test, CRUD + streaming + crash/reopen)

**Analogs:** `tests/test_clear_recovery.py` and
`tests/test_blob_store_read_contract.py`.

Copy the `_SimulatedClearInterruption(BaseException)` marker and reopen pattern
(`test_clear_recovery.py:27-31,234-326`) for process-loss-style boundaries.
For each write/overwrite/delete/clear boundary inject ordinary `Exception` and
`BaseException` where meaningful, then assert old-complete or new-complete
visibility, exact residue ownership, and typed recoverable cleanup.

**Read tracer pattern** (`test_blob_store_read_contract.py:507-561`):

```python
events = []
def get_raw_with_event(blob_key):
    events.append("repository")
    return original_get_raw(blob_key)
@contextmanager
def snapshot_with_event(locator, metadata):
    events.append("snapshot")
    with original_snapshot(locator, metadata) as snapshot:
        yield snapshot
...
assert events == ["repository", "snapshot", "digest", "handler"]
```

Adapt it to assert one bounded reacquisition only on proven generation change,
and no retry for same-generation integrity failures. Keep exact manifest/payload
bytes and mtimes where stable across reopen.

---

### `tests/test_blob_store_reconciliation.py` (test, batch + file I/O + report/apply)

**Analogs:** adversarial journal tests in `tests/test_clear_recovery.py`.

Build bounded owned operation records/manifests through helpers analogous to
`_journal_for` and `_write_untrusted_journal`
(`test_clear_recovery.py:952-974`). For dry-run, compare complete pre/post
state: raw manifest/evidence bytes, payload bytes or hashes, mtimes, directory
membership, signing-key bytes, backend counters, call counts, and mutation
spies. Assert deterministic machine report plus human summary.

For apply, interrupt after every checkpoint, reopen, resume, and assert no
repeated destructive calls. Mutate authority between dry-run and apply and
assert blocked/conflict. Cover malformed/oversized/duplicate/unknown-version,
wrong-store/key, locator escape, symlink, provenance-free, and unsupported
inventory evidence. Ambiguous residue remains untouched unless safe quarantine
is proven by backend-native containment.

---

### `tests/test_blob_store_concurrency.py` (test, event-driven synchronization)

**Analogs:** event-controlled admission tests in
`tests/test_clear_recovery.py:565-602,641-679,1872-1927` and existing SQLite
thread tests in `tests/test_sqlite_concurrency.py:84-118`.

Use exact seam events, not timing sleeps:

```python
entered = threading.Event()
release = threading.Event()

def pause_before_cas(*_args):
    entered.set()
    assert release.wait(timeout=5)
```

Cover same-key write/write (one CAS winner), write/delete, read/write and
read/delete (complete snapshot or one typed outcome), distinct-key overlap
while key A is paused, clear/post-snapshot writes, and registry retirement.
Use bounded `join(timeout=5)` and collect thread errors explicitly. Validate
independent instances, not only one object, so local locks cannot hide missing
backend CAS.

---

### `tests/test_blob_store_close_contract.py` (test, request-response + event-driven drain)

**Analog:** constructor/resource ownership tests in
`tests/test_blob_store_read_contract.py:90-288`.

Copy the close spies and cancellation identity assertions:

```python
def close_backend_spy(backend):
    closed_backends.append(backend)
    original_backend_close(backend)
...
assert closed_backends.count(initialized_backend) == 1
```

Add an admitted operation paused by an event, begin `close()`, assert a second
operation is rejected, then release and assert deterministic drain. Repeated
close must be harmless and exactly-once. Internally created JSON/SQLite
backends and `GuardedHandlerIO` are closed once; caller-injected backends are
never closed. Assert close does not clear stored user data and no operation uses
resources after closure.

---

### `src/cacheness/config.py` (config/model, request-response + transform)

**Analog:** existing grouped configuration classes in
`src/cacheness/config.py:9-14,33-112,351-425`.

Keep lifecycle limits as a configuration-owned public value object, following
the module's dataclass and validation conventions rather than defining a second
policy type in `storage/`. The import and grouped-default pattern is:

```python
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Union
from pathlib import Path

@dataclass
class CacheStorageConfig:
    cache_dir: str = "./cache"
    max_cache_size_mb: Optional[int] = 2000

@dataclass
class CacheMetadataConfig:
    metadata_backend: str = "auto"
    default_ttl_hours: float = 24

class CacheConfig:
    storage: CacheStorageConfig = field(default_factory=CacheStorageConfig)
    metadata: CacheMetadataConfig = field(default_factory=CacheMetadataConfig)
```

Add the frozen `LifecycleLimits` dataclass beside these configuration models,
with the seven exact defaults locked by Plan 03-03:
`max_operation_record_bytes=1_048_576`, `max_operation_field_bytes=8_192`,
`manifest_page_size=256`, `operation_page_size=256`,
`max_reconcile_actions=10_000`, `orphan_grace_seconds=300.0`, and
`close_wait_seconds=30.0`. Preserve caller-supplied values; reject booleans,
non-finite values, zero, and negative values rather than silently clamping. The
existing local validation style (`config.py:44-47,80-102`) raises a precise
`ValueError` from `__post_init__`; use that for constructor-time limits
validation. Compose one `lifecycle_limits` field into `CacheConfig` with a
default factory and preserve a supplied instance by identity using `is None`,
not truthiness or reconstruction. Keep the type import stable for all lifecycle
modules.

---

### `src/cacheness/__init__.py` (provider/barrel, request-response imports)

**Analog:** the existing conditional public-export barrel in
`src/cacheness/__init__.py:32-74,205-241`.

Import the configuration-owned lifecycle type directly from `.config` with the
other config exports, and add it to the stable `__all__` list. Do not re-export
it through `storage/__init__.py`; the plan deliberately gives configuration
ownership to the top-level API. Preserve optional import guards and aliases:

```python
from .config import (
    CacheBlobConfig,
    CacheMetadataConfig,
    CacheStorageConfig,
    CompressionConfig,
    SerializationConfig,
    HandlerConfig,
    SecurityConfig,
    ConfigValidationError,
    validate_config,
    validate_config_strict,
)

__all__ = [
    "cacheness",
    "CacheConfig",
    "CacheBlobConfig",
    "CacheMetadataConfig",
    "CacheStorageConfig",
    # ... existing public exports ...
]
```

Add `LifecycleLimits` alongside these config names, retaining the existing
convenience import surface and avoiding a broad import of new storage modules
that could introduce a cycle.

---

### `tests/test_config_validation.py` (test, request-response + transform)

**Analog:** the existing configuration validation suite itself, especially
`tests/test_config_validation.py:7-49,56-85,171-295`.

Use its direct `cacheness.config` imports, temporary-directory fixture, valid
config fixture, banner grouping, and exact `pytest.raises(..., match=...)`
assertions:

```python
from cacheness.config import (
    CacheConfig,
    CacheStorageConfig,
    CacheMetadataConfig,
    ConfigValidationError,
    validate_config,
    validate_config_strict,
)

@pytest.fixture
def valid_config():
    """Provide a valid configuration instance."""
    return CacheConfig()

def test_invalid_stream_threshold(self):
    with pytest.raises(ValueError, match="must be non-negative"):
        CacheBlobConfig(stream_threshold_bytes=-1)
```

Extend the suite with a focused `LifecycleLimits` block: assert all seven
defaults exactly, assert every limit accepts an explicit caller value, and
parameterize each invalid edge (non-finite, non-positive, or field-specific
invalid value) with the stable field name in the message. Also test that the
top-level `cacheness.LifecycleLimits` identity is the config class, preventing
duplicate ownership or a storage-only type.

---

### `tests/test_blob_store_read_contract.py` (test, request-response + file I/O + event-driven fault injection)

**Analog:** the existing direct-read and constructor/ownership contract suite
in `tests/test_blob_store_read_contract.py:507-583,586-634,838-946` and
`tests/test_blob_store_read_contract.py:90-288`.

Preserve the event tracer that proves the read ordering (repository, one
snapshot, digest, then handler) and the rule that absence returns `None`
without payload I/O:

```python
events: list[str] = []
...
def get_raw_with_event(blob_key: str):
    events.append("repository")
    return original_get_raw(blob_key)

@contextmanager
def snapshot_with_event(locator, metadata):
    events.append("snapshot")
    with original_snapshot(locator, metadata) as snapshot:
        yield snapshot

assert store.get(key) == "tracer payload"
assert events == ["repository", "snapshot", "digest", "handler"]
```

Use its parametrized typed-failure matrix (`:838-946`) to ensure CAS
retirement/patch errors remain typed and preserve the original cause; use the
constructor tests to cover Plan 03-09's ownership regressions. Add the revised
Plan 03-02/03-08 cases here without replacing these existing guarantees:
one retryable read acquisition after a concurrent generation change, no second
payload snapshot, and read/close behavior that leaves injected resources owned
by the caller.

---

### `tests/test_clear_recovery.py` (test, batch + file I/O + event-driven recovery)

**Analog:** the predecessor recovery and adversarial-journal suite in
`tests/test_clear_recovery.py:29-65,234-326,945-974,1538-1590`.

Keep the explicit process-loss exception, managed payload-byte capture, and
reopen helper used to make recovery deterministic:

```python
class _SimulatedClearInterruption(BaseException):
    """Represent loss of control between durable clear phases."""

def _put_json_payloads(store: BlobStore) -> tuple[list[str], dict[Path, bytes]]:
    keys = [store.put("first", key="first"), store.put("second", key="second")]
    payload_bytes = {}
    for key in keys:
        entry = store.get_metadata(key)
        assert entry is not None
        payload_path = Path(entry["metadata"]["actual_path"])
        payload_bytes[payload_path] = payload_path.read_bytes()
    return keys, payload_bytes

def _reopen_json_store(root: Path) -> BlobStore:
    return BlobStore(root, backend="json")
```

Extend the legacy-compatibility tests to prove exact Phase 1 clear journals
remain readable, while malformed, future-schema, topology-mismatched, or
provenance-free evidence is preserved byte-for-byte and causes zero mutation.
Retain the raw-bound-before-JSON pattern (`:1545-1580`) and the attacker-journal
helper (`:952-974`); these are the closest concrete tests for bounded parsing,
authenticated evidence, and “report/no destructive guess” recovery. Add the
Plan 03-06 assertion that ordinary mutations no longer use the predecessor's
global clear path.

## Shared Patterns

### Authority and CAS

**Sources:** `src/cacheness/storage/manifest.py:238-395`,
`src/cacheness/storage/manifest_repository.py:38-60,236-307`,
`src/cacheness/storage/blob_store.py:1006-1076`.

**Apply to:** `src/cacheness/storage/manifest_repository.py`,
`src/cacheness/storage/blob_store.py`, `tests/test_manifest_repository_cas.py`,
`tests/test_blob_store_read_contract.py`, and all lifecycle mutation tests.

Authenticate the current committed manifest before deriving the expectation;
compare expected generation plus exact-record digest in one repository-local
publication. Treat successful CAS as the only authority transition. A failed
comparison is a typed conflict and never overwrites or revokes the winner.

### Bounded Authenticated Evidence

**Sources:** `src/cacheness/storage/manifest.py:80-228`,
`src/cacheness/storage/clear_recovery.py:34-52,543-642`.

**Apply to:** `src/cacheness/config.py`,
`tests/test_config_validation.py`, `src/cacheness/storage/operation_record.py`,
`src/cacheness/storage/operation_repository.py`,
`src/cacheness/storage/reconciliation.py`, and
`tests/test_clear_recovery.py`.

Reject oversized bytes before parsing, duplicate/unknown fields, unsupported
versions, invalid owner/topology, bad locators, and out-of-bound collections.
Operation records carry provenance and progress but never payload contents.
Malformed or unauthenticated evidence is reported/blocked, not guessed around.

### Contained Immutable I/O

**Sources:** `src/cacheness/storage/guarded_handler_io.py:134-211,336-367`,
`src/cacheness/storage/path_security.py:285-314,437-476,615-750`.

**Apply to:** `src/cacheness/storage/blob_store.py`,
`tests/test_blob_store_read_contract.py`, and `tests/test_clear_recovery.py`.

All managed reads/writes/deletes/quarantines go through no-follow,
root-identity-checked `ManagedFileOps`; use exclusive streamed generation
publication and one private read snapshot. Preserve descriptor cleanup and
directory fsyncs. Never deserialize payloads during reconciliation.

### Typed Failures and Cause Preservation

**Sources:** `src/cacheness/error_handling.py:107-126,155-299`,
`src/cacheness/storage/manifest_repository.py:63-73`,
`tests/test_blob_manifest_backends.py:172-195`.

**Apply to:** `src/cacheness/storage/lifecycle.py`,
`src/cacheness/storage/operation_repository.py`,
`src/cacheness/storage/reconciliation.py`, `src/cacheness/storage/blob_store.py`,
`tests/test_blob_store_read_contract.py`, and `tests/test_clear_recovery.py`.

Use domain subclasses with stable `context["reason"]`, preserve
`__cause__`, and distinguish conflict, backend failure, integrity/version,
unsupported capability, and recoverable cleanup. Existing typed errors are
re-raised unchanged when already classified.

### Deterministic Concurrency and Ownership

**Sources:** `src/cacheness/storage/clear_recovery.py:149-195`,
`src/cacheness/storage/path_security.py:220-236`,
`tests/test_clear_recovery.py:565-602`,
`tests/test_blob_store_read_contract.py:150-288`.

**Apply to:** `src/cacheness/storage/coordination.py`,
`src/cacheness/storage/blob_store.py`, `tests/test_blob_store_concurrency.py`,
`tests/test_blob_store_close_contract.py`,
`tests/test_blob_store_read_contract.py`, and `tests/test_clear_recovery.py`.

### Configuration ownership and public API stability

**Sources:** `src/cacheness/config.py:9-14,33-112,351-425,773-815`,
`src/cacheness/__init__.py:32-67,221-241`,
`tests/test_config_validation.py:39-49,171-295`.

**Apply to:** `src/cacheness/config.py`, `src/cacheness/__init__.py`,
`tests/test_config_validation.py`, and every lifecycle module importing limits.

Configuration models use focused dataclasses and constructor-time validation;
the top-level package imports those symbols directly and enumerates its stable
public API in `__all__`. Keep `LifecycleLimits` defined once in config,
re-exported from the top-level barrel, and verified by identity/default/edge
tests. Do not create a second storage-owned limits class or add a storage barrel
re-export.

### Read and recovery regression harnesses

**Sources:** `tests/test_blob_store_read_contract.py:507-583,838-946`,
`tests/test_clear_recovery.py:29-65,234-326,945-974,1538-1590`.

**Apply to:** `tests/test_blob_store_read_contract.py` and
`tests/test_clear_recovery.py`, with related lifecycle and reconciliation tests.

Use event-traced ordering and typed-failure parameterization for direct reads;
use reopen-after-interruption, byte snapshots, raw bound checks, and untrusted
journal fixtures for recovery. Tests should assert state and bytes are
unchanged on malformed or untrusted evidence, use bounded waits for concurrency,
and distinguish `BaseException` process-loss simulation from ordinary errors.

Use refcounted per-key entries, a short aggregate barrier only for clear/
reconciliation, event/barrier seam hooks with bounded waits, and explicit
instance admission/drain. Local locks do not replace backend CAS. Track
resource ownership from construction and release exactly once.

## No Analog Found

No file is without a usable role/data-flow analog. The five proposed modules
`lifecycle.py`, `operation_record.py`, `operation_repository.py`,
`reconciliation.py`, and `coordination.py` are new seams, so their analogs are
compositions of the existing BlobStore, manifest, clear-recovery, and guarded
I/O implementations rather than copy-ready modules. Their exact public names
are delegated by CONTEXT; planners should preserve the responsibilities and
ordering above.

## Metadata

**Analog search scope:** `src/cacheness/storage/`, `src/cacheness/error_handling.py`,
`src/cacheness/metadata.py`, `tests/test_blob_manifest*.py`,
`tests/test_blob_store_read_contract.py`, `tests/test_clear_recovery.py`,
`tests/test_filesystem_containment.py`, `tests/test_sqlite_concurrency.py`,
and registry/concurrency tests.  
**Files scanned:** 16 implementation/test analog files plus required phase
artifacts.  
**Pattern extraction date:** 2026-08-30
