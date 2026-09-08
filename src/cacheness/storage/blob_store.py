"""Authority-backed object storage with JSON projection compatibility.

``BlobStore`` chooses one lifecycle authority at construction.  The lifecycle
engine is the sole payload/manifest reader and writer; the optional JSON
backend is a revision-bound projection for compatibility, never read-back
authority or a recovery path.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import logging
import secrets
import stat
import uuid
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from cacheness.config import CacheConfig, CompressionConfig
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheBlobLifecycleTimeoutError,
    CacheBlobManifestMalformedError,
    CacheBlobManifestUnauthenticatedError,
    CacheBlobManifestUnsupportedVersionError,
    CacheBlobMigrationRequiredError,
    CacheManifestIntegrityError,
    CacheManifestUnsupportedVersionError,
    CacheReason,
    CacheStorageError,
)
from .backends import JsonBackend
from .backends.blob_backends import InMemoryBlobBackend, InMemoryHandlerIO
from .composition import StoreTopology
from .coordination import InstanceAdmission, KeyCoordinatorRegistry
from .guarded_handler_io import GuardedHandlerIO
from .handlers import HandlerRegistry
from .integrity import (
    ManifestKeyError,
    ManifestKeyProvider,
    ManifestSigningKeyProvider,
    verify_hmac_sha256,
)
from .legacy_manifest import LegacyManifestIdentity, recognize_legacy_fixture_tree
from .lifecycle import AuthorityLifecycleEngine
from .lifecycle_authority import EntryExpectation
from .manifest import (
    BlobManifestV1,
    canonical_signing_bytes_from_record,
    decode_canonical_manifest_record,
)
from .manifest_repository import JsonProjectionExporter
from .memory_lifecycle_authority import InMemoryLifecycleAuthority
from .path_security import encode_physical_name
from .reconciliation import ReconciliationReport, _AuthorityReconciler
from .read_contract import BlobEntryInfo


logger = logging.getLogger(__name__)

_IMMUTABLE_METADATA_PATCH_FIELDS = frozenset(
    {
        "schema_version", "key", "cache_key", "generation", "state", "locator",
        "actual_path", "handler_type", "data_type", "payload_format",
        "payload_format_version", "storage_format", "digest_algorithm", "digest",
        "byte_size", "file_size", "created_at", "handler_metadata", "user_metadata",
        "signature_algorithm", "signature",
    }
)
_RETIRED_SCHEDULER_CONTROL_SENTINELS = (
    (".cacheness-clear-journal-v1.json", stat.S_IFREG),
    (".cacheness-inventory-v2", stat.S_IFREG),
    ("operations", stat.S_IFDIR),
)


class _EphemeralManifestKeyProvider:
    """One process-local signing key for an explicitly memory-only topology."""

    def __init__(self) -> None:
        self._key = secrets.token_bytes(32)

    def get_key(self) -> bytes:
        return self._key

    def get_or_initialize_new_store(self) -> bytes:
        return self._key

    def initialize_new_store(self) -> bytes:
        return self._key


def _retired_scheduler_control(root: Path) -> str | None:
    """Classify exact retired control evidence without opening it."""
    try:
        root_stat = root.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISDIR(root_stat.st_mode):
        return None
    for name, expected_type in _RETIRED_SCHEDULER_CONTROL_SENTINELS:
        try:
            candidate_stat = (root / name).lstat()
        except FileNotFoundError:
            continue
        if stat.S_IFMT(candidate_stat.st_mode) == expected_type:
            return name
    return None


def _ordinary_admitted(method: Callable) -> Callable:
    """Order ordinary work locally and reject immutable legacy evidence first."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        self._require_canonical_store()
        with self._instance_admission.operation():
            return method(self, *args, **kwargs)
    return wrapped


class BlobStore:
    """Store typed blobs through one authority-owned lifecycle.

    Local coordination only bounds same-process ordering and admission/close
    ownership.  Manifest transitions, snapshot reads, cleanup, and recovery
    are delegated to the selected ``LifecycleAuthority`` engine.
    """

    def __init__(
        self,
        topology: StoreTopology,
        *,
        cache_dir: Union[str, Path] = ".blobstore",
        compression: str = "lz4",
        compression_level: int = 3,
        content_addressable: bool = False,
        config: CacheConfig | None = None,
        manifest_key_provider: ManifestSigningKeyProvider | None = None,
    ) -> None:
        """Create a direct store from one explicit, validated topology."""
        if not isinstance(topology, StoreTopology):
            raise TypeError("topology must be a StoreTopology")
        configured_path = (
            config.storage.cache_dir
            if config is not None and cache_dir == ".blobstore"
            else cache_dir
        )
        self.cache_dir = Path(configured_path)
        self.topology = topology.resolve()
        self.payload_backend = self.topology.payload
        self.lifecycle_authority = self.topology.authority
        self.projections = self.topology.projections
        self.guarded_handler_io = None
        self._initialized = False
        self._guarded_handler_io_released = False
        try:
            self._initialize(
                compression,
                compression_level,
                content_addressable,
                config,
                manifest_key_provider,
            )
        except BaseException:
            self._close_failed_initialization_resources()
            raise
        logger.debug("BlobStore initialized at %s", self.cache_dir)

    def _initialize(
        self,
        compression: str,
        compression_level: int,
        content_addressable: bool,
        config: CacheConfig | None,
        manifest_key_provider: ManifestSigningKeyProvider | None,
    ) -> None:
        retired_control = _retired_scheduler_control(self.cache_dir)
        if retired_control is not None:
            raise CacheBlobMigrationRequiredError(
                "Retired scheduler control requires an explicit local-store rebuild",
                context={"control_class": "retired_scheduler_control"},
                reason=CacheReason.BLOB_MIGRATION_REQUIRED,
            )
        self._legacy_identity: LegacyManifestIdentity | None = None
        if (self.cache_dir / "provenance.json").is_file():
            self._legacy_identity = recognize_legacy_fixture_tree(self.cache_dir)
        self.compression = compression
        self.compression_level = compression_level
        self.content_addressable = content_addressable
        self.config = config or CacheConfig(
            cache_dir=self.cache_dir,
            compression=CompressionConfig(
                pickle_compression_codec=compression,
                pickle_compression_level=compression_level,
                blosc2_array_clevel=compression_level,
            ),
        )
        self.lifecycle_limits = self.config.lifecycle_limits
        self._instance_admission = InstanceAdmission(self.lifecycle_limits)
        self._key_coordinator = KeyCoordinatorRegistry()
        self._immutable_metadata_patch_fields = _IMMUTABLE_METADATA_PATCH_FIELDS
        self.handlers = HandlerRegistry()
        self._manifest_key_provider = manifest_key_provider or (
            _EphemeralManifestKeyProvider()
            if self._is_memory_topology()
            else ManifestKeyProvider(
                self.cache_dir / "blob_manifest_hmac_key.bin",
                lifecycle_limits=self.lifecycle_limits,
            )
        )
        self.capabilities = self.topology.capabilities
        self._projection_exporter = (
            JsonProjectionExporter(
                self.lifecycle_authority,
                projection.metadata_file,
                page_size=self.lifecycle_limits.manifest_page_size,
            )
            if (
                (projection := next(
                    (
                        item
                        for item in self.projections
                        if type(item) is JsonBackend
                    ),
                    None,
                ))
                is not None
            )
            else None
        )
        self.lifecycle = AuthorityLifecycleEngine(self, self.lifecycle_authority)
        # Kept as a direct engine alias for private timing seams only. It is
        # never selected conditionally and cannot represent another authority.
        self._authority_lifecycle = self.lifecycle

    def _export_compatible_projection(self) -> None:
        if self._projection_exporter is None:
            return
        try:
            self._projection_exporter.export()
        except (CacheBlobBackendError, CacheBlobLifecycleConflictError) as error:
            logger.warning("BlobStore JSON projection remains dirty after authority commit: %s", error)

    def _close_failed_initialization_resources(self) -> None:
        try:
            self.topology.close()
        except Exception:
            logger.exception("Failed to close owned topology participants")
        if self.guarded_handler_io is not None:
            try:
                self.guarded_handler_io.close()
            except Exception:
                logger.exception("Failed to close BlobStore managed-root descriptor")

    @property
    def legacy_identity(self) -> LegacyManifestIdentity | None:
        """Expose exact legacy evidence without ever using it as storage truth."""
        return self._legacy_identity

    @staticmethod
    def inspect_legacy_fixture_tree(root: str | Path) -> LegacyManifestIdentity:
        """Inspect known legacy evidence before an explicit future migration."""
        return recognize_legacy_fixture_tree(root)

    @_ordinary_admitted
    def initialize(self) -> None:
        """Initialize once before workers start; never upgrade existing schemas.

        Ordinary single-process first writes call this same path. Concurrent
        first initialization is not a supported availability guarantee.
        """
        self.lifecycle_authority.preflight_mutation()
        if self._initialized:
            return
        initializer = getattr(self.lifecycle_authority, "initialize", None)
        if callable(initializer):
            initializer()
        self._materialize_authority_store()
        empty = not self.lifecycle_authority.list_entries()
        self._authority_manifest_key(initialize_new_store=empty)
        self._initialized = True

    @contextmanager
    def open_entry(self, key: str):
        """Acquire one verified entry; policy can inspect metadata before read()."""
        self._require_canonical_store()
        with self._instance_admission.operation():
            with self.lifecycle.open_entry(key) as entry:
                yield entry

    @_ordinary_admitted
    def get_entry_info(self, key: str) -> BlobEntryInfo | None:
        """Inspect authenticated metadata without reading/deserializing payloads."""
        return self.lifecycle.get_entry_info(key)

    @_ordinary_admitted
    def put_entry(self, data: Any, key=None, metadata=None) -> BlobEntryInfo:
        """Commit an entry and own its cleanup, returning the exact receipt."""
        result = self._put_with_result_admitted(data, key=key, metadata=metadata)
        return replace(
            self.lifecycle.entry_info(result.promoted),
            previous_locator=None if result.previous is None else result.previous.locator,
        )

    @_ordinary_admitted
    def put(self, data: Any, key: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store one native handler payload through the selected authority."""
        return self._put_with_result_admitted(data, key=key, metadata=metadata).key


    def _put_with_result_admitted(
        self,
        data: Any,
        key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Store one payload after exactly one public or facade admission."""
        blob_key = (
            self._compute_content_hash(data) if self.content_addressable
            else key if key is not None else self._generate_unique_key()
        )
        with self._key_coordinator.hold(self._storage_id_for_key(blob_key)):
            result = self.lifecycle.put(
                data,
                key=blob_key,
                metadata=metadata,
            )
        self._export_compatible_projection()
        return result


    @_ordinary_admitted
    def get(self, key: str) -> Optional[Any]:
        """Read one authority snapshot, or return ``None`` when absent."""
        return self.lifecycle.get(key)

    @_ordinary_admitted
    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Read signed public metadata without deserializing the payload."""
        return self.lifecycle.get_metadata(key)

    @_ordinary_admitted
    def update_metadata(self, key: str, metadata: Dict[str, Any]) -> bool:
        """Promote an authority-owned metadata revision."""
        updated = self.lifecycle.update_metadata(key, metadata)
        if updated:
            self._export_compatible_projection()
        return updated

    @_ordinary_admitted
    def delete(
        self, key: str, *, expected: EntryExpectation | None = None
    ) -> bool:
        """Tombstone one observed generation and delete its exact payload."""
        with self._key_coordinator.hold(self._storage_id_for_key(key)):
            deleted = self.lifecycle.delete(key=key, expected=expected)
        if deleted:
            self._export_compatible_projection()
        return deleted

    @_ordinary_admitted
    def exists(self, key: str) -> bool:
        """Verify a signed payload snapshot exists through the authority."""
        return self.lifecycle.exists(key)

    @_ordinary_admitted
    def list(
        self, prefix: Optional[str] = None, metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """List committed authority entries, optionally filtering signed metadata."""
        return self.lifecycle.list(prefix, metadata_filter)

    def clear(self) -> int:
        """Clear one authority-owned membership snapshot."""
        with self._instance_admission.clear_operation() as release_snapshot:
            self._require_canonical_store()
            try:
                try:
                    token = self.lifecycle.begin_clear()
                finally:
                    release_snapshot()
                cleared = self.lifecycle.complete_clear(token)
            except (CacheBlobBackendError, CacheBlobLifecycleConflictError):
                raise
            except (CacheStorageError, OSError) as exc:
                raise CacheBlobBackendError(
                    "BlobStore clear lifecycle could not complete",
                    context={"operation": "clear"},
                ) from exc
        self._export_compatible_projection()
        return cleared

    def reconcile(
        self, *, apply: bool = False, resume_token: str | None = None, now: Any | None = None
    ) -> ReconciliationReport:
        """Inspect or settle only authority-recorded lifecycle debt."""
        with self._instance_admission.operation():
            self._require_canonical_store()
            report = _AuthorityReconciler(self, lifecycle_limits=self.lifecycle_limits).reconcile(
                apply=apply, resume_token=resume_token, now=now
            )
        if apply:
            self._export_compatible_projection()
        return report

    def close(self) -> None:
        """Drain local admission and release only resources this store owns."""
        should_release = self._instance_admission.begin_close()
        if not should_release:
            return
        closed = False
        try:
            self._release_owned_resources()
            closed = True
        except CacheStorageError:
            raise
        except CacheBlobLifecycleTimeoutError:
            raise
        except Exception as exc:
            raise CacheBlobBackendError(
                "BlobStore close could not release an owned resource",
                context={"operation": "close"},
            ) from exc
        finally:
            self._instance_admission.finish_close(closed=closed)

    def _release_owned_resources(self) -> None:
        if self.guarded_handler_io is not None and not self._guarded_handler_io_released:
            self.guarded_handler_io.close()
            self._guarded_handler_io_released = True
        self.topology.close()

    def __enter__(self):
        self._instance_admission.require_open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _materialize_authority_store(self) -> GuardedHandlerIO:
        """Open contained payload I/O after authority root validation."""
        if self.guarded_handler_io is None:
            if isinstance(self.payload_backend, InMemoryBlobBackend):
                self.guarded_handler_io = InMemoryHandlerIO(self.payload_backend)
            else:
                self.cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
                self.guarded_handler_io = GuardedHandlerIO(self.cache_dir)
        return self.guarded_handler_io

    def _authority_manifest_key(self, *, initialize_new_store: bool = False) -> bytes:
        """Read or initialize the authority trust root at its lifecycle boundary."""
        context = {
            "operation": "initialize_manifest_key" if initialize_new_store else "get_manifest_key",
            "provider": type(self._manifest_key_provider).__name__,
        }
        try:
            initialize_or_get = getattr(self._manifest_key_provider, "get_or_initialize_new_store", None)
            if initialize_new_store and callable(initialize_or_get):
                key = initialize_or_get()
            else:
                try:
                    key = self._manifest_key_provider.get_key()
                except ManifestKeyError:
                    if not initialize_new_store:
                        raise
                    initializer = getattr(self._manifest_key_provider, "initialize_new_store", None)
                    if not callable(initializer):
                        raise
                    key = initializer()
        except Exception as exc:
            raise CacheBlobManifestUnauthenticatedError(
                "Canonical BlobStore signing key is unavailable",
                context=context,
                reason=CacheReason.MANIFEST_SIGNING_KEY_INVALID,
            ) from exc
        if type(key) is not bytes or len(key) != 32:
            raise CacheBlobManifestUnauthenticatedError(
                "Canonical BlobStore signing key is invalid",
                context=context,
                reason=CacheReason.MANIFEST_SIGNING_KEY_INVALID,
            )
        return key

    def _manifest_key(self, *, initialize_new_store: bool = False) -> bytes:
        """Compatibility spelling for direct trust-root inspection callers."""
        return self._authority_manifest_key(initialize_new_store=initialize_new_store)

    def _authenticated_authority_manifest(
        self, raw: bytes, *, allow_tombstone: bool = False
    ) -> BlobManifestV1:
        """Authenticate canonical authority bytes before trusting any locator."""
        try:
            raw_record = decode_canonical_manifest_record(raw)
        except CacheManifestUnsupportedVersionError as exc:
            raise CacheBlobManifestUnsupportedVersionError(
                "Authority manifest schema version is unsupported"
            ) from exc
        except CacheManifestIntegrityError as exc:
            raise CacheBlobManifestMalformedError("Authority manifest is malformed") from exc
        if not verify_hmac_sha256(
            canonical_signing_bytes_from_record(raw_record),
            raw_record.get("signature"),
            self._authority_manifest_key(),
        ):
            raise CacheBlobManifestUnauthenticatedError("Authority manifest cannot be authenticated")
        try:
            manifest = BlobManifestV1.from_mapping(raw_record)
        except CacheManifestUnsupportedVersionError as exc:
            raise CacheBlobManifestUnsupportedVersionError(
                "Authority manifest schema version is unsupported"
            ) from exc
        except CacheManifestIntegrityError as exc:
            raise CacheBlobManifestMalformedError("Authority manifest is malformed") from exc
        if manifest.canonical_bytes() != raw:
            raise CacheBlobManifestMalformedError("Authority manifest is not canonical")
        return manifest

    def _storage_id_for_key(self, key: str) -> str:
        return encode_physical_name(key, namespace="blob-store")

    def _delete_or_prove_absent(self, locator: Path) -> None:
        """Use durable exact deletion, otherwise prove the managed path absent."""
        guarded_io = self._materialize_authority_store()
        memory_delete = getattr(guarded_io, "delete_or_prove_absent", None)
        if callable(memory_delete):
            memory_delete(locator)
            return
        cleanup_error: Exception | None = None
        try:
            if guarded_io.file_ops.delete_durable(locator):
                return
        except Exception as exc:
            cleanup_error = exc
        try:
            if not guarded_io.file_ops.exists(locator):
                return
        except Exception as exc:
            cleanup_error = exc
        context = {"operation": "payload_cleanup"}
        if cleanup_error is not None:
            context["cleanup_error"] = type(cleanup_error).__name__
        raise CacheStorageError("Could not prove managed payload cleanup", context=context)

    def _resolve_payload_handler(self, manifest: BlobManifestV1) -> Any:
        """Resolve one signed handler/payload contract before opening bytes."""
        resolver = getattr(self.handlers, "resolve_payload_contract", None)
        try:
            if callable(resolver):
                return resolver(
                    manifest.handler_type, manifest.payload_format, manifest.payload_format_version
                )
            handler = self.handlers.get_handler_by_type(manifest.handler_type)
            declared_format = manifest.handler_metadata.get("storage_format")
            if declared_format != manifest.payload_format or manifest.payload_format_version != 1:
                raise CacheManifestUnsupportedVersionError(
                    "Canonical manifest declares an unsupported native payload contract"
                )
            return handler
        except (CacheManifestUnsupportedVersionError, ValueError) as exc:
            from cacheness.error_handling import CacheBlobPayloadUnsupportedVersionError
            raise CacheBlobPayloadUnsupportedVersionError(
                "Canonical BlobStore payload contract is unsupported"
            ) from exc

    @staticmethod
    def _handler_metadata(manifest: BlobManifestV1) -> Dict[str, Any]:
        return {
            **dict(manifest.user_metadata), **dict(manifest.handler_metadata),
            "cache_key": manifest.key, "data_type": manifest.handler_type,
            "storage_format": manifest.payload_format, "file_size": manifest.byte_size,
            "created_at": manifest.created_at,
        }

    def _manifest_entry_data(self, manifest: BlobManifestV1) -> Dict[str, Any]:
        return {
            "cache_key": manifest.key,
            "data_type": manifest.handler_type,
            "file_size": manifest.byte_size,
            "created_at": manifest.created_at,
            "metadata": {
                **dict(manifest.user_metadata), **dict(manifest.handler_metadata),
                "actual_path": manifest.locator,
            },
        }

    def _compute_content_hash(self, data: Any) -> str:
        import pickle
        try:
            serialized = pickle.dumps(data)
        except Exception:
            serialized = repr(data).encode()
        return hashlib.sha256(serialized).hexdigest()[:16]

    def _require_canonical_store(self) -> None:
        if self._legacy_identity is not None:
            self._legacy_identity.require_explicit_migration()

    def _is_memory_topology(self) -> bool:
        return isinstance(self.payload_backend, InMemoryBlobBackend) and isinstance(
            self.lifecycle_authority, InMemoryLifecycleAuthority
        )

    @staticmethod
    def _generate_unique_key() -> str:
        return uuid.uuid4().hex[:16]
