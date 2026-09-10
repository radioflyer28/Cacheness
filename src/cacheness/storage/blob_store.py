"""Authority-backed object storage with JSON projection compatibility.

``BlobStore`` chooses one lifecycle authority at construction.  The lifecycle
engine is the sole payload/manifest reader and writer; the optional JSON
backend is a revision-bound projection for compatibility, never read-back
authority or a recovery path.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import logging
import secrets
import stat
import uuid
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from cacheness.config import CacheConfig, CacheStorageConfig, CompressionConfig
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
from .catalog import (
    CatalogCursor,
    CatalogPage,
    CatalogQuery,
    CatalogSchema,
    DEFAULT_PAGE_SIZE,
    validate_catalog_mapping,
    validate_catalog_page_request,
)
from .composition import BackendRole, ParticipantCapabilities, StoreTopology
from .coordination import InstanceAdmission
from .handlers import HandlerRegistry
from .integrity import (
    ManifestKeyError,
    ManifestKeyProvider,
    ManifestSigningKeyProvider,
)
from .legacy_manifest import LegacyManifestIdentity, recognize_legacy_fixture_tree
from .lifecycle import (
    AuthorityLifecycleEngine,
    _DEFAULT_CATALOG_SCHEMA_FINGERPRINT,
    _DEFAULT_CATALOG_SCHEMA_ID,
    _DEFAULT_CATALOG_SCHEMA_REVISION,
)
from .lifecycle_authority import EntryExpectation
from .manifest import (
    BlobManifest,
    verify_current_manifest,
)
from .path_security import encode_physical_name
from .projections import (
    ProjectionCapabilityError,
    ProjectionController,
    ProjectionOutcome,
)
from .reconciliation import ReconciliationReport, _AuthorityReconciler
from .read_contract import BlobEntry, BlobReceipt
from .sqlite_lifecycle_authority import SqliteLifecycleAuthority


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
            storage=CacheStorageConfig(cache_dir=self.cache_dir),
            compression=CompressionConfig(
                pickle_compression_codec=compression,
                pickle_compression_level=compression_level,
                blosc2_array_clevel=compression_level,
            ),
        )
        self.lifecycle_limits = self.config.lifecycle_limits
        self._instance_admission = InstanceAdmission(self.lifecycle_limits)
        self._immutable_metadata_patch_fields = _IMMUTABLE_METADATA_PATCH_FIELDS
        self.handlers = HandlerRegistry(self.config)
        if manifest_key_provider is not None:
            self._manifest_key_provider = manifest_key_provider
        elif self._is_memory_topology():
            self._manifest_key_provider = _EphemeralManifestKeyProvider()
        elif self.topology.qualified_profile.requirements.coordination_scope == "multiple_hosts":
            raise CacheBlobBackendError(
                "A multi-host BlobStore requires an external manifest signing key",
                context={"operation": "blob_store_composition", "stage": "signing_key"},
            )
        else:
            self._manifest_key_provider = ManifestKeyProvider(
                self.cache_dir / "blob_manifest_hmac_key.bin",
                lifecycle_limits=self.lifecycle_limits,
            )
        self.capabilities = self.topology.capabilities
        self.lifecycle = AuthorityLifecycleEngine(self, self.lifecycle_authority)
        # Kept as a direct engine alias for private timing seams only. It is
        # never selected conditionally and cannot represent another authority.
        self._authority_lifecycle = self.lifecycle

        # Only explicit ProjectionSink participants are driven by the generic
        # controller. A projection never participates in lifecycle authority.
        self._projection_controllers = self._create_projection_controllers()

    def _create_projection_controllers(self) -> tuple[ProjectionController, ...]:
        """Build only explicit derived controllers from topology projections."""
        controllers: list[ProjectionController] = []
        names: set[str] = set()
        for projection in self.projections:
            query = getattr(projection, "projection_query", None)
            schema = getattr(projection, "projection_schema", None)
            if query is None and schema is None:
                continue
            if query is None or schema is None:
                raise TypeError(
                    "ProjectionSink must declare projection_query and projection_schema together"
                )
            controller = ProjectionController(
                self,
                projection,
                page_size=getattr(projection, "projection_page_size", DEFAULT_PAGE_SIZE),
                work_cap=getattr(projection, "projection_work_cap", None),
                query=query,
                schema=schema,
                source_store_id=getattr(projection, "source_store_id", None),
                capabilities=ParticipantCapabilities.from_participant(
                    projection, BackendRole.PROJECTION.value
                ),
            )
            if controller.projection_name in names:
                raise TypeError("ProjectionSink names must be unique within one BlobStore")
            names.add(controller.projection_name)
            controllers.append(controller)
        return tuple(controllers)

    def _receipt_for_result(self, result: Any) -> BlobReceipt:
        """Freeze the canonical authority result before any derived attempt."""
        if result.promoted is None:
            raise CacheBlobLifecycleConflictError("Committed put lacks an authority entry")
        return BlobReceipt(
            operation_id=result.operation_id,
            key=result.promoted.key,
            generation=result.promoted.generation,
            locator=result.promoted.locator,
            expectation=result.promoted.expectation,
            catalog_revision=result.promoted.expectation.revision or 0,
            projections={},
        )

    def _run_post_commit_projections(self, receipt: BlobReceipt) -> BlobReceipt:
        """Attempt derived work after authority commit without any rollback path."""
        outcomes: dict[str, ProjectionOutcome] = {}
        for controller in self._projection_controllers:
            attempt = controller.best_effort(receipt)
            if attempt.outcome is not None:
                outcomes[controller.projection_name] = attempt.outcome
        return receipt.with_projection_outcomes(outcomes)

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

    @property
    def projection_store_id(self) -> str:
        """Return a stable non-path checkpoint identity for this store topology.

        Continuation cursors remain authenticated by the authority's manifest
        key. This value binds a projection checkpoint before a terminal page
        has a cursor to carry the authority's opaque store identity.
        """
        source = (
            f"{type(self.lifecycle_authority).__module__}."
            f"{type(self.lifecycle_authority).__qualname__}:"
            f"{self.cache_dir.absolute()}"
        )
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

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
        if self._initialized:
            self.lifecycle_authority.preflight_mutation()
            return
        initializer = getattr(self.lifecycle_authority, "initialize", None)
        if callable(initializer):
            initializer()
        # The explicit authority initializer creates only a fresh, current
        # layout.  Validate it afterwards so a new PostgreSQL authority is
        # provisioned through this public boundary, while existing layouts
        # remain read-only validation failures rather than implicit upgrades.
        self.lifecycle_authority.preflight_mutation()
        self._materialize_authority_store()
        if self.topology.qualified_profile.requirements.coordination_scope == "multiple_hosts":
            # A remote authority must not materialize its catalog merely to
            # decide whether an application-owned shared key exists. The
            # injected provider is already an explicit topology prerequisite.
            self._authority_manifest_key()
        else:
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
    def get_entry_info(self, key: str) -> BlobEntry | None:
        """Inspect authenticated metadata without reading/deserializing payloads."""
        return self.lifecycle.get_entry_info(key)

    @_ordinary_admitted
    def put_entry(
        self,
        data: Any,
        key=None,
        metadata=None,
        *,
        catalog_schema: CatalogSchema | None = None,
        catalog_values: Dict[str, Any] | None = None,
    ) -> BlobReceipt:
        """Commit an entry and own its cleanup, returning the exact receipt."""
        result = self._put_with_result_admitted(
            data,
            key=key,
            metadata=metadata,
            catalog_schema=catalog_schema,
            catalog_values=catalog_values,
        )
        return self._run_post_commit_projections(self._receipt_for_result(result))

    @_ordinary_admitted
    def put(
        self,
        data: Any,
        key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        catalog_schema: CatalogSchema | None = None,
        catalog_values: Dict[str, Any] | None = None,
    ) -> str:
        """Store one native handler payload through the selected authority."""
        result = self._put_with_result_admitted(
            data,
            key=key,
            metadata=metadata,
            catalog_schema=catalog_schema,
            catalog_values=catalog_values,
        )
        self._run_post_commit_projections(self._receipt_for_result(result))
        return result.key


    def _put_with_result_admitted(
        self,
        data: Any,
        key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        catalog_schema: CatalogSchema | None = None,
        catalog_values: Dict[str, Any] | None = None,
    ):
        """Store one payload after validation and exactly one public admission."""
        catalog = self._prepare_catalog_write(catalog_schema, catalog_values)
        blob_key = (
            self._compute_content_hash(data) if self.content_addressable
            else key if key is not None else self._generate_unique_key()
        )
        return self.lifecycle.put(
            data,
            key=blob_key,
            metadata=metadata,
            catalog_schema_id=catalog[0],
            catalog_schema_revision=catalog[1],
            catalog_schema_fingerprint=catalog[2],
            catalog_values=catalog[3],
        )

    @staticmethod
    def _prepare_catalog_write(
        catalog_schema: CatalogSchema | None,
        catalog_values: Dict[str, Any] | None,
    ) -> tuple[str, int, str, dict[str, Any]]:
        """Freeze catalog inputs before handlers, authority, or payload I/O run."""
        supplied = {} if catalog_values is None else catalog_values
        if catalog_schema is None:
            values = validate_catalog_mapping(supplied, schema=None)
            return (
                _DEFAULT_CATALOG_SCHEMA_ID,
                _DEFAULT_CATALOG_SCHEMA_REVISION,
                _DEFAULT_CATALOG_SCHEMA_FINGERPRINT,
                values,
            )
        if not isinstance(catalog_schema, CatalogSchema):
            raise TypeError("catalog_schema must be a CatalogSchema or None")
        values = catalog_schema.validate_mapping(supplied, materialize_defaults=True)
        return (
            catalog_schema.schema_id,
            catalog_schema.revision,
            catalog_schema.fingerprint,
            values,
        )


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
        return self.lifecycle.update_metadata(key, metadata)

    @_ordinary_admitted
    def update_catalog(
        self,
        key: str,
        *,
        catalog_schema: CatalogSchema,
        catalog_values: Dict[str, Any],
        expected: EntryExpectation | None = None,
        replace: bool = False,
    ) -> BlobReceipt | None:
        """Patch or replace authenticated catalog values without rewriting bytes.

        Catalog patches preserve the selected payload generation and use the
        exact observed authority record as their compare-and-swap precondition.
        ``replace=True`` records only the supplied stored fields; the default
        patches the currently stored mapping.
        """
        if not isinstance(catalog_schema, CatalogSchema):
            raise TypeError("catalog_schema must be a CatalogSchema")
        if type(replace) is not bool:
            raise TypeError("replace must be a bool")
        patch = self._prepare_catalog_patch(
            catalog_schema, catalog_values, replace=replace
        )
        result = self.lifecycle.update_catalog(
            key,
            catalog_schema=catalog_schema,
            catalog_values=patch,
            expected=expected,
            replace_values=replace,
        )
        if result is None:
            return None
        return self._run_post_commit_projections(self._receipt_for_result(result))

    @staticmethod
    def _prepare_catalog_patch(
        catalog_schema: CatalogSchema,
        catalog_values: Dict[str, Any],
        *,
        replace: bool,
    ) -> dict[str, Any]:
        """Validate a bounded replacement or patch before authority access."""
        patch = validate_catalog_mapping(catalog_values, schema=None)
        if replace:
            return catalog_schema.validate_mapping(patch, materialize_defaults=False)
        for name, value in patch.items():
            declared = catalog_schema.field_map.get(name)
            if declared is not None:
                declared.validate(value)
        return patch

    @_ordinary_admitted
    def delete(
        self, key: str, *, expected: EntryExpectation | None = None
    ) -> bool:
        """Tombstone one observed generation and delete its exact payload."""
        deleted = self.lifecycle.delete(key=key, expected=expected)
        return deleted

    @_ordinary_admitted
    def exists(self, key: str) -> bool:
        """Verify a signed payload snapshot exists through the authority."""
        return self.lifecycle.exists(key)

    @_ordinary_admitted
    def list(self, prefix: Optional[str] = None) -> List[str]:
        """List local authority entries by key prefix only.

        The remote reference topology cannot return a complete catalog in one
        value. Call :meth:`list_page` there and follow its authenticated
        continuation cursor instead.
        """
        return self.lifecycle.list(prefix)

    @_ordinary_admitted
    def list_page(
        self,
        *,
        schema: CatalogSchema,
        cursor: str | None = None,
        limit: int = DEFAULT_PAGE_SIZE,
        work_cap: int | None = None,
    ) -> CatalogPage:
        """Return one bounded committed catalog page for remote enumeration.

        ``CatalogPage`` carries the authority-authenticated continuation cursor,
        so callers cannot accidentally turn a remote catalog into a silently
        truncated list. It remains authority membership, never S3 inventory.
        """
        query = CatalogQuery(page_size=limit, cursor=cursor)
        return self.query_catalog(
            query,
            schema=schema,
            cursor=cursor,
            limit=limit,
            work_cap=work_cap,
        )

    @_ordinary_admitted
    def query_catalog(
        self,
        query: CatalogQuery,
        *,
        schema: CatalogSchema,
        cursor: str | None = None,
        limit: int | None = None,
        work_cap: int | None = None,
    ) -> CatalogPage:
        """Return a bounded portable page from signed current descriptors.

        The authority is the sole membership catalog.  It keyset-enumerates
        current descriptor bytes; this facade authenticates each descriptor
        before applying native predicates and exposes their exact expectations,
        without creating a metadata mirror or claiming an acceleration index.
        """
        if not isinstance(query, CatalogQuery):
            # Keep the error type and input validation boundary in catalog.py.
            validate_catalog_page_request(
                query, schema=schema, cursor=cursor, limit=1, work_cap=1
            )
        effective_cursor = query.cursor if cursor is None else cursor
        effective_limit = query.page_size if limit is None else limit
        if work_cap is None:
            effective_work_cap = (
                max(effective_limit, DEFAULT_PAGE_SIZE)
                if isinstance(effective_limit, int) and not isinstance(effective_limit, bool)
                else DEFAULT_PAGE_SIZE
            )
        else:
            effective_work_cap = work_cap
        validate_catalog_page_request(
            query,
            schema=schema,
            cursor=effective_cursor,
            limit=effective_limit,
            work_cap=effective_work_cap,
        )
        signing_key = self._authority_manifest_key()
        if effective_cursor is not None:
            CatalogCursor.inspect(effective_cursor, signing_key=signing_key)
        return self.lifecycle_authority.catalog_page(
            query,
            effective_cursor,
            schema=schema,
            limit=effective_limit,
            work_cap=effective_work_cap,
            signing_key=signing_key,
            manifest_loader=self._authenticated_authority_manifest,
        )

    @_ordinary_admitted
    def refresh_projection(self, name: str, receipt: BlobReceipt):
        """Run one caller-requested bounded derived refresh.

        A projection failure raises a committed-partial error that carries this
        exact receipt. No authority row, payload generation, or cleanup debt is
        changed by the refresh attempt.
        """
        return self._projection_controller(name).refresh(receipt)

    @_ordinary_admitted
    def rebuild_projection(
        self,
        name: str,
        *,
        requested: str = "offline",
        workers_stopped: bool = False,
    ):
        """Explicitly rebuild one derived projection into an isolated destination."""
        if isinstance(self.lifecycle_authority, SqliteLifecycleAuthority) and (
            requested != "offline" or workers_stopped is not True
        ):
            raise ProjectionCapabilityError(
                "SQLite projection rebuild requires explicit offline maintenance "
                "with workers stopped"
            )
        return self._projection_controller(name).rebuild(requested=requested)

    def _projection_controller(self, name: str) -> ProjectionController:
        if not isinstance(name, str) or not name:
            raise ValueError("Projection name must be a non-empty string")
        for controller in self._projection_controllers:
            if controller.projection_name == name:
                return controller
        raise KeyError(f"No configured projection named {name!r}")

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

    def _materialize_authority_store(self) -> Any:
        """Materialize guarded generation I/O from the selected payload only."""
        if self.guarded_handler_io is None:
            self.guarded_handler_io = self.payload_backend.materialize_handler_io()
            required_methods = (
                "stage",
                "publish_generation",
                "open_snapshot",
                "delete_or_prove_absent",
                "close",
            )
            if not all(
                callable(getattr(self.guarded_handler_io, name, None))
                for name in required_methods
            ):
                raise CacheBlobBackendError(
                    "Payload participant returned incomplete guarded generation I/O"
                )
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

    def _authenticated_authority_manifest(
        self, raw: bytes, *, allow_tombstone: bool = False
    ) -> BlobManifest:
        """Authenticate canonical authority bytes before trusting any locator."""
        try:
            manifest = BlobManifest.from_canonical_bytes(raw)
        except CacheManifestUnsupportedVersionError as exc:
            raise CacheBlobManifestUnsupportedVersionError(
                "Authority manifest schema version is unsupported"
            ) from exc
        except CacheManifestIntegrityError as exc:
            raise CacheBlobManifestMalformedError("Authority manifest is malformed") from exc
        try:
            verify_current_manifest(manifest, self._authority_manifest_key())
        except CacheManifestIntegrityError as exc:
            raise CacheBlobManifestUnauthenticatedError(
                "Authority manifest cannot be authenticated"
            ) from exc
        if manifest.canonical_bytes() != raw:
            raise CacheBlobManifestMalformedError("Authority manifest is not canonical")
        return manifest

    def _storage_id_for_key(self, key: str) -> str:
        return encode_physical_name(key, namespace="blob-store")

    def _delete_or_prove_absent(self, locator: Path) -> None:
        """Delegate exact cleanup to the selected payload participant's I/O."""
        self._materialize_authority_store().delete_or_prove_absent(locator)

    def delete_migration_payload(self, locator: str) -> None:
        """Delete one authority-correlated retired migration locator idempotently.

        This narrow maintenance-only primitive is intentionally not an ordinary
        BlobStore deletion path: callers must obtain the exact locator from the
        selected migration authority before requesting external cleanup.
        """
        if not isinstance(locator, str) or not locator:
            raise ValueError("migration payload locator must be a non-empty string")
        self._materialize_authority_store().delete_or_prove_absent(locator)

    def _resolve_payload_handler(self, manifest: BlobManifest) -> Any:
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
    def _handler_metadata(manifest: BlobManifest) -> Dict[str, Any]:
        return {
            **dict(manifest.user_metadata), **dict(manifest.handler_metadata),
            "cache_key": manifest.key, "data_type": manifest.handler_type,
            "storage_format": manifest.payload_format, "file_size": manifest.byte_size,
            "created_at": manifest.created_at,
        }

    def _manifest_entry_data(self, manifest: BlobManifest) -> Dict[str, Any]:
        return {
            "cache_key": manifest.key,
            "data_type": manifest.handler_type,
            "file_size": manifest.byte_size,
            "created_at": manifest.created_at,
            "metadata": {
                **dict(manifest.user_metadata), **dict(manifest.handler_metadata),
                "actual_path": manifest.locator,
            },
            "catalog": {
                "schema_id": manifest.catalog_schema_id,
                "schema_revision": manifest.catalog_schema_revision,
                "schema_fingerprint": manifest.catalog_schema_fingerprint,
                "values": dict(manifest.catalog_values),
                "presence": manifest.catalog_presence,
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
        require_worker_access = getattr(
            self.lifecycle_authority, "require_ordinary_worker_access", None
        )
        if callable(require_worker_access):
            require_worker_access()

    def _is_memory_topology(self) -> bool:
        return self.topology.qualified_profile.pair == ("memory", "memory")

    @staticmethod
    def _generate_unique_key() -> str:
        return uuid.uuid4().hex[:16]
