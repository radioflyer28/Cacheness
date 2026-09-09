"""Cache policy built on a single internal :class:`BlobStore` engine.

``UnifiedCache`` deliberately owns only cache policy: key derivation, TTL,
entry-size eviction, and hit/miss accounting. Payload publication, catalog
membership, recovery, and resource ownership belong to its composed
``BlobStore``. The facade does not expose or maintain a second metadata
authority.
"""

from __future__ import annotations

import hashlib
import hmac
import inspect
import json
import logging
import secrets
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

from .cache_policy import (
    CacheLookupResult,
    CacheMaintenancePhase,
    CacheMaintenanceResult,
    CacheMaintenanceState,
    CacheOutcome,
    CachePutResult,
    CacheRemovalReport,
    CacheStatistics,
    _CacheOutcomeRecorder,
    _CacheRemovalCandidate,
    execute_exact_removals,
)
from .config import (
    CacheConfig,
    _DEFAULT_TTL,
    validate_config_strict,
)
from .error_handling import (
    CacheBlobBackendError,
    CacheBlobIntegrityError,
    CacheBlobLifecycleConflictError,
    CacheBlobLifecycleTimeoutError,
    CacheBlobManifestUnsupportedVersionError,
    CacheBlobMigrationRequiredError,
    CacheBlobPayloadUnsupportedVersionError,
    CacheBlobStoreClosedError,
    CacheManifestUnsupportedVersionError,
)
from .handlers import HandlerRegistry
from .serialization import create_unified_cache_key
from .storage.blob_store import BlobStore
from .storage.catalog import (
    DEFAULT_PAGE_SIZE,
    CatalogEntry,
    CatalogField,
    CatalogPage,
    CatalogPredicate,
    CatalogQuery,
    CatalogSchema,
    CatalogStaleCursorError,
    validate_catalog_page_request,
    validate_catalog_query,
)
from .storage.composition import StoreTopology
from .storage.path_security import encode_physical_name
from .storage.read_contract import CacheReadFailureCategory, classify_cache_read_failure


logger = logging.getLogger(__name__)


_CACHE_POLICY_SCHEMA = CatalogSchema(
    fields=(
        CatalogField("cache_namespace", "string", required=True, queryable=True),
        CatalogField("cache_prefix", "string", required=True, queryable=True),
        CatalogField("function_namespace", "string", queryable=True),
    ),
    schema_id="cacheness-cache-policy",
    revision=1,
)
_CACHE_NAMESPACE = "unified-cache"
_REMOVAL_PAGE_SIZE = DEFAULT_PAGE_SIZE
_REMOVAL_WORK_CAP = DEFAULT_PAGE_SIZE
_RESTART_REMOVAL_CURSOR = "cache-policy:restart"


def _clear_coordinated(method: Callable) -> Callable:
    """Reject facade work after close without adding a lifecycle lock."""

    @wraps(method)
    def wrapped(self: "UnifiedCache", *args: Any, **kwargs: Any) -> Any:
        if self._closed:
            raise CacheBlobStoreClosedError("Cache is closed")
        return method(self, *args, **kwargs)

    return wrapped


_clear_read_coordinated = _clear_coordinated


def _normalize_function_args(
    func: Callable, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Normalize equivalent function calling conventions for cache keys."""

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return {**{f"__arg_{index}": value for index, value in enumerate(args)}, **kwargs}
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


class UnifiedCache:
    """A narrow cache-policy layer backed by one private ``BlobStore``.

    This class intentionally does not accept a metadata backend. Applications
    that need direct persistence compose and use ``BlobStore``; cache callers
    use this facade for policy only.
    """

    def __init__(
        self, config: CacheConfig, *, store: BlobStore | StoreTopology
    ) -> None:
        """Compose one caller-selected store for this cache policy instance."""

        if not isinstance(config, CacheConfig):
            raise TypeError("config must be a CacheConfig")
        validate_config_strict(config)
        self.config = config
        self.cache_dir = Path(self.config.storage.cache_dir)
        self._closed = False
        self._outcome_recorder = _CacheOutcomeRecorder()
        self._maintenance_secret = secrets.token_bytes(32)
        if isinstance(store, BlobStore):
            # An application has already selected, composed, and owns this
            # store. Validate its declared profile without initializing or
            # otherwise mutating the caller's lifecycle resource.
            if not hasattr(store.topology, "qualified_profile"):
                raise TypeError("injected BlobStore must expose a qualified topology")
            self.store = store
            self.handlers = store.handlers
            self._owned_store: BlobStore | None = None
        elif isinstance(store, StoreTopology):
            # Preflight profile/capability declarations before BlobStore can
            # resolve participants or materialize any storage resource.
            store.qualification_report()
            store.capability_report()
            root = self.cache_dir / ".cacheness" / "blobstore"
            self.handlers = HandlerRegistry(self.config)
            self.store = BlobStore(store, cache_dir=root, config=self.config)
            self.store.handlers = self.handlers
            self._owned_store = self.store
        else:
            raise TypeError("store must be a BlobStore or StoreTopology")
        self._cache_blob_store = self.store
        self.actual_backend = "-".join(
            self._cache_blob_store.topology.qualified_profile.pair
        )
        logger.info(
            "Unified cache initialized at %s using BlobStore topology %s",
            self.cache_dir,
            self.actual_backend,
        )

    def initialize(self) -> None:
        """Initialize a cache-created store before sharing the cache with workers.

        An injected store remains caller-owned, including its initialization
        boundary.  It is already a validated BlobStore composition, so the
        facade has no participant lifecycle work to perform for that form.
        """

        if self._closed:
            raise CacheBlobStoreClosedError("Cache is closed")
        if self._owned_store is not None:
            self._owned_store.initialize()

    def _create_cache_key(self, params: Mapping[str, Any]) -> str:
        """Return the deterministic public cache identity for ``params``."""

        return create_unified_cache_key(dict(params), self.config)

    @staticmethod
    def function_namespace(func: Callable) -> str:
        """Return the stable, queryable namespace for one decorated function."""

        target = inspect.unwrap(func)
        module = getattr(target, "__module__", None)
        qualname = getattr(target, "__qualname__", None)
        if not isinstance(module, str) or not module:
            raise TypeError("cached functions require a non-empty __module__")
        if not isinstance(qualname, str) or not qualname:
            raise TypeError("cached functions require a non-empty __qualname__")
        return f"{module}.{qualname}"

    def _function_cache_key(
        self, func: Callable, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[str, str]:
        """Derive one policy-owned key from function identity and bound arguments."""

        namespace = self.function_namespace(func)
        normalized = _normalize_function_args(func, args, kwargs)
        return (
            namespace,
            self._create_cache_key(
                {
                    "__function_namespace__": namespace,
                    "__function_arguments__": normalized,
                }
            ),
        )

    def _get_cache_file_path(self, cache_key: str, prefix: str = "") -> Path:
        """Return an opaque path-like diagnostic identity, not a payload locator."""

        return self.cache_dir / self._storage_id_for_cache_key(cache_key, prefix)

    @staticmethod
    def _storage_id_for_cache_key(cache_key: str, prefix: str = "") -> str:
        return encode_physical_name(cache_key, prefix, namespace="unified-cache")

    @staticmethod
    def _plain_value(value: Any) -> Any:
        """Copy frozen storage metadata into policy-owned, mutable values."""

        if isinstance(value, Mapping):
            return {key: UnifiedCache._plain_value(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [UnifiedCache._plain_value(item) for item in value]
        return value

    @staticmethod
    def _canonical_cache_key_params(params: Mapping[str, Any]) -> dict[str, str]:
        """Store inspectable key parameters only when policy enables it."""

        return {str(key): repr(value) for key, value in params.items()}

    def _cache_entry(self, snapshot: Any) -> dict[str, Any]:
        """Render an authenticated BlobStore snapshot for cache policy."""

        raw = self._plain_value(snapshot.metadata)
        metadata = raw.get("metadata", {}) if isinstance(raw, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        created_at = raw.get("created_at") if isinstance(raw, dict) else None
        file_size = raw.get("file_size", 0) if isinstance(raw, dict) else 0
        data_type = raw.get("data_type") if isinstance(raw, dict) else None
        return {
            "cache_key": snapshot.key,
            "generation": snapshot.generation,
            "data_type": data_type,
            "file_size": file_size if isinstance(file_size, int) else 0,
            "created_at": created_at,
            "metadata": metadata,
            "prefix": metadata.get("prefix", ""),
            "description": metadata.get("description", ""),
        }

    @staticmethod
    def _raise_malformed_policy_facts(message: str) -> None:
        """Raise one typed fail-closed result for unauthenticated policy input."""

        raise CacheBlobIntegrityError(message)

    def _policy_entry_from_snapshot(self, snapshot: Any) -> dict[str, Any]:
        """Validate the authenticated intrinsic facts needed for cache policy."""

        raw = self._plain_value(snapshot.metadata)
        if not isinstance(raw, dict):
            self._raise_malformed_policy_facts("Cache policy snapshot metadata is invalid")
        file_size = raw.get("file_size")
        created_at = raw.get("created_at")
        catalog = raw.get("catalog")
        if (
            not isinstance(file_size, int)
            or isinstance(file_size, bool)
            or file_size < 0
            or not isinstance(created_at, str)
            or not created_at
            or not isinstance(catalog, dict)
            or catalog.get("schema_id") != _CACHE_POLICY_SCHEMA.schema_id
            or catalog.get("schema_revision") != _CACHE_POLICY_SCHEMA.revision
            or catalog.get("schema_fingerprint") != _CACHE_POLICY_SCHEMA.fingerprint
        ):
            self._raise_malformed_policy_facts(
                "Cache policy requires authenticated intrinsic catalog facts"
            )
        try:
            datetime.fromisoformat(created_at)
            values = _CACHE_POLICY_SCHEMA.read_mapping(catalog.get("values", {}))
        except (TypeError, ValueError) as error:
            raise CacheBlobIntegrityError(
                "Cache policy intrinsic catalog facts are malformed"
            ) from error
        if values.get("cache_namespace") != _CACHE_NAMESPACE:
            self._raise_malformed_policy_facts("Cache policy namespace is invalid")
        return self._cache_entry(snapshot)

    @staticmethod
    def _candidate_from_snapshot(snapshot: Any) -> _CacheRemovalCandidate:
        """Retain the exact expectation from one authenticated entry snapshot."""

        return _CacheRemovalCandidate(snapshot.key, snapshot.expectation)

    def _candidate_from_catalog_entry(
        self, entry: CatalogEntry
    ) -> _CacheRemovalCandidate:
        """Reject incomplete catalog policy facts before lifecycle mutation."""

        if (
            entry.expectation is None
            or entry.byte_size is None
            or entry.created_at is None
            or entry.schema_id != _CACHE_POLICY_SCHEMA.schema_id
            or entry.schema_revision != _CACHE_POLICY_SCHEMA.revision
        ):
            self._raise_malformed_policy_facts(
                "Catalog entry lacks authenticated cache-policy facts"
            )
        try:
            datetime.fromisoformat(entry.created_at)
        except ValueError as error:
            raise CacheBlobIntegrityError(
                "Catalog entry creation time is malformed"
            ) from error
        return _CacheRemovalCandidate(entry.key, entry.expectation)

    def _remove_exact_candidates(
        self,
        candidates: tuple[_CacheRemovalCandidate, ...],
        *,
        complete: bool = True,
        continuation: str | None = None,
    ) -> CacheRemovalReport:
        """Delegate every exact cache-policy removal to BlobStore.delete."""

        return execute_exact_removals(
            candidates,
            delete=lambda candidate: self._cache_blob_store.delete(
                candidate.key, expected=candidate.expectation
            ),
            complete=complete,
            continuation=continuation,
        )

    @staticmethod
    def _is_expired_at(created_at: str, ttl_hours: object) -> bool:
        """Evaluate one validated cache-policy timestamp without storage I/O."""

        if ttl_hours is None:
            return False
        if not isinstance(ttl_hours, (int, float)) or isinstance(ttl_hours, bool):
            raise TypeError("ttl_hours must be a number, None, or the default sentinel")
        created = datetime.fromisoformat(created_at)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) > created + timedelta(hours=ttl_hours)

    def _authority_snapshot_entry(
        self, cache_key: str
    ) -> tuple[Any | None, dict[str, Any] | None]:
        snapshot = self._cache_blob_store.get_entry_info(cache_key)
        return (snapshot, self._cache_entry(snapshot)) if snapshot is not None else (None, None)

    def _is_expired(
        self,
        cache_key: str,
        ttl_hours: object = _DEFAULT_TTL,
        entry: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Determine expiration without touching storage.

        ``None`` means no expiry. A negative numeric TTL is useful to force an
        immediate policy miss in tests and administrative callers.
        """

        if entry is None:
            _, entry = self._authority_snapshot_entry(cache_key)
        if entry is None:
            return True
        if ttl_hours is None:
            return False
        ttl = (
            self.config.policy.default_ttl_hours
            if ttl_hours is _DEFAULT_TTL
            else ttl_hours
        )
        if not isinstance(ttl, (int, float)) or isinstance(ttl, bool):
            raise TypeError("ttl_hours must be a number, None, or the default sentinel")
        created_at = entry.get("created_at")
        if isinstance(created_at, datetime):
            created_at = created_at.isoformat()
        if not isinstance(created_at, str) or not created_at:
            return True
        return self._is_expired_at(created_at, ttl)

    def _retire_exact_authority_snapshot(
        self, cache_key: str, snapshot: Any | None
    ) -> bool:
        """Delete only the observed generation, never a replacement by key alone."""

        if snapshot is None:
            return False
        report = self._remove_exact_candidates(
            (self._candidate_from_snapshot(snapshot),)
        )
        return report.removed == 1

    def _cleanup_expired(
        self,
        *,
        cursor: str | None = None,
        page_size: int = _REMOVAL_PAGE_SIZE,
        work_cap: int = _REMOVAL_WORK_CAP,
    ) -> CacheRemovalReport:
        """Select one bounded page of expired entries and exact-delete it."""

        # Catalog cursors are revision-bound.  An exact deletion below advances
        # that revision, so the cache-policy restart token deliberately starts
        # a fresh bounded scan instead of presenting a cursor it invalidated.
        query_cursor = None if cursor == _RESTART_REMOVAL_CURSOR else cursor
        try:
            page = self._query_cache_catalog(
                CatalogQuery(page_size=page_size),
                cursor=query_cursor,
                page_size=page_size,
                work_cap=work_cap,
            )
        except CatalogStaleCursorError:
            if query_cursor is None:
                raise
            return CacheRemovalReport(
                retryable=1,
                complete=False,
                continuation=_RESTART_REMOVAL_CURSOR,
            )
        ttl = self.config.policy.default_ttl_hours
        candidates = tuple(
            self._candidate_from_catalog_entry(entry)
            for entry in page.entries
            if self._is_expired_at(entry.created_at, ttl)
        )
        report = self._remove_exact_candidates(
            candidates,
            complete=page.exhausted,
            continuation=page.cursor,
        )
        if report.removed and not page.exhausted:
            return replace(
                report,
                complete=False,
                continuation=_RESTART_REMOVAL_CURSOR,
            )
        if report.removed:
            logger.info("Cleaned up %s expired cache entries", report.removed)
        return report

    def _query_cache_catalog(
        self,
        query: CatalogQuery,
        *,
        cursor: str | None,
        page_size: int,
        work_cap: int,
    ) -> CatalogPage:
        """Validate a cache-policy page before the BlobStore authority is touched."""

        validate_catalog_query(query, schema=_CACHE_POLICY_SCHEMA)
        validate_catalog_page_request(
            query,
            schema=_CACHE_POLICY_SCHEMA,
            cursor=cursor,
            limit=page_size,
            work_cap=work_cap,
        )
        return self._cache_blob_store.query_catalog(
            query,
            schema=_CACHE_POLICY_SCHEMA,
            cursor=cursor,
            limit=page_size,
            work_cap=work_cap,
        )

    def _maintenance_composition_fingerprint(self) -> str:
        """Bind policy continuation to this store without making it authority."""

        source = ":".join(
            (
                self._cache_blob_store.projection_store_id,
                self.actual_backend,
                _CACHE_POLICY_SCHEMA.fingerprint,
            )
        )
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    def _maintenance_configuration_fingerprint(self) -> str:
        """Bind continuation evidence to the exact finite policy configuration."""

        policy = self.config.policy
        payload = {
            "catalog_page_size": policy.catalog_page_size,
            "default_ttl_hours": policy.default_ttl_hours,
            "maintenance_work_cap": policy.maintenance_work_cap,
            "max_authoritative_bytes": policy.max_authoritative_bytes,
            "max_maintenance_state_bytes": policy.max_maintenance_state_bytes,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    def _seal_maintenance_state(
        self,
        *,
        phase: CacheMaintenancePhase,
        authority_revision: int | None,
        cursor: str | None,
        observed_bytes: int = 0,
        excess_bytes: int = 0,
        candidate_key: str | None = None,
        candidate_expectation: Any | None = None,
        candidate_byte_size: int | None = None,
    ) -> CacheMaintenanceState:
        """Create bounded, instance-signed continuation evidence.

        Signing protects the opaque policy token from accidental or malicious
        modification.  It does not authorize storage work on its own: every
        later operation still validates authoritative catalog facts and routes
        mutation through ``BlobStore.delete(expected=...)``.
        """

        unsigned = CacheMaintenanceState(
            phase=phase,
            composition_fingerprint=self._maintenance_composition_fingerprint(),
            configuration_fingerprint=self._maintenance_configuration_fingerprint(),
            authority_revision=authority_revision,
            cursor=cursor,
            observed_bytes=observed_bytes,
            excess_bytes=excess_bytes,
            candidate_key=candidate_key,
            candidate_expectation=candidate_expectation,
            candidate_byte_size=candidate_byte_size,
            signature="0" * 64,
        )
        if unsigned.encoded_size > self.config.policy.max_maintenance_state_bytes:
            raise CacheBlobIntegrityError(
                "Cache maintenance continuation exceeds its configured byte bound"
            )
        signature = hmac.new(
            self._maintenance_secret,
            unsigned.encoded.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        return replace(unsigned, signature=signature)

    def _validate_maintenance_state(
        self, state: CacheMaintenanceState
    ) -> CacheMaintenanceState:
        """Reject foreign or modified continuation evidence before authority I/O."""

        if not isinstance(state, CacheMaintenanceState):
            raise ValueError("maintenance state must be a CacheMaintenanceState")
        if state.composition_fingerprint != self._maintenance_composition_fingerprint():
            raise ValueError("maintenance state belongs to another cache composition")
        if state.configuration_fingerprint != self._maintenance_configuration_fingerprint():
            raise ValueError("maintenance state does not match the active policy configuration")
        if state.encoded_size > self.config.policy.max_maintenance_state_bytes:
            raise ValueError("maintenance state exceeds the configured byte bound")
        expected_signature = hmac.new(
            self._maintenance_secret,
            state.encoded.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(state.signature, expected_signature):
            raise ValueError("maintenance state signature is invalid")
        return state

    @staticmethod
    def _maintenance_restart(
        cause: BaseException,
        removal: CacheRemovalReport | None = None,
    ) -> CacheMaintenanceResult:
        """Return a typed restart outcome without claiming size completion."""

        return CacheMaintenanceResult(
            complete=False,
            retryable=True,
            removal=CacheRemovalReport() if removal is None else removal,
            cause=cause,
        )

    def _maintenance_pending(
        self,
        *,
        phase: CacheMaintenancePhase,
        authority_revision: int | None,
        cursor: str | None,
        observed_bytes: int = 0,
        excess_bytes: int = 0,
        candidate_key: str | None = None,
        candidate_expectation: Any | None = None,
        candidate_byte_size: int | None = None,
        removal: CacheRemovalReport | None = None,
    ) -> CacheMaintenanceResult:
        """Return one caller-driven continuation after exactly this work step."""

        state = self._seal_maintenance_state(
            phase=phase,
            authority_revision=authority_revision,
            cursor=cursor,
            observed_bytes=observed_bytes,
            excess_bytes=excess_bytes,
            candidate_key=candidate_key,
            candidate_expectation=candidate_expectation,
            candidate_byte_size=candidate_byte_size,
        )
        report = CacheRemovalReport() if removal is None else removal
        report = replace(report, complete=False, continuation=state.encoded)
        return CacheMaintenanceResult(
            complete=False,
            retryable=True,
            removal=report,
            state=state,
        )

    def _maintenance_page(self, cursor: str | None) -> CatalogPage:
        """Read one policy-bounded authenticated catalog page."""

        policy = self.config.policy
        return self._query_cache_catalog(
            CatalogQuery(page_size=policy.catalog_page_size),
            cursor=cursor,
            page_size=policy.catalog_page_size,
            work_cap=policy.maintenance_work_cap,
        )

    @staticmethod
    def _require_maintenance_revision(
        page: CatalogPage, expected_revision: int | None
    ) -> int:
        """Treat changed authority facts as a typed restart, never stale input."""

        if expected_revision is not None and page.revision != expected_revision:
            raise CacheBlobLifecycleConflictError(
                "Authoritative catalog revision changed during size maintenance"
            )
        return page.revision

    def _maintenance_candidate_and_size(
        self, entry: CatalogEntry
    ) -> tuple[_CacheRemovalCandidate, int]:
        """Extract only authenticated intrinsic facts for one eviction choice."""

        candidate = self._candidate_from_catalog_entry(entry)
        if entry.byte_size is None or entry.byte_size < 0:
            self._raise_malformed_policy_facts(
                "Catalog entry has no authenticated byte size"
            )
        return candidate, entry.byte_size

    def _scan_size_maintenance(
        self, state: CacheMaintenanceState, *, verify: bool
    ) -> CacheMaintenanceResult:
        """Run one bounded inventory or verification page.

        A complete scan transitions to a separate fresh verification scan
        before making a completion claim.  No scan follows a continuation or
        mutation cursor after the current method returns.
        """

        page = self._maintenance_page(state.cursor)
        revision = self._require_maintenance_revision(page, state.authority_revision)
        observed_bytes = state.observed_bytes
        for entry in page.entries:
            _, byte_size = self._maintenance_candidate_and_size(entry)
            observed_bytes += byte_size
        if not page.exhausted:
            return self._maintenance_pending(
                phase=state.phase,
                authority_revision=revision,
                cursor=page.cursor,
                observed_bytes=observed_bytes,
                excess_bytes=state.excess_bytes,
            )

        limit = self.config.policy.max_authoritative_bytes
        if verify:
            if observed_bytes <= limit:
                return CacheMaintenanceResult(
                    complete=True,
                    retryable=False,
                    removal=CacheRemovalReport(),
                )
            return self._maintenance_pending(
                phase=CacheMaintenancePhase.EVICT,
                authority_revision=revision,
                cursor=None,
                excess_bytes=observed_bytes - limit,
            )

        if observed_bytes <= limit:
            return self._maintenance_pending(
                phase=CacheMaintenancePhase.VERIFY,
                authority_revision=revision,
                cursor=None,
            )
        return self._maintenance_pending(
            phase=CacheMaintenancePhase.EVICT,
            authority_revision=revision,
            cursor=None,
            excess_bytes=observed_bytes - limit,
        )

    def _evict_size_maintenance(
        self, state: CacheMaintenanceState
    ) -> CacheMaintenanceResult:
        """Select or exact-delete at most one candidate in canonical page order."""

        if state.candidate_key is not None:
            candidate = _CacheRemovalCandidate(
                state.candidate_key, state.candidate_expectation
            )
            report = self._remove_exact_candidates((candidate,))
            if report.failed:
                return self._maintenance_restart(report.failures[0].cause, report)
            if report.conflicted or report.removed != 1:
                return self._maintenance_restart(
                    CacheBlobLifecycleConflictError(
                        "Exact cache eviction candidate changed before deletion"
                    ),
                    report,
                )
            remaining = max(0, state.excess_bytes - state.candidate_byte_size)
            return self._maintenance_pending(
                phase=(
                    CacheMaintenancePhase.VERIFY
                    if remaining == 0
                    else CacheMaintenancePhase.EVICT
                ),
                authority_revision=None,
                cursor=None,
                excess_bytes=remaining,
                removal=report,
            )

        page = self._maintenance_page(state.cursor)
        revision = self._require_maintenance_revision(page, state.authority_revision)
        for entry in page.entries:
            candidate, byte_size = self._maintenance_candidate_and_size(entry)
            if byte_size > 0:
                return self._maintenance_pending(
                    phase=CacheMaintenancePhase.EVICT,
                    authority_revision=revision,
                    cursor=page.cursor,
                    excess_bytes=state.excess_bytes,
                    candidate_key=candidate.key,
                    candidate_expectation=candidate.expectation,
                    candidate_byte_size=byte_size,
                )
        if page.exhausted:
            return self._maintenance_restart(
                CacheBlobIntegrityError(
                    "Authoritative size inventory cannot satisfy a positive eviction excess"
                )
            )
        return self._maintenance_pending(
            phase=CacheMaintenancePhase.EVICT,
            authority_revision=revision,
            cursor=page.cursor,
            excess_bytes=state.excess_bytes,
        )

    def _run_size_maintenance(self, state: CacheMaintenanceState) -> CacheMaintenanceResult:
        """Execute one finite inventory, eviction, or verification step only."""

        try:
            if state.phase is CacheMaintenancePhase.INVENTORY:
                return self._scan_size_maintenance(state, verify=False)
            if state.phase is CacheMaintenancePhase.EVICT:
                return self._evict_size_maintenance(state)
            if state.phase is CacheMaintenancePhase.VERIFY:
                return self._scan_size_maintenance(state, verify=True)
            raise ValueError("maintenance state contains an unsupported phase")
        except (
            CacheBlobBackendError,
            CacheBlobIntegrityError,
            CacheBlobLifecycleConflictError,
            CacheBlobLifecycleTimeoutError,
            CacheBlobStoreClosedError,
            CatalogStaleCursorError,
        ) as error:
            return self._maintenance_restart(error)

    @_clear_coordinated
    def maintain_size(self) -> CacheMaintenanceResult:
        """Start one bounded size-maintenance step from canonical catalog order."""

        return self._start_size_maintenance()

    def _start_size_maintenance(self) -> CacheMaintenanceResult:
        """Run one maintenance step without changing a completed put receipt.

        This private path is used only after ``BlobStore.put_entry`` has
        committed.  It intentionally bypasses the facade close guard so the
        existing maintenance result boundary can report a concurrent close as
        typed, retryable incomplete work rather than hiding the receipt.
        """

        state = self._seal_maintenance_state(
            phase=CacheMaintenancePhase.INVENTORY,
            authority_revision=None,
            cursor=None,
        )
        return self._run_size_maintenance(state)

    @_clear_coordinated
    def resume_maintenance(self, state: CacheMaintenanceState) -> CacheMaintenanceResult:
        """Resume exactly one validated policy-maintenance step.

        This method neither schedules background work nor loops through later
        pages.  Callers decide whether and when to submit the returned state.
        """

        return self._run_size_maintenance(self._validate_maintenance_state(state))

    def _put_for_cache_key(
        self,
        data: Any,
        *,
        cache_key: str,
        prefix: str,
        description: str,
        key_params: Mapping[str, Any],
        function_namespace: str | None = None,
    ) -> CachePutResult:
        """Commit one value under a policy-owned key and catalog namespace.

        BlobStore commits the payload and catalog before policy maintenance
        begins.  The one returned policy step may be incomplete or retryable,
        but it cannot revoke, relabel, or roll back the committed generation.
        """

        metadata: dict[str, Any] = {"prefix": prefix, "description": description}
        if self.config.metadata.store_cache_key_params:
            metadata["cache_key_params"] = self._canonical_cache_key_params(key_params)
        catalog_values: dict[str, str] = {
            "cache_namespace": _CACHE_NAMESPACE,
            "cache_prefix": prefix,
        }
        if function_namespace is not None:
            catalog_values["function_namespace"] = function_namespace
        receipt = self._cache_blob_store.put_entry(
            data,
            key=cache_key,
            metadata=metadata,
            catalog_schema=_CACHE_POLICY_SCHEMA,
            catalog_values=catalog_values,
        )
        maintenance = self._start_size_maintenance()
        return CachePutResult(receipt=receipt, maintenance=maintenance)

    @_clear_coordinated
    def put(
        self, data: Any, prefix: str = "", description: str = "", **kwargs: Any
    ) -> CachePutResult:
        """Commit one value and return receipt plus one bounded policy outcome."""

        return self._put_for_cache_key(
            data,
            cache_key=self._create_cache_key(kwargs),
            prefix=prefix,
            description=description,
            key_params=kwargs,
        )

    @_clear_read_coordinated
    def lookup_call(
        self,
        func: Callable,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        cache_key: str | None = None,
    ) -> CacheLookupResult:
        """Look up one normalized decorated call through the shared policy boundary."""

        if cache_key is None:
            _, cache_key = self._function_cache_key(func, args, kwargs)
        return self.lookup(cache_key=cache_key)

    @_clear_coordinated
    def put_call(
        self,
        func: Callable,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        value: Any,
        *,
        namespace: str | None = None,
        cache_key: str | None = None,
    ) -> CachePutResult:
        """Commit one decorated result with its authenticated function namespace."""

        if (namespace is None) != (cache_key is None):
            raise ValueError("function namespace and cache key must be supplied together")
        if namespace is None:
            namespace, cache_key = self._function_cache_key(func, args, kwargs)
        assert cache_key is not None
        return self._put_for_cache_key(
            value,
            cache_key=cache_key,
            prefix="",
            description=f"Cached result for {namespace}",
            key_params={
                "__function_namespace__": namespace,
                "__function_cache_key__": cache_key,
            },
            function_namespace=namespace,
        )

    def _record_outcome(self, outcome: CacheOutcome) -> None:
        """Best-effort record an already-final lookup outcome.

        Derived statistics must never alter canonical BlobStore state or turn a
        completed lookup into a failed read. ``RuntimeError`` is the narrow
        observer-unavailable signal used for a recorder close race; arbitrary
        programming errors still propagate from the lookup itself.
        """

        if not self.config.metadata.enable_cache_stats:
            return
        try:
            self._outcome_recorder.record(outcome)
        except RuntimeError:
            logger.debug("Cache statistics observer was unavailable", exc_info=True)

    @staticmethod
    def _outcome_for_storage_failure(error: BaseException) -> CacheOutcome | None:
        """Map declared direct-read failures into the public policy vocabulary."""

        category = classify_cache_read_failure(error)
        if category is CacheReadFailureCategory.INTEGRITY:
            return CacheOutcome.CORRUPT
        if category is CacheReadFailureCategory.LIFECYCLE_CONFLICT:
            return CacheOutcome.CONFLICT
        if category in {
            CacheReadFailureCategory.MANIFEST_UNSUPPORTED_VERSION,
            CacheReadFailureCategory.PAYLOAD_UNSUPPORTED_VERSION,
            CacheReadFailureCategory.UNSUPPORTED_VERSION,
            CacheReadFailureCategory.BACKEND_FAILURE,
            CacheReadFailureCategory.MIGRATION_REQUIRED,
        }:
            return CacheOutcome.BACKEND_ERROR
        if isinstance(
            error, (CacheBlobLifecycleTimeoutError, CacheBlobStoreClosedError)
        ):
            return CacheOutcome.BACKEND_ERROR
        return None

    @_clear_read_coordinated
    def statistics(self) -> CacheStatistics:
        """Return a frozen derived outcome snapshot without observing storage."""

        if not self.config.metadata.enable_cache_stats:
            return CacheStatistics()
        return self._outcome_recorder.snapshot()

    @_clear_read_coordinated
    def lookup(
        self,
        cache_key: Optional[str] = None,
        ttl_hours: object = _DEFAULT_TTL,
        prefix: str = "",
        **kwargs: Any,
    ) -> CacheLookupResult:
        """Observe one BlobStore snapshot and return its policy outcome."""

        del prefix  # Public policy identity is not a physical path prefix.
        if cache_key is None:
            cache_key = self._create_cache_key(kwargs)
        try:
            with self._cache_blob_store.open_entry(cache_key) as snapshot:
                if snapshot is None:
                    result = CacheLookupResult(CacheOutcome.ABSENT)
                else:
                    entry = self._policy_entry_from_snapshot(snapshot)
                    if self._is_expired(cache_key, ttl_hours, entry):
                        removal = self._remove_exact_candidates(
                            (self._candidate_from_snapshot(snapshot),)
                        )
                        result = CacheLookupResult(
                            CacheOutcome.EXPIRED, removal=removal
                        )
                    else:
                        result = CacheLookupResult(CacheOutcome.HIT, value=snapshot.read())
        except (
            CacheBlobIntegrityError,
            CacheBlobLifecycleConflictError,
            CacheBlobBackendError,
            CacheBlobManifestUnsupportedVersionError,
            CacheBlobPayloadUnsupportedVersionError,
            CacheManifestUnsupportedVersionError,
            CacheBlobMigrationRequiredError,
            CacheBlobLifecycleTimeoutError,
            CacheBlobStoreClosedError,
        ) as error:
            outcome = self._outcome_for_storage_failure(error)
            if outcome is None:
                raise
            result = CacheLookupResult(outcome, cause=error)
        self._record_outcome(result.outcome)
        return result

    @_clear_coordinated
    def invalidate(
        self, cache_key: Optional[str] = None, prefix: str = "", **kwargs: Any
    ) -> CacheRemovalReport:
        """Invalidate one observed cache generation if it is still current."""

        del prefix
        if cache_key is None:
            cache_key = self._create_cache_key(kwargs)
        snapshot = self._cache_blob_store.get_entry_info(cache_key)
        if snapshot is None:
            return CacheRemovalReport()
        return self._remove_exact_candidates((self._candidate_from_snapshot(snapshot),))

    @_clear_coordinated
    def invalidate_where(
        self,
        query: CatalogQuery,
        *,
        cursor: str | None = None,
        page_size: int = _REMOVAL_PAGE_SIZE,
        work_cap: int = _REMOVAL_WORK_CAP,
    ) -> CacheRemovalReport:
        """Remove one validated, bounded authoritative catalog page exactly."""

        # Catalog cursors are revision-bound.  A successful exact deletion from
        # this page advances that revision, so later work must begin a fresh
        # bounded scan instead of reusing the cursor that described this page.
        query_cursor = None if cursor == _RESTART_REMOVAL_CURSOR else cursor
        try:
            page = self._query_cache_catalog(
                query,
                cursor=query_cursor,
                page_size=page_size,
                work_cap=work_cap,
            )
        except CatalogStaleCursorError:
            if query_cursor is None:
                raise
            return CacheRemovalReport(
                retryable=1,
                complete=False,
                continuation=_RESTART_REMOVAL_CURSOR,
            )
        candidates = tuple(
            self._candidate_from_catalog_entry(entry) for entry in page.entries
        )
        report = self._remove_exact_candidates(
            candidates,
            complete=page.exhausted,
            continuation=page.cursor,
        )
        if report.removed and not page.exhausted:
            return replace(
                report,
                complete=False,
                continuation=_RESTART_REMOVAL_CURSOR,
            )
        return report

    @_clear_coordinated
    def invalidate_function(
        self,
        func: Callable,
        *,
        cursor: str | None = None,
        page_size: int = _REMOVAL_PAGE_SIZE,
        work_cap: int = _REMOVAL_WORK_CAP,
    ) -> CacheRemovalReport:
        """Remove one bounded page in a function's authoritative namespace."""

        namespace = self.function_namespace(func)
        return self.invalidate_where(
            CatalogQuery(
                predicates=(
                    CatalogPredicate("function_namespace", "eq", namespace),
                ),
                page_size=page_size,
            ),
            cursor=cursor,
            page_size=page_size,
            work_cap=work_cap,
        )

    @_clear_coordinated
    def clear_all(
        self,
        *,
        cursor: str | None = None,
        page_size: int = _REMOVAL_PAGE_SIZE,
        work_cap: int = _REMOVAL_WORK_CAP,
    ) -> CacheRemovalReport:
        """Remove one bounded global page through exact BlobStore deletion."""

        return self.invalidate_where(
            CatalogQuery(page_size=page_size),
            cursor=cursor,
            page_size=page_size,
            work_cap=work_cap,
        )

    def close(self) -> None:
        """Close only a cache-created store and then close this policy facade.

        BlobStore preserves its own typed close outcomes.  The facade neither
        enumerates topology participants nor turns a storage close failure
        into an aggregate cache lifecycle report.
        """

        if self._closed:
            return
        if self._owned_store is not None:
            self._owned_store.close()
        self._closed = True

    def __enter__(self) -> "UnifiedCache":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.close()
        return False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
