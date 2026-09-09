"""Cache policy built on a single internal :class:`BlobStore` engine.

``UnifiedCache`` deliberately owns only cache policy: key derivation, TTL,
entry-size eviction, and hit/miss accounting. Payload publication, catalog
membership, recovery, and resource ownership belong to its composed
``BlobStore``. The facade does not expose or maintain a second metadata
authority.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

from .cache_policy import (
    CacheLookupResult,
    CacheOutcome,
    CacheRemovalReport,
    CacheStatistics,
    _CacheOutcomeRecorder,
    _CacheRemovalCandidate,
    execute_exact_removals,
)
from .config import CacheConfig, _DEFAULT_TTL, create_cache_config
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
        bound = inspect.signature(func).bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except (TypeError, ValueError):
        return {**{f"__arg_{index}": value for index, value in enumerate(args)}, **kwargs}


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
        self.config = config
        self.cache_dir = Path(self.config.storage.cache_dir)
        self.handlers = HandlerRegistry(self.config)
        self._closed = False
        self._outcome_recorder = _CacheOutcomeRecorder()
        self._owns_store = isinstance(store, StoreTopology)
        if isinstance(store, BlobStore):
            self.store = store
        elif isinstance(store, StoreTopology):
            root = self.cache_dir / ".cacheness" / "blobstore"
            self.store = BlobStore(store, cache_dir=root, config=self.config)
        else:
            raise TypeError("store must be a BlobStore or StoreTopology")
        self._cache_blob_store = self.store
        self._cache_blob_store.handlers = self.handlers
        self.actual_backend = "-".join(
            self._cache_blob_store.topology.qualified_profile.pair
        )
        logger.info(
            "Unified cache initialized at %s using BlobStore topology %s",
            self.cache_dir,
            self.actual_backend,
        )

    def initialize(self) -> None:
        """Initialize the internal store before sharing this cache with workers."""

        if self._closed:
            raise CacheBlobStoreClosedError("Cache is closed")
        self._cache_blob_store.initialize()

    def _create_cache_key(self, params: Mapping[str, Any]) -> str:
        """Return the deterministic public cache identity for ``params``."""

        return create_unified_cache_key(dict(params), self.config)

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
            self.config.metadata.default_ttl_hours
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

        page = self._query_cache_catalog(
            CatalogQuery(page_size=page_size),
            cursor=cursor,
            page_size=page_size,
            work_cap=work_cap,
        )
        ttl = self.config.metadata.default_ttl_hours
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

    def _enforce_size_limit(self) -> None:
        """Keep policy size eviction above storage without a second catalog."""

        max_size_mb = self.config.storage.max_cache_size_mb
        if max_size_mb is None:
            return
        candidates: list[tuple[str, str, Any, int]] = []
        total_size = 0
        for cache_key in self._cache_blob_store.list():
            try:
                snapshot, entry = self._authority_snapshot_entry(cache_key)
            except CacheBlobLifecycleConflictError:
                continue
            if snapshot is None or entry is None:
                continue
            size = entry["file_size"] if entry["file_size"] > 0 else 0
            total_size += size
            candidates.append(
                (str(entry.get("created_at") or ""), cache_key, snapshot, size)
            )
        limit = int(max_size_mb * 1024 * 1024)
        if total_size <= limit:
            return
        target = int(limit * 0.8)
        removed = 0
        for _, cache_key, snapshot, size in sorted(candidates):
            if total_size <= target:
                break
            if self._retire_exact_authority_snapshot(cache_key, snapshot):
                total_size -= size
                removed += 1
        if removed:
            logger.info("Cache size policy evicted %s entries", removed)

    @_clear_coordinated
    def put(
        self, data: Any, prefix: str = "", description: str = "", **kwargs: Any
    ) -> str:
        """Store data and return its cache-policy key.

        BlobStore commits the payload and catalog. A subsequent eviction error
        is policy debt and cannot revoke that committed generation.
        """

        cache_key = self._create_cache_key(kwargs)
        metadata: dict[str, Any] = {"prefix": prefix, "description": description}
        if self.config.metadata.store_cache_key_params:
            metadata["cache_key_params"] = self._canonical_cache_key_params(kwargs)
        self._cache_blob_store.handlers = self.handlers
        self._cache_blob_store.put_entry(
            data,
            key=cache_key,
            metadata=metadata,
            catalog_schema=_CACHE_POLICY_SCHEMA,
            catalog_values={
                "cache_namespace": _CACHE_NAMESPACE,
                "cache_prefix": prefix,
            },
        )
        if not self._closed:
            try:
                self._enforce_size_limit()
            except Exception as error:  # policy follow-up never rolls back a commit
                logger.warning("Blob committed; cache size policy remains pending: %s", error)
        return cache_key

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
        self._cache_blob_store.handlers = self.handlers
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

        try:
            page = self._query_cache_catalog(
                query,
                cursor=cursor,
                page_size=page_size,
                work_cap=work_cap,
            )
        except CatalogStaleCursorError:
            if cursor is None:
                raise
            return CacheRemovalReport(
                retryable=1,
                complete=False,
                continuation=cursor,
            )
        candidates = tuple(
            self._candidate_from_catalog_entry(entry) for entry in page.entries
        )
        return self._remove_exact_candidates(
            candidates,
            complete=page.exhausted,
            continuation=page.cursor,
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

    @_clear_read_coordinated
    def list_entries(self) -> list[dict[str, Any]]:
        """List cache-policy views rendered from canonical BlobStore snapshots."""

        entries: list[dict[str, Any]] = []
        for cache_key in self._cache_blob_store.list():
            snapshot, entry = self._authority_snapshot_entry(cache_key)
            if snapshot is None or entry is None:
                continue
            entries.append(
                {
                    "cache_key": cache_key,
                    "data_type": entry["data_type"],
                    "description": entry["description"],
                    "metadata": entry["metadata"],
                    "created": entry["created_at"],
                    "last_accessed": entry["created_at"],
                    "size_mb": round(entry["file_size"] / (1024 * 1024), 3),
                    "expired": self._is_expired(cache_key, entry=entry),
                }
            )
        return entries

    @_clear_read_coordinated
    def get_stats(self) -> dict[str, Any]:
        """Return cache policy counters plus canonical BlobStore inventory."""

        entries = self.list_entries()
        statistics = self.statistics()
        return {
            "cache_hits": statistics.hit,
            "cache_misses": statistics.misses,
            "total_entries": len(entries),
            "dataframe_entries": sum(
                entry["data_type"] == "dataframe" for entry in entries
            ),
            "array_entries": sum(entry["data_type"] == "array" for entry in entries),
            "total_size_mb": round(sum(entry["size_mb"] for entry in entries), 2),
            "hit_rate": statistics.hit_rate,
            "cache_dir": str(self.cache_dir),
            "max_size_mb": self.config.storage.max_cache_size_mb,
            "default_ttl_hours": self.config.metadata.default_ttl_hours,
            "backend_type": self.actual_backend,
        }

    def close(self) -> None:
        """Close the composed store exactly once."""

        if self._closed:
            return
        self._closed = True
        if self._owns_store:
            self._cache_blob_store.close()

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

    @classmethod
    def for_api(
        cls,
        cache_dir: Optional[str] = None,
        ttl_hours: int = 6,
        ignore_errors: bool = True,
        **kwargs: Any,
    ) -> "UnifiedCache":
        """Create a cache with API-oriented TTL and compression defaults."""

        del ignore_errors
        config = create_cache_config(
            cache_dir=cache_dir or "./cache",
            default_ttl_hours=ttl_hours,
            pickle_compression_codec="zstd",
            pickle_compression_level=3,
            **kwargs,
        )
        return cls(config)


_global_cache: Optional[UnifiedCache] = None


def get_cache(config: Optional[CacheConfig] = None) -> UnifiedCache:
    """Return the process-global policy facade, creating it once."""

    global _global_cache
    if _global_cache is None:
        _global_cache = UnifiedCache(config)
    return _global_cache


def reset_cache(config: Optional[CacheConfig] = None) -> UnifiedCache:
    """Replace and close the process-global policy facade."""

    global _global_cache
    if _global_cache is not None:
        _global_cache.close()
    _global_cache = UnifiedCache(config)
    return _global_cache
