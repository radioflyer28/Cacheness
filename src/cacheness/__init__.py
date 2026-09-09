"""Canonical public surface for cache policy, direct storage, and SQL caching.

``UnifiedCache`` is a policy facade over a caller-selected ``BlobStore``
composition. Applications construct it with nested ``CacheConfig`` values,
explicitly initialize it, and use the one ``cached`` decorator when function
results need the same policy.

Direct object storage and SQL pull-through remain intentionally separate
surfaces; neither becomes an alternate cache lifecycle owner.
"""

from .cache_policy import (
    CacheLookupResult,
    CacheMaintenanceResult,
    CacheMaintenanceState,
    CacheOutcome,
    CachePutResult,
    CacheRemovalReport,
    CacheStatistics,
)
from .config import CacheConfig, CachePolicyConfig
from .core import UnifiedCache
from .decorators import cached
from .sql_cache import SqlCache, SqlCacheAdapter
from .storage import BlobStore, RoleRegistry, StoreTopology

__version__ = "0.3.14"
__author__ = "radioflyer28"
__email__ = "akgithub.2drwc@aleeas.com"


__all__ = [
    "UnifiedCache",
    "CacheConfig",
    "CachePolicyConfig",
    "cached",
    "CacheOutcome",
    "CacheLookupResult",
    "CacheStatistics",
    "CacheRemovalReport",
    "CacheMaintenanceState",
    "CacheMaintenanceResult",
    "CachePutResult",
    "BlobStore",
    "RoleRegistry",
    "StoreTopology",
    "SqlCache",
    "SqlCacheAdapter",
]
