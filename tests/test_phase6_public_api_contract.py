"""Public-contract tracer for the canonical Phase 6 cache surface."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

import cacheness
from cacheness import (
    BlobStore,
    CacheConfig,
    CacheLookupResult,
    CacheMaintenanceResult,
    CacheMaintenanceState,
    CacheOutcome,
    CachePolicyConfig,
    CachePutResult,
    CacheRemovalReport,
    CacheStatistics,
    RoleRegistry,
    SqlCache,
    SqlCacheAdapter,
    StoreTopology,
    UnifiedCache,
    cached,
)
from cacheness.config import CacheStorageConfig
from cacheness.storage.composition import BackendRef


CANONICAL_PUBLIC_NAMES = (
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
)


def _memory_topology() -> StoreTopology:
    """Build the supported same-process topology for a public-cache tracer."""

    return StoreTopology(
        payload=BackendRef(name="memory"),
        authority=BackendRef(name="memory"),
    )


def test_canonical_imports_run_one_explicit_cache_lifecycle(tmp_path) -> None:
    """One documented import/config/result surface can complete a cache lifecycle."""

    assert tuple(cacheness.__all__) == CANONICAL_PUBLIC_NAMES
    namespace: dict[str, object] = {}
    exec("from cacheness import *", namespace)
    assert tuple(name for name in CANONICAL_PUBLIC_NAMES if name in namespace) == (
        CANONICAL_PUBLIC_NAMES
    )

    config = CacheConfig(
        storage=CacheStorageConfig(cache_dir=str(tmp_path / "cache")),
        policy=CachePolicyConfig(max_authoritative_bytes=1024 * 1024),
    )
    cache = UnifiedCache(config, store=_memory_topology())
    try:
        cache.initialize()
        written = cache.put({"surface": "canonical"}, request_id="tracer")
        lookup = cache.lookup(cache_key=written.receipt.key)
        statistics = cache.statistics()

        assert isinstance(written, CachePutResult)
        assert isinstance(lookup, CacheLookupResult)
        assert lookup.outcome is CacheOutcome.HIT
        assert lookup.value == {"surface": "canonical"}
        assert isinstance(statistics, CacheStatistics)
        assert statistics.hit == 1
    finally:
        cache.close()


def test_cache_requires_an_explicit_blobstore_composition_before_io(tmp_path) -> None:
    """An omitted composition is rejected before the cache can initialize storage."""

    config = CacheConfig(storage=CacheStorageConfig(cache_dir=str(tmp_path / "cache")))

    with pytest.raises(TypeError, match="store"):
        UnifiedCache(config)  # type: ignore[call-arg]


def test_optional_yaml_capability_fails_only_when_requested() -> None:
    """Blocking PyYAML cannot prevent base import or change the canonical surface."""

    script = textwrap.dedent(
        f"""
        import builtins

        original_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "yaml" or name.startswith("yaml."):
                raise ImportError("blocked optional dependency: yaml")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = blocked_import
        import cacheness
        from cacheness.config import load_config_from_yaml

        assert tuple(cacheness.__all__) == {CANONICAL_PUBLIC_NAMES!r}
        try:
            load_config_from_yaml("unused.yaml")
        except ImportError as error:
            assert "install" in str(error).lower()
        else:
            raise AssertionError("YAML loading unexpectedly succeeded")
        """
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(part for part in sys.path if part)

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr


def test_import_order_does_not_change_public_surface_or_store_identity(tmp_path) -> None:
    """Optional storage import order cannot select a cache identity or alter exports."""

    import cacheness.storage  # noqa: F401 - deliberate order probe

    assert tuple(cacheness.__all__) == CANONICAL_PUBLIC_NAMES
    cache = UnifiedCache(
        CacheConfig(storage=CacheStorageConfig(cache_dir=str(tmp_path / "cache"))),
        store=_memory_topology(),
    )
    try:
        assert cache.store is cache._cache_blob_store
    finally:
        cache.close()


def test_sql_cache_remains_a_separate_supported_surface() -> None:
    """SQL pull-through remains separately importable, never a UnifiedCache route."""

    assert SqlCache.__module__ == "cacheness.sql_cache"
    assert SqlCacheAdapter.__module__ == "cacheness.sql_cache"
    assert not issubclass(SqlCache, UnifiedCache)
    assert cached.__module__ == "cacheness.decorators"
    assert BlobStore.__module__ == "cacheness.storage.blob_store"
    assert RoleRegistry.__module__ == "cacheness.storage.composition"
