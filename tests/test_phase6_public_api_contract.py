"""Public-contract tracer for the canonical Phase 6 cache surface."""

from __future__ import annotations

import inspect
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
from cacheness.config import (
    CacheMetadataConfig,
    CacheStorageConfig,
    SecurityConfig,
    load_config_from_json,
    load_config_from_yaml,
    save_config_to_json,
    save_config_to_yaml,
)
from cacheness.error_handling import CacheMigrationOrRebuildRequiredError
from cacheness.storage.composition import BackendRef
from cacheness.storage.manifest import StoreVersionDimensions


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
    assert namespace["CacheRemovalReport"] is CacheRemovalReport
    assert namespace["CacheMaintenanceState"] is CacheMaintenanceState
    assert namespace["CacheMaintenanceResult"] is CacheMaintenanceResult

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


def test_config_requires_an_explicit_blobstore_composition_before_io(tmp_path) -> None:
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


@pytest.mark.parametrize(
    ("suffix", "save_config", "load_config"),
    (
        ("json", save_config_to_json, load_config_from_json),
        ("yaml", save_config_to_yaml, load_config_from_yaml),
    ),
)
def test_config_round_trip_preserves_every_cache_policy_setting(
    tmp_path, suffix, save_config, load_config
) -> None:
    """Persisted policy settings retain their cache semantics after loading."""

    if suffix == "yaml":
        pytest.importorskip("yaml")
    config = CacheConfig(
        storage=CacheStorageConfig(cache_dir=str(tmp_path / "cache")),
        policy=CachePolicyConfig(
            default_ttl_hours=None,
            max_authoritative_bytes=7,
            catalog_page_size=3,
            maintenance_work_cap=3,
            max_maintenance_state_bytes=512,
        ),
    )
    path = tmp_path / f"cache-config.{suffix}"

    save_config(config, path)

    assert load_config(path).policy == config.policy


@pytest.mark.parametrize(
    "first_import",
    (
        "import cacheness",
        "import cacheness.storage\nimport cacheness",
    ),
)
def test_ordering_does_not_change_public_surface_or_store_identity(
    first_import: str,
) -> None:
    """Optional storage import order cannot select a cache identity or alter exports."""

    script = (
        f"{first_import}\n"
        "from cacheness import CacheConfig, StoreTopology, UnifiedCache\n"
        "from cacheness.storage.composition import BackendRef\n"
        f"assert tuple(cacheness.__all__) == {CANONICAL_PUBLIC_NAMES!r}\n"
        "cache = UnifiedCache(\n"
        "    CacheConfig(),\n"
        "    store=StoreTopology(\n"
        "        payload=BackendRef(name='memory'),\n"
        "        authority=BackendRef(name='memory'),\n"
        "    ),\n"
        ")\n"
        "try:\n"
        "    assert cache.store is cache._cache_blob_store\n"
        "finally:\n"
        "    cache.close()\n"
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


def test_sqlcache_remains_a_separate_supported_surface() -> None:
    """SQL pull-through remains separately importable, never a UnifiedCache route."""

    assert SqlCache.__module__ == "cacheness.sql_cache"
    assert SqlCacheAdapter.__module__ == "cacheness.sql_cache"
    assert not issubclass(SqlCache, UnifiedCache)
    assert cached.__module__ == "cacheness.decorators"
    assert BlobStore.__module__ == "cacheness.storage.blob_store"
    assert RoleRegistry.__module__ == "cacheness.storage.composition"


@pytest.mark.parametrize(
    "name",
    (
        "cacheness",
        "get_cache",
        "reset_cache",
        "create_cache_config",
        "cache_function",
        "memoize",
        "CacheContext",
    ),
)
def test_removed_development_public_names_do_not_delegate(name: str) -> None:
    """The cutover removes overlapping owners rather than preserving adapters."""

    assert name not in cacheness.__all__
    assert not hasattr(cacheness, name)


def test_removed_cache_compatibility_methods_and_flat_config_are_absent() -> None:
    """The canonical lifecycle uses typed results and nested configuration only."""

    assert not hasattr(UnifiedCache, "for_api")
    assert not hasattr(UnifiedCache, "get")
    assert not hasattr(UnifiedCache, "get_stats")
    assert not hasattr(UnifiedCache, "list_entries")

    parameters = inspect.signature(CacheConfig).parameters
    for name in (
        "cache_dir",
        "default_ttl_hours",
        "metadata_backend",
        "blob_backend",
        "hash_path_content",
    ):
        assert name not in parameters

    config = CacheConfig()
    for name in ("cache_dir", "default_ttl_hours", "blob_backend"):
        assert not hasattr(config, name)

    with pytest.raises(TypeError):
        CacheConfig(cache_dir="deprecated")


def test_cache_policy_limits_have_no_competing_nested_configuration() -> None:
    """TTL and size limits are accepted only by the policy configuration."""

    assert "max_cache_size_mb" not in inspect.signature(CacheStorageConfig).parameters
    assert "default_ttl_hours" not in inspect.signature(CacheMetadataConfig).parameters

    with pytest.raises(TypeError):
        CacheStorageConfig(max_cache_size_mb=1)
    with pytest.raises(TypeError):
        CacheMetadataConfig(default_ttl_hours=1)

    policy = CachePolicyConfig(default_ttl_hours=1, max_authoritative_bytes=1)
    assert policy.default_ttl_hours == 1
    assert policy.max_authoritative_bytes == 1


@pytest.mark.parametrize(
    ("config_type", "retired_name"),
    (
        (CacheStorageConfig, "create_cache_dir"),
        (CacheStorageConfig, "temp_dir"),
        (CacheMetadataConfig, "enable_metadata"),
        (CacheMetadataConfig, "enable_memory_cache"),
        (CacheMetadataConfig, "memory_cache_type"),
        (CacheMetadataConfig, "memory_cache_maxsize"),
        (CacheMetadataConfig, "memory_cache_ttl_seconds"),
        (CacheMetadataConfig, "memory_cache_stats"),
        (SecurityConfig, "signing_key_file"),
        (SecurityConfig, "use_in_memory_key"),
        (SecurityConfig, "signature_version"),
        (SecurityConfig, "delete_invalid_signatures"),
    ),
)
def test_runtime_inert_configuration_options_are_removed(
    config_type: type, retired_name: str
) -> None:
    """The pre-production cutover rejects configuration with no runtime owner."""

    assert retired_name not in inspect.signature(config_type).parameters
    with pytest.raises(TypeError):
        config_type(**{retired_name: object()})


def test_explicit_version_dimensions_reject_unsupported_store_layouts() -> None:
    """Current records retain version facts and refuse implicit layout upgrades."""

    current = StoreVersionDimensions()
    assert set(current.to_mapping()) == {
        "manifest_schema_version",
        "payload_format_version",
        "sqlite_user_version",
        "store_epoch",
        "store_format_version",
    }

    with pytest.raises(CacheMigrationOrRebuildRequiredError, match="offline"):
        StoreVersionDimensions(store_format_version=current.store_format_version + 1)
