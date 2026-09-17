"""Executable public contract for the explicit Phase 6 cache surface."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

import cacheness
from cacheness import CacheConfig, CacheLookupResult, CacheOutcome, CachePutResult
from cacheness import StoreTopology, UnifiedCache
from cacheness import error_handling
from cacheness.config import CacheStorageConfig
from cacheness.storage.composition import BackendRef


def _memory_topology() -> StoreTopology:
    """Build the documented same-process topology for public API checks."""

    return StoreTopology(
        payload=BackendRef(name="memory"),
        authority=BackendRef(name="memory"),
    )


class TestPublicExports:
    """Freeze canonical exports and explicitly reject retired cache routes."""

    def test_star_import_exposes_only_declared_canonical_names(self):
        namespace: dict[str, object] = {}
        exec("from cacheness import *", namespace)

        assert set(cacheness.__all__).issubset(namespace)
        for retired in (
            "SQLAlchemyDataAdapter",
            "SQLAlchemyPullThroughCache",
            "SQLAlchemySqlCacheAdapter",
            "cacheness",
            "get_cache",
            "reset_cache",
        ):
            assert retired not in cacheness.__all__
            assert not hasattr(cacheness, retired)

    def test_role_registry_is_the_only_public_blob_selection_surface(self):
        """Public composition retains roles, not retired backend registries."""

        retired = {
            "create_metadata_backend",
            "register_metadata_backend",
            "unregister_metadata_backend",
            "register_blob_backend",
            "unregister_blob_backend",
            "get_blob_backend",
            "list_blob_backends",
            "BlobBackend",
            "FilesystemBlobBackend",
            "InMemoryBlobBackend",
        }

        import cacheness.storage as storage

        assert retired.isdisjoint(cacheness.__all__)
        assert all(not hasattr(cacheness, name) for name in retired)
        assert {"RoleRegistry", "BackendRole", "BackendRef", "StoreTopology"}.issubset(
            storage.__all__
        )
        assert cacheness.RoleRegistry is storage.RoleRegistry
        assert cacheness.StoreTopology is storage.StoreTopology
        assert storage.BackendRef(name="memory").name == "memory"

    def test_storage_handler_protocol_is_alias_free(self):
        """The generic extension protocol is storage-oriented and unambiguous."""
        import cacheness.storage as storage

        assert {"FormatHandler", "FormatHandlerError"}.issubset(storage.__all__)
        assert hasattr(storage, "FormatHandler")
        assert hasattr(storage, "FormatHandlerError")
        assert not hasattr(storage, "CacheHandler")
        assert not hasattr(storage, "CacheHandlerError")

    def test_format_handler_error_has_one_public_identity(self):
        """All handler failures share the cross-cutting CacheError hierarchy."""
        import cacheness.storage as storage
        from cacheness.error_handling import CacheError, FormatHandlerError
        from cacheness.interfaces import CacheReadError, CacheWriteError

        assert storage.FormatHandlerError is FormatHandlerError
        assert issubclass(CacheWriteError, FormatHandlerError)
        assert issubclass(CacheReadError, FormatHandlerError)
        assert issubclass(FormatHandlerError, CacheError)

    def test_explicit_constructor_returns_typed_results(self, tmp_path):
        """The package surface exposes one explicit cache lifecycle."""

        cache = UnifiedCache(
            CacheConfig(storage=CacheStorageConfig(cache_dir=tmp_path / "cache")),
            store=_memory_topology(),
        )
        try:
            cache.initialize()
            written = cache.put({"surface": "canonical"}, request_id="public")
            lookup = cache.lookup(cache_key=written.receipt.key)

            assert isinstance(written, CachePutResult)
            assert isinstance(lookup, CacheLookupResult)
            assert lookup.outcome is CacheOutcome.HIT
            assert lookup.value == {"surface": "canonical"}
        finally:
            cache.close()

    def test_removed_constructor_and_result_compatibility_routes_remain_absent(self):
        """No public adapter revives flat config or raw get/put semantics."""

        assert "cache_dir" not in inspect.signature(CacheConfig).parameters
        for name in ("for_api", "get", "get_stats", "list_entries"):
            assert not hasattr(UnifiedCache, name)

        with pytest.raises(TypeError):
            CacheConfig(cache_dir="deprecated")
        with pytest.raises(TypeError, match="store"):
            UnifiedCache(CacheConfig())  # type: ignore[call-arg]

    def test_public_exception_inheritance_and_reason_values(self):
        """Typed errors retain the documented fail-closed cache vocabulary."""

        expected_reasons = {
            "path_traversal",
            "path_absolute",
            "path_drive",
            "path_unc",
            "path_rooted",
            "path_outside_root",
            "path_race",
            "invalid_identifier",
            "invalid_query_field",
            "invalid_query_value",
            "invalid_legacy_array",
            "unsafe_object_array",
            "unsupported_legacy_layout",
            "read_only_legacy_store",
            "invalid_legacy_signature",
            "manifest_invalid",
            "manifest_bounds",
            "manifest_unsupported_version",
            "manifest_signature_invalid",
            "manifest_signing_key_invalid",
            "blob_lifecycle_conflict",
            "blob_backend_failure",
            "blob_backend_capability_unsupported",
            "blob_migration_cleanup_required",
            "blob_migration_confirmation_required",
            "blob_migration_evidence_invalid",
            "blob_migration_evidence_mismatch",
            "blob_migration_offline_decision_required",
            "blob_migration_plan_stale",
            "blob_migration_required",
            "blob_recoverable_cleanup",
            "blob_committed_partial",
            "blob_reconciliation_blocked",
            "blob_reconciliation_conflict",
            "blob_reconciliation_checkpoint_invalid",
            "blob_lock_release_failure",
            "blob_store_closed",
            "blob_close_timeout",
            "blob_lifecycle_timeout",
            "metadata_corrupt",
            "catalog_validation_failed",
            "catalog_query_invalid",
            "catalog_cursor_invalid",
            "catalog_cursor_stale",
            "migration_or_rebuild_required",
        }

        assert {reason.value for reason in error_handling.CacheReason} == expected_reasons
        assert issubclass(
            error_handling.CacheUnsafePathError, error_handling.CacheStorageError
        )
        assert issubclass(
            error_handling.CacheQueryValidationError,
            error_handling.CacheMetadataError,
        )


def test_retired_sqlcache_surface_is_naturally_absent_when_dependency_is_blocked():
    """Optional imports stay isolated without retaining the retired product."""

    script = textwrap.dedent(
        """
        import builtins

        original_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "sqlalchemy" or name.startswith("sqlalchemy."):
                raise ImportError("blocked optional dependency: sqlalchemy")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = blocked_import
        import cacheness
        import importlib

        assert "SqlCache" not in cacheness.__all__
        assert "SqlCacheAdapter" not in cacheness.__all__
        assert not hasattr(cacheness, "SqlCache")
        assert not hasattr(cacheness, "SqlCacheAdapter")
        try:
            exec("from cacheness import SqlCache", {})
        except ImportError:
            pass
        else:
            raise AssertionError("SqlCache import remained available")
        try:
            importlib.import_module("cacheness.sql_cache")
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError("cacheness.sql_cache module remained available")
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


def test_public_storage_docs_keep_the_d16_transport_boundary_explicit():
    """Published guidance retains one provider/handler seam and Phase 8 limits."""

    repository_root = Path(__file__).parents[1]
    documented = {
        "README.md": ("release qualification guide",),
        "docs/API_REFERENCE.md": (
            "BlobStore",
            "CacheOutcome",
            "Release qualification",
        ),
        "docs/PLUGIN_DEVELOPMENT.md": (
            "store.handlers.register_handler",
            "private staging",
            "persisted payload identities",
        ),
        "docs/SECURITY.md": (
            "ExpectedBucketOwner",
            "stable bucket",
            "name whose ownership",
            "bucket policy",
            "Phase 8",
        ),
        "docs/RELEASE_QUALIFICATION.md": (
            "128 MiB",
            "opaque corroborating transport evidence",
            "Phase 8",
        ),
    }
    retired_payload_apis = ("S3BlobBackend", "BlobBackend", "register_blob_backend")

    for relative_path, required_terms in documented.items():
        text = " ".join(
            (repository_root / relative_path).read_text(encoding="utf-8").split()
        )
        assert all(term in text for term in required_terms)
        assert all(retired not in text for retired in retired_payload_apis)

    plugin_source = (repository_root / "docs/PLUGIN_DEVELOPMENT.md").read_text(
        encoding="utf-8"
    )
    assert '"actual_path": str(' in plugin_source
    assert '"file_path": str(' not in plugin_source


def test_api_reference_imports_are_current_barrel_exports() -> None:
    """The focused reference names only imports consumers can use today."""

    repository_root = Path(__file__).parents[1]
    source = (repository_root / "docs" / "API_REFERENCE.md").read_text(
        encoding="utf-8"
    )

    assert "## Current public imports" in source
    assert "from cacheness import CacheConfig, UnifiedCache, cached" in source
    assert "from cacheness.storage import (" in source

    top_level = (
        "BlobStore",
        "CacheConfig",
        "CacheLookupResult",
        "CacheOutcome",
        "CachePutResult",
        "StoreTopology",
        "UnifiedCache",
        "cached",
    )
    storage = (
        "BackendRef",
        "BlobEntry",
        "BlobReceipt",
        "CatalogQuery",
        "CatalogSchema",
        "FormatHandler",
        "FormatHandlerError",
        "HandlerRegistry",
        "MigrationPlan",
        "OfflineMigrationService",
    )

    import cacheness.storage as storage_module

    assert all(hasattr(cacheness, name) for name in top_level)
    assert all(hasattr(storage_module, name) for name in storage)
    assert all(name in source for name in (*top_level, *storage))
    assert (
        "Cacheness no longer ships SqlCache or a range-aware SQL pull-through cache."
        in source
    )
    assert "CacheHandler" not in source
