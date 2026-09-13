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
            "sql_cache_fetch_failed",
            "sql_cache_gap_detection_failed",
            "sql_cache_upsert_failed",
            "missing_optional_dependency",
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


def test_optional_sqlcache_surface_remains_separate_when_dependency_is_blocked():
    """Optional SQL support fails at construction without changing cache exports."""

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
        from cacheness import SqlCache, SqlCacheAdapter

        assert SqlCache is cacheness.SqlCache
        assert SqlCacheAdapter is cacheness.SqlCacheAdapter
        try:
            SqlCache("sqlite:///:memory:", None, None)
        except Exception as error:
            assert "install" in str(error).lower()
        else:
            raise AssertionError("SQL cache construction unexpectedly succeeded")
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
        "README.md": ("128 MiB", "opaque transport evidence", "Phase 8"),
        "docs/API_REFERENCE.md": ("owner pinning", "SHA-256", "Phase 8"),
        "docs/PLUGIN_DEVELOPMENT.md": (
            "store.handlers.register_handler",
            "custom endpoint",
            "Phase 8",
        ),
        "docs/SECURITY.md": (
            "ExpectedBucketOwner",
            "stable bucket",
            "name whose ownership",
            "bucket policy",
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
