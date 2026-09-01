"""Executable compatibility contract for the supported package facade."""

import inspect
import os
import subprocess
import sys
import textwrap

import pytest

import cacheness
from cacheness import error_handling


class TestPublicExports:
    """Freeze documented exports, aliases, and import semantics."""

    def test_star_import_exposes_every_declared_public_name(self):
        namespace = {}
        exec("from cacheness import *", namespace)

        assert set(cacheness.__all__).issubset(namespace)
        assert namespace["SQLAlchemyDataAdapter"] is namespace[
            "SQLAlchemySqlCacheAdapter"
        ]
        assert namespace["SQLAlchemyPullThroughCache"] is namespace["SqlCache"]

    def test_documented_constructor_decorator_and_registry_signatures(self):
        assert "cache_dir" in inspect.signature(cacheness.CacheConfig).parameters
        assert "handler" in inspect.signature(cacheness.register_handler).parameters
        assert "handler_name" in inspect.signature(cacheness.unregister_handler).parameters
        assert callable(cacheness.cached)
        assert isinstance(cacheness.list_handlers(), list)

    def test_public_configuration_result_shapes(self):
        config = cacheness.CacheConfig()

        assert isinstance(config.storage.cache_dir, str)
        assert cacheness.create_cache_config().storage

    def test_public_exception_inheritance_and_reason_values(self):
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
            "blob_migration_required",
            "blob_recoverable_cleanup",
            "blob_reconciliation_blocked",
            "blob_reconciliation_conflict",
            "blob_reconciliation_checkpoint_invalid",
            "blob_lock_release_failure",
            "blob_store_closed",
            "blob_close_timeout",
        }

        assert {reason.value for reason in error_handling.CacheReason} == expected_reasons
        assert issubclass(
            error_handling.CacheUnsafePathError, error_handling.CacheStorageError
        )
        assert issubclass(
            error_handling.CacheQueryValidationError,
            error_handling.CacheMetadataError,
        )
        assert issubclass(
            error_handling.CacheLegacyFormatError,
            error_handling.CacheSerializationError,
        )

        error = error_handling.CacheUnsafePathError(
            "unsafe path", reason=error_handling.CacheReason.PATH_TRAVERSAL
        )
        assert error.context["reason"] == "path_traversal"


@pytest.mark.parametrize("blocked_module", ["yaml", "sqlalchemy", "pandas"])
def test_optional_public_names_remain_importable_when_dependency_is_blocked(
    blocked_module,
):
    """Optional dependencies fail at use time rather than hiding package names."""
    script = textwrap.dedent(
        f"""
        import builtins

        blocked_module = {blocked_module!r}
        original_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == blocked_module or name.startswith(blocked_module + "."):
                raise ImportError(f"blocked optional dependency: {{blocked_module}}")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = blocked_import
        import cacheness
        from cacheness import (
            SQLAlchemyDataAdapter,
            SQLAlchemyPullThroughCache,
            SQLAlchemySqlCacheAdapter,
            SqlCache,
            SqlCacheAdapter,
            load_config_from_yaml,
            save_config_to_yaml,
        )

        assert SQLAlchemyDataAdapter is SqlCacheAdapter
        assert SQLAlchemySqlCacheAdapter is SqlCacheAdapter
        assert SQLAlchemyPullThroughCache is SqlCache
        assert callable(load_config_from_yaml)
        assert callable(save_config_to_yaml)

        if blocked_module == "yaml":
            try:
                save_config_to_yaml(None, "unused.yaml")
            except ImportError as error:
                assert "install" in str(error).lower()
            else:
                raise AssertionError("YAML use unexpectedly succeeded")
        else:
            try:
                SqlCache("sqlite:///:memory:", None, None)
            except Exception as error:
                assert "install" in str(error).lower()
            else:
                raise AssertionError("SQL cache construction unexpectedly succeeded")
        """
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in sys.path if part
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
