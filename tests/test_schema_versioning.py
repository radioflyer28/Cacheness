"""Tests for schema versioning and namespace registry on MetadataBackend."""

import pytest
from datetime import datetime, timezone

from cacheness.metadata import (
    MetadataBackend,
    NamespaceInfo,
    validate_namespace_id,
    DEFAULT_NAMESPACE,
    NAMESPACE_ID_PATTERN,
    Migration,
)


class TestValidateNamespaceId:
    """Tests for validate_namespace_id()."""

    def test_valid_simple(self):
        assert validate_namespace_id("default") == "default"

    def test_valid_underscore(self):
        assert validate_namespace_id("my_project") == "my_project"

    def test_valid_numeric(self):
        assert validate_namespace_id("project123") == "project123"

    def test_valid_all_digits(self):
        assert validate_namespace_id("42") == "42"

    def test_valid_max_length(self):
        ns_id = "a" * 48
        assert validate_namespace_id(ns_id) == ns_id

    def test_invalid_too_long(self):
        with pytest.raises(ValueError, match="must match"):
            validate_namespace_id("a" * 49)

    def test_invalid_empty(self):
        with pytest.raises(ValueError, match="must match"):
            validate_namespace_id("")

    def test_invalid_uppercase(self):
        with pytest.raises(ValueError, match="must match"):
            validate_namespace_id("MyProject")

    def test_invalid_hyphen(self):
        with pytest.raises(ValueError, match="must match"):
            validate_namespace_id("my-project")

    def test_invalid_space(self):
        with pytest.raises(ValueError, match="must match"):
            validate_namespace_id("my project")

    def test_invalid_dot(self):
        with pytest.raises(ValueError, match="must match"):
            validate_namespace_id("my.project")

    def test_invalid_type_int(self):
        with pytest.raises(ValueError, match="must be a string"):
            validate_namespace_id(123)  # type: ignore[arg-type]

    def test_invalid_type_none(self):
        with pytest.raises(ValueError, match="must be a string"):
            validate_namespace_id(None)  # type: ignore[arg-type]

    def test_invalid_special_chars(self):
        with pytest.raises(ValueError, match="must match"):
            validate_namespace_id("drop;table")


class TestNamespaceInfo:
    """Tests for NamespaceInfo dataclass."""

    def test_defaults(self):
        info = NamespaceInfo(namespace_id="test")
        assert info.namespace_id == "test"
        assert info.display_name == ""
        assert info.schema_version == 1
        assert isinstance(info.created_at, datetime)
        assert info.signature is None

    def test_custom_fields(self):
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        info = NamespaceInfo(
            namespace_id="prod",
            display_name="Production Cache",
            schema_version=2,
            created_at=ts,
            signature="abc123",
        )
        assert info.namespace_id == "prod"
        assert info.display_name == "Production Cache"
        assert info.schema_version == 2
        assert info.created_at == ts
        assert info.signature == "abc123"


class TestMetadataBackendSchemaVersioning:
    """Tests for schema versioning on MetadataBackend ABC."""

    class DummyBackend(MetadataBackend):
        """Minimal concrete backend for testing ABC methods."""

        def __init__(self):
            self._versions: dict[str, int] = {}
            self._migration_log: list[tuple[str, int, int]] = []

        # Implement all abstract methods as stubs
        def load_metadata(self):
            return {"entries": {}}

        def save_metadata(self, metadata):
            pass

        def get_entry(self, cache_key):
            return None

        def put_entry(self, cache_key, entry_data):
            pass

        def remove_entry(self, cache_key):
            return False

        def update_entry_metadata(self, cache_key, updates):
            return False

        def list_entries(self):
            return []

        def get_stats(self):
            return {}

        def update_access_time(self, cache_key):
            pass

        def increment_hits(self):
            pass

        def increment_misses(self):
            pass

        def cleanup_expired(self, ttl_seconds):
            return 0

        def cleanup_by_size(self, target_size_mb):
            return {"count": 0, "removed_entries": []}

        def clear_all(self):
            return 0

        # Override schema versioning to use in-memory dict
        def get_schema_version(self, namespace_id=DEFAULT_NAMESPACE):
            return self._versions.get(namespace_id, 0)

        def set_schema_version(self, namespace_id, version):
            self._versions[namespace_id] = version

        def get_migrations(self):
            def migrate_v0_v1(backend, ns_id):
                backend._migration_log.append((ns_id, 0, 1))

            def migrate_v1_v2(backend, ns_id):
                backend._migration_log.append((ns_id, 1, 2))

            return [
                (0, 1, migrate_v0_v1),
                (1, 2, migrate_v1_v2),
            ]

    def test_default_schema_version_is_zero(self):
        backend = self.DummyBackend()
        assert backend.get_schema_version() == 0

    def test_set_and_get_schema_version(self):
        backend = self.DummyBackend()
        backend.set_schema_version("test_ns", 3)
        assert backend.get_schema_version("test_ns") == 3

    def test_run_migrations_from_zero(self):
        backend = self.DummyBackend()
        final = backend.run_migrations("myns")
        assert final == 2
        assert backend.get_schema_version("myns") == 2
        assert backend._migration_log == [("myns", 0, 1), ("myns", 1, 2)]

    def test_run_migrations_partial(self):
        """If already at v1, only v1->v2 should run."""
        backend = self.DummyBackend()
        backend._versions["myns"] = 1
        final = backend.run_migrations("myns")
        assert final == 2
        assert backend._migration_log == [("myns", 1, 2)]

    def test_run_migrations_none_needed(self):
        """If at latest version, no migrations run."""
        backend = self.DummyBackend()
        backend._versions["myns"] = 2
        final = backend.run_migrations("myns")
        assert final == 2
        assert backend._migration_log == []

    def test_run_migrations_default_namespace(self):
        backend = self.DummyBackend()
        final = backend.run_migrations()
        assert final == 2
        assert backend.get_schema_version(DEFAULT_NAMESPACE) == 2


class TestMetadataBackendNamespaceRegistry:
    """Tests for namespace registry defaults on MetadataBackend ABC."""

    class MinimalBackend(MetadataBackend):
        """Backend with only abstract methods implemented (no overrides)."""

        def load_metadata(self):
            return {"entries": {}}

        def save_metadata(self, metadata):
            pass

        def get_entry(self, cache_key):
            return None

        def put_entry(self, cache_key, entry_data):
            pass

        def remove_entry(self, cache_key):
            return False

        def update_entry_metadata(self, cache_key, updates):
            return False

        def list_entries(self):
            return []

        def get_stats(self):
            return {}

        def update_access_time(self, cache_key):
            pass

        def increment_hits(self):
            pass

        def increment_misses(self):
            pass

        def cleanup_expired(self, ttl_seconds):
            return 0

        def cleanup_by_size(self, target_size_mb):
            return {"count": 0, "removed_entries": []}

        def clear_all(self):
            return 0

    def test_list_namespaces_default(self):
        """Default implementation returns single 'default' namespace."""
        backend = self.MinimalBackend()
        namespaces = backend.list_namespaces()
        assert len(namespaces) == 1
        assert namespaces[0].namespace_id == DEFAULT_NAMESPACE
        assert namespaces[0].display_name == "Default"

    def test_get_namespace_default_exists(self):
        backend = self.MinimalBackend()
        ns = backend.get_namespace(DEFAULT_NAMESPACE)
        assert ns is not None
        assert ns.namespace_id == DEFAULT_NAMESPACE

    def test_get_namespace_nonexistent(self):
        backend = self.MinimalBackend()
        assert backend.get_namespace("nonexistent") is None

    def test_namespace_exists_default(self):
        backend = self.MinimalBackend()
        assert backend.namespace_exists(DEFAULT_NAMESPACE) is True

    def test_namespace_exists_nonexistent(self):
        backend = self.MinimalBackend()
        assert backend.namespace_exists("nonexistent") is False

    def test_create_namespace_raises(self):
        """Default create_namespace raises NotImplementedError."""
        backend = self.MinimalBackend()
        with pytest.raises(NotImplementedError, match="does not support namespaces"):
            backend.create_namespace("new_ns")

    def test_drop_namespace_raises(self):
        """Default drop_namespace raises NotImplementedError."""
        backend = self.MinimalBackend()
        with pytest.raises(NotImplementedError, match="does not support namespaces"):
            backend.drop_namespace("some_ns")

    def test_schema_version_default_is_zero(self):
        """Default get_schema_version returns 0 (untracked)."""
        backend = self.MinimalBackend()
        assert backend.get_schema_version() == 0

    def test_set_schema_version_noop(self):
        """Default set_schema_version is a no-op."""
        backend = self.MinimalBackend()
        backend.set_schema_version("default", 5)  # Should not raise
        assert backend.get_schema_version() == 0  # Still 0 (not persisted)

    def test_get_migrations_empty(self):
        """Default get_migrations returns empty list."""
        backend = self.MinimalBackend()
        assert backend.get_migrations() == []

    def test_run_migrations_noop(self):
        """With no migrations, run_migrations returns current version."""
        backend = self.MinimalBackend()
        assert backend.run_migrations() == 0


class TestNamespaceIdPattern:
    """Tests for the NAMESPACE_ID_PATTERN regex directly."""

    @pytest.mark.parametrize(
        "valid_id",
        ["a", "abc", "a1", "test_ns", "project_42", "a" * 48, "0", "___"],
    )
    def test_valid_patterns(self, valid_id):
        assert NAMESPACE_ID_PATTERN.match(valid_id) is not None

    @pytest.mark.parametrize(
        "invalid_id",
        ["", "A", "ABC", "a-b", "a.b", "a b", "a" * 49, "caché", "名前"],
    )
    def test_invalid_patterns(self, invalid_id):
        assert NAMESPACE_ID_PATTERN.match(invalid_id) is None
