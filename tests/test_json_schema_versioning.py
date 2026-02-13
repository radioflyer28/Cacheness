"""Tests for JSON backend schema versioning and namespace registry."""

import json
import os
from pathlib import Path

import pytest

from cacheness.metadata import (
    JsonBackend,
    NamespaceInfo,
    DEFAULT_NAMESPACE,
    validate_namespace_id,
)


@pytest.fixture
def json_backend(tmp_path):
    """Create a fresh JSON backend in a temp directory."""
    metadata_file = tmp_path / "metadata.json"
    backend = JsonBackend(metadata_file)
    yield backend
    backend.close()


@pytest.fixture
def json_backend_path(tmp_path):
    """Return a metadata file path for manual backend construction."""
    return tmp_path / "metadata.json"


# ── Schema versioning ─────────────────────────────────────────────────


class TestJsonSchemaVersioning:
    """Test schema versioning on JsonBackend."""

    def test_default_namespace_registered_on_init(self, json_backend):
        """Creating a backend should register the 'default' namespace."""
        ns = json_backend.get_namespace(DEFAULT_NAMESPACE)
        assert ns is not None
        assert ns.namespace_id == DEFAULT_NAMESPACE
        assert ns.display_name == "Default"
        assert ns.schema_version >= 1

    def test_get_schema_version_default(self, json_backend):
        """Schema version should be 1 after init."""
        version = json_backend.get_schema_version(DEFAULT_NAMESPACE)
        assert version == 1

    def test_set_schema_version(self, json_backend):
        """set_schema_version should persist."""
        json_backend.set_schema_version(DEFAULT_NAMESPACE, 42)
        assert json_backend.get_schema_version(DEFAULT_NAMESPACE) == 42

    def test_get_schema_version_unknown_namespace(self, json_backend):
        """Unknown namespace returns 0."""
        assert json_backend.get_schema_version("nonexistent") == 0

    def test_set_schema_version_unknown_namespace_noop(self, json_backend):
        """Setting version on unknown namespace is silently ignored."""
        json_backend.set_schema_version("nonexistent", 5)
        assert json_backend.get_schema_version("nonexistent") == 0

    def test_get_migrations_empty(self, json_backend):
        """JSON backend currently has no migrations."""
        assert json_backend.get_migrations() == []

    def test_run_migrations_returns_current_version(self, json_backend):
        """run_migrations with no migrations returns current version unchanged."""
        version = json_backend.run_migrations(DEFAULT_NAMESPACE)
        assert version == 1

    def test_idempotent_init(self, json_backend_path):
        """Opening the same directory twice should not fail or change version."""
        backend1 = JsonBackend(json_backend_path)
        v1 = backend1.get_schema_version(DEFAULT_NAMESPACE)
        backend1.close()

        backend2 = JsonBackend(json_backend_path)
        v2 = backend2.get_schema_version(DEFAULT_NAMESPACE)
        backend2.close()

        assert v1 == v2 == 1

    def test_registry_file_created(self, json_backend, tmp_path):
        """The cacheness_namespaces.json file should exist after init."""
        registry_file = tmp_path / "cacheness_namespaces.json"
        assert registry_file.exists()

    def test_registry_file_valid_json(self, json_backend, tmp_path):
        """The registry file should contain valid JSON with expected structure."""
        registry_file = tmp_path / "cacheness_namespaces.json"
        with open(registry_file) as f:
            data = json.load(f)
        assert "namespaces" in data
        assert DEFAULT_NAMESPACE in data["namespaces"]

    def test_schema_version_persists_across_restarts(self, json_backend_path):
        """Schema version changes should survive backend restarts."""
        backend = JsonBackend(json_backend_path)
        backend.set_schema_version(DEFAULT_NAMESPACE, 7)
        backend.close()

        backend2 = JsonBackend(json_backend_path)
        assert backend2.get_schema_version(DEFAULT_NAMESPACE) == 7
        backend2.close()


# ── Namespace registry ─────────────────────────────────────────────────


class TestJsonNamespaceRegistry:
    """Test namespace registry on JsonBackend."""

    def test_list_namespaces_has_default(self, json_backend):
        """list_namespaces should include 'default' after init."""
        namespaces = json_backend.list_namespaces()
        ids = [ns.namespace_id for ns in namespaces]
        assert DEFAULT_NAMESPACE in ids

    def test_create_namespace(self, json_backend, tmp_path):
        """create_namespace should register and create a metadata file."""
        ns = json_backend.create_namespace("project_alpha", "Project Alpha")
        assert ns.namespace_id == "project_alpha"
        assert ns.display_name == "Project Alpha"
        assert ns.schema_version == 1

        # Per-namespace metadata file should exist
        ns_file = tmp_path / "project_alpha_metadata.json"
        assert ns_file.exists()

        # Should appear in list
        ids = [n.namespace_id for n in json_backend.list_namespaces()]
        assert "project_alpha" in ids

    def test_create_namespace_duplicate_raises(self, json_backend):
        """Creating a namespace that already exists should raise ValueError."""
        json_backend.create_namespace("dup_ns")
        with pytest.raises(ValueError, match="already exists"):
            json_backend.create_namespace("dup_ns")

    def test_create_default_namespace_raises(self, json_backend):
        """Cannot re-create the default namespace."""
        with pytest.raises(ValueError, match="pre-registered"):
            json_backend.create_namespace(DEFAULT_NAMESPACE)

    def test_create_namespace_invalid_id(self, json_backend):
        """Invalid namespace IDs should be rejected."""
        with pytest.raises(ValueError, match="Invalid namespace ID"):
            json_backend.create_namespace("UPPERCASE")
        with pytest.raises(ValueError, match="Invalid namespace ID"):
            json_backend.create_namespace("has-dash")
        with pytest.raises(ValueError, match="Invalid namespace ID"):
            json_backend.create_namespace("")

    def test_drop_namespace(self, json_backend, tmp_path):
        """drop_namespace should remove file and registry entry."""
        json_backend.create_namespace("to_drop")
        ns_file = tmp_path / "to_drop_metadata.json"
        assert ns_file.exists()

        result = json_backend.drop_namespace("to_drop")
        assert result is True
        assert not ns_file.exists()
        assert not json_backend.namespace_exists("to_drop")

    def test_drop_nonexistent_namespace(self, json_backend):
        """Dropping a namespace that doesn't exist returns False."""
        assert json_backend.drop_namespace("nonexistent") is False

    def test_drop_default_raises(self, json_backend):
        """Cannot drop the 'default' namespace."""
        with pytest.raises(ValueError, match="Cannot drop"):
            json_backend.drop_namespace(DEFAULT_NAMESPACE)

    def test_get_namespace(self, json_backend):
        """get_namespace should return NamespaceInfo or None."""
        json_backend.create_namespace("lookup_ns", "Lookup Test")
        ns = json_backend.get_namespace("lookup_ns")
        assert ns is not None
        assert ns.namespace_id == "lookup_ns"
        assert ns.display_name == "Lookup Test"
        assert ns.schema_version == 1

        assert json_backend.get_namespace("nonexistent") is None

    def test_namespace_exists(self, json_backend):
        """namespace_exists should return correct boolean."""
        assert json_backend.namespace_exists(DEFAULT_NAMESPACE) is True
        assert json_backend.namespace_exists("nope") is False

        json_backend.create_namespace("check_me")
        assert json_backend.namespace_exists("check_me") is True

    def test_multiple_namespaces(self, json_backend, tmp_path):
        """Can create and list multiple namespaces."""
        json_backend.create_namespace("ns_one")
        json_backend.create_namespace("ns_two")
        json_backend.create_namespace("ns_three")

        namespaces = json_backend.list_namespaces()
        ids = {ns.namespace_id for ns in namespaces}
        assert ids == {DEFAULT_NAMESPACE, "ns_one", "ns_two", "ns_three"}

        # Each should have its own file
        assert (tmp_path / "ns_one_metadata.json").exists()
        assert (tmp_path / "ns_two_metadata.json").exists()
        assert (tmp_path / "ns_three_metadata.json").exists()

    def test_namespace_metadata_file_has_correct_structure(
        self, json_backend, tmp_path
    ):
        """Per-namespace metadata files should have the standard JSON structure."""
        json_backend.create_namespace("struct_test")
        ns_file = tmp_path / "struct_test_metadata.json"

        with open(ns_file) as f:
            data = json.load(f)
        assert "entries" in data
        assert isinstance(data["entries"], dict)
        assert data["cache_hits"] == 0
        assert data["cache_misses"] == 0

    def test_default_namespace_maps_to_existing_file(self, json_backend):
        """Default namespace should map to the original metadata file."""
        ns_file = json_backend._metadata_file_for_namespace(DEFAULT_NAMESPACE)
        assert ns_file == json_backend.metadata_file

    def test_other_namespace_has_suffixed_file(self, json_backend, tmp_path):
        """Non-default namespaces should use suffixed filenames."""
        ns_file = json_backend._metadata_file_for_namespace("my_project")
        assert ns_file == tmp_path / "my_project_metadata.json"

    def test_drop_namespace_then_recreate(self, json_backend, tmp_path):
        """Can drop and re-create the same namespace."""
        json_backend.create_namespace("recyclable")
        json_backend.drop_namespace("recyclable")
        assert not json_backend.namespace_exists("recyclable")

        ns = json_backend.create_namespace("recyclable", "Second time")
        assert ns.display_name == "Second time"
        assert json_backend.namespace_exists("recyclable")

    def test_set_schema_version_on_created_namespace(self, json_backend):
        """Schema version can be set on namespaces created via create_namespace."""
        json_backend.create_namespace("versioned_ns")
        assert json_backend.get_schema_version("versioned_ns") == 1

        json_backend.set_schema_version("versioned_ns", 5)
        assert json_backend.get_schema_version("versioned_ns") == 5


# ── Backward compatibility ─────────────────────────────────────────────


class TestJsonBackwardCompatibility:
    """Ensure existing JsonBackend operations still work correctly."""

    def test_put_and_get_entry(self, json_backend):
        """Basic put/get should still work after namespace additions."""
        json_backend.put_entry(
            "abc123",
            {
                "description": "test entry",
                "data_type": "pickle",
                "metadata": {"key": "value"},
            },
        )
        entry = json_backend.get_entry("abc123")
        assert entry is not None
        assert entry["description"] == "test entry"
        assert entry["data_type"] == "pickle"

    def test_list_entries(self, json_backend):
        """list_entries should still work."""
        json_backend.put_entry(
            "key1",
            {
                "data_type": "pickle",
                "file_size": 1024,
            },
        )
        entries = json_backend.list_entries()
        assert len(entries) == 1
        assert entries[0]["cache_key"] == "key1"

    def test_stats(self, json_backend):
        """get_stats should still work."""
        stats = json_backend.get_stats()
        assert "total_entries" in stats
        assert "cache_hits" in stats

    def test_clear_all(self, json_backend):
        """clear_all should still work."""
        json_backend.put_entry("key1", {"data_type": "pickle"})
        count = json_backend.clear_all()
        assert count == 1
        assert json_backend.get_entry("key1") is None

    def test_legacy_db_without_registry_auto_upgrades(self, tmp_path):
        """A pre-namespace metadata file should auto-register 'default' on open."""
        # Create a legacy metadata file without registry
        metadata_file = tmp_path / "metadata.json"
        legacy_data = {
            "entries": {"old_key": {"data_type": "pickle", "description": "old"}},
            "cache_hits": 5,
            "cache_misses": 2,
        }
        with open(metadata_file, "w") as f:
            json.dump(legacy_data, f)

        # No registry file yet
        registry_file = tmp_path / "cacheness_namespaces.json"
        assert not registry_file.exists()

        # Open backend — should auto-create registry
        backend = JsonBackend(metadata_file)
        assert registry_file.exists()
        assert backend.namespace_exists(DEFAULT_NAMESPACE)
        assert backend.get_schema_version(DEFAULT_NAMESPACE) == 1

        # Existing entries should still be accessible
        entry = backend.get_entry("old_key")
        assert entry is not None
        assert entry["data_type"] == "pickle"

        backend.close()

    def test_corrupted_registry_recovery(self, tmp_path):
        """If registry file is corrupted, backend should recover gracefully."""
        metadata_file = tmp_path / "metadata.json"
        registry_file = tmp_path / "cacheness_namespaces.json"

        # Create a corrupted registry
        with open(registry_file, "w") as f:
            f.write("NOT VALID JSON{{{")

        backend = JsonBackend(metadata_file)
        # Should have recovered and seeded 'default'
        assert backend.namespace_exists(DEFAULT_NAMESPACE)
        assert backend.get_schema_version(DEFAULT_NAMESPACE) == 1
        backend.close()
