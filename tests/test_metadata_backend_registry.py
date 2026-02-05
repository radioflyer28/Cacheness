"""
Tests for Phase 2.2: Metadata Backend Registry

Tests the metadata backend registration system for custom backends.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock

# Import the registry functions
from cacheness.storage.backends import (
    register_metadata_backend,
    unregister_metadata_backend,
    get_metadata_backend,
    list_metadata_backends,
    _metadata_backend_registry,
    _initialize_builtin_backends,
    MetadataBackend,
    JsonBackend,
)

# Conditionally import optional backends
try:
    from cacheness.storage.backends import SqliteBackend
    _HAS_SQLITE = True
except ImportError:
    _HAS_SQLITE = False

try:
    from cacheness.storage.backends import InMemoryBackend
    _HAS_MEMORY = True
except ImportError:
    _HAS_MEMORY = False

# Import module-level API
import cacheness


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def clean_registry():
    """Provide a clean registry state for each test."""
    # Save current registry state
    original_registry = _metadata_backend_registry.copy()
    
    # Reset to builtin backends only
    _metadata_backend_registry.clear()
    _initialize_builtin_backends()
    
    yield _metadata_backend_registry
    
    # Restore original registry state
    _metadata_backend_registry.clear()
    _metadata_backend_registry.update(original_registry)


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class MockMetadataBackend(MetadataBackend):
    """A mock metadata backend for testing."""
    
    def __init__(self, cache_dir: Path = None, **options):
        self.cache_dir = cache_dir
        self.options = options
        self._entries = {}
        self._hits = 0
        self._misses = 0
    
    def load_metadata(self):
        return {"entries": self._entries, "stats": self.get_stats()}
    
    def save_metadata(self, metadata):
        if "entries" in metadata:
            self._entries = metadata["entries"]
    
    def get_entry(self, cache_key: str):
        return self._entries.get(cache_key)
    
    def put_entry(self, cache_key: str, entry_data):
        self._entries[cache_key] = entry_data
    
    def remove_entry(self, cache_key: str):
        if cache_key in self._entries:
            del self._entries[cache_key]
    
    def list_entries(self):
        return list(self._entries.values())
    
    def get_stats(self):
        return {"entries": len(self._entries), "hits": self._hits, "misses": self._misses}
    
    def update_access_time(self, cache_key: str):
        if cache_key in self._entries:
            self._entries[cache_key]["last_access"] = "now"
    
    def increment_hits(self):
        self._hits += 1
    
    def increment_misses(self):
        self._misses += 1
    
    def cleanup_expired(self, ttl_seconds: float) -> int:
        return 0  # Mock doesn't track TTL
    
    def clear_all(self) -> int:
        count = len(self._entries)
        self._entries.clear()
        return count
    
    def close(self) -> None:
        pass


class AnotherMockBackend(MetadataBackend):
    """Another mock backend for testing multiple registrations."""
    
    def __init__(self, cache_dir: Path = None, **options):
        self.cache_dir = cache_dir
        self._entries = {}
        self._hits = 0
        self._misses = 0
    
    def load_metadata(self):
        return {"entries": {}, "stats": {}}
    
    def save_metadata(self, metadata):
        pass
    
    def get_entry(self, cache_key: str):
        return None
    
    def put_entry(self, cache_key: str, entry_data):
        pass
    
    def remove_entry(self, cache_key: str):
        pass
    
    def list_entries(self):
        return []
    
    def get_stats(self):
        return {}
    
    def update_access_time(self, cache_key: str):
        pass
    
    def increment_hits(self):
        pass
    
    def increment_misses(self):
        pass
    
    def cleanup_expired(self, ttl_seconds: float) -> int:
        return 0
    
    def clear_all(self) -> int:
        return 0
    
    def close(self) -> None:
        pass


class NotABackend:
    """A class that doesn't inherit from MetadataBackend."""
    
    def get(self, key: str):
        return None


# =============================================================================
# Test Classes
# =============================================================================

class TestBuiltinBackends:
    """Test that builtin backends are registered on module load."""
    
    def test_json_backend_registered(self, clean_registry):
        """Test that json backend is registered by default."""
        backends = list_metadata_backends()
        names = [b["name"] for b in backends]
        assert "json" in names
    
    def test_sqlite_backend_registered(self, clean_registry):
        """Test that sqlite backend is registered by default."""
        backends = list_metadata_backends()
        names = [b["name"] for b in backends]
        assert "sqlite" in names
    
    def test_memory_backend_registered(self, clean_registry):
        """Test that memory backend is registered by default."""
        backends = list_metadata_backends()
        names = [b["name"] for b in backends]
        assert "memory" in names
    
    def test_builtin_backends_marked_correctly(self, clean_registry):
        """Test that builtin backends have is_builtin=True."""
        backends = list_metadata_backends()
        for backend in backends:
            if backend["name"] in ("json", "sqlite", "memory"):
                assert backend["is_builtin"] is True


class TestRegisterMetadataBackend:
    """Test the register_metadata_backend function."""
    
    def test_register_custom_backend(self, clean_registry):
        """Test registering a custom metadata backend."""
        register_metadata_backend("mock", MockMetadataBackend)
        
        backends = list_metadata_backends()
        names = [b["name"] for b in backends]
        assert "mock" in names
    
    def test_register_backend_not_builtin(self, clean_registry):
        """Test that registered backends are marked as not builtin."""
        register_metadata_backend("mock", MockMetadataBackend)
        
        backends = list_metadata_backends()
        mock_info = next(b for b in backends if b["name"] == "mock")
        assert mock_info["is_builtin"] is False
    
    def test_register_backend_class_info(self, clean_registry):
        """Test that backend class info is captured."""
        register_metadata_backend("mock", MockMetadataBackend)
        
        backends = list_metadata_backends()
        mock_info = next(b for b in backends if b["name"] == "mock")
        assert mock_info["class"] == "MockMetadataBackend"
    
    def test_register_duplicate_raises_error(self, clean_registry):
        """Test that registering duplicate name raises ValueError."""
        register_metadata_backend("mock", MockMetadataBackend)
        
        with pytest.raises(ValueError, match="already registered"):
            register_metadata_backend("mock", AnotherMockBackend)
    
    def test_register_duplicate_with_force(self, clean_registry):
        """Test that force=True allows overwriting existing backend."""
        register_metadata_backend("mock", MockMetadataBackend)
        register_metadata_backend("mock", AnotherMockBackend, force=True)
        
        backends = list_metadata_backends()
        mock_info = next(b for b in backends if b["name"] == "mock")
        assert mock_info["class"] == "AnotherMockBackend"
    
    def test_register_invalid_class_raises_error(self, clean_registry):
        """Test that non-MetadataBackend class raises ValueError."""
        with pytest.raises(ValueError, match="must inherit from MetadataBackend"):
            register_metadata_backend("invalid", NotABackend)
    
    def test_register_non_class_raises_error(self, clean_registry):
        """Test that non-class raises ValueError."""
        with pytest.raises(ValueError, match="must be a class"):
            register_metadata_backend("invalid", "not a class")
    
    def test_register_multiple_backends(self, clean_registry):
        """Test registering multiple custom backends."""
        register_metadata_backend("mock1", MockMetadataBackend)
        register_metadata_backend("mock2", AnotherMockBackend)
        
        backends = list_metadata_backends()
        names = [b["name"] for b in backends]
        assert "mock1" in names
        assert "mock2" in names
    
    def test_cannot_overwrite_builtin_without_force(self, clean_registry):
        """Test that builtin backends cannot be overwritten without force."""
        with pytest.raises(ValueError, match="already registered"):
            register_metadata_backend("json", MockMetadataBackend)
    
    def test_can_overwrite_builtin_with_force(self, clean_registry):
        """Test that builtin backends can be overwritten with force."""
        register_metadata_backend("json", MockMetadataBackend, force=True)
        
        backends = list_metadata_backends()
        json_info = next(b for b in backends if b["name"] == "json")
        assert json_info["class"] == "MockMetadataBackend"


class TestUnregisterMetadataBackend:
    """Test the unregister_metadata_backend function."""
    
    def test_unregister_custom_backend(self, clean_registry):
        """Test unregistering a custom backend."""
        register_metadata_backend("mock", MockMetadataBackend)
        
        result = unregister_metadata_backend("mock")
        
        assert result is True
        backends = list_metadata_backends()
        names = [b["name"] for b in backends]
        assert "mock" not in names
    
    def test_unregister_nonexistent_returns_false(self, clean_registry):
        """Test that unregistering nonexistent backend returns False."""
        result = unregister_metadata_backend("nonexistent")
        assert result is False
    
    def test_unregister_builtin_backend(self, clean_registry):
        """Test that builtin backends can be unregistered."""
        result = unregister_metadata_backend("json")
        
        assert result is True
        backends = list_metadata_backends()
        names = [b["name"] for b in backends]
        assert "json" not in names
    
    def test_unregister_then_reregister(self, clean_registry):
        """Test unregistering then re-registering a backend."""
        register_metadata_backend("mock", MockMetadataBackend)
        unregister_metadata_backend("mock")
        register_metadata_backend("mock", AnotherMockBackend)
        
        backends = list_metadata_backends()
        mock_info = next(b for b in backends if b["name"] == "mock")
        assert mock_info["class"] == "AnotherMockBackend"


class TestGetMetadataBackend:
    """Test the get_metadata_backend function."""
    
    def test_get_json_backend(self, clean_registry, temp_dir):
        """Test getting a JSON backend instance."""
        metadata_file = temp_dir / "metadata.json"
        backend = get_metadata_backend("json", metadata_file=metadata_file)
        
        assert backend is not None
        assert isinstance(backend, JsonBackend)
    
    @pytest.mark.skipif(not _HAS_SQLITE, reason="SQLite backend not available")
    def test_get_sqlite_backend(self, clean_registry, temp_dir):
        """Test getting a SQLite backend instance."""
        db_file = temp_dir / "cache.db"
        backend = get_metadata_backend("sqlite", db_file=db_file)
        
        assert backend is not None
        assert isinstance(backend, SqliteBackend)
        backend.close()
    
    @pytest.mark.skipif(not _HAS_MEMORY, reason="Memory backend not available")
    def test_get_memory_backend(self, clean_registry):
        """Test getting a memory backend instance."""
        backend = get_metadata_backend("memory")
        
        assert backend is not None
        assert isinstance(backend, InMemoryBackend)
    
    def test_get_custom_backend(self, clean_registry, temp_dir):
        """Test getting a custom backend instance."""
        register_metadata_backend("mock", MockMetadataBackend)
        
        backend = get_metadata_backend("mock", cache_dir=temp_dir)
        
        assert backend is not None
        assert isinstance(backend, MockMetadataBackend)
        assert backend.cache_dir == temp_dir
    
    def test_get_backend_with_options(self, clean_registry, temp_dir):
        """Test passing options to backend constructor."""
        register_metadata_backend("mock", MockMetadataBackend)
        
        backend = get_metadata_backend("mock", cache_dir=temp_dir, custom_option="value")
        
        assert backend.options.get("custom_option") == "value"
    
    def test_get_nonexistent_raises_error(self, clean_registry):
        """Test that getting nonexistent backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown metadata backend"):
            get_metadata_backend("nonexistent")
    
    def test_get_creates_new_instance(self, clean_registry, temp_dir):
        """Test that each get creates a new instance."""
        register_metadata_backend("mock", MockMetadataBackend)
        
        backend1 = get_metadata_backend("mock", cache_dir=temp_dir)
        backend2 = get_metadata_backend("mock", cache_dir=temp_dir)
        
        assert backend1 is not backend2


class TestListMetadataBackends:
    """Test the list_metadata_backends function."""
    
    def test_list_returns_list(self, clean_registry):
        """Test that list_metadata_backends returns a list."""
        result = list_metadata_backends()
        assert isinstance(result, list)
    
    def test_list_contains_builtins(self, clean_registry):
        """Test that list contains builtin backends."""
        backends = list_metadata_backends()
        names = [b["name"] for b in backends]
        
        assert "json" in names
        assert "sqlite" in names
        assert "memory" in names
    
    def test_list_entry_format(self, clean_registry):
        """Test that list entries have correct format."""
        backends = list_metadata_backends()
        
        for backend in backends:
            assert "name" in backend
            assert "class" in backend
            assert "is_builtin" in backend
    
    def test_list_includes_custom_backends(self, clean_registry):
        """Test that custom backends appear in list."""
        register_metadata_backend("mock", MockMetadataBackend)
        
        backends = list_metadata_backends()
        names = [b["name"] for b in backends]
        assert "mock" in names
    
    def test_list_order_builtins_first(self, clean_registry):
        """Test that builtin backends appear before custom ones."""
        register_metadata_backend("aaa_mock", MockMetadataBackend)
        
        backends = list_metadata_backends()
        
        # Find indices of builtin and custom
        builtin_indices = [i for i, b in enumerate(backends) if b["is_builtin"]]
        custom_indices = [i for i, b in enumerate(backends) if not b["is_builtin"]]
        
        # All builtin indices should be less than all custom indices
        if builtin_indices and custom_indices:
            assert max(builtin_indices) < min(custom_indices)


class TestModuleLevelAPI:
    """Test the module-level API in cacheness package."""
    
    def test_register_metadata_backend_exported(self):
        """Test that register_metadata_backend is exported from cacheness."""
        assert hasattr(cacheness, "register_metadata_backend")
    
    def test_unregister_metadata_backend_exported(self):
        """Test that unregister_metadata_backend is exported from cacheness."""
        assert hasattr(cacheness, "unregister_metadata_backend")
    
    def test_get_metadata_backend_exported(self):
        """Test that get_metadata_backend is exported from cacheness."""
        assert hasattr(cacheness, "get_metadata_backend")
    
    def test_list_metadata_backends_exported(self):
        """Test that list_metadata_backends is exported from cacheness."""
        assert hasattr(cacheness, "list_metadata_backends")
    
    def test_metadata_backend_base_class_exported(self):
        """Test that MetadataBackend base class is exported."""
        assert hasattr(cacheness, "MetadataBackend")
    
    def test_module_level_functions_work(self, clean_registry, temp_dir):
        """Test that module-level functions work correctly."""
        # These should work without raising errors
        backends = cacheness.list_metadata_backends()
        assert len(backends) >= 3  # At least builtin backends
        
        backend = cacheness.get_metadata_backend("memory")
        assert backend is not None


class TestBackendFunctionality:
    """Test that registered backends function correctly."""
    
    def test_custom_backend_put_get(self, clean_registry, temp_dir):
        """Test put_entry/get_entry with custom backend."""
        register_metadata_backend("mock", MockMetadataBackend)
        backend = get_metadata_backend("mock", cache_dir=temp_dir)
        
        # Store metadata
        backend.put_entry("test_key", {"name": "test", "size": 100})
        
        # Retrieve metadata
        result = backend.get_entry("test_key")
        assert result is not None
        assert result["name"] == "test"
        assert result["size"] == 100
    
    def test_custom_backend_exists(self, clean_registry, temp_dir):
        """Test get_entry returns None for missing keys."""
        register_metadata_backend("mock", MockMetadataBackend)
        backend = get_metadata_backend("mock", cache_dir=temp_dir)
        
        assert backend.get_entry("test_key") is None
        backend.put_entry("test_key", {"name": "test"})
        assert backend.get_entry("test_key") is not None
    
    def test_custom_backend_delete(self, clean_registry, temp_dir):
        """Test remove_entry with custom backend."""
        register_metadata_backend("mock", MockMetadataBackend)
        backend = get_metadata_backend("mock", cache_dir=temp_dir)
        
        backend.put_entry("test_key", {"name": "test"})
        assert backend.get_entry("test_key") is not None
        
        backend.remove_entry("test_key")
        assert backend.get_entry("test_key") is None
    
    def test_custom_backend_list_entries(self, clean_registry, temp_dir):
        """Test list_entries with custom backend."""
        register_metadata_backend("mock", MockMetadataBackend)
        backend = get_metadata_backend("mock", cache_dir=temp_dir)
        
        backend.put_entry("key1", {"name": "test1"})
        backend.put_entry("key2", {"name": "test2"})
        
        entries = backend.list_entries()
        assert len(entries) == 2
        names = [e["name"] for e in entries]
        assert "test1" in names
        assert "test2" in names
    
    def test_custom_backend_clear_all(self, clean_registry, temp_dir):
        """Test clear_all with custom backend."""
        register_metadata_backend("mock", MockMetadataBackend)
        backend = get_metadata_backend("mock", cache_dir=temp_dir)
        
        backend.put_entry("key1", {"name": "test1"})
        backend.put_entry("key2", {"name": "test2"})
        
        count = backend.clear_all()
        assert count == 2
        assert len(backend.list_entries()) == 0


class TestBackendValidation:
    """Test backend class validation during registration."""
    
    def test_validate_missing_get_method(self, clean_registry):
        """Test that backend without get method fails validation."""
        class InvalidBackend(MetadataBackend):
            def put(self, key, metadata): pass
            def delete(self, key): pass
            def exists(self, key): pass
            def list_keys(self): pass
            def clear(self): pass
            def close(self): pass
            def get_stats(self): pass
            # Missing: get
        
        # Should work because it inherits MetadataBackend (which has abstract methods)
        # The validation is handled by MetadataBackend's abstract method enforcement
        # This test verifies the type check works
        with pytest.raises(TypeError):
            InvalidBackend()  # Can't instantiate abstract class
    
    def test_instance_instead_of_class(self, clean_registry, temp_dir):
        """Test that passing an instance instead of class raises error."""
        instance = MockMetadataBackend(cache_dir=temp_dir)
        
        with pytest.raises(ValueError, match="must be a class"):
            register_metadata_backend("mock", instance)


class TestConcurrency:
    """Test thread safety of backend registry operations."""
    
    def test_concurrent_registration(self, clean_registry):
        """Test that concurrent registrations don't corrupt registry."""
        import threading
        import time
        
        errors = []
        
        def register_backend(name):
            try:
                # Create a unique backend class
                class TempBackend(MockMetadataBackend):
                    pass
                
                register_metadata_backend(name, TempBackend)
            except Exception as e:
                errors.append(e)
        
        threads = []
        for i in range(10):
            t = threading.Thread(target=register_backend, args=(f"backend_{i}",))
            threads.append(t)
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # Check no errors occurred
        assert len(errors) == 0
        
        # Check all backends were registered
        backends = list_metadata_backends()
        names = [b["name"] for b in backends]
        for i in range(10):
            assert f"backend_{i}" in names


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for metadata backend registry."""
    
    def test_full_workflow(self, clean_registry, temp_dir):
        """Test complete workflow: register, use, unregister."""
        # Register custom backend
        register_metadata_backend("mock", MockMetadataBackend)
        
        # List should include it
        backends = list_metadata_backends()
        names = [b["name"] for b in backends]
        assert "mock" in names
        
        # Create instance and use it
        backend = get_metadata_backend("mock", cache_dir=temp_dir)
        backend.put_entry("key1", {"data": "value"})
        assert backend.get_entry("key1")["data"] == "value"
        
        # Unregister
        unregister_metadata_backend("mock")
        
        # Should no longer be available
        backends = list_metadata_backends()
        names = [b["name"] for b in backends]
        assert "mock" not in names
        
        # Getting it should raise error
        with pytest.raises(ValueError):
            get_metadata_backend("mock")
    
    def test_replace_builtin_backend(self, clean_registry, temp_dir):
        """Test replacing a builtin backend with custom implementation."""
        # Replace json backend
        register_metadata_backend("json", MockMetadataBackend, force=True)
        
        # Get should return our mock
        backend = get_metadata_backend("json", cache_dir=temp_dir)
        assert isinstance(backend, MockMetadataBackend)
        
        # Verify the class has been replaced
        backends = list_metadata_backends()
        json_info = next(b for b in backends if b["name"] == "json")
        assert json_info["class"] == "MockMetadataBackend"
        # Note: is_builtin is determined by name, not class, so it remains True
        assert json_info["is_builtin"] is True
