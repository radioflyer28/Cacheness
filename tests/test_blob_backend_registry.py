"""
Tests for Phase 2.3: Blob Storage Backend Registry

Tests the blob storage backend registration system for custom backends.
"""

import pytest
import tempfile
from pathlib import Path
from io import BytesIO

# Import the registry functions
from cacheness.storage.backends.blob_backends import (
    BlobBackend,
    FilesystemBlobBackend,
    register_blob_backend,
    unregister_blob_backend,
    get_blob_backend,
    list_blob_backends,
    _blob_backend_registry,
    _initialize_builtin_blob_backends,
)

# Import module-level API
import cacheness


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def clean_registry():
    """Provide a clean registry state for each test."""
    # Save current registry state
    original_registry = _blob_backend_registry.copy()
    
    # Reset to builtin backends only
    _blob_backend_registry.clear()
    _initialize_builtin_blob_backends()
    
    yield _blob_backend_registry
    
    # Restore original registry state
    _blob_backend_registry.clear()
    _blob_backend_registry.update(original_registry)


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class MockBlobBackend(BlobBackend):
    """A mock blob backend for testing."""
    
    def __init__(self, base_dir: str = None, **options):
        self.base_dir = base_dir
        self.options = options
        self._storage = {}
    
    def write_blob(self, blob_id: str, data: bytes) -> str:
        blob_path = f"mock://{blob_id}"
        self._storage[blob_path] = data
        return blob_path
    
    def read_blob(self, blob_path: str) -> bytes:
        if blob_path not in self._storage:
            raise FileNotFoundError(f"Blob not found: {blob_path}")
        return self._storage[blob_path]
    
    def delete_blob(self, blob_path: str) -> bool:
        if blob_path in self._storage:
            del self._storage[blob_path]
            return True
        return False
    
    def exists(self, blob_path: str) -> bool:
        return blob_path in self._storage


class AnotherMockBackend(BlobBackend):
    """Another mock backend for testing multiple registrations."""
    
    def __init__(self, **options):
        self._storage = {}
    
    def write_blob(self, blob_id: str, data: bytes) -> str:
        return f"another://{blob_id}"
    
    def read_blob(self, blob_path: str) -> bytes:
        return b""
    
    def delete_blob(self, blob_path: str) -> bool:
        return False
    
    def exists(self, blob_path: str) -> bool:
        return False


class NotABackend:
    """A class that doesn't inherit from BlobBackend."""
    
    def write_blob(self, blob_id: str, data: bytes) -> str:
        return ""


# =============================================================================
# Test Classes
# =============================================================================

class TestBuiltinBlobBackends:
    """Test that builtin blob backends are registered on module load."""
    
    def test_filesystem_backend_registered(self, clean_registry):
        """Test that filesystem backend is registered by default."""
        backends = list_blob_backends()
        names = [b["name"] for b in backends]
        assert "filesystem" in names
    
    def test_builtin_backends_marked_correctly(self, clean_registry):
        """Test that builtin backends have is_builtin=True."""
        backends = list_blob_backends()
        for backend in backends:
            if backend["name"] in ("filesystem",):
                assert backend["is_builtin"] is True


class TestRegisterBlobBackend:
    """Test the register_blob_backend function."""
    
    def test_register_custom_backend(self, clean_registry):
        """Test registering a custom blob backend."""
        register_blob_backend("mock", MockBlobBackend)
        
        backends = list_blob_backends()
        names = [b["name"] for b in backends]
        assert "mock" in names
    
    def test_register_backend_not_builtin(self, clean_registry):
        """Test that registered backends are marked as not builtin."""
        register_blob_backend("mock", MockBlobBackend)
        
        backends = list_blob_backends()
        mock_info = next(b for b in backends if b["name"] == "mock")
        assert mock_info["is_builtin"] is False
    
    def test_register_backend_class_info(self, clean_registry):
        """Test that backend class info is captured."""
        register_blob_backend("mock", MockBlobBackend)
        
        backends = list_blob_backends()
        mock_info = next(b for b in backends if b["name"] == "mock")
        assert mock_info["class"] == "MockBlobBackend"
    
    def test_register_duplicate_raises_error(self, clean_registry):
        """Test that registering duplicate name raises ValueError."""
        register_blob_backend("mock", MockBlobBackend)
        
        with pytest.raises(ValueError, match="already registered"):
            register_blob_backend("mock", AnotherMockBackend)
    
    def test_register_duplicate_with_force(self, clean_registry):
        """Test that force=True allows overwriting existing backend."""
        register_blob_backend("mock", MockBlobBackend)
        register_blob_backend("mock", AnotherMockBackend, force=True)
        
        backends = list_blob_backends()
        mock_info = next(b for b in backends if b["name"] == "mock")
        assert mock_info["class"] == "AnotherMockBackend"
    
    def test_register_invalid_class_raises_error(self, clean_registry):
        """Test that non-BlobBackend class raises ValueError."""
        with pytest.raises(ValueError, match="must inherit from BlobBackend"):
            register_blob_backend("invalid", NotABackend)
    
    def test_register_non_class_raises_error(self, clean_registry):
        """Test that non-class raises ValueError."""
        with pytest.raises(ValueError, match="must be a class"):
            register_blob_backend("invalid", "not a class")
    
    def test_register_multiple_backends(self, clean_registry):
        """Test registering multiple custom backends."""
        register_blob_backend("mock1", MockBlobBackend)
        register_blob_backend("mock2", AnotherMockBackend)
        
        backends = list_blob_backends()
        names = [b["name"] for b in backends]
        assert "mock1" in names
        assert "mock2" in names
    
    def test_cannot_overwrite_builtin_without_force(self, clean_registry):
        """Test that builtin backends cannot be overwritten without force."""
        with pytest.raises(ValueError, match="already registered"):
            register_blob_backend("filesystem", MockBlobBackend)
    
    def test_can_overwrite_builtin_with_force(self, clean_registry):
        """Test that builtin backends can be overwritten with force."""
        register_blob_backend("filesystem", MockBlobBackend, force=True)
        
        backends = list_blob_backends()
        fs_info = next(b for b in backends if b["name"] == "filesystem")
        assert fs_info["class"] == "MockBlobBackend"


class TestUnregisterBlobBackend:
    """Test the unregister_blob_backend function."""
    
    def test_unregister_custom_backend(self, clean_registry):
        """Test unregistering a custom backend."""
        register_blob_backend("mock", MockBlobBackend)
        
        result = unregister_blob_backend("mock")
        
        assert result is True
        backends = list_blob_backends()
        names = [b["name"] for b in backends]
        assert "mock" not in names
    
    def test_unregister_nonexistent_returns_false(self, clean_registry):
        """Test that unregistering nonexistent backend returns False."""
        result = unregister_blob_backend("nonexistent")
        assert result is False
    
    def test_unregister_builtin_backend(self, clean_registry):
        """Test that builtin backends can be unregistered."""
        result = unregister_blob_backend("filesystem")
        
        assert result is True
        backends = list_blob_backends()
        names = [b["name"] for b in backends]
        assert "filesystem" not in names
    
    def test_unregister_then_reregister(self, clean_registry):
        """Test unregistering then re-registering a backend."""
        register_blob_backend("mock", MockBlobBackend)
        unregister_blob_backend("mock")
        register_blob_backend("mock", AnotherMockBackend)
        
        backends = list_blob_backends()
        mock_info = next(b for b in backends if b["name"] == "mock")
        assert mock_info["class"] == "AnotherMockBackend"


class TestGetBlobBackend:
    """Test the get_blob_backend function."""
    
    def test_get_filesystem_backend(self, clean_registry, temp_dir):
        """Test getting a filesystem backend instance."""
        backend = get_blob_backend("filesystem", base_dir=temp_dir)
        
        assert backend is not None
        assert isinstance(backend, FilesystemBlobBackend)
    
    def test_get_custom_backend(self, clean_registry, temp_dir):
        """Test getting a custom backend instance."""
        register_blob_backend("mock", MockBlobBackend)
        
        backend = get_blob_backend("mock", base_dir=str(temp_dir))
        
        assert backend is not None
        assert isinstance(backend, MockBlobBackend)
        assert backend.base_dir == str(temp_dir)
    
    def test_get_backend_with_options(self, clean_registry, temp_dir):
        """Test passing options to backend constructor."""
        register_blob_backend("mock", MockBlobBackend)
        
        backend = get_blob_backend("mock", base_dir=str(temp_dir), custom_option="value")
        
        assert backend.options.get("custom_option") == "value"
    
    def test_get_nonexistent_raises_error(self, clean_registry):
        """Test that getting nonexistent backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown blob backend"):
            get_blob_backend("nonexistent")
    
    def test_get_creates_new_instance(self, clean_registry, temp_dir):
        """Test that each get creates a new instance."""
        register_blob_backend("mock", MockBlobBackend)
        
        backend1 = get_blob_backend("mock", base_dir=str(temp_dir))
        backend2 = get_blob_backend("mock", base_dir=str(temp_dir))
        
        assert backend1 is not backend2


class TestListBlobBackends:
    """Test the list_blob_backends function."""
    
    def test_list_returns_list(self, clean_registry):
        """Test that list_blob_backends returns a list."""
        result = list_blob_backends()
        assert isinstance(result, list)
    
    def test_list_contains_builtins(self, clean_registry):
        """Test that list contains builtin backends."""
        backends = list_blob_backends()
        names = [b["name"] for b in backends]
        
        assert "filesystem" in names
        assert "memory" in names
    
    def test_list_entry_format(self, clean_registry):
        """Test that list entries have correct format."""
        backends = list_blob_backends()
        
        for backend in backends:
            assert "name" in backend
            assert "class" in backend
            assert "is_builtin" in backend
    
    def test_list_includes_custom_backends(self, clean_registry):
        """Test that custom backends appear in list."""
        register_blob_backend("mock", MockBlobBackend)
        
        backends = list_blob_backends()
        names = [b["name"] for b in backends]
        assert "mock" in names
    
    def test_list_order_builtins_first(self, clean_registry):
        """Test that builtin backends appear before custom ones."""
        register_blob_backend("aaa_mock", MockBlobBackend)
        
        backends = list_blob_backends()
        
        # Find indices of builtin and custom
        builtin_indices = [i for i, b in enumerate(backends) if b["is_builtin"]]
        custom_indices = [i for i, b in enumerate(backends) if not b["is_builtin"]]
        
        # All builtin indices should be less than all custom indices
        if builtin_indices and custom_indices:
            assert max(builtin_indices) < min(custom_indices)


class TestModuleLevelAPI:
    """Test the module-level API in cacheness package."""
    
    def test_register_blob_backend_exported(self):
        """Test that register_blob_backend is exported from cacheness."""
        assert hasattr(cacheness, "register_blob_backend")
    
    def test_unregister_blob_backend_exported(self):
        """Test that unregister_blob_backend is exported from cacheness."""
        assert hasattr(cacheness, "unregister_blob_backend")
    
    def test_get_blob_backend_exported(self):
        """Test that get_blob_backend is exported from cacheness."""
        assert hasattr(cacheness, "get_blob_backend")
    
    def test_list_blob_backends_exported(self):
        """Test that list_blob_backends is exported from cacheness."""
        assert hasattr(cacheness, "list_blob_backends")
    
    def test_blob_backend_base_class_exported(self):
        """Test that BlobBackend base class is exported."""
        assert hasattr(cacheness, "BlobBackend")
    
    def test_filesystem_blob_backend_exported(self):
        """Test that FilesystemBlobBackend is exported."""
        assert hasattr(cacheness, "FilesystemBlobBackend")
    
    def test_module_level_functions_work(self, clean_registry, temp_dir):
        """Test that module-level functions work correctly."""
        # These should work without raising errors
        backends = cacheness.list_blob_backends()
        assert len(backends) >= 1  # At least builtin backends
        
        backend = cacheness.get_blob_backend("filesystem", base_dir=str(temp_dir))
        assert backend is not None


class TestFilesystemBlobBackend:
    """Test the FilesystemBlobBackend implementation."""
    
    def test_write_and_read_blob(self, temp_dir):
        """Test writing and reading a blob."""
        backend = FilesystemBlobBackend(temp_dir)
        
        data = b"Hello, World!"
        blob_path = backend.write_blob("test_key", data)
        
        result = backend.read_blob(blob_path)
        assert result == data
    
    def test_exists(self, temp_dir):
        """Test exists method."""
        backend = FilesystemBlobBackend(temp_dir)
        
        blob_path = backend.write_blob("test_key", b"data")
        
        assert backend.exists(blob_path)
        assert not backend.exists("nonexistent")
    
    def test_delete_blob(self, temp_dir):
        """Test deleting a blob."""
        backend = FilesystemBlobBackend(temp_dir)
        
        blob_path = backend.write_blob("test_key", b"data")
        assert backend.exists(blob_path)
        
        result = backend.delete_blob(blob_path)
        assert result is True
        assert not backend.exists(blob_path)
    
    def test_delete_nonexistent(self, temp_dir):
        """Test deleting a nonexistent blob."""
        backend = FilesystemBlobBackend(temp_dir)
        result = backend.delete_blob("nonexistent")
        assert result is False
    
    def test_read_nonexistent_raises(self, temp_dir):
        """Test reading nonexistent blob raises FileNotFoundError."""
        backend = FilesystemBlobBackend(temp_dir)
        
        with pytest.raises(FileNotFoundError):
            backend.read_blob("nonexistent")
    
    def test_get_size(self, temp_dir):
        """Test get_size method."""
        backend = FilesystemBlobBackend(temp_dir)
        
        data = b"Hello, World!"
        blob_path = backend.write_blob("test_key", data)
        
        assert backend.get_size(blob_path) == len(data)
    
    def test_write_blob_stream(self, temp_dir):
        """Test writing from a stream."""
        backend = FilesystemBlobBackend(temp_dir)
        
        data = b"Stream data"
        stream = BytesIO(data)
        
        blob_path = backend.write_blob_stream("stream_key", stream)
        result = backend.read_blob(blob_path)
        
        assert result == data
    
    def test_read_blob_stream(self, temp_dir):
        """Test reading as a stream."""
        backend = FilesystemBlobBackend(temp_dir)
        
        data = b"Stream data"
        blob_path = backend.write_blob("stream_key", data)
        
        stream = backend.read_blob_stream(blob_path)
        result = stream.read()
        stream.close()
        
        assert result == data
    
    def test_nested_blob_id(self, temp_dir):
        """Test blob ID with path separators."""
        backend = FilesystemBlobBackend(temp_dir)
        
        data = b"nested data"
        blob_path = backend.write_blob("folder/subfolder/key", data)
        
        assert backend.exists(blob_path)
        assert backend.read_blob(blob_path) == data


class TestBackendValidation:
    """Test backend class validation during registration."""
    
    def test_instance_instead_of_class(self, clean_registry, temp_dir):
        """Test that passing an instance instead of class raises error."""
        instance = MockBlobBackend(base_dir=str(temp_dir))
        
        with pytest.raises(ValueError, match="must be a class"):
            register_blob_backend("mock", instance)


class TestConcurrency:
    """Test thread safety of backend registry operations."""
    
    def test_concurrent_registration(self, clean_registry):
        """Test that concurrent registrations don't corrupt registry."""
        import threading
        
        errors = []
        
        def register_backend(name):
            try:
                # Create a unique backend class
                class TempBackend(MockBlobBackend):
                    pass
                
                register_blob_backend(name, TempBackend)
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
        backends = list_blob_backends()
        names = [b["name"] for b in backends]
        for i in range(10):
            assert f"backend_{i}" in names


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for blob backend registry."""
    
    def test_full_workflow(self, clean_registry, temp_dir):
        """Test complete workflow: register, use, unregister."""
        # Register custom backend
        register_blob_backend("mock", MockBlobBackend)
        
        # List should include it
        backends = list_blob_backends()
        names = [b["name"] for b in backends]
        assert "mock" in names
        
        # Create instance and use it
        backend = get_blob_backend("mock", base_dir=str(temp_dir))
        blob_path = backend.write_blob("key1", b"value")
        assert backend.read_blob(blob_path) == b"value"
        
        # Unregister
        unregister_blob_backend("mock")
        
        # Should no longer be available
        backends = list_blob_backends()
        names = [b["name"] for b in backends]
        assert "mock" not in names
        
        # Getting it should raise error
        with pytest.raises(ValueError):
            get_blob_backend("mock")
    
    def test_filesystem_backend_large_blob(self, temp_dir):
        """Test filesystem backend with large blob."""
        backend = FilesystemBlobBackend(temp_dir)
        
        # Create 1MB of data
        data = b"x" * (1024 * 1024)
        
        blob_path = backend.write_blob("large_blob", data)
        result = backend.read_blob(blob_path)
        
        assert len(result) == len(data)
        assert result == data
    
    def test_replace_builtin_backend(self, clean_registry, temp_dir):
        """Test replacing a builtin backend with custom implementation."""
        # Replace filesystem backend
        register_blob_backend("filesystem", MockBlobBackend, force=True)
        
        # Get should return our mock
        backend = get_blob_backend("filesystem", base_dir=str(temp_dir))
        assert isinstance(backend, MockBlobBackend)
        
        # Verify the class has been replaced
        backends = list_blob_backends()
        fs_info = next(b for b in backends if b["name"] == "filesystem")
        assert fs_info["class"] == "MockBlobBackend"
        # Note: is_builtin is determined by name, so it remains True
        assert fs_info["is_builtin"] is True
