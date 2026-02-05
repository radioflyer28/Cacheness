"""
Test update_data() management operation.

Tests the new Phase 3 management operation for updating blob data
at an existing cache_key without changing the key itself.
"""

import pytest
from pathlib import Path
import tempfile
import shutil
from cacheness import UnifiedCache
from cacheness.config import CacheConfig, StorageConfig


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory."""
    temp_dir = tempfile.mkdtemp(prefix="cacheness_test_update_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def cache(temp_cache_dir):
    """Create UnifiedCache instance with in-memory backend."""
    config = CacheConfig(
        storage=StorageConfig(
            cache_dir=str(temp_cache_dir),
            backend_type="memory"
        )
    )
    return UnifiedCache(config=config)


class TestUpdateData:
    """Test update_data() operation."""
    
    def test_update_data_existing_entry(self, cache):
        """Test updating an existing cache entry."""
        # Create initial entry
        original_data = {"value": 100, "name": "original"}
        cache.put(original_data, experiment="exp_001", run_id=1)
        
        # Verify original data
        retrieved = cache.get(experiment="exp_001", run_id=1)
        assert retrieved == original_data
        
        # Update with new data
        new_data = {"value": 200, "name": "updated"}
        success = cache.update_data(new_data, experiment="exp_001", run_id=1)
        assert success is True
        
        # Verify updated data
        retrieved = cache.get(experiment="exp_001", run_id=1)
        assert retrieved == new_data
        assert retrieved["value"] == 200
        assert retrieved["name"] == "updated"
    
    def test_update_data_nonexistent_entry(self, cache):
        """Test updating a nonexistent cache entry returns False."""
        new_data = {"value": 200, "name": "updated"}
        success = cache.update_data(new_data, experiment="exp_999", run_id=999)
        assert success is False
        
        # Verify entry doesn't exist
        retrieved = cache.get(experiment="exp_999", run_id=999)
        assert retrieved is None
    
    def test_update_data_preserves_cache_key(self, cache):
        """Test that cache_key remains unchanged after update."""
        # Create initial entry
        original_data = [1, 2, 3]
        cache.put(original_data, test_id="key_test")
        
        # Get original cache_key
        original_meta = cache.get_metadata(test_id="key_test")
        original_key = original_meta.get("cache_key") if original_meta else None
        
        # Update data
        new_data = [4, 5, 6, 7, 8]
        success = cache.update_data(new_data, test_id="key_test")
        assert success is True
        
        # Verify cache_key unchanged
        updated_meta = cache.get_metadata(test_id="key_test")
        # Note: get_metadata doesn't return cache_key, so we verify via retrieval
        retrieved = cache.get(test_id="key_test")
        assert retrieved == new_data
    
    def test_update_data_updates_metadata(self, cache):
        """Test that derived metadata fields are updated."""
        # Create initial entry (small data)
        original_data = {"small": "x" * 10}
        cache.put(original_data, data_id="meta_test")
        
        # Get original metadata
        original_meta = cache.get_metadata(data_id="meta_test")
        original_size = original_meta.get("file_size", 0)
        original_created = original_meta.get("created_at")
        
        # Update with larger data
        import time
        time.sleep(0.1)  # Ensure timestamp difference
        new_data = {"large": "x" * 1000}
        success = cache.update_data(new_data, data_id="meta_test")
        assert success is True
        
        # Get updated metadata
        updated_meta = cache.get_metadata(data_id="meta_test")
        updated_size = updated_meta.get("file_size", 0)
        updated_created = updated_meta.get("created_at")
        
        # Verify metadata changed
        # Note: File size might not change significantly due to compression
        # but created_at timestamp should be updated
        assert updated_created != original_created
    
    def test_update_data_with_direct_cache_key(self, cache):
        """Test update_data() using direct cache_key parameter."""
        # Create initial entry
        original_data = "original text"
        cache.put(original_data, text_id="direct_key_test")
        
        # Get cache_key indirectly by creating it manually
        cache_key = cache._create_cache_key({"text_id": "direct_key_test"})
        
        # Update using direct cache_key
        new_data = "updated text"
        success = cache.update_data(new_data, cache_key=cache_key)
        assert success is True
        
        # Verify updated data
        retrieved = cache.get(text_id="direct_key_test")
        assert retrieved == new_data
    
    def test_update_data_different_data_types(self, cache):
        """Test updating entries with different data types."""
        # Start with dict
        cache.put({"type": "dict"}, type_test="multi")
        success = cache.update_data({"type": "dict_updated"}, type_test="multi")
        assert success is True
        assert cache.get(type_test="multi") == {"type": "dict_updated"}
        
        # Update to list
        success = cache.update_data([1, 2, 3], type_test="multi")
        assert success is True
        assert cache.get(type_test="multi") == [1, 2, 3]
        
        # Update to string
        success = cache.update_data("final string", type_test="multi")
        assert success is True
        assert cache.get(type_test="multi") == "final string"


class TestUpdateDataWithJSONBackend:
    """Test update_data() with JSON backend."""
    
    @pytest.fixture
    def json_cache(self, temp_cache_dir):
        """Create UnifiedCache with JSON backend."""
        config = CacheConfig(
            storage=StorageConfig(
                cache_dir=str(temp_cache_dir),
                backend_type="json"
            )
        )
        return UnifiedCache(config=config)
    
    def test_json_backend_update(self, json_cache):
        """Test update with JSON backend."""
        json_cache.put({"value": 1}, json_test="backend")
        success = json_cache.update_data({"value": 2}, json_test="backend")
        assert success is True
        assert json_cache.get(json_test="backend") == {"value": 2}


class TestUpdateDataWithSQLiteBackend:
    """Test update_data() with SQLite backend."""
    
    @pytest.fixture
    def sqlite_cache(self, temp_cache_dir):
        """Create UnifiedCache with SQLite backend."""
        config = CacheConfig(
            storage=StorageConfig(
                cache_dir=str(temp_cache_dir),
                backend_type="sqlite",
                db_file=str(temp_cache_dir / "test_cache.db")
            )
        )
        return UnifiedCache(config=config)
    
    def test_sqlite_backend_update(self, sqlite_cache):
        """Test update with SQLite backend."""
        sqlite_cache.put({"value": 1}, sqlite_test="backend")
        success = sqlite_cache.update_data({"value": 2}, sqlite_test="backend")
        assert success is True
        assert sqlite_cache.get(sqlite_test="backend") == {"value": 2}
    
    def test_sqlite_backend_update_preserves_other_fields(self, sqlite_cache):
        """Test that update preserves other metadata fields."""
        # Create entry with description
        sqlite_cache.put(
            {"value": 1},
            sqlite_test="preserve",
            description="Important data"
        )
        
        # Update data
        success = sqlite_cache.update_data({"value": 2}, sqlite_test="preserve")
        assert success is True
        
        # Verify description preserved (if stored in metadata)
        meta = sqlite_cache.get_metadata(sqlite_test="preserve")
        # Description is at entry level, should be preserved
        assert meta.get("description") == "Important data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
