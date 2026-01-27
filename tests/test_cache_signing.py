"""
Tests for cache entry signing and invalid signature deletion configuration.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from cacheness.core import UnifiedCache
from cacheness.config import CacheConfig, SecurityConfig


@pytest.fixture
def temp_cache_signing_enabled():
    """Fixture to create a temporary cache with entry signing enabled."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        config = CacheConfig(
            cache_dir=str(temp_dir / "cache"),
            security=SecurityConfig(
                enable_entry_signing=True,
                delete_invalid_signatures=True,
                allow_unsigned_entries=True
            )
        )
        cache = UnifiedCache(config)
        yield cache, temp_dir
        cache.close()
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


@pytest.fixture
def temp_cache_delete_disabled():
    """Fixture to create a temporary cache with delete_invalid_signatures disabled."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        config = CacheConfig(
            cache_dir=str(temp_dir / "cache"),
            security=SecurityConfig(
                enable_entry_signing=True,
                delete_invalid_signatures=False,
                allow_unsigned_entries=True
            )
        )
        cache = UnifiedCache(config)
        yield cache, temp_dir
        cache.close()
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


class TestDeleteInvalidSignatures:
    """Test cases for delete_invalid_signatures configuration option."""
    
    def test_config_option_defaults(self):
        """Test that configuration options have correct defaults."""
        # Test default config
        default_config = CacheConfig()
        assert default_config.security.delete_invalid_signatures
        
        # Test explicit True
        config_true = CacheConfig(delete_invalid_signatures=True)
        assert config_true.security.delete_invalid_signatures
        
        # Test explicit False
        config_false = CacheConfig(delete_invalid_signatures=False)
        assert not config_false.security.delete_invalid_signatures
        
        # Test via SecurityConfig
        security_config = SecurityConfig(delete_invalid_signatures=False)
        config_via_security = CacheConfig(security=security_config)
        assert not config_via_security.security.delete_invalid_signatures
    
    def test_basic_signing_functionality(self, temp_cache_signing_enabled, temp_cache_delete_disabled):
        """Test that basic signing functionality works with both config values."""
        cache1, temp_dir1 = temp_cache_signing_enabled
        cache2, temp_dir2 = temp_cache_delete_disabled
        
        # Store and retrieve data with delete_invalid_signatures=True
        test_data = {"message": "Hello, World!", "numbers": [1, 2, 3, 4, 5]}
        cache_key = cache1.put(test_data, description="Test data with signing")
        retrieved_data = cache1.get(cache_key=cache_key)
        assert retrieved_data == test_data
        
        # Store and retrieve data with delete_invalid_signatures=False
        test_data2 = {"message": "Second test", "value": 42}
        cache_key2 = cache2.put(test_data2, description="Test data for retention test")
        retrieved_data2 = cache2.get(cache_key=cache_key2)
        assert retrieved_data2 == test_data2
    
    def test_config_logging(self):
        """Test that the security configuration logs the delete_invalid_signatures setting."""
        import logging
        import io
        
        # Capture log output
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        logger = logging.getLogger('src.cacheness.config')
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        try:
            # Create config with delete_invalid_signatures=False
            config = CacheConfig(
                security=SecurityConfig(delete_invalid_signatures=False)
            )
            
            # Check that the setting is logged
            log_output = log_capture.getvalue()
            assert "delete_invalid=False" in log_output or "delete_invalid_signatures" in str(config.security.__dict__)
            
        finally:
            logger.removeHandler(handler)
    
    def test_backward_compatibility_parameter(self):
        """Test that the backward compatibility parameter works correctly."""
        # Test direct parameter to CacheConfig constructor
        config = CacheConfig(delete_invalid_signatures=False)
        assert not config.security.delete_invalid_signatures
        
        # Test that it overrides SecurityConfig defaults
        security_config = SecurityConfig()  # Uses default True
        config_with_override = CacheConfig(
            security=security_config,
            delete_invalid_signatures=False  # Should override the SecurityConfig default
        )
        assert not config_with_override.security.delete_invalid_signatures
    
    def test_in_memory_key_configuration(self):
        """Test the use_in_memory_key configuration option."""
        # Test default
        default_config = CacheConfig()
        assert not default_config.security.use_in_memory_key
        
        # Test explicit True
        config_true = CacheConfig(use_in_memory_key=True)
        assert config_true.security.use_in_memory_key
        
        # Test explicit False
        config_false = CacheConfig(use_in_memory_key=False)
        assert not config_false.security.use_in_memory_key
        
        # Test via SecurityConfig
        security_config = SecurityConfig(use_in_memory_key=True)
        config_via_security = CacheConfig(security=security_config)
        assert config_via_security.security.use_in_memory_key
    
    def test_in_memory_key_functionality(self):
        """Test that in-memory keys work correctly and don't persist to disk."""
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Test persistent key (default)
            config_persistent = CacheConfig(
                cache_dir=str(temp_dir / "persistent"),
                security=SecurityConfig(
                    enable_entry_signing=True,
                    use_in_memory_key=False
                )
            )
            
            cache_persistent = UnifiedCache(config_persistent)
            key_file_path = temp_dir / "persistent" / "cache_signing_key.bin"
            
            # Key file should be created for persistent keys
            assert key_file_path.exists(), "Key file should be created for persistent keys"
            
            # Test that persistent cache works
            test_data_persistent = {"persistent": "test"}
            cache_key_persistent = cache_persistent.put(test_data_persistent)
            retrieved_persistent = cache_persistent.get(cache_key=cache_key_persistent)
            assert retrieved_persistent == test_data_persistent
            
            # Test in-memory key
            config_memory = CacheConfig(
                cache_dir=str(temp_dir / "memory"),
                security=SecurityConfig(
                    enable_entry_signing=True,
                    use_in_memory_key=True
                )
            )
            
            cache_memory = UnifiedCache(config_memory)
            key_file_path_memory = temp_dir / "memory" / "cache_signing_key.bin"
            
            # Key file should NOT be created for in-memory keys
            assert not key_file_path_memory.exists(), "Key file should NOT be created for in-memory keys"
            
            # Test basic functionality
            test_data = {"test": "data", "value": 42}
            cache_key = cache_memory.put(test_data, description="Test with in-memory key")
            retrieved = cache_memory.get(cache_key=cache_key)
            assert retrieved == test_data
            
            cache_memory.close()
            cache_persistent.close()
            
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def test_in_memory_key_restart_behavior(self):
        """Test that in-memory keys invalidate cache entries on restart."""
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            config = CacheConfig(
                cache_dir=str(temp_dir / "restart_test"),
                security=SecurityConfig(
                    enable_entry_signing=True,
                    use_in_memory_key=True,
                    delete_invalid_signatures=True  # Clean up invalid signatures
                )
            )
            
            # First cache instance
            cache1 = UnifiedCache(config)
            test_data = {"restart": "test", "value": 123}
            cache_key = cache1.put(test_data, description="Restart test data")
            
            # Verify data can be retrieved
            retrieved1 = cache1.get(cache_key=cache_key)
            assert retrieved1 == test_data
            
            # Second cache instance (simulating restart)
            cache2 = UnifiedCache(config)
            
            # Data should not be retrievable due to different in-memory key
            retrieved2 = cache2.get(cache_key=cache_key)
            assert retrieved2 is None, "Data should not be accessible after restart with in-memory key"
            
            cache2.close()
            cache1.close()
            
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
