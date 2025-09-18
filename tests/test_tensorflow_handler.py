"""
Tests for TensorFlow tensor handler.

These tests verify that TensorFlow tensors can be cached and retrieved correctly
using the blosc2 tensor compression format.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from cacheness.handlers import TensorFlowTensorHandler, TENSORFLOW_AVAILABLE, BLOSC2_AVAILABLE
from cacheness.config import CacheConfig


class TestTensorFlowTensorHandler:
    """Test the TensorFlow tensor handler functionality."""

    @pytest.fixture(autouse=True)
    def setup_handler(self):
        """Set up test fixtures."""
        self.config = CacheConfig()
        self.handler = TensorFlowTensorHandler()

    @pytest.mark.skipif(
        not TENSORFLOW_AVAILABLE or not BLOSC2_AVAILABLE,
        reason="TensorFlow or blosc2 not available"
    )
    def test_can_handle_tensorflow_tensor(self):
        """Test that the handler correctly identifies TensorFlow tensors."""
        import tensorflow as tf
        
        # Test with regular tensor
        tensor = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
        assert self.handler.can_handle(tensor)
        
        # Test with Variable
        variable = tf.Variable([[1, 2], [3, 4]], dtype=tf.float32)
        assert self.handler.can_handle(variable)
        
        # Test with non-tensor data
        assert not self.handler.can_handle([1, 2, 3])
        assert not self.handler.can_handle(np.array([1, 2, 3]))

    @pytest.mark.skipif(
        not TENSORFLOW_AVAILABLE or not BLOSC2_AVAILABLE,
        reason="TensorFlow or blosc2 not available"
    )
    def test_put_and_get_tensor(self):
        """Test storing and retrieving a TensorFlow tensor."""
        import tensorflow as tf
        import blosc2
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test tensor
            original_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
            file_path = Path(temp_dir) / "test_tensor"
            
            # Store tensor
            metadata = self.handler.put(original_tensor, file_path, self.config)
            
            # Verify metadata
            assert metadata["storage_format"] == "blosc2_tensor"
            assert "file_size" in metadata
            assert metadata["actual_path"].endswith(".b2tr")
            assert metadata["metadata"]["shape"] == [2, 2]
            assert "float32" in metadata["metadata"]["dtype"]
            assert not metadata["metadata"]["was_variable"]
            
            # Verify file exists
            b2tr_path = Path(metadata["actual_path"])
            assert b2tr_path.exists()
            
            # Load tensor back
            loaded_tensor = self.handler.get(file_path, metadata)
            
            # Verify tensor equality
            assert isinstance(loaded_tensor, tf.Tensor)
            assert loaded_tensor.shape == original_tensor.shape
            assert loaded_tensor.dtype == original_tensor.dtype
            np.testing.assert_array_equal(loaded_tensor.numpy(), original_tensor.numpy())

    @pytest.mark.skipif(
        not TENSORFLOW_AVAILABLE or not BLOSC2_AVAILABLE,
        reason="TensorFlow or blosc2 not available"
    )
    def test_put_and_get_variable(self):
        """Test storing and retrieving a TensorFlow Variable."""
        import tensorflow as tf
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test variable
            original_variable = tf.Variable([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
            file_path = Path(temp_dir) / "test_variable"
            
            # Store variable
            metadata = self.handler.put(original_variable, file_path, self.config)
            
            # Verify metadata indicates it was a variable
            assert metadata["metadata"]["was_variable"] is True
            
            # Load variable back
            loaded_tensor = self.handler.get(file_path, metadata)
            
            # Verify it's converted back to Variable
            assert isinstance(loaded_tensor, tf.Variable)
            assert loaded_tensor.shape == original_variable.shape
            assert loaded_tensor.dtype == original_variable.dtype
            np.testing.assert_array_equal(loaded_tensor.numpy(), original_variable.numpy())

    @pytest.mark.skipif(
        not TENSORFLOW_AVAILABLE or not BLOSC2_AVAILABLE,
        reason="TensorFlow or blosc2 not available"
    )
    def test_large_tensor_compression(self):
        """Test that large tensors are compressed efficiently."""
        import tensorflow as tf
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create large tensor
            large_tensor = tf.constant(np.random.random((100, 100)), dtype=tf.float32)
            file_path = Path(temp_dir) / "large_tensor"
            
            # Store tensor
            metadata = self.handler.put(large_tensor, file_path, self.config)
            
            # Check compression is working (file should be smaller than raw data)
            raw_size = large_tensor.numpy().nbytes
            compressed_size = metadata["file_size"]
            
            # Blosc2 should provide some compression for random data
            assert compressed_size < raw_size
            
            # Load and verify
            loaded_tensor = self.handler.get(file_path, metadata)
            np.testing.assert_array_equal(loaded_tensor.numpy(), large_tensor.numpy())

    def test_can_handle_without_tensorflow(self):
        """Test handler behavior when TensorFlow is not available."""
        with patch('cacheness.handlers.TENSORFLOW_AVAILABLE', False):
            handler = TensorFlowTensorHandler()
            assert not handler.can_handle([1, 2, 3])
            
    def test_can_handle_without_blosc2(self):
        """Test handler behavior when blosc2 is not available."""
        with patch('cacheness.handlers.BLOSC2_AVAILABLE', False):
            handler = TensorFlowTensorHandler()
            assert not handler.can_handle([1, 2, 3])

    def test_file_extension(self):
        """Test that the handler returns the correct file extension."""
        assert self.handler.get_file_extension(self.config) == "b2tr"

    def test_data_type(self):
        """Test that the handler returns the correct data type."""
        assert self.handler.data_type == "tensorflow_tensor"

    @pytest.mark.skipif(
        not TENSORFLOW_AVAILABLE or not BLOSC2_AVAILABLE,
        reason="TensorFlow or blosc2 not available"
    )
    def test_error_handling_invalid_path(self):
        """Test error handling for invalid file paths."""
        import tensorflow as tf
        from cacheness.interfaces import CacheReadError
        
        # Try to load from non-existent file
        fake_metadata = {
            "storage_format": "blosc2_tensor",
            "metadata": {"was_variable": False}
        }
        
        with pytest.raises(CacheReadError):
            self.handler.get(Path("/nonexistent/path.b2tr"), fake_metadata)

    @pytest.mark.skipif(
        not TENSORFLOW_AVAILABLE or not BLOSC2_AVAILABLE,
        reason="TensorFlow or blosc2 not available"
    )
    def test_different_dtypes(self):
        """Test handling different tensor data types."""
        import tensorflow as tf
        
        dtypes_to_test = [tf.int32, tf.int64, tf.float32, tf.float64]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for i, dtype in enumerate(dtypes_to_test):
                # Create tensor with specific dtype
                tensor = tf.constant([[1, 2], [3, 4]], dtype=dtype)
                file_path = Path(temp_dir) / f"tensor_{i}"
                
                # Store and load
                metadata = self.handler.put(tensor, file_path, self.config)
                loaded_tensor = self.handler.get(file_path, metadata)
                
                # Verify dtype is preserved
                assert loaded_tensor.dtype == tensor.dtype
                np.testing.assert_array_equal(loaded_tensor.numpy(), tensor.numpy())
