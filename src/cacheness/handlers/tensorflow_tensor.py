"""Handler for TensorFlow tensors using blosc2.save_tensor/load_tensor."""

from pathlib import Path
from typing import Any


from ._compat import (
    CacheHandler,
    CacheWriteError,
    CacheReadError,
    HandlerResult,
    BlobReadContext,
    cache_operation_context,
    logger,
    _lazy_import_tensorflow,
    BLOSC2_AVAILABLE,
    blosc2,
)


class TensorFlowTensorHandler(CacheHandler):
    """Handler for TensorFlow tensors using blosc2.save_tensor/load_tensor."""

    priority: int = 45

    def can_handle(self, data: Any) -> bool:
        """Check if data is a TensorFlow tensor that can be cached."""
        # Quick check for obviously non-tensor types before importing TensorFlow
        if isinstance(data, (str, int, float, bool, list, tuple, dict)):
            return False

        # Also exclude numpy arrays (they have their own handler)
        if hasattr(data, "__array__") and hasattr(data, "dtype"):
            # This is likely a numpy array or similar
            return False

        # Only import TensorFlow if we have a potential tensor-like object
        tf_module, tf_available = _lazy_import_tensorflow()
        if not tf_available or tf_module is None:
            return False
        if not BLOSC2_AVAILABLE or blosc2 is None:
            return False

        # Check if it's a TensorFlow tensor (EagerTensor, Variable, etc.)
        try:
            return isinstance(data, (tf_module.Tensor, tf_module.Variable))
        except Exception:  # intentionally broad — TF isinstance may fail
            return False

    def put(self, data: Any, file_path: Path, config: Any) -> HandlerResult:
        """Store TensorFlow tensor using blosc2.save_tensor with proper error handling."""
        tf_module, tf_available = _lazy_import_tensorflow()
        if not tf_available or tf_module is None:
            raise CacheWriteError(
                "TensorFlow not available for storing tensor",
                handler_type="tensorflow_tensor",
                data_type=type(data).__name__,
            )

        with cache_operation_context(
            "store_tensorflow_tensor",
            shape=data.shape.as_list()
            if hasattr(data.shape, "as_list")
            else str(data.shape),
            dtype=str(data.dtype),
        ):
            try:
                # Convert to tensor if it's a Variable
                if isinstance(data, tf_module.Variable):
                    tensor_data = data.value()
                else:
                    tensor_data = data

                b2tr_path = file_path.with_suffix("").with_suffix(".b2tr")

                logger.debug(
                    f"Writing TensorFlow tensor to {b2tr_path} with blosc2 compression"
                )

                # Use blosc2.save_tensor for optimized tensor storage
                blosc2.save_tensor(
                    tensor_data.numpy(),  # Convert to numpy for blosc2
                    str(b2tr_path),
                    cparams={
                        "clevel": config.compression.blosc2_array_clevel,
                        "codec": getattr(
                            blosc2.Codec,
                            config.compression.blosc2_array_codec.upper(),
                            blosc2.Codec.LZ4,
                        ),
                    },
                )

                file_size = b2tr_path.stat().st_size
                logger.debug(
                    f"TensorFlow tensor written successfully: {file_size} bytes"
                )

                return HandlerResult(
                    storage_format="blosc2_tensor",
                    file_size=file_size,
                    actual_path=str(b2tr_path),
                    compression_codec=config.compression.blosc2_array_codec,
                    extra={
                        "shape": tensor_data.shape.as_list()
                        if hasattr(tensor_data.shape, "as_list")
                        else list(tensor_data.shape),
                        "dtype": str(tensor_data.dtype),
                        "was_variable": isinstance(data, tf_module.Variable),
                    },
                )

            except Exception as e:  # intentionally broad — re-raises as CacheWriteError
                raise CacheWriteError(
                    f"Failed to write TensorFlow tensor with blosc2: {e}",
                    handler_type="tensorflow_tensor",
                    data_type=type(data).__name__,
                ) from e

    def get(self, file_path: Path, metadata: BlobReadContext) -> Any:
        """Load TensorFlow tensor from blosc2 tensor file with proper error handling."""
        tf_module, tf_available = _lazy_import_tensorflow()

        with cache_operation_context(
            "load_tensorflow_tensor", file_path=str(file_path)
        ):
            try:
                if not tf_available or tf_module is None:
                    raise CacheReadError(
                        "TensorFlow not available for loading tensor",
                        handler_type="tensorflow_tensor",
                    )

                if not BLOSC2_AVAILABLE or blosc2 is None:
                    raise CacheReadError(
                        "blosc2 not available for loading tensor",
                        handler_type="tensorflow_tensor",
                    )

                logger.debug(f"Reading TensorFlow tensor from {file_path}")

                # Load tensor using blosc2.load_tensor
                numpy_array = blosc2.load_tensor(str(file_path))

                # Convert numpy array back to TensorFlow tensor
                tensor = tf_module.constant(numpy_array)

                # If it was originally a Variable, convert back to Variable
                if metadata.get("metadata", {}).get("was_variable", False):
                    tensor = tf_module.Variable(tensor)

                logger.debug(f"Loaded TensorFlow tensor with shape {tensor.shape}")
                return tensor

            except Exception as e:  # intentionally broad — re-raises as CacheReadError
                raise CacheReadError(
                    f"Failed to load TensorFlow tensor from {file_path}: {e}",
                    handler_type="tensorflow_tensor",
                ) from e

    def get_file_extension(self, config: Any) -> str:
        """Return the file extension for TensorFlow tensor files."""
        return "b2tr"

    @property
    def data_type(self) -> str:
        """Return the data type handled by this handler."""
        return "tensorflow_tensor"
