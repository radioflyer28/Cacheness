"""
Cache Format Handlers
====================

This module contains specialized handlers for different data types in the cache system.
Each handler implements focused interfaces following the Interface Segregation Principle.
"""

import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional
import logging

# Import focused interfaces
from .interfaces import (
    CacheHandler,
    CacheHandlerError,
    CacheWriteError,
    CacheReadError,
    CacheFormatError,
)
from .error_handling import cache_operation_context

# DataFrame libraries with fallback
try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    pl = None  # type: ignore
    POLARS_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    pd = None  # type: ignore
    PANDAS_AVAILABLE = False

# Import compressed pickle utilities
from .compress_pickle import (
    write_file as write_compressed_pickle,
    read_file as read_compressed_pickle,
    is_pickleable,
    BLOSC_AVAILABLE,
    optimize_compression_params,
)

# Optional dependency - blosc2 for array compression
try:
    import blosc2

    BLOSC2_AVAILABLE = True
except ImportError:
    blosc2 = None  # type: ignore
    BLOSC2_AVAILABLE = False

# Optional dependency - TensorFlow for tensor compression (completely lazy loaded)
TENSORFLOW_AVAILABLE = False
tf = None
_tensorflow_import_attempted = False

def _lazy_import_tensorflow():
    """Lazy import TensorFlow to avoid slow startup times and system issues."""
    global tf, TENSORFLOW_AVAILABLE, _tensorflow_import_attempted
    
    if _tensorflow_import_attempted:
        return tf, TENSORFLOW_AVAILABLE
        
    _tensorflow_import_attempted = True
    
    try:
        import tensorflow as tf_module
        tf = tf_module
        TENSORFLOW_AVAILABLE = True
        logger.debug("TensorFlow successfully imported")
    except ImportError:
        tf = None
        TENSORFLOW_AVAILABLE = False
        logger.debug("TensorFlow not available (ImportError)")
    except Exception as e:
        tf = None
        TENSORFLOW_AVAILABLE = False
        logger.warning(f"TensorFlow import failed with unexpected error: {e}")
    
    return tf, TENSORFLOW_AVAILABLE

logger = logging.getLogger(__name__)

# Log DataFrame backend availability with debug info
if POLARS_AVAILABLE and PANDAS_AVAILABLE:
    logger.info("📊 Both Polars and Pandas available for DataFrame caching")
elif POLARS_AVAILABLE:
    logger.info("📊 Polars available for DataFrame caching")
elif PANDAS_AVAILABLE:
    logger.info("📊 Pandas available for DataFrame caching (Polars not found)")
else:
    logger.warning(
        "⚠️  Neither Polars nor Pandas available - DataFrame caching disabled"
    )


class PolarsDataFrameHandler(CacheHandler):
    """Handler for Polars DataFrames using Parquet format."""

    def can_handle(self, data: Any) -> bool:
        """Check if data is a Polars DataFrame that can be saved to Parquet."""
        if not POLARS_AVAILABLE or pl is None:
            logger.debug("Polars not available, cannot handle DataFrame")
            return False
        if not isinstance(data, pl.DataFrame):
            return False

        return self.validate_dataframe(data)

    def validate_dataframe(self, data: Any) -> bool:
        """Validate that the DataFrame can be cached in Parquet format."""
        try:
            import io

            data.write_parquet(io.BytesIO())
            logger.debug(f"Polars DataFrame validation passed: shape={data.shape}")
            return True
        except Exception as e:
            logger.debug(f"Polars DataFrame validation failed: {e}")
            return False

    def put(self, data: Any, file_path: Path, config: Any) -> Dict[str, Any]:
        """Store Polars DataFrame as Parquet with proper error handling."""
        with cache_operation_context(
            "store_polars_dataframe", shape=data.shape, columns=len(data.columns)
        ):
            try:
                parquet_path = file_path.with_suffix("").with_suffix(".parquet")
                compression = config.compression.parquet_compression

                logger.debug(
                    f"Writing Polars DataFrame to {parquet_path} with {compression} compression"
                )
                data.write_parquet(parquet_path, compression=compression)

                file_size = parquet_path.stat().st_size
                logger.debug(
                    f"Polars DataFrame written successfully: {file_size} bytes"
                )

                return {
                    "storage_format": "parquet",
                    "file_size": file_size,
                    "actual_path": str(parquet_path),
                    "metadata": {
                        "shape": data.shape,
                        "columns": data.columns,
                        "dtypes": [str(dtype) for dtype in data.dtypes],
                        "compression": compression,
                        "backend": "polars",
                    },
                }

            except Exception as e:
                raise CacheWriteError(
                    f"Failed to write Polars DataFrame to Parquet: {e}",
                    handler_type="polars_dataframe",
                    data_type=type(data).__name__,
                ) from e

    def get(self, file_path: Path, metadata: Dict[str, Any]) -> Any:
        """Load Polars DataFrame from Parquet with proper error handling."""
        with cache_operation_context("load_polars_dataframe", file_path=str(file_path)):
            try:
                if not POLARS_AVAILABLE or pl is None:
                    raise CacheReadError(
                        "Polars not available for loading DataFrame",
                        handler_type="polars_dataframe",
                    )

                logger.debug(f"Reading Polars DataFrame from {file_path}")
                df = pl.read_parquet(file_path)

                # Log successful read with basic stats
                logger.debug(f"Polars DataFrame loaded successfully: shape={df.shape}")
                return df

            except Exception as e:
                if isinstance(e, CacheReadError):
                    raise
                raise CacheReadError(
                    f"Failed to read Polars DataFrame from Parquet: {e}",
                    handler_type="polars_dataframe",
                ) from e

    def get_file_extension(self, config: Any) -> str:
        """Get file extension for Polars DataFrames."""
        return ".parquet"

    @property
    def data_type(self) -> str:
        return "polars_dataframe"


class PandasSeriesHandler(CacheHandler):
    """Handler for Pandas Series using Parquet format."""

    def can_handle(self, data: Any) -> bool:
        """Check if data is a Pandas Series."""
        if not PANDAS_AVAILABLE or pd is None:
            return False
        if not isinstance(data, pd.Series):
            return False

        # Try to convert to Parquet to see if it's compatible
        try:
            temp_df = data.to_frame()
            import io

            temp_df.to_parquet(
                io.BytesIO()
            )  # Keep index for Series compatibility check
            return True
        except Exception:
            # This Series has mixed types that can't be handled by Parquet
            return False

    def put(self, data: Any, file_path: Path, config: Any) -> Dict[str, Any]:
        """Store Pandas Series as Parquet."""
        # Convert Series to DataFrame for Parquet storage
        df = data.to_frame()

        # Use the same Parquet storage logic as DataFrame handler
        parquet_path = file_path.with_suffix("").with_suffix(".parquet")
        df.to_parquet(
            parquet_path,
            compression=config.compression.parquet_compression,
            # Keep index=True for Series to preserve index data
        )

        file_size = parquet_path.stat().st_size
        return {
            "file_size": file_size,
            "actual_path": str(parquet_path),
            "storage_format": "parquet",
            "metadata": {
                "shape": list(df.shape),
                "columns": list(df.columns),
                "dtypes": [str(dtype) for dtype in df.dtypes],
                "compression": config.compression.parquet_compression,
                "backend": "pandas",
                "is_series": True,
                "series_name": data.name,
            },
        }

    def get(self, file_path: Path, metadata: Dict[str, Any]) -> Any:
        """Load Pandas Series from Parquet."""
        df = pd.read_parquet(file_path)

        # Convert back to Series - since we preserved the index, use it
        series = df.iloc[:, 0]  # Get the first (and only) column

        # Restore the original Series name
        if metadata.get("is_series") and "series_name" in metadata:
            series.name = metadata["series_name"]

        return series

    def get_file_extension(self, config: Any) -> str:
        """Get file extension for Series (Parquet)."""
        return ".parquet"

    @property
    def file_extension(self) -> str:
        """Get file extension for Series (Parquet)."""
        return ".parquet"

    @property
    def data_type(self) -> str:
        return "pandas_series"


class PolarsSeriesHandler(CacheHandler):
    """Handler for Polars Series using Parquet format."""

    def can_handle(self, data: Any) -> bool:
        """Check if data is a Polars Series."""
        if not POLARS_AVAILABLE or pl is None:
            return False
        if not isinstance(data, pl.Series):
            return False

        # Try to convert to Parquet to see if it's compatible
        try:
            temp_df = data.to_frame()
            import io

            temp_df.write_parquet(io.BytesIO())
            return True
        except Exception:
            # This Series has mixed types that can't be handled by Parquet
            return False

    def put(self, data: Any, file_path: Path, config: Any) -> Dict[str, Any]:
        """Store Polars Series as Parquet."""
        # Convert Series to DataFrame for Parquet storage
        df = data.to_frame()

        # Use the same Parquet storage logic as DataFrame handler
        parquet_path = file_path.with_suffix("").with_suffix(".parquet")
        df.write_parquet(
            parquet_path, compression=config.compression.parquet_compression
        )

        file_size = parquet_path.stat().st_size
        return {
            "file_size": file_size,
            "actual_path": str(parquet_path),
            "storage_format": "parquet",
            "metadata": {
                "shape": list(df.shape),
                "columns": list(df.columns),
                "dtypes": [str(dtype) for dtype in df.dtypes],
                "compression": config.compression.parquet_compression,
                "backend": "polars",
                "is_series": True,
                "series_name": data.name,
            },
        }

    def get(self, file_path: Path, metadata: Dict[str, Any]) -> Any:
        """Load Polars Series from Parquet."""
        df = pl.read_parquet(file_path)

        # Convert back to Series
        series = df.to_series(0)  # Get the first (and only) column

        # Restore the original Series name
        if metadata.get("is_series") and "series_name" in metadata:
            series = series.alias(metadata["series_name"])

        return series

    def get_file_extension(self, config: Any) -> str:
        """Get file extension for Series (Parquet)."""
        return ".parquet"

    @property
    def file_extension(self) -> str:
        """Get file extension for Series (Parquet)."""
        return ".parquet"

    @property
    def data_type(self) -> str:
        return "polars_series"


class PandasDataFrameHandler(CacheHandler):
    """Handler for Pandas DataFrames using Parquet format."""

    def can_handle(self, data: Any) -> bool:
        """Check if data is a Pandas DataFrame."""
        if not PANDAS_AVAILABLE or pd is None:
            return False
        if not isinstance(data, pd.DataFrame):
            return False

        # Check if DataFrame can be written to Parquet
        try:
            import io

            data.to_parquet(io.BytesIO())
            return True
        except Exception:
            # DataFrame has types that can't be written to Parquet, let ObjectHandler handle it
            return False

    def put(self, data: Any, file_path: Path, config: Any) -> Dict[str, Any]:
        """Store Pandas DataFrame as Parquet."""
        parquet_path = file_path.with_suffix("").with_suffix(".parquet")
        data.to_parquet(
            parquet_path,
            compression=config.compression.parquet_compression,
            # Keep index=True by default to preserve DataFrame index
        )

        return {
            "storage_format": "parquet",
            "file_size": parquet_path.stat().st_size,
            "actual_path": str(parquet_path),
            "metadata": {
                "shape": data.shape,
                "columns": data.columns.tolist(),
                "dtypes": [str(dtype) for dtype in data.dtypes],
                "compression": config.compression.parquet_compression,
                "backend": "pandas",
            },
        }

    def get(self, file_path: Path, metadata: Dict[str, Any]) -> Any:
        """Load Pandas DataFrame from Parquet."""
        if not PANDAS_AVAILABLE or pd is None:
            raise ImportError("Pandas not available for loading DataFrame")

        return pd.read_parquet(file_path)

    def get_file_extension(self, config: Any) -> str:
        """Get file extension for Pandas DataFrames and Series."""
        return ".parquet"

    @property
    def data_type(self) -> str:
        return "pandas_dataframe"


class ArrayHandler(CacheHandler):
    """Handler for NumPy arrays using blosc2 or NPZ format."""

    def can_handle(self, data: Any) -> bool:
        """Check if data is a NumPy array or dict of arrays."""
        if isinstance(data, np.ndarray):
            return True
        if isinstance(data, dict):
            return all(isinstance(v, np.ndarray) for v in data.values())
        return False

    def put(self, data: Any, file_path: Path, config: Any) -> Dict[str, Any]:
        """Store array(s) using optimal format."""
        if isinstance(data, np.ndarray):
            return self._put_single_array(data, file_path, config)
        elif isinstance(data, dict):
            return self._put_array_dict(data, file_path, config)
        else:
            raise ValueError(
                "ArrayHandler can only handle np.ndarray or Dict[str, np.ndarray]"
            )

    def _put_single_array(
        self, data: np.ndarray, file_path: Path, config: Any
    ) -> Dict[str, Any]:
        """Store a single numpy array, trying blosc2 first, then NPZ fallback."""
        # Try blosc2 compression first if enabled
        if config.compression.use_blosc2_arrays and BLOSC2_AVAILABLE:
            try:
                # Update file path for blosc2 format
                blosc2_path = file_path.with_suffix("").with_suffix(".b2nd")
                self._write_blosc2_array(data, blosc2_path, config)

                return {
                    "storage_format": "blosc2",
                    "file_size": blosc2_path.stat().st_size,
                    "actual_path": str(blosc2_path),
                    "metadata": {
                        "shape": data.shape,
                        "dtype": str(data.dtype),
                        "storage_format": "blosc2",
                        "compression": config.compression.blosc2_array_codec,
                    },
                }
            except Exception as e:
                logger.warning(f"blosc2 compression failed, falling back to NPZ: {e}")

        # Fallback to NPZ format
        npz_path = file_path.with_suffix("").with_suffix(".npz")
        if config.compression.npz_compression:
            np.savez_compressed(npz_path, data=data)
        else:
            np.savez(npz_path, data=data)

        return {
            "storage_format": "npz",
            "file_size": npz_path.stat().st_size,
            "actual_path": str(npz_path),
            "metadata": {
                "shape": data.shape,
                "dtype": str(data.dtype),
                "storage_format": "npz",
                "compression": "zlib" if config.compression.npz_compression else "none",
            },
        }

    def _put_array_dict(
        self, data: Dict[str, np.ndarray], file_path: Path, config: Any
    ) -> Dict[str, Any]:
        """Store a dictionary of arrays using NPZ format."""
        # Filter to only numpy arrays
        array_data = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}

        npz_path = file_path.with_suffix("").with_suffix(".npz")
        if config.compression.npz_compression:
            np.savez_compressed(npz_path, **array_data)
        else:
            np.savez(npz_path, **array_data)

        return {
            "storage_format": "npz",
            "file_size": npz_path.stat().st_size,
            "actual_path": str(npz_path),
            "metadata": {
                "arrays": {
                    key: {"shape": arr.shape, "dtype": str(arr.dtype)}
                    for key, arr in array_data.items()
                },
                "storage_format": "npz",
                "compression": "zlib" if config.compression.npz_compression else "none",
            },
        }

    def _write_blosc2_array(
        self, data: np.ndarray, file_path: Path, config: Any
    ) -> None:
        """Write numpy array to file using blosc2 compression with metadata."""
        if not BLOSC2_AVAILABLE:
            raise ImportError(
                "blosc2 is required for array compression but is not available"
            )

        # Compress the array data using blosc2.compress2 (no 2GB limit)
        compressed_data = blosc2.compress2(
            data,
            cparams={
                'typesize': data.dtype.itemsize,
                'clevel': config.compression.blosc2_array_clevel,
                'codec': getattr(
                    blosc2.Codec,
                    config.compression.blosc2_array_codec.upper(),
                    blosc2.Codec.LZ4,
                ),
            },
        )

        # Write compressed data and metadata to file
        with open(file_path, "wb") as f:
            # Write shape and dtype info first
            shape_bytes = str(data.shape).encode("utf-8")
            dtype_bytes = str(data.dtype).encode("utf-8")

            # Write metadata lengths and data
            f.write(len(shape_bytes).to_bytes(4, "little"))
            f.write(shape_bytes)
            f.write(len(dtype_bytes).to_bytes(4, "little"))
            f.write(dtype_bytes)
            f.write(compressed_data)

    def _read_blosc2_array(self, file_path: Path) -> np.ndarray:
        """Read numpy array from blosc2 compressed file with metadata."""
        if not BLOSC2_AVAILABLE:
            raise ImportError(
                "blosc2 is required for array decompression but is not available"
            )

        with open(file_path, "rb") as f:
            # Read shape and dtype metadata
            shape_len = int.from_bytes(f.read(4), "little")
            shape_str = f.read(shape_len).decode("utf-8")
            dtype_len = int.from_bytes(f.read(4), "little")
            dtype_str = f.read(dtype_len).decode("utf-8")

            # Read compressed data
            compressed_data = f.read()

            # Decompress and reconstruct array
            # Decompress the data using blosc2.decompress2 (no 2GB limit)
            decompressed = blosc2.decompress2(compressed_data)
            shape = eval(shape_str)  # Convert string tuple back to tuple
            dtype = np.dtype(dtype_str)
            return np.frombuffer(decompressed, dtype=dtype).reshape(shape)

    def get(self, file_path: Path, metadata: Dict[str, Any]) -> Any:
        """Load array(s) from file with format detection."""
        storage_format = metadata.get("storage_format", "npz")

        # Try the expected format first
        if storage_format == "blosc2":
            try:
                # Try blosc2 format
                blosc2_path = file_path.with_suffix("").with_suffix(".b2nd")
                if blosc2_path.exists():
                    return self._read_blosc2_array(blosc2_path)
            except Exception as e:
                logger.debug(f"Failed to read blosc2 file: {e}")

        # Try NPZ format (fallback or primary)
        npz_path = file_path.with_suffix("").with_suffix(".npz")
        if npz_path.exists():
            data = np.load(npz_path, allow_pickle=True)

            # Return single array if only one, otherwise return dict
            arrays = {key: data[key] for key in data.files}
            if len(arrays) == 1:
                return list(arrays.values())[0]
            return arrays

        raise FileNotFoundError(f"No valid array file found for {file_path}")

    def get_file_extension(self, config: Any) -> str:
        """Get file extension for arrays (determined dynamically)."""
        if config.compression.use_blosc2_arrays and BLOSC2_AVAILABLE:
            return ".b2nd"
        return ".npz"

    @property
    def data_type(self) -> str:
        return "array"


class TensorFlowTensorHandler(CacheHandler):
    """Handler for TensorFlow tensors using blosc2.save_tensor/load_tensor."""

    def can_handle(self, data: Any) -> bool:
        """Check if data is a TensorFlow tensor that can be cached."""
        # Quick check for obviously non-tensor types before importing TensorFlow
        if isinstance(data, (str, int, float, bool, list, tuple, dict)):
            return False
        
        # Also exclude numpy arrays (they have their own handler)
        if hasattr(data, '__array__') and hasattr(data, 'dtype'):
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
        except Exception:
            return False

    def put(self, data: Any, file_path: Path, config: Any) -> Dict[str, Any]:
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
            shape=data.shape.as_list() if hasattr(data.shape, 'as_list') else str(data.shape),
            dtype=str(data.dtype)
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
                        'clevel': config.compression.blosc2_array_clevel,
                        'codec': getattr(
                            blosc2.Codec,
                            config.compression.blosc2_array_codec.upper(),
                            blosc2.Codec.LZ4,
                        ),
                    }
                )

                file_size = b2tr_path.stat().st_size
                logger.debug(
                    f"TensorFlow tensor written successfully: {file_size} bytes"
                )

                return {
                    "storage_format": "blosc2_tensor",
                    "file_size": file_size,
                    "actual_path": str(b2tr_path),
                    "metadata": {
                        "shape": tensor_data.shape.as_list() if hasattr(tensor_data.shape, 'as_list') else list(tensor_data.shape),
                        "dtype": str(tensor_data.dtype),
                        "storage_format": "blosc2_tensor",
                        "compression": config.compression.blosc2_array_codec,
                        "was_variable": isinstance(data, tf_module.Variable),
                    },
                }

            except Exception as e:
                raise CacheWriteError(
                    f"Failed to write TensorFlow tensor with blosc2: {e}",
                    handler_type="tensorflow_tensor",
                    data_type=type(data).__name__,
                ) from e

    def get(self, file_path: Path, metadata: Dict[str, Any]) -> Any:
        """Load TensorFlow tensor from blosc2 tensor file with proper error handling."""
        tf_module, tf_available = _lazy_import_tensorflow()
        
        with cache_operation_context("load_tensorflow_tensor", file_path=str(file_path)):
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

            except Exception as e:
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


class ObjectHandler(CacheHandler):
    """Handler for general Python objects using compressed pickle."""

    def can_handle(self, data: Any) -> bool:
        """Check if data can be pickled (and isn't handled by other handlers)."""
        # Don't handle DataFrames if specialized handlers can handle them
        if POLARS_AVAILABLE and pl is not None and isinstance(data, pl.DataFrame):
            # Check if PolarsDataFrameHandler would reject this (Object types, etc.)
            try:
                import io

                data.write_parquet(io.BytesIO())
                return False  # Specialized handler can handle it
            except Exception:
                return is_pickleable(data)  # We can handle it as fallback

        if PANDAS_AVAILABLE and pd is not None and isinstance(data, pd.DataFrame):
            # Check if PandasDataFrameHandler would reject this (complex objects, etc.)
            try:
                import io

                data.to_parquet(io.BytesIO())
                return False  # Specialized handler can handle it
            except Exception:
                return is_pickleable(data)  # We can handle it as fallback

        # For Series, only handle if specialized handlers can't (i.e., mixed-type Series)
        if POLARS_AVAILABLE and pl is not None and isinstance(data, pl.Series):
            # Check if PolarsSeriesHandler would reject this (mixed types)
            try:
                temp_df = data.to_frame()
                import io

                temp_df.write_parquet(io.BytesIO())
                return False  # Specialized handler can handle it
            except Exception:
                return is_pickleable(data)  # We can handle it as fallback

        if PANDAS_AVAILABLE and pd is not None and isinstance(data, pd.Series):
            # Check if PandasSeriesHandler would reject this (mixed types)
            try:
                temp_df = data.to_frame()
                import io

                temp_df.to_parquet(
                    io.BytesIO()
                )  # Keep index for proper compatibility check
                return False  # Specialized handler can handle it
            except Exception:
                return is_pickleable(data)  # We can handle it as fallback

        # Don't handle arrays - let ArrayHandler do that
        if isinstance(data, np.ndarray):
            return False
        if isinstance(data, dict) and all(
            isinstance(v, np.ndarray) for v in data.values()
        ):
            return False

        return is_pickleable(data)

    def put(self, data: Any, file_path: Path, config: Any) -> Dict[str, Any]:
        """Store object using compressed pickle or fallback to standard pickle."""
        if not is_pickleable(data):
            raise ValueError(
                f"Object of type {type(data)} is not pickleable and cannot be cached"
            )

        # Ensure clean path without existing extension
        if BLOSC_AVAILABLE:
            pickle_path = file_path.with_suffix("").with_suffix(
                f".pkl.{config.compression.pickle_compression_codec}"
            )

            # Optimize compression parameters based on data characteristics
            compression_params = optimize_compression_params(
                data,
                codec=config.compression.pickle_compression_codec,
                base_clevel=config.compression.pickle_compression_level,
                enable_multithreading=getattr(
                    config.compression, "enable_multithreading", True
                ),
                auto_optimize_threads=getattr(
                    config.compression, "auto_optimize_threads", True
                ),
            )

            # Use compressed pickle with optimized parameters
            write_compressed_pickle(
                data,
                pickle_path,
                nparray=False,  # Don't use numpy optimization for general objects
                **compression_params,
            )
            storage_format = "compressed_pickle"
        else:
            # Fall back to standard pickle when blosc is not available
            pickle_path = file_path.with_suffix("").with_suffix(".pkl")
            import pickle

            with open(pickle_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            storage_format = "pickle"

        metadata = {
            "object_type": str(type(data)),
            "storage_format": storage_format,
            "compression_codec": config.compression.pickle_compression_codec
            if BLOSC_AVAILABLE
            else None,
        }

        return {
            "storage_format": storage_format,
            "file_size": pickle_path.stat().st_size,
            "actual_path": str(pickle_path),
            "metadata": metadata,
        }

    def get(self, file_path: Path, metadata: Dict[str, Any]) -> Any:
        """Load object using compressed pickle or standard pickle."""
        storage_format = metadata.get("storage_format", "compressed_pickle")

        if storage_format == "pickle" or not BLOSC_AVAILABLE:
            # Standard pickle fallback
            pickle_path = file_path.with_suffix("").with_suffix(".pkl")
            if not pickle_path.exists():
                # Try compressed version as fallback
                pickle_path = file_path.with_suffix("").with_suffix(
                    f".pkl.{metadata.get('compression_codec', 'zstd')}"
                )

            if not pickle_path.exists():
                raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

            if pickle_path.suffix == ".pkl":
                # Standard pickle
                import pickle

                with open(pickle_path, "rb") as f:
                    return pickle.load(f)
            else:
                # Compressed pickle
                return read_compressed_pickle(pickle_path, nparray=False)
        else:
            # Compressed pickle
            codec = metadata.get("compression_codec", "zstd")
            pickle_path = file_path.with_suffix("").with_suffix(f".pkl.{codec}")
            if not pickle_path.exists():
                raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

            return read_compressed_pickle(pickle_path, nparray=False)

    def get_file_extension(self, config: Any) -> str:
        """Get file extension for objects."""
        if BLOSC_AVAILABLE:
            return f".pkl.{config.compression.pickle_compression_codec}"
        else:
            return ".pkl"

    @property
    def data_type(self) -> str:
        return "object"


class HandlerRegistry:
    """Registry for cache handlers with configurable selection order."""

    def __init__(self, config: Optional[Any] = None):
        self.config = config
        self.handlers = []

        # If config specifies handler priority, use that order
        if config and hasattr(config, "handlers") and config.handlers.handler_priority:
            self._setup_handlers_from_config(config)
        else:
            self._setup_default_handlers(config)

    def _setup_default_handlers(self, config):
        """Setup handlers in default priority order."""
        # Add Series handlers first (higher priority than DataFrame handlers)
        if self._should_enable_handler("polars_series", config):
            if POLARS_AVAILABLE:
                self.handlers.append(PolarsSeriesHandler())

        if self._should_enable_handler("pandas_series", config):
            if PANDAS_AVAILABLE:
                self.handlers.append(PandasSeriesHandler())

        # Add DataFrame handlers if available
        if self._should_enable_handler("polars_dataframes", config):
            if POLARS_AVAILABLE:
                self.handlers.append(PolarsDataFrameHandler())

        if self._should_enable_handler("pandas_dataframes", config):
            if PANDAS_AVAILABLE:
                self.handlers.append(PandasDataFrameHandler())

        # Add other handlers
        # Note: TensorFlow handler disabled due to system compatibility issues
        # if self._should_enable_handler("tensorflow_tensors", config):
        #     # Check availability with lazy loading
        #     _, tf_available = _lazy_import_tensorflow()
        #     if tf_available and BLOSC2_AVAILABLE:
        #         self.handlers.append(TensorFlowTensorHandler())

        if self._should_enable_handler("numpy_arrays", config):
            self.handlers.append(ArrayHandler())

        if self._should_enable_handler("object_pickle", config):
            self.handlers.append(ObjectHandler())  # Keep as fallback

    def _setup_handlers_from_config(self, config):
        """Setup handlers based on config.handlers.handler_priority order."""
        handler_map = {
            "polars_series": lambda: PolarsSeriesHandler()
            if POLARS_AVAILABLE
            else None,
            "pandas_series": lambda: PandasSeriesHandler()
            if PANDAS_AVAILABLE
            else None,
            "polars_dataframes": lambda: PolarsDataFrameHandler()
            if POLARS_AVAILABLE
            else None,
            "pandas_dataframes": lambda: PandasDataFrameHandler()
            if PANDAS_AVAILABLE
            else None,
            # Note: TensorFlow handler disabled due to system compatibility issues
            # "tensorflow_tensors": lambda: TensorFlowTensorHandler()
            # if _lazy_import_tensorflow()[1] and BLOSC2_AVAILABLE
            # else None,
            "numpy_arrays": lambda: ArrayHandler(),
            "object_pickle": lambda: ObjectHandler(),
        }

        # Get priority list from config structure
        priority_list = config.handlers.handler_priority or []

        for handler_name in priority_list:
            if handler_name in handler_map and self._should_enable_handler(
                handler_name, config
            ):
                handler = handler_map[handler_name]()
                if handler is not None:
                    self.handlers.append(handler)
                    logger.debug(f"Registered {handler_name} handler with priority")

        # Add any missing default handlers that weren't specified in priority
        remaining_handlers = set(handler_map.keys()) - set(priority_list)
        for handler_name in remaining_handlers:
            if self._should_enable_handler(handler_name, config):
                handler = handler_map[handler_name]()
                if handler is not None:
                    self.handlers.append(handler)
                    logger.debug(f"Registered {handler_name} handler as default")

    def _should_enable_handler(self, handler_name: str, config) -> bool:
        """Check if a handler should be enabled based on config."""
        if config is None:
            return True  # Enable all by default

        handler_enable_map = {
            "polars_series": "enable_polars_series",
            "pandas_series": "enable_pandas_series",
            "polars_dataframes": "enable_polars_dataframes",
            "pandas_dataframes": "enable_pandas_dataframes",
            "tensorflow_tensors": "enable_tensorflow_tensors",
            "numpy_arrays": "enable_numpy_arrays",
            "object_pickle": "enable_object_pickle",
        }

        config_attr = handler_enable_map.get(handler_name)
        if config_attr:
            # Use the config structure
            return getattr(config.handlers, config_attr, True)
        return True

    def get_handler(self, data: Any) -> CacheHandler:
        """Get the appropriate handler for the given data."""
        for handler in self.handlers:
            if handler.can_handle(data):
                return handler

        raise ValueError(f"No handler available for data type: {type(data)}")

    def get_handler_by_type(self, data_type: str) -> CacheHandler:
        """Get handler by data type string."""
        for handler in self.handlers:
            if handler.data_type == data_type:
                return handler

        raise ValueError(f"No handler found for data type: {data_type}")
