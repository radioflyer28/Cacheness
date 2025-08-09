"""
Cache Format Handlers
====================

This module contains specialized handlers for different data types in the cache system.
Each handler implements the Strategy pattern for format-specific operations.
"""

import numpy as np
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
import logging

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
    BLOSC_AVAILABLE
)

# Optional dependency - blosc2 for array compression
try:
    import blosc2
    BLOSC2_AVAILABLE = True
except ImportError:
    blosc2 = None  # type: ignore
    BLOSC2_AVAILABLE = False

logger = logging.getLogger(__name__)

# Log DataFrame backend availability
if POLARS_AVAILABLE and PANDAS_AVAILABLE:
    logger.info("📊 Both Polars and Pandas available for DataFrame caching")
elif POLARS_AVAILABLE:
    logger.info("📊 Polars available for DataFrame caching")
elif PANDAS_AVAILABLE:
    logger.info("📊 Pandas available for DataFrame caching (Polars not found)")
else:
    logger.warning("⚠️  Neither Polars nor Pandas available - DataFrame caching disabled")

logger = logging.getLogger(__name__)


class CacheHandler(ABC):
    """Abstract base class for cache format handlers."""
    
    @abstractmethod
    def can_handle(self, data: Any) -> bool:
        """Check if this handler can process the given data type."""
        pass
        
    @abstractmethod
    def put(self, data: Any, file_path: Path, config: Any) -> Dict[str, Any]:
        """Store data and return metadata."""
        pass
        
    @abstractmethod
    def get(self, file_path: Path, metadata: Dict[str, Any]) -> Any:
        """Retrieve data from file."""
        pass
        
    @abstractmethod
    def get_file_extension(self, config: Any) -> str:
        """Get the file extension for this format."""
        pass
    
    @property
    @abstractmethod
    def data_type(self) -> str:
        """Return the data type identifier for this handler."""
        pass


class PolarsDataFrameHandler(CacheHandler):
    """Handler for Polars DataFrames using Parquet format."""
    
    def can_handle(self, data: Any) -> bool:
        """Check if data is a Polars DataFrame."""
        if not POLARS_AVAILABLE or pl is None:
            return False
        return isinstance(data, pl.DataFrame)
    
    def put(self, data: Any, file_path: Path, config: Any) -> Dict[str, Any]:
        """Store Polars DataFrame as Parquet."""
        parquet_path = file_path.with_suffix('').with_suffix('.parquet')
        data.write_parquet(parquet_path, compression=config.parquet_compression)
        
        return {
            "storage_format": "parquet",
            "file_size": parquet_path.stat().st_size,
            "actual_path": str(parquet_path),
            "metadata": {
                "shape": data.shape,
                "columns": data.columns,
                "dtypes": [str(dtype) for dtype in data.dtypes],
                "compression": config.parquet_compression,
                "backend": "polars"
            }
        }
    
    def get(self, file_path: Path, metadata: Dict[str, Any]) -> Any:
        """Load Polars DataFrame from Parquet."""
        if not POLARS_AVAILABLE or pl is None:
            raise ImportError("Polars not available for loading DataFrame")
        return pl.read_parquet(file_path)
    
    def get_file_extension(self, config: Any) -> str:
        """Get file extension for Polars DataFrames."""
        return ".parquet"
    
    @property
    def data_type(self) -> str:
        return "polars_dataframe"


class PandasDataFrameHandler(CacheHandler):
    """Handler for Pandas DataFrames using Parquet format."""
    
    def can_handle(self, data: Any) -> bool:
        """Check if data is a Pandas DataFrame."""
        if not PANDAS_AVAILABLE or pd is None:
            return False
        return isinstance(data, pd.DataFrame)
    
    def put(self, data: Any, file_path: Path, config: Any) -> Dict[str, Any]:
        """Store Pandas DataFrame as Parquet."""
        parquet_path = file_path.with_suffix('').with_suffix('.parquet')
        data.to_parquet(parquet_path, compression=config.parquet_compression, index=False)
        
        return {
            "storage_format": "parquet", 
            "file_size": parquet_path.stat().st_size,
            "actual_path": str(parquet_path),
            "metadata": {
                "shape": data.shape,
                "columns": data.columns.tolist(),
                "dtypes": [str(dtype) for dtype in data.dtypes],
                "compression": config.parquet_compression,
                "backend": "pandas"
            }
        }
    
    def get(self, file_path: Path, metadata: Dict[str, Any]) -> Any:
        """Load Pandas DataFrame from Parquet."""
        if not PANDAS_AVAILABLE or pd is None:
            raise ImportError("Pandas not available for loading DataFrame")
        return pd.read_parquet(file_path)
    
    def get_file_extension(self, config: Any) -> str:
        """Get file extension for Pandas DataFrames."""
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
            raise ValueError("ArrayHandler can only handle np.ndarray or Dict[str, np.ndarray]")
    
    def _put_single_array(self, data: np.ndarray, file_path: Path, config: Any) -> Dict[str, Any]:
        """Store a single numpy array, trying blosc2 first, then NPZ fallback."""
        # Try blosc2 compression first if enabled
        if config.use_blosc2_arrays and BLOSC2_AVAILABLE:
            try:
                # Update file path for blosc2 format
                blosc2_path = file_path.with_suffix('').with_suffix('.b2nd')
                self._write_blosc2_array(data, blosc2_path, config)
                
                return {
                    "storage_format": "blosc2",
                    "file_size": blosc2_path.stat().st_size,
                    "actual_path": str(blosc2_path),
                    "metadata": {
                        "shape": data.shape, 
                        "dtype": str(data.dtype),
                        "storage_format": "blosc2",
                        "compression": config.blosc2_array_codec
                    }
                }
            except Exception as e:
                logger.warning(f"blosc2 compression failed, falling back to NPZ: {e}")
        
        # Fallback to NPZ format
        npz_path = file_path.with_suffix('').with_suffix('.npz')
        if config.npz_compression:
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
                "compression": "zlib" if config.npz_compression else "none"
            }
        }
    
    def _put_array_dict(self, data: Dict[str, np.ndarray], file_path: Path, config: Any) -> Dict[str, Any]:
        """Store a dictionary of arrays using NPZ format."""
        # Filter to only numpy arrays
        array_data = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
        
        npz_path = file_path.with_suffix('').with_suffix('.npz')
        if config.npz_compression:
            np.savez_compressed(npz_path, **array_data)
        else:
            np.savez(npz_path, **array_data)
        
        return {
            "storage_format": "npz",
            "file_size": npz_path.stat().st_size,
            "actual_path": str(npz_path),
            "metadata": {
                "arrays": {key: {"shape": arr.shape, "dtype": str(arr.dtype)} 
                          for key, arr in array_data.items()},
                "storage_format": "npz",
                "compression": "zlib" if config.npz_compression else "none"
            }
        }
    
    def _write_blosc2_array(self, data: np.ndarray, file_path: Path, config: Any) -> None:
        """Write numpy array to file using blosc2 compression with metadata."""
        if not BLOSC2_AVAILABLE:
            raise ImportError("blosc2 is required for array compression but is not available")
        
        # Compress the array data
        compressed_data = blosc2.compress(
            data, 
            typesize=data.dtype.itemsize,
            clevel=config.blosc2_array_clevel,
            codec=getattr(blosc2.Codec, config.blosc2_array_codec.upper(), blosc2.Codec.LZ4)
        )
        
        # Write compressed data and metadata to file
        with open(file_path, 'wb') as f:
            # Write shape and dtype info first
            shape_bytes = str(data.shape).encode('utf-8')
            dtype_bytes = str(data.dtype).encode('utf-8')
            
            # Write metadata lengths and data
            f.write(len(shape_bytes).to_bytes(4, 'little'))
            f.write(shape_bytes)
            f.write(len(dtype_bytes).to_bytes(4, 'little'))
            f.write(dtype_bytes)
            f.write(compressed_data)
    
    def _read_blosc2_array(self, file_path: Path) -> np.ndarray:
        """Read numpy array from blosc2 compressed file with metadata."""
        if not BLOSC2_AVAILABLE:
            raise ImportError("blosc2 is required for array decompression but is not available")
        
        with open(file_path, 'rb') as f:
            # Read shape and dtype metadata
            shape_len = int.from_bytes(f.read(4), 'little')
            shape_str = f.read(shape_len).decode('utf-8')
            dtype_len = int.from_bytes(f.read(4), 'little')
            dtype_str = f.read(dtype_len).decode('utf-8')
            
            # Read compressed data
            compressed_data = f.read()
            
            # Decompress and reconstruct array
            decompressed = blosc2.decompress(compressed_data)
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
                blosc2_path = file_path.with_suffix('').with_suffix('.b2nd')
                if blosc2_path.exists():
                    return self._read_blosc2_array(blosc2_path)
            except Exception as e:
                logger.debug(f"Failed to read blosc2 file: {e}")
        
        # Try NPZ format (fallback or primary)
        npz_path = file_path.with_suffix('').with_suffix('.npz')
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
        if config.use_blosc2_arrays and BLOSC2_AVAILABLE:
            return ".b2nd"
        return ".npz"
    
    @property
    def data_type(self) -> str:
        return "array"


class ObjectHandler(CacheHandler):
    """Handler for general Python objects using compressed pickle."""
    
    def can_handle(self, data: Any) -> bool:
        """Check if data can be pickled (and isn't handled by other handlers)."""
        # Don't handle DataFrames or arrays - let specialized handlers do that
        if POLARS_AVAILABLE and pl is not None and isinstance(data, pl.DataFrame):
            return False
        if PANDAS_AVAILABLE and pd is not None and isinstance(data, pd.DataFrame):
            return False
        if isinstance(data, np.ndarray):
            return False
        if isinstance(data, dict) and all(isinstance(v, np.ndarray) for v in data.values()):
            return False
        
        return is_pickleable(data)
    
    def put(self, data: Any, file_path: Path, config: Any) -> Dict[str, Any]:
        """Store object using compressed pickle or fallback to standard pickle."""
        if not is_pickleable(data):
            raise ValueError(f"Object of type {type(data)} is not pickleable and cannot be cached")
        
        # Ensure clean path without existing extension
        if BLOSC_AVAILABLE:
            pickle_path = file_path.with_suffix('').with_suffix('.pkl.lz4')
            # Use compressed pickle with lz4 for best performance
            write_compressed_pickle(
                data, 
                pickle_path, 
                nparray=False,  # Don't use numpy optimization for general objects
                codec=config.pickle_compression_codec,
                clevel=config.pickle_compression_level
            )
            storage_format = "compressed_pickle"
        else:
            # Fall back to standard pickle when blosc is not available
            pickle_path = file_path.with_suffix('').with_suffix('.pkl')
            import pickle
            with open(pickle_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            storage_format = "pickle"
        
        metadata = {
            "object_type": str(type(data)),
            "storage_format": storage_format
        }
        
        return {
            "storage_format": storage_format,
            "file_size": pickle_path.stat().st_size,
            "actual_path": str(pickle_path),
            "metadata": metadata
        }
    
    def get(self, file_path: Path, metadata: Dict[str, Any]) -> Any:
        """Load object using compressed pickle or standard pickle."""
        storage_format = metadata.get("storage_format", "compressed_pickle")
        
        if storage_format == "pickle" or not BLOSC_AVAILABLE:
            # Standard pickle fallback
            pickle_path = file_path.with_suffix('').with_suffix('.pkl')
            if not pickle_path.exists():
                # Try compressed version as fallback
                pickle_path = file_path.with_suffix('').with_suffix('.pkl.lz4')
            
            if not pickle_path.exists():
                raise FileNotFoundError(f"Pickle file not found: {pickle_path}")
            
            if pickle_path.suffix == '.pkl':
                # Standard pickle
                import pickle
                with open(pickle_path, 'rb') as f:
                    return pickle.load(f)
            else:
                # Compressed pickle
                return read_compressed_pickle(pickle_path, nparray=False)
        else:
            # Compressed pickle
            pickle_path = file_path.with_suffix('').with_suffix('.pkl.lz4')
            if not pickle_path.exists():
                raise FileNotFoundError(f"Pickle file not found: {pickle_path}")
            
            return read_compressed_pickle(pickle_path, nparray=False)
    
    def get_file_extension(self, config: Any) -> str:
        """Get file extension for objects."""
        if BLOSC_AVAILABLE:
            return ".pkl.lz4"
        else:
            return ".pkl"
    
    @property
    def data_type(self) -> str:
        return "object"


class HandlerRegistry:
    """Registry for cache handlers with automatic selection."""
    
    def __init__(self):
        self.handlers = []
        
        # Add DataFrame handlers if available
        if POLARS_AVAILABLE:
            self.handlers.append(PolarsDataFrameHandler())
        if PANDAS_AVAILABLE:
            self.handlers.append(PandasDataFrameHandler())
        
        # Add other handlers
        self.handlers.extend([
            ArrayHandler(), 
            ObjectHandler(),  # Keep as fallback
        ])
    
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
