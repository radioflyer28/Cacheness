"""
Type Handlers
=============

Specialized handlers for different data types in the cache system.
Each handler implements focused interfaces following the Interface Segregation Principle.

Available handlers:
- ArrayHandler: NumPy arrays with blosc2 compression
- DataFrameHandler: Pandas/Polars DataFrames in Parquet format
- ObjectHandler: Generic Python objects via pickle/dill
- SeriesHandler: Pandas Series
- TensorHandler: TensorFlow tensors (optional)

Usage:
    from cacheness.storage.handlers import HandlerRegistry, ArrayHandler

    # Get a handler for data
    registry = HandlerRegistry()
    handler = registry.get_handler(data)

    # Or use specific handler directly
    handler = ArrayHandler()
    if handler.can_handle(my_array):
        metadata = handler.put(my_array, path, config)
"""

# Re-export from parent handlers.py for backward compatibility
from cacheness.handlers import (
    # Handler classes
    ArrayHandler,
    ObjectHandler,
    HandlerRegistry,
)

# Re-export from interfaces
from cacheness.interfaces import (
    CacheHandler,
    CacheabilityChecker,
    CacheWriter,
    CacheReader,
    FormatProvider,
    DataFrameHandler,
    SeriesHandler,
    CacheHandlerError,
    CacheWriteError,
    CacheReadError,
    CacheFormatError,
)

# Optional handlers
try:
    from cacheness.handlers import PandasDataFrameHandler, PandasSeriesHandler

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    from cacheness.handlers import PolarsDataFrameHandler

    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False

try:
    from cacheness.handlers import TensorFlowTensorHandler

    _HAS_TENSORFLOW = True
except ImportError:
    _HAS_TENSORFLOW = False

__all__ = [
    # Base interfaces
    "CacheHandler",
    "CacheabilityChecker",
    "CacheWriter",
    "CacheReader",
    "FormatProvider",
    "DataFrameHandler",
    "SeriesHandler",
    # Errors
    "CacheHandlerError",
    "CacheWriteError",
    "CacheReadError",
    "CacheFormatError",
    # Core handlers
    "ArrayHandler",
    "ObjectHandler",
    "HandlerRegistry",
]

if _HAS_PANDAS:
    __all__.extend(["PandasDataFrameHandler", "PandasSeriesHandler"])

if _HAS_POLARS:
    __all__.append("PolarsDataFrameHandler")

if _HAS_TENSORFLOW:
    __all__.append("TensorFlowTensorHandler")
