"""Handler package for type-aware serialization.

Re-exports all handler classes and the registry for backward compatibility.
"""

from .numpy_array import ArrayHandler
from .bytes_handler import BytesHandler
from .object_handler import ObjectHandler
from .registry import HandlerRegistry

# Optional DataFrame/Series handlers
try:
    from .pandas_dataframe import PandasDataFrameHandler
except ImportError:
    pass

try:
    from .pandas_series import PandasSeriesHandler
except ImportError:
    pass

try:
    from .polars_dataframe import PolarsDataFrameHandler
except ImportError:
    pass

try:
    from .polars_series import PolarsSeriesHandler
except ImportError:
    pass

try:
    from .tensorflow_tensor import TensorFlowTensorHandler
except ImportError:
    pass

# Re-export compat symbols that downstream code may reference
from ._compat import (
    BLOSC2_AVAILABLE,
    BLOSC_AVAILABLE,
    DILL_AVAILABLE,
    PANDAS_AVAILABLE,
    POLARS_AVAILABLE,
    _lazy_import_tensorflow,
)

__all__ = [
    "ArrayHandler",
    "BytesHandler",
    "ObjectHandler",
    "HandlerRegistry",
    "PandasDataFrameHandler",
    "PandasSeriesHandler",
    "PolarsDataFrameHandler",
    "PolarsSeriesHandler",
    "TensorFlowTensorHandler",
    "BLOSC2_AVAILABLE",
    "BLOSC_AVAILABLE",
    "DILL_AVAILABLE",
    "PANDAS_AVAILABLE",
    "POLARS_AVAILABLE",
    "_lazy_import_tensorflow",
]
