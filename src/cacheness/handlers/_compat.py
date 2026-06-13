"""Shared imports and optional dependency detection for handler modules."""

import logging

# Import focused interfaces from parent package
from ..interfaces import (  # noqa: F401
    CacheHandler,
    HandlerResult,
    BlobReadContext,
    CacheWriteError,
    CacheReadError,
)
from ..error_handling import cache_operation_context  # noqa: F401
from ..compress_pickle import (
    BLOSC_AVAILABLE,  # noqa: F401
    is_pickleable,  # noqa: F401
    is_dill_serializable,  # noqa: F401
    optimize_compression_params,  # noqa: F401
    write_file as write_compressed_pickle,  # noqa: F401
    read_file as read_compressed_pickle,  # noqa: F401
)

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

# Optional dependency - blosc2 for array compression
try:
    import blosc2

    BLOSC2_AVAILABLE = True
except (ImportError, PermissionError):
    blosc2 = None  # type: ignore
    BLOSC2_AVAILABLE = False

# Optional dependency - dill for enhanced object serialization
try:
    import dill

    DILL_AVAILABLE = True
except ImportError:
    dill = None  # type: ignore
    DILL_AVAILABLE = False

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
    except Exception as e:  # intentionally broad — TF import may fail any way
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
