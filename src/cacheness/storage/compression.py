"""
Compression Utilities
====================

Functions to compress and decompress cached data using various codecs.

Supported compression codecs:
- blosc2/blosclz: Fast compression with good ratios (default)
- lz4: Very fast compression
- lz4hc: High compression variant of lz4
- zstd: Zstandard compression (excellent ratio)
- zlib: Standard zlib compression
- gzip: Gzip compression
- snappy: Google's fast compressor

Usage:
    from cacheness.storage.compression import write_file, read_file
    
    # Write compressed data
    write_file(data, Path("cache/data.pkl"), codec="lz4", clevel=3)
    
    # Read compressed data
    data = read_file(Path("cache/data.pkl"))
"""

# Re-export from parent compress_pickle.py
from ..compress_pickle import (
    write_file,
    read_file,
    is_pickleable,
    is_dill_serializable,
    verify_dill_serializable,
    list_available_codecs,
    optimize_compression_params,
    CompressionError,
    DecompressionError,
    BLOSC_AVAILABLE,
    DILL_AVAILABLE,
)

__all__ = [
    # Core functions
    "write_file",
    "read_file",
    "is_pickleable",
    "is_dill_serializable",
    "verify_dill_serializable",
    "list_available_codecs",
    "optimize_compression_params",
    # Errors
    "CompressionError",
    "DecompressionError",
    # Feature flags
    "BLOSC_AVAILABLE",
    "DILL_AVAILABLE",
]
