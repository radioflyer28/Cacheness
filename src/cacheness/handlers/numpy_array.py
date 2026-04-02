"""Handler for NumPy arrays using blosc2 or NPZ format."""

import ast
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ._compat import (
    CacheHandler,
    HandlerResult,
    BlobReadContext,
    logger,
    BLOSC2_AVAILABLE,
    blosc2,
)


class ArrayHandler(CacheHandler):
    """Handler for NumPy arrays using blosc2 or NPZ format."""

    priority: int = 50

    def can_handle(self, data: Any) -> bool:
        """Check if data is a NumPy array or dict of arrays."""
        if isinstance(data, np.ndarray):
            return True
        if isinstance(data, dict):
            return all(isinstance(v, np.ndarray) for v in data.values())
        return False

    def put(self, data: Any, file_path: Path, config: Any) -> HandlerResult:
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
    ) -> HandlerResult:
        """Store a single numpy array, trying blosc2 first, then NPZ fallback."""
        # Try blosc2 compression first if enabled
        if config.compression.use_blosc2_arrays and BLOSC2_AVAILABLE:
            try:
                # Update file path for blosc2 format
                blosc2_path = file_path.with_suffix("").with_suffix(".b2nd")
                self._write_blosc2_array(data, blosc2_path, config)

                return HandlerResult(
                    storage_format="blosc2_array",
                    file_size=blosc2_path.stat().st_size,
                    actual_path=str(blosc2_path),
                    compression_codec=config.compression.blosc2_array_codec,
                    extra={
                        "shape": data.shape,
                        "dtype": str(data.dtype),
                    },
                )
            except Exception as e:  # intentionally broad — blosc2 fallback to NPZ
                logger.warning(f"blosc2 compression failed, falling back to NPZ: {e}")

        # Fallback to NPZ format
        npz_path = file_path.with_suffix("").with_suffix(".npz")
        if config.compression.npz_compression:
            np.savez_compressed(npz_path, data=data)
        else:
            np.savez(npz_path, data=data)

        return HandlerResult(
            storage_format="npz",
            file_size=npz_path.stat().st_size,
            actual_path=str(npz_path),
            compression_codec="zlib" if config.compression.npz_compression else "none",
            extra={
                "shape": data.shape,
                "dtype": str(data.dtype),
            },
        )

    def _put_array_dict(
        self, data: Dict[str, np.ndarray], file_path: Path, config: Any
    ) -> HandlerResult:
        """Store a dictionary of arrays using NPZ format."""
        # Filter to only numpy arrays
        array_data = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}

        npz_path = file_path.with_suffix("").with_suffix(".npz")
        if config.compression.npz_compression:
            np.savez_compressed(npz_path, **array_data)
        else:
            np.savez(npz_path, **array_data)

        return HandlerResult(
            storage_format="npz",
            file_size=npz_path.stat().st_size,
            actual_path=str(npz_path),
            compression_codec="zlib" if config.compression.npz_compression else "none",
            extra={
                "arrays": {
                    key: {"shape": arr.shape, "dtype": str(arr.dtype)}
                    for key, arr in array_data.items()
                },
            },
        )

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
                "typesize": data.dtype.itemsize,
                "clevel": config.compression.blosc2_array_clevel,
                "codec": getattr(
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
            shape = ast.literal_eval(shape_str)  # Convert string tuple back to tuple
            dtype = np.dtype(dtype_str)
            return np.frombuffer(decompressed, dtype=dtype).reshape(shape)

    def get(self, file_path: Path, metadata: BlobReadContext) -> Any:
        """Load array(s) from file with format detection."""
        storage_format = metadata.get("storage_format", "npz")

        # Try the expected format first (accept legacy "blosc2" for backward compat)
        if storage_format in ("blosc2_array", "blosc2"):
            try:
                # Try blosc2 format
                blosc2_path = file_path.with_suffix("").with_suffix(".b2nd")
                if blosc2_path.exists():
                    return self._read_blosc2_array(blosc2_path)
            except Exception as e:  # intentionally broad — blosc2 read fallback to NPZ
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

    # -- Zero-disk inline fast-paths ----------------------------------

    def put_bytes(self, data: Any, config: Any) -> tuple[bytes, HandlerResult]:
        """Serialize a NumPy array to bytes in-memory (no disk I/O).

        Only supports single ``np.ndarray`` with blosc2 available and enabled.
        Dict-of-arrays and NPZ-only configurations fall back to the disk path.
        """
        if not isinstance(data, np.ndarray):
            raise NotImplementedError("put_bytes only supports single np.ndarray")
        if not (config.compression.use_blosc2_arrays and BLOSC2_AVAILABLE):
            raise NotImplementedError("put_bytes requires blosc2 for arrays")

        # Same binary format as _write_blosc2_array: shape|dtype|compressed
        compressed_data = blosc2.compress2(
            data,
            cparams={
                "typesize": data.dtype.itemsize,
                "clevel": config.compression.blosc2_array_clevel,
                "codec": getattr(
                    blosc2.Codec,
                    config.compression.blosc2_array_codec.upper(),
                    blosc2.Codec.LZ4,
                ),
            },
        )
        shape_bytes = str(data.shape).encode("utf-8")
        dtype_bytes = str(data.dtype).encode("utf-8")

        parts = [
            len(shape_bytes).to_bytes(4, "little"),
            shape_bytes,
            len(dtype_bytes).to_bytes(4, "little"),
            dtype_bytes,
            compressed_data,
        ]
        blob = b"".join(parts)

        result = HandlerResult(
            storage_format="blosc2_array",
            file_size=len(blob),
            actual_path="",
            compression_codec=config.compression.blosc2_array_codec,
            extra={
                "shape": data.shape,
                "dtype": str(data.dtype),
            },
        )
        return blob, result

    def get_bytes(self, blob: bytes, metadata: BlobReadContext) -> Any:
        """Deserialize a NumPy array from bytes in-memory (no disk I/O).

        Expects the binary format produced by :meth:`put_bytes`
        (shape|dtype|blosc2-compressed data).
        """
        storage_format = metadata.get("storage_format", "npz")
        if storage_format not in ("blosc2_array", "blosc2"):
            raise NotImplementedError(
                f"get_bytes does not support format {storage_format!r}"
            )

        offset = 0
        shape_len = int.from_bytes(blob[offset : offset + 4], "little")
        offset += 4
        shape_str = blob[offset : offset + shape_len].decode("utf-8")
        offset += shape_len
        dtype_len = int.from_bytes(blob[offset : offset + 4], "little")
        offset += 4
        dtype_str = blob[offset : offset + dtype_len].decode("utf-8")
        offset += dtype_len
        compressed_data = blob[offset:]

        decompressed = blosc2.decompress2(compressed_data)
        shape = ast.literal_eval(shape_str)
        dtype = np.dtype(dtype_str)
        return np.frombuffer(decompressed, dtype=dtype).reshape(shape)
