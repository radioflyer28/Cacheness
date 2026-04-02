"""Handler for general Python objects using compressed pickle."""

from pathlib import Path
from typing import Any, Dict

import numpy as np

from ._compat import (
    CacheHandler,
    HandlerResult,
    BlobReadContext,
    is_pickleable,
    is_dill_serializable,
    write_compressed_pickle,
    read_compressed_pickle,
    optimize_compression_params,
    BLOSC_AVAILABLE,
    DILL_AVAILABLE,
    dill,
)


class ObjectHandler(CacheHandler):
    """Handler for general Python objects using compressed pickle."""

    priority: int = 100

    def can_handle(self, data: Any, config: Any = None) -> bool:
        """Check if data can be pickled or dill-serialized.

        ObjectHandler is always last in the registry's priority list.  By the
        time this method runs every specialised handler (DataFrame, Series,
        TensorFlow, Bytes) has already declined, so we don't need to re-test
        their logic.  The only types we must still skip are ``np.ndarray``,
        dict-of-arrays, and buffer types (``bytes``/``bytearray``/
        ``memoryview``), because their respective handlers never reject those.
        """
        # BytesHandler always accepts these — don't claim them
        if isinstance(data, (bytes, bytearray, memoryview)):
            return False

        # ArrayHandler always accepts these — don't claim them
        if isinstance(data, np.ndarray):
            return False
        if (
            isinstance(data, dict)
            and data
            and all(isinstance(v, np.ndarray) for v in data.values())
        ):
            return False

        # Try pickle first
        if is_pickleable(data):
            return True

        # Then try dill as fallback if enabled
        if (
            config
            and hasattr(config, "handlers")
            and config.handlers.enable_dill_fallback
        ):
            return is_dill_serializable(data)

        return False

    def put(self, data: Any, file_path: Path, config: Any) -> HandlerResult:
        """Store object using compressed pickle with dill fallback."""
        # Determine serialization method: try pickle first, then dill if enabled
        use_pickle = is_pickleable(data)
        use_dill = False
        serializer_name = "pickle"

        if not use_pickle:
            # Only try dill if enabled in config
            if (
                config
                and hasattr(config, "handlers")
                and config.handlers.enable_dill_fallback
            ):
                use_dill = is_dill_serializable(data)
                if use_dill:
                    serializer_name = "dill"

            if not use_dill:
                dill_status = (
                    " (dill disabled)"
                    if not (
                        config
                        and hasattr(config, "handlers")
                        and config.handlers.enable_dill_fallback
                    )
                    else ""
                )
                raise ValueError(
                    f"Object of type {type(data)} cannot be serialized with pickle{dill_status}"
                )

        # Check if we should use compression based on codec, size threshold, and availability
        should_compress = (
            BLOSC_AVAILABLE and config.compression.pickle_compression_codec != "none"
        )
        if should_compress:
            # Get a rough estimate of object size by pickling it first
            import pickle

            try:
                if use_dill and DILL_AVAILABLE and dill is not None:
                    test_data = dill.dumps(data, protocol=dill.HIGHEST_PROTOCOL)
                else:
                    test_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

                # Only compress if object is larger than threshold
                if len(test_data) < config.compression.compression_threshold_bytes:
                    should_compress = False
            except Exception:  # intentionally broad — size estimation may fail any way
                # If we can't estimate size, use compression anyway (but respect "none" codec)
                should_compress = (
                    BLOSC_AVAILABLE
                    and config.compression.pickle_compression_codec != "none"
                )

        # Ensure clean path without existing extension
        if should_compress:
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

            # Use the appropriate serializer with compression
            if use_dill:
                # Use dill for serialization but still apply blosc compression
                self._write_compressed_dill(data, pickle_path, compression_params)
            else:
                # Use compressed pickle with optimized parameters
                write_compressed_pickle(
                    data,
                    pickle_path,
                    nparray=False,  # Don't use numpy optimization for general objects
                    **compression_params,
                )
            storage_format = f"compressed_{serializer_name}"
        else:
            # Fall back to standard serialization when blosc is not available
            pickle_path = file_path.with_suffix("").with_suffix(".pkl")

            if use_dill and DILL_AVAILABLE and dill is not None:
                with open(pickle_path, "wb") as f:
                    dill.dump(data, f, protocol=dill.HIGHEST_PROTOCOL)
            else:
                import pickle

                with open(pickle_path, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            storage_format = serializer_name

        return HandlerResult(
            storage_format=storage_format,
            file_size=pickle_path.stat().st_size,
            actual_path=str(pickle_path),
            compression_codec=config.compression.pickle_compression_codec
            if BLOSC_AVAILABLE
            else None,
            serializer=serializer_name,
            object_type=str(type(data)),
        )

    def _write_compressed_dill(
        self, data: Any, file_path: Path, compression_params: Dict[str, Any]
    ) -> None:
        """Write object using dill serialization with blosc compression."""
        if not DILL_AVAILABLE or dill is None:
            raise ValueError("dill is not available for serialization")

        if not BLOSC_AVAILABLE:
            raise ValueError("blosc is not available for compression")

        # Use similar approach to write_compressed_pickle but with dill
        # Serialize with dill first
        pickled_data = dill.dumps(data, protocol=dill.HIGHEST_PROTOCOL)

        # Use the same compression approach as compress_pickle
        from ..compress_pickle import blosc

        # Filter compression params to only include valid blosc2 parameters
        # For pickled data, set typesize to 1 (byte) since it's arbitrary binary data
        valid_blosc_params = {
            "typesize": 1,  # Always use 1 for pickled binary data
        }
        for key, value in compression_params.items():
            if key in ["clevel", "codec", "filter"]:
                valid_blosc_params[key] = value

        compressed_data = blosc.compress(pickled_data, **valid_blosc_params)

        # Write to file
        with open(file_path, "wb") as f:
            f.write(compressed_data)

    def _read_compressed_dill(self, file_path: Path) -> Any:
        """Read object using dill deserialization with blosc decompression."""
        if not DILL_AVAILABLE or dill is None:
            raise ValueError("dill is not available for deserialization")

        if not BLOSC_AVAILABLE:
            raise ValueError("blosc is not available for decompression")

        # Read compressed data
        with open(file_path, "rb") as f:
            compressed_data = f.read()

        # Use the same decompression approach as compress_pickle
        from ..compress_pickle import blosc

        decompressed_data = blosc.decompress(compressed_data)

        # Deserialize with dill
        return dill.loads(decompressed_data)

    def get(self, file_path: Path, metadata: BlobReadContext) -> Any:
        """Load object using compressed pickle/dill or standard pickle/dill."""
        storage_format = metadata.get("storage_format", "compressed_pickle")
        serializer = metadata.get("serializer", "pickle")

        if storage_format in ["pickle", "dill"] or not BLOSC_AVAILABLE:
            # Standard serialization fallback
            pickle_path = file_path.with_suffix("").with_suffix(".pkl")
            if not pickle_path.exists():
                # Try compressed version as fallback
                pickle_path = file_path.with_suffix("").with_suffix(
                    f".pkl.{metadata.get('compression_codec', 'zstd')}"
                )

            if not pickle_path.exists():
                raise FileNotFoundError(f"Serialized file not found: {pickle_path}")

            if pickle_path.suffix == ".pkl":
                # Standard serialization
                if serializer == "dill" and DILL_AVAILABLE and dill is not None:
                    with open(pickle_path, "rb") as f:
                        return dill.load(f)
                else:
                    import pickle

                    with open(pickle_path, "rb") as f:
                        return pickle.load(f)
            else:
                # Compressed serialization
                if serializer == "dill":
                    return self._read_compressed_dill(pickle_path)
                else:
                    return read_compressed_pickle(pickle_path, nparray=False)
        else:
            # Compressed serialization
            codec = metadata.get("compression_codec", "zstd")
            pickle_path = file_path.with_suffix("").with_suffix(f".pkl.{codec}")
            if not pickle_path.exists():
                raise FileNotFoundError(f"Serialized file not found: {pickle_path}")

            if serializer == "dill":
                return self._read_compressed_dill(pickle_path)
            else:
                return read_compressed_pickle(pickle_path, nparray=False)

    # -- Zero-disk inline fast-paths ----------------------------------

    def put_bytes(self, data: Any, config: Any) -> tuple[bytes, HandlerResult]:
        """Serialize a Python object to bytes in-memory (no disk I/O).

        Mirrors the logic in :meth:`put` but performs all serialization and
        optional blosc compression entirely in memory.
        """
        import pickle as _pickle

        # Determine serializer — same logic as put()
        use_pickle = is_pickleable(data)
        use_dill = False
        serializer_name = "pickle"

        if not use_pickle:
            if (
                config
                and hasattr(config, "handlers")
                and config.handlers.enable_dill_fallback
            ):
                use_dill = is_dill_serializable(data)
                if use_dill:
                    serializer_name = "dill"
            if not use_dill:
                raise NotImplementedError("Object cannot be serialized in-memory")

        # Serialize to bytes
        if use_dill and DILL_AVAILABLE and dill is not None:
            raw = dill.dumps(data, protocol=dill.HIGHEST_PROTOCOL)
        else:
            raw = _pickle.dumps(data, protocol=_pickle.HIGHEST_PROTOCOL)

        # Optionally compress with blosc
        should_compress = (
            BLOSC_AVAILABLE
            and config.compression.pickle_compression_codec != "none"
            and len(raw) >= config.compression.compression_threshold_bytes
        )
        if should_compress:
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
            from ..compress_pickle import blosc as _blosc

            valid_blosc_params: Dict[str, Any] = {"typesize": 1}
            for key, value in compression_params.items():
                if key in ["clevel", "filter"]:
                    valid_blosc_params[key] = value
                elif key == "codec" and isinstance(value, str):
                    # Convert string codec to blosc2.Codec enum
                    codec_map = {
                        "lz4": _blosc.Codec.LZ4,
                        "lz4hc": _blosc.Codec.LZ4HC,
                        "zstd": _blosc.Codec.ZSTD,
                        "zlib": _blosc.Codec.ZLIB,
                        "blosclz": _blosc.Codec.BLOSCLZ,
                    }
                    valid_blosc_params["codec"] = codec_map.get(
                        value.lower(), _blosc.Codec.LZ4
                    )
                elif key == "codec":
                    valid_blosc_params["codec"] = value
            blob = _blosc.compress(raw, **valid_blosc_params)
            storage_format = f"compressed_{serializer_name}"
        else:
            blob = raw
            storage_format = serializer_name

        result = HandlerResult(
            storage_format=storage_format,
            file_size=len(blob),
            actual_path="",
            compression_codec=config.compression.pickle_compression_codec
            if BLOSC_AVAILABLE
            else None,
            serializer=serializer_name,
            object_type=str(type(data)),
        )
        return blob, result

    def get_bytes(self, blob: bytes, metadata: BlobReadContext) -> Any:
        """Deserialize a Python object from bytes in-memory (no disk I/O).

        Mirrors the logic in :meth:`get` but reads entirely from the
        provided bytes buffer.
        """
        import pickle as _pickle

        storage_format = metadata.get("storage_format", "compressed_pickle")
        serializer = metadata.get("serializer", "pickle")

        is_compressed = storage_format.startswith("compressed") and BLOSC_AVAILABLE
        if is_compressed:
            from ..compress_pickle import blosc as _blosc

            raw = _blosc.decompress(blob)
        else:
            raw = blob

        if serializer == "dill" and DILL_AVAILABLE and dill is not None:
            return dill.loads(raw)
        return _pickle.loads(raw)

    def get_file_extension(self, config: Any) -> str:
        """Get file extension for objects."""
        if BLOSC_AVAILABLE and config.compression.pickle_compression_codec != "none":
            return f".pkl.{config.compression.pickle_compression_codec}"
        else:
            return ".pkl"

    @property
    def data_type(self) -> str:
        return "object"
