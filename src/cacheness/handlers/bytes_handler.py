"""Handler for raw bytes, bytearray, and memoryview objects."""

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
)


class BytesHandler(CacheHandler):
    """Handler for raw bytes, bytearray, and memoryview objects.

    Stores binary data as-is without any serialization or transformation.
    This is the preferred handler when the caller has **pre-serialized**
    data (protobuf, msgpack, custom binary formats) or opaque payloads
    that should be written verbatim.

    The handler sits just before :class:`ObjectHandler` in the default
    priority chain so that ``bytes`` objects are stored as raw ``.bin``
    files rather than being unnecessarily pickled.
    """

    def can_handle(self, data: Any, config: Any = None) -> bool:
        """Accept ``bytes``, ``bytearray``, and ``memoryview``."""
        return isinstance(data, (bytes, bytearray, memoryview))

    def put(self, data: Any, file_path: Path, config: Any) -> HandlerResult:
        """Write raw bytes to disk.

        Args:
            data: A ``bytes``, ``bytearray``, or ``memoryview`` object.
            file_path: Base file path (extension will be replaced with ``.bin``).
            config: Cache configuration (unused — data is written verbatim).

        Returns:
            HandlerResult with ``storage_format="raw_bytes"``.
        """
        with cache_operation_context("store_bytes", size=len(data)):
            try:
                bin_path = file_path.with_suffix("").with_suffix(".bin")
                raw = bytes(data) if not isinstance(data, bytes) else data

                bin_path.write_bytes(raw)
                file_size = bin_path.stat().st_size

                logger.debug("Wrote %d raw bytes to %s", file_size, bin_path)

                return HandlerResult(
                    storage_format="raw_bytes",
                    file_size=file_size,
                    actual_path=str(bin_path),
                )
            except Exception as e:  # intentionally broad — re-raises as CacheWriteError
                raise CacheWriteError(f"Failed to write bytes data: {e}") from e

    def get(self, file_path: Path, metadata: BlobReadContext) -> bytes:
        """Read raw bytes from disk.

        Args:
            file_path: Path to the ``.bin`` file.
            metadata: Handler metadata (unused).

        Returns:
            The bytes exactly as they were stored.
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise CacheReadError(f"Bytes file not found: {file_path}")

            data = path.read_bytes()
            logger.debug("Read %d raw bytes from %s", len(data), path)
            return data
        except CacheReadError:
            raise
        except Exception as e:  # intentionally broad — re-raises as CacheReadError
            raise CacheReadError(f"Failed to read bytes data: {e}") from e

    # -- Zero-disk inline fast-paths ----------------------------------

    def put_bytes(self, data: Any, config: Any) -> tuple[bytes, HandlerResult]:
        """Serialize raw bytes in-memory (passthrough — no transformation)."""
        raw = bytes(data) if not isinstance(data, bytes) else data
        result = HandlerResult(
            storage_format="raw_bytes",
            file_size=len(raw),
            actual_path="",
        )
        return raw, result

    def get_bytes(self, blob: bytes, metadata: BlobReadContext) -> bytes:
        """Deserialize raw bytes in-memory (passthrough)."""
        return blob

    def get_file_extension(self, config: Any) -> str:
        """Return the file extension for raw bytes files."""
        return ".bin"

    @property
    def data_type(self) -> str:
        """Return the data type identifier."""
        return "bytes"
