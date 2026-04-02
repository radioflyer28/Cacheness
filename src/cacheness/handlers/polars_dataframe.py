"""Handler for Polars DataFrames using Parquet format."""

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
    POLARS_AVAILABLE,
    pl,
)


class PolarsDataFrameHandler(CacheHandler):
    """Handler for Polars DataFrames using Parquet format."""

    priority: int = 30

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
        except Exception as e:  # intentionally broad — parquet compatibility check
            logger.debug(f"Polars DataFrame validation failed: {e}")
            return False

    def put(self, data: Any, file_path: Path, config: Any) -> HandlerResult:
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

                return HandlerResult(
                    storage_format="parquet",
                    file_size=file_size,
                    actual_path=str(parquet_path),
                    compression_codec=compression,
                    extra={
                        "shape": data.shape,
                        "columns": data.columns,
                        "dtypes": [str(dtype) for dtype in data.dtypes],
                        "backend": "polars",
                    },
                )

            except Exception as e:  # intentionally broad — re-raises as CacheWriteError
                raise CacheWriteError(
                    f"Failed to write Polars DataFrame to Parquet: {e}",
                    handler_type="polars_dataframe",
                    data_type=type(data).__name__,
                ) from e

    def get(self, file_path: Path, metadata: BlobReadContext) -> Any:
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

            except Exception as e:  # intentionally broad — re-raises as CacheReadError
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
