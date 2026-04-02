"""Handler for Pandas Series using Parquet format."""

from pathlib import Path
from typing import Any

from ._compat import (
    CacheHandler,
    CacheWriteError,
    CacheReadError,
    HandlerResult,
    BlobReadContext,
    cache_operation_context,
    PANDAS_AVAILABLE,
    pd,
)


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
        except Exception:  # intentionally broad — parquet compatibility check
            # This Series has mixed types that can't be handled by Parquet
            return False

    def put(self, data: Any, file_path: Path, config: Any) -> HandlerResult:
        """Store Pandas Series as Parquet with proper error handling."""
        with cache_operation_context(
            "store_pandas_series", shape=data.shape, name=data.name
        ):
            try:
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
                return HandlerResult(
                    storage_format="parquet",
                    file_size=file_size,
                    actual_path=str(parquet_path),
                    compression_codec=config.compression.parquet_compression,
                    extra={
                        "shape": list(df.shape),
                        "columns": list(df.columns),
                        "dtypes": [str(dtype) for dtype in df.dtypes],
                        "backend": "pandas",
                        "is_series": True,
                        "series_name": data.name,
                    },
                )

            except Exception as e:  # intentionally broad — re-raises as CacheWriteError
                raise CacheWriteError(
                    f"Failed to write Pandas Series to Parquet: {e}",
                    handler_type="pandas_series",
                    data_type=type(data).__name__,
                ) from e

    def get(self, file_path: Path, metadata: BlobReadContext) -> Any:
        """Load Pandas Series from Parquet with proper error handling."""
        with cache_operation_context("load_pandas_series", file_path=str(file_path)):
            try:
                if not PANDAS_AVAILABLE or pd is None:
                    raise CacheReadError(
                        "Pandas not available for loading Series",
                        handler_type="pandas_series",
                    )

                df = pd.read_parquet(file_path)

                # Convert back to Series - since we preserved the index, use it
                series = df.iloc[:, 0]  # Get the first (and only) column

                # Restore the original Series name
                if metadata.get("is_series") and "series_name" in metadata:
                    series.name = metadata["series_name"]

                return series

            except Exception as e:  # intentionally broad — re-raises as CacheReadError
                if isinstance(e, CacheReadError):
                    raise
                raise CacheReadError(
                    f"Failed to read Pandas Series from Parquet: {e}",
                    handler_type="pandas_series",
                ) from e

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
