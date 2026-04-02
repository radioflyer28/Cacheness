"""Handler for Pandas DataFrames using Parquet format."""

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


class PandasDataFrameHandler(CacheHandler):
    """Handler for Pandas DataFrames using Parquet format."""

    def can_handle(self, data: Any) -> bool:
        """Check if data is a Pandas DataFrame."""
        if not PANDAS_AVAILABLE or pd is None:
            return False
        if not isinstance(data, pd.DataFrame):
            return False

        # Check if DataFrame can be written to Parquet
        try:
            import io

            data.to_parquet(io.BytesIO())
            return True
        except Exception:  # intentionally broad — parquet compatibility check
            # DataFrame has types that can't be written to Parquet, let ObjectHandler handle it
            return False

    def put(self, data: Any, file_path: Path, config: Any) -> HandlerResult:
        """Store Pandas DataFrame as Parquet with proper error handling."""
        with cache_operation_context(
            "store_pandas_dataframe",
            shape=data.shape,
            columns=len(data.columns),
        ):
            try:
                parquet_path = file_path.with_suffix("").with_suffix(".parquet")
                compression = config.compression.parquet_compression

                data.to_parquet(
                    parquet_path,
                    compression=compression,
                    # Keep index=True by default to preserve DataFrame index
                )

                file_size = parquet_path.stat().st_size
                return HandlerResult(
                    storage_format="parquet",
                    file_size=file_size,
                    actual_path=str(parquet_path),
                    compression_codec=compression,
                    extra={
                        "shape": data.shape,
                        "columns": data.columns.tolist(),
                        "dtypes": [str(dtype) for dtype in data.dtypes],
                        "backend": "pandas",
                    },
                )

            except Exception as e:  # intentionally broad — re-raises as CacheWriteError
                raise CacheWriteError(
                    f"Failed to write Pandas DataFrame to Parquet: {e}",
                    handler_type="pandas_dataframe",
                    data_type=type(data).__name__,
                ) from e

    def get(self, file_path: Path, metadata: BlobReadContext) -> Any:
        """Load Pandas DataFrame from Parquet with proper error handling."""
        with cache_operation_context("load_pandas_dataframe", file_path=str(file_path)):
            try:
                if not PANDAS_AVAILABLE or pd is None:
                    raise CacheReadError(
                        "Pandas not available for loading DataFrame",
                        handler_type="pandas_dataframe",
                    )

                return pd.read_parquet(file_path)

            except Exception as e:  # intentionally broad — re-raises as CacheReadError
                if isinstance(e, CacheReadError):
                    raise
                raise CacheReadError(
                    f"Failed to read Pandas DataFrame from Parquet: {e}",
                    handler_type="pandas_dataframe",
                ) from e

    def get_file_extension(self, config: Any) -> str:
        """Get file extension for Pandas DataFrames and Series."""
        return ".parquet"

    @property
    def data_type(self) -> str:
        return "pandas_dataframe"
