"""Registry for cache handlers with configurable selection order."""

import logging
from typing import Any, Optional

from ._compat import (
    CacheHandler,
    POLARS_AVAILABLE,
    PANDAS_AVAILABLE,
)
from .polars_series import PolarsSeriesHandler
from .pandas_series import PandasSeriesHandler
from .polars_dataframe import PolarsDataFrameHandler
from .pandas_dataframe import PandasDataFrameHandler
from .numpy_array import ArrayHandler
from .bytes_handler import BytesHandler
from .object_handler import ObjectHandler

logger = logging.getLogger(__name__)


class HandlerRegistry:
    """
    Registry for cache handlers with configurable selection order.

    Supports:
    - Built-in handlers (DataFrame, Array, Object)
    - Custom handler registration via `register_handler()`
    - Handler unregistration via `unregister_handler()`
    - Handler listing via `list_handlers()`
    - Priority-based handler selection

    Example:
        >>> registry = HandlerRegistry()
        >>>
        >>> # Register custom handler
        >>> class MyHandler(CacheHandler):
        ...     def can_handle(self, data): return isinstance(data, MyType)
        ...     # ... other methods
        >>>
        >>> registry.register_handler(MyHandler(), priority=0)  # Highest priority
        >>>
        >>> # List all handlers
        >>> for info in registry.list_handlers():
        ...     print(f"{info['name']}: priority={info['priority']}")
    """

    def __init__(self, config: Optional[Any] = None):
        self.config = config
        self.handlers = []

        # If config specifies handler priority, use that order
        if config and hasattr(config, "handlers") and config.handlers.handler_priority:
            self._setup_handlers_from_config(config)
        else:
            self._setup_default_handlers(config)

    def _setup_default_handlers(self, config):
        """Setup handlers in default priority order."""
        # Add Series handlers first (higher priority than DataFrame handlers)
        if self._should_enable_handler("polars_series", config):
            if POLARS_AVAILABLE:
                self.handlers.append(PolarsSeriesHandler())

        if self._should_enable_handler("pandas_series", config):
            if PANDAS_AVAILABLE:
                self.handlers.append(PandasSeriesHandler())

        # Add DataFrame handlers if available
        if self._should_enable_handler("polars_dataframes", config):
            if POLARS_AVAILABLE:
                self.handlers.append(PolarsDataFrameHandler())

        if self._should_enable_handler("pandas_dataframes", config):
            if PANDAS_AVAILABLE:
                self.handlers.append(PandasDataFrameHandler())

        # Add other handlers
        # Note: TensorFlow handler disabled due to system compatibility issues
        # if self._should_enable_handler("tensorflow_tensors", config):
        #     # Check availability with lazy loading
        #     _, tf_available = _lazy_import_tensorflow()
        #     if tf_available and BLOSC2_AVAILABLE:
        #         self.handlers.append(TensorFlowTensorHandler())

        if self._should_enable_handler("numpy_arrays", config):
            self.handlers.append(ArrayHandler())

        if self._should_enable_handler("bytes", config):
            self.handlers.append(BytesHandler())

        if self._should_enable_handler("object_pickle", config):
            self.handlers.append(ObjectHandler())  # Keep as fallback

    def _setup_handlers_from_config(self, config):
        """Setup handlers based on config.handlers.handler_priority order."""
        handler_map = {
            "polars_series": lambda: PolarsSeriesHandler()
            if POLARS_AVAILABLE
            else None,
            "pandas_series": lambda: PandasSeriesHandler()
            if PANDAS_AVAILABLE
            else None,
            "polars_dataframes": lambda: PolarsDataFrameHandler()
            if POLARS_AVAILABLE
            else None,
            "pandas_dataframes": lambda: PandasDataFrameHandler()
            if PANDAS_AVAILABLE
            else None,
            # Note: TensorFlow handler disabled due to system compatibility issues
            # "tensorflow_tensors": lambda: TensorFlowTensorHandler()
            # if _lazy_import_tensorflow()[1] and BLOSC2_AVAILABLE
            # else None,
            "numpy_arrays": lambda: ArrayHandler(),
            "bytes": lambda: BytesHandler(),
            "object_pickle": lambda: ObjectHandler(),
        }

        # Get priority list from config structure
        priority_list = config.handlers.handler_priority or []

        for handler_name in priority_list:
            if handler_name in handler_map and self._should_enable_handler(
                handler_name, config
            ):
                handler = handler_map[handler_name]()
                if handler is not None:
                    self.handlers.append(handler)
                    logger.debug(f"Registered {handler_name} handler with priority")

        # Add any missing default handlers that weren't specified in priority
        remaining_handlers = set(handler_map.keys()) - set(priority_list)
        for handler_name in remaining_handlers:
            if self._should_enable_handler(handler_name, config):
                handler = handler_map[handler_name]()
                if handler is not None:
                    self.handlers.append(handler)
                    logger.debug(f"Registered {handler_name} handler as default")

    def _should_enable_handler(self, handler_name: str, config) -> bool:
        """Check if a handler should be enabled based on config."""
        if config is None:
            return True  # Enable all by default

        handler_enable_map = {
            "polars_series": "enable_polars_series",
            "pandas_series": "enable_pandas_series",
            "polars_dataframes": "enable_polars_dataframes",
            "pandas_dataframes": "enable_pandas_dataframes",
            "tensorflow_tensors": "enable_tensorflow_tensors",
            "numpy_arrays": "enable_numpy_arrays",
            "bytes": "enable_bytes_handler",
            "object_pickle": "enable_object_pickle",
        }

        config_attr = handler_enable_map.get(handler_name)
        if config_attr:
            # Use the config structure
            return getattr(config.handlers, config_attr, True)
        return True

    def get_handler(self, data: Any) -> CacheHandler:
        """Get the appropriate handler for the given data."""
        for handler in self.handlers:
            # Try to pass config to can_handle if the method supports it
            try:
                if handler.can_handle(data, self.config):
                    return handler
            except TypeError:
                # Fallback for handlers that don't accept config parameter
                if handler.can_handle(data):
                    return handler

        raise ValueError(f"No handler available for data type: {type(data)}")

    def get_handler_by_type(self, data_type: str) -> CacheHandler:
        """Get handler by data type string."""
        for handler in self.handlers:
            if handler.data_type == data_type:
                return handler

        raise ValueError(f"No handler found for data type: {data_type}")

    def register_handler(
        self,
        handler: CacheHandler,
        priority: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        Register a custom handler with optional priority.

        Args:
            handler: Handler instance implementing CacheHandler interface
            priority: Position in handler list (0 = highest priority, None = append to end)
            name: Optional name for the handler (defaults to handler.data_type)

        Raises:
            ValueError: If handler doesn't implement required interface
            ValueError: If handler with same name already exists

        Example:
            >>> class ParquetHandler(CacheHandler):
            ...     @property
            ...     def data_type(self): return "parquet"
            ...     # ... other methods
            >>>
            >>> registry.register_handler(ParquetHandler(), priority=0)
        """
        # Validate handler implements required interface
        self._validate_handler(handler)

        handler_name = name or handler.data_type

        # Check for duplicate registration
        existing_names = [h.data_type for h in self.handlers]
        if handler_name in existing_names:
            raise ValueError(
                f"Handler '{handler_name}' already registered. "
                f"Use unregister_handler() first or provide a unique name."
            )

        # Insert at priority position or append
        if priority is not None:
            if priority < 0:
                priority = 0
            if priority >= len(self.handlers):
                self.handlers.append(handler)
            else:
                self.handlers.insert(priority, handler)
            logger.info(f"Registered handler '{handler_name}' at priority {priority}")
        else:
            self.handlers.append(handler)
            logger.info(f"Registered handler '{handler_name}' at end of priority list")

    def unregister_handler(self, handler_name: str) -> bool:
        """
        Remove a handler by name (data_type).

        Args:
            handler_name: The data_type of the handler to remove

        Returns:
            True if handler was removed, False if not found

        Example:
            >>> registry.unregister_handler("parquet")
            True
        """
        for i, handler in enumerate(self.handlers):
            if handler.data_type == handler_name:
                self.handlers.pop(i)
                logger.info(f"Unregistered handler '{handler_name}'")
                return True

        logger.warning(f"Handler '{handler_name}' not found for unregistration")
        return False

    def list_handlers(self) -> list:
        """
        List all registered handlers with their priority and capabilities.

        Returns:
            List of dictionaries with handler information:
            - name: Handler data_type
            - priority: Position in handler list (lower = higher priority)
            - class: Handler class name
            - is_builtin: Whether it's a built-in handler

        Example:
            >>> for info in registry.list_handlers():
            ...     print(f"{info['priority']}: {info['name']} ({info['class']})")
            0: polars_dataframe (PolarsDataFrameHandler)
            1: pandas_dataframe (PandasDataFrameHandler)
            2: numpy_array (ArrayHandler)
            3: object (ObjectHandler)
        """
        builtin_types = {
            "polars_dataframe",
            "pandas_dataframe",
            "polars_series",
            "pandas_series",
            "numpy_array",
            "object",
            "tensorflow_tensor",
        }

        result = []
        for i, handler in enumerate(self.handlers):
            result.append(
                {
                    "name": handler.data_type,
                    "priority": i,
                    "class": handler.__class__.__name__,
                    "is_builtin": handler.data_type in builtin_types,
                }
            )

        return result

    def _validate_handler(self, handler: Any) -> None:
        """
        Validate that handler implements required CacheHandler interface.

        Raises:
            ValueError: If handler is missing required methods/properties
        """
        required_methods = ["can_handle", "put", "get", "get_file_extension"]
        required_properties = ["data_type"]

        missing = []

        for method in required_methods:
            if not callable(getattr(handler, method, None)):
                missing.append(f"method '{method}'")

        for prop in required_properties:
            if not hasattr(handler, prop):
                missing.append(f"property '{prop}'")

        if missing:
            raise ValueError(
                f"Handler {handler.__class__.__name__} missing required: {', '.join(missing)}. "
                f"Handlers must implement the CacheHandler interface."
            )
