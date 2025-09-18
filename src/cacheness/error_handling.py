"""
Standardized Error Handling for Cacheness
========================================

This module provides consistent error handling patterns and logging across all cache operations.
"""

import functools
import logging
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, Union
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class CacheError(Exception):
    """Base exception for all cache-related errors."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.context = context or {}
        super().__init__(message)

        # Log error with context for debugging
        context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
        logger.error(
            f"Cache error: {message}" + (f" ({context_str})" if context_str else "")
        )


class CacheConfigurationError(CacheError):
    """Raised when cache configuration is invalid."""

    pass


class CacheStorageError(CacheError):
    """Raised when cache storage operations fail."""

    pass


class CacheSerializationError(CacheError):
    """Raised when cache key serialization fails."""

    pass


class CacheHandlerError(CacheError):
    """Raised when cache handler operations fail."""

    pass


class CacheIntegrityError(CacheError):
    """Raised when cache integrity verification fails."""

    pass


class CacheMetadataError(CacheError):
    """Raised when cache metadata operations fail."""

    pass


def with_error_handling(
    error_type: Type[CacheError] = CacheError,
    context: Optional[Dict[str, Any]] = None,
    reraise: bool = True,
    default_return: Any = None,
):
    """
    Decorator for standardized error handling in cache operations.

    Args:
        error_type: Type of CacheError to raise
        context: Additional context to include in error
        reraise: Whether to reraise the exception after logging
        default_return: Value to return if not reraising
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except CacheError:
                # Re-raise cache errors as-is
                raise
            except Exception as e:
                # Convert other exceptions to cache errors
                error_context = (context or {}).copy()
                error_context.update(
                    {
                        "function": func.__name__,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()),
                        "original_error": str(e),
                        "original_error_type": type(e).__name__,
                    }
                )

                error_msg = f"Error in {func.__name__}: {e}"
                cache_error = error_type(error_msg, error_context)

                if reraise:
                    raise cache_error from e
                else:
                    logger.warning(f"Suppressed error in {func.__name__}: {e}")
                    return default_return

        return wrapper

    return decorator


@contextmanager
def cache_operation_context(operation: str, **context):
    """
    Context manager for cache operations with standardized logging and error handling.

    Args:
        operation: Description of the operation
        **context: Additional context for logging
    """
    logger.debug(f"Starting cache operation: {operation}", extra=context)
    start_time = None

    try:
        import time

        start_time = time.time()
        yield

        if start_time:
            duration = time.time() - start_time
            logger.debug(
                f"Cache operation completed: {operation} ({duration:.3f}s)",
                extra=context,
            )

    except CacheError:
        logger.error(f"Cache operation failed: {operation}", extra=context)
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error in cache operation: {operation} - {e}", extra=context
        )
        # Let decorators handle error type conversion - just re-raise
        raise


def log_cache_performance(func: Callable) -> Callable:
    """Decorator to log performance metrics for cache operations."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import time

        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            # Log performance info
            logger.debug(
                f"Cache operation {func.__name__} completed in {duration:.3f}s"
            )

            return result

        except Exception as e:
            duration = time.time() - start_time
            logger.warning(
                f"Cache operation {func.__name__} failed after {duration:.3f}s: {e}"
            )
            raise

    return wrapper


def validate_file_path(file_path: Union[str, Path], must_exist: bool = False) -> Path:
    """
    Validate and normalize file paths with proper error handling.

    Args:
        file_path: File path to validate
        must_exist: Whether the file must already exist

    Returns:
        Validated Path object

    Raises:
        CacheStorageError: If path validation fails
    """
    try:
        path = Path(file_path)

        # Validate path components
        if not path.name:
            raise CacheStorageError(
                "Invalid file path: empty filename", {"file_path": str(file_path)}
            )

        # Check existence if required
        if must_exist and not path.exists():
            raise CacheStorageError(
                f"Required file does not exist: {path}", {"file_path": str(file_path)}
            )

        # Ensure parent directory exists if creating new file
        if not must_exist:
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {path.parent}")

        return path

    except OSError as e:
        raise CacheStorageError(
            f"File system error: {e}",
            {"file_path": str(file_path), "must_exist": must_exist},
        ) from e


def safe_file_operation(
    operation: str, file_path: Path, func: Callable, *args, **kwargs
):
    """
    Perform file operations with proper error handling and cleanup.

    Args:
        operation: Description of the operation
        file_path: File being operated on
        func: Function to call
        *args, **kwargs: Arguments for the function

    Returns:
        Result of the function call
    """
    with cache_operation_context(operation, file_path=str(file_path)):
        try:
            return func(*args, **kwargs)
        except PermissionError as e:
            raise CacheStorageError(
                f"Permission denied for {operation}: {file_path}",
                {"operation": operation, "file_path": str(file_path)},
            ) from e
        except OSError as e:
            raise CacheStorageError(
                f"File system error during {operation}: {e}",
                {"operation": operation, "file_path": str(file_path)},
            ) from e


def handle_import_errors(
    module_name: str, required_for: Optional[str] = None
) -> Callable:
    """
    Decorator to handle import errors gracefully with informative messages.

    Args:
        module_name: Name of the module being imported
        required_for: What functionality requires this module

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ImportError as e:
                error_msg = f"Missing dependency '{module_name}'"
                if required_for:
                    error_msg += f" required for {required_for}"

                context = {
                    "module_name": module_name,
                    "required_for": required_for,
                    "function": func.__name__,
                }

                logger.error(f"{error_msg}. Install with: pip install {module_name}")
                raise CacheConfigurationError(error_msg, context) from e

        return wrapper

    return decorator


def log_configuration_validation(config_class: str):
    """
    Decorator to log configuration validation results.

    Args:
        config_class: Name of the configuration class being validated
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                result = func(self, *args, **kwargs)
                logger.debug(f"{config_class} configuration validated successfully")
                return result
            except Exception as e:
                logger.error(f"{config_class} configuration validation failed: {e}")
                raise

        return wrapper

    return decorator


class ErrorSummary:
    """Utility class for collecting and summarizing cache errors."""

    def __init__(self):
        self.errors = []
        self.warnings = []

    def add_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Add an error to the summary."""
        self.errors.append(
            {
                "error": error,
                "type": type(error).__name__,
                "message": str(error),
                "context": context or {},
                "traceback": traceback.format_exc(),
            }
        )

    def add_warning(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Add a warning to the summary."""
        self.warnings.append({"message": message, "context": context or {}})

    def has_errors(self) -> bool:
        """Check if any errors were recorded."""
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        """Check if any warnings were recorded."""
        return len(self.warnings) > 0

    def log_summary(self):
        """Log a summary of all errors and warnings."""
        if self.has_errors():
            logger.error(f"Cache operation completed with {len(self.errors)} error(s)")
            for i, error_info in enumerate(self.errors, 1):
                logger.error(
                    f"Error {i}: {error_info['type']}: {error_info['message']}"
                )

        if self.has_warnings():
            logger.warning(
                f"Cache operation completed with {len(self.warnings)} warning(s)"
            )
            for i, warning_info in enumerate(self.warnings, 1):
                logger.warning(f"Warning {i}: {warning_info['message']}")

        if not self.has_errors() and not self.has_warnings():
            logger.debug(
                "Cache operation completed successfully with no errors or warnings"
            )

    def get_error_report(self) -> Dict[str, Any]:
        """Get a detailed error report."""
        return {
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
        }
