"""
Standardized Error Handling for Cacheness
========================================

This module provides consistent error handling patterns and logging across all cache operations.
"""

import functools
import logging
import traceback
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, Union
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class CacheReason(str, Enum):
    """Stable machine-readable reasons for public cache boundary failures."""

    PATH_TRAVERSAL = "path_traversal"
    PATH_ABSOLUTE = "path_absolute"
    PATH_DRIVE = "path_drive"
    PATH_UNC = "path_unc"
    PATH_ROOTED = "path_rooted"
    PATH_OUTSIDE_ROOT = "path_outside_root"
    PATH_RACE = "path_race"
    INVALID_IDENTIFIER = "invalid_identifier"
    INVALID_QUERY_FIELD = "invalid_query_field"
    INVALID_QUERY_VALUE = "invalid_query_value"
    INVALID_LEGACY_ARRAY = "invalid_legacy_array"
    UNSAFE_OBJECT_ARRAY = "unsafe_object_array"
    UNSUPPORTED_LEGACY_LAYOUT = "unsupported_legacy_layout"
    READ_ONLY_LEGACY_STORE = "read_only_legacy_store"
    INVALID_LEGACY_SIGNATURE = "invalid_legacy_signature"
    SQL_CACHE_FETCH_FAILED = "sql_cache_fetch_failed"
    SQL_CACHE_GAP_DETECTION_FAILED = "sql_cache_gap_detection_failed"
    SQL_CACHE_UPSERT_FAILED = "sql_cache_upsert_failed"
    MISSING_OPTIONAL_DEPENDENCY = "missing_optional_dependency"
    MANIFEST_INVALID = "manifest_invalid"
    MANIFEST_BOUNDS = "manifest_bounds"
    MANIFEST_UNSUPPORTED_VERSION = "manifest_unsupported_version"
    MANIFEST_SIGNATURE_INVALID = "manifest_signature_invalid"
    MANIFEST_SIGNING_KEY_INVALID = "manifest_signing_key_invalid"
    BLOB_MANIFEST_MALFORMED = MANIFEST_INVALID
    BLOB_MANIFEST_UNAUTHENTICATED = MANIFEST_SIGNATURE_INVALID
    BLOB_PAYLOAD_MISSING = MANIFEST_INVALID
    BLOB_PAYLOAD_TAMPERED = MANIFEST_INVALID
    BLOB_MANIFEST_UNSUPPORTED_VERSION = MANIFEST_UNSUPPORTED_VERSION
    BLOB_PAYLOAD_UNSUPPORTED_VERSION = MANIFEST_UNSUPPORTED_VERSION
    BLOB_LIFECYCLE_CONFLICT = "blob_lifecycle_conflict"
    BLOB_BACKEND_FAILURE = "blob_backend_failure"
    BLOB_BACKEND_CAPABILITY_UNSUPPORTED = "blob_backend_capability_unsupported"
    BLOB_MIGRATION_REQUIRED = "blob_migration_required"
    BLOB_RECOVERABLE_CLEANUP = "blob_recoverable_cleanup"
    BLOB_RECONCILIATION_BLOCKED = "blob_reconciliation_blocked"
    BLOB_RECONCILIATION_CONFLICT = "blob_reconciliation_conflict"
    BLOB_RECONCILIATION_CHECKPOINT_INVALID = "blob_reconciliation_checkpoint_invalid"
    BLOB_STORE_CLOSED = "blob_store_closed"
    BLOB_CLOSE_TIMEOUT = "blob_close_timeout"
    BLOB_LIFECYCLE_TIMEOUT = "blob_lifecycle_timeout"
    BLOB_LOCK_RELEASE_FAILURE = "blob_lock_release_failure"
    METADATA_CORRUPT = "metadata_corrupt"


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


def _context_with_reason(
    context: Optional[Dict[str, Any]], reason: CacheReason
) -> Dict[str, Any]:
    """Return error context with its public reason code made authoritative."""
    error_context = dict(context or {})
    error_context["reason"] = reason.value
    return error_context


class CacheUnsafePathError(CacheStorageError):
    """Raised when a filesystem path or persisted locator is unsafe."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        reason: CacheReason,
    ):
        super().__init__(message, _context_with_reason(context, reason))


class CacheQueryValidationError(CacheMetadataError):
    """Raised when public metadata query input is invalid."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        reason: CacheReason,
    ):
        super().__init__(message, _context_with_reason(context, reason))


class CacheLegacyFormatError(CacheSerializationError):
    """Raised when a legacy payload layout is unsafe or unsupported."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        reason: CacheReason,
    ):
        super().__init__(message, _context_with_reason(context, reason))


class CacheBlobIntegrityError(CacheIntegrityError):
    """Base class for direct BlobStore evidence and payload integrity failures."""


class CacheManifestIntegrityError(CacheBlobIntegrityError):
    """Raised when canonical BlobStore evidence is malformed or tampered with."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        reason: CacheReason = CacheReason.MANIFEST_INVALID,
    ):
        super().__init__(message, _context_with_reason(context, reason))


class CacheBlobManifestMalformedError(CacheManifestIntegrityError):
    """Raised when a canonical BlobStore manifest is malformed."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        reason: CacheReason = CacheReason.BLOB_MANIFEST_MALFORMED,
    ):
        super().__init__(message, context, reason=reason)


class CacheBlobManifestUnauthenticatedError(CacheManifestIntegrityError):
    """Raised when a canonical BlobStore manifest cannot be authenticated."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        reason: CacheReason = CacheReason.BLOB_MANIFEST_UNAUTHENTICATED,
    ):
        super().__init__(message, context, reason=reason)


class CacheBlobPayloadMissingError(CacheManifestIntegrityError):
    """Raised when an authenticated BlobStore payload is missing."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        reason: CacheReason = CacheReason.BLOB_PAYLOAD_MISSING,
    ):
        super().__init__(message, context, reason=reason)


class CacheBlobPayloadTamperedError(CacheManifestIntegrityError):
    """Raised when a BlobStore payload fails an integrity check."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        reason: CacheReason = CacheReason.BLOB_PAYLOAD_TAMPERED,
    ):
        super().__init__(message, context, reason=reason)


class CacheManifestUnsupportedVersionError(CacheStorageError):
    """Raised when a manifest or native payload version is not supported."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        reason: CacheReason = CacheReason.MANIFEST_UNSUPPORTED_VERSION,
    ):
        super().__init__(message, _context_with_reason(context, reason))


class CacheBlobManifestUnsupportedVersionError(CacheManifestUnsupportedVersionError):
    """Raised when a BlobStore manifest schema version is unsupported."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        reason: CacheReason = CacheReason.BLOB_MANIFEST_UNSUPPORTED_VERSION,
    ):
        super().__init__(message, context, reason=reason)


class CacheBlobPayloadUnsupportedVersionError(CacheManifestUnsupportedVersionError):
    """Raised when a BlobStore payload format version is unsupported."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        reason: CacheReason = CacheReason.BLOB_PAYLOAD_UNSUPPORTED_VERSION,
    ):
        super().__init__(message, context, reason=reason)


class CacheBlobLifecycleConflictError(CacheStorageError):
    """Raised when a direct BlobStore read sees a non-committed generation."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        reason: CacheReason = CacheReason.BLOB_LIFECYCLE_CONFLICT,
    ):
        super().__init__(message, _context_with_reason(context, reason))


class CacheBlobBackendError(CacheStorageError):
    """Raised when a manifest repository cannot complete a backend operation."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        reason: CacheReason = CacheReason.BLOB_BACKEND_FAILURE,
    ):
        super().__init__(message, _context_with_reason(context, reason))


class CacheBlobLockReleaseError(CacheStorageError):
    """Raised when an otherwise successful lifecycle lock cannot be released."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        reason: CacheReason = CacheReason.BLOB_LOCK_RELEASE_FAILURE,
    ):
        super().__init__(message, _context_with_reason(context, reason))


class CacheBlobStoreClosedError(CacheStorageError):
    """Raised when a BlobStore instance rejects new work during close."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        reason: CacheReason = CacheReason.BLOB_STORE_CLOSED,
    ):
        super().__init__(message, _context_with_reason(context, reason))


class CacheBlobCloseTimeoutError(CacheStorageError):
    """Raised when a finite BlobStore close drain cannot complete in time."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        reason: CacheReason = CacheReason.BLOB_CLOSE_TIMEOUT,
    ):
        super().__init__(message, _context_with_reason(context, reason))


class CacheBlobLifecycleTimeoutError(CacheStorageError):
    """Raised when a bounded lifecycle authority cannot be admitted in time."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        reason: CacheReason = CacheReason.BLOB_LIFECYCLE_TIMEOUT,
    ):
        super().__init__(message, _context_with_reason(context, reason))


class CacheBlobMigrationRequiredError(CacheStorageError):
    """Raised for an exact legacy layout that requires explicit migration."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        reason: CacheReason = CacheReason.BLOB_MIGRATION_REQUIRED,
    ):
        super().__init__(message, _context_with_reason(context, reason))


class CacheBlobRecoverableCleanupError(CacheStorageError):
    """Raised when published authority survives but cleanup needs resumption."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        reason: CacheReason = CacheReason.BLOB_RECOVERABLE_CLEANUP,
    ):
        super().__init__(message, _context_with_reason(context, reason))


class CacheBlobReconciliationError(CacheStorageError):
    """Raised when reconciliation cannot safely apply a requested repair."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        reason: CacheReason = CacheReason.BLOB_RECONCILIATION_BLOCKED,
    ):
        super().__init__(message, _context_with_reason(context, reason))


class CacheBlobReconciliationConflictError(CacheBlobReconciliationError):
    """Raised when exact evidence changes during reconciliation revalidation."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        reason: CacheReason = CacheReason.BLOB_RECONCILIATION_CONFLICT,
    ):
        super().__init__(message, _context_with_reason(context, reason))


class CacheBlobReconciliationCheckpointError(CacheBlobIntegrityError):
    """Raised when persisted reconciliation action progress is malformed."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        reason: CacheReason = CacheReason.BLOB_RECONCILIATION_CHECKPOINT_INVALID,
    ):
        super().__init__(message, _context_with_reason(context, reason))


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
