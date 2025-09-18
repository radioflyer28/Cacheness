"""
Tests for the error_handling module.

This module tests all error handling functionality including:
- Exception hierarchy and custom errors
- Error handling decorators
- File operation safety
- Path validation
- Error context management
- Performance logging
- Import error handling
- Error summarization
"""

import logging
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from contextlib import redirect_stderr
import io

from cacheness.error_handling import (
    CacheError,
    CacheConfigurationError,
    CacheStorageError,
    CacheSerializationError,
    CacheHandlerError,
    CacheIntegrityError,
    CacheMetadataError,
    with_error_handling,
    cache_operation_context,
    log_cache_performance,
    validate_file_path,
    safe_file_operation,
    handle_import_errors,
    log_configuration_validation,
    ErrorSummary,
)


class TestCacheErrorHierarchy:
    """Test the cache error exception hierarchy."""

    def test_cache_error_base_initialization(self):
        """Test CacheError base class initialization."""
        # Test without context
        error = CacheError("Test message")
        assert str(error) == "Test message"
        assert error.context == {}

        # Test with context
        context = {"key1": "value1", "key2": 42}
        error = CacheError("Test message", context)
        assert str(error) == "Test message"
        assert error.context == context

    def test_cache_error_logging(self, caplog):
        """Test that CacheError logs errors with context."""
        with caplog.at_level(logging.ERROR):
            context = {"operation": "test", "file": "test.txt"}
            CacheError("Test error", context)
            
        assert "Cache error: Test error" in caplog.text
        assert "operation=test" in caplog.text
        assert "file=test.txt" in caplog.text

    def test_specific_error_types(self):
        """Test all specific cache error types inherit correctly."""
        error_types = [
            CacheConfigurationError,
            CacheStorageError,
            CacheSerializationError,
            CacheHandlerError,
            CacheIntegrityError,
            CacheMetadataError,
        ]

        for error_type in error_types:
            error = error_type("Test message", {"type": error_type.__name__})
            assert isinstance(error, CacheError)
            assert isinstance(error, error_type)
            assert str(error) == "Test message"
            assert error.context["type"] == error_type.__name__


class TestWithErrorHandlingDecorator:
    """Test the with_error_handling decorator."""

    def test_decorator_reraises_cache_errors(self):
        """Test that decorator re-raises CacheError types as-is."""
        @with_error_handling()
        def failing_function():
            raise CacheStorageError("Original cache error")

        with pytest.raises(CacheStorageError) as exc_info:
            failing_function()
        
        assert str(exc_info.value) == "Original cache error"

    def test_decorator_converts_other_exceptions(self):
        """Test that decorator converts non-cache exceptions."""
        @with_error_handling(error_type=CacheHandlerError)
        def failing_function():
            raise ValueError("Original error")

        with pytest.raises(CacheHandlerError) as exc_info:
            failing_function()
        
        assert "Error in failing_function: Original error" in str(exc_info.value)
        assert exc_info.value.context["function"] == "failing_function"
        assert exc_info.value.context["original_error"] == "Original error"
        assert exc_info.value.context["original_error_type"] == "ValueError"

    def test_decorator_with_context(self):
        """Test decorator with additional context."""
        context = {"operation": "test_op", "file": "test.txt"}
        
        @with_error_handling(context=context)
        def failing_function():
            raise RuntimeError("Test error")

        with pytest.raises(CacheError) as exc_info:
            failing_function()
        
        assert exc_info.value.context["operation"] == "test_op"
        assert exc_info.value.context["file"] == "test.txt"
        assert exc_info.value.context["function"] == "failing_function"

    def test_decorator_no_reraise_mode(self, caplog):
        """Test decorator in no-reraise mode."""
        with caplog.at_level(logging.WARNING):
            @with_error_handling(reraise=False, default_return="default")
            def failing_function():
                raise ValueError("Test error")

            result = failing_function()
        
        assert result == "default"
        assert "Suppressed error in failing_function: Test error" in caplog.text

    def test_decorator_preserves_successful_returns(self):
        """Test that decorator doesn't interfere with successful operations."""
        @with_error_handling()
        def successful_function(value):
            return value * 2

        result = successful_function(21)
        assert result == 42

    def test_decorator_with_args_and_kwargs(self):
        """Test decorator properly handles function arguments."""
        @with_error_handling()
        def function_with_args(arg1, arg2, kwarg1=None, kwarg2=None):
            raise ValueError("Test error")

        with pytest.raises(CacheError) as exc_info:
            function_with_args("a", "b", kwarg1="c", kwarg2="d")
        
        assert exc_info.value.context["args_count"] == 2
        assert set(exc_info.value.context["kwargs_keys"]) == {"kwarg1", "kwarg2"}


class TestCacheOperationContext:
    """Test the cache_operation_context context manager."""

    def test_successful_operation(self, caplog):
        """Test context manager with successful operation."""
        with caplog.at_level(logging.DEBUG):
            with cache_operation_context("test_operation", key="value"):
                time.sleep(0.01)  # Small delay to test duration logging
        
        logs = caplog.text
        assert "Starting cache operation: test_operation" in logs
        assert "Cache operation completed: test_operation" in logs
        assert "s)" in logs  # Check duration is logged

    def test_cache_error_propagation(self, caplog):
        """Test context manager with CacheError."""
        with caplog.at_level(logging.ERROR):
            with pytest.raises(CacheStorageError):
                with cache_operation_context("test_operation"):
                    raise CacheStorageError("Test cache error")
        
        assert "Cache operation failed: test_operation" in caplog.text

    def test_non_cache_error_conversion(self, caplog):
        """Test context manager logs but doesn't convert non-cache errors."""
        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError):  # Should not be converted anymore
                with cache_operation_context("test_operation", file="test.txt"):
                    raise ValueError("Test error")
        
        assert "Unexpected error in cache operation: test_operation" in caplog.text


class TestLogCachePerformance:
    """Test the log_cache_performance decorator."""

    def test_successful_operation_logging(self, caplog):
        """Test performance logging for successful operations."""
        with caplog.at_level(logging.DEBUG):
            @log_cache_performance
            def test_function():
                time.sleep(0.01)
                return "success"

            result = test_function()
        
        assert result == "success"
        assert "Cache operation test_function completed in" in caplog.text
        assert "s" in caplog.text

    def test_failed_operation_logging(self, caplog):
        """Test performance logging for failed operations."""
        with caplog.at_level(logging.WARNING):
            @log_cache_performance
            def failing_function():
                time.sleep(0.01)
                raise ValueError("Test error")

            with pytest.raises(ValueError):
                failing_function()
        
        assert "Cache operation failing_function failed after" in caplog.text
        assert "Test error" in caplog.text


class TestValidateFilePath:
    """Test the validate_file_path function."""

    def test_valid_file_path(self):
        """Test validation of valid file paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_file.txt"
            result = validate_file_path(file_path)
            
            assert isinstance(result, Path)
            assert result == file_path
            assert result.parent.exists()  # Parent directory should be created

    def test_existing_file_requirement(self):
        """Test must_exist parameter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with existing file
            existing_file = Path(temp_dir) / "existing.txt"
            existing_file.write_text("test")
            
            result = validate_file_path(existing_file, must_exist=True)
            assert result == existing_file
            
            # Test with non-existing file
            non_existing = Path(temp_dir) / "non_existing.txt"
            with pytest.raises(CacheStorageError) as exc_info:
                validate_file_path(non_existing, must_exist=True)
            
            assert "Required file does not exist" in str(exc_info.value)
            assert str(non_existing) in exc_info.value.context["file_path"]

    def test_empty_filename_error(self):
        """Test error on empty filename."""
        with pytest.raises(CacheStorageError) as exc_info:
            validate_file_path("")
        
        assert "Invalid file path: empty filename" in str(exc_info.value)

    def test_string_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path_str = str(Path(temp_dir) / "test.txt")
            result = validate_file_path(file_path_str)
            
            assert isinstance(result, Path)
            assert str(result) == file_path_str

    @patch('pathlib.Path.mkdir')
    def test_os_error_handling(self, mock_mkdir):
        """Test OSError handling during path validation."""
        mock_mkdir.side_effect = OSError("Permission denied")
        
        with pytest.raises(CacheStorageError) as exc_info:
            validate_file_path("/invalid/path/file.txt")
        
        assert "File system error" in str(exc_info.value)
        assert "Permission denied" in str(exc_info.value)


class TestSafeFileOperation:
    """Test the safe_file_operation function."""

    def test_successful_file_operation(self, caplog):
        """Test successful file operation."""
        with caplog.at_level(logging.DEBUG):
            def mock_operation(content):
                return f"processed: {content}"
            
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = Path(temp_dir) / "test.txt"
                result = safe_file_operation(
                    "test write", file_path, mock_operation, "test content"
                )
        
        assert result == "processed: test content"
        assert "Starting cache operation: test write" in caplog.text

    def test_permission_error_handling(self):
        """Test handling of permission errors."""
        def failing_operation():
            raise PermissionError("Access denied")
        
        file_path = Path("/test/file.txt")
        with pytest.raises(CacheStorageError) as exc_info:
            safe_file_operation("test operation", file_path, failing_operation)
        
        assert "Permission denied for test operation" in str(exc_info.value)
        assert exc_info.value.context["operation"] == "test operation"
        assert exc_info.value.context["file_path"] == str(file_path)

    def test_os_error_handling(self):
        """Test handling of general OS errors."""
        def failing_operation():
            raise OSError("Disk full")
        
        file_path = Path("/test/file.txt")
        with pytest.raises(CacheStorageError) as exc_info:
            safe_file_operation("test operation", file_path, failing_operation)
        
        assert "File system error during test operation" in str(exc_info.value)
        assert "Disk full" in str(exc_info.value)


class TestHandleImportErrors:
    """Test the handle_import_errors decorator."""

    def test_successful_import_operation(self):
        """Test decorator doesn't interfere with successful operations."""
        @handle_import_errors("test_module")
        def successful_function():
            return "success"
        
        result = successful_function()
        assert result == "success"

    def test_import_error_handling(self, caplog):
        """Test import error conversion."""
        with caplog.at_level(logging.ERROR):
            @handle_import_errors("missing_module", "test functionality")
            def failing_function():
                raise ImportError("No module named 'missing_module'")
            
            with pytest.raises(CacheConfigurationError) as exc_info:
                failing_function()
        
        assert "Missing dependency 'missing_module'" in str(exc_info.value)
        assert "required for test functionality" in str(exc_info.value)
        assert exc_info.value.context["module_name"] == "missing_module"
        assert exc_info.value.context["required_for"] == "test functionality"
        assert "Install with: pip install missing_module" in caplog.text

    def test_import_error_without_required_for(self):
        """Test import error handling without required_for parameter."""
        @handle_import_errors("missing_module")
        def failing_function():
            raise ImportError("No module named 'missing_module'")
        
        with pytest.raises(CacheConfigurationError) as exc_info:
            failing_function()
        
        assert "Missing dependency 'missing_module'" in str(exc_info.value)
        assert "required for" not in str(exc_info.value)
        assert exc_info.value.context["required_for"] is None


class TestLogConfigurationValidation:
    """Test the log_configuration_validation decorator."""

    def test_successful_validation_logging(self, caplog):
        """Test logging for successful configuration validation."""
        with caplog.at_level(logging.DEBUG):
            class TestConfig:
                @log_configuration_validation("TestConfig")
                def validate(self):
                    return True
            
            config = TestConfig()
            result = config.validate()
        
        assert result is True
        assert "TestConfig configuration validated successfully" in caplog.text

    def test_failed_validation_logging(self, caplog):
        """Test logging for failed configuration validation."""
        with caplog.at_level(logging.ERROR):
            class TestConfig:
                @log_configuration_validation("TestConfig")
                def validate(self):
                    raise ValueError("Invalid configuration")
            
            config = TestConfig()
            with pytest.raises(ValueError):
                config.validate()
        
        assert "TestConfig configuration validation failed" in caplog.text
        assert "Invalid configuration" in caplog.text


class TestErrorSummary:
    """Test the ErrorSummary class."""

    def test_error_summary_initialization(self):
        """Test ErrorSummary initialization."""
        summary = ErrorSummary()
        assert summary.errors == []
        assert summary.warnings == []
        assert not summary.has_errors()
        assert not summary.has_warnings()

    def test_add_error(self):
        """Test adding errors to summary."""
        summary = ErrorSummary()
        error = ValueError("Test error")
        context = {"key": "value"}
        
        summary.add_error(error, context)
        
        assert summary.has_errors()
        assert len(summary.errors) == 1
        
        error_info = summary.errors[0]
        assert error_info["error"] == error
        assert error_info["type"] == "ValueError"
        assert error_info["message"] == "Test error"
        assert error_info["context"] == context
        assert "traceback" in error_info

    def test_add_warning(self):
        """Test adding warnings to summary."""
        summary = ErrorSummary()
        context = {"operation": "test"}
        
        summary.add_warning("Test warning", context)
        
        assert summary.has_warnings()
        assert len(summary.warnings) == 1
        
        warning_info = summary.warnings[0]
        assert warning_info["message"] == "Test warning"
        assert warning_info["context"] == context

    def test_log_summary_with_errors(self, caplog):
        """Test logging summary with errors."""
        with caplog.at_level(logging.ERROR):
            summary = ErrorSummary()
            summary.add_error(ValueError("Error 1"))
            summary.add_error(TypeError("Error 2"))
            
            summary.log_summary()
        
        logs = caplog.text
        assert "Cache operation completed with 2 error(s)" in logs
        assert "Error 1: ValueError: Error 1" in logs
        assert "Error 2: TypeError: Error 2" in logs

    def test_log_summary_with_warnings(self, caplog):
        """Test logging summary with warnings."""
        with caplog.at_level(logging.WARNING):
            summary = ErrorSummary()
            summary.add_warning("Warning 1")
            summary.add_warning("Warning 2")
            
            summary.log_summary()
        
        logs = caplog.text
        assert "Cache operation completed with 2 warning(s)" in logs
        assert "Warning 1: Warning 1" in logs
        assert "Warning 2: Warning 2" in logs

    def test_log_summary_success(self, caplog):
        """Test logging summary with no errors or warnings."""
        with caplog.at_level(logging.DEBUG):
            summary = ErrorSummary()
            summary.log_summary()
        
        assert "Cache operation completed successfully with no errors or warnings" in caplog.text

    def test_get_error_report(self):
        """Test getting detailed error report."""
        summary = ErrorSummary()
        summary.add_error(ValueError("Test error"), {"key": "value"})
        summary.add_warning("Test warning", {"operation": "test"})
        
        report = summary.get_error_report()
        
        assert report["error_count"] == 1
        assert report["warning_count"] == 1
        assert len(report["errors"]) == 1
        assert len(report["warnings"]) == 1
        assert report["errors"][0]["type"] == "ValueError"
        assert report["warnings"][0]["message"] == "Test warning"


class TestErrorHandlingIntegration:
    """Integration tests for error handling components."""

    def test_decorator_with_context_manager(self, caplog):
        """Test decorator and context manager working together."""
        with caplog.at_level(logging.DEBUG):
            @with_error_handling(error_type=CacheStorageError)
            def operation_with_context():
                with cache_operation_context("test_operation"):
                    raise ValueError("Test error")
            
            # Now the decorator should properly convert to CacheStorageError
            with pytest.raises(CacheStorageError):
                operation_with_context()
        
        logs = caplog.text
        assert "Starting cache operation: test_operation" in logs
        assert "Unexpected error in cache operation" in logs

    def test_file_operations_with_validation_and_safety(self):
        """Test file validation and safe operations together."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.txt"
            
            # Validate path
            validated_path = validate_file_path(file_path)
            
            # Use safe operation
            def write_content(content):
                validated_path.write_text(content)
                return "written"
            
            result = safe_file_operation(
                "write test", validated_path, write_content, "test content"
            )
            
            assert result == "written"
            assert validated_path.read_text() == "test content"

    def test_performance_logging_with_error_handling(self, caplog):
        """Test performance logging combined with error handling."""
        with caplog.at_level(logging.WARNING):
            @log_cache_performance
            @with_error_handling(reraise=False, default_return="fallback")
            def slow_failing_operation():
                time.sleep(0.01)
                raise ValueError("Simulated error")
            
            result = slow_failing_operation()
        
        assert result == "fallback"
        logs = caplog.text
        # The performance decorator should not log "failed after" when the error is suppressed
        assert "Suppressed error" in logs

    def test_comprehensive_error_context_propagation(self):
        """Test that error context is properly propagated through layers."""
        @with_error_handling(
            error_type=CacheHandlerError,
            context={"layer": "handler", "operation": "process"}
        )
        def handler_operation():
            with cache_operation_context("handler_process", item_id="123"):
                raise RuntimeError("Core failure")
        
        # Now the decorator should properly convert to CacheHandlerError
        with pytest.raises(CacheHandlerError) as exc_info:
            handler_operation()
        
        # Check that context from decorator is preserved
        context = exc_info.value.context
        assert context["layer"] == "handler"
        assert context["operation"] == "process"
        assert context["function"] == "handler_operation"
        assert context["original_error_type"] == "RuntimeError"
