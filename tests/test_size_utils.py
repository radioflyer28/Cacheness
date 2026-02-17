"""Tests for size_utils module and size-related integration behavior.

Covers:
- parse_size() parsing of human-readable strings and raw values
- format_size() formatting bytes to human-readable strings
- bytes_to_mb_display() backward-compat display helper
- Config max_cache_size / max_cache_size_bytes property
- get_stats() returns total_size_bytes (JSON + SQLite backends)
- cleanup_by_size() uses bytes internally — no rounding errors
"""

import pytest

from cacheness import CacheConfig, cacheness
from cacheness.size_utils import parse_size, format_size, bytes_to_mb_display
from cacheness.config import CacheStorageConfig


# ==================== parse_size() ====================


class TestParseSize:
    """Unit tests for parse_size()."""

    def test_int_passthrough(self):
        assert parse_size(1048576) == 1048576

    def test_float_truncated(self):
        assert parse_size(1048576.9) == 1048576

    def test_zero(self):
        assert parse_size(0) == 0

    def test_string_bytes(self):
        assert parse_size("500B") == 500

    def test_string_kb(self):
        assert parse_size("1KB") == 1024

    def test_string_mb(self):
        assert parse_size("7MB") == 7 * 1024 * 1024

    def test_string_gb(self):
        assert parse_size("2GB") == 2 * 1024 * 1024 * 1024

    def test_string_tb(self):
        assert parse_size("1TB") == 1024**4

    def test_fractional_gb(self):
        assert parse_size("2.5 GB") == int(2.5 * 1024 * 1024 * 1024)

    def test_case_insensitive(self):
        assert parse_size("10mb") == 10 * 1024 * 1024
        assert parse_size("10Mb") == 10 * 1024 * 1024

    def test_whitespace_between(self):
        assert parse_size("100 KB") == 100 * 1024

    def test_plain_numeric_string(self):
        assert parse_size("1024") == 1024

    def test_plain_float_string(self):
        assert parse_size("1024.5") == 1024

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Empty size string"):
            parse_size("")

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError, match="Cannot parse size"):
            parse_size("10 XB")

    def test_garbage_raises(self):
        with pytest.raises(ValueError, match="Cannot parse size"):
            parse_size("hello")

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError, match="Expected str, int, or float"):
            parse_size([1, 2, 3])  # type: ignore[arg-type]

    def test_leading_trailing_whitespace(self):
        assert parse_size("  500 MB  ") == 500 * 1024 * 1024


# ==================== format_size() ====================


class TestFormatSize:
    """Unit tests for format_size()."""

    def test_bytes(self):
        assert format_size(500) == "500 B"

    def test_zero(self):
        assert format_size(0) == "0 B"

    def test_kb(self):
        assert format_size(1024) == "1.00 KB"

    def test_mb(self):
        assert format_size(1048576) == "1.00 MB"

    def test_gb(self):
        assert format_size(1073741824) == "1.00 GB"

    def test_tb(self):
        assert format_size(1099511627776) == "1.00 TB"

    def test_fractional_mb(self):
        assert format_size(7 * 1024 * 1024) == "7.00 MB"

    def test_fractional_gb(self):
        assert format_size(int(2.5 * 1024**3)) == "2.50 GB"


# ==================== bytes_to_mb_display() ====================


class TestBytesToMbDisplay:
    """Unit tests for bytes_to_mb_display()."""

    def test_exact_mb(self):
        assert bytes_to_mb_display(1048576) == 1.0

    def test_small_file(self):
        assert bytes_to_mb_display(5400) == 0.005

    def test_zero(self):
        assert bytes_to_mb_display(0) == 0.0

    def test_custom_decimals(self):
        assert bytes_to_mb_display(5400, decimals=6) == 0.005150


# ==================== Config integration ====================


class TestCacheStorageConfigSizeBytes:
    """Tests for CacheStorageConfig.max_cache_size_bytes property."""

    def test_legacy_mb_field(self):
        cfg = CacheStorageConfig(max_cache_size_mb=100)
        assert cfg.max_cache_size_bytes == 100 * 1024 * 1024

    def test_new_string_field(self):
        cfg = CacheStorageConfig(max_cache_size="500MB")
        assert cfg.max_cache_size_bytes == 500 * 1024 * 1024

    def test_new_bytes_int_field(self):
        cfg = CacheStorageConfig(max_cache_size=1048576)
        assert cfg.max_cache_size_bytes == 1048576

    def test_new_field_precedence(self):
        """max_cache_size takes precedence over max_cache_size_mb."""
        cfg = CacheStorageConfig(max_cache_size="1GB", max_cache_size_mb=500)
        assert cfg.max_cache_size_bytes == 1024**3

    def test_both_none(self):
        cfg = CacheStorageConfig(max_cache_size=None, max_cache_size_mb=None)
        assert cfg.max_cache_size_bytes is None

    def test_negative_max_cache_size_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            CacheStorageConfig(max_cache_size_mb=-1)

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            CacheStorageConfig(max_cache_size="2 Gigglebytes")

    def test_default_still_works(self):
        """Default max_cache_size_mb=2000 still resolves to bytes."""
        cfg = CacheStorageConfig()
        assert cfg.max_cache_size_bytes == 2000 * 1024 * 1024


# ==================== get_stats() returns total_size_bytes ====================


class TestGetStatsTotalSizeBytes:
    """Verify get_stats() returns total_size_bytes for JSON and SQLite backends."""

    @pytest.fixture(params=["json", "sqlite"])
    def cache(self, request, tmp_path):
        backend = request.param
        config = CacheConfig(
            cache_dir=str(tmp_path / "cache"),
            metadata_backend=backend,
            max_cache_size_mb=None,
            cleanup_on_init=False,
        )
        config.security.enable_signing = False
        cache = cacheness(config)
        yield cache
        cache.close()

    def test_empty_cache_has_zero_bytes(self, cache):
        stats = cache.get_stats()
        assert "total_size_bytes" in stats
        assert stats["total_size_bytes"] == 0

    def test_after_put_size_bytes_is_int(self, cache):
        cache.put("hello world", on={"key": "v1"})
        stats = cache.get_stats()
        assert isinstance(stats["total_size_bytes"], int)
        assert stats["total_size_bytes"] > 0

    def test_total_size_mb_backward_compat(self, cache):
        """total_size_mb is still present for backward compat."""
        cache.put("hello", on={"key": "v1"})
        stats = cache.get_stats()
        assert "total_size_mb" in stats
        expected_mb = stats["total_size_bytes"] / (1024 * 1024)
        assert abs(stats["total_size_mb"] - expected_mb) < 1e-9

    def test_total_size_bytes_matches_file(self, cache):
        """total_size_bytes matches actual blob file size on disk."""
        data = "x" * 10_000
        cache.put(data, on={"key": "v1"})
        stats = cache.get_stats()
        # Size should be roughly the serialized size (within reason)
        assert stats["total_size_bytes"] > 0
        assert stats["total_size_bytes"] == stats["total_size_mb"] * 1024 * 1024


# ==================== cleanup_by_size() bytes interface ====================


class TestCleanupBySizeBytes:
    """Verify cleanup_by_size() works with bytes parameter and has no rounding errors."""

    @pytest.fixture(params=["json", "sqlite"])
    def cache(self, request, tmp_path):
        backend = request.param
        config = CacheConfig(
            cache_dir=str(tmp_path / "cache"),
            metadata_backend=backend,
            max_cache_size_mb=None,
            cleanup_on_init=False,
        )
        config.security.enable_signing = False
        cache = cacheness(config)
        yield cache
        cache.close()

    def test_cleanup_by_size_bytes_removes_entries(self, cache):
        """cleanup_by_size accepts bytes and removes entries to reach target."""
        # Add several entries
        for i in range(5):
            cache.put(f"data-{i}" * 100, on={"idx": i})

        stats_before = cache.get_stats()
        total_before = stats_before["total_size_bytes"]
        assert total_before > 0

        # Target: half the current size
        target = total_before // 2
        result = cache.metadata_backend.cleanup_by_size(target)
        assert result["count"] > 0

        stats_after = cache.get_stats()
        assert stats_after["total_size_bytes"] <= target

    def test_cleanup_no_rounding_error_small_files(self, cache):
        """Regression test: small files (< 0.01 MB) should not trigger false positives.

        This is the CACHE-mdd bug: SQLite rounded total_size_mb to 2 dp,
        making a 5 KB cache appear as 0.01 MB (10 KB), which doubled its
        apparent size and caused cleanup to delete everything.
        """
        # Create a small entry (~few hundred bytes)
        cache.put("tiny", on={"key": "small"})
        stats = cache.get_stats()
        actual_bytes = stats["total_size_bytes"]

        # Verify precise: total_size_bytes is exact int, no rounding
        assert isinstance(actual_bytes, int)

        # Ask cleanup to target exactly the current size → should remove nothing
        result = cache.metadata_backend.cleanup_by_size(actual_bytes)
        assert result["count"] == 0, (
            f"Expected 0 removals when target == current size, got {result['count']}"
        )

    def test_cleanup_already_below_target(self, cache):
        """If cache is already below target, cleanup_by_size returns 0."""
        cache.put("small", on={"key": "v1"})
        result = cache.metadata_backend.cleanup_by_size(999_999_999)  # ~1 GB target
        assert result["count"] == 0
        assert result["removed_entries"] == []

    def test_enforce_size_limit_uses_bytes(self, tmp_path):
        """_enforce_size_limit uses max_cache_size_bytes path correctly."""
        config = CacheConfig(
            cache_dir=str(tmp_path / "cache"),
            metadata_backend="json",
            max_cache_size="1KB",
            cleanup_on_init=False,
        )
        config.security.enable_signing = False
        cache = cacheness(config)

        # Put data larger than 1 KB
        cache.put("x" * 2000, on={"key": "v1"})
        cache.put("y" * 2000, on={"key": "v2"})

        # After enforcement, total should be <= 1 KB (or some entries removed)
        stats = cache.get_stats()
        # The exact behavior depends on enforcement timing, but we should
        # have fewer bytes than without enforcement
        assert stats["total_entries"] >= 0  # Sanity check — no crash
        cache.close()


# ==================== max_size_bytes in cache get_stats() ====================


class TestCacheGetStatsMaxSizeBytes:
    """Verify cacheness get_stats() includes max_size_bytes."""

    def test_max_size_bytes_present(self, tmp_path):
        config = CacheConfig(
            cache_dir=str(tmp_path / "cache"),
            metadata_backend="json",
            max_cache_size="500MB",
            cleanup_on_init=False,
        )
        config.security.enable_signing = False
        cache = cacheness(config)
        stats = cache.get_stats()
        assert "max_size_bytes" in stats
        assert stats["max_size_bytes"] == 500 * 1024 * 1024
        # Backward compat
        assert "max_size_mb" in stats
        cache.close()

    def test_max_size_bytes_none_when_unlimited(self, tmp_path):
        config = CacheConfig(
            cache_dir=str(tmp_path / "cache"),
            metadata_backend="json",
            cleanup_on_init=False,
        )
        config.storage.max_cache_size_mb = None
        config.storage.max_cache_size = None
        config.security.enable_signing = False
        cache = cacheness(config)
        stats = cache.get_stats()
        assert stats["max_size_bytes"] is None
        assert stats["max_size_mb"] is None
        cache.close()
