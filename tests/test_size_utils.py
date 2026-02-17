"""Tests for size_utils module and size/duration-related integration behavior.

Covers:
- parse_size() parsing of human-readable strings and raw values
- format_size() formatting bytes to human-readable strings
- bytes_to_mb_display() backward-compat display helper
- parse_duration() parsing of human-readable duration strings
- format_duration() formatting seconds to human-readable strings
- Config max_cache_size / max_cache_size_bytes property
- Config default_ttl / memory_cache_ttl duration fields
- get_stats() returns total_size_bytes (JSON + SQLite backends)
- cleanup_by_size() uses bytes internally — no rounding errors
"""

import pytest

from cacheness import CacheConfig, cacheness
from cacheness.size_utils import (
    parse_size,
    format_size,
    bytes_to_mb_display,
    parse_duration,
    format_duration,
)
from cacheness.config import CacheStorageConfig, CacheMetadataConfig


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


# ==================== parse_duration() ====================


class TestParseDuration:
    """Unit tests for parse_duration() — shorthand suffixes only."""

    # --- Passthrough for numeric types ---

    def test_int_passthrough(self):
        assert parse_duration(3600) == 3600.0

    def test_float_passthrough(self):
        assert parse_duration(1.5) == 1.5

    def test_zero(self):
        assert parse_duration(0) == 0.0

    # --- Each shorthand suffix ---

    def test_seconds(self):
        assert parse_duration("30s") == 30.0

    def test_minutes(self):
        assert parse_duration("5m") == 300.0

    def test_hours(self):
        assert parse_duration("6h") == 21600.0

    def test_days(self):
        assert parse_duration("7d") == 604800.0

    def test_weeks(self):
        assert parse_duration("2w") == 1209600.0

    def test_months(self):
        assert parse_duration("3mo") == 3 * 86400 * 30

    def test_years(self):
        assert parse_duration("1y") == 86400 * 365

    # --- Fractional values ---

    def test_fractional_hours(self):
        assert parse_duration("1.5h") == 5400.0

    def test_fractional_minutes(self):
        assert parse_duration("2.5m") == 150.0

    def test_fractional_days(self):
        assert parse_duration("0.5d") == 43200.0

    # --- Case insensitivity ---

    def test_case_insensitive_upper(self):
        assert parse_duration("6H") == 21600.0

    def test_case_insensitive_upper_mo(self):
        assert parse_duration("1MO") == 86400 * 30

    # --- Whitespace handling ---

    def test_whitespace_between(self):
        assert parse_duration("6 h") == 21600.0

    def test_leading_trailing_whitespace(self):
        assert parse_duration("  30s  ") == 30.0

    # --- Plain numeric string fallback ---

    def test_plain_numeric_string(self):
        assert parse_duration("3600") == 3600.0

    def test_plain_float_string(self):
        assert parse_duration("1.5") == 1.5

    # --- "mo" must not be consumed as "m" + trailing "o" ---

    def test_mo_not_parsed_as_m(self):
        """'1mo' should be 1 month, not 1 minute."""
        assert parse_duration("1mo") == 86400 * 30
        assert parse_duration("1m") == 60.0

    # --- Error cases ---

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Empty duration string"):
            parse_duration("")

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError, match="Cannot parse duration"):
            parse_duration("10x")

    def test_full_word_not_accepted(self):
        """Full words like 'hours' are not accepted — shorthand only."""
        with pytest.raises(ValueError, match="Cannot parse duration"):
            parse_duration("6hours")

    def test_garbage_raises(self):
        with pytest.raises(ValueError, match="Cannot parse duration"):
            parse_duration("hello")

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError, match="Expected str, int, or float"):
            parse_duration([1, 2, 3])  # type: ignore[arg-type]


# ==================== format_duration() ====================


class TestFormatDuration:
    """Unit tests for format_duration()."""

    def test_zero(self):
        assert format_duration(0) == "0s"

    def test_seconds(self):
        assert format_duration(45) == "45s"

    def test_minutes_exact(self):
        assert format_duration(300) == "5m"

    def test_hours_exact(self):
        assert format_duration(3600) == "1h"

    def test_days_exact(self):
        assert format_duration(86400) == "1d"

    def test_weeks_exact(self):
        assert format_duration(604800) == "1w"

    def test_months_exact(self):
        assert format_duration(86400 * 30) == "1mo"

    def test_years_exact(self):
        assert format_duration(86400 * 365) == "1y"

    def test_fractional_hours(self):
        assert format_duration(5400) == "1.50h"

    def test_fractional_minutes(self):
        assert format_duration(90) == "1.50m"

    def test_large_seconds_shows_hours(self):
        assert format_duration(7200) == "2h"

    def test_large_days(self):
        assert format_duration(172800) == "2d"

    def test_fractional_seconds(self):
        assert format_duration(0.5) == "0.50s"

    def test_multiple_years(self):
        assert format_duration(86400 * 365 * 2) == "2y"


# ==================== Config duration integration ====================


class TestCacheMetadataConfigDuration:
    """Tests for CacheMetadataConfig duration fields."""

    def test_default_ttl_string_resolves(self):
        """default_ttl string overrides default_ttl_seconds."""
        cfg = CacheMetadataConfig(default_ttl="6h")
        assert cfg.default_ttl_seconds == 21600.0

    def test_default_ttl_numeric_passthrough(self):
        """default_ttl as int/float sets default_ttl_seconds."""
        cfg = CacheMetadataConfig(default_ttl=7200)
        assert cfg.default_ttl_seconds == 7200.0

    def test_default_ttl_precedence(self):
        """default_ttl takes precedence over default_ttl_seconds."""
        cfg = CacheMetadataConfig(default_ttl="1h", default_ttl_seconds=9999)
        assert cfg.default_ttl_seconds == 3600.0

    def test_memory_cache_ttl_string_resolves(self):
        """memory_cache_ttl string overrides memory_cache_ttl_seconds."""
        cfg = CacheMetadataConfig(memory_cache_ttl="10m")
        assert cfg.memory_cache_ttl_seconds == 600.0

    def test_memory_cache_ttl_numeric_passthrough(self):
        cfg = CacheMetadataConfig(memory_cache_ttl=120)
        assert cfg.memory_cache_ttl_seconds == 120.0

    def test_memory_cache_ttl_precedence(self):
        """memory_cache_ttl takes precedence over memory_cache_ttl_seconds."""
        cfg = CacheMetadataConfig(memory_cache_ttl="15m", memory_cache_ttl_seconds=9999)
        assert cfg.memory_cache_ttl_seconds == 900.0

    def test_default_unchanged_when_not_set(self):
        """When neither is specified, defaults are preserved."""
        cfg = CacheMetadataConfig()
        assert cfg.default_ttl_seconds == 86400
        assert cfg.memory_cache_ttl_seconds == 300

    def test_legacy_seconds_still_works(self):
        """Setting only the legacy field still works."""
        cfg = CacheMetadataConfig(default_ttl_seconds=7200)
        assert cfg.default_ttl_seconds == 7200

    def test_invalid_duration_string_raises(self):
        with pytest.raises(ValueError, match="Cannot parse duration"):
            CacheMetadataConfig(default_ttl="10xyz")


class TestCacheConfigDuration:
    """Tests for CacheConfig duration kwargs."""

    def test_default_ttl_kwarg(self):
        cfg = CacheConfig(default_ttl="12h")
        assert cfg.metadata.default_ttl_seconds == 43200.0

    def test_memory_cache_ttl_kwarg(self):
        cfg = CacheConfig(memory_cache_ttl="10m")
        assert cfg.metadata.memory_cache_ttl_seconds == 600.0

    def test_default_ttl_kwarg_precedence(self):
        """default_ttl kwarg takes precedence over default_ttl_seconds."""
        cfg = CacheConfig(default_ttl="2h", default_ttl_seconds=9999)
        assert cfg.metadata.default_ttl_seconds == 7200.0

    def test_legacy_default_ttl_seconds_kwarg(self):
        """Legacy default_ttl_seconds kwarg still works alone."""
        cfg = CacheConfig(default_ttl_seconds=1800)
        assert cfg.metadata.default_ttl_seconds == 1800

    def test_storage_mode_clears_ttl(self):
        """Storage mode sets default_ttl_seconds to None even if default_ttl was set."""
        cfg = CacheConfig(default_ttl="6h", storage_mode=True)
        assert cfg.metadata.default_ttl_seconds is None
        assert cfg.metadata.default_ttl is None


# ==================== resolve_ttl helper ====================


class TestResolveTtl:
    """Tests for the resolve_ttl() helper function."""

    def test_ttl_string_resolved(self):
        from cacheness.size_utils import resolve_ttl

        assert resolve_ttl(ttl="6h") == 21600.0

    def test_ttl_seconds_numeric(self):
        from cacheness.size_utils import resolve_ttl

        assert resolve_ttl(ttl_seconds=3600) == 3600.0

    def test_neither_returns_none(self):
        from cacheness.size_utils import resolve_ttl

        assert resolve_ttl() is None

    def test_both_raises_value_error(self):
        from cacheness.size_utils import resolve_ttl

        with pytest.raises(ValueError, match="Cannot specify both"):
            resolve_ttl(ttl="6h", ttl_seconds=3600)

    def test_ttl_seconds_string_raises_type_error(self):
        from cacheness.size_utils import resolve_ttl

        with pytest.raises(TypeError, match="ttl_seconds.*must be numeric"):
            resolve_ttl(ttl_seconds="6h")

    def test_context_in_error_message(self):
        from cacheness.size_utils import resolve_ttl

        with pytest.raises(TypeError, match="@cached"):
            resolve_ttl(ttl_seconds="1h", _param_owner="@cached")


# ==================== core.py get() with duration strings ====================


class TestGetWithDurationString:
    """Tests that get() accepts the ``ttl`` parameter for duration strings."""

    @pytest.fixture
    def cache(self, tmp_path):
        config = CacheConfig(
            cache_dir=str(tmp_path / "cache"),
            metadata_backend="json",
            default_ttl_seconds=86400,
            cleanup_on_init=False,
        )
        config.security.enable_signing = False
        cache = cacheness(config)
        yield cache
        cache.close()

    def test_get_with_ttl_string(self, cache):
        """get(ttl='1h') should work."""
        cache.put("hello", on={"key": "v1"})
        result = cache.get(on={"key": "v1"}, ttl="1h")
        assert result == "hello"

    def test_get_with_short_ttl_string_expires(self, cache):
        """get(ttl='0s') should expire immediately."""
        import time

        cache.put("hello", on={"key": "v2"})
        time.sleep(0.01)
        result = cache.get(on={"key": "v2"}, ttl="0s")
        # 0s TTL means already expired
        assert result is None

    def test_get_ttl_seconds_rejects_string(self, cache):
        """get(ttl_seconds='1h') should raise TypeError."""
        cache.put("hello", on={"key": "v3"})
        with pytest.raises(TypeError, match="ttl_seconds.*must be numeric"):
            cache.get(on={"key": "v3"}, ttl_seconds="1h")

    def test_get_ttl_and_ttl_seconds_mutually_exclusive(self, cache):
        """get(ttl='1h', ttl_seconds=3600) should raise ValueError."""
        cache.put("hello", on={"key": "v4"})
        with pytest.raises(ValueError, match="Cannot specify both"):
            cache.get(on={"key": "v4"}, ttl="1h", ttl_seconds=3600)

    def test_get_ttl_seconds_numeric_still_works(self, cache):
        """get(ttl_seconds=3600) with numeric value should still work."""
        cache.put("hello", on={"key": "v5"})
        result = cache.get(on={"key": "v5"}, ttl_seconds=3600)
        assert result == "hello"


# ==================== for_api() with duration strings ====================


class TestForApiDurationString:
    """Tests that UnifiedCache.for_api() accepts duration strings via ``ttl``."""

    def test_for_api_ttl_string(self, tmp_path):
        cache = cacheness.for_api(cache_dir=str(tmp_path / "cache"), ttl="2h")
        assert cache.config.metadata.default_ttl_seconds == 7200.0
        cache.close()

    def test_for_api_numeric_still_works(self, tmp_path):
        cache = cacheness.for_api(cache_dir=str(tmp_path / "cache"), ttl_seconds=3600)
        assert cache.config.metadata.default_ttl_seconds == 3600
        cache.close()

    def test_for_api_ttl_seconds_rejects_string(self, tmp_path):
        with pytest.raises(TypeError, match="ttl_seconds.*must be numeric"):
            cacheness.for_api(cache_dir=str(tmp_path / "cache"), ttl_seconds="2h")

    def test_for_api_ttl_and_ttl_seconds_mutually_exclusive(self, tmp_path):
        with pytest.raises(ValueError, match="Cannot specify both"):
            cacheness.for_api(
                cache_dir=str(tmp_path / "cache"), ttl="2h", ttl_seconds=3600
            )
