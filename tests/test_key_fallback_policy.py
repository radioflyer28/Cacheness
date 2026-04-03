"""Tests for key_fallback_policy configuration and 3-mode behavior."""

import logging
import warnings

import pytest

from cacheness.config import SecurityConfig
from cacheness.error_handling import CacheSecurityError
from cacheness.security import CacheEntrySigner, create_cache_signer


class TestKeyFallbackPolicyModes:
    """Test each key_fallback_policy mode with write failures."""

    def test_raise_mode_raises_on_write_failure(self, tmp_path):
        """key_fallback_policy='raise' raises CacheSecurityError on key write failure."""
        blocker = tmp_path / "blocker"
        blocker.write_text("block")
        impossible_key = blocker / "subdir" / "key.bin"
        with pytest.raises(CacheSecurityError, match="Failed to persist"):
            CacheEntrySigner(key_file_path=impossible_key, key_fallback_policy="raise")

    def test_warn_mode_logs_warning_on_write_failure(self, tmp_path, caplog):
        """key_fallback_policy='warn' logs WARNING and uses in-memory key."""
        blocker = tmp_path / "blocker"
        blocker.write_text("block")
        impossible_key = blocker / "subdir" / "key.bin"
        with caplog.at_level(logging.WARNING, logger="cacheness.security"):
            signer = CacheEntrySigner(
                key_file_path=impossible_key, key_fallback_policy="warn"
            )
        assert signer.secret_key is not None
        assert len(signer.secret_key) == 32
        assert "Failed to persist signing key" in caplog.text

    def test_fallback_mode_silent_on_write_failure(self, tmp_path, caplog):
        """key_fallback_policy='fallback' silently uses in-memory key."""
        blocker = tmp_path / "blocker"
        blocker.write_text("block")
        impossible_key = blocker / "subdir" / "key.bin"
        with caplog.at_level(logging.DEBUG, logger="cacheness.security"):
            signer = CacheEntrySigner(
                key_file_path=impossible_key, key_fallback_policy="fallback"
            )
        assert signer.secret_key is not None
        assert len(signer.secret_key) == 32
        assert "Failed to persist signing key" not in caplog.text

    def test_raise_mode_on_corrupt_key(self, tmp_path):
        """key_fallback_policy='raise' raises CacheSecurityError for corrupt key file."""
        key_path = tmp_path / "key.bin"
        key_path.write_bytes(b"tooshort")  # 8 bytes, not 32
        with pytest.raises(CacheSecurityError, match="Invalid signing key length"):
            CacheEntrySigner(key_file_path=key_path, key_fallback_policy="raise")

    def test_warn_mode_on_corrupt_key(self, tmp_path, caplog):
        """key_fallback_policy='warn' logs warning for corrupt key and regenerates."""
        key_path = tmp_path / "key.bin"
        key_path.write_bytes(b"tooshort")
        with caplog.at_level(logging.WARNING, logger="cacheness.security"):
            signer = CacheEntrySigner(
                key_file_path=key_path, key_fallback_policy="warn"
            )
        assert signer.secret_key is not None
        assert len(signer.secret_key) == 32
        assert "Invalid key length" in caplog.text

    def test_fallback_mode_on_corrupt_key(self, tmp_path, caplog):
        """key_fallback_policy='fallback' silently regenerates corrupt key."""
        key_path = tmp_path / "key.bin"
        key_path.write_bytes(b"tooshort")
        with caplog.at_level(logging.WARNING, logger="cacheness.security"):
            signer = CacheEntrySigner(
                key_file_path=key_path, key_fallback_policy="fallback"
            )
        assert signer.secret_key is not None
        assert len(signer.secret_key) == 32
        assert "Invalid key length" not in caplog.text


class TestKeyFallbackPolicyConfig:
    """Test SecurityConfig field and validation."""

    def test_default_policy_is_warn(self):
        """Default key_fallback_policy is 'warn'."""
        cfg = SecurityConfig()
        assert cfg.key_fallback_policy == "warn"

    def test_invalid_policy_raises_valueerror(self):
        """Invalid key_fallback_policy value raises ValueError."""
        with pytest.raises(ValueError, match="key_fallback_policy must be one of"):
            SecurityConfig(key_fallback_policy="invalid")


class TestDeprecationShim:
    """Test backward compatibility for raise_on_key_fallback."""

    def test_raise_on_key_fallback_true_maps_to_raise(self):
        """raise_on_key_fallback=True maps to key_fallback_policy='raise' with deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = SecurityConfig(raise_on_key_fallback=True)
        assert cfg.key_fallback_policy == "raise"
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message).lower()

    def test_raise_on_key_fallback_false_no_shim(self):
        """raise_on_key_fallback=False (default) does NOT trigger deprecation."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = SecurityConfig(raise_on_key_fallback=False)
        assert cfg.key_fallback_policy == "warn"  # new default, not "fallback"
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) == 0

    def test_new_field_takes_precedence(self):
        """key_fallback_policy takes precedence when both are set."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cfg = SecurityConfig(
                raise_on_key_fallback=True, key_fallback_policy="fallback"
            )
        # new field was already "fallback" (not default "warn")
        # so the shim condition (key_fallback_policy == "warn") is False
        # and the new field wins
        assert cfg.key_fallback_policy == "fallback"


class TestFactoryIntegration:
    """Test create_cache_signer and BlobStore passthrough."""

    def test_create_cache_signer_passes_policy(self, tmp_path):
        """create_cache_signer passes key_fallback_policy to CacheEntrySigner."""
        signer = create_cache_signer(cache_dir=tmp_path, key_fallback_policy="warn")
        assert signer.key_fallback_policy == "warn"
        assert signer.secret_key is not None

    def test_create_cache_signer_raise_mode(self, tmp_path):
        """create_cache_signer with raise policy propagates errors."""
        blocker = tmp_path / "blocker"
        blocker.write_text("block")
        with pytest.raises(CacheSecurityError):
            create_cache_signer(
                cache_dir=blocker / "subdir",
                key_fallback_policy="raise",
            )
