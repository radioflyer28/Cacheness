"""
Tests for non-destructive get mode (delete_on_error=False).

Verifies that when delete_on_error=False, get() and get_with_metadata()
return None on deserialization/corruption errors but preserve cache entries
instead of auto-deleting them.
"""

import os
import tempfile

import pytest

from cacheness import CacheConfig, cacheness


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cache_destructive(tmp_path):
    """Cache with default delete_on_error=True (destructive on error)."""
    config = CacheConfig(cache_dir=str(tmp_path), delete_on_error=True)
    cache = cacheness(config)
    yield cache
    cache.close()


@pytest.fixture
def cache_non_destructive(tmp_path):
    """Cache with delete_on_error=False (preserves entries on error)."""
    config = CacheConfig(cache_dir=str(tmp_path), delete_on_error=False)
    cache = cacheness(config)
    yield cache
    cache.close()


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestDeleteOnErrorConfig:
    """Test delete_on_error configuration option."""

    def test_default_is_true(self):
        """delete_on_error defaults to True for backwards compatibility."""
        config = CacheConfig()
        assert config.delete_on_error is True
        assert config.metadata.delete_on_error is True

    def test_can_set_false_via_flat_kwarg(self):
        """delete_on_error can be set via flat CacheConfig kwarg."""
        config = CacheConfig(delete_on_error=False)
        assert config.delete_on_error is False
        assert config.metadata.delete_on_error is False

    def test_can_set_via_metadata_config(self):
        """delete_on_error can be set via CacheMetadataConfig."""
        from cacheness.config import CacheMetadataConfig

        meta = CacheMetadataConfig(delete_on_error=False)
        config = CacheConfig(metadata=meta)
        assert config.metadata.delete_on_error is False

    def test_flat_kwarg_overrides_metadata_config(self):
        """Flat kwarg takes precedence over sub-config."""
        from cacheness.config import CacheMetadataConfig

        meta = CacheMetadataConfig(delete_on_error=True)
        config = CacheConfig(metadata=meta, delete_on_error=False)
        assert config.metadata.delete_on_error is False


# ---------------------------------------------------------------------------
# Deserialization error tests
# ---------------------------------------------------------------------------


class TestDeserializationErrorHandling:
    """Test get() behavior when deserialization fails."""

    def _corrupt_blob(self, cache, cache_key):
        """Corrupt the blob file for a cache entry so deserialization fails."""
        file_path = cache._get_cache_file_path(cache_key)
        # Find the actual blob file (may have extension)
        parent = os.path.dirname(file_path)
        base = os.path.basename(file_path)
        for f in os.listdir(parent):
            if f.startswith(base) or f == base:
                full = os.path.join(parent, f)
                if os.path.isfile(full):
                    with open(full, "wb") as fh:
                        fh.write(b"CORRUPTED_GARBAGE_DATA_12345")
                    return full
        # Fallback: write directly to the base path
        with open(file_path, "wb") as fh:
            fh.write(b"CORRUPTED_GARBAGE_DATA_12345")
        return file_path

    def test_destructive_deletes_on_deserialization_error(self, cache_destructive):
        """With delete_on_error=True, get() deletes the entry on error."""
        cache = cache_destructive
        cache.put(on={"key": "test_val"}, data="hello world")
        cache_key = cache._create_cache_key({"key": "test_val"})

        # Verify data is there
        assert cache.get(on={"key": "test_val"}) == "hello world"

        # Corrupt the blob
        self._corrupt_blob(cache, cache_key)

        # get() should return None AND delete the entry
        result = cache.get(on={"key": "test_val"})
        assert result is None

        # Entry metadata should be gone
        meta = cache.get_metadata(on={"key": "test_val"})
        assert meta is None

    def test_non_destructive_preserves_on_deserialization_error(
        self, cache_non_destructive
    ):
        """With delete_on_error=False, get() returns None but keeps the entry."""
        cache = cache_non_destructive
        cache.put(on={"key": "test_val"}, data="hello world")
        cache_key = cache._create_cache_key({"key": "test_val"})

        # Verify data is there
        assert cache.get(on={"key": "test_val"}) == "hello world"

        # Corrupt the blob
        self._corrupt_blob(cache, cache_key)

        # get() should return None but NOT delete the entry
        result = cache.get(on={"key": "test_val"})
        assert result is None

        # Entry metadata should still exist
        meta = cache.get_metadata(on={"key": "test_val"})
        assert meta is not None

    def test_non_destructive_get_with_metadata(self, cache_non_destructive):
        """get_with_metadata() also respects delete_on_error=False."""
        cache = cache_non_destructive
        cache.put(on={"key": "gwm"}, data=[1, 2, 3])
        cache_key = cache._create_cache_key({"key": "gwm"})

        self._corrupt_blob(cache, cache_key)

        # get_with_metadata returns None on error
        result = cache.get_with_metadata(on={"key": "gwm"})
        assert result is None

        # But entry metadata is preserved
        meta = cache.get_metadata(on={"key": "gwm"})
        assert meta is not None

    def test_non_destructive_allows_re_put(self, cache_non_destructive):
        """After a non-destructive failure, the entry can be overwritten."""
        cache = cache_non_destructive
        cache.put(on={"key": "reput"}, data="original")
        cache_key = cache._create_cache_key({"key": "reput"})

        self._corrupt_blob(cache, cache_key)

        # get() fails but preserves
        assert cache.get(on={"key": "reput"}) is None

        # Re-put should work
        cache.put(on={"key": "reput"}, data="fixed")
        assert cache.get(on={"key": "reput"}) == "fixed"


# ---------------------------------------------------------------------------
# Hash integrity tests
# ---------------------------------------------------------------------------


class TestHashIntegrityWithDeleteOnError:
    """Test hash mismatch behavior with delete_on_error flag."""

    def _tamper_blob(self, cache, cache_key):
        """Modify blob contents so hash no longer matches (but file is still parseable)."""
        file_path = cache._get_cache_file_path(cache_key)
        parent = os.path.dirname(file_path)
        base = os.path.basename(file_path)
        for f in os.listdir(parent):
            if f.startswith(base) or f == base:
                full = os.path.join(parent, f)
                if os.path.isfile(full):
                    # Append a byte to change hash without destroying format
                    with open(full, "ab") as fh:
                        fh.write(b"\x00")
                    return full
        return None

    def test_hash_mismatch_destructive(self, tmp_path):
        """With delete_on_error=True, hash mismatch deletes the entry."""
        config = CacheConfig(
            cache_dir=str(tmp_path),
            verify_cache_integrity=True,
            delete_on_error=True,
        )
        cache = cacheness(config)

        cache.put(on={"key": "hash_d"}, data="test data")
        cache_key = cache._create_cache_key({"key": "hash_d"})

        self._tamper_blob(cache, cache_key)

        result = cache.get(on={"key": "hash_d"})
        assert result is None

        # Entry should be gone
        meta = cache.get_metadata(on={"key": "hash_d"})
        assert meta is None

        cache.close()

    def test_hash_mismatch_non_destructive(self, tmp_path):
        """With delete_on_error=False, hash mismatch returns None but preserves entry."""
        config = CacheConfig(
            cache_dir=str(tmp_path),
            verify_cache_integrity=True,
            delete_on_error=False,
        )
        cache = cacheness(config)

        cache.put(on={"key": "hash_nd"}, data="test data")
        cache_key = cache._create_cache_key({"key": "hash_nd"})

        self._tamper_blob(cache, cache_key)

        result = cache.get(on={"key": "hash_nd"})
        assert result is None

        # Entry metadata should still exist
        meta = cache.get_metadata(on={"key": "hash_nd"})
        assert meta is not None

        cache.close()


# ---------------------------------------------------------------------------
# FileNotFoundError and OSError are not affected
# ---------------------------------------------------------------------------


class TestUnaffectedErrorPaths:
    """FileNotFoundError and OSError behavior is independent of delete_on_error."""

    def test_file_not_found_still_cleans_metadata(self, tmp_path):
        """FileNotFoundError always cleans up metadata (blob is already gone)."""
        # Disable integrity verification so the FileNotFoundError path is reached
        # (otherwise hash check intercepts the missing file first)
        config = CacheConfig(
            cache_dir=str(tmp_path),
            delete_on_error=False,
            verify_cache_integrity=False,
        )
        cache = cacheness(config)
        cache.put(on={"key": "fnf"}, data="data")
        cache_key = cache._create_cache_key({"key": "fnf"})

        # Delete the blob file directly
        file_path = cache._get_cache_file_path(cache_key)
        parent = os.path.dirname(file_path)
        for f in os.listdir(parent):
            full = os.path.join(parent, f)
            if os.path.isfile(full):
                os.remove(full)

        # get() returns None and cleans metadata (even with delete_on_error=False)
        result = cache.get(on={"key": "fnf"})
        assert result is None

        meta = cache.get_metadata(on={"key": "fnf"})
        assert meta is None

        cache.close()


# ---------------------------------------------------------------------------
# Logging tests
# ---------------------------------------------------------------------------


class TestDeleteOnErrorLogging:
    """Verify appropriate warning messages are logged."""

    def _corrupt_blob(self, cache, cache_key):
        """Corrupt the blob file for a cache entry."""
        file_path = cache._get_cache_file_path(cache_key)
        parent = os.path.dirname(file_path)
        base = os.path.basename(file_path)
        for f in os.listdir(parent):
            if f.startswith(base) or f == base:
                full = os.path.join(parent, f)
                if os.path.isfile(full):
                    with open(full, "wb") as fh:
                        fh.write(b"CORRUPTED_GARBAGE_DATA_12345")
                    return full
        with open(file_path, "wb") as fh:
            fh.write(b"CORRUPTED_GARBAGE_DATA_12345")
        return file_path

    def test_non_destructive_logs_retained_message(self, cache_non_destructive, caplog):
        """Non-destructive mode logs 'retained due to delete_on_error=False'."""
        cache = cache_non_destructive
        cache.put(on={"key": "logtest"}, data="test")
        cache_key = cache._create_cache_key({"key": "logtest"})
        self._corrupt_blob(cache, cache_key)

        import logging

        with caplog.at_level(logging.WARNING):
            cache.get(on={"key": "logtest"})

        assert any(
            "delete_on_error=False" in record.message for record in caplog.records
        ), (
            f"Expected 'delete_on_error=False' in log messages, got: {[r.message for r in caplog.records]}"
        )
