"""Tests for the put_bytes / get_bytes handler protocol (CACHE-728).

Validates that:
- CacheWriter.put_bytes and CacheReader.get_bytes raise NotImplementedError
  by default (backward-compatible ABC contract).
- BytesHandler, ObjectHandler, and ArrayHandler implement the zero-disk
  in-memory fast-path correctly.
- Round-trip serialization produces identical data.
- core.py _try_direct_inline and _read_inline_blob use the fast-path when
  available and fall back to the disk path when not.
"""

import pickle
import secrets
import tempfile
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

from cacheness.handlers import BytesHandler, ObjectHandler, ArrayHandler
from cacheness.core import UnifiedCache, CacheConfig
from cacheness.config import SecurityConfig
from cacheness.encryption import encrypt_blob
from cacheness.interfaces import CacheWriter, CacheReader, HandlerResult
from cacheness.storage import BlobStore


# ---------------------------------------------------------------------------
# ABC contract: default raises NotImplementedError
# ---------------------------------------------------------------------------


class _StubWriter(CacheWriter):
    """Minimal concrete CacheWriter (only put is abstract)."""

    def put(self, data, file_path, config):
        return HandlerResult(storage_format="stub", file_size=0, actual_path="")


class _StubReader(CacheReader):
    """Minimal concrete CacheReader (only get is abstract)."""

    def get(self, file_path, metadata):
        return None


class TestABCContract:
    """Default put_bytes / get_bytes should raise NotImplementedError."""

    def test_put_bytes_default_raises(self):
        writer = _StubWriter()
        with pytest.raises(NotImplementedError):
            writer.put_bytes(b"hello", config=None)

    def test_get_bytes_default_raises(self):
        reader = _StubReader()
        with pytest.raises(NotImplementedError):
            reader.get_bytes(b"hello", metadata={})


# ---------------------------------------------------------------------------
# BytesHandler
# ---------------------------------------------------------------------------


class TestBytesHandlerBytes:
    """BytesHandler put_bytes / get_bytes are trivial passthroughs."""

    def test_put_bytes_returns_raw(self):
        handler = BytesHandler()
        blob, result = handler.put_bytes(b"raw data", config=None)
        assert blob == b"raw data"
        assert result.storage_format == "raw_bytes"
        assert result.file_size == len(b"raw data")
        assert result.actual_path == ""

    def test_put_bytes_bytearray(self):
        handler = BytesHandler()
        blob, result = handler.put_bytes(bytearray(b"\x00\x01\x02"), config=None)
        assert blob == b"\x00\x01\x02"
        assert isinstance(blob, bytes)

    def test_put_bytes_memoryview(self):
        handler = BytesHandler()
        blob, result = handler.put_bytes(memoryview(b"mv"), config=None)
        assert blob == b"mv"

    def test_get_bytes_passthrough(self):
        handler = BytesHandler()
        assert handler.get_bytes(b"hello", metadata={}) == b"hello"

    def test_round_trip(self):
        handler = BytesHandler()
        original = b"round trip test \x00\xff"
        blob, _ = handler.put_bytes(original, config=None)
        recovered = handler.get_bytes(blob, metadata={})
        assert recovered == original


# ---------------------------------------------------------------------------
# ObjectHandler
# ---------------------------------------------------------------------------


class TestObjectHandlerBytes:
    """ObjectHandler put_bytes / get_bytes with pickle serialization."""

    @pytest.fixture()
    def config(self, tmp_path):
        return CacheConfig(cache_dir=tmp_path)

    def test_put_bytes_simple_object(self, config):
        handler = ObjectHandler()
        data = {"key": [1, 2, 3], "nested": {"a": True}}
        blob, result = handler.put_bytes(data, config)

        assert isinstance(blob, bytes)
        assert len(blob) > 0
        assert result.actual_path == ""
        assert result.serializer == "pickle"

    def test_get_bytes_simple_object(self, config):
        handler = ObjectHandler()
        data = {"key": [1, 2, 3], "nested": {"a": True}}
        blob, result = handler.put_bytes(data, config)
        recovered = handler.get_bytes(
            blob, dict(result._as_legacy_dict().get("metadata", {}))
        )
        assert recovered == data

    def test_round_trip_list(self, config):
        handler = ObjectHandler()
        data = [1, "two", 3.0, None, True]
        blob, result = handler.put_bytes(data, config)
        meta = {
            "storage_format": result.storage_format,
            "serializer": result.serializer,
        }
        recovered = handler.get_bytes(blob, meta)
        assert recovered == data

    def test_round_trip_large_object_compressed(self, config):
        """Objects exceeding compression threshold should be compressed."""
        handler = ObjectHandler()
        # Create object larger than default threshold (1024 bytes)
        data = list(range(5000))
        blob, result = handler.put_bytes(data, config)
        meta = {
            "storage_format": result.storage_format,
            "serializer": result.serializer,
        }
        recovered = handler.get_bytes(blob, meta)
        assert recovered == data

    def test_storage_format_reflects_compression(self, config):
        handler = ObjectHandler()
        # Small object — below compression threshold
        small_blob, small_result = handler.put_bytes(42, config)
        assert small_result.storage_format == "pickle"

        # Large object — above compression threshold
        big_data = list(range(5000))
        big_blob, big_result = handler.put_bytes(big_data, config)
        # May or may not be compressed depending on blosc availability
        assert big_result.storage_format in ("pickle", "compressed_pickle")


# ---------------------------------------------------------------------------
# ArrayHandler
# ---------------------------------------------------------------------------


class TestArrayHandlerBytes:
    """ArrayHandler put_bytes / get_bytes with blosc2 compression."""

    @pytest.fixture()
    def config(self, tmp_path):
        return CacheConfig(cache_dir=tmp_path)

    def test_put_bytes_1d_array(self, config):
        handler = ArrayHandler()
        data = np.arange(100, dtype=np.float64)
        try:
            blob, result = handler.put_bytes(data, config)
        except NotImplementedError:
            pytest.skip("blosc2 not available or not enabled")
        assert isinstance(blob, bytes)
        assert result.storage_format == "blosc2_array"
        assert result.actual_path == ""

    def test_round_trip_1d(self, config):
        handler = ArrayHandler()
        data = np.arange(100, dtype=np.float64)
        try:
            blob, result = handler.put_bytes(data, config)
        except NotImplementedError:
            pytest.skip("blosc2 not available")
        meta = {
            "storage_format": result.storage_format,
            "shape": result.extra.get("shape"),
            "dtype": result.extra.get("dtype"),
        }
        recovered = handler.get_bytes(blob, meta)
        np.testing.assert_array_equal(recovered, data)

    def test_round_trip_2d(self, config):
        handler = ArrayHandler()
        data = np.random.rand(10, 20).astype(np.float32)
        try:
            blob, result = handler.put_bytes(data, config)
        except NotImplementedError:
            pytest.skip("blosc2 not available")
        meta = {"storage_format": result.storage_format}
        recovered = handler.get_bytes(blob, meta)
        np.testing.assert_array_equal(recovered, data)

    def test_dict_of_arrays_raises(self, config):
        handler = ArrayHandler()
        data = {"a": np.array([1, 2]), "b": np.array([3, 4])}
        with pytest.raises(NotImplementedError):
            handler.put_bytes(data, config)

    def test_get_bytes_npz_format_raises(self, config):
        handler = ArrayHandler()
        with pytest.raises(NotImplementedError):
            handler.get_bytes(b"fake", metadata={"storage_format": "npz"})


# ---------------------------------------------------------------------------
# Integration: round-trip through UnifiedCache (inline path)
# ---------------------------------------------------------------------------


class TestInlineRoundTrip:
    """End-to-end: put → inline → get using the zero-disk path."""

    @pytest.fixture()
    def cache(self, tmp_path):
        return UnifiedCache(
            CacheConfig(
                cache_dir=tmp_path,
                max_inline_size=1_000_000,  # Large enough to inline everything
            )
        )

    def test_bytes_round_trip(self, cache):
        original = b"inline bytes test"
        cache.put(original, cache_key="bytes-rt")
        recovered = cache.get(cache_key="bytes-rt")
        assert recovered == original

    def test_object_round_trip(self, cache):
        original = {"msg": "hello", "nums": [1, 2, 3]}
        cache.put(original, cache_key="obj-rt")
        recovered = cache.get(cache_key="obj-rt")
        assert recovered == original

    def test_array_round_trip(self, cache):
        original = np.arange(50, dtype=np.float64)
        cache.put(original, cache_key="arr-rt")
        recovered = cache.get(cache_key="arr-rt")
        np.testing.assert_array_equal(recovered, original)

    def test_inline_entry_has_no_blob_file(self, cache, tmp_path):
        """When direct inline succeeds, no blob file should exist on disk."""
        cache.put(b"small", cache_key="no-file")
        # Check that no blob files were written in the cache directory
        # (exclude cache_signing_key.bin which is a metadata artifact)
        blob_files = [
            p for p in tmp_path.rglob("*.bin") if p.name != "cache_signing_key.bin"
        ]
        assert len(blob_files) == 0, f"Unexpected blob files: {blob_files}"

    def test_storage_mode_round_trip(self, tmp_path):
        """Storage-mode put/get should also use the direct inline path."""
        cache = UnifiedCache(
            CacheConfig(
                cache_dir=tmp_path,
                storage_mode=True,
                max_inline_size=1_000_000,
            )
        )
        cache.put(b"storage-bytes", cache_key="sm-rt")
        assert cache.get(cache_key="sm-rt") == b"storage-bytes"

    def test_update_data_round_trip(self, cache):
        """update_data should also use the direct inline path."""
        cache.put(b"v1", cache_key="upd-rt")
        cache.update_data(b"v2", cache_key="upd-rt")
        assert cache.get(cache_key="upd-rt") == b"v2"


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------


class TestFallback:
    """Handlers that don't implement put_bytes/get_bytes fall back to disk."""

    @pytest.fixture()
    def cache(self, tmp_path):
        return UnifiedCache(
            CacheConfig(
                cache_dir=tmp_path,
                max_inline_size=1_000_000,
            )
        )

    def test_dataframe_falls_back(self, cache):
        """DataFrame handlers don't have put_bytes — should still work."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        cache.put(df, cache_key="df-fallback")
        recovered = cache.get(cache_key="df-fallback")
        assert recovered is not None
        pd.testing.assert_frame_equal(recovered, df)


class TestEncryptedBlobReads:
    """Encrypted blob reads prefer get_bytes before plaintext temp fallback."""

    def _encrypted_store(self, tmp_path):
        (tmp_path / "cache_signing_key.bin").write_bytes(secrets.token_bytes(32))
        config = CacheConfig(
            cache_dir=tmp_path,
            security=SecurityConfig(
                enable_entry_signing=True,
                enable_content_encryption=True,
                encryption_key_file="cache_signing_key.bin",
                allow_unsigned_entries=True,
            ),
        )
        return BlobStore(
            cache_dir=tmp_path,
            backend="json",
            enable_signing=True,
            config=config,
        )

    def _write_encrypted_blob(self, store, path, plaintext):
        ciphertext, iv, algo = encrypt_blob(plaintext, store._encryption_key)
        path.write_bytes(ciphertext)
        return {
            "encryption_algorithm": algo.decode(),
            "encryption_iv": iv.hex(),
            "storage_format": "pickle",
            "serializer": "pickle",
        }

    def test_read_blob_tries_handler_get_bytes_before_temp_file(self, tmp_path):
        store = self._encrypted_store(tmp_path)
        blob_path = tmp_path / "encrypted.pkl"
        metadata = self._write_encrypted_blob(
            store, blob_path, pickle.dumps({"secure": "bytes"})
        )

        handler = MagicMock()
        handler.get_bytes.return_value = {"secure": "bytes"}
        handler.get.side_effect = AssertionError("temp-file fallback was used")
        store.handlers.get_handler_by_type = MagicMock(return_value=handler)

        assert store._read_blob(blob_path, "object", metadata) == {"secure": "bytes"}
        handler.get_bytes.assert_called_once()
        handler.get.assert_not_called()

    def test_read_blob_temp_fallback_uses_mkstemp_and_unlinks(
        self, tmp_path, monkeypatch
    ):
        store = self._encrypted_store(tmp_path)
        blob_path = tmp_path / "encrypted.pkl"
        metadata = self._write_encrypted_blob(store, blob_path, b"fallback-plaintext")
        temp_paths = []

        real_mkstemp = tempfile.mkstemp

        def tracking_mkstemp(*args, **kwargs):
            fd, name = real_mkstemp(*args, **kwargs)
            temp_paths.append(Path(name))
            return fd, name

        monkeypatch.setattr(tempfile, "mkstemp", tracking_mkstemp)

        class TempOnlyHandler:
            def get_bytes(self, blob, metadata):
                raise NotImplementedError

            def get(self, file_path, metadata):
                file_path = Path(file_path)
                assert file_path.parent == tmp_path
                assert file_path.read_bytes() == b"fallback-plaintext"
                return "fallback-ok"

        store.handlers.get_handler_by_type = MagicMock(return_value=TempOnlyHandler())

        assert store._read_blob(blob_path, "temp-only", metadata) == "fallback-ok"
        assert len(temp_paths) == 1
        assert not temp_paths[0].exists()
