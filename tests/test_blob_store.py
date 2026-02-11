"""
Tests for BlobStore enhancements (CACHE-xf3).

Tests:
- xxhash-based content-addressable keys
- File hash computation on put()
- Entry signing and verification
- verify_integrity() method
- Config-aware HandlerRegistry
- Thread safety
"""

import os
import threading
from pathlib import Path

import pytest
import xxhash

from cacheness.storage import BlobStore


@pytest.fixture
def blob_dir(tmp_path):
    """Return a temporary directory for blob storage."""
    return tmp_path / "blobs"


@pytest.fixture
def store(blob_dir):
    """Create a basic BlobStore for testing."""
    return BlobStore(cache_dir=blob_dir, backend="json")


@pytest.fixture
def sqlite_store(blob_dir):
    """Create a SQLite-backed BlobStore for testing."""
    return BlobStore(cache_dir=blob_dir, backend="sqlite")


@pytest.fixture
def signed_store(blob_dir):
    """Create a BlobStore with signing enabled."""
    return BlobStore(
        cache_dir=blob_dir,
        backend="json",
        enable_signing=True,
        use_in_memory_key=True,
    )


# ── Basic put/get (regression) ─────────────────────────────────────


class TestBlobStoreBasic:
    """Regression tests — basic put/get should still work."""

    def test_put_get_string(self, store):
        key = store.put("hello world", key="greeting")
        result = store.get(key)
        assert result == "hello world"

    def test_put_get_dict(self, store):
        data = {"a": 1, "b": [2, 3]}
        key = store.put(data, key="mydict")
        result = store.get(key)
        assert result == data

    def test_put_get_auto_key(self, store):
        key = store.put(42)
        assert len(key) == 16  # uuid hex[:16]
        assert store.get(key) == 42

    def test_delete(self, store):
        key = store.put("delete me", key="del")
        assert store.exists(key)
        assert store.delete(key)
        assert not store.exists(key)

    def test_list(self, store):
        store.put("a", key="prefix-one")
        store.put("b", key="prefix-two")
        store.put("c", key="other")
        keys = store.list(prefix="prefix-")
        assert sorted(keys) == ["prefix-one", "prefix-two"]

    def test_clear(self, store):
        store.put("x", key="k1")
        store.put("y", key="k2")
        removed = store.clear()
        assert removed >= 2
        assert store.list() == []

    def test_get_nonexistent(self, store):
        assert store.get("nope") is None

    def test_update_metadata(self, store):
        key = store.put("data", key="meta-test")
        assert store.update_metadata(key, {"tag": "v1"})
        meta = store.get_metadata(key)
        nested = meta.get("metadata", {})
        assert nested.get("tag") == "v1"

    def test_context_manager(self, blob_dir):
        with BlobStore(cache_dir=blob_dir, backend="json") as s:
            s.put("ctx", key="ctx-key")
        # After close, directory still exists
        assert blob_dir.exists()


# ── xxhash content-addressable keys ────────────────────────────────


class TestXxhashKeys:
    """Content-addressable keys now use xxhash instead of SHA-256."""

    def test_content_addressable_uses_xxhash(self, blob_dir):
        store = BlobStore(cache_dir=blob_dir, backend="json", content_addressable=True)
        key = store.put("deterministic")
        # Same data → same key
        key2 = store.put("deterministic")
        assert key == key2

    def test_content_addressable_key_is_xxhash(self, blob_dir):
        """Verify the key is actually an xxhash, not SHA-256."""
        import pickle

        data = "test-data"
        store = BlobStore(cache_dir=blob_dir, backend="json", content_addressable=True)
        key = store.put(data)

        # Manually compute expected xxhash
        serialized = pickle.dumps(data)
        expected = xxhash.xxh3_64(serialized).hexdigest()[:16]
        assert key == expected

    def test_different_data_different_keys(self, blob_dir):
        store = BlobStore(cache_dir=blob_dir, backend="json", content_addressable=True)
        k1 = store.put("aaa")
        k2 = store.put("bbb")
        assert k1 != k2


# ── File hash computation ──────────────────────────────────────────


class TestFileHash:
    """File hash (xxhash) should be computed on put and stored in metadata."""

    def test_file_hash_stored_on_put(self, store):
        key = store.put("hash-me", key="hashed")
        meta = store.get_metadata(key)
        assert meta is not None
        # file_hash is in nested metadata (JsonBackend stores custom fields there)
        nested = meta.get("metadata", {})
        file_hash = meta.get("file_hash") or nested.get("file_hash")
        assert file_hash is not None
        # xxhash hex digest is 16 chars
        assert len(file_hash) == 16

    def test_file_hash_matches_actual_file(self, store):
        key = store.put("verify-hash", key="vhash")
        meta = store.get_metadata(key)
        nested = meta.get("metadata", {})
        file_hash = meta.get("file_hash") or nested.get("file_hash")

        # Calculate hash of the actual file
        actual_path = Path(meta.get("actual_path") or nested.get("actual_path"))
        hasher = xxhash.xxh3_64()
        with open(actual_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        assert hasher.hexdigest() == file_hash

    def test_file_hash_none_for_missing_file(self, store):
        """_calculate_file_hash returns None for nonexistent files."""
        result = store._calculate_file_hash(Path("/nonexistent/path.pkl"))
        assert result is None


# ── Entry signing ──────────────────────────────────────────────────


class TestEntrySigning:
    """HMAC-SHA256 entry signing on put, verification on get."""

    def test_signed_entry_has_signature(self, signed_store):
        key = signed_store.put("signed-data", key="sig-test")
        meta = signed_store.get_metadata(key)
        nested = meta.get("metadata", {})
        sig = meta.get("entry_signature") or nested.get("entry_signature")
        assert sig is not None
        assert len(sig) == 64  # SHA-256 hex = 64 chars

    def test_signed_entry_roundtrip(self, signed_store):
        """Signed entry can be retrieved successfully."""
        key = signed_store.put({"x": 1}, key="sig-rt")
        result = signed_store.get(key)
        assert result == {"x": 1}

    def test_tampered_entry_returns_none(self, signed_store):
        """Tampering with metadata causes get() to return None."""
        key = signed_store.put("tamper-test", key="tamper")
        # Tamper with the metadata
        entry = signed_store.backend.get_entry(key)
        entry["file_size"] = 999999  # Tamper
        signed_store.backend.put_entry(key, entry)
        # get() should detect tampered signature and return None
        result = signed_store.get(key)
        assert result is None

    def test_unsigned_store_no_signature(self, store):
        """Non-signed store should not add signatures."""
        key = store.put("no-sig", key="nosig")
        meta = store.get_metadata(key)
        nested = meta.get("metadata", {})
        assert meta.get("entry_signature") is None
        assert nested.get("entry_signature") is None
        assert meta.get("entry_signature") is None
        assert nested.get("entry_signature") is None

    def test_signer_attribute(self, signed_store, store):
        assert signed_store.signer is not None
        assert store.signer is None


# ── verify_integrity ───────────────────────────────────────────────


class TestVerifyIntegrity:
    """Integrity verification: orphans, dangling, size/hash mismatches."""

    def test_clean_store_passes(self, store):
        store.put("a", key="k1")
        store.put("b", key="k2")
        report = store.verify_integrity()
        assert report["orphaned_blobs"] == []
        assert report["dangling_entries"] == []
        assert report["size_mismatches"] == []

    def test_detects_dangling_metadata(self, store):
        key = store.put("data", key="dangling")
        # Delete the blob file but keep metadata
        meta = store.get_metadata(key)
        nested = meta.get("metadata", {})
        actual_path = meta.get("actual_path") or nested.get("actual_path")
        os.remove(actual_path)

        report = store.verify_integrity()
        assert len(report["dangling_entries"]) == 1
        assert report["dangling_entries"][0]["cache_key"] == key

    def test_detects_orphaned_blobs(self, store):
        key = store.put("orphan-data", key="orphan")
        # Remove metadata but keep the file
        store.backend.remove_entry(key)

        report = store.verify_integrity()
        assert len(report["orphaned_blobs"]) >= 1

    def test_detects_size_mismatch(self, store):
        key = store.put("size-check", key="sizecheck")
        # Tamper with file_size in metadata
        entry = store.backend.get_entry(key)
        entry["file_size"] = 1  # Wrong size
        store.backend.put_entry(key, entry)

        report = store.verify_integrity()
        assert len(report["size_mismatches"]) == 1

    def test_detects_hash_mismatch(self, store):
        key = store.put("hash-check", key="hashcheck")
        # Tamper with file_hash in nested metadata (where JsonBackend stores it)
        entry = store.backend.get_entry(key)
        nested = entry.get("metadata", {})
        nested["file_hash"] = "badhash000000000"
        entry["metadata"] = nested
        store.backend.put_entry(key, entry)

        report = store.verify_integrity(verify_hashes=True)
        assert "hash_mismatches" in report
        assert len(report["hash_mismatches"]) == 1

    def test_repair_removes_orphans(self, store):
        key = store.put("repair-me", key="repair-orphan")
        meta = store.get_metadata(key)
        nested = meta.get("metadata", {})
        actual_path = meta.get("actual_path") or nested.get("actual_path")
        # Remove metadata → blob becomes orphan
        store.backend.remove_entry(key)
        assert os.path.exists(actual_path)

        report = store.verify_integrity(repair=True)
        assert report["repaired"]["orphans_deleted"] >= 1
        assert not os.path.exists(actual_path)

    def test_repair_removes_dangling(self, store):
        key = store.put("repair-dangling", key="repair-dang")
        meta = store.get_metadata(key)
        nested = meta.get("metadata", {})
        actual_path = meta.get("actual_path") or nested.get("actual_path")
        # Remove blob file → metadata becomes dangling
        os.remove(actual_path)

        report = store.verify_integrity(repair=True)
        assert report["repaired"]["dangling_removed"] == 1
        # Metadata should be removed
        assert store.get_metadata(key) is None

    def test_empty_store_passes(self, store):
        report = store.verify_integrity()
        assert report["orphaned_blobs"] == []
        assert report["dangling_entries"] == []
        assert report["size_mismatches"] == []


# ── Config-aware HandlerRegistry ───────────────────────────────────


class TestConfigAwareHandlers:
    """HandlerRegistry should receive config for proper handler configuration."""

    def test_handler_registry_has_config(self, store):
        """HandlerRegistry should be initialized with config."""
        assert store.handlers.config is not None

    def test_custom_config_passed_through(self, blob_dir):
        """Custom CacheConfig should propagate to handler registry."""
        from cacheness.config import CacheConfig, CompressionConfig

        cfg = CacheConfig(
            cache_dir=blob_dir,
            compression=CompressionConfig(pickle_compression_codec="zstd"),
        )
        s = BlobStore(cache_dir=blob_dir, backend="json", config=cfg)
        assert s.config is cfg
        assert s.handlers.config is cfg


# ── Thread safety ──────────────────────────────────────────────────


class TestThreadSafety:
    """BlobStore should be safe for concurrent access."""

    def test_concurrent_puts(self, store):
        """Multiple threads putting data should not raise."""
        errors = []

        def worker(i):
            try:
                store.put(f"data-{i}", key=f"thread-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        keys = store.list()
        assert len(keys) == 10

    def test_concurrent_gets(self, store):
        """Multiple threads getting data should not raise."""
        keys = [store.put(f"data-{i}", key=f"get-{i}") for i in range(10)]
        errors = []
        results = [None] * 10

        def worker(i):
            try:
                results[i] = store.get(keys[i])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        for i in range(10):
            assert results[i] == f"data-{i}"

    def test_has_lock(self, store):
        """BlobStore should have an RLock."""
        assert hasattr(store, "_lock")
        assert isinstance(store._lock, type(threading.RLock()))


# ── SQLite backend ─────────────────────────────────────────────────


class TestSqliteBackend:
    """BlobStore with SQLite backend."""

    def test_put_get_sqlite(self, sqlite_store):
        key = sqlite_store.put("sqlite-data", key="sqltest")
        result = sqlite_store.get(key)
        assert result == "sqlite-data"

    def test_file_hash_sqlite(self, sqlite_store):
        key = sqlite_store.put("hash-sqlite", key="sqlhash")
        meta = sqlite_store.get_metadata(key)
        assert meta is not None
        # Check file_hash at top level or in nested metadata
        nested = meta.get("metadata", {})
        file_hash = meta.get("file_hash") or nested.get("file_hash")
        assert file_hash is not None

    def test_verify_integrity_sqlite(self, sqlite_store):
        sqlite_store.put("a", key="si1")
        sqlite_store.put("b", key="si2")
        report = sqlite_store.verify_integrity()
        assert report["orphaned_blobs"] == []
        assert report["dangling_entries"] == []
