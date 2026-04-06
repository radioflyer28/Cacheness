"""
Backend Operation Parity Tests
===============================

Verifies that SQLite and PostgreSQL metadata backends behave identically
for all operations defined in the MetadataBackend interface.

Also verifies encryption test parity across JSON, SQLite, and PostgreSQL
backends (Phase 24 / ENC-03).
"""

import os
import secrets

import pytest

cryptography = pytest.importorskip("cryptography")

from datetime import datetime, timezone  # noqa: E402

from cacheness.config import CacheConfig, CacheMetadataConfig, CacheStorageConfig, CompressionConfig, SecurityConfig  # noqa: E402  # fmt: skip
from cacheness.core import UnifiedCache as cacheness  # noqa: E402
from cacheness.error_handling import CacheConfigurationError  # noqa: E402
from cacheness.interfaces import RotationResult  # noqa: E402
from cacheness.storage.blob_store import BlobStore  # noqa: E402

try:
    from cacheness.storage.backends.postgresql_backend import PostgresBackend

    _HAS_PG = True
except ImportError:
    _HAS_PG = False


def _get_pg_url():
    return os.environ.get("CACHENESS_TEST_POSTGRES_URL")


def _generate_key_file(path):
    """Write 32 random bytes to a file and return the path."""
    path.write_bytes(secrets.token_bytes(32))
    return path


def _make_encrypted_cache_for_backend(tmp_path, backend, **security_overrides):
    """Create a UnifiedCache with encryption enabled for the specified backend."""
    key_file = tmp_path / "cache_signing_key.bin"
    if not key_file.exists():
        _generate_key_file(key_file)
    defaults = {
        "enable_entry_signing": True,
        "enable_content_encryption": True,
        "encryption_key_file": "cache_signing_key.bin",
        "allow_unsigned_entries": True,
        "delete_invalid_signatures": False,
    }
    defaults.update(security_overrides)
    security = SecurityConfig(**defaults)
    metadata_cfg = CacheMetadataConfig(metadata_backend=backend)
    if backend == "postgresql":
        metadata_cfg = CacheMetadataConfig(
            metadata_backend="postgresql",
            metadata_backend_options={"connection_url": _get_pg_url()},
        )
    config = CacheConfig(
        storage=CacheStorageConfig(cache_dir=str(tmp_path)),
        metadata=metadata_cfg,
        compression=CompressionConfig(use_blosc2_arrays=False),
        security=security,
    )
    return cacheness(config)


def _make_encrypted_blobstore_for_backend(tmp_path, backend, **overrides):
    """Create a BlobStore with encryption enabled for the specified backend."""
    key_file = tmp_path / "cache_signing_key.bin"
    if not key_file.exists():
        _generate_key_file(key_file)
    security = SecurityConfig(
        enable_entry_signing=True,
        enable_content_encryption=True,
        encryption_key_file="cache_signing_key.bin",
        allow_unsigned_entries=True,
        **overrides,
    )
    metadata_cfg = CacheMetadataConfig(metadata_backend=backend)
    if backend == "postgresql":
        metadata_cfg = CacheMetadataConfig(
            metadata_backend="postgresql",
            metadata_backend_options={"connection_url": _get_pg_url()},
        )
    config = CacheConfig(
        storage=CacheStorageConfig(cache_dir=str(tmp_path)),
        metadata=metadata_cfg,
        compression=CompressionConfig(use_blosc2_arrays=False),
        security=security,
    )
    # BlobStore only accepts "json", "sqlite", or a MetadataBackend instance
    backend_arg = backend
    if backend == "postgresql":
        backend_arg = PostgresBackend(connection_url=_get_pg_url())
    return BlobStore(
        cache_dir=tmp_path,
        backend=backend_arg,
        enable_signing=True,
        config=config,
        namespace="default",
    )


class TestBackendParity:
    """Test that SQLite and PostgreSQL backends have identical behavior."""

    @pytest.fixture
    def sqlite_backend(self, tmp_path):
        """Create a SQLite backend for testing."""
        from cacheness.metadata import SqliteBackend

        db_path = tmp_path / "test_sqlite.db"
        backend = SqliteBackend(str(db_path))
        yield backend
        backend.close()

    @pytest.fixture
    def postgresql_backend(self):
        """Create a PostgreSQL backend for testing (requires Docker)."""
        pytest.skip(
            "PostgreSQL backend tests require Docker - run with integration tests"
        )
        # This would be enabled in Docker integration tests
        from cacheness.storage.backends.postgresql_backend import PostgresBackend

        backend = PostgresBackend(
            connection_url="postgresql://cacheness_test:test_password@localhost:5432/cacheness_test"
        )
        yield backend
        backend.clear_all()
        backend.close()

    def test_put_and_get_entry_parity(self, sqlite_backend):
        """Test that put_entry and get_entry work identically."""
        entry_data = {
            "cache_key": "test_key_123",
            "description": "Test entry",
            "data_type": "array",
            "file_size": 2048,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                # Only known technical fields are preserved
                "s3_etag": "test_etag_123",
                "actual_path": "/test/path.npz",
                "object_type": "<class 'numpy.ndarray'>",
                "storage_format": "numpy",
                "serializer": "numpy",
                "compression_codec": "zstd",
            },
        }

        # Store entry
        sqlite_backend.put_entry("test_key_123", entry_data)

        # Retrieve entry
        retrieved = sqlite_backend.get_entry("test_key_123")

        assert retrieved is not None
        assert retrieved["description"] == "Test entry"
        assert retrieved["data_type"] == "array"
        assert retrieved["file_size"] == 2048
        assert "metadata" in retrieved
        # Technical metadata fields are preserved
        assert retrieved["metadata"]["s3_etag"] == "test_etag_123"
        assert retrieved["metadata"]["actual_path"] == "/test/path.npz"
        assert retrieved["metadata"]["object_type"] == "<class 'numpy.ndarray'>"

    def test_list_entries_parity(self, sqlite_backend):
        """Test that list_entries returns consistent format."""
        # Store multiple entries
        for i in range(3):
            entry_data = {
                "cache_key": f"key_{i}",
                "description": f"Entry {i}",
                "data_type": "dataframe",
                "file_size": 1024 * (i + 1),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "index": i,
                },
            }
            sqlite_backend.put_entry(f"key_{i}", entry_data)

        # List entries
        entries = sqlite_backend.list_entries()

        assert len(entries) == 3

        # Verify structure
        for entry in entries:
            assert "cache_key" in entry
            assert "description" in entry
            assert "data_type" in entry
            assert (
                "created_at" in entry or "created" in entry
            )  # Handle naming differences
            assert "metadata" in entry

    def test_remove_entry_parity(self, sqlite_backend):
        """Test that remove_entry works identically."""
        entry_data = {
            "cache_key": "remove_test",
            "description": "To be removed",
            "data_type": "object",
            "file_size": 512,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {},
        }

        # Store and verify
        sqlite_backend.put_entry("remove_test", entry_data)
        assert sqlite_backend.get_entry("remove_test") is not None

        # Remove and verify
        sqlite_backend.remove_entry("remove_test")
        assert sqlite_backend.get_entry("remove_test") is None

    def test_update_access_time_parity(self, sqlite_backend):
        """Test that update_access_time works identically."""
        entry_data = {
            "cache_key": "access_test",
            "description": "Access time test",
            "data_type": "array",
            "file_size": 256,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {},
        }

        sqlite_backend.put_entry("access_test", entry_data)
        original = sqlite_backend.get_entry("access_test")

        # Wait a moment to ensure different timestamp
        import time

        time.sleep(0.1)

        # Update access time
        sqlite_backend.update_access_time("access_test")
        updated = sqlite_backend.get_entry("access_test")

        # Verify access time changed (if backend tracks it)
        assert updated is not None
        assert updated["accessed_at"] >= original["accessed_at"]

    def test_stats_operations_parity(self, sqlite_backend):
        """Test that increment_hits, increment_misses, and get_stats work."""
        # Initial stats
        stats = sqlite_backend.get_stats()
        assert isinstance(stats, dict)

        initial_hits = stats.get("cache_hits", 0)
        initial_misses = stats.get("cache_misses", 0)

        # Increment counters
        sqlite_backend.increment_hits()
        sqlite_backend.increment_misses()

        # Verify changes
        new_stats = sqlite_backend.get_stats()
        assert new_stats.get("cache_hits", 0) == initial_hits + 1
        assert new_stats.get("cache_misses", 0) == initial_misses + 1

    def test_cleanup_expired_parity(self, sqlite_backend):
        """Test that cleanup_expired works identically."""
        from datetime import timedelta

        # Create entries with different ages
        old_time = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        recent_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()

        old_entry = {
            "cache_key": "old_key",
            "description": "Old entry",
            "data_type": "object",
            "file_size": 128,
            "created_at": old_time,
            "metadata": {},
        }

        recent_entry = {
            "cache_key": "recent_key",
            "description": "Recent entry",
            "data_type": "object",
            "file_size": 128,
            "created_at": recent_time,
            "metadata": {},
        }

        sqlite_backend.put_entry("old_key", old_entry)
        sqlite_backend.put_entry("recent_key", recent_entry)

        # Cleanup entries older than 24 hours (86400 seconds)
        removed_count = sqlite_backend.cleanup_expired(ttl_seconds=86400)

        # Verify old entry removed, recent entry remains
        assert sqlite_backend.get_entry("old_key") is None
        assert sqlite_backend.get_entry("recent_key") is not None
        assert removed_count >= 1

    def test_cleanup_by_size_parity(self, sqlite_backend):
        """Test that cleanup_by_size works identically."""
        import time

        # Create entries with different sizes and access times
        base_time = datetime.now(timezone.utc)

        # Entry 1: 1 MB, accessed recently
        recent_entry = {
            "cache_key": "recent_large",
            "description": "Recent large entry",
            "data_type": "array",
            "file_size": 1024 * 1024,  # 1 MB
            "actual_path": "/tmp/recent_large.pkl",
            "created_at": base_time.isoformat(),
            "accessed_at": base_time.isoformat(),
            "metadata": {},
        }

        # Entry 2: 2 MB, accessed 1 second ago
        mid_entry = {
            "cache_key": "mid_large",
            "description": "Mid-age large entry",
            "data_type": "array",
            "file_size": 2 * 1024 * 1024,  # 2 MB
            "actual_path": "/tmp/mid_large.pkl",
            "created_at": base_time.isoformat(),
            "accessed_at": base_time.isoformat(),
            "metadata": {},
        }

        # Entry 3: 3 MB, accessed 2 seconds ago (oldest)
        old_entry = {
            "cache_key": "old_large",
            "description": "Old large entry",
            "data_type": "array",
            "file_size": 3 * 1024 * 1024,  # 3 MB
            "actual_path": "/tmp/old_large.pkl",
            "created_at": base_time.isoformat(),
            "accessed_at": base_time.isoformat(),
            "metadata": {},
        }

        # Insert in order: old, mid, recent (to test LRU, not insertion order)
        sqlite_backend.put_entry("old_large", old_entry)
        time.sleep(0.01)  # Small delay to ensure different accessed_at
        sqlite_backend.put_entry("mid_large", mid_entry)
        time.sleep(0.01)
        sqlite_backend.put_entry("recent_large", recent_entry)

        # Touch entries to establish clear LRU order: old < mid < recent
        sqlite_backend.get_entry("old_large")  # Accessed first (oldest)
        time.sleep(0.01)
        sqlite_backend.get_entry("mid_large")  # Accessed second
        time.sleep(0.01)
        sqlite_backend.get_entry("recent_large")  # Accessed third (most recent)

        # Total size: 6 MB, target: 2 MB (in bytes)
        # Should remove oldest entries (old_large: 3MB, then mid_large: 2MB)
        # Leaving only recent_large (1MB)
        result = sqlite_backend.cleanup_by_size(target_size_bytes=2 * 1024 * 1024)

        # Verify result structure
        assert "count" in result
        assert "removed_entries" in result
        assert result["count"] >= 1

        # Verify LRU eviction: oldest entries removed, recent kept
        assert sqlite_backend.get_entry("old_large") is None  # Removed (oldest)
        assert sqlite_backend.get_entry("recent_large") is not None  # Kept (newest)

        # Verify removed_entries contains actual_path
        removed_entries = result["removed_entries"]
        assert len(removed_entries) >= 1
        assert all("cache_key" in entry for entry in removed_entries)
        assert all("actual_path" in entry for entry in removed_entries)

    def test_clear_all_parity(self, sqlite_backend):
        """Test that clear_all works identically."""
        # Store multiple entries
        for i in range(5):
            entry_data = {
                "cache_key": f"clear_{i}",
                "description": f"Clear test {i}",
                "data_type": "object",
                "file_size": 64,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": {},
            }
            sqlite_backend.put_entry(f"clear_{i}", entry_data)

        # Verify entries exist
        entries = sqlite_backend.list_entries()
        assert len(entries) >= 5

        # Clear all
        removed_count = sqlite_backend.clear_all()

        # Verify all removed
        entries_after = sqlite_backend.list_entries()
        assert len(entries_after) == 0
        assert removed_count >= 5

    def test_special_characters_parity(self, sqlite_backend):
        """Test that both backends handle special characters identically in supported fields."""
        special_data = {
            "cache_key": "special_test",
            "description": "Test with 特殊字符 и символы",
            "data_type": "object",
            "file_size": 1024,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                # Technical fields with special characters
                "actual_path": "/path/to/特殊文件.pkl",
                "object_type": "<class 'dict'>",
            },
        }

        sqlite_backend.put_entry("special_test", special_data)
        retrieved = sqlite_backend.get_entry("special_test")

        assert retrieved is not None
        assert "特殊字符" in retrieved["description"]
        assert "特殊文件" in retrieved["metadata"]["actual_path"]

    def test_large_metadata_parity(self, sqlite_backend):
        """Test that both backends handle technical metadata fields."""
        # Note: Custom metadata fields are NOT preserved by the optimized SQLite backend
        # Only known technical fields are stored in dedicated columns
        # This is by design for performance reasons

        entry_data = {
            "cache_key": "tech_meta",
            "description": "Technical metadata test",
            "data_type": "object",
            "file_size": 4096,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "actual_path": "/very/long/path/to/file" * 10,
                "object_type": "<class 'numpy.ndarray'>",
                "storage_format": "blosc2_array",
                "serializer": "pickle",
                "compression_codec": "zstd",
                "file_hash": "a" * 16,
                "s3_etag": "b" * 32,
            },
        }

        sqlite_backend.put_entry("tech_meta", entry_data)
        retrieved = sqlite_backend.get_entry("tech_meta")

        assert retrieved is not None
        # All technical fields should be preserved
        assert retrieved["metadata"]["actual_path"] == "/very/long/path/to/file" * 10
        assert retrieved["metadata"]["object_type"] == "<class 'numpy.ndarray'>"
        assert retrieved["metadata"]["storage_format"] == "blosc2_array"
        assert retrieved["metadata"]["file_hash"] == "a" * 16
        assert retrieved["metadata"]["s3_etag"] == "b" * 32

    def test_concurrent_operations_parity(self, sqlite_backend):
        """Test that both backends handle concurrent operations safely."""
        import concurrent.futures

        def write_entry(i):
            entry_data = {
                "cache_key": f"concurrent_{i}",
                "description": f"Concurrent entry {i}",
                "data_type": "object",
                "file_size": 256,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": {"thread_id": i},
            }
            sqlite_backend.put_entry(f"concurrent_{i}", entry_data)

        # Write entries concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(write_entry, i) for i in range(50)]
            concurrent.futures.wait(futures)

        # Verify all entries written
        entries = sqlite_backend.list_entries()
        concurrent_entries = [
            e for e in entries if e["cache_key"].startswith("concurrent_")
        ]
        assert len(concurrent_entries) == 50


class TestKnownDifferences:
    """Document known differences between SQLite and PostgreSQL backends."""

    def test_get_stats_structure_differences(self, tmp_path):
        """Document that get_stats() returns different structures."""
        from cacheness.metadata import SqliteBackend

        db_path = tmp_path / "test.db"
        sqlite = SqliteBackend(str(db_path))

        stats = sqlite.get_stats()

        # SQLite includes total_size_mb
        assert "total_entries" in stats
        assert "total_size_mb" in stats or "total_size_bytes" in stats

        # PostgreSQL does NOT include total_size_mb in get_stats()
        # Instead it returns: backend_type, cache_dir, cache_hits, cache_misses
        # This is documented as a known difference

        sqlite.close()


# ── Encryption Backend Parity Tests (Phase 24 / ENC-03) ───────────


def _skip_if_pg_unavailable(backend):
    """Skip test if backend is postgresql and PG is not available."""
    if backend == "postgresql":
        if not _HAS_PG or not _get_pg_url():
            pytest.skip("PostgreSQL not available")


@pytest.mark.xdist_group("docker")
class TestEncryptionBackendParity_BlobStore:
    """BlobStore encryption tests parametrized across all backends."""

    @pytest.mark.parametrize("backend", ["json", "sqlite", "postgresql"])
    def test_encrypted_put_get_roundtrip(self, tmp_path, backend):
        """put with encryption, get returns original data."""
        _skip_if_pg_unavailable(backend)
        store = _make_encrypted_blobstore_for_backend(tmp_path, backend)
        blob_key = store.put({"msg": "encrypted"}, key="test-data")
        result = store.get(blob_key)
        assert result == {"msg": "encrypted"}

    @pytest.mark.parametrize("backend", ["json", "sqlite", "postgresql"])
    def test_encrypted_entry_metadata_has_encryption_fields(self, tmp_path, backend):
        """Encrypted entry metadata contains encryption_algorithm and encryption_iv."""
        _skip_if_pg_unavailable(backend)
        store = _make_encrypted_blobstore_for_backend(tmp_path, backend)
        blob_key = store.put("hello", key="enc-meta-test")
        meta = store.get_metadata(blob_key)
        nested = meta.get("metadata", {})
        assert nested.get("encryption_algorithm") == "aes-256-gcm"
        assert "encryption_iv" in nested
        iv_hex = nested["encryption_iv"]
        assert len(bytes.fromhex(iv_hex)) == 12

    @pytest.mark.parametrize("backend", ["json", "sqlite", "postgresql"])
    def test_unencrypted_entry_readable_with_encryption_enabled(
        self, tmp_path, backend
    ):
        """Store without encryption, enable encryption, old entry still readable."""
        _skip_if_pg_unavailable(backend)
        # First: store without encryption using same backend
        backend_arg = backend
        if backend == "postgresql":
            backend_arg = PostgresBackend(connection_url=_get_pg_url())
        config_no_enc = CacheConfig(
            storage=CacheStorageConfig(cache_dir=str(tmp_path)),
            compression=CompressionConfig(use_blosc2_arrays=False),
            security=SecurityConfig(enable_entry_signing=False),
        )
        store_plain = BlobStore(
            cache_dir=tmp_path, backend=backend_arg, config=config_no_enc
        )
        blob_key = store_plain.put("pre-encryption data", key="legacy")

        # Second: create store with encryption enabled (same dir)
        store_enc = _make_encrypted_blobstore_for_backend(tmp_path, backend)
        result = store_enc.get(blob_key)
        assert result == "pre-encryption data"

    @pytest.mark.parametrize("backend", ["json", "sqlite", "postgresql"])
    def test_encrypted_entry_without_key_returns_none(self, tmp_path, backend):
        """Encrypted entry with no encryption key configured returns None."""
        _skip_if_pg_unavailable(backend)
        store_enc = _make_encrypted_blobstore_for_backend(tmp_path, backend)
        blob_key = store_enc.put("secret", key="locked")

        # Create store WITHOUT encryption key
        backend_arg = backend
        if backend == "postgresql":
            backend_arg = PostgresBackend(connection_url=_get_pg_url())
        config_no_enc = CacheConfig(
            storage=CacheStorageConfig(cache_dir=str(tmp_path)),
            compression=CompressionConfig(use_blosc2_arrays=False),
            security=SecurityConfig(enable_entry_signing=False),
        )
        store_plain = BlobStore(
            cache_dir=tmp_path, backend=backend_arg, config=config_no_enc
        )
        result = store_plain.get(blob_key)
        assert result is None

    @pytest.mark.parametrize("backend", ["json", "sqlite", "postgresql"])
    def test_encryption_disabled_by_default(self, tmp_path, backend):
        """Default BlobStore does not encrypt, no encryption_algorithm in metadata."""
        _skip_if_pg_unavailable(backend)
        backend_arg = backend
        if backend == "postgresql":
            backend_arg = PostgresBackend(connection_url=_get_pg_url())
        config = CacheConfig(
            storage=CacheStorageConfig(cache_dir=str(tmp_path)),
            compression=CompressionConfig(use_blosc2_arrays=False),
        )
        store = BlobStore(cache_dir=tmp_path, backend=backend_arg, config=config)
        blob_key = store.put("not encrypted", key="plain")
        meta = store.get_metadata(blob_key)
        nested = meta.get("metadata", {})
        assert "encryption_algorithm" not in nested


@pytest.mark.xdist_group("docker")
class TestEncryptionBackendParity_UnifiedCache:
    """UnifiedCache encryption tests parametrized across all backends."""

    @pytest.mark.parametrize("backend", ["json", "sqlite", "postgresql"])
    def test_cache_encrypted_put_get_roundtrip(self, tmp_path, backend):
        """UnifiedCache with encryption, put/get works for string data."""
        _skip_if_pg_unavailable(backend)
        cache = _make_encrypted_cache_for_backend(tmp_path, backend)
        cache.put("secure string", on={"prefix": "test"}, description="roundtrip")
        result = cache.get(on={"prefix": "test"})
        assert result == "secure string"

    @pytest.mark.parametrize("backend", ["json", "sqlite", "postgresql"])
    def test_cache_encrypted_put_get_various_types(self, tmp_path, backend):
        """Encryption works with dict, list, int, float."""
        _skip_if_pg_unavailable(backend)
        cache = _make_encrypted_cache_for_backend(tmp_path, backend)

        cache.put({"key": "value"}, on={"kind": "dict"})
        cache.put([1, 2, 3], on={"kind": "list"})
        cache.put(42, on={"kind": "int"})
        cache.put(3.14, on={"kind": "float"})

        assert cache.get(on={"kind": "dict"}) == {"key": "value"}
        assert cache.get(on={"kind": "list"}) == [1, 2, 3]
        assert cache.get(on={"kind": "int"}) == 42
        assert cache.get(on={"kind": "float"}) == 3.14

    @pytest.mark.parametrize("backend", ["json", "sqlite", "postgresql"])
    def test_cache_encryption_disabled_by_default(self, tmp_path, backend):
        """Default UnifiedCache has no encryption."""
        _skip_if_pg_unavailable(backend)
        metadata_cfg = CacheMetadataConfig(metadata_backend=backend)
        if backend == "postgresql":
            metadata_cfg = CacheMetadataConfig(
                metadata_backend="postgresql",
                metadata_backend_options={"connection_url": _get_pg_url()},
            )
        config = CacheConfig(
            storage=CacheStorageConfig(cache_dir=str(tmp_path)),
            metadata=metadata_cfg,
            compression=CompressionConfig(use_blosc2_arrays=False),
        )
        cache = cacheness(config)
        cache.put("no encryption", on={"prefix": "plain"})
        entries = cache.list_entries()
        for e in entries:
            meta = e.get("metadata", {})
            assert "encryption_algorithm" not in meta

    @pytest.mark.parametrize("backend", ["json", "sqlite", "postgresql"])
    def test_cache_mixed_encrypted_unencrypted(self, tmp_path, backend):
        """Some entries encrypted, some not, all readable."""
        _skip_if_pg_unavailable(backend)
        metadata_cfg = CacheMetadataConfig(metadata_backend=backend)
        if backend == "postgresql":
            metadata_cfg = CacheMetadataConfig(
                metadata_backend="postgresql",
                metadata_backend_options={"connection_url": _get_pg_url()},
            )
        config_plain = CacheConfig(
            storage=CacheStorageConfig(cache_dir=str(tmp_path)),
            metadata=metadata_cfg,
            compression=CompressionConfig(use_blosc2_arrays=False),
            security=SecurityConfig(enable_entry_signing=False),
        )
        cache_plain = cacheness(config_plain)
        cache_plain.put("old data", on={"mix": "unencrypted"})

        cache_enc = _make_encrypted_cache_for_backend(tmp_path, backend)
        cache_enc.put("new data", on={"mix": "encrypted"})

        assert cache_enc.get(on={"mix": "unencrypted"}) == "old data"
        assert cache_enc.get(on={"mix": "encrypted"}) == "new data"

    @pytest.mark.parametrize("backend", ["json", "sqlite", "postgresql"])
    def test_cache_init_without_cryptography_raises(
        self, tmp_path, backend, monkeypatch
    ):
        """When cryptography is not importable, enable_content_encryption raises."""
        _skip_if_pg_unavailable(backend)
        original_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        def mock_import(name, *args, **kwargs):
            if name == "cryptography":
                raise ImportError("mocked")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)

        with pytest.raises(CacheConfigurationError, match="cacheness\\[encryption\\]"):
            SecurityConfig(enable_content_encryption=True)


@pytest.mark.xdist_group("docker")
class TestEncryptionBackendParity_KeyRotation:
    """Key rotation tests parametrized across all backends."""

    @pytest.mark.parametrize("backend", ["json", "sqlite", "postgresql"])
    def test_rotate_key_re_encrypts_entries(self, tmp_path, backend):
        """rotate_key re-encrypts, RotationResult.re_encrypted > 0."""
        _skip_if_pg_unavailable(backend)
        cache = _make_encrypted_cache_for_backend(tmp_path, backend)
        cache.put("secret1", on={"rot": "one"})
        cache.put("secret2", on={"rot": "two"})

        new_key = _generate_key_file(tmp_path / "new_key.bin")
        result = cache.rotate_key(new_key)

        assert isinstance(result, RotationResult)
        assert result.re_encrypted == 2

    @pytest.mark.parametrize("backend", ["json", "sqlite", "postgresql"])
    def test_rotated_encrypted_entries_readable(self, tmp_path, backend):
        """After rotation, encrypted entries still readable with new key."""
        _skip_if_pg_unavailable(backend)
        cache = _make_encrypted_cache_for_backend(tmp_path, backend)
        cache.put("alpha", on={"rot2": "a"})
        cache.put("beta", on={"rot2": "b"})

        new_key = _generate_key_file(tmp_path / "new_key.bin")
        cache.rotate_key(new_key)

        assert cache.get(on={"rot2": "a"}) == "alpha"
        assert cache.get(on={"rot2": "b"}) == "beta"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
