"""Tests for put_file / get_file on UnifiedCache and BlobStore.

CACHE-zsw: File storage convenience API for arbitrary files.
"""

import tempfile
import shutil
import gc
import time
import uuid
import pytest
from pathlib import Path

from cacheness.core import UnifiedCache
from cacheness import CacheConfig
from cacheness.config import CacheMetadataConfig
from cacheness.storage.blob_store import BlobStore
from cacheness.custom_metadata import (
    custom_metadata_model,
    CustomMetadataBase,
)
from cacheness.metadata import Base
from sqlalchemy import Column, String, Float


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def temp_dir(request):
    """Create a temporary directory for cache and test files."""
    d = tempfile.mkdtemp()

    def cleanup():
        gc.collect()
        time.sleep(0.2)
        if Path(d).exists():
            try:
                shutil.rmtree(d)
            except PermissionError:
                time.sleep(0.5)
                shutil.rmtree(d)

    request.addfinalizer(cleanup)
    return Path(d)


@pytest.fixture
def cache(temp_dir):
    """UnifiedCache with store_full_metadata=True."""
    config = CacheConfig(
        cache_dir=str(temp_dir / "cache"),
        metadata_backend="sqlite",
        store_full_metadata=True,
    )
    c = UnifiedCache(config)
    yield c
    c.close()


@pytest.fixture
def cache_no_meta(temp_dir):
    """UnifiedCache with store_full_metadata=False."""
    config = CacheConfig(
        cache_dir=str(temp_dir / "cache_nometa"),
        metadata_backend="sqlite",
        store_full_metadata=False,
    )
    c = UnifiedCache(config)
    yield c
    c.close()


@pytest.fixture
def blob_store(temp_dir):
    """Standalone BlobStore."""
    store = BlobStore(cache_dir=str(temp_dir / "blobs"))
    yield store
    store.close()


@pytest.fixture
def sample_file(temp_dir):
    """Create a sample text file."""
    f = temp_dir / "sample.txt"
    f.write_text("hello world\n", encoding="utf-8")
    return f


@pytest.fixture
def sample_binary(temp_dir):
    """Create a sample binary file."""
    f = temp_dir / "image.png"
    # Minimal PNG header + some bytes
    f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    return f


@pytest.fixture
def sample_csv(temp_dir):
    """Create a sample CSV file."""
    f = temp_dir / "data.csv"
    f.write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")
    return f


# ── UnifiedCache: put_file / get_file ───────────────────────────────


class TestUnifiedCachePutFile:
    def test_round_trip_text_file(self, cache, sample_file):
        """Store and retrieve a text file."""
        key = cache.put_file(sample_file)
        assert isinstance(key, str) and len(key) == 16

        data = cache.get_file(cache_key=key)
        assert data == sample_file.read_bytes()

    def test_round_trip_binary_file(self, cache, sample_binary):
        """Store and retrieve a binary file."""
        key = cache.put_file(sample_binary)
        data = cache.get_file(cache_key=key)
        assert data == sample_binary.read_bytes()

    def test_metadata_populated(self, cache, sample_file):
        """File metadata auto-populated in metadata_dict."""
        key = cache.put_file(sample_file)
        entry = cache.metadata_backend.get_entry(key)
        assert entry is not None
        meta = cache._extract_metadata_dict(entry)
        assert meta["original_filename"] == "sample.txt"
        assert meta["mime_type"] == "text/plain"
        assert meta["original_size"] == len(sample_file.read_bytes())

    def test_mime_type_png(self, cache, sample_binary):
        """PNG file gets correct MIME type."""
        key = cache.put_file(sample_binary)
        entry = cache.metadata_backend.get_entry(key)
        meta = cache._extract_metadata_dict(entry)
        assert meta["mime_type"] == "image/png"

    def test_mime_type_csv(self, cache, sample_csv):
        """CSV file gets a CSV-related MIME type (varies by platform)."""
        key = cache.put_file(sample_csv)
        entry = cache.metadata_backend.get_entry(key)
        meta = cache._extract_metadata_dict(entry)
        # Windows may return 'application/vnd.ms-excel' instead of 'text/csv'
        assert meta["mime_type"] in ("text/csv", "application/vnd.ms-excel")

    def test_explicit_cache_key(self, cache, sample_file):
        """Explicit cache_key is respected."""
        key = cache.put_file(sample_file, cache_key="myfile123456abcd")
        assert key == "myfile123456abcd"
        data = cache.get_file(cache_key="myfile123456abcd")
        assert data == sample_file.read_bytes()

    def test_with_on_param(self, cache, sample_file):
        """on= parameter changes cache key."""
        k1 = cache.put_file(sample_file, on={"version": "v1"})
        k2 = cache.put_file(sample_file, on={"version": "v2"})
        assert k1 != k2

    def test_with_kwargs(self, cache, sample_file):
        """Extra kwargs participate in key derivation."""
        k1 = cache.put_file(sample_file, run="A")
        k2 = cache.put_file(sample_file, run="B")
        assert k1 != k2

    def test_with_description(self, cache, sample_file):
        """Description is stored but doesn't affect key."""
        k1 = cache.put_file(sample_file, description="first", tag="x")
        k2 = cache.put_file(sample_file, description="second", tag="x")
        assert k1 == k2  # description not part of key

    def test_file_not_found(self, cache, temp_dir):
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            cache.put_file(temp_dir / "nonexistent.txt")

    def test_directory_raises(self, cache, temp_dir):
        """Directory path raises IsADirectoryError."""
        with pytest.raises(IsADirectoryError, match="directory"):
            cache.put_file(temp_dir)

    def test_str_path_accepted(self, cache, sample_file):
        """String path works (not just pathlib.Path)."""
        key = cache.put_file(str(sample_file))
        assert cache.get_file(cache_key=key) == sample_file.read_bytes()

    def test_works_without_store_full_metadata(self, cache_no_meta, sample_file):
        """put_file works even without store_full_metadata (metadata not queryable)."""
        key = cache_no_meta.put_file(sample_file)
        data = cache_no_meta.get_file(cache_key=key)
        assert data == sample_file.read_bytes()

    def test_user_kwargs_override_file_meta(self, cache, sample_file):
        """User-supplied kwargs override auto-populated file metadata."""
        key = cache.put_file(sample_file, original_filename="custom_name.dat")
        entry = cache.metadata_backend.get_entry(key)
        meta = cache._extract_metadata_dict(entry)
        assert meta["original_filename"] == "custom_name.dat"

    def test_unknown_extension_fallback_mime(self, cache, temp_dir):
        """Unknown file extension gets application/octet-stream."""
        f = temp_dir / "data.xyz789"
        f.write_bytes(b"mystery")
        key = cache.put_file(f)
        entry = cache.metadata_backend.get_entry(key)
        meta = cache._extract_metadata_dict(entry)
        assert meta["mime_type"] == "application/octet-stream"


class TestUnifiedCacheGetFile:
    def test_miss_returns_none(self, cache):
        """Cache miss returns None."""
        assert cache.get_file(cache_key="0000000000000000") is None

    def test_get_file_to_dest(self, cache, sample_file, temp_dir):
        """get_file with dest= writes file to disk."""
        key = cache.put_file(sample_file)
        dest = temp_dir / "output" / "restored.txt"
        result = cache.get_file(cache_key=key, dest=dest)
        assert isinstance(result, Path)
        assert result == dest
        assert result.read_bytes() == sample_file.read_bytes()

    def test_get_file_to_dest_dir(self, cache, sample_file, temp_dir):
        """get_file with dest= pointing to a directory uses original filename."""
        key = cache.put_file(sample_file)
        dest_dir = temp_dir / "output_dir"
        dest_dir.mkdir()
        result = cache.get_file(cache_key=key, dest=dest_dir)
        assert isinstance(result, Path)
        assert result.name == "sample.txt"
        assert result.read_bytes() == sample_file.read_bytes()

    def test_get_file_to_dest_dir_fallback_name(
        self, cache_no_meta, sample_file, temp_dir
    ):
        """When metadata_dict is unavailable, falls back to <key>.bin."""
        key = cache_no_meta.put_file(sample_file)
        dest_dir = temp_dir / "output_dir2"
        dest_dir.mkdir()
        result = cache_no_meta.get_file(cache_key=key, dest=dest_dir)
        assert isinstance(result, Path)
        assert result.name == f"{key}.bin"

    def test_get_file_creates_parent_dirs(self, cache, sample_file, temp_dir):
        """get_file creates intermediate directories for dest path."""
        key = cache.put_file(sample_file)
        dest = temp_dir / "deep" / "nested" / "dir" / "out.txt"
        result = cache.get_file(cache_key=key, dest=dest)
        assert result.exists()
        assert result.read_bytes() == sample_file.read_bytes()

    def test_get_file_with_on(self, cache, sample_file, temp_dir):
        """get_file with on= param for key lookup."""
        cache.put_file(sample_file, on={"version": "v1"})
        data = cache.get_file(on={"version": "v1"})
        assert data == sample_file.read_bytes()

    def test_get_file_with_kwargs(self, cache, sample_file):
        """get_file with kwargs for key lookup."""
        cache.put_file(sample_file, run="A")
        data = cache.get_file(run="A")
        assert data == sample_file.read_bytes()

    def test_get_file_type_error_on_non_bytes(self, cache):
        """get_file raises TypeError if the entry isn't bytes."""
        cache.put({"dict": "data"}, cache_key="notbytes12345678")
        with pytest.raises(TypeError, match="Expected bytes"):
            cache.get_file(cache_key="notbytes12345678")

    def test_get_file_with_ttl(self, cache, sample_file):
        """get_file supports TTL parameters."""
        key = cache.put_file(sample_file)
        # Should not expire — ample TTL
        data = cache.get_file(cache_key=key, ttl="1h")
        assert data == sample_file.read_bytes()


# ── BlobStore: put_file / get_file ──────────────────────────────────


class TestBlobStorePutFile:
    def test_round_trip(self, blob_store, sample_file):
        """Store and retrieve file via BlobStore."""
        key = blob_store.put_file(sample_file)
        data = blob_store.get_file(key)
        assert data == sample_file.read_bytes()

    def test_metadata_populated(self, blob_store, sample_file):
        """File metadata stored in blob entry."""
        key = blob_store.put_file(sample_file)
        entry = blob_store.get_metadata(key)
        assert entry is not None
        nested = entry.get("metadata", {})
        assert nested["original_filename"] == "sample.txt"
        assert nested["original_size"] == len(sample_file.read_bytes())

    def test_explicit_key(self, blob_store, sample_file):
        """Explicit key is respected."""
        key = blob_store.put_file(sample_file, key="my-doc")
        assert key == "my-doc"
        assert blob_store.get_file("my-doc") == sample_file.read_bytes()

    def test_user_metadata_merged(self, blob_store, sample_file):
        """User metadata is merged with file metadata."""
        key = blob_store.put_file(sample_file, metadata={"project": "alpha"})
        entry = blob_store.get_metadata(key)
        nested = entry.get("metadata", {})
        assert nested["project"] == "alpha"
        assert nested["original_filename"] == "sample.txt"

    def test_file_not_found(self, blob_store, temp_dir):
        with pytest.raises(FileNotFoundError):
            blob_store.put_file(temp_dir / "nope.bin")

    def test_directory_raises(self, blob_store, temp_dir):
        with pytest.raises(IsADirectoryError):
            blob_store.put_file(temp_dir)


class TestBlobStoreGetFile:
    def test_miss_returns_none(self, blob_store):
        assert blob_store.get_file("nonexistent") is None

    def test_get_to_dest(self, blob_store, sample_file, temp_dir):
        """get_file with dest= writes file."""
        key = blob_store.put_file(sample_file)
        dest = temp_dir / "restored.txt"
        result = blob_store.get_file(key, dest=dest)
        assert isinstance(result, Path)
        assert result.read_bytes() == sample_file.read_bytes()

    def test_get_to_dest_dir(self, blob_store, sample_file, temp_dir):
        """dest= directory uses original filename from metadata."""
        key = blob_store.put_file(sample_file)
        dest_dir = temp_dir / "out"
        dest_dir.mkdir()
        result = blob_store.get_file(key, dest=dest_dir)
        assert result.name == "sample.txt"

    def test_type_error_on_non_bytes(self, blob_store):
        """get_file raises TypeError for non-bytes blobs."""
        key = blob_store.put({"not": "bytes"})
        with pytest.raises(TypeError, match="Expected bytes"):
            blob_store.get_file(key)


# ── Query integration ───────────────────────────────────────────────


class TestFileQueryIntegration:
    def test_query_by_mime_type(self, cache, sample_file, sample_binary):
        """Files stored via put_file are queryable by mime_type."""
        cache.put_file(sample_file, tag="a")
        cache.put_file(sample_binary, tag="b")

        results = cache.query_meta(mime_type="text/plain")
        assert results is not None and len(results) == 1
        assert results[0]["metadata_dict"]["original_filename"] == "sample.txt"

    def test_query_by_original_filename(self, cache, sample_file):
        """Files are queryable by original_filename."""
        cache.put_file(sample_file, tag="f1")
        results = cache.query_meta(original_filename="sample.txt")
        assert results is not None and len(results) == 1

    def test_plain_get_returns_bytes(self, cache, sample_file):
        """Plain get() on a file entry returns bytes."""
        key = cache.put_file(sample_file)
        data = cache.get(cache_key=key)
        assert isinstance(data, bytes)
        assert data == sample_file.read_bytes()


# ── Custom metadata integration ─────────────────────────────────────


def _make_file_model():
    """Create a unique ORM model class for test isolation."""
    suffix = uuid.uuid4().hex[:8]
    schema_name = f"file_meta_{suffix}"
    table_name = f"custom_file_meta_{suffix}"

    @custom_metadata_model(schema_name)
    class _FileModel(Base, CustomMetadataBase):
        __tablename__ = table_name
        __table_args__ = {"extend_existing": True}

        project = Column(String(100), nullable=False, index=True)
        version = Column(String(50), nullable=False)
        score = Column(Float, nullable=True)

    return _FileModel, schema_name


class TestPutFileCustomMetadata:
    def test_custom_metadata_orm(self, temp_dir, sample_file):
        """put_file with custom_metadata stores ORM objects."""
        Model, schema_name = _make_file_model()
        config = CacheConfig(
            cache_dir=str(temp_dir / "cache_orm"),
            metadata=CacheMetadataConfig(
                metadata_backend="sqlite",
                store_full_metadata=True,
            ),
        )
        cache = UnifiedCache(config)
        try:
            meta_obj = Model(project="alpha", version="1.0", score=0.95)
            key = cache.put_file(
                sample_file,
                custom_metadata=meta_obj,
                run="orm_test",
            )

            # Verify file round-trip
            data = cache.get_file(cache_key=key)
            assert data == sample_file.read_bytes()

            # Verify ORM metadata was stored
            with cache.query_custom_session(schema_name) as query:
                rows = query.all()
                assert len(rows) == 1
                assert rows[0].project == "alpha"
                assert rows[0].version == "1.0"
                assert rows[0].score == 0.95
                assert rows[0].cache_key == key
        finally:
            cache.close()

    def test_custom_metadata_with_on(self, temp_dir, sample_file):
        """put_file with custom_metadata AND on= works without ValueError."""
        Model, schema_name = _make_file_model()
        config = CacheConfig(
            cache_dir=str(temp_dir / "cache_orm_on"),
            metadata=CacheMetadataConfig(
                metadata_backend="sqlite",
                store_full_metadata=True,
            ),
        )
        cache = UnifiedCache(config)
        try:
            meta_obj = Model(project="beta", version="2.0", score=0.88)
            key = cache.put_file(
                sample_file,
                on={"experiment": "exp_01"},
                custom_metadata=meta_obj,
            )

            # Verify retrieval via on=
            data = cache.get_file(on={"experiment": "exp_01"})
            assert data == sample_file.read_bytes()

            # Verify ORM metadata
            with cache.query_custom_session(schema_name) as query:
                rows = query.all()
                assert len(rows) == 1
                assert rows[0].project == "beta"
        finally:
            cache.close()


# ── Move semantics ──────────────────────────────────────────────────


class TestPutFileMoveUnifiedCache:
    def test_move_deletes_source(self, cache, sample_file):
        """put_file(move=True) deletes source after store."""
        original_bytes = sample_file.read_bytes()
        key = cache.put_file(sample_file, move=True)

        # Source should be gone
        assert not sample_file.exists()
        # Data should be in cache
        assert cache.get_file(cache_key=key) == original_bytes

    def test_copy_preserves_source(self, cache, sample_file):
        """put_file(move=False) preserves source (default)."""
        cache.put_file(sample_file)
        assert sample_file.exists()

    def test_move_with_on(self, cache, sample_file):
        """move=True works with on= key derivation."""
        original_bytes = sample_file.read_bytes()
        key = cache.put_file(sample_file, on={"job": "m1"}, move=True)
        assert not sample_file.exists()
        assert cache.get_file(on={"job": "m1"}) == original_bytes

    def test_move_with_explicit_key(self, cache, sample_file):
        """move=True works with explicit cache_key."""
        cache.put_file(sample_file, cache_key="movein123456abcd", move=True)
        assert not sample_file.exists()
        assert cache.get_file(cache_key="movein123456abcd") is not None


class TestGetFileMoveUnifiedCache:
    def test_move_writes_and_deletes_entry(self, cache, sample_file, temp_dir):
        """get_file(move=True) writes to dest and deletes cache entry."""
        original_bytes = sample_file.read_bytes()
        key = cache.put_file(sample_file)

        dest = temp_dir / "moved_out.txt"
        result = cache.get_file(cache_key=key, dest=dest, move=True)

        assert result == dest
        assert dest.read_bytes() == original_bytes
        # Cache entry should be gone
        assert cache.get_file(cache_key=key) is None

    def test_move_without_dest_raises(self, cache, sample_file):
        """get_file(move=True) without dest raises ValueError."""
        key = cache.put_file(sample_file)
        with pytest.raises(ValueError, match="move=True requires dest"):
            cache.get_file(cache_key=key, move=True)

    def test_copy_out_preserves_entry(self, cache, sample_file, temp_dir):
        """get_file(move=False) preserves cache entry (default)."""
        key = cache.put_file(sample_file)
        dest = temp_dir / "copy_out.txt"
        cache.get_file(cache_key=key, dest=dest)
        # Entry still in cache
        assert cache.get_file(cache_key=key) is not None

    def test_move_with_on(self, cache, sample_file, temp_dir):
        """get_file(move=True) works with on= key lookup."""
        original_bytes = sample_file.read_bytes()
        cache.put_file(sample_file, on={"run": "r1"})
        dest = temp_dir / "moved.txt"
        result = cache.get_file(on={"run": "r1"}, dest=dest, move=True)
        assert result == dest
        assert dest.read_bytes() == original_bytes
        assert cache.get_file(on={"run": "r1"}) is None

    def test_move_miss_returns_none(self, cache, temp_dir):
        """get_file(move=True) on miss returns None without error."""
        dest = temp_dir / "nowhere.txt"
        assert (
            cache.get_file(cache_key="0000000000000000", dest=dest, move=True) is None
        )
        assert not dest.exists()

    def test_move_to_dest_dir(self, cache, sample_file, temp_dir):
        """get_file(move=True, dest=dir) resolves filename and deletes entry."""
        key = cache.put_file(sample_file)
        dest_dir = temp_dir / "move_dir"
        dest_dir.mkdir()
        result = cache.get_file(cache_key=key, dest=dest_dir, move=True)
        assert result.name == "sample.txt"
        assert result.exists()
        assert cache.get_file(cache_key=key) is None


class TestGetFileOverwriteUnifiedCache:
    def test_overwrite_true_default(self, cache, sample_file, temp_dir):
        """Default overwrite=True silently overwrites existing dest."""
        key = cache.put_file(sample_file)
        dest = temp_dir / "existing.txt"
        dest.write_text("old content")

        result = cache.get_file(cache_key=key, dest=dest)
        assert result == dest
        assert dest.read_bytes() == sample_file.read_bytes()

    def test_overwrite_false_on_existing(self, cache, sample_file, temp_dir):
        """overwrite=False raises FileExistsError if dest exists."""
        key = cache.put_file(sample_file)
        dest = temp_dir / "existing2.txt"
        dest.write_text("old content")

        with pytest.raises(FileExistsError, match="Destination already exists"):
            cache.get_file(cache_key=key, dest=dest, overwrite=False)
        # Original content unchanged
        assert dest.read_text() == "old content"

    def test_overwrite_false_on_new_file(self, cache, sample_file, temp_dir):
        """overwrite=False succeeds when dest doesn't exist."""
        key = cache.put_file(sample_file)
        dest = temp_dir / "new_file.txt"

        result = cache.get_file(cache_key=key, dest=dest, overwrite=False)
        assert result == dest
        assert dest.read_bytes() == sample_file.read_bytes()

    def test_overwrite_false_with_dir_dest(self, cache, sample_file, temp_dir):
        """overwrite=False with dir dest raises when resolved file exists."""
        key = cache.put_file(sample_file)
        dest_dir = temp_dir / "ow_dir"
        dest_dir.mkdir()
        # Pre-create the file that would be resolved
        (dest_dir / "sample.txt").write_text("old")

        with pytest.raises(FileExistsError):
            cache.get_file(cache_key=key, dest=dest_dir, overwrite=False)


# ── BlobStore: move + overwrite ─────────────────────────────────────


class TestBlobStorePutFileMove:
    def test_move_deletes_source(self, blob_store, sample_file):
        """put_file(move=True) deletes source."""
        original_bytes = sample_file.read_bytes()
        key = blob_store.put_file(sample_file, move=True)
        assert not sample_file.exists()
        assert blob_store.get_file(key) == original_bytes

    def test_copy_preserves_source(self, blob_store, sample_file):
        """put_file(move=False) preserves source (default)."""
        blob_store.put_file(sample_file)
        assert sample_file.exists()


class TestBlobStoreGetFileMove:
    def test_move_writes_and_deletes_entry(self, blob_store, sample_file, temp_dir):
        """get_file(move=True) writes to dest and deletes blob."""
        original_bytes = sample_file.read_bytes()
        key = blob_store.put_file(sample_file)

        dest = temp_dir / "blob_moved.txt"
        result = blob_store.get_file(key, dest=dest, move=True)

        assert result == dest
        assert dest.read_bytes() == original_bytes
        assert blob_store.get_file(key) is None

    def test_move_without_dest_raises(self, blob_store, sample_file):
        """get_file(move=True) without dest raises ValueError."""
        key = blob_store.put_file(sample_file)
        with pytest.raises(ValueError, match="move=True requires dest"):
            blob_store.get_file(key, move=True)

    def test_move_miss_returns_none(self, blob_store, temp_dir):
        """get_file(move=True) on miss returns None."""
        dest = temp_dir / "nowhere.txt"
        assert blob_store.get_file("nonexistent", dest=dest, move=True) is None


class TestBlobStoreGetFileOverwrite:
    def test_overwrite_true_default(self, blob_store, sample_file, temp_dir):
        """Default overwrite=True overwrites existing dest."""
        key = blob_store.put_file(sample_file)
        dest = temp_dir / "existing.txt"
        dest.write_text("old")

        result = blob_store.get_file(key, dest=dest)
        assert dest.read_bytes() == sample_file.read_bytes()

    def test_overwrite_false_raises(self, blob_store, sample_file, temp_dir):
        """overwrite=False raises FileExistsError on existing dest."""
        key = blob_store.put_file(sample_file)
        dest = temp_dir / "existing2.txt"
        dest.write_text("old")

        with pytest.raises(FileExistsError, match="Destination already exists"):
            blob_store.get_file(key, dest=dest, overwrite=False)

    def test_overwrite_false_succeeds_on_new(self, blob_store, sample_file, temp_dir):
        """overwrite=False succeeds when dest doesn't exist."""
        key = blob_store.put_file(sample_file)
        dest = temp_dir / "new.txt"
        result = blob_store.get_file(key, dest=dest, overwrite=False)
        assert result == dest
        assert dest.read_bytes() == sample_file.read_bytes()
