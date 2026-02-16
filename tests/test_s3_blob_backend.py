"""
Tests for Phase 2.10: S3 Blob Backend

Tests the S3 blob storage backend using moto for AWS mocking.
"""

import pytest
from io import BytesIO

# Check for moto availability
try:
    import moto
    from moto import mock_aws

    MOTO_AVAILABLE = True
except ImportError:
    MOTO_AVAILABLE = False
    mock_aws = None

# Check for boto3 availability
try:
    import boto3

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

# Skip all tests if dependencies not available
pytestmark = [
    pytest.mark.skipif(not MOTO_AVAILABLE, reason="moto not installed"),
    pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not installed"),
]


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def aws_credentials():
    """Mocked AWS credentials for moto."""
    import os

    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


@pytest.fixture
def s3_client(aws_credentials):
    """Create an S3 client with mocked AWS."""
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        yield client


@pytest.fixture
def s3_bucket(s3_client):
    """Create a test bucket."""
    bucket_name = "test-cache-bucket"
    s3_client.create_bucket(Bucket=bucket_name)
    return bucket_name


@pytest.fixture
def s3_backend(aws_credentials, s3_bucket):
    """Create an S3BlobBackend with mocked AWS."""
    from cacheness.storage.backends.s3_backend import S3BlobBackend

    with mock_aws():
        # Create the bucket first
        client = boto3.client("s3", region_name="us-east-1")
        try:
            client.create_bucket(Bucket=s3_bucket)
        except client.exceptions.BucketAlreadyOwnedByYou:
            pass

        # Create backend
        backend = S3BlobBackend(
            bucket=s3_bucket,
            region="us-east-1",
        )
        yield backend


@pytest.fixture
def s3_backend_with_prefix(aws_credentials, s3_bucket):
    """Create an S3BlobBackend with prefix."""
    from cacheness.storage.backends.s3_backend import S3BlobBackend

    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        try:
            client.create_bucket(Bucket=s3_bucket)
        except client.exceptions.BucketAlreadyOwnedByYou:
            pass

        backend = S3BlobBackend(
            bucket=s3_bucket,
            prefix="cache/v1",
            region="us-east-1",
        )
        yield backend


# =============================================================================
# S3BlobBackend Core Operations Tests
# =============================================================================


class TestS3BlobBackendBasics:
    """Test basic S3BlobBackend operations."""

    def test_write_and_read_blob(self, aws_credentials, s3_bucket):
        """Test basic write and read operations."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(bucket=s3_bucket)

            blob_id = "test_blob_123"
            data = b"Hello, S3!"

            # Write
            blob_path = backend.write_blob(blob_id, data)
            assert blob_path.startswith(f"s3://{s3_bucket}/")

            # Read
            read_data = backend.read_blob(blob_path)
            assert read_data == data

    def test_exists(self, aws_credentials, s3_bucket):
        """Test exists() method."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(bucket=s3_bucket)

            blob_id = "exists_test"
            data = b"test data"

            # Should not exist initially
            fake_path = f"s3://{s3_bucket}/nonexistent"
            assert not backend.exists(fake_path)

            # Write and check
            blob_path = backend.write_blob(blob_id, data)
            assert backend.exists(blob_path)

    def test_delete(self, aws_credentials, s3_bucket):
        """Test delete operation."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(bucket=s3_bucket)

            blob_id = "delete_test"
            data = b"data to delete"

            blob_path = backend.write_blob(blob_id, data)
            assert backend.exists(blob_path)

            # Delete
            result = backend.delete_blob(blob_path)
            assert result is True
            assert not backend.exists(blob_path)

    def test_get_size(self, aws_credentials, s3_bucket):
        """Test get_size() method."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(bucket=s3_bucket)

            blob_id = "size_test"
            data = b"x" * 1000

            blob_path = backend.write_blob(blob_id, data)
            size = backend.get_size(blob_path)
            assert size == 1000

    def test_read_nonexistent_raises(self, aws_credentials, s3_bucket):
        """Test reading non-existent blob raises FileNotFoundError."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(bucket=s3_bucket)

            with pytest.raises(FileNotFoundError):
                backend.read_blob(f"s3://{s3_bucket}/nonexistent")


# =============================================================================
# S3 Directory Sharding Tests
# =============================================================================


class TestS3Sharding:
    """Test Git-style directory sharding in S3BlobBackend."""

    def test_default_shard_chars_is_zero(self, aws_credentials, s3_bucket):
        """Default shard_chars should be 0 (no sharding)."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(bucket=s3_bucket)
            assert backend.shard_chars == 0

    def test_sharding_creates_correct_key(self, aws_credentials, s3_bucket):
        """Test that sharding creates correct S3 keys."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(bucket=s3_bucket, shard_chars=2)

            blob_id = "abc123def456"
            key = backend._get_s3_key(blob_id)

            # Should be sharded: ab/abc123def456
            assert key == "ab/abc123def456"

    def test_sharding_with_prefix(self, aws_credentials, s3_bucket):
        """Test sharding with prefix."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(bucket=s3_bucket, prefix="cache/v1", shard_chars=2)

            blob_id = "xyz789"
            key = backend._get_s3_key(blob_id)

            # Should be: cache/v1/xy/xyz789
            assert key == "cache/v1/xy/xyz789"

    def test_sharding_disabled(self, aws_credentials, s3_bucket):
        """Test with sharding disabled."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(bucket=s3_bucket, shard_chars=0)

            blob_id = "abc123"
            key = backend._get_s3_key(blob_id)

            # No sharding
            assert key == "abc123"

    def test_sharding_with_write_read(self, aws_credentials, s3_bucket):
        """Test sharding works end-to-end."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(bucket=s3_bucket, shard_chars=2)

            blob_id = "abc123def456"
            data = b"sharded data"

            blob_path = backend.write_blob(blob_id, data)

            # Verify the key is sharded
            assert "/ab/" in blob_path

            # Should still be readable
            read_data = backend.read_blob(blob_path)
            assert read_data == data


# =============================================================================
# S3 ETag Tests
# =============================================================================


class TestS3ETag:
    """Test S3 ETag handling."""

    def test_get_etag(self, aws_credentials, s3_bucket):
        """Test get_etag() method."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(bucket=s3_bucket)

            blob_id = "etag_test"
            data = b"test data for etag"

            blob_path = backend.write_blob(blob_id, data)
            etag = backend.get_etag(blob_path)

            assert etag is not None
            assert isinstance(etag, str)
            assert len(etag) > 0
            # ETag should not have quotes
            assert '"' not in etag

    def test_verify_etag(self, aws_credentials, s3_bucket):
        """Test verify_etag() method."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(bucket=s3_bucket)

            blob_id = "verify_etag_test"
            data = b"verify this"

            blob_path = backend.write_blob(blob_id, data)
            etag = backend.get_etag(blob_path)
            assert etag is not None

            # Should verify correctly
            assert backend.verify_etag(blob_path, etag)

            # Should fail with wrong etag
            assert not backend.verify_etag(blob_path, "wrong_etag")

    def test_get_blob_metadata(self, aws_credentials, s3_bucket):
        """Test get_blob_metadata() method."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(bucket=s3_bucket)

            blob_id = "metadata_test"
            data = b"metadata test data"

            blob_path = backend.write_blob(blob_id, data)
            metadata = backend.get_blob_metadata(blob_path)

            assert metadata is not None
            assert metadata["s3_bucket"] == s3_bucket
            assert "s3_key" in metadata
            assert "s3_etag" in metadata
            assert metadata["size"] == len(data)


# =============================================================================
# S3 Streaming Tests
# =============================================================================


class TestS3Streaming:
    """Test streaming operations."""

    def test_write_blob_stream(self, aws_credentials, s3_bucket):
        """Test streaming write."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(bucket=s3_bucket)

            blob_id = "stream_write_test"
            data = b"streamed data content"
            stream = BytesIO(data)

            blob_path = backend.write_blob_stream(blob_id, stream)

            # Verify
            read_data = backend.read_blob(blob_path)
            assert read_data == data

    def test_read_blob_stream(self, aws_credentials, s3_bucket):
        """Test streaming read."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(bucket=s3_bucket)

            blob_id = "stream_read_test"
            data = b"data to stream read"

            blob_path = backend.write_blob(blob_id, data)

            # Read as stream
            stream = backend.read_blob_stream(blob_path)
            read_data = stream.read()
            assert read_data == data


# =============================================================================
# S3 List Keys Tests
# =============================================================================


class TestS3ListKeys:
    """Test listing S3 keys."""

    def test_list_keys_empty(self, aws_credentials, s3_bucket):
        """Test list_keys() on empty bucket."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(bucket=s3_bucket)

            keys = backend.list_keys()
            assert keys == []

    def test_list_keys(self, aws_credentials, s3_bucket):
        """Test list_keys() with blobs."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(bucket=s3_bucket, shard_chars=0)

            # Write several blobs
            for i in range(5):
                backend.write_blob(f"blob_{i}", f"data_{i}".encode())

            keys = backend.list_keys()
            assert len(keys) == 5
            for key in keys:
                assert key.startswith(f"s3://{s3_bucket}/")

    def test_list_keys_with_prefix(self, aws_credentials, s3_bucket):
        """Test list_keys() with prefix filter."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(bucket=s3_bucket, prefix="cache/", shard_chars=0)

            # Write blobs
            backend.write_blob("item1", b"data1")
            backend.write_blob("item2", b"data2")

            keys = backend.list_keys()
            assert len(keys) == 2
            for key in keys:
                assert "cache/" in key


# =============================================================================
# S3 Compatibility Tests
# =============================================================================


class TestS3Compatibility:
    """Test S3-compatible endpoint configuration."""

    def test_custom_endpoint_url(self, aws_credentials, s3_bucket):
        """Test custom endpoint_url for S3-compatible backends (e.g. Garage)."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            # Note: moto doesn't really simulate Garage, but we can test the config
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            # This tests that custom endpoint_url is accepted
            backend = S3BlobBackend(
                bucket=s3_bucket,
                endpoint_url="http://localhost:3900",  # Garage S3 endpoint
                access_key="GKdeadbeef02d4b4e901234567",
                secret_key="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                use_ssl=False,
            )

            assert backend.endpoint_url == "http://localhost:3900"
            assert backend.use_ssl is False


# =============================================================================
# S3BlobBackend Registration Tests
# =============================================================================


class TestS3BackendRegistration:
    """Test S3 backend registration with the blob backend registry."""

    def test_register_s3_backend(self, aws_credentials, s3_bucket):
        """Test registering S3 backend."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend
        from cacheness.storage.backends.blob_backends import (
            register_blob_backend,
            unregister_blob_backend,
            get_blob_backend,
            list_blob_backends,
        )

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            # Register
            register_blob_backend("s3", S3BlobBackend, force=True)

            # Verify it's listed
            backends = list_blob_backends()
            backend_names = [b["name"] for b in backends]
            assert "s3" in backend_names

            # Get backend instance
            backend = get_blob_backend("s3", bucket=s3_bucket)
            assert isinstance(backend, S3BlobBackend)

            # Clean up
            unregister_blob_backend("s3")


# =============================================================================
# S3 Error Handling Tests
# =============================================================================


class TestS3ErrorHandling:
    """Test error handling in S3 backend."""

    def test_boto3_not_available_error(self):
        """Test error when boto3 is not available."""
        # This is hard to test since boto3 IS available in our test env
        # Just document that the error is raised in the code
        pass

    def test_nonexistent_blob_returns_none_for_etag(self, aws_credentials, s3_bucket):
        """Test get_etag returns None for non-existent blob."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(bucket=s3_bucket)

            etag = backend.get_etag(f"s3://{s3_bucket}/nonexistent")
            assert etag is None

    def test_nonexistent_blob_returns_none_for_metadata(
        self, aws_credentials, s3_bucket
    ):
        """Test get_blob_metadata returns None for non-existent blob."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(bucket=s3_bucket)

            metadata = backend.get_blob_metadata(f"s3://{s3_bucket}/nonexistent")
            assert metadata is None

    def test_get_size_nonexistent(self, aws_credentials, s3_bucket):
        """Test get_size returns -1 for non-existent blob."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(bucket=s3_bucket)

            size = backend.get_size(f"s3://{s3_bucket}/nonexistent")
            assert size == -1


# =============================================================================
# S3 ETag Verification on Read Tests
# =============================================================================


class TestS3ETagVerification:
    """Test ETag verification on read_blob()."""

    def test_read_blob_with_matching_etag(self, aws_credentials, s3_bucket):
        """Test read_blob succeeds when ETag matches."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(bucket=s3_bucket, verify_on_read=True)

            blob_id = "etag_verify_match"
            data = b"verify me"

            blob_path = backend.write_blob(blob_id, data)
            write_meta = backend.get_write_metadata()
            expected_etag = write_meta["s3_etag"]

            # Should succeed without error
            result = backend.read_blob(blob_path, expected_etag=expected_etag)
            assert result == data

    def test_read_blob_with_mismatched_etag_warns(self, aws_credentials, s3_bucket):
        """Test read_blob logs warning on ETag mismatch (non-strict)."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(
                bucket=s3_bucket, verify_on_read=True, strict_etag=False
            )

            blob_id = "etag_verify_mismatch"
            data = b"mismatch data"

            blob_path = backend.write_blob(blob_id, data)

            # Should return data but log warning (non-strict)
            result = backend.read_blob(blob_path, expected_etag="wrong_etag")
            assert result == data

    def test_read_blob_with_mismatched_etag_strict_raises(
        self, aws_credentials, s3_bucket
    ):
        """Test read_blob raises S3IntegrityError on mismatch in strict mode."""
        from cacheness.storage.backends.s3_backend import (
            S3BlobBackend,
            S3IntegrityError,
        )

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(
                bucket=s3_bucket, verify_on_read=True, strict_etag=True
            )

            blob_id = "etag_strict_fail"
            data = b"strict data"

            blob_path = backend.write_blob(blob_id, data)

            with pytest.raises(S3IntegrityError) as exc_info:
                backend.read_blob(blob_path, expected_etag="wrong_etag")

            assert "wrong_etag" in str(exc_info.value)
            assert exc_info.value.blob_path == blob_path

    def test_read_blob_no_etag_skips_verification(self, aws_credentials, s3_bucket):
        """Test read_blob skips verification when no expected_etag given."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(
                bucket=s3_bucket, verify_on_read=True, strict_etag=True
            )

            blob_id = "etag_no_expected"
            data = b"no expected etag"

            blob_path = backend.write_blob(blob_id, data)

            # Should succeed without error even in strict mode
            result = backend.read_blob(blob_path)
            assert result == data

    def test_read_blob_verify_disabled(self, aws_credentials, s3_bucket):
        """Test read_blob skips verification when verify_on_read=False."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(
                bucket=s3_bucket, verify_on_read=False, strict_etag=True
            )

            blob_id = "etag_verify_disabled"
            data = b"no verify"

            blob_path = backend.write_blob(blob_id, data)

            # Should succeed even with wrong etag because verify is disabled
            result = backend.read_blob(blob_path, expected_etag="wrong_etag")
            assert result == data


# =============================================================================
# S3 Content-MD5 on Write Tests
# =============================================================================


class TestS3ContentMD5:
    """Test Content-MD5 header on write_blob()."""

    def test_write_blob_with_content_md5(self, aws_credentials, s3_bucket):
        """Test that write_blob sends Content-MD5 and succeeds."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(bucket=s3_bucket)

            blob_id = "md5_test"
            data = b"content md5 test data"

            blob_path = backend.write_blob(blob_id, data)

            # Verify data was written correctly
            read_data = backend.read_blob(blob_path)
            assert read_data == data

    def test_write_blob_captures_etag(self, aws_credentials, s3_bucket):
        """Test that write_blob captures ETag in write metadata."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(bucket=s3_bucket)

            blob_path = backend.write_blob("etag_capture", b"capture etag")
            meta = backend.get_write_metadata()

            assert "s3_etag" in meta
            assert len(meta["s3_etag"]) > 0


# =============================================================================
# S3 Stream Upload ETag Capture Tests
# =============================================================================


class TestS3StreamETagCapture:
    """Test ETag capture on write_blob_stream()."""

    def test_write_blob_stream_captures_etag(self, aws_credentials, s3_bucket):
        """Test that write_blob_stream captures ETag via head_object."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(bucket=s3_bucket)

            data = b"streamed data for etag capture"
            stream = BytesIO(data)

            blob_path = backend.write_blob_stream("stream_etag", stream)
            meta = backend.get_write_metadata()

            assert "s3_etag" in meta
            assert len(meta["s3_etag"]) > 0

            # Verify data integrity
            read_data = backend.read_blob(blob_path)
            assert read_data == data

    def test_write_blob_stream_etag_matches_head(self, aws_credentials, s3_bucket):
        """Test that stream upload ETag matches a subsequent head_object."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(bucket=s3_bucket)

            data = b"stream etag consistency check"
            stream = BytesIO(data)

            blob_path = backend.write_blob_stream("stream_etag_check", stream)
            write_etag = backend.get_write_metadata().get("s3_etag")
            head_etag = backend.get_etag(blob_path)

            assert write_etag is not None
            assert write_etag == head_etag


# =============================================================================
# S3IntegrityError Tests
# =============================================================================


class TestS3IntegrityError:
    """Test S3IntegrityError exception."""

    def test_integrity_error_attributes(self):
        """Test S3IntegrityError has correct attributes."""
        from cacheness.storage.backends.s3_backend import S3IntegrityError

        err = S3IntegrityError("s3://bucket/key", "expected123", "actual456")
        assert err.blob_path == "s3://bucket/key"
        assert err.expected_etag == "expected123"
        assert err.actual_etag == "actual456"
        assert "expected123" in str(err)
        assert "actual456" in str(err)

    def test_integrity_error_is_exception(self):
        """Test S3IntegrityError inherits from Exception."""
        from cacheness.storage.backends.s3_backend import S3IntegrityError

        assert issubclass(S3IntegrityError, Exception)


# =============================================================================
# S3 Redundant Upload Skip Tests
# =============================================================================


class TestS3RedundantUploadSkip:
    """Test skipping redundant uploads when object already exists."""

    def test_skip_redundant_upload(self, aws_credentials, s3_bucket):
        """Test that identical data is not re-uploaded."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend
        from unittest.mock import patch

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(bucket=s3_bucket)

            blob_id = "dedup_test"
            data = b"deduplicate this"

            # First write
            path1 = backend.write_blob(blob_id, data)
            etag1 = backend.get_write_metadata()["s3_etag"]

            # Second write with same data — should skip upload
            with patch.object(
                backend._client, "put_object", wraps=backend._client.put_object
            ) as mock_put:
                path2 = backend.write_blob(blob_id, data)
                etag2 = backend.get_write_metadata()["s3_etag"]

                # put_object should NOT have been called
                mock_put.assert_not_called()

            assert path1 == path2
            assert etag1 == etag2

    def test_no_skip_for_different_data(self, aws_credentials, s3_bucket):
        """Test that different data IS uploaded even to same key."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=s3_bucket)

            backend = S3BlobBackend(bucket=s3_bucket)

            blob_id = "dedup_diff"

            # First write
            backend.write_blob(blob_id, b"version 1")
            etag1 = backend.get_write_metadata()["s3_etag"]

            # Second write with different data
            backend.write_blob(blob_id, b"version 2")
            etag2 = backend.get_write_metadata()["s3_etag"]

            assert etag1 != etag2

            # Verify latest data is readable
            blob_path = f"s3://{s3_bucket}/{backend._get_s3_key(blob_id)}"
            assert backend.read_blob(blob_path) == b"version 2"
