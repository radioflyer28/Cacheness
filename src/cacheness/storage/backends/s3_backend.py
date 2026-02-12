"""
S3 Blob Storage Backend
=======================

S3-compatible blob storage backend for distributed caching.
Supports Amazon S3, MinIO, and other S3-compatible services.

Requirements:
    pip install cacheness[s3]
    # or
    pip install boto3

Usage:
    from cacheness.storage.backends.s3_backend import S3BlobBackend
    from cacheness import register_blob_backend

    # Register S3 backend
    register_blob_backend("s3", S3BlobBackend)

    # Use with Amazon S3
    backend = get_blob_backend(
        "s3",
        bucket="my-cache-bucket",
        region="us-east-1",
    )

    # Use with MinIO
    backend = get_blob_backend(
        "s3",
        bucket="local-cache",
        endpoint_url="http://minio.internal:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        use_ssl=False,
    )
"""

import logging
from typing import BinaryIO, Dict, List, Optional, Any

from .blob_backends import BlobBackend

logger = logging.getLogger(__name__)


# Check for boto3 availability
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None
    ClientError = None
    NoCredentialsError = None


class S3BlobBackend(BlobBackend):
    """
    S3-compatible blob storage backend.

    Stores blobs in Amazon S3 or S3-compatible services (MinIO, etc.).
    Supports Git-style directory sharding for better performance with
    large numbers of objects.

    Attributes:
        bucket: S3 bucket name
        prefix: Optional prefix (folder) for all objects
        region: AWS region (default: us-east-1)
        endpoint_url: Custom endpoint URL for S3-compatible services (MinIO)
        shard_chars: Number of leading chars for directory sharding (default: 2)

    Example:
        # Amazon S3
        backend = S3BlobBackend(
            bucket="my-cache-bucket",
            prefix="cache/v1/",
            region="us-east-1",
        )

        # MinIO
        backend = S3BlobBackend(
            bucket="local-cache",
            endpoint_url="http://minio:9000",
            access_key="minioadmin",
            secret_key="minioadmin",
            use_ssl=False,
        )
    """

    # Multipart upload threshold (5MB minimum for S3)
    MULTIPART_THRESHOLD = 5 * 1024 * 1024  # 5MB
    MULTIPART_CHUNKSIZE = 8 * 1024 * 1024  # 8MB chunks

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: str = "us-east-1",
        endpoint_url: Optional[str] = None,
        use_ssl: bool = True,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        shard_chars: int = 2,
        **kwargs,
    ):
        """
        Initialize S3 blob backend.

        Args:
            bucket: S3 bucket name
            prefix: Optional key prefix (e.g., "cache/v1/")
            region: AWS region (default: us-east-1)
            endpoint_url: Custom endpoint for S3-compatible services (MinIO)
            use_ssl: Use HTTPS (default: True)
            access_key: AWS access key (optional, falls back to credential chain)
            secret_key: AWS secret key (optional, falls back to credential chain)
            shard_chars: Number of leading chars for directory sharding (default: 2)
            **kwargs: Additional boto3 client options
        """
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for S3 backend. "
                "Install with: pip install cacheness[s3] or pip install boto3"
            )

        self.bucket = bucket
        self.prefix = (
            prefix.rstrip("/") + "/" if prefix and not prefix.endswith("/") else prefix
        )
        self.region = region
        self.endpoint_url = endpoint_url
        self.use_ssl = use_ssl
        self.shard_chars = shard_chars

        # Build client config
        client_kwargs = {
            "region_name": region,
            "use_ssl": use_ssl,
        }

        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url

        # Only add credentials if explicitly provided
        # Otherwise boto3 will use its credential chain
        if access_key and secret_key:
            client_kwargs["aws_access_key_id"] = access_key
            client_kwargs["aws_secret_access_key"] = secret_key

        # Create S3 client
        self._client = boto3.client("s3", **client_kwargs)

        logger.debug(
            f"S3BlobBackend initialized: bucket={bucket}, prefix={self.prefix}, "
            f"region={region}, endpoint={endpoint_url}, shard_chars={shard_chars}"
        )

    def _get_s3_key(self, blob_id: str) -> str:
        """
        Get the S3 object key for a blob ID with directory sharding.

        Args:
            blob_id: Unique blob identifier

        Returns:
            Full S3 key including prefix and shard directory

        Example (shard_chars=2, prefix="cache/"):
            blob_id = "abc123def456"
            returns: "cache/ab/abc123def456"
        """
        # Apply Git-style sharding
        if self.shard_chars > 0 and len(blob_id) >= self.shard_chars:
            shard_dir = blob_id[: self.shard_chars]
            return f"{self.prefix}{shard_dir}/{blob_id}"

        return f"{self.prefix}{blob_id}"

    def write_blob(self, blob_id: str, data: bytes) -> str:
        """
        Write blob to S3.

        Args:
            blob_id: Unique identifier for the blob
            data: Raw bytes to store

        Returns:
            S3 URI in format s3://bucket/key

        Raises:
            Exception: If upload fails
        """
        s3_key = self._get_s3_key(blob_id)

        try:
            response = self._client.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=data,
            )

            etag = response.get("ETag", "").strip('"')

            logger.debug(
                f"Wrote blob {blob_id} ({len(data)} bytes) to s3://{self.bucket}/{s3_key} "
                f"(ETag: {etag})"
            )

            return f"s3://{self.bucket}/{s3_key}"

        except ClientError as e:
            logger.error(f"Failed to write blob {blob_id} to S3: {e}")
            raise

    def read_blob(self, blob_path: str) -> bytes:
        """
        Read blob from S3.

        Args:
            blob_path: S3 URI (s3://bucket/key) or just the key

        Returns:
            Raw bytes of the blob

        Raises:
            FileNotFoundError: If blob doesn't exist
        """
        s3_key = self._parse_blob_path(blob_path)

        try:
            response = self._client.get_object(
                Bucket=self.bucket,
                Key=s3_key,
            )

            data = response["Body"].read()
            logger.debug(
                f"Read blob from s3://{self.bucket}/{s3_key} ({len(data)} bytes)"
            )
            return data

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("NoSuchKey", "404"):
                raise FileNotFoundError(f"Blob not found: {blob_path}")
            logger.error(f"Failed to read blob from S3: {e}")
            raise

    def delete_blob(self, blob_path: str) -> bool:
        """
        Delete blob from S3.

        Args:
            blob_path: S3 URI (s3://bucket/key) or just the key

        Returns:
            True if deleted (S3 delete always returns success even if key doesn't exist)
        """
        s3_key = self._parse_blob_path(blob_path)

        try:
            self._client.delete_object(
                Bucket=self.bucket,
                Key=s3_key,
            )

            logger.debug(f"Deleted blob from s3://{self.bucket}/{s3_key}")
            return True

        except ClientError as e:
            logger.error(f"Failed to delete blob from S3: {e}")
            return False

    def exists(self, blob_path: str) -> bool:
        """
        Check if blob exists in S3.

        Args:
            blob_path: S3 URI (s3://bucket/key) or just the key

        Returns:
            True if blob exists
        """
        s3_key = self._parse_blob_path(blob_path)

        try:
            self._client.head_object(
                Bucket=self.bucket,
                Key=s3_key,
            )
            return True

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("NoSuchKey", "404"):
                return False
            # Re-raise unexpected errors
            raise

    def write_blob_stream(self, blob_id: str, stream: BinaryIO) -> str:
        """
        Write blob from stream to S3 (supports multipart upload for large objects).

        Args:
            blob_id: Unique identifier for the blob
            stream: File-like object with read() method

        Returns:
            S3 URI in format s3://bucket/key
        """
        s3_key = self._get_s3_key(blob_id)

        try:
            # Use upload_fileobj which handles multipart automatically
            self._client.upload_fileobj(
                stream,
                self.bucket,
                s3_key,
            )

            logger.debug(f"Wrote blob stream {blob_id} to s3://{self.bucket}/{s3_key}")
            return f"s3://{self.bucket}/{s3_key}"

        except ClientError as e:
            logger.error(f"Failed to upload stream {blob_id} to S3: {e}")
            raise

    def read_blob_stream(self, blob_path: str) -> BinaryIO:
        """
        Read blob as stream from S3.

        Args:
            blob_path: S3 URI (s3://bucket/key) or just the key

        Returns:
            File-like object with read() method

        Raises:
            FileNotFoundError: If blob doesn't exist
        """
        s3_key = self._parse_blob_path(blob_path)

        try:
            response = self._client.get_object(
                Bucket=self.bucket,
                Key=s3_key,
            )

            # Return the streaming body wrapped in BytesIO for consistent interface
            return response["Body"]

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("NoSuchKey", "404"):
                raise FileNotFoundError(f"Blob not found: {blob_path}")
            raise

    def get_size(self, blob_path: str) -> int:
        """
        Get blob size using HEAD request (efficient, doesn't download data).

        Args:
            blob_path: S3 URI (s3://bucket/key) or just the key

        Returns:
            Size in bytes, or -1 if not found
        """
        s3_key = self._parse_blob_path(blob_path)

        try:
            response = self._client.head_object(
                Bucket=self.bucket,
                Key=s3_key,
            )
            return response.get("ContentLength", -1)

        except ClientError:
            return -1

    def get_etag(self, blob_path: str) -> Optional[str]:
        """
        Get S3 ETag for a blob (HEAD request).

        The ETag is S3's content hash (MD5 for single uploads, composite for multipart).

        Args:
            blob_path: S3 URI (s3://bucket/key) or just the key

        Returns:
            ETag string (without quotes), or None if not found
        """
        s3_key = self._parse_blob_path(blob_path)

        try:
            response = self._client.head_object(
                Bucket=self.bucket,
                Key=s3_key,
            )
            etag = response.get("ETag", "")
            return etag.strip('"') if etag else None

        except ClientError:
            return None

    def verify_etag(self, blob_path: str, expected_etag: str) -> bool:
        """
        Verify blob integrity via ETag comparison.

        Args:
            blob_path: S3 URI or key
            expected_etag: Expected ETag value

        Returns:
            True if ETag matches
        """
        actual_etag = self.get_etag(blob_path)
        if actual_etag is None:
            return False
        return actual_etag == expected_etag.strip('"')

    def get_blob_metadata(self, blob_path: str) -> Optional[Dict[str, Any]]:
        """
        Get full metadata for a blob.

        Returns:
            Dictionary with s3_key, s3_bucket, s3_etag, size, last_modified
            or None if blob doesn't exist
        """
        s3_key = self._parse_blob_path(blob_path)

        try:
            response = self._client.head_object(
                Bucket=self.bucket,
                Key=s3_key,
            )

            return {
                "s3_bucket": self.bucket,
                "s3_key": s3_key,
                "s3_etag": response.get("ETag", "").strip('"'),
                "size": response.get("ContentLength", 0),
                "last_modified": response.get("LastModified"),
                "content_type": response.get("ContentType", "application/octet-stream"),
            }

        except ClientError:
            return None

    def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """
        List blob keys in the bucket.

        Args:
            prefix: Optional additional prefix to filter by

        Returns:
            List of S3 URIs
        """
        search_prefix = self.prefix
        if prefix:
            search_prefix = f"{self.prefix}{prefix}"

        keys = []
        paginator = self._client.get_paginator("list_objects_v2")

        try:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=search_prefix):
                for obj in page.get("Contents", []):
                    keys.append(f"s3://{self.bucket}/{obj['Key']}")

            return keys

        except ClientError as e:
            logger.error(f"Failed to list objects in S3: {e}")
            return []

    def _parse_blob_path(self, blob_path: str) -> str:
        """
        Parse blob path to extract S3 key.

        Handles both s3://bucket/key URIs and plain keys.

        Args:
            blob_path: S3 URI or key

        Returns:
            S3 key (without s3://bucket/ prefix)
        """
        if blob_path.startswith("s3://"):
            # Parse s3://bucket/key format
            path = blob_path[5:]  # Remove "s3://"
            if "/" in path:
                bucket, key = path.split("/", 1)
                if bucket != self.bucket:
                    logger.warning(
                        f"Blob path bucket '{bucket}' doesn't match configured bucket '{self.bucket}'"
                    )
                return key
            return ""

        # Assume it's already a key
        return blob_path

    def close(self) -> None:
        """Close S3 client (boto3 handles connection pooling automatically)."""
        # boto3 clients manage their own connection pools
        pass


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "S3BlobBackend",
    "BOTO3_AVAILABLE",
]
