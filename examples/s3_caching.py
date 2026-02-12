import boto3
from botocore.exceptions import ClientError
from cacheness import cached, cacheness
import pandas as pd

# Initialize cache for S3 files
s3_cache = cacheness(
    cache_dir="./s3_cache",
    default_ttl_seconds=259200,  # 72 hours - Cache S3 files for 3 days
    max_cache_size_mb=5000,  # 5GB cache limit for large files
    metadata_config={
        "backend": "sqlite",  # Better performance for many files
        "store_cache_key_params": True,  # Track S3 paths and versions
    },
)


class S3DataManager:
    """S3 file manager with intelligent caching and ETag-based invalidation."""

    def __init__(self, bucket_name, aws_profile=None):
        self.bucket_name = bucket_name
        if aws_profile:
            self.session = boto3.Session(profile_name=aws_profile)
        else:
            self.session = boto3.Session()
        self.s3_client = self.session.client("s3")

    def _get_s3_file_metadata(self, s3_key, version_id=None):
        """Get S3 file metadata without downloading content."""
        try:
            params = {"Bucket": self.bucket_name, "Key": s3_key}
            if version_id:
                params["VersionId"] = version_id

            response = self.s3_client.head_object(**params)
            return {
                "etag": response.get("ETag", "").strip('"'),
                "last_modified": response.get("LastModified"),
                "size": response.get("ContentLength", 0),
                "content_type": response.get("ContentType", "binary/octet-stream"),
            }
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                raise FileNotFoundError(f"S3 object not found: {s3_key}")
            raise

    def _is_cache_valid(self, s3_key, version_id=None):
        """Check if cached file is still valid by comparing ETags."""
        try:
            # Get current S3 file metadata
            current_metadata = self._get_s3_file_metadata(s3_key, version_id)
            current_etag = current_metadata["etag"]

            # Check if we have a cached entry
            cache_key_params = {"s3_key": s3_key}
            if version_id:
                cache_key_params["version_id"] = version_id

            cached_entry = s3_cache.get(**cache_key_params, key_prefix="s3_file")

            if cached_entry is None:
                return False  # No cache entry exists

            # Compare ETags
            cached_etag = cached_entry.get("etag", "")
            if current_etag != cached_etag:
                print(
                    f"File {s3_key} has changed (ETag: {cached_etag} -> {current_etag})"
                )
                # Invalidate the stale cache entry
                s3_cache.invalidate(**cache_key_params, key_prefix="s3_file")
                return False

            return True  # Cache is valid

        except Exception as e:
            print(f"Error checking cache validity for {s3_key}: {e}")
            return False  # Assume invalid on error

    def download_file_with_validation(
        self, s3_key, version_id=None, force_refresh=False
    ):
        """Download file with ETag-based cache validation."""
        if not force_refresh and self._is_cache_valid(s3_key, version_id):
            print(f"Using valid cached version of {s3_key}")
            cache_key_params = {"s3_key": s3_key}
            if version_id:
                cache_key_params["version_id"] = version_id
            return s3_cache.get(**cache_key_params, key_prefix="s3_file")

        # Download fresh copy
        return self.download_file(s3_key, version_id)

    @cached(cache=s3_cache, ttl_seconds=86400, key_prefix="s3_file")  # 24 hours
    def download_file(self, s3_key, version_id=None):
        """Download file from S3 with caching."""
        try:
            print(f"Downloading {s3_key} from S3...")

            params = {"Bucket": self.bucket_name, "Key": s3_key}
            if version_id:
                params["VersionId"] = version_id

            response = self.s3_client.get_object(**params)
            content = response["Body"].read()

            # Return content with metadata including ETag
            return {
                "content": content,
                "content_type": response.get("ContentType", "binary/octet-stream"),
                "last_modified": response.get("LastModified"),
                "size": response.get("ContentLength", len(content)),
                "etag": response.get("ETag", "").strip('"'),
            }

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise FileNotFoundError(f"S3 object not found: {s3_key}")
            raise

    def load_dataframe_with_validation(
        self, s3_key, file_format="csv", force_refresh=False, **read_kwargs
    ):
        """Load DataFrame from S3 with ETag-based cache validation."""
        print(f"Loading DataFrame from S3: {s3_key}")

        # Download with validation
        file_data = self.download_file_with_validation(
            s3_key, force_refresh=force_refresh
        )
        content = file_data["content"]

        # Parse based on format
        if file_format.lower() == "csv":
            return pd.read_csv(BytesIO(content), **read_kwargs)
        elif file_format.lower() == "parquet":
            return pd.read_parquet(BytesIO(content), **read_kwargs)
        elif file_format.lower() in ["xlsx", "excel"]:
            return pd.read_excel(BytesIO(content), **read_kwargs)
        elif file_format.lower() == "json":
            return pd.read_json(BytesIO(content), **read_kwargs)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

    @cached(cache=s3_cache, ttl_seconds=172800, key_prefix="s3_dataframe")  # 48 hours
    def load_dataframe(self, s3_key, file_format="csv", **read_kwargs):
        """Load DataFrame from S3 with caching."""
        print(f"Loading DataFrame from S3: {s3_key}")

        # Download the file
        file_data = self.download_file(s3_key)
        content = file_data["content"]

        # Parse based on format
        if file_format.lower() == "csv":
            return pd.read_csv(BytesIO(content), **read_kwargs)
        elif file_format.lower() == "parquet":
            return pd.read_parquet(BytesIO(content), **read_kwargs)
        elif file_format.lower() in ["xlsx", "excel"]:
            return pd.read_excel(BytesIO(content), **read_kwargs)
        elif file_format.lower() == "json":
            return pd.read_json(BytesIO(content), **read_kwargs)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

    @cached(
        cache=s3_cache, ttl_seconds=604800, key_prefix="s3_list"
    )  # 168 hours - 1 week
    def list_files(self, prefix="", max_keys=1000, file_extension=None):
        """List S3 files with caching."""
        print(f"Listing S3 files with prefix: {prefix}")

        paginator = self.s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=self.bucket_name, Prefix=prefix, MaxKeys=max_keys
        )

        files = []
        for page in pages:
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]
                    if file_extension and not key.endswith(file_extension):
                        continue

                    files.append(
                        {
                            "key": key,
                            "size": obj["Size"],
                            "last_modified": obj["LastModified"],
                            "etag": obj["ETag"].strip('"'),
                        }
                    )

        return files

    def get_cache_stats(self):
        """Get caching statistics."""
        stats = s3_cache.get_stats()
        print("S3 Cache Statistics:")
        print(f"  Total files cached: {stats['total_entries']}")
        print(f"  Cache size: {stats['total_size_mb']:.1f} MB")
        print(f"  Hit rate: {stats['hit_rate']:.1%}")
        return stats


# Usage
from io import BytesIO

# Initialize S3 manager
s3_manager = S3DataManager("my-data-bucket", aws_profile="default")

# Basic download and caching
file_data = s3_manager.download_file("data/customer_data.csv")
print(f"Downloaded {file_data['size']} bytes, ETag: {file_data['etag']}")

# Smart caching with ETag validation
# First call downloads the file
df = s3_manager.load_dataframe_with_validation(
    "analytics/sales_data.parquet", file_format="parquet"
)
print(f"Loaded DataFrame: {df.shape}")

# Second call checks ETag first - uses cache if file unchanged
df_cached = s3_manager.load_dataframe_with_validation(
    "analytics/sales_data.parquet", file_format="parquet"
)

# If file was updated in S3, cache is automatically invalidated and fresh copy downloaded
df_fresh = s3_manager.load_dataframe_with_validation(
    "analytics/sales_data.parquet", file_format="parquet"
)

# Force refresh (skip cache validation)
df_forced = s3_manager.load_dataframe_with_validation(
    "analytics/sales_data.parquet", file_format="parquet", force_refresh=True
)

# Traditional caching (no ETag checking) - faster but may be stale
df_traditional = s3_manager.load_dataframe(
    "analytics/sales_data.parquet", file_format="parquet"
)

# List files with caching
csv_files = s3_manager.list_files(prefix="exports/", file_extension=".csv")
print(f"Found {len(csv_files)} CSV files")

# Check file validity manually
is_valid = s3_manager._is_cache_valid("data/customer_data.csv")
print(f"Cache is valid: {is_valid}")

# Cache management
s3_manager.get_cache_stats()

# Clear specific cached files
s3_cache.invalidate(s3_key="data/customer_data.csv")

# Or clear all S3 cache
# s3_cache.clear_all()
