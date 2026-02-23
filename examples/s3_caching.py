#!/usr/bin/env python3
"""
S3 File Caching
================

Cache files downloaded from S3 so repeated reads are served from disk.

Prerequisites:
    uv add boto3
    AWS credentials configured (``aws configure``)

Usage:
    uv run python examples/s3_caching.py
"""

from io import BytesIO

import boto3
import pandas as pd

from cacheness import cached, cacheness, CacheConfig

# -- Dedicated cache instance for S3 downloads --------------------------------
s3_config = CacheConfig(
    cache_dir="./s3_cache",
    default_ttl="3d",
    max_cache_size="5gb",
    metadata_backend="sqlite",
)
s3_cache = cacheness(s3_config)


class S3DataManager:
    """Thin wrapper around S3 with a local disk cache."""

    def __init__(self, bucket: str, profile: str | None = None):
        session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        self.s3 = session.client("s3")
        self.bucket = bucket

    @cached(cache_instance=s3_cache, ttl="1d", key_prefix="s3_file")
    def download(self, key: str):
        """Download a single S3 object (cached for 24 h)."""
        print(f"  Downloading s3://{self.bucket}/{key} ...")
        resp = self.s3.get_object(Bucket=self.bucket, Key=key)
        content = resp["Body"].read()
        return {
            "content": content,
            "size": len(content),
            "etag": resp.get("ETag", "").strip('"'),
        }

    @cached(cache_instance=s3_cache, ttl="2d", key_prefix="s3_df")
    def read_csv(self, key: str, **kwargs):
        """Read a CSV from S3 into a DataFrame (cached for 48 h)."""
        data = self.download(key)
        return pd.read_csv(BytesIO(data["content"]), **kwargs)

    @cached(cache_instance=s3_cache, ttl="1w", key_prefix="s3_list")
    def list_files(self, prefix: str = "", ext: str | None = None):
        """List objects under *prefix* (cached for 1 week)."""
        print(f"  Listing s3://{self.bucket}/{prefix} ...")
        paginator = self.s3.get_paginator("list_objects_v2")
        files: list[dict] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if ext and not obj["Key"].endswith(ext):
                    continue
                files.append({"key": obj["Key"], "size": obj["Size"]})
        return files


if __name__ == "__main__":
    mgr = S3DataManager("my-data-bucket")

    # Download & cache a file
    blob = mgr.download("data/sample.csv")
    print(f"Downloaded {blob['size']} bytes, ETag={blob['etag']}")

    blob2 = mgr.download("data/sample.csv")  # cached
    print(f"Cached {blob2['size']} bytes")

    # List CSV files
    csvs = mgr.list_files(prefix="exports/", ext=".csv")
    print(f"\nFound {len(csvs)} CSV files")
