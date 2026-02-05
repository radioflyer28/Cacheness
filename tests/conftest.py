"""
Fixtures for PostgreSQL and S3 integration testing.

These fixtures automatically connect to Docker containers started via docker-compose.
"""

import os
from pathlib import Path
import boto3
import psycopg
import pytest
from sqlalchemy import create_engine, text
from moto import mock_aws


# ==================== Environment Configuration ====================

def get_postgres_url() -> str:
    """Get PostgreSQL connection URL from environment."""
    user = os.getenv("POSTGRES_USER", "cacheness")
    password = os.getenv("POSTGRES_PASSWORD", "cacheness_dev_password")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "cacheness_test")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"


def get_s3_config() -> dict:
    """Get S3/MinIO configuration from environment."""
    return {
        "endpoint_url": os.getenv("S3_ENDPOINT_URL", "http://localhost:9000"),
        "access_key": os.getenv("S3_ACCESS_KEY", "minioadmin"),
        "secret_key": os.getenv("S3_SECRET_KEY", "minioadmin"),
        "bucket": os.getenv("S3_BUCKET", "cache-bucket"),
        "region": os.getenv("S3_REGION", "us-east-1"),
    }


# ==================== PostgreSQL Fixtures ====================

@pytest.fixture(scope="session")
def postgres_available() -> bool:
    """Check if PostgreSQL is available."""
    try:
        with psycopg.connect(
            dbname=os.getenv("POSTGRES_DB", "cacheness_test"),
            user=os.getenv("POSTGRES_USER", "cacheness"),
            password=os.getenv("POSTGRES_PASSWORD", "cacheness_dev_password"),
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
        ):
            return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def postgres_engine(postgres_available):
    """Create SQLAlchemy engine for PostgreSQL."""
    if not postgres_available:
        pytest.skip("PostgreSQL not available")
    engine = create_engine(get_postgres_url())
    yield engine
    engine.dispose()


@pytest.fixture
def postgres_connection(postgres_engine):
    """Provide a PostgreSQL connection for a test."""
    connection = postgres_engine.connect()
    yield connection
    connection.close()


@pytest.fixture
def postgres_clean_db(postgres_connection):
    """Provide a clean PostgreSQL database for a test."""
    # Create test schema
    postgres_connection.execute(text("CREATE SCHEMA IF NOT EXISTS cache"))
    postgres_connection.commit()
    
    yield postgres_connection
    
    # Cleanup
    postgres_connection.execute(text("DROP SCHEMA IF EXISTS cache CASCADE"))
    postgres_connection.commit()


# ==================== S3/MinIO Fixtures ====================

@pytest.fixture(scope="session")
def s3_available() -> bool:
    """Check if MinIO/S3 is available."""
    try:
        config = get_s3_config()
        client = boto3.client(
            "s3",
            endpoint_url=config["endpoint_url"],
            aws_access_key_id=config["access_key"],
            aws_secret_access_key=config["secret_key"],
            region_name=config["region"],
        )
        client.head_bucket(Bucket=config["bucket"])
        return True
    except Exception:
        return False


@pytest.fixture
def s3_client(s3_available):
    """Provide an S3 client for MinIO testing."""
    if not s3_available:
        pytest.skip("MinIO/S3 not available")
    
    config = get_s3_config()
    client = boto3.client(
        "s3",
        endpoint_url=config["endpoint_url"],
        aws_access_key_id=config["access_key"],
        aws_secret_access_key=config["secret_key"],
        region_name=config["region"],
    )
    yield client


@pytest.fixture
def s3_bucket(s3_client):
    """Provide a clean S3 bucket for a test."""
    config = get_s3_config()
    bucket = config["bucket"]
    
    # Clear bucket before test
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        if "Contents" in page:
            for obj in page["Contents"]:
                s3_client.delete_object(Bucket=bucket, Key=obj["Key"])
    
    yield bucket
    
    # Cleanup after test
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        if "Contents" in page:
            for obj in page["Contents"]:
                s3_client.delete_object(Bucket=bucket, Key=obj["Key"])


# ==================== Mocked S3 Fixtures ====================

@pytest.fixture
def mock_s3_client():
    """Provide a mocked S3 client (no container needed)."""
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="test-bucket")
        yield client


@pytest.fixture
def mock_s3_bucket(mock_s3_client):
    """Provide a mocked S3 bucket for a test."""
    yield "test-bucket"


# ==================== Integration Fixtures ====================

@pytest.fixture
def cacheness_postgres_config(postgres_available) -> dict:
    """Provide Cacheness configuration for PostgreSQL backend."""
    if not postgres_available:
        pytest.skip("PostgreSQL not available")
    
    return {
        "metadata_backend": "postgresql",
        "metadata_config": {
            "connection_string": get_postgres_url(),
        },
    }


@pytest.fixture
def cacheness_s3_config(s3_available) -> dict:
    """Provide Cacheness configuration for S3 backend."""
    if not s3_available:
        pytest.skip("S3 not available")
    
    config = get_s3_config()
    return {
        "blob_backend": "s3",
        "blob_config": {
            "bucket": config["bucket"],
            "endpoint_url": config["endpoint_url"],
            "aws_access_key_id": config["access_key"],
            "aws_secret_access_key": config["secret_key"],
            "region_name": config["region"],
        },
    }


@pytest.fixture
def cacheness_postgres_s3_config(postgres_available, s3_available) -> dict:
    """Provide Cacheness configuration for PostgreSQL + S3 integration."""
    if not postgres_available:
        pytest.skip("PostgreSQL not available")
    if not s3_available:
        pytest.skip("S3 not available")
    
    config = get_s3_config()
    return {
        "metadata_backend": "postgresql",
        "metadata_config": {
            "connection_string": get_postgres_url(),
        },
        "blob_backend": "s3",
        "blob_config": {
            "bucket": config["bucket"],
            "endpoint_url": config["endpoint_url"],
            "aws_access_key_id": config["access_key"],
            "aws_secret_access_key": config["secret_key"],
            "region_name": config["region"],
        },
    }


# ==================== Config File Fixtures ====================

@pytest.fixture
def config_dir() -> Path:
    """Get the config directory path."""
    workspace_root = Path(__file__).parent.parent
    return workspace_root / "config"


@pytest.fixture
def cacheness_config_from_yaml(config_dir):
    """Load Cacheness configuration from test_config.yaml."""
    try:
        from cacheness.config import load_config_from_yaml
    except ImportError:
        pytest.skip("cacheness not installed")
    
    config_file = config_dir / "test_config.yaml"
    if not config_file.exists():
        pytest.skip(f"Config file not found: {config_file}")
    
    return load_config_from_yaml(str(config_file))


@pytest.fixture
def cacheness_config_from_json(config_dir):
    """Load Cacheness configuration from test_config.json."""
    try:
        from cacheness.config import load_config_from_json
    except ImportError:
        pytest.skip("cacheness not installed")
    
    config_file = config_dir / "test_config.json"
    if not config_file.exists():
        pytest.skip(f"Config file not found: {config_file}")
    
    return load_config_from_json(str(config_file))


@pytest.fixture
def cacheness_cache_from_yaml(cacheness_config_from_yaml, postgres_available, s3_available):
    """Provide a Cache instance configured from test_config.yaml."""
    if not postgres_available:
        pytest.skip("PostgreSQL not available")
    if not s3_available:
        pytest.skip("S3 not available")
    
    try:
        from cacheness import cacheness as create_cache
    except ImportError:
        pytest.skip("cacheness not installed")
    
    cache = create_cache(config=cacheness_config_from_yaml)
    cache.clear_all()  # Clear any cached data from previous test runs
    yield cache
    cache.clear_all()  # Cleanup after test


@pytest.fixture
def cacheness_cache_from_json(cacheness_config_from_json, postgres_available, s3_available):
    """Provide a Cache instance configured from test_config.json."""
    if not postgres_available:
        pytest.skip("PostgreSQL not available")
    if not s3_available:
        pytest.skip("S3 not available")
    
    try:
        from cacheness import cacheness as create_cache
    except ImportError:
        pytest.skip("cacheness not installed")
    
    cache = create_cache(config=cacheness_config_from_json)
    cache.clear_all()  # Clear any cached data from previous test runs
    yield cache
    cache.clear_all()  # Cleanup after test


@pytest.fixture
def cacheness_local_sqlite_fs(config_dir):
    """Provide a Cache instance using local SQLite + Filesystem config (no containers)."""
    try:
        from cacheness import cacheness as create_cache
        from cacheness.config import load_config_from_yaml
    except ImportError:
        pytest.skip("cacheness not installed")
    
    config_file = config_dir / "local_sqlite_fs.yaml"
    if not config_file.exists():
        pytest.skip(f"Config file not found: {config_file}")
    
    config = load_config_from_yaml(str(config_file))
    cache = create_cache(config=config)
    yield cache
    # Cleanup could go here

