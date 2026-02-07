# Local Test Environment Setup: PostgreSQL + S3/MinIO

This guide explains how to set up a local test environment for testing the PostgreSQL metadata backend and S3 blob backend integration.

## Quick Start: Docker Compose

### Prerequisites

- **Docker** and **Docker Compose** installed
- For Windows: WSL 2 recommended (though native Windows works)

### Start Services

```bash
# Start PostgreSQL and MinIO
docker-compose up -d

# Verify services are healthy
docker-compose ps

# Expected output:
# NAME                      COMMAND                 SERVICE             STATUS
# cacheness-postgres        postgres -c fsync...    postgres            Up (healthy)
# cacheness-minio           minio server /data      minio               Up (healthy)
# cacheness-minio-init      /bin/sh -c /usr/b...    minio-init          Exited 0
```

### Connection Details

**PostgreSQL:**
- Host: `localhost`
- Port: `5432`
- Database: `cacheness_test`
- User: `cacheness`
- Password: `cacheness_dev_password` ⚠️ **(dev only)**
- Connection string: `postgresql://cacheness:cacheness_dev_password@localhost:5432/cacheness_test`

**MinIO (S3-compatible):**
- S3 API: `http://localhost:9000`
- Console: `http://localhost:9001`
- Access Key: `minioadmin` ⚠️ **(dev only)**
- Secret Key: `minioadmin` ⚠️ **(dev only)**
- Buckets created: `cache-bucket`, `test-bucket`

> **🔒 Security Note:** These credentials are hardcoded for local testing only and are safe because services only listen on localhost. Never use these credentials in production. See [config/README.md](../config/README.md) for production security guidance.

### Stop Services

```bash
docker-compose down

# With volume cleanup (removes persistent data)
docker-compose down -v
```

---

## Configuration Files

The test environment includes pre-configured YAML and JSON files in the [`config/`](../config/) directory:

### Available Configurations

1. **`config/test_config.yaml`** - PostgreSQL + MinIO (requires Docker containers)
2. **`config/test_config.json`** - Same as above in JSON format
3. **`config/local_sqlite_fs.yaml`** - SQLite + Filesystem (no containers needed)

### Using Configuration Files

Load configuration in your tests or scripts:

```python
from cacheness import Cache
from cacheness.config import CacheConfig

# Load from YAML
config = CacheConfig.from_yaml("config/test_config.yaml")
cache = Cache(config=config)

# Load from JSON
config = CacheConfig.from_json("config/test_config.json")
cache = Cache(config=config)

# Use in tests with fixtures
def test_with_config(cacheness_cache_from_yaml):
    """Test using YAML config fixture."""
    cacheness_cache_from_yaml.put({"data": "test"}, key="mykey")
    assert cacheness_cache_from_yaml.get(key="mykey") == {"data": "test"}
```

### Configuration Features

The config files include:
- **Signing key** for cache entry verification
- **Backend settings** (PostgreSQL + S3 or SQLite + Filesystem)
- **Security options** (`delete_invalid_signatures`)
- **Compression settings** (`compress`, `compress_level`)
- **Handler options** (`enable_dill_fallback`, `tensorflow_enabled`)
- **Sharding configuration** (`shard_chars`)

### Customizing Configuration

Copy and modify for your needs:

```bash
# Create custom config
cp config/test_config.yaml config/my_config.yaml

# Edit with your settings
# - Change signing_key
# - Adjust connection strings
# - Modify performance settings
```

Use environment variables for sensitive data:

```yaml
# In config file
signing_key: "${CACHE_SIGNING_KEY}"
metadata_config:
  connection_string: "${POSTGRES_CONNECTION_STRING}"
```

See [config/README.md](../config/README.md) for full documentation.

---

## Running Integration Tests

### Option 1: Using Docker Compose Directly

```bash
# Ensure containers are running
docker-compose up -d

# Run all integration tests
pytest tests/test_postgresql_integration.py -v

# Run S3 integration tests
pytest tests/test_s3_integration.py -v

# Run PostgreSQL + S3 integration tests
pytest tests/test_postgres_s3_integration.py -v

# Run all tests (unit + integration)
pytest tests/ -v
```

### Option 2: Using Configuration Files

```bash
# Test with YAML config
pytest tests/ -v --config=config/test_config.yaml

# Or use fixture in your tests
def test_with_yaml_config(cacheness_cache_from_yaml):
    # Cache is pre-configured from test_config.yaml
    pass
```

### Option 3: Using Environment Variables

If your containers are running on different hosts/ports, override via environment:

```bash
# Custom PostgreSQL
export POSTGRES_HOST=my-postgres-host
export POSTGRES_PORT=5432
export POSTGRES_USER=cacheness
export POSTGRES_PASSWORD=mypassword
export POSTGRES_DB=cacheness_test

# Custom MinIO/S3
export S3_ENDPOINT_URL=http://my-minio-host:9000
export S3_ACCESS_KEY=minioadmin
export S3_SECRET_KEY=minioadmin
export S3_BUCKET=cache-bucket
export S3_REGION=us-east-1

pytest tests/ -v
```

### Option 4: Skip Integration Tests

To run only unit tests without containers:

```bash
# Skip any tests requiring PostgreSQL or S3
pytest tests/ -v --ignore=tests/test_postgresql_integration.py --ignore=tests/test_s3_integration.py --ignore=tests/test_postgres_s3_integration.py
```

---

## DevContainer Setup (Optional)

For full IDE integration with VS Code:

### Prerequisites

- **VS Code** with Remote - Containers extension
- **Docker** and **Docker Compose**

### Start DevContainer

1. Open the workspace in VS Code
2. Click **Remote-Containers: Reopen in Container** (command palette: `Ctrl+Shift+P`)
3. VS Code will build and start the container
4. Automatically installs dependencies with `pip install -e '.[dev,s3,postgresql,cloud]'`

### Inside DevContainer

The container provides:
- Python 3.11 with `venv`
- All development dependencies
- Docker access (can run docker-compose from inside)
- Automatic port forwarding for PostgreSQL (5432), MinIO S3 API (9000), MinIO Console (9001)

```bash
# From inside the container, start services
docker-compose up -d

# Run tests
pytest tests/ -v

# Or just the integration tests
pytest tests/test_postgres_s3_integration.py -v
```

---

## Fixture Usage in Tests

The test fixtures are defined in [`conftest_integration.py`](../tests/conftest_integration.py). Here's how to use them:

### Configuration File Fixtures

```python
def test_with_yaml_config(cacheness_cache_from_yaml):
    """Test using pre-configured cache from test_config.yaml."""
    cacheness_cache_from_yaml.put({"data": "value"}, key="test")
    result = cacheness_cache_from_yaml.get(key="test")
    assert result == {"data": "value"}


def test_with_json_config(cacheness_cache_from_json):
    """Test using pre-configured cache from test_config.json."""
    # Cache is already configured with signing key, backends, etc.
    pass


def test_local_dev(cacheness_local_sqlite_fs):
    """Test using local SQLite+Filesystem (no containers)."""
    # Fast local testing without Docker
    cacheness_local_sqlite_fs.put({"x": 1}, key="local")
    assert cacheness_local_sqlite_fs.get(key="local") == {"x": 1}


def test_load_config_manually(config_dir):
    """Test loading config files directly."""
    from cacheness.config import CacheConfig
    
    config_file = config_dir / "test_config.yaml"
    config = CacheConfig.from_yaml(str(config_file))
    
    # Verify config loaded correctly
    assert config.signing_key is not None
    assert config.metadata_backend == "postgresql"
    assert config.blob_backend == "s3"
```

### PostgreSQL Fixtures

```python
def test_postgres_metadata(postgres_connection):
    """Test that requires a PostgreSQL connection."""
    result = postgres_connection.execute(text("SELECT 1"))
    assert result.fetchone()[0] == 1


def test_postgres_clean_db(postgres_clean_db):
    """Test with a clean PostgreSQL schema for isolation."""
    # Database is clean and ready for use
    # Automatically cleaned up after test
    pass
```

### S3/MinIO Fixtures

```python
def test_minio_upload(s3_client, s3_bucket):
    """Test S3 operations against MinIO."""
    s3_client.put_object(
        Bucket=s3_bucket,
        Key="test-file.txt",
        Body=b"test content",
    )
    
    response = s3_client.get_object(Bucket=s3_bucket, Key="test-file.txt")
    assert response["Body"].read() == b"test content"


def test_mocked_s3(mock_s3_client, mock_s3_bucket):
    """Test S3 operations with mocked S3 (no container needed)."""
    mock_s3_client.put_object(
        Bucket=mock_s3_bucket,
        Key="test-file.txt",
        Body=b"test content",
    )
    # Works exactly like real S3 but in-memory
```

### Integration Fixtures

```python
def test_cacheness_postgres_s3(cacheness_postgres_s3_config):
    """Test Cacheness with PostgreSQL metadata + S3 blobs."""
    from cacheness import Cache
    
    cache = Cache(**cacheness_postgres_s3_config)
    
    @cache
    def expensive_function(x):
        return x * 2
    
    result = expensive_function(5)
    assert result == 10
    
    # Cache hit on second call
    result = expensive_function(5)
    assert result == 10
```

---

## Troubleshooting

### PostgreSQL Connection Refused

```bash
# Check if container is running
docker-compose ps

# If not running, start it
docker-compose up -d postgres

# Check logs
docker-compose logs postgres

# Verify connectivity
psql -h localhost -U cacheness -d cacheness_test -W
```

### MinIO Not Responding

```bash
# Check if container is running
docker-compose ps

# If not running, start it
docker-compose up -d minio minio-init

# Check logs
docker-compose logs minio

# Verify connectivity
aws s3 ls --endpoint-url http://localhost:9000
```

### Port Already in Use

If port 5432 or 9000 is already in use:

```bash
# Option 1: Stop the conflicting service
# Find what's using port 5432 on Windows:
netstat -ano | findstr :5432
# Kill the process by PID if needed

# Option 2: Modify docker-compose.yml to use different ports
# Change "5432:5432" to "5433:5432" etc.
```

### Slow/Hanging Tests

MinIO and PostgreSQL need time to become healthy. The fixtures check this automatically:

```python
# This will skip if services aren't ready
def test_needs_postgres(postgres_available):
    if not postgres_available:
        pytest.skip("PostgreSQL not available")
```

To wait for services manually:

```bash
# Option 1: Wait for healthchecks to pass
docker-compose up -d && sleep 10 && pytest tests/ -v

# Option 2: Explicit wait
docker-compose up -d
docker wait cacheness-minio-init  # Wait for bucket creation
pytest tests/ -v
```

---

## Cleanup

### Remove Containers Only (Keep Data)

```bash
docker-compose stop
```

### Remove Containers and Data

```bash
docker-compose down -v
```

### Remove Everything (Including Docker Images)

```bash
docker-compose down -v --rmi all
```

---

## Advanced: Custom Docker Compose Override

For persistent development with customizations, create `docker-compose.override.yml`:

```yaml
version: '3.8'

services:
  postgres:
    environment:
      # Custom password
      POSTGRES_PASSWORD: my_secure_password
    ports:
      - "5433:5432"  # Use different port if 5432 is busy

  minio:
    ports:
      - "9010:9000"  # Use different port if 9000 is busy
      - "9011:9001"
```

This file is auto-loaded and won't be committed to git (add to `.gitignore`).

---

## CI/CD Integration

For GitHub Actions or other CI systems:

```yaml
# Example: .github/workflows/integration-tests.yml
name: Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_DB: cacheness_test
          POSTGRES_USER: cacheness
          POSTGRES_PASSWORD: cacheness_dev_password
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      minio:
        image: minio/minio:latest
        env:
          MINIO_ROOT_USER: minioadmin
          MINIO_ROOT_PASSWORD: minioadmin
        options: >-
          --health-cmd "curl -f http://localhost:9000/minio/health/live"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 9000:9000

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -e '.[dev,s3,postgresql,cloud]'
      
      - name: Run integration tests
        run: pytest tests/ -v
        env:
          POSTGRES_HOST: postgres
          S3_ENDPOINT_URL: http://minio:9000
```

---

## Next Steps

- See [BACKEND_SELECTION.md](../BACKEND_SELECTION.md) for backend selection guide
