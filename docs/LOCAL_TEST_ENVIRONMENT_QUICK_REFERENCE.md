# Local Test Environment: Quick Reference

## Configuration Files

Pre-configured YAML/JSON files in `config/`:

```bash
config/test_config.yaml       # PostgreSQL + MinIO (needs Docker)
config/test_config.json       # Same as above (JSON format)
config/local_sqlite_fs.yaml   # SQLite + Filesystem (no Docker)
```

**Load in Python:**
```python
from cacheness import Cache
from cacheness.config import CacheConfig

config = CacheConfig.from_yaml("config/test_config.yaml")
cache = Cache(config=config)
```

**Use in tests:**
```python
def test_with_config(cacheness_cache_from_yaml):
    cacheness_cache_from_yaml.put({"data": "value"}, key="test")
```

See [config/README.md](../config/README.md) for details.

## One-Liner Setup

### Windows PowerShell
```powershell
scripts\setup_local_env.bat
```

### macOS/Linux
```bash
python scripts/setup_local_env.py
```

Or use the Makefile:
```bash
make setup-env
```

## Docker Compose Commands

```bash
# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Stop and remove data
docker-compose down -v
```

## Connection Details

| Service     | Host/URL                          | Port | User        | Password             |
|-------------|-----------------------------------|------|-------------|----------------------|
| PostgreSQL  | localhost                         | 5432 | cacheness   | cacheness_dev_pass   |
| MinIO API   | http://localhost:9000             | 9000 | minioadmin  | minioadmin           |
| MinIO Web   | http://localhost:9001             | 9001 | minioadmin  | minioadmin           |

## Common Test Commands

```bash
# All tests
make test

# Integration tests only
make test-integration

# PostgreSQL tests
make test-postgres

# S3 tests
make test-s3

# Quick unit tests (no containers)
make test-quick
```

## Environment Variables

Copy `.env.example` to `.env` to customize:

```bash
cp .env.example .env
# Edit .env with custom values
```

Then load:
```bash
# Linux/macOS
export $(cat .env | xargs)

# Windows PowerShell
Get-Content .env | ForEach-Object { 
    $name, $value = $_.split("=")
    [System.Environment]::SetEnvironmentVariable($name, $value)
}
```

## Test Fixtures

Use fixtures in your tests:

```python
def test_with_postgres(postgres_connection):
    """Test requiring PostgreSQL."""
    pass

def test_with_s3(s3_client, s3_bucket):
    """Test requiring MinIO."""
    pass

def test_integration(cacheness_postgres_s3_config):
    """Test PostgreSQL + S3 integration."""
    pass
```

See [conftest_integration.py](../tests/conftest_integration.py) for all available fixtures.

## Troubleshooting

### Services won't start
```bash
# Check Docker daemon
docker ps

# Check logs
docker-compose logs postgres
docker-compose logs minio

# Free up ports (Windows)
netstat -ano | findstr :5432
# Kill process if needed
```

### Tests hang
- Services may need more time to become healthy
- Check `docker-compose ps` for "healthy" status
- Wait 10-30 seconds after starting

### Port conflicts
Edit `docker-compose.yml` and change ports:
```yaml
postgres:
  ports:
    - "5433:5432"  # Use 5433 instead of 5432
```

## Full Documentation

See [docs/LOCAL_TEST_ENVIRONMENT.md](../docs/LOCAL_TEST_ENVIRONMENT.md) for complete setup guide.
