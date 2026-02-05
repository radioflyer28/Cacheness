# Cacheness Configuration Files

This directory contains configuration files for different test and development scenarios.

## Configuration Files

### `test_config.yaml` / `test_config.json`
PostgreSQL + MinIO (S3) configuration for integration testing with Docker containers.

**Usage:**
```python
from cacheness import CacheConfig

# Load from YAML
config = CacheConfig.from_yaml("config/test_config.yaml")

# Load from JSON
config = CacheConfig.from_json("config/test_config.json")

# Create cache with config
cache = Cache(config=config)
```

**Requires:**
- PostgreSQL running on localhost:5432 (see docker-compose.yml)
- MinIO running on localhost:9000 (see docker-compose.yml)

### `local_sqlite_fs.yaml`
SQLite + Filesystem configuration for quick local development without containers.

**Usage:**
```python
config = CacheConfig.from_yaml("config/local_sqlite_fs.yaml")
cache = Cache(config=config)
```

**No dependencies** - works out of the box.

## Environment Variable Substitution

Configuration files support environment variable substitution:

```yaml
metadata_config:
  connection_string: "${POSTGRES_CONNECTION_STRING}"

blob_config:
  aws_access_key_id: "${S3_ACCESS_KEY}"
  aws_secret_access_key: "${S3_SECRET_KEY}"
```

Set environment variables before loading:
```bash
export POSTGRES_CONNECTION_STRING="postgresql://user:pass@host:5432/db"
export S3_ACCESS_KEY="minioadmin"
export S3_SECRET_KEY="minioadmin"
```

## Security: Development vs Production

### ⚠️ Development Credentials (Current Configs)

The provided config files contain **hardcoded credentials suitable ONLY for local testing**:
- PostgreSQL: `cacheness` / `cacheness_dev_password`
- MinIO: `minioadmin` / `minioadmin`
- Signing key: Hardcoded dev key

**These are safe because:**
- Services only accessible on localhost
- No external network exposure
- Standard practice for development Docker containers
- Similar to how official Docker images provide default credentials

**Do NOT use these credentials in:**
- Production environments
- Staging environments accessible from internet
- Any publicly accessible servers
- CI/CD deployments that touch production data

### 🔒 Production Security

For production, implement proper security:

#### 1. Use Environment Variables
```yaml
metadata_config:
  connection_string: "${POSTGRES_CONNECTION_STRING}"
blob_config:
  aws_access_key_id: "${AWS_ACCESS_KEY_ID}"
  aws_secret_access_key: "${AWS_SECRET_ACCESS_KEY}"
signing_key: "${CACHE_SIGNING_KEY}"
```

#### 2. Use Secret Management Services
- **AWS**: Secrets Manager or Parameter Store
- **Azure**: Key Vault
- **GCP**: Secret Manager
- **HashiCorp**: Vault
- **Kubernetes**: Secrets with encryption at rest

#### 3. Generate Strong Credentials

**Signing key:**
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

**Database password:**
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

#### 4. Use IAM Roles/Managed Identities

Instead of credentials, use:
- **AWS**: IAM roles for EC2/ECS/Lambda
- **Azure**: Managed identities
- **GCP**: Service accounts

Example AWS S3 with IAM:
```yaml
blob_config:
  bucket: "my-prod-bucket"
  region_name: "us-east-1"
  # No credentials needed - uses IAM role
```

#### 5. Network Security
- Run PostgreSQL/MinIO in private subnets
- Use VPC/Security Groups to restrict access
- Enable SSL/TLS for all connections
- Use private endpoints for cloud services

### When to Use Secret Management Tools

**Use SOPS/sealed-secrets/etc when:**
- Committing configs to git that will be used in staging/production
- Sharing configs across team members for non-local environments
- Deploying to Kubernetes clusters
- Managing multiple environment configurations

**Skip secret management tools when:**
- Local development only (current use case)
- Credentials are localhost Docker defaults
- No risk of credential exposure

### Example Production Config

```yaml
# production_config.yaml - with environment variables
cache_dir: "/var/cache/cacheness"

metadata_backend: "postgresql"
metadata_config:
  connection_string: "${POSTGRES_CONNECTION_STRING}"  # From secret manager

blob_backend: "s3"
blob_config:
  bucket: "${S3_BUCKET_NAME}"
  region_name: "${AWS_REGION}"
  # Uses IAM role - no credentials needed
  shard_chars: 2

signing_key: "${CACHE_SIGNING_KEY}"  # From secret manager
delete_invalid_signatures: true  # Strict in production

handlers:
  enable_dill_fallback: false  # Disabled for security in production
  tensorflow_enabled: true

compress: true
compress_level: 5
verbose: false  # Less logging in production
```

## Signing Keys

**Development:**
```yaml
signing_key: "dev_test_key_do_not_use_in_production_12345678901234567890"
```

**Production:**
Generate a secure key:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Store in environment variable and reference:
```yaml
signing_key: "${CACHE_SIGNING_KEY}"
```

## Testing with Config Files

### In pytest fixtures

```python
@pytest.fixture
def cache_with_config():
    """Provide a cache instance using test config."""
    from cacheness import Cache, CacheConfig
    
    config = CacheConfig.from_yaml("config/test_config.yaml")
    cache = Cache(config=config)
    yield cache
    # Cleanup if needed

def test_with_config(cache_with_config):
    """Test using configured cache."""
    cache_with_config.put({"data": "value"}, key="test")
    assert cache_with_config.get(key="test") == {"data": "value"}
```

### Command-line override

Load config and override specific parameters:
```python
config = CacheConfig.from_yaml("config/test_config.yaml")

# Override signing key
config.signing_key = os.getenv("CACHE_SIGNING_KEY", config.signing_key)

# Override blob bucket
config.blob_config.bucket = os.getenv("S3_BUCKET", config.blob_config.bucket)

cache = Cache(config=config)
```

## Docker Compose Integration

The `.devcontainer/devcontainer.json` automatically mounts this directory, making configs available inside the container.

## Best Practices

1. **Never commit production keys** - Use environment variables
2. **Use separate configs** for different environments (dev/test/prod)
3. **Set `delete_invalid_signatures: false`** in dev for debugging
4. **Use lower `compress_level`** in dev for faster iteration
5. **Enable `verbose: true`** in dev for detailed logging
