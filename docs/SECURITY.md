# Security Guide

This guide covers cache entry signing, integrity protection, and security best practices for Cacheness.

## Overview

Cacheness provides cryptographic signing for cache metadata entries to prevent tampering with the SQLite database or JSON metadata files. This ensures cache integrity and detects unauthorized modifications.

## Quick Start

```python
from cacheness import cacheness, CacheConfig, SecurityConfig

# Default configuration - signing enabled with enhanced field set
cache = cacheness()  # ✅ Entry signing active with 11 security fields

# High-security configuration
secure_config = CacheConfig(
    security=SecurityConfig(
        enable_entry_signing=True,        # Enable HMAC signing
        use_in_memory_key=True,           # No key persistence
        delete_invalid_signatures=True,   # Auto-cleanup
        custom_signed_fields=None         # Use default enhanced fields
    )
)
cache = cacheness(secure_config)
```

## Cache Entry Signing

### How It Works

1. **Signature Generation**: When storing cache entries, Cacheness creates HMAC-SHA256 signatures of critical metadata fields
2. **Signature Storage**: The signature is stored alongside the cache entry metadata
3. **Signature Verification**: On every cache retrieval, the signature is verified against the current metadata
4. **Tamper Detection**: If verification fails, the entry is treated as corrupted/tampered

### Default Signed Fields

Cacheness signs **11 fields** by default for enhanced security:

| Field | Purpose | Security Value |
|-------|---------|----------------|
| `cache_key` | Unique identifier | Prevents key substitution |
| `data_type` | Type classification | Prevents type confusion |
| `prefix` | Key prefix | Namespace protection |
| `file_size` | Size verification | Detects partial corruption |
| `file_hash` | Content integrity | Detects file tampering |
| `object_type` | Original object type | Prevents type spoofing |
| `storage_format` | Serialization format | Detects format tampering |
| `serializer` | Serializer used | Prevents deserialize attacks |
| `compression_codec` | Compression method | Detects compression tampering |
| `actual_path` | File location | Prevents path substitution |
| `created_at` | Timestamp | Prevents replay attacks |

### Custom Field Selection

You can customize which fields are signed based on your security requirements:

```python
# Minimal security - only essential fields (faster)
minimal_config = CacheConfig(
    security=SecurityConfig(
        custom_signed_fields=["cache_key", "file_hash", "data_type", "file_size"]
    )
)

# Maximum security - all available fields (comprehensive)
paranoid_config = CacheConfig(
    security=SecurityConfig(
        custom_signed_fields=[
            "cache_key", "file_hash", "data_type", "file_size",
            "created_at", "prefix", "description", "actual_path",
            "object_type", "storage_format", "serializer", "compression_codec"
        ]
    )
)

# Default enhanced security (recommended)
enhanced_config = CacheConfig(
    security=SecurityConfig(
        custom_signed_fields=None  # Uses default 11 fields
    )
)
```

## Key Management

### Persistent Keys (Default)

```python
# Default behavior - key persisted to disk
config = CacheConfig(
    security=SecurityConfig(
        use_in_memory_key=False  # Default
    )
)
```

**Characteristics:**
- ✅ Key stored in `cache_signing_key.bin` with restrictive permissions (0600)
- ✅ Cache entries remain valid across process restarts
- ✅ Good for development and persistent environments
- ⚠️ Key file could potentially be compromised if disk access is breached

### In-Memory Keys (Enhanced Security)

```python
# High-security mode - no key persistence
config = CacheConfig(
    security=SecurityConfig(
        use_in_memory_key=True  # Enhanced security
    )
)
```

**Characteristics:**
- ✅ No cryptographic material written to disk
- ✅ New key generated for each process
- ✅ Cache entries invalidated on restart
- ✅ Ideal for high-security environments
- ✅ Perfect for containerized applications
- ⚠️ Cache doesn't survive process restarts

### Key Generation

```python
# Automatic key generation
cache = cacheness()  # Auto-generates 32-byte HMAC-SHA256 key

# Key file location
print(cache.config.security.signing_key_file)  # "cache_signing_key.bin"

# Custom key file location
config = CacheConfig(
    security=SecurityConfig(
        signing_key_file="custom_signing_key.bin"
    )
)
```

## Signature Verification

### Automatic Verification

Every cache retrieval automatically verifies signatures:

```python
# Store data with signature
cache.put({"results": [1, 2, 3]}, experiment="exp_001")

# Retrieve data - signature automatically verified
data = cache.get(experiment="exp_001")  # ✅ Signature valid

# If signature verification fails:
# - delete_invalid_signatures=True → Entry deleted, returns None
# - delete_invalid_signatures=False → Warning logged, data returned
```

### Handling Invalid Signatures

```python
# Auto-delete invalid signatures (recommended for production)
config_strict = CacheConfig(
    security=SecurityConfig(
        delete_invalid_signatures=True  # Default
    )
)

# Retain invalid signatures for debugging
config_debug = CacheConfig(
    security=SecurityConfig(
        delete_invalid_signatures=False
    )
)
```

## Security Best Practices

### Production Environments

```python
# Recommended production configuration
production_config = CacheConfig(
    security=SecurityConfig(
        enable_entry_signing=True,
        custom_signed_fields=None,          # Use default enhanced fields (6 fields)
        use_in_memory_key=True,             # No key persistence
        delete_invalid_signatures=True,     # Auto-cleanup
        allow_unsigned_entries=False        # Strict mode
    )
)
```

### Development Environments

```python
# Development-friendly configuration
dev_config = CacheConfig(
    security=SecurityConfig(
        enable_entry_signing=True,
        custom_signed_fields=None,          # Use default enhanced fields
        use_in_memory_key=False,            # Persistent across restarts
        delete_invalid_signatures=False,    # Keep for debugging
        allow_unsigned_entries=True         # Backward compatibility
    )
)
```

### Containerized Applications

```python
# Perfect for containers/microservices
container_config = CacheConfig(
    security=SecurityConfig(
        use_in_memory_key=True,             # No persistent state
        delete_invalid_signatures=True,     # Clean startup
        custom_signed_fields=None           # Use default enhanced fields
    )
)
```

## Backward Compatibility

### Migration from Unsigned Caches

```python
# Gradual migration - allows both signed and unsigned entries
migration_config = CacheConfig(
    security=SecurityConfig(
        enable_entry_signing=True,
        allow_unsigned_entries=True,        # Accept legacy entries
        delete_invalid_signatures=False     # Don't delete during migration
    )
)
```

### Disabling Signing

```python
# Disable signing entirely
no_signing_config = CacheConfig(
    security=SecurityConfig(
        enable_entry_signing=False
    )
)
```

## Security Considerations

### Threat Model

**Protected Against:**
- ✅ Cache metadata tampering (database/JSON modification)
- ✅ File hash manipulation
- ✅ Cache key collision attacks
- ✅ Timestamp manipulation
- ✅ Unauthorized cache entry creation

**Not Protected Against:**
- ❌ Direct file system access to cached data files
- ❌ Complete database replacement
- ❌ Process memory attacks
- ❌ OS-level privilege escalation

### Risk Assessment

| Configuration | Security Level | Performance Impact | Use Case |
|---------------|----------------|-------------------|----------|
| `enable_entry_signing=False` | Low | None | Development only |
| `custom_signed_fields=[4 minimal fields]` | Medium | Minimal | Basic integrity |
| `custom_signed_fields=None` (default) | High | Low | **Recommended default** |
| `custom_signed_fields=[8 maximum fields]` | Very High | Medium | High-security environments |
| `use_in_memory_key=True` | Very High | None | Production/containers |

### Custom Field Performance

Different field combinations have varying performance characteristics:

```python
# Minimal fields - fastest signing/verification
minimal_fields = ["cache_key", "file_hash", "data_type", "file_size"]

# Default enhanced fields - balanced performance/security
default_fields = None  # Uses built-in enhanced set (6 fields)

# Maximum security fields - comprehensive but slower
maximum_fields = [
    "cache_key", "file_hash", "data_type", "file_size",
    "created_at", "prefix", "description", "actual_path"
]

config = CacheConfig(
    security=SecurityConfig(custom_signed_fields=minimal_fields)
)
```

### Key Rotation

```python
# For key rotation, simply delete the key file and restart
import os
os.remove("cache_signing_key.bin")  # Forces new key generation

# Or use in-memory keys for automatic rotation
config = CacheConfig(
    security=SecurityConfig(use_in_memory_key=True)
)
```

## Performance Impact

### Signing Overhead

| Operation | Overhead | Notes |
|-----------|----------|-------|
| **cache.put()** | ~0.1ms | HMAC generation |
| **cache.get()** | ~0.1ms | HMAC verification |
| **Key generation** | ~5ms | One-time cost |

### Field Count Performance

```python
import time

# Benchmark different field configurations
field_configs = {
    "minimal": ["cache_key", "file_hash", "data_type", "file_size"],
    "default": None,  # Uses default enhanced (6 fields)
    "maximum": ["cache_key", "file_hash", "data_type", "file_size", 
               "created_at", "prefix", "description", "actual_path"]
}

for name, fields in field_configs.items():
    config = CacheConfig(
        security=SecurityConfig(custom_signed_fields=fields)
    )
    cache = cacheness(config)
    
    start = time.time()
    for i in range(1000):
        cache.put(f"data_{i}", test_field=i)
    elapsed = time.time() - start
    print(f"{name}: {elapsed:.2f}s for 1000 operations")
```

**Typical Results:**
- **minimal**: ~1.2s (4 fields)
- **default**: ~1.4s (6 fields) - **Recommended**
- **maximum**: ~1.8s (8 fields)

## Troubleshooting

### Common Issues

**Signature verification failures after restart:**
```python
# Check if using in-memory keys
print(cache.config.security.use_in_memory_key)  # True = expected behavior

# Switch to persistent keys if needed
config = CacheConfig(
    security=SecurityConfig(use_in_memory_key=False)
)
```

**Cache not accessible after key file deletion:**
```python
# Expected behavior - regenerate cache or restore key file
# Set allow_unsigned_entries=True for migration period
```

**Performance concerns with maximum fields:**
```python
# Switch to default enhanced fields for better performance
config = CacheConfig(
    security=SecurityConfig(custom_signed_fields=None)  # Use default 6 fields
)

# Or use minimal fields for fastest performance
config = CacheConfig(
    security=SecurityConfig(
        custom_signed_fields=["cache_key", "file_hash", "data_type", "file_size"]
    )
)
```

### Diagnostic Information

```python
# Get signer information
if cache.signer:
    info = cache.signer.get_field_info()
    print(f"Signed fields: {info['signed_fields']}")
    print(f"In-memory key: {info['use_in_memory_key']}")
    print(f"Key file exists: {info['key_exists']}")
    
    # Check field count for performance analysis
    field_count = len(info['signed_fields'])
    print(f"Signing {field_count} fields")
    if field_count <= 4:
        print("Performance: Optimal (minimal fields)")
    elif field_count <= 6:
        print("Performance: Good (default enhanced)")
    else:
        print("Performance: Slower (maximum security)")
```

## Examples

### High-Security ML Pipeline

```python
from cacheness import cacheness, CacheConfig, SecurityConfig

# Secure configuration for sensitive ML data
ml_config = CacheConfig(
    cache_dir="./secure_ml_cache",
    security=SecurityConfig(
        enable_entry_signing=True,
        use_in_memory_key=True,             # No key persistence
        delete_invalid_signatures=True,     # Auto-cleanup
        custom_signed_fields=[              # Maximum protection - all 8 fields
            "cache_key", "file_hash", "data_type", "file_size",
            "created_at", "prefix", "description", "actual_path"
        ],
        allow_unsigned_entries=False        # Strict mode
    )
)

cache = cacheness(ml_config)

# Store sensitive model data
sensitive_model = train_confidential_model(private_data)
cache.put(sensitive_model, 
          project="confidential", 
          model="neural_net", 
          version="1.0")

# Data automatically protected with cryptographic signatures
# Cache invalidated on restart for maximum security
```

### Development with Debugging

```python
# Developer-friendly configuration
dev_config = CacheConfig(
    security=SecurityConfig(
        enable_entry_signing=True,
        use_in_memory_key=False,            # Survive restarts
        delete_invalid_signatures=False,    # Keep for debugging
        custom_signed_fields=None           # Use default enhanced fields
    )
)

cache = cacheness(dev_config)

# Debug signature issues
try:
    data = cache.get(experiment="test")
    if data is None:
        print("Cache miss or signature verification failed")
        # Check logs for signature warnings
except Exception as e:
    print(f"Cache error: {e}")
```

### Container Deployment

```python
# Perfect for containerized microservices
container_config = CacheConfig(
    security=SecurityConfig(
        use_in_memory_key=True,             # No persistent state
        delete_invalid_signatures=True,     # Clean startup
        custom_signed_fields=None,          # Use default enhanced fields
    )
)

# Each container instance gets fresh signing keys
# Old cache entries automatically cleaned up
cache = cacheness(container_config)
```

This security model provides robust protection against cache tampering while maintaining flexibility for different deployment scenarios.
