# Security Guide

This guide covers cache entry signing, integrity protection, and security best practices for Cacheness.

## Overview

Cacheness provides cryptographic signing for cache metadata entries to prevent tampering with the SQLite database or JSON metadata files. This ensures cache integrity and detects unauthorized modifications.

## Trusted Payload and Executable Serializer Boundary

Cacheness treats application payloads and cache control data differently. Cache
metadata, persisted locators, and legacy headers are untrusted inputs: they are
validated and malformed metadata or a malformed legacy header fails closed.
Application payloads are trusted only when they come from a producer that your
application trusts.

### Local store-owner boundary

For filesystem-backed stores, the OS principal that owns the store is also part
of the trusted deployment boundary. Cacheness validates lifecycle-control files,
contained paths, signatures, digests, and file identities and fails closed when
it observes substitution. It cannot preserve per-key coordination if that same
principal deliberately deletes or rebinds every live coordination object.

Do not modify `.cacheness-*` control objects, lifecycle lock files, metadata
authority files, or their platform authority records while a store is open.
Use filesystem permissions to prevent other principals from modifying the store.
Applications that require protection from the store owner or coordination across
different security principals need an external transactional authority rather
than the local filesystem topology.

On Windows, this local-store boundary is additionally limited to processes in
one OS user and one interactive or service session. The current-user registry
authority and local mutex are intentionally not a cross-user, cross-service, or
cross-session coordinator. Deployment ACLs must exclude other principals, and
deployments must not configure the same local store for more than one session.
Use an external transactional authority whenever that topology is required.

### Amazon S3 transport boundary (D-16)

Amazon S3 is an explicitly composed payload participant; it does not become a
metadata authority, visibility authority, or cross-resource transaction. For a
production S3 topology, configure an explicit bucket and region, use a stable
bucket name whose ownership you control, and grant only narrowly scoped IAM
credentials and bucket-policy permissions for the configured prefix.

The obstore 0.11.1 cutover intentionally does **not** expose
`ExpectedBucketOwner` or another owner-pinning workaround. Treat that as a
cross-account defense-in-depth residual risk: Cacheness does not fork the SDK,
add a custom SigV4 signer, retain a legacy SDK escape hatch, or create a second
payload participant. Production custom endpoint overrides are rejected; only
loopback mocked-S3 test fixtures may use them. Compatible S3 services require
their own qualification.

Publication is one direct conditional create at an immutable generation
locator. The configurable direct-create limit has a **128 MiB initial default**;
oversized payloads fail clearly rather than activating multipart, temporary
objects, or a fallback path. Before deserialization, Cacheness verifies the
signed canonical SHA-256 digest and byte size. ETag and optional version are
signed, generation-bound opaque transport evidence, not content hashes. The
developer transport comparison is read-only and noncanonical; it cannot prove
the canonical digest, authorize adoption, or select lifecycle visibility.

Phase 8 retains real AWS and compatible-service qualification, platform and
packaging matrices, RSS/performance budgets, and SHA-256-versus-XXH3 benchmark
evidence. Mocked S3 and skipped live tests do not satisfy those gates.

### Pickle and dill are executable serialization

`pickle` and `dill` can execute arbitrary code while deserializing. Do not load
them from an untrusted producer, even when a cache entry has a valid signature,
HMAC, or content digest. Those integrity controls detect unauthorized changes
and establish authenticity for the configured trust boundary; they do not
sandbox deserialization or make hostile executable serialization safe.

Treat a missing key, an invalid signature, or a failed integrity check as a
failed cache entry. Retaining invalid evidence for debugging never authorizes
its payload for deserialization.

### Safe array defaults and explicit object-array opt-in

Ordinary NumPy arrays use native NPZ and load with `allow_pickle=False`.
Object-dtype arrays need pickle semantics, so they cross into `ObjectHandler`
only for trusted application payloads and only when every predicate below is
set together:

```python
from cacheness import (
    CacheConfig,
    CacheMetadataConfig,
    HandlerConfig,
    SecurityConfig,
)

trusted_object_array_config = CacheConfig(
    handlers=HandlerConfig(
        allow_trusted_object_arrays=True,
        enable_object_pickle=True,
    ),
    metadata=CacheMetadataConfig(verify_cache_integrity=True),
    security=SecurityConfig(
        enable_entry_signing=True,
        allow_unsigned_entries=False,
    ),
)
```

This opt-in preserves an authenticity and integrity gate for trusted payloads;
it is not a safe-unpickling mode for hostile data. Keep the default
`allow_trusted_object_arrays=False` unless your application controls every
payload producer and key-management boundary.

### Native formats and legacy compatibility

Native handlers are the format owners for new payloads. The former
Cacheness raw-array header is a read-only compatibility artifact, not a
format for new writes. A declared malformed or unsafe legacy header is rejected
instead of guessing another sidecar or enabling pickle. This guide does not
promise a replacement container or a stored-data migration workflow.

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
| `serializer` | Serializer used | Binds the declared serializer metadata |
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
# - delete_invalid_signatures=True → Entry removed, returns None
# - delete_invalid_signatures=False → Evidence retained, returns None
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

## Unsigned Entry Compatibility

### Allowing Existing Unsigned Entries

```python
# Compatibility setting - allows both signed and unsigned entries
migration_config = CacheConfig(
    security=SecurityConfig(
        enable_entry_signing=True,
        allow_unsigned_entries=True,        # Accept legacy entries
        delete_invalid_signatures=False     # Don't delete during migration
    )
)
```

This setting does not make untrusted executable payloads safe and does not
perform stored-data migration.

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
