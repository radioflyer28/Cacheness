# Security Guide

This guide covers cache entry signing, integrity protection, namespace-aware key management, and security best practices for Cacheness.

## Overview

Cacheness provides cryptographic signing for cache metadata entries to prevent tampering with the SQLite database or JSON metadata files. This ensures cache integrity and detects unauthorized modifications.

## Quick Start

```python
from cacheness import cacheness, CacheConfig, SecurityConfig

# Default configuration - signing enabled with 10 metadata fields
cache = cacheness()  # ✅ Entry signing active (v2 signature, 10 fields)

# High-security configuration
secure_config = CacheConfig(
    security=SecurityConfig(
        enable_entry_signing=True,        # Enable HMAC signing (default)
        use_in_memory_key=True,           # No key persistence
        delete_invalid_signatures=True,   # Auto-cleanup (default)
        allow_unsigned_entries=False,      # Reject unsigned entries
    )
)
cache = cacheness(secure_config)
```

## Cache Entry Signing

### How It Works

1. **Signature Generation**: When storing cache entries, Cacheness creates HMAC-SHA256 signatures of critical metadata fields
2. **Signature Storage**: The signature is stored alongside the cache entry metadata as `entry_signature`
3. **Signature Verification**: On every cache retrieval, the signature is verified against the current metadata
4. **Tamper Detection**: If verification fails, the entry is treated as corrupted/tampered

### Signature Versions

Signed fields are managed via version-based field lists. The current version is **v2**.

**Signature format:** `v{N}:{hex_hmac_sha256}` (e.g., `v2:abcdef01...`). Legacy bare-hex signatures are treated as v1.

#### v2 Fields (Current — 10 fields)

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
| `created_at` | Timestamp | Prevents replay attacks |

#### v1 Fields (Legacy — 11 fields)

v1 includes all v2 fields **plus** `actual_path`. Legacy entries without a stored signature version are verified with the v1 field list. New entries always use v2.

> **Note:** Field selection is not configurable. The signer uses the version-appropriate field list automatically. This design prevents misconfiguration and ensures consistent verification.

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
# Automatic key generation (32-byte random secret via secrets.token_bytes)
cache = cacheness()  # Auto-generates signing key

# Key file location
print(cache.config.security.signing_key_file)  # "cache_signing_key.bin"

# Custom key file location
config = CacheConfig(
    security=SecurityConfig(
        signing_key_file="custom_signing_key.bin"
    )
)
```

### Key Rotation

```python
# For key rotation, simply delete the key file and restart
import os
os.remove("cache_signing_key.bin")  # Forces new key generation

# Or use in-memory keys for automatic rotation per process
config = CacheConfig(
    security=SecurityConfig(use_in_memory_key=True)
)
```

> **Important:** After key rotation, existing entries will fail signature verification. Set `allow_unsigned_entries=True` or `delete_invalid_signatures=True` during the transition period.

## Signing Keys and Namespaces

### Current Behavior

All namespaces sharing the same `cache_dir` share the **same signing key file** (`cache_signing_key.bin` by default). The key file path is derived from `cache_dir`, not from the namespace:

```
cache_dir/
├── cache_signing_key.bin    ← shared by ALL namespaces
├── cache_metadata.json      ← default namespace metadata
├── analytics/               ← "analytics" namespace blob storage
│   └── ...
└── ml_pipeline/             ← "ml_pipeline" namespace blob storage
    └── ...
```

This means:
- **Cross-namespace verification works** — an entry signed by namespace A can be verified by namespace B (same key)
- **No cryptographic isolation** between namespaces sharing a `cache_dir`
- The signed `prefix` field provides logical namespace attribution but not cryptographic separation

### Per-Namespace Key Configuration

To achieve cryptographic isolation between namespaces, use separate `cache_dir` paths or separate `signing_key_file` names:

```python
from cacheness import cacheness, CacheConfig, SecurityConfig

# Namespace A with its own signing key
config_a = CacheConfig(
    cache_dir="./cache",
    namespace="team_alpha",
    security=SecurityConfig(
        signing_key_file="signing_key_alpha.bin"
    )
)
cache_a = cacheness(config_a)

# Namespace B with its own signing key
config_b = CacheConfig(
    cache_dir="./cache",
    namespace="team_beta",
    security=SecurityConfig(
        signing_key_file="signing_key_beta.bin"
    )
)
cache_b = cacheness(config_b)
```

### Multi-Tenant Deployment Patterns

#### Shared Key (Simple — Default)

All namespaces share a single key. Suitable when namespaces are used for organizational purposes and all tenants are trusted.

```python
# All namespaces use the same key (default)
cache_prod = cacheness(CacheConfig(namespace="production"))
cache_staging = cacheness(CacheConfig(namespace="staging"))
# Both use cache_signing_key.bin in their cache_dir
```

#### Isolated Keys (Secure — Multi-Tenant)

Each namespace has its own signing key. Use when namespaces represent different security domains or untrusted tenants.

```python
# Each namespace gets its own cache directory and key
for tenant in ["client_a", "client_b", "client_c"]:
    config = CacheConfig(
        cache_dir=f"./cache/{tenant}",
        namespace=tenant,
        security=SecurityConfig(
            signing_key_file="signing_key.bin",  # unique per cache_dir
            allow_unsigned_entries=False,
        )
    )
    cache = cacheness(config)
```

#### Per-Namespace Key File (Shared Directory)

When tenants must share a `cache_dir` but need separate keys:

```python
for tenant in ["client_a", "client_b"]:
    config = CacheConfig(
        cache_dir="./shared_cache",
        namespace=tenant,
        security=SecurityConfig(
            signing_key_file=f"signing_key_{tenant}.bin",
        )
    )
    cache = cacheness(config)
```

> **Note:** Key files are always stored relative to `cache_dir`. With per-namespace key files, you get `cache_dir/signing_key_client_a.bin` and `cache_dir/signing_key_client_b.bin`.

### Migration Guide: Adding Namespaces to an Existing Cache

When migrating from a single-namespace cache to multi-namespace:

1. **Existing entries stay in the `default` namespace** — signed with the original key
2. **New namespaces can use the same key** (shared model) or separate keys (isolated model)
3. **No re-signing needed** — existing entries remain valid under the `default` namespace

```python
# Step 1: Existing cache continues working (default namespace)
legacy_cache = cacheness(CacheConfig(
    cache_dir="./existing_cache",
    security=SecurityConfig(
        allow_unsigned_entries=True,  # Accept pre-signing entries
    )
))

# Step 2: New namespace uses same key (entries are cross-verifiable)
new_cache = cacheness(CacheConfig(
    cache_dir="./existing_cache",
    namespace="v2_pipeline",
    security=SecurityConfig(
        allow_unsigned_entries=False,  # New namespace: strict from the start
    )
))
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

### Storage Mode Behavior

In storage mode (`storage_mode=True`), entries are **never deleted** even if signature verification fails — only `None` is returned. This preserves the storage-mode guarantee of data durability.

### The `allow_unsigned_entries` Flag

Controls handling of entries that have **no** signature (e.g., created before signing was enabled):

| `allow_unsigned_entries` | Entry has no signature | Entry has invalid signature |
|---|---|---|
| `True` (default) | ✅ Accepted | Depends on `delete_invalid_signatures` |
| `False` | ❌ Rejected (deleted) | Depends on `delete_invalid_signatures` |

## Security Best Practices

### Production Environments

```python
# Recommended production configuration
production_config = CacheConfig(
    security=SecurityConfig(
        enable_entry_signing=True,
        use_in_memory_key=True,             # No key persistence
        delete_invalid_signatures=True,     # Auto-cleanup
        allow_unsigned_entries=False,        # Strict mode
    )
)
```

### Development Environments

```python
# Development-friendly configuration
dev_config = CacheConfig(
    security=SecurityConfig(
        enable_entry_signing=True,
        use_in_memory_key=False,            # Persistent across restarts
        delete_invalid_signatures=False,    # Keep for debugging
        allow_unsigned_entries=True,         # Backward compatibility
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
        delete_invalid_signatures=False,    # Don't delete during migration
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
- ❌ Direct file system access to cached data files (blobs are not encrypted)
- ❌ Complete database replacement
- ❌ Process memory attacks
- ❌ OS-level privilege escalation

### Risk Assessment

| Configuration | Security Level | Performance Impact | Use Case |
|---|---|---|---|
| `enable_entry_signing=False` | Low | None | Development only |
| Default (`enable_entry_signing=True`) | High | ~0.1ms per op | **Recommended** |
| `use_in_memory_key=True` | Very High | None | Production/containers |
| `allow_unsigned_entries=False` | Very High | None | Strict environments |

## Performance Impact

### Signing Overhead

| Operation | Overhead | Notes |
|-----------|----------|-------|
| **cache.put()** | ~0.1ms | HMAC generation (10 fields) |
| **cache.get()** | ~0.1ms | HMAC verification (10 fields) |
| **Key generation** | ~5ms | One-time cost |

Signing overhead is negligible compared to I/O. The 10-field v2 signature provides comprehensive protection with minimal performance impact.

## Troubleshooting

### Common Issues

**Signature verification failures after restart:**
```python
# Check if using in-memory keys (expected behavior with in-memory keys)
print(cache.config.security.use_in_memory_key)  # True = expected

# Switch to persistent keys if you need entries to survive restarts
config = CacheConfig(
    security=SecurityConfig(use_in_memory_key=False)
)
```

**Cache not accessible after key file deletion:**
```python
# Expected behavior - old entries can't be verified
# Option 1: Allow unsigned entries during migration
config = CacheConfig(
    security=SecurityConfig(allow_unsigned_entries=True)
)

# Option 2: Delete invalid and rebuild cache
config = CacheConfig(
    security=SecurityConfig(delete_invalid_signatures=True)
)
```

### Diagnostic Information

```python
# Get signer information
if cache.signer:
    info = cache.signer.get_field_info()
    print(f"Signature version: {info['signature_version']}")
    print(f"Signed fields ({len(info['signed_fields'])}): {info['signed_fields']}")
    print(f"In-memory key: {info['use_in_memory_key']}")
    print(f"Key file exists: {info['key_exists']}")
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
        allow_unsigned_entries=False,        # Strict mode
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

### Multi-Tenant with Isolated Keys

```python
from cacheness import cacheness, CacheConfig, SecurityConfig

def create_tenant_cache(tenant_id: str):
    """Create a cache with tenant-specific signing key."""
    return cacheness(CacheConfig(
        cache_dir=f"./cache/{tenant_id}",
        namespace=tenant_id,
        security=SecurityConfig(
            enable_entry_signing=True,
            allow_unsigned_entries=False,
            delete_invalid_signatures=True,
        )
    ))

# Each tenant has completely isolated signing keys and blob storage
cache_acme = create_tenant_cache("acme_corp")
cache_globex = create_tenant_cache("globex_inc")
```

### Container Deployment

```python
# Perfect for containerized microservices
container_config = CacheConfig(
    security=SecurityConfig(
        use_in_memory_key=True,             # No persistent state
        delete_invalid_signatures=True,     # Clean startup
    )
)

# Each container instance gets fresh signing keys
# Old cache entries automatically cleaned up
cache = cacheness(container_config)
```
