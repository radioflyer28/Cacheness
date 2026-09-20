# Security guide

This guide describes boundaries Cacheness enforces. It does not make hostile
payloads safe, turn storage ownership into an access-control service, or expand
the topology and release claims in [Release qualification](RELEASE_QUALIFICATION.md).

## Trusted Payload and Executable Serializer Boundary

Cacheness treats persisted metadata, locators, legacy headers, and lifecycle-control objects as untrusted inputs. It uses safe parsing and path containment, validates a handler's private staging path and returned regular artifact, and must fail closed on malformed metadata, a malformed legacy header, an unsafe path, inode substitution, a failed signature, or an integrity mismatch. Canonical integrity is SHA-256 plus size; signed ETag/version values are opaque corroborating transport evidence, never a replacement content hash.

Application payloads are **trusted application payloads** only when every
producer and key-management boundary is controlled by the application. HMAC
signatures and integrity verification detect unauthorized changes at that
boundary, but do not sandbox deserialization or make hostile data safe.

### Pickle and dill are executable serialization

`pickle` and `dill` can execute arbitrary code while deserializing. Do not
load them from an untrusted producer, even if an entry has a valid HMAC,
signature, or digest. A missing key, invalid signature, or failed integrity
check is a failed entry; retaining invalid evidence for debugging never
authorizes its payload for deserialization.

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

This explicit configuration is an authenticity/integrity gate for trusted
payloads. It is not a safe-unpickling mode. Keep
`allow_trusted_object_arrays=False` unless the application controls every
payload producer and its key-management boundary.

### Native formats and legacy compatibility

Native handlers are the format owner for new payloads. The former Cacheness
raw-array header is read-only compatibility, not a format for new writes. A
malformed or unsafe legacy header is rejected instead of guessing another
sidecar or enabling pickle. This guide does not promise a replacement
container or a stored-data conversion workflow.

## Store-owner and remote transport boundaries

For a filesystem-backed store, the OS principal that owns the store is part of
the trusted deployment boundary. Do not modify `.cacheness-*` control objects,
lifecycle locks, metadata authority files, or platform authority records while
a store is open. Cacheness detects observable substitution and fails closed,
but cannot preserve coordination if that same owner deliberately deletes or
rebinds every live control object. Use filesystem permissions for other
principals and an external transactional authority for cross-principal work.

On Windows, the local-store boundary is limited to one OS user and one
interactive or service session. It is not a cross-user, cross-service, or
cross-session coordinator.

Amazon S3 is a payload participant, not a metadata/visibility authority or a
cross-resource transaction. A future qualified deployment needs an explicit
stable bucket, a bucket name whose ownership you control, narrow IAM
credentials, and a restrictive bucket policy for the configured prefix.
`ExpectedBucketOwner` is deliberately not exposed; Cacheness does not add a
custom signer, a legacy SDK escape hatch, or a second payload participant.
Production custom endpoint overrides are unsupported. Phase 8 retained this
boundary and its verification machinery, but remote qualification remains the
separate nonpassing path documented in the release guide.

## Operating securely

- Treat keys and signing material as application secrets. Configure filesystem
  permissions so other OS principals cannot modify the store or read key files.
- Use only contained paths and suffixes declared by the selected format
  handler; custom handlers must perform I/O only through their private staging
  location.
- Treat cache misses caused by failed verification as failures to investigate,
  not as permission to deserialize raw retained bytes.
- Stop workers before offline migration or rebuild. Ordinary store opens
  validate and fail rather than silently upgrading historical layouts.
- For the exact payload ceiling, topology-specific recovery limits, and current
  remote/platform/publication status, use the single
  [Release qualification](RELEASE_QUALIFICATION.md) owner.
