# Phase 18: Encryption at Rest - Context

**Gathered:** 2026-04-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement AES-256-GCM encryption of cached blob data at rest. Encryption is disabled by default — existing users see zero behavior change. When enabled, blob content is encrypted after handler compression and decrypted before handler decompression. Signing remains independent and operates on metadata. Unencrypted entries remain readable when encryption is later enabled (migration path).

**Primary threat model:** Storing cached data on remote servers (S3, PostgreSQL, libSQL cloud replicas) that are not fully trusted — either because the server could be compromised, or because the storage provider shouldn't have access to the data. Encryption happens **client-side** before data leaves the local process, so the server only ever stores ciphertext. A server breach exposes only ciphertext, useless without the client-held master key. This is especially important for the tiered cache pattern (local + remote UnifiedCache) and libSQL embedded replicas that sync metadata to Turso Cloud.

</domain>

<decisions>
## Implementation Decisions

### D-01: Encryption Architecture — Injection Point
Encrypt/decrypt at the BlobStore layer — encrypt after handler compression, decrypt before handler decompression. Single integration point for all data types. Handlers are unaware of encryption.

### D-02: Encryption Architecture — Algorithm
AES-256-GCM via `cryptography` library (AEAD with authentication). Added as optional dependency `cacheness[encryption]`.

### D-03: Encryption Architecture — Key Derivation
Reuse HKDF from Phase 16 with different info string: `b"cacheness-aes-gcm-v1:{namespace_id}"`. Domain separation from signing keys — same master key, cryptographically isolated derived keys.

### D-04: Encryption Architecture — Nonce/IV Strategy
Random 12-byte IV per blob via `os.urandom(12)`. Stored in entry metadata. Standard AES-GCM practice, no counter management needed.

### D-05: Configuration — Config Location
Extend `SecurityConfig` with `enable_content_encryption: bool = False` and `encryption_key_file: str = "cache_signing_key.bin"` (reuses signing key file by default). Keeps all security config in one dataclass.

### D-06: Configuration — Key File
Same key file as signing by default. HKDF derives separate keys for signing vs encryption from the same master key. One key to manage, cryptographically isolated.

### D-07: Configuration — Missing Dependency
Raise `CacheConfigurationError` at init if encryption enabled but `cryptography` not installed. Fail fast with clear message: "Install cacheness[encryption] for AES-GCM support."

### D-08: Configuration — Per-Namespace
Encryption is a global setting, not per-namespace. Same pattern as signing.

### D-09: Migration — Detection
Metadata field `encryption_algorithm` — if present and non-null, entry is encrypted. If absent, entry is unencrypted (pre-encryption). Same version-detection pattern as Phase 16 signature versioning.

### D-10: Migration — Unencrypted Reads
Read transparently — check metadata `encryption_algorithm`. If absent, skip decrypt, read as-is. Satisfies SC-5 (unencrypted entries remain readable).

### D-11: Migration — Re-encryption
Don't auto-re-encrypt old entries. New puts are encrypted, old entries stay unencrypted. Keeps scope minimal.

### D-12: Migration — Signing Interaction
Encrypt-then-sign: signing operates on metadata (including encryption_algorithm, iv), not blob content. Content integrity via AES-GCM authentication tag. Signature verifies metadata integrity.

### D-13: Error Handling — Decryption Failure
Return None + log warning — same as signature verification failure. Entry is effectively corrupted. `delete_invalid_signatures` setting controls auto-delete behavior.

### D-14: Error Handling — Key Rotation
Extend `rotate_key()` to re-encrypt: decrypt with old key, re-encrypt with new key. Add `re_encrypted` count to `RotationResult`.

### D-15: Error Handling — S3/Remote Backends
Encryption happens transparently in BlobStore before blob backend write. Remote backends receive already-encrypted bytes. No backend-specific changes needed.

### D-16: Error Handling — Performance Documentation
Add brief note in SECURITY.md documenting expected overhead. No full benchmark required.

### Agent's Discretion
- Internal class/function naming for encryption utilities
- Exact metadata field names (suggested: `encryption_algorithm`, `encryption_iv`, `encryption_tag`)
- Whether to create a new file (e.g., `encryption.py`) or extend `security.py`
- Test fixture patterns and helper functions

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_hkdf_sha256()` in `security.py` — RFC 5869 HKDF, can derive encryption keys with different info string
- `CacheEntrySigner` pattern — class-based security component with init/sign/verify lifecycle
- `SecurityConfig` dataclass — existing security configuration container
- `create_cache_signer()` factory — pattern for creating security components from config

### Established Patterns
- Security components initialized in `BlobStore._init_signer()` and `UnifiedCache._init_entry_signer()`
- Metadata stored in nested dict within entry_data: `entry_data["metadata"]` contains `actual_path`, `compression_codec`, etc.
- Signature versioning: `signature_version` field gates behavior (v2 = master key, v3 = HKDF-derived)
- Thread safety via `threading.RLock()` — security operations are thread-safe
- Optional dependencies via `try/except` with availability flags (e.g., `PANDAS_AVAILABLE`)

### Integration Points
- **Write path:** `BlobStore.put()` → handler writes compressed file → `blob_backend.write_blob_from_path()` moves to managed dir. Encryption inserts between handler output and blob backend write.
- **Read path:** `BlobStore.get()` → `blob_backend` reads bytes → handler decompresses. Decryption inserts between blob backend read and handler input.
- **Config wiring:** `CacheMetadataConfig.security` → `SecurityConfig` → passed to `BlobStore` and `UnifiedCache`
- **Key rotation:** `UnifiedCache.rotate_key()` and `BlobStore.rotate_key()` — extend to re-encrypt

</code_context>

<specifics>
## Specific Ideas

- Encryption key derived via HKDF with info string `b"cacheness-aes-gcm-v1:{namespace_id}"` for domain separation from signing
- AES-GCM provides authenticated encryption — the 16-byte tag serves as content integrity check, complementing HMAC signature over metadata
- `cryptography` library as optional dependency in `[project.optional-dependencies]` group named `encryption`
- Metadata fields: `encryption_algorithm`, `encryption_iv` (hex), `encryption_tag` (hex)

</specifics>

<deferred>
## Deferred Ideas

- Batch re-encryption utility for migrating existing unencrypted entries
- Per-namespace encryption toggle
- Automated key rotation with re-encryption
- Alternative encryption algorithms (ChaCha20-Poly1305)

</deferred>
