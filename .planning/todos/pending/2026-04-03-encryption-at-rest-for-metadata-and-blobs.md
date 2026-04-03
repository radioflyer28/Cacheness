---
created: 2026-04-03T20:10:58.493Z
title: Encryption at rest for metadata and blobs
area: general
files:
  - docs/SECURITY.md
  - docs/FUTURE_IMPROVEMENTS.md
  - docs/LIBSQL_BACKEND.md
---

## Problem

Cacheness provides integrity protection (HMAC signing, hash verification) but not confidentiality. Blobs are stored as plaintext files on disk. Anyone with filesystem access can read cached data — DataFrames, NumPy arrays, pickled objects, etc. Metadata (cache keys, data types, timestamps) is also plaintext in JSON/SQLite.

## Solution

Two complementary layers:

1. **Metadata encryption** — via the planned libSQL backend's `encryption_key` parameter. Encrypts the entire metadata database at rest. Depends on the libSQL backend being implemented first.

2. **Blob encryption** — AES-256-GCM envelope encryption in the blob write path:
   - Generate per-blob DEK (data encryption key)
   - Encrypt blob with DEK after handler serialization
   - Encrypt DEK with master key (derived via existing HKDF infrastructure)
   - Store encrypted blob + encrypted DEK
   - Transparent to handlers — encrypt after write, decrypt before read

Key management leverages existing `SecurityConfig` and HKDF per-namespace key derivation.

`file_hash` should be computed on ciphertext (not plaintext) so integrity verification doesn't require decryption.

See SECURITY.md "Encryption at Rest" and FUTURE_IMPROVEMENTS.md section 11 for full design.
