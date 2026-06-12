---
created: 2026-04-03T20:10:58.493Z
title: Encryption at rest for metadata and blobs
area: general
status: completed
completed: 2026-06-12
resolution: Shipped in Phase 23 (ENC-01/ENC-02, D-01/D-02) — AES-256-GCM blob encryption with HKDF keys, hash-over-ciphertext. Residual gaps tracked separately.
files:
  - docs/SECURITY.md
  - docs/FUTURE_IMPROVEMENTS.md
  - docs/LIBSQL_BACKEND.md
---

> **Closed 2026-06-12:** Blob encryption shipped in Phase 23 (`src/cacheness/encryption.py`, inline-blob D-01/D-02 paths). The 2026-06 code review confirmed the crypto fundamentals are sound (AES-256-GCM, random nonces, HKDF domain separation, hash-over-ciphertext) but found residual gaps now tracked as **TASK-14** in `docs/CODE_REVIEW_ACTIONS.md` (findings S2/S3: decrypted plaintext written to temp files on disk during reads; encrypted reads bypass the blob-backend abstraction, breaking memory:// and s3://). Metadata-at-rest encryption (libSQL `encryption_key`) remains unimplemented — still listed in CONCERNS.md.

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
