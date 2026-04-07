# Phase 25: Inline Blob Encryption - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-06
**Phase:** 25-inline-blob-encryption
**Areas discussed:** Encryption insertion point, Decryption strategy, Key rotation for inline, Inline ciphertext storage format, Hash computation

---

## Encryption Insertion Point

| Option | Description | Selected |
|--------|-------------|----------|
| Inside _try_direct_inline() | Encrypt right after handler.put_bytes(). Return ciphertext as blob_data + encryption fields in the dict. Self-contained. | ✓ |
| In the caller (core.py put()) | Return plaintext from _try_direct_inline(), encrypt in put() before storing. Mirrors _write_blob() pattern. | |

**User's choice:** Inside _try_direct_inline()
**Notes:** Self-contained approach — caller doesn't need to know about encryption.

---

## Decryption Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Decrypt upfront (before dispatch) | Check encryption_algorithm in entry metadata. If encrypted + key available, decrypt blob_data to plaintext before dispatching to fast or slow path. | ✓ |
| Slow path only (let fast path fail) | Let fast path fail on ciphertext, fall through to slow path which writes temp file and calls _read_blob() (which already decrypts). | |

**User's choice:** Decrypt upfront (before dispatch)
**Notes:** One decrypt point, both paths get clean plaintext. Simple and consistent.

---

## Key Rotation for Inline

| Option | Description | Selected |
|--------|-------------|----------|
| Inline branch in rotate_key() | When actual_path is None and has encryption metadata: read blob_data, decrypt with old key, re-encrypt with new key, update entry. Same loop, minimal change. | ✓ |
| Extract helper with type dispatch | Move entire re-encryption loop into a helper, dispatch by entry type. Cleaner but more refactoring. | |

**User's choice:** Inline branch in rotate_key()
**Notes:** Minimal code change, same loop structure.

---

## Inline Ciphertext Storage Format

| Option | Description | Selected |
|--------|-------------|----------|
| Base64 encode ciphertext | Store blob_data as base64 string when encrypted. Decode on read. JSON-safe. | |
| Raw bytes (let backend handle) | Keep blob_data as raw bytes. JSON backend doesn't support inline blobs anyway (only SQLite/PG). | ✓ |

**User's choice:** Raw bytes (let backend handle)
**Notes:** JSON backend confirmed to not support blob_data (L139 comment). Only SQLite/PG have binary columns.

---

## Hash Computation

| Option | Description | Selected |
|--------|-------------|----------|
| Recompute hash on plaintext | Hash the plaintext before encryption, matching file-backed flow. | ✓ |
| Hash the ciphertext | Simpler since that's what's stored, but differs from file-backed behavior. | |
| Skip hash (GCM is enough) | GCM authentication tag provides integrity. Hash adds no value. | |

**User's choice:** Recompute hash on plaintext
**Notes:** Consistent with file-backed behavior where file_hash represents original data integrity.

---

## Agent's Discretion

- Guard clause for missing encryption key on read (return None vs raise)
- Error handling for decryption failures
- Size check ordering (before vs after encryption) for max_inline_size

## Deferred Ideas

None — discussion stayed within phase scope.
