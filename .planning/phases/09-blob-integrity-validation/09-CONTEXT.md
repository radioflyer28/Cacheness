---
phase: 09-blob-integrity-validation
type: context
source: auto-generated (infrastructure phase)
---

# Phase 09: Blob Integrity Validation — Context

## Phase Goal
Validate blob content integrity end-to-end — users can trust that stored blobs have not been tampered with or corrupted.

## Current State Analysis

### file_hash in HMAC Signed Fields ✅
`file_hash` is already included in both v1 and v2 signed field lists in `security.py`:
- `SIGNED_FIELDS_BY_VERSION[1]` includes `file_hash`
- `SIGNED_FIELDS_BY_VERSION[2]` includes `file_hash`

This means: tampering with blob content changes `file_hash`, which would invalidate the HMAC signature.

### verify_integrity() — Hash Mismatch Detection ✅
`blob_store.py:verify_integrity()` already detects blob content modification via xxhash comparison when `verify_hashes=True`. Test coverage exists in `test_cache_integrity_verification.py::test_detects_hash_mismatch`.

### Gap: verify_integrity() Lacks Signature Verification
`verify_integrity()` does NOT verify HMAC signatures. HMAC verification only happens during `get()` via `_verify_entry()`. A bulk audit won't catch sophisticated tampering where both blob and metadata `file_hash` are modified consistently (without the HMAC key).

## Key Files
- `src/cacheness/security.py` — `CacheEntrySigner.SIGNED_FIELDS_BY_VERSION`, `sign_entry()`, `verify_entry()`
- `src/cacheness/storage/blob_store.py` — `BlobStore.verify_integrity()`
- `src/cacheness/_verification_mixin.py` — `verify_integrity()` (delegates to BlobStore), `_verify_entry()`, `_extract_signable_fields()`
- `src/cacheness/interfaces.py` — `IntegrityReport` dataclass
- `tests/test_cache_integrity_verification.py` — existing integrity tests

## Decisions
- D-01: Add `verify_signatures` parameter to `verify_integrity()` (not a separate method)
- D-02: Add `signature_failures` field to `IntegrityReport` (follows existing pattern of optional fields)
- D-03: Signature verification in `verify_integrity()` reuses existing `CacheEntrySigner.verify_entry()` logic
