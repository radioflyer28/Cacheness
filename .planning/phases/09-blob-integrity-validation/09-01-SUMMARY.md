---
phase: 09-blob-integrity-validation
plan: 01
status: complete
---

# Summary: Blob Integrity Validation

## What Was Built

End-to-end blob integrity validation via `verify_integrity(verify_signatures=True)`. Confirms that `file_hash` is covered by the HMAC signature (both v1 and v2 fields), and adds HMAC signature verification to the bulk integrity audit.

## Files Created/Modified

| File | Change |
|------|--------|
| `src/cacheness/interfaces.py` | Added `signature_failures` field to `IntegrityReport` |
| `src/cacheness/storage/blob_store.py` | Added `verify_signatures` parameter to `verify_integrity()` |
| `src/cacheness/_verification_mixin.py` | Signature verification loop using `_extract_signable_fields()` for proper normalization |
| `tests/test_cache_integrity_verification.py` | 5 new tests in `TestVerifyIntegritySignatures` class |

## Key Decisions

- **Signature verification in mixin, not blob_store**: Needs `_extract_signable_fields()` for `created_at` timezone normalization consistency with signing path
- **No new `blob_hmac` field needed**: `file_hash` is already in signed fields, so HMAC covers blob integrity transitively
- **`verify_signatures=False` by default**: Backward compatible — no behavior change unless explicitly requested

## Test Results

- 5 new tests: confirmatory (file_hash in signed fields), signature failure detection, clean signature, default behavior, end-to-end blob tampering
- Full suite: 1636 passed, 101 skipped, 0 failures
