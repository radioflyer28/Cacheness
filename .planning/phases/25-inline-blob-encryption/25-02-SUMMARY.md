---
plan: 25-02
status: completed
commit: 5a8a8ce
---

## Summary

Added inline blob handling to `rotate_key()` and comprehensive encryption tests.

### Task 1: rotate_key() inline branch (INLINE-03)

Replaced `if not actual_path_str: continue` with an if/elif/else branch:
- **File-backed path** (unchanged): read from disk, decrypt, re-encrypt, write back
- **Inline path** (new): decrypt `blob_data` with old key, re-encrypt with new key, update `blob_data`, `encryption_iv`, `file_hash`, `file_size`
- **Neither**: skip (continue)

Hash is computed on ciphertext (consistent with verification mixin which hashes `blob_data` directly).

### Task 2: TestInlineBlobEncryption tests

Added 9 tests to `test_inline_blobs.py`:
- 4 roundtrip tests (dict, string, list, int)
- Encryption metadata verification (aes-256-gcm, IV length)
- Ciphertext verification (blob_data not deserializable as pickle)
- File hash verification (valid 16-char hex)
- Key rotation test (IV changes, blob_data changes, data still readable)
- Backward compatibility (unencrypted inline still works)

### D-05 Correction

Original plan specified "hash plaintext before encryption" but the verification mixin hashes `blob_data` directly (which is ciphertext when encrypted). Fixed to hash after encryption in both write path (`_try_direct_inline`) and rotation path (`rotate_key`), consistent with file-backed entries where the hash is computed on the ciphertext file on disk.

### Test Results
- 112 passed (35 inline + 77 core/encryption/integrity), 0 failures
