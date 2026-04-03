# Requirements: v0.9.0 Completeness & Hardening

## Management Operations

### MGMT-01: put_batch() API
**Priority:** Must-have
**Description:** Implement `put_batch()` on UnifiedCache for batch put operations. The existing `get_batch()`, `delete_batch()`, and `touch_batch()` APIs are already implemented — `put_batch()` is the only missing batch operation.
**Acceptance Criteria:**
- `put_batch(items)` accepts a list of (key_kwargs, data) tuples and stores them efficiently
- Uses backend-level transactions where available (SQLite)
- Returns a list of results (success/failure per item)
- Tests cover JSON and SQLite backends, multiple data types, error handling

### MGMT-02: Update CONCERNS.md for Implemented APIs
**Priority:** Must-have
**Description:** The CONCERNS.md "Missing management operations" section lists `update_blob_data()`, `touch()`, `get_metadata()`, `get_batch()`, `delete_batch()` as missing — but all were implemented in v0.8.0. Update CONCERNS.md to reflect reality after `put_batch()` ships.
**Acceptance Criteria:**
- CONCERNS.md accurately reflects current implementation state
- Implemented APIs are marked as addressed with version references

## Documentation

### DOC-01: Threading Model Documentation
**Priority:** Must-have
**Description:** Fix contradictory threading documentation. API_REFERENCE.md says "thread-safe for all operations" while TROUBLESHOOTING.md says "not thread-safe". Both are outdated after v0.8.0 RLock work. Document the actual concurrency model accurately.
**Acceptance Criteria:**
- API_REFERENCE.md Thread Safety section reflects post-v0.8.0 reality (RLock in management ops, backend-level protection for put/get)
- TROUBLESHOOTING.md Thread Safety Issues section is consistent with API_REFERENCE.md
- Concurrency boundaries clearly documented: what's safe, what's not, which backends provide what guarantees
- No contradictions between documentation files

## Security

### SEC-01: Pickle/Dill Deserialization Safety
**Priority:** Must-have
**Description:** 7 unprotected `pickle.loads()` and `dill.loads()` calls exist across `compress_pickle.py` and `handlers/object_handler.py`. While HMAC signing covers metadata, blob content can be tampered with independently. Add defense-in-depth measures.
**Acceptance Criteria:**
- Document the threat model clearly (SECURITY.md): signing covers metadata + file_hash, so blobs are protected when verify_hashes=True (default since v0.7.0)
- Add explicit security warnings at each deserialization call site as code comments
- Add a "Deserialization Security" section to SECURITY.md documenting the layered defense
- Verify that verify_hashes=True (default) prevents blob tampering — add a test that modifies a blob file and confirms get() rejects it

### SEC-02: Windows Key File Permissions
**Priority:** Should-have
**Description:** `chmod(0o600)` on the signing key file is a no-op on Windows. The key file may be readable by other local users. Use Windows-native mechanisms or clearly document the limitation.
**Acceptance Criteria:**
- On Windows, use `icacls` to restrict key file permissions to the current user only
- Graceful fallback if `icacls` fails (log warning, continue)
- Test that verifies the permission-setting code path runs on Windows
- Document the Windows key file permission behavior in SECURITY.md
