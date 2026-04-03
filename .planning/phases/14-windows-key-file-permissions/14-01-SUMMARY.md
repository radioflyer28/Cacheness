# Phase 14: Windows Key File Permissions — Summary

## Goal
Make signing key file permissions effective on Windows (where `chmod 0o600` is a no-op).

## Changes

### `src/cacheness/security.py`
- Added `_set_key_file_permissions()` static method with cross-platform logic
- **Windows:** Uses `icacls` to remove inherited permissions and grant only the current user `(R,W)`
- **Unix/macOS:** Uses `chmod(0o600)` as before
- Both paths are best-effort with warning on failure

### `tests/test_namespace_signing.py`
- Added `TestKeyFilePermissions` class with 2 tests:
  1. `test_key_file_permissions_applied_on_generate` — verifies ACL is set when key is generated
  2. `test_set_key_file_permissions_static_method` — tests the static method directly

### `docs/SECURITY.md`
- Updated Key Management section to document both Unix and Windows permission behavior

## Verification
- 2 new tests pass on Windows
- All existing security/signing tests still pass (0 regressions)
