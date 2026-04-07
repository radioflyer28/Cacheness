---
phase: 26-integration-hardening
plan: 01
status: completed
started: 2025-04-07
completed: 2025-04-07
---

## Summary

Added 4 config validation checks that catch bad configuration combos at construction time, raising `CacheConfigurationError` with actionable fix suggestions.

## Changes

### src/cacheness/config.py
- Added 3 checks to `SecurityConfig.__post_init__` (after existing encryption/cryptography validation):
  - **Combo 1:** encryption + empty `encryption_key_file` (when not in-memory) 
  - **Combo 2:** encryption + `enable_entry_signing=False` (encrypted data needs integrity signing)
  - **Combo 3:** `use_in_memory_key=True` + encryption (non-persistent keys make encrypted cache unreadable)
- Added 1 check to `CacheConfig.__post_init__`:
  - **Combo 5:** JSON metadata backend + `max_inline_size > 0` (JSON doesn't support inline blobs)

### tests/test_core.py
- Added `TestConfigValidation` class with 6 tests:
  - `test_encryption_no_key_file_raises`
  - `test_encryption_signing_disabled_raises`
  - `test_in_memory_key_encryption_raises`
  - `test_json_backend_inline_blobs_raises`
  - `test_valid_encryption_config_no_error`
  - `test_error_messages_are_actionable`

## Deviations

- **Combo 2 revised:** Plan specified `encryption + allow_unsigned_entries=True` as the check. However, `allow_unsigned_entries=True` is the DEFAULT — enforcing this as a hard error would break ALL existing encryption users and tests. Revised to check `encryption + enable_entry_signing=False` instead, which catches the actually dangerous combo (encrypted data with no integrity signing) without breaking backward compatibility.
- **Combo 4 dropped:** `enable_entry_signing=True + allow_unsigned_entries=True` — both are defaults, standalone check is impractical. When encryption is active, this is subsumed by Combo 2.

## Key Files

- `src/cacheness/config.py` — 4 validation checks
- `tests/test_core.py` — 6 new tests

## Self-Check: PASSED
- All 6 TestConfigValidation tests pass
- Default SecurityConfig() and CacheConfig() construct without error
- Existing encryption tests unaffected (107 pass in Tier 1)
- Full suite: 1773 passed, 122 skipped, 0 failures
