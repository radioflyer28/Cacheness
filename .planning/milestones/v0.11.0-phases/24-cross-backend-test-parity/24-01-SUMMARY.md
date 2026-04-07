---
phase: 24-cross-backend-test-parity
plan: 01
status: complete
requirements_completed: [ENC-03]
---

## Summary

Parametrized 12 encryption integration tests across JSON, SQLite, and PostgreSQL backends in `tests/test_backend_parity.py`. Updated an outdated incompatibility comment in `tests/test_concurrent_security.py`.

## What Was Built

**File:** `tests/test_backend_parity.py` (~320 lines added)

### Helper Functions

- `_get_pg_url()` — reads `CACHENESS_TEST_POSTGRES_URL` env var
- `_generate_key_file(path)` — writes 32 random bytes for encryption key
- `_make_encrypted_cache_for_backend(tmp_path, backend, **security_overrides)` — creates UnifiedCache with encryption for any backend; PG uses `metadata_backend_options={"connection_url": url}`
- `_make_encrypted_blobstore_for_backend(tmp_path, backend, **overrides)` — creates BlobStore with encryption; PG passes PostgresBackend instance directly (BlobStore doesn't accept "postgresql" string)
- `_skip_if_pg_unavailable(backend)` — pytest.skip helper for PG tests

### Test Classes and Methods

**TestEncryptionBackendParity_BlobStore** (5 tests × 3 backends = 15):
- `test_encrypted_put_get_roundtrip` — encrypt → put → get roundtrip
- `test_encrypted_entry_metadata_has_encryption_fields` — verify `is_encrypted` and `encryption_key_id` metadata
- `test_unencrypted_entry_readable_with_encryption_enabled` — unencrypted data still readable
- `test_encrypted_entry_without_key_returns_none` — no key → returns None
- `test_encryption_disabled_by_default` — default BlobStore has no encryption

**TestEncryptionBackendParity_UnifiedCache** (5 tests × 3 backends = 15):
- `test_cache_encrypted_put_get_roundtrip` — full UnifiedCache encrypt/decrypt cycle
- `test_cache_encrypted_put_get_various_types` — multiple Python types (dict, list, int, str)
- `test_cache_encryption_disabled_by_default` — no encryption config → no encryption
- `test_cache_mixed_encrypted_unencrypted` — mixed-mode cache access
- `test_cache_init_without_cryptography_raises` — missing `cryptography` → CacheConfigurationError

**TestEncryptionBackendParity_KeyRotation** (2 tests × 3 backends = 6):
- `test_rotate_key_re_encrypts_entries` — key rotation re-encrypts all entries
- `test_rotated_encrypted_entries_readable` — data readable after rotation

All classes decorated with `@pytest.mark.xdist_group("docker")`. All tests parametrized with `@pytest.mark.parametrize("backend", ["json", "sqlite", "postgresql"])`.

**File:** `tests/test_concurrent_security.py` (2 lines changed)
- Updated docstring from "encryption+SQLite has a known incompatibility" to "Encryption works with all backends since Phase 23 added encryption schema columns"

## Key Decisions

- **BlobStore PG workaround** — BlobStore.__init__ only accepts "json", "sqlite", or a MetadataBackend instance (not "postgresql" string). PG tests create a PostgresBackend instance directly.
- **Existing tests untouched** — Per D-08, test_encryption_at_rest.py stays as JSON-only baseline; new parity tests live in test_backend_parity.py
- **xdist_group("docker")** — All 3 classes grouped for Docker resource sharing, matching existing PG test patterns

## Verification

```
# Encryption parity tests only (JSON + SQLite pass, PG skipped without Docker)
uv run pytest tests/test_backend_parity.py -x -q -k "Encryption"
24 passed, 12 skipped, 12 deselected in 7.11s

# Full test suite regression check
uv run pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py
1756 passed, 118 skipped, 27 warnings in 111.61s
```

## Commits

- `831cbc4` — feat(ENC-03): parametrize 12 encryption tests across JSON/SQLite/PostgreSQL
- `9e75baa` — fix(ENC-03): update outdated encryption+SQLite incompatibility comment
- `18273cb` — style(ENC-03): fix ruff E402 import ordering in test_backend_parity
