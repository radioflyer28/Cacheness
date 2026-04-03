---
phase: 16-hkdf-key-derivation
plan: 01
status: complete
started: 2026-04-03T18:25:00Z
completed: 2026-04-03T18:40:00Z
---

## Summary

Implemented per-namespace cryptographic key isolation via HKDF-SHA256 (RFC 5869). Each namespace now derives its own signing key from the master key + namespace ID, preventing cross-namespace key compromise.

## Tasks Completed

### Task 1: HKDF derivation + config + signature versioning
- Added `_hkdf_sha256()` module-level function (stdlib hmac+hashlib, RFC 5869)
- Added v3 to `SIGNED_FIELDS_BY_VERSION` (same fields as v2, different key)
- `CacheEntrySigner.__init__` accepts `namespace_id` and `use_hkdf_derivation`
- Stores `self.master_key` and `self.derived_key` (HKDF output)
- `sign_entry()` uses derived_key, produces `v3:` when HKDF enabled
- `verify_entry()` routes v1/v2 → master_key, v3 → derived_key
- `sign_namespace()` produces `ns2:` (HKDF) or `ns1:` (shared key)
- `verify_namespace()` handles both ns1 and ns2 formats
- `create_cache_signer()` factory passes namespace_id and use_hkdf_derivation
- `get_field_info()` includes hkdf and namespace info
- `SecurityConfig` gains `use_hkdf_derivation: bool = True`
- **Commit:** `44011e7`

### Task 2: Wire namespace_id through core.py and blob_store.py
- `core.py._init_entry_signer` passes `namespace_id=self.namespace` and `use_hkdf_derivation=self.config.security.use_hkdf_derivation`
- `blob_store.py.BlobStore.__init__` accepts `use_hkdf_derivation` parameter
- `blob_store.py._init_signer` passes `namespace_id=self._namespace` and `use_hkdf_derivation`
- **Commit:** `e582d5a`

### Task 3: Tests + docs
- Created `tests/test_hkdf_derivation.py` with 14 tests across 4 classes
- Updated `tests/test_namespace_signing.py` — ns1: assertions → accept ns2: (HKDF default)
- Added "Per-Namespace Key Derivation (HKDF)" section to `docs/SECURITY.md`
- **Commit:** `3d9ac0b`

## Key Files

### Created
- `tests/test_hkdf_derivation.py` — 14 new HKDF tests

### Modified
- `src/cacheness/security.py` — HKDF function, v3 signing, ns2 namespaces, factory params
- `src/cacheness/config.py` — `use_hkdf_derivation` field on SecurityConfig
- `src/cacheness/core.py` — namespace_id wiring in _init_entry_signer
- `src/cacheness/storage/blob_store.py` — namespace_id + hkdf wiring in _init_signer
- `tests/test_namespace_signing.py` — updated assertions for ns2 default
- `docs/SECURITY.md` — HKDF documentation section

## Deviations from Plan

**[Rule 1 - Bug] Existing namespace signing tests hardcoded ns1: prefix** — Found during Task 3. The `test_namespace_signing.py` tests had assertions checking `startswith("ns1:")` which broke with HKDF enabled by default (produces `ns2:`). Fixed by updating to `startswith(("ns1:", "ns2:"))` for integration tests and `startswith("ns")` for the unit roundtrip test.

**Total deviations:** 1 auto-fixed (bug). **Impact:** Minimal — test assertion updates only, no production code affected.

## Test Results

- **Full suite:** 1678 passed, 101 skipped, 0 failures
- **New tests:** 14 (test_hkdf_derivation.py)
- **Regression:** None

## Self-Check: PASSED
