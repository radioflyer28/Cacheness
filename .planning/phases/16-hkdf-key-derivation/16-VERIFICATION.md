---
status: passed
phase: 16-hkdf-key-derivation
requirement_ids: [SEC-01]
verified: 2026-04-03
---

# Phase 16 Verification: HKDF Key Derivation

## Requirement Coverage

### SEC-01: Per-namespace key derivation via HKDF
**Status:** PASSED

| Criterion | Evidence | Status |
|-----------|----------|--------|
| Derive namespace-specific signing keys from master key + namespace ID | `_hkdf_sha256()` in security.py:38, `self.derived_key` in CacheEntrySigner.__init__ | ✅ |
| Existing entries with shared key must still verify | `verify_entry()` routes v1/v2 → `self.master_key` | ✅ |
| New entries use derived key | `sign_entry()` uses `self.derived_key`, produces v3 | ✅ |
| Different namespaces produce different signatures | test_different_namespaces_different_signatures passes | ✅ |
| Opt-out path (disable HKDF) | `use_hkdf_derivation=False` → v2 signatures, test confirms | ✅ |
| BlobStore also gets derived keys | blob_store.py passes namespace_id to create_cache_signer | ✅ |

## Must-Haves Verification

| Truth | Verified | Evidence |
|-------|----------|----------|
| Each namespace derives its own signing key via HKDF-SHA256 | ✅ | `_hkdf_sha256()` called with namespace_id in info parameter |
| Entries signed with shared master key (v2) still verify | ✅ | test_v2_entries_verify_after_hkdf_enabled passes |
| New entries in different namespaces produce different signatures | ✅ | test_different_namespaces_different_signatures passes |
| Disabling use_hkdf_derivation reverts to shared-key behavior | ✅ | test_hkdf_disabled_uses_v2 passes |
| BlobStore gets HKDF-derived keys transparently | ✅ | blob_store.py:247 passes namespace_id=self._namespace |
| Namespace signatures use derived key (ns2 format) | ✅ | test_ns2_format_when_hkdf_enabled passes |

## Artifact Verification

| Artifact | Contains | Verified |
|----------|----------|----------|
| src/cacheness/security.py | `def _hkdf_sha256` | ✅ (line 38) |
| src/cacheness/config.py | `use_hkdf_derivation` | ✅ (line 412) |
| tests/test_hkdf_derivation.py | 14 tests across 4 classes | ✅ |

## Key-Link Verification

| From | To | Pattern | Verified |
|------|----|---------|----------|
| core.py | security.py | `namespace_id=self\.namespace` | ✅ (line 340) |
| blob_store.py | security.py | `namespace_id=self\._namespace` | ✅ (line 247) |

## Test Results

- **Full suite:** 1678 passed, 101 skipped, 0 failures
- **New tests:** 14 (tests/test_hkdf_derivation.py)
- **Baseline delta:** +14 tests (1664 → 1678)
- **Regressions:** None

## Score

**6/6 must-haves verified. All requirements satisfied.**
