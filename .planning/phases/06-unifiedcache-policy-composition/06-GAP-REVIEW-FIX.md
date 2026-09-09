---
phase: 06-unifiedcache-policy-composition
fixed_at: 2026-09-09T09:04:38Z
review_path: .planning/phases/06-unifiedcache-policy-composition/06-GAP-REVIEW.md
findings_in_scope: 5
fixed: 5
skipped: 0
status: all_fixed
---

# Phase 6: Gap Review Fix Report

**Fixed at:** 2026-09-09T09:04:38Z
**Source review:** `.planning/phases/06-unifiedcache-policy-composition/06-GAP-REVIEW.md`

## Summary

- Findings in scope: 5
- Fixed: 5
- Skipped: 0

## Fixed Issues

### CR-01: The migrated signing suite no longer exercises signing behavior

**Files modified:** `src/cacheness/config.py`, `tests/test_cache_signing.py`, `tests/test_legacy_array_security.py`, `tests/test_config_validation.py`, `tests/test_phase6_public_api_contract.py`
**Commit:** `2f9f5dc`

Replaced configuration-value tests with a direct BlobStore authority-manifest
HMAC mutation test. It asserts the typed unauthenticated-manifest failure before
payload access and proves the original authority entry and manifest key bytes
are retained. Removed only the four SecurityConfig fields with no runtime
consumer (`signing_key_file`, `use_in_memory_key`, `signature_version`, and
`delete_invalid_signatures`) and migrated their direct tests. BlobStore remains
the manifest-signing authority; no legacy signer authority was restored.

### CR-02: The order-isolation matrix never runs SqlCache before the Phase 6 public contract

**Files modified:** `tests/test_phase6_suite_isolation.py`
**Commit:** `4ae973b`

Added a genuine reverse order beginning with SqlCache and a pair-order assertion
which proves every public/Phase-6/SqlCache pair executes in both directions.

### CR-03: A symlinked live module can turn the exact-three runner into an unrelated exclusion

**Files modified:** `tools/run_phase6_local_suite.py`, `tests/test_phase6_suite_isolation.py`
**Commits:** `9dd5ca7`, `5c48048`

The runner now rejects symlinked and non-regular lexical configured entries via
`lstat`, and forms ignores from validated literal paths. The regression test
uses a live-module symlink to an unrelated repository test. The closure fix
also checks every ancestor with `lstat`, requires every ancestor to be a real
directory, and confirms the strict resolved candidate remains under the
resolved repository root. Its regression redirects `tests/integration` to an
outside directory containing the expected basenames and fails before pytest.

### CR-04: Canonical inventory completeness is checked against an oracle derived from itself

**Files modified:** `tools/verify_phase6_contracts.py`, `tests/test_phase6_contract_verifier.py`
**Commit:** `3e0c302`

Added an independent Plan 09-11 literal inventory oracle and derives the
executed inventory from it. The self-test now executes modified verifier source
that omits a canonical node, modeling initialization after a source omission.

### CR-05: The compatibility AST audit misses aliases of a known UnifiedCache

**Files modified:** `tools/verify_phase6_contracts.py`, `tests/test_phase6_contract_verifier.py`
**Commit:** `c8b1e2f`

The AST audit now propagates proven cache aliases and recognizes helpers that
return or yield known UnifiedCache values. Mutation tests cover both paths and
also confirm an unproven receiver is not falsely treated as a cache.

## Verification

Verification ran in the main checkout because `workflow.use_worktrees` is
`false`.

- Focused reviewed scope: 282 passed, 2 platform-specific skips.
- Final CR-01 focused scope: 86 passed.
- CR-03 ancestor-hardening isolation scope: 11 passed.
- Fixed Phase 6 verifier: completed successfully in the isolated project
  environment.
- Scoped Ruff across reviewed and modified paths: passed.
- Complete non-live suite: attempted with the documented isolated, frozen,
  all-extras environment but did not pass due environment failures outside the
  changes: six Moto/S3 setup errors from missing Botocore `endpoints` data;
  spawned workers selecting Homebrew Python 3.11 and missing `pytest` or
  `_sqlite3`; and resulting Python-ABI/import failures in subprocess and SQL
  tests. No lifecycle, concurrency, dependency, or topology behavior was
  changed to mask these failures.

---

_Fixed: 2026-09-09T09:04:38Z_
_Fixer: the agent (gsd-code-fixer)_
