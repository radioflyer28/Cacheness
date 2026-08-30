---
phase: 01
fixed_at: 2026-08-30T00:26:23Z
review_path: .planning/phases/01-compatibility-and-security-baseline/01-REVIEW.md
iteration: 2
findings_in_scope: 4
fixed: 4
skipped: 0
status: all_fixed
---

# Phase 01: Code Review Fix Report

**Fixed at:** 2026-08-30T00:26:23Z  
**Source review:** `.planning/phases/01-compatibility-and-security-baseline/01-REVIEW.md`  
**Iteration:** 2

## Summary

- Findings in scope: 4
- Fixed: 4
- Skipped: 0

## Fixed Issues

### CR-01: A failed digest during overwrite destroys the previously valid entry

**Files modified:** `src/cacheness/core.py`, `tests/test_cache_integrity.py`  
**Commit:** `86c886d`

Replacement payloads now use unique candidate locators. Digest validation completes
before metadata replacement; an invalid digest removes only the candidate, leaving
the old metadata and payload readable. The regression snapshots the old entry and
bytes across an injected overwrite-digest failure.

### CR-02: `BlobStore.clear()` can delete an arbitrary prefix of payloads while retaining all metadata

**Files modified:** `src/cacheness/storage/path_security.py`, `src/cacheness/storage/blob_store.py`, `tests/test_filesystem_containment.py`  
**Commit:** `e37cac8`

`clear()` now stages payloads into same-root tombstones before metadata deletion,
restores the exact metadata and payload pre-state after staged deletion or partial
metadata-clear failure, and only finalizes tombstone deletion after metadata clears.
Regressions cover every payload position, metadata deletion after one row mutates,
staging-write failure, and tombstone-finalization failure.

### CR-03: Numeric metadata filters still misclassify `nan` and `inf`

**Files modified:** `src/cacheness/core.py`, `src/cacheness/error_handling.py`, `src/cacheness/query_validation.py`, `tests/test_query_meta.py`, `tests/test_query_meta_security.py`  
**Commit:** `cc25f9d`

Non-finite caller thresholds now raise `CacheQueryValidationError` with the stable
`invalid_query_value` reason before a SQLite session opens. SQLite comparisons accept
only JSON scalar integer/real suffixes whose floating representation is finite, so
stored `nan` and infinities cannot enter threshold results. Regressions cover NaN,
both infinities, signed zero, exponent notation, and finite boundary magnitudes.

### CR-04: Signing failure commits an unsigned entry even when unsigned entries are forbidden

**Files modified:** `src/cacheness/core.py`, `tests/test_cache_integrity.py`, `tests/test_legacy_array_security.py`  
**Commit:** `944c381`

Strict signing now rejects unavailable signers, signer exceptions, and empty signer
results before metadata publication, deleting the uncommitted candidate only. This
preserves an overwritten entry's exact metadata and bytes. Explicit
`allow_unsigned_entries=True` remains the compatibility policy that permits unsigned
publication; the trusted object-array configuration inherits the strict behavior.

## Verification

Verification ran in the **main checkout**; no isolated-worktree environment was
used.

- Combined Phase 1 matrix passed:
  `uv run pytest -q -o log_cli=false tests/test_cache_integrity.py tests/test_filesystem_containment.py tests/test_blob_backend_registry.py tests/test_legacy_array_security.py tests/test_query_meta.py tests/test_query_meta_security.py tests/test_config_validation.py tests/test_phase1_quality_gates.py -x`
  (one expected Windows-junction skip).
- Full suite passed: `uv run pytest -q -o log_cli=false` (exit 0). Expected optional
  PostgreSQL/TensorFlow skips, one Windows-junction skip, the collection warning for
  the dataclass test fixture, and the known shutdown-only `SqliteBackend.__del__`
  warning remained.
- Phase quality gate passed: `tests/test_phase1_quality_gates.py` (7 tests).
- Ruff baseline: `uv run ruff check src tests --output-format concise` reported 118
  existing findings, matching the documented Phase 1 baseline.
- Parse checks and `git diff --check` passed before each fix; final `git diff --check`
  also passed.

## Post-verification Contract Adjustment

The full suite initially exposed that the public `CacheReason` exact-set assertion
did not include CR-03's new `invalid_query_value` reason. The working tree therefore
also updates `tests/test_public_api_contract.py`; its focused contract/query run and
the subsequent full suite pass. This companion test-only adjustment is intentionally
uncommitted for the orchestrator to review with this report.

---

_Fixed: 2026-08-30T00:26:23Z_  
_Fixer: the agent (gsd-code-fixer)_  
_Iteration: 2_
