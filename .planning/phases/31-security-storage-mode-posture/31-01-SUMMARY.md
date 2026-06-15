---
phase: 31-security-storage-mode-posture
plan: 01
subsystem: security
tags: [signing, hmac, hkdf, cache-integrity, documentation]

requires:
  - phase: 31-security-storage-mode-posture
    provides: Phase 31 security posture context and SEC-01 decisions
provides:
  - SecurityConfig.minimum_signature_version with compatibility default
  - Central minimum signature-version enforcement in CacheEntrySigner.verify_entry
  - UnifiedCache and BlobStore signer construction using the same minimum-version policy
  - SEC-01 regression tests for v2 compatibility, strict v3 rejection, and downgraded prefixes
  - Security docs for unsigned-entry stripping risk and strict v3 guidance
affects: [security, signing, blob-store, unified-cache, phase-31]

tech-stack:
  added: []
  patterns:
    - Centralized signer policy enforcement through CacheEntrySigner
    - Compatibility-default security config with opt-in strict mode

key-files:
  created:
    - .planning/phases/31-security-storage-mode-posture/deferred-items.md
  modified:
    - src/cacheness/config.py
    - src/cacheness/security.py
    - src/cacheness/core.py
    - src/cacheness/storage/blob_store.py
    - docs/SECURITY.md
    - tests/test_cache_signing.py
    - tests/test_key_rotation_api.py

key-decisions:
  - "SEC-01 keeps minimum_signature_version defaulting to 1 for old-cache compatibility."
  - "New deployments are documented to use minimum_signature_version=3 with allow_unsigned_entries=False where metadata may be attacker-writable."
  - "Minimum-version rejection is implemented in CacheEntrySigner.verify_entry so UnifiedCache and BlobStore share the policy."

patterns-established:
  - "Signer-level policy: verification policy belongs in CacheEntrySigner, not duplicated in every caller."
  - "Strict signing posture: allow_unsigned_entries=False plus minimum_signature_version=3 for attacker-writable metadata stores."

requirements-completed: [SEC-01]

duration: 12min
completed: 2026-06-15
---

# Phase 31 Plan 01: Signature Minimum Version Summary

**Configurable signature-version floor with strict v3 downgrade rejection and explicit unsigned-entry stripping guidance**

## Performance

- **Duration:** 12 min
- **Started:** 2026-06-15T01:47:59Z
- **Completed:** 2026-06-15T01:59:40Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- Added `SecurityConfig.minimum_signature_version` with default `1`, preserving legacy v1/v2 compatibility.
- Enforced the configured minimum inside `CacheEntrySigner.verify_entry()` before payload/key selection.
- Threaded the policy through UnifiedCache and BlobStore signer creation, including rotation-created signer instances.
- Added SEC-01 tests for default compatibility, strict v3 rejection of valid v2 signatures, and downgraded `v3:` to `v2:` prefixes.
- Updated `docs/SECURITY.md` to recommend `minimum_signature_version=3` for new deployments and document metadata-write signature stripping risk when `allow_unsigned_entries=True`.

## Task Commits

1. **Task 1: Add SEC-01 downgrade and compatibility regressions** - `6e7d745` (test)
2. **Task 2: Enforce minimum signature version and document unsigned-entry risk** - `c26c112` (feat)

**Plan metadata:** pending final docs commit

## Files Created/Modified

- `src/cacheness/config.py` - Adds and validates `SecurityConfig.minimum_signature_version`; supports flat `CacheConfig(minimum_signature_version=...)`.
- `src/cacheness/security.py` - Stores signer minimum version and rejects parsed signature versions below it.
- `src/cacheness/core.py` - Passes the configured minimum version into UnifiedCache signer construction and rotation signer construction.
- `src/cacheness/storage/blob_store.py` - Passes the configured minimum version into BlobStore signer construction and rotation signer construction.
- `docs/SECURITY.md` - Documents strict v3 guidance and unsigned-entry stripping risk.
- `tests/test_cache_signing.py` - Adds direct signer downgrade and default compatibility regression coverage.
- `tests/test_key_rotation_api.py` - Adds end-to-end strict-minimum read rejection for legacy v2 signatures.
- `.planning/phases/31-security-storage-mode-posture/deferred-items.md` - Records unrelated verification findings.

## Decisions Made

- Kept old-cache compatibility by default with `minimum_signature_version=1`.
- Used signer-level enforcement so every verifier path that calls `CacheEntrySigner.verify_entry()` receives the same policy.
- Documented `allow_unsigned_entries=True` as a compatibility posture, not a defense against metadata-write attackers.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Threaded SEC-01 policy through UnifiedCache construction**
- **Found during:** Task 2 (Enforce minimum signature version and document unsigned-entry risk)
- **Issue:** The plan's task file list omitted `src/cacheness/core.py`, but UnifiedCache constructs both the active signer and rotation signer there. Without updating it, UnifiedCache would silently keep the default minimum even when configured otherwise.
- **Fix:** Passed `self.config.security.minimum_signature_version` into both UnifiedCache signer factory calls.
- **Files modified:** `src/cacheness/core.py`
- **Verification:** Scoped SEC-01 tests pass; strict v3 read rejects legacy v2 entries.
- **Committed in:** `c26c112`

---

**Total deviations:** 1 auto-fixed (Rule 2)
**Impact on plan:** Required for SEC-01 correctness. No Phase 32 work or default behavior change was introduced.

## Issues Encountered

- `uv` failed against the default Windows cache path, so verification and quality commands were rerun with repo-local `UV_CACHE_DIR` / `UV_PYTHON_INSTALL_DIR`.
- The exact plan pytest command expands to the broader suite because of repository pytest configuration and fails outside this plan on `tests/test_compress_pickle.py` when `blosc2` is unavailable. The scoped SEC-01 run passes.
- `ty check` on the plan's touched-file list still reports pre-existing diagnostics in `_verification_mixin.py`, `config.py`, and `blob_store.py`. Introduced test typing diagnostics were resolved.

## Verification

- `uv run --python 3.12 pytest -o addopts='' tests/test_cache_signing.py tests/test_key_rotation_api.py -x -q --ignore=tests/test_tensorflow_handler.py` - PASSED, 24 passed.
- `uv run --python 3.12 pytest tests/test_cache_signing.py tests/test_key_rotation_api.py -x -q --ignore=tests/test_tensorflow_handler.py` - FAILED outside plan scope after broad addopts expansion, 7 `tests/test_compress_pickle.py` failures due missing `blosc2`.
- `uv run --python 3.12 ruff format src/cacheness/config.py src/cacheness/security.py src/cacheness/_verification_mixin.py src/cacheness/storage/blob_store.py tests/test_cache_signing.py tests/test_key_rotation_api.py` - PASSED.
- `uv run --python 3.12 ruff check --fix src/cacheness/config.py src/cacheness/security.py src/cacheness/_verification_mixin.py src/cacheness/storage/blob_store.py tests/test_cache_signing.py tests/test_key_rotation_api.py` - PASSED.
- `uv run --python 3.12 ruff check src/cacheness/config.py src/cacheness/security.py src/cacheness/_verification_mixin.py src/cacheness/storage/blob_store.py tests/test_cache_signing.py tests/test_key_rotation_api.py` - PASSED.
- `uv run --python 3.12 ty check src/cacheness/config.py src/cacheness/security.py src/cacheness/_verification_mixin.py src/cacheness/storage/blob_store.py tests/test_cache_signing.py tests/test_key_rotation_api.py` - FAILED on pre-existing type diagnostics documented in deferred items.

## Known Stubs

None. Stub scan only matched existing log/error strings containing "not available".

## Threat Flags

None. This plan narrows an existing metadata-signature trust boundary and does not add new endpoints, auth paths, file-access surfaces, or schema changes.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

SEC-01 is ready for downstream Phase 31 signing work. Plan 31-04 can rely on signer-level minimum-version enforcement while it converges canonical signing fields.

## Self-Check: PASSED

- Verified summary and deferred-items files exist.
- Verified all key source, docs, and test files exist.
- Verified task commits `6e7d745` and `c26c112` are present in git history.

---
*Phase: 31-security-storage-mode-posture*
*Completed: 2026-06-15*
