---
phase: 31-security-storage-mode-posture
plan: 04
subsystem: security
tags: [signing, hmac, blob-store, compatibility, canonical-fields]

requires:
  - phase: 31-security-storage-mode-posture
    provides: SEC-01 minimum signature-version policy and SEC-02 BlobStore read hardening
provides:
  - Shared canonical signing field extraction for UnifiedCache and BlobStore
  - BlobStore new-entry signing through the canonical helper
  - Explicit legacy flattened BlobStore signature verification fallback
  - SEC-04 regressions for canonical field parity, legacy compatibility, and minimum-version interaction
affects: [security, signing, blob-store, unified-cache, key-rotation, phase-31]

tech-stack:
  added: []
  patterns:
    - Shared signing-field helper module
    - Canonical-first verification with named legacy compatibility fallback

key-files:
  created:
    - src/cacheness/signing_fields.py
    - .planning/phases/31-security-storage-mode-posture/31-04-SUMMARY.md
  modified:
    - src/cacheness/_verification_mixin.py
    - src/cacheness/storage/blob_store.py
    - tests/test_cache_signing.py
    - tests/test_blob_store.py
    - .planning/phases/31-security-storage-mode-posture/deferred-items.md

key-decisions:
  - "SEC-04 canonical signing fields live in src/cacheness/signing_fields.py and are shared by UnifiedCache and BlobStore."
  - "BlobStore new writes use canonical fields; old flattened BlobStore signatures are accepted only through an explicit legacy verifier."
  - "Minimum signature-version policy remains signer-level, so legacy compatibility cannot bypass a stricter configured minimum."

patterns-established:
  - "Use extract_signable_fields(cache_key, entry, metadata) for new cache-entry signatures."
  - "Use extract_legacy_blobstore_signable_fields only for read-time compatibility with pre-SEC-04 BlobStore signatures."

requirements-completed: [SEC-04]

duration: 8min
completed: 2026-06-15
---

# Phase 31 Plan 04: Shared Canonical Signing Fields Summary

**UnifiedCache and BlobStore now sign new metadata through one canonical field helper while preserving old flattened BlobStore signatures through an explicit compatibility verifier.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-06-15T02:54:02Z
- **Completed:** 2026-06-15T03:02:01Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- Added SEC-04 regressions for UnifiedCache/BlobStore canonical field-set parity, old flattened BlobStore signatures, and minimum-version interaction with legacy signatures.
- Created `src/cacheness/signing_fields.py` with the shared canonical extraction helper and a named legacy BlobStore shape helper.
- Updated UnifiedCache verification to delegate to the shared helper through its existing `_extract_signable_fields()` method.
- Updated BlobStore `put()` and rotation re-signing to use canonical fields for new signatures.
- Added BlobStore canonical-first signature verification with explicit fallback for old flattened signatures.

## Task Commits

1. **Task 1: Add canonical-signing parity and compatibility regressions** - `d1d35dd` (test)
2. **Task 2: Share canonical signing helper and wire BlobStore compatibility** - `8e118a6` (feat)

**Plan metadata:** pending final docs commit or skipped by GSD commit helper.

## Files Created/Modified

- `src/cacheness/signing_fields.py` - New shared signing-field helper plus explicit legacy BlobStore shape helper.
- `src/cacheness/_verification_mixin.py` - Keeps the existing mixin method but delegates canonical extraction to the shared helper.
- `src/cacheness/storage/blob_store.py` - Uses canonical fields for new signatures and verifies legacy flattened signatures through a named fallback.
- `tests/test_cache_signing.py` - Adds UnifiedCache/BlobStore canonical field-set parity coverage.
- `tests/test_blob_store.py` - Adds old flattened signature compatibility and minimum-version interaction regressions.
- `.planning/phases/31-security-storage-mode-posture/deferred-items.md` - Records environment and pre-existing type-check caveats.

## Decisions Made

- Preserved `VerificationMixin._extract_signable_fields()` as a delegating method so existing internal call sites remain stable.
- Kept legacy BlobStore compatibility verify-only; old entries are not re-signed on read.
- Let `CacheEntrySigner.verify_entry()` remain the sole minimum-version enforcement point, including compatibility fallback attempts.

## Deviations from Plan

None - plan executed as written. Environment verification caveats were documented in deferred items.

## Issues Encountered

- The literal plan pytest command fails before collection in this Windows environment because xdist cannot access `C:\Users\akriz\AppData\Local\Temp\pytest-of-akriz`. Controlled equivalent commands with addopts cleared, pytest cache disabled, and repo-local `--basetemp` passed.
- `ty check` still reports pre-existing diagnostics in `_verification_mixin.py`, `blob_store.py`, `test_namespace_signing.py`, and `test_cross_system_compatibility.py`. New `signing_fields.py` diagnostics were fixed before commit.
- `gsd-tools query state.advance-plan`, `state.update-progress`, and `state.record-session` could not parse or update this project's custom STATE.md shape. Working handlers updated metrics, decisions, roadmap, and requirements; duplicated metric/decision prefixes and the stale current-position text were patched manually.

## Verification

- RED: `uv run --python 3.12 pytest -o addopts='' -p no:cacheprovider --basetemp .uv-cache\pytest-31-04-red-1 tests/test_cache_signing.py tests/test_blob_store.py tests/test_namespace_signing.py -x -q --ignore=tests/test_tensorflow_handler.py` - FAILED as expected on missing `cacheness.signing_fields`.
- Task 2 gate 1: `uv run --python 3.12 pytest -o addopts='' -p no:cacheprovider --basetemp .uv-cache\pytest-31-04-green-1 tests/test_cache_signing.py tests/test_blob_store.py tests/test_namespace_signing.py -x -q --ignore=tests/test_tensorflow_handler.py` - PASSED, 75 passed.
- Task 2 gate 2: `uv run --python 3.12 pytest -o addopts='' -p no:cacheprovider --basetemp .uv-cache\pytest-31-04-green-2 tests/test_cache_signing.py tests/test_blob_store.py tests/test_cross_system_compatibility.py -x -q --ignore=tests/test_tensorflow_handler.py` - PASSED, 66 passed.
- Plan gate: `uv run --python 3.12 pytest -o addopts='' -p no:cacheprovider --basetemp .uv-cache\pytest-31-04-plan-gate tests/test_cache_signing.py tests/test_blob_store.py tests/test_namespace_signing.py tests/test_cross_system_compatibility.py -x -q --ignore=tests/test_tensorflow_handler.py` - PASSED, 83 passed.
- Literal plan command: `uv run --python 3.12 pytest tests/test_cache_signing.py tests/test_blob_store.py tests/test_namespace_signing.py -x -q --ignore=tests/test_tensorflow_handler.py` - FAILED before collection on Windows xdist temp ACL.
- `uv run --python 3.12 ruff format src/cacheness/signing_fields.py src/cacheness/_verification_mixin.py src/cacheness/storage/blob_store.py tests/test_cache_signing.py tests/test_blob_store.py tests/test_namespace_signing.py tests/test_cross_system_compatibility.py` - PASSED.
- `uv run --python 3.12 ruff check --fix src/cacheness/signing_fields.py src/cacheness/_verification_mixin.py src/cacheness/storage/blob_store.py tests/test_cache_signing.py tests/test_blob_store.py tests/test_namespace_signing.py tests/test_cross_system_compatibility.py` - PASSED.
- `uv run --python 3.12 ruff check src/cacheness/signing_fields.py src/cacheness/_verification_mixin.py src/cacheness/storage/blob_store.py tests/test_cache_signing.py tests/test_blob_store.py tests/test_namespace_signing.py tests/test_cross_system_compatibility.py` - PASSED.
- `uv run --python 3.12 ty check src/cacheness/signing_fields.py src/cacheness/_verification_mixin.py src/cacheness/storage/blob_store.py tests/test_cache_signing.py tests/test_blob_store.py tests/test_namespace_signing.py tests/test_cross_system_compatibility.py` - FAILED on pre-existing diagnostics documented in deferred items; no `signing_fields.py` diagnostics remain.

## Known Stubs

None. Stub scan only matched ordinary optional defaults, internal empty collections, and test assertions.

## Threat Flags

None. This plan narrows an existing metadata-signature trust boundary and does not add endpoints, auth paths, schema changes, or new external file-access surfaces.

## User Setup Required

None - no external service configuration required.

## TDD Gate Compliance

- RED commit exists: `d1d35dd` (`test(31-04): add canonical signing regressions`)
- GREEN commit exists after RED: `8e118a6` (`feat(31-04): share canonical signing fields`)

## Next Phase Readiness

SEC-04 is complete. Plan 31-03 can rotate keys using the shared canonical helper without preserving BlobStore's old flattened signing shape for new signatures.

## Self-Check: PASSED

- Verified summary and key source/test files exist.
- Verified task commits `d1d35dd` and `8e118a6` are present in git history.
- Verified task commits did not delete tracked files.

---
*Phase: 31-security-storage-mode-posture*
*Completed: 2026-06-15*
