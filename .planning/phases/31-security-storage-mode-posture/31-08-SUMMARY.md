---
phase: 31-security-storage-mode-posture
plan: 08
subsystem: security
tags: [key-rotation, signing, encryption, interrupted-rotation, blob-store]

requires:
  - phase: 31-security-storage-mode-posture
    provides: SEC-03 two-phase key rotation staging from Plan 31-03
provides:
  - Hard-interruption regressions for staged-key metadata before active-key replacement
  - UnifiedCache staged signer fallback for interrupted rotation reads
  - BlobStore staged signer fallback with legacy signature compatibility
  - Encrypted BlobStore read fallback using staged encryption key after active decrypt failure
affects: [security, signing, encryption, blob-store, unified-cache, phase-31]

tech-stack:
  added: []
  patterns:
    - Active-key-first verification with staged-key fallback only while <keyfile>.new exists
    - Active-key-first encrypted read with staged encryption fallback only after decrypt failure
    - Hard-interruption tests using BaseException to bypass controlled rollback

key-files:
  created:
    - .planning/phases/31-security-storage-mode-posture/31-08-SUMMARY.md
  modified:
    - src/cacheness/core.py
    - src/cacheness/_verification_mixin.py
    - src/cacheness/storage/blob_store.py
    - tests/test_key_rotation_api.py
    - tests/test_encryption_at_rest.py
    - tests/test_blob_store.py

key-decisions:
  - "Interrupted rotation fallback accepts staged signatures only when the protected sibling <keyfile>.new still exists."
  - "Active signer and active encryption key remain the normal read path; staged signer/decryption are fallback-only."
  - "Startup does not replace keys, delete <keyfile>.new, rewrite metadata, or complete rotation implicitly."

patterns-established:
  - "Hard-interruption regression pattern: raise BaseException from os.replace(<keyfile>.new, active_key) while allowing .rotating blob publishes."
  - "Staged-key fallback pattern: construct a private signer from <keyfile>.new and use it only after active-key verification fails."

requirements-completed: [SEC-03]

duration: 11min
completed: 2026-06-15
---

# Phase 31 Plan 08: Hard-Interruption Key Rotation Gap Closure Summary

**Interrupted key rotation reads now preserve signed and encrypted entries when metadata was staged before active key publication.**

## Performance

- **Duration:** 11 min
- **Started:** 2026-06-15T13:56:54Z
- **Completed:** 2026-06-15T14:07:17Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- Added RED regressions for UnifiedCache, BlobStore, encrypted UnifiedCache, and encrypted BlobStore hard interruptions after staged metadata/signatures but before active key replacement.
- Implemented UnifiedCache startup detection of leftover `<keyfile>.new`, including staged namespace and entry-signature verification fallback.
- Implemented BlobStore staged signer fallback for canonical and legacy signatures, plus staged encrypted-read fallback after active-key decryption fails.
- Preserved public API semantics: no automatic key replacement, staged-key deletion, metadata rewrite, or implicit rotation completion on startup.

## Task Commits

1. **Task 1: Add SEC-03 hard-interruption regressions** - `5d7c0c4` (test)
2. **Task 2: Add staged-key read fallback for interrupted rotation** - `ea4c89f` (feat)

**Plan metadata:** pending final docs commit.

## Files Created/Modified

- `src/cacheness/core.py` - Initializes interrupted-rotation staged signer/encryption fallback, accepts staged namespace signatures, and shares fallback state with the internal BlobStore.
- `src/cacheness/_verification_mixin.py` - Verifies entries with the staged signer before integrity hooks or invalid-signature deletion run.
- `src/cacheness/storage/blob_store.py` - Initializes standalone staged fallback state, verifies active then staged signatures, and retries encrypted blob decryption with the staged key after active-key failure.
- `tests/test_key_rotation_api.py` - Adds hard-interruption regressions for signed UnifiedCache and BlobStore entries.
- `tests/test_encryption_at_rest.py` - Adds encrypted UnifiedCache hard-interruption regression.
- `tests/test_blob_store.py` - Adds encrypted BlobStore hard-interruption regression.

## Decisions Made

- Kept `<keyfile>.new` as a fallback credential only; readers never publish it over the active key.
- Required both an active persistent key file and a 32-byte staged key file before creating fallback state.
- Preserved active-key-first behavior for signatures and decryption to avoid broadening staged-key trust.
- Preserved SEC-04 legacy BlobStore verification order for both active and staged signers.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The first RED run used the literal plan pytest command and project addopts expanded collection broadly; it still failed on the intended hard-interruption read failure. Later verification used controlled Windows-safe equivalents with `-o addopts=''`, repo-local `--basetemp`, and repo-local uv cache directories.
- `uv run ... ruff format` initially failed against the default Windows uv cache path. Rerunning with repo-local `UV_CACHE_DIR=.uv-cache` and `UV_PYTHON_INSTALL_DIR=.uv-python` succeeded.
- `ty check src/cacheness/core.py src/cacheness/_verification_mixin.py src/cacheness/storage/blob_store.py` still reports pre-existing mixin/Path diagnostics in those files. No new diagnostics point at the staged fallback helpers.

## Verification

- RED: `uv run --python 3.12 pytest -o addopts='' -p no:cacheprovider --basetemp .uv-cache\pytest-31-08-red tests/test_key_rotation_api.py tests/test_encryption_at_rest.py tests/test_blob_store.py -k "hard_interruption_after_staged" -x -q --ignore=tests/test_tensorflow_handler.py` - FAILED as expected: reopened UnifiedCache returned `None` and deleted the staged-signature entry.
- GREEN focused: `uv run --python 3.12 pytest -o addopts='' -p no:cacheprovider --basetemp .uv-cache\pytest-31-08-green-focused tests/test_key_rotation_api.py tests/test_encryption_at_rest.py tests/test_blob_store.py -k "hard_interruption_after_staged" -x -q --ignore=tests/test_tensorflow_handler.py` - PASSED, 4 passed.
- Plan focused gate: `uv run --python 3.12 pytest -o addopts='' -p no:cacheprovider --basetemp .uv-cache\pytest-31-08-plan-focused tests/test_key_rotation_api.py tests/test_encryption_at_rest.py tests/test_blob_store.py -x -q --ignore=tests/test_tensorflow_handler.py` - PASSED, 89 passed.
- Broader gate: `uv run --python 3.12 pytest -o addopts='' -p no:cacheprovider --basetemp .uv-cache\pytest-31-08-plan-b tests/test_key_rotation.py tests/test_key_rotation_api.py tests/test_encryption_at_rest.py tests/test_blob_store.py tests/test_cache_signing.py tests/test_handler_bytes_protocol.py -x -q --ignore=tests/test_tensorflow_handler.py` - PASSED, 130 passed, 3 skipped.
- Final focused sweep: `uv run --python 3.12 pytest -o addopts='' -p no:cacheprovider --basetemp .uv-cache\pytest-31-08-final-sweep tests/test_cache_signing.py tests/test_blob_store.py tests/test_key_rotation.py tests/test_key_rotation_api.py tests/test_storage_mode.py tests/test_atomic_writes.py tests/test_write_intent.py tests/test_encryption_at_rest.py tests/test_handler_bytes_protocol.py -x -q --ignore=tests/test_tensorflow_handler.py` - PASSED, 203 passed, 3 skipped.
- `uv run --python 3.12 ruff format src/cacheness/core.py src/cacheness/_verification_mixin.py src/cacheness/storage/blob_store.py tests/test_key_rotation_api.py tests/test_encryption_at_rest.py tests/test_blob_store.py` - PASSED.
- `uv run --python 3.12 ruff check --fix src/cacheness/core.py src/cacheness/_verification_mixin.py src/cacheness/storage/blob_store.py tests/test_key_rotation_api.py tests/test_encryption_at_rest.py tests/test_blob_store.py` - PASSED.
- `uv run --python 3.12 ruff check src/cacheness/core.py src/cacheness/_verification_mixin.py src/cacheness/storage/blob_store.py tests/test_key_rotation_api.py tests/test_encryption_at_rest.py tests/test_blob_store.py` - PASSED.
- `uv run --python 3.12 ty check src/cacheness/core.py src/cacheness/_verification_mixin.py src/cacheness/storage/blob_store.py` - FAILED on pre-existing diagnostics documented above.

## Known Stubs

None. Stub scan only matched existing optional-backend log strings containing "not available" in `src/cacheness/core.py`.

## Threat Flags

None. This plan stays inside the existing staged-key trust boundary documented in the plan threat model and does not add new endpoints, auth paths, schemas, or external file-access surfaces.

## User Setup Required

None - no external service configuration required.

## TDD Gate Compliance

- RED commit exists: `5d7c0c4` (`test(31-08): add hard-interruption rotation regressions`)
- GREEN commit exists after RED: `ea4c89f` (`feat(31-08): add interrupted rotation staged-key fallback`)

## Next Phase Readiness

SEC-03 gap closure is complete. Phase 31 is ready for re-verification and Phase 32 can proceed with small independent release-polish fixes.

## Self-Check: PASSED

- Verified summary and all key source/test files exist.
- Verified task commits `5d7c0c4` and `ea4c89f` are present in git history.
- Verified task commits did not delete tracked files.

---
*Phase: 31-security-storage-mode-posture*
*Completed: 2026-06-15*
