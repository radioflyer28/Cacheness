---
phase: 31-security-storage-mode-posture
plan: 03
subsystem: security
tags: [key-rotation, signing, encryption, atomic-writes, blob-store]

requires:
  - phase: 31-security-storage-mode-posture
    provides: SEC-01 minimum signature-version policy, SEC-02 backend-routed encrypted reads, and SEC-04 shared canonical signing fields
provides:
  - Two-phase UnifiedCache.rotate_key key staging
  - Two-phase BlobStore.rotate_key key staging
  - Startup warning for leftover <keyfile>.new staged keys
  - Local encrypted blob rotation through .rotating plus os.replace
  - Fault regressions for interrupted rotation preserving old active keys and readable entries
affects: [security, signing, encryption, blob-store, unified-cache, phase-31]

tech-stack:
  added: []
  patterns:
    - Staged key-file publication through <keyfile>.new
    - Old-key verification before rotation mutation
    - Best-effort rollback before key publication on controlled rotation failure
    - Local encrypted blob publication through <blob>.rotating and os.replace

key-files:
  created:
    - .planning/phases/31-security-storage-mode-posture/31-03-SUMMARY.md
  modified:
    - src/cacheness/core.py
    - src/cacheness/storage/blob_store.py
    - src/cacheness/security.py
    - tests/test_key_rotation.py
    - tests/test_key_rotation_api.py
    - tests/test_encryption_at_rest.py
    - tests/test_blob_store.py

key-decisions:
  - "SEC-03 writes new key bytes to <keyfile>.new and replaces the active key only after rotation succeeds."
  - "UnifiedCache and BlobStore verify existing signed entries with the old signer before re-signing with the staged signer."
  - "Local encrypted blob rotation writes ciphertext to <blob>.rotating and publishes with os.replace."
  - "Leftover <keyfile>.new files are logged as interrupted rotations on startup; full resume remains out of scope."

patterns-established:
  - "Rotation publication pattern: stage key, verify old signature, mutate/sign with staged key, publish active key last."
  - "Encrypted local blob rotation pattern: snapshot old ciphertext, write .rotating, os.replace, and roll back on controlled failure."

requirements-completed: [SEC-03]

duration: 14min
completed: 2026-06-15
---

# Phase 31 Plan 03: Two-Phase Key Rotation Summary

**UnifiedCache and BlobStore key rotation now stage new keys, verify old signatures first, and publish encrypted local blobs atomically.**

## Performance

- **Duration:** 14 min
- **Started:** 2026-06-15T03:07:04Z
- **Completed:** 2026-06-15T03:20:30Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- Added SEC-03 regressions for interrupted UnifiedCache and BlobStore rotation preserving the old active key and keeping old entries readable.
- Added startup logging coverage for leftover `<keyfile>.new` files.
- Added encrypted local blob rotation tests proving committed blob paths are not rewritten in place.
- Implemented staged key writes through `<keyfile>.new` and active key replacement only after successful rotation.
- Implemented old-signer verification before each entry is re-signed with the staged signer.
- Reworked encrypted local blob re-encryption to publish through `<blob>.rotating` and `os.replace`.

## Task Commits

1. **Task 1: Add SEC-03 interrupted-rotation fault regressions** - `68c817d` (test)
2. **Task 2: Implement two-phase rotation and staged key replacement** - `d9183bf` (feat)

**Plan metadata:** pending final docs commit or skipped by GSD commit helper.

## Files Created/Modified

- `src/cacheness/security.py` - Adds staged-key path/write helpers and startup error logging for leftover `<keyfile>.new`.
- `src/cacheness/core.py` - Implements staged UnifiedCache rotation, old-key verification, encrypted blob `.rotating` publication, and rollback before key publication.
- `src/cacheness/storage/blob_store.py` - Implements the same staged rotation policy for BlobStore.
- `tests/test_key_rotation.py` - Adds leftover staged-key startup logging regression.
- `tests/test_key_rotation_api.py` - Adds interrupted rotation regressions for UnifiedCache and BlobStore.
- `tests/test_encryption_at_rest.py` - Adds UnifiedCache encrypted local blob atomic publication regression.
- `tests/test_blob_store.py` - Adds BlobStore encrypted local blob atomic publication regression.

## Decisions Made

- Kept `<keyfile>.new` as the only staged key filename, matching D-16 and D-21.
- Used the staged key file to construct the new signer, then rewrote `new_signer.key_file_path` to the active key path only after `os.replace`.
- Treated a controlled mid-rotation failure as a failed rotation result that keeps the old active signer/key in memory and removes the staged key file.
- Logged leftover staged keys at signer startup without attempting automatic resume.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added rollback before active key publication**
- **Found during:** Task 2 (Implement two-phase rotation and staged key replacement)
- **Issue:** Staging the key and replacing it last is not sufficient by itself when a controlled failure occurs after some metadata or blobs have been published under the staged key. Old entries would not remain reliably usable under the still-active old key.
- **Fix:** Snapshot touched metadata and local encrypted blob ciphertext before mutation, then restore those snapshots and remove the staged key if rotation fails before active key replacement.
- **Files modified:** `src/cacheness/core.py`, `src/cacheness/storage/blob_store.py`
- **Verification:** Interrupted-rotation regressions pass for both APIs; encrypted blob rotation tests pass.
- **Committed in:** `d9183bf`

---

**Total deviations:** 1 auto-fixed (Rule 2)
**Impact on plan:** Required for SEC-03 correctness. No public API signatures, package dependencies, or Phase 32 scope were changed.

## Issues Encountered

- The literal plan pytest command still fails before collection in this Windows environment because xdist cannot access `C:\Users\akriz\AppData\Local\Temp\pytest-of-akriz`. Controlled equivalent runs cleared addopts, disabled pytest cache, and used repo-local `--basetemp`.
- `ty check` on the full touched-file list still reports pre-existing diagnostics in `core.py`, `blob_store.py`, and legacy tests. The new rotation helper diagnostics were resolved; remaining diagnostics match the same pre-existing class documented by earlier Phase 31 summaries.

## Verification

- RED: `uv run --python 3.12 pytest -o addopts='' -p no:cacheprovider --basetemp .uv-cache\pytest-31-03-red tests/test_key_rotation.py tests/test_key_rotation_api.py tests/test_encryption_at_rest.py tests/test_core.py tests/test_blob_store.py -x -q --ignore=tests/test_tensorflow_handler.py` - FAILED as expected on missing interrupted-rotation startup warning.
- Focused GREEN: `uv run --python 3.12 pytest -o addopts='' -p no:cacheprovider --basetemp .uv-cache\pytest-31-03-green-focused tests/test_key_rotation.py tests/test_key_rotation_api.py tests/test_encryption_at_rest.py tests/test_blob_store.py -x -q --ignore=tests/test_tensorflow_handler.py` - PASSED, 91 passed.
- Plan gate A: `uv run --python 3.12 pytest -o addopts='' -p no:cacheprovider --basetemp .uv-cache\pytest-31-03-postformat-a tests/test_key_rotation.py tests/test_key_rotation_api.py tests/test_encryption_at_rest.py tests/test_core.py -x -q --ignore=tests/test_tensorflow_handler.py` - PASSED, 106 passed.
- Plan gate B: `uv run --python 3.12 pytest -o addopts='' -p no:cacheprovider --basetemp .uv-cache\pytest-31-03-postformat-b tests/test_key_rotation.py tests/test_blob_store.py -x -q --ignore=tests/test_tensorflow_handler.py` - PASSED, 55 passed.
- Literal plan command: `uv run --python 3.12 pytest tests/test_key_rotation.py tests/test_key_rotation_api.py tests/test_encryption_at_rest.py tests/test_core.py -x -q --ignore=tests/test_tensorflow_handler.py` - FAILED before collection on Windows xdist temp ACL.
- `uv run --python 3.12 ruff format src/cacheness/core.py src/cacheness/storage/blob_store.py src/cacheness/security.py tests/test_key_rotation.py tests/test_key_rotation_api.py tests/test_core.py tests/test_encryption_at_rest.py tests/test_blob_store.py` - PASSED.
- `uv run --python 3.12 ruff check --fix src/cacheness/core.py src/cacheness/storage/blob_store.py src/cacheness/security.py tests/test_key_rotation.py tests/test_key_rotation_api.py tests/test_core.py tests/test_encryption_at_rest.py tests/test_blob_store.py` - PASSED.
- `uv run --python 3.12 ruff check src/cacheness/core.py src/cacheness/storage/blob_store.py src/cacheness/security.py tests/test_key_rotation.py tests/test_key_rotation_api.py tests/test_core.py tests/test_encryption_at_rest.py tests/test_blob_store.py` - PASSED.
- `uv run --python 3.12 ty check src/cacheness/core.py src/cacheness/storage/blob_store.py src/cacheness/security.py tests/test_key_rotation.py tests/test_key_rotation_api.py tests/test_core.py tests/test_encryption_at_rest.py tests/test_blob_store.py` - FAILED on pre-existing diagnostics documented above.

## Known Stubs

None. Stub scan only matched existing optional-backend log strings containing "not available".

## Threat Flags

None. This plan narrows existing key-file and encrypted local blob trust boundaries. It does not add endpoints, auth paths, schema changes, or new external file-access surfaces beyond staged sibling key/blob files required by SEC-03.

## User Setup Required

None - no external service configuration required.

## TDD Gate Compliance

- RED commit exists: `68c817d` (`test(31-03): add interrupted rotation regressions`)
- GREEN commit exists after RED: `d9183bf` (`feat(31-03): implement two-phase key rotation`)

## Next Phase Readiness

SEC-03 is complete. Plan 31-06 can rely on key rotation preserving active keys on controlled failures and local encrypted blob rewrites avoiding in-place publication.

## Self-Check: PASSED

- Verified summary and key source/test files exist.
- Verified task commits `68c817d` and `d9183bf` are present in git history.
- Verified task commits did not delete tracked files.

---
*Phase: 31-security-storage-mode-posture*
*Completed: 2026-06-15*
