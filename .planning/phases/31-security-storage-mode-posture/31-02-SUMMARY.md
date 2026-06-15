---
phase: 31-security-storage-mode-posture
plan: 02
subsystem: security
tags: [encryption, blob-store, handler-bytes, backend-abstraction, plaintext-temp]

requires:
  - phase: 31-security-storage-mode-posture
    provides: SEC-01 signer policy from Phase 31 Plan 01
provides:
  - Backend-routed encrypted BlobStore reads through blob_backend.read_blob
  - In-memory encrypted plaintext deserialization via handler.get_bytes
  - Hardened plaintext temp fallback using tempfile.mkstemp under cache_dir
  - SEC-02 regressions for encrypted memory backends, bytes-first reads, and temp cleanup
affects: [security, encrypted-reads, blob-store, handler-protocol, phase-31]

tech-stack:
  added: []
  patterns:
    - Shared BlobStore encrypted-read helper for public get and low-level _read_blob
    - Bytes-first encrypted deserialization with NotImplementedError-only temp fallback

key-files:
  created:
    - .planning/phases/31-security-storage-mode-posture/31-02-SUMMARY.md
  modified:
    - src/cacheness/storage/blob_store.py
    - tests/test_blob_store.py
    - tests/test_encryption_at_rest.py
    - tests/test_handler_bytes_protocol.py
    - .planning/phases/31-security-storage-mode-posture/deferred-items.md

key-decisions:
  - "Encrypted BlobStore reads preserve backend abstraction by using blob_backend.read_blob for ciphertext."
  - "Encrypted reads try handler.get_bytes on decrypted plaintext before any temp-file fallback."
  - "Temp fallback is limited to handlers that raise NotImplementedError and uses mkstemp under cache_dir with best-effort POSIX 0600 permissions."

patterns-established:
  - "Encrypted read helper: pass backend path strings to blob backends and display Path values only to handlers/temp fallback."
  - "SEC-02 tests should exercise both BlobStore.get and _read_blob because UnifiedCache delegates through the low-level path."

requirements-completed: [SEC-02]

duration: 12min
completed: 2026-06-15
---

# Phase 31 Plan 02: Encrypted Backend-Routed In-Memory Reads Summary

**Encrypted BlobStore reads now use backend-routed ciphertext reads and avoid plaintext temp files when handlers support byte deserialization.**

## Performance

- **Duration:** 12 min
- **Started:** 2026-06-15T02:23:28Z
- **Completed:** 2026-06-15T02:33:50Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Added SEC-02 regressions for encrypted `InMemoryBlobBackend` reads, encrypted object reads without `NamedTemporaryFile`, `handler.get_bytes()` preference, and `mkstemp` fallback cleanup.
- Reworked `BlobStore.get()` and `_read_blob()` so encrypted ciphertext is read through `self.blob_backend.read_blob(...)` instead of direct local `Path.read_bytes()`.
- Added a shared encrypted-read helper that decrypts in memory, calls `handler.get_bytes(plaintext, metadata)` first, and falls back to a cache-directory temp file only on `NotImplementedError`.
- Hardened fallback temp files by using `tempfile.mkstemp`, best-effort POSIX `0600`, and guaranteed unlink in `finally`.

## Task Commits

1. **Task 1: Add SEC-02 backend-routed and plaintext-temp regressions** - `be1b76f` (test)
2. **Task 2: Route encrypted reads through blob backends and prefer bytes** - `46099af` (feat)

**Plan metadata:** pending final docs commit or skipped by GSD commit helper.

## Files Created/Modified

- `src/cacheness/storage/blob_store.py` - Adds shared encrypted read helper, backend-routed ciphertext reads, bytes-first handler path, and hardened temp fallback.
- `tests/test_blob_store.py` - Adds encrypted memory backend round-trip regression proving URI paths are not treated as filesystem paths.
- `tests/test_encryption_at_rest.py` - Adds public encrypted object read regression blocking the old plaintext `NamedTemporaryFile` path.
- `tests/test_handler_bytes_protocol.py` - Adds low-level `_read_blob()` coverage for `get_bytes()` preference and `mkstemp` fallback cleanup.
- `.planning/phases/31-security-storage-mode-posture/deferred-items.md` - Records environment/baseline verification caveats.

## Decisions Made

- Kept hash semantics unchanged: stored `file_hash` continues to cover the persisted ciphertext.
- Used backend path strings for blob backend calls and separate `Path` values only where handlers or temp fallback require filesystem paths.
- Restricted plaintext temp fallback to `NotImplementedError` from `get_bytes()`, matching D-12.

## Deviations from Plan

None - plan executed as written. Environment and baseline verification caveats were documented in deferred items.

## Issues Encountered

- The literal plan pytest command fails before collection in this Windows environment because xdist uses an ACL-denied temp root. The controlled targeted run with addopts cleared, pytest cache disabled, and repo-local basetemp passes.
- `ty check` still reports pre-existing diagnostics in `blob_store.py` signing/config typing and legacy test typing. New helper-specific diagnostics were resolved.
- `ruff format` rewrapped a post-`pytest.importorskip` import; the `# noqa: E402` marker was moved to the import line that Ruff validates.

## Verification

- RED: `uv run --python 3.12 pytest -o addopts='' -p no:cacheprovider --basetemp .uv-cache\pytest-31-02-red tests/test_blob_store.py::TestEncryptedBackendReads tests/test_encryption_at_rest.py::TestBlobStoreEncryption::test_encrypted_object_get_uses_in_memory_bytes_path tests/test_handler_bytes_protocol.py::TestEncryptedBlobReads -x -q --ignore=tests/test_tensorflow_handler.py` - FAILED as expected on `Blob file missing: memory:\encrypted-memory.pkl`.
- Focused GREEN: `uv run --python 3.12 pytest -o addopts='' -p no:cacheprovider --basetemp .uv-cache\pytest-31-02-final-focused tests/test_blob_store.py::TestEncryptedBackendReads tests/test_encryption_at_rest.py::TestBlobStoreEncryption::test_encrypted_object_get_uses_in_memory_bytes_path tests/test_handler_bytes_protocol.py::TestEncryptedBlobReads -x -q --ignore=tests/test_tensorflow_handler.py` - PASSED, 4 passed.
- Plan target: `uv run --python 3.12 pytest -o addopts='' -p no:cacheprovider --basetemp .uv-cache\pytest-31-02-plan-final tests/test_blob_store.py tests/test_encryption_at_rest.py tests/test_handler_bytes_protocol.py -x -q --ignore=tests/test_tensorflow_handler.py` - PASSED, 87 passed, 3 skipped.
- Literal plan command: `uv run --python 3.12 pytest tests/test_blob_store.py tests/test_encryption_at_rest.py tests/test_handler_bytes_protocol.py -x -q --ignore=tests/test_tensorflow_handler.py` - FAILED before collection due ACL-denied xdist temp root.
- `uv run --python 3.12 ruff format src/cacheness/storage/blob_store.py tests/test_blob_store.py tests/test_encryption_at_rest.py tests/test_handler_bytes_protocol.py` - PASSED.
- `uv run --python 3.12 ruff check --fix src/cacheness/storage/blob_store.py tests/test_blob_store.py tests/test_encryption_at_rest.py tests/test_handler_bytes_protocol.py` - PASSED.
- `uv run --python 3.12 ruff check src/cacheness/storage/blob_store.py tests/test_blob_store.py tests/test_encryption_at_rest.py tests/test_handler_bytes_protocol.py` - PASSED.
- `uv run --python 3.12 ty check src/cacheness/storage/blob_store.py tests/test_blob_store.py tests/test_encryption_at_rest.py tests/test_handler_bytes_protocol.py` - FAILED on pre-existing diagnostics documented in deferred items.

## Known Stubs

None. Stub scan only matched existing optional initializers, test helper empty values, and optional-dependency skip messages.

## Threat Flags

None. This plan narrows the existing encrypted read trust boundary by routing through configured blob backends and reducing plaintext-on-disk exposure; it does not add endpoints, auth paths, schema changes, or new external file-access surfaces.

## User Setup Required

None - no external service configuration required.

## TDD Gate Compliance

- RED commit exists: `be1b76f` (`test(31-02): add encrypted backend read regressions`)
- GREEN commit exists after RED: `46099af` (`feat(31-02): route encrypted blob reads through backend`)

## Next Phase Readiness

SEC-02 is ready for downstream Phase 31 work. Plan 31-04 can rely on BlobStore encrypted reads preserving backend abstraction while it converges signing fields, and Plan 31-03 can reuse backend-routed encrypted reads during rotation hardening.

## Self-Check: PASSED

- Verified summary, deferred-items, source, and test files exist.
- Verified task commits `be1b76f` and `46099af` are present in git history.
- Verified task commits did not delete tracked files.

---
*Phase: 31-security-storage-mode-posture*
*Completed: 2026-06-15*
