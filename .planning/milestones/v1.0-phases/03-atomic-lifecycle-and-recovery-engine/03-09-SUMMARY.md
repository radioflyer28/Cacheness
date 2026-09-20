---
phase: 03-atomic-lifecycle-and-recovery-engine
plan: "09"
subsystem: storage-lifecycle-validation
tags: [blobstore, lifecycle-authority, sqlite, crash-recovery, concurrency, platform-evidence]
requires:
  - phase: 03-08
    provides: Scheduler-free, single-authority BlobStore composition
provides:
  - Deterministic public crash/reopen and concurrency evidence for lifecycle authority
  - Executable Python 3.11/3.13 and platform evidence commands with explicit unavailable semantics
affects: [03-11, 03-10, phase-999.1, BlobStore, lifecycle-authority]
actuals:
  tokens: 11675
  tasks: 2
  commits: 11
tech-stack:
  added: []
  patterns:
    - Crash tests terminate public BlobStore operations at named authority boundaries and assert whole recovered state
    - Native-platform evidence distinguishes PASS, FAIL, and UNAVAILABLE; unavailable is never a support claim
key-files:
  created: []
  modified:
    - tests/_lifecycle_test_support.py
    - tests/test_blob_store_atomic_lifecycle.py
    - tests/test_blob_store_reconciliation.py
    - tests/test_blob_store_concurrency.py
    - tests/test_sqlite_lifecycle_authority.py
    - tests/test_phase3_windows_contract.py
    - tests/test_blob_store_close_contract.py
    - tests/test_phase3_scheduler_retirement.py
    - src/cacheness/storage/sqlite_lifecycle_authority.py
    - verify_platform.py
    - pyproject.toml
    - uv.lock
key-decisions:
  - "A non-Windows platform result is canonical UNAVAILABLE with exit code 2, not a passing skip or native Windows qualification."
  - "A native Windows denial claim needs a genuinely distinct logon-session or service token; same-session evidence alone is insufficient."
  - "NumPy is declared as a base runtime dependency because package import paths require it."
patterns-established:
  - "Crash and race validation use deterministic hooks, barriers, and full authority-state assertions rather than timing sleeps."
  - "Platform evidence stays machine-readable, repeatable, and separate from future native qualification."
requirements-completed: [STOR-03, STOR-04, STOR-05, STOR-06, STOR-07]
platform-evidence:
  command: ".venv/bin/python verify_platform.py --phase3"
  non_windows_status: UNAVAILABLE
  unavailable_exit_code: 2
  windows_qualified: false
  backlog_phase: 999.1
coverage:
  - id: D1
    description: Deterministic crash, reopen, rollback, and reconciliation evidence preserves one authoritative whole state without payload deserialization.
    requirement: STOR-03
    verification:
      - kind: integration
        ref: ".venv/bin/pytest -q tests/test_blob_store_atomic_lifecycle.py tests/test_blob_store_reconciliation.py tests/test_blob_store_concurrency.py tests/test_sqlite_lifecycle_authority.py -x"
        status: pass
    human_judgment: false
  - id: D2
    description: Same-key CAS, distinct-key overlap, close, busy-deadline, and retired-scheduler behavior remain bounded and deterministic.
    requirement: STOR-07
    verification:
      - kind: integration
        ref: ".venv/bin/pytest -q tests/test_phase3_windows_contract.py tests/test_blob_store_concurrency.py tests/test_blob_store_close_contract.py tests/test_phase3_scheduler_retirement.py -k 'not native_windows' -x"
        status: pass
    human_judgment: false
  - id: D3
    description: Python 3.11/3.13 and non-Windows platform evidence run with explicit pass or unavailable semantics.
    requirement: STOR-05
    verification:
      - kind: integration
        ref: "uv run --python 3.11 --frozen pytest -q tests/test_blob_store_concurrency.py tests/test_blob_store_close_contract.py tests/test_sqlite_lifecycle_authority.py tests/test_phase3_scheduler_retirement.py -x"
        status: pass
      - kind: other
        ref: ".venv/bin/python verify_platform.py --phase3 (Darwin: UNAVAILABLE, exit 2, byte-identical twice)"
        status: pass
    human_judgment: false
duration: 8min
completed: 2026-09-05
status: complete
---

# Phase 03 Plan 09: Deterministic Crash, Race, and Platform Evidence Summary

**BlobStore lifecycle authority now has deterministic crash/reopen and race evidence on Python 3.11/3.13, while the platform runner truthfully emits non-Windows `UNAVAILABLE` rather than claiming Windows qualification.**

## Performance

- **Duration:** 8min close-out verification; implementation was completed in the recorded task commits.
- **Completed:** 2026-09-05
- **Tasks:** 2/2
- **Files modified:** 12
- **Verification:** 38 crash/recovery/concurrency tests passed on the repository Python 3.13 environment; 31 platform/close/retirement tests passed with 1 native-only target deselected; 29 required focused tests passed under CPython 3.11.16; Phase 3 Ruff delta was clean.

## Accomplishments

- Added deterministic subprocess crash coverage across authority intent, candidate publication, promotion, cleanup debt, clear, reconciliation, and rollback boundaries, asserting old-or-new committed state and indexed recovery evidence without handler deserialization.
- Exercised CAS conflicts, ABA-safe lineage changes, distinct-key payload overlap, clear/reconcile interleavings, fork/close ownership, and SQLite's one absolute busy deadline without timing-based race assertions.
- Added machine-readable Phase 3 platform evidence with explicit PASS, FAIL, and `UNAVAILABLE` states; on Darwin it exits 2 with canonical byte-identical `UNAVAILABLE` JSON and does not claim native Windows support.
- Exercised the real Python 3.11 focused suite and restored the repository's pinned CPython 3.13 virtual environment after that run.

## Task Commits

Each task's implementation was already completed atomically before this replan close-out.

1. **Task 1: Complete deterministic crash, reopen, and interleaving coverage**
   - `9abeb10` — `test(03-09): add deterministic crash boundary matrix`
   - `f9e64e7` — `feat(03-09): add deterministic public crash harness`
   - `5735e1a` — `test(03-09): add crash reopen authority matrix`
   - `5165dc1` — `test(03-09): add transactional rollback fault coverage`
   - `d74ec66` — `feat(03-09): add authority crash and rollback seams`
   - `5488478` — `test(03-09): complete deterministic lifecycle interleavings`
2. **Task 2: Add honest Python-version and native-platform evidence commands**
   - `f05e6c1` — `test(03-09): add platform evidence runner contract`
   - `7888814` — `fix(03-09): declare NumPy runtime dependency`
   - `a846c10` — `feat(03-09): implement platform evidence runner`
   - `7d59ff4` — `test(03-09): require distinct second-token evidence`
   - `d0dcc73` — `feat(03-09): verify distinct second-token evidence`

## Files Created/Modified

- `tests/_lifecycle_test_support.py` — public subprocess crash harness and authority boundary hooks.
- `tests/test_blob_store_atomic_lifecycle.py` and `tests/test_blob_store_reconciliation.py` — whole-state crash/recovery and no-deserialization assertions.
- `tests/test_blob_store_concurrency.py`, `tests/test_blob_store_close_contract.py`, and `tests/test_sqlite_lifecycle_authority.py` — deterministic interleavings, close/fork, and bounded busy/rollback coverage.
- `tests/test_phase3_windows_contract.py` and `tests/test_phase3_scheduler_retirement.py` — platform contract and scheduler-negative reachability gates.
- `src/cacheness/storage/sqlite_lifecycle_authority.py` — fault seam used by transactional rollback coverage.
- `verify_platform.py` — canonical Phase 3 evidence runner and native Windows proof contract.
- `pyproject.toml` and `uv.lock` — NumPy declared for package-import correctness.

## Decisions Made

- The non-Windows result from `.venv/bin/python verify_platform.py --phase3` is `UNAVAILABLE`, exits 2, and is evidence of an unavailable required system only; it is not native Windows qualification.
- The repository requires a real `uv run --python 3.11 --frozen` execution as well as its pinned CPython 3.13 run. Missing interpreter evidence is not treated as success.
- Native Windows qualification still requires the backlog Phase 999.1 evidence: protected NTFS root, Python 3.11, same-session contention, and a genuinely different token/session denial.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical runtime dependency] Declared NumPy as a base dependency.**

- **Found during:** Task 2
- **Issue:** Cacheness imports NumPy from normal package-import paths while its project metadata treated NumPy as optional, so a nominal base install could not reliably import the library.
- **Fix:** Added the existing NumPy requirement to `pyproject.toml` and regenerated its locked declaration.
- **Files modified:** `pyproject.toml`, `uv.lock`
- **Verification:** Both focused Python 3.11 and repository Python 3.13 gates imported and executed Cacheness successfully.
- **Committed in:** `7888814`

**Total deviations:** 1 auto-fixed (Rule 2).

## Issues Encountered

- `uv run --python 3.11 --frozen` correctly rebuilt `.venv` around CPython 3.11. The frozen repository sync was then run to restore the pinned CPython 3.13 environment before final repository-runtime verification.

## Known Stubs

None.

## Next Phase Readiness

- Plan 03-11 can consume this completed normal summary and capture an attested D-32 qualification record.
- The remaining native Windows evidence is intentionally deferred, not passed: no eligible Windows environment was available, so `windows_qualified` remains `false`. Backlog Phase 999.1 is required before any Windows-qualified release.

## Self-Check: PASSED

- All eleven Task 1/Task 2 commits exist in history and their files match the 03-09 implementation scope.
- The plan summary includes the required `platform-evidence` mapping with the exact runner command, `UNAVAILABLE`, exit code 2, `windows_qualified: false`, and backlog Phase 999.1.
- Final focused verification passed on CPython 3.11 and CPython 3.13; repeated Darwin runner output was byte-identical and exited 2.
