---
phase: 03-atomic-lifecycle-and-recovery-engine
plan: "01"
subsystem: lifecycle-authority compatibility contracts
tags: [blobstore, lifecycle-authority, sqlite, compatibility, ruff]
requires:
  - phase: 02-canonical-storage-and-integrity-contract
    provides: authenticated committed-manifest reads and native handler payload contracts
provides:
  - Frozen public and reconciliation-v1 compatibility baseline for the authority migration
  - Fail-closed scheduler release evidence and explicit disposable-store rebuild boundary
  - Confirmed versioned SQLite lifecycle-authority locator and schema identity
  - Exact-scope Phase 3 Ruff delta gate
affects: [03-02, 03-03, 03-04, 03-05, 03-07, phase-3-validation]
actuals:
  tokens: 14577
  tasks: 3
  commits: 5
tech-stack:
  added: []
  patterns:
    - Characterize caller-visible contracts before lifecycle refactoring.
    - Gate Phase 3 Python edits with a removal-tolerant, no-new-findings Ruff baseline.
    - Confirm one-way persistent authority identity before the first database is created.
key-files:
  created:
    - docs/lifecycle-authority.md
    - tests/_lifecycle_test_support.py
    - tests/fixtures/phase3_ruff_baseline.json
    - tests/test_lifecycle_authority_contract.py
    - tests/test_phase3_release_evidence.py
    - tests/test_phase3_ruff_delta.py
    - tools/verify_phase3_ruff_delta.py
  modified:
    - tests/test_blob_store_read_contract.py
key-decisions:
  - "Confirmed the sole local lifecycle authority locator as .cacheness/lifecycle-authority-v1.sqlite3 with SQLite application ID 0x43414348, user_version 1, and a generated store-identity row."
  - "Scheduler stores are unreleased development state; the only supported disposal action is explicit rebuild, never scheduler replay."
requirements-completed: [STOR-03, STOR-04, STOR-05, STOR-06, STOR-07]
coverage:
  - id: D1
    description: Phase 3-owned Python changes are constrained by a deterministic, removal-tolerant Ruff delta gate.
    verification:
      - kind: unit
        ref: tests/test_phase3_ruff_delta.py
        status: pass
      - kind: other
        ref: .venv/bin/python tools/verify_phase3_ruff_delta.py
        status: pass
    human_judgment: false
  - id: D2
    description: Public error, metadata, reconciliation-v1, release/rebuild, lazy-inspection, and Windows deployment boundaries are characterized before authority creation.
    verification:
      - kind: unit
        ref: tests/test_phase3_release_evidence.py
        status: pass
      - kind: unit
        ref: tests/test_blob_store_read_contract.py
        status: pass
      - kind: unit
        ref: tests/test_lifecycle_authority_contract.py
        status: pass
    human_judgment: false
  - id: D3
    description: The versioned SQLite lifecycle-authority identity is explicitly human-confirmed before the first authority database is created.
    verification:
      - kind: manual_procedural
        ref: checkpoint:decision confirm-recommended
        status: pass
    human_judgment: false
duration: 1h 12m
completed: 2026-09-05
status: complete
---

# Phase 03 Plan 01: Lifecycle Authority Compatibility Baseline Summary

**Frozen the public lifecycle boundary, release/rebuild evidence, authority test
fixtures, and the one-way SQLite authority identity before transactional authority
creation.**

## Performance

- **Duration:** 1h 12m (including the blocking-human persisted-identity decision)
- **Started:** 2026-09-05T00:14:47Z
- **Completed:** 2026-09-05T01:26:58Z
- **Tasks:** 3/3
- **Files modified:** 8

## Accomplishments

- Added a scoped, stable-fingerprint Ruff delta gate that permits only removal of
  pre-existing findings and requires every Phase 3-new Python file to be clean.
- Characterized public error/reconciliation-v1 shapes, unreleased scheduler evidence,
  read-only empty-store inspection, and the Windows pre-provisioned-root contract.
- Confirmed the reserved authority database contract:
  `.cacheness/lifecycle-authority-v1.sqlite3`, application ID `0x43414348`,
  `user_version = 1`, and a generated store-identity row.

## Task Commits

Each implementation task was committed atomically:

1. **Task 1: Freeze the exact Phase 3 Ruff baseline before production edits**
   - `61d8a20` (`test`) — failing Ruff delta verifier tests
   - `1423493` (`feat`) — verifier and checked-in baseline
2. **Task 2: Characterize release history, public shapes, lazy inspection, and Windows scope**
   - `11d6e98` (`test`) — lifecycle authority characterization tests
   - `004033f` (`feat`) — compatibility and release-boundary documentation
3. **Task 3: Confirm persisted authority identity**
   - Human decision: `confirm-recommended`; no production artifact is created until
     Plan 03-02 implements the authority.

## Files Created/Modified

- `docs/lifecycle-authority.md` — public compatibility, rebuild, locator, and Windows scope contract.
- `tests/_lifecycle_test_support.py` — deterministic lifecycle, payload, and fault observers.
- `tests/test_lifecycle_authority_contract.py` — Wave 0 lifecycle observable-shape and missing-authority tests.
- `tests/test_phase3_release_evidence.py` — fail-closed scheduler release/rebuild evidence.
- `tests/test_phase3_ruff_delta.py`, `tests/fixtures/phase3_ruff_baseline.json`, and `tools/verify_phase3_ruff_delta.py` — exact-scope Ruff gate.
- `tests/test_blob_store_read_contract.py` — frozen legacy-v1 metadata dictionary assertion.

## Decisions Made

- Confirmed `confirm-recommended` at the blocking-human checkpoint. The first
  persistent authority database will use the reserved, versioned locator
  `.cacheness/lifecycle-authority-v1.sqlite3`, SQLite application ID `0x43414348`,
  schema `user_version = 1`, and a generated store identity. A change after creation
  requires an explicit locator/schema migration.
- Kept the authority database absent in this plan. Its absence is expected: this
  checkpoint establishes its identity before Plan 03-02 creates it.

## Verification

- `70 passed` — `.venv/bin/pytest -q tests/test_phase3_ruff_delta.py tests/test_phase3_release_evidence.py tests/test_blob_store_read_contract.py tests/test_lifecycle_authority_contract.py -x`
- `Phase 3 Ruff delta: no unmatched findings` — `.venv/bin/python tools/verify_phase3_ruff_delta.py`
- Inspected `docs/lifecycle-authority.md`: the explicit rebuild-only path, lazy
  inspection boundary, Windows scope, authority locator, application ID, schema
  version, and generated store-identity contract are present.

## TDD Gate Compliance

- Task 1 and Task 2 each have an ordered `test(03-01)` RED commit followed by a
  `feat(03-01)` GREEN commit. No violation found.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. The authority database is intentionally not created until the next plan; a
read-only existence check therefore reported it absent, as required by the checkpoint
ordering.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 03-02 can create the confirmed SQLite authority and generated store-identity
  row without guessing a persisted locator or schema contract.
- The six flagged Phase 3 edge probes remain visible and routed to the later
  fault, recovery, idempotency, reconciliation, and concurrency test plans.

## Self-Check: PASSED

- All eight plan-owned artifacts exist.
- All four Task 1/Task 2 commits exist in repository history.
- No known stubs, skipped tests, unrun verification, or new threat surface were found.

---

*Phase: 03-atomic-lifecycle-and-recovery-engine*
*Completed: 2026-09-05*
