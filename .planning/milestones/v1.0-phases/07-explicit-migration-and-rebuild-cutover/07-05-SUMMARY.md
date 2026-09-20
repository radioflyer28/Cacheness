---
phase: 07-explicit-migration-and-rebuild-cutover
plan: "05"
subsystem: storage-migration
tags: [offline-maintenance, migration, canonical-json, hmac, recovery]
requires:
  - phase: 07-03
    provides: "Canonical bounded migration plans and explicit release compatibility"
provides:
  - "Authenticated, bounded, path-contained maintenance-run evidence"
  - "Explicit resumable offline steps with output revalidation and typed diagnostics"
  - "Non-secret signing fingerprints and strict stopped-worker acknowledgement binding"
affects: [migration, rebuild, blobstore, lifecycle-authority, phase-07]
actuals:
  tokens: 19676
  tasks: 2
  commits: 5
tech-stack:
  added: []
  patterns:
    - "Maintenance evidence is canonical, domain-separated HMAC data and never lifecycle authority."
    - "Resume re-derives only plan-bound expected outputs, verifies them, and advances one legal recorded state."
key-files:
  created:
    - tests/test_migration_run_evidence.py
  modified:
    - src/cacheness/storage/migration_evidence.py
    - src/cacheness/storage/migration.py
    - src/cacheness/error_handling.py
key-decisions:
  - "Evidence uses an existing signing provider's read-only get_key path and persists a SHA-256 fingerprint, never key bytes."
  - "Resume accepts only the exact configured run ID and derived evidence path; it never discovers a latest run or adopts incidental candidate state."
patterns-established:
  - "Create maintenance evidence only in inspected state, then conditionally checkpoint exact previous bytes through legal transitions."
  - "Before skipping a staged step, rebuild expected descriptors from the plan and authenticate every recorded candidate output."
requirements-completed: [MIGR-03, MIGR-04, MIGR-05]
coverage:
  - id: D1
    description: "Bounded canonical maintenance evidence rejects forgery, hostile paths, pre-advanced state, and secret disclosure."
    requirement: MIGR-05
    verification:
      - kind: unit
        ref: "tests/test_migration_run_evidence.py"
        status: pass
    human_judgment: false
  - id: D2
    description: "Explicit restart requires the exact run/evidence pair and revalidates staged output before the next idempotent offline step."
    requirement: MIGR-04
    verification:
      - kind: integration
        ref: "tests/test_migration_run_evidence.py#test_resume_requires_exact_run_and_evidence_then_revalidates_next_step"
        status: pass
      - kind: integration
        ref: "tests/test_migration_cutover.py"
        status: pass
    human_judgment: false
duration: 16m
completed: 2026-09-09
status: complete
---

# Phase 07 Plan 05: Authenticated Maintenance Evidence Summary

**Offline migration evidence is now canonical, HMAC-authenticated, path-contained, and resumable only through an exact recorded run that revalidates its outputs.**

## Performance

- **Duration:** 16m
- **Started:** 2026-09-09T20:06:20-04:00
- **Completed:** 2026-09-10T00:21:52Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Added `MaintenanceEvidenceStore`, `StoppedWorkerAcknowledgement`, full maintenance-state vocabulary, strict canonical JSON limits, migration-domain HMAC verification, atomic same-directory evidence replacement, and non-secret key fingerprints.
- Added typed stale-plan, evidence-invalid, evidence-mismatch, confirmation, offline-decision, and cleanup diagnostics without logging or serializing signing bytes.
- Implemented exact-run resume that authenticates evidence, confirms current source/destination bindings, revalidates deterministic staged candidate outputs, and advances only a legal next offline step. Activation remains separately requested and authority-owned.

## Task Commits

1. **Task 1: Authenticate and contain canonical maintenance evidence** - `5b4f10e` (test), `817ac61` (feat)
2. **Task 2: Resume only explicit revalidated idempotent steps** - `50fa06a` (test), `378d9c3` (feat)
3. **Task 2 hardening: Reject pre-advanced maintenance evidence** - `e675c9d` (fix)

## Files Created/Modified

- `src/cacheness/storage/migration_evidence.py` - Owns bounded canonical evidence, safe evidence-file containment, domain-separated HMAC, legal checkpoints, and redacted reports.
- `src/cacheness/storage/migration.py` - Binds offline actions to acknowledgements, existing signing providers, exact evidence, stale-plan checks, and explicit resume revalidation.
- `src/cacheness/error_handling.py` - Defines branchable offline-maintenance diagnostics and stable reasons.
- `tests/test_migration_run_evidence.py` - Covers forgery, bounds, path safety, redaction, exact checkpointing, restart, stale-source, and candidate-tampering behavior.

## Decisions Made

- Maintenance evidence corroborates an explicit offline workflow only; it neither makes a candidate visible nor becomes a second lifecycle authority.
- A restart reconstructs expected candidate descriptors from the authenticated plan and validates recorded output bytes. It never uses a blob listing, catalog membership, or newest work-directory file as progress evidence.
- An active verified run is not implicitly activated by resume; authority activation stays a separate explicit operator action.

## Verification

- `uv run --frozen pytest -q tests/test_migration_run_evidence.py tests/test_migration_cutover.py -x -o log_cli=false` — passed (10 tests).
- `uv run --frozen ruff check src/cacheness/storage/migration.py src/cacheness/storage/migration_evidence.py src/cacheness/error_handling.py tests/test_migration_run_evidence.py tests/test_migration_cutover.py --output-format concise` — passed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical Functionality] Rejected pre-advanced evidence and bounded all receipt collections**
- **Found during:** Task 2 hardening review
- **Issue:** The evidence store could create a first record beyond inspection, allowing an authenticated but uninspected resume state; authority and cleanup receipt counts also lacked their own semantic caps.
- **Fix:** Restricted first creation to `inspected`, retained legal checkpoint transitions for later states, and capped/typed authority and cleanup receipt collections.
- **Files modified:** `src/cacheness/storage/migration_evidence.py`, `tests/test_migration_run_evidence.py`
- **Verification:** Focused migration evidence, cutover, plan-contract, and inspection tests plus Ruff passed.
- **Committed in:** `e675c9d`

**2. [Rule 3 - Blocking] Removed an unused revalidation result before focused lint**
- **Found during:** Task 2 verification
- **Issue:** A redundant local candidate variable caused the required Ruff verification to fail.
- **Fix:** Retained the candidate revalidation call and removed only its unused binding.
- **Files modified:** `src/cacheness/storage/migration.py`
- **Verification:** Focused Ruff command passed.
- **Committed in:** `378d9c3`

---

**Total deviations:** 2 auto-fixed (1 Rule 2 missing critical functionality, 1 Rule 3 blocking verification issue).
**Impact on plan:** Both changes harden the planned fail-closed evidence boundary; they add no lock, queue, lease, daemon, lifecycle authority, or support claim.

## Issues Encountered

- The sandbox could not open uv's existing shared cache for one final verification attempt. Re-running the identical commands with approved cache access passed; no dependency or lockfile changed.

## Known Stubs

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 07 can now use explicit authenticated evidence for later rollback, finalize, purge, and rebuild work. Those actions must continue to treat evidence as corroboration only and keep lifecycle visibility exclusively with the selected authority.

## Self-Check: PASSED

- All four implementation/test artifacts and this summary exist on disk.
- TDD and hardening commits `5b4f10e`, `817ac61`, `50fa06a`, `378d9c3`, and `e675c9d` exist in git history.

---
*Phase: 07-explicit-migration-and-rebuild-cutover*
*Completed: 2026-09-09*
