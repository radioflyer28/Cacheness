---
phase: 03-atomic-lifecycle-and-recovery-engine
plan: "11"
subsystem: platform-evidence-attestation
tags: [windows-qualification, attestation, sha256, atomic-publication, python-311]
requires:
  - phase: 03-09
    provides: Immutable repository-runtime platform evidence and normal completed summary
  - phase: 03-12
    provides: Digest-bound command-role contract for qualification attestation
provides:
  - Fixed-command capture and byte-level verification of unavailable Windows evidence
  - Atomically published current-host UNAVAILABLE/NOT_QUALIFIED qualification record
  - D-32 validation boundary preserving Phase 999.1 native qualification work
affects: [03-10, phase-999.1, windows-qualification, release-validation]
actuals:
  tokens: 12284
  tasks: 2
  commits: 3
tech-stack:
  added: []
  patterns:
    - Fixed argv evidence helpers bind raw stdout and parsed canonical JSON with SHA-256
    - Qualification summaries consume structured command-role contracts instead of prose
key-files:
  created:
    - tools/capture_phase3_windows_qualification.py
    - tests/test_phase3_windows_qualification_attestation.py
    - .planning/phases/03-atomic-lifecycle-and-recovery-engine/03-09-WINDOWS-QUALIFICATION.md
  modified:
    - .planning/phases/03-atomic-lifecycle-and-recovery-engine/03-VALIDATION.md
key-decisions:
  - "The fixed Python 3.11 qualification argv is non-overridable and stays distinct from the completed repository-runtime command."
  - "Darwin exit 2/UNAVAILABLE is recorded as NOT_QUALIFIED with native_evidence false, never as Windows support."
  - "Phase 999.1 still requires eligible native Windows PASS/exit 0 evidence before a Windows-qualified release."
patterns-established:
  - "Qualification evidence is atomically published through a fsynced sibling temporary file, replace, parent fsync, and byte-level read-back."
  - "Downstream consumers verify exact mapped fields and digests from completed summaries and addenda."
requirements-completed: [STOR-03, STOR-04, STOR-05, STOR-06, STOR-07]
windows-qualification:
  artifact_path: ".planning/phases/03-atomic-lifecycle-and-recovery-engine/03-09-WINDOWS-QUALIFICATION.md"
  artifact_sha256: f6010b5d9abf950194f58dab5bedda8b65d1c15de54030e4da31429ef4854f45
  contract_summary_path: ".planning/phases/03-atomic-lifecycle-and-recovery-engine/03-12-SUMMARY.md"
  contract_summary_sha256: 9073d17f177937306974da6ece6a039ac1261db1017f1bc7b3d6a7a8c57e1929
  qualification_attestation_command: "uv run --python 3.11 --frozen python verify_platform.py --phase3 --require-system Windows --require-python 3.11"
  runner_status: UNAVAILABLE
  runner_exit_code: 2
  milestone_status: NOT_QUALIFIED
  native_evidence: false
  backlog_phase: 999.1
  future_required_status: PASS
  future_required_exit_code: 0
coverage:
  - id: D1
    description: Fixed-command helper atomically captures and structurally verifies the current host's unavailable runner evidence.
    requirement: STOR-03
    verification:
      - kind: integration
        ref: "uv run --python 3.11 --frozen pytest -q tests/test_phase3_windows_qualification_attestation.py -x"
        status: pass
    human_judgment: false
  - id: D2
    description: D-32 record remains UNAVAILABLE/NOT_QUALIFIED without native evidence or a Windows support claim.
    requirement: STOR-07
    verification:
      - kind: integration
        ref: ".venv/bin/python tools/capture_phase3_windows_qualification.py capture … && verify …"
        status: pass
    human_judgment: false
duration: 13min
completed: 2026-09-05
status: complete
---

# Phase 03 Plan 11: Windows Qualification Attestation Summary

**A fixed Python 3.11 runner now atomically attests the current Darwin host as `UNAVAILABLE` and `NOT_QUALIFIED`, with byte-level provenance while preserving Phase 999.1's native Windows proof obligation.**

## Performance

- **Duration:** 13min
- **Started:** 2026-09-05T23:27:10Z
- **Completed:** 2026-09-05T23:40:00Z
- **Tasks:** 2/2
- **Files modified:** 4
- **Verification:** 16 focused adversarial tests passed under frozen Python 3.11; the real capture and immediate verify passed with runner exit 2, `UNAVAILABLE`, `NOT_QUALIFIED`, and `native_evidence: false`; direct Ruff was clean for both new Python files.

## Accomplishments

- Added a non-overridable argv helper that runs without a shell, checks the real exit code and one UTF-8 JSON object, then binds exact stdout bytes and canonical parsed JSON with SHA-256.
- Bound capture and verification to the completed 03-09 and 03-12 summaries plus the additive command-role contract, rejecting swapped roles, stale/tampered bytes, malformed output, and support-claim contradictions.
- Published the real current-host qualification record atomically and revised D-32 validation language to distinguish the completed unavailable-evidence obligation from future native Windows qualification.

## Task Commits

Each task was committed atomically:

1. **Task 1: Prove fixed-command capture and byte-level attestation end to end**
   - `a33c66f` — `test(03-11): define Windows qualification attestation contract`
   - `da1aad7` — `feat(03-11): attest fixed Windows qualification evidence`
2. **Task 2: Capture the real record and revise Phase 3 validation semantics**
   - `29ed072` — `fix(03-11): record unavailable Windows qualification`

## Files Created/Modified

- `tools/capture_phase3_windows_qualification.py` — fixed-command runner, structural contract parser, atomic publisher, and fresh verifier.
- `tests/test_phase3_windows_qualification_attestation.py` — happy-path, malformed, tamper, stale, binding, support-claim, atomic-failure, and live repeatability checks.
- `03-09-WINDOWS-QUALIFICATION.md` — current-host stdout byte record and digest-bound qualification status.
- `03-VALIDATION.md` — D-32 sign-off distinguishes unavailable evidence from future native Windows release qualification.

## Decisions Made

- Kept the completed 03-09 repository-runtime command separate from the only valid Python 3.11 qualification command.
- Kept `UNAVAILABLE` at exit 2 as `NOT_QUALIFIED`; no native protected-root, same-session, different-token, or scheduler-retirement proof is inferred.
- Required Phase 999.1 to later obtain native Windows Python 3.11 `PASS` at exit 0 before any Windows-qualified release.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed an inherited virtualenv activation marker from the fixed-command child environment.**

- **Found during:** Task 2
- **Issue:** Invoking the helper through the repository `.venv` leaked `VIRTUAL_ENV` to `uv`, causing `uv` to reject the required `--python 3.11` selector before the fixed runner executed.
- **Fix:** Removed only `VIRTUAL_ENV` from the child environment; the argv remains fixed and no caller value can replace it.
- **Files modified:** `tools/capture_phase3_windows_qualification.py`, `tests/test_phase3_windows_qualification_attestation.py`
- **Verification:** The exact capture and immediate verify command passed against the real current host.
- **Committed in:** `29ed072`

**Total deviations:** 1 auto-fixed (Rule 1).
**Impact on plan:** The fix preserves the required Python 3.11 command rather than broadening it or changing qualification semantics.

## Issues Encountered

- The sandbox cannot access uv's existing shared cache, so the real capture/verify command was run with approved external cache access. It produced the required current-host `UNAVAILABLE` result and did not alter the fixed argv.

## Known Stubs

None.

## Next Phase Readiness

- Plan 03-10 can consume this summary's `windows-qualification` mapping, the immutable 03-12 contract, and the captured record for final Phase 3 acceptance.
- Native Windows qualification remains intentionally unproven. Phase 999.1 must run the fixed command on an eligible Windows Python 3.11 environment and produce `PASS`/exit 0 with protected-root, same-session, and distinct-token evidence.

## Self-Check: PASSED

- The helper, tests, qualification artifact, and validation contract exist and have the recorded qualification and contract-summary SHA-256 bindings.
- The three task commits exist, and the capture/verify run re-executed the fixed command with exit 2 and `UNAVAILABLE` evidence.
