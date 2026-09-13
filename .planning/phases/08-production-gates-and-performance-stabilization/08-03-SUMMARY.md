---
phase: 08-production-gates-and-performance-stabilization
plan: 03
subsystem: qualification
tags: [python, platform-matrix, evidence, pytest, macos, windows]
requires:
  - phase: 08-01
    provides: bounded Phase 8 evidence envelopes and deterministic gate boundary
provides:
  - Exact stable, advisory, and TensorFlow-aware Python qualification matrix validation
  - Runtime-bound Linux, macOS-boundary, and Windows-nonclaim platform evidence rows
  - ADR 0001 progress classifications preserved in platform evidence aggregation
affects: [phase-08-release-evidence, ci-platform-matrix, QUAL-03]
actuals:
  tokens: 8842.5
  tasks: 2
  commits: 6
tech-stack:
  added: []
  patterns:
    - Standalone, fixed-command platform evidence runner with exact runtime identity checks
    - Platform-role aggregation that distinguishes qualification from non-native availability
key-files:
  created:
    - tools/run_phase8_platform_gates.py
    - tests/qualification/test_phase8_platform.py
  modified:
    - tools/phase8_evidence.py
key-decisions:
  - "Linux 3.11–3.14 is the only full core qualification matrix; Python 3.15 remains advisory."
  - "macOS records only 3.11 and 3.14 public-topology boundary smoke, while Windows is an explicit Phase 999.1 UNAVAILABLE nonclaim."
  - "Success, conflict, and typed-retryable contention are valid ADR progress outcomes and do not trigger portability or coordination changes."
patterns-established:
  - "Platform evidence binds requested and actual operating-system/interpreter identities before a row can pass."
  - "Feature-profile aggregation accepts TensorFlow only on its explicit compatible stable-minor subset."
requirements-completed: [QUAL-03]
coverage:
  - id: D1
    description: Stable, advisory, and TensorFlow-compatible Python matrix contract
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: tests/qualification/test_phase8_platform.py#test_python_stable_matrix_is_exact_and_complete
        status: pass
      - kind: unit
        ref: tests/qualification/test_phase8_platform.py#test_python_tensorflow_profile_has_explicit_stable_compatibility
        status: pass
    human_judgment: false
  - id: D2
    description: Distinct Linux full-gate, macOS boundary-smoke, and Windows nonclaim roles
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: tests/qualification/test_phase8_platform.py#test_platform_linux_matrix_and_macos_boundaries_are_distinct
        status: pass
      - kind: unit
        ref: tests/qualification/test_phase8_platform.py#test_platform_windows_is_non_native_unavailable_without_substitute_execution
        status: pass
    human_judgment: false
duration: 12m 13s
completed: 2026-09-13
status: complete
---

# Phase 08 Plan 03: Platform Qualification Matrix Summary

**Fail-closed platform evidence now requires Linux 3.11–3.14 rows, records macOS boundary smoke, and preserves Windows as a Phase 999.1 nonclaim.**

## Performance

- **Duration:** 12m 13s
- **Started:** 2026-09-13T23:44:48Z
- **Completed:** 2026-09-13T23:57:01Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Added a standalone platform evidence runner that binds requested Python/OS labels to the actual runtime before fixed commands can run.
- Encoded the stable 3.11–3.14 matrix, 3.15 advisory boundary, and TensorFlow-only 3.11–3.13 compatibility subset.
- Separated Linux full gates, macOS 3.11/3.14 public topology smoke, and an unconditional Windows `UNAVAILABLE`/`NOT_QUALIFIED` record for Phase 999.1.
- Preserved ADR 0001 success, conflict, and typed-retryable outcomes as valid progress classifications without changing lifecycle coordination.

## Task Commits

1. **Task 1: Encode the stable, advisory, and feature-aware Python matrix**
   - `f19a876` (`test`) — failing matrix contract
   - `9c2e1e1` (`feat`) — runner and validated platform envelope
2. **Task 2: Separate Linux qualification, macOS boundary smoke, and Windows nonclaim**
   - `d7b16eb` (`test`) — failing platform-role contract
   - `c18d216` (`feat`) — fixed role aggregation and non-native Windows evidence
3. **Matrix hardening discovered during final verification**
   - `8715e49` (`test`) — failing advisory-substitution regression coverage
   - `ddafc6e` (`fix`) — published advisory and TensorFlow compatibility enforcement

## Files Created/Modified

- `tools/run_phase8_platform_gates.py` — fixed platform commands, matrix aggregation, runtime identity checks, and canonical nonclaim output.
- `tests/qualification/test_phase8_platform.py` — unit contracts for all matrix, role, nonclaim, and ADR-progress behavior.
- `tools/phase8_evidence.py` — strict payload validation for the new bounded `platform` evidence class.

## Decisions Made

- Linux full-gate evidence cannot be substituted with macOS smoke or advisory interpreters.
- The macOS smoke exercises only public memory/memory and SQLite/filesystem `BlobStore` composition at the supported oldest/newest boundaries.
- Windows always exits `2` with Phase 999.1 metadata; the runner never invokes a non-native substitute.
- No retryable contention state is translated into a portability failure, corruption claim, or request for new coordination.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical functionality] Added strict `platform` evidence-envelope validation.**

- **Found during:** Task 1
- **Issue:** The shared Phase 8 envelope contract allowed `PASS` only for deterministic and packaging evidence, so a valid platform row could not be safely emitted or independently validated.
- **Fix:** Added an allow-listed, bounded platform payload with expected/actual OS and Python identity, feature profile, role, advisory flag, backlog phase, command profile, and reason.
- **Files modified:** `tools/phase8_evidence.py`
- **Verification:** Focused platform contracts and Ruff pass.
- **Committed in:** `9c2e1e1`

**2. [Rule 1 - Bug] Removed an unused runner import exposed by the configured lint gate.**

- **Found during:** Task 1
- **Issue:** The new runner imported `json` without using it.
- **Fix:** Removed the unused import.
- **Files modified:** `tools/run_phase8_platform_gates.py`
- **Verification:** `uv run ruff check tools/run_phase8_platform_gates.py tools/phase8_evidence.py tests/qualification/test_phase8_platform.py`
- **Committed in:** `9c2e1e1`

**3. [Rule 1 - Bug] Rejected advisory matrix substitutions outside published ranges.**

- **Found during:** Final matrix verification
- **Issue:** An arbitrary non-stable advisory core row, or an advisory TensorFlow row outside 3.11–3.13, could be ignored rather than rejected by aggregation.
- **Fix:** Allow only the declared 3.15 advisory core row and check TensorFlow compatibility before advisory handling.
- **Files modified:** `tests/qualification/test_phase8_platform.py`, `tools/run_phase8_platform_gates.py`
- **Verification:** Focused platform contracts and Ruff pass.
- **Committed in:** `8715e49`, `ddafc6e`

---

**Total deviations:** 3 auto-fixed (1 Rule 2, 2 Rule 1).
**Impact on plan:** Both changes are required for a trustworthy evidence boundary; neither changes payload storage, portability behavior, or lifecycle coordination.

## TDD Gate Compliance

- RED commits: `f19a876`, `d7b16eb`, `8715e49`
- GREEN commits: `9c2e1e1`, `c18d216`, `ddafc6e`
- REFACTOR commits: none required

## Verification

- `uv run pytest -q -o log_cli=false tests/qualification/test_phase8_platform.py -x` — 11 passed.
- `uv run ruff check tools/run_phase8_platform_gates.py tools/phase8_evidence.py tests/qualification/test_phase8_platform.py` — passed.
- Direct macOS public topology smoke (memory/memory and SQLite/filesystem) — passed on the current macOS host.

The current macOS smoke is not Linux matrix evidence. No Windows native evidence was run or claimed.

## User Setup Required

None - no external service configuration is required for this plan.

## Next Phase Readiness

Later release tooling can consume strict platform row evidence without treating a local macOS smoke, an advisory interpreter, or an unavailable Windows result as a full-matrix pass. Actual Linux 3.11–3.14 jobs and Phase 999.1 native Windows evidence remain separate external qualification work.

## Self-Check: PASSED

- All three implementation/test artifacts and this summary exist on disk.
- All six RED/GREEN and regression-fix commits resolve in the repository history.

---
*Phase: 08-production-gates-and-performance-stabilization*
*Completed: 2026-09-13*
