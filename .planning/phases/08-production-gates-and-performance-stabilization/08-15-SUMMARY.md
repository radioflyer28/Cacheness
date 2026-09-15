---
phase: 08-production-gates-and-performance-stabilization
plan: 15
subsystem: qualification tooling
tags: [release-qualification, preflight, postgresql, amazon-s3, fixed-verifier]
requires:
  - phase: 08-14
    provides: release-evidence aggregation with controlled-Linux performance deferred
provides:
  - Pure protected-live configuration preflight with sanitized NOT_RUN reporting
  - Literal Plan 08-15 and threat-selector bindings in the Phase 8 verifier
affects: [08-11 live qualification, 08-12 release publication]
actuals:
  tokens: 23174
  tasks: 2
  commits: 3
tech-stack:
  added: []
  patterns:
    - Local configuration validation separated from every network, cleanup, evidence, and qualification effect
    - Literal fixed-plan and threat-to-test inventory for additive gap plans
key-files:
  created:
    - .planning/phases/08-production-gates-and-performance-stabilization/08-15-SUMMARY.md
  modified:
    - tools/run_phase8_qualification.py
    - tests/qualification/test_phase8_evidence.py
    - tools/verify_phase8_contracts.py
    - tests/test_phase8_contract_verifier.py
key-decisions:
  - "Preflight reports configuration readiness only; every result states service_state NOT_RUN."
  - "BACK-05 remains unqualified until the later real PostgreSQL/Amazon-S3 qualification run produces its QUALIFIED/CLEAN evidence."
  - "Plan 08-15 and five threat mappings are literal verifier inventory, never discovered at runtime."
patterns-established:
  - "Protected-live preflight: parse and validate local policy only, then make no service, test, filesystem, cleanup, namespace, or evidence effect."
requirements-completed: []
coverage:
  - id: D1
    description: Configuration-only protected-live preflight emits bounded sanitized NOT_RUN results without invoking effect-bearing qualification seams.
    verification:
      - kind: unit
        ref: tests/qualification/test_phase8_evidence.py#four preflight contracts
        status: pass
    human_judgment: false
  - id: D2
    description: Fixed Phase 8 manifest binds Plan 08-15 and each of its five threats to literal preflight tests.
    verification:
      - kind: unit
        ref: tests/test_phase8_contract_verifier.py#test_fixed_manifest_binds_live_configuration_preflight_gap
        status: pass
    human_judgment: false
duration: 10m
completed: 2026-09-15
status: complete
---

# Phase 08 Plan 15: Protected-Live Preflight Summary

**A side-effect-free local configuration gate now proves only protected-live readiness while explicitly preserving NOT_RUN service status and the mandatory later real-service qualification.**

## Performance

- **Duration:** 10m
- **Started:** 2026-09-15T12:53:45-04:00
- **Completed:** 2026-09-15T13:03:11-04:00
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Added a `--preflight` path that validates required local PostgreSQL/AWS configuration, exact source identity, and reviewed cleanup policy without opening a connection, resolving credentials, starting pytest, creating a namespace, cleaning resources, or writing evidence.
- Made every `CONFIGURED`, `UNAVAILABLE`, and `INVALID` result bounded, sanitized, and explicitly `"service_state": "NOT_RUN"`.
- Bound Plan 08-15 and T-08-15-01 through T-08-15-05 to literal executable tests in the closed Phase 8 verifier.

## Task Commits

1. **Task 1: Add one pure, sanitized live-configuration preflight path** - `684ee0b` (test), `6640deb` (feat)
2. **Task 2: Bind the additive preflight plan into the fixed Phase 8 verifier** - `8577d49` (test)

## Files Created/Modified

- `tools/run_phase8_qualification.py` - adds the configuration-only preflight without changing the side-effecting qualification mode.
- `tests/qualification/test_phase8_evidence.py` - adversarial contracts for redaction, source/policy validation, and effect-free CLI behavior.
- `tools/verify_phase8_contracts.py` - fixed Plan 08-15 and five threat-selector bindings.
- `tests/test_phase8_contract_verifier.py` - closed-inventory assertion for the new gap plan.

## Decisions Made

- Preflight is intentionally weaker than qualification: it cannot establish credentials, identity, IAM authorization, resource state, or service availability.
- `BACK-05` and `QUAL-03` remain pending real-service evidence; this plan supplies their prerequisite gate and does not mark either requirement complete.
- Controlled Linux performance remains deferred under D-23/SEED-006; no macOS timing claim was added.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The sandbox could not initialize the shared uv cache. Re-running the planned isolated verification with approved workspace cache access passed all seven selected tests and both Ruff gates.

## User Setup Required

None - this plan deliberately performs no external service configuration or contact.

## Next Phase Readiness

- Plan 08-11 can invoke `--preflight` before its protected real-service qualification checkpoint.
- The later real PostgreSQL/Amazon-S3 run remains mandatory to produce `QUALIFIED`/`CLEAN` evidence. This local preflight cannot close `BACK-05`.

## Self-Check: PASSED

- Confirmed all four modified implementation/test files and all three task commits exist.
- Confirmed selected preflight and fixed-manifest tests pass, and Ruff check/format pass for all four files.

---
*Phase: 08-production-gates-and-performance-stabilization*
*Completed: 2026-09-15*
