# Phase 28: Silent Data-Loss Remediation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md - this log preserves the alternatives considered.

**Date:** 2026-06-12
**Phase:** 28-Silent Data-Loss Remediation
**Areas discussed:** Write-intent recovery boundary, JSON failure behavior, Cache-key remediation source, Test packaging, Plan slicing and order, clear_all cleanup strategy, Compatibility and release notes, Verification strictness

---

## Todo Folding

Folded into Phase 28:

- Fix `clear_all()` not deleting blob files
- Fix write-intent journal path resolution and safety checks
- Property-based stress testing for cache key serialization
- JSON backend - propagate save failures, preserve corrupt files
- Stabilize cache keys - remove unstable `hash()`/`str()` fallbacks

Reviewed but deferred:

- Tiered pull-through cache

---

## Write-Intent Recovery Boundary

| Option | Description | Selected |
|--------|-------------|----------|
| TASK-2 only | Keep Phase 28 to the action doc's required fixes. | |
| TASK-2 + SEED-006 | Also record intent before blob write in `put()` and storage mode. | |
| Separate plans in Phase 28 | Plan TASK-2 first, then SEED-006 as a follow-up in the same phase. | |
| Source from docs | Let review docs and seed determine the boundary. | yes |

**User's choice:** Source from docs.
**Notes:** Context records TASK-2 as mandatory and SEED-006 as a companion
follow-up slice.

---

## JSON Failure Behavior

| Option | Description | Selected |
|--------|-------------|----------|
| TASK-3 exactly | Follow the action doc and avoid extra health-state work. | |
| TASK-3 plus health flag | Also expose backend dirty/save-failed state. | |
| All JSON writes raise | Make access-time/hit/miss writes raise too. | |
| Source from docs | Let TASK-3 and R3/R4 define behavior. | yes |

**User's choice:** Source from docs.
**Notes:** Context records TASK-3 exactly, with R3/R4 as rationale.

---

## Cache-Key Remediation Source

| Option | Description | Selected |
|--------|-------------|----------|
| TASK-4 exactly | Follow only the action doc. | |
| FINDINGS expands TASK-4 | Expand object fallback behavior from U1. | |
| TASK-4 minimum, U1 guardrails | Use TASK-4 as concrete minimum and U1 as rationale. | |
| Source from docs | Let action and findings docs define the balance. | yes |

**User's choice:** Source from docs.
**Notes:** TASK-4 is the concrete contract; U1 is the guardrail.

---

## Test Packaging

| Option | Description | Selected |
|--------|-------------|----------|
| Source from codebase docs | Use `.planning/codebase/TESTING.md` organization. | |
| Focused cache-key files | Put key tests in existing cache-key/serialization files. | |
| One new Phase 28 test file | Create a phase-specific test file. | |
| Source from docs and existing tests | Use action-doc acceptance commands plus current test organization. | yes |

**User's choice:** Source from docs and existing tests.
**Notes:** Planner chooses exact files after reading current tests.

---

## Additional Doc-Sourced Areas

### Plan slicing and order

`CODE_REVIEW_ACTIONS.md` says each task is self-contained, tasks run in order
within a wave, and one task equals one beads issue equals one atomic commit.
Phase 28 should preserve TASK-1 through TASK-4 granularity and Wave 1 order,
with SEED-006 as a companion follow-up slice tied to TASK-2.

### clear_all cleanup strategy

TASK-1 allows either backend enumeration or careful recursive filesystem
filtering. The planner should choose after reading current APIs, while
preserving the explicit exclusion list and no-lock contract.

### Compatibility and release notes

TASK-4 mandates commit message and CHANGELOG compatibility note. TASK-3 says
commit message should mention the new contract if old swallow-behavior tests
are updated.

### Verification strictness

Every task with a "Verify first" step must run that step before implementation.
If observed behavior differs from the action doc, stop and report.

## the agent's Discretion

- Exact test file placement after reading current tests.
- TASK-1 enumeration method after reading current blob backend APIs.
- Whether implementation creates user-visible docs beyond the explicitly
  required TASK-4 CHANGELOG note.

## Deferred Ideas

- Tiered pull-through cache remains future scope.
