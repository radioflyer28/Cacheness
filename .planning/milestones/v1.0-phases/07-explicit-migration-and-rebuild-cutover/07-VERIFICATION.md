---
phase: 07-explicit-migration-and-rebuild-cutover
verified: 2026-09-12T03:08:09Z
status: passed
score: 9/9 must-haves verified
roadmap_score: 5/5 roadmap truths verified
requirement_score: 4/4 requirements satisfied
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 8/9
  gaps_closed:
    - "SECURITY_THREAT_NODES['T-07-21-03'] now preserves the forged-debt selector and appends the exact forward-fence/resume selector; both affected exact-map tests enforce the complete ordered tuple."
  gaps_remaining: []
  regressions: []
deferred:
  - truth: "Live PostgreSQL/AWS S3, Windows, supported-Python matrix, packaging, and performance qualification."
    addressed_in: "Phase 8"
    evidence: "Phase 8 goal is reproducible release evidence across supported installations, Python versions, backends, failures, and operational scale."
decision_coverage:
  honored: 22
  total: 22
  not_honored: []
---

# Phase 7: Explicit Migration and Rebuild Cutover Verification Report

**Phase Goal:** With workers stopped, users can explicitly migrate or rebuild supported versioned stores without silent mutation or losing the only valid copy of stored data; the tooling establishes future release migration discipline even though current pre-production layouts may be unsupported.
**Verified:** 2026-09-12T03:08:09Z
**Status:** passed
**Re-verification:** Yes — after Plan 07-24 gap closure.

## Goal Achievement

### Observable Roadmap Truths

| # | Roadmap truth | Status | Evidence |
|---|---|---|---|
| 1 | Users can inspect without mutation and receive human- and machine-readable plans with counts, bytes, incompatibilities, and actions. | ✓ VERIFIED | Previously verified implementation remains present. The independently run fixed `--quick` verifier executed its exact MIGR-03 nodes and reported PASS. |
| 2 | Supported same-backend migrations use offline copy-verify-switch, retain the prior copy, and never upgrade on ordinary open/initialize. | ✓ VERIFIED | Previously verified implementation remains unchanged. The fixed quick verifier reported MIGR-04 PASS. |
| 3 | Interrupted bounded migration or rebuild resumes idempotently for durably attributed effects without losing the only valid generation. | ✓ VERIFIED | The shipped forward-fence/resume behavior remains backed by its passing exact behavioral selector, and Plan 24 now binds that selector to the applicable terminal-state threat without removing forged-debt evidence. The fixed quick verifier reported MIGR-05 PASS. |
| 4 | Incompatible formats and cross-backend moves have an explicit, scoped, confirmed rebuild path. | ✓ VERIFIED | Previously verified registered-handler/destination-BlobStore rebuild path remains unchanged. The fixed quick verifier reported MIGR-06 PASS. |
| 5 | The source-version window is explicit and projections remain derived rather than becoming cutover authority. | ✓ VERIFIED | Previously verified release-window and projection-separation implementation remains unchanged; fixed inventory validation passed. |

**Roadmap score:** 5/5 truths verified.

### Plan 07-24 Gap-Closure Truths

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | `T-07-21-03` preserves forged-debt evidence and also owns the forward-fence/resume selector. | ✓ VERIFIED | `tools/verify_phase7_contracts.py:325-328` contains the exact ordered two-selector tuple. Both referenced test functions exist in `tests/test_rebuild_workflow.py`. |
| 2 | The Plan 23 exact-map self-test enforces the complete tuple. | ✓ VERIFIED | `tests/test_phase7_contract_verifier.py:392-430` defines both exact selectors and asserts tuple equality. The named test passed independently. |
| 3 | The current-three-gap exact-map assertion agrees with the expanded tuple and fixed inventory remains unchanged. | ✓ VERIFIED | `tests/test_phase7_contract_verifier.py:437-504` asserts the same ordered tuple plus the reviewed 23-plan/54-gap-threat/100-total-threat inventory. Both exact tests and the fixed quick verifier passed. |
| 4 | Previously verified Plan 23 production cleanup-debt fencing and resume behavior remain intact. | ✓ VERIFIED | Plan 24's two commits touch only the verifier and its test; no production lifecycle or behavioral test file changed. The fixed verifier executed the exact behavior selector and reported MIGR-05 PASS. |

**Combined score:** 9/9 must-haves verified (0 present-but-behavior-unverified).

## Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `tools/verify_phase7_contracts.py` | Complete exact selector ownership for `T-07-21-03` | ✓ VERIFIED | Exists, substantive, consumed by the fixed verifier, and contains forged-debt first plus forward-fence/resume second. |
| `tests/test_phase7_contract_verifier.py` | Fail-closed complete-tuple assertions | ✓ VERIFIED | Exists, active, substantive, imports the verifier, and both affected tests assert exact ordered equality. |

The generic artifact query reported 2/2 passed. Its key-link query missed the first link because the PLAN regex assumes a single line; direct inspection and executable equality tests prove the multiline tuple wiring.

## Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `tools/verify_phase7_contracts.py` | `tests/test_rebuild_workflow.py` | Literal `SECURITY_THREAT_NODES['T-07-21-03']` tuple | ✓ WIRED | Both exact pytest selectors exist and execute through the fixed verifier. |
| `tests/test_phase7_contract_verifier.py` | `tools/verify_phase7_contracts.py` | Two exact equality assertions | ✓ WIRED | Both the Plan 23 mapping test and current-three-gap test compare `(forged_debt, forward_fence)` in order. |

## Data-Flow Trace

| Flow | Source | Result | Status |
|---|---|---|---|
| Terminal-state threat evidence | Literal `T-07-21-03` tuple | Fixed manifest resolves and runs both exact rebuild-workflow nodes | ✓ FLOWING |
| Self-test fail-closed contract | Exact selector strings in verifier test | Equality comparison rejects omission, substitution, or reordering | ✓ FLOWING |

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| Both exact-map tests enforce the complete ordered tuple | `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase7_contract_verifier.py::test_fixed_manifest_maps_rebuild_cleanup_debt_fence_exactly tests/test_phase7_contract_verifier.py::test_fixed_manifest_maps_current_three_gap_repairs_exactly -x -o log_cli=false -o addopts=` | 2 passed | ✓ PASS |
| Scoped quality gate | `uv run --isolated --all-extras --group dev --frozen ruff check tools/verify_phase7_contracts.py tests/test_phase7_contract_verifier.py --output-format concise` | All checks passed | ✓ PASS |

## Probe Execution

| Probe | Command | Result | Status |
|---|---|---|---|
| Fixed Phase 7 quick verifier | `uv run --isolated --all-extras --group dev --frozen python tools/verify_phase7_contracts.py --quick` | Exit 0; MIGR-03, MIGR-04, MIGR-05, MIGR-06, and fixed inventory all PASS | ✓ PASS |
| Full deterministic non-live suite | Orchestrator-owned post-Plan-24 run | Reported exit 0 by the orchestrator | ✓ PASS (orchestrator evidence) |

Live PostgreSQL/S3, Windows, supported-Python matrix, packaging, and performance were not run because they are Phase 8 qualification boundaries.

## Requirements Coverage

| Requirement | Source plans | Description | Status | Evidence |
|---|---|---|---|---|
| MIGR-03 | Phase 7 plans | Non-mutating inventory and human/machine plans | ✓ SATISFIED | Previously verified implementation; fixed exact quick contract passed. |
| MIGR-04 | Phase 7 plans | Explicit offline same-backend copy-verify-switch with no implicit upgrade | ✓ SATISFIED | Previously verified implementation; fixed exact quick contract passed. |
| MIGR-05 | Phase 7 plans including 07-24 | Safe explicit-evidence resume without loss or adoption | ✓ SATISFIED | Production behavioral selectors pass, `T-07-21-03` owns both required selectors, and both exact-map tests enforce the binding. |
| MIGR-06 | Phase 7 plans | Explicit confirmed rebuild path | ✓ SATISFIED | Previously verified implementation; fixed exact quick contract passed. |

All four Phase 7 requirements are mapped and claimed by plans; none is orphaned.

## Test Quality Audit

| Test file | Linked requirement | Active / skipped | Circular | Strongest assertion | Verdict |
|---|---|---|---|---|---|
| `tests/test_phase7_contract_verifier.py` | MIGR-05 evidence manifest | Active; no linked skip markers | No | Exact value/ordered tuple plus executable fixed-manifest validation | SUFFICIENT |
| `tests/test_rebuild_workflow.py` | MIGR-05 lifecycle behavior | Active | No | Multi-step behavioral and value assertions | SUFFICIENT (previously verified; executed by quick verifier) |

No disabled requirement-only test, circular oracle, or weak existence-only assertion was found for the closure contract.

## Anti-Patterns and Prohibitions

No `TBD`, `FIXME`, `XXX`, placeholder implementation, or disabled linked test appears in the two Plan 24 files. Scoped Ruff passes.

The complete Plan 24 commit range changes only `tools/verify_phase7_contracts.py` and `tests/test_phase7_contract_verifier.py`. It adds no production lifecycle state, authority, lock, queue, lease, journal, sidecar, listing/adoption rule, stronger cross-resource atomicity claim, production obstore adoption, Phase 3 contention change, or Phase 8 qualification claim. Fixed Plan 01-23 paths and the 54-gap-threat/100-total-threat inventory remain unchanged.

## Decision Coverage

All **22/22** trackable `07-CONTEXT.md` decisions are represented in shipped artifacts according to the non-blocking decision-coverage gate.

## Human Verification Required

N/A — infrastructure/library verification with no user-facing visual or external-service behavior in this closure. All closure criteria are deterministic and exercised programmatically.

## Deferred Items

Live PostgreSQL/AWS S3, Windows, supported Python versions, packaging, and performance qualification are explicitly Phase 8 work. ADR 0001's accepted invisible/unattributed pre-checkpoint orphan, non-cross-resource ACID, topology-specific bounded outcomes, and the pre-existing Phase 3 SQLite bootstrap contention behavior remain accepted boundaries rather than Phase 7 gaps.

## Gaps Summary

No remaining gaps. Plan 07-24 closes the sole prior blocker by making the fixed verifier own and fail closed on the complete Plan 21 terminal-state threat evidence tuple. Phase 7 achieves its declared stopped-worker migration/rebuild goal without reopening topology-specific limits or adding coordination machinery.

---

_Verified: 2026-09-12T03:08:09Z_
_Verifier: the agent (gsd-verifier)_
