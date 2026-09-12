---
phase: 07-explicit-migration-and-rebuild-cutover
verified: 2026-09-12T02:05:49Z
status: gaps_found
score: 8/9 must-haves verified
roadmap_score: 5/5 roadmap truths verified
requirement_score: 4/4 requirements satisfied
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 4/5
  gaps_closed:
    - "Authenticated rebuild cleanup debt now fences direct stage, verify, and accept before source, payload, authority, or evidence effects."
    - "Resume now dispatches exact cleanup only from REBUILDING or REBUILD_VERIFYING and rejects debt in every other state before participant effects."
    - "REBUILD_ACCEPTED evidence with cleanup debt is now invalid, while permitted explicit resume settles exact receipts to debt-free ABORTED."
  gaps_remaining:
    - "The exact forward-fence/resume regression is not mapped to the required Plan 21 terminal-state threat T-07-21-03, and the Plan 23 verifier self-test does not assert that binding."
  regressions:
    - "The full fixed verifier again encountered the pre-existing Phase 3 SQLite fresh-root bootstrap contention failure; the exact test also failed on one rerun. This is outside the Phase 7 migration implementation and does not justify new coordination under ADR 0001, but it keeps the aggregate probe non-green."
gaps:
  - truth: "The fixed verifier maps the exact forward-fence/resume regression to MIGR-05, A-MIGR05, D-16/D-19/D-20/D-21, the applicable Plan 21 terminal-state threat, and every Plan 23 threat, so the lifecycle invariant cannot false-green."
    status: failed
    reason: "The production behavior and Plan 23 threat mappings are present, but tools/verify_phase7_contracts.py leaves T-07-21-03 mapped only to test_rebuild_cleanup_retry_rejects_forged_debt_without_payload_access. The required forward-fence/resume selector is absent, and test_fixed_manifest_maps_rebuild_cleanup_debt_fence_exactly never asserts the T-07-21-03 binding. The fixed --quick verifier therefore reports MIGR-05 and the inventory PASS despite this exact contract omission."
    artifacts:
      - path: "tools/verify_phase7_contracts.py"
        issue: "SECURITY_THREAT_NODES['T-07-21-03'] omits test_rebuild_cleanup_debt_fences_forward_methods_and_resume_settles_exact_receipts."
      - path: "tests/test_phase7_contract_verifier.py"
        issue: "The Plan 23 exact-mapping self-test asserts MIGR-05, A-MIGR05, D-16/D-19/D-20/D-21, and T-07-23-01..04, but not the explicitly required T-07-21-03 edge."
    missing:
      - "Add the exact forward-fence/resume selector to SECURITY_THREAT_NODES['T-07-21-03'] without removing the existing applicable selector."
      - "Make test_fixed_manifest_maps_rebuild_cleanup_debt_fence_exactly assert the complete T-07-21-03 mapping so later removal fails closed."
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
**Verified:** 2026-09-12T02:05:49Z
**Status:** gaps_found
**Re-verification:** Yes — after Plan 07-23.

## Goal Achievement

### Observable Roadmap Truths

| # | Roadmap truth | Status | Evidence |
|---|---|---|---|
| 1 | Users can inspect without mutation and receive human- and machine-readable plans with counts, bytes, incompatibilities, and actions. | ✓ VERIFIED | The previous implementation remains present; Plan 23 does not touch inspection. The independently run fixed `--quick` verifier executed the exact MIGR-03 contract and reported PASS. |
| 2 | Supported same-backend migrations use offline copy-verify-switch, retain the prior copy, and never upgrade on ordinary open/initialize. | ✓ VERIFIED | The previous directed-transform and ordinary-open evidence remains wired. Plan 23 changes only rebuild cleanup-debt fencing, and the fixed `--quick` verifier reported MIGR-04 PASS. |
| 3 | Interrupted bounded migration or rebuild resumes idempotently for durably attributed effects without losing the only valid generation. | ✓ VERIFIED | `stage_rebuild()`, `verify_rebuild()`, and `accept_rebuild()` read authenticated evidence then call `_require_rebuild_cleanup_settlement()` before source revalidation or participant/authority/evidence effects (`migration.py:2912-2917`, `3024-3029`, `3135-3141`). `resume()` permits settlement only in `REBUILDING`/`REBUILD_VERIFYING` before calling `_settle_rebuild_cleanup_debt()` (`migration.py:4352-4368`). The exact behavioral test passed. |
| 4 | Incompatible formats and cross-backend moves have an explicit, scoped, confirmed rebuild path. | ✓ VERIFIED | The prior registered-handler/destination-BlobStore rebuild path remains present and was not altered except for the debt fence. The fixed `--quick` verifier reported MIGR-06 PASS. |
| 5 | The source-version window is explicit and projections remain derived rather than becoming cutover authority. | ✓ VERIFIED | The release-window and projection-separation implementation remains unchanged; the fixed exact contract inventory passed. |

**Roadmap score:** 5/5 truths verified.

### Plan 07-23 Gap-Closure Truths

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | Authentic debt fences direct stage/verify/accept before effects and directs the caller to explicit resume. | ✓ VERIFIED | One private guard is called immediately after the one evidence read in all three public methods. The exact multi-step regression snapshots evidence bytes, authority identity, destination identity, source values, and participant calls; it passed. |
| 2 | Resume dispatches cleanup only from the two defined nonterminal states and rejects terminal/other debt before payload effects. | ✓ VERIFIED | State classification precedes `_settle_rebuild_cleanup_debt()`. The same exact regression exercises rejected forward states and permitted explicit settlement; it passed. |
| 3 | Accepted evidence cannot carry debt, while defined-state exact debt remains idempotently settleable. | ✓ VERIFIED | `MaintenanceRunEvidence.__post_init__()` rejects `REBUILD_ACCEPTED` plus debt (`migration_evidence.py:719-720`). Construction and canonical-load regression passed; permitted settlement reaches debt-free `ABORTED`. |
| 4 | The fixed verifier owns every exact required requirement/decision/assumption/threat binding. | ✗ FAILED | MIGR-05, A-MIGR05, D-16/D-19/D-20/D-21, and Plan 23 threats include the new selector, but `T-07-21-03` does not. The self-test omits that assertion, and the fixed quick verifier still reports PASS. |

**Combined score:** 8/9 must-haves verified (0 present-but-behavior-unverified).

## Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `src/cacheness/storage/migration.py` | Existing coordinator reordered around one private debt guard and state-limited resume settlement | ✓ VERIFIED | Substantive and wired. The guard precedes forward work; resume validates state before exact receipt-bound settlement. No new authority or coordination mechanism was added. |
| `src/cacheness/storage/migration_evidence.py` | Fail-closed accepted-state cleanup-debt invariant | ✓ VERIFIED | Substantive and wired through normal construction and canonical decoding. |
| `tests/test_rebuild_workflow.py` | Authentic-debt forward fence, pre-effect rejection, exact settlement, and evidence regressions | ✓ VERIFIED | Both exact named tests are active and independently passed. Assertions are behavioral/value-level, not existence checks. |
| `tools/verify_phase7_contracts.py` | Literal Plan 23 inventory and all exact requirement/decision/threat mappings | ⚠ PARTIAL | Plan 23, 54 gap threats, 100 total threats, MIGR-05, decisions, assumption, and Plan 23 threats are mapped. Required `T-07-21-03` cross-binding is absent. |
| `tests/test_phase7_contract_verifier.py` | Adversarial exact binding self-test | ⚠ PARTIAL | The exact self-test passes but never checks `T-07-21-03`; syntactic key-link verification therefore returned a false positive. |

The generic artifact query reported 5/5 present/substantive. Manual Level 3 semantic tracing supersedes its pattern-only result for the two partial verifier artifacts.

## Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| Direct rebuild methods | Authenticated evidence model | Single read, immediate cleanup-debt guard, then state-specific validation | ✓ WIRED | Guard runs before source revalidation, destination access, authority changes, or evidence replacement. |
| Explicit resume | Existing lifecycle authority and exact payload participant | State-limited dispatch then authenticated receipt/operation replay settlement | ✓ WIRED | Only `REBUILDING` and `REBUILD_VERIFYING` with debt can enter settlement; no listing/adoption path exists. |
| Evidence decoder | Accepted-state invariant | `MaintenanceRunEvidence.__post_init__()` on construction/from-record | ✓ WIRED | Accepted evidence with debt fails closed. |
| Fixed verifier | Exact Plan 23 behavior test | MIGR-05, A-MIGR05, decisions, and Plan 23 threats | ⚠ PARTIAL | Most links are exact, but the required Plan 21 `T-07-21-03` terminal-state edge is missing. |

## Data-Flow Trace

| Flow | Source | Result | Status |
|---|---|---|---|
| Forward rebuild call with authentic debt | Signed `MaintenanceRunEvidence` read from the exact run path | Typed offline-decision error before participant/authority/evidence mutation | ✓ FLOWING |
| Permitted explicit resume | Authenticated receipt batches plus canonical `read_mutation(operation_id)` replay | Exact delete/absence proof, checkpointed retirement, debt-free terminal `ABORTED` | ✓ FLOWING |
| Accepted evidence decoding | Canonical record with `REBUILD_ACCEPTED` and nonempty debt | `ValueError` before the record can be represented | ✓ FLOWING |
| Verifier threat evidence | Fixed selector tables | New regression reaches Plan 23 threats but not `T-07-21-03` | ✗ DISCONNECTED EDGE |

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| Forward methods fence debt and explicit resume settles exact receipts | Exact named `pytest` selector for `test_rebuild_cleanup_debt_fences_forward_methods_and_resume_settles_exact_receipts` | 1 passed | ✓ PASS |
| Accepted evidence rejects cleanup debt | Exact named `pytest` selector for `test_rebuild_evidence_rejects_accepted_cleanup_debt` | 1 passed | ✓ PASS |
| Plan 23 exact verifier mapping self-test | Exact named `pytest` selector for `test_fixed_manifest_maps_rebuild_cleanup_debt_fence_exactly` | 1 passed, despite omitting the promised `T-07-21-03` assertion | ✗ INSUFFICIENT CONTRACT |
| Scoped lint | `ruff check` on the five Plan 23 source/test files | All checks passed | ✓ PASS |

## Probe Execution

| Probe | Command | Result | Status |
|---|---|---|---|
| Fixed Phase 7 quick verifier | `uv run --isolated --all-extras --group dev --frozen python tools/verify_phase7_contracts.py --quick` | MIGR-03 through MIGR-06 and fixed inventory reported PASS | ⚠ FALSE-GREEN for the omitted `T-07-21-03` edge |
| Fixed Phase 7 full verifier | Same command with `--all` | Phase 7 requirement nodes reported PASS; aggregate exited 1 on `test_fresh_root_bootstrap_converges_through_sqlite` | ✗ NON-PASS |
| Exact unrelated failing selector | Exact Phase 3 SQLite bootstrap test | Failed again with one contender receiving `CacheBlobMigrationRequiredError` | ⚠ PRE-EXISTING PHASE 3 / OUTSIDE PHASE 7 |

The aggregate failure is the already documented Phase 3 fresh-root contention test. It is not caused by Plan 23 and is not converted into a new Phase 7 lifecycle gap or a demand for another lock/queue/sidecar under ADR 0001. It remains visible because the full probe is not green.

## Requirements Coverage

| Requirement | Source plans | Description | Status | Evidence |
|---|---|---|---|---|
| MIGR-03 | Phase 7 plans | Non-mutating inventory and human/machine plans | ✓ SATISFIED | Previous verified implementation; exact fixed quick contract passed. |
| MIGR-04 | Phase 7 plans | Explicit offline same-backend copy-verify-switch with no implicit upgrade | ✓ SATISFIED | Previous verified implementation; exact fixed quick contract passed. |
| MIGR-05 | Phase 7 plans including 07-23 | Safe resume from explicit evidence without loss or adoption | ✓ SATISFIED | The remaining behavioral loss path is closed and its exact behavioral tests pass. The blocker is the plan-required fixed-verifier cross-binding, not production behavior. |
| MIGR-06 | Phase 7 plans | Explicit confirmed rebuild path | ✓ SATISFIED | Previous verified rebuild path remains present; exact fixed quick contract passed. |

All four Phase 7 requirements are mapped to this phase and claimed by plans; none is orphaned.

## Test Quality Audit

| Test file | Linked requirement | Active / skipped | Circular | Strongest assertion | Verdict |
|---|---|---|---|---|---|
| `tests/test_rebuild_workflow.py` | MIGR-05 | Active; 0 skipped markers in linked tests | No | Behavioral: authentic failure/debt creation, pre-effect state snapshots, exact settlement, terminal evidence rejection | SUFFICIENT |
| `tests/test_phase7_contract_verifier.py` | MIGR-05 evidence manifest | Active; 0 skipped markers | No | Structural mapping equality and adversarial source mutation | INSUFFICIENT for the promised `T-07-21-03` binding |

The verifier tests write isolated mutated copies to prove fail-closed manifest validation; they do not generate expected values from the system under test and are not circular. No requirement-linked disabled test was found.

## Anti-Patterns and Prohibitions

No unreferenced `TBD`, `FIXME`, or `XXX`, placeholder implementation, or disabled requirement-linked test appears in the five Plan 23 files. Scoped Ruff passes.

The production diff adds one private guard only. It adds no lifecycle state, persistence field, schema, lock, queue, lease, journal, sidecar, listing/adoption source, online-worker protocol, cross-resource ACID claim, production obstore integration, or live/platform/performance qualification claim. Ordinary store operations remain outside this explicit maintenance path.

## Decision Coverage

All **22/22** trackable `07-CONTEXT.md` decisions are represented in shipped artifacts according to the non-blocking decision-coverage gate. This lexical coverage does not repair the missing exact `T-07-21-03` verifier edge.

## Human Verification Required

None — this is an infrastructure/library phase, and both the repaired behavior and the remaining verifier-binding defect are deterministically testable.

## Deferred Items

Live PostgreSQL/AWS S3, Windows, supported Python versions, packaging, and performance qualification are explicitly Phase 8 work. ADR 0001's accepted invisible/unattributed pre-checkpoint orphan, non-cross-resource ACID, and topology-specific bounded progress remain accepted limits rather than Phase 7 gaps.

## Gaps Summary

The prior data-loss path is closed in production code with strong behavioral evidence. One small but blocking plan-contract gap remains: bind the exact forward-fence/resume selector to `T-07-21-03` and make the Plan 23 self-test assert that edge. This requires only two narrow verifier/test edits and no lifecycle or coordination change.

---

_Verified: 2026-09-12T02:05:49Z_
_Verifier: the agent (gsd-verifier)_
