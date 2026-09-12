---
phase: 07-explicit-migration-and-rebuild-cutover
reviewed: 2026-09-12T01:57:20Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - src/cacheness/storage/migration.py
  - src/cacheness/storage/migration_evidence.py
  - tests/test_rebuild_workflow.py
  - tools/verify_phase7_contracts.py
  - tests/test_phase7_contract_verifier.py
findings:
  critical: 0
  warning: 2
  info: 0
  total: 2
status: issues_found
---

# Phase 7: Code Review Report

**Reviewed:** 2026-09-12T01:57:20Z
**Depth:** standard
**Files Reviewed:** 5
**Status:** issues_found

## Summary

The production repair closes the previously demonstrated data-loss path without
adding lifecycle authority or coordination machinery: direct forward operations
authenticate and fence cleanup debt before source or payload access, explicit
resume limits settlement to the two failure-producing states, and accepted
evidence rejects debt. The focused implementation tests pass.

Two verification defects remain. The fixed verifier omits one exact threat
mapping required by Plan 07-23, and the new end-to-end test does not exercise
resume's rejection of any forbidden debt-bearing state or settlement from
`REBUILD_VERIFYING`. These do not invalidate the production fence, but they leave
the claimed fail-closed state matrix incompletely enforced against regression.

This review does not request cross-resource ACID, stronger progress guarantees,
or any lock, queue, lease, journal, sidecar, listing authority, or new lifecycle
mechanism.

## Narrative Findings (AI reviewer)

## Warnings

### WR-01 [WARNING]: Plan 21 terminal-state threat is not bound to the new forward-fence selector

**Files:** `tools/verify_phase7_contracts.py:325` and
`tests/test_phase7_contract_verifier.py:419-426`

**Issue:** Plan 07-23 explicitly requires the new
`test_rebuild_cleanup_debt_fences_forward_methods_and_resume_settles_exact_receipts`
selector to be mapped to `T-07-21-03` as well as the Plan 23 threats. The verifier
still maps `T-07-21-03` only to the older forged-debt test, and the new verifier
self-test checks the Plan 23 threat rows but never asserts the required Plan 21
binding. Consequently, `validate_fixed_manifest()` can pass while the literal
cross-plan ownership contract in 07-23 is absent.

**Fix:** Add the forward-fence selector to
`SECURITY_THREAT_NODES["T-07-21-03"]`, then update both exact mapping assertions
in `test_fixed_manifest_maps_current_three_gap_repairs_exactly` and
`test_fixed_manifest_maps_rebuild_cleanup_debt_fence_exactly` so removing that
cross-plan selector fails validation.

### WR-02 [WARNING]: Resume's state-before-effect contract is only partially tested

**File:** `tests/test_rebuild_workflow.py:973-1058`

**Issue:** The first loop exercises only the three direct forward methods. The
sole `resume()` assertion uses debt in `REBUILDING`. Plan 07-23 requires explicit
coverage that debt in `REBUILD_STAGED`, `REBUILD_VERIFIED`, `REBUILD_ACCEPTED`,
and unrelated states fails before participant access, and that both permitted
states (`REBUILDING` and `REBUILD_VERIFYING`) settle exact receipts. The accepted
evidence model test covers construction/loading rejection, but it does not call
`resume()`; no new test spies on participant access for the other forbidden
states, and no test settles debt from `REBUILD_VERIFYING`. A future reordering or
state-set regression in `resume()` could therefore retain the exact mapped test
name and still false-green the stated state matrix.

**Fix:** Parameterize authenticated evidence across every representable forbidden
debt state and assert `resume()` raises before `get_entry_info`, `delete`, or
`delete_migration_payload`. Add a second settlement case for authentic
`REBUILD_VERIFYING` debt, asserting exact receipt retirement and debt-free
`ABORTED` evidence. Keep these checks in the exact mapped Plan 23 selector (or add
equally exact selectors to the verifier mapping).

---

_Reviewed: 2026-09-12T01:57:20Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
