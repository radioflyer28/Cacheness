---
phase: 07-explicit-migration-and-rebuild-cutover
verified: 2026-09-12T00:32:01Z
status: gaps_found
score: 4/5 roadmap must-haves verified
requirement_score: 3/4 requirements satisfied
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 3/5
  gaps_closed:
    - "Same-numeric-version/different-format migration now uses the destination handler identity and executes the exact directed transform."
    - "Typed S3 abort operational failures now checkpoint exact attributed debt, report truthful per-call deletion counts, and settle on explicit retry while integrity and ownership mismatches fail closed."
    - "Rebuild cleanup debt now has an authenticated receipt-bound settlement path and reaches terminal ABORTED only after every recorded receipt is settled."
  gaps_remaining:
    - "Authentic rebuild cleanup debt does not fence direct stage/verify/accept progression and can reach REBUILD_ACCEPTED; a later resume deletes an accepted entry before its evidence checkpoint fails."
  regressions: []
gaps:
  - truth: "An interrupted bounded migration or rebuild resumes idempotently for durably attributed effects without losing the only valid generation."
    status: failed
    reason: "Direct stage_rebuild(), verify_rebuild(), and accept_rebuild() preserve authentic cleanup debt instead of requiring settlement. resume() dispatches debt before validating the rebuild state, so accepted evidence with debt causes exact deletion of an accepted destination entry followed by a failed REBUILD_ACCEPTED evidence checkpoint."
    artifacts:
      - path: "src/cacheness/storage/migration.py"
        issue: "The direct rebuild methods carry cleanup_debt through forward states, and resume settles debt before rejecting terminal REBUILD_ACCEPTED."
      - path: "src/cacheness/storage/migration_evidence.py"
        issue: "The evidence model forbids debt only in ABORTED, allowing REBUILD_ACCEPTED evidence with outstanding cleanup debt."
      - path: "tools/verify_phase7_contracts.py"
        issue: "MIGR-05 and Plan 21 mappings omit the direct-method debt-fence behavior, so the fixed verifier can report MIGR-05 PASS while the data-loss path exists."
    missing:
      - "At the existing maintenance coordinator seam, reject direct stage_rebuild(), verify_rebuild(), and accept_rebuild() whenever authenticated rebuild cleanup debt exists and direct the caller to explicit resume settlement."
      - "Permit resume settlement only from the defined nonterminal rebuild cleanup states and reject cleanup debt in REBUILD_ACCEPTED before any payload effect."
      - "Make MaintenanceRunEvidence reject REBUILD_ACCEPTED with cleanup debt."
      - "Add one exact regression proving all forward methods fail before payload/authority mutation, resume alone settles the exact receipts, and the fixed verifier maps that selector to MIGR-05 and the applicable decisions/threat."
deferred:
  - truth: "Live PostgreSQL/AWS S3, Windows, supported-Python matrix, packaging, and performance qualification."
    addressed_in: "Phase 8"
    evidence: "Phase 8 owns reproducible real-service, platform, packaging, and operational-scale release evidence."
decision_coverage:
  honored: 22
  total: 22
  not_honored: []
---

# Phase 7: Explicit Migration and Rebuild Cutover Verification Report

**Phase Goal:** With workers stopped, users can explicitly migrate or rebuild supported versioned stores without silent mutation or losing the only valid copy of stored data; the tooling establishes future release migration discipline even though current pre-production layouts may be unsupported.
**Verified:** 2026-09-12T00:32:01Z
**Status:** gaps_found
**Re-verification:** Yes — after Plans 07-20 through 07-22.

## Goal Achievement

### Observable Truths

| # | Roadmap truth | Status | Evidence |
|---|---|---|---|
| 1 | Users can inspect without mutation and receive human- and machine-readable plans with counts, bytes, incompatibilities, and actions. | ✓ VERIFIED | Previous evidence remains present and wired; the fixed exact-selector run continued to report MIGR-03 PASS. No Phase 20-22 change touched the inspection model or ordinary-open boundary. |
| 2 | Supported same-backend migrations use offline copy-verify-switch, retain the prior copy, and never upgrade on ordinary open/initialize. | ✓ VERIFIED | `_configured_destination_contract()` now always selects the destination handler's declared format/version (`migration.py:1967-1990`). The same-version MCAP regression passed and proves one directed transform plus an authenticated `mcap-v2@1` candidate manifest. |
| 3 | Interrupted bounded migration can resume idempotently for durably attributed effects without losing the only valid generation. | ✗ FAILED | Migration abort and immediate rebuild-debt resume now work, but an independently reproduced public-method sequence reaches `REBUILD_ACCEPTED` with debt; `resume()` deletes the accepted `first` entry and then raises an evidence-transition error, leaving accepted evidence with the entry absent. |
| 4 | Incompatible formats and cross-backend moves have an explicit, scoped, confirmed rebuild path. | ✓ VERIFIED | The distinct include-all/exact-confirmation handler-backed rebuild remains present and wired. The new gap concerns recovery ordering after authentic debt, not absence of the explicit rebuild path. |
| 5 | The source-version window is explicit and projections remain derived rather than becoming cutover authority. | ✓ VERIFIED | The current/immediately-previous release policy, rebuild-only historical posture, and post-acceptance derived projections remain unchanged and covered by the fixed selectors. |

**Score:** 4/5 truths verified (0 present-but-behavior-unverified).

The accepted post-publication/pre-authority-checkpoint invisible orphan remains outside guaranteed cleanup and is not a gap. The blocker uses already authenticated, authority-attributed receipts and is repairable inside the existing coordinator/evidence seam without new coordination or stronger cross-resource ACID.

## Previous Gap Closure

| Previous blocker | Implementation evidence | Behavioral evidence | Status |
|---|---|---|---|
| Same-version/different-format target collapsed to identity | Destination identity is unconditionally `(handler.payload_format, handler.payload_format_version)` at `migration.py:1967-1990`. | `test_same_version_different_format_uses_exact_directed_transform_and_destination_manifest` passed. | ✓ CLOSED |
| Typed S3 abort errors escaped before debt checkpoint | Abort imports and narrowly handles `CacheBlobBackendError` alongside `OSError`; exact debt and per-call completion accounting remain authority-attributed. | S3 snapshot/delete retry, truthful accounting, and fail-closed ownership tests passed. | ✓ CLOSED |
| Rebuild cleanup debt had no settlement path | `_settle_rebuild_cleanup_debt()` validates debt against authenticated receipts and authority replay, deletes/proves exact absence, checkpoints progress, and writes terminal ABORTED only after full retirement (`migration.py:2813-2885`). | Settlement, terminal-ABORTED rejection, forged-debt, and changed-owner tests passed. | ✓ CLOSED, but forward-method fencing is incomplete |

## Requirements Coverage

| Requirement | Status | Evidence / blocker |
|---|---|---|
| MIGR-03 | ✓ SATISFIED | Non-mutating bounded inventory and one canonical human/machine plan remain implemented and exactly mapped. |
| MIGR-04 | ✓ SATISFIED | Same-version/different-format migrations now select the declared destination identity and execute one exact handler-owned transform; ordinary opens remain validation-only. |
| MIGR-05 | ✗ BLOCKED | Authentic rebuild cleanup debt can progress to acceptance, after which resume deletes an accepted entry before failing its evidence checkpoint. This violates safe deterministic recovery and the no-loss phase goal. |
| MIGR-06 | ✓ SATISFIED | A distinct explicit, scoped, confirmed rebuild through registered source handlers and destination BlobStore exists. |

**Coverage:** 3/4 requirements satisfied. All four Phase 7 requirement IDs are claimed by plans and mapped in REQUIREMENTS.md; none is orphaned.

## Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `src/cacheness/storage/migration.py` | Exact destination selection and deterministic migration/rebuild recovery | ⚠ PARTIAL | Destination selection, S3 abort debt, exact rebuild settlement, and truthful counts are substantive and wired. Forward rebuild methods preserve debt through `REBUILD_STAGED`, `REBUILD_VERIFIED`, and `REBUILD_ACCEPTED`; resume performs cleanup before terminal-state rejection. |
| `src/cacheness/storage/migration_evidence.py` | Authenticated bounded evidence with legal terminal invariants | ⚠ PARTIAL | ABORTED debt is rejected at lines 719-725, but REBUILD_ACCEPTED debt is allowed. |
| `src/cacheness/storage/memory_lifecycle_authority.py` | Stable operation replay for exact cleanup ownership | ✓ VERIFIED | Promoted mutation replay retains a copied `EntrySnapshot`, preventing later key owners from rewriting historical operation evidence. |
| `tests/test_migration_cutover.py` | Destination transform and migration-abort regressions | ✓ VERIFIED | The exact same-version transform, partial-count, and fail-closed ownership tests are active and passed. |
| `tests/test_migration_remote_contract.py` | Deterministic S3 abort debt/retry contract | ✓ VERIFIED | The Moto-backed snapshot and delete failures checkpoint debt, avoid listing, and settle on retry; the exact test passed. |
| `tests/test_rebuild_workflow.py` | Receipt-bound rebuild cleanup recovery | ⚠ PARTIAL | Existing tests prove immediate `resume()` settlement and forged/changed-owner refusal, but none attempts forward methods while debt exists. |
| `tools/verify_phase7_contracts.py` | Exact fail-closed Phase 7 evidence manifest | ✗ PARTIAL | It maps the three planned repairs exactly but omits the direct-method debt-fence invariant; it reports MIGR-05 PASS despite the reproduced loss path. |

The artifact verifier reported 10/10 Plan 20-22 artifacts present/substantive and 10/10 declared key links pattern-wired. Behavioral tracing supersedes those syntactic results for the incomplete rebuild recovery link.

## Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| Destination handler | Candidate manifest | Declared current format/version then exact directed transform | ✓ WIRED | Equal numeric versions no longer collapse distinct formats. |
| S3 participant failure | Migration cleanup debt | Narrow typed operational-error normalization and exact authority attribution | ✓ WIRED | Both snapshot and delete failure/retry cases pass without listing/adoption. |
| Rebuild receipts | Cleanup settlement | Authenticated receipt validation plus existing authority operation replay | ✓ WIRED | Immediate explicit resume settles exact effects and preserves changed current owners. |
| Rebuild cleanup debt | Forward rebuild methods | Debt guard before stage/verify/accept | ✗ NOT WIRED | Each method carries debt forward rather than refusing progression. |
| Rebuild cleanup debt | Terminal-state validation | State validation before participant deletion | ✗ BROKEN | `resume()` checks debt first at `migration.py:4334-4337`; accepted evidence can trigger deletion before the checkpoint rejects the transition. |
| Exact fixed verifier | Direct-method debt fence | Exact regression selector in MIGR-05/decision/threat maps | ✗ NOT WIRED | The current map covers immediate settlement only, allowing a false green. |

## Data-Flow Trace

| Flow | Source | Result | Status |
|---|---|---|---|
| Same-version format migration | Destination handler contract plus authenticated source manifest | Directed transformed bytes and destination-identity manifest | ✓ FLOWING |
| Migration abort | Authority-attributed candidate descriptors | Exact debt, truthful completed count, explicit retry to ABORTED | ✓ FLOWING |
| Immediate rebuild cleanup recovery | Authenticated `BlobReceipt` batches and authority replay | Exact delete/absence proof, debt retirement, terminal ABORTED | ✓ FLOWING |
| Debt-bearing rebuild progression | Authentic REBUILDING evidence | Debt propagates through stage, verify, and accept | ✗ UNSAFE FLOW |
| Resume after debt-bearing acceptance | REBUILD_ACCEPTED evidence with exact debt | Accepted payload is deleted; evidence update then fails | ✗ DATA LOSS / DISAGREEMENT |

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| Planned gap repairs | Eight exact Plan 20-21 selectors in one isolated frozen pytest invocation | 11 passed (parametrization included) | ✓ PASS |
| Fixed-manifest mappings and false-PASS renderer | Two exact verifier self-tests | 2 passed | ✓ PASS, but scope incomplete |
| Direct-method rebuild-debt sequence | Independent Python reproduction using the public stage/verify/accept/resume methods and existing store fixtures | Reached `REBUILD_ACCEPTED` with one debt item; resume raised `CacheBlobMigrationEvidenceMismatchError`; `destination.get("first")` became `None`; evidence stayed `REBUILD_ACCEPTED` | ✗ FAIL |

## Probe Execution

| Probe | Command | Result | Status |
|---|---|---|---|
| Fixed Phase 7 verifier | `uv run --isolated --all-extras --group dev --frozen python tools/verify_phase7_contracts.py --all` | The tool rendered MIGR-03 through MIGR-06 PASS, then exited 1 because the known nondeterministic Phase 3 clear/delete test failed in its embedded non-live suite. | ⚠ NON-PASS; unrelated suite failure does not explain the Phase 7 gap, while MIGR-05's rendered PASS demonstrates the missing evidence dimension |

The previously recorded direct deterministic non-live run passed 1,345 tests with 9 skips. This verification observed the already documented `test_clear_and_delete_converge_after_an_exact_snapshot` interleaving failure. Per ADR 0001 and the phase boundary, it is not reclassified as a Phase 7 migration gap and does not justify another race-coordination mechanism.

## Review Finding Adjudication

| Finding | Verdict | Independent evidence |
|---|---|---|
| Current CR-01: debt does not fence direct rebuild progression | CONFIRMED BLOCKER | Static trace shows debt copied at `migration.py:2995`, `3031`, `3101`, and `3144`; independent execution reached accepted-with-debt and then deleted the accepted entry before evidence failure. |
| Current WR-01: fixed verifier omits the direct-method fence | CONFIRMED WARNING / part of blocker closure | MIGR-05 maps immediate settlement and validation selectors, but no selected test attempts stage/verify/accept while debt exists. The mapping self-test passes because it validates the incomplete reviewed set. |
| Prior CR-01: same-version target identity | CLOSED | Production destination selection and exact behavioral test agree. |
| Prior CR-02: typed S3 abort debt | CLOSED | Narrow exception handling and deterministic S3 retry test agree. |
| Prior CR-03: rebuild debt settlement | CLOSED WITH NEW EDGE GAP | The settlement path works when invoked immediately; it is unsafe after forward progression because debt is not fenced. |
| Prior WR-01: abort deletion count | CLOSED | Exact partial and replay accounting assertions pass. |
| Prior WR-02: handler `name=` alias | DEFERRED | Explicitly recorded in `deferred-items.md`; it is unrelated to Phase 7 migration/rebuild safety. |

## Test Quality Audit

| Test area | Active / skipped | Assertion strength | Verdict |
|---|---|---|---|
| Same-version destination transform | Active; no skip marker | Behavioral: transform count, authenticated manifest identity, bytes, activation round-trip | SUFFICIENT |
| Typed S3 abort and migration debt | Active; no skip marker | Behavioral: both failure sites, zero listing, exact debt, retry settlement | SUFFICIENT |
| Rebuild cleanup settlement | Active; no skip marker | Behavioral only for immediate resume and altered ownership | INSUFFICIENT: omits direct progression with authentic debt |
| Fixed verifier self-tests | Active; no skip marker | Structural and subprocess assertions | INSUFFICIENT: prove exactness of the declared map, not completeness of the missing lifecycle invariant |

No requirement-linked disabled tests or circular expected-value generator was found. The defect is a missing behavioral dimension, not a skipped or weakly asserted implementation test.

## Anti-Patterns and Prohibitions

No unreferenced `TBD`, `FIXME`, or `XXX`, placeholder implementation, or disabled requirement-linked test was found in the Plan 20-22 implementation inventory.

No new lock, queue, lease, sidecar, journal, authority, listing/adoption path, cross-resource ACID claim, production obstore adoption, or perfect invisible-orphan reclamation was introduced. The required repair is a narrow ordering/state guard at the existing maintenance coordinator and evidence boundaries.

## Decision Coverage

All **22/22** trackable CONTEXT.md decisions are represented in shipped artifacts according to the non-blocking decision-coverage gate. Lexical coverage does not override the observed D-19/D-20/D-21 recovery failure.

## Human Verification Required

None — this is an infrastructure/library phase and the remaining failure is deterministically reproduced.

## Deferred Items

Live PostgreSQL/AWS S3, Windows, supported Python versions, packaging, and performance remain Phase 8 work. The accepted invisible/unattributed pre-checkpoint orphan, lack of perfect orphan reclamation, non-cross-resource ACID, typed bounded contention, production obstore adoption, and the unrelated handler alias decision are not reopened as Phase 7 gaps.

## Gaps Summary

One finite blocker remains. Authentic rebuild cleanup debt must fence forward rebuild progression and be settled only from permitted nonterminal recovery states. Terminal accepted evidence must reject debt, and `resume()` must reject terminal-state debt before any external deletion. The exact regression must be added to the fixed verifier so MIGR-05 cannot false-green.

This repair requires no new lifecycle authority, state machine, journal, lock, queue, lease, sidecar, listing/adoption behavior, or stronger atomicity guarantee.

---

_Verified: 2026-09-12T00:32:01Z_
_Verifier: the agent (gsd-verifier)_
