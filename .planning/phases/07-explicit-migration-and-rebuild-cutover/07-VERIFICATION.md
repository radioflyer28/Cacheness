---
phase: 07-explicit-migration-and-rebuild-cutover
verified: 2026-09-11T20:37:01Z
status: gaps_found
score: 3/5 roadmap must-haves verified
requirement_score: 2/4 requirements satisfied
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 2/5
  gaps_closed:
    - "Migration runs now split before mutation, checkpoint bounded authority-attributed candidate batches, and resume or abort attributed STAGING work."
    - "Rebuild writes now use deterministic operation IDs, canonical lifecycle-authority replay, bounded exact receipts, and resumable pre-acceptance states."
    - "Changed payload contracts now reach a concrete store-local handler transformation and publish a destination-identity manifest for the tested version-changing edge."
    - "Shareable plans now contain digest bindings rather than raw catalog or manifest data, and execution authenticates fresh source state."
    - "Fresh PostgreSQL initialization and mutation preflight now re-check persisted activated-offline state."
    - "The fixed verifier now binds claims to AST-validated exact pytest selectors."
  gaps_remaining:
    - "Same-numeric-version/different-format migration is still misclassified as an identity copy."
    - "S3 migration abort operational failures can escape before cleanup debt is checkpointed."
    - "Rebuild cleanup debt is written into terminal ABORTED evidence with no deterministic settlement path."
  regressions: []
gaps:
  - truth: "Supported same-backend metadata and payload format migrations use offline copy-verify-switch semantics and produce the declared destination contract."
    status: failed
    reason: "When source and destination payload formats differ but share the same numeric version, _configured_destination_contract() substitutes the source identity if the destination handler can read it. Inspection/staging then bypass the declared transform and preserve the old format."
    artifacts:
      - path: "src/cacheness/storage/migration.py"
        issue: "Lines 1980-1991 use source readability plus numeric-version equality to redefine the destination identity."
      - path: "tests/test_migration_cutover.py"
        issue: "The transform integration test covers mcap-v1@1 -> mcap-v2@2, not differing formats with the same version."
    missing:
      - "Always derive the target payload format/version from the destination handler's declared current contract; use source support only as a readability decision."
      - "Add a same-version/different-format migration test proving one exact directed transform and a destination-format manifest."
  - truth: "Abort persists every authority-attributed candidate cleanup failure as retryable debt across supported payload participants."
    status: failed
    reason: "OfflineMigrationService.abort() catches OSError and ValueError, but S3 open/delete/absence-proof failures are CacheBlobBackendError, which is neither. Those failures escape before updated STAGING evidence and an AbortReceipt are written."
    artifacts:
      - path: "src/cacheness/storage/migration.py"
        issue: "Lines 3943-3967 omit CacheBlobBackendError from both the snapshot and outer cleanup-failure handling."
      - path: "src/cacheness/storage/backends/s3_backend.py"
        issue: "S3 snapshot and exact-delete operational failures intentionally normalize to CacheBlobBackendError."
      - path: "tests/test_migration_cutover.py"
        issue: "Abort recovery is exercised with OSError only; no deterministic S3 participant error verifies debt checkpointing and retry."
    missing:
      - "Treat typed payload-participant operational failures as retryable cleanup debt while continuing to fail closed for ownership and integrity conflicts."
      - "Add deterministic S3 snapshot/delete failure coverage proving exact debt persistence and later attributed retry settlement without listing or adoption."
  - truth: "Durable authority-attributed rebuild cleanup debt has a deterministic exact settlement path."
    status: failed
    reason: "_abort_rebuild_after_failure() enters ABORTED when a receipt is merely represented by cleanup debt. ABORTED has no legal transition, rebuild resume rejects it, and no production path parses and settles rebuild debt."
    artifacts:
      - path: "src/cacheness/storage/migration.py"
        issue: "Lines 2750-2780 treat debt representation as terminal coverage; resume lines 4219-4243 handle no ABORTED rebuild cleanup path."
      - path: "src/cacheness/storage/migration_evidence.py"
        issue: "ABORTED is terminal in the legal transition map."
      - path: "tests/test_rebuild_workflow.py"
        issue: "test_rebuild_verification_failure_cleans_only_exact_receipts_and_persists_debt asserts terminal ABORTED debt but never restores the participant and retries exact cleanup."
    missing:
      - "Keep rebuild failure evidence in an existing resumable cleanup condition until every exact receipt is retired, or permit a narrowly receipt-bound retry from ABORTED."
      - "Remove debt only after exact deletion or proven absence and enter terminal ABORTED only after settlement."
      - "Extend the failure test to retry and prove evidence plus payload state converge."
deferred:
  - truth: "Live PostgreSQL/AWS S3, Windows, supported-Python matrix, packaging, and performance qualification."
    addressed_in: "Phase 8"
    evidence: "Phase 8 goal and BACK-05/QUAL-01 through QUAL-07 own reproducible release, real-service, platform, and operational-scale evidence."
decision_coverage:
  honored: 22
  total: 22
  not_honored: []
---

# Phase 7: Explicit Migration and Rebuild Cutover Verification Report

**Phase Goal:** With workers stopped, users can explicitly migrate or rebuild supported versioned stores without silent mutation or losing the only valid copy of stored data; the tooling establishes future release migration discipline even though current pre-production layouts may be unsupported.
**Verified:** 2026-09-11T20:37:01Z
**Status:** gaps_found
**Re-verification:** Yes — after Plans 07-12 through 07-19 attempted the six prior gap closures.

## Goal Achievement

### Observable Truths

| # | Roadmap truth | Status | Evidence |
|---|---|---|---|
| 1 | Users can inspect without mutation and receive human- and machine-readable plans with counts, bytes, incompatibilities, and actions. | ✓ VERIFIED | Revision-bound authority inventory feeds one canonical `MigrationPlan`; the named inventory and confidential-plan tests passed. |
| 2 | Supported same-backend migrations use offline copy-verify-switch, retain the prior copy, and never upgrade on ordinary open/initialize. | ✗ FAILED | The tested version-changing transform works, but direct invocation reproduced `mcap-v2@1` resolving to source identity `mcap-v1@1`; the declared transformation is bypassed when numeric versions match. |
| 3 | Interrupted bounded migration resumes idempotently for durably attributed effects without losing the only valid generation. | ✗ FAILED | Local attributed STAGING recovery passes, but S3 `CacheBlobBackendError` can escape migration abort before debt checkpointing, and rebuild cleanup debt becomes terminal without settlement. |
| 4 | Incompatible formats and cross-backend moves have an explicit, scoped, confirmed rebuild path. | ✓ VERIFIED | Include-all/exact-confirmation rebuild, registered source handler reads, canonical destination lifecycle replay, pre-acceptance projection suppression, and explicit acceptance are substantive and tested. |
| 5 | The source-version window is explicit and projections remain derived rather than becoming cutover authority. | ✓ VERIFIED | Current/immediately-previous release policy and rebuild-only historical posture are explicit; projection work remains post-acceptance and cannot authorize publication. |

**Score:** 3/5 truths verified (0 present-but-behavior-unverified).

The accepted post-publication/pre-authority-checkpoint invisible orphan is not a gap. The failures above concern already attributed effects or incorrect destination-contract selection and can be fixed within the existing stopped-worker, single-authority design.

## Requirements Coverage

| Requirement | Status | Evidence / blocking issue |
|---|---|---|
| MIGR-03 | ✓ SATISFIED | Bounded non-mutating inventory and one confidential canonical plan model produce both machine and human output. |
| MIGR-04 | ✗ BLOCKED | Same-version/different-format payload migration can silently become an identity copy; S3 abort does not reliably checkpoint typed operational cleanup failures. |
| MIGR-05 | ✗ BLOCKED | Rebuild cleanup debt has no deterministic settlement route after terminal ABORTED, and S3 migration abort can lose the retry checkpoint for attributed work. |
| MIGR-06 | ✓ SATISFIED | A distinct explicit, confirmed rebuild path exists for incompatible/cross-backend cases and uses registered handlers plus destination BlobStore lifecycle. |

**Coverage:** 2/4 requirements satisfied. No Phase 7 requirement is orphaned; all four IDs appear in plan frontmatter and REQUIREMENTS.md.

## Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `src/cacheness/storage/migration_authority.py` | Bounded candidate descriptors and receipts | ✓ VERIFIED | Exists, substantive, and used by migration staging/verification. The former 256-entry whole-store cap is gone. |
| `src/cacheness/storage/migration_evidence.py` | Authenticated bounded continuation, rebuild receipts, and cleanup debt | ⚠ PARTIAL | Receipt/debt fields are substantive, but terminal ABORTED cannot settle rebuild debt. |
| `src/cacheness/storage/migration.py` | Explicit migration/rebuild coordinator | ✗ PARTIAL | Core workflows are wired, but destination selection and two cleanup paths violate required contracts. |
| Memory/SQLite/PostgreSQL lifecycle authorities | Canonical publication and exact replay | ✓ VERIFIED | Candidate batches, exact mutation replay, and post-load PostgreSQL worker fences are implemented through the existing authority seam. |
| `src/cacheness/handlers.py` | Store-local handler resolution and concrete transforms | ⚠ PARTIAL | Directed transform resolution is wired; the migration coordinator can incorrectly classify a changed same-version format as exact-copy. The optional registration alias remains inconsistent. |
| `docs/STORAGE_MIGRATION.md` and public exports | One explicit operator surface and accurate bounded guarantees | ✓ VERIFIED | Public Python workflow, non-claims, confidentiality boundary, and Phase 8 ownership are present. |
| `tools/verify_phase7_contracts.py` | Exact executable evidence manifest | ✓ VERIFIED | Uses AST-validated `path::test_name` selectors and an exact threat inventory; its current mapping nonetheless omits the three newly identified behaviors. |

## Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| Authority raw inventory | Canonical plan/report | Revision-bound authenticated assessments | ✓ WIRED | Real authority pages feed plan entries and renderings. |
| Configured destination handler | Compatibility edge and transform | `_configured_destination_contract` then transform resolver | ✗ BROKEN | Source readability can redefine a same-version destination format as source identity. |
| Attributed migration candidates | S3 cleanup debt | `abort` snapshot/delete exception normalization | ✗ BROKEN | `CacheBlobBackendError` is outside the caught exception types. |
| Rebuild receipts | Exact cleanup settlement | `_abort_rebuild_after_failure` plus `resume` | ✗ NOT WIRED | Debt is recorded, but no callable path retires it after ABORTED. |
| Digest-only plan records | Authenticated live source state | Execution-time inventory/manifest reread | ✓ WIRED | Catalog/manifest drift is checked before candidate mutation. |
| Persisted PostgreSQL publication state | Ordinary worker admission | Post-open `require_ordinary_worker_access` | ✓ WIRED | Both initialization and direct mutation preflight use the canonical fence. |
| Fixed claim maps | Exact pytest functions | AST selector validation and deterministic pytest union | ✓ WIRED | Removed/renamed mapped functions fail closed. |

## Data-Flow Trace

| Flow | Source | Result | Status |
|---|---|---|---|
| Inspection | Authority inventory pages | Canonical assessments, totals, plan bytes, human rendering | ✓ FLOWING |
| Migration candidate | Authenticated source manifest/payload | Immutable candidate, bounded authority receipt, later activation | ⚠ PARTIAL — incorrect target format for same numeric version |
| Migration abort | Authority-attributed candidate receipt | Exact participant deletion and STAGING debt | ✗ DISCONNECTED for typed S3 operational errors |
| Rebuild cleanup | Authenticated `BlobReceipt` batches | Exact delete/proven absence and evidence convergence | ✗ DISCONNECTED after debt enters ABORTED |
| Rebuild publication | Authenticated source handler value | Projection-free canonical BlobStore receipt, verify, accept | ✓ FLOWING |

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| Entry-complete inspection | Named `test_raw_inventory_pages_every_canonical_entry_without_manifest_filtering` | Passed | ✓ PASS |
| Attributed STAGING resume/abort | Named `test_resume_and_abort_staging_use_only_authority_attributed_batches` | Passed | ✓ PASS |
| Version-changing transform | Named `test_migration_executes_one_handler_transform_and_publishes_destination_manifest_identity` | Passed | ✓ PASS for `mcap-v1@1 -> mcap-v2@2` |
| Same-version format target selection | Direct call with destination `mcap-v2@1` readable from source `mcap-v1@1` | Returned `('mcap-v1', 1)` | ✗ FAIL |
| Rebuild failure cleanup | Named `test_rebuild_verification_failure_cleans_only_exact_receipts_and_persists_debt` | Passed while asserting terminal ABORTED debt | ✗ INSUFFICIENT — no settlement behavior |
| Confidential plan binding | Named `test_machine_plan_digest_binds_sensitive_catalog_and_manifest_without_serializing_them` | Passed | ✓ PASS |
| Fixed all-mode verifier | `uv run --isolated --all-extras --group dev --frozen python tools/verify_phase7_contracts.py --all` | Exit 0; MIGR-03 through MIGR-06 reported PASS | ⚠ MISLEADING PASS — exact mappings do not include the three failing behaviors |

## Probe Execution

No conventional `scripts/**/probe-*.sh` is declared. The phase-specific fixed verifier was rerun independently above and exited zero; its evidence incompleteness is captured as a test-quality issue rather than accepted as goal proof.

## Review Finding Adjudication

| Finding | Verdict | Independent evidence |
|---|---|---|
| CR-01 same-version format migration | CONFIRMED BLOCKER | Production branch at `migration.py:1980-1991`; direct invocation returned the source format despite a different declared destination format. |
| CR-02 S3 abort failure checkpoint | CONFIRMED BLOCKER | S3 snapshot/delete paths raise `CacheBlobBackendError`; it is neither `OSError` nor `ValueError`, the only types caught by migration abort. |
| CR-03 terminal rebuild cleanup debt | CONFIRMED BLOCKER | Debt counts as terminal coverage, ABORTED has no transition, rebuild resume rejects ABORTED, and `rebuild:` debt has no settlement consumer. |
| WR-01 partial abort deleted count | CONFIRMED WARNING | Partial abort returns `deleted_entries=len(candidates)` even when debt remains. State remains STAGING, so this is misleading accounting rather than loss of authority state. |
| WR-02 handler registration aliases | CONFIRMED WARNING | `name=` participates in duplicate validation/logging but is not stored; lookup/list/unregister use `handler.data_type`. Phase 7's normal data-type registration path remains functional. |

## Test Quality Audit

| Test area | Active / skipped | Assertion strength | Verdict |
|---|---|---|---|
| Inspection and confidential plans | Active; no disabled requirement test found | Value + behavioral | Sufficient for MIGR-03 |
| Migration compatibility/transform | Active | Behavioral for differing name and differing version | INSUFFICIENT: no differing-format/same-version case |
| Migration abort recovery | Active | Behavioral for local `OSError` | INSUFFICIENT: no `CacheBlobBackendError`/deterministic S3 abort case |
| Rebuild recovery | Active | Behavioral for response loss and state resume | INSUFFICIENT: cleanup-debt test stops at terminal ABORTED and never retries |
| Fixed verifier self-tests | Active | Structural + subprocess | INSUFFICIENT: exact selectors are exact but the claim map itself omits these behaviors |

No requirement-linked disabled test or circular expected-value generator was found. The issue is missing behavioral dimensions, not selector syntax or disabled coverage.

## Anti-Patterns and Prohibitions

No unreferenced `TBD`, `FIXME`, or `XXX` marker, placeholder implementation, or disabled requirement-linked test was found in the Phase 7 implementation inventory.

The inspected code continues to honor the architectural prohibitions: no ordinary implicit migration, no second lifecycle authority, no new lock/queue/lease/sidecar, no listing-based adoption, no universal native converter, no production obstore adoption, and no Phase 8 qualification claim. The three recommended repairs require only narrower contract selection and settlement through existing evidence/authority seams.

## Decision Coverage

All **22/22** trackable CONTEXT.md decisions are represented in shipped artifacts according to the non-blocking GSD decision-coverage gate. This lexical coverage signal does not override the behavioral failures in D-03, D-10, D-16, D-19, and D-21.

## Human Verification Required

None — this is an infrastructure/library phase. All acceptance criteria and current failures are deterministically verifiable.

## Deferred Items

Live PostgreSQL/AWS S3, Windows, supported Python versions, packaging, and performance remain correctly assigned to Phase 8. The accepted invisible/unattributed pre-checkpoint orphan, lack of perfect orphan reclamation, absence of obstore adoption, non-cross-resource ACID, and typed bounded contention are not reopened as gaps.

## Gaps Summary

Three finite blockers remain:

1. Destination payload identity must come from the destination handler even when source and destination use the same numeric version.
2. Migration abort must turn typed S3 participant operational failures into existing exact cleanup debt and later settle it.
3. Rebuild cleanup debt must remain recoverable until exact deletion/proven absence, reaching terminal ABORTED only after settlement.

These repairs do not require a new lifecycle authority, journal, lock, queue, lease, online writer protocol, listing-based adoption, or stronger ACID guarantee.

---

_Verified: 2026-09-11T20:37:01Z_
_Verifier: the agent (gsd-verifier)_
