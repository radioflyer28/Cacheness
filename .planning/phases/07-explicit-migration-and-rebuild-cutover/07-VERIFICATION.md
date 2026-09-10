---
phase: 07-explicit-migration-and-rebuild-cutover
verified: 2026-09-10T13:18:20Z
status: gaps_found
score: 2/5 roadmap must-haves verified
requirement_score: 1/4 requirements satisfied
behavior_unverified: 0
overrides_applied: 0
gaps:
  - truth: "Supported same-backend migration is bounded and resumable across every staged external effect."
    status: failed
    reason: "Candidate staging writes all external payloads before any bounded batch receipt is checkpointed, while the final candidate digest rejects more than 256 entries; STAGING cannot be aborted and partial abort progress is not durable."
    artifacts:
      - path: "src/cacheness/storage/migration_authority.py"
        issue: "candidate_digest rejects whole-store candidates above 256 entries."
      - path: "src/cacheness/storage/migration.py"
        issue: "stage checkpoints only STAGING before writes and one final receipt afterward; abort excludes STAGING and does not checkpoint per-deletion progress."
    missing:
      - "Authenticated bounded candidate-batch receipts checkpointed after each external batch."
      - "Incremental whole-store digest/count aggregation without a 256-entry store cap."
      - "Resumable STAGING abort and idempotent already-absent deletion handling with cleanup debt."
  - truth: "Interrupted rebuilds resume or clean only their exact run-owned destination entries."
    status: failed
    reason: "Rebuild destination receipts live only in a local list until the full stage completes; resume has no rebuild-state path, and verification failure passes an empty receipt set to cleanup before recording ABORTED."
    artifacts:
      - path: "src/cacheness/storage/migration.py"
        issue: "stage_rebuild has no per-batch durable receipts; verify_rebuild calls cleanup with (); resume handles only migration states."
      - path: "src/cacheness/storage/migration_evidence.py"
        issue: "The state vocabulary contains rebuild states but no exact staged destination receipt collection sufficient for recovery."
    missing:
      - "Persist exact BlobReceipt identity for each bounded rebuild batch."
      - "Resume branches for REBUILDING and rebuild verification states."
      - "Receipt-bound cleanup/debt on rebuild failure and crash tests around every boundary."
  - truth: "Declared compatibility destinations and handler-owned transformations determine the produced destination contract."
    status: failed
    reason: "MigrationCompatibilityEdge.supports checks only the source, stage copies source bytes while changing only the locator, and production migration/rebuild code never calls resolve_payload_transformation or transform_payload."
    artifacts:
      - path: "src/cacheness/storage/migration.py"
        issue: "Destination compatibility dimensions are ignored during classification/execution; changed contracts are not transformed."
      - path: "src/cacheness/handlers.py"
        issue: "Transformation resolver exists but is orphaned from production migration/rebuild execution."
    missing:
      - "Bind eligibility to exact source and destination contract dimensions."
      - "Invoke exactly one handler-owned directed transformation for changed payload contracts and construct the destination-version manifest."
      - "An end-to-end differing-format test asserting transform invocation and destination identity."
  - truth: "Machine plans do not disclose raw credentials or secret-bearing application metadata."
    status: failed
    reason: "MigrationPlan serializes catalog_values verbatim and embeds the complete signed manifest as base64; both can contain arbitrary user/handler metadata, contradicting the no-credentials plan claim."
    artifacts:
      - path: "src/cacheness/storage/migration.py"
        issue: "_entry_record emits catalog_values and full manifest bytes into shareable canonical JSON."
      - path: "docs/STORAGE_MIGRATION.md"
        issue: "The runbook claims secrets do not appear in plans without enforcing that boundary."
    missing:
      - "Digest-bound non-secret plan records, with manifest/catalog reread and authentication at execution."
      - "A documented protected-artifact policy if sensitive metadata must be retained."
      - "Secret-sentinel tests through MigrationPlan.to_canonical_bytes()."
  - truth: "A fresh PostgreSQL-backed worker cannot initialize while persisted publication state is activated_offline."
    status: failed
    reason: "Before identity load, publication_state returns synthetic IDLE. BlobStore.initialize is admitted on that result, then authority.initialize/open loads persisted state, but the subsequent preflight_mutation only calls open and does not re-run require_ordinary_worker_access."
    artifacts:
      - path: "src/cacheness/storage/backends/postgresql_lifecycle_authority.py"
        issue: "publication_state returns IDLE when store_identity is None; preflight_mutation validates schema only."
      - path: "src/cacheness/storage/blob_store.py"
        issue: "initialize does not repeat the canonical worker fence after persisted PostgreSQL identity/state is loaded."
    missing:
      - "Re-run the ordinary-worker publication-state fence after PostgreSQL initialization/open validation and before _initialized is set or payload resources are materialized."
      - "Deterministic fresh-instance activated_offline initialize regression coverage."
  - truth: "The fixed Phase 7 verifier binds each requirement, decision, threat, assumption, and prohibition to exact executable evidence."
    status: failed
    reason: "Coverage maps contain test filenames, not path::test selectors; manifest validation proves only that the file exists in the fixed set, so removing a mapped behavior while leaving unrelated tests can still print PASS."
    artifacts:
      - path: "tools/verify_phase7_contracts.py"
        issue: "MIGRATION_REQUIREMENT_NODES, DECISION_NODES, SECURITY_THREAT_NODES, and assumption maps are file-granular."
      - path: "tests/test_phase7_contract_verifier.py"
        issue: "Omission tests remove map/path members but do not remove or rename one mapped test function while keeping its file."
    missing:
      - "Exact path::test_name mappings validated against parsed test functions."
      - "Execute the exact mapped nodes and add remove/rename self-tests."
deferred:
  - truth: "Live PostgreSQL/AWS S3, Windows, supported-Python matrix, and performance qualification."
    addressed_in: "Phase 8"
    evidence: "Phase 8 goal and QUAL/BACK-05 requirements own reproducible release evidence, real services, platforms, and operational scale."
decision_coverage:
  honored: 22
  total: 22
  not_honored: []
---

# Phase 7: Explicit Migration and Rebuild Cutover Verification Report

**Phase Goal:** With workers stopped, users can explicitly migrate or rebuild supported versioned stores without silent mutation or losing the only valid copy of stored data; the tooling establishes future release migration discipline even though current pre-production layouts may be unsupported.
**Verified:** 2026-09-10T13:18:20Z
**Status:** gaps_found
**Re-verification:** No — initial independent goal verification

## Goal Achievement

### Observable Truths

| # | Roadmap truth | Status | Evidence |
|---|---|---|---|
| 1 | Non-mutating inspection produces human and machine plans with complete classifications, counts, bytes, incompatibilities, and actions. | ✓ VERIFIED | Revision-bound inventory is paged through authority implementations; `MigrationPlan` owns both canonical JSON and human rendering. The named raw-inventory spot-check passed. |
| 2 | Same-backend migration uses offline copy-verify-switch, retains the prior copy, and never upgrades on ordinary open/initialize. | ✗ FAILED | Happy-path activation and ordinary no-upgrade tests exist, but `candidate_digest()` caps the complete candidate at 256 after all payload writes, `abort()` rejects STAGING, and PostgreSQL fresh initialization can pass the pre-load synthetic-IDLE fence. |
| 3 | Interrupted migration resumes idempotently from explicit evidence without losing the only valid generation or adopting unexplained state. | ✗ FAILED | Migration and rebuild effects are not checkpointed per bounded batch. Rebuild resume is absent; migration abort cannot retire partial STAGING effects or resume partial deletion. |
| 4 | Incompatible formats and cross-backend moves have an explicit scoped, confirmed rebuild path rather than implicit deletion/universal physical migration. | ✗ FAILED | Include-all confirmation and a normal handler-backed rebuild path exist, but differing destination contracts are not executed through the declared transformation seam and interrupted rebuilds are not resumable/cleanable. |
| 5 | The supported source window is explicit and projections remain derived, never a second cutover authority. | ✓ VERIFIED | Release-window/matrix models are explicit; schemas 8/4 are pinned as the first baseline; projection rebuild is post-activation and cannot authorize publication. No later-phase evidence contradicts this. |

**Score:** 2/5 roadmap truths verified.

The goal is not achieved: the normal one-entry flows work, but already-promised recovery and compatibility behavior fails at deterministic local boundaries. None of these failures requires stronger cross-resource ACID, online writer coordination, or a new lifecycle authority.

## Requirements Coverage

| Requirement | Status | Evidence |
|---|---|---|
| MIGR-03 | ✓ SATISFIED | Bounded non-mutating raw inventory, canonical machine plan, and human rendering from the same model are implemented and exercised. |
| MIGR-04 | ✗ BLOCKED | Copy-verify-switch cannot handle a 257-entry store without stranding untracked external effects; PostgreSQL initialize has an activated-offline fence gap. |
| MIGR-05 | ✗ BLOCKED | Exact per-batch maintenance evidence is missing for migration and rebuild; rebuild resume/cleanup and partial abort progress are not recoverable. Plan serialization also violates the promised no-secret-output boundary. |
| MIGR-06 | ✗ BLOCKED | Confirmed rebuild exists for the happy path, but declared destination compatibility/handler transformation is ignored and interrupted rebuild cannot resume safely. |

**Requirement score:** 1/4 satisfied.

No Phase 7 requirement is orphaned: MIGR-03 through MIGR-06 all appear in plan frontmatter and REQUIREMENTS.md.

## Required Artifacts

| Artifact group | Expected | Status | Details |
|---|---|---|---|
| `migration.py`, `migration_authority.py`, `migration_evidence.py` | Complete explicit migration/rebuild coordinator, authority receipts, resumable evidence | ✗ PARTIAL | Substantive and imported, but batch attribution/recovery and destination-contract execution are incomplete. |
| Memory/SQLite authorities | Whole-store activation, retained prior, rollback/finalize | ✓ VERIFIED | Authority-owned candidate/prior publication is substantive and wired; happy-path lifecycle tests pass. |
| PostgreSQL authority | Same semantic authority contract and worker fencing | ✗ PARTIAL | Deterministic authority transitions exist, but fresh-instance BlobStore initialization can pass before persisted activated-offline state is checked. |
| Handler transformation seam | Exact store-local directed transform | ⚠️ ORPHANED | The resolver and base method exist and unit tests call the resolver, but production migration/rebuild never calls them. |
| Public storage exports and runbook | One explicit Python-only operator surface | ✓ VERIFIED | Public exports resolve and normal workflow documentation/tests are present; the plan confidentiality claim needs correction. |
| Fixed verifier and validation ledger | Exact executable coverage for all fixed contracts | ✗ PARTIAL | The tool runs and reports PASS, but its semantic mappings are only file-level. |

The automated artifact helper reported all declared source artifacts present and all pattern links green (except Plan 02's planning-file paths, which the helper cannot resolve). Those are existence/pattern results only and do not override the behavioral failures above.

## Key Link and Data-Flow Verification

| Flow | Status | Details |
|---|---|---|
| Authority raw inventory → authenticated assessments → canonical plan/report | ✓ FLOWING | Real authority pages feed plan entries and aggregate renderings. |
| Source payload → staged candidate → verified receipt → authority activation | ✗ BROKEN AT SCALE/INTERRUPTION | Payload writes precede durable batch evidence; the final receipt rejects more than 256 entries. |
| Rebuild source `open_entry()` → destination `put_entry()` → verification/acceptance | ✗ RECOVERY DISCONNECTED | Normal values flow, but exact destination receipts disappear on crash and cleanup is called with an empty set after verification failure. |
| Compatibility edge → destination format identity → handler transformation | ✗ NOT WIRED | Edge destination is not consulted and the production coordinator never invokes transformation resolution/execution. |
| Activated-offline PostgreSQL state → fresh BlobStore.initialize fence | ✗ NOT WIRED | The pre-initialize check sees synthetic IDLE and no post-load worker fence is performed. |
| Canonical activation → optional projection rebuild | ✓ FLOWING | Projection outcome is derived and cannot revoke authority publication. |

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| Entry-complete bounded inventory | `uv run --frozen pytest -q tests/test_migration_inspection.py::test_raw_inventory_pages_every_canonical_entry_without_manifest_filtering` | Passed (parameterized; included in 4 total cases) | ✓ PASS |
| Explicit whole-store happy-path activation | `uv run --frozen pytest -q tests/test_migration_cutover.py::test_memory_tracer_requires_explicit_whole_store_activation` | Passed | ✓ PASS |
| Normal registered-handler rebuild path | `uv run --frozen pytest -q tests/test_rebuild_workflow.py::test_rebuild_uses_registered_source_handler_and_destination_blobstore_lifecycle` | Passed | ✓ PASS |
| Fixed verifier quick mode | `uv run --frozen python tools/verify_phase7_contracts.py --quick` | Exit 0; printed all four MIGR IDs PASS | ⚠️ MISLEADING PASS — CR-06 explains why this is not sufficient evidence |

## Review Finding Adjudication

| Finding | Verdict | Independent evidence |
|---|---|---|
| CR-01 stranded migration candidates | CONFIRMED BLOCKER | `_MAX_CANDIDATE_ENTRIES = 256`; all writes occur before `candidate_digest()` and the STAGED checkpoint; abort's allowed-state set omits STAGING. |
| CR-02 rebuild cannot resume/clean interruption | CONFIRMED BLOCKER | `stage_rebuild()` holds receipts locally; `resume()` has no rebuild branch; `verify_rebuild()` invokes `_abort_rebuild_after_failure(..., ())`. |
| CR-03 destination/transform ignored | CONFIRMED BLOCKER | `MigrationCompatibilityEdge.supports()` compares only `versions == source`; production references to `resolve_payload_transformation`/`transform_payload` are absent. |
| CR-04 plan metadata confidentiality | CONFIRMED BLOCKER | `_entry_record()` emits thawed `catalog_values` and base64 full manifest into `to_canonical_bytes()`. Existing secret tests cover evidence/errors, not arbitrary plan catalog metadata. |
| CR-05 PostgreSQL initialization fence | CONFIRMED BLOCKER | `publication_state()` returns IDLE while identity is unloaded; post-initialize `preflight_mutation()` only calls `open()`, so it does not invoke the worker-access fence. |
| CR-06 verifier evidence binding | CONFIRMED BLOCKER | Fixed maps contain filenames only and `validate_fixed_manifest()` checks membership/existence, not exact test function selectors. |
| WR-01 evidence directory durability | CONFIRMED WARNING | `_atomic_write()` fsyncs the temporary file and calls `os.replace()` but never fsyncs the parent directory. Canonical store safety remains intact, so this is not promoted to a phase blocker. |
| WR-02 inherited failing transform accepted | CONFIRMED WARNING | Registration tests only callability; subclasses inherit the callable base method that always raises `CacheFormatError`. Require a concrete override when edges are declared. |

## Test Quality Audit

| Test area | Active/Skipped | Assertion level | Verdict |
|---|---|---|---|
| Inspection/plan | Active; no requirement-linked disabled marker found | Value + behavioral | Sufficient for MIGR-03 |
| Migration cutover/evidence | Active; no requirement-linked disabled marker found | Behavioral happy paths | Insufficient boundary coverage: no 257-entry, post-first-batch crash, STAGING abort, or partial-abort retry case |
| Rebuild | Active; no requirement-linked disabled marker found | Behavioral happy paths/failure cleanup | Insufficient recovery coverage: no persisted per-entry receipt, REBUILDING resume, or verification cleanup proof |
| Handler transforms | Active | Resolver/value assertions | Insufficient integration: no production differing-contract transformation test |
| PostgreSQL authority | Active deterministic adapter tests | SQL/value/behavioral | Missing fresh-authority BlobStore.initialize fence test |
| Fixed verifier self-tests | Active | Structural mutation tests | Insufficient binding: removing/renaming a mapped function while leaving its file is not tested |

No circular expected-value generator was found in the requirement-linked tests. The central weakness is not disabled tests; it is that strong happy-path assertions omit the exact recovery and wiring boundaries named above.

## Anti-Patterns and Prohibitions

No `TBD`, `FIXME`, `XXX`, `TODO`, `HACK`, placeholder text, or empty implementation marker was found in the Phase 7 production/test inventory.

Two Phase 01 prohibitions are observably violated: partial/unattributed candidates are possible during STAGING, and canonical plans can serialize raw credential-bearing metadata. The remaining architecture prohibitions (no second lifecycle authority, no new queue/lease/sidecar/online writer protocol, no projection authority, no live-service qualification claim) are supported by the inspected code and fixed-scope documentation.

## Decision Coverage

The non-blocking decision-coverage gate reported: **22/22 trackable CONTEXT decisions honored by shipped artifacts**. This is a substring/translation coverage signal only; it does not negate the behavioral defects in D-09, D-10, D-19, D-20, D-21, and D-22.

## Human Verification

N/A — this is an infrastructure/library phase with no user-facing visual flow. The observable failures are deterministic code-contract defects, not matters requiring manual UAT.

## Deferred Qualification

Live PostgreSQL/AWS S3, native Windows, supported-Python-version matrices, and performance/scale distributions remain correctly assigned to Phase 8. They were not used as failures or as passing evidence here. The accepted clear/delete snapshot race is also not reopened by this verification.

## Gaps Summary

Six finite blockers prevent Phase 7's goal: durable bounded attribution for migration staging/abort, durable rebuild recovery, real destination-contract/handler-transform execution, plan metadata confidentiality, a post-load PostgreSQL worker fence, and exact test-node binding in the fixed verifier. WR-01 and WR-02 remain warnings. These gaps can be closed within the existing single-authority, stopped-worker architecture.

---

_Verified: 2026-09-10T13:18:20Z_
_Verifier: the agent (gsd-verifier)_
