---
phase: 7
slug: explicit-migration-and-rebuild-cutover
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-09-09
updated: 2026-09-10
---

# Phase 7 — Validation Evidence

Phase 7 has fixed deterministic local evidence for explicit offline migration and rebuild. It does not qualify live PostgreSQL/AWS S3, Windows, a Python-version matrix, or performance distributions.

## Fixed Acceptance Boundary

tools/verify_phase7_contracts.py owns literal reviewed tuples for production/public/documentation paths, Plans 01–11, focused test nodes, requirements, decisions, 46 declared security threats, four flagged assumptions, and the eight Plan 01 prohibitions. It rejects missing, duplicate, or root-escaping entries before execution and never derives scope from Git, test discovery, comments, strings, a latest run, a candidate, or a live-service observation.

Self-tests mutate source inventory and exercise adversarial executable fixtures for ordinary maintenance reachability, a second authority, coordination primitives, candidate/listing adoption, handler reads before verification, unbounded inventory, secret output, false qualification, altered coverage evidence, and false PASS rendering.

## Execution Evidence

| Command | Environment | Exit | Result |
|---|---|---:|---|
| uv run --frozen pytest -q tests/test_phase7_contract_verifier.py -x -o log_cli=false | frozen base Python 3.13.15 | 0 | 14 verifier self-tests passed |
| uv run --frozen python tools/verify_phase7_contracts.py --quick | frozen base Python 3.13.15 | 0 | fixed local quick inventory passed; deterministic PostgreSQL contract explicitly NOT RUN because base lacks optional psycopg |
| uv run --isolated --all-extras --group dev --frozen python tools/verify_phase7_contracts.py --all | isolated locked all-extras/dev Python 3.13.15 | 0 | fixed Phase 7 inventory, active detector comparison, deterministic non-live suite, and scoped Ruff passed |
| uv run --isolated --all-extras --group dev --frozen pytest -q [15 fixed Phase 7 modules] --junitxml=… -o log_cli=false | isolated locked all-extras/dev Python 3.13.15 | 0 | 140 tests; 0 failures, 0 errors, 0 skips |
| uv run --isolated --all-extras --group dev --frozen pytest -q --ignore=tests/integration/test_postgresql_authority.py --ignore=tests/integration/test_remote_topology.py --ignore=tests/integration/test_s3_generation.py --junitxml=… -o log_cli=false | isolated locked all-extras/dev Python 3.13.15 | 0 | 1311 tests; 0 failures, 0 errors, 9 documented platform/optional skips |
| uv run --isolated --all-extras --group dev --frozen ruff check [fixed Phase 7 Python inventory] | isolated locked all-extras/dev Python 3.13.15 | 0 | all scoped paths clean |

The all gate invokes the active installed gsd-core/bin/lib/api-coverage.cjs --json over the fixed Roadmap/Plan scope and compares its typed result to 07-COVERAGE.md.

### Non-live suite inventory

The non-live suite excludes exactly these Phase 8-owned real-service modules. They are NOT RUN / NOT QUALIFIED, never skips or passes:

- tests/integration/test_postgresql_authority.py
- tests/integration/test_remote_topology.py
- tests/integration/test_s3_generation.py

## Plan Task Ledger

| Task ID | Named evidence | Status |
|---|---|---|
| 07-01-T1 | tests/test_migration_cutover.py | ✅ |
| 07-01-T2 | tests/test_stored_compatibility.py | ✅ |
| 07-02-T1 | accepted D-02 in 07-CONTEXT.md; fixed decision map | ✅ |
| 07-02-T2 | accepted D-08 in 07-CONTEXT.md; fixed decision map | ✅ |
| 07-02-T3 | accepted D-15 in 07-CONTEXT.md; fixed decision map | ✅ |
| 07-03-T1 | tests/test_migration_plan_contract.py | ✅ |
| 07-03-T2 | plan-contract, inspection, and stored-compatibility tests | ✅ |
| 07-04-T1 | inspection and lifecycle-authority tests | ✅ |
| 07-04-T2 | PostgreSQL lifecycle-authority contract | ✅ deterministic adapter only |
| 07-05-T1 | tests/test_migration_run_evidence.py | ✅ |
| 07-05-T2 | evidence and cutover tests | ✅ |
| 07-06-T1 | accepted RQ-01; schema 8/4 verifier pin | ✅ |
| 07-06-T2 | cutover and lifecycle-authority tests | ✅ |
| 07-06-T3 | cutover, evidence, and projection SQL tests | ✅ |
| 07-07-T1 | PostgreSQL lifecycle-authority contract | ✅ deterministic adapter only |
| 07-07-T2 | remote-contract and S3 backend tests | ✅ deterministic adapter only |
| 07-08-T1 | cutover and lifecycle-authority tests | ✅ |
| 07-08-T2 | cutover and evidence tests | ✅ |
| 07-09-T1 | tests/test_rebuild_workflow.py | ✅ |
| 07-09-T2 | tests/test_handler_registration.py | ✅ |
| 07-09-T3 | rebuild, handler, and BlobStore read-contract tests | ✅ |
| 07-10-T1 | public-contract and stored-compatibility tests | ✅ |
| 07-10-T2 | test_external_api_coverage_declaration_is_detector_backed | ✅ |
| 07-11-T1 | verifier self-tests and fixed quick gate | ✅ |
| 07-11-T2 | fixed all gate, focused inventory, non-live suite, scoped Ruff | ✅ |

## Requirement, Decision, and Research Coverage

| Item | Named executable evidence | Status |
|---|---|---|
| MIGR-03 | inspection, plan-contract, and public-contract tests | ✅ |
| MIGR-04 | cutover, stored-compatibility, and lifecycle-authority tests | ✅ |
| MIGR-05 | evidence, cutover, projection SQL, and remote-contract tests | ✅ |
| MIGR-06 | rebuild, handler-registration, and BlobStore read-contract tests | ✅ |
| D-01…D-22 | literal DECISION_NODES map checked for exact equality and non-empty fixed-module evidence | ✅ |
| RQ-01 | SQLite 8 and PostgreSQL schema/capability 4 AST pins plus authority contracts | ✅ deterministic only |
| RQ-02 | public no-CLI test plus literal public/docs inventory | ✅ |
| RQ-03 | cutover/evidence tests and BlobStore ordinary-worker-fence audit | ✅ |
| RQ-04 | handler registration, rebuild, and BlobStore read-contract tests | ✅ |

The four former flagged assumptions remain the exact FLAGGED_ASSUMPTION_NODES map keyed by MIGR-03 through MIGR-06; the verifier fails if any key disappears.

## Security and Prohibition Coverage

All 46 declared threat IDs have literal non-empty SECURITY_THREAT_NODES entries; the verifier rejects any missing or unexpected ID. The reviewed set is exactly T-07-01…T-07-24; T-07-26…T-07-29; T-07-31…T-07-34; T-07-36…T-07-40; T-07-42…T-07-50.

Named maps use inventory evidence for T-07-04/T-07-12…T-07-15; evidence security for T-07-05/T-07-11/T-07-16…T-07-20; authority/cutover for T-07-01/T-07-21…T-07-23/T-07-31…T-07-34; remote adapters for T-07-26…T-07-29; handler/rebuild for T-07-36…T-07-40; public boundary for T-07-42…T-07-45; and final verifier tamper/support evidence for T-07-46…T-07-50.

The ordered eight-row PLAN01_PROHIBITIONS inventory is mutation-tested. Its executable backstops are:

1. ordinary paths cannot reach offline maintenance;
2. listings, candidates, and projections cannot authorize activation;
3. migration code cannot add a lock, queue, or second authority;
4. behavior tests reject historical readers, force paths, and universal conversions;
5. cutover tests reject partial activation and retain the prior store through finalize;
6. evidence/cutover tests reject stale, unauthenticated, unexplained, and incomplete inputs;
7. evidence/public/verifier tests reject raw signing keys, credentials, and provider paths in output; and
8. bounded inventory/evidence and Phase 8 non-claim checks reject unbounded calls and inflated qualification language.

## Known Pre-existing Suite Observation

Before the non-live inventory correction, the initial all-extras gate returned exit 1 because it incorrectly included the three Phase 8 live modules (nine missing-fixture errors) and also observed one tests/test_blob_store_concurrency.py::test_clear_and_delete_converge_after_an_exact_snapshot CacheBlobLifecycleConflictError. The correction changed only the fixed Phase 8 live-module inventory; it did not alter lifecycle/storage code or turn that failure into a pass. The repeated fixed non-live suite passed 1311 tests with zero failures/errors. The one-off concurrency observation remains in deferred-items.md as pre-existing Phase 3 evidence; ADR 0001 and the approved Wave 1 boundary prohibit reopening a race-patch loop here.

## Phase 8 Non-claims

- Live PostgreSQL and AWS S3: NOT QUALIFIED. Deterministic adapter coverage is not real-service evidence.
- Windows/native platform evidence and the supported Python-version matrix: NOT QUALIFIED. The deterministic run retained nine documented platform/optional skips.
- Performance distributions, budgets, and tail-latency claims: NOT QUALIFIED. No Phase 7 timing is a runtime correctness or progress guarantee.

## Sign-off

- [x] Every plan task has named deterministic evidence.
- [x] Fixed requirement, decision, threat, assumption, and prohibition maps have no silent row drop.
- [x] Integrity, recovery/progress, and performance claims remain separate.
- [x] wave_0_complete: true
- [x] nyquist_compliant: true

Approval: deterministic local Phase 7 gate complete; Phase 8 owns every listed non-qualification boundary.

