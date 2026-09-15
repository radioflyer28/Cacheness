---
phase: 08
slug: production-gates-and-performance-stabilization
status: planned
nyquist_compliant: true
wave_0_complete: false
created: 2026-09-13
updated: 2026-09-15
---

# Phase 08 — Validation Strategy

> Nyquist contract for canonical Plans 08-01 through 08-10 and 08-13 through
> 08-16. Plans 08-11/08-12 are superseded by D-24 and preserved for SEED-007;
> they are not completion evidence. No live credential, exact live run, controlled-
> Linux capture, or immutable release is claimed here. D-23 defers controlled-Linux
> qualification to SEED-006, and D-24 defers BACK-05/publication to SEED-007.

## Validation Invariants

1. Preserve ADR 0001’s four separate claim classes: safety, deterministic recovery,
   topology-specific progress, and performance. A result in one class never proves
   another.
2. Preserve the Phase 07.1 participant boundary: built-in payload operations flow
   through `ObstoreGenerationIO`; there is no production boto3 fallback, parallel
   lifecycle authority, cache-side retry coordinator, or digest-semantics change.
3. Every qualifying artifact binds one exact 40-character source SHA plus its
   reviewed relevant-source digest. `UNAVAILABLE`, `NOT_QUALIFIED`, skipped,
   deselected, stale, scheduled-diagnostic, prerelease, Windows, mocked, and remote
   latency results never fill a qualifying slot.
4. Deterministic PostgreSQL authority gaps are test-first work in Wave 2 and must
   pass before Wave 3 captures coverage floors.
5. Current Phase 8 closure has no external checkpoint. Controlled Linux remains
   `DEFERRED`/`NOT_QUALIFIED` under SEED-006; live services remain
   `DEFERRED`/`NOT_QUALIFIED` and publication remains `DEFERRED`/`NOT_PUBLISHED`
   under SEED-007. These are closed local-milestone nonclaims, not inferred passes.

## Executable Local-Readiness Record

Plan 08-16 produces only
`08-LOCAL-READINESS.json`, with schema
`cacheness-phase8-local-readiness-v1`, through this fixed command:

```bash
uv run --isolated --all-extras --group dev --frozen python \
  tools/verify_phase8_contracts.py --local-ready \
  --output .planning/phases/08-production-gates-and-performance-stabilization/08-LOCAL-READINESS.json
```

The record is `LOCAL_READY` only when one clean revision and reviewed-source
digest bind four exact local evidence classes: deterministic integrity/recovery,
coverage plus scoped Ruff, structural call/RSS bounds, and a source-free base
wheel public round trip. It stores only those compact pass records plus observed
OS/Python/machine facts. It has no field that can carry a live-service,
workflow-run, tag, draft, asset, Linux-matrix, Windows, controlled-performance,
or publication result.

The record must include exactly these nonclaims: `QUAL-06` is
`DEFERRED`/`NOT_QUALIFIED` to SEED-006; `BACK-05` is
`DEFERRED`/`NOT_QUALIFIED` to SEED-007; publication is
`DEFERRED`/`NOT_PUBLISHED` to SEED-007. Missing, stale, dirty, malformed,
duplicate, wrong-class, wrong-source, skipped, or contradictory local evidence
fails closed. The retained live collector/aggregator and publication controller
remain the stricter SEED-007 path; mocked/local/preflight evidence is never an
input to it.

## Evidence Classes and Non-Substitution Rules

| Evidence class | Producer | Blocking scope | Required proof | Never substitutes for |
|---|---|---|---|---|
| Deterministic/local | `tools/run_phase8_local_gates.py`, fixed pytest selectors, `tools/verify_phase8_contracts.py` | PR and release | Exact selector inventory, no skip/deselection, canonical same-SHA envelope | Live services, controlled timing, other platforms |
| Packaging | `tools/run_phase8_packaging.py` | PR and release | Fresh isolated wheel install and representative public round trip for base and each literal optional group | Source-tree imports, all-extras transitivity, service qualification |
| Platform | `tools/run_phase8_platform_gates.py`, `quality.yml` | PR and release according to row role | Full stable Linux 3.11-3.14 rows, macOS 3.11/3.14 boundary smoke, TensorFlow only on compatible stable rows | Windows support, prerelease support, controlled performance |
| Coverage/quality | `tools/verify_phase8_coverage.py`, Coverage.py JSON/XML, direct Ruff commands | PR and release | Named behavior selectors plus post-gap repository/critical statement and branch floors; read-only baseline verification | Behavior omitted merely because percentages are green |
| Structural | `tools/run_phase8_scale_gates.py` | PR and release | Exact authority/participant call formulas and peak-memory bounds at fixed tiers | Wall-clock latency or stronger progress guarantees |
| Controlled performance | `benchmarks/phase8_benchmarks.py`, `performance.yml` | Deferred to SEED-006; not current-release blocking | Current release records exact `DEFERRED`/`NOT_QUALIFIED`; harness/workflow/preflight remain available for later stable distributions and baseline comparison | macOS diagnostics, runtime deadlines, remote latency, safety/recovery/progress |
| Live service | `tools/run_phase8_qualification.py`, `live_qualification.yml` | Deferred to SEED-007 | Current local report requires `DEFERRED` plus `NOT_QUALIFIED`; future proof still requires real PostgreSQL + authoritative Amazon S3, exact fixed suite/SHA/source, `QUALIFIED` and `CLEAN` | Mocks, collection-only, scheduled diagnostics, configuration preflight, local proof |
| Publication | `tools/verify_phase8_release.py` plus final immutable-release verifier | Deferred to SEED-007 | Current local report requires `DEFERRED` plus `NOT_PUBLISHED`; future proof still requires exact tag, published immutable state, exact asset set/states/digests | A local readiness manifest or unexecuted publication controller |

Remote-service latency, prerelease Python, and macOS performance observations are
diagnostic only. Windows and controlled-Linux performance remain explicitly
nonqualified. Those rows cannot enter the blocking aggregate.

## Test Infrastructure

| Property | Value |
|---|---|
| Framework | pytest 8.4.1, pytest-cov 6.2.1, Coverage.py 7.10.3, Ruff 0.12.9 in the researched environment |
| Config | `pyproject.toml`; Wave 3 enables branch coverage and strict baseline validation |
| Per-task command | Each plan task’s literal `<automated>` command |
| Full deterministic command | `uv run --isolated --all-extras --group dev --frozen python tools/verify_phase8_contracts.py --all` |
| Feedback target | Focused deterministic self-tests under 30 seconds; live execution is a separately bounded external job |
| Current status | Plans 08-01 through 08-10 and 08-13 through 08-15 are complete; 08-11/08-12 are superseded; Plan 08-16 is the only remaining local-readiness closure |

## Phase Requirements to Automated Evidence

| Requirement | Required truth | Automated evidence | Evidence class | Current planning status |
|---|---|---|---|---|
| BACK-05 | Real PostgreSQL and authoritative Amazon S3 behavior passes the frozen suite at one exact candidate SHA with exact bounded cleanup; compatible S3 services remain unclaimed | Existing runner/workflow/preflight/controller plus future SEED-007 exact execution | Live service + publication | `DEFERRED`/`NOT_QUALIFIED`; tooling exists, but no current live or publication evidence is claimed |
| QUAL-01 | Clean minimal wheel imports every guaranteed public symbol and performs generic-object plus NumPy public round trips | `tests/packaging/test_wheel_matrix.py -k base`; `tools/run_phase8_packaging.py` | Packaging | Wave 0 extension missing; existing `tests/test_full_suite_environment.py` is only an analog |
| QUAL-02 | Every literal optional group installs independently and exercises its public feature; TensorFlow incompatibility on 3.14 is explicit, never skipped | `tests/packaging/test_wheel_matrix.py`; exact packaging manifest emitted by `tools/run_phase8_packaging.py` | Packaging + platform | Wave 0 missing |
| QUAL-03 | CI definitions cover stable Python/platform roles, deterministic backends, packaging, branch coverage, scoped Ruff, structural gates, and protected live qualification without class substitution | Static workflow contracts plus the Plan 16 exact local-readiness command; external execution remains separate evidence | Current local classes + workflow contract | Local path closes; live/platform claims remain explicit nonclaims until eligible execution |
| QUAL-04 | Finite integrity/recovery/commit-boundary regressions remain green; shared-worker fixtures initialize before contention; terminal outcomes remain success, exact conflict, or typed retryable outcome | Inherited `tools/verify_phase071_contracts.py --all` plus `tests/test_phase8_lifecycle_coverage.py` | Deterministic/local | Inherited verifier exists; Phase 8 named-gap file missing |
| QUAL-05 | Named lifecycle/cache-policy gaps plus repository and critical statement/branch counts do not regress | `tests/test_phase8_lifecycle_coverage.py`; `tests/test_phase8_cache_policy_coverage.py`; `tests/test_phase8_coverage_gate.py`; read-only `tools/verify_phase8_coverage.py` | Deterministic + coverage | Wave 0 tests/verifier/baseline missing; capture forbidden until Wave 2 passes |
| QUAL-06 | Checked baseline contains stable distributions, tails, RSS, environment/source identity, layer-separated workloads, and hash comparisons for named tiers; thresholds remain performance-only | Existing harness/workflow/preflight self-tests preserve future execution capability; SEED-006 owns actual controlled capture | Controlled performance | `DEFERRED`/`NOT_QUALIFIED`; not a current-milestone requirement and macOS diagnostics do not qualify it |
| QUAL-07 | Inventory/aggregate operations meet exact call formulas and bounded peak-memory contracts without timing claims | `tests/performance/test_complexity_contracts.py`; `tests/performance/test_memory_bounds.py`; `tools/run_phase8_scale_gates.py` | Structural | Wave 0 consolidation missing; inherited behavior tests are analogs |

## Deterministic PostgreSQL Authority Gate — Must Precede Coverage Capture

Plan 08-04 Task 1 must own a Phase 8 selector for each row below. Existing Phase 5/07.1
tests are useful fixtures, but coverage baselining may not proceed until the Phase 8
inventory proves the families explicitly and `tools/verify_phase8_contracts.py`
names those selectors literally.

| Gap family | Test-first behavior | Existing analog | Required automated evidence |
|---|---|---|---|
| DB-API error classification | Map declared SQLSTATE/driver classes to typed bounded-progress outcomes; preserve unknown driver failures as backend errors; preserve cause; redact secret-bearing messages/context | `tests/contracts/test_postgresql_lifecycle_authority.py::test_inventory_preserves_retryable_postgresql_progress_causes`; authority `_raise_driver_error` | Add parametrized Phase 8 selector covering SQLSTATE and class routes, unknown error, cause, and redaction; run `tests/test_phase8_lifecycle_coverage.py -k postgresql_error_classification -x` |
| Exact replay | Identical operation/proof replay is idempotent and exact; changed operation descriptors, proof bytes, digest, size, or transport evidence conflict without new mutation | `test_postgresql_verification_replays_exact_transport_evidence`, `test_prepare_and_verification_use_parameterized_cas_statements`, `test_postgresql_read_mutation_returns_exact_prepared_and_promoted_replay` | Add Phase 8 replay matrix; run `tests/test_phase8_lifecycle_coverage.py -k postgresql_replay -x` |
| Bounded pagination | Inventory and reconciliation use explicit `limit`/work caps, canonical cursors, stable high-water bounds, no skipped unemitted rows, and typed stale/reinspection outcomes | `test_inventory_preserves_retryable_postgresql_progress_causes`, `test_reconciliation_cursor_does_not_skip_unemitted_debt_rows`, existing `inventory_page(limit=1, work_cap=1024)` contract | Add Phase 8 boundary/cursor matrix; run `tests/test_phase8_lifecycle_coverage.py -k postgresql_pagination -x` |
| Transactional rollback | Every initialization/read/transition failure exits the connection transaction as failure, rolls back partial writes, preserves domain cause, and does not commit/leave partial authority state | `test_initialize_rolls_back_and_redacts_driver_details`, `test_initialize_rejects_an_existing_partial_layout_before_ddl`, `test_pool_lease_receives_transition_failure_for_rollback` | Add failure injection across prepare/verify/promote/clear/reconcile paths; run `tests/test_phase8_lifecycle_coverage.py -k postgresql_transaction_rollback -x` |

**Wave dependency:** 08-04-01 must pass all four families before 08-05-02 may execute.
The coverage capture tool must preflight the literal selector inventory and refuse to
write a baseline when a selector is absent, skipped, deselected, or failing. The
pre-gap research percentages are diagnostic and never qualify this dependency.

## Exact Plan/Task Verification Map

| Task ID | Wave | Requirement focus | Automated command | Evidence class | Status |
|---|---:|---|---|---|---|
| 08-01-01 | 1 | QUAL-03/04 exact-commit tracer | `uv run pytest -q -o log_cli=false tests/test_phase8_release_tracer.py -k tracer -x` | Deterministic/local | Wave 0 missing |
| 08-01-02 | 1 | Truthful evidence parsing/nonclaims | `uv run pytest -q -o log_cli=false tests/test_phase8_release_tracer.py -x` | Deterministic/local | Wave 0 missing |
| 08-02-01 | 2 | QUAL-01 base wheel | `uv run pytest -q -o log_cli=false tests/packaging/test_wheel_matrix.py -k base -x` | Packaging | Wave 0 missing |
| 08-02-02 | 2 | QUAL-02 literal optional groups | `uv run pytest -q -o log_cli=false tests/packaging/test_wheel_matrix.py -x` | Packaging | Wave 0 missing |
| 08-03-01 | 2 | Python/TensorFlow compatibility roles | `uv run pytest -q -o log_cli=false tests/qualification/test_phase8_platform.py -k python -x` | Platform | Wave 0 missing |
| 08-03-02 | 2 | Linux/macOS/Windows nonclaim | `uv run pytest -q -o log_cli=false tests/qualification/test_phase8_platform.py -x` | Platform | Wave 0 missing |
| 08-04-01 | 2 | QUAL-04/05 lifecycle + PostgreSQL gaps | `uv run pytest -q -o log_cli=false tests/test_phase8_lifecycle_coverage.py -x` | Deterministic/local | Wave 0 missing; blocks 08-05-02 |
| 08-04-02 | 2 | QUAL-05 cache policy | `uv run pytest -q -o log_cli=false tests/test_phase8_cache_policy_coverage.py -x` | Deterministic/local | Wave 0 missing; blocks 08-05-02 |
| 08-05-01 | 3 | QUAL-05 ratchet/parser/Ruff scope | `uv run pytest -q -o log_cli=false tests/test_phase8_coverage_gate.py -x` | Coverage/quality | Wave 0 missing |
| 08-05-02 | 3 | QUAL-05 post-gap baseline | `uv run python tools/verify_phase8_coverage.py --report build/phase8/coverage.json --baseline tests/qualification/phase8_coverage_baseline.json` | Coverage/quality | Blocked until 08-04 passes |
| 08-06-01 | 2 | QUAL-07 call formulas | `uv run pytest -q -o log_cli=false tests/performance/test_complexity_contracts.py -x` | Structural | Wave 0 missing |
| 08-06-02 | 2 | QUAL-07 peak memory | `uv run pytest -q -o log_cli=false tests/performance/test_memory_bounds.py tests/performance/test_complexity_contracts.py -x` | Structural | Wave 0 missing |
| 08-07-01 | 2 | QUAL-06 workloads | `uv run pytest -q -o log_cli=false tests/performance/test_phase8_benchmarks.py -k workloads -x` | Performance self-test | Wave 0 missing |
| 08-07-02 | 2 | QUAL-06 distributions/hash/baseline | `uv run pytest -q -o log_cli=false tests/performance/test_phase8_benchmarks.py -k 'distribution or hash or baseline' -x` | Performance self-test | Wave 0 missing |
| 08-07-03 | 2 | Controlled workflow contract | `uv run pytest -q -o log_cli=false tests/performance/test_phase8_benchmarks.py -k workflow -x` | Performance self-test | Wave 0 missing |
| 08-08-01 | 2 | BACK-05 runner/source/obstore | `uv run pytest -q -o log_cli=false tests/qualification/test_phase8_evidence.py -k runner -x` | Live harness self-test | Wave 0 missing |
| 08-08-02 | 2 | BACK-05 cleanup/redaction | `uv run pytest -q -o log_cli=false tests/qualification/test_phase8_evidence.py -k 'cleanup or redact or secret' -x` | Live harness self-test | Wave 0 missing |
| 08-08-03 | 2 | Exact-SHA live workflow | `uv run pytest -q -o log_cli=false tests/qualification/test_phase8_live_workflow.py tests/qualification/test_phase8_evidence.py -x` | Live orchestration self-test | Wave 0 missing |
| 08-09-01 | 4 | QUAL-03 fixed quality CI | `uv run pytest -q -o log_cli=false tests/qualification/test_phase8_quality_workflow.py -x` | Deterministic/platform/workflow | Wave 0 missing |
| 08-09-02 | 4 | Qualification/nonclaim docs | `uv run pytest -q -o log_cli=false tests/qualification/test_phase8_quality_workflow.py -k documentation -x` | Documentation contract | Wave 0 missing |
| 08-10-01 | 5 | Fixed source/selector/decision verifier | `uv run pytest -q -o log_cli=false tests/test_phase8_contract_verifier.py -x` | Deterministic/local | Wave 0 missing |
| 08-10-02 | 5 | Exact-SHA workflow dispatch/run-ID wait/fixed artifact collection | `uv run pytest -q -o log_cli=false tests/qualification/test_phase8_release.py -k 'dispatch or run_id or artifact_collection' -x` | External-orchestration self-test | Wave 0 missing |
| 08-10-03 | 5 | Same-SHA aggregate and published immutable release verifier | `uv run pytest -q -o log_cli=false tests/qualification/test_phase8_release.py tests/test_phase8_contract_verifier.py -x` | Publication self-test | Wave 0 missing |
| 08-13-01 | 6 | Read-only exact-commit controlled-runner preflight | `uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false tests/performance/test_phase8_benchmarks.py -k preflight_runner -x` | Future controlled-runner capability | Complete; retained for SEED-006 and does not write a baseline |
| 08-14-01 | 7 | Required release preflight/collection and explicit performance deferral | `uv run --isolated --group dev --frozen pytest -q -o log_cli=false tests/qualification/test_phase8_release.py::test_preflight_collection_reports_exact_current_inventory tests/qualification/test_phase8_release.py::test_required_release_collection_excludes_deferred_performance tests/qualification/test_phase8_release.py::test_aggregate_records_exact_deferred_performance_nonclaim tests/qualification/test_phase8_release.py::test_deferred_performance_artifacts_and_macos_diagnostics_are_rejected -x` | Release-policy self-test | Must pass before external collection |
| 08-14-02 | 7 | Fixed inventory and qualification documentation cutover | `uv run --isolated --group dev --frozen pytest -q -o log_cli=false tests/test_phase8_contract_verifier.py::test_fixed_manifest_covers_deferred_performance_decision tests/test_phase8_contract_verifier.py::test_deferred_performance_status_is_nonblocking_but_not_qualified tests/test_phase8_contract_verifier.py::test_all_mode_still_blocks_on_unavailable_current_evidence tests/test_phase8_contract_verifier.py::test_release_documentation_preserves_seed006_nonclaim -x` | Contract/documentation self-test | Must bind Plan 14, D-23, and current requirement set |
| 08-14-03 | 7 | Complete publication controller before exact candidate selection | `uv run --isolated --group dev --frozen pytest -q -o log_cli=false tests/qualification/test_phase8_release.py::test_publication_preflight_fails_before_mutation tests/qualification/test_phase8_release.py::test_prepare_draft_uploads_exact_non_deferred_asset_set tests/qualification/test_phase8_release.py::test_prepare_draft_verifies_uploaded_states_digests_and_content tests/qualification/test_phase8_release.py::test_publish_requires_exact_approved_prepublication_digest tests/qualification/test_phase8_release.py::test_publish_and_verify_requires_immutable_exact_remote_state tests/qualification/test_phase8_release.py::test_postpublication_mismatch_records_unqualified_incident -x` | Publication-controller self-test | Must commit before Plan 08-11 so D-10 evidence is not invalidated by later tracked tooling changes |
| 08-15-01 | 8 | Configuration-only protected-live preflight and purity contract | `uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false tests/qualification/test_phase8_evidence.py::test_preflight_accepts_sanitized_configuration_without_external_effects tests/qualification/test_phase8_evidence.py::test_preflight_rejects_missing_invalid_and_disallowed_configuration_without_external_effects tests/qualification/test_phase8_evidence.py::test_preflight_requires_clean_source_and_reviewed_cleanup_contract tests/qualification/test_phase8_evidence.py::test_preflight_cli_never_runs_or_writes_qualification_evidence -x` | Live-harness self-test | Must pass without PostgreSQL/AWS access or evidence writes before the exact candidate is selected |
| 08-15-02 | 8 | Fixed plan/threat/selector binding | `uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false tests/test_phase8_contract_verifier.py::test_fixed_manifest_binds_live_configuration_preflight_gap tests/test_phase8_contract_verifier.py::test_fixed_manifest_rejects_source_mutation_that_drops_plan_or_threat tests/test_phase8_contract_verifier.py::test_selector_validation_rejects_removed_renamed_duplicate_and_unowned_nodes -x` | Contract self-test | Binds Plan 15 and T-08-15-01 through T-08-15-05 without changing decisions or evidence statuses |
| 08-11 | — | Exact-SHA live collection | Not run | Deferred live service | Superseded by D-24; preserved intact for SEED-007 and never counted as PASS |
| 08-12 | — | Immutable publication | Not run | Deferred publication | Superseded by D-24; preserved intact for SEED-007 and remains NOT_PUBLISHED |
| 08-16-01 | 9 | Fixed local-readiness status/manifest contract and D-24 bindings | `uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false tests/test_phase8_contract_verifier.py tests/qualification/test_phase8_release.py -x` | Local-readiness contract | Must preserve strict future release commands while adding the separate local boundary |
| 08-16-02 | 9 | Exact local suite, base wheel, coverage/Ruff, structural, integrity/recovery, and nonclaims | `uv run --isolated --all-extras --group dev --frozen python tools/verify_phase8_contracts.py --local-ready --output .planning/phases/08-production-gates-and-performance-stabilization/08-LOCAL-READINESS.json` | Local readiness | Exit 0 requires every local class and exact D-23/D-24 nonclaim record |

## Exact-SHA Dispatch, Wait, and Artifact Collection Gate

The protected release-candidate path must implement and test this sequence for
the required quality and live workflows. The retained performance workflow is not
dispatched by current release collection; the aggregate records its D-23 deferral.

1. Accept a required input containing exactly one 40-character candidate SHA.
2. Dispatch the workflow whose definition is itself read from a trusted reviewed ref,
   passing that SHA as data; record the returned/discovered workflow run ID.
3. In the job, checkout the explicit SHA, use detached source, and fail unless
   `git rev-parse HEAD` equals the input before any service call.
4. The controller selects that exact run ID, verifies `headSha`, event, workflow,
   status, and conclusion, and waits for that run with failure propagation.
5. Download the exact artifact name using the run ID. Never call artifact download
   without a run ID and never choose “latest.”
6. Parse the bounded envelope and verify schema, exact revision, recomputed
   relevant-source digest, RC (not scheduled) role, fixed test inventory/no skips,
   sanitized service identity, `QUALIFIED`, and `CLEAN`.
7. Persist the run ID, workflow identity, artifact digest, and source identity in the
   aggregate. Persist no credential or raw payload.

`tests/qualification/test_phase8_live_workflow.py` must statically reject missing
SHA input, implicit/default checkout, untrusted PR triggers, absent protected
environment, “latest” artifact selection, missing run-ID wait/download, and any
scheduled artifact eligible for release qualification. `tests/qualification/test_phase8_release.py`
must reject old-SHA, wrong-run, duplicate, source-drift, incomplete, unclean, and
diagnostic artifacts.

## Immutable Publication Verification Gate

The final transition is not complete when `release_qualification.json` is merely
written. It is complete only after the authorized operator and verifier prove:

- the intended tag exists and resolves to the exact candidate SHA;
- the release starts as a draft while assets remain mutable;
- the exact allow-listed qualifying assets are present, with no extra diagnostic,
  failed, unavailable, scheduled, raw-log, or secret-bearing artifact;
- every release asset reports state `uploaded`, its API `sha256:` digest equals the
  locally recomputed digest, and its content revalidates under the Phase 8 parsers;
- the draft is published only after those prepublication checks;
- the published release is non-draft and immutable, the tag still resolves to the
  exact SHA, and `gh release verify` plus `gh release verify-asset` succeeds for
  every local qualifying asset;
- a final exact state/assets/digests record is retained without mutating the release.

The deterministic verifier implementation must reuse the reviewed shape of
`tools/verify_phase5_contracts.py`: literal source/test inventories, fixed local
contracts, read-only external-status reporting, and no promotion of absent live
proof. Release-state tests must reuse `tests/test_phase3_release_evidence.py`’s
Git/tag/archive inspection style, replacing historical scheduler assertions with
fake CLI/API release fixtures for exact tag, state, asset set, and digest rejection.
The deterministic tests exercise fakes only; they do not claim repository settings,
authorization, or a published release exists.

## Original Wave 0 / Test-First Gaps

The unchecked rows below preserve the planning-time scaffold inventory; completed
plan summaries are authoritative for implementation status. D-23 changes only the
current release-blocking controlled-performance boundary.

- [ ] `tests/test_phase8_release_tracer.py` and `tools/phase8_evidence.py` — strict
      evidence schema, canonicalization, exact source, class separation, and
      nonclaim states.
- [ ] `tests/packaging/test_wheel_matrix.py` — base wheel and one fresh environment
      per literal optional group, with public round trips.
- [ ] `tests/qualification/test_phase8_platform.py` — stable/advisory/feature-aware
      matrices and Windows nonclaim.
- [ ] `tests/test_phase8_lifecycle_coverage.py` — lifecycle risks plus explicit
      PostgreSQL DB-API classification, replay, bounded pagination, and transaction
      rollback matrices. This gap must close before coverage baseline capture.
- [ ] `tests/test_phase8_cache_policy_coverage.py` — typed policy outcomes,
      invalidation/clear accounting, stale continuation, immutable/storage-free stats.
- [ ] `tests/test_phase8_coverage_gate.py`, `tools/verify_phase8_coverage.py`, and
      `tests/qualification/phase8_coverage_baseline.json` — strict JSON schema,
      literal selectors, capture/verify separation, repository/critical statement
      and branch floors, safe direct Ruff argv. Baseline is created only after the
      preceding two test files pass.
- [ ] `tests/performance/test_complexity_contracts.py`,
      `tests/performance/test_memory_bounds.py`, and `tools/run_phase8_scale_gates.py`
      — exact formulas and independent peak-memory bounds.
- [ ] `tests/performance/test_phase8_benchmarks.py`,
      `benchmarks/phase8_workloads.py`, `benchmarks/phase8_benchmarks.py`, and
      `.github/workflows/performance.yml` — fake-data harness contracts before any
      controlled capture.
- [ ] `tests/qualification/test_phase8_evidence.py`,
      `tests/qualification/test_phase8_live_workflow.py`, and
      `tools/run_phase8_qualification.py` — fixed suite, redaction, exact cleanup,
      exact-SHA dispatch/run-ID wait/download.
- [ ] `tests/qualification/test_phase8_quality_workflow.py` and
      `.github/workflows/quality.yml` — pinned, least-privilege matrix contract.
- [ ] `tests/test_phase8_contract_verifier.py`, `tools/verify_phase8_contracts.py`,
      `tests/qualification/test_phase8_release.py`, and
      `tools/verify_phase8_release.py` — fixed source/selector verifier, exact-SHA
      dispatch/run-ID collection, same-SHA aggregation, immutable publication
      state/assets/digests verifier.
- [ ] `docs/RELEASE_QUALIFICATION.md` — marker-bounded evidence/nonclaim/operator
      contract asserted by tests.

## Deferred External Prerequisites (Not Current Checkpoints)

| Checkpoint | Type | Owner | Preflight | Resume evidence | Failure disposition |
|---|---|---|---|---|---|
| `pyperf` legitimacy | `checkpoint:human-verify` | Maintainer | Review official project provenance and seam `SUS` reason before locking | Explicit approval or documented rejection/fallback | Do not install on silence |
| Controlled Linux runner | Deferred; no current checkpoint | Future SEED-006 owner | Existing read-only preflight remains the eligibility gate when the seed is promoted | No current resume evidence; QUAL-06 remains `DEFERRED`/`NOT_QUALIFIED` and macOS numbers remain diagnostic only |
| Protected live environment | SEED-007 prerequisite | Cloud/repository admin | Existing configuration-only preflight, then real identity/IAM/service proof | Sanitized exact-SHA `QUALIFIED`/`CLEAN` evidence | Keep BACK-05 deferred and unqualified |
| Exact live RC execution | SEED-007 prerequisite | Release operator | Existing exact-SHA dispatch/run-ID collector | Validated fixed artifacts from the exact run | Keep remote support unqualified |
| Immutable release enablement/authority | SEED-007 prerequisite | Repository admin/release operator | Existing policy/auth/tag/asset preflight | Exact verified draft | Keep publication deferred and not published |
| Final publish transition | SEED-007 prerequisite | Release operator/reviewer | Existing report-digest-bound approval | Published immutable release passes read-only verification | Leave publication not published |

Credentials, endpoints, account IDs, reviewer identities, and physical runner details
are deliberately absent here. The plan must request/verify them at the owning
checkpoint rather than inventing values.

## Sampling Rate

- **Per task commit:** run the task’s exact focused command; for Python edits also run
  direct Ruff lint/format on changed paths plus the fixed critical scope.
- **After Wave 2:** run all new deterministic, packaging, platform, structural,
  benchmark-harness, and live-harness self-tests. Specifically require all four
  PostgreSQL gap selectors before Wave 3.
- **After Wave 3:** generate branch-aware coverage evidence and run read-only
  verification against the newly reviewed post-gap baseline.
- **After Wave 4:** validate workflow/docs contracts; do not dispatch live or
  controlled jobs from an untrusted PR.
- **After Wave 5:** run fixed verifier quick/all modes plus adversarial dispatcher,
  collector, aggregator, and publication-verifier tests.
- **After Wave 6:** require the read-only controlled-runner preflight contract, canonical fingerprint/digest drift and adversarial tests, optional-lock-free Git/worktree non-mutation tests, Ruff lint plus format-check on every modified Python file, and the fixed-verifier binding.
- **After Wave 7:** require Plan 08-14's exact current requirement/decision/threat
  inventory, quality/live-only release collection, explicit SEED-006 deferral record,
  and rejection of macOS or controlled-performance artifact substitution.
- **After Wave 8:** require Plan 08-15's configuration-only live preflight, explicit no-network/no-mutation/no-write tests, sanitized bounded output, and literal plan/threat/selector binding.
- **After Wave 9 / phase gate:** require Plan 16's exact local-readiness command to
  pass deterministic integrity/recovery, base wheel/import, coverage/Ruff, and
  structural evidence for one source identity. Require exact D-23/D-24 deferral
  records and reject any live, publication, Linux, Windows, or performance promotion.

## Validation Sign-Off Criteria

- [ ] Every canonical Plan 01-10 and 13-16 code-producing task has its exact automated command and evidence class; superseded Plans 11/12 remain historical future-seed inputs only.
- [ ] Wave 0 creates every missing test/harness/workflow verifier before relying on it.
- [ ] PostgreSQL DB-API classification, replay, pagination, and rollback selectors
      pass before coverage floors are captured.
- [ ] Coverage capture is explicit; ordinary verification is non-mutating.
- [ ] Existing exact-SHA live dispatch and immutable-publication commands remain
      unchanged and tested for SEED-007; Phase 8 does not invoke them.
- [ ] `tools/verify_phase8_contracts.py` and release tests demonstrably reuse the
      Phase 5 fixed-verifier and Phase 3 release-evidence analogs.
- [ ] No absent external prerequisite, mock, collection-only run, diagnostic artifact,
      or manual assertion is counted as automated qualification.
- [ ] QUAL-06 remains visibly `DEFERRED`/`NOT_QUALIFIED` with SEED-006; BACK-05
      remains `DEFERRED`/`NOT_QUALIFIED` and publication remains
      `DEFERRED`/`NOT_PUBLISHED` with SEED-007. None is a PASS.
- [ ] ADR 0001 safety/recovery/progress/performance distinctions and the Phase 07.1
      obstore participant boundary remain intact.

**Approval state:** Nyquist-compliant D-24 local-readiness plan; Plan 08-16
execution evidence remains pending and has no current external checkpoint.
