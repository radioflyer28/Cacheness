---
phase: 11
slug: clean-supplemental-documentation-and-normalize-validation-ev
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-09-17
governing_decision: docs/adr/0001-topology-specific-storage-guarantees.md
---

# Phase 11 — Validation Strategy

> Per-phase validation contract for the TensorFlow cutover, documentation and
> evidence normalization, and final milestone-audit refresh. Production storage
> lifecycle code is outside this phase.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.4.1 |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false <touched selectors> -x` |
| **Full suite command** | `uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false -m 'not (live_postgresql or live_aws_s3 or live_remote)'` |
| **Estimated runtime** | Focused selectors under 60 seconds; final frozen non-live suite measured during execution |

---

## Sampling Rate

- **After every task commit:** Run the narrow selectors for touched package,
  documentation, qualification, or evidence contracts; run scoped Ruff for
  touched Python files.
- **After manifest/lock work:** Run `uv lock --check` after plain `uv lock`.
- **After every plan wave:** Run the affected package/documentation/
  qualification/validation contract cluster, not the complete suite.
- **Phase 3 stop gate:** Run the finite six-file command once after runtime and
  packaging removal stabilize. A genuine integrity or recovery defect stops
  Phase 11 and is classified under ADR 0001; it is not repaired here.
- **Before `$gsd-verify-work`:** Inspect a fresh source-free wheel and run the
  frozen non-live suite exactly once before refreshing the milestone audit.
- **Max focused feedback latency:** 60 seconds.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 11-W0-01 | 11-01 | 1 | D-05–D-08 | T-11-01 | Retired TensorFlow runtime/config/export/package surface is unreachable while retained handlers still round-trip | unit + package contract | `uv run --isolated --all-extras --group dev --frozen pytest -q --collect-only tests/packaging/test_wheel_matrix.py::test_tensorflow_surface_is_absent_from_built_wheel_and_metadata tests/packaging/test_wheel_matrix.py::test_retained_local_round_trips_survive_tensorflow_cutover tests/test_phase10_sqlcache_removal.py` | ✅ extend existing | ✅ green |
| 11-W0-02 | 11-01 | 1 | D-01–D-07 | T-11-02 | Platform/TensorFlow and pandas/custom-metadata dispositions have separate selectors plus one combined six-file contract | documentation contract | `uv run --isolated --all-extras --group dev --frozen pytest -q --collect-only tests/test_phase9_documentation.py::test_platform_and_tensorflow_supplements_are_consolidated_or_deleted tests/test_phase9_documentation.py::test_pandas_and_custom_metadata_supplements_are_consolidated_or_deleted tests/test_phase9_documentation.py::test_supplemental_documentation_is_consolidated_or_deleted tests/test_phase9_documentation.py::test_current_guidance_has_no_supported_tensorflow_claim tests/test_phase9_examples.py` | ✅ extend existing | ✅ green |
| 11-W0-03 | 11-02 | 1 | D-05, D-18 | T-11-04 | CI and qualification expose exactly the `core` feature profile, reject both retired profile names, and keep retained jobs parseable | workflow contract | `uv run --isolated --all-extras --group dev --frozen pytest -q --collect-only tests/qualification/test_phase8_quality_workflow.py tests/qualification/test_phase8_platform.py tests/test_phase9_quality_workflow.py` | ✅ update existing | ✅ green |
| 11-W0-04 | 11-02 | 1 | D-07, D-09–D-12 | T-11-05 | Phase 3, seed, Phase 1/5/6, Phase 7/8/9, and combined discovery each have a selector valid at their own task boundary | planning/evidence contract | `uv run --isolated --all-extras --group dev --frozen pytest -q --collect-only tests/test_phase9_evidence_metadata.py` | ✅ extend existing | ✅ green |
| 11-W0-05 | 11-02 | 1 | D-18–D-20 | T-11-06 | Phase 11 validation and milestone audit have non-rerunning evidence-derived parsers without a preselected verdict | evidence/audit contract | `uv run --isolated --all-extras --group dev --frozen pytest -q --collect-only tests/test_phase9_evidence_metadata.py tests/qualification/test_phase8_release.py` | ✅ extend existing | ✅ green |
| 11-REG-01 | 11-06 | 4 | D-13–D-17 | T-11-16 | Current lifecycle integrity/recovery and cache-over-store behavior remain green under declared typed outcomes | focused regression | `uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false tests/test_phase3_gap_acceptance.py tests/test_phase3_local_workflows.py tests/test_blob_store_atomic_lifecycle.py tests/test_blob_store_reconciliation.py tests/test_unified_cache_lifecycle_authority.py tests/test_integration.py -x` | ✅ | ✅ green |
| 11-EVID-03 | 11-06 | 4 | D-13–D-17 | T-11-17 | The compact Phase 3 record is canonical before any other normalized record is required | schema contract | `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase9_evidence_metadata.py::test_phase11_phase3_validation_is_canonical -x` | ❌ W0 selector | ✅ green |
| 11-EVID-SEED | 11-07 | 5 | D-07 | T-11-20 | SEED-005 is fulfilled and linked without requiring unfinished validation normalization | seed contract | `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase9_evidence_metadata.py::test_phase11_seed_resolution_is_canonical -x` | ❌ W0 selector | ✅ green |
| 11-EVID-156 | 11-08 | 6 | D-09–D-12 | T-11-22..24 | Phase 1/5/6 normalization is independently canonical after Task 1 | schema contract | `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase9_evidence_metadata.py::test_phase11_phase_1_5_6_validations_are_canonical -x` | ❌ W0 selector | ✅ green |
| 11-EVID-789 | 11-08 | 6 | D-09–D-12 | T-11-22..24 | Phase 7/8/9 normalization is canonical and the combined seven-file/seed inventory passes only after Task 2 | schema contract | `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase9_evidence_metadata.py::test_phase11_phase_7_8_9_validations_are_canonical tests/test_phase9_evidence_metadata.py::test_phase11_validation_discovery_and_seed_resolution_are_canonical -x` | ❌ W0 selectors | ✅ green |
| 11-PKG-01 | 11-09 | 7 | D-05, D-18 | T-11-25 | Fresh wheel has no TensorFlow extra/dependency/module/export and retained local BlobStore/UnifiedCache journeys pass | installed-artifact contract | `uv run --isolated --all-extras --group dev --frozen pytest -q tests/packaging/test_wheel_matrix.py -x` | ✅ extend existing | ✅ green |
| 11-FINAL-01 | 11-09 | 7 | D-18–D-20 | T-11-26..28 | All bounded local gates pass once before the audit derives its verdict | full non-live regression | `uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false -m 'not (live_postgresql or live_aws_s3 or live_remote)'` | ✅ | ✅ green |
| 11-FINAL-RECORD | 11-09 | 7 | D-18–D-20 | T-11-26 | The validation record captures the exact one-run command/result, canonical schema, approval, and nonclaims without rerunning the suite | evidence parser | `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase9_evidence_metadata.py::test_phase11_validation_record_matches_final_acceptance_evidence -x` | ❌ W0 selector | ✅ green |
| 11-AUDIT-01 | 11-10 | 8 | D-19–D-20 | T-11-29..31 | The audit is freshly derived, internally consistent, cites Phase 11, closes only targeted debt, and preserves all deferrals | audit parser | `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase9_evidence_metadata.py::test_phase11_refreshed_milestone_audit_is_evidence_derived -x` | ❌ W0 selector | ↪ explicitly superseded for 11-09; Plan 11-10 is the sole audit-derivation owner |

*Status: ✅ green · ↪ explicitly superseded for a later plan*

---

## Focused Acceptance Evidence

Recorded on 2026-09-19 while this record remains draft and before the
milestone audit refresh. These checks are local-only; they do not qualify live
PostgreSQL/Amazon-S3, controlled-Linux performance, native Windows, or
immutable publication.

- `uv lock --check` — exit 0.
- `uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false tests/packaging/test_wheel_matrix.py tests/test_handler_registration.py tests/test_phase9_documentation.py tests/test_phase9_examples.py tests/test_full_suite_environment.py::test_documented_full_suite_command_uses_locked_extras_and_dev_group tests/test_phase1_quality_gates.py::test_validation_artifact_records_terminal_approval_and_gap_wave_history tests/qualification/test_phase8_quality_workflow.py tests/qualification/test_phase8_platform.py tests/test_phase9_quality_workflow.py tests/test_phase9_evidence_metadata.py tests/qualification/test_phase8_release.py -x` — exit 0; 111 passed. This is the digest-bound source-free wheel, retained handler, documentation/example, workflow/platform, evidence/release, canonical-guide, and Phase 1 approval cluster.
- `uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false tests/test_phase3_gap_acceptance.py tests/test_phase3_local_workflows.py tests/test_blob_store_atomic_lifecycle.py tests/test_blob_store_reconciliation.py tests/test_unified_cache_lifecycle_authority.py tests/test_integration.py -x` — exit 0; 50 passed. This finite local Phase 3 evidence gate exercised no external service and authorizes no lifecycle repair.
- `uv run --isolated --all-extras --group dev --frozen ruff check src/cacheness/handlers.py src/cacheness/config.py src/cacheness/storage/handlers/__init__.py tools/run_phase8_packaging.py tools/phase8_evidence.py tools/run_phase8_platform_gates.py tools/run_phase8_local_gates.py tests/packaging/test_wheel_matrix.py tests/qualification/test_phase8_quality_workflow.py tests/qualification/test_phase8_platform.py tests/qualification/test_phase8_release.py tests/test_phase9_quality_workflow.py tests/test_phase9_evidence_metadata.py tests/test_phase9_documentation.py tests/test_full_suite_environment.py tests/test_phase1_quality_gates.py tests/test_phase10_sqlcache_removal.py` — exit 0; all checks passed.

## Final Frozen Non-Live Acceptance

The command below ran once on 2026-09-19 while this record was draft and the
milestone audit retained original provenance and unresolved target debt. It
exited 0 and reached 100%. The observed summary contained exactly three
expected skips (Windows junction fixture, unavailable device-node creation,
and native Windows evidence target) plus one
`PytestCollectionWarning` for `TestDataClassForConsistency` having an
`__init__` constructor; it did not report a test failure.

`uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false -m 'not (live_postgresql or live_aws_s3 or live_remote)'`

## Explicit Nonclaims

- Live PostgreSQL and Amazon S3 qualification remains `UNAVAILABLE` and
  `NOT_QUALIFIED`; no local, mock, or frozen-suite evidence substitutes for
  real-service evidence required by `BACK-05` and `SEED-007`.
- Controlled-Linux performance remains `DEFERRED` and `NOT_QUALIFIED` pending
  `QUAL-06` and `SEED-006`; this macOS-local run is not a performance claim.
- Native Windows remains `UNAVAILABLE` and `NOT_QUALIFIED` pending Phase
  999.1; portable wheel tags and skipped Windows-only tests are not host
  qualification.
- Immutable publication remains `NOT_PUBLISHED`; this local wheel inspection
  is not an immutable release publication.
- The milestone audit remains at its original provenance and outstanding debt
  in this plan. Plan 11-10 alone may derive its final verdict.

---

## Wave 0 Requirements

- [x] Extend `tests/packaging/test_wheel_matrix.py` with negative wheel member,
  metadata, import/export, optional-extra, and retained-round-trip assertions.
- [x] Extend `tests/test_phase9_documentation.py` with the exact supplemental
  deletion set, canonical destinations, bounded current-reference scan, and no
  supported TensorFlow claims.
- [x] Update workflow/profile contracts before deleting TensorFlow-specific CI
  jobs and qualification profiles.
- [x] Add separate planning/evidence selectors for Phase 3, fulfilled SEED-005,
  Phases 1/5/6, and Phases 7/8/9, then a combined seven-file/seed selector that
  runs only after all prerequisites exist.
- [x] Add a non-rerunning Phase 11 validation-record parser for the exact
  full-suite command/result, canonical schema, green/superseded rows, approval,
  and explicit nonclaims.
- [x] Add an audit parser proving fresh provenance, Phase 11 citation, resolved
  targeted debt, recomputed Nyquist inventory, internally consistent scores,
  and preserved `BACK-05`, `QUAL-06`, native-Windows, and publication deferrals.

---

## Manual-Only Verifications

All implementation behaviors have automated verification. The final milestone
verdict is derived after the automated layered gate; it must not be selected in
advance or treated as a substitute for the gate evidence.

---

## Validation Sign-Off

- [x] Every planned task has an automated verify command or an explicit Wave 0 dependency.
- [x] Sampling continuity: no three consecutive tasks lack automated verification.
- [x] Wave 0 covers every missing negative, discovery, seed, and audit assertion.
- [x] No watch-mode flags or timing-only race assertions are used.
- [x] TensorFlow is absent from runtime/config/exports, package metadata, lockfile, CI, current docs, and current codebase maps; qualification accepts exactly `core` and rejects both `tensorflow` and `non_tensorflow` profile inputs/rows.
- [x] Retained handlers, `BlobStore`, and `UnifiedCache` package journeys remain green.
- [x] Phase 3's finite gate passes once, or a genuine integrity/recovery failure stops Phase 11 without lifecycle edits.
- [x] Normalized validation artifacts preserve supersession provenance and all explicit external/platform/performance nonclaims.
- [x] Fresh wheel, scoped Ruff, lock freshness, exact documentation scans, and the frozen non-live suite pass before the audit refresh.
- [x] `nyquist_compliant: true` is set only after every required row is green or explicitly superseded with evidence.

**Approval:** approved 2026-09-19 after the measured frozen non-live suite.
