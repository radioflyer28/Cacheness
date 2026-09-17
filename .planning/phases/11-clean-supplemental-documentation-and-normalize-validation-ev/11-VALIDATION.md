---
phase: 11
slug: clean-supplemental-documentation-and-normalize-validation-ev
status: draft
nyquist_compliant: false
wave_0_complete: false
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
| 11-W0-01 | TBD | 0 | D-05–D-08 | T-11-01 | Retired TensorFlow runtime/config/export/package surface is unreachable while retained handlers still round-trip | unit + package contract | `uv run --isolated --all-extras --group dev --frozen pytest -q tests/packaging/test_wheel_matrix.py tests/test_handler_registration.py -x` | ✅ extend existing | ⬜ pending |
| 11-W0-02 | TBD | 0 | D-01–D-07 | T-11-02 | Current docs contain no stale TensorFlow or removed-constructor guidance and canonical examples remain exact | documentation contract | `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase9_documentation.py tests/test_phase9_examples.py -x` | ✅ extend existing | ⬜ pending |
| 11-W0-03 | TBD | 0 | D-05, D-18 | T-11-03 | CI and qualification profiles no longer advertise TensorFlow while retained jobs remain parseable | workflow contract | `uv run --isolated --all-extras --group dev --frozen pytest -q tests/qualification/test_phase8_quality_workflow.py tests/qualification/test_phase8_platform.py tests/test_phase9_quality_workflow.py -x` | ✅ update existing | ⬜ pending |
| 11-W0-04 | TBD | 0 | D-07, D-09–D-12 | T-11-04 | SEED-005 is fulfilled and target validation files discover as canonical without erasing supersession or nonclaims | planning/evidence contract | `uv run --isolated --all-extras --group dev --frozen pytest -q <phase-11 evidence selector> -x` | ❌ W0 | ⬜ pending |
| 11-W0-05 | TBD | 0 | D-19–D-20 | T-11-05 | Audit can close targeted debt without promoting BACK-05, QUAL-06, Windows, or publication | audit contract | `uv run --isolated --all-extras --group dev --frozen pytest -q <phase-11 audit selector> -x` | ❌ W0 | ⬜ pending |
| 11-REG-01 | TBD | 2 | D-13–D-17 | T-11-06 | Current lifecycle integrity/recovery and cache-over-store behavior remain green under declared typed outcomes | focused regression | `uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false tests/test_phase3_gap_acceptance.py tests/test_phase3_local_workflows.py tests/test_blob_store_atomic_lifecycle.py tests/test_blob_store_reconciliation.py tests/test_unified_cache_lifecycle_authority.py tests/test_integration.py -x` | ✅ | ⬜ pending |
| 11-PKG-01 | TBD | 3 | D-05, D-18 | T-11-03 | Fresh wheel has no TensorFlow extra/dependency/module/export and retained local BlobStore/UnifiedCache journeys pass | installed-artifact contract | `uv run --isolated --all-extras --group dev --frozen pytest -q tests/packaging/test_wheel_matrix.py -x` | ✅ extend existing | ⬜ pending |
| 11-FINAL-01 | TBD | 3 | D-18–D-20 | T-11-01..06 | All bounded local gates pass once before the audit derives its verdict | full non-live regression | `uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false -m 'not (live_postgresql or live_aws_s3 or live_remote)'` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Extend `tests/packaging/test_wheel_matrix.py` with negative wheel member,
  metadata, import/export, optional-extra, and retained-round-trip assertions.
- [ ] Extend `tests/test_phase9_documentation.py` with the exact supplemental
  deletion set, canonical destinations, bounded current-reference scan, and no
  supported TensorFlow claims.
- [ ] Update workflow/profile contracts before deleting TensorFlow-specific CI
  jobs and qualification profiles.
- [ ] Add a Phase 11 planning/evidence contract for fulfilled SEED-005 and
  canonical discovery of Phases 1, 3, 5, 6, 7, 8, and 9 validation records.
- [ ] Add audit-boundary assertions proving the targeted debt is resolved while
  `BACK-05`, `QUAL-06`, native Windows, and immutable publication remain
  deferred or nonqualified.

---

## Manual-Only Verifications

All implementation behaviors have automated verification. The final milestone
verdict is derived after the automated layered gate; it must not be selected in
advance or treated as a substitute for the gate evidence.

---

## Validation Sign-Off

- [ ] Every planned task has an automated verify command or an explicit Wave 0 dependency.
- [ ] Sampling continuity: no three consecutive tasks lack automated verification.
- [ ] Wave 0 covers every missing negative, discovery, seed, and audit assertion.
- [ ] No watch-mode flags or timing-only race assertions are used.
- [ ] TensorFlow is absent from runtime/config/exports, package metadata, lockfile, CI, qualification profiles, current docs, and current codebase maps.
- [ ] Retained handlers, `BlobStore`, and `UnifiedCache` package journeys remain green.
- [ ] Phase 3's finite gate passes once, or a genuine integrity/recovery failure stops Phase 11 without lifecycle edits.
- [ ] Normalized validation artifacts preserve supersession provenance and all explicit external/platform/performance nonclaims.
- [ ] Fresh wheel, scoped Ruff, lock freshness, exact documentation scans, and the frozen non-live suite pass before the audit refresh.
- [ ] `nyquist_compliant: true` is set only after every required row is green or explicitly superseded with evidence.

**Approval:** pending
