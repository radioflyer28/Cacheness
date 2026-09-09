---
phase: 7
slug: explicit-migration-and-rebuild-cutover
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-09-09
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.4.1 |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest -q tests/test_migration_inspection.py tests/test_migration_plan_contract.py tests/test_migration_run_evidence.py -o log_cli=false` |
| **Full suite command** | `uv run pytest -q -o log_cli=false` |
| **Estimated runtime** | Focused commands should remain below 60 seconds; full-suite timing is measured, not a runtime correctness deadline |

---

## Sampling Rate

- **After every task commit:** Run the affected new test module plus its nearest existing lifecycle or handler contract module.
- **After every plan wave:** Run `uv run pytest -q tests/test_migration_*.py tests/test_rebuild_workflow.py tests/test_stored_compatibility.py tests/test_handler_registration.py -o log_cli=false`.
- **Before `$gsd-verify-work`:** The deterministic full suite must be green.
- **Max feedback latency:** 60 seconds for focused task sampling; split tests when a focused command exceeds it rather than converting the sampling budget into public timeout behavior.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 07-TBD-01 | TBD | 0+ | MIGR-03 | T-07-PLAN-TAMPER / T-07-EXHAUSTION | Inspection is bounded and non-mutating; plans have canonical authenticated structure and stable reasons | unit + contract | `uv run pytest -q tests/test_migration_inspection.py tests/test_migration_plan_contract.py -o log_cli=false` | ❌ W0 | ⬜ pending |
| 07-TBD-02 | TBD | 0+ | MIGR-04 | T-07-STALE-SOURCE / T-07-PARTIAL-ACTIVATION | Stale sources and incomplete candidates never activate; ordinary open never migrates | fault injection + contract | `uv run pytest -q tests/test_migration_cutover.py tests/test_stored_compatibility.py -o log_cli=false` | ❌ W0 / ✅ existing compatibility | ⬜ pending |
| 07-TBD-03 | TBD | 0+ | MIGR-05 | T-07-EVIDENCE-TAMPER / T-07-KEY-DISCLOSURE | Resume revalidates explicit evidence; prior store and signing identity remain intact | fault injection + integration | `uv run pytest -q tests/test_migration_run_evidence.py tests/test_migration_cutover.py tests/test_projection_sql_atomicity.py -o log_cli=false` | ❌ W0 / ✅ existing projection | ⬜ pending |
| 07-TBD-04 | TBD | 0+ | MIGR-06 | T-07-EXCLUSION-AMBIGUITY / T-07-UNSAFE-DESERIALIZATION | Rebuild is exact and confirmed; custom formats authenticate before handler invocation | integration + contract | `uv run pytest -q tests/test_rebuild_workflow.py tests/test_handler_registration.py -o log_cli=false` | ❌ W0 / ✅ existing handler | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_migration_inspection.py` — bounded raw inventory, canonical identity, historical rebuild-only classification, and byte-for-byte non-mutation.
- [ ] `tests/test_migration_plan_contract.py` — canonical JSON/human rendering, stable reasons, compatibility matrix, and exact totals.
- [ ] `tests/test_migration_run_evidence.py` — evidence authentication, redaction, run/source/destination mismatch, corruption, and explicit resume.
- [ ] `tests/test_migration_cutover.py` — stage, verify, activate, rollback, finalize, purge, and interruption injection.
- [ ] `tests/test_rebuild_workflow.py` — cross-backend and custom-handler rebuild plus exact exclusion confirmation.
- [ ] Extend `tests/test_lifecycle_authority_contract.py` and `tests/contracts/test_postgresql_lifecycle_authority.py` for the narrow maintenance capability.
- [ ] Extend `tests/test_stored_compatibility.py` so an initialized current store is inspectable without a manually fabricated marker.

---

## Manual-Only Verifications

All Phase 7 deterministic behaviors have automated verification. Live PostgreSQL/S3 qualification and the supported Python-version/platform matrix remain Phase 8 gates and must not be reported as Phase 7 passes.

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verification or Wave 0 dependencies.
- [ ] Sampling continuity: no 3 consecutive tasks without automated verification.
- [ ] Wave 0 covers all missing references.
- [ ] No watch-mode flags.
- [ ] Focused feedback latency remains below 60 seconds.
- [ ] Integrity, recovery, progress, and performance assertions remain separate.
- [ ] `nyquist_compliant: true` set in frontmatter.

**Approval:** pending
