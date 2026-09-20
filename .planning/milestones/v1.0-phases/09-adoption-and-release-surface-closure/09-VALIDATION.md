---
phase: 09
slug: adoption-and-release-surface-closure
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-09-16
updated: 2026-09-16
---

# Phase 09 — Validation Strategy

> Per-task feedback contract and completed evidence record. Phase 09 had no
> separate Wave 0 plan: each new test was created red and executed in its
> owning TDD task. The original task provenance remains in the table below;
> the current validated state is corroborated by `09-VERIFICATION.md` (44/44
> must-haves) and the current package/documentation contracts.

## Test Infrastructure

| Property | Value |
|---|---|
| Framework | pytest 8.4.1 |
| Config | `pyproject.toml` |
| Quick run | `uv run pytest -q -o log_cli=false tests/test_phase9_examples.py tests/test_phase9_documentation.py tests/test_phase9_quality_workflow.py tests/test_phase9_evidence_metadata.py tests/test_interfaces.py tests/test_handler_registration.py tests/test_public_api_contract.py tests/test_security_documentation.py -x` |
| Full suite | `uv run pytest -q -o log_cli=false -m "not (live_postgresql or live_aws_s3 or live_remote)" -x` |
| Feedback target | Focused task commands under 60 seconds; packaging/full-suite gates at plan or phase boundaries |

## Sampling Rate

- After every task commit: run the exact `<automated>` command in that task.
- After each wave: run the quick suite plus affected wheel/document/workflow contracts.
- Before verification: run the complete targeted suite, targeted Ruff, source-free wheel matrix, and full non-live suite.
- No watch-mode command and no manual-only acceptance criterion is allowed.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat refs | Secure behavior | Test type | Automated command | Test ownership | Status |
|---|---:|---:|---|---|---|---|---|---|---|
| 09-01-01 | 01 | 1 | CACH-06 | T-09-04 | Independent cross-cutting FormatHandlerError rename preserves translation/context without alias | unit | `uv run pytest -q -o log_cli=false tests/test_error_handling.py -x` | existing test edited in task | ✅ green |
| 09-01-02 | 01 | 1 | CACH-06 | T-09-01–03 | Atomic core protocol/barrel/interface-error cutover; no committed import break or alias | unit + integration | `uv run pytest -q -o log_cli=false tests/test_interfaces.py tests/test_handler_registration.py -x` plus dual-barrel import probe | existing tests edited in task | ✅ green |
| 09-02-01 | 02 | 2 | CACH-06 | T-09-05–07 | Source-free quiet import, alias absence, exact durable identity/reopen | packaging + integration | `uv run pytest -q -o log_cli=false tests/packaging/test_wheel_matrix.py tests/test_stored_compatibility.py tests/test_public_api_contract.py tests/test_guarded_handler_io.py -x` | existing tests edited in task | ✅ green |
| 09-02-02 | 02 | 2 | CACH-06 | T-09-08 | Current guidance retains private handler I/O and one authority | static contract | `uv run python -c "...current guidance assertions..."` | inline task assertion | ✅ green |
| 09-03-01 | 03 | 2 | CACH-06 | T-09-09–12 | Four exact network-free disposable journeys and safe MCAP round trip | subprocess integration | `uv run pytest -q -o log_cli=false tests/test_phase9_examples.py -x` | `tests/test_phase9_examples.py` created in task | ✅ green |
| 09-04-01 | 04 | 3 | CACH-06 | T-09-09A/B | First deletion batch preserves canonical behavior and Phase 10 boundary | integration + file contract | Phase 9 example pytest plus five exact absence checks | existing harness from 09-03 | ✅ green |
| 09-05-01 | 05 | 3 | CACH-06 | T-09-10A/B | Second deletion batch removes S3 promotion without remote/SqlCache changes | integration + file contract | Phase 9 example pytest plus five exact absence checks | existing harness from 09-03 | ✅ green |
| 09-06-01 | 06 | 4 | CACH-06 | T-09-13/14 | Blocking non-live CI invokes exact harness; examples/README.md links exactly four canonical files without obsolete paths or SqlCache promotion | workflow + index contract | `uv run pytest -q -o log_cli=false tests/test_phase9_quality_workflow.py tests/test_phase9_examples.py tests/qualification/test_phase8_quality_workflow.py -x` | `tests/test_phase9_quality_workflow.py` created and `examples/README.md` edited in task | ✅ green |
| 09-06-02 | 06 | 4 | CACH-06 | T-09-15/16 | Superseded harness/scripts absent while the canonical index/example contract remains green and Phase 10 files remain untouched | integration + file contract | Phase 9 workflow/example pytest plus four exact absence checks | existing Phase 9 tests | ✅ green |
| 09-07-01 | 07 | 4 | CACH-06 | T-09-17/18/20 | README current APIs, exact base-only checkout command, task-local extras, task navigation, bounded claims | documentation contract | `uv run pytest -q -o log_cli=false tests/test_phase9_documentation.py tests/test_phase9_examples.py tests/test_public_api_contract.py -x` | `tests/test_phase9_documentation.py` created in task and asserts `uv sync --frozen --no-default-groups` | ✅ green |
| 09-07-02 | 07 | 4 | CACH-06 | T-09-19 | Storage/cache/operator guides preserve initialization and offline maintenance | documentation + executable examples | same focused documentation/example/public command | owning documentation test extended | ✅ green |
| 09-08-01 | 08 | 5 | CACH-06 | T-09-21–24 | One truthful matrix, trusted-payload boundary, bounds, and exact nonclaims | documentation + security | `uv run pytest -q -o log_cli=false tests/test_phase9_documentation.py tests/test_security_documentation.py tests/qualification/test_phase8_quality_workflow.py -x` | existing/new documentation contracts edited in task | ✅ green |
| 09-09-01 | 09 | 6 | CACH-06 | T-09-25/26 | API imports match barrels; tutorial retains contained path I/O and stable identities | public API + integration | `uv run pytest -q -o log_cli=false tests/test_public_api_contract.py tests/test_phase9_documentation.py tests/test_handler_registration.py tests/test_guarded_handler_io.py -x` | existing tests edited in task | ✅ green |
| 09-09-02 | 09 | 6 | CACH-06 | T-09-27 | Obsolete docs absent and navigation has no dead/legacy branch | documentation + file contract | documentation/public/security pytest plus three exact absence checks | existing Phase 9 documentation test | ✅ green |
| 09-10-01 | 10 | 7 | CACH-06 | T-09-28–30 | Evidence refresh cites existing scope and preserves all nonclaims | artifact contract | `uv run pytest -q -o log_cli=false tests/test_phase9_evidence_metadata.py -x` | `tests/test_phase9_evidence_metadata.py` created in task | ✅ green |
| 09-10-02 | 10 | 7 | CACH-06 | T-09-31 | Exactly one dormant Narwhals seed and no dependency addition | artifact + dependency contract | evidence pytest plus exact seed/dependency probe | owning evidence test extended | ✅ green |

## Owning-Task Test Creation

The following missing tests are deliberately created and run in their owning
TDD tasks, so there is no detached Wave 0 plan:

- `09-03-01` creates `tests/test_phase9_examples.py` before the four examples.
- `09-06-01` creates `tests/test_phase9_quality_workflow.py` before workflow edits.
- `09-07-01` creates `tests/test_phase9_documentation.py` before README/docs edits.
- `09-10-01` creates `tests/test_phase9_evidence_metadata.py` before metadata refresh.
- `09-02-01` extends the existing wheel and stored-compatibility tests before package/evidence edits.

## Manual-Only Verifications

None. Human prose review is useful but no acceptance behavior depends only on it.

## Validation Sign-Off

- [x] Every planned task has an owning automated command.
- [x] Every new test is created and executed in its owning TDD task.
- [x] No three consecutive tasks lack automated verification.
- [x] No watch-mode flags or external-service requirements exist.
- [x] All task commands green after execution, as independently confirmed by `09-VERIFICATION.md`.
- [x] Targeted Ruff, source-free wheel matrix, and full non-live suite green.
- [x] `nyquist_compliant: true` set after execution evidence exists.

No external API integration is introduced. PostgreSQL and Amazon S3 remain
reference-only and `NOT_QUALIFIED`; controlled-Linux performance and Windows
remain `NOT_QUALIFIED`; immutable publication remains `NOT_PUBLISHED`.
