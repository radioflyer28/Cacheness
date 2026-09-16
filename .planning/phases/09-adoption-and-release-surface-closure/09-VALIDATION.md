---
phase: 09
slug: adoption-and-release-surface-closure
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-09-16
---

# Phase 09 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.4.1 |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest -q -o log_cli=false tests/test_phase9_examples.py tests/test_interfaces.py tests/test_handler_registration.py tests/test_guarded_handler_io.py tests/test_public_api_contract.py tests/test_security_documentation.py -x` |
| **Full suite command** | `uv run pytest -q -o log_cli=false -m "not (live_postgresql or live_aws_s3 or live_remote)" -x` |
| **Estimated runtime** | Quick suite under 60 seconds; full non-live runtime measured during execution |

---

## Sampling Rate

- **After every task commit:** Run the focused command named in that task's `<verify>` block.
- **After every plan wave:** Run the quick suite above plus the affected wheel, documentation, or example contract tests.
- **Before `$gsd-verify-work`:** The full non-live suite, targeted Ruff scope, source-free wheel matrix, and exact four-example harness must be green.
- **Max feedback latency:** 60 seconds for task-level focused tests; longer packaging and full-suite gates run at wave/phase boundaries.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 09-01-01 | 01 | 1 | CACH-06 | T-09-01 | Public cutover exposes only `FormatHandler` while stored identities remain unchanged | unit + integration | `uv run pytest -q tests/test_interfaces.py tests/test_handler_registration.py tests/test_guarded_handler_io.py tests/test_stored_compatibility.py -x` | ✅ existing; edits required | ⬜ pending |
| 09-01-02 | 01 | 1 | CACH-06 | T-09-02 | Minimal source-free import is quiet and optional dataframe behavior remains request-bound | packaging | `uv run pytest -q tests/packaging/test_wheel_matrix.py -x` | ✅ existing; edits required | ⬜ pending |
| 09-02-01 | 02 | 2 | CACH-06 | T-09-03 | Published examples are network-free, path-contained, disposable, and self-verifying | subprocess integration | `uv run pytest -q tests/test_phase9_examples.py -x` | ❌ Wave 0 | ⬜ pending |
| 09-02-02 | 02 | 2 | CACH-06 | — | CI invokes the exact canonical example-file harness | workflow contract | `uv run pytest -q tests/test_phase9_quality_workflow.py -x` | ❌ Wave 0 | ⬜ pending |
| 09-03-01 | 03 | 3 | CACH-06 | T-09-04 | Supported docs retain trusted-payload and fail-closed integrity boundaries without overstating topology qualification | documentation contract | `uv run pytest -q tests/test_phase9_documentation.py tests/test_public_api_contract.py tests/test_security_documentation.py -x` | ❌ Wave 0 plus existing tests | ⬜ pending |
| 09-04-01 | 04 | 4 | CACH-06 | T-09-05 | Evidence refresh preserves recorded nonclaims and does not fabricate qualification | artifact contract | `uv run pytest -q tests/test_phase9_evidence_metadata.py -x` | ❌ Wave 0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_phase9_examples.py` — replace the Phase 6 harness with a literal four-file execution and residue checks.
- [ ] `tests/test_phase9_documentation.py` — enforce current imports, task navigation, one guarantees owner, checkout-first installation, and bounded claims.
- [ ] `tests/test_phase9_quality_workflow.py` — prove CI invokes the exact Phase 9 example harness.
- [ ] `tests/test_phase9_evidence_metadata.py` — constrain Phase 3/8 metadata refresh to existing evidence and preserved nonclaims.
- [ ] Extend `tests/packaging/test_wheel_matrix.py` — assert new/old public names and silent minimal import.
- [ ] Extend `tests/test_stored_compatibility.py` — assert identity-preserving durable reopen across the source rename.

---

## Manual-Only Verifications

All Phase 9 acceptance behaviors have automated verification. Human review may assess prose quality, but no requirement depends only on that review.

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Task-level feedback latency remains under 60 seconds
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
