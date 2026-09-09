---
phase: 06
slug: unifiedcache-policy-composition
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-09-08
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.4.1 |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run --frozen pytest <changed-test-file> -q -o log_cli=false` |
| **Full suite command** | `uv run --frozen pytest -q -o log_cli=false` |
| **Estimated runtime** | Quick checks under 15 seconds; full suite measured during execution |

## Sampling Rate

- **After every task commit:** Run the mapped Phase 6 test file plus the nearest
  existing regression file.
- **After every plan wave:** Run all Phase 6 tests and the Phase 3–5 lifecycle
  contract suites affected by that wave.
- **Before `$gsd-verify-work`:** The full suite and scoped Ruff checks must be
  green, with any unrelated baseline exception identified explicitly.
- **Max feedback latency:** 60 seconds for per-task sampling; split a command if
  the measured feedback loop exceeds this bound.

## Per-Requirement Verification Map

Task IDs and threat references are assigned by the Phase 6 plans. Every planned
task must bind to one or more rows below and add its exact command to the final
per-task map before execution completes.

| Requirement | Secure behavior | Test type | Automated command | File exists | Status |
|-------------|-----------------|-----------|-------------------|-------------|--------|
| CACH-01 | Cache operations reach canonical state only through BlobStore/shared engine | contract/adversarial | `uv run --frozen pytest tests/contracts/test_phase6_topology_policy.py -q -o log_cli=false` | ❌ W0 | ⬜ pending |
| CACH-02 | Policy stays in UnifiedCache and derived state cannot authorize lifecycle effects | unit/architecture | `uv run --frozen pytest tests/test_phase6_policy_contract.py -q -o log_cli=false` | ❌ W0 | ⬜ pending |
| CACH-03 | Every removal path uses exact-generation lifecycle deletion and one report | adversarial | `uv run --frozen pytest tests/test_phase6_removal_contract.py -q -o log_cli=false` | ❌ W0 | ⬜ pending |
| CACH-04 | Cached `None` is a hit and one lookup performs one BlobStore read | unit | `uv run --frozen pytest tests/test_phase6_lookup_contract.py -q -o log_cli=false` | ❌ W0 | ⬜ pending |
| CACH-05 | Immutable statistics distinguish all locked outcomes without catalog scans | unit | `uv run --frozen pytest tests/test_phase6_statistics.py -q -o log_cli=false` | ❌ W0 | ⬜ pending |
| CACH-06 | Canonical positive exports and removed negative surface are both enforced | public contract | `uv run --frozen pytest tests/test_phase6_public_api_contract.py tests/test_phase6_decorator_contract.py -q -o log_cli=false` | ❌ W0 | ⬜ pending |
| CACH-07 regression | SqlCache remains a separate import and behavioral subsystem | regression | `uv run --frozen pytest tests/test_sql_cache.py -q -o log_cli=false` | ✅ | ⬜ pending |

### Fixed Retained Regression Evidence

| Source criterion | Exact test node | Acceptance evidence | File exists | Status |
|------------------|-----------------|---------------------|-------------|--------|
| ROADMAP Phase 6 SC-06 — strict malformed direct projection observations still reject malformed evidence | `tests/test_catalog_projection.py::test_json_projection_rejects_incompatible_derived_documents` | `uv run --frozen pytest tests/test_catalog_projection.py::test_json_projection_rejects_incompatible_derived_documents -q -o log_cli=false` exits zero with the named test passing; the test writes `{"format_version": 999}` and requires `load_projection_checkpoint()` to raise `JsonProjectionError` matching `incompatible shape`, proving the document is not admitted as an observation. Record the actual outcome here during Plan 08 and leave this row open on failure or skip. | ✅ | ⬜ pending |

## Required Adversarial Cases

- Two identical decorated calls returning `None` execute the function once and
  record one absent outcome followed by one hit.
- A BlobStore spy observes exactly one `open_entry()` per lookup, including
  expired and corrupt cases.
- Selection observes generation A, generation B wins before deletion, the
  exact delete conflicts, B remains readable, and the removal report records
  the conflict/retryable result.
- Malformed predicates fail before query or delete call counts change.
- More entries than one page stop at the configured total-work cap; the opaque
  continuation resumes without duplicate deletion or unbounded materialization.
- Clearing one decorated function removes only its function namespace and
  returns the actual removal report.
- Statistics/projection failure cannot authorize deletion, change a successful
  BlobStore receipt, or revoke a canonical commit.
- The fixed node
  `tests/test_catalog_projection.py::test_json_projection_rejects_incompatible_derived_documents`
  rejects an incompatible derived document; Plan 08 records its actual pass/fail
  evidence and cannot substitute a broader file-level run for this named node.
- A close race preserves a committed write and reports the declared typed
  partial/retryable outcome.
- Removed aliases, factories, and alternate decorators are absent and no hidden
  wrapper constructs a cache.
- The remote candidate translates declared retryable outcomes without claiming
  Phase 8 live PostgreSQL/Amazon S3 qualification.

## Wave 0 Requirements

- [ ] `tests/test_phase6_lookup_contract.py` — presence, outcome, one-read, and
  cached-`None` cases.
- [ ] `tests/test_phase6_removal_contract.py` — exact deletion, structured
  report, pagination, and replacement-race cases.
- [ ] `tests/test_phase6_statistics.py` — immutable per-outcome aggregate.
- [ ] `tests/test_phase6_decorator_contract.py` — explicit ownership,
  suppression, function isolation, and clear reporting.
- [ ] `tests/test_phase6_public_api_contract.py` — canonical positive and legacy
  negative surface.
- [ ] `tests/test_phase6_policy_contract.py` — TTL, bounded size enforcement,
  invalidation ownership, and fail-closed preflight.
- [ ] `tests/contracts/test_phase6_topology_policy.py` — supported local profiles
  and deterministic remote-candidate semantics.

No new test framework or external-service fixture is required.

## Manual-Only Verifications

All Phase 6 behaviors have automated verification. Real-service qualification
is a Phase 8 gate, not a manual Phase 6 substitute.

## Validation Sign-Off

- [ ] Every final plan task has an automated command or explicit Wave 0 dependency
- [ ] Sampling continuity: no three consecutive tasks lack automated verification
- [ ] Wave 0 covers every missing test file above
- [x] No watch-mode flags
- [ ] Measured per-task feedback latency remains below 60 seconds
- [ ] `nyquist_compliant: true` set after post-execution validation

**Approval:** pending execution
