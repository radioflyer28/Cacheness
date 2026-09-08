---
phase: 05
slug: payload-backends-and-supported-topology-qualification
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-09-08
---

# Phase 05 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.4.1 |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `.venv/bin/pytest -q tests/contracts/test_payload_generation_io.py tests/test_supported_topologies.py -x` |
| **Full suite command** | `.venv/bin/pytest -q -o log_cli=false` |
| **Estimated runtime** | Quick contract target under 30 seconds; full-suite baseline measured during execution |

## Sampling Rate

- **After every task commit:** Run the narrowest affected contract module, including `.venv/bin/pytest -q tests/contracts/test_payload_generation_io.py tests/test_supported_topologies.py -x` once those files exist.
- **After every plan wave:** Run the complete Phase 5 contract/fault suite plus `.venv/bin/pytest -q -o log_cli=false`.
- **Before `$gsd-verify-work`:** The local suite and real-service qualification runner must be green. `UNAVAILABLE` and `NOT_QUALIFIED` leave BACK-05 open.
- **Max feedback latency:** 30 seconds for quick local contract sampling; live-service qualification is a separate bounded gate.

## Per-Task Verification Map

The planner populates task rows after PLAN.md files define the final task graph. The minimum required coverage is:

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 05-W0-01 | TBD | 0 | BACK-01 | TBD | Immutable generations, contained verified reads, bounded I/O | contract + fault | `.venv/bin/pytest -q tests/contracts/test_payload_generation_io.py tests/test_payload_faults.py -x` | ❌ W0 | ⬜ pending |
| 05-W0-02 | TBD | 0 | BACK-04 | TBD | Exactly three supported pairings; unsupported cross-pairs reject before I/O | unit + integration | `.venv/bin/pytest -q tests/test_supported_topologies.py tests/contracts/test_topology_lifecycle.py -x` | ❌ W0 | ⬜ pending |
| 05-W0-03 | TBD | 0 | BACK-05 | TBD | Real PostgreSQL and AWS S3 evidence is sanitized and unavailable is non-passing | live qualification | checked-in Phase 5 qualification runner | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

## Wave 0 Requirements

- [ ] Reusable payload-generation I/O contract for memory, filesystem, and S3.
- [ ] Tier-aware lifecycle-authority contract for memory, SQLite, and PostgreSQL.
- [ ] Exact supported-topology matrix and negative cross-pair tests.
- [ ] Deterministic S3 fault boundary for ambiguous acceptance, multipart failure, pagination, deletion ambiguity, and malformed metadata.
- [ ] Real PostgreSQL fixture with a unique test-owned schema and bounded cleanup.
- [ ] Real AWS S3 fixture with a unique test-owned prefix and bounded cleanup.
- [ ] Combined remote multi-client suite with an externally supplied shared signer.
- [ ] Sanitized evidence writer whose `UNAVAILABLE` and `NOT_QUALIFIED` results are non-passing.
- [ ] Documented pytest live markers; ordinary developer runs may deselect them, but the qualification runner may not.

## Manual-Only Verifications

None. Live services require external configuration, but qualification remains an automated rerunnable gate and must not be converted into a manual approval.

## Validation Sign-Off

- [ ] All tasks have `<automated>` verification or Wave 0 dependencies.
- [ ] Sampling continuity: no 3 consecutive tasks without automated verification.
- [ ] Wave 0 covers all missing references.
- [ ] No watch-mode flags.
- [ ] Quick feedback latency target is under 30 seconds.
- [ ] Real-service absence remains visibly non-passing.
- [ ] `nyquist_compliant: true` set in frontmatter after plan/task mapping is complete.

**Approval:** pending
