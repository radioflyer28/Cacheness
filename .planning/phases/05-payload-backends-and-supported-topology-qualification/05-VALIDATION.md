---
phase: 05
slug: payload-backends-and-supported-topology-qualification
status: planned
nyquist_compliant: true
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

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 05-01-01 | 05-01 | 1 | BACK-04 | T-05-01..04 | Explicit memory profile tracer; qualification distinct from construction | integration | `uv run --frozen pytest -q tests/test_supported_topologies.py -x -o log_cli=false` | ❌ planned | ⬜ pending |
| 05-01-02 | 05-01 | 1 | BACK-04 | T-05-01..04 | Empty, adjacent, Cartesian, and order cases reject before I/O | unit | `uv run --frozen pytest -q tests/test_supported_topologies.py tests/test_topology_capabilities.py -x -o log_cli=false` | ❌ planned | ⬜ pending |
| 05-02-01 | 05-02 | 2 | BACK-01 | T-05-05..08 | Complete memory/filesystem generation-I/O contract | contract | `uv run --frozen pytest -q tests/contracts/test_payload_generation_io.py -x -o log_cli=false` | ❌ planned | ⬜ pending |
| 05-02-02 | 05-02 | 2 | BACK-01, BACK-04 | T-05-05..08 | Deterministic integrity/recovery/progress fault taxonomy | fault | `uv run --frozen pytest -q tests/test_payload_faults.py tests/contracts/test_payload_generation_io.py -x -o log_cli=false` | ❌ planned | ⬜ pending |
| 05-03-01 | 05-03 | 2 | BACK-01 | T-05-09..13 | Conditional S3 publish and contained verified read | contract | `uv run --frozen pytest -q tests/contracts/test_s3_generation_io.py -x -o log_cli=false` | ❌ planned | ⬜ pending |
| 05-03-02 | 05-03 | 2 | BACK-01 | T-05-09..13 | Multipart/ambiguity bounds and exact verification | fault contract | `uv run --frozen pytest -q tests/contracts/test_s3_generation_io.py -x -o log_cli=false` | ❌ planned | ⬜ pending |
| 05-03-03 | 05-03 | 2 | BACK-01 | T-05-09..13 | Bounded inventory and exact deletion/absence proof | contract | `uv run --frozen pytest -q tests/test_s3_blob_backend.py tests/contracts/test_s3_generation_io.py -x -o log_cli=false` | partial existing | ⬜ pending |
| 05-04-01 | 05-04 | 2 | BACK-04, BACK-05 | T-05-14..18 | Explicit PostgreSQL schema/version initialization | driver contract | `uv run --frozen --extra postgresql pytest -q tests/contracts/test_postgresql_lifecycle_authority.py -x -o log_cli=false` | ❌ planned | ⬜ pending |
| 05-04-02 | 05-04 | 2 | BACK-04, BACK-05 | T-05-14..18 | Exact transactional prepare/verify/promote/abort CAS | driver contract | `uv run --frozen --extra postgresql pytest -q tests/contracts/test_postgresql_lifecycle_authority.py tests/test_lifecycle_authority_contract.py -x -o log_cli=false` | ❌ planned | ⬜ pending |
| 05-05-01 | 05-05 | 3 | BACK-04, BACK-05 | T-05-19..23 | Bounded catalog/debt/clear/reconciliation workflows | driver contract | `uv run --frozen --extra postgresql pytest -q tests/contracts/test_postgresql_lifecycle_authority.py -x -o log_cli=false` | ❌ planned | ⬜ pending |
| 05-05-02 | 05-05 | 3 | BACK-04, BACK-05 | T-05-19..23 | Tier-aware semantic contract and typed SQLSTATE outcomes | contract | `uv run --frozen --extra postgresql pytest -q tests/contracts/test_lifecycle_authority.py tests/contracts/test_postgresql_lifecycle_authority.py tests/test_lifecycle_authority_contract.py -x -o log_cli=false` | ❌ planned | ⬜ pending |
| 05-06-01 | 05-06 | 4 | BACK-01, BACK-04 | T-05-24..28 | Remote profile uses the one composition root/engine | integration | `uv run --frozen --extra cloud pytest -q tests/test_supported_topologies.py tests/test_blob_store_composition.py -x -o log_cli=false` | partial existing | ⬜ pending |
| 05-06-02 | 05-06 | 4 | BACK-01, BACK-04 | T-05-24..28 | Common topology recovery and bounded non-authoritative inventory | contract + fault | `uv run --frozen --extra cloud pytest -q tests/contracts/test_topology_lifecycle.py tests/test_payload_faults.py -x -o log_cli=false` | ❌ planned | ⬜ pending |
| 05-07-01 | 05-07 | 4 | BACK-05 | T-05-29..33 | Sanitized non-passing unavailable/failure evidence | runner contract | `uv run --frozen --extra cloud pytest -q tests/qualification/test_live_evidence.py -x -o log_cli=false` | ❌ planned | ⬜ pending |
| 05-07-02 | 05-07 | 4 | BACK-05 | T-05-29..33 | Exact-run service ownership and bounded cleanup | fixture contract | `uv run --frozen --extra cloud pytest -q tests/qualification/test_live_evidence.py -x -o log_cli=false` | ❌ planned | ⬜ pending |
| 05-08-01 | 05-08 | 5 | BACK-04, BACK-05 | T-05-34..38 | Genuine PostgreSQL and AWS S3 service matrices | live collection | `uv run --frozen --extra cloud pytest --collect-only -q tests/integration/test_postgresql_authority.py tests/integration/test_s3_generation.py -o log_cli=false` | ❌ planned | ⬜ pending |
| 05-08-02 | 05-08 | 5 | BACK-04, BACK-05 | T-05-34..38 | Two independent remote clients and recovery | live collection | `uv run --frozen --extra cloud pytest --collect-only -q tests/integration/test_remote_topology.py -o log_cli=false` | ❌ planned | ⬜ pending |
| 05-09-01 | 05-09 | 6 | BACK-01, BACK-04, BACK-05 | T-05-39..43 | Exact docs/runtime/API coverage matrix | documentation contract | `uv run --frozen --extra cloud pytest -q tests/test_phase5_contract_verifier.py tests/test_supported_topologies.py -x -o log_cli=false` | ❌ planned | ⬜ pending |
| 05-09-02 | 05-09 | 6 | BACK-01, BACK-04, BACK-05 | T-05-39..43 | Local architecture verifier cannot spoof live status | verifier | `uv run --frozen --extra cloud pytest -q tests/test_phase5_contract_verifier.py -x -o log_cli=false && uv run --frozen --extra cloud python tools/verify_phase5_contracts.py` | ❌ planned | ⬜ pending |
| 05-10-01 | 05-10 | 7 | BACK-05 | T-05-44..48 | Real-service zero-skip QUALIFIED evidence and cleanup | live qualification | `uv run --frozen --extra cloud python tools/run_phase5_qualification.py --output .planning/phases/05-payload-backends-and-supported-topology-qualification/05-LIVE-QUALIFICATION.json` | ❌ external gate | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

## Wave 0 Requirements

- [ ] Plan 05-02 creates the reusable local payload-generation contract; Plan 05-03 adds S3.
- [ ] Plan 05-05 creates the tier-aware memory/SQLite/PostgreSQL authority contract.
- [ ] Plan 05-01 creates the exact matrix and negative cross-pair/edge tests.
- [ ] Plan 05-03 creates deterministic S3 ambiguity/multipart/pagination/delete coverage.
- [ ] Plan 05-07 creates unique run-owned PostgreSQL and AWS S3 fixtures with bounded cleanup.
- [ ] Plan 05-08 creates the combined independent-client/shared-signer suite.
- [ ] Plan 05-07 creates the sanitized non-passing evidence writer and strict markers.
- [ ] Plan 05-10 executes the non-substitutable real-service gate; absence remains open.

## Manual-Only Verifications

None. Live services require external configuration, but qualification remains an automated rerunnable gate and must not be converted into a manual approval.

## Validation Sign-Off

- [x] All 20 tasks have explicit `<automated>` verification; real-service execution remains the Plan 05-10 external precondition.
- [ ] Sampling continuity: no 3 consecutive tasks without automated verification.
- [x] Planned contract/fixture files cover every missing test reference and are ordered before their consumers.
- [ ] No watch-mode flags.
- [ ] Quick feedback latency target is under 30 seconds.
- [x] Real-service absence remains visibly non-passing.
- [x] `nyquist_compliant: true` set after the complete 20-task map.

**Approval:** plan structure mapped; execution evidence pending
