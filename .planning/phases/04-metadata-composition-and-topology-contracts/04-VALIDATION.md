---
phase: 04
slug: metadata-composition-and-topology-contracts
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-09-07
---

# Phase 04 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest `>=8.4.1` |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run --frozen pytest -q -o log_cli=false tests/test_catalog_schema.py tests/test_catalog_query_contract.py tests/test_blob_store_composition.py` |
| **Full suite command** | `uv run --frozen pytest -q -o log_cli=false` |
| **Estimated runtime** | Quick suite under 30 seconds; full-suite baseline must be measured during execution |

---

## Sampling Rate

- **After every task commit:** Run the task's targeted test module plus the closest existing authority/read-contract regression module.
- **After every plan wave:** Run `uv run --frozen pytest -q -o log_cli=false` and `uv run ruff check src tests`.
- **Before `$gsd-verify-work`:** Full suite and the Phase 4 contract matrix must be green, with any pre-existing lint baseline distinguished from Phase 4 changes.
- **Max feedback latency:** 30 seconds for per-task targeted checks.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 04-W0-01 | TBD | 0 | BACK-07 | T-04-01 | Exact type validation, bounded schema inputs, and typed rejection of unsupported layouts | unit | `uv run --frozen pytest -q tests/test_catalog_schema.py` | ❌ W0 | ⬜ pending |
| 04-W0-02 | TBD | 0 | BACK-07 | T-04-02, T-04-03 | Bound typed predicates; authenticated revision-bound cursors | contract | `uv run --frozen pytest -q tests/test_catalog_query_contract.py` | ❌ W0 | ⬜ pending |
| 04-W0-03 | TBD | 0 | BACK-02, BACK-03 | T-04-04 | Projections never become authority; one selector path preserves exact injected resources and explicit ownership | contract | `uv run --frozen pytest -q tests/test_metadata_role_contract.py tests/test_blob_store_composition.py` | ❌ W0 | ⬜ pending |
| 04-W0-04 | TBD | 0 | BACK-06 | T-04-05 | Invalid capability claims fail before payload mutation | contract | `uv run --frozen pytest -q tests/test_topology_capabilities.py` | ❌ W0 | ⬜ pending |
| 04-W0-05 | TBD | 0 | BACK-07 | T-04-06 | Projection failure preserves canonical receipt and isolated rebuild state | fault/integration | `uv run --frozen pytest -q tests/test_catalog_projection.py` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_catalog_schema.py` — schema validation, opaque undeclared fields, current-layout reopen, missing/default semantics, explicit evolution boundaries, and non-mutating migration-required rejection of unsupported layouts.
- [ ] `tests/test_catalog_query_contract.py` — portable predicates, bounds, deterministic key/generation ordering, cursor authentication, mismatch, and stale-revision behavior across memory and SQLite.
- [ ] `tests/test_metadata_role_contract.py` — authority/projection role matrix for JSON, memory, SQLite, and PostgreSQL-facing metadata implementations.
- [ ] `tests/test_blob_store_composition.py` — one-root injection, registered-name parity, ambiguity rejection, close ownership, initialization-failure cleanup, and absence of superseded parallel selectors.
- [ ] `tests/test_topology_capabilities.py` — participant/composed capability reporting and construction-time rejection of impossible guarantees.
- [ ] `tests/test_catalog_projection.py` — duplicate/interrupted pull, checkpointing, committed-partial outcomes, and isolated rebuild publication.
- [ ] Shared fixtures for schemas, bounded pages, fault injection, registered roles, and caller-owned resources.

---

## Manual-Only Verifications

All core Phase 4 behaviors have automated verification. Historical compatibility tests may be rewritten or retired when they assert a removed pre-production API; explicit version detection and future migration-tool seams remain tested. A live PostgreSQL service is not used to claim PostgreSQL authority qualification in this phase; that service-backed topology evidence belongs to Phase 5.

---

## Validation Sign-Off

- [ ] All planned tasks have an automated verify command or an explicit Wave 0 dependency.
- [ ] Sampling continuity: no three consecutive tasks lack automated verification.
- [ ] Wave 0 covers all missing test references.
- [ ] Commands contain no watch-mode flags.
- [ ] Targeted feedback latency is below 30 seconds.
- [ ] Python 3.11 and 3.13 public catalog-value/cursor serialization checks are recorded.
- [ ] `nyquist_compliant: true` is set in frontmatter after validation.

**Approval:** pending
