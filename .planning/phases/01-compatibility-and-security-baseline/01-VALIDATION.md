---
phase: 01
slug: compatibility-and-security-baseline
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-08-29
---

# Phase 01 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.4.1 |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest -q -o log_cli=false <phase-test-file> -x` |
| **Full suite command** | `uv run pytest -q -o log_cli=false` |
| **Estimated runtime** | Focused checks under 30 seconds; full-suite baseline measured during execution |

---

## Sampling Rate

- **After every task commit:** Run the focused command mapped to that task.
- **After every plan wave:** Run `uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false tests/test_public_api_contract.py tests/test_stored_compatibility.py tests/test_filesystem_containment.py tests/test_legacy_array_security.py tests/test_query_meta.py tests/test_query_meta_security.py tests/test_security_documentation.py tests/test_phase10_sqlcache_removal.py -x`.
- **Before phase verification:** Run `uv run pytest -q -o log_cli=false` and `uv run ruff check src tests`.
- **Max feedback latency:** 30 seconds for focused checks; record and split any focused command that exceeds it.

---

## Per-Task Verification Map

Task and wave assignments are finalized by the planner. Every requirement already has an automated target and a Wave 0 file dependency.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 01-01, 01-08..01-12 | 01-01, 01-08..01-12 | 0-6 | MIGR-01 | T-01-09..T-01-11, T-01-28..T-01-36 | Supported exports, aliases, signatures, configs, registries, decorators, errors, and selected `0.3.x` artifacts remain executable | characterization + fixture integration | `uv run pytest -q -o log_cli=false tests/test_public_api_contract.py tests/test_stored_compatibility.py -x` | ✅ W0 | ✅ green |
| 01-06 | 01-06 | 4 | CACH-07 (historical) | T-01-20..T-01-23 | Historical `SqlCache` failure-isolation characterization established the original risk boundary; the retired surface is now absent rather than executable. | historical selector → direct-removal negative contract | `uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false tests/test_phase10_sqlcache_removal.py -x` | Phase 10 | ⚪ superseded — Phase 10 direct-removal/negative contract |
| 01-02, 01-03 | 01-02, 01-03 | 1-2 | SECU-01 | T-01-01..T-01-07, T-01-12, T-01-37 | Every filesystem operation rejects escape forms without mutating rejected metadata or payload evidence | unit + filesystem integration | `uv run pytest -q -o log_cli=false tests/test_filesystem_containment.py -x` | ✅ W0 | ✅ green |
| 01-04 | 01-04 | 3 | SECU-02 | T-01-13..T-01-16 | Legacy headers use bounded non-executing parsing; ordinary NPZ loading disallows pickle; declared invalid artifacts fail closed | unit + stored fixture | `uv run pytest -q -o log_cli=false tests/test_legacy_array_security.py -x` | ✅ W0 | ✅ green |
| 01-05 | 01-05 | 4 | SECU-06 | T-01-17..T-01-19 | Query fields prevalidate before database access while documented numeric/string semantics remain intact | unit + SQLite integration | `uv run pytest -q -o log_cli=false tests/test_query_meta.py tests/test_query_meta_security.py -x` | ✅ W0 | ✅ green |
| 01-07 | 01-07 | 7 | SECU-07 | T-01-24 | Documentation states trusted-payload limits, unsafe serializer risks, object-array opt-in, integrity limitations, and safe defaults | documentation contract | `uv run pytest -q -o log_cli=false tests/test_security_documentation.py -x` | ✅ W0 | ✅ green |

*Status: ✅ green*

Phase-gate threats T-01-25..T-01-27 cover validation sign-off, unsafe-construct regression, and Ruff-baseline integrity; they are tracked by Plan 01-07's final gate rather than attributed to a single phase requirement.

## Gap-Closure Waves 8-10

The completed rows and measured counts below are historical Phase 1 evidence,
not current approval. During Waves 8-9, the only accepted validation state is
`status: draft`, `nyquist_compliant: false`, `wave_0_complete: false`, and
`Approval: pending`.

| Wave | Plans | Review findings | Threat refs | Status |
|------|-------|-----------------|-------------|--------|
| 8 | 01-13 | CR-01, CR-06 | T-01-38..T-01-40 | ✅ complete |
| 9 | 01-14 | CR-04, CR-05 | T-01-41..T-01-45 | ✅ complete |
| 10 | 01-15 | CR-02, CR-03, CR-04 | T-01-46..T-01-48 | ✅ complete |

Executors do not modify `01-REVIEW.md` or `01-REVIEW-FIX.md`, and do not
restore approval. Only the orchestrator renews review and finalizes validation
after Plans 01-13 through 01-15 execute.

STOR-03, STOR-04, STOR-05, STOR-06, and CACH-03 remain **INCOMPLETE**
downstream requirements. The gap waves do not introduce a canonical manifest,
generation/CAS model, or general lifecycle/reconciliation engine.

---

## Required Test Dimensions

- Snapshot metadata and outside-root bytes before hostile read, `exists`, delete, and list calls; assert no mutation after typed rejection.
- Run a host-independent corpus containing POSIX traversal/absolute paths plus Windows drive, UNC, rooted-backslash, and mixed-separator forms.
- Use call-order spies to prove invalid query fields fail before session creation or database execution.
- Cover truncated/oversized headers, invalid UTF-8 and tuple grammar, invalid rank/dim/dtype, object dtype, decompression failure, byte mismatch, and an invalid declared artifact with a valid alternative sidecar.
- Test imports under the full environment and with YAML, SQLAlchemy, and pandas unavailable, including `from cacheness import *`.
- Cover first/middle/last missing-range failures, empty fetches, custom gap-detector failures, and bulk-upsert fallback success/failure.

---

## Wave 0 Requirements

- [x] `tests/test_public_api_contract.py` — public export, alias, signature, exception/reason, and optional-dependency contract.
- [x] `tests/test_stored_compatibility.py` plus safe `0.3.x` fixtures — current and representative legacy format characterization.
- [x] `tests/test_filesystem_containment.py` — reusable hostile path corpus, outside-root evidence, and symlink fixtures.
- [x] `tests/test_legacy_array_security.py` — valid/corrupt legacy raw-array fixtures generated without evaluating metadata.
- [x] `tests/test_query_meta_security.py` — hostile field corpus and pre-database call-order spies.
- [x] Historical `tests/test_sql_cache_failure_contract.py` — original deterministic failing-adapter/gap-detector characterization; superseded by the Phase 10 direct-removal/negative contract in `tests/test_phase10_sqlcache_removal.py`.
- [x] `tests/test_security_documentation.py` — trusted-payload and unsafe-serializer documentation assertions.

---

## Manual-Only Verifications

All phase behaviors have automated verification. Human review may improve documentation wording but is not required to prove the security boundary.

---

## Historical Validation Sign-Off

- [x] All tasks have automated verification or explicit Wave 0 dependencies.
- [x] Sampling continuity: no three consecutive tasks lack an automated check.
- [x] Wave 0 covers every missing test reference.
- [x] No watch-mode flags are used.
- [x] Focused feedback latency remains under 30 seconds.
- [x] Full suite and Ruff gate are green without hiding new failures.
- [x] `nyquist_compliant: true` and `wave_0_complete: true` are set after validation.

## Completed Evidence

- Focused Phase 1 suite: passed in 7.6 seconds before status evidence was recorded.
- Full pytest: 960 passed, 27 skipped.
- Ruff: 118 findings across `src tests`, within the measured 123-finding baseline; all Phase 1-created Python files had zero findings.
- `tests/test_phase1_quality_gates.py` parses Ruff JSON and proves each unsafe-construct sentinel against a synthetic violating snippet before scanning production code.

**Historical evidence:** retained alongside the renewed approval below.

## Gap-Closure Validation Sign-Off

- [x] CR-01 through CR-06 and CR-R1 through CR-R4 have current adversarial regression evidence.
- [x] Focused, full-suite, and executable parsed-Ruff evidence was rerun after Waves 8-10 and the final review fixes.
- [x] Renewed independent code review is clean with zero findings.
- [x] Renewed independent security review is SECURED with 47/47 threats closed.
- [x] Historical terminal state: `status: complete`, `nyquist_compliant: true`, and `wave_0_complete: true` were restored by the orchestrator after review convergence.

## Renewed Approval Evidence

- Final implementation commit: `cf3be4b`.
- Full suite: `uv run pytest -q -o log_cli=false` passed on 2026-08-30 with 27 expected optional/platform skips and one existing collection warning.
- Phase quality gate: `tests/test_phase1_quality_gates.py` passed (7 tests).
- Final focused review matrix passed, including clear recovery, metadata, integrity, containment, query, custom-metadata, and concurrent-access coverage.
- Final security re-audit: SECURED, 47/47 declared threats closed, no unregistered flags.
- Downstream `STOR-03..STOR-06`, `CACH-03`, `BACK-03`, and `BACK-06` remain incomplete and are not claimed by this approval.

**Approval:** approved 2026-08-30

## Canonical Validation Normalization (Phase 11)

The current frontmatter is `status: validated` with the original completed
evidence retained above. The only removed executable selector is `01-06`:
its historical `SqlCache` characterization has an explicit supersession mapping
to Phase 10's direct-removal/negative contract. No historical role or result is
erased.

This normalization remains local and deterministic only. PostgreSQL, Amazon S3,
controlled-Linux performance, and Windows remain `NOT_QUALIFIED`; immutable
publication remains `NOT_PUBLISHED` under the existing deferred owners.
