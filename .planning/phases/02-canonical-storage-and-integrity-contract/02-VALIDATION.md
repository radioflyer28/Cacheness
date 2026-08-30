---
phase: 02
slug: canonical-storage-and-integrity-contract
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-08-30
---

# Phase 02 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.4.1 |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest -q -o log_cli=false <phase-test-file> -x` |
| **Full suite command** | `uv run pytest -q -o log_cli=false` |
| **Estimated runtime** | Focused files under 30 seconds; full suite approximately 30 seconds |

---

## Sampling Rate

- **After every task commit:** Run the targeted Phase 2 test file plus `uv run ruff check <changed-source-and-test-paths>`.
- **After every plan wave:** Run all Phase 2 test files plus the Phase 1 compatibility, containment, and clear-recovery regressions.
- **Before `$gsd-verify-work`:** Full pytest, the eight-fixture compatibility validator, and Ruff on Phase 2-created files must be green.
- **Max feedback latency:** 30 seconds for focused checks; split a target that exceeds it.

---

## Per-Task Verification Map

Task/plan identifiers are finalized by the planner; the requirement-to-target contract is fixed here.

| Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| STOR-01 | T-02 manifest parity | JSON, memory, and SQLite expose identical typed manifest semantics and canonical signed bytes | backend contract | `uv run pytest -q -o log_cli=false tests/test_blob_manifest_backends.py -x` | ✅ | ✅ green |
| STOR-02 | T-02 committed visibility | Only authenticated committed manifests can reach a payload snapshot | ordering/contract | `uv run pytest -q -o log_cli=false tests/test_blob_store_read_contract.py -x` | ✅ | ✅ green |
| STOR-08 | T-02 outcome collapse | Missing, corrupt, conflict, unsupported-version, and backend failures remain distinct | API contract | `uv run pytest -q -o log_cli=false tests/test_blob_store_read_contract.py -x` | ✅ | ✅ green |
| SECU-03 | T-02 verify ordering | Manifest authenticity, size, and SHA-256 digest verification precede handler deserialization on one snapshot | adversarial integration | `uv run pytest -q -o log_cli=false tests/test_blob_store_integrity.py -x` | ✅ | ✅ green |
| SECU-04 | T-02 signing downgrade | Missing/invalid/unsafe key material, absent/invalid signatures, and unsupported signer configuration fail closed | unit/platform | `uv run pytest -q -o log_cli=false tests/test_blob_store_integrity.py -x` | ✅ | ✅ green |
| SECU-05 | T-02 incomplete signature projection | Mutating any security-critical manifest field invalidates the signature | parameterized unit | `uv run pytest -q -o log_cli=false tests/test_blob_manifest.py -x` | ✅ | ✅ green |
| SECU-08 | T-02 cache translation collapse | Public typed integrity errors exist and a pure seam classifies them without rewiring `UnifiedCache` | unit contract | `uv run pytest -q -o log_cli=false tests/test_blob_store_translation_seam.py -x` | ✅ | ✅ green |
| MIGR-02 | T-02 version ambiguity | Manifest schema and native payload format versions are independent, exact identifiers | golden/contract | `uv run pytest -q -o log_cli=false tests/test_blob_manifest.py -x` | ✅ | ✅ green |
| MIGR-07 | T-02 future-version guessing | Unknown versions and exact legacy conversion-needed outcomes are typed and non-mutating | compatibility | `uv run pytest -q -o log_cli=false tests/test_blob_store_legacy_contract.py -x` | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

## Required Adversarial Dimensions

- Golden canonical bytes across insertion order, Unicode, empty/optional metadata, backend restart, and reopen.
- Byte-bounded parsing that rejects oversized documents, excessive depth/count/string sizes, invalid UTF-8, duplicate keys, non-string keys, floats/non-finite numbers, boolean-as-integer confusion, negative/oversized sizes, and malformed digests/signatures before payload access.
- Call-order spies proving no handler lookup/invocation, snapshot, access update, delete, rewrite, or signing-key creation occurs after an earlier failure.
- Independent tampering of every signed critical field, plus same-length payload replacement, truncation, extension, missing payload, and replacement around guarded snapshot creation.
- JSON, memory, and SQLite parity, including arbitrary user/handler metadata that the current SQLite `put_entry()` shape can lose.
- Pre/post hashes and mtimes for all eight Phase 1 compatibility fixture trees, proving legacy inspection is read-only.
- Representative NPZ, Parquet, pickle/dill, and Blosc2-native handler paths, proving no new Cacheness payload header is introduced.

---

## Wave 0 Requirements

- [x] `tests/test_blob_manifest.py` — golden codec, bounds, independent versions, and complete signed projection.
- [x] `tests/test_blob_manifest_backends.py` — JSON/memory/SQLite parity and persistent reopen.
- [x] `tests/test_blob_store_read_contract.py` — ordered outcomes for every direct public read surface.
- [x] `tests/test_blob_store_integrity.py` — digest/signature/key/tamper matrix and one-snapshot proof.
- [x] `tests/test_blob_store_legacy_contract.py` — exact eight-fixture read-only outcomes.
- [x] `tests/test_blob_store_translation_seam.py` — pure future `UnifiedCache` classification boundary.

---

## Manual-Only Verifications

All Phase 2 behaviors have automated verification. Non-POSIX key-permission behavior must use platform-specific automated tests or a typed unsupported-capability result rather than subjective inspection.

---

## Validation Sign-Off

- [x] Every plan task has an automated check or explicit Wave 0 dependency.
- [x] Sampling continuity: no three consecutive tasks lack an automated check.
- [x] Wave 0 covers every missing test reference.
- [x] No watch-mode flags are used.
- [x] Focused feedback latency remains under 30 seconds.
- [x] Full suite and compatibility corpus validator pass without suppressing failures.
- [x] Phase 2-created Python files are Ruff-clean.
- [x] `nyquist_compliant: true` and `wave_0_complete: true` are set only after execution evidence is complete.

**Approval:** validated 2026-08-30

## Validation Audit 2026-08-30

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |

The consolidated Phase 2 requirement gate passed across 461 automated cases, with
one expected Windows-junction skip. No manual-only requirement remains.
