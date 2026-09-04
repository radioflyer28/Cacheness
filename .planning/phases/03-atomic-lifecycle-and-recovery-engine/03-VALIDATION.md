---
phase: 03
slug: atomic-lifecycle-and-recovery-engine
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-09-04
---

# Phase 03 — Validation Strategy

> Replacement validation contract for the transactional lifecycle-authority replan.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.4.1 |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `.venv/bin/pytest -q tests/test_lifecycle_authority_contract.py tests/test_sqlite_lifecycle_authority.py -x` |
| **Full phase command** | `.venv/bin/pytest -q tests/test_lifecycle_authority_contract.py tests/test_sqlite_lifecycle_authority.py tests/test_blob_store_atomic_lifecycle.py tests/test_blob_store_reconciliation.py tests/test_blob_store_concurrency.py tests/test_blob_store_close_contract.py -o log_cli=false` |
| **Repository gate** | `uv run pytest -q -o log_cli=false` |
| **Estimated runtime** | Establish from Wave 0; focused feedback target under 30 seconds |

---

## Sampling Rate

- **After every task commit:** Run the common authority contract plus the directly changed behavior file.
- **After every plan wave:** Run the full Phase 3 command.
- **Before `$gsd-verify-work`:** Run the full Phase 3 command, full repository suite, compatibility corpus, supported Python 3.11/3.13 checks, and changed-path Ruff.
- **Max feedback latency:** 30 seconds for focused tests; split longer crash/platform matrices into explicit gate jobs.

---

## Requirement Verification Map

| Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| STOR-03 | T-03-01 partial visibility | Every fault boundary exposes the old or new complete generation; promotion is one authority transaction | contract + subprocess crash | `.venv/bin/pytest -q tests/test_lifecycle_authority_contract.py tests/test_blob_store_atomic_lifecycle.py -x` | Contract file ❌ W0; BlobStore file requires rewrite | ⬜ pending |
| STOR-04 | T-03-02 unindexed residue | Durable intent precedes persistent payload side effects; exact indexed work/debt survives reopen | fault + recovery | `.venv/bin/pytest -q tests/test_sqlite_lifecycle_authority.py tests/test_blob_store_reconciliation.py -x` | SQLite file ❌ W0; reconciliation requires rewrite | ⬜ pending |
| STOR-05 | T-03-03 stale cleanup | Overwrite/delete/clear/close are idempotent and never reclaim a current generation | interface + integration | `.venv/bin/pytest -q tests/test_blob_store_atomic_lifecycle.py tests/test_blob_store_close_contract.py -x` | Existing files require scheduler-decoupled rewrite | ⬜ pending |
| STOR-06 | T-03-04 unsafe repair | Dry-run is non-mutating; apply is indexed, bounded, exact, resumable, and revalidated | adversarial + reopen | `.venv/bin/pytest -q tests/test_blob_store_reconciliation.py -x` | Existing file requires replacement internals | ⬜ pending |
| STOR-07 | T-03-05 race/ABA | Same-key CAS has one winner; ABA is rejected; distinct-key payload work overlaps outside transactions | deterministic concurrency | `.venv/bin/pytest -q tests/test_blob_store_concurrency.py -x` | Existing file requires receipt/barrier test replacement | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

## Required State and Failure Dimensions

- Common contract runs against `InMemoryLifecycleAuthority` and `SqliteLifecycleAuthority` for absent/create, overwrite, tombstone/delete, operation idempotency, rollback, cleanup debt, clear targets, reconciliation checkpoints, malformed rows, unknown versions, and close ownership.
- Subprocess crash matrix interrupts before/after intent commit, candidate creation/fsync, verification, promotion transaction commit, cleanup, debt retirement, clear progress, and reconciliation checkpointing.
- Every crash reopen runs SQLite integrity and foreign-key checks, performs two equal dry-runs, resumes an interrupted apply, and proves the only valid generation survives.
- Same-key write/write and write/delete release promotion barriers together and assert exactly one expected-lineage winner.
- ABA test prepares against absence, performs create/delete, then proves the stale prepared create cannot promote against the later absent lineage.
- Distinct-key test pauses key A at every payload boundary and proves key B completes; counters assert payload overlap and no authority transaction is open at the pause.
- SQLite busy tests hold an independent writer and prove the absolute deadline, typed timeout, rollback, and preserved cause.
- Clear captures exact key/generation targets transactionally; later creates and overwrites are not deleted.
- Reconciliation uses captured high-water IDs and monotonic keyset cursors with independent row/action/byte limits; no filename inventory or offset pagination is accepted.

---

## Wave 0 Requirements

- [ ] `tests/test_lifecycle_authority_contract.py` — common semantic contract for memory and SQLite adapters.
- [ ] `tests/test_sqlite_lifecycle_authority.py` — schema identity/migration, `journal_mode=DELETE`, `synchronous=EXTRA`, pragmas, busy deadline, rollback, crash recovery, local topology, close, and fork ownership.
- [ ] Shared deterministic authority/payload fault hooks and spawned-process helpers.
- [ ] Characterization tests for public error reasons, reconciliation report shape, JSON metadata projection shape, and released scheduler-format prevalence.
- [ ] Benchmark probes for authority transaction duration, clear snapshot duration, busy wait, and distinct-key payload overlap; budgets are set from measured baselines.
- [ ] Retire or rewrite scheduler-internal tests in `test_manifest_repository_cas.py`, `test_clear_recovery.py`, and adjacent Phase 3 files so the test surface is the lifecycle-authority interface and `BlobStore` behavior.

---

## Manual-Only Verifications

No lifecycle behavior is accepted as manual-only. Native Windows/macOS/Linux and supported-Python execution is an automated platform gate; until Phase 8 installs that CI matrix, Phase 3 records unavailable native-host evidence explicitly and does not substitute a platform fake for final platform proof.

---

## Validation Sign-Off

- [ ] Every planned task has an automated verify command or explicit Wave 0 dependency.
- [ ] Sampling continuity: no three consecutive tasks lack an automated check.
- [ ] Wave 0 covers every missing test reference and compatibility question.
- [ ] No watch-mode flags or timing-only race assertions are used.
- [ ] Focused feedback latency is measured and remains under 30 seconds.
- [ ] Common contract passes unchanged against both current authority adapters.
- [ ] Full deterministic crash, rollback, ABA, clear, reconciliation, and concurrency matrices pass.
- [ ] No authority transaction spans handler or payload backend I/O.
- [ ] No file-native receipt, inventory, head, tail, anchor, cursor, pending-control, or authority-lock protocol remains reachable.
- [ ] Full repository suite, compatibility corpus, supported-Python checks, and changed-path Ruff pass.
- [ ] `nyquist_compliant: true` and `wave_0_complete: true` are set only after implementation evidence is complete.

**Approval:** pending replacement-plan execution and verification
