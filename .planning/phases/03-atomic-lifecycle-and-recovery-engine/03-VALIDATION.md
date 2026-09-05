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
| **Full phase command** | `.venv/bin/pytest -q tests/test_lifecycle_authority_contract.py tests/test_sqlite_lifecycle_authority.py tests/test_blob_store_read_contract.py tests/test_blob_store_atomic_lifecycle.py tests/test_blob_store_reconciliation.py tests/test_blob_store_concurrency.py tests/test_blob_store_close_contract.py tests/test_manifest_repository_cas.py tests/test_phase3_scheduler_retirement.py tests/test_phase3_windows_contract.py tests/test_phase3_ruff_delta.py -o log_cli=false` |
| **Repository gate** | `uv run pytest -q -o log_cli=false`, `.venv/bin/python tools/verify_phase3_ruff_delta.py`, and direct Ruff on every Phase 3-new Python file |
| **Estimated runtime** | Establish from Wave 0; focused feedback target under 30 seconds |

---

## Sampling Rate

- **After every task commit:** Run the common authority contract plus the directly changed behavior file.
- **After every plan wave:** Run the full Phase 3 command.
- **Before `$gsd-verify-work`:** Run the full Phase 3 command, full repository suite, compatibility corpus, a real `uv run --python 3.11 --frozen` focused run, the repository Python 3.13 run, attached native-Windows evidence, the checked-in exact-scope Ruff delta gate, and direct Ruff on every Phase 3-new Python file.
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
- Empty compatible store inspection is lazy: get/get_metadata/exists/list and repeated dry-run reconciliation return normal empty results without creating a root, authority database, JSON, journal, temporary, or control object. Established/legacy/scheduler/future/corrupt/wrong-object evidence without authority fails typed and unchanged before any mkdir/open.
- Windows D-22 is limited to one logon session on a local NTFS root provisioned before Cacheness starts, with inheritance disabled and the protected DACL's sole ordinary mutation grant bound to the current token's `S-1-5-5-X-Y` logon SID rather than the persistent account SID. Read-only inspection of an absent/empty root remains zero-mutation empty; any Windows mutation against an absent root fails typed and unchanged with the documented actionable offline PowerShell/`icacls.exe` provisioning command. Cacheness only validates the token/root/DACL before authority database creation/open and every mutation and never creates the Windows root, changes mode, disables inheritance, or edits ACEs. Native evidence must prove no directory/mode/ACL write, unchanged rejection of absent/unsafe/drifted roots, same-session multiprocess SQLite success, and a different-session/service-token denial for authority open and root mutation. If that second token cannot be exercised, the gate is `UNAVAILABLE`, not pass.

---

## Wave 0 Requirements

- [ ] `tests/test_lifecycle_authority_contract.py` — common semantic contract for memory and SQLite adapters.
- [ ] `tests/test_sqlite_lifecycle_authority.py` — schema identity/migration, `journal_mode=DELETE`, `synchronous=EXTRA`, pragmas, busy deadline, rollback, crash recovery, local topology, close, and fork ownership.
- [ ] Shared deterministic authority/payload fault hooks and spawned-process helpers.
- [ ] Characterization tests for public error reasons, reconciliation report shape, JSON metadata projection shape, and released scheduler-format prevalence.
- [ ] Exact before/after filesystem characterization for lazy empty inspection, read-only reopen, and every established authority-missing artifact class.
- [ ] `tests/test_phase3_release_evidence.py` — deterministic repository/package/tag/fixture/changelog release gate with exact contradiction reporting.
- [ ] `tests/test_phase3_ruff_delta.py`, `tools/verify_phase3_ruff_delta.py`, and `tests/fixtures/phase3_ruff_baseline.json` — exact Phase 3-owned scope, stable line-independent fingerprints, removal-tolerant/no-new-findings comparison, and clean-new-file enforcement established before production edits.
- [ ] `tests/test_phase3_windows_contract.py` plus `verify_platform.py --phase3` — pre-provisioned-root enforcement, zero runtime directory/mode/ACL mutation, actionable offline provisioning failure, unchanged absent/unsafe/drift rejection, current-token logon-SID/DACL validation, same-session positive behavior, and different-session/service-token denial.
- [ ] Benchmark probes for authority transaction duration, clear snapshot duration, busy wait, and distinct-key payload overlap; budgets are set from measured baselines.
- [ ] Retire or rewrite scheduler-internal tests in `test_manifest_repository_cas.py`, `test_clear_recovery.py`, and adjacent Phase 3 files so the test surface is the lifecycle-authority interface and `BlobStore` behavior.

---

## Manual-Only Verifications

No lifecycle behavior is accepted as manual-only. Native Windows and supported-Python execution use automated commands. Because Phase 8 owns the permanent CI matrix, Plan 09 has a blocking evidence checkpoint: a non-Windows host, or a Windows host unable to exercise a genuinely different logon-session/service token, records `UNAVAILABLE` and cannot approve the native-Windows gate. Python 3.11 must actually execute through repository/uv interpreter selection; missing-interpreter output is a blocker, not a passing skip.

---

## Validation Sign-Off

- [ ] Every planned task has an automated verify command or explicit Wave 0 dependency.
- [ ] Sampling continuity: no three consecutive tasks lack an automated check.
- [ ] Wave 0 covers every missing test reference and compatibility question.
- [ ] No watch-mode flags or timing-only race assertions are used.
- [ ] Focused feedback latency is measured and remains under 30 seconds.
- [ ] Common contract passes unchanged against both current authority adapters.
- [ ] Full deterministic crash, rollback, ABA, clear, reconciliation, and concurrency matrices pass.
- [ ] Canonical reconciliation machine reports use a versioned v2 envelope and every finding includes authoritative/expected generation, operation provenance, residue type/role, proposed action, reason, disposition, and checkpoint state; the exact legacy v1 dictionary remains available through its characterized adapter.
- [ ] No authority transaction spans handler or payload backend I/O.
- [ ] No file-native receipt, inventory, head, tail, anchor, cursor, pending-control, or authority-lock protocol remains reachable.
- [ ] Full repository suite and compatibility corpus pass; real Python 3.11/3.13 and native-Windows evidence is attached; the exact-scope Ruff delta contains no unmatched finding and every Phase 3-new Python file passes direct Ruff (repository-wide debt remains Phase 8).
- [ ] `nyquist_compliant: true` and `wave_0_complete: true` are set only after implementation evidence is complete.

**Approval:** pending replacement-plan execution and verification
