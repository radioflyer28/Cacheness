---
phase: 03
slug: atomic-lifecycle-and-recovery-engine
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-09-04
replanned: 2026-09-06
replan_target: 03-21 through 03-25
governing_decision: docs/adr/0001-topology-specific-storage-guarantees.md
windows-qualification:
  current_host_status: UNAVAILABLE
  milestone_status: NOT_QUALIFIED
  native_evidence: false
  windows_qualified: false
  backlog_phase: 999.1
  future_required_status: PASS
  future_required_exit_code: 0
  future_powershell_command: "uv run --python 3.11 --frozen python verify_platform.py --phase3 --require-system Windows --require-python 3.11 --phase3-root $env:CACHENESS_PHASE3_WINDOWS_ROOT"
  root_environment_variable: CACHENESS_PHASE3_WINDOWS_ROOT
  second_token_environment_variable: CACHENESS_PHASE3_WINDOWS_SECOND_TOKEN_COMMAND_JSON
---

# Phase 03 — Validation Strategy

> ADR 0001 validation contract for the SQLite-authority/local-filesystem topology.

## Current authoritative gap acceptance: Plans 03-21 through 03-25

This section supersedes task numbering and bootstrap availability assumptions in
the historical Plan 20 strategy below. Implementation is pending; no checkbox or
passing verdict is inferred from planning. Current verification is gaps_found.
Planning/review was inline at the user's request, not independent agent review.

Shared-process tests initialize the local SQLite/filesystem store before workers
start. Plans 22 and 24 require explicit compatibility approvals. Incomplete
initialization must fail typed unchanged, not be adopted or made universally
available through another filesystem protocol. Canonical cache reads use
authenticated same-generation BlobStore metadata; optional projection failure
cannot revoke a valid blob. Memory tests claim one-process behavior only.

### Task coverage and commands

All commands below run from an isolated disposable checkout with its own copied
fixtures. Never run the compatibility/full suite against the original workspace's
dirty sqlite-columns-v0314 WAL/SHM. Record lstat identity/size/mtime/ctime and SHA-256
before/after, and retain those original files unchanged. Each implementation task
runs its PLAN verify command before commit; no three tasks lack automated checks.
Checkpoint tasks instead require recorded human approval before runtime edits.

| Plan / tasks | Behavior | Class | Required tests |
|---|---|---|---|
| 21.1–2 | Aborted memory mutation; three-debt page/action-size-1 resume, retirement/insertion, apply twice | Recovery + integrity | test_blob_store_reconciliation.py; test_lifecycle_authority_contract.py |
| 21.3 | Raw strict point/list/query decoder, signed-cache evidence preservation | Integrity | test_cached_query_meta.py; test_query_meta_security.py; test_cache_integrity.py |
| 22.1–2 | Approved initialize/close/spawn/reopen, single-process convenience, incomplete/foreign state preserved | Progress contract + integrity | test_sqlite_bootstrap_concurrency.py; test_unified_cache_lifecycle_authority.py; read/legacy contracts |
| 22.3 | Primary/extended/unknown SQLite error codes and caused rollback outcomes | Recovery + progress | test_sqlite_lifecycle_authority.py; test_sqlite_authority_admission.py |
| 23.1 | Single verified entry, pre-deserialization metadata, None vs absence, snapshot/close lifetime | Integrity | test_blob_store_read_contract.py; test_blob_store_close_contract.py; test_blob_store_integrity.py |
| 23.2 | Exact committed receipt, pre/post-promotion failures, metadata round trip, engine-owned debt cleanup | Integrity + recovery | test_blob_store_atomic_lifecycle.py; reconciliation/concurrency suites |
| 24.1–3 | Approved post-commit behavior, cache engine delegation, projection corruption/outage, stale policy, separate namespaces | Integrity + recovery | unified cache lifecycle/adversarial, cached query, integrity, projection CAS/SQL and query_meta suites |
| 25.1 | Compact public direct-store and cache workflow | User acceptance | test_phase3_local_workflows.py (created in 25.1) |
| 25.2 | Exact committed full qualification; protected evidence and platform limits | Evidence | Groups below |

Plan 25 runs these groups independently at the SAME implementation commit:

1. **Gap/public workflow:** `uv run --isolated --python 3.11 --all-extras --group dev --frozen pytest -q tests/test_blob_store_reconciliation.py tests/test_cached_query_meta.py tests/test_sqlite_bootstrap_concurrency.py tests/test_sqlite_lifecycle_authority.py tests/test_blob_store_read_contract.py tests/test_unified_cache_lifecycle_authority.py tests/test_unified_cache_adversarial_lifecycle.py tests/test_phase3_local_workflows.py -o log_cli=false`
2. **Integrity/compatibility/recovery:** `uv run --isolated --python 3.11 --all-extras --group dev --frozen pytest -q tests/test_lifecycle_authority_contract.py tests/test_blob_store_atomic_lifecycle.py tests/test_blob_store_integrity.py tests/test_blob_store_close_contract.py tests/test_blob_store_legacy_contract.py tests/test_cache_integrity.py tests/test_query_meta_security.py tests/test_projection_mutation_contract.py tests/test_projection_sql_atomicity.py tests/test_filesystem_containment.py tests/test_sqlite_metadata_bootstrap_atomicity.py tests/test_phase3_postreview_concurrency.py -o log_cli=false`. Retain the existing partial-stream crash and file-fsync/before-directory-fsync crash tests from Plan 20; confirm selection and collected names in the ledger.
3. **Progress:** `uv run --isolated --python 3.11 --all-extras --group dev --frozen pytest -q tests/test_blob_store_concurrency.py tests/test_sqlite_authority_admission.py tests/test_sqlite_concurrency.py tests/test_sqlite_concurrency_temp.py tests/test_query_meta.py -o log_cli=false`. Initialized independent-process barriers prove supported sharing; safe success/conflict/typed retryable timeout are accepted, never every contender within 0.187 seconds.
4. **Development runtime smoke:** Repeat group 1 with `--python 3.13`; record actual interpreters. The full supported-version matrix remains Phase 8.
5. **Performance only:** `uv run --isolated --python 3.11 --all-extras --group dev --frozen python benchmarks/lifecycle_authority_benchmark.py --verify-baseline benchmarks/lifecycle_authority_baseline.json`. Report environment/distributions and regressions separately. Do not change runtime failure semantics or relax the baseline to pass.
6. **Lint:** `uv run --isolated --python 3.11 --all-extras --group dev --frozen python tools/verify_phase3_ruff_delta.py`; then `uv run --isolated --python 3.11 --all-extras --group dev --frozen ruff check src/cacheness/storage/memory_lifecycle_authority.py src/cacheness/storage/sqlite_lifecycle_authority.py src/cacheness/storage/blob_store.py src/cacheness/storage/lifecycle.py src/cacheness/storage/read_contract.py src/cacheness/storage/__init__.py src/cacheness/metadata.py src/cacheness/core.py tests/test_blob_store_reconciliation.py tests/test_cached_query_meta.py tests/test_sqlite_bootstrap_concurrency.py tests/test_sqlite_lifecycle_authority.py tests/test_unified_cache_lifecycle_authority.py tests/test_unified_cache_adversarial_lifecycle.py tests/test_blob_store_read_contract.py tests/test_blob_store_atomic_lifecycle.py tests/test_blob_store_close_contract.py tests/test_cache_integrity.py tests/test_phase3_local_workflows.py`.
7. **Repository gate:** `uv run --isolated --python 3.11 --all-extras --group dev --frozen pytest -q -o log_cli=false` from the clean detached implementation commit. Record counts, skips and all failures, not just the latest rerun.

### Completion and stop conditions

The finite failure matrix is in `03-GAP-REPLAN.md`. Existing tests may change only
when their exact premise is superseded by an approved contract; record the mapping
and retain independent integrity assertions. New test files are explicit tasks,
not assumed fixtures. Protected original sidecars and Windows qualification
artifacts remain unchanged. Windows is still UNAVAILABLE/NOT_QUALIFIED; Phase
999.1 must supply native evidence before a Windows-qualified release.

If qualification finds a defect, keep it failed, classify the actual invariant,
topology and reproduction, and stop. Do not enter another automatic fix/review
cycle, add coordination machinery or erase an intermittent failure with reruns.
Missing Python/env or nonzero test/lint exit is not a pass. A fresh phase verifier
must close `03-VERIFICATION.md`; planning and execution summaries cannot do so.

## Historical Plan 20 validation target (retained evidence)

---

## ADR 0001 Authoritative Validation Target

Phase 3 supports SQLite authority plus local filesystem blobs on one host with
multiple processes. Memory authority is same-process-only. PostgreSQL,
multihost, S3 authority/composition, and native Windows qualification are not
Phase 3 gates; established compatibility behavior remains protected.

SQLite is the sole transactional lifecycle authority. Its short transaction
owns entry visibility, intent, debt, clear progress, reconciliation checkpoints,
projection dirtiness, and authority revision. Immutable filesystem payload
effects occur outside SQLite and are made crash-consistent by durable intent,
idempotent completion, and deterministic reconciliation. Validation must not
describe the combined resources as one ACID transaction.

| Requirement / evidence | Class | Acceptance target |
|---|---|---|
| STOR-03 | Integrity | A read observes the old or new authenticated complete generation, never a mixture. |
| STOR-04 | Recovery + integrity | Pre-promotion interruption preserves the old generation plus exact intent, including process loss during `create_stream_durable_exclusive` copying and after file fsync but before containing-directory fsync; post-promotion interruption preserves the new generation plus cleanup debt. |
| STOR-05 | Integrity + recovery | Overwrite/delete/clear/close repeat safely and never reclaim a live locator. |
| STOR-06 | Recovery | Dry-run is non-mutating; apply uses indexed high-water/keyset work, exact revalidation, checkpoints, and resume. |
| STOR-07 | Integrity + progress | SQLite lineage CAS prevents disagreement; a contender may succeed, conflict, or return a typed retryable timeout. |
| Latency/throughput distributions | Performance | Dedicated benchmark regression evidence only; no runtime-default or universal-success assertion. |

### Independent command groups

- **Task 3 focused benchmark/config:** before Task 3 commits, run `uv run --isolated --python 3.11 --all-extras --group dev --frozen pytest -q tests/test_config_validation.py -o log_cli=false`, then the performance command, Phase 3 Ruff-delta verifier, and direct Ruff on Task 3's owned Python/test files. This is focused local evidence, not phase qualification.
- **Task 4 correctness:** after Task 3 commits, run `uv run --isolated --python 3.11 --all-extras --group dev --frozen pytest -q tests/test_lifecycle_authority_contract.py tests/test_blob_store_read_contract.py tests/test_blob_store_atomic_lifecycle.py tests/test_projection_mutation_contract.py tests/test_projection_sql_atomicity.py tests/test_filesystem_containment.py -o log_cli=false` from the clean detached Task 1/2/3 commit.
- **Task 4 recovery:** run `uv run --isolated --python 3.11 --all-extras --group dev --frozen pytest -q tests/test_sqlite_lifecycle_authority.py tests/test_blob_store_atomic_lifecycle.py tests/test_blob_store_reconciliation.py tests/test_blob_store_close_contract.py tests/test_sqlite_metadata_bootstrap_atomicity.py tests/test_phase3_postreview_concurrency.py -k "interruption or crash or streaming or fsync or rollback or cleanup or reconcile or clear or close or bootstrap" -o log_cli=false`; the partial-stream and file-fsync/directory-fsync-gap cases must be selected.
- **Task 4 progress:** run `uv run --isolated --python 3.11 --all-extras --group dev --frozen pytest -q tests/test_blob_store_concurrency.py tests/test_sqlite_authority_admission.py tests/test_sqlite_concurrency.py tests/test_sqlite_concurrency_temp.py -o log_cli=false`; the pass condition is safe success/conflict/typed-retryable-timeout accounting, not every contender succeeding within a fixed small interval.
- **Task 4 performance:** run `uv run --isolated --python 3.11 --all-extras --group dev --frozen python benchmarks/lifecycle_authority_benchmark.py --verify-baseline benchmarks/lifecycle_authority_baseline.json`; this checks named statistical workload envelopes only.
- **Task 4 repository qualification:** after all four groups pass independently, run the Phase 3 Ruff-delta verifier, scoped direct Ruff, and `uv run --isolated --python 3.11 --all-extras --group dev --frozen pytest -q -o log_cli=false` from the same clean detached worktree. Protect the original workspace's compatibility WAL/SHM sidecars with before/after lstat and SHA-256 fingerprints.

### Task and validation ownership

| Task | Files it may modify | Validation responsibility |
|---|---|---|
| Task 1 | Five runtime/tracer files named in 03-20 | End-to-end SQLite/filesystem tracer, including the two partial-publication crash boundaries. |
| Task 2 | Five integrity/concurrency test files named in 03-20 | Retained fail-closed regressions and guarantee-class rewrites. |
| Task 3 | Lifecycle benchmark, lifecycle baseline JSON, config validation test | Focused local benchmark/config verification before commit. |
| Task 4 | `03-20-SUMMARY.md` only | Read-only detached-worktree qualification of committed Tasks 1-3 and protected-sidecar identity proof. |

### Stop-condition gate

Validation fails if a proposed fix adds or retains an authority-wide lock,
FIFO queue, sidecar, lease, lifecycle truth, or statement-by-statement timeout
stage solely to make a stress schedule finish within the old measured deadline.
Process-local coordination is accepted only when optional, bounded, and proven
to improve performance; removing it must leave SQLite/CAS correctness intact.
The runtime contention timeout remains caller-configurable operational policy.
The recorded 0.187-second value is benchmark/regression evidence only.

### Plan 03-19 finding disposition

- Retain atomic metadata bootstrap and fail-closed query-metadata parsing as
  integrity/recovery coverage.
- Retain caused typed SQLite BUSY/LOCKED translation, rollback, safe bootstrap
  reclassification, projection snapshot cleanup, and uncertain-commit recovery.
- Remove authority-wide FIFO admission, bootstrap readiness scheduling,
  exhaustive timeout-stage census/allowlists, and the dirty pre-dispatch-stage
  patch unless an independently demonstrated integrity or recovery invariant
  requires a smaller replacement.
- Keep `03-VERIFICATION.md` at `gaps_found` until a fresh verifier runs these
  commands; this validation strategy does not fabricate a passing verdict.

## Legacy Pre-ADR Validation Detail (Historical)

The remaining detail below records the prior transactional-authority test
inventory. When it requires all contenders to succeed, couples runtime behavior
to 0.187 seconds, or treats a typed retryable timeout as corruption, the ADR 0001
target above supersedes it.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.4.1 |
| **Config file** | `pyproject.toml` |
| **Quick run command** | Use the independent correctness and recovery commands in the ADR 0001 target above. |
| **Full phase command** | Run correctness, recovery, and progress separately, then the frozen repository gate from a clean detached worktree. |
| **Repository gate** | `uv run --isolated --python 3.11 --all-extras --group dev --frozen pytest -q -o log_cli=false`, `uv run --isolated --python 3.11 --all-extras --group dev --frozen python tools/verify_phase3_ruff_delta.py`, and direct Ruff from the clean worktree. |
| **Estimated runtime** | Establish from Wave 0; focused feedback target under 30 seconds |

---

## Sampling Rate

- **After every task commit:** Run the common authority contract plus the directly changed behavior file.
- **After every plan wave:** Run the full Phase 3 command.
- **Before `$gsd-verify-work`:** Follow the authoritative command groups above, then run the frozen full suite plus Ruff-delta checks from a clean detached worktree at the exact implementation commit. Do not run the compatibility corpus against the original workspace's dirty WAL/SHM sidecars; fingerprint those protected files before and after instead. Retain the existing `UNAVAILABLE`/`NOT_QUALIFIED` current-host artifact unchanged; no new native-Windows capture is a Phase 3 gate.
- **Max feedback latency:** 30 seconds for focused tests; split longer crash/platform matrices into explicit gate jobs.

---

## Requirement Verification Map

| Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| STOR-03 | T-03-01 partial visibility | Every fault boundary exposes the old or new complete generation; promotion is one authority transaction | contract + subprocess crash | `.venv/bin/pytest -q tests/test_lifecycle_authority_contract.py tests/test_blob_store_atomic_lifecycle.py -x` | Contract file ❌ W0; BlobStore file requires rewrite | ⬜ pending |
| STOR-04 | T-03-02 unindexed residue | Durable intent precedes persistent payload side effects; exact indexed work/debt survives reopen, including partial exclusive streaming and the file-fsync/directory-fsync gap | fault + recovery | Use the authoritative Task 4 recovery command above | Existing files; Plan 03-20 adds the two missing publication-boundary cases | ⬜ pending |
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
- D-32 closes only the current-host evidence obligation: this milestone must atomically capture and freshly verify the fixed Python 3.11 command's exit 2, canonical `UNAVAILABLE` JSON, `NOT_QUALIFIED`, `native_evidence: false`, and backlog Phase 999.1 record. Protected-NTFS-root proof, Python 3.11 native execution, same-session contention, different-session/service-token denial, and scheduler-retirement PASS remain unsatisfied prerequisites for a future Windows-qualified release. The future PowerShell command is `uv run --python 3.11 --frozen python verify_platform.py --phase3 --require-system Windows --require-python 3.11 --phase3-root $env:CACHENESS_PHASE3_WINDOWS_ROOT`; it requires `CACHENESS_PHASE3_WINDOWS_ROOT` and `CACHENESS_PHASE3_WINDOWS_SECOND_TOKEN_COMMAND_JSON` to be supplied by the eligible environment.

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

No lifecycle behavior is accepted as manual-only. Native Windows and supported-Python execution use automated commands. Because Phase 8 owns the permanent CI matrix, this milestone records a non-Windows host, or a Windows host unable to exercise a genuinely different logon-session/service token, as atomically attested `UNAVAILABLE`/`NOT_QUALIFIED` evidence with `native_evidence: false`; it cannot approve a native-Windows gate. Python 3.11 must actually execute through repository/uv interpreter selection; missing-interpreter output is a blocker, not a passing skip. The current artifact verifies only an unavailable system and Phase 999.1 must later run `uv run --python 3.11 --frozen python verify_platform.py --phase3 --require-system Windows --require-python 3.11 --phase3-root $env:CACHENESS_PHASE3_WINDOWS_ROOT` with the provisioned-root and distinct-token environment variables before any Windows-qualified release.

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
- [ ] The full repository suite passes from a clean detached worktree; compatibility checks use disposable fixture copies and leave the original workspace's dirty WAL/SHM sidecars fingerprint-identical; Python 3.11 executes; the exact-scope Ruff delta contains no unmatched finding; and the existing `UNAVAILABLE`/`NOT_QUALIFIED` artifact remains honest and unchanged. Native protected-root, same-session, different-token, and scheduler-retirement `PASS` evidence remains required by Phase 999.1 before a Windows-qualified release.
- [ ] `nyquist_compliant: true` and `wave_0_complete: true` are set only after implementation evidence is complete.

**Approval:** the existing D-32 artifact remains `UNAVAILABLE`/`NOT_QUALIFIED` and is not a native Windows qualification. Final Phase 3 approval remains pending execution and fresh verification of Plan 03-20 against the authoritative ADR 0001 target; Phase 999.1 remains mandatory before any Windows-qualified release.
