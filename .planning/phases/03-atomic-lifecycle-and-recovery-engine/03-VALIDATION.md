---
phase: 03
slug: atomic-lifecycle-and-recovery-engine
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-30
---

# Phase 03 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.4.1 |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest -q -o log_cli=false tests/test_manifest_repository_cas.py tests/test_blob_store_atomic_lifecycle.py -x` |
| **Full phase command** | `uv run pytest -q -o log_cli=false tests/test_manifest_repository_cas.py tests/test_blob_store_atomic_lifecycle.py tests/test_blob_store_reconciliation.py tests/test_blob_store_concurrency.py tests/test_blob_store_close_contract.py tests/test_blob_manifest.py tests/test_blob_manifest_backends.py tests/test_blob_store_read_contract.py tests/test_blob_store_integrity.py tests/test_clear_recovery.py tests/test_filesystem_containment.py -x` |
| **Full suite command** | `uv run pytest -q -o log_cli=false` |
| **Estimated runtime** | Focused files under 30 seconds; full phase gate under 60 seconds on the reference environment |

---

## Sampling Rate

- **After every task commit:** Run the smallest new test module for the touched lifecycle seam plus its nearest Phase 2 or clear-recovery regression file.
- **After every plan wave:** Run the full Phase 3 command.
- **Before phase verification:** Run the full phase command, full pytest suite, targeted Ruff, and `uv run python tests/fixtures/compat/validate_corpus.py --expected-through sqlite-columns-v0314`.
- **Max feedback latency:** 30 seconds for a focused task gate; split slower targets.

---

## Per-Task Verification Map

Task/plan identifiers are finalized by the planner; this requirement-to-target contract is fixed.

| Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| STOR-03 | Lifecycle authority/CAS | Every injected write boundary exposes the old complete or new complete generation; reads never mix manifest and payload | fault + crash/reopen | `uv run pytest -q -o log_cli=false tests/test_blob_store_atomic_lifecycle.py -k 'write or read_acquisition' -x` | ❌ W0 | ⬜ pending |
| STOR-04 | Evidence/provenance | Pre-authority failure preserves old authority and detectable owned residue; post-authority failure preserves new authority and cleanup debt | fault matrix + reopen | `uv run pytest -q -o log_cli=false tests/test_blob_store_atomic_lifecycle.py -k 'failure or recovery or reopen' -x` | ❌ W0 | ⬜ pending |
| STOR-05 | Destructive convergence | Overwrite/delete/clear/close are idempotent, generation-conditional, and ownership-aware | contract + integration | `uv run pytest -q -o log_cli=false tests/test_blob_store_atomic_lifecycle.py tests/test_blob_store_close_contract.py -k 'idempotent or delete or clear or close' -x` | ❌ W0 | ⬜ pending |
| STOR-06 | Reconciliation safety | Dry-run is byte-for-byte non-mutating; apply is bounded, checkpointed, resumable, and evidence-gated | adversarial + reopen | `uv run pytest -q -o log_cli=false tests/test_blob_store_reconciliation.py -x` | ❌ W0 | ⬜ pending |
| STOR-07 | Race determinism | Same-key races have one deterministic CAS winner; distinct keys overlap; local coordination entries retire | deterministic concurrency | `uv run pytest -q -o log_cli=false tests/test_blob_store_concurrency.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

## Required Fault and Race Dimensions

- Inject before/after serialization, operation-record persistence, candidate publication, candidate verification, CAS comparison/write/acknowledgement, authority checkpointing, old-payload reclamation, and operation retirement.
- Exercise ordinary exceptions and process-loss-style `BaseException`; reopen a fresh store after simulated process loss.
- Force write/write, write/delete, read/write, read/delete, clear/post-snapshot write, and close/in-flight interleavings with `Event`/`Barrier` hooks and timeouts, never `sleep()` as the oracle.
- Assert exact manifest/payload/operation bytes and mtimes where stable, no cross-generation reclamation, and one bounded read retry only.
- Prove a blocked key A does not prevent an unrelated key B from completing.

---

## Wave 0 Requirements

- [ ] `tests/test_manifest_repository_cas.py` — exact create-if-absent, replace-if-record/generation, conditional tombstone retirement, independent-instance conflict, and local adapter atomicity.
- [ ] `tests/test_blob_store_atomic_lifecycle.py` — write/overwrite/delete/clear authority and complete fault/reopen matrix.
- [ ] `tests/test_blob_store_reconciliation.py` — dry-run immutability, deterministic reports, bounded pages, apply revalidation, checkpoints, safe quarantine/report policy.
- [ ] `tests/test_blob_store_concurrency.py` — forced same-key/distinct-key races, one-retry acquisition, and coordination-registry retirement.
- [ ] `tests/test_blob_store_close_contract.py` — admission/drain, idempotent close, owned-resource cleanup, injected-backend retention, and no stored-data clear.
- [ ] Shared explicit lifecycle fault hooks/fixtures and bounded inventory/call counters.

---

## Manual-Only Verifications

All Phase 3 behaviors must have automated local reference coverage. Backends whose
topology cannot yet provide the required CAS or quarantine primitives return typed
unsupported-capability outcomes; their full service matrices remain Phases 4 and 5.

---

## Validation Sign-Off

- [ ] All tasks have an automated verify command or explicit Wave 0 dependency.
- [ ] Sampling continuity: no three consecutive tasks lack an automated check.
- [ ] Wave 0 covers every missing test reference.
- [ ] No watch-mode flags or timing-only race assertions are used.
- [ ] Focused feedback latency remains under 30 seconds.
- [ ] Fault, crash/reopen, and race matrices run without unconditional skips.
- [ ] Full suite, compatibility corpus, and targeted Ruff pass.
- [ ] `nyquist_compliant: true` and `wave_0_complete: true` are set only after evidence is complete.

**Approval:** pending
