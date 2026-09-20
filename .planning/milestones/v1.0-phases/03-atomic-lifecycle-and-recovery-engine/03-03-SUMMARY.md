---
phase: 03-atomic-lifecycle-and-recovery-engine
plan: "03"
subsystem: lifecycle authority
tags: [sqlite, lifecycle-authority, atomicity, rollback, windows, dacl]
requires:
  - phase: 03-02
    provides: durable authority tracer, immutable entry snapshots, and the shared memory/SQLite transition contract
provides:
  - Hardened SQLite LifecycleAuthority identity, containment, durability, timeout, and ownership semantics
  - Atomic authority transitions with rollback and exact uncertain-commit classification
  - An executable Windows local/current-logon-session topology contract with offline-only root provisioning
affects: [03-04, 03-05, 03-09, blobstore-lifecycle]
actuals:
  tokens: 25532
  tasks: 3
  commits: 7
tech-stack:
  added: []
  patterns:
    - SQLite alone owns rollback journals and cross-process write serialization; authority code never manages sidecars.
    - SQLite indexed fields corroborate bounded canonical manifests and every state-changing transition is a short explicit transaction.
    - Windows capability validation only inspects an offline-provisioned root and current-token logon-SID DACL.
key-files:
  created:
    - tests/test_sqlite_lifecycle_authority.py
    - tests/test_phase3_windows_contract.py
  modified:
    - src/cacheness/storage/lifecycle_authority.py
    - src/cacheness/storage/sqlite_lifecycle_authority.py
    - src/cacheness/storage/memory_lifecycle_authority.py
    - src/cacheness/config.py
    - docs/lifecycle-authority.md
    - tests/test_lifecycle_authority_contract.py
key-decisions:
  - "SQLite initialization verifies journal_mode=DELETE, synchronous=EXTRA, foreign_keys, trusted_schema, the confirmed application ID, user_version 1, and generated store identity before use."
  - "Promotion and complete lifecycle transitions advance entries, operation state, cleanup debt, projection dirtiness, and authority revision inside one short SQLite transaction."
  - "Windows supports only an offline-provisioned local root whose protected DACL gives ordinary mutation rights solely to the current token logon SID; Cacheness validates but never edits it."
patterns-established:
  - "Use the one absolute monotonic busy deadline across connection, BEGIN IMMEDIATE, rollback, and uncertain-commit classification."
  - "Treat incomplete or malformed authority rows and DACL evidence as typed fail-closed outcomes before locator or payload access."
requirements-completed: [STOR-03, STOR-04, STOR-05, STOR-07]
coverage:
  - id: D1
    description: SQLite authorities verify their contained identity and durability pragmas, reject invalid topology and identity unchanged, and enforce close/fork/busy ownership.
    requirement: STOR-03
    verification:
      - kind: integration
        ref: tests/test_sqlite_lifecycle_authority.py
        status: pass
      - kind: unit
        ref: tests/test_lifecycle_authority_contract.py
        status: pass
    human_judgment: false
  - id: D2
    description: Promotion, rollback, uncertain-commit classification, clear, reconciliation, and projection transitions preserve whole authority state across SQLite and memory adapters.
    requirement: STOR-04
    verification:
      - kind: integration
        ref: tests/test_lifecycle_authority_contract.py#test_sqlite_promotion_rolls_back_every_participating_authority_row
        status: pass
      - kind: unit
        ref: tests/test_lifecycle_authority_contract.py#test_sqlite_classifies_an_uncertain_commit_by_reopening_exact_operation_state
        status: pass
    human_judgment: false
  - id: D3
    description: The local current-session topology rejects unsupported sharing and exposes only idempotent, authority-owned lifecycle transitions.
    requirement: STOR-05
    verification:
      - kind: unit
        ref: tests/test_phase3_windows_contract.py
        status: pass
      - kind: integration
        ref: tests/test_lifecycle_authority_contract.py#test_complete_clear_reconciliation_and_projection_transitions_use_authority_state
        status: pass
    human_judgment: false
  - id: D4
    description: Exact lineage, generation, and manifest-digest corroboration yields deterministic same-key conflicts without a global lock.
    requirement: STOR-07
    verification:
      - kind: integration
        ref: tests/test_lifecycle_authority_contract.py#test_common_authority_transition_contract
        status: pass
    human_judgment: false
duration: 25min
completed: 2026-09-05
status: complete
---

# Phase 03 Plan 03: Durable SQLite Authority Summary

**A contained SQLite lifecycle authority now verifies its durable identity, executes complete atomic state transitions, and enforces the Windows one-logon-session deployment boundary without a second lock authority.**

## Performance

- **Duration:** 25min
- **Started:** 2026-09-05T01:55:57Z
- **Completed:** 2026-09-05T02:20:47Z
- **Tasks:** 3/3
- **Files modified:** 8

## Accomplishments

- Added connection-local SQLite policy enforcement for the confirmed authority database, including contained-object checks, read-back durability pragmas, identity validation, an absolute busy deadline, and deterministic PID/fork/close behavior.
- Made durable transition state atomic: prepared mutations, verified promotion, cleanup debt, authority revision, clear/reconciliation state, and non-authoritative projection revision commit or roll back together; uncertain commits reopen and classify one exact operation.
- Added the explicit local/current-session topology capability. On Windows it validates a pre-provisioned protected DACL against the current token logon SID, returns an offline PowerShell/`icacls.exe` instruction on any missing or unsafe evidence, and never creates or changes the root/DACL.

## Task Commits

1. **Task 1: Harden SQLite identity, connection policy, busy deadline, and ownership**
   - `3043af1` (`test`) — failing durability, topology, identity, busy, fork, and close coverage
   - `311cc2f` (`feat`) — contained SQLite authority identity, pragma, deadline, and ownership implementation
2. **Task 2: Make every complete authority transition atomic and rollback-safe**
   - `126940b` (`test`) — failing rollback, uncertain-commit, malformed-row, and complete-transition coverage
   - `4ca87af` (`feat`) — atomic SQLite and parity memory transition implementation
   - `9312c82` (`docs`) — corrected the transition contract comment after all declared transitions were implemented
3. **Task 3: Enforce the Windows one-user/session capability and deployment contract**
   - `55ec27d` (`test`) — failing platform-neutral Windows capability and deployment-contract coverage
   - `7577d98` (`feat`) — topology validation, read-only DACL proof, and offline provisioning documentation

## Files Created/Modified

- `src/cacheness/storage/sqlite_lifecycle_authority.py` — contained SQLite creation/open, pragma and identity checks, short atomic transactions, ambiguity classification, and Windows DACL inspection.
- `src/cacheness/storage/lifecycle_authority.py` — digest/generation-correlated snapshots and complete transition interface.
- `src/cacheness/storage/memory_lifecycle_authority.py` — in-memory adapter parity for atomic transition semantics.
- `src/cacheness/config.py` — caller-owned authority busy limit and topology capability contract with JSON/YAML round-trip support.
- `tests/test_sqlite_lifecycle_authority.py`, `tests/test_lifecycle_authority_contract.py`, and `tests/test_phase3_windows_contract.py` — adversarial identity, transaction, and Windows deployment proof.
- `docs/lifecycle-authority.md` — confirmed identity plus exact offline Windows provisioning and validation contract.

## Decisions Made

- The confirmed identity is `.cacheness/lifecycle-authority-v1.sqlite3`, application ID `0x43414348`, user version 1, and a generated store identity; JSON is never an authorization source.
- `BEGIN IMMEDIATE` is the sole cross-process lifecycle commit boundary. SQLite retains exclusive ownership of all journals and sidecars.
- Windows native proof is intentionally deferred to Plan 09; this plan supplies fail-closed, platform-neutral capability enforcement and deployment instructions only.

## TDD Gate Compliance

Each task has its ordered `test(03-03)` RED commit followed by its `feat(03-03)` GREEN commit. The task-two comment correction was committed separately after confirming that no authority transition remained a placeholder.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Kept the in-memory reference adapter semantically aligned with the durable authority contract.**

- **Found during:** Task 2 (Make every complete authority transition atomic and rollback-safe)
- **Issue:** The planned SQLite transition fields and complete-transition semantics would otherwise make common adapter-contract tests disagree with the in-memory reference implementation.
- **Fix:** Added the same bounded generation/digest corroboration, projection/revision, cleanup-debt, clear, and reconciliation transition semantics to the in-memory adapter.
- **Files modified:** `src/cacheness/storage/memory_lifecycle_authority.py`
- **Verification:** Full authority contract suite passed across SQLite and memory adapters.
- **Committed in:** `4ca87af`

---

**Total deviations:** 1 auto-fixed (1 Rule 2 correctness fix)
**Impact on plan:** Required to preserve the planned cross-adapter contract; no new authority architecture or backend was introduced.

## Issues Encountered

- The configuration module has two pre-existing Ruff `F841` findings. The Phase 3 Ruff-delta check confirmed this plan introduced no new findings.

## User Setup Required

None for non-Windows installations. Windows deployment requires the operator to run the documented offline provisioning sequence before the first authority mutation; native enforcement evidence remains a Plan 09 acceptance gate.

## Next Phase Readiness

- Plans 03-04 and 03-05 can build atop a complete, durable authority transaction contract and exact state snapshots.
- Plan 03-09 must collect the native Windows DACL and cross-session/service-token evidence; it cannot treat platform-neutral tests as a substitute.

## Verification

- `5 passed, 2 deselected` — `.venv/bin/pytest -q tests/test_phase3_windows_contract.py -k "rejects_unsupported_before_mutation or contract_shape" -x`
- `29 passed` — `.venv/bin/pytest -q tests/test_lifecycle_authority_contract.py tests/test_sqlite_lifecycle_authority.py -x`
- `70 passed` — `.venv/bin/pytest -q tests/test_config_validation.py -x`
- `Phase 3 Ruff delta: no unmatched findings` — `.venv/bin/python tools/verify_phase3_ruff_delta.py`
- `compileall` passed — `.venv/bin/python -m compileall -q src/cacheness`

## Self-Check: PASSED

- All eight implementation/test artifacts and this summary exist.
- All seven ordered task and correction commits exist in repository history.

---
*Phase: 03-atomic-lifecycle-and-recovery-engine*
*Completed: 2026-09-05*
