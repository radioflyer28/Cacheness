---
phase: 05-payload-backends-and-supported-topology-qualification
plan: "06"
subsystem: storage-composition
tags: [blobstore, topology, postgresql, s3, bounded-pagination, reconciliation]
requires:
  - phase: 05-02
    provides: exact local topology profiles and composition root
  - phase: 05-03
    provides: bounded S3 payload inventory participant
  - phase: 05-05
    provides: PostgreSQL lifecycle authority and page primitives
provides:
  - built-in S3/PostgreSQL role construction through the one BlobStore composition root
  - bounded remote catalog enumeration and non-authoritative S3 inventory evidence
  - common lifecycle contracts for memory, SQLite/filesystem, and PostgreSQL/S3 profiles
affects: [phase-05-07, phase-05-08, phase-07-migration-rebuild]
actuals:
  tokens: 13566
  tasks: 3
  commits: 6
tech-stack:
  added: []
  patterns:
    - lazy optional-dependency participant factories
    - authority-authenticated page APIs for multi-host workflows
    - signed, bounded report-only inventory continuation
key-files:
  created:
    - tests/contracts/test_topology_lifecycle.py
  modified:
    - src/cacheness/storage/composition.py
    - src/cacheness/storage/blob_store.py
    - src/cacheness/storage/lifecycle.py
    - src/cacheness/storage/reconciliation.py
    - src/cacheness/storage/backends/__init__.py
    - src/cacheness/storage/__init__.py
    - tests/test_supported_topologies.py
    - tests/test_lifecycle_authority_contract.py
    - tests/test_metadata_backend_registry.py
    - tests/test_postgresql_backend.py
key-decisions:
  - "Remote participant construction is lazy and identity-based; registry availability is separate from immutable profile qualification."
  - "PostgreSQL/S3 uses an application-supplied shared manifest key and never creates a host-local default key."
  - "S3 inventory is a single bounded, resumable, report-only evidence page; it cannot establish catalog membership or authorize cleanup."
patterns-established:
  - "Multi-host public enumeration uses list_page with an authority cursor; list_entries remains unreachable for remote profiles."
  - "Report-only inventory cursors are validated and authenticated inside the existing reconciliation resume token."
requirements-completed: [BACK-01, BACK-04]
coverage:
  - id: D1
    description: "All exact topology profiles construct one AuthorityLifecycleEngine and complete their lifecycle contract."
    requirement: BACK-01
    verification:
      - kind: integration
        ref: "tests/contracts/test_topology_lifecycle.py"
        status: pass
    human_judgment: false
  - id: D2
    description: "The remote PostgreSQL/S3 candidate has bounded authority catalog pages and report-only S3 inventory continuation."
    requirement: BACK-04
    verification:
      - kind: integration
        ref: "tests/contracts/test_topology_lifecycle.py"
        status: pass
    human_judgment: false
  - id: D3
    description: "The Phase 4 owned matrix reflects the qualified PostgreSQL authority and current bounded reconciliation report shape."
    verification:
      - kind: integration
        ref: "tools/verify_phase4_cutover.py"
        status: pass
    human_judgment: false
duration: 39min
completed: 2026-09-08
status: complete
---

# Phase 05 Plan 06: Remote Profile Composition and Bounded Workflows Summary

**The PostgreSQL/S3 candidate now composes through the same BlobStore lifecycle engine as the local profiles, while remote enumeration and S3 inventory remain bounded, resumable, and non-authoritative.**

## Performance

- **Duration:** 39 min
- **Started:** 2026-09-08T09:56:00-04:00
- **Completed:** 2026-09-08T10:34:29-04:00
- **Tasks:** 3
- **Files modified:** 11

## Accomplishments

- Registered S3 and PostgreSQL participants lazily through the existing `RoleRegistry`, with explicit instance identities and typed multi-host progress outcomes.
- Required a caller-supplied shared signing key for the remote topology and preserved base-package imports when optional remote dependencies are absent.
- Added `BlobStore.list_page()` for authority-authenticated, cursor-bounded remote catalog access; remote `list()` refuses before it could reach `list_entries()`.
- Added a single S3 inventory page to reconciliation as signed, bounded evidence. Unattributed objects are `REPORT_ONLY` findings and never become catalog, visibility, revocation, or deletion authority.
- Proved all three exact profiles use `AuthorityLifecycleEngine`, including a PostgreSQL spy that fails if an unbounded authority listing is attempted.
- Repaired the stale Phase 4 owned-matrix expectations so they characterize the current report cursor and qualified PostgreSQL authority registration rather than retired compatibility assumptions.

## Task Commits

Each TDD task was committed atomically:

1. **Task 1: Construct the PostgreSQL/Amazon-S3 candidate through the same BlobStore engine**
   - `2214133` (`test`): add the red remote topology composition contract.
   - `107f182` (`feat`): compose the qualified remote topology.
2. **Task 2: Prove common lifecycle recovery and bounded S3 evidence consumption across all profiles**
   - `3c08cfe` (`test`): add the red bounded remote lifecycle contract.
   - `53190c1` (`feat`): bound remote topology workflows and report-only inventory evidence.
   - `1aea67a` (`test`): cover exact local profile engine ownership.
3. **Post-wave integration repair: Align stale Phase 4 ownership assertions**
   - `fabebf3` (`test`): assert the current inventory cursor field and PostgreSQL authority registration without registering a projection.

## Files Created/Modified

- `src/cacheness/storage/composition.py` - lazily constructs built-in S3/PostgreSQL participants and declares remote progress outcomes.
- `src/cacheness/storage/blob_store.py` - requires remote external signing material and exposes authority-backed `list_page()`.
- `src/cacheness/storage/lifecycle.py` - rejects remote unbounded listing before calling `list_entries()`.
- `src/cacheness/storage/reconciliation.py` - carries one bounded S3 evidence page and its validated continuation cursor in the existing signed resume token.
- `src/cacheness/storage/backends/__init__.py` - provides lazy composition re-exports without an import cycle.
- `src/cacheness/storage/__init__.py` - conditionally exposes remote participant classes when their optional dependencies exist.
- `tests/test_supported_topologies.py` - verifies remote composition and exact local engine ownership.
- `tests/contracts/test_topology_lifecycle.py` - verifies cursor-bound remote workflows, signed continuation, and report-only inventory treatment.
- `tests/test_lifecycle_authority_contract.py` - asserts the current exact `inventory_cursor` dictionary field.
- `tests/test_metadata_backend_registry.py` - asserts PostgreSQL resolves as the qualified lifecycle authority and not a projection.
- `tests/test_postgresql_backend.py` - keeps the derived metadata classification separate while asserting authority registration and unavailable projection construction.

## Decisions Made

- The remote candidate is an exact named profile, not a runtime support claim: profile requirements and observed release evidence stay distinct.
- An external shared signing-key provider is mandatory for a multi-host topology; deriving a local key would break its trust boundary.
- The normal remote public listing shape is `list_page()`. Replacing the pre-production unbounded shape avoids silent truncation and avoids materializing the catalog.
- The S3 page is not joined into canonical membership. Reconciliation correlates only locators already attributed by its bounded authority work; all other observed S3 locators remain report-only.

## Verification

- `uv run --frozen --extra cloud pytest -q tests/test_supported_topologies.py tests/test_blob_store_composition.py -x -o log_cli=false` — passed (30 tests).
- `uv run --frozen --extra cloud pytest -q tests/contracts/test_topology_lifecycle.py tests/test_payload_faults.py -x -o log_cli=false` — passed (18 tests).
- `uv run --frozen --extra cloud pytest -q tests/test_supported_topologies.py tests/test_blob_store_composition.py tests/contracts/test_topology_lifecycle.py tests/test_payload_faults.py -x -o log_cli=false` — passed (50 tests).
- `uv run --frozen --extra cloud pytest -q tests/test_blob_store_reconciliation.py tests/test_phase3_local_workflows.py tests/test_clear_recovery.py tests/test_filesystem_containment.py -x -o log_cli=false` — passed (with two expected Windows skips).
- `uv run --frozen --extra cloud ruff check src/cacheness/storage/composition.py src/cacheness/storage/blob_store.py src/cacheness/storage/lifecycle.py src/cacheness/storage/reconciliation.py src/cacheness/storage/backends/__init__.py src/cacheness/storage/__init__.py tests/test_supported_topologies.py tests/contracts/test_topology_lifecycle.py` — passed.
- `python -m py_compile` passed for every modified storage module; `git diff --check a15bc07..HEAD` passed.
- Architecture scan confirms the S3 adapter declares no lifecycle-authority transition calls; the only `list_entries()` locations are locally guarded and the remote contract spy proves they are unreachable for PostgreSQL/S3.
- `uv run --frozen --extra cloud python tools/verify_phase4_cutover.py` — passed (593 tests, 3 expected skips); its full-tree collection diagnostic completed without collection errors.
- The focused Phase 5 suite plus the three repaired ownership-contract modules passed; Ruff passed for each repaired test file.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking issue] Broke the new barrel import cycle with lazy composition exports.**
- **Found during:** Task 1 (remote participant composition).
- **Issue:** Eager composition re-exports in `storage.backends` formed a circular import once the remote factories were made public.
- **Fix:** Replaced those eager composition imports with narrow lazy `__getattr__` re-exports; optional backend classes retain their existing guarded exports.
- **Files modified:** `src/cacheness/storage/backends/__init__.py`.
- **Verification:** Base `cacheness`/`cacheness.storage` import and focused composition tests passed.
- **Committed in:** `107f182`.

### Approved Scope Deviation

**2. Added the minimal real reconciliation seam required for bounded S3 evidence.**
- **Found during:** Task 2 (bounded S3 evidence consumption).
- **Reason:** `BlobStore` delegates reconciliation to `src/cacheness/storage/reconciliation.py`; implementing the plan's bounded, signed, report-only inventory requirement anywhere else would duplicate lifecycle ownership or add a coordinator.
- **Approval:** Parent explicitly approved this single-file scope extension.
- **Fix:** Added exactly one S3 inventory page as report-only evidence and bound its validated continuation cursor to the existing HMAC-signed resume token. It neither materializes an inventory nor makes S3 listing authoritative.
- **Files modified:** `src/cacheness/storage/reconciliation.py`, `tests/contracts/test_topology_lifecycle.py`.
- **Verification:** Remote contract proves two separately resumed bounded pages, no unbounded authority listing, and only `REPORT_ONLY` handling for unknown objects.
- **Committed in:** `53190c1`.

### Post-wave Integration Repair

**3. Updated Phase 4 owned-matrix assertions for the shipped qualified profile.**
- **Found during:** Post-wave Phase 4 cutover verification.
- **Issue:** Three tests still asserted the retired reconciliation dictionary shape and PostgreSQL's former unregistered-authority state.
- **Fix:** Asserted the exact `inventory_cursor` report field; asserted the built-in PostgreSQL lifecycle authority registration; and preserved that no PostgreSQL projection participant is registered. No lifecycle implementation or compatibility shim changed.
- **Files modified:** `tests/test_lifecycle_authority_contract.py`, `tests/test_metadata_backend_registry.py`, `tests/test_postgresql_backend.py`.
- **Verification:** The exact Phase 4 cutover verifier passed with 593 tests and 3 expected skips; focused Phase 5 contracts and Ruff also passed.
- **Committed in:** `fabebf3`.

**Total deviations:** 1 automatic blocking fix, 1 explicitly approved minimal scope extension, and 1 post-wave assertion repair.

## Known Stubs

None.

## Issues Encountered

The task plan named the public `BlobStore` and engine files, but the existing reconciliation implementation is the actual execution seam. The approved narrowly scoped edit preserved the one-engine design instead of wrapping it in a second workflow layer.

## User Setup Required

None. Applications selecting PostgreSQL/S3 must provide their normal optional dependencies, service configuration, initialized authority schema, and an external shared manifest signing key.

## Next Phase Readiness

- The remote candidate has a single composition path and bounded maintenance semantics ready for qualification evidence and performance work.
- Phase 7 retains the explicit version/migration/rebuild seam; this plan introduced no compatibility adapter or implicit migration.

## Self-Check: PASSED

- Confirmed the eleven changed source/test files and this summary exist.
- Confirmed commits `2214133`, `107f182`, `3c08cfe`, `53190c1`, `1aea67a`, and `fabebf3` exist in repository history.

---
*Phase: 05-payload-backends-and-supported-topology-qualification*
*Completed: 2026-09-08*
