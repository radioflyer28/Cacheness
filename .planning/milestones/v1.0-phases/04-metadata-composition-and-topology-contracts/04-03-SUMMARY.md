---
phase: 04-metadata-composition-and-topology-contracts
plan: "03"
subsystem: storage composition
tags: [topology, composition, ownership, capabilities, memory]
requires:
  - phase: 04-02
    provides: Native catalog and format-2 storage contracts
provides:
  - One role-aware StoreTopology resolver for named and exact injected participants
  - Explicit ownership, pre-I/O capability minima, and conservative composed guarantees
  - A production-quality same-process memory BlobStore tracer
affects: [04-04, 04-05, 04-06, 04-08, BlobStore, catalog authority]
actuals:
  tokens: 14637
  tasks: 2
  commits: 2
tech-stack:
  added: []
  patterns:
    - A single typed topology root separates payload, authority, and projection roles
    - Named participants declare preflight capabilities; injected participants preserve identity and caller ownership by default
    - In-memory topology is explicitly ephemeral and same-process, without durability or acceleration claims
key-files:
  created:
    - src/cacheness/storage/composition.py
  modified:
    - src/cacheness/storage/backends/__init__.py
    - src/cacheness/storage/backends/blob_backends.py
    - src/cacheness/storage/blob_store.py
    - src/cacheness/storage/memory_lifecycle_authority.py
    - tests/test_blob_store_composition.py
    - tests/test_topology_capabilities.py
key-decisions:
  - "StoreTopology is the sole direct BlobStore composition root; injected participants retain exact identity and default caller ownership."
  - "Memory topology provides only same-process ephemeral behavior, with canonical scan but no durability or index acceleration."
  - "Capability minima are checked before named factory construction or participant I/O."
patterns-established:
  - "Role validation and capability composition occur at topology resolution rather than in BlobStore selectors."
  - "Ownership ledger teardown closes resolver-owned resources once in reverse creation order while preserving caller-owned injections."
requirements-completed: [BACK-02, BACK-03, BACK-06, BACK-07]
coverage:
  - id: D1
    description: Registered participants and exact injected instances resolve through one role-aware StoreTopology path.
    requirement: BACK-03
    verification:
      - kind: unit
        ref: uv run --frozen pytest -q tests/test_blob_store_composition.py tests/test_metadata_role_contract.py -k "registered or injected or role or option or resolve" -x
        status: pass
    human_judgment: false
  - id: D2
    description: Ownership teardown, partial-construction unwind, and capability minima are explicit and fail before named participant I/O.
    requirement: BACK-06
    verification:
      - kind: unit
        ref: uv run --frozen pytest -q tests/test_topology_capabilities.py tests/test_blob_store_composition.py -k "capability or minimum or ownership or close or unwind" -x
        status: pass
    human_judgment: false
  - id: D3
    description: Metadata authority and projection roles are distinct so JSON and PostgreSQL projections cannot be used as Phase 4 lifecycle authority.
    requirement: BACK-02
    verification:
      - kind: unit
        ref: tests/test_metadata_role_contract.py
        status: pass
    human_judgment: false
  - id: D4
    description: Direct BlobStore memory tracer performs put, info, and close through registered and exact-injection paths while advertising only same-process guarantees.
    requirement: BACK-07
    verification:
      - kind: integration
        ref: uv run --frozen pytest -q tests/test_blob_store_composition.py tests/test_topology_capabilities.py tests/test_metadata_role_contract.py -o log_cli=false
        status: pass
    human_judgment: false
duration: 12m 43s
completed: 2026-09-08
status: complete
---

# Phase 04 Plan 03: Role-Aware Topology and Memory Tracer Summary

**A single role-aware StoreTopology now composes direct BlobStore participants with exact injection, ownership and capability contracts, and a truthful same-process memory tracer.**

## Performance

- **Duration:** 12m 43s
- **Started:** 2026-09-08T01:45:35Z
- **Completed:** 2026-09-08T01:58:18Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- Added `BackendRef`, a role registry, `StoreTopology`, and one resolver for registered names and exact injected payload, authority, and projection participants.
- Added explicit caller-versus-store ownership, reverse-order closing, partial-construction unwind, participant/composed capability reports, and pre-I/O minimum validation.
- Routed direct `BlobStore` construction through the topology root and proved a registered-name and injected-instance memory-only put → info → close tracer without persistence claims.

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement one role-aware registry and resolver** - `20ab890` (feat)
2. **Task 2: Enforce ownership/capabilities and execute the memory-only tracer** - `74acb5f` (feat)

## Files Created/Modified

- `src/cacheness/storage/composition.py` - Typed topology references, role registry/resolution, ownership ledger, capability declarations, minima, and composed guarantees.
- `src/cacheness/storage/backends/__init__.py` - Exposes topology-aware backend registration alongside existing factories pending final cutover.
- `src/cacheness/storage/backends/blob_backends.py` - Declares in-memory payload capabilities and deterministic close behavior.
- `src/cacheness/storage/blob_store.py` - Consumes `StoreTopology` as its direct composition root and supports ephemeral in-memory handler I/O.
- `src/cacheness/storage/memory_lifecycle_authority.py` - Reports the memory authority's same-process capabilities.
- `tests/test_blob_store_composition.py` - Covers name resolution, exact injection, ownership, close/unwind, and the memory tracer.
- `tests/test_topology_capabilities.py` - Covers applicability, minima rejection, and topology-qualified guarantees.

## Decisions Made

- `StoreTopology` is the sole direct `BlobStore` composition root; it does not merge options into injected objects or silently fall back to legacy selectors.
- Injected resources stay caller-owned unless ownership is explicitly transferred. Resolver-created resources are closed once, in reverse construction order.
- Memory participants expose ephemeral, same-process behavior and canonical scan support only. They do not claim durable persistence, cross-process sharing, or index acceleration.
- JSON and PostgreSQL metadata paths remain projection roles in this phase; they cannot be selected as lifecycle authority.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Test contract] Scoped the legacy-selector retirement assertion to the direct BlobStore path.**
- **Found during:** Task 1 (Implement one role-aware registry and resolver).
- **Issue:** The Wave 0 assertion also rejected deprecated factory exports whose atomic source/export deletion is intentionally owned by final cutover Plan 08.
- **Fix:** Kept the direct BlobStore prohibition while allowing the deliberately retained factories outside that path.
- **Files modified:** `tests/test_blob_store_composition.py`
- **Verification:** Registered, injected, role, option, and resolver contract tests passed.
- **Committed in:** `20ab890`.

**2. [Rule 2 - Critical capability enforcement] Moved named capability validation ahead of factory construction.**
- **Found during:** Task 2 (Enforce ownership/capabilities and execute the memory-only tracer).
- **Issue:** A named participant with an impossible capability minimum could otherwise be constructed before rejection, violating the plan's pre-I/O safety boundary.
- **Fix:** Added declared named-participant capabilities for preflight validation and retained actual capability checks for injected instances, with owned-resource unwind on failure.
- **Files modified:** `src/cacheness/storage/composition.py`, `tests/test_topology_capabilities.py`
- **Verification:** Capability/minimum/ownership/close/unwind tests passed.
- **Committed in:** `74acb5f`.

**3. [Rule 1 - Bug] Applied private permissions to the temporary memory snapshot path.**
- **Found during:** Task 2 (Enforce ownership/capabilities and execute the memory-only tracer).
- **Issue:** The initial memory handler I/O implementation attempted to set permissions on the buffered file writer instead of its filesystem path.
- **Fix:** Applied the permission mode to the temporary snapshot path before handing it to the handler.
- **Files modified:** `src/cacheness/storage/blob_store.py`
- **Verification:** The full composition, topology, and metadata-role suite passed.
- **Committed in:** `74acb5f`.

---

**Total deviations:** 3 auto-fixed (2 Rule 1 bugs, 1 Rule 2 critical enforcement).
**Impact on plan:** All changes preserve the clean-cutover boundary, pre-I/O capability rejection, and sole Phase 3 lifecycle authority; no coordinator, lock, queue, or persistence expansion was introduced.

## Issues Encountered

- The main checkout's Git index is sandbox-protected. The orchestrator created the two verified task commits above; no unrelated working-tree artifacts were staged.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Later Phase 4 plans can consume the resolved topology, lifecycle authority, and capability vocabulary without adding competing lifecycle coordination.
- SQLite persistent proof remains intentionally deferred to Plan 04; PostgreSQL authority and S3 topology remain outside Phase 4.

---
*Phase: 04-metadata-composition-and-topology-contracts*
*Completed: 2026-09-08*

## Self-Check: PASSED

- The summary and all seven implementation/test artifacts exist.
- Task commits `20ab890` and `74acb5f` are reachable in repository history.
- `uv run --frozen pytest -q tests/test_blob_store_composition.py tests/test_topology_capabilities.py tests/test_metadata_role_contract.py -o log_cli=false` passes (34 tests).
- Targeted Ruff and `uv run --frozen python tools/verify_phase4_ruff_delta.py` pass.
