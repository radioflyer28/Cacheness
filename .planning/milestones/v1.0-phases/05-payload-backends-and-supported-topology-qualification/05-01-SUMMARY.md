---
phase: 05-payload-backends-and-supported-topology-qualification
plan: "01"
subsystem: storage-composition
tags: [blobstore, topology-qualification, lifecycle, sqlite, s3]
requires:
  - phase: 04-metadata-composition-and-topology-contracts
    provides: StoreTopology as the sole direct BlobStore composition root
provides:
  - Immutable three-profile support catalog independent of participant registration
  - Pre-construction rejection of unsupported authority/payload pairs
  - Read-only qualification reports separate from capability constructibility
affects: [05-02, 05-03, 05-04, 05-05, 05-06, 05-07, 05-08, 05-09]
actuals:
  tokens: 7250
  tasks: 2
  commits: 6
tech-stack:
  added: []
  patterns:
    - Immutable support profiles keyed by explicit authority/payload identities
    - Qualification before participant construction, distinct from capability inspection
key-files:
  created:
    - tests/test_supported_topologies.py
  modified:
    - src/cacheness/storage/composition.py
    - tests/test_blob_store_composition.py
    - tests/test_catalog_projection.py
    - tests/test_catalog_query_contract.py
    - tests/test_sqlite_bootstrap_concurrency.py
key-decisions:
  - "Built-in support is exactly memory/memory, SQLite/filesystem, and PostgreSQL/S3; registration never implies a supported pairing."
  - "Profile records contain immutable requirements only; observed live-service status remains external release evidence."
  - "Capability reports remain a pre-construction constructibility inspection and do not advertise qualification."
patterns-established:
  - "Resolve StoreTopology qualification from explicit named or injected participant identities before factories or payload I/O."
  - "Keep JSON projections out of authority/payload profile identity and lifecycle authority permissions."
requirements-completed: [BACK-04]
coverage:
  - id: D1
    description: Immutable three-profile catalog and public memory/memory BlobStore tracer
    requirement: BACK-04
    verification:
      - kind: integration
        ref: tests/test_supported_topologies.py#test_memory_profile_resolves_before_the_public_store_round_trip
        status: pass
    human_judgment: false
  - id: D2
    description: Deterministic pre-I/O rejection for unqualified Cartesian, incomplete, duplicate, and projection-order declarations
    requirement: BACK-04
    verification:
      - kind: unit
        ref: tests/test_supported_topologies.py
        status: pass
    human_judgment: false
duration: 6min
completed: 2026-09-08
status: complete
---

# Phase 5 Plan 01: Supported Topology Catalog Summary

**An immutable three-profile catalog now separates supported BlobStore topologies from merely constructible participants, with memory/memory proven through the existing lifecycle engine.**

## Performance

- **Duration:** 6 min
- **Started:** 2026-09-08T11:31:37Z
- **Completed:** 2026-09-08T11:37:26Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- Added frozen qualification records for memory/memory, SQLite/filesystem, and PostgreSQL/S3, including topology-specific coordination, atomicity, progress, prerequisites, projection, and evidence requirements.
- Made `StoreTopology.resolve()` reject all unqualified pairs before named factories can construct participants or perform I/O.
- Proved the named memory profile completes a real public `BlobStore` put/get through `AuthorityLifecycleEngine`; corrected SQLite's canonical-query capability declaration.
- Added the exact rejection matrix for cross-pairs, empty or incomplete descriptors, duplicate projections, and registration/projection-order independence.
- Updated current Phase 4 composition fixtures to declare memory, filesystem, or SQLite identity explicitly where they inject a production participant.

## Task Commits

1. **Task 1: Resolve one memory/memory store through an explicit qualified profile**
   - `f55c739` — `test(05-01): add failing supported topology tracer`
   - `6daa8b1` — `feat(05-01): add immutable topology qualification catalog`
2. **Task 2: Reject empty, adjacent, Cartesian, and order-dependent topology declarations before I/O**
   - `051bebb` — `test(05-01): add failing topology rejection matrix`
   - `42f9fcc` — `feat(05-01): reject duplicate topology declarations`
3. **Post-wave integration repair: Declare identities in current supported fixtures**
   - `33aef26` — `test(05-01): declare supported topology fixture identities`

## Files Created/Modified

- `src/cacheness/storage/composition.py` — immutable profile catalog, qualification report, pre-factory allow-list lookup, and duplicate projection validation.
- `tests/test_supported_topologies.py` — public tracer and exact support-boundary matrix.

## Decisions Made

- Support is a catalog lookup on explicit authority/payload identities, not a deduction from registry membership, structural protocol conformance, or capability minima.
- The remote profile declares real PostgreSQL/Amazon S3 prerequisites and its evidence schema, but it contains no mutable readiness or qualification result.
- JSON is a derived projection capability only and cannot change the selected profile or gain authority permissions.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Preserved capability-minimum error precedence before profile lookup**

- **Found during:** Task 1 focused capability regression verification
- **Issue:** The initial qualification lookup masked an existing pre-construction `CapabilityRequirementError` with an injected-identity error.
- **Fix:** Kept static capability-minimum evaluation before qualification lookup; both paths remain pre-factory and pre-I/O.
- **Files modified:** `src/cacheness/storage/composition.py`
- **Verification:** `tests/test_topology_capabilities.py` passes with the supported-topology matrix.
- **Committed in:** `6daa8b1`

**2. [Rule 1 - Bug] Repaired current supported-contract fixtures after strict identity enforcement**

- **Found during:** Post-wave Phase 4 cutover matrix verification
- **Issue:** Current Phase 4 tests injected valid memory, filesystem, and SQLite participants without the newly required explicit qualification identity, masking their intended supported topology.
- **Fix:** Declared the matching support identity in current test fixtures; registry-only application names now assert pre-construction rejection rather than creating an unsupported alias.
- **Files modified:** `tests/test_blob_store_composition.py`, `tests/test_catalog_projection.py`, `tests/test_catalog_query_contract.py`, `tests/test_sqlite_bootstrap_concurrency.py`
- **Verification:** `uv run --frozen python tools/verify_phase4_cutover.py` reports 610 passed, 3 skipped.
- **Committed in:** `33aef26`

**Total deviations:** 2 auto-fixed (Rule 1)

## Issues Encountered

None beyond the corrected validation precedence above.

## User Setup Required

None. The PostgreSQL/S3 row is declarative only; its real-service evidence remains a later Phase 5 gate.

## Next Phase Readiness

Plan 05-02 can use the immutable catalog to test local payload participants without inferring support from construction. The PostgreSQL/S3 profile remains intentionally unqualified until its later real-service evidence is produced.

## Self-Check: PASSED

- Confirmed the two owned implementation/test files exist and the four task commits are present in git history.
- Focused verification passed: `uv run --frozen pytest -q tests/test_supported_topologies.py tests/test_topology_capabilities.py -x -o log_cli=false`.
- Targeted lint passed: `uv run --frozen ruff check src/cacheness/storage/composition.py tests/test_supported_topologies.py`.

---
*Phase: 05-payload-backends-and-supported-topology-qualification*
*Completed: 2026-09-08*
