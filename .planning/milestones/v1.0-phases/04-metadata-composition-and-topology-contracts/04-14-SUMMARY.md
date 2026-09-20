---
phase: 04-metadata-composition-and-topology-contracts
plan: "14"
subsystem: storage-composition
tags: [blobstore, projection, json, topology, ast-audit, ruff]
requires:
  - phase: 04-13
    provides: public-cutover evidence and the bounded Phase 4 release matrix
provides:
  - replay-safe, derived-only JSON ProjectionSink registration
  - truthful deferred PostgreSQL projection construction
  - alias-aware executable-consumer audit with adversarial fixtures
  - refreshed Python 3.11/3.13 release evidence
affects: [phase-05-backend-qualification, phase-07-migration, phase-08-qualification]
actuals:
  tokens: 10373
  tasks: 3
  commits: 3
tech-stack:
  added: []
  patterns:
    - versioned same-directory JSON replacement for derived projection state
    - source-shared AST audit fixtures for executable consumer checks
key-files:
  created:
    - tests/test_phase4_cutover_verifier.py
  modified:
    - src/cacheness/metadata.py
    - src/cacheness/storage/composition.py
    - tools/verify_phase4_cutover.py
    - .planning/phases/04-metadata-composition-and-topology-contracts/04-VALIDATION.md
key-decisions:
  - "JSON is the sole constructible Phase 4 built-in projection and persists only derived checkpointed state."
  - "PostgreSQL remains derived-role classified but is unregistered until Phase 5 supplies a qualified ProjectionSink."
  - "The clean-cutover audit follows AST bindings and aliases, not source-text matching."
  - "The accidental root metadata.py near-copy is removed rather than preserved as a compatibility module."
patterns-established:
  - "Apply a derived batch before checkpoint advancement; persist its identity so restart replay is a no-op."
  - "Release-audit source fixtures must call the same visitor as repository scanning."
requirements-completed: [BACK-02, BACK-03, BACK-06, BACK-07]
coverage:
  - id: D1
    description: JSON is a structurally valid, replay-safe derived projection through RoleRegistry and StoreTopology.
    requirement: BACK-02
    verification:
      - kind: integration
        ref: tests/test_catalog_projection.py#test_json_projection_applies_reopens_and_resolves_through_topology
        status: pass
      - kind: integration
        ref: tests/test_catalog_projection.py#test_json_projection_replays_a_pending_batch_after_checkpoint_failure
        status: pass
    human_judgment: false
  - id: D2
    description: PostgreSQL is classified derived-only but has no premature constructible projection registration.
    requirement: BACK-06
    verification:
      - kind: unit
        ref: tests/test_postgresql_backend.py#test_postgresql_is_deferred_until_a_qualified_projection_sink_exists
        status: pass
    human_judgment: false
  - id: D3
    description: The consumer audit rejects retired direct, alias, and implementation-star imports while ignoring clean root-star and strings.
    requirement: BACK-03
    verification:
      - kind: unit
        ref: tests/test_phase4_cutover_verifier.py
        status: pass
      - kind: other
        ref: uv run --frozen python tools/verify_phase4_cutover.py --audit
        status: pass
    human_judgment: false
  - id: D4
    description: The expanded exact owned matrix and frozen Ruff delta pass on Python 3.11 and 3.13 without a full-tree success claim.
    requirement: BACK-07
    verification:
      - kind: integration
        ref: uv run --frozen --python 3.11 python tools/verify_phase4_cutover.py --all
        status: pass
      - kind: integration
        ref: uv run --frozen --python 3.13 python tools/verify_phase4_cutover.py --all
        status: pass
    human_judgment: false
status: complete
---

# Phase 04 Plan 14: Projection Truthfulness and Release Evidence Summary

**A replay-safe JSON ProjectionSink is the only constructible Phase 4 built-in projection; PostgreSQL is honestly deferred and the release audit now catches executable retired-API aliases.**

## Performance

- **Completed:** 2026-09-08T08:47:06Z
- **Tasks:** 3/3
- **Files modified:** 11 (including the deliberate removal of untracked root `metadata.py`)
- **Release matrix:** 607 passed, 6 skipped across 42 modules on both CPython 3.11.16 and 3.13.15

## Accomplishments

- Implemented a versioned, exact-shape JSON projection document with same-directory atomic replacement, pending-batch replay, and checkpoint-after-apply semantics.
- Made composition truthful: JSON advertises bounded refresh only; PostgreSQL remains derived-role classified but unregistered until Phase 5 qualification.
- Added table-driven AST audit fixtures for direct imports, module aliases, bound aliases, implementation-module star imports, clean root-star, and string-only negatives.
- Removed the confirmed accidental root `metadata.py` near-copy and recorded accurate two-interpreter evidence without changing migration/version boundaries.

## Task Commits

1. **Task 1: Resolve and exercise the truthful built-in JSON projection while deferring PostgreSQL construction** — `bc624d8` (`feat`)
2. **Task 2: Make the executable-consumer AST audit complete for direct and aliased retired uses** — `4d8faba` (`test`)
3. **Task 3: Remove the accidental root module and refresh exact two-interpreter release evidence** — `4e794a1` (`docs`)

## Files Created/Modified

- `src/cacheness/metadata.py` — derived-only JSON sink with validated document, replay-safe pending state, and checkpoint persistence.
- `src/cacheness/storage/composition.py` — JSON-only built-in projection registration; PostgreSQL construction deferred.
- `tools/verify_phase4_cutover.py` — alias-aware AST source audit shared by fixtures and tree scanning.
- `tests/test_phase4_cutover_verifier.py` — adversarial audit fixtures covering every documented syntax boundary.
- `docs/CATALOG_AND_TOPOLOGY.md` and `04-VALIDATION.md` — truthful built-in inventory and current evidence/non-claims.

## Decisions Made

- Projection documents retain only source identity, derived catalog entries, one pending batch identity, and the last validated checkpoint; no authority data or lifecycle control enters the JSON sink.
- PostgreSQL receives no placeholder registration: typed unavailable named resolution is more truthful than a structurally invalid factory result.
- Root-package star imports remain clean because the public `__all__` excludes retired symbols; implementation-module star imports are failures because they can bind retired surfaces.

## Deviations from Plan

None - plan executed as specified. The initial red JSON protocol assertion and the AST fixture module both failed before implementation, then passed after their respective task changes.

## Known Stubs

None.

## Issues Encountered

- A sandboxed Ruff invocation could not open the shared `uv` cache; the same scoped command completed successfully with the required cache access. No source or baseline diagnostics were changed to work around it.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 5 can qualify real PostgreSQL and S3 backend pairs without inheriting a placeholder PostgreSQL projection. Phase 7 retains the explicit offline migration/rebuild boundary. The three pandas-dependent SQL-cache modules remain a classified non-green diagnostic for Phase 8.

## Self-Check: PASSED

All listed files exist, the accidental root module is absent, and Task 1-3
commits `bc624d8`, `4d8faba`, and `4e794a1` resolve in the repository history.

---
*Phase: 04-metadata-composition-and-topology-contracts*
*Plan: 14*
*Completed: 2026-09-08*
