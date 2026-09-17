---
phase: 09-adoption-and-release-surface-closure
plan: 07
subsystem: adoption-documentation
tags: [documentation, onboarding, blobstore, unified-cache, uv]
requires:
  - phase: 09-adoption-and-release-surface-closure
    provides: Four canonical local examples, alias-free FormatHandler, and exact-file CI coverage
provides:
  - Concise local-ready BlobStore-first README with two current quick starts
  - Task-first documentation navigation and executable current workflow guides
  - Explicit local storage, cache-policy, initialization, and maintenance boundaries
affects: [09-08, 09-09, 09-10, phase-10]
actuals:
  tokens: 22565
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - Base checkout installation remains frozen and excludes default dependency groups.
    - README and guides link exact examples and one qualification owner instead of restating lifecycle or release guarantees.
key-files:
  created:
    - tests/test_phase9_documentation.py
  modified:
    - README.md
    - docs/README.md
    - docs/BLOB_STORE.md
    - docs/CACHE_POLICY.md
    - docs/STORAGE_INITIALIZATION.md
    - docs/STORAGE_MIGRATION.md
key-decisions:
  - "The gateway has exactly two local quick starts: direct BlobStore storage and UnifiedCache function policy."
  - "Task guides describe the qualified local topology while RELEASE_QUALIFICATION.md remains the sole detailed claim matrix."
  - "Migration prose retains stopped-worker, copy-verify-switch or confirmed-rebuild discipline without adding lifecycle mechanics."
patterns-established:
  - "Adoption documentation is contract-tested for current imports, no promoted SqlCache route, and frozen base-only checkout setup."
requirements-completed: [CACH-06]
coverage:
  - id: D1
    description: The README provides a concise local-ready BlobStore-first gateway with exactly two explicit quick starts and frozen base-only checkout installation.
    requirement: CACH-06
    verification:
      - kind: integration
        ref: tests/test_phase9_documentation.py#test_readme_is_a_local_ready_blobstore_first_gateway
        status: pass
      - kind: integration
        ref: tests/test_phase9_documentation.py#test_base_checkout_install_is_frozen_and_has_no_extra_opt_in
        status: pass
    human_judgment: false
  - id: D2
    description: Task-first navigation and detailed guides publish current BlobStore, UnifiedCache, initialization, and migration boundaries without promoting SqlCache.
    requirement: CACH-06
    verification:
      - kind: integration
        ref: tests/test_phase9_documentation.py#test_task_first_navigation_links_current_journeys_and_single_matrix
        status: pass
      - kind: integration
        ref: tests/test_phase9_documentation.py#test_task_guides_own_their_current_capabilities
        status: pass
      - kind: integration
        ref: uv run pytest -q -o log_cli=false tests/test_phase9_documentation.py tests/test_phase9_examples.py tests/test_public_api_contract.py -x
        status: pass
    human_judgment: false
duration: 7min
completed: 2026-09-17
status: complete
---

# Phase 9 Plan 07: Onboarding and Task Guides Summary

**Cacheness now opens with a compact local-ready BlobStore-first journey, while task guides accurately show explicit catalog storage, cache policy, initialization, and stopped-worker maintenance.**

## Performance

- **Duration:** 7 min
- **Started:** 2026-09-17T03:40:01Z
- **Completed:** 2026-09-17T03:46:26Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- Replaced the cache-first README with exactly two current, explicit local quick starts: direct in-memory `BlobStore` storage and `UnifiedCache` plus `@cached(cache=cache)` policy.
- Made checkout plus `uv sync --frozen --no-default-groups` the primary local-ready setup, with a clearly secondary local-wheel route and one detailed qualification owner.
- Reorganized documentation around storing objects, caching results, adding a format, and operating or migrating a store.
- Rewrote stale direct storage/cache guidance against catalog receipts, typed cache outcomes, exact-generation removal, deliberate initialization, and explicit offline maintenance.

## Task Commits

1. **Task 1: Make the README gateway and task navigation execute the current story**
   - `cab6ec0` — `test(09-07): add failing adoption documentation contract`
   - `38b78f3` — `docs(09-07): publish BlobStore-first onboarding gateway`
2. **Task 2: Rewrite store, cache, initialization, and migration guides**
   - `89a3f71` — `test(09-07): add failing task-guide documentation contract`
   - `d1ebce7` — `docs(09-07): align store and cache task guides`

## Verification

- `uv run pytest -q -o log_cli=false tests/test_phase9_documentation.py tests/test_phase9_examples.py tests/test_public_api_contract.py -x` — passed (16 tests).
- `git diff --check` — passed.
- Confirmed that the README and task guides do not promote `SqlCache` or an unqualified remote workflow.

## Decisions Made

- The README links, rather than restates, detailed qualification claims so local adoption language cannot accidentally imply publication or broader support.
- The local SQLite-plus-filesystem guide calls the SQLite catalog the visibility authority and calls its external-payload guarantee crash consistency, not cross-resource ACID.
- Direct storage and cache policy remain separate logical roles; `UnifiedCache` reuses a `BlobStore` engine rather than adding another lifecycle owner.

## Deviations from Plan

None - plan executed exactly as written. The initial broad red documentation contract was split into one red commit per planned task before the tracer green commit, preserving the plan's atomic TDD boundary.

## Issues Encountered

- The final test rerun initially could not open the shared uv cache in the filesystem sandbox. Re-running the unchanged command with authorized cache access passed.

## Known Stubs

None.

## User Setup Required

None - no external service configuration is required.

## Next Phase Readiness

- Plan 09-08 can consolidate the detailed guarantees owner against a concise README and task-first navigation without duplicating local topology claims.
- Phase 10 still owns direct SqlCache implementation and asset removal; this plan only removed it from the promoted documentation journey.

## Self-Check: PASSED

- All seven documentation/test artifacts exist and Task commits `cab6ec0`,
  `38b78f3`, `89a3f71`, and `d1ebce7` are present in Git history.

*Phase: 09-adoption-and-release-surface-closure*
*Completed: 2026-09-17*
