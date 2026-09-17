---
phase: 09-adoption-and-release-surface-closure
plan: 09
subsystem: documentation
tags: [public-api, blob-store, format-handler, documentation, mcap]
requires:
  - phase: 09-adoption-and-release-surface-closure
    provides: "Qualification boundaries and local-ready task-first documentation"
provides:
  - "Current barrel-derived API reference for BlobStore, UnifiedCache, catalog, migration, results, handlers, and focused errors"
  - "One safe store-local MCAP-style FormatHandler tutorial tied to an executable example"
  - "Removal of obsolete backend, configuration, and development-planning guides"
affects: [phase-10-sqlcache-removal, public-documentation, extension-guidance]
actuals:
  tokens: 53102
  tasks: 2
  commits: 2
tech-stack:
  added: []
  patterns:
    - "Public API references are asserted against the actual cacheness and cacheness.storage barrels."
    - "Generic format documentation teaches one store-local, path-based handler journey."
key-files:
  created: []
  modified:
    - docs/API_REFERENCE.md
    - docs/PLUGIN_DEVELOPMENT.md
    - docs/README.md
    - tests/test_public_api_contract.py
    - tests/test_phase9_documentation.py
    - docs/BACKEND_SELECTION.md (deleted)
    - docs/CONFIGURATION.md (deleted)
    - docs/DEVELOPMENT_PLANNING.md (deleted)
key-decisions:
  - "The focused API reference names only current barrel imports and links qualification claims to their single owner."
  - "Format extension guidance uses persisted payload identities and one store-local registry, not global registration or lifecycle APIs."
  - "Pre-cutover guide material is removed from the supported surface rather than retained as redirects or archives."
patterns-established:
  - "Document a custom format through the executable MCAP-style example and private contained staging contract."
  - "Route topology and configuration questions to direct store/cache task guides plus release qualification."
requirements-completed: [CACH-06]
coverage:
  - id: D1
    description: "Current public API reference is bound to actual top-level and storage barrel exports."
    requirement: CACH-06
    verification:
      - kind: unit
        ref: tests/test_public_api_contract.py#test_api_reference_imports_are_current_barrel_exports
        status: pass
    human_judgment: false
  - id: D2
    description: "MCAP-style FormatHandler tutorial teaches stable identities, safe private staging, and store-local registration."
    requirement: CACH-06
    verification:
      - kind: unit
        ref: tests/test_phase9_documentation.py#test_mcap_extension_tutorial_uses_one_store_local_safe_format_path
        status: pass
      - kind: integration
        ref: tests/test_handler_registration.py tests/test_guarded_handler_io.py
        status: pass
    human_judgment: false
  - id: D3
    description: "Obsolete pre-cutover document branches are deleted and navigation leads to current task guides."
    requirement: CACH-06
    verification:
      - kind: unit
        ref: tests/test_phase9_documentation.py#test_navigation_has_no_pre_cutover_configuration_or_backend_branches
        status: pass
    human_judgment: false
duration: 2m
completed: 2026-09-17
status: complete
---

# Phase 09 Plan 09: Public API and format extension closure Summary

**Barrel-derived API documentation, one safe MCAP-style format tutorial, and removal of obsolete pre-cutover guidance.**

## Performance

- **Duration:** 2m
- **Started:** 2026-09-17T04:17:17Z
- **Completed:** 2026-09-17T04:18:46Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- Replaced the legacy mixed API inventory with exact current `cacheness` and
  `cacheness.storage` imports, direct-store/cache-result guidance, catalog and
  stopped-worker migration references, format extension entry points, and
  focused errors.
- Replaced legacy plugin/backend narratives with a single MCAP-style
  `FormatHandler` tutorial that demonstrates stable persisted identities, a
  safe `.mcap` artifact, private contained staging, a regular-file boundary,
  a store-local priority registration, and a round trip.
- Deleted obsolete backend-selection, configuration, and development-planning
  branches and made documentation navigation route readers to the direct-store,
  cache-policy, and qualification guides.

## Task Commits

Each task was committed atomically:

1. **Task 1: Publish the current API reference and MCAP-style FormatHandler tutorial** - `b791a67` (docs)
2. **Task 2: Delete obsolete document branches and repair navigation** - `0c02fb8` (docs)

## Files Created/Modified

- `docs/API_REFERENCE.md` - Focused current API inventory linked to exact barrels and task guides.
- `docs/PLUGIN_DEVELOPMENT.md` - Practical single-format extension tutorial tied to `examples/custom_mcap_format.py`.
- `docs/README.md` - Routes configuration and topology questions to current owners.
- `docs/BACKEND_SELECTION.md` - Deleted obsolete pre-cutover guidance.
- `docs/CONFIGURATION.md` - Deleted obsolete pre-cutover guidance.
- `docs/DEVELOPMENT_PLANNING.md` - Deleted obsolete historical planning guidance.
- `tests/test_public_api_contract.py` - Binds documented imports to actual barrels.
- `tests/test_phase9_documentation.py` - Covers the extension tutorial and navigation cleanup.

## Decisions Made

- The API reference is a focused current import inventory, not an encyclopedic
  list of old constructors or a second owner of qualification claims.
- The only generic extension journey is one store-local `FormatHandler`; the
  persisted payload identity controls compatibility, not the Python base-class
  name.
- Historical material remains in Git/planning history, not the public docs
  surface. Dedicated `SqlCache` material remains physically untouched for the
  Phase 10 removal scope.

## Verification

Passed:

```text
uv run pytest -q -o log_cli=false tests/test_public_api_contract.py tests/test_phase9_documentation.py tests/test_handler_registration.py tests/test_guarded_handler_io.py tests/test_security_documentation.py -x
# 32 passed
```

The same verification confirmed the three obsolete documents are absent.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

The sandbox initially denied access to the shared `uv` cache and Git index;
the approved execution environment resolved both without changing source or
test scope.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 9 has one final plan after this public-reference closure.
- The inherited Phase 5/07.1 documentation verifier relocation is intentionally
  outside this plan's owned files and is handled by the phase-level follow-up.

## Self-Check: PASSED

- `docs/API_REFERENCE.md` and `docs/PLUGIN_DEVELOPMENT.md` exist.
- Removed documentation branches are absent.
- Task commits `b791a67` and `0c02fb8` exist in Git history.

---
*Phase: 09-adoption-and-release-surface-closure*
*Completed: 2026-09-17*
