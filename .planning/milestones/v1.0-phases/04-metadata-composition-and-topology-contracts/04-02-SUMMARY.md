---
phase: 04-metadata-composition-and-topology-contracts
plan: "02"
subsystem: storage catalog contracts
tags: [catalog, manifest, hmac, cursor, metadata]
requires:
  - phase: 04-01
    provides: Wave 0 red contracts and the Phase 4 Ruff-delta baseline
provides:
  - Immutable native catalog schema, bounded metadata values, predicates, cursors, and pages
  - Authenticated format-2 manifest vocabulary with independent version dimensions
  - Non-mutating unsupported-layout classification and a frozen BlobReceipt result
affects: [04-03, 04-04, 04-05, 04-06, 04-08, BlobStore, catalog authority]
actuals:
  tokens: 25648
  tasks: 2
  commits: 2
tech-stack:
  added: []
  patterns:
    - Native frozen catalog values validate declared fields while preserving bounded opaque metadata
    - Canonical JSON and HMAC bind explicit format-2 catalog descriptor fields without version lockstep
    - Opaque cursors authenticate every catalog snapshot identity before continuation
key-files:
  created:
    - src/cacheness/storage/catalog.py
  modified:
    - src/cacheness/error_handling.py
    - src/cacheness/storage/manifest.py
    - src/cacheness/storage/read_contract.py
    - src/cacheness/storage/__init__.py
    - tests/test_catalog_schema.py
key-decisions:
  - "Declared defaults materialize only for new writes; portable predicates inspect stored presence and never infer an absent default."
  - "Store format, epoch, manifest schema, SQLite user version, and payload format are explicit independent dimensions."
  - "Unsupported development, future, mixed, incomplete, and foreign layouts produce typed migration-or-rebuild evidence without mutation."
  - "BlobReceipt is the new frozen public commit-result vocabulary; BlobEntryInfo is not re-exported as an alias."
patterns-established:
  - "Catalog query validation completes before authority dispatch and returns typed cursor/revision outcomes."
  - "New manifest wire encoding is explicit rather than a dataclass-derived wire format."
requirements-completed: [BACK-02, BACK-07]
coverage:
  - id: D1
    description: Native catalog schemas preserve opaque mappings, materialize write defaults, enforce exact scalar types, and distinguish absence from null/default.
    requirement: BACK-07
    verification:
      - kind: unit
        ref: uv run --frozen pytest -q tests/test_catalog_schema.py tests/test_catalog_query_contract.py -o log_cli=false
        status: pass
    human_judgment: false
  - id: D2
    description: Format-2 manifests authenticate independent version dimensions and reject unsupported layouts without mutation.
    requirement: BACK-02
    verification:
      - kind: unit
        ref: tests/test_catalog_schema.py#test_current_manifest_keeps_version_dimensions_independent_and_authenticated
        status: pass
      - kind: unit
        ref: tests/test_catalog_schema.py#test_unsupported_development_layout_is_rejected_read_only
        status: pass
    human_judgment: false
  - id: D3
    description: BlobReceipt is frozen and the storage public surface does not re-export BlobEntryInfo as its compatibility alias.
    requirement: BACK-07
    verification:
      - kind: unit
        ref: tests/test_catalog_schema.py#test_blob_receipt_is_frozen_and_not_the_legacy_entry_info_alias
        status: pass
    human_judgment: false
duration: 16m 10s
completed: 2026-09-08
status: complete
---

# Phase 04 Plan 02: Native Catalog and Format 2 Summary

**Native catalog schemas now provide bounded typed metadata and portable query vocabulary, alongside an authenticated format-2 descriptor and frozen BlobReceipt result.**

## Performance

- **Duration:** 16m 10s
- **Started:** 2026-09-08T01:13:18Z
- **Completed:** 2026-09-08T01:29:28Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- Added immutable native catalog fields, schemas, exact signed-64/string/boolean validation, bounded opaque metadata, stored-presence defaults, finite AND-only predicates, authenticated cursors, and bounded pages.
- Defined the format-2 manifest descriptor with separate store epoch, manifest schema, SQLite user-version, and handler payload-format dimensions; its classifier rejects unsupported layouts without initialization or mutation.
- Published the frozen BlobReceipt semantic result and removed BlobEntryInfo from the storage package’s public exports.

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement native schema and portable query value contracts** - `5bd94a4` (feat)
2. **Task 2: Establish format 2, independent version dimensions, and BlobReceipt** - `014373c` (feat)

## Files Created/Modified

- `src/cacheness/storage/catalog.py` - Native schema, bounded values, query predicates, cursor, and page contracts.
- `src/cacheness/error_handling.py` - Typed catalog validation, query, cursor, and migration/rebuild errors.
- `src/cacheness/storage/manifest.py` - Format-2 manifest encoder/decoder, HMAC verification, and read-only layout classifier.
- `src/cacheness/storage/read_contract.py` - Frozen BlobReceipt result.
- `src/cacheness/storage/__init__.py` - Current catalog, manifest, and receipt public exports.
- `tests/test_catalog_schema.py` - Format-2, independent-version, receipt, and non-mutation coverage.

## Decisions Made

- Materialize defaults only when creating a new record; additive reads may present effective defaults, but queries only inspect stored values and presence.
- Use format version 2 to avoid adopting the development format-1 layout, while retaining distinct version fields rather than forcing numeric lockstep.
- Treat migration/rebuild as an explicit future maintenance interface: classifier failures make no path, byte, or metadata mutation.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Restored catalog-presence immutability after JSON decode.**
- **Found during:** Task 2 verification.
- **Issue:** Canonical JSON decoded the stored presence tuple as a list, which failed frozen-manifest validation.
- **Fix:** Reconstructed the presence tuple before current-manifest validation.
- **Files modified:** `src/cacheness/storage/manifest.py`
- **Verification:** Format-2 manifest round-trip test passes.
- **Committed in:** `014373c`.

**2. [Rule 1 - Bug] Normalized invalid predicate values to the portable query error type.**
- **Found during:** Task 1 verification.
- **Issue:** Exact scalar validation leaked a catalog-schema error from the public query boundary.
- **Fix:** Translated invalid predicate and membership values to `CatalogQueryValidationError` while preserving their cause.
- **Files modified:** `src/cacheness/storage/catalog.py`
- **Verification:** Query validation tests pass before authority dispatch.
- **Committed in:** `5bd94a4`.

**3. [Rule 1 - Bug] Corrected the receipt contract fixture to use the authority’s absent expectation constructor.**
- **Found during:** Task 2 verification.
- **Issue:** The new test referred to a nonexistent `EntryExpectation.missing()` helper.
- **Fix:** Used the existing `EntryExpectation.absent()` contract.
- **Files modified:** `tests/test_catalog_schema.py`
- **Verification:** Frozen receipt contract passes.
- **Committed in:** `014373c`.

---

**Total deviations:** 3 auto-fixed (3 Rule 1 bugs).
**Impact on plan:** The corrections preserve the planned immutable wire, query-boundary, and receipt contracts without adding a lifecycle coordinator or compatibility adapter.

## Issues Encountered

- The Task 1 selector matched format-boundary tests by module keyword, so the shared Wave 0 contracts were completed and verified before their task commits were split.
- This executor could not create `.git/index.lock` on the main checkout. The orchestrator staged the already verified files into the two exact task commits above; this was an execution-permission boundary, not a code-scope deviation.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plans 04-03 and 04-04 can compose participants and persist the native catalog descriptor through the existing authority transaction.
- The current format boundary is explicit and read-only; Phase 7 remains responsible for future offline migration and rebuild execution.

---
*Phase: 04-metadata-composition-and-topology-contracts*
*Completed: 2026-09-08*

## Self-Check: PASSED

- All six implementation/test artifacts and this summary exist.
- Task commits `5bd94a4` and `014373c` are reachable in repository history.
