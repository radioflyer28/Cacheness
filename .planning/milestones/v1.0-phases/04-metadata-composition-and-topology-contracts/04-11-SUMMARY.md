---
phase: 04-metadata-composition-and-topology-contracts
plan: "11"
subsystem: storage-catalog
tags: [blob-store, catalog, cursor, projections, topology-capabilities]
requires:
  - phase: 04-10
    provides: one composition root with role-local participant capabilities
provides:
  - Post-commit projection outcomes that retain committed receipt evidence for ordinary sink failures
  - Per-sink rebuild qualification without inflating whole-topology capability reports
  - Finite, strict, authenticated catalog cursor inspection before authority dispatch
affects: [phase-04-plan-12, phase-04-plan-13, phase-05, phase-06]
actuals:
  tokens: 6936
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - External derived-sink boundaries catch Exception, not BaseException, after canonical promotion.
    - Opaque cursor envelopes validate encoded size, decoded size, exact shape, and every field before HMAC comparison or authority access.
key-files:
  created: []
  modified:
    - src/cacheness/storage/projections.py
    - src/cacheness/storage/blob_store.py
    - src/cacheness/storage/catalog.py
    - tests/test_catalog_projection.py
    - tests/test_catalog_query_contract.py
key-decisions:
  - "Ordinary projection failures are named derived outcomes or typed committed-partial errors; BaseException control flow remains visible."
  - "Named rebuild checks the selected projection's ParticipantCapabilities while aggregate topology reporting remains conservative."
  - "Cursor limits model the closed envelope's worst valid JSON representation, including JSON escaping, so creation never emits a token inspection rejects."
patterns-established:
  - "Do not retry, roll back, delete, or otherwise alter canonical authority state after a projection failure."
  - "Bound caller-controlled cursor bytes before base64, UTF-8, JSON, HMAC, manifest loading, or authority catalog dispatch."
requirements-completed: [BACK-02, BACK-06, BACK-07]
coverage:
  - id: D1
    description: "Ordinary derived-sink failures preserve committed receipt evidence, while BaseException control flow propagates."
    requirement: BACK-07
    verification:
      - kind: integration
        ref: "tests/test_catalog_projection.py"
        status: pass
    human_judgment: false
  - id: D2
    description: "A named capable projection can rebuild even when the whole topology truthfully reports a weaker aggregate rebuild capability."
    requirement: BACK-06
    verification:
      - kind: integration
        ref: "tests/test_catalog_projection.py#test_named_rebuild_uses_its_own_sink_capabilities"
        status: pass
    human_judgment: false
  - id: D3
    description: "Public catalog cursors are strictly decoded and fully bounded before canonical authority work."
    requirement: BACK-07
    verification:
      - kind: integration
        ref: "tests/test_catalog_query_contract.py"
        status: pass
    human_judgment: false
duration: 22m
completed: 2026-09-08
status: complete
---

# Phase 04 Plan 11: Projection and Cursor Boundary Summary

**Derived projection failures now retain exact committed receipt evidence, and catalog cursor envelopes are finite and rejected before decoder, parser, or authority work.**

## Performance

- **Duration:** 22m
- **Started:** 2026-09-08T06:33:00Z
- **Completed:** 2026-09-08T06:55:37Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Replaced the narrow projection error tuple with an external-boundary `Exception` catch, preserving immutable receipt identity in dirty outcomes and committed-partial errors while leaving `BaseException` untouched.
- Passed each `ProjectionController` its own `ParticipantCapabilities`; selected projection rebuilds no longer inherit a different sink's unavailable mode.
- Added fixed cursor envelope bounds, strict URL-safe base64 decoding, exact field/signature validation, and constructor symmetry so no validly created cursor is later rejected for size.
- Proved oversized or malformed cursor input cannot invoke `catalog_page` or canonical manifest loading.

## Task Commits

1. **Task 1 RED: Preserve exact committed receipts across every ordinary projection failure** - `eac7c42` (`test`)
2. **Task 1 GREEN: Preserve exact committed receipts across every ordinary projection failure** - `013159c` (`feat`)
3. **Task 2 RED: Reject oversized or malformed cursors before decode, JSON, and authority dispatch** - `e0d263c` (`test`)
4. **Task 2 GREEN: Reject oversized or malformed cursors before decode, JSON, and authority dispatch** - `a3902e1` (`feat`)

## Files Created/Modified

- `src/cacheness/storage/projections.py` - Translates all ordinary external projection failures into truthful derived outcomes and safely discards failed isolated rebuild candidates.
- `src/cacheness/storage/blob_store.py` - Supplies each controller the selected projection's capability declaration instead of the aggregate report.
- `src/cacheness/storage/catalog.py` - Enforces closed-shape encoded, decoded, field, signature, and numeric cursor bounds before HMAC and authority use.
- `tests/test_catalog_projection.py` - Covers custom sink exceptions, BaseException propagation, exact receipt identity, and mixed-capability named rebuilds.
- `tests/test_catalog_query_contract.py` - Covers cursor boundary values, strict decoding, malformed signatures, oversized tokens, and no-dispatch behavior.

## Decisions Made

- Projection delivery is derived-only: post-commit failures preserve the canonical outcome and never schedule retries, undo authority state, or create a second authority.
- Aggregate topology reports remain minimum guarantees; explicit named rebuilds use only the selected sink's declaration.
- Cursor size limits cover worst-case JSON escaping for all six finite textual fields plus the fixed HMAC signature, ensuring emitted tokens are inspectable.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Tracking] Restored the completed-phase count after plan advancement**
- **Found during:** Completion tracking
- **Issue:** The state advancement command reduced `completed_phases` from the roadmap's three completed phases to two while Phase 4 remains in progress.
- **Fix:** Restored the derived count to three so STATE.md remains consistent with ROADMAP.md.
- **Files modified:** `.planning/STATE.md`

**Total deviations:** 1 auto-fixed (1 Rule 1 tracking correction).

## Issues Encountered

- `uv run --frozen python tools/verify_phase4_ruff_delta.py` did not pass because its frozen Phase 4 scope has drifted outside this plan: it reports unrelated `src/cacheness/storage/backends/s3_backend.py`, `tests/test_s3_blob_backend.py`, and other later Phase 4 scope additions. Scoped Ruff and all task test commands passed. This remains recorded in the cross-phase Windows ledger for Phase 4 release verification.

## Verification

- PASS — `uv run --frozen pytest -q tests/test_catalog_projection.py tests/test_blob_store_composition.py -o log_cli=false` (30 passed).
- PASS — `uv run --frozen pytest -q tests/test_catalog_query_contract.py -o log_cli=false` (41 passed).
- PASS — final focused matrix over projection, composition, and cursor contracts (74 passed).
- PASS — scoped Ruff over all Plan 04-11 source and test files.
- OUTSTANDING — Phase 4 Ruff-delta helper scope drift; no finding belongs to a Plan 04-11 file.

## Next Phase Readiness

- Plan 04-12 can migrate the remaining consumers against receipt-preserving projections and finite canonical cursor inputs.
- Before Phase 4 release sign-off, refresh the frozen Ruff-delta scope or reconcile it with the authorized Phase 4 file set; do not suppress or relabel unrelated findings.

## Self-Check: PASSED

- The summary and all five owned source/test files exist.
- TDD commits `eac7c42`, `013159c`, `e0d263c`, and `a3902e1` exist in repository history.

---
*Phase: 04-metadata-composition-and-topology-contracts*
*Completed: 2026-09-08*
