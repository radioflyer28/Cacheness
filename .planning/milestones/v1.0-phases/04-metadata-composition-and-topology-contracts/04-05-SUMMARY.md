---
phase: 04-metadata-composition-and-topology-contracts
plan: "05"
subsystem: derived catalog projections
tags: [catalog, projections, checkpoints, topology, blobstore, sqlite]
requires:
  - phase: 04-04
    provides: authenticated revision-bound canonical catalog pages from the lifecycle authority
provides:
  - One derived-only projection controller with bounded page pulls and checkpoint-after-apply ordering
  - Immutable named projection outcomes on BlobReceipt and typed committed-partial refresh failures
  - Explicit refresh and isolated rebuild boundaries that require declared capabilities and offline SQLite maintenance
affects: [04-06, 04-07, 04-08, BlobStore, UnifiedCache, catalog authority]
actuals:
  tokens: 11744.25
  tasks: 2
  commits: 3
tech-stack:
  added: []
  patterns:
    - Derived ProjectionSink participants consume canonical CatalogPage scans without lifecycle authority permissions
    - Projection batches apply idempotently before durable checkpoint advancement
    - Committed BlobReceipt values attribute derived status without redefining canonical state
key-files:
  created:
    - src/cacheness/storage/projections.py
    - docs/CATALOG_AND_TOPOLOGY.md
  modified:
    - src/cacheness/storage/composition.py
    - src/cacheness/storage/blob_store.py
    - src/cacheness/storage/read_contract.py
    - src/cacheness/error_handling.py
    - tests/test_catalog_projection.py
key-decisions:
  - "Projection sinks are strictly derived BackendRole.PROJECTION participants and cannot authorize lifecycle, cleanup, or canonical-query completeness."
  - "Projection checkpoints bind store, epoch, schema, query, revision, and cursor; apply precedes checkpoint so replay is idempotent."
  - "A projection error preserves the committed BlobReceipt; explicit refresh raises a typed committed-partial error carrying that receipt unchanged."
  - "SQLite projection rebuilds require explicitly requested offline maintenance with stopped workers; no migration or live-service claim is made."
patterns-established:
  - "Use ProjectionController for bounded canonical pulls, best-effort post-commit delivery, caller-requested refresh, and isolated publication."
  - "Treat JSON/ORM/PostgreSQL-facing materialization as attributable derived work, never a lifecycle rollback condition."
requirements-completed: [BACK-02, BACK-06, BACK-07]
coverage:
  - id: D1
    description: Projection participants remain role-separated from lifecycle authority and consume bounded revision-pinned catalog pages with checkpoint-after-apply ordering.
    requirement: BACK-02
    verification:
      - kind: unit
        ref: uv run --frozen pytest -q tests/test_catalog_projection.py tests/test_metadata_role_contract.py -o log_cli=false
        status: pass
    human_judgment: false
  - id: D2
    description: Projection failures leave authority entries committed, attach immutable named receipt outcomes, and preserve the exact receipt in explicit committed-partial errors.
    requirement: BACK-07
    verification:
      - kind: integration
        ref: tests/test_catalog_projection.py#test_post_commit_projection_failure_preserves_the_authority_receipt
        status: pass
    human_judgment: false
  - id: D3
    description: Refresh and isolated rebuild remain explicit capability-qualified operations; SQLite publication requires offline stopped-worker maintenance.
    requirement: BACK-06
    verification:
      - kind: unit
        ref: tests/test_catalog_projection.py -k "refresh or rebuild or offline"
        status: pass
    human_judgment: false
duration: 13m 52s
completed: 2026-09-07
status: complete
---

# Phase 04 Plan 05: Derived Catalog Projection Summary

**Bounded, revision-pinned derived projection delivery now preserves canonical BlobStore commits, exposes immutable receipt outcomes, and requires explicit capability-qualified refresh or rebuild recovery.**

## Performance

- **Duration:** 13m 52s
- **Started:** 2026-09-07T22:36:14-04:00
- **Completed:** 2026-09-07T22:50:06-04:00
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- Added `ProjectionController`, `ProjectionCheckpoint`, bounded idempotent batches, and named projection outcomes around canonical `CatalogPage` pulls without adding a second lifecycle authority, queue, or cross-resource transaction.
- Reused the shared composition role vocabulary so JSON, ORM, and PostgreSQL-facing participants remain derived-only; checkpoints bind the source, schema, query, revision, and continuation cursor.
- Extended frozen `BlobReceipt` values with immutable named outcomes, introduced a typed committed-partial error, and kept best-effort failures from revoking the committed generation.
- Documented catalog/topology boundaries and added explicit refresh/rebuild APIs, with SQLite rebuild publication restricted to stopped-worker offline maintenance.

## Task Commits

1. **Task 1 RED: Add failing projection checkpoint contract** — `f3734b6` (`test`)
2. **Task 1: Pull bounded revision snapshots into role-checked projections** — `497d3ed` (`feat`)
3. **Task 2: Preserve commits through projection failure and expose refresh/rebuild** — `0fe5eb5` (`feat`)

## Files Created/Modified

- `src/cacheness/storage/projections.py` — Derived-only bounded pull, checkpoint, status, refresh, and isolated rebuild controller.
- `src/cacheness/storage/composition.py` — Shared `ProjectionSink` protocol anchored to the existing projection role.
- `src/cacheness/storage/blob_store.py` — Post-authority projection invocation, immutable receipt attribution, and explicit refresh/rebuild facades.
- `src/cacheness/storage/read_contract.py` — Immutable named projection outcomes on frozen receipts.
- `src/cacheness/error_handling.py` — Typed committed-partial projection error and stable reason.
- `docs/CATALOG_AND_TOPOLOGY.md` — Current catalog authority, projection semantics, and Phase 5–7 boundaries.
- `tests/test_catalog_projection.py` — Checkpoint identity, committed failure, receipt immutability, refresh, and rebuild coverage.

## Decisions Made

- A projection sink is never a lifecycle authority: it has no canonical read completeness, promotion, cleanup, or recovery permission.
- Apply-before-checkpoint provides deterministic page replay without another coordinator; a source cursor remains authority-authenticated.
- Derived failure is a status or a typed committed-partial result, not a rollback signal. The canonical receipt remains attributable evidence of the committed generation.
- SQLite rebuild publication is deliberately maintenance-only. No background retry queue, live PostgreSQL/S3 qualification claim, or historic migration executor was introduced.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The main checkout blocks direct Git-index writes. The orchestrator committed the verified RED and GREEN task boundaries individually; no unrelated files were staged.
- The default uv cache was sandbox-protected during verification, so tests used a temporary writable uv cache. Dependency resolution and the frozen lockfile remained unchanged.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Later Phase 4 work can consume derived projection status and explicit recovery APIs without consulting projections for canonical lifecycle state.
- Phase 5 remains responsible for real PostgreSQL/S3 qualification, Phase 6 for cache policy composition, and Phase 7 for migration/rebuild execution.

---
*Phase: 04-metadata-composition-and-topology-contracts*
*Completed: 2026-09-07*

## Self-Check: PASSED

- All seven declared implementation, documentation, and test files exist.
- Task commits `f3734b6`, `497d3ed`, and `0fe5eb5` are reachable in Git history.
- The full projection/role suite, BlobStore composition regression suite, targeted Ruff, and Phase 4 Ruff-delta gate pass.
