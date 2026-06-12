# Phase 28: Silent Data-Loss Remediation - Context

**Gathered:** 2026-06-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 28 repairs the Wave 1 silent data-loss and broken-guarantee findings from
the 2026-06-12 code review. It covers `clear_all()` blob cleanup, write-intent
crash recovery, JSON metadata persistence/corruption behavior, stable
persistent cache keys, and cache-key property testing.

This phase is remediation work. Do not implement the tiered pull-through cache
here; that todo was reviewed and deferred because Phase 28 handles its
prerequisites.

</domain>

<decisions>
## Implementation Decisions

### Source-of-Truth Rule

- **D-01:** Phase 28 decisions are sourced from `docs/CODE_REVIEW_ACTIONS.md`
  and `docs/CODE_REVIEW_FINDINGS.md` unless this context explicitly says
  otherwise.
- **D-02:** `docs/CODE_REVIEW_ACTIONS.md` is the concrete implementation
  contract. `docs/CODE_REVIEW_FINDINGS.md` provides rationale, severity, and
  guardrails.

### Plan Slicing and Order

- **D-03:** Preserve the action-doc granularity: one task equals one beads
  issue equals one atomic commit.
- **D-04:** Plan Phase 28 as ordered Wave 1 slices: TASK-1, TASK-2, TASK-3,
  TASK-4.
- **D-05:** Include SEED-006 as a companion follow-up slice tied to TASK-2, not
  as a silent expansion of TASK-2 acceptance criteria.

### `clear_all()` Cleanup Strategy

- **D-06:** Follow TASK-1. Replace root-only globbing in `_clear_blob_files()`.
- **D-07:** Planner may choose either `blob_backend.list_blobs(namespace)` for
  every namespace directory or careful `rglob` under `cache_dir` after reading
  current blob backend APIs.
- **D-08:** Preserve TASK-1 exclusions: do not delete `*.db`, `*.db-wal`,
  `*.db-shm`, `cache_metadata.json*`, anything under `.intents/`, or
  `.cache_signing_key*`.
- **D-09:** Keep `_clear_blob_files()` no-lock; the caller holds the lock.

### Write-Intent Recovery Boundary

- **D-10:** TASK-2 is mandatory.
- **D-11:** SEED-006 is included in Phase 28 because `ROADMAP.md` and the seed
  tie it directly to TASK-2.
- **D-12:** Plan TASK-2 first, then SEED-006 as a follow-up/second slice if the
  first slice lands cleanly.
- **D-13:** Do not weaken TASK-2's storage-mode invariant. Storage-mode work
  must not introduce deletion or expiry of committed durable entries.

### JSON Failure Behavior

- **D-14:** Follow TASK-3 exactly. Use findings R3/R4 as rationale.
- **D-15:** Data-critical `put_entry()` and `remove_entry()` raise on
  persistence failure.
- **D-16:** Stats/access-time writes remain best-effort.
- **D-17:** Corrupt JSON metadata is preserved as `*.corrupt-<timestamp>` and
  logged at error level before the backend starts empty.
- **D-18:** Missing parent directories get a `mkdir(parents=True,
  exist_ok=True)` retry before data-critical failure bubbles.
- **D-19:** Do not add broader backend health/dirty-state behavior unless the
  planner finds it is already naturally present in the code.

### Cache-Key Remediation Source

- **D-20:** Follow TASK-4 as the concrete implementation contract. Use finding
  U1 as rationale and guardrail.
- **D-21:** Do not use Python `hash()` or raw default `str()`/repr as persistent
  key material.
- **D-22:** Large tuples must be deterministic.
- **D-23:** Process-stable scalar hash use is allowed only for the types listed
  in TASK-4.
- **D-24:** Default memory-address repr must become a stable low-quality marker
  with a warning.
- **D-25:** The TASK-4 compatibility note is mandatory in the commit message and
  CHANGELOG.

### Test Packaging and Verification

- **D-26:** Use TASK acceptance commands plus `.planning/codebase/TESTING.md`.
  The planner chooses exact test files after reading current tests.
- **D-27:** Every folded task with a "Verify first" step must run it before
  implementation. If actual behavior differs from the doc, stop and report
  instead of improvising.
- **D-28:** Acceptance tests lock in each fix. Storage-mode-touching work must
  include `tests/test_storage_mode.py` in Tier-1.
- **D-29:** TASK-4 requires the cross-subprocess `PYTHONHASHSEED` regression.
- **D-30:** Property-based cache-key stress testing is folded into Phase 28 and
  should follow the existing Hypothesis patterns in the test suite.

### Compatibility and Release Notes

- **D-31:** TASK-4 explicitly requires a commit message and CHANGELOG note
  because affected cache keys change.
- **D-32:** TASK-3 requires the commit message to mention the new persistence
  contract if existing tests asserting swallow behavior are updated.
- **D-33:** Other tasks require atomic commit messages. Add docs/CHANGELOG only
  when the implementation creates user-visible behavior beyond the task docs.

### Folded Todos

- **Fix clear_all() not deleting blob files** - Folded into TASK-1 / REL-01.
- **Fix write-intent journal path resolution and safety checks** - Folded into
  TASK-2 / REL-02 through REL-04.
- **JSON backend - propagate save failures, preserve corrupt files** - Folded
  into TASK-3 / REL-05 through REL-06.
- **Stabilize cache keys - remove unstable hash()/str() fallbacks** - Folded
  into TASK-4 / KEY-01.
- **Property-based stress testing for cache key serialization** - Folded into
  KEY-02.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Scope

- `.planning/PROJECT.md` - v0.12.0 milestone boundaries and project
  constraints.
- `.planning/REQUIREMENTS.md` - Phase 28 requirements REL-01 through REL-06 and
  KEY-01 through KEY-02.
- `.planning/ROADMAP.md` - Phase 28 goal and success criteria.
- `.planning/STATE.md` - Current milestone state and pending todo summary.

### Review Docs

- `docs/CODE_REVIEW_FINDINGS.md` - Findings R1, R2, R3, R4, R8, R17, U1, and
  Wave 1 recommended fix order.
- `docs/CODE_REVIEW_ACTIONS.md` - TASK-1 through TASK-4, global rules,
  verify-first steps, acceptance criteria, test commands, and compatibility
  notes.

### Folded Todos and Seed

- `.planning/todos/pending/2026-06-12-fix-clear-all-not-deleting-blob-files.md`
  - TASK-1 todo.
- `.planning/todos/pending/2026-06-12-fix-write-intent-journal-path-resolution.md`
  - TASK-2 todo.
- `.planning/todos/pending/2026-06-12-json-backend-propagate-save-failures.md`
  - TASK-3 todo.
- `.planning/todos/pending/2026-06-12-stabilize-cache-keys-remove-unstable-hash-fallbacks.md`
  - TASK-4 todo.
- `.planning/todos/pending/2026-04-03-property-based-stress-testing-for-cache-key-serialization.md`
  - KEY-02 property-testing todo.
- `.planning/seeds/SEED-006-record-write-intent-before-blob-write.md` - Companion
  follow-up slice for TASK-2.

### Codebase Maps

- `.planning/codebase/ARCHITECTURE.md` - Layers and integration points.
- `.planning/codebase/TESTING.md` - Existing test organization and tiered test
  strategy.
- `.planning/codebase/CONCERNS.md` - Existing known concerns and property-test
  gap context.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `src/cacheness/storage/blob_store.py` - `_clear_blob_files()` and BlobStore
  cleanup path.
- `src/cacheness/core.py` - `UnifiedCache.__init__`,
  `_cleanup_stale_intents()`, `put()`, `clear_all()`,
  `clear_all_namespaces()`, and cache-key creation.
- `src/cacheness/write_intent.py` - `WriteIntentJournal` intent recording and
  cleanup.
- `src/cacheness/_storage_mode_mixin.py` - storage-mode write path and storage
  invariant.
- `src/cacheness/metadata/json_backend.py` - JSON load/save behavior.
- `src/cacheness/serialization.py` - cache-key serialization fallbacks.

### Established Patterns

- Verify-first, then fix, then targeted tests.
- Use existing test files and source-to-test mapping rather than phase-specific
  test islands unless the planner finds a clear reason.
- Use `uv` only for Python/test commands.
- Always ignore `tests/test_tensorflow_handler.py` on Windows.
- Preserve storage-mode durability semantics.

### Integration Points

- `clear_all_namespaces()` currently calls `_clear_blob_files()` directly, so
  TASK-1 must confirm namespace directories are cleared there too.
- `UnifiedCache.__init__` must run stale-intent cleanup independently of
  expired-entry cleanup.
- `cleanup_stale_intents()` needs cache-dir-relative path resolution and an
  entry-exists preservation guard.
- JSON backend mutators must distinguish data-critical writes from best-effort
  telemetry writes.
- Cache-key changes must be validated across subprocesses with different
  `PYTHONHASHSEED` values.

</code_context>

<specifics>
## Specific Ideas

No freeform implementation preferences were added. The user repeatedly asked to
source decisions from `CODE_REVIEW_FINDINGS.md` and `CODE_REVIEW_ACTIONS.md`.

</specifics>

<deferred>
## Deferred Ideas

- Tiered pull-through cache remains future scope. Phase 28 handles reliability
  prerequisites only.

### Reviewed Todos (not folded)

- **Tiered pull-through cache** - Reviewed by todo matching but not folded. It
  depends on v0.12.0 reliability/parity prerequisites and belongs in a later
  feature phase.

</deferred>

---

*Phase: 28-Silent Data-Loss Remediation*
*Context gathered: 2026-06-12*
