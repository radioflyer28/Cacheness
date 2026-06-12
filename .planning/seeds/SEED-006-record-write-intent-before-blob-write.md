---
id: SEED-006
status: dormant
planted: 2026-06-12
planted_during: v0.11.0 complete / pre-v0.12 planning
trigger_when: when TASK-2 (write-intent fixes) is being executed or any crash-recovery rework
scope: small
---

# SEED-006: Record write intent BEFORE blob write (journal coverage gap)

## Why This Matters

In `put()`, `record_intent()` runs **after** `_write_blob()` completes — a crash during handler serialization/encryption/rename leaves an orphan blob the journal never knew about. The journal only covers the blob-complete→metadata-commit window. Code review finding **R8**. Deferred from TASK-2 because it interacts with the path-resolution and entry-exists fixes there — design together, not piecemeal.

## When to Surface

**Trigger:** when TASK-2 is being executed (same files), or any crash-recovery rework.

## Scope Estimate

**Small** — record intent with the *planned* path before invoking the handler, in both `core.put()` and `_storage_mode_put()`; adjust stale-cleanup expectations (intent may reference a never-created blob).

## Breadcrumbs

- docs/CODE_REVIEW_FINDINGS.md R8
- docs/CODE_REVIEW_ACTIONS.md TASK-2 (companion changes) + out-of-scope list
- src/cacheness/core.py:884 (record_intent call site)
- src/cacheness/_storage_mode_mixin.py:75

## Notes

If TASK-2's executor has budget, fold this in as a follow-up commit in the same worktree — the test scaffolding overlaps heavily.
