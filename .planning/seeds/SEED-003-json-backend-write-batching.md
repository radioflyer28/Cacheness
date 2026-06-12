---
id: SEED-003
status: dormant
planted: 2026-06-12
planted_during: v0.11.0 complete / pre-v0.12 planning
trigger_when: performance-focused milestone, or when JSON backend write amplification becomes a user complaint
scope: medium
---

# SEED-003: JSON backend write batching/debouncing

## Why This Matters

Every `get()` on the JSON backend rewrites the entire metadata file **twice** (`update_access_time` + `increment_hits` each trigger a full-file save) — read-heavy workloads degrade and churn the disk beyond the already-documented O(n²) write scaling. Code review finding **U5**. Deferred because the fix is a perf design decision: dirty-flag + flush-on-close/interval vs. in-memory-only access times with a documented caveat — each changes durability semantics.

## When to Surface

**Trigger:** performance-focused milestone, or JSON-backend perf complaints.

## Scope Estimate

**Medium** — needs a decided flush policy, crash-semantics documentation, and careful interaction with TASK-3's new raise-on-error contract for data-critical writes.

## Breadcrumbs

- docs/CODE_REVIEW_FINDINGS.md U5
- src/cacheness/metadata/json_backend.py (`_save_to_disk`, `update_access_time`, `increment_hits`)
- .planning/codebase/CONCERNS.md (JSON O(n²) writes — known)

## Notes

Must land AFTER TASK-3 (raise-on-error for put/remove) so the batching layer doesn't reintroduce silent loss for data-critical writes.
