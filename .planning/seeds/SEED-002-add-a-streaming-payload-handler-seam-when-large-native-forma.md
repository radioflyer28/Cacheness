---
id: SEED-002
status: dormant
planted: 2026-09-09
planted_during: v1.0 / Phase 7
trigger_when: when relevant
scope: unknown
audit_acknowledged:
  milestone: v1.0
  at: 2026-09-20
  status: dormant
---

# SEED-002: Add a streaming payload-handler seam when large native formats justify it

## Why This Matters

_To be filled in. Run `$gsd-capture --seed --enrich SEED-002` to add context._

## When to Surface

**Trigger:** when relevant

This seed will surface during `$gsd-new-milestone` when the milestone scope matches.

## Scope Estimate

**Unknown** — run `$gsd-capture --seed --enrich SEED-002` to estimate effort.

## Breadcrumbs

- `src/cacheness/interfaces.py` — Defines the current path-based guarded write and private read-snapshot interface.
- `src/cacheness/storage/blob_store.py` — Owns payload lifecycle and invokes handlers only through managed storage.
- `src/cacheness/storage/backends/blob_backends.py` — Contains blob-backend streaming capabilities that are not yet a handler-level interface.
- `benchmarks/lifecycle_authority_benchmark.py` — Existing location for evidence-driven lifecycle performance work.
- `.planning/milestones/v1.0-ROADMAP.md` — Archived Phase 8 requires measured, bounded aggregate behavior rather than speculative optimization.

## Notes

_Captured via one-shot seed capture. Enrich with trigger, why, and scope at your convenience._
