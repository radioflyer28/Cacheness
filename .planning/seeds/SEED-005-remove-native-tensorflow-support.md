---
id: SEED-005
status: dormant
planted: 2026-09-13
planted_during: v1.0 Phase 08 — Production Gates and Performance Stabilization
trigger_when: when relevant
scope: unknown
---

# SEED-005: Remove native TensorFlow support

## Why This Matters

_To be filled in. Run `$gsd-capture --seed --enrich SEED-005` to add context._

## When to Surface

**Trigger:** when relevant

This seed will surface during `$gsd-new-milestone` when the milestone scope matches.

## Scope Estimate

**Unknown** — run `$gsd-capture --seed --enrich SEED-005` to estimate effort.

## Breadcrumbs

- `src/cacheness/handlers.py` — current native TensorFlow handler implementation
- `src/cacheness/config.py` — current TensorFlow handler configuration surface
- `pyproject.toml` — current TensorFlow optional dependency groups
- `.planning/phases/08-production-gates-and-performance-stabilization/08-02-SUMMARY.md` — current packaging qualification history
- `.planning/phases/08-production-gates-and-performance-stabilization/08-03-SUMMARY.md` — current platform qualification history

## Notes

Captured during Phase 8 after deciding that removal is worthwhile but not part of
the current production-gates phase. Phase 8 should leave TensorFlow support
untouched and continue from its pre-removal plan chain.

