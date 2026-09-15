---
id: SEED-006
status: dormant
planted: 2026-09-15
planted_during: v1.0 Phase 08 — Production Gates and Performance Stabilization
trigger_when: when relevant
scope: unknown
---

# SEED-006: Qualify performance on a controlled Linux runner

## Why This Matters

_To be filled in. Run `$gsd-capture --seed --enrich SEED-006` to add context._

## When to Surface

**Trigger:** when relevant

This seed will surface during `$gsd-new-milestone` when the milestone scope matches.

## Scope Estimate

**Unknown** — run `$gsd-capture --seed --enrich SEED-006` to estimate effort.

## Breadcrumbs

- `benchmarks/phase8_benchmarks.py` — controlled-runner preflight and benchmark harness
- `benchmarks/phase8_workloads.py` — canonical performance workload inventory
- `.github/workflows/performance.yml` — dedicated Linux performance workflow contract
- `.planning/phases/08-production-gates-and-performance-stabilization/08-11-PLAN.md` — original release-blocking baseline capture plan
- `docs/RELEASE_QUALIFICATION.md` — current qualification claims and nonclaims

## Notes

Deferred from Phase 8 because no eligible `cacheness-perf-linux-x64` runner is
available. macOS measurements may remain useful diagnostic evidence, but they do
not establish Linux equivalence or a cross-platform performance guarantee.

