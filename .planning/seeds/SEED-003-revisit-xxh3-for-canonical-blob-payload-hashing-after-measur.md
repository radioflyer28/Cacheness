---
id: SEED-003
status: dormant
planted: 2026-09-12
planted_during: Phase 07.1 — Obstore Payload Participant Unification
trigger_when: when relevant
scope: unknown
---

# SEED-003: Revisit XXH3 for canonical blob payload hashing after measuring SHA-256 cost and integrity tradeoffs

## Why This Matters

_To be filled in. Run `$gsd-capture --seed --enrich SEED-003` to add context._

## When to Surface

**Trigger:** when relevant

This seed will surface during `$gsd-new-milestone` when the milestone scope matches.

## Scope Estimate

**Unknown** — run `$gsd-capture --seed --enrich SEED-003` to estimate effort.

## Breadcrumbs

- `src/cacheness/storage/manifest.py` — current persisted payload digest algorithm is SHA-256.
- `src/cacheness/storage/integrity.py` — canonical payload digest/size implementation.
- `src/cacheness/serialization.py` and `src/cacheness/file_hashing.py` — existing XXH3 uses for cache keys and file fingerprints.
- `.planning/phases/05-payload-backends-and-supported-topology-qualification/05-CONTEXT.md` — decision D-10 makes SHA-256 plus byte size canonical and treats ETags as corroborating evidence only.

## Notes

_Captured via one-shot seed capture. Enrich with trigger, why, and scope at your convenience._
