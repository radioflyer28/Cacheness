---
id: SEED-001
status: dormant
planted: 2026-09-09
planted_during: v1.0 / Phase 7
trigger_when: when relevant
scope: unknown
---

# SEED-001: Rename cache-first handler terminology before the public interface freezes

## Why This Matters

_To be filled in. Run `$gsd-capture --seed --enrich SEED-001` to add context._

## When to Surface

**Trigger:** when relevant

This seed will surface during `$gsd-new-milestone` when the milestone scope matches.

## Scope Estimate

**Unknown** — run `$gsd-capture --seed --enrich SEED-001` to estimate effort.

## Breadcrumbs

- `CONTEXT.md` — Defines BlobStore as the storage engine and cache instances as policy consumers.
- `src/cacheness/interfaces.py` — Declares the current cache-named payload handler interface.
- `src/cacheness/handlers.py` — Implements the current handler registry and payload-contract resolution.
- `src/cacheness/storage/__init__.py` — Exposes `CacheHandler` and `HandlerRegistry` from the BlobStore-oriented package.
- `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md` — Treats handler-owned payload formats as independently versioned migration contracts.

## Notes

_Captured via one-shot seed capture. Enrich with trigger, why, and scope at your convenience._
