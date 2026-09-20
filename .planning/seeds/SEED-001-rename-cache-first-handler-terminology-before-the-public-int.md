---
id: SEED-001
status: fulfilled
planted: 2026-09-09
planted_during: v1.0 / Phase 7
fulfilled: 2026-09-20
fulfilled_during: v1.0 / Phase 9 — Adoption and Release Surface Closure
---

# SEED-001: Rename cache-first handler terminology before the public interface freezes

## Resolution

**Fulfilled in Phase 9 Plan 01.** The public handler protocol and error base
were renamed to `FormatHandler` and `FormatHandlerError` without compatibility
aliases. The store-local registration API remains
`store.handlers.register_handler(...)`; stable stored handler identities were
not changed. See
[09-01-SUMMARY.md](../milestones/v1.0-phases/09-adoption-and-release-surface-closure/09-01-SUMMARY.md).

## Why This Matters

Cache-first terminology was misleading once `BlobStore` became the storage
foundation shared with cache-policy instances.

## When to Surface

No future trigger: the terminology cutover is complete.

## Scope Estimate

Completed in Phase 9 Plan 01.

## Breadcrumbs

- `CONTEXT.md` — Defines BlobStore as the storage engine and cache instances as policy consumers.
- `src/cacheness/interfaces.py` — Declares the current cache-named payload handler interface.
- `src/cacheness/handlers.py` — Implements the current handler registry and payload-contract resolution.
- `src/cacheness/storage/__init__.py` — Exposes `CacheHandler` and `HandlerRegistry` from the BlobStore-oriented package.
- `.planning/milestones/v1.0-phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md` — Treats handler-owned payload formats as independently versioned migration contracts.

## Notes

Retained as historical context, not promotable future work.
