# Phase 6: UnifiedCache Policy Composition - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution
> agents. Decisions are captured in CONTEXT.md; this log preserves the
> alternatives considered.

**Date:** 2026-09-08
**Phase:** 6-unifiedcache-policy-composition
**Areas discussed:** Public cache surface, Presence and outcome semantics,
Invalidation and eviction policy, Decorator ownership and cleanup

---

## Public Cache Surface

| Option | Description | Selected |
|--------|-------------|----------|
| One explicit constructor and decorator | Keep `UnifiedCache`, `CacheConfig`, and `cached`; remove redundant pre-production aliases and globals | ✓ |
| Keep legacy aliases as wrappers | Route old names through the new surface | |
| Retain the current broad surface | Preserve constructors, aliases, globals, and decorator variants | |

**Choice:** One explicit constructor and decorator (recommended default).
**Notes:** The project is pre-production, so compatibility wrappers would
preserve the overlapping ownership surface this milestone is removing.

---

## Presence and Outcome Semantics

| Option | Description | Selected |
|--------|-------------|----------|
| Typed lookup result with shared outcomes | Represent presence independently of value and reuse one outcome vocabulary for statistics | ✓ |
| Sentinel-only private handling | Fix cached `None` internally but keep public diagnostics fragmented | |
| Continue overloading `None` | Treat cached `None` as indistinguishable from absence | |

**Choice:** Typed lookup result with shared outcomes (recommended default).
**Notes:** `None` is a valid application value. Conflict and backend errors
remain typed rather than collapsing into ordinary absence.

---

## Invalidation and Eviction Policy

| Option | Description | Selected |
|--------|-------------|----------|
| Authority-backed bounded policy | Select from authoritative facts and delete exact generations through BlobStore | ✓ |
| Projection-driven LRU | Let derived access metadata independently choose deletions | |
| Separate cache cleanup engine | Duplicate cleanup/lifecycle sequencing in UnifiedCache | |

**Choice:** Authority-backed bounded policy (recommended default).
**Notes:** Prefer deterministic oldest-entry eviction if exact LRU would require
a write on every read or a second authoritative projection.

---

## Decorator Ownership and Cleanup

| Option | Description | Selected |
|--------|-------------|----------|
| Explicit cache binding | `cached` receives an application-owned cache and returns actual lifecycle clear results | ✓ |
| Implicit process-global cache | Decorators acquire mutable module-global ownership | |
| Per-call cache construction | Each call creates and closes a separate cache | |

**Choice:** Explicit cache binding (recommended default).
**Notes:** Decorator lookup consumes the same presence result once; cached
`None` is returned without recomputation.

## the agent's Discretion

- Exact class and enum names for lookup, statistics, outcomes, and removal
  reports.
- Exact bounded page/work defaults and internal module split.
- Exact authoritative cache-policy metadata field names.

## Deferred Ideas

- General cache-policy plugins and distributed invalidation remain v2 work.
- Migration/rebuild belongs to Phase 7; live-service and performance gates
  belong to Phase 8.
