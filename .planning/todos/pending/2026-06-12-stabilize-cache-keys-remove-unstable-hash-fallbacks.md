---
created: 2026-06-12T15:45:54.188Z
title: Stabilize cache keys - remove unstable hash()/str() fallbacks
area: general
resolves_phase: 28
files:
  - src/cacheness/serialization.py:330
  - src/cacheness/serialization.py:363
---

## Problem

Code review finding **U1** (🔴 HIGH — silent permanent cache misses): `_serialize_with_config()` fallback 5 uses `hash(obj)`, which is PYTHONHASHSEED-randomized for anything containing strings (e.g. tuples longer than `max_tuple_recursive_length=10`, frozensets) and id-based for default objects; fallback 6 uses `str(obj)` whose default repr embeds memory addresses. Either fallback ⇒ persistent disk-cache keys change every process run: 100% miss rate for affected parameters plus unbounded growth of unreachable entries. Completely silent.

## Solution

Execute **TASK-4** in `docs/CODE_REVIEW_ACTIONS.md` — xxhash recursively-serialized elements for large tuples, restrict `hash()` to process-stable types, sanitize default-repr strings, warn on unstable fallbacks. Note compatibility impact (keys change for affected objects) in commit + CHANGELOG. Acceptance includes cross-subprocess key-equality test with differing PYTHONHASHSEED.

Related: existing todo "Property-based stress testing for cache key serialization" (2026-04-03) — that testing work would lock in this fix; consider doing them together.
