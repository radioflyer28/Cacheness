---
created: 2026-04-03T20:47:59.531Z
title: Property-based stress testing for cache key serialization
area: testing
files:
  - src/cacheness/serialization.py
  - src/cacheness/core.py
  - src/cacheness/decorators.py
---

## Problem

Cache key generation (`create_unified_cache_key` in `serialization.py`, `_generate_cache_key` in `decorators.py`) is a critical correctness boundary — if two different inputs produce the same key, data is silently corrupted. If the same input produces different keys across Python versions or dependency upgrades (NumPy, Pandas, Polars), cache misses silently invalidate data.

Current tests are example-based: they check known inputs against expected outputs. This misses edge cases in:
- **Argument serialization:** Complex nested objects, NaN handling, dtype coercion, empty containers, unicode, very large/small numbers
- **Cross-version stability:** Python 3.12 → 3.13 → 3.14 may change `repr()`, `hash()`, or pickle protocols
- **Dependency drift:** NumPy/Pandas/Polars version upgrades may change how objects serialize (e.g., NumPy 2.0 changed dtype repr)
- **Collision resistance:** No systematic check that distinct inputs produce distinct keys
- **Roundtrip consistency:** Key generated pre-put must match key generated pre-get for the same logical arguments

These are exactly the class of bugs that property-based testing / fuzzing / Hypothesis excels at finding — silent, data-corrupting edge cases that example tests miss.

## Solution

Build a Hypothesis-based property test suite for cache key derivation:

1. **Hypothesis strategies** for cache key inputs:
   - Arbitrary Python primitives (int, float, str, bytes, bool, None)
   - Containers (list, dict, tuple, set) with nested structures
   - NumPy arrays with varying dtypes, shapes, NaN/inf values
   - Pandas DataFrames/Series with edge-case dtypes (categorical, nullable int, timezone-aware datetime)
   - Polars DataFrames/Series
   - Mixed-type argument combinations matching `@cached` decorator signatures

2. **Properties to verify:**
   - **Determinism:** Same input → same key (always)
   - **Collision resistance:** Different inputs → different keys (with high probability)
   - **Stability across calls:** Key generated twice in same process is identical
   - **Roundtrip:** `put(data, **kwargs)` then `get(**kwargs)` retrieves the same data
   - **Prefix stripping:** `prefix`, `description`, `custom_metadata`, `ttl_seconds` params are excluded from key hash
   - **Argument order independence:** `f(a=1, b=2)` and `f(b=2, a=1)` produce the same key

3. **Regression oracle:** Snapshot known key outputs for specific inputs, detect when upgrades change them. This catches Python/NumPy/Pandas version drift before it silently invalidates caches.

4. **Integration with CI:** Run Hypothesis with a small example database in CI (fast), with a larger database for nightly/pre-release runs.

**Inspiration:** [Antithesis property-based testing concepts](https://antithesis.com/docs/resources/property_based_testing/) — "sometimes" properties, deterministic simulation, autonomous fault injection.

**Dependencies:** `hypothesis` (already popular in the Python ecosystem, MIT licensed). Consider `hypothesis[numpy]` and `hypothesis[pandas]` extras for typed array/dataframe strategies.
