---
id: SEED-008
status: dormant
planted: 2026-09-17
planted_during: v1.0 Phase 09 — Adoption and Release Surface Closure
trigger_when: when a future dataframe format/developer-kit milestone is selected
scope: medium
---

# SEED-008: Investigate Narwhals for dataframe format-handler compatibility

## Why This Matters

Phase 8 decision D-08 retained the existing optional dataframe behavior: the
`dataframes` extra qualifies the current pandas, PyArrow, and Polars ecosystem
around the existing Parquet handlers. The Phase 8 context deferred a Narwhals
investigation rather than treating it as a new handler feature or package
decision.

A future investigation can determine whether Narwhals offers a useful, narrow
compatibility seam across pandas, PyArrow, and Polars. Parquet remains handler-owned:
native `FormatHandler` implementations retain their stable
data and payload identities, select native serialization, and keep existing
round trips authoritative. A compatibility layer must not turn all dataframe
values into one synthetic identity or make a library choice appear to be a
storage-lifecycle guarantee.

## When to Surface

**Trigger:** when a future dataframe format/developer-kit milestone needs a
cross-library compatibility decision, a PyArrow-facing user workflow, or a
reusable handler-extension policy.

## Decision Questions

1. Can Narwhals provide a small optional compatibility seam for pandas,
   PyArrow, and Polars without changing the current `pandas_dataframe`,
   `pandas_series`, `polars_dataframe`, `polars_series`, or `parquet` payload
   identities?
2. Does its supported version range align with Cacheness's `dataframes` extra
   and quiet base-import / focused optional-dependency policy?
3. What public contract would safely distinguish a native object round trip
   from an explicitly requested conversion, especially for PyArrow tables and
   mixed pandas/Polars workflows?
4. Does its benefit justify an optional dependency and the maintenance burden,
   or should the existing focused native handlers remain the whole supported
   surface?

## Acceptance Evidence

- A short, reproducible comparison records supported pandas, PyArrow, and
  Polars versions, conversion behavior, error behavior, and memory boundaries.
- Isolated `dataframes`-extra tests prove base imports remain quiet and native
  pandas/Polars Parquet round trips retain their exact current identities.
- If a compatibility API is proposed, tests distinguish native round trips
  from explicit conversions and preserve safe suffix/path containment through
  the existing handler boundary.
- The resulting proposal states whether Narwhals is adopted, rejected, or kept
  deferred, with migration/format effects explicitly recorded before any
  dependency change.

## Explicit Non-Goals

- Do not install Narwhals or add it to `pyproject.toml` or `uv.lock` from this
  seed.
- Do not implement an adapter, modify retained dataframe handlers, or change
  Parquet/native payload identities during this investigation capture.
- Do not add a new lifecycle authority, backend topology claim, or remote
  qualification claim.
- Reusable conformance tooling, multiple third-party format examples, and a
  broad extension policy remain future developer-kit work rather than this
  seed's implied current implementation.

## Breadcrumbs

- `src/cacheness/handlers.py` — retained pandas/Polars `FormatHandler`
  implementations and their `parquet` payload format.
- `pyproject.toml` — the optional `dataframes` extra.
- `.planning/phases/08-production-gates-and-performance-stabilization/08-CONTEXT.md`
  — D-08 and the deferred Narwhals investigation.
- `.planning/phases/08-production-gates-and-performance-stabilization/08-VERIFICATION.md`
  — qualification of retained behavior, not a dataframe-handler redesign.
- `.planning/phases/09-adoption-and-release-surface-closure/09-CONTEXT.md`
  — Phase 9's minimum format-handler surface and deferred developer-kit scope.

## Notes

This is a dormant investigation, not an adoption decision. It deliberately
does not alter Cacheness's current BlobStore lifecycle, cache-policy layering,
or Phase 8 local-readiness scope.
