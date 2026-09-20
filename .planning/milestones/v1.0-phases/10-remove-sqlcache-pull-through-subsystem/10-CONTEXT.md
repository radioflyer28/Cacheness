# Phase 10: Remove SqlCache Pull-Through Subsystem - Context

**Gathered:** 2026-09-17
**Status:** Ready for planning

<domain>
## Phase Boundary

Remove the unrelated table/range-oriented `SqlCache` product before the first
supported Cacheness release. The phase deletes its runtime, public exports,
dedicated tests/examples/docs, DuckDB surface, and now-unused dependencies while
preserving the supported `BlobStore` foundation, `UnifiedCache` policy layer,
SQL-backed lifecycle authorities/catalog metadata, dataframe format handlers,
and future schema/format migration tooling.

This is direct pre-production removal, not deprecation, compatibility work, a
replacement SQL-query cache, or a redesign of storage lifecycle authority.

</domain>

<decisions>
## Implementation Decisions

### Removed import and data behavior
- **D-01:** Delete `cacheness.sql_cache`, its top-level exports, aliases, and every compatibility hook. Old imports fail naturally with `ModuleNotFoundError` or `ImportError`; do not retain a tombstone module, `__getattr__` shim, or tailored runtime exception. — **Reversibility:** costly — undoing this after the first supported release would reintroduce a separate public product and maintenance contract.
- **D-02:** Existing canonical guidance explains alternatives by use case: `UnifiedCache` for object/function caching and `BlobStore` for object persistence. It must explicitly state that Cacheness has no in-package replacement for range-aware SQL pull-through.
- **D-03:** Caller-owned development databases and tables are left untouched and unsupported. Phase 10 must not discover, delete, mutate, export, or migrate them, and must not retain implementation solely to provide cleanup tooling.
- **D-04:** Phase 10 does not change the package version. Version selection and publication remain release-process work.

### Dependencies and extras
- **D-05:** Delete the `sql` optional extra rather than repurposing it or retaining an empty compatibility extra.
- **D-06:** Remove every DuckDB product dependency and claim: manifest and lockfile entries, runtime branches, tests, examples, docs, packaging matrices, and CI references. DuckDB is not retained as a development-only dependency.
- **D-07:** Do not redesign the remaining SQLAlchemy/PostgreSQL dependency organization. Remove only entries proven exclusive to `SqlCache`; preserve dependencies required by lifecycle authorities, catalog metadata, format handlers, and supported backends.
- **D-08:** Prune every additional manifest/lock dependency whose last supported use disappears with the deleted subsystem. Retain anything still used by `BlobStore`, `UnifiedCache`, handlers, authorities, or development tooling.

### Documentation and history
- **D-09:** Delete dedicated SqlCache documents, examples, and tests outright. Do not archive them inside the repository or leave unsupported files with warning banners.
- **D-10:** Preserve truthful historical planning artifacts, completed phase records, and dated audits. Update a historical document only if it falsely presents `SqlCache` as part of the current post-Phase-10 product; do not rewrite history merely to remove the name.
- **D-11:** Surgically remove SqlCache-specific material from mixed current-facing documents while retaining useful content about pandas/dataframes, handlers, or supported storage/cache behavior.
- **D-12:** Record the cutover as a concise note in existing canonical API/migration guidance and the docs index. Do not create a standalone long-lived SqlCache removal guide.

### Regression and package contracts
- **D-13:** Tests must prove complete absence: no top-level export, package attribute, source module, importable `cacheness.sql_cache`, compatibility hook/tombstone, or wheel member.
- **D-14:** Scan current-facing source, tests, examples, package metadata, README, and documentation for dangling references. Use an explicit narrow allowlist for the concise removal note and truthful historical planning/dated-audit references; do not impose repository-wide historical erasure.
- **D-15:** Rewrite useful older public-surface, suite-isolation, lint, and packaging contracts to enforce the new boundary. Do not delete or exclude a mixed test merely because it previously asserted SqlCache behavior.
- **D-16:** Final acceptance includes a freshly built and isolated wheel: supported `BlobStore`/`UnifiedCache` imports and representative local round trips pass, SqlCache and DuckDB are absent, and installed dependency metadata matches the pruned manifest.

### the agent's Discretion
- Exact task slicing and ordering of source deletion, contract inversion, documentation cleanup, and dependency lock refresh.
- Exact wording and location within the existing canonical API/migration pages, provided the note remains concise and follows D-02/D-12.
- The explicit reference allowlist structure and test helpers, provided historical artifacts remain preserved and all current-facing surfaces are covered.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope and product boundary
- `.planning/ROADMAP.md` — Phase 10 goal, success criteria, and fixed boundary.
- `.planning/REQUIREMENTS.md` — `CACH-07` removal requirement and protected remaining capabilities.
- `.planning/PROJECT.md` — pre-production cutover, one-engine product definition, and explicit decision to remove rather than merge SqlCache.
- `.planning/phases/09-adoption-and-release-surface-closure/09-CONTEXT.md` — current published product surface and Phase 10 ownership carried forward from Phase 9.

### Architectural guardrails
- `docs/adr/0001-topology-specific-storage-guarantees.md` — must not be weakened or broadened while removing an unrelated subsystem.
- `docs/API_REFERENCE.md` — canonical supported API guidance to update with the bounded removal note.
- `docs/STORAGE_MIGRATION.md` — canonical migration/rebuild guidance; removal wording must not imply SqlCache table migration support.
- `docs/README.md` — canonical documentation index from which obsolete SqlCache pages/links must disappear.

### Package and mixed-document surfaces
- `pyproject.toml` — current `duckdb-engine` and `sql` extra declarations plus remaining SQL dependency groups.
- `src/cacheness/__init__.py` — canonical public export surface.
- `docs/PANDAS_API_AUDIT.md` — mixed current-facing document requiring surgical cleanup rather than wholesale deletion.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 9 wheel and public-API contracts already exercise source-free package imports and canonical local journeys; extend them to assert the removed module and dependencies are absent.
- Existing documentation-contract tests already scan promoted guidance and link ownership; reuse their explicit-inventory/allowlist style for dangling-reference checks.
- `uv` manifest/lock workflows provide the supported way to regenerate and prove dependency removal.

### Established Patterns
- Public symbols are enumerated explicitly in `src/cacheness/__init__.py`; Phase 10 removes `SqlCache` and `SqlCacheAdapter` directly from imports and `__all__`.
- Pre-production compatibility reset means direct deletion is preferred over aliases, deprecation shims, or tombstones.
- Historical planning evidence remains truthful history, while current-facing docs and executable examples must describe only supported behavior.
- Supported package proofs use fresh isolated wheels rather than relying only on source-tree imports.

### Integration Points
- `src/cacheness/sql_cache.py` is the standalone implementation to delete; it also owns its adapters, builders, result/failure types, and `SQLAlchemyDataAdapter` alias.
- `src/cacheness/error_handling.py` contains SqlCache-only error codes that must be classified and removed if no remaining caller uses them.
- `pyproject.toml` currently declares `duckdb-engine` in base and optional groups and defines the `sql` extra; `uv.lock` must converge after manifest cleanup.
- Dedicated tests include `tests/test_sql_cache.py`, `tests/test_sql_cache_failure_contract.py`, and `tests/test_sql_cache_documentation.py`; mixed Phase 1/6/public API contracts must be inverted or repaired rather than blindly deleted.
- Dedicated examples include `beginner_sql_cache.py`, `simple_stock_cache.py`, and `stock_cache_example.py`; additional mixed examples and current docs require reference-level classification.
- Remaining SQLAlchemy/PostgreSQL code supports lifecycle authorities, metadata/catalog behavior, and optional PostgreSQL topology and must survive the removal.

</code_context>

<specifics>
## Specific Ideas

- A stale `from cacheness import SqlCache` should fail because the symbol is absent, and `import cacheness.sql_cache` should fail because no module ships.
- The removal note must avoid falsely claiming that `UnifiedCache` reproduces range-gap SQL caching.
- Wheel inspection must verify both file membership and installed dependency metadata, not merely import behavior.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 10-remove-sqlcache-pull-through-subsystem*
*Context gathered: 2026-09-17*
