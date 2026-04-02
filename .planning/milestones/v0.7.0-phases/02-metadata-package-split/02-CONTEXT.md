# Phase 2: Metadata Package Split - Context

**Gathered:** 2026-04-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Split `src/cacheness/metadata.py` (3,046 lines) into a `src/cacheness/metadata/` package with per-backend files and shared modules. Preserve all existing import paths. PostgresBackend stays in `storage/backends/` — this split covers only the contents of `metadata.py` itself.

</domain>

<decisions>
## Implementation Decisions

### Module Structure
- **D-01:** Mirror the Phase 1 `_compat.py` pattern — shared imports, ORM models, namespace utilities, and optional dependency detection all go in `_compat.py`
- **D-02:** Package structure:
  - `_compat.py` — shared imports, SQLAlchemy `Base`, `CacheEntry`, `CacheEntryMixin`, `CacheStatsMixin`, `CacheStats`, namespace model factory (`_get_namespace_models`), `NamespaceInfo`, `validate_namespace_id`, `DEFAULT_NAMESPACE`, `NAMESPACE_ID_PATTERN`, `SQLALCHEMY_AVAILABLE`
  - `base.py` — `MetadataBackend` ABC + `CachedMetadataBackend` wrapper
  - `json_backend.py` — `JsonBackend`
  - `sqlite_backend.py` — `SqliteBackend`
  - `__init__.py` — re-exports all public names + `create_metadata_backend()` factory

### PostgresBackend Location
- **D-03:** PostgresBackend stays in `storage/backends/postgresql_backend.py`. It already works, no unnecessary churn.

### Agent's Discretion
- CachedMetadataBackend placement (with base.py vs own file) — agent decides based on cohesion
- Import organization within _compat.py — agent decides grouping

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Source Code
- `src/cacheness/metadata.py` — The file being split (3,046 lines, 5 classes + factory)
- `src/cacheness/__init__.py` — Re-exports metadata names (lines 35+, 83+)
- `src/cacheness/core.py` — Imports `DEFAULT_NAMESPACE`, `create_metadata_backend`, `Base` (lines 22, 195, 455)
- `src/cacheness/custom_metadata.py` — Heavy consumer of `Base`, `SQLALCHEMY_AVAILABLE`, `CacheEntry`, `DEFAULT_NAMESPACE`, `_get_namespace_models`
- `src/cacheness/config.py` — Imports `validate_namespace_id`, `DEFAULT_NAMESPACE` (line 14)
- `src/cacheness/storage/backends/base.py` — Imports `MetadataBackend` (line 9, 18)
- `src/cacheness/storage/backends/__init__.py` — Imports `JsonBackend`, `create_metadata_backend`, `SqliteBackend` (lines 65, 69)

### Prior Phase Artifacts
- `.planning/phases/01-handler-package-split/01-01-SUMMARY.md` — Established `_compat.py` pattern and conditional re-export approach

### Project References
- `.planning/REQUIREMENTS.md` — DECO-02 requirement definition
- `docs/BACKEND_SELECTION.md` — Backend comparison and selection guide

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 1 `handlers/_compat.py` pattern: shared imports, optional dep detection, availability flags — directly applicable
- Phase 1 `handlers/__init__.py` pattern: conditional try/except re-exports — directly applicable

### Established Patterns
- `_compat.py` aggregates shared imports to avoid duplication across sibling modules
- `__init__.py` re-exports all public names for backward compatibility
- Availability flags (`SQLALCHEMY_AVAILABLE`) as module-level constants
- Rename-then-create-directory workflow for replacing .py with package/

### Integration Points
- 18 downstream import sites reference `from .metadata import ...` or `from cacheness.metadata import ...`
- `custom_metadata.py` is the heaviest consumer (7 import sites with `Base`, `CacheEntry`, `DEFAULT_NAMESPACE`, `_get_namespace_models`)
- `storage/backends/base.py` and `storage/backends/__init__.py` import `MetadataBackend`, `JsonBackend`, `create_metadata_backend`
- `core.py` imports `DEFAULT_NAMESPACE`, `create_metadata_backend`, `SQLALCHEMY_AVAILABLE`, `Base`
- `config.py` imports `validate_namespace_id`, `DEFAULT_NAMESPACE`

</code_context>

<specifics>
## Specific Ideas

- Follow the exact same workflow as Phase 1: rename metadata.py → _metadata_legacy.py, create metadata/ directory, create files, verify imports, delete legacy file
- The `_compat.py` should be the "big" module here since it holds ORM models — this is expected and keeps SQLAlchemy isolated

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-metadata-package-split*
*Context gathered: 2026-04-02*
