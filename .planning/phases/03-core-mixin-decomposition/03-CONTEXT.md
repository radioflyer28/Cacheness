# Phase 3: Core Mixin Decomposition - Context

**Gathered:** 2026-04-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Decompose `src/cacheness/core.py` (3,307 lines) into 4 focused mixin classes — verification, statistics, custom metadata, and storage mode — while preserving the single `UnifiedCache` class API and all import paths. Mixins are runtime behavior only; initialization and orchestration (put/get) stay in `UnifiedCache`. No new public API surface.

</domain>

<decisions>
## Implementation Decisions

### Mixin Granularity
- **D-01:** Create exactly 4 mixins per ROADMAP: VerificationMixin, StatsMixin, CustomMetadataMixin, StorageModeMixin
- **D-02:** Query operations, file operations, batch operations, convenience API, and update/delete operations stay in `UnifiedCache` — not extracted into mixins
- **D-03:** StatsMixin stays as its own mixin (~35 lines) for pattern consistency, not merged with Verification

### File Organization
- **D-04:** Flat files beside `core.py` in `src/cacheness/`: `_verification_mixin.py`, `_stats_mixin.py`, `_custom_metadata_mixin.py`, `_storage_mode_mixin.py`. No `core/` sub-package — `from cacheness.core import UnifiedCache` works unchanged without any `__init__.py` re-export layer

### Core Trunk Boundary
- **D-05:** `put()` and `get()` (the main orchestration methods) stay in `UnifiedCache`. Mixins provide leaf logic only — called via `self._verify_entry()`, `self._record_hit()`, etc.
- **D-06:** All `_init_*` helpers stay in core.py inside `UnifiedCache.__init__()`. Mixins are purely runtime behavior — no initialization, no `__init__` methods
- **D-07:** `_sign_current_namespace()` stays in core.py with the init helpers (called from `_init_entry_signer()`)
- **D-08:** Module-level helpers (`_PutCleanup` class, `_normalize_function_args` function) stay in core.py alongside the code that uses them

### Signing Method Ownership
- **D-09:** `_sign_entry_if_enabled()` and `_extract_signable_fields()` move to VerificationMixin — they form a coherent sign/verify concern despite being called from the write path (put)

### Mixin State Access
- **D-10:** Plain cooperative inheritance — mixins use `self.config`, `self.metadata_backend`, `self.signer`, etc. directly. No Protocol class, no ABC, no type annotations for mixin state. Standard Python mixin pattern
- **D-11:** Cross-mixin method dependencies via `self.*` are standard mixin behavior — no documentation or special handling needed. MRO guarantees resolution

### MRO Ordering
- **D-12:** Agent decides MRO order based on logical dependency (e.g., verification first since signing is used by storage mode). No naming conflicts between mixins

### DECO-04 Validation
- **D-13:** Trust the existing test suite (1,604 tests) for DECO-04 validation. No explicit import smoke test added. Since we're not creating a `core/` package, the only import path at stake (`from cacheness.core import UnifiedCache`) stays unchanged

### Import Style
- **D-14:** Mixin files use relative imports (`from .interfaces import ...`, `from .metadata import ...`) matching core.py's existing style

### Mixin Visibility
- **D-15:** Mixin files are private (underscore-prefixed filenames). Mixin classes are not exported from `cacheness/__init__.py`. Users interact only with `UnifiedCache`

### Post-Extraction Size
- **D-16:** ~2,400 lines remaining in core.py after extraction is acceptable. The goal is focused mixins for 4 concerns, not minimizing core.py line count. Further splits are a future milestone

### Init-Mixin Coupling
- **D-17:** Init methods in core.py setting up state consumed by mixin methods in other files is acceptable coupling — expected for cooperative mixins

### Circular Import Risk
- **D-18:** One-directional import flow (core.py imports mixin files; mixin files import from metadata/handlers/interfaces/config; no mixin imports from core.py or another mixin). No circular import risk

### Agent's Discretion
- Exact MRO ordering of the 4 mixins in the class declaration
- How to handle imports within each mixin file (which specific names to import)
- Whether to include module-level docstrings in mixin files

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Source Code
- `src/cacheness/core.py` — The file being decomposed (3,307 lines, UnifiedCache class + helpers)
- `src/cacheness/__init__.py` — Re-exports UnifiedCache and other public names
- `src/cacheness/interfaces.py` — ABCs, TypedDicts, dataclass contracts (HandlerResult, IntegrityReport, SignableFields)
- `src/cacheness/config.py` — CacheConfig, _DEFAULT_TTL
- `src/cacheness/security.py` — HMAC-SHA256 entry signing (EntrySigner)
- `src/cacheness/custom_metadata.py` — Custom metadata models and utilities
- `src/cacheness/metadata/` — Metadata backend package (Phase 2 output)
- `src/cacheness/handlers/` — Handler package (Phase 1 output)

### Prior Phase Artifacts
- `.planning/phases/02-metadata-package-split/02-CONTEXT.md` — Established `_compat.py` pattern and flat sibling module approach
- `.planning/phases/01-handler-package-split/01-01-SUMMARY.md` — Established `_compat.py` and conditional re-export patterns

### Project References
- `.planning/REQUIREMENTS.md` — DECO-03 and DECO-04 requirement definitions
- `.planning/ROADMAP.md` — Phase 3 success criteria (5 criteria listed)
- `.planning/codebase/STRUCTURE.md` — Codebase layout reference

</canonical_refs>

<code_context>
## Existing Code Insights

### Method Assignment (Planned)

**VerificationMixin** (~330 lines):
- `_verify_entry()` — integrity hash + signature verification
- `verify_integrity()` — public integrity check API
- `_extract_signable_fields()` — build signable data dict
- `_calculate_file_hash()` — xxhash file hashing
- `_sign_entry_if_enabled()` — sign entry during put (D-09)

**StatsMixin** (~35 lines):
- `_record_hit()` — increment hit counter
- `_record_miss()` — increment miss counter
- `get_stats()` — return stats dict

**CustomMetadataMixin** (~350 lines):
- `_supports_custom_metadata()` — check backend support
- `_normalize_custom_metadata()` — normalize input to iterable
- `_store_custom_metadata()` — write custom metadata via SQLAlchemy
- `_get_custom_metadata()` — read custom metadata for cache key
- `_get_registered_schemas()` — list registered schema types
- `query_custom()` — public query with filters
- `query_custom_session()` — context manager for advanced queries
- `query_custom_metadata()` — query with model class
- `get_custom_metadata_for_entry()` — get metadata for specific entry

**StorageModeMixin** (~190 lines):
- `_storage_mode_put()` — put without eviction/stats
- `_storage_mode_get()` — get without TTL/stats/auto-delete
- `_storage_mode_get_with_metadata()` — get with metadata, no TTL

### Stays in UnifiedCache (~2,400 lines)
- `__init__()` and all `_init_*` helpers
- `put()`, `get()`, `get_with_metadata()`, `get_metadata()` — orchestration
- Query operations, file operations, batch operations, convenience API
- Update/delete operations, management operations
- Dunder methods, `for_api()` factory
- `_PutCleanup` class, `_normalize_function_args()` function

### Established Patterns
- Phase 1/2: `_compat.py` for shared imports — NOT needed here (mixins import directly from sibling modules)
- Phase 1/2: Package split with `__init__.py` — NOT used here (flat files, no package)
- Mixin files use relative imports matching core.py's existing style

### Integration Points
- core.py imports 4 mixin classes at top level for class declaration
- Each mixin uses `self.*` to access attrs/methods from UnifiedCache and other mixins
- No mixin imports from core.py — one-directional dependency

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches for mixin implementation.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 03-core-mixin-decomposition*
*Context gathered: 2026-04-03*
