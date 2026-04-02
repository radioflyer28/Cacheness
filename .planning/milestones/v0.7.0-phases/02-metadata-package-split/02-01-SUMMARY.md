---
phase: 02-metadata-package-split
plan: 01
subsystem: infra
tags: [refactor, metadata, package-split, python-packaging, sqlalchemy, orm]

requires:
  - phase: 01-handler-package-split
    provides: "Validated _compat.py package-split pattern"
provides:
  - "metadata/ package with one file per backend"
  - "Shared _compat.py with ORM models, namespace utilities, availability flags"
  - "base.py with MetadataBackend ABC and CachedMetadataBackend wrapper"
  - "Separate json_backend.py and sqlite_backend.py files"
  - "Re-exporting __init__.py with create_metadata_backend factory"
  - "Fixed handlers/_compat.py missing re-exports from Phase 1"
affects: [03-core-mixin-decomposition]

tech-stack:
  added: []
  patterns:
    - "_compat.py shared imports pattern with ORM models and namespace utils"
    - "Direct stdlib imports in each module rather than re-exporting through _compat"
    - "noqa: F401 on _compat re-exports to prevent ruff from removing them"

key-files:
  created:
    - src/cacheness/metadata/__init__.py
    - src/cacheness/metadata/_compat.py
    - src/cacheness/metadata/base.py
    - src/cacheness/metadata/json_backend.py
    - src/cacheness/metadata/sqlite_backend.py
  modified:
    - src/cacheness/handlers/_compat.py

key-decisions:
  - "Each module imports its own stdlib deps directly (threading, datetime, etc.) rather than re-exporting through _compat — cleaner than Phase 1 approach"
  - "SQLite migration functions (_sqlite_migrate_v1_to_v2, _sqlite_migrate_v2_to_v3) placed in sqlite_backend.py, not json_backend.py where they were incorrectly extracted initially"
  - "Used noqa: F401 on handlers/_compat.py re-exports to prevent ruff from stripping them"
  - "Migration type alias uses Callable[['object', str], None] instead of forward-ref to MetadataBackend to avoid circular reference"

patterns-established:
  - "Direct stdlib imports per module: each backend file imports its own standard library deps, _compat only exports shared project types (ORM models, constants, flags)"
  - "noqa: F401 is required on all _compat re-exports to prevent ruff from removing 'unused' imports that are actually consumed by sibling modules"

requirements-completed: [DECO-02]

duration: 20min
completed: 2026-04-02
---

# Phase 2: Metadata Package Split Summary

**Split monolithic metadata.py (3,046 lines) into 5-file metadata/ package and fixed pre-existing handlers/_compat.py import bug — all 1,604 tests passing.**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-04-02
- **Completed:** 2026-04-02
- **Tasks:** 2 (create package + verify)
- **Files created:** 5
- **Files deleted:** 1 (metadata.py)
- **Files modified:** 1 (handlers/_compat.py)

## Accomplishments
- Decomposed 3,046-line monolithic metadata.py into 5 focused modules
- _compat.py (339 lines): ORM models, namespace utilities, availability flags
- base.py (729 lines): MetadataBackend ABC, CachedMetadataBackend, create_entry_cache
- json_backend.py (741 lines): JsonBackend implementation
- sqlite_backend.py (1,230 lines): SqliteBackend + schema migration functions
- __init__.py (162 lines): re-exports + create_metadata_backend factory
- Preserved all import paths — `from cacheness.metadata import MetadataBackend, JsonBackend, SqliteBackend` works unchanged
- Fixed pre-existing handlers/_compat.py bug: added missing re-exports for CacheHandler, HandlerResult, BlobReadContext, CacheWriteError, CacheReadError, cache_operation_context, and compress_pickle utilities
- Full test suite: 1,604 passed, 101 skipped, 0 failures

## Task Commits

1. **Task 1+2: Create metadata package + verify** — `56be995` (feat)

## Files Created/Modified
- `src/cacheness/metadata/__init__.py` — Re-exports all public names + create_metadata_backend factory
- `src/cacheness/metadata/_compat.py` — ORM models, namespace validation, availability flags
- `src/cacheness/metadata/base.py` — MetadataBackend ABC, CachedMetadataBackend wrapper, create_entry_cache
- `src/cacheness/metadata/json_backend.py` — JsonBackend implementation
- `src/cacheness/metadata/sqlite_backend.py` — SqliteBackend + migration functions (_sqlite_migrate_v1_to_v2, _sqlite_migrate_v2_to_v3)
- `src/cacheness/handlers/_compat.py` — Added missing interface re-exports (CacheHandler, HandlerResult, etc.)

## Decisions Made
- Improved on Phase 1 pattern: each module imports its own stdlib deps directly instead of routing through _compat
- Migration functions belong in sqlite_backend.py since they're SQLite-specific (initially misplaced in json_backend.py by extraction script)
- Used `Callable[["object", str], None]` for Migration type alias to avoid forward reference to MetadataBackend

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] handlers/_compat.py missing re-exports**
- **Found during:** Task 1 (import verification)
- **Issue:** Phase 1 created handlers/_compat.py but omitted re-exports for CacheHandler, HandlerResult, BlobReadContext, CacheWriteError, CacheReadError, cache_operation_context, and compress_pickle utilities. All handler modules import these from ._compat. This blocked entire package import.
- **Fix:** Added imports from ..interfaces, ..error_handling, and ..compress_pickle with noqa: F401 annotations
- **Files modified:** src/cacheness/handlers/_compat.py
- **Verification:** All 1,604 tests pass
- **Committed in:** 56be995 (part of task commit)

**2. [Rule 3 - Blocking] SQLite migration functions misplaced in json_backend.py**
- **Found during:** Task 1 (ruff check)
- **Issue:** Extraction script placed _sqlite_migrate_v1_to_v2 and _sqlite_migrate_v2_to_v3 at the end of json_backend.py instead of sqlite_backend.py
- **Fix:** Moved migration functions to sqlite_backend.py before the SqliteBackend class definition
- **Files modified:** json_backend.py, sqlite_backend.py
- **Verification:** ruff check passes, all tests pass
- **Committed in:** 56be995

---

**Total deviations:** 2 auto-fixed (2x Rule 3 - Blocking)
**Impact on plan:** Both fixes were necessary for correctness. No scope creep.

## Issues Encountered
- Initial extraction script placed code at wrong boundaries — migration functions ended up in json_backend.py
- ruff --fix auto-removed the handlers/_compat.py re-exports (they appeared "unused" to ruff) — required adding noqa: F401 annotations

## User Setup Required
None — no external service configuration required.

## Next Phase Readiness
- Phase 3 (Core Mixin Decomposition) can proceed — metadata/ package is stable
- Pattern refined: direct stdlib imports per module is cleaner than Phase 1's _compat re-export approach
- handlers/ and metadata/ packages fully verified with 1,604 passing tests
