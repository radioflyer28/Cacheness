# Roadmap: Cacheness v0.7.0 Cleanup & Hardening

## Overview

This milestone transforms Cacheness from a well-tested but structurally monolithic library into a maintainable, secure, and debuggable codebase — without changing any public API semantics. Work proceeds in three structural phases (decomposing the three largest files), followed by two behavioral phases (exception handling and security hardening), and concludes with additive test coverage and handler guardrails. Research confirms this ordering avoids two-variable debugging and keeps git history useful.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Handler Package Split** - Split handlers.py into a handlers/ package with one file per handler
- [ ] **Phase 2: Metadata Package Split** - Split metadata.py into a metadata/ package with one file per backend
- [ ] **Phase 3: Core Mixin Decomposition** - Decompose core.py into focused mixins preserving UnifiedCache API
- [ ] **Phase 4: Exception Handling** - Extend exception hierarchy and narrow broad catches to specific types
- [ ] **Phase 5: Security Hardening** - Enable blob content hashing by default and make key fallback configurable
- [ ] **Phase 6: Test Gaps & Handler Robustness** - Add thread safety tests, key rotation tests, and handler ordering guardrails

## Phase Details

### Phase 1: Handler Package Split
**Goal**: handlers.py becomes a handlers/ package with one file per handler, validating the package-split pattern for phases 2-3
**Depends on**: Nothing (first phase)
**Requirements**: DECO-01
**Success Criteria** (what must be TRUE):
  1. Each handler class (DataFrame, NumPy, Polars, Series, Object, Bytes, Dill) lives in its own file under `src/cacheness/handlers/`
  2. `handlers/__init__.py` re-exports all public names — `from cacheness.handlers import HandlerRegistry, get_handler` works unchanged
  3. All 1,427+ existing tests pass without modification to test files
**Plans:** 1 plan
Plans:
- [ ] 01-01-PLAN.md — Create handlers package, split all handler classes into individual files, re-export via __init__.py

### Phase 2: Metadata Package Split
**Goal**: metadata.py becomes a metadata/ package with one file per backend and a shared base module
**Depends on**: Phase 1
**Requirements**: DECO-02
**Success Criteria** (what must be TRUE):
  1. Each backend (JSON, SQLite, PostgreSQL) lives in its own file under `src/cacheness/metadata/`
  2. SQLAlchemy models and shared base class are in a dedicated shared module
  3. `from cacheness.metadata import MetadataBackend, JsonMetadataBackend, SqliteMetadataBackend` works unchanged
  4. All 1,427+ existing tests pass without modification to test files
**Plans**: TBD

### Phase 3: Core Mixin Decomposition
**Goal**: core.py decomposes into focused mixins while preserving the single UnifiedCache class API and all import paths
**Depends on**: Phase 2
**Requirements**: DECO-03, DECO-04
**Success Criteria** (what must be TRUE):
  1. `from cacheness.core import UnifiedCache` works unchanged
  2. UnifiedCache uses mixin classes for verification, statistics, custom metadata, and storage mode concerns
  3. No mixin defines `__init__` — all initialization stays in `UnifiedCache.__init__()`
  4. All existing import paths across handlers, metadata, and core continue working (DECO-04 complete validation)
  5. All 1,427+ existing tests pass without modification to test files
**Plans**: TBD

### Phase 4: Exception Handling
**Goal**: Exception handling is narrowed from broad catches to specific types, making failures visible and debuggable
**Depends on**: Phase 3
**Requirements**: ERRH-01, ERRH-02
**Success Criteria** (what must be TRUE):
  1. `CacheSecurityError` and `CacheBackendError` exist in `error_handling.py` and are importable from `cacheness`
  2. Zero `except Exception` catches remain in source files except those explicitly annotated `# intentionally broad`
  3. Previously-swallowed errors in deserialization and backend operations now raise specific, catchable exception types
  4. All 1,427+ existing tests pass — no behavioral regressions in normal operation paths
**Plans**: TBD

### Phase 5: Security Hardening
**Goal**: Blob integrity verification and key management are production-ready defaults
**Depends on**: Phase 4
**Requirements**: SECU-01, SECU-02
**Success Criteria** (what must be TRUE):
  1. New cache entries have blob content hashes computed and stored on `put()`
  2. `get()` verifies blob hash when present — missing hash passes (does not fail), invalid hash raises error
  3. In-memory key fallback can be configured to raise an error instead of silently degrading to a transient key
  4. `verify_cache_integrity()` includes blob hash checks for entries that have stored hashes
**Plans**: TBD

### Phase 6: Test Gaps & Handler Robustness
**Goal**: Thread safety, key rotation, and handler ordering have explicit test coverage and guardrails
**Depends on**: Phase 5
**Requirements**: TEST-01, TEST-02, TEST-03
**Success Criteria** (what must be TRUE):
  1. Thread safety smoke tests exercise concurrent `put()`/`get()` with `ThreadPoolExecutor` against all metadata backends
  2. Key rotation tests verify graceful behavior when signing key is deleted and regenerated
  3. Handlers have numeric priority values controlling type detection order, with conflict detection warnings for overlapping `can_handle()` matches
  4. Total test count is ≥1,427 — no regressions, new tests added
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6

| Phase | Plans Complete | Status | Completed |
|-------|---------------|--------|-----------|
| 1. Handler Package Split | 0/0 | Not started | - |
| 2. Metadata Package Split | 0/0 | Not started | - |
| 3. Core Mixin Decomposition | 0/0 | Not started | - |
| 4. Exception Handling | 0/0 | Not started | - |
| 5. Security Hardening | 0/0 | Not started | - |
| 6. Test Gaps & Handler Robustness | 0/0 | Not started | - |
