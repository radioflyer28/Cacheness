# Requirements: Cacheness v0.7.0 Cleanup & Hardening

**Defined:** 2026-04-02
**Core Value:** Improve reliability, security, and maintainability without changing public API semantics

## v0.7.0 Requirements

Requirements for the cleanup & hardening milestone. Each maps to roadmap phases.

### Code Decomposition

- [ ] **DECO-01**: Split `handlers.py` (1,425 lines) into a `handlers/` package with one file per handler and registry in `__init__.py`
- [ ] **DECO-02**: Split `metadata.py` (2,562 lines) into a `metadata/` package with one file per backend (JSON, SQLite, PostgreSQL) and shared base
- [ ] **DECO-03**: Decompose `core.py` (3,307 lines) into focused mixins (verification, statistics, custom metadata, storage mode) while preserving single `UnifiedCache` class API
- [ ] **DECO-04**: All existing import paths (`from cacheness.handlers import ...`, `from cacheness.metadata import ...`, `from cacheness.core import UnifiedCache`) continue working via `__init__.py` re-exports

### Security Hardening

- [ ] **SECU-01**: Enable blob content hashing by default — hash computed on `put()`, verified on `get()` when hash is present (missing hash = pass, not fail)
- [ ] **SECU-02**: Make in-memory key fallback behavior configurable — add option to raise error instead of silently degrading to transient key

### Error Handling

- [ ] **ERRH-01**: Extend exception hierarchy with 2-3 new types (`CacheSecurityError`, `CacheBackendError`) in `error_handling.py`
- [ ] **ERRH-02**: Narrow broad `except Exception` catches across source files (~30 instances in core.py, compress_pickle.py, custom_metadata.py) to specific exception types, preserving intentional safety nets with `# intentionally broad` comments

### Testing & Robustness

- [ ] **TEST-01**: Add thread safety smoke tests — concurrent `put()`/`get()` with `ThreadPoolExecutor` against all metadata backends
- [ ] **TEST-02**: Add key rotation scenario tests — delete key file, restart, verify entries signed with old key are handled gracefully
- [ ] **TEST-03**: Add handler ordering guardrails — numeric priority on handlers, conflict detection warnings when overlapping `can_handle()` matches

## Future Requirements

Deferred to subsequent milestones:

- **SECU-03**: Per-namespace key derivation via HKDF (needs migration path design)
- **SECU-04**: Windows key file ACLs via `icacls` (document limitation for now)
- **PERF-01**: Async/await support (AsyncUnifiedCache)
- **EVIC-01**: Advanced eviction policies (LRU, LFU, size-based)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Blob content encryption at rest | Feature addition, not hardening |
| Adding `cryptography` package | C-extension bloat for simple HKDF achievable with stdlib |
| Automatic key rotation | Distributed systems problem, out of scope for disk cache |
| Full thread safety on `put()`/`get()` | Would serialize all operations, defeating performance |
| Deep inheritance hierarchy for UnifiedCache | Mixins preferred; avoids tight coupling |
| Splitting `compress_pickle.py` | Stable, low churn — not worth the refactoring risk |
| TensorFlow handler fixes | Platform issues (Windows hangs), low priority |
| JSON backend O(n²) writes | Documented limitation; mitigation is "use SQLite" |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DECO-01 | Phase 1: Handler Package Split | Pending |
| DECO-02 | Phase 2: Metadata Package Split | Pending |
| DECO-03 | Phase 3: Core Mixin Decomposition | Pending |
| DECO-04 | Phase 3: Core Mixin Decomposition | Pending |
| ERRH-01 | Phase 4: Exception Handling | Pending |
| ERRH-02 | Phase 4: Exception Handling | Pending |
| SECU-01 | Phase 5: Security Hardening | Pending |
| SECU-02 | Phase 5: Security Hardening | Pending |
| TEST-01 | Phase 6: Test Gaps & Handler Robustness | Pending |
| TEST-02 | Phase 6: Test Gaps & Handler Robustness | Pending |
| TEST-03 | Phase 6: Test Gaps & Handler Robustness | Pending |

**Coverage:**
- v0.7.0 requirements: 11 total
- Mapped to phases: 11 ✓
- Unmapped: 0

---
*Requirements defined: 2026-04-02*
*Last updated: 2026-04-02 after milestone v0.7.0 scoping*
