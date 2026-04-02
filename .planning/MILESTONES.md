# Milestones

## v0.7.0 Cleanup & Hardening (Shipped: 2026-04-02)

**Phases completed:** 6 phases, 6 commits, 39 files changed (+6480/-5661 lines)
**Test baseline:** 1604 → 1616 tests (12 new, 0 regressions)
**Git range:** `5ab2d10..b538183`
**Timeline:** 2026-04-02

**Key accomplishments:**

1. Split `handlers.py` (1700 lines) into 11-file `handlers/` package with per-handler modules
2. Split `metadata.py` (3046 lines) into `metadata/` package with per-backend modules
3. Decomposed `core.py` into 4 focused mixins (verification, stats, custom metadata, storage mode)
4. Narrowed ~30 broad `except Exception` catches to specific types; added `CacheSecurityError` and `CacheBackendError`
5. Security hardening: `verify_hashes` defaults to `True`, configurable `raise_on_key_fallback`
6. Added thread safety tests, key rotation tests, and handler priority with conflict detection warnings

---
