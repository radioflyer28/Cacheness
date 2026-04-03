# Milestones

## v0.9.0 Completeness & Hardening (Shipped: 2026-04-03)

**Phases completed:** 4 phases, 4 plans, 20 files changed (+606/-56 lines)
**Test baseline:** 1641 → 1651 tests (10 new, 0 regressions)
**Git range:** `8f1a340..e7be809` (11 commits)

**Key accomplishments:**

1. Completed batch API surface — `put_batch()` alongside existing `get_batch()`, `delete_batch()`, `touch_batch()` (8 tests)
2. Fixed contradictory threading documentation — discovered 22+ methods use RLock, corrected API_REFERENCE.md and TROUBLESHOOTING.md
3. Documented 3-layer deserialization defense model (xxhash + HMAC + signature verification) in SECURITY.md with code comments at all pickle/dill sites
4. Cross-platform key file permissions — `icacls` on Windows replaces no-op `chmod(0o600)` (2 tests)
5. Updated CONCERNS.md — 5 items marked as resolved with version references

---

## v0.8.0 API & Robustness (Shipped: 2026-04-03)

**Phases completed:** 4 phases, 4 plans, 28 files changed (+1740/-36 lines)
**Test baseline:** 1616 → 1641 tests (25 new, 0 regressions)
**Git range:** `c801520..914a1a0` (14 commits)

**Key accomplishments:**

1. Swapped SqliteBackend `Lock` → `RLock` with consistent lock acquisition across all public methods — re-entrant calls no longer deadlock
2. Added crash-safe `WriteIntentJournal` — intent files track in-flight blob writes, orphaned blobs auto-cleaned on cache init
3. Added `verify_signatures=True` parameter to `verify_integrity()` — end-to-end HMAC signature verification confirms blob tamper detection via `file_hash`
4. Added `delete_by_prefix()` API with backend-optimized SQL `LIKE` queries for bulk key-prefix deletion

---

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
