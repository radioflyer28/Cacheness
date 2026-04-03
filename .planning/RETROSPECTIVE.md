# Retrospective

## Milestone: v0.7.0 — Cleanup & Hardening

**Shipped:** 2026-04-02
**Phases:** 6 | **Plans:** 3 formal + 3 inline

### What Was Built

1. Split `handlers.py` (1700 lines) into 11-file `handlers/` package with per-handler modules
2. Split `metadata.py` (3046 lines) into `metadata/` package with per-backend modules
3. Decomposed `core.py` into 4 focused mixins (verification, stats, custom metadata, storage mode)
4. Narrowed ~30 broad `except Exception` catches to specific types; added `CacheSecurityError` and `CacheBackendError`
5. Security hardening: `verify_hashes` defaults to `True`, configurable `raise_on_key_fallback`
6. Added 12 new tests: thread safety (6), key rotation (6), handler priority with conflict detection

### What Worked

- **`_compat.py` pattern** for shared imports in package splits — avoided circular imports, reused in both handlers/ and metadata/ packages
- **Inline execution for simple phases** — Phases 4-6 didn't need formal plan files; executing directly was faster without loss of quality
- **Tiered testing** — running Tier 1 tests after each change kept feedback loops short (~5-15s), full suite only needed once before push
- **Sequential phase ordering** (structural → behavioral → additive) — avoided two-variable debugging; each phase had a stable base

### What Was Inefficient

- **Phase 6 full suite failure diagnosis** — the 65KB test output was difficult to parse; finding the one failing test required re-running with better filtering
- **Phases 4-6 had no formal summaries** — the CLI only counted 3 phases because only 3 had on-disk SUMMARY.md files. Inline execution should still produce lightweight summaries
- **Output truncation** — large terminal outputs were truncated, requiring multiple read attempts

### Patterns Established

- `_compat.py` as the canonical shared-imports module name for package splits
- `# intentionally broad` annotation comment for `except Exception` catches that are deliberate
- Handler `priority: int` class attribute for explicit type detection ordering
- `HandlerRegistry.get_handler()` warns when multiple handlers match the same type

### Key Lessons

- Always produce a SUMMARY.md even for inline phases — otherwise tooling undercounts
- Filter large test output at the source (`--tb=line`, `Select-String -Pattern FAILED`) rather than reading 65KB files
- The `verify_hashes` default change (False→True) is a behavioral change that needs careful test update — existing tests may assume the old default

### Cost Observations

- Sessions: 3 (Phase 1-3 session, Phase 4-5 session, Phase 5-6 session)
- Notable: All 6 phases completed and pushed in one day of work

---

## Milestone: v0.8.0 — API & Robustness

**Shipped:** 2026-04-03
**Phases:** 4 | **Plans:** 4

### What Was Built

1. SqliteBackend `Lock` → `RLock` swap with consistent lock acquisition across all public methods — re-entrant calls no longer deadlock
2. Crash-safe `WriteIntentJournal` — intent files track in-flight blob writes, orphaned blobs auto-cleaned on cache init
3. `verify_signatures=True` parameter on `verify_integrity()` — end-to-end HMAC signature verification confirms blob tamper detection via `file_hash`
4. `delete_by_prefix()` API with SQL `LIKE` optimization on SQLite, Python filter fallback on JSON

### What Worked

- **Codebase scout before planning** — reading source code to understand existing patterns (e.g., discovering `file_hash` is already in HMAC signed fields) avoided unnecessary work in Phase 9
- **Direct implementation with incremental testing** — skipping full GSD execute-phase overhead for focused 1-plan phases was efficient
- **Phase ordering with dependencies** — Phase 7 (RLock) foundational fix enabled Phases 8-10 to safely use locking

### What Was Inefficient

- **Phase 9 created_at timezone normalization** — initial implementation in blob_store.py failed because signing path normalizes created_at via `_extract_signable_fields()`. Moving to `_verification_mixin.py` fixed it but required a refactor mid-phase.
- **Phase 10 cache key slash semantics** — using `models/bert` as cache key failed because `/` is treated as a directory separator. Fixed by switching to `_` separator. This was a surprising edge case.
- **No VERIFICATION.md files produced** — phases executed without the GSD verification step, making the milestone audit rely solely on SUMMARYs and test results.

### Patterns Established

- `WriteIntentJournal` as the crash-safety pattern: record intent → do work → clear intent
- `keys_by_prefix()` ABC method with SQL LIKE override pattern for backend-optimized queries
- `verify_signatures` parameter pattern for opt-in expensive verification
- Signature verification belongs in the mixin that has `_extract_signable_fields()`, not in blob_store

### Key Lessons

- Always check timezone/normalization paths when implementing signature verification — signing and verification must use identical field extraction
- Cache keys with `/` create subdirectories — prefix deletion works on explicit cache keys, not hashed kwarg-derived keys
- SqliteBackend uses namespace-specific table names (`self._entries_table`), not `self.namespace_id` — different from expected patterns

### Cost Observations

- Sessions: 2 (Phase 7 session, Phases 8-10 session)
- Notable: All 4 phases completed in 2 sessions, 25 new tests added

---

## Cross-Milestone Trends

| Metric | v0.7.0 | v0.8.0 |
|--------|--------|--------|
| Phases | 6 | 4 |
| Plans (formal) | 3 | 4 |
| Tests added | 12 | 25 |
| Test total | 1616 | 1641 |
| Files changed | 39 | 28 |
| Lines +/- | +6480/-5661 | +1740/-36 |
