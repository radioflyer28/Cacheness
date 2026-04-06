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

## Milestone: v0.9.0 — Completeness & Hardening

**Shipped:** 2026-04-03
**Phases:** 4 | **Plans:** 4

### What Was Built

1. `put_batch()` API completing the batch operations surface (`put_batch`, `get_batch`, `delete_batch`, `touch_batch`)
2. Threading model documentation fixed — corrected false claim that `_lock` is never acquired (it's acquired by 22+ methods)
3. Deserialization security documented — 3-layer defense model (xxhash + HMAC + signature verification) at all pickle/dill sites
4. Cross-platform key file permissions — `icacls` on Windows replaces no-op `chmod(0o600)`

### What Worked

- **Discovery-first approach** — Explore subagent found that most "missing" APIs already existed, reducing MGMT scope from 6+ items to just `put_batch()`
- **Documentation as the deliverable** — Phases 12-13 proved that documenting existing defenses is more valuable than adding unnecessary code (existing 3-layer model was already sound)
- **CONCERNS.md as scope source** — using the codebase audit document to drive milestone requirements produced highly targeted, actionable work

### What Was Inefficient

- **Git commit timeouts** — several commits timed out in the terminal, requiring background terminal checks and retry logic. Intermittent issue, likely Windows GPG or hook overhead.
- **gsd-tools stale data** — `roadmap analyze` returned v0.7.0 data instead of v0.9.0. The tooling couldn't parse the multi-milestone ROADMAP correctly, requiring manual phase tracking.
- **Planning overhead for documentation phases** — full discuss→plan→execute cycle was unnecessary for phases that were purely documentation fixes

### Patterns Established

- `_set_key_file_permissions()` as cross-platform static method pattern — OS detection at call site, graceful fallback on both platforms
- CONCERNS.md as milestone scope driver — items listed as "missing" or "broken" become requirements, resolved items get version annotations
- Documentation phases can skip formal planning — investigate, fix, commit

### Key Lessons

- Always verify documentation claims against actual code — TROUBLESHOOTING.md's false threading claim persisted across 2 milestones because nobody checked
- `icacls /inheritance:r /grant:r` is the Windows equivalent of `chmod 0o600` — no pywin32 dependency needed
- Small milestones (4 phases) complete in a single session, ideal for cleanup/hardening work

### Cost Observations

- Sessions: 2 (planning + phases 11-12, phases 13-14 + lifecycle)
- Notable: Entire milestone from requirements definition through push in ~2 hours

---

## Milestone: v0.10.0 — Security & Architecture

**Shipped:** 2026-04-06
**Phases:** 8 (7 complete + 1 superseded) | **Plans:** 9

### What Was Built

1. Per-namespace HKDF-SHA256 key derivation — cryptographic isolation between namespaces
2. Configurable key fallback policy (raise/warn/fallback) with backward-compatible deprecation shim
3. Key rotation API (`rotate_key()`) on both UnifiedCache and BlobStore with `RotationResult` tracking
4. AES-256-GCM encryption at rest with HKDF-derived per-namespace encryption keys
5. Core decomposition II — core.py from 2874 to 1422 lines via 11 mixin files
6. 18 concurrency & integration tests — thread safety stress, atomic writes, cross-phase feature composition

### What Worked

- **Milestone audit as quality gate** — running `/gsd-audit-milestone` mid-milestone caught doc gaps (missing VERIFICATION.md for 3 phases) and requirement gaps (TEST-01/TEST-02 unsatisfied). This drove Phases 21-22 as targeted gap closure rather than discovering issues at ship time.
- **Outside-GSD work + retroactive verification** — Phases 18-19 were implemented outside GSD but retroactively verified in Phase 21. This proved that formal verification can be decoupled from implementation.
- **Phase supersession** — Phase 20 was planned but superseded by Phase 22 (a more comprehensive version). Clean supersession tracking prevented wasted effort.
- **Cross-phase integration testing** — Phase 22 tested all three security features together (fallback + HKDF + encryption), catching composition issues that per-feature tests wouldn't find.

### What Was Inefficient

- **Phases 18-19 outside GSD** — no SUMMARY.md files, requiring Phase 21 to create retroactive verification artifacts. If they'd been tracked through GSD, the audit would have passed on first run.
- **REQUIREMENTS.md/ROADMAP.md checkbox staleness** — TEST-01/TEST-02 checkboxes not updated when Phase 22 completed, requiring manual fixes during audit. Phase completion should auto-update requirement checkboxes.
- **bd hook version mismatch** — pre-commit hook was from bd 0.55.4 but bd was at 0.63.3, blocking commits until hooks were reinstalled.

### Patterns Established

- `_hkdf_sha256()` as stdlib-only HKDF implementation (hmac + hashlib, no external crypto dependency for signing)
- `key_fallback_policy` 3-mode pattern with deprecation shim for old boolean field
- `RotationResult` dataclass for tracking bulk re-signing operations
- `encryption.py` as standalone encryption module with `encrypt_blob()`/`decrypt_blob()`
- Milestone audit → gap closure phases → re-audit as the quality assurance loop

### Key Lessons

- Always run milestone audit before declaring "done" — it catches systematic gaps that per-phase verification misses
- Phase supersession is cleaner than modifying existing phases — Phase 22 replaced Phase 20 with better scope
- Integration tests that combine multiple features are high-value but easy to forget — plan them explicitly

### Cost Observations

- Sessions: ~5 (planning, phases 15-17, phases 18-19 outside GSD, phase 21 retroactive, phase 22 + audit)
- Notable: Largest milestone yet (8 phases, 76 new tests, +9464 lines) — took 4 days vs 1 day for prior milestones

---

## Cross-Milestone Trends

| Metric | v0.7.0 | v0.8.0 | v0.9.0 | v0.10.0 |
|--------|--------|--------|--------|---------|
| Phases | 6 | 4 | 4 | 8 (7+1 superseded) |
| Plans (formal) | 3 | 4 | 4 | 9 |
| Tests added | 12 | 25 | 10 | 76 |
| Test total | 1616 | 1641 | 1651 | 1727 |
| Files changed | 39 | 28 | 20 | 67 |
| Lines +/- | +6480/-5661 | +1740/-36 | +606/-56 | +9464/-2220 |
