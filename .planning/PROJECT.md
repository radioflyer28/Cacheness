# Cacheness

## What This Is

Cacheness is a Python disk caching library with pluggable metadata backends (JSON/SQLite/PostgreSQL) and handler-based type-aware serialization (DataFrames, NumPy arrays, TensorFlow tensors, etc.). It features cryptographic signing (HMAC-SHA256 with per-namespace HKDF key derivation), optional AES-256-GCM encryption at rest, and configurable security policies. Also serves as a standalone persistent key-value store via storage mode.

## Current State

**Shipped:** v0.10.0 Security & Architecture (2026-04-06)

Full security infrastructure complete: per-namespace HKDF key derivation, configurable key fallback policies, key rotation API with atomic re-signing, and AES-256-GCM encryption at rest. Core decomposed from 2874 to 1422 lines via 11 mixin extractions. Thread safety verified under concurrent rotation, encrypted access, and sustained multi-threaded access. Test suite at 1727 passed / 101 skipped / 0 failures.

## Core Value

Improve reliability, security, and maintainability of Cacheness without changing its public API semantics — make the library safer, more debuggable, and easier to evolve.

## Requirements

### Validated

- ✓ Pluggable metadata backends (JSON, SQLite, PostgreSQL) — existing
- ✓ Type-aware handler-based serialization (DataFrames, NumPy, Polars, Series, Objects, Bytes, Dill) — existing
- ✓ Pluggable blob backends (Filesystem, S3, In-memory) — existing
- ✓ Decorator-based caching (`@cached`, `@cache_if`) — existing
- ✓ Direct put/get API with deterministic cache keys — existing
- ✓ Storage mode (persistent key-value store, no eviction) — existing
- ✓ HMAC-SHA256 metadata signing and verification — existing
- ✓ TTL-based cache expiration — existing
- ✓ Namespace isolation — existing
- ✓ Custom metadata support (linked tables) — existing
- ✓ Cache integrity verification and repair — existing
- ✓ Configuration via dataclasses with human-readable size/duration parsing — existing
- ✓ S3-compatible blob storage — existing
- ✓ Comprehensive test suite (1,427 tests) — existing
- ✓ Handler package split (handlers.py → handlers/ package) — v0.7.0
- ✓ Metadata package split (metadata.py → metadata/ package) — v0.7.0
- ✓ Core mixin decomposition (core.py → 4 mixins) — v0.7.0
- ✓ Narrowed exception handling with CacheSecurityError/CacheBackendError — v0.7.0
- ✓ Blob content hashing enabled by default — v0.7.0
- ✓ Configurable key fallback behavior (raise_on_key_fallback) — v0.7.0
- ✓ Thread safety smoke tests — v0.7.0
- ✓ Key rotation scenario tests — v0.7.0
- ✓ Handler priority values with conflict detection — v0.7.0

- ✓ Concurrency safety: SqliteBackend RLock swap — v0.8.0
- ✓ Crash-safe write intent journal for orphaned blob prevention — v0.8.0
- ✓ End-to-end blob integrity validation via HMAC signature verification — v0.8.0
- ✓ `delete_by_prefix()` with SQL LIKE optimization — v0.8.0

- ✓ `put_batch()` with batch storage API — v0.9.0
- ✓ Accurate threading model documentation — v0.9.0
- ✓ Deserialization security documentation and verification — v0.9.0
- ✓ Cross-platform key file permissions (Windows icacls) — v0.9.0
- ✓ CONCERNS.md updated to reflect all resolved items — v0.9.0

- ✓ Per-namespace key derivation via HKDF — v0.10.0
- ✓ Configurable key fallback policy (raise/warn/fallback) — v0.10.0
- ✓ Blob content encryption at rest (AES-256-GCM) — v0.10.0
- ✓ Key rotation API with atomic re-signing — v0.10.0
- ✓ Further core.py decomposition (2874 → 1422 lines) — v0.10.0
- ✓ Thread safety under concurrent access (verified) — v0.10.0
- ✓ Cross-platform atomic write verification — v0.10.0

### Active

(None — next milestone not yet planned)

### Out of Scope

- Async/await support (AsyncUnifiedCache) — large feature addition, separate milestone
- Advanced eviction policies (LRU, LFU, size-based) — large feature addition, separate milestone
- TensorFlow handler fixes — low priority, platform issues (Windows hangs)
- JSON backend O(n²) write performance — documented limitation, mitigation is "use SQLite"
- Export/import cache — low priority convenience feature
- Automated key rotation — needs background task infrastructure, defer until demand


## Context

- **Codebase state:** ~20,000 lines of Python across ~55 source files. Well-tested (1,727 passed, 101 skipped).
- **Codebase map:** `.planning/codebase/` contains 7 detailed analysis documents from 2026-04-02.
- **Structure:** Code decomposed into `handlers/` (11 files), `metadata/` (5 files), and 15 mixin/helper files alongside `core.py` (1,422 lines).
- **Security:** Per-namespace HKDF key derivation. AES-256-GCM encryption at rest. Configurable key fallback policy (raise/warn/fallback). Key rotation API. HMAC-SHA256 signing. End-to-end signature verification. Cross-platform key file permissions. 3-layer deserialization defense.
- **Concurrency:** SqliteBackend uses RLock for re-entrant safety. Write intent journal prevents orphaned blobs. Thread safety verified under concurrent rotation, encryption, and sustained access. Atomic writes verified on Windows.
- **Error handling:** All `except Exception` catches narrowed or annotated `# intentionally broad`.
- **Tests:** 1,727 tests covering thread safety, key rotation, encryption, concurrency stress, atomic writes, cross-phase integration, and all core functionality.

## Testing Philosophy & Workflow

### Red-Green-Red TDD

All new code follows the **red-green-red** cycle:

1. **Red** — Write a failing test that defines the expected behavior
2. **Green** — Write the minimum code to make the test pass
3. **Red** — Refactor, confirm the test still passes, then write the next failing test

Tests are written *before* implementation, not after. A feature is not done until its tests pass and existing tests remain green.

### Tiered Test Execution

Run the smallest useful set of tests at each stage — fast feedback, full coverage before push:

| Tier | When | What | Time |
|------|------|------|------|
| **Tier 1** | After each code change | Tests directly exercising modified code | ~5-15s |
| **Tier 2** | After planned changes, before full suite | Add regression-risk and cross-cutting tests | ~30-60s |
| **Full suite** | Once before push | All 1,427+ tests | ~48s parallel |

Use the source-to-test mapping in `copilot-instructions.md` to select Tier 1 files. Add cross-cutting tests (integrity, parity, fault injection) for Tier 2.

### Test Quality Standards

New tests must match the quality and coverage strategies already employed:

- **One test file per concern area** — not per source file. Group by behavior domain (`test_cache_integrity.py`, `test_namespace_isolation.py`).
- **Parametrize across backends** — any test touching metadata must run against JSON, SQLite, and (where dockerized) PostgreSQL.
- **Fault injection** — use `patch.object` to simulate failures (disk errors, metadata write failures, network issues). See `tests/test_fault_injection.py`.
- **Property-based testing** — use Hypothesis for invariant verification where applicable. See `tests/test_property_based.py`.
- **Backend parity** — when adding behavior to one backend, verify all backends behave identically via `tests/test_backend_parity.py`.
- **Optional dependency guards** — use `@pytest.mark.skipif(not DEP_AVAILABLE, ...)` and `pytest.importorskip()` for optional deps (numpy, pandas, polars, dill, blosc2).
- **Fixture lifecycle** — always close caches and clean up temp dirs. Use context managers or `yield` + explicit `cache.close()`.
- **xdist grouping** — tests sharing Docker resources (PostgreSQL, real S3) use `@pytest.mark.xdist_group("docker")`.
- **No sleeping in tests** — use deterministic waits or mock time. Exception: Windows file-handle cleanup may require minimal `gc.collect()` + sleep.

### Test Naming

- Test functions: `test_<behavior_under_test>` — descriptive enough to serve as documentation
- Test classes: group related tests as `class Test<Feature>:`
- Parametrized IDs: use `pytest.param(..., id="descriptive-label")`

## Constraints

- **Backward compatibility**: All existing public APIs must remain unchanged. Imports from `cacheness` must continue to work.
- **Test baseline**: Must maintain ≥1,427 passing tests. Zero regressions.
- **Package manager**: uv only — no pip, no python directly.
- **Python version**: ≥3.12 (linting target)
- **Platform**: Must work on Windows and Unix. Windows-specific quirks documented in `docs/WINDOWS_COMPATIBILITY.md`.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Mixins over inheritance for core.py decomposition | Preserves single `UnifiedCache` class API, no breaking changes | ✓ Good — 4 mixins extracted cleanly |
| Package-per-backend for metadata.py | Clear separation, each backend file independent | ✓ Good — 5-file package |
| Package-per-handler for handlers.py | Handlers are independent, registry stays in `registry.py` | ✓ Good — 11-file package |
| Cleanup + hardening only (no async/eviction) | Scope control — async and eviction are large features deserving their own milestones | ✓ Good |
| `_compat.py` pattern for shared imports | Avoids circular imports in package splits | ✓ Good — reused across handlers/ and metadata/ |
| Inline execution for phases 4-6 | Simple enough to not need formal plan files | ✓ Good — faster delivery |
| RLock over ReadWriteLock | ReadWriteLock needs call-chain audit to avoid deadlocks; RLock sufficient for current patterns | ✓ Good — simple, no deadlocks |
| File-based intent journal over metadata table | Backend-agnostic, crash-safe, no schema migration, lazy dir creation | ✓ Good — 12 tests pass |
| No new blob_hmac field | file_hash already in HMAC signed fields — transitive blob integrity guaranteed | ✓ Good — no unnecessary complexity |
| Document existing deserialization defenses vs add new code | 3-layer model already covers threats — documenting > adding unnecessary code | ✓ Good |
| icacls over win32security for Windows key permissions | Zero new dependencies, available on all Windows | ✓ Good |
| HKDF-SHA256 for per-namespace key derivation | RFC 5869, stdlib-only (hmac+hashlib), no new deps | ✓ Good — cryptographic namespace isolation |
| 3-mode key fallback policy over boolean | More expressive (raise/warn/fallback), backward-compatible deprecation shim | ✓ Good |
| AES-256-GCM for encryption at rest | Authenticated encryption, HKDF-derived per-namespace keys, opt-in | ✓ Good |
| Phase 20 superseded by Phase 22 | Phase 20 never executed; Phase 22 (gap closure) addressed same TEST-01/TEST-02 requirements more comprehensively | ✓ Good |
| Retroactive verification (Phase 21) for outside-GSD work | Phases 18-19 complete but lacked VERIFICATION.md — Phase 21 created them | ✓ Good |
| Further mixin extraction (11 mixins) | core.py 2874→1422 lines, clean separation of concerns | ✓ Good |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-06 after v0.10.0 milestone completed*
