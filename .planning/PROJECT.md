# Cacheness — Cleanup & Hardening Milestone

## What This Is

Cacheness is a Python disk caching library with pluggable metadata backends (JSON/SQLite/PostgreSQL) and handler-based type-aware serialization (DataFrames, NumPy arrays, TensorFlow tensors, etc.). It also serves as a standalone persistent key-value store via storage mode. This milestone addresses technical debt, security gaps, error handling inconsistencies, test coverage gaps, and code structure issues identified by the codebase audit.

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

### Active

- [ ] Decompose `core.py` (3,307 lines) into focused mixins/delegates
- [ ] Split `metadata.py` (2,562 lines) into per-backend modules
- [ ] Split `handlers.py` (1,425 lines) into per-handler modules
- [ ] Narrow broad `except Exception` catches (~30 instances) to specific exception types
- [ ] Enable blob content hashing by default (not just metadata signing)
- [ ] Implement proper signing key file permissions on Windows (ACLs or documented limitation)
- [ ] Add per-namespace key derivation (HKDF from master key + namespace ID)
- [ ] Make in-memory key fallback behavior configurable (fail-loud option)
- [ ] Add thread safety tests for concurrent `put()`/`get()` operations
- [ ] Add key rotation scenario tests
- [ ] Improve handler type detection robustness (document ordering constraints, add guardrails)

### Out of Scope

- Async/await support (AsyncUnifiedCache) — large feature addition, separate milestone
- Advanced eviction policies (LRU, LFU, size-based) — large feature addition, separate milestone
- TensorFlow handler fixes — low priority, platform issues (Windows hangs)
- JSON backend O(n²) write performance — documented limitation, mitigation is "use SQLite"
- Export/import cache — low priority convenience feature

## Context

- **Codebase state:** 16,613 lines of Python across 27 source files. Well-tested (1,427 passed, 102 skipped).
- **Codebase map:** `.planning/codebase/` contains 7 detailed analysis documents from 2026-04-02.
- **Key concern:** The three largest files (`core.py`, `metadata.py`, `handlers.py`) contain 7,294 lines combined — nearly half the codebase. Decomposition is the highest-impact structural change.
- **Security gaps:** Signing covers metadata but not blob content by default. Windows key permissions are ineffective. Namespace key isolation is opt-in rather than default.
- **Error handling:** 30+ broad `except Exception` catches mask bugs and make debugging difficult.
- **Test gaps:** No concurrency tests despite documented thread-safety limitations. No key rotation tests.

## Constraints

- **Backward compatibility**: All existing public APIs must remain unchanged. Imports from `cacheness` must continue to work.
- **Test baseline**: Must maintain ≥1,427 passing tests. Zero regressions.
- **Package manager**: uv only — no pip, no python directly.
- **Python version**: ≥3.12 (linting target)
- **Platform**: Must work on Windows and Unix. Windows-specific quirks documented in `docs/WINDOWS_COMPATIBILITY.md`.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Mixins over inheritance for core.py decomposition | Preserves single `UnifiedCache` class API, no breaking changes | — Pending |
| Package-per-backend for metadata.py | Clear separation, each backend file independent | — Pending |
| Package-per-handler for handlers.py | Handlers are independent, registry stays in `__init__.py` | — Pending |
| Cleanup + hardening only (no async/eviction) | Scope control — async and eviction are large features deserving their own milestones | ✓ Good |

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
*Last updated: 2026-04-02 after initialization*
