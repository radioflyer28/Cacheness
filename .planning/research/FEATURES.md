# Feature Landscape

**Domain:** Python disk caching library — cleanup & hardening milestone
**Researched:** 2026-04-02
**Overall confidence:** HIGH (based on direct codebase analysis + established Python ecosystem practices)

This document maps the feature space for improving reliability, security, and maintainability of an existing, well-tested Python caching library (1,427 tests, 16,613 LOC). All items below are scoped to the **Cleanup & Hardening** milestone — no new public API features.

---

## 1. Code Decomposition — Monolithic File Splitting

**Context:** `core.py` (3,307 lines), `metadata.py` (2,562 lines), `handlers.py` (1,425 lines) = 7,294 lines in three files, nearly half the codebase.

### Table Stakes (Must-Do)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Split `handlers.py` into `handlers/` package with one file per handler | Independent handlers sharing a file is pure organizational debt. Lowest-risk decomposition because handlers are stateless and self-contained. | **Low** | Keep `HandlerRegistry` in `handlers/__init__.py`, re-export all handler classes. 8 handlers → 8 files + `__init__.py` + `base.py`. |
| Split `metadata.py` into `metadata/` package with per-backend modules | Three full backend implementations (JSON 500L, SQLite 800L, PostgreSQL 700L) in one file. Each backend is already independent behind the ABC. | **Low-Medium** | `metadata/base.py` (ABC + shared utilities), `metadata/json_backend.py`, `metadata/sqlite_backend.py`, `metadata/pg_backend.py`, `metadata/__init__.py` (re-exports). Must preserve all existing import paths via `__init__.py`. |
| Preserve all existing imports from `cacheness.handlers` and `cacheness.metadata` | Any decomposition that breaks `from cacheness.metadata import SqliteBackend` is a regression. | **Low** | Re-export from `__init__.py`. Test with `import cacheness; dir(cacheness.metadata)`. |

### Differentiators (Nice-to-Have)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Decompose `core.py` via mixin classes | Reduces the 3,307-line monolith to ~1,500-line coordinator + focused mixins (`_VerificationMixin`, `_StatisticsMixin`, `_StorageModeMixin`, `_CustomMetadataMixin`). | **Medium-High** | Mixin-based decomposition preserves the single `UnifiedCache` class API. Risk: mixin interactions (shared state via `self`) require careful method boundary design. Python's MRO is well-defined but debuggability suffers with deep mixin chains. **Limit to 4-5 mixins max.** |
| Extract delegate classes instead of mixins for `core.py` | Delegates (`self._verifier = CacheVerifier(self)`) give explicit dependency injection and cleaner testing. | **Medium-High** | More testable than mixins (can unit-test delegates independently) but requires passing `self` or specific attributes to delegates. More refactoring than mixins. |

### Anti-Features (Avoid)

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Deep inheritance hierarchy for `UnifiedCache` | Multiple inheritance levels create fragile base class problems. `super()` chains become undebuggable. | Prefer flat mixins (one level) or composition via delegates. |
| Splitting `core.py` into separate public classes | Breaking `UnifiedCache` into `CacheReader`, `CacheWriter`, `CacheManager` changes the public API. | Keep `UnifiedCache` as the single entry point. Internal decomposition only. |
| Moving files without re-export aliases | Breaks downstream imports. Even internal test imports break. | Always re-export from `__init__.py` with explicit `__all__`. |
| Decomposing `compress_pickle.py` in this milestone | 888 lines but low churn, stable, and not causing maintenance pain. | Leave for a future milestone unless a bug forces touching it. |

### Complexity Assessment

- **handlers.py split:** Low risk, mechanical file moves. ~2-4 hours.
- **metadata.py split:** Low-medium risk, slightly more cross-file references. ~4-6 hours.
- **core.py decomposition:** Highest risk item in the milestone. Mixin boundaries must be drawn carefully to avoid circular attribute access. ~1-2 days.

### Dependencies

- `handlers.py` split and `metadata.py` split are **independent** — can be done in parallel.
- `core.py` decomposition should happen **after** handlers and metadata splits (fewer merge conflicts, cleaner base to work from).
- All three must maintain backward-compatible imports.

---

## 2. Security Hardening

**Context:** HMAC-SHA256 metadata signing exists but has gaps: blob content not signed by default, Windows key permissions are a no-op, all namespaces share one signing key, in-memory key fallback is silent.

### Table Stakes (Must-Do)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Default-on blob content hashing | Current signing covers metadata fields but NOT the actual serialized blob file. An attacker with filesystem access can replace a blob while keeping the metadata signature intact. The `file_hash` field already exists in `EntrySummary` and is computed during `put()` — but verification on `get()` is optional and off by default. | **Low-Medium** | Wire `file_hash` verification into `get()` path by default. Must handle missing hashes for pre-existing entries gracefully (skip verification, log warning). Add `verify_blob_hash` config option (default `True`) for opt-out. |
| Make in-memory key fallback configurable | `security.py` lines 155-160: disk write failure silently falls back to in-memory key. Entries become unverifiable after restart. | **Low** | Add `signing_key_fallback` config: `"memory"` (current default), `"error"` (raise `CacheSecurityError`). Set `"memory"` as default for backward compat, document `"error"` as recommended for production. |

### Differentiators (Nice-to-Have)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Per-namespace key derivation (HKDF) | Cryptographic isolation between namespaces sharing a `cache_dir`. Currently one key signs everything — a compromised namespace key compromises all namespaces. | **Medium** | Use `hashlib`-based HKDF (stdlib, no new deps): derive `namespace_key = HKDF(master_key, info=namespace_id)`. Must handle key migration for existing entries (old entries use master key directly). The Python `hmac` module is sufficient — no need for `cryptography` package. |
| Windows key file ACLs | `chmod(0o600)` is a no-op on Windows. Key file may be world-readable. | **Medium** | Two options: (a) Use `icacls` subprocess call — no new dependency but fragile. (b) Use `pywin32` (`win32security`) — robust but adds optional dependency. (c) Document as known limitation with manual `icacls` command. **Recommend option (c) for this milestone** — actual ACL implementation is complex and low-impact (local attacker with file access has bigger problems). |

### Anti-Features (Avoid)

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Encrypting blob content at rest | Encryption is a feature addition, not hardening. Adds complexity (key management, performance overhead, recovery). Signing detects tampering; encryption prevents reading. Different threat models. | Document as future feature. Signing + integrity verification is sufficient for the "detect tampering" use case. |
| Adding `cryptography` package dependency | Heavy C extension dependency for HKDF when `hashlib` can do the same thing via `hmac.new()` with a derived key. | Use stdlib `hashlib`/`hmac` for key derivation. Python 3.12+ `hashlib` has everything needed. |
| Automatic key rotation | Complex distributed systems problem (re-sign all entries, handle mixed-version entries during rotation window). | Document manual key rotation procedure. Test that old-key entries degrade gracefully (warning, not crash). |

### Complexity Assessment

- **Blob hash verification default-on:** Low-medium. The hash is already computed; this is wiring it into the read path.
- **Configurable key fallback:** Low. Config option + conditional raise.
- **HKDF namespace keys:** Medium. Key derivation is simple; migration path for existing entries is the complexity.
- **Windows ACLs:** Medium if implemented, low if documented-only.

### Dependencies

- Blob hash verification is **independent** — can proceed without other security work.
- Key fallback configuration depends on the `CacheConfig` dataclass — coordinate with any config refactoring.
- HKDF namespace keys depend on having the security module stabilized first.
- Windows ACLs are **independent** of all other security work.

---

## 3. Exception Handling Cleanup

**Context:** 30+ `except Exception` catches across `core.py` (30+), `handlers.py` (20+), `decorators.py` (12), `custom_metadata.py` (6), `compress_pickle.py` (10+). An exception hierarchy already exists in `error_handling.py` (`CacheError` → `CacheConfigurationError`, `CacheStorageError`, `CacheSerializationError`, `CacheHandlerError`, `CacheIntegrityError`, `CacheMetadataError`) but is underutilized.

### Table Stakes (Must-Do)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Narrow `except Exception` catches in `core.py` to specific types | 30+ broad catches mask bugs. Example: `_init_auto_backend()` catches all exceptions when trying SQLite and silently falls back to JSON — a permissions error looks identical to a missing module. | **Medium** | Audit each catch site. Categories: (1) **Narrow to specific** — `OSError`, `PermissionError`, `ValueError`, `KeyError`, `TypeError`, `sqlite3.Error`, `json.JSONDecodeError`. (2) **Keep broad but re-raise critical** — catch `Exception`, re-raise `KeyboardInterrupt`, `SystemExit`, `MemoryError`. (3) **Intentionally broad** — top-level decorator safety nets that must never crash user code. Document each decision. |
| Narrow `except Exception` catches in `handlers.py` | 20+ broad catches in handler `put()`/`get()` methods. Deserialization errors masked as generic failures. | **Medium** | Most should be `OSError | pickle.UnpicklingError | blosc2.Error | ValueError`. Some TensorFlow catches are intentionally broad (TF raises surprising exception types). |
| Ensure all swallowed exceptions log at WARNING or higher | Several catches use `logger.debug()` for errors that affect correctness (e.g., metadata write failures). | **Low** | Audit logging levels. Rule: if the catch changes behavior (fallback, skip, delete), log at WARNING minimum. If it's a harmless retry, DEBUG is fine. |

### Differentiators (Nice-to-Have)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Add missing exception types to hierarchy | Current hierarchy lacks `CacheSecurityError` (for signing/verification failures), `CacheBackendError` (for backend-specific failures), and `CacheKeyError` (for cache key computation failures). | **Low** | Add 2-3 new exception classes to `error_handling.py`. Subclass from `CacheError`. |
| Structured error context on all `CacheError` raises | `CacheError` already accepts a `context` dict — but most raise sites don't pass it. Adding `context={"cache_key": key, "backend": "sqlite"}` aids debugging. | **Low-Medium** | Low per-site change, but many sites to update. Can be done incrementally. |
| Use `cache_operation_context()` context manager consistently | `error_handling.py` already defines `cache_operation_context()` for standardized try/except/log — but it's used in only a few places. | **Medium** | Replacing raw try/except with the context manager across 30+ sites is mechanical but touches many lines. Good for consistency but not urgent. |

### Anti-Features (Avoid)

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Catching and wrapping every stdlib exception in `CacheError` | Overly aggressive wrapping hides the original traceback and makes debugging harder. `OSError` should propagate as `OSError` when it's meaningful. | Only wrap when the exception crosses a significant abstraction boundary (handler → cache, backend → cache). Let specific exceptions propagate naturally within a layer. |
| Adding retry logic to exception handling | Retry is a feature, not error handling. Retries mask transient failures and add latency. | If retry is needed, build it as a separate decorator/wrapper, not inline in catch blocks. |
| Making all cache operations return `Result[T, Error]` instead of raising | Functional error handling would change the entire public API. Pythonic convention is exceptions. | Keep exceptions. Consider `Result` types only if a future async milestone warrants it. |

### Complexity Assessment

- **Narrowing `except Exception` in core.py:** Medium. Each of the 30+ sites needs individual analysis to determine the right exception type. Risk of changing behavior if a previously-caught exception type is no longer caught.
- **Narrowing in handlers.py:** Medium. Same analysis needed, but handlers are more isolated (lower blast radius per change).
- **Adding exception types:** Low. Simple class definitions.
- **Logging level audit:** Low. Mechanical find-and-replace.

### Dependencies

- Exception hierarchy additions should happen **before** narrowing catches (so the new types are available).
- Narrowing catches in `core.py` should happen **after** `core.py` decomposition (fewer merge conflicts, smaller files to audit).
- Narrowing in `handlers.py` and `decorators.py` is **independent** — can proceed in parallel.

---

## 4. Concurrent Testing Strategies

**Context:** `UnifiedCache._lock` (RLock) is used in ~18 management methods but NOT in `put()`/`get()`. Thread safety is documented as limited. No concurrency tests exist despite the library supporting multi-process access via SQLite WAL mode.

### Table Stakes (Must-Do)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Thread-safety smoke tests for concurrent `put()`/`get()` | No tests verify behavior under concurrent access. Users may assume thread safety because a lock exists. Need to establish the behavioral contract: what works, what doesn't, what corrupts. | **Medium** | Use `concurrent.futures.ThreadPoolExecutor` with 4-8 workers. Test scenarios: concurrent puts to different keys, concurrent gets of same key, concurrent put+get of same key, concurrent put+evict. Assert no crashes, no data corruption. Don't assert ordering — that's not the contract. |
| Document actual thread-safety guarantees per backend | JSON backend: has `threading.Lock()` internally but `UnifiedCache` doesn't acquire `_lock` for `put()`/`get()`. SQLite backend: WAL mode provides concurrent reads + serialized writes. PostgreSQL: inherently concurrent. | **Low** | Write explicit thread-safety docs per backend. Tests should match the documented guarantees, not aspirational ones. |

### Differentiators (Nice-to-Have)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Stress tests with high contention | 50+ concurrent operations testing for deadlocks, resource exhaustion, file handle leaks. | **Medium** | Use `pytest-timeout` to catch deadlocks. Parameterize across backends. On Windows, be careful with file handle limits. |
| Multi-process safety tests | Fork `multiprocessing.Process` workers against the same `cache_dir`. Critical for SQLite WAL correctness. | **Medium-High** | `multiprocessing` + `pytest` interaction is tricky (fixture sharing, temp dir cleanup). Use `tmp_path_factory` at session scope. Skip on Windows if `fork()` not available (use `spawn` instead). |
| Key rotation scenario tests | Delete key file, restart cache, verify old entries produce warnings (not crashes) on verification. | **Low-Medium** | Straightforward test: create cache, sign entries, delete key file, re-instantiate, `get()` should return data with warning (not error). |

### Anti-Features (Avoid)

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Adding full thread safety to `put()`/`get()` | Acquiring `_lock` in `put()`/`get()` would serialize all operations, destroying performance for the common case (single-threaded use). Backend-level locking is more granular. | Document that thread safety is provided at the backend level. `UnifiedCache`-level lock is for management operations only. |
| Using `asyncio`-based concurrency tests | Cacheness is synchronous. Testing with `asyncio` would require `loop.run_in_executor()` wrapping, adding complexity without testing real concurrency. | Use `threading`/`multiprocessing` for concurrency tests. Async tests belong in a future async milestone. |
| Benchmarking under concurrency | Performance benchmarking is a separate concern from correctness testing. | Keep concurrency tests focused on correctness (no corruption, no crashes). Benchmarks go in `benchmarks/`. |

### Complexity Assessment

- **Thread-safety smoke tests:** Medium. Test design is straightforward; flaky test prevention (race conditions in assertions) is the challenge.
- **Documentation:** Low. Synthesize existing knowledge from `TROUBLESHOOTING.md` and code comments.
- **Stress tests:** Medium. More test infrastructure than logic.
- **Multi-process tests:** Medium-high. Cross-platform process management is tricky.
- **Key rotation tests:** Low-medium. Linear test scenario.

### Dependencies

- Thread-safety tests are **independent** of code changes — they test existing behavior.
- Key rotation tests depend on understanding the security module — coordinate with security hardening.
- Multi-process tests require SQLite WAL mode — depends on SQLite backend being available (it is, by default).
- Concurrency tests should be written **before** any `core.py` decomposition (establish baseline behavior, then verify it's preserved after refactoring).

---

## 5. Handler/Plugin Ordering Robustness

**Context:** `HandlerRegistry` iterates handlers in registration order, using the first `can_handle()` match. Adding a handler that matches broadly (e.g., anything with `.dtype`) can shadow existing handlers. Priority is implicit (registration order) with optional explicit ordering via `config.handlers.handler_priority`. The TensorFlow handler uses elaborate early-return checks to avoid matching NumPy arrays.

### Table Stakes (Must-Do)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Document handler ordering constraints | Current ordering relies on implicit knowledge: Series before DataFrame (subtype match), DataFrame before Array (both have `.dtype`), Array before Object (fallback). This is documented only in code comments, not in user-facing docs. | **Low** | Add a "Handler Priority" section to `docs/PLUGIN_DEVELOPMENT.md` or create `docs/HANDLER_ORDERING.md`. Include the full default priority chain with rationale. |
| Add handler conflict detection warnings | When `register_handler()` is called, check if the new handler's `can_handle()` overlaps with existing handlers by testing against a standard set of probe values. Log a warning if overlap detected. | **Medium** | Probe set: `np.array([1])`, `pd.DataFrame()`, `pd.Series()`, `pl.DataFrame()`, `pl.Series()`, `b"bytes"`, `"string"`, `42`, `{"dict": 1}`. Run each through old and new handler — if both match, warn. **Don't block registration** — just warn. |
| Guard ObjectHandler as always-last | `ObjectHandler.can_handle()` accepts anything pickleable — it MUST be last. Currently enforced by registration order convention. Add an explicit check in `register_handler()` that prevents registering after ObjectHandler unless `force=True`. | **Low** | Check `self.handlers[-1].__class__.__name__ == "ObjectHandler"` and insert before it. Or tag ObjectHandler with `is_fallback = True` attribute. |

### Differentiators (Nice-to-Have)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Explicit numeric priority on handlers | Replace implicit ordering with explicit `priority: int` attribute on each handler class. `HandlerRegistry` sorts by priority. Lower number = higher priority. | **Medium** | Already partially supported via `config.handlers.handler_priority` (name-based ordering). Adding numeric priority to the handler class itself would be more robust. Risk: existing custom handlers don't have a `priority` attribute — need a default. |
| Handler capability introspection | Add `handler.supported_types() -> list[type]` method for explicit type declarations instead of relying solely on `can_handle()` runtime checks. | **Medium** | Would allow static analysis of handler overlap without needing probe values. But some handlers (ObjectHandler) genuinely accept "anything," making this less useful. |
| `can_handle()` type narrowing for Series/DataFrame | Currently `PolarsSeriesHandler.can_handle()` must run before `PolarsDataFrameHandler.can_handle()` because `pl.Series` is a valid DataFrame-like. Add explicit negative checks: `DataFrameHandler.can_handle()` returns `False` for `Series` types. | **Low-Medium** | Makes ordering less fragile. Each handler explicitly rejects types it shouldn't handle, rather than relying on registration order. |

### Anti-Features (Avoid)

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Full plugin discovery system (entry points, stevedore) | Over-engineering for 8 built-in handlers. Entry point discovery adds import-time overhead and a dependency. | Keep `register_handler()` API. Users explicitly register custom handlers. |
| Handler chain-of-responsibility with `next_handler` | GoF pattern that adds complexity. Current linear scan is simple and fast for 8 handlers. | Keep linear `for handler in self.handlers` scan. O(n) with n=8 is not a problem. |
| Automatic handler ordering via topological sort on type dependencies | Complex to implement, hard to debug, brittle when type hierarchies change. | Use explicit numeric priority. Manual ordering by humans is fine for <20 handlers. |

### Complexity Assessment

- **Documentation:** Low. Write it down.
- **Conflict detection warnings:** Medium. Needs a good probe set and must not break existing `register_handler()` call sites.
- **ObjectHandler guard:** Low. Small check in `register_handler()`.
- **Numeric priority:** Medium. Refactor internal representation; maintain backward compat.
- **Type narrowing in `can_handle()`:** Low-medium per handler, but touches all handlers.

### Dependencies

- Documentation is **independent** — can be done anytime.
- Conflict detection depends on having the handler types and probe values available — **independent** of other handler changes.
- ObjectHandler guard is **independent**.
- Numeric priority refactor should happen **after** `handlers.py` split into package (easier to modify individual handler files).
- Type narrowing in `can_handle()` should happen **after** `handlers.py` split.

---

## Feature Dependencies (Cross-Cutting)

```
handlers.py split ──┐
                    ├──→ core.py decomposition ──→ exception narrowing (core.py)
metadata.py split ──┘

exception hierarchy additions ──→ exception narrowing (all files)

concurrency tests ──→ (baseline established) ──→ core.py decomposition ──→ concurrency tests (re-run)

security: blob hash default-on ─── (independent)
security: key fallback config ─── (independent)
security: HKDF namespace keys ──→ key rotation tests

handler ordering docs ─── (independent)
handler conflict detection ──→ after handlers.py split
handler ObjectHandler guard ─── (independent)
```

### Recommended Phase Ordering

Based on dependencies and risk:

1. **Phase 1: Low-risk structural splits** — `handlers.py` → package, `metadata.py` → package (parallel, independent)
2. **Phase 2: Testing baseline** — Concurrency tests, key rotation tests (parallel, test existing behavior before changing it)
3. **Phase 3: Exception hierarchy + narrowing** — Add missing exception types, then narrow catches in `handlers.py`, `decorators.py`, `custom_metadata.py` (lower-risk files first)
4. **Phase 4: Security hardening** — Blob hash default-on, key fallback config, HKDF namespace keys
5. **Phase 5: core.py decomposition** — Highest-risk change, done last with full test baseline in place
6. **Phase 6: Exception narrowing in core.py** — After decomposition, narrower files to audit
7. **Phase 7: Handler ordering robustness** — After handlers.py split, add conflict detection and ObjectHandler guard

### MVP Recommendation

Prioritize for maximum reliability impact with minimum risk:

1. **handlers.py split** — Low risk, immediate maintainability win
2. **metadata.py split** — Low risk, immediate maintainability win
3. **Exception hierarchy additions** — Low effort, enables later narrowing
4. **Blob hash verification default-on** — Direct security improvement, low complexity
5. **Concurrency smoke tests** — Establishes behavioral baseline
6. **core.py decomposition** — Highest impact but highest risk — do last with full safety net

**Defer if time-constrained:**
- Windows ACL implementation (document limitation instead)
- Multi-process safety tests (complex infrastructure)
- Numeric handler priority (current implicit ordering works)
- Structured error context on all raise sites (incremental, no deadline)

---

## Sources

- Direct codebase analysis: `src/cacheness/core.py`, `handlers.py`, `metadata.py`, `security.py`, `error_handling.py`, `interfaces.py`
- Project audit: `.planning/codebase/CONCERNS.md` (2026-04-02)
- Project definition: `.planning/PROJECT.md`
- Python standard library documentation for `hmac`, `hashlib`, `threading`, `concurrent.futures` — HIGH confidence
- Python mixin patterns: community consensus from Real Python, Python docs, PyCon talks — HIGH confidence (well-established pattern)
- OWASP guidance on caching security: blob integrity verification is standard practice — MEDIUM confidence (general guidance applied to specific context)
