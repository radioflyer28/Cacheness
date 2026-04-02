# Domain Pitfalls — Cacheness Cleanup & Hardening

**Domain:** Python library refactoring & hardening (existing codebase with 1,427 tests, stable public API)
**Researched:** 2026-04-02
**Overall confidence:** HIGH (pitfalls derived from codebase audit + established Python packaging patterns)

---

## Critical Pitfalls

Mistakes that cause test suite regressions, broken downstream consumers, or security vulnerabilities that are harder to fix than the original problem.

---

### P1: Mixin MRO and `super()` Chain Breakage During core.py Decomposition

**Severity:** HIGH
**Applies to:** core.py decomposition (mixins)
**Description:** When extracting `UnifiedCache` methods into mixins (`_StorageModeMixin`, `_VerificationMixin`, `_StatisticsMixin`, `_CustomMetadataMixin`), Python's Method Resolution Order (MRO) determines which `super()` calls reach which class. If any mixin calls `super().__init__()` or `super().some_method()` and the MRO doesn't include a class that defines that method, you get `AttributeError` at runtime — not at import time. This is invisible until the specific code path executes.

**Warning signs:**
- Tests pass individually but fail when run together (MRO differs depending on import order in some edge cases)
- `AttributeError: 'UnifiedCache' object has no attribute '_signer'` in mixin methods that assume another mixin has initialized state
- Mixins that access `self._metadata_backend` or `self._blob_store` before `UnifiedCache.__init__()` has run

**Prevention:**
1. **Do NOT use `super().__init__()` in mixins.** Mixins should be method-only — no `__init__` in any mixin class. All initialization stays in `UnifiedCache.__init__()`.
2. **Explicit attribute access, not inheritance.** Mixins access `self.xxx` assuming `UnifiedCache.__init__` has already set it up. Document this contract at the top of each mixin.
3. **MRO sanity test:** Add a test that asserts `UnifiedCache.__mro__` is exactly the expected tuple. If someone reorders the base classes, this test catches it immediately.
4. **Run the full test suite after each mixin extraction** — don't batch all extractions before testing.

**Cacheness-specific risk:** `UnifiedCache` has 67+ methods and accesses `self._metadata_backend`, `self._blob_store`, `self._handler_registry`, `self._signer`, `self._lock`, and `self._config` extensively. Mixins will all share this state. A mixin that accidentally shadows one of these attributes (via a local assignment or property) will silently break other mixins.

---

### P2: Circular Imports When Splitting metadata.py into a Package

**Severity:** HIGH
**Applies to:** metadata.py decomposition (package split)
**Description:** `metadata.py` currently defines `MetadataBackend` (ABC), `CachedMetadataBackend` (wrapper), `JsonBackend`, `SqliteBackend`, and the `create_metadata_backend()` factory — all in one file. Splitting into `metadata/__init__.py`, `metadata/base.py`, `metadata/json_backend.py`, `metadata/sqlite_backend.py` introduces circular import risk because:
- The factory function imports all backends
- Backends import the base class
- `core.py` imports both the factory and `DEFAULT_NAMESPACE` from metadata
- `__init__.py` (package level) re-exports everything

**Warning signs:**
- `ImportError: cannot import name 'MetadataBackend' from partially initialized module`
- Tests work when run individually but fail with `import cacheness` at the top
- `from cacheness.metadata import JsonBackend` works but `from cacheness import JsonBackend` doesn't

**Prevention:**
1. **Base class in its own module** (`metadata/base.py`) with zero imports from sibling modules.
2. **Factory function uses lazy imports** — `create_metadata_backend()` imports backend classes inside the function body, not at module level. This is already the pattern for the security module (see `core.py` line 356).
3. **`__init__.py` re-exports with explicit imports** — don't use `from .json_backend import *`. List every name.
4. **Test the import chain:** `python -c "from cacheness.metadata import JsonBackend, SqliteBackend, MetadataBackend, create_metadata_backend"` as an automated check.
5. **Preserve the SQLAlchemy ORM model classes** (`CacheEntry`, `CacheStats`, `CacheNamespace`, `Base`) in a shared module (e.g., `metadata/models.py`) — both SQLite and PostgreSQL backends use them.

**Cacheness-specific risk:** The SQLAlchemy `Base = declarative_base()` is defined at module level in `metadata.py` (line ~200). Tests import `Base` directly (`from cacheness.metadata import Base`). If `Base` moves to `metadata/models.py`, every test that imports it breaks unless `metadata/__init__.py` re-exports it.

---

### P3: Re-Export Breakage — Tests and Examples Import Internal Paths

**Severity:** HIGH
**Applies to:** All decomposition work (core.py, metadata.py, handlers.py)
**Description:** The test suite and examples import directly from internal module paths:
- `from cacheness.core import UnifiedCache, CacheConfig, _normalize_function_args`
- `from cacheness.metadata import JsonBackend, SqliteBackend, Base, DEFAULT_NAMESPACE`
- `from cacheness.handlers import ArrayHandler, ObjectHandler, BytesHandler, HandlerRegistry`

After decomposition, these paths change (e.g., `cacheness.metadata.json_backend.JsonBackend`). If the old paths aren't preserved via re-exports in `__init__.py`, tests break silently — Python's import system won't warn you, it just raises `ImportError`.

**Warning signs:**
- `ImportError` in tests that worked before the split
- Grep reveals 50+ import sites across tests and examples that reference the old module paths
- `from cacheness.metadata import Base` stops working after `Base` moves to `metadata/models.py`

**Prevention:**
1. **Before splitting:** Run `grep -rn "from cacheness\.\(core\|metadata\|handlers\) import" tests/ examples/` and catalog every import path. This is the backwards-compatibility contract.
2. **After splitting:** Every old import path must still work via re-exports in the new package's `__init__.py`.
3. **Add an import compatibility test:** A single test file that imports every public and semi-public name from the old paths and asserts they resolve.
4. **Don't move private functions that tests import.** `_normalize_function_args` is imported by `test_cross_system_compatibility.py`. Either keep it importable from `cacheness.core` or update the test.

---

### P4: Enabling Blob Hashing by Default Breaks Existing Caches

**Severity:** HIGH
**Applies to:** Security hardening (blob content hashing)
**Description:** Currently, HMAC signing covers metadata fields but NOT blob file content. The project plans to make blob content hashing "default-on." If existing caches were created without blob hashes, enabling hashing by default means `verify_cache_integrity()` will report every pre-existing entry as corrupt — because there's no stored hash to compare against.

**Warning signs:**
- Users upgrade Cacheness and `verify_cache_integrity()` reports 100% corruption
- `get()` with `delete_on_error=True` (the default) auto-deletes entries that fail verification
- Production caches silently emptied on library upgrade

**Prevention:**
1. **"Hash if present" verification.** Verify blob hash only when a stored hash exists. Missing hash = skip blob verification (with a warning). This preserves backward compatibility.
2. **Write-path only.** Enable hashing on `put()` for new entries. Existing entries get hashed when they're next written/updated.
3. **Migration utility.** Provide `cache.rehash_entries()` that backfills blob hashes for existing entries without re-serializing data.
4. **Versioned integrity policy.** Store the integrity policy version in metadata. Entries created under policy v1 (no blob hash) are verified under v1 rules.
5. **Never make `get()` fail on missing hashes.** The `delete_on_error=True` default makes this extremely dangerous.

---

### P5: Narrowing `except Exception` Catches Exposes Previously-Swallowed Failures

**Severity:** HIGH
**Applies to:** Error handling cleanup
**Description:** The 30+ `except Exception` catches in `core.py` exist because the code has evolved to handle a wide variety of failure modes — file permission errors, serialization failures, database connection problems, handler type mismatches, corrupt metadata, etc. Narrowing these to specific types (e.g., `except (OSError, pickle.UnpicklingError)`) will cause previously-swallowed exceptions to propagate to the caller. This is intentional — but some of those catches serve as safety nets for failure modes you haven't encountered yet.

**Warning signs:**
- `TypeError` or `ValueError` propagating from deep in a handler when a user passes unexpected data types
- `sqlite3.OperationalError` or `sqlalchemy.exc.OperationalError` escaping when the database is locked
- Users report "Cacheness used to silently handle this, now it crashes"
- `_init_auto_backend()` (line ~282) catches `Exception` to fall back from SQLite to JSON — narrowing this wrong breaks the auto-detection

**Prevention:**
1. **Audit each catch individually.** Don't batch-narrow. For each `except Exception`, trace every code path in the try block and identify which specific exceptions can actually occur.
2. **Preserve intentional safety nets.** Some catches are defensive — `_init_auto_backend()` must catch broadly because SQLAlchemy can raise many different exception types during initialization. Add a comment explaining why it's broad.
3. **Add `except Exception` logging before removing.** Before narrowing, temporarily add `logger.warning(f"Caught {type(e).__name__}: {e}")` to each broad catch. Run the full test suite. This reveals which exception types actually occur.
4. **Create a custom exception hierarchy.** Define `CacheError`, `CacheIOError`, `CacheSerializationError`, `CacheIntegrityError`, `CacheBackendError`. Wrap caught exceptions in the appropriate custom type. This gives callers something stable to catch.
5. **Don't narrow catches inside `get()`.** The `get()` method has `delete_on_error=True` behavior that must handle ANY deserialization failure gracefully. Narrowing its catches risks leaving corrupted entries in place instead of cleaning them up.

---

## Moderate Pitfalls

Mistakes that cause significant rework, test instability, or subtle bugs — but won't break production users immediately.

---

### P6: Test-Implementation Coupling Across the 1,427-Test Suite

**Severity:** MEDIUM
**Applies to:** All decomposition and refactoring work
**Description:** Tests that assert internal state (`cache._metadata_backend`, `cache._handler_registry`, `cache._signer`) or mock internal methods (`patch.object(UnifiedCache, '_write_blob')`) break when the internal structure changes. With 1,427 tests, even 5% coupling to internals means ~70 tests need updating during decomposition.

**Warning signs:**
- Tests use `patch.object(UnifiedCache, '_some_private_method')` and the method moves to a mixin
- Tests directly access `cache._metadata_backend._data` (JSON backend internals)
- Tests assert that `isinstance(cache._metadata_backend, SqliteBackend)` — still valid, but path import changes
- Fault injection tests (`test_fault_injection.py`) mock internal methods by name

**Prevention:**
1. **Inventory mock targets before refactoring.** `grep -rn "patch.object\|patch(" tests/` to find every mock site. These are the fragile points.
2. **When moving methods to mixins, keep the method name identical.** `UnifiedCache.verify_cache_integrity` should still exist even if the implementation lives in `_VerificationMixin`. The MRO handles this automatically.
3. **Mocks should target the lowest stable interface.** Mock at the backend level (`patch.object(JsonBackend, 'save_entry')`) rather than at the `UnifiedCache` level, because backends aren't being restructured.
4. **Fix coupling in batches per concern.** Don't try to decouple all 1,427 tests at once. Fix the tests that break from each specific refactoring step.

---

### P7: handlers.py Split Breaks Registration Order and Type Detection

**Severity:** MEDIUM
**Applies to:** handlers.py decomposition (package split)
**Description:** `HandlerRegistry` iterates handlers in registration order and returns the first `can_handle()` match. The registration order is currently implicit — it's the order classes appear in `handlers.py` and get registered in `_register_default_handlers()`. Splitting handlers into separate files means the registration order is determined by import order in `handlers/__init__.py`, which is less visible and easier to accidentally change.

**Warning signs:**
- After split, a Polars DataFrame is handled by `ObjectHandler` instead of `PolarsDataFrameHandler` (because `ObjectHandler` was registered first)
- Tests pass but serialization format changes silently (parquet → pickle) because a different handler matched
- TensorFlow handler's elaborate `can_handle()` early-return checks stop working because handler ordering changed

**Prevention:**
1. **Explicit registration with numbered priority.** Change `_register_default_handlers()` to register handlers with explicit priority values, not insertion order. e.g., `registry.register(PolarsDataFrameHandler, priority=10)`.
2. **If not adding priorities:** Keep `_register_default_handlers()` in `handlers/__init__.py` with a comment block documenting exactly why each handler is registered in that order.
3. **Add a handler ordering test.** Assert that `registry.get_handler_info()` returns handlers in exactly the expected order. Any reordering breaks this test.
4. **Add a type-dispatch test.** For each supported type (ndarray, DataFrame, Series, bytes, dict, etc.), assert which handler is selected. This catches silent handler substitution.

---

### P8: Per-Namespace Key Derivation Invalidates Existing Signed Entries

**Severity:** MEDIUM
**Applies to:** Security hardening (per-namespace key derivation)
**Description:** Adding HKDF-based per-namespace key derivation (master key + namespace ID → derived key) means the signing key for each namespace changes. All existing entries in non-default namespaces were signed with the master key directly. After the change, signature verification for those entries will fail because the verification now uses the derived key.

**Warning signs:**
- Entries in the default namespace verify fine (if the derivation includes an identity case for default), but all other namespaces report signature failures
- `verify_cache_integrity()` shows 100% signature failures in non-default namespaces after upgrade
- Using `get()` with `delete_on_error=True` silently purges all entries in custom namespaces

**Prevention:**
1. **Fallback verification.** Try derived key first, then fall back to master key for verification. If master key succeeds, re-sign with derived key (lazy migration).
2. **Migration function.** `cache.migrate_signing_keys()` re-signs all entries with namespace-derived keys.
3. **Version the signing scheme.** Store `signing_version: 1` or `signing_version: 2` in entry metadata. Version 1 = master key, version 2 = derived key.
4. **Default namespace uses master key directly.** For backward compatibility, derive only for non-default namespaces.

---

### P9: Windows ACL Implementation Creates Platform-Conditional Security

**Severity:** MEDIUM
**Applies to:** Security hardening (Windows key permissions)
**Description:** Replacing the no-op `chmod(0o600)` on Windows with actual ACLs (via `win32security` or `icacls` subprocess) introduces a platform-conditional code path that's hard to test in CI. `win32security` requires `pywin32` (heavy optional dependency). `icacls` is a subprocess call that can fail silently or behave differently across Windows versions.

**Warning signs:**
- `pywin32` not installed → no error, key file remains world-readable
- `icacls` command syntax differs between Windows Server and Windows 10/11
- Tests pass on developer machine but fail on CI runner (different Windows version or permissions model)
- Security feature is silently absent on half the target platform

**Prevention:**
1. **Prefer `icacls` over `pywin32`** — it's universally available on Windows and doesn't require an extra dependency.
2. **Test ACL setting works.** After applying ACLs, verify by reading them back (`icacls <file>` output parsing).
3. **Document the limitation clearly** if you can't verify ACLs in CI. "On Windows, key file permissions are best-effort" is acceptable if documented.
4. **Don't make this a hard requirement.** If ACL setting fails, log a warning and continue (matching the existing `chmod` behavior). Don't crash.
5. **Consider documenting this as a known limitation** and advising Windows users to set permissions manually, rather than building fragile automation.

---

### P10: Premature Abstraction in Decomposition

**Severity:** MEDIUM
**Applies to:** All decomposition work
**Description:** The temptation during decomposition is to refactor while splitting — introducing new ABCs, protocols, or indirection layers "while we're in there." This transforms a low-risk structural change (moving code to new files) into a high-risk behavioral change (new abstractions that all existing code must now conform to).

**Warning signs:**
- Creating `Protocol` classes or ABCs for things that have exactly one implementation
- Introducing a "plugin system" for handlers when the current list-based registry works fine
- Adding generic type parameters to class hierarchies that don't need them
- Creating `utils.py` or `helpers.py` during decomposition (usually means extracting code that should stay where it is)

**Prevention:**
1. **Phase 1: Move code only.** No new abstractions, no renaming, no refactoring. Just cut-paste to new files with re-exports.
2. **Phase 2: Clean up imports.** After all code is moved and tests pass, clean up internal imports.
3. **Phase 3 (only if needed): Refactor.** Only add abstractions if they solve a concrete problem that manifests after the split.
4. **Rule of thumb:** If you're adding a class that doesn't correspond to an existing class in the monolith, you're adding scope.

---

### P11: Concurrent Test Flakiness from Shared State

**Severity:** MEDIUM
**Applies to:** Test refactoring, adding thread safety tests
**Description:** Adding concurrency tests for `put()`/`get()` requires shared temp directories, shared database files, and shared `UnifiedCache` instances across threads. These tests are inherently flaky on CI due to timing, file system latency (especially on Windows), and SQLite lock contention.

**Warning signs:**
- Tests pass 95% of the time locally but fail intermittently on CI
- `sqlite3.OperationalError: database is locked` in concurrent tests
- File cleanup failures on Windows (`PermissionError: [WinError 32]` — file in use by another thread)
- Tests that use `time.sleep()` for synchronization

**Prevention:**
1. **Use `threading.Barrier` and `threading.Event` for synchronization** — not sleep.
2. **Each concurrent test gets its own temp directory** — no sharing between test functions.
3. **Test concurrency at the backend level, not the cache level.** The backends (SQLite WAL, JSON threading.Lock) are where thread safety actually lives.
4. **Mark concurrent tests with `@pytest.mark.xdist_group("concurrent")`** to prevent pytest-xdist from running them in parallel with other tests that might interfere.
5. **Set tight iteration counts.** 10 concurrent threads × 10 operations is enough to detect races. 100 × 1000 causes timeout flakes.

---

## Minor Pitfalls

Issues that cause friction, minor bugs, or tech debt — but are straightforward to fix when noticed.

---

### P12: `__init__.py` Re-Export Bloat After Package Split

**Severity:** LOW
**Applies to:** All decomposition work
**Description:** After splitting `metadata.py` into a package, the `metadata/__init__.py` must re-export every name that was previously importable from `cacheness.metadata`. It's easy to miss names, especially for rarely-used exports like `NamespaceInfo`, `validate_namespace_id`, or `DEFAULT_NAMESPACE`.

**Prevention:**
- Before splitting, `dir(cacheness.metadata)` in a Python REPL and record the full list of public names.
- After splitting, assert `set(dir(cacheness.metadata))` is a superset of the pre-split set.

---

### P13: Exception Hierarchy Designed Too Late

**Severity:** LOW
**Applies to:** Error handling cleanup
**Description:** Exception narrowing works best when there's a target hierarchy to narrow INTO. Designing the hierarchy after narrowing catches means you'll restructure the exception types twice.

**Prevention:**
- Design the exception hierarchy FIRST: `CacheError` → `CacheIOError`, `CacheSerializationError`, `CacheIntegrityError`, `CacheBackendError`, `CacheKeyError`.
- Then narrow each `except Exception` to catch the appropriate custom type.
- This also gives downstream users a stable exception API to code against.

---

### P14: `conftest.py` Fixture Interdependencies Break After Refactoring

**Severity:** LOW
**Applies to:** Test refactoring
**Description:** `conftest.py` fixtures may internally reference module paths or create backend instances in ways that assume the pre-split module structure. Fixtures that create `JsonBackend(...)` or `SqliteBackend(...)` directly need their import paths updated.

**Prevention:**
- Audit `conftest.py` before refactoring. List every import and every fixture that creates a backend or handler directly.
- Update fixtures immediately after the decomposition step, before running tests.

---

### P15: In-Memory Key Fallback Remains Silent After "Configurable" Change

**Severity:** LOW
**Applies to:** Security hardening (configurable key fallback)
**Description:** Making the in-memory key fallback "configurable" (fail-loud option) risks leaving the default as-is — silent fallback. If the purpose is to improve security posture, the default should be fail-loud for new installations while preserving silent fallback for existing configs.

**Prevention:**
- Default to fail-loud (`raise CacheSecurityError(...)`) when signing is explicitly enabled.
- Default to silent fallback when signing is auto-detected (backward compatibility).
- Log at WARNING level in all cases, not just when falling back.

---

## Integration Pitfalls

How changes in one area interact with changes in another. These are the most commonly missed pitfalls in multi-concern refactoring milestones.

---

### P16: Decomposition + Exception Narrowing Interaction

**Severity:** HIGH
**Applies to:** core.py decomposition + error handling cleanup (cross-cutting)
**Description:** If you decompose core.py into mixins AND narrow exception catches simultaneously, you lose the ability to isolate which change caused a test failure. A `TypeError` might propagate because (a) the exception catch was narrowed, or (b) a mixin's method can't find an attribute due to MRO issues. Debugging becomes a two-variable problem.

**Prevention:**
- **Decomposition first, exception narrowing second.** Complete the structural split, get all 1,427 tests passing, THEN narrow exception catches.
- Never do both in the same phase.

---

### P17: Security Hardening + Decomposition Interaction

**Severity:** MEDIUM
**Applies to:** Security hardening + core.py decomposition (cross-cutting)
**Description:** Security methods (`sign_entry`, `verify_signature`, `verify_cache_integrity`) may move to a `_VerificationMixin` while also being modified for blob hashing changes. If the decomposition happens concurrently with security hardening, you get merge conflicts in code that's both moving and changing.

**Prevention:**
- **Decompose first.** Move security-related methods to the mixin without modifying their behavior.
- Then apply security hardening changes to the methods in their new location.
- This makes git blame still useful for understanding what changed vs. what moved.

---

### P18: Handler Split + Handler Ordering Guardrails Interaction

**Severity:** MEDIUM
**Applies to:** handlers.py decomposition + handler type detection robustness (cross-cutting)
**Description:** Adding ordering guardrails (priority-based registration, type-dispatch validation) while simultaneously splitting handlers into separate files creates two moving targets. If the guardrail relies on class introspection (`__subclasses__()`, registry inspection), the module boundary changes can affect class discovery.

**Prevention:**
- Split handlers first (preserving exact registration order).
- Then add guardrails as a separate change.

---

## Phase Mapping

Which pitfalls should be addressed or watched for in which milestone area:

| Milestone Area | Critical Pitfalls | Moderate Pitfalls | Minor Pitfalls |
|---|---|---|---|
| **core.py decomposition** | P1 (MRO), P3 (re-exports), P16 (decomp+exceptions) | P6 (test coupling), P10 (premature abstraction), P17 (decomp+security) | P12 (re-export bloat), P14 (fixture paths) |
| **metadata.py package split** | P2 (circular imports), P3 (re-exports) | P10 (premature abstraction) | P12 (re-export bloat), P14 (fixture paths) |
| **handlers.py package split** | P3 (re-exports) | P7 (registration order), P10 (premature abstraction), P18 (split+guardrails) | P12 (re-export bloat) |
| **Error handling cleanup** | P5 (narrowing catches) | P16 (decomp+exceptions) | P13 (hierarchy timing) |
| **Blob content hashing** | P4 (breaks existing caches) | — | — |
| **Windows key permissions** | — | P9 (platform-conditional security) | — |
| **Per-namespace key derivation** | — | P8 (invalidates signed entries) | P15 (fallback behavior) |
| **Configurable key fallback** | — | — | P15 (fallback behavior) |
| **Thread safety tests** | — | P11 (concurrent test flakiness) | — |
| **Handler type detection guardrails** | — | P7 (registration order), P18 (split+guardrails) | — |

## Recommended Phase Ordering (Pitfall-Informed)

Based on the interaction pitfalls (P16, P17, P18), the safest phase ordering is:

1. **Decomposition first** (core.py → mixins, metadata.py → package, handlers.py → package) — structural changes only, no behavioral changes
2. **Error handling cleanup second** — now that code is in its final locations, narrow catches without conflating with move-related failures
3. **Security hardening third** — blob hashing, key derivation, Windows permissions, fallback behavior — applied to the decomposed, stabilized codebase
4. **Test additions last** — thread safety tests, key rotation tests — the codebase is stable and the tests won't need updating due to subsequent structural changes

This ordering minimizes the two-variable debugging problem (P16) and ensures git blame remains useful (P17).

---

## Sources

- Cacheness codebase audit (`.planning/codebase/CONCERNS.md`, 2026-04-02)
- Python MRO documentation (C3 linearization): [Python Data Model — MRO](https://docs.python.org/3/howto/mro.html)
- Python import system (circular imports): [Python Import System](https://docs.python.org/3/reference/import.html)
- HMAC-SHA256 compatibility patterns: established cryptographic library migration practices
- pytest-xdist concurrent test isolation: established pytest patterns

**Confidence:** HIGH — pitfalls are derived from direct codebase analysis (grep results, import chains, exception sites) rather than generic advice. The codebase-specific risks (SQLAlchemy `Base` re-export, `_normalize_function_args` test import, `delete_on_error` interaction with blob hashing) are concrete and verified against the source.

---

*Pitfalls audit: 2026-04-02*
