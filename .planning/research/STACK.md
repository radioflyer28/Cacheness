# Stack Research: Cleanup & Hardening Milestone

**Project:** Cacheness
**Researched:** 2026-04-02
**Overall Confidence:** HIGH
**Research Mode:** Ecosystem (subsequent milestone — stack delta only)

## Executive Summary

The cleanup & hardening milestone requires **zero new runtime dependencies**. Every capability needed — HKDF key derivation, Windows file permissions, concurrent testing, exception narrowing, code decomposition — can be achieved with Python's stdlib and the existing dependency set.

This is the ideal outcome for a hardening milestone: reducing risk by *not* adding attack surface.

## Stack Additions by Area

### 1. Code Decomposition (core.py → mixins, metadata.py → package, handlers.py → package)

**New dependencies needed: NONE**

| Capability | Source | Already Available |
|------------|--------|-------------------|
| Mixin pattern | Python stdlib (`abc`, class MRO) | ✓ `abc` used in `interfaces.py` |
| Package structure | Python stdlib (`__init__.py`, relative imports) | ✓ `src/cacheness/storage/` already uses this pattern |
| Re-exports for backward compat | Python `__init__.py` with `__all__` | ✓ Standard Python |

**Rationale:** Decomposition is purely structural refactoring. The `storage/` subdirectory already demonstrates the package pattern (backends in `blob_backends.py`, store in `blob_store.py`, registry in `backend_registry.py`). Apply the same pattern to `metadata.py` and `handlers.py`. For `core.py`, extract mixins into a `core/` package with `__init__.py` re-exporting `UnifiedCache`.

**What NOT to add:**
- ~~`importlib` plugins~~ — Cacheness uses direct imports, not plugin discovery. Over-engineering.
- ~~Abstract factory frameworks~~ — The existing `abc.ABC` + registry pattern is sufficient.

---

### 2. Security Hardening

#### 2a. HKDF Key Derivation (per-namespace keys from master key)

**New dependencies needed: NONE**

| Capability | Source | Already Available |
|------------|--------|-------------------|
| HKDF Extract + Expand (RFC 5869) | Python stdlib `hmac` + `hashlib` | ✓ Both used in `security.py` |

**Rationale:** HKDF is a two-step process (extract → expand) built entirely on HMAC. The implementation is ~15 lines of Python using `hmac.new()` and `hashlib`. Verified working with Python 3.12 stdlib — produces deterministic, namespace-isolated 32-byte keys from a master key.

**Why NOT `cryptography` package:**
- `cryptography` (v46.0.3) is a transitive dependency via `moto[s3]` in dev deps, but is **not** a runtime dependency.
- Adding it as a core dep would introduce a C-extension build requirement, increasing install complexity and binary size significantly.
- HKDF is simple enough that stdlib implementation is correct, auditable, and has zero risk of version conflicts.
- The existing `security.py` already uses `hmac` and `hashlib` for HMAC-SHA256 signing — HKDF fits naturally.

**Implementation approach:**
```python
# In security.py — ~15 lines total
def hkdf_derive_key(master_key: bytes, info: bytes, length: int = 32) -> bytes:
    """RFC 5869 HKDF-SHA256: derive namespace-specific key from master key."""
    # Extract: PRK = HMAC-SHA256(salt=None, IKM=master_key)
    prk = hmac.new(b'\x00' * 32, master_key, hashlib.sha256).digest()
    # Expand: OKM = T(1) || T(2) || ... (truncated to length)
    n = (length + 31) // 32
    okm, t = b'', b''
    for i in range(1, n + 1):
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        okm += t
    return okm[:length]
```

#### 2b. Blob Content Hashing (default-on integrity)

**New dependencies needed: NONE**

| Capability | Source | Already Available |
|------------|--------|-------------------|
| Content hashing | `xxhash` (fast) or `hashlib.sha256` (crypto) | ✓ `xxhash>=3.5.0` is a core dep |
| File hashing | `src/cacheness/file_hashing.py` | ✓ Already implements parallel file hashing |

**Rationale:** The infrastructure already exists. `file_hashing.py` uses `ProcessPoolExecutor` with `xxhash` for fast file hashing. The change is making this default-on in `SecurityConfig` rather than opt-in. No new code paths, just a config default change.

#### 2c. Windows File Permissions (signing key protection)

**New dependencies needed: NONE**

| Capability | Source | Already Available |
|------------|--------|-------------------|
| Windows ACLs | `subprocess` + `icacls` (Windows built-in) | ✓ `subprocess` is stdlib |
| Platform detection | `sys.platform` / `os.name` | ✓ stdlib |

**Rationale:** Two viable approaches exist:

| Approach | Pros | Cons |
|----------|------|------|
| **`icacls` via subprocess** (RECOMMENDED) | Zero deps, Windows built-in since Vista, simple invocation | Subprocess overhead (~50ms), requires parsing output for errors |
| `pywin32` (`win32security`) | Native API, richer control | Heavy C-extension dep (~30MB), overkill for one chmod equivalent |

**Why `icacls`:**
```python
# Equivalent to chmod 600 on Windows
import subprocess, os
if os.name == 'nt':
    username = os.environ.get('USERNAME', '')
    subprocess.run(
        ['icacls', str(key_path), '/inheritance:r',
         '/grant:r', f'{username}:(R,W)'],
        capture_output=True, check=True
    )
```

This runs once at key generation time — the ~50ms subprocess overhead is irrelevant. Avoids adding `pywin32` (30MB+ compiled extension) for a single use case.

**What NOT to add:**
- ~~`pywin32`~~ — Massive dependency for one `chmod` equivalent. `icacls` achieves the same result.
- ~~`pypiwin32`~~ — Unofficial wrapper, maintenance concerns.

#### 2d. Configurable Key Fallback Behavior

**New dependencies needed: NONE**

This is a config change in `SecurityConfig` dataclass (already in `config.py`) and a conditional raise in `security.py`. Pure logic change.

---

### 3. Error Handling Cleanup

**New dependencies needed: NONE**

| Capability | Source | Already Available |
|------------|--------|-------------------|
| Specific exception types | Python stdlib exception hierarchy | ✓ |
| Custom exception classes | Defined in source | ✓ `error_handling.py` exists |
| Context managers for cleanup | `contextlib` | ✓ Already used |

**Rationale:** Narrowing `except Exception` to specific types (`OSError`, `ValueError`, `pickle.UnpicklingError`, `orjson.JSONDecodeError`, `sqlalchemy.exc.OperationalError`, etc.) requires only knowledge of what exceptions the called code raises. No libraries involved.

**Key exception types to narrow to (already available in existing deps):**

| Current Broad Catch | Narrow To | Source |
|---------------------|-----------|--------|
| Blob read failures | `OSError`, `FileNotFoundError` | stdlib |
| Pickle/dill deserialization | `pickle.UnpicklingError`, `ModuleNotFoundError`, `AttributeError` | stdlib |
| JSON metadata parse | `orjson.JSONDecodeError`, `json.JSONDecodeError` | `orjson` (existing) / stdlib |
| SQLite operations | `sqlalchemy.exc.OperationalError`, `sqlalchemy.exc.IntegrityError` | `sqlalchemy` (existing) |
| Compression failures | `blosc2.Error`, `RuntimeError` | `blosc2` (existing) |
| Type detection | `TypeError`, `AttributeError` | stdlib |

---

### 4. Test Coverage Additions

**New dependencies needed: NONE**

| Capability | Source | Already Available |
|------------|--------|-------------------|
| Concurrent put/get testing | `concurrent.futures.ThreadPoolExecutor` | ✓ stdlib, already used in `test_backend_parity.py` |
| Thread creation | `threading.Thread`, `threading.Barrier` | ✓ stdlib, already used in `test_blob_store.py` |
| Key rotation scenarios | No special tooling | ✓ Test with `CacheEntrySigner` directly |
| Property-based testing | `hypothesis` | ✓ `>=6.151.5` in dev deps |
| Fault injection | `unittest.mock.patch` | ✓ stdlib, already used in `test_fault_injection.py` |

**Rationale:** The test infrastructure is already comprehensive. Concurrent tests exist at the blob store and backend level — the gap is at the `UnifiedCache` level (`put()`/`get()` under contention). The same `ThreadPoolExecutor` pattern from `test_backend_parity.py` applies.

**What NOT to add:**
- ~~`pytest-asyncio`~~ — No async code exists or is planned for this milestone.
- ~~`pytest-timeout`~~ — Concurrency tests should use deterministic synchronization (barriers, events), not timeouts.
- ~~`pytest-repeat`~~ — Stress testing is better done with parametrized loops in specific tests.
- ~~`locust` / `pytest-benchmark`~~ — Performance benchmarking is out of scope; the `benchmarks/` directory already handles this.

---

### 5. Handler Type Detection Robustness

**New dependencies needed: NONE**

| Capability | Source | Already Available |
|------------|--------|-------------------|
| Type checking | `isinstance()`, `hasattr()` | ✓ stdlib |
| Handler registry | `HandlerRegistry` class | ✓ `handlers.py` |
| Priority/ordering | Python list ordering | ✓ Current registration order |

**Rationale:** The robustness issue is about *logic*, not *tooling*. The fix involves:
1. Documenting handler registration order constraints
2. Adding guardrail checks (e.g., assert specific handlers registered before generic ones)
3. Possibly adding a `priority` attribute to handlers for explicit ordering

No external libraries needed.

---

## Summary: Stack Delta

| Area | New Runtime Deps | New Dev Deps | Stdlib Additions Used |
|------|-----------------|--------------|----------------------|
| Code decomposition | 0 | 0 | — |
| HKDF key derivation | 0 | 0 | `hmac` (already used) |
| Blob content hashing | 0 | 0 | — (uses existing `xxhash`) |
| Windows ACLs | 0 | 0 | `subprocess` + `icacls` |
| Key fallback config | 0 | 0 | — |
| Error handling cleanup | 0 | 0 | — |
| Concurrent testing | 0 | 0 | `concurrent.futures`, `threading` |
| Key rotation tests | 0 | 0 | — |
| Handler robustness | 0 | 0 | — |
| **TOTAL** | **0** | **0** | — |

## Anti-Recommendations: What NOT to Add

| Library | Why Considered | Why Rejected |
|---------|---------------|--------------|
| `cryptography` | HKDF implementation | 7MB+ C-extension; stdlib `hmac` implements HKDF in 15 lines. Already a transitive dev dep (via `moto`), but adding as runtime dep adds build complexity for no benefit. |
| `pywin32` / `pypiwin32` | Windows ACLs for key file | 30MB+ compiled extension for a single `chmod` equivalent. `icacls` subprocess achieves the same with zero deps. |
| `pytest-asyncio` | Async testing | No async code in this milestone. Out of scope. |
| `pytest-timeout` | Prevent test hangs | Concurrency tests should use barriers/events, not timeouts. The TF hang issue is handled by `--ignore`. |
| `pytest-repeat` | Stress/flaky testing | Parametrized loops in specific tests are simpler and more explicit. |
| `attrs` / `pydantic` | Config validation | `dataclasses` already used throughout. Adding a validation framework for config changes is over-engineering. |

## Integration Points with Existing Stack

| New Capability | Integrates With | How |
|----------------|----------------|-----|
| HKDF key derivation | `security.py` (`hmac`, `hashlib`) | New function in existing module, called from `CacheEntrySigner.__init__()` |
| Blob content hashing default | `file_hashing.py` (`xxhash`), `config.py` (`SecurityConfig`) | Change `verify_blob_hash` default from `False` to `True` |
| Windows ACLs | `security.py` (`os`, `subprocess`) | Platform-specific branch in `_generate_new_key()` |
| Configurable fallback | `config.py` (`SecurityConfig`), `security.py` | New `SecurityConfig.allow_memory_key_fallback` field |
| Exception narrowing | All source files | Uses exception types from existing deps (`sqlalchemy`, `orjson`, `blosc2`) |
| Concurrent tests | `test_backend_parity.py` pattern | Same `ThreadPoolExecutor` + assertions pattern |

## Version Pinning Notes

No version changes needed. Current pins are all adequate:

| Package | Current Pin | Status | Notes |
|---------|------------|--------|-------|
| `xxhash` | `>=3.5.0` | ✓ Current | Used for blob content hashing (already available) |
| `sqlalchemy` | `>=2.0.0` | ✓ Current | Exception types stable across 2.x |
| `orjson` | `>=3.8.0` | ✓ Current | `JSONDecodeError` available since 3.x |
| `blosc2` | `>=3.5.1` | ✓ Current | Error types stable |
| `hypothesis` | `>=6.151.5` | ✓ Current | Sufficient for any new property tests |

## Confidence Assessment

| Area | Confidence | Reason |
|------|------------|--------|
| HKDF via stdlib | HIGH | Verified working with Python 3.12 stdlib `hmac`. RFC 5869 is well-specified. |
| Windows ACLs via icacls | HIGH | `icacls` is a Windows built-in since Vista. Well-documented. |
| No new deps needed | HIGH | Every capability verified against existing stack. |
| Exception narrowing types | HIGH | Exception types are stable public API in all referenced libraries. |
| Concurrent test patterns | HIGH | Already used in `test_backend_parity.py` and `test_blob_store.py`. |

## Sources

- Python 3.12 `hmac` module documentation — HKDF implementation basis
- RFC 5869 (HMAC-based Extract-and-Expand Key Derivation Function) — HKDF specification
- Microsoft `icacls` documentation — Windows ACL command reference
- Existing codebase: `security.py`, `file_hashing.py`, `config.py`, `test_backend_parity.py`
- `pyproject.toml` and `uv.lock` — current dependency inventory
