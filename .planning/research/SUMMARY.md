# Research Synthesis — v1.0 Cleanup & Hardening

**Milestone:** v1.0 Cleanup & Hardening
**Synthesized:** 2026-04-02
**Sources:** STACK.md, FEATURES.md, ARCHITECTURE.md, PITFALLS.md
**Overall Confidence:** HIGH

---

## 1. Executive Summary

The v1.0 Cleanup & Hardening milestone is a zero-dependency structural refactoring and security improvement pass on an already-stable codebase (1,427 tests, 16,613 LOC). Every planned capability — HKDF key derivation, blob integrity verification, exception narrowing, Windows file permissions, concurrent testing — is achievable using Python's stdlib and existing runtime dependencies. No new packages are needed.

The work decomposes into two categories: **structural refactors** (splitting three monolithic files into packages/mixins) and **behavioral changes** (security hardening, exception narrowing). Research unanimously recommends completing all structural work first, then applying behavioral changes to the decomposed codebase. This avoids the two-variable debugging problem (is a test failing because code moved, or because behavior changed?) and keeps git history useful.

The highest-risk item is the `core.py` mixin decomposition (3,900 lines, 85+ methods, tight state coupling). The lowest-risk items are the `handlers.py` split (stateless, self-contained handlers) and test additions (purely additive).

---

## 2. Stack Decisions

**Zero new runtime dependencies.** This is the defining constraint and ideal outcome for a hardening milestone.

| Capability | How | Source |
|------------|-----|--------|
| HKDF key derivation | `hmac` + `hashlib` (stdlib, ~15 lines) | Already used in `security.py` |
| Blob content hashing | `xxhash` (existing core dep) + `file_hashing.py` | Already computed on `put()` |
| Windows key permissions | `icacls` via `subprocess` (stdlib) | One-time call at key generation |
| Exception narrowing | Existing exception hierarchy in `error_handling.py` | Extend with 2-3 new types |
| Concurrent testing | `concurrent.futures`, `threading` (stdlib) | Already used in existing tests |
| Code decomposition | Standard Python packages + `__init__.py` re-exports | Pattern proven in `storage/` |

**Explicitly rejected:** `cryptography` (C-extension bloat for simple HKDF), `pywin32` (30MB for one chmod equivalent), `pytest-asyncio` (no async code), `pytest-timeout` (use deterministic synchronization instead).

---

## 3. Recommended Phase Order

Research from ARCHITECTURE.md and PITFALLS.md converges on this ordering. The key insight: structural changes are low-risk individually but create merge conflict compounding as a group, while behavioral changes are easier to audit on smaller files.

| Phase | Scope | Risk | Effort | Rationale |
|-------|-------|------|--------|-----------|
| **1. handlers.py → handlers/ package** | 1,676 lines → 8 files + init | LOW | ~1 day | Smallest scope, validates the package-split pattern. Handlers are stateless and self-contained. |
| **2. metadata.py → metadata/ package** | 2,562 lines → 4 files + init | MEDIUM | ~1-2 days | Shared base class (SQLAlchemy models) adds coordination. Validates pattern for complex cases. |
| **3. core.py → mixin decomposition** | 3,900 lines → 8 mixins + core | HIGH | ~2-3 days | Largest, highest coupling. Benefits from experience in phases 1-2. All 1,427 tests must pass. |
| **4. Exception narrowing** | ~50 `except Exception` sites | MEDIUM | ~1-2 days | Behavioral change — must happen after code is in final locations (P16). |
| **5. Security hardening** | Blob hashing, HKDF, key fallback, Windows ACLs | MEDIUM | ~2 days | Behavioral + new features. Applied to stable, decomposed codebase (P17). |
| **6. Test gap coverage** | Thread safety, key rotation tests | LOW | ~1 day | Additive. Can run in parallel with phases 1-3. Key rotation tests land with/after phase 5. |

**Critical ordering constraints (from pitfall analysis):**
- Decomposition before exception narrowing (P16: isolate structural vs behavioral failures)
- Decomposition before security hardening (P17: avoid merge conflicts in moving+changing code)
- Handler split before handler ordering guardrails (P18: split first, then add guardrails)
- Exception hierarchy design before narrowing catches (P13: have target types ready)

---

## 4. Key Risks

Top 5 risks synthesized from PITFALLS.md, ranked by severity and likelihood:

### Risk 1: Mixin MRO & State Coupling (P1)
**Impact:** `AttributeError` at runtime, invisible until specific code paths execute.
**Mitigation:** No `__init__` in mixins. All initialization stays in `UnifiedCache.__init__()`. Add MRO assertion test. Extract one mixin at a time with full test runs between each.

### Risk 2: Import Path Breakage (P2, P3)
**Impact:** `ImportError` in tests and downstream consumers. 50+ import sites across tests reference internal module paths.
**Mitigation:** Catalog all imports before splitting (`grep -rn "from cacheness\.\(core\|metadata\|handlers\) import"`). Every old path must resolve via `__init__.py` re-exports. Add import compatibility test.

### Risk 3: Blob Hashing Default-On Breaks Existing Caches (P4)
**Impact:** `verify_cache_integrity()` reports 100% corruption on pre-existing entries. `get()` with `delete_on_error=True` silently purges caches on upgrade.
**Mitigation:** "Hash if present" verification — verify only when stored hash exists. Write-path only for new entries. Never make `get()` fail on missing hashes.

### Risk 4: Exception Narrowing Exposes Swallowed Failures (P5)
**Impact:** Previously-silent errors propagate to callers. "Cacheness used to handle this, now it crashes."
**Mitigation:** Audit each catch individually with logging before narrowing. Preserve intentional safety nets (`_init_auto_backend`, `__del__`, hook invocations) with explicit `# intentionally broad` comments. Don't narrow catches inside `get()`.

### Risk 5: Per-Namespace Key Derivation Invalidates Signatures (P8)
**Impact:** All entries in non-default namespaces fail verification after upgrade.
**Mitigation:** Fallback verification (try derived key → fall back to master key → re-sign with derived key on next write). Version the signing scheme in metadata.

---

## 5. Table Stakes vs Nice-to-Have

### Table Stakes (Must-Do for v1.0)

| Item | Category | Why |
|------|----------|-----|
| Split `handlers.py` into per-handler files | Decomposition | Organizational debt, lowest-risk decomposition |
| Split `metadata.py` into per-backend files | Decomposition | Three independent backends sharing a file |
| Decompose `core.py` via mixins | Decomposition | 3,900-line monolith is the primary maintenance burden |
| Preserve all existing import paths | Decomposition | Non-negotiable backward compatibility |
| Default-on blob content hashing | Security | Metadata signing without blob verification has a gap |
| Configurable in-memory key fallback | Security | Silent fallback is a security-relevant default |
| Narrow `except Exception` in core.py | Error handling | 30+ broad catches mask bugs |
| Narrow `except Exception` in handlers.py | Error handling | 20+ broad catches mask deserialization failures |
| Thread-safety smoke tests | Testing | No tests verify concurrent behavior despite lock existence |

### Nice-to-Have (Differentiators)

| Item | Category | Complexity | Notes |
|------|----------|------------|-------|
| Per-namespace key derivation (HKDF) | Security | Medium | Migration path for existing entries is the complexity |
| Windows key file ACLs via `icacls` | Security | Medium | Consider document-only for v1.0 (P9) |
| Explicit numeric priority on handlers | Robustness | Medium | Currently implicit registration order works |
| Handler conflict detection warnings | Robustness | Medium | Useful but not urgent |
| Structured error context on `CacheError` raises | Error handling | Low-Medium | Many sites to update, can be incremental |
| Multi-process safety tests | Testing | Medium-High | Cross-platform process management is tricky |
| Stress tests with high contention | Testing | Medium | Diminishing returns beyond smoke tests |

### Anti-Features (Explicitly Avoid)

- Encrypting blob content at rest (feature addition, not hardening)
- Adding `cryptography` package (C-extension bloat)
- Automatic key rotation (distributed systems problem)
- Full thread safety on `put()`/`get()` (serializes all operations)
- Deep inheritance hierarchy for `UnifiedCache`
- Splitting `compress_pickle.py` (stable, low churn)
- Result/Either types replacing exceptions (API change)

---

## 6. Open Questions

These need design decisions before or during implementation:

| Question | Context | Options | Recommendation |
|----------|---------|---------|----------------|
| Mixins vs delegates for core.py? | Both decompose the monolith. Mixins share `self`, delegates use explicit injection. | A) Mixins (simpler, less refactoring) B) Delegates (more testable, more refactoring) | **Mixins** — lower risk for a hardening milestone. Delegates are a future consideration. |
| Package (`core/`) vs sibling files (`_core_*.py`) for mixins? | Package requires import chain update. Sibling files avoid it. | A) `core/` package B) `_core_*.py` sibling files | **Decide during phase 3 planning.** Sibling files are lower risk but messier at package root. |
| Mixin type safety approach? | Mixins lack type info about attributes from other mixins. | A) Bare `self` access B) `Protocol`-based contracts C) `TYPE_CHECKING` imports | **Start with bare `self` access**, add Protocol only if `ty check` complains. |
| Windows ACLs: implement or document? | `icacls` works but is fragile. `pywin32` is heavy. Documentation is honest. | A) `icacls` B) Document as known limitation | **Document for v1.0**, stretch goal to implement `icacls`. |
| Default namespace key derivation? | Should default namespace use master key directly (backward compat) or derived key? | A) Master key for default B) Derived key for all | **Master key for default** — backward compatibility, zero migration for common case. |
| `compress_pickle.py` exception narrowing? | ~10 broad catches, not analyzed in detail. | A) Include in phase 4 B) Defer | **Include if time permits**, defer if not — low churn file. |

---

## 7. Cross-Cutting Concerns

Themes that appeared across multiple research areas:

### Backward Compatibility is the Primary Constraint
Every research area identified import path preservation as critical. The `storage/` re-export layer, test imports, and public API all depend on stable paths. The pattern: always re-export from `__init__.py`, never remove an importable name, test the import chain explicitly.

### "Move Code, Then Change Behavior" Ordering
ARCHITECTURE.md, PITFALLS.md, and FEATURES.md all converge on the same conclusion: structural refactoring (moving code between files) must complete before behavioral changes (narrowing exceptions, changing defaults). P16 and P17 formalize why — two-variable debugging is error-prone and destroys git blame utility.

### Migration Paths for Every Default Change
Blob hashing default-on (P4), per-namespace key derivation (P8), and key fallback behavior (P15) all share a pattern: changing a default breaks existing data. The universal mitigation is "new behavior on write, graceful degradation on read" — verify hashes only if present, try new key then old key, warn but don't crash.

### The Test Suite is Both Asset and Constraint
1,427 tests provide excellent regression detection but also create ~70+ potential breakpoints during refactoring (tests that reference internal paths, mock private methods, or assert internal state). The test suite must be treated as a backward-compatibility contract, not just a verification tool.

### Security Hardening is Incremental, Not Transformational
No single security change is large. Blob hashing is a config default flip. Key fallback is a conditional + config field. HKDF is ~15 lines. Windows ACLs are documentation or a subprocess call. The risk comes from interactions (signing scheme migration), not individual changes.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack decisions | **HIGH** | Zero new deps confirmed across all research areas |
| Phase ordering | **HIGH** | Architecture and pitfalls research independently converge on same order |
| Decomposition patterns | **HIGH** (handlers, metadata) / **MEDIUM** (core) | Handlers and metadata have clean boundaries; core mixin extraction has state coupling edge cases |
| Exception narrowing | **MEDIUM** | 50 sites identified but each needs per-site analysis during implementation |
| Security hardening | **HIGH** | Integration points clear, existing infrastructure supports changes |
| Test coverage | **HIGH** | Straightforward additive work with known patterns |

**Gaps remaining:**
- Per-site exception narrowing analysis (deferred to implementation)
- `compress_pickle.py` exception audit (not deeply analyzed)
- core.py mixin boundary edge cases (will surface during extraction)
- PostgreSQL backend location decision (future milestone concern)

---

*Research synthesis complete. Ready for requirements definition and roadmap creation.*
