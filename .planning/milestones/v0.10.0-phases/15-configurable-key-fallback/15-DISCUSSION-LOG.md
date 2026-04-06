# Phase 15: Configurable Key Fallback - Discussion Log

**Date:** 2026-04-03
**Participants:** User + Agent

---

## Codebase Scout

**Agent scouted** `src/cacheness/security.py`, `src/cacheness/config.py`, `src/cacheness/core.py`, `src/cacheness/error_handling.py`, and test files.

**Key findings:**
- `_generate_new_key()` at security.py:133-167 is the single decision point for fallback behavior
- `SecurityConfig` has 6 fields including `raise_on_key_fallback: bool = False`
- `_init_entry_signer()` in core.py catches `(OSError, ValueError)` and sets `self.signer = None` — even `raise` mode is caught at the core level
- Existing tests in `test_key_rotation.py` cover raise and no-raise scenarios

---

## Gray Areas Identified

Four gray areas identified for discussion:
1. Config API design (new field vs modify existing)
2. Warn vs Fallback behavior definition
3. Error semantics (which errors for which scenarios)
4. Test coverage scope

User selected: **All 4** for discussion.

---

## Q1: Config API Design

**Options presented:**
- A) New `key_fallback_policy: str` field + deprecation shim for old `raise_on_key_fallback`
- B) Repurpose `raise_on_key_fallback` to accept `str | bool`
- C) Replace `raise_on_key_fallback` outright (breaking change)

**User chose:** A — New str field + deprecation shim *(recommended)*

**Rationale:** Clean API, backward compatible, clear migration path.

---

## Q2: Warn Behavior Definition

**Options presented:**
- A) Warn = WARNING log + in-memory key (cache works normally)
- B) Warn = WARNING log + disabled signing (signer = None)

**User chose:** A — WARNING log + in-memory key *(recommended)*

**Rationale:** Users get visibility (warning) without losing signing functionality. Old silent behavior available as `"fallback"` mode.

---

## Q3: Error Semantics

**Options presented:**
- A) CacheSecurityError for all key issues (write failures + corrupt key)
- B) Separate KeyWriteError vs KeyCorruptionError

**User chose:** A — CacheSecurityError for all key issues *(recommended)*

**Rationale:** Single error type is simpler. No downstream code distinguishes between key failure causes.

---

## Q4: Test Coverage Plan

**Options presented:**
- A) Minimal (3-4 tests, basic mode smoke tests)
- B) Comprehensive (~8-10 tests, all modes + edge cases + deprecation + integration)

**User chose:** B — Comprehensive *(recommended)*

**Rationale:** Security config changes warrant thorough coverage. Tests also serve as documentation for the 3-mode API.

---

*End of discussion — proceed to planning.*
