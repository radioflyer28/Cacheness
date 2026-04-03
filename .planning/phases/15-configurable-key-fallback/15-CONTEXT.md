# Phase 15: Configurable Key Fallback - Context

**Gathered:** 2026-04-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace the binary `raise_on_key_fallback: bool` with a 3-mode `key_fallback_policy` configuration. Users explicitly control what happens when the signing key file cannot be written: raise an error, log a warning and continue with in-memory key, or silently fall back. No new crypto features — this is config/behavior only.

</domain>

<decisions>
## Implementation Decisions

### Config API Design
- **D-01:** Add new field `key_fallback_policy: str = "warn"` to `SecurityConfig` accepting `"raise"`, `"warn"`, or `"fallback"`.
- **D-02:** Deprecate `raise_on_key_fallback: bool` with a backward-compatibility shim — if the old field is explicitly set, it maps to the new field (`True` → `"raise"`, `False` → `"fallback"`). Log a deprecation warning when the old field is used.
- **D-03:** New field takes precedence if both are set.

### Warn vs Fallback Behavior
- **D-04:** `"warn"` mode: log at WARNING level + use in-memory key + cache operates normally. This is the new default (replaces the old `raise_on_key_fallback=False` which was silent).
- **D-05:** `"fallback"` mode: silently use in-memory key, no log output (matches old `raise_on_key_fallback=False` behavior exactly).
- **D-06:** `"raise"` mode: raise `CacheSecurityError` (matches old `raise_on_key_fallback=True` behavior exactly).

### Error Semantics
- **D-07:** `CacheSecurityError` is used for all key-related failures in `"raise"` mode — both write failures and corrupt key files (wrong size).
- **D-08:** No new error types needed.

### Test Coverage
- **D-09:** Comprehensive test suite: each mode (raise/warn/fallback) with write failures, deprecation shim both directions, corrupt key file handling, integration with `UnifiedCache._init_entry_signer` catch semantics. Target ~8-10 new tests.

### Agent's Discretion
- Exact deprecation warning message wording
- Whether to use `Literal["raise", "warn", "fallback"]` type annotation or plain `str` with runtime validation
- Test file location (extend existing `test_key_rotation.py` or new `test_key_fallback_policy.py`)

</decisions>

<specifics>
## Specific Ideas

- Default changes from silent fallback to warning — this is a behavioral change for users who had `raise_on_key_fallback=False` (default). They'll now see a WARNING log they didn't see before. This is intentional and desired.
- The `"fallback"` option preserves the old silent behavior for users who explicitly want no noise.

</specifics>

<canonical_refs>
## Canonical References

### Security Architecture
- `src/cacheness/security.py` — `_generate_new_key()` (lines 133-167) is the decision point. `_load_or_generate_key()` (lines 109-131) is the entry point.
- `src/cacheness/config.py` — `SecurityConfig` dataclass (lines ~394-408) defines all key-related fields.
- `src/cacheness/core.py` — `_init_entry_signer()` (lines 329-351) catches `CacheSecurityError` and sets `self.signer = None`.
- `src/cacheness/error_handling.py` — `CacheSecurityError` definition.

### Documentation
- `docs/SECURITY.md` — public-facing security documentation to update.

### Existing Tests
- `tests/test_key_rotation.py` — `test_raise_on_key_fallback()` and `test_no_raise_on_key_fallback_by_default()` test current behavior.
- `tests/test_cache_signing.py` — in-memory key tests.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `CacheSecurityError` — already exists, used for `raise` mode
- `_generate_new_key()` — the exact function to modify (security.py:133-167)
- `SecurityConfig` — dataclass to extend with new field

### Established Patterns
- Config via `@dataclass` in `config.py` — follow existing field pattern
- `# intentionally broad` annotation for catch clauses — maintain this pattern
- `create_cache_signer()` factory function passes config through

### Integration Points
- `SecurityConfig` in `config.py` → new field + deprecation shim
- `CacheEntrySigner.__init__()` → accept new parameter
- `_generate_new_key()` → implement 3-mode behavior
- `_init_entry_signer()` in `core.py` → may need adjustment for `warn` vs `raise` semantics
- `create_cache_signer()` → pass through new parameter

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 15-configurable-key-fallback*
