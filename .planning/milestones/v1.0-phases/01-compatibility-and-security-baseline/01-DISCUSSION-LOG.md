# Phase 1: Compatibility and Security Baseline - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-29
**Phase:** 01-compatibility-and-security-baseline
**Areas discussed:** Compatibility envelope, path-containment failures, legacy arrays and unsafe parsing, query and `SqlCache` failure contracts

---

## Compatibility Envelope

| Question | Options considered | Selected |
|----------|--------------------|----------|
| Public API baseline | Exported/documented surface and public examples/tests; every callable non-private symbol; narrower curated manifest | Exported/documented surface and public examples/tests |
| Stored-format window | `0.3.14` plus safely identifiable earlier `0.3.x`; only `0.3.14`; every reconstructable historical format | `0.3.14` plus safely identifiable earlier `0.3.x` |
| Security conflicts | Safe adapters with immediate typed rejection for unsafe input; one release even for unsafe behavior; immediate breaking changes | Safe adapters with immediate typed rejection |
| Optional symbols | Stable top-level names with use-time dependency errors; conditional exports; explicit submodules only | Stable top-level names with use-time dependency errors |
| Adapter duration | Full milestone until announced major/`1.0`; one release; indefinitely | Full milestone until announced major/`1.0` |
| Known defects | Test corrected intent and document changes; encode expected failures first; preserve unless vulnerable | Test corrected intent and document changes |
| Relative configuration paths | Preserve representation and resolve at runtime; always save absolute; save relative to config file | Preserve representation and resolve at runtime |
| Error stability | Stable classes and reason codes; exact messages too; broad base family only | Stable classes and reason codes |

**User's choice:** Selected the recommended compatibility boundary in all eight decisions.
**Notes:** Accidental quirks are not contracts. Security fixes may intentionally change behavior without a deprecation period.

---

## Path-Containment Failures

| Question | Options considered | Selected |
|----------|--------------------|----------|
| Unsafe read-like operations | Typed error for every operation; mutation errors but read misses; configurable strictness | Typed error for every operation |
| Existing external locator | Refuse and preserve; delete metadata; copy under root | Refuse and preserve |
| Symlinks | Resolve root but reject entry-path symlinks; allow currently contained symlinks; reject all symlinks | Resolve root but reject entry-path symlinks |
| Direct blob identifiers | Restricted opaque identifiers; contained nested paths; sanitized arbitrary identifiers | Restricted opaque identifiers |

**User's choice:** Chose strict, explicit rejection and an opaque identifier model.
**Notes:** Unsafe locators remain untouched for the later migration/reconciliation workflow rather than being mutated during ordinary access.

---

## Legacy Arrays and Unsafe Parsing

| Question | Options considered | Selected |
|----------|--------------------|----------|
| Legacy tuple header | Safe parse and validate; reject/rebuild; compatibility flag | Safe parse and validate |
| NumPy object arrays | Explicit trusted-object path; normal handler with signed trust marker; automatic pickle | Explicit trusted-object path |
| Invalid declared artifact | Typed fail-closed error; try another sidecar; configurable fallback | Typed fail-closed error |
| Custom Blosc2 raw-array header | Legacy read-only; formalize it; remove support | Legacy read-only |

**User's choice:** Keep safe reads for identifiable legacy artifacts but never write the custom raw-array container again.
**Notes:** The user clarified that native handler libraries own serialization. Blosc2 may compress serialized pickle bytes or store native compatible data structures; Cacheness should coordinate formats, not invent them. Validation is structural and byte-consistency based, not a new default filesize policy.

---

## Query and `SqlCache` Failure Contracts

| Question | Options considered | Selected |
|----------|--------------------|----------|
| `query_meta()` semantics | Preserve numeric threshold/string equality; all exact equality; explicit operators | Preserve documented semantics |
| Invalid query fields | Typed pre-database error; empty result; ignore invalid filters | Typed pre-database error |
| Missing-range fetch failure | Strict default with explicit best effort; best effort by default; new result wrapper | Strict default with explicit best effort |
| Fallback boundary | Equivalent internal fallback only; automatic fallback everywhere; no fallback | Equivalent internal fallback only |

**User's choice:** Preserve documented query behavior while making unsafe filters and incomplete SQL results explicit.
**Notes:** Caller-supplied adapter and gap-detector failures do not silently fall back unless the caller enabled that policy.

---

## the agent's Discretion

None. Exact implementation mechanics remain for research and planning within the locked decisions.

## Deferred Ideas

None. Future native array format and canonical manifest design remain in the already-defined Phase 2 scope.
