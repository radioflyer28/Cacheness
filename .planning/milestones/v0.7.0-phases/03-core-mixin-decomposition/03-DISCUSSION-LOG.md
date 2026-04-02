# Phase 3: Core Mixin Decomposition - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-03
**Phase:** 03-core-mixin-decomposition
**Areas discussed:** Mixin granularity, File organization, Core trunk boundary, Mixin state access, Init helper placement, Signing method ownership, Module-level helpers, MRO ordering, DECO-04 validation scope, Mixin import style, Post-extraction size, Mixin visibility, Init-mixin coupling, Circular import risk, Cross-mixin method dependencies, Sign namespace boundary

---

## Mixin Granularity

| Option | Description | Selected |
|--------|-------------|----------|
| 4 per ROADMAP | Verification, Statistics, CustomMetadata, StorageMode. Keep it scoped — query/file/batch/convenience stay in UnifiedCache. | ✓ |
| 6-7 mixins | Add QueryMixin, FileOpsMixin, ConvenienceAPIMixin. Thinner UnifiedCache but more files/MRO complexity. | |
| 3 mixins (merge stats) | Merge Statistics into UnifiedCache (it's only 35 lines). Create Verification, CustomMetadata, StorageMode. | |

**User's choice:** 4 per ROADMAP
**Notes:** User also confirmed StatsMixin should stay as its own mixin for consistency, not merged with VerificationMixin.

---

## File Organization

| Option | Description | Selected |
|--------|-------------|----------|
| Flat files in src/cacheness/ | core.py stays as a file, mixins go into _verification_mixin.py, _stats_mixin.py, etc. beside it. Simplest approach. | ✓ |
| core/ sub-package | core.py becomes core/ package like handlers/ and metadata/. Consistent with Phase 1/2 but adds package nesting. | |

**User's choice:** Flat files in src/cacheness/
**Notes:** No package restructuring needed — `from cacheness.core import UnifiedCache` works without changes.

---

## Core Trunk Boundary

| Option | Description | Selected |
|--------|-------------|----------|
| put/get stay in UnifiedCache | put()/get() stay in UnifiedCache and call into mixin methods via self. Mixins are leaf logic only. | ✓ |
| put/get move to a mixin too | Move put()/get() into a CacheOperationsMixin. UnifiedCache becomes nearly empty. | |

**User's choice:** put/get stay in UnifiedCache
**Notes:** None.

---

## Mixin State Access

| Option | Description | Selected |
|--------|-------------|----------|
| Plain self access | Mixins just use self.config, self.metadata_backend, etc. No Protocol, no ABC. Standard cooperative inheritance. | ✓ |
| Protocol-typed access | Define Protocol class listing attributes mixins depend on. Better IDE autocomplete but adds boilerplate. | |

**User's choice:** Plain self access
**Notes:** None.

---

## Init Helper Placement

| Option | Description | Selected |
|--------|-------------|----------|
| Keep init in core.py | Keep all _init_* methods in core.py inside UnifiedCache. Mixins are purely runtime behavior. | ✓ |
| Move init helpers to their mixin | _init_custom_metadata_support() to CustomMetadataMixin, _init_entry_signer() to VerificationMixin. | |

**User's choice:** Keep init in core.py
**Notes:** None.

---

## Signing Method Ownership

| Option | Description | Selected |
|--------|-------------|----------|
| VerificationMixin | _sign_entry_if_enabled() and _extract_signable_fields() move to VerificationMixin. Coherent sign/verify concern. | ✓ |
| Stay in core.py | Signing is part of put()'s workflow, not verification. Keep in core.py with put(). | |

**User's choice:** VerificationMixin
**Notes:** Sign and verify form a coherent domain despite being called from different paths (write vs read).

---

## Module-Level Helpers

| Option | Description | Selected |
|--------|-------------|----------|
| Keep in core.py | _PutCleanup and _normalize_function_args are tightly coupled to core.py's put() workflow. | ✓ |
| Move to _core_utils.py | Reduces core.py line count but adds a file for ~80 lines. | |

**User's choice:** Keep in core.py
**Notes:** None.

---

## MRO Ordering

| Option | Description | Selected |
|--------|-------------|----------|
| Agent decides order | No naming conflicts. List in logical dependency order (verification first). Agent decides exact order. | ✓ |
| Alphabetical order | Predictable, no thought needed. | |

**User's choice:** Agent decides order
**Notes:** None.

---

## DECO-04 Validation Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Trust test suite | 1,604 existing tests import extensively. If any path breaks, tests fail. | ✓ |
| Add import smoke test | Explicit import test as belt-and-suspenders. | |

**User's choice:** Trust test suite
**Notes:** Since we're not creating a core/ package, the import risk is minimal.

---

## Mixin Import Style

| Option | Description | Selected |
|--------|-------------|----------|
| Relative (match core.py) | from .interfaces import ..., from .metadata import ... Consistent with existing style. | ✓ |
| Absolute imports | from cacheness.interfaces import ... Slightly more explicit but inconsistent. | |

**User's choice:** Relative (match core.py)
**Notes:** None.

---

## Post-Extraction Size

| Option | Description | Selected |
|--------|-------------|----------|
| Acceptable — scope is 4 mixins | ~2,400 lines is fine. The goal is focused mixins, not minimizing core.py. | ✓ |
| Extract more to get under 2,000 | Pull out QueryMixin and FileOpsMixin. Expands scope beyond ROADMAP. | |

**User's choice:** Acceptable — scope is 4 mixins
**Notes:** Further splits can be a future milestone.

---

## Mixin Visibility

| Option | Description | Selected |
|--------|-------------|----------|
| Private (underscore files, no export) | Files named _verification_mixin.py. Classes not exported from __init__.py. | ✓ |
| Public (exported) | Files without underscore, classes exported. Users could subclass mixins. | |

**User's choice:** Private (underscore files, no export)
**Notes:** None.

---

## Init-Mixin Coupling

| Option | Description | Selected |
|--------|-------------|----------|
| Acceptable coupling | Init in core.py sets state, mixin methods read it. Expected for cooperative mixins. | ✓ |
| Add coupling comments | Document which mixin consumes each init-set attribute. | |

**User's choice:** Acceptable coupling
**Notes:** None.

---

## Circular Import Risk

| Option | Description | Selected |
|--------|-------------|----------|
| No concern | One-directional import flow. core.py → mixin files → sibling modules. No mixin imports from core.py. | ✓ |

**User's choice:** No concern
**Notes:** Confirmed one-directional dependency eliminates risk.

---

## Cross-Mixin Method Dependencies

| Option | Description | Selected |
|--------|-------------|----------|
| Standard mixin behavior — no concern | Cross-class calls via self.* are the whole point of mixins. MRO guarantees resolution. | ✓ |
| Document dependencies in comments | Add brief comments listing which UnifiedCache methods each mixin depends on. | |

**User's choice:** Standard mixin behavior — no concern
**Notes:** None.

---

## Sign Namespace Boundary

| Option | Description | Selected |
|--------|-------------|----------|
| Keep in core.py (with init) | Called from __init__ flow. Keep with other init helpers for consistency. | ✓ |
| Move to VerificationMixin | It's signing logic. Move to VerificationMixin. | |

**User's choice:** Keep in core.py (with init)
**Notes:** Follows D-06 (all init helpers stay in core.py).

---

## Agent's Discretion

- Exact MRO ordering of the 4 mixins in the class declaration
- Import organization within each mixin file
- Whether to include module-level docstrings in mixin files

## Deferred Ideas

None — discussion stayed within phase scope.
