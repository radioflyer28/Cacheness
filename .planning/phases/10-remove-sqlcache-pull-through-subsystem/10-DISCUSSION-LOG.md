# Phase 10: Remove SqlCache Pull-Through Subsystem - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-17
**Phase:** 10-remove-sqlcache-pull-through-subsystem
**Areas discussed:** Removed import behavior, Dependencies and extras, Documentation boundary, Regression contract

---

## Removed import behavior

| Decision | Alternatives considered | Selected |
|----------|-------------------------|----------|
| Old imports | Natural absence; tombstone module; top-level tombstone | Natural absence ✓ |
| Replacement guidance | Alternatives by use case; generic note; point everything to UnifiedCache | Alternatives by use case ✓ |
| Existing SQL tables | Leave untouched; export tool; cleanup tooling | Leave untouched ✓ |
| Versioning | No phase bump; development minor; 1.0.0 | No phase bump ✓ |

**User's choice:** Direct deletion with normal import failures, accurate use-case guidance, no caller-data mutation, and no version bump.
**Notes:** `UnifiedCache` and `BlobStore` are not presented as substitutes for range-aware SQL pull-through.

---

## Dependencies and extras

| Decision | Alternatives considered | Selected |
|----------|-------------------------|----------|
| `sql` extra | Delete; repurpose; empty compatibility extra | Delete ✓ |
| DuckDB | Remove every claim/dependency; runtime only; keep for development | Remove every claim/dependency ✓ |
| Remaining SQL groups | Remove only SqlCache-only entries; redesign all; collapse groups | Remove only SqlCache-only entries ✓ |
| Other orphan packages | Prune every proven orphan; remove only DuckDB; retain broad lock | Prune every proven orphan ✓ |

**User's choice:** Fully remove the subsystem's dependency surface without redesigning dependencies owned by supported storage/catalog behavior.
**Notes:** Manifest and lockfile must agree.

---

## Documentation boundary

| Decision | Alternatives considered | Selected |
|----------|-------------------------|----------|
| Dedicated assets | Delete; archive; retain with banners | Delete ✓ |
| Historical records | Preserve truthful history; scrub mentions; delete artifacts | Preserve truthful history ✓ |
| Mixed current docs | Surgical cleanup; delete whole file; leave unchanged | Surgical cleanup ✓ |
| Removal guidance | Existing canonical docs; standalone guide; README only; none | Existing canonical docs ✓ |

**User's choice:** Remove unsupported product material while retaining accurate history and useful mixed documentation.
**Notes:** No new long-lived legacy-product document.

---

## Regression contract

| Decision | Alternatives considered | Selected |
|----------|-------------------------|----------|
| Import/module absence | Complete proof; `__all__` only; source deletion only | Complete proof ✓ |
| Reference scope | Current-facing allowlist; repository-wide zero; runtime only | Current-facing allowlist ✓ |
| Older mixed gates | Rewrite/invert; delete; exclude | Rewrite/invert ✓ |
| Package acceptance | Fresh isolated wheel; source tests only; metadata only | Fresh isolated wheel ✓ |

**User's choice:** A strict current-product ratchet with explicit historical exceptions and artifact-level wheel evidence.
**Notes:** The final proof includes supported local round trips and installed dependency metadata.

---

## the agent's Discretion

- Exact implementation task ordering and test-helper structure.
- Exact concise wording placement in existing canonical guidance.
- Exact form of the explicit current-facing reference allowlist.

## Deferred Ideas

None.
