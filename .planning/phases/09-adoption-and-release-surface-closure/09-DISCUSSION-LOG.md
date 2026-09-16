# Phase 9: Adoption and Release Surface Closure - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-16
**Phase:** 09-adoption-and-release-surface-closure
**Areas discussed:** Product story and documentation structure, installation and release posture, executable examples, custom handler onboarding

---

## Product Story and Documentation Structure

| Question | Options considered | Selected |
| --- | --- | --- |
| README role | Concise gateway with two quick starts; comprehensive README; tutorial-first README | Concise gateway with two quick starts |
| Documentation organization | Task-first guides plus component references; component-only references; progressive manual | Task-first guides plus component references |
| Obsolete documents | Remove from supported documentation; historical archive; rewrite every document in place | Remove from supported documentation |
| Qualification visibility | Short README status box plus detailed guarantees page; full matrix in README; references only | Status box plus detailed guarantees page |

**User's choice:** Selected the recommended concise, task-first, deletion-oriented structure with prominent but layered qualification disclosure.
**Notes:** BlobStore remains the primary storage product; UnifiedCache is policy over it. Historical material remains available through Git and planning artifacts rather than supported-doc navigation.

---

## Installation and Release Posture

| Question | Options considered | Selected |
| --- | --- | --- |
| Primary pre-publication installation | Checked-out repository with `uv`; build/install local wheel first; retain registry install with warning | Checked-out repository with `uv` |
| Release wording | Local-ready development version; alpha release candidate; unreleased v1.0 | Local-ready development version |
| PostgreSQL/S3 visibility | Guarantees/reference only; advanced experimental guides; omit entirely | Guarantees/reference only |
| Optional dependencies | Capability-specific extras; broad recommended bundle; all-extras development install | Capability-specific extras |

**User's choice:** Present honest checkout-based local use and introduce optional dependencies only where required.
**Notes:** PostgreSQL/S3 remain visible as candidate integrations only where their `NOT_QUALIFIED` status is explicit. No current registry artifact is represented as containing the refactored architecture.

---

## Executable Examples

| Question | Options considered | Selected |
| --- | --- | --- |
| Canonical example set | Four focused journeys; two README quick starts only; broad feature gallery | Four focused journeys |
| Existing example inventory | Audit/delete aggressively; retain unsupported archive; modernize everything | Audit/delete aggressively |
| Documentation/CI identity | Execute exact published files; import shared helper functions; duplicate snippets in tests | Execute exact published files |
| Runtime behavior | Self-verifying and disposable; persistent tutorial data; display-only | Self-verifying and disposable |

**User's choice:** Keep a small exact set that is both published documentation and executable evidence.
**Notes:** The four journeys are memory BlobStore, durable filesystem-plus-SQLite BlobStore with catalog metadata, UnifiedCache/decorator, and a custom MCAP-style format handler.

---

## Custom Handler Onboarding

| Question | Options considered | Selected |
| --- | --- | --- |
| Tutorial depth | Practical tutorial plus minimum safety contract; example only; complete developer kit now | Practical tutorial plus minimum safety contract |
| Existing protocol name | Serialization-handler prose only; retain CacheHandler terminology; rename public protocol | Rename public protocol |
| Replacement name | FormatHandler; SerializationHandler; BlobHandler | FormatHandler |
| Persisted identities | Preserve current identifiers; rename identifiers and migrate/rebuild | Preserve current identifiers |

**User's choice:** Perform a direct, alias-free `CacheHandler` to `FormatHandler` API cutover while preserving persisted data/payload identities.
**Notes:** The user challenged the initial recommendation to defer the rename. Discussion established that a pre-production direct cutover avoids freezing misleading cache-only terminology. Persisted identities already use accurate values such as `array`, `object`, `npz`, and `parquet`, so changing them would add migration cost without semantic benefit. Per-store `store.handlers.register_handler(...)` remains the accepted registration shape.

---

## the agent's Discretion

- Exact filenames, navigation ordering, concise prose, bounded example-runner mechanics, and edit sequencing for the alias-free `FormatHandler` cutover.

## Deferred Ideas

- Remove `SqlCache` in a separate Phase 10 before milestone completion. The user noted that its pull-through table/range model is outside the BlobStore problem and materially expands maintenance burden. The removal should be direct and alias-free.
- Defer the complete handler developer kit and reusable conformance suite beyond Phase 9's practical tutorial.
