# Project Retrospective

*A living record of completed planning milestones. Historical observations do not expand current product guarantees.*

## Milestone: v1.0 — Local Readiness

**Closed:** 2026-09-20
**Phases:** 12 accepted | **Plans:** 165 canonical | **Recorded tasks:** 327
**Scope:** local readiness; no immutable release publication

### What Was Built

- One `BlobStore` lifecycle authority with signed manifests, immutable payload generations, typed outcomes, exact deletion, and attributable recovery.
- `UnifiedCache` as cache policy over a `BlobStore` engine, plus direct application-defined catalog metadata and explicit stopped-worker migration/rebuild tooling.
- One guarded obstore participant for built-in filesystem, memory, and S3 payload mechanics; custom format handlers retain a private path-based seam.
- A source-free wheel and reproducible local package, test, lint, examples, and documentation gates. `SqlCache` and native TensorFlow support were removed.

### What Worked

- ADR 0001 bounded claims by topology and kept external payload effects separate from authority transactions.
- The obstore participant removed duplicate low-level file/S3 mechanics while leaving lifecycle ownership in `BlobStore`.
- Exact-revision validation and a final milestone audit made local evidence and remote/platform nonclaims distinguishable.

### What Was Inefficient

- Early Phase 3 race fixes repeatedly added coordination across filesystem, SQLite, projections, and cache surfaces; the architecture had to be narrowed before finite local qualification became tractable.
- Generated milestone metadata counted the superseded Phase 03-19 plan and copied historical one-liners into the current accomplishments list; closeout required manual normalization.
- The open-artifact acknowledgment writer did not support heading-delimited deferred-item files, requiring precise in-place status markers and a rescan.

### Patterns Established

- One lifecycle authority; immutable payloads are external participants with attributable recovery, not a distributed ACID transaction.
- Initialize before shared workers; perform schema cutover with stopped workers; do not turn performance targets into runtime progress guarantees.
- Keep obsolete pre-production surfaces out of the supported package while preserving explicit future migration tooling and dated evidence.
- A milestone may close on an approved local scope only when deferred live-service, platform, performance, and publication claims remain visibly nonpassing.

### Key Lessons

1. State the topology and finite safety invariants before adding synchronization; an unbounded race-repair loop is not a substitute for one authority boundary.
2. Treat `BlobStore` as the engine for cache instances, while allowing cache and direct-persistence instances to have separate retention namespaces.
3. Preserve historical failures and overrides as evidence, but prevent them from becoming active work or stronger release claims by accident.

### Cost Observations

- Session and model-mix totals are not reliably captured in the milestone artifacts; no estimate is asserted.
- Rework was concentrated in the Phase 3 coordination redesign and later evidence/metadata normalization.

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Phases | Key change |
|-----------|--------|------------|
| v1.0 | 12 accepted | Moved from overlapping lifecycle surfaces to ADR-bounded single authority and exact local evidence |

### Cumulative Quality

| Milestone | Evidence |
|-----------|----------|
| v1.0 | 42/42 in-scope requirements; 12/12 accepted phase closures; 7/7 integrations; 8/8 executable flows; Phase 11 fresh wheel and non-live suite passed |

### Top Lessons (Verified Across Milestones)

Not yet applicable; one milestone has closed.
