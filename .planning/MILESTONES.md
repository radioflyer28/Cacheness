# Project Milestones: Cacheness

## v1.0 Local Readiness (Shipped: 2026-09-20)

**Delivered:** A locally qualified `BlobStore` foundation with `UnifiedCache` as its policy layer, customizable catalog metadata, explicit offline migration/rebuild tooling, and one guarded obstore payload participant. This is a planning milestone, not an immutable published release.

**Phases completed:** 12 phases (165 canonical plans, 327 recorded tasks). The raw plan-file count is 166 because superseded Phase 03-19 remains as history.

**Key accomplishments:**

- Established one authoritative storage lifecycle with immutable generations, exact cleanup and attributable recovery, bounded by ADR 0001's topology-specific guarantees.
- Composed cache policy over `BlobStore` and added direct application-defined catalog metadata without a second lifecycle authority.
- Unified built-in filesystem, memory and S3 payload mechanics under obstore while retaining private path-based format handlers and store-local registration.
- Delivered explicit stopped-worker versioned migration/rebuild tooling and a reproducible local package, test, lint and documentation qualification path.
- Removed the unrelated `SqlCache` subsystem and dormant TensorFlow integration; published four executable local examples and current BlobStore-first guidance.

**Evidence:** [v1.0 milestone audit](milestones/v1.0-MILESTONE-AUDIT.md) passed for the approved local scope (42/42 in-scope requirements; 12/12 accepted phase closures). Phase 3 was directly qualified at `5282dca` under the user's process override; its historical independent-verifier record is `historical_superseded`, not `passed`.

**Known verification overrides:** 14 newly acknowledged artifact records, 0 carried from a prior close (see [STATE.md](STATE.md#deferred-items)). They include six still-relevant future-work seeds and eight historical phase observations; SEED-001 was separately marked fulfilled.

### Known Gaps

- `BACK-05` — live PostgreSQL/Amazon S3 qualification remains `NOT_QUALIFIED` under SEED-007.
- `QUAL-06` — controlled-Linux performance budgets remain `NOT_QUALIFIED` under SEED-006.
- Native Windows qualification remains `UNAVAILABLE`/`NOT_QUALIFIED` (backlog Phase 999.1). Immutable release publication remains `NOT_PUBLISHED` under SEED-007. A local Git tag is not publication.

**Timeline:** 2026-08-29 to 2026-09-20 (22 days). **Git range:** project initialization `2d79f3f` through the v1.0 closeout commit. **Next:** select the next milestone from the preserved backlog and seeds; qualify remote services before advertising them or publishing a release.

---
