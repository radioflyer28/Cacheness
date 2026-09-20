# Phase 7: Explicit Migration and Rebuild Cutover - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-09
**Phase:** 7-explicit-migration-and-rebuild-cutover
**Areas discussed:** Supported version window, Plan and incompatibility behavior, Cutover and retirement, Resume and operator evidence

---

## Supported version window

| Question | Options presented | Selected |
|----------|-------------------|----------|
| Existing persisted source layouts | Current canonical baseline only; selected transitional layouts; all historical layouts | Current canonical baseline only |
| Direct support window | Current and immediately previous release; current and two previous; every release indefinitely | Current and immediately previous release |
| Independently versioned qualification | Explicit compatibility matrix; store version only; best-effort detection | Explicit compatibility matrix |
| Outside supported window | Inspect and report rebuild-only; reject inspection; force override | Inspect and report rebuild-only |

**User's choice:** `1a 2a 3a 4a`
**Notes:** The user asked which formats were in scope. The distinction was clarified among the overall store layout, topology-specific authority schemas, signed manifest schema, handler-owned native payload contracts, application catalog schema, and derived projections. The user then clarified whether the contract targeted older-to-next-release or next-release-to-future migration. The agreed intent is the latter: historical layouts are rebuild-only, while the current canonical post-refactor layout becomes the first supported release baseline.

---

## Plan and incompatibility behavior

| Question | Options presented | Selected |
|----------|-------------------|----------|
| Inventory detail | Entry-complete classification; aggregate plan; sampled inspection | Entry-complete classification |
| Incompatible entry in supported source | Block whole-store migration; migrate compatible entries; silently skip incompatible entries | Block whole-store migration |
| Rebuild exclusions | Exact confirmed scope; category-only confirmation; warning-only exclusions | Exact confirmed scope |
| Source changed after inspection | Reject stale plan; refresh automatically; permit override | Reject stale plan |

**User's choice:** `1a 2a 3a 4a`
**Notes:** Migration is whole-store and fail-closed. Partial selection belongs to a separately generated and confirmed rebuild plan.

---

## Cutover and retirement

| Question | Options presented | Selected |
|----------|-------------------|----------|
| Copy and activation boundary | Stage/verify then explicit activate; auto-activate after verification; external switch | Stage/verify then explicit activate |
| Rollback eligibility | Offline only; while destination unchanged; never | Offline only |
| Prior-store retention | Explicit operator action; configured period; until next migration | Explicit operator action |
| Finalize semantics | End rollback eligibility, purge separately; finalize and purge; verify only | End rollback eligibility, purge separately |

**User's choice:** `1a 2a 3a 4a`
**Notes:** Restarting writers ends rollback eligibility. No timer or ordinary lifecycle cleanup may delete migration recovery material.

---

## Resume and operator evidence

| Question | Options presented | Selected |
|----------|-------------------|----------|
| Evidence location | Explicit external work directory; destination-internal directory; default user-state directory | Explicit external work directory |
| Authoritative representation | Canonical JSON plus rendered human report; human-first; separate formats | Canonical JSON plus rendered human report |
| Continuation | Explicit run ID/evidence path; locate latest automatically; rerun original command | Explicit run ID/evidence path |
| Missing or corrupt evidence | Fail closed with diagnostics; reconstruct from candidate; restart automatically | Fail closed with diagnostics |
| Abort semantics | Remove only unactivated run-owned candidate; remove any run destination; mark abandoned only | Remove only unactivated run-owned candidate |

**User's choice:** `1a 2a 3a 4a 5a`
**Notes:** Maintenance evidence is not runtime authority. Signing secrets remain behind the configured provider and never appear in plans or logs.

---

## the agent's Discretion

None of the discussed operator-visible behavior was delegated. Normal implementation details remain open only within the recorded constraints.

## Deferred Ideas

None.
