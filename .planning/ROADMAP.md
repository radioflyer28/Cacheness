# Roadmap: Cacheness

## Milestones

- ✅ **v1.0 Local Readiness** — 12 phases, 165 canonical plans (completed 2026-09-20). [Milestone record](MILESTONES.md), [full roadmap archive](milestones/v1.0-ROADMAP.md), [requirements archive](milestones/v1.0-REQUIREMENTS.md), and [audit](milestones/v1.0-MILESTONE-AUDIT.md).
- **Next milestone** — not yet scoped. The backlog below and [seeds](seeds/) remain candidates; no deferred qualification is implicitly complete.

## Phases

<details>
<summary>✅ v1.0 Local Readiness — completed 2026-09-20</summary>

- [x] Phase 1: Compatibility and Security Baseline — 15/15 plans
- [x] Phase 2: Canonical Storage and Integrity Contract — 7/7 plans
- [x] Phase 3: Atomic Lifecycle and Recovery Engine — 24/24 canonical plans; direct local qualification at `5282dca`, independent verifier historical/superseded
- [x] Phase 4: Metadata Composition and Topology Contracts — 14/14 plans
- [x] Phase 5: Payload Backends and Supported Topology Qualification — 9/9 canonical plans
- [x] Phase 6: UnifiedCache Policy Composition — 11/11 plans
- [x] Phase 7: Explicit Migration and Rebuild Cutover — 24/24 plans
- [x] Phase 07.1: Obstore Payload Participant Unification (INSERTED) — 11/11 plans
- [x] Phase 8: Production Gates and Performance Stabilization — 17/17 canonical plans; remote/performance/publication gates deferred
- [x] Phase 9: Adoption and Release Surface Closure — 11/11 plans
- [x] Phase 10: Remove SqlCache Pull-Through Subsystem — 9/9 plans
- [x] Phase 11: Clean Supplemental Documentation and Normalize Validation Evidence — 13/13 plans

</details>

The v1.0 closeout is an approved local-scope planning milestone, not a published immutable release. BACK-05, QUAL-06, native Windows, and publication remain explicit nonclaims. See [ADR 0001](../docs/adr/0001-topology-specific-storage-guarantees.md) before changing lifecycle guarantees.

## Backlog

### Phase 999.1: Qualify native Windows lifecycle authority (BACKLOG)

**Goal:** Run the Phase 3 native-Windows release qualification in an eligible Python 3.11 NTFS environment and attach the machine-readable evidence.
**Requirements:** TBD
**Plans:** 0 plans

Plans:

- [ ] Verify protected-DACL provisioning, same-session SQLite contention, and different-token denial; promote with $gsd-review-backlog when ready.

### Phase 999.2: Formalize custom payload handler contract and developer kit (BACKLOG)

**Goal:** Give third-party format authors one canonical, BlobStore-first extension contract with accurate per-store registration documentation, stable payload identity/version guidance, representative native-format examples, and reusable conformance tests.
**Requirements:** TBD
**Plans:** 0 plans

Plans:

- [ ] Replace stale global cache-handler examples with `store.handlers.register_handler(...)`, document lifecycle ownership and migration compatibility responsibilities, and provide a contract-test kit; promote with $gsd-review-backlog when ready.
