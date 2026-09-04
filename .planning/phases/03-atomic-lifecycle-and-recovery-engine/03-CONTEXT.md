# Phase 3: Atomic Lifecycle and Recovery Engine - Context

**Gathered:** 2026-08-30; architecture replanned 2026-09-04
**Status:** Ready for transactional-authority replanning

<domain>
## Phase Boundary

Build the backend-neutral lifecycle engine below direct `BlobStore` operations so
write, overwrite, delete, clear, close, and recovery preserve an old or new complete
generation. This phase defines atomic publication, idempotent reclamation, durable
recovery evidence, dry-run/resumable reconciliation, and same-key coordination. It
does not yet implement the complete PostgreSQL/S3 backend matrix, rewire
`UnifiedCache`, or execute stored-format migrations.

</domain>

<decisions>
## Implementation Decisions

### Immutable Generations and Publication

- **D-01:** Payload generations are immutable after candidate creation. A write serializes to a new generation-specific locator, verifies the candidate, and conditionally publishes one committed canonical manifest that names it. Payload bytes at an already published generation locator are never overwritten in place. — **Reversibility:** one-way — In-place generations would invalidate the same-snapshot and recovery guarantees and require another stored-layout migration.
- **D-02:** Publication uses an expected-generation compare-and-swap contract. Creating an absent key expects absence; overwriting expects the authenticated committed generation read by the operation. A failed comparison returns a typed conflict and never revokes the winner.
- **D-03:** Readers observe only committed manifests from Phase 2. Prepared intent and cleanup records are recovery evidence, not alternate normal-read authorities. A read may take one bounded retry only when it detects a generation changed during acquisition; it must not guess between generations.
- **D-04:** The authority point is the successful conditional manifest publication. Before it, the previous committed generation remains authoritative; after it, the new committed generation remains authoritative even if retirement of the old payload fails.

### Failure Residue and Operation Evidence

- **D-05:** Every multi-step mutation creates bounded, versioned operation evidence before the first externally persistent side effect. It records operation ID, logical key, expected generation, candidate/new generation, relevant locators, intended transition, and progress without copying payload contents.
- **D-06:** Operation evidence and generated locators carry enough authenticated or strictly validated provenance to distinguish library-owned residue from unrelated files/objects. Recovery never infers ownership from a loose filename pattern or handler payload contents.
- **D-07:** Failures before authority publication preserve the previous generation and leave the candidate plus intent detectable for rollback/reconciliation. Failures after authority publication preserve the new generation and leave old-generation reclamation as explicit cleanup debt. — **Reversibility:** costly — External operators and later migration tooling will depend on this pre/post-authority classification.
- **D-08:** Serialization failure before a candidate exists leaves no persistent lifecycle record. Once operation evidence or a candidate is persisted, cleanup failure is surfaced as a typed recoverable outcome rather than hidden behind the original operation result.

### Delete, Clear, and Close Convergence

- **D-09:** Delete conditionally publishes a signed tombstone for the expected generation before reclaiming payload bytes and retiring the tombstone. Repeated delete is successful for an already absent key, resumes reclamation for the same tombstone, and conflicts rather than deleting a newer generation.
- **D-10:** Clear is a resumable store-wide operation built from an authenticated bounded snapshot of keys and per-entry expected generations. It may use a store-wide admission barrier while establishing/committing that clear, but ordinary operations on distinct keys are not globally serialized. Keys created after the clear snapshot are not accidentally deleted.
- **D-11:** Cleanup is idempotent: missing already-reclaimed payloads or operation records count as completed steps when provenance and generation still match; a locator now owned by a different generation is a conflict and is never removed.
- **D-12:** `close()` is ownership-aware and idempotent. It stops new operations on that instance, waits for or deterministically cancels its in-flight work, flushes owned durable state, and releases owned resources. It does not clear stored user data and does not close caller-injected backends.

### Reconciliation Policy

- **D-13:** Reconciliation is dry-run by default and returns both a stable machine-readable report and a human-readable summary. Findings include authoritative generation, operation provenance, residue type, proposed action, reason, and whether the action is safe, blocked, or requires confirmation.
- **D-14:** Apply mode is resumable and checkpoints each completed action. Re-running after interruption converges without repeating destructive work or losing the only valid copy.
- **D-15:** Automatic repair is limited to outcomes proven by authenticated manifests, validated operation records, generation checks, and contained/backend-owned locators. Ambiguous, malformed, unsupported-version, or provenance-free evidence is quarantined only when a safe backend-native quarantine is possible; otherwise it is reported untouched.
- **D-16:** Reconciliation never deserializes trusted application payloads merely to decide ownership or lifecycle state, never guesses future schemas, and never silently deletes the last valid generation.

### Same-Key Coordination

- **D-17:** In-process work uses a bounded per-key coordination mechanism, not one global mutex. Coordination entries are retired when unused so unbounded key cardinality does not become a memory leak.
- **D-18:** Backend generation compare-and-swap is the correctness boundary across processes/instances; local locks are an optimization and ordering aid, not the cross-process guarantee. A topology that cannot truthfully provide the required conditional publication fails with a typed unsupported-capability outcome rather than downgrading.
- **D-19:** Same-key write/write races have one conditional-publication winner; losers receive a typed conflict and clean or report only their own candidate residue. Write/delete races obey the same expected-generation rule. Reads return a complete committed generation or a typed lifecycle/conflict outcome, never mixed bytes and metadata.
- **D-20:** Operations on distinct keys proceed independently except for an explicit store-wide clear/reconciliation admission barrier. Lock acquisition order for multi-key work is deterministic to avoid deadlocks.

### Local Authority Trust and Windows Scope

- **D-21:** The OS principal that owns a local store is inside the trusted deployment boundary for lifecycle-control availability. Cacheness validates control-object type, containment, identity, and authenticated contents and fails closed when observable substitution occurs, but it does not promise continued operation or immutable per-key authority if that same principal deliberately deletes or rebinds every lifecycle authority object while the store is live. Ordinary Cacheness processes never perform such rebinding. — **Reversibility:** costly — Defending against a hostile store owner would require an external coordinator, privileged mandatory controls, or store-wide serialization and therefore changes the architecture or deployment model.
- **D-22:** In this milestone, Windows local-store coordination is supported only among processes running as one OS user in one interactive or service session. Cross-user, cross-service, and cross-session access to the same local store is unsupported and must fail or be prevented by deployment ACLs; supporting it later requires an explicit global authority namespace, ACL/security-descriptor contract, and native Windows validation. Advertised Windows compatibility otherwise remains in force. — **Reversibility:** reversible — A later milestone may broaden the topology after implementing and validating that authority contract.

### Transactional Authority Replan

- **D-23:** The file-native lifecycle scheduler built from operation receipts, inventory events, heads, anchors, cursors, pending-control records, and staged control files is rejected. The replacement removes that protocol rather than wrapping or incrementally repairing it. D-05 through D-20 remain behavioral requirements only where they do not prescribe that discarded mechanism. — **Reversibility:** costly — Reintroducing file-native transactional coordination would require new proof that it is smaller and more reliable than the transactional authority.
- **D-24:** `BlobStore` depends on one deep lifecycle-authority module whose interface exposes complete entry-state transactions, not individual receipt/checkpoint/storage primitives. The authority atomically owns canonical manifest state, mutation intent, cleanup debt, clear membership/progress, and reconciliation checkpoints. Callers never coordinate those records themselves.
- **D-25:** Payload generations remain immutable native handler output outside the authority transaction. A durable authority intent is committed before the first persistent payload side effect; a later authority transaction conditionally promotes the verified generation and records any cleanup debt. Recovery queries indexed intent/debt rows and never discovers protocol state by scanning filenames.
- **D-26:** The local persistent reference adapter uses Python's standard-library SQLite engine as the transactional authority, with explicit durability configuration and schema/version migration. JSON remains a supported metadata projection and compatibility representation, but it is not an independent cross-process transaction authority. Configuration that requests durable multi-process guarantees without a capable authority fails with a typed unsupported-capability outcome rather than silently downgrading.
- **D-27:** The in-memory authority adapter provides deterministic same-process behavior for tests and ephemeral stores. PostgreSQL and S3-capable authority adapters use their native transaction or conditional-write facilities through the same lifecycle-authority interface in Phases 4 and 5; Phase 3 must not recreate the discarded filesystem scheduler inside those adapters.
- **D-28:** Distinct-key serialization, payload verification, and cleanup proceed concurrently. A backend may serialize only the short authority commit itself (SQLite has one writer), and no authority transaction or global/store/family lease may span handler serialization, payload upload/fsync, payload verification, or reclamation.
- **D-29:** Clear stores its exact target generations and progress as transactional authority rows. Reconciliation pages indexed authority rows using stable database cursors and explicit work/byte/action limits; it does not maintain a second file inventory or infer provenance.
- **D-30:** The implementation must delete or retire the abandoned file-native scheduler modules and tests rather than carry both lifecycle engines. Replacement tests exercise behavior through the lifecycle-authority interface and `BlobStore`; low-level receipt/inventory tests are historical evidence, not the new test surface.
- **D-31:** Windows local persistence uses the same SQLite transactional authority and the D-22 one-user/session trust scope. No custom Win32/POSIX lock-file authority, inode/handle identity registry, extended-attribute reservation, or platform-specific receipt protocol is part of the replacement.

### the agent's Discretion

- Exact class/module names for the lifecycle authority, transaction records, and reconciliation reports.
- Whether local per-key coordination uses lock striping or dynamically retained locks, provided unrelated keys remain concurrent and retention is bounded.
- Exact bounded database busy retry/backoff values and authority-row encoding, provided failure outcomes remain deterministic and testable.
- Whether safe quarantine is implemented as a contained rename, backend namespace move, or immutable report-only disposition for a backend that cannot move atomically.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Product and Requirements

- `.planning/PROJECT.md` — Establishes `BlobStore` lifecycle ownership, compatibility, reliability, concurrency, and backend-neutral constraints.
- `.planning/ROADMAP.md` § Phase 3 — Defines old-or-new generation visibility, failure recovery, idempotent operations, reconciliation, and race success criteria.
- `.planning/REQUIREMENTS.md` — Phase 3 requirements `STOR-03` through `STOR-07`.

### Prior Locked Contracts

- `.planning/phases/02-canonical-storage-and-integrity-contract/02-CONTEXT.md` — Locks canonical manifests, committed-only visibility, signed generation/state/locator fields, typed conflicts, and read ordering.
- `.planning/phases/02-canonical-storage-and-integrity-contract/02-VERIFICATION.md` — Verifies the Phase 2 authority/read boundary and the absence of any Cacheness payload wrapper/header.
- `.planning/phases/01-compatibility-and-security-baseline/01-CONTEXT.md` — Locks containment, evidence preservation, public error reasons, native handler ownership, and trusted-payload boundaries.
- `docs/SECURITY.md` — Defines the trusted application payload model and boundary-hardening obligations.

### Codebase Evidence

- `.planning/codebase/ARCHITECTURE.md` — Maps current two-step lifecycle boundaries and disconnected backend seams.
- `.planning/codebase/INTEGRATIONS.md` — Describes backend/service integration constraints that later phases must implement without weakening this engine.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `src/cacheness/storage/manifest.py`: Signed canonical generation, lifecycle state, locator, digest, and size fields provide the authority record this phase transitions.
- `src/cacheness/storage/manifest_repository.py`: Exact raw-record adapters are the seam to extend with conditional publication; current local adapters already preserve canonical bytes.
- `src/cacheness/storage/clear_recovery.py`: Phase 1 bounded journals, admission locks, prepared/committed recovery, and tombstone reclamation are proven patterns to generalize rather than replace with inference.
- `src/cacheness/storage/guarded_handler_io.py`: Candidate publication, contained snapshots, cleanup, and owned-resource behavior can support immutable generation I/O.
- `src/cacheness/error_handling.py` and `src/cacheness/storage/read_contract.py`: Stable typed lifecycle, conflict, integrity, and backend outcomes already exist.

### Established Patterns

- Phase 2 treats exact raw canonical manifest bytes as the metadata authority and rejects non-committed states before payload access.
- Candidate payloads are created separately from authority publication, and clear recovery already distinguishes prepared from committed operations across reopen.
- Reads preserve evidence and authenticate before using locators; recovery must retain the same fail-closed ordering and containment rules.
- Native handlers own payload formats. Lifecycle records describe and coordinate payloads but never wrap or reinterpret their bytes.

### Integration Points

- `BlobStore.put`, `delete`, `clear`, `close`, and metadata mutation must route through one lifecycle-operation engine.
- `ManifestRepository` needs a backend-neutral expected-generation/conditional-write seam without claiming Phase 4's full advertised backend implementation.
- `GuardedHandlerIO` needs generation-specific candidate/publication/reclamation primitives with ownership checks.
- Direct read surfaces must coordinate with active transitions while preserving the Phase 2 typed outcome and one-snapshot contract.
- Reconciliation must consume manifest repositories, operation evidence, and payload inventory through bounded interfaces; full S3/PostgreSQL capability plumbing remains Phase 4/5.

</code_context>

<specifics>
## Specific Ideas

- The lifecycle engine belongs to `BlobStore`; cache policy must eventually consume it rather than duplicate it.
- Correctness is old-complete or new-complete at every failure boundary, with residue preserved as evidence until a deterministic recovery action succeeds.
- Native NumPy, Blosc2, PyArrow/Parquet, pickle, and dill payloads remain untouched by lifecycle framing.

</specifics>

<deferred>
## Deferred Ideas

- Concrete capability declarations and full metadata-backend conditional publication across JSON, memory, SQLite, and PostgreSQL — Phase 4.
- Full filesystem, memory, and S3 payload lifecycle implementation and matrix verification — Phase 5.
- `UnifiedCache` delegation, TTL/eviction/invalidation policy, and miss/statistics translation — Phase 6.
- Stored-format inventory and copy-verify-switch migration execution — Phase 7; Phase 3 reconciliation handles lifecycle inconsistency, not format migration.

</deferred>

---

*Phase: 3-atomic-lifecycle-and-recovery-engine*
*Context gathered: 2026-08-30*
