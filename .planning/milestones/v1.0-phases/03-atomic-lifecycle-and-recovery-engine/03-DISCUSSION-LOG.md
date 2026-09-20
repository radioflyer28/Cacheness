# Phase 3: Atomic Lifecycle and Recovery Engine - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-30
**Phase:** 3-atomic-lifecycle-and-recovery-engine
**Areas discussed:** Generation publication, failure residue, delete/clear/close convergence, reconciliation policy, same-key coordination
**Mode:** `--auto`; the recommended option was selected for each question under the user's standing authorization.

---

## Generation Publication

| Option | Description | Selected |
|--------|-------------|----------|
| Immutable candidate plus conditional manifest publication | Preserve old authority until one expected-generation swap publishes the verified new generation | ✓ |
| In-place replacement | Replace payload/metadata at stable locations and rely on rollback | |
| Prepared record visible to readers | Let readers interpret transitional manifests | |

**Selection:** Immutable candidate plus conditional manifest publication.
**Notes:** This follows Phase 2's committed-only read contract and makes one authority point testable.

## Failure Residue

| Option | Description | Selected |
|--------|-------------|----------|
| Durable intent and provenance-marked residue | Record bounded transition evidence before persistent side effects and classify pre/post authority failures | ✓ |
| Infer from backend contents | Reconstruct intent from filenames, object names, or payload contents | |
| Best-effort rollback only | Attempt cleanup immediately and discard recovery state | |

**Selection:** Durable intent and provenance-marked residue.
**Notes:** Recovery may never guess provenance or erase the only valid copy.

## Delete, Clear, and Close Convergence

| Option | Description | Selected |
|--------|-------------|----------|
| Tombstone then reclaim | Conditionally revoke a generation, then idempotently reclaim owned payload and lifecycle evidence | ✓ |
| Delete payload first | Remove bytes before changing manifest authority | |
| Delete manifest first | Remove authority without durable reclamation evidence | |

**Selection:** Tombstone then reclaim, with bounded clear snapshots and ownership-aware idempotent close.
**Notes:** A store-wide clear may use an explicit admission barrier; normal distinct-key work remains concurrent.

## Reconciliation Policy

| Option | Description | Selected |
|--------|-------------|----------|
| Evidence-gated dry-run-first reconciliation | Report by default; apply only provenance-proven repair/reclaim actions with resumable checkpoints | ✓ |
| Aggressive automatic cleanup | Delete anything not reachable from the current manifest set | |
| Report only | Never offer safe automated repair | |

**Selection:** Evidence-gated dry-run-first reconciliation.
**Notes:** Ambiguous or future evidence stays untouched or is safely quarantined only when the backend proves that operation.

## Same-Key Coordination

| Option | Description | Selected |
|--------|-------------|----------|
| Per-key coordination plus CAS | Coordinate locally by key and use backend generation checks as the cross-instance correctness boundary | ✓ |
| One global store lock | Serialize all keys through a single mutex | |
| Unchecked last writer wins | Allow overwrites/deletes without expected-generation checks | |

**Selection:** Per-key coordination plus backend generation compare-and-swap.
**Notes:** Unsupported topologies fail explicitly rather than advertising a guarantee they cannot enforce.

## the agent's Discretion

- Exact lifecycle/journal/report class names and bounded record encoding.
- Lock striping versus dynamically retained per-key locks, bounded retry values, and safe backend-specific quarantine representation.

## Deferred Ideas

- Full backend capability and CAS implementations: Phases 4 and 5.
- UnifiedCache policy composition: Phase 6.
- Stored-format migration execution: Phase 7.
