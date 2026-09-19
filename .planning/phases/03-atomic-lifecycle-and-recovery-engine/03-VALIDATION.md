---
phase: 03
slug: atomic-lifecycle-and-recovery-engine
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-09-04
updated: 2026-09-19
governing_decision: docs/adr/0001-topology-specific-storage-guarantees.md
qualified_commit: 5282dcabc7157037d95144527a220f51e51c9803
execution_mode: direct_primary_agent
qualification_provenance: user-approved direct qualification; no independent verifier ran
completion_disposition: scoped-local-qualification-confirmed
nonclaim_boundary: >-
  This record confirms only declared local topology evidence. It does not
  qualify live services, native Windows, controlled-Linux performance, or
  immutable publication.
---

# Phase 03 — Canonical Validation Record

> Compact current evidence for the approved direct qualification. Historical
> implementation detail, failed development attempts, and the prior draft
> strategy remain in Git history and the dated ledger; they are not live
> validation authority.

## Scope and Provenance

The qualified durable topology is **SQLite/local filesystem**: a SQLite
lifecycle authority and immutable local-filesystem generations on one host,
including initialized independent workers. The only memory claim is
single-process memory authority plus memory payloads; it is not crash-durable
or multi-process evidence.

The user approved direct qualification of bounded Plans 03-21 through 03-25.
The controlling qualification revision is
`5282dcabc7157037d95144527a220f51e51c9803` (`5282dca`), with production work
at `c37f418` and one test-only initialization correction at the qualified
revision. `execution_mode: direct_primary_agent` is deliberate provenance: no
independent verifier, checker, reviewer, or subagent ran, and this record does
not relabel the historical process as independent verification.

## ADR 0001 Guarantee Boundary

[ADR 0001](../../../docs/adr/0001-topology-specific-storage-guarantees.md)
controls this record.

| Class | Qualified local boundary |
| --- | --- |
| Integrity | Readers observe a complete old or new immutable generation; authority promotion selects visibility; malformed metadata, unsafe paths, and integrity failures fail closed. |
| Recovery | Durable intent and exact cleanup debt make incomplete local transitions deterministic to reconcile without guessing from timing or payload presence. |
| Progress | Initialized same-key contention may complete, conflict, or produce a typed retryable timeout; universal contender success is not required. |
| Performance | Benchmarks remain named-workload regression evidence only and never become a lifecycle correctness deadline. |
| ACID scope | SQLite is transactional authority; filesystem effects use immutable publication plus reconciliation, not a cross-resource ACID transaction. |

## Evidence Chain

- [`03-25-SUMMARY.md`](03-25-SUMMARY.md) records the direct-primary-agent
  disposition, exact qualified commit, and the final Python 3.11/3.13,
  integrity, progress, Ruff, and benchmark results.
- [`phase3-direct-implementation-2026-09-06.md`](../../../docs/phase3-direct-implementation-2026-09-06.md)
  is the dated implementation and qualification ledger, including the initial
  failed uninitialized-worker schedule and its test-only correction.
- Phase 07.1 corroborates the same single lifecycle authority after payload
  participant unification; it does not turn obstore publication into a second
  authority or cross-resource transaction.
- Phase 8's `LOCAL_READY` evidence corroborates deterministic local readiness
  while retaining service, platform, performance, and publication nonclaims.

## Current Finite Confirmation

Phase 11 ran this command once after the runtime, package, CI, and
documentation cutovers stabilized:

```bash
uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false tests/test_phase3_gap_acceptance.py tests/test_phase3_local_workflows.py tests/test_blob_store_atomic_lifecycle.py tests/test_blob_store_reconciliation.py tests/test_unified_cache_lifecycle_authority.py tests/test_integration.py -x
```

**Result:** 50 passed; exit 0 on 2026-09-19. This confirms only the named
current local integrity, recovery, and cache-over-store contracts; it is not a
new full qualification or an authorization to modify lifecycle behavior.

## Per-Task Verification Map

| Task ID | Evidence | Class | Runnable command or source | Status |
| --- | --- | --- | --- | --- |
| 03-25-DIRECT | Bounded direct implementation and exact-commit qualification at `5282dca` | Integrity, recovery, progress, performance | `03-25-SUMMARY.md` and the dated direct ledger | ✅ green |
| 11-REG-01 | Finite post-cutover current confirmation | Integrity, recovery, cache-over-store | Exact six-file command above | ✅ green |
| 03-HISTORICAL-DRAFT | Former broad Plan 20 narrative and pending task map | Historical planning | Git history and dated ledger | explicitly superseded |

## Explicit Nonclaims

- Native **Windows** remains `UNAVAILABLE`/`NOT_QUALIFIED`; Phase 999.1 owns
  native evidence and no portable wheel tag substitutes for it.
- Live **PostgreSQL** and **Amazon S3** lifecycle qualification remain
  `DEFERRED`/`NOT_QUALIFIED`; local or mocked evidence is not a live-service
  claim.
- Multi-host behavior, a full supported-Python matrix, broad catalog schemas,
  and general stored-data migration are outside this Phase 3 record.
- **controlled-Linux** performance remains `DEFERRED`/`NOT_QUALIFIED` under
  SEED-006; local timing does not qualify it.
- Immutable release publication remains `DEFERRED`/`NOT_PUBLISHED` under
  SEED-007.
- The finite gates are not proof of every interleaving, wait-free operation,
  starvation freedom, universal contender success, or one ACID transaction
  across SQLite and filesystem resources.

## ADR Stop Conditions

If a future finite gate exposes a genuine integrity or recovery defect in the
declared topology, stop, report its invariant, topology, reproduction, and
failing node, then plan separately. Do not respond by adding a lock, queue,
lease, sidecar, retry/timing mechanism, projection gate, second lifecycle
authority, or cross-resource commit protocol. A documented typed contention
outcome or timing variation is a progress/performance result, not evidence
that the safety or recovery contract failed.

## Validation Sign-Off

- ✅ The approved SQLite/local filesystem and single-process memory scope is
  stated without topology inflation.
- ✅ Integrity, recovery, progress, performance, and ACID boundaries follow
  ADR 0001.
- ✅ Exact direct-primary-agent provenance, `5282dca`, evidence chain, and the
  one-run Phase 11 command/result are recorded.
- ✅ Windows, PostgreSQL, Amazon S3, controlled-Linux, and publication
  nonclaims remain explicit.

**Approval:** user-approved direct qualification at `5282dca`; this current
evidence normalization adds no independent-verifier claim.
