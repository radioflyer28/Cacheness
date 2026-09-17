# Explicit storage migration and rebuild

<!-- migration-runbook:start -->

`BlobStore` is the only payload and canonical-catalog lifecycle owner.
`UnifiedCache` is a policy layer over one `BlobStore`; it does not migrate a
store. This guide exposes the single supported maintenance surface: the
Python-library `cacheness.storage.OfflineMigrationService` API.

The stopped-worker maintenance sequence is explicit: inventory the source,
stage a copy into the selected destination, verify every candidate, then switch
the authority selection through activation. If the source is incompatible or
crosses a supported boundary, use the separately confirmed rebuild path rather
than treating rebuild as a partial migration.

There is no CLI, `project.scripts` entry, global maintenance service, or hidden
latest-run lookup. If a command-line adapter is added in a future release, it
must be a stateless renderer over these same Python models and must require the
same explicit work directory, run ID, stopped-worker acknowledgement, evidence
path, and confirmation values.

## Before creating a maintenance run

Stop every ordinary worker that can open, read, query, initialize, write, or
delete from either participating store. Preserve the source root, its signing
provider configuration, and any authority sidecar files. Choose an
operator-owned work directory that is separate from both the source and
destination roots. The work directory stores authenticated maintenance evidence
only: it is not a payload location, lifecycle authority, or candidate-discovery
mechanism.

Ordinary opens of an existing `BlobStore` and `initialize()` validate the
layout. They never migrate, rebuild, adopt, or purge an unsupported layout. A
deliberate initializer may create a fresh current store before workers share
it, but a failed version check is a typed migration-or-rebuild requirement, not
an invitation to remove files or retry with different constructor options.

Every run supplies all safety inputs explicitly:

```python
from pathlib import Path

from cacheness.storage import OfflineMigrationService, render_migration_report


service = OfflineMigrationService(
    source=source_store,
    destination=destination_store,
    work_directory=Path("./operator-maintenance/run-2026-09-10"),
    run_id="release-2-cutover-001",
    stopped_workers_acknowledged=True,
    compatibility_edges=published_compatibility_edges,
)

inspection = service.inspect()                 # read-only, fixed source revision
plan = service.plan(inspection)                # immutable migration model
canonical_plan = plan.to_canonical_bytes()     # canonical JSON representation
human_report = render_migration_report(plan)   # rendered from the same model
```

`published_compatibility_edges` must contain the exact release-specific,
per-contract migration edges published by the target release. Do not invent an
edge, wildcard, force flag, or format conversion. The
`MigrationCompatibilityEdge.current_to_current_for_test()` helper exists only
for deterministic test fixtures; it is not a production format transition.

The plan and run evidence retain identifiers, exact counts and bytes, stable
reasons, intended actions, and a signing-provider fingerprint. They never
serialize signing material, connection strings, credentials, or provider file
locations. Human text is a rendering of validated model data; it is never a
second plan format.

### Shareable plans and protected maintenance evidence

Canonical plans are shareable authorization records, not exports of source
metadata. Each entry has a fixed allowlist of safe identifiers, generation,
payload digest, byte size, disposition, reason, and separate SHA-256 bindings
for the authenticated source catalog and signed manifest. The raw catalog
values, full manifest, handler metadata, managed locator, provider credentials,
and signing material are never in canonical plan bytes. The executor reads and
authenticates the current source manifest again, recomputes both bindings, and
refuses with `source_state_drift` before a candidate write if either binding or
the source inventory changes. A caller-provided digest cannot authorize an
unauthenticated manifest.

The main authenticated run evidence records only bounded lifecycle facts,
digests, receipt references, and stable error reasons. Existing
authority-owned candidate evidence may retain a bounded exact *target* manifest
after that candidate has been attributed by the lifecycle authority. Such a
target manifest can contain application or handler metadata needed for exact
resume or abort, but it is protected maintenance evidence: plans, reports,
diagnostics, and logs never render its descriptor body. No maintenance artifact
may contain signing keys, key-provider material, cloud credentials, service or
database credentials, or equivalent infrastructure secrets.

The entry, payload-byte, and evidence-byte limits define one independently
recoverable maintenance run. A larger store must be split before it stages any
candidate. The accepted recovery boundary is also deliberate: if a payload is
published and the process dies before the authority checkpoint, it is an
invisible, unattributed, unadopted orphan. It is outside the exact-reclamation
guarantee; it never becomes visible data, while authority visibility and
integrity checks remain fail-closed under ADR 0001.

## Version window and inspection

The current canonical post-refactor layout is the first released baseline. It
does not promise a physical migration from pre-refactor or transitional
development layouts. Each release directly supports only its current and
immediately previous released layouts. Older released layouts proceed through
their declared successive steps. An unsupported source remains inspectable but
is rebuild-only; migration refuses it without a force override.

The compatibility matrix considers independent persisted contracts instead of
trusting one top-level version:

- store layout;
- topology-specific lifecycle-authority schema/capability;
- signed manifest schema;
- handler-owned payload contract; and
- application catalog schema when transformation is declared.

`inspect()` does not mutate either store. It returns an entry-complete,
revision-bound `MigrationInspection`; every canonical entry is classified as
`migratable`, `rebuildable`, `blocked`, or `unverifiable` with a stable
`MigrationReason`. Aggregate counts and byte totals come from the same fixed
inventory. If an identity or revision changes before any later action, the plan
is stale: stop and inspect again rather than refreshing or overriding it.

## Same-backend migration: stage, verify, then activate

A supported migration replaces an entire store selection only after every
candidate entry has been copied and verified. It never makes a partial
candidate visible. Run these operations while workers remain stopped:

```python
service.stage(plan)     # immutable, run-owned candidate effects
service.verify(plan)    # verifies the complete candidate receipt
service.activate(plan)  # the selected authority performs the visibility change
```

Candidate files or objects, maintenance evidence, bucket listings, and object
metadata never select visible data. The selected lifecycle authority is the
only visibility decision. The authority records the candidate and retains the
prior selection before it changes canonical visibility.

After `activate()`, the authority state is `activated_offline`. This is a
deliberate handoff state, not a worker-serving state. Every ordinary BlobStore
entry point—including open, read, catalog query, initialization, payload
mutation, deletion, and ordinary reconciliation—remains blocked until the
operator resolves the activation. Only narrow maintenance receipt/evidence
operations plus the explicit rollback-or-finalize decision are available.

## Resolve an activated-offline handoff

Do not restart workers while the state is `activated_offline`. Choose one
explicitly while the worker stop remains in force:

```python
# Restore the exact retained prior selection. This remains an offline action.
rollback_receipt = service.rollback(plan)

# Or accept the activated selection and permanently end rollback eligibility.
final_confirmation = service.finalize_confirmation(plan)
finalize_receipt = service.finalize(plan, confirmation=final_confirmation)
```

`rollback()` only accepts the exact activated receipt and restores the retained
prior selection. `finalize()` requires the exact confirmation bound to the run,
plan, and activated receipt. Finalization permits the authority to leave the
offline handoff and seals rollback before workers may restart. Finalization
does **not** delete the retained prior store.

Physical retirement is a later, independent action. It requires a different
confirmation that binds every retained-prior identity, locator, manifest digest,
entry count, and byte total:

```python
purge_confirmation = service.purge_confirmation(plan)
purge_receipt = service.purge(plan, confirmation=purge_confirmation)
```

`purge()` is idempotent delete-or-prove-absent cleanup. A failed or ambiguous
external deletion becomes authenticated, retryable cleanup debt; it does not
change the active selection or reinterpret an already successful activation.
Do not treat `finalize()` as a purge request and do not purge before the exact
separate confirmation.

Before activation, an operator may instead call `service.abort(plan)`. Abort
can remove only an authenticated candidate proven to be owned by that run; it
keeps the source and run evidence. Once activated, abort refuses and directs
the operator to rollback or finalize.

## Resume only an exact recorded run

After interruption, reconstruct the same service and supply the exact run ID
and evidence path recorded for that run:

```python
result = service.resume(
    plan,
    run_id="release-2-cutover-001",
    evidence_path=service.evidence_path,
)
```

Resume authenticates evidence, validates its run/plan/identity bindings, and
revalidates recorded candidate outputs before it continues one legal step. It
does not scan a work directory for a newest record, infer progress from blob
presence, adopt a candidate, or repair a mismatched catalog. Missing, corrupt,
or mismatched evidence fails closed with a typed diagnostic; preserve the
artifacts and re-inspect or follow the narrow error-specific recovery action.

## Explicit rebuild for unsupported or cross-backend data

Rebuild is not a looser migration. It is the explicit path for incompatible
payload contracts, unsupported source layouts, and cross-backend moves. It
uses source handlers only after authenticating their payload bytes, writes
through the destination `BlobStore` lifecycle, and never deletes source data.
Handlers own every declared payload transformation edge; this service does not
provide a universal NPZ, Blosc2, Parquet, pickle, or dill converter.

Start with every inspected entry. To omit anything, regenerate a plan with
`RebuildExclusion.exact_keys(...)` or a closed disposition category. The
resulting plan resolves the exact excluded keys, reasons, counts, and bytes;
it must receive its own exact confirmation before any write:

```python
rebuild_plan = service.create_rebuild_plan(inspection)
rebuild_confirmation = service.rebuild_confirmation(rebuild_plan)
service.confirm_rebuild(rebuild_plan, confirmation=rebuild_confirmation)
service.stage_rebuild(rebuild_plan)
service.verify_rebuild(rebuild_plan)
service.accept_rebuild(rebuild_plan)
```

An exclusion is never a wildcard or an open-ended predicate. It is a newly
generated immutable plan input, not a way to reinterpret a prior plan. Unknown
authenticated catalog attributes are retained through the rebuild. An explicit
derived projection rebuild may run only after canonical migration activation or
rebuild acceptance; its result is derived state and can never select, revoke,
or redefine the canonical BlobStore outcome.

## Diagnostics and recovery boundaries

| Typed result or error | Operator response |
| --- | --- |
| `CacheMigrationOrRebuildRequiredError` | Preserve the root. Inspect it; do not open it with a different backend selection. |
| `CacheBlobMigrationPlanStaleError` | Stop. The source or destination changed; create a fresh inspection and plan. |
| `CacheBlobMigrationEvidenceError` or `CacheBlobMigrationEvidenceMismatchError` | Preserve evidence and candidate effects. Supply the exact run/evidence pair or re-inspect; never adopt unexplained state. |
| `CacheBlobMigrationOfflineDecisionRequiredError` | Keep workers stopped and make the named explicit decision or confirmation. |
| `CacheBlobMigrationCleanupError` or an incomplete `PurgeReceipt` | Retain authenticated cleanup debt and retry the exact purge action later. |

## Topology guarantees and non-claims

The maintenance API follows [ADR 0001](adr/0001-topology-specific-storage-guarantees.md).
One lifecycle authority controls canonical selection; immutable payload and
maintenance-evidence effects are external to that authority transaction.

- Memory authority plus memory blobs is same-process and ephemeral. It has no
  crash-durability promise.
- SQLite authority plus local filesystem blobs is the supported durable local
  topology: SQLite transactions cover authority state, while immutable files
  are reconciled external effects. This is not cross-resource ACID. Under
  contention, a typed retryable timeout may be a correct outcome.
- PostgreSQL authority with filesystem or S3 participants has deterministic
  offline adapter contracts in this phase. PostgreSQL transactions end at
  authority state; remote payload effects remain verifiable external effects.
  Real PostgreSQL and AWS S3 are not qualified in Phase 7. Phase 8 alone owns
  their live-service qualification, compatible-service scope, Windows evidence,
  and performance qualification.

This guide makes no promise of automatic or seamless migration, online-writer
coordination, cross-resource ACID, universal payload conversion, live remote
service support, Windows qualification, or a performance guarantee. Correctness
comes from the declared topology, immutable generations, authority receipts,
and deterministic reconciliation—not from a new coordination service.

## Decision contract summary (D-01 through D-22)

1. **D-01–D-05:** the first release baseline begins now; the direct window is
   current plus immediately previous release, and historical development
   layouts remain inspection/rebuild-only without compatibility readers.
2. **D-06–D-11:** inspection is complete and non-mutating; a whole migration
   blocks partial publication; stale state stops work; handlers own transforms;
   authenticated unknown catalog attributes survive; projections remain derived.
3. **D-12–D-16:** stage, verify, and activate are distinct; activation enters
   `activated_offline`; rollback needs stopped workers; prior data is retained;
   finalization ends rollback but does not delete; separately confirmed purge
   records retryable cleanup debt without revising canonical activation.
4. **D-17–D-22:** work directories contain only maintenance evidence; canonical
   JSON and human reports share a model; exact run ID/evidence-path resume is
   revalidated; evidence failures fail closed; abort is run-owned and
   pre-activation only; configured signing identity is preserved without
   serializing signing material.

<!-- migration-runbook:end -->
