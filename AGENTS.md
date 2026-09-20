<!-- GSD:project-start source:PROJECT.md -->

## Project

**Cacheness** is a Python blob-storage and caching library for arbitrary
objects, arrays, dataframes, and function results. `BlobStore` is the storage
foundation; `UnifiedCache` uses a `BlobStore` as its engine and adds cache
policy. A direct-persistence store and a cache-policy store may be separate
instances/namespaces.

**Core value:** applications store and retrieve data through one
backend-neutral lifecycle, with cache policy layered above it without
compromising integrity or cleanup correctness.

### Hard constraints

- This is a pre-production cutover. Do not add compatibility shims for retired
  development surfaces or layouts; retain explicit version and offline
  migration/rebuild tooling for future supported releases.
- `BlobStore` owns lifecycle, authority selection, immutable payload
  publication, recovery, and exact cleanup. `UnifiedCache` owns only keys,
  TTL, outcomes, statistics, invalidation, and bounded maintenance policy.
- Do not make cache policy a second catalog or lifecycle authority.
- Keep application payloads trusted, but enforce safe parsing, path
  containment, signing, and fail-closed integrity boundaries.
- The supported topology families are filesystem, memory, and S3 payloads with
  memory, SQLite, JSON, and PostgreSQL catalog/authority roles as declared by
  topology capability contracts. Do not infer cross-resource ACID or universal
  availability from a shared interface.
- Maintain Python 3.11+ support. Local readiness is qualified; live remote
  service and controlled-Linux performance release qualification remain
  deliberately deferred.

<!-- GSD:project-end -->

<!-- storage-guarantees:start source:docs/adr/0001-topology-specific-storage-guarantees.md -->

## Storage Lifecycle Design Guardrail

Before planning or changing storage lifecycle, concurrency, recovery, backend
topology, timeouts, or `UnifiedCache`/`BlobStore` composition, read
`docs/adr/0001-topology-specific-storage-guarantees.md`. It defines mandatory
safety invariants, topology-specific progress guarantees, and stop conditions
that prevent performance targets from becoming unsupported atomicity or
availability promises.

<!-- storage-guarantees:end -->

## Current Architecture

```text
application ──> BlobStore lifecycle ──> one LifecycleAuthority
                    │                   └─ immutable obstore participant
                    └─ store-local HandlerRegistry

application ──> UnifiedCache policy ──> one BlobStore
```

- `BlobStore` is the only coordinator of visibility, descriptors, intent,
  cleanup debt, reconciliation, exact deletion, and recovery.
- A `StoreTopology` resolves one authority and one payload participant.
  Filesystem paths, payload listings, and projections are not authority.
- Handlers receive only a private contained staging/snapshot path. Register a
  custom file format through `store.handlers.register_handler(...)`; never
  bypass `BlobStore` publication with an application-managed path.
- `UnifiedCache` may accept a caller-owned `BlobStore` or build an owned store
  from a topology. It cannot turn policy maintenance or projections into a
  commit precondition.

## Technology and dependencies

- Python `>=3.11`, with Python 3.13 pinned for this checkout; use `uv` and the
  committed `uv.lock` for dependency operations.
- `obstore==0.11.1` implements built-in immutable payload object I/O; it does
  not own metadata transactions or lifecycle coordination.
- SQLAlchemy supports SQLite metadata/projection work and the PostgreSQL
  authority path. `psycopg` is optional support for the PostgreSQL path.
- NumPy is a base dependency. Optional handler integrations include Blosc2,
  pandas/PyArrow, Polars, dill, and orjson. PostgreSQL and S3 remain optional
  topology integrations. Keep extras and guarded imports synchronized.
- The five published extras are `recommended`, `dataframes`, `s3`,
  `postgresql`, and `cloud`. Do not add a compatibility extra.

## Implementation conventions

- Use lowercase `snake_case.py`, PascalCase classes/exceptions, and typed
  public boundaries. Place public re-exports deliberately in package
  `__init__.py` files.
- Keep functions focused. Do not extend a coordinator when a focused helper or
  existing participant contract is the real seam.
- Use domain exceptions from `error_handling.py` or handler-specific exceptions
  from `interfaces.py`; catch narrow operational errors and preserve causes
  with `raise ... from exc` when translating them.
- Keep optional imports lazy or capability-guarded. New code must not add to
  the current Ruff baseline; run scoped `ruff check` over touched Python files.
- Tests are `test_<subject>.py` under `tests/`. Prefer real temporary local
  stores and small fakes at external boundaries. Guard live services with their
  explicit markers; do not turn normal tests into remote-service tests.

## Lifecycle rules for agents

- Read ADR 0001 before touching lifecycle/topology/concurrency/recovery code.
  Stop and escalate rather than adding another lock, queue, bootstrap state,
  projection gate, or cross-resource commit protocol to chase a stronger
  guarantee.
- Authority transactions establish canonical membership. External payload
  effects are reconciled with attributable intent/debt; they are not a single
  distributed transaction.
- Initialize before shared workers. Schema migration and layout cutover require
  stopped workers and explicit maintenance evidence; ordinary opens validate
  rather than implicitly upgrade.
- Treat ETag/version information as opaque transport evidence. Canonical
  payload verification remains the signed manifest digest and size.
- Preserve historical planning/audit material when updating current maps. Do
  not rewrite dated evidence merely to simplify a current product description.

## Validation expectations

- Use `uv run --isolated --all-extras --group dev --frozen pytest ...` for
  project tests and `uv lock --check` after manifest work.
- Fresh-wheel qualification must inspect the built artifact, installed metadata,
  normal old-import absence, and local `BlobStore`/`UnifiedCache` round trips;
  source-tree success alone is insufficient.
- Preserve user-owned ignored files and unrelated planning configuration.

<!-- GSD:skills-start source:skills/ -->

## Project Skills

- **Spike findings for cacheness** (implementation patterns, constraints, and
  gotchas) → `Skill("spike-findings-cacheness")`

<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using edit, write, or other file-changing tools, start work through a
GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, documentation updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repository edits outside a GSD workflow unless the user
explicitly asks to bypass it.

<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate a profile.
> This section is managed by `generate-claude-profile`; do not edit manually.

<!-- GSD:profile-end -->
