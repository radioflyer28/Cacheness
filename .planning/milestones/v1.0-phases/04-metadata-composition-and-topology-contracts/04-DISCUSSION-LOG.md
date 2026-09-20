# Phase 4: Metadata Composition and Topology Contracts - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-07
**Phase:** 4-metadata-composition-and-topology-contracts
**Areas discussed:** Catalog schema, query and indexing semantics, backend selection and capability contracts, derived projections and external ORM integration

---

## Catalog Schema

### Schema requirement

| Option | Description | Selected |
|--------|-------------|----------|
| Optional declared schema atop mapping | Keep arbitrary mappings and optionally declare validated/queryable fields. | ✓ |
| Declared schema required | Require every catalog entry to conform to a declared schema. | |
| Mapping only | Retain mappings and callbacks without a declarative model. | |

**User's choice:** Optional declared schema atop the existing mapping.

### Undeclared fields

| Option | Description | Selected |
|--------|-------------|----------|
| Preserve but treat as opaque | Round-trip undeclared values without portable validation/query promises. | ✓ |
| Reject undeclared fields | Enforce a closed schema. | |
| Drop undeclared fields | Persist only declared values. | |

**User's choice:** Preserve undeclared fields as opaque data.

### Declaration API

| Option | Description | Selected |
|--------|-------------|----------|
| Small Cacheness-native schema | Define type, required/default, validation, and query/index intent in a native contract. | ✓ |
| Python dataclasses | Use dataclass fields plus additional conventions. | |
| External model protocol | Make Pydantic or similar models the primary declaration API. | |
| Plain validator callbacks | Validate without an introspectable schema. | |

**User's choice:** Small Cacheness-native schema with a future adapter seam.

### Evolution

| Option | Description | Selected |
|--------|-------------|----------|
| Backward-readable, explicit migration | Validate new mutations while retaining readable old entries; migrate incompatible changes offline. | ✓ |
| Validate everything on reopen | Reject old entries until they match the current schema. | |
| Automatically rewrite on read | Upgrade catalog metadata lazily during reads. | |

**User's choice:** Backward-readable evolution with explicit incompatible-change migration.

**Notes:** The user selected the recommended compatibility-preserving option for all four questions.

---

## Query and Indexing Semantics

### Portable operators

| Option | Description | Selected |
|--------|-------------|----------|
| Constrained typed filters | Equality, comparison/range, membership, and existence over declared fields, combined with AND. | ✓ |
| Full boolean expressions | Include nested AND/OR/NOT immediately. | |
| Equality only | Provide exact matching only. | |
| Backend-native queries | Expose SQL/JSON-specific expressions instead of a portable contract. | |

**User's choice:** Constrained typed AND filters.

### Ordering and pagination

| Option | Description | Selected |
|--------|-------------|----------|
| Stable canonical cursor | Deterministic key/generation ordering with bounded pages. | ✓ |
| Portable sorting by any field | Require cross-backend sorting and collation rules. | |
| Offset and limit | Use positional pagination. | |
| No ordering guarantee | Return backend iteration order. | |

**User's choice:** Stable canonical cursor.

### Index selection

| Option | Description | Selected |
|--------|-------------|----------|
| Explicit schema intent | Declare queryable/indexed fields; keep indexes rebuildable. | ✓ |
| Index every declared field | Create indexes automatically for all fields. | |
| Backend chooses automatically | Leave index policy undocumented and adapter-specific. | |
| No secondary indexes | Scan the catalog for every query. | |

**User's choice:** Explicit schema index intent.

### Query consistency

| Option | Description | Selected |
|--------|-------------|----------|
| Canonical-complete results | Use same-transaction indexes or scan canonical catalog state. | ✓ |
| Eventual results with freshness marker | Permit lagged external indexes to answer. | |
| Best effort with fallback | Fall back only when index failure is detected. | |
| Backend-specific consistency | Let each adapter define completeness independently. | |

**User's choice:** Canonically complete portable results.

**Notes:** External independently updated indexes may not silently omit portable-query results.

---

## Backend Selection and Capability Contracts

### Composition API

| Option | Description | Selected |
|--------|-------------|----------|
| One typed store configuration | Compose payload, catalog authority, projection, and requirements in one validated root. | ✓ |
| Independent constructor arguments | Assemble roles independently and validate afterward. | |
| Factory names only | Require string registration for all backends. | |
| Injected instances only | Remove name-based primary construction. | |

**User's choice:** One typed store configuration with compatibility adapters.

### Conflicting selectors

| Option | Description | Selected |
|--------|-------------|----------|
| Reject ambiguity | Preserve exact injected instances and fail conflicting name/instance selection. | ✓ |
| Injected instance wins | Silently ignore the name. | |
| Named configuration wins | Silently replace the injected object. | |
| Merge options into instance | Reconfigure caller-owned state. | |

**User's choice:** Reject ambiguous selectors.

### Capability negotiation

| Option | Description | Selected |
|--------|-------------|----------|
| Inspect always, require optionally | Expose actual capabilities and reject unmet caller minima or impossible pairings. | ✓ |
| Require an explicit topology tier | Make every caller name a guarantee profile. | |
| Infer strongest tier automatically | Advertise the strongest inferred promise. | |
| Backend-name allowlists | Validate combinations by implementation name. | |

**User's choice:** Inspect capabilities always and make minimum requirements optional.

### Resource ownership

| Option | Description | Selected |
|--------|-------------|----------|
| Explicit ownership | Own factory-created resources; borrow injected resources unless ownership transfers explicitly. | ✓ |
| Store always owns them | Close injected resources with the store. | |
| Caller always owns them | Require callers to close factory-created resources. | |
| Reference-count shared instances | Add automatic shared-resource coordination. | |

**User's choice:** Explicit ownership.

**Notes:** Selection and ownership must be observable and must not rely on precedence or hidden sharing.

---

## Derived Projections and External ORM Integration

### ORM/index role

| Option | Description | Selected |
|--------|-------------|----------|
| Derived-only integration | Keep authoritative custom fields native; rebuild external ORM rows/indexes from commits. | ✓ |
| Two extension modes | Also allow same-session transactional plugins. | |
| ORM-defined authority | Let external models replace the native catalog. | |
| No ORM contract | Omit an external integration contract. | |

**User's choice:** Derived-only integration in Phase 4.

### Update delivery

| Option | Description | Selected |
|--------|-------------|----------|
| Checkpointed pull | Idempotently consume bounded pages of committed catalog state. | ✓ |
| Synchronous callbacks only | Depend on post-commit callbacks for delivery. | |
| Required background worker | Require an operational convergence process. | |
| Exactly-once delivery | Add a durable event-coordination protocol. | |

**User's choice:** Checkpointed pull from canonical catalog state.

### Projection failure

| Option | Description | Selected |
|--------|-------------|----------|
| Committed receipt with projection status | Preserve canonical success and expose warning or typed committed-partial status. | ✓ |
| Fail the whole write | Report total failure after canonical commit. | |
| Warnings only | Hide projection status from programmatic callers. | |
| Wait for projection convergence | Couple write success to derived infrastructure. | |

**User's choice:** Committed receipt with explicit projection status.

### Rebuild behavior

| Option | Description | Selected |
|--------|-------------|----------|
| Capability-qualified rebuild | Build isolated state and report online catch-up versus offline maintenance support. | ✓ |
| Always rebuild online | Require convergence while writes continue. | |
| Always stop writers | Require offline rebuilding for every adapter. | |
| Repair during normal operations | Hide projection repair inside reads/writes. | |

**User's choice:** Explicit, capability-qualified rebuild.

**Notes:** Projection lag or failure never authorizes canonical deletion, rollback, or a second lifecycle coordinator.

---

## the agent's Discretion

- Exact API names and module placement.
- Finite portable value types, validation error classes, page-size defaults, and opaque cursor encoding within the recorded constraints.
- Internal schema fingerprint and projection checkpoint representation.

## Deferred Ideas

- Transactional external ORM plugins that join an authority database session.
- Full PostgreSQL/S3 topology qualification in Phase 5.
- Complete UnifiedCache policy composition in Phase 6.
- Stored-format/schema migration execution in Phase 7.

---

## Post-Discussion Planning Override

During planning, the user clarified that Cacheness is not yet used in production and
approved dropping backward compatibility to reduce Phase 4 complexity. This supersedes
the discussion's runtime-compatibility assumptions, including compatibility adapters for
old constructors/selectors and automatic readability of pre-Phase-4 development layouts.

The following remains required:

- one clean typed composition and catalog API;
- explicit schema/format version identification;
- typed migration/rebuild-required failure for unsupported layouts, without mutation;
- Phase 7 migration and rebuild tooling for future released versions.

The following may be removed instead of adapted:

- overlapping pre-production constructors, aliases, factories, and backend selectors;
- runtime shims for development-only catalog layouts;
- historical tests whose sole purpose is preserving a removed pre-production API.
