# Phase 6: UnifiedCache Policy Composition - Context

**Gathered:** 2026-09-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Publish one coherent cache-policy API whose storage work is performed by
`BlobStore`. Phase 6 finishes TTL, bounded size eviction, predicate and
decorator invalidation, cached-`None` handling, statistics, and cache-facing
error semantics across the supported topology contracts delivered by Phases 3
through 5.

This phase may remove redundant pre-production cache aliases and constructors.
It does not create a policy plugin framework, merge `SqlCache`, require a store
to serve cache and durable-retention roles simultaneously, change payload
formats, migrate stored schemas, qualify real services, or add another
lifecycle/cleanup coordinator. ADR 0001 remains binding.

</domain>

<decisions>
## Implementation Decisions

### Public Cache Surface

- **D-01:** The canonical cache surface is imported from `cacheness` and is
  centered on `UnifiedCache`, `CacheConfig`, and one decorator named `cached`.
  Remove redundant pre-production class aliases, global-cache factories,
  alternate decorator names, and overlapping constructor paths rather than
  routing them through compatibility wrappers. — **Reversibility:** costly —
  This intentionally changes application call sites and defines the first
  coherent public cache contract for later releases.
- **D-02:** Cache construction is explicit. A cache selects or receives one
  supported `BlobStore` composition and exposes deliberate initialization and
  close boundaries. No hidden process-global cache is part of the canonical
  API.
- **D-03:** Configuration has one nested `CacheConfig` vocabulary. Flat legacy
  constructor/config aliases may be removed; storage topology, handler,
  security, and cache-policy settings remain visibly separated. Unsupported
  topology or optional-feature requests fail before policy operations begin.
- **D-04:** Preserve clear optional-export warnings and typed missing-dependency
  errors for requested features. Do not preserve misleading availability flags
  or aliases merely because they existed before the pre-production cutover.

### Presence and Outcome Semantics

- **D-05:** Introduce one presence-bearing cache lookup result and a shared
  outcome vocabulary. At minimum it distinguishes `hit`, `absent`, `expired`,
  `corrupt`, `conflict`, and `backend_error`; exact names are implementation
  discretion. A hit carries the stored value even when that value is `None`.
- **D-06:** `UnifiedCache`, the decorator, and statistics consume that same
  lookup boundary. They must not infer absence from `value is None` or perform a
  second storage read to recover presence.
- **D-07:** Absence and policy expiry are normal cache misses. Canonical
  corruption may become a separately classified, non-destructive cache miss.
  Conflicts and backend errors retain typed causes and are not silently relabeled
  as absence. Error-suppression behavior, if offered by the decorator, must be
  explicit and still record the actual outcome.
- **D-08:** Statistics expose one immutable documented aggregate/result model,
  not backend-specific dictionaries or independently authoritative counters.
  It records hits plus the five non-hit outcome classes separately and derives
  totals/rates from those values. Statistics loss or close races never alter
  canonical entry state.

### TTL, Invalidation, and Size Eviction

- **D-09:** TTL is cache policy expressed over authoritative entry facts stored
  through `BlobStore`. Expiry is checked from the single entry snapshot used by
  the lookup. An expired lookup attempts exact-generation lifecycle deletion
  and reports the policy outcome without treating cleanup conflict as
  corruption.
- **D-10:** Single-key invalidation, predicate invalidation, size eviction,
  decorator clearing, and global clear all select entries through bounded
  supported catalog operations and remove them through the same exact-generation
  BlobStore lifecycle primitive. No path deletes payload files, metadata rows,
  or S3 objects directly.
- **D-11:** Invalidation and clear return one structured removal report naming
  attempted, removed, conflicted/retryable, and failed work. Decorator clear
  returns this report so callers see what was actually removed rather than an
  unconditional success or a separately counted estimate.
- **D-12:** V1 size enforcement is deterministic and bounded. Candidate
  selection may use authoritative cache-policy fields and stable canonical
  ordering; a derived projection or in-memory statistics layer cannot authorize
  deletion. If exact LRU would require a second authority or a write on every
  read, prefer an explicitly documented deterministic oldest-entry policy for
  V1. — **Reversibility:** costly — Eviction ordering is user-visible policy and
  later changes can alter retention behavior.
- **D-13:** Predicate invalidation uses the Phase 4 portable query contract where
  possible and bounded client-side evaluation only where explicitly supported.
  Malformed predicates fail closed before deletion. Pagination resumes from
  opaque authority-owned cursors and never materializes an unbounded catalog.

### Decorator Ownership and Lifecycle

- **D-14:** `cached` binds to an explicit `UnifiedCache` instance. It does not
  silently acquire a mutable module-global cache or create a second lifecycle
  owner. The application owns cache initialization and close.
- **D-15:** Decorated calls use deterministic function/key policy already owned
  by the cache layer, consume the presence-bearing result once, and invoke the
  function only for declared miss outcomes or explicit error-suppression policy.
  A cached `None` is returned without recomputation.
- **D-16:** Decorator helpers expose one clear operation backed by the cache's
  bounded predicate/prefix invalidation and return its structured removal
  report. Function identity remains part of the key namespace; clearing one
  decorated function must not clear unrelated entries.
- **D-17:** Closing after a canonical commit preserves the existing typed
  committed-partial contract for explicitly requested external metadata or
  derived statistics. Policy-layer close cannot revoke the BlobStore commit or
  create a repair prerequisite.

### the agent's Discretion

- Exact result, outcome enum, statistics, and removal-report class names and
  whether convenience value access is a method or property.
- Exact authoritative metadata field names for TTL, entry size, and stable
  eviction order, provided they use the existing versioned catalog vocabulary.
- Exact page sizes and per-operation work limits within configured lifecycle
  bounds.
- Internal module split between cache policy, decorator support, and public
  exports, provided `core.py` becomes thinner and storage sequencing remains in
  `BlobStore`/the shared lifecycle engine.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Governing Product and Guarantee Contracts

- `docs/adr/0001-topology-specific-storage-guarantees.md` — Mandatory lifecycle
  authority, topology-specific progress, ACID boundary, and race-loop stop
  conditions.
- `.planning/PROJECT.md` — BlobStore-first product, cache-policy layer,
  pre-production compatibility reset, and `SqlCache` exclusion.
- `.planning/ROADMAP.md` §§ Phase 6 — Phase goal, carried-forward scope, and six
  success criteria.
- `.planning/REQUIREMENTS.md` §§ Cache Composition — CACH-01 through CACH-06 and
  the Phase 3/5 traceability boundary.
- `CONTEXT.md` — Canonical domain vocabulary for store, cache, catalog,
  authority, projection, and policy.

### Delivered Storage and Composition Contracts

- `.planning/phases/05-payload-backends-and-supported-topology-qualification/05-CONTEXT.md`
  — Exact supported profiles, bounded progress outcomes, and one-engine rules.
- `.planning/phases/05-payload-backends-and-supported-topology-qualification/05-VERIFICATION.md`
  — Verified Phase 5 baseline and explicit Phase 8 BACK-05 deferral.
- `.planning/phases/04-metadata-composition-and-topology-contracts/04-CONTEXT.md`
  — Portable catalog query, stable cursors, authoritative attributes, and
  derived projection rules.
- `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-CONTEXT.md` —
  Immutable generation lifecycle, exact deletion, cleanup debt, and committed
  partial outcomes.
- `docs/CATALOG_AND_TOPOLOGY.md` — Published topology/catalog capabilities and
  unsupported-combination behavior.
- `docs/STORAGE_INITIALIZATION.md` — Explicit initialization and maintenance
  boundaries.
- `docs/phase3-direct-implementation-2026-09-06.md` — Direct qualification
  ledger for the thin local cache-over-BlobStore slice that Phase 6 extends.

### Current Cache Implementation and Maps

- `src/cacheness/core.py` — Current `UnifiedCache` policy facade and remaining
  list-driven cleanup/eviction/statistics behavior.
- `src/cacheness/decorators.py` — Current global-cache, cached-`None`, and
  decorator-clear seams to replace.
- `src/cacheness/config.py` — Current nested plus flat compatibility-heavy
  configuration surface.
- `src/cacheness/__init__.py` — Current public exports and optional-feature
  behavior.
- `.planning/codebase/ARCHITECTURE.md` — Component boundaries and historical
  duplication warnings; verify stale claims against current source.
- `.planning/codebase/STACK.md` — Runtime, optional dependencies, and packaging
  constraints.
- `.planning/codebase/INTEGRATIONS.md` — External service/configuration seams;
  historical wiring claims may predate Phases 4–5.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `src/cacheness/storage/blob_store.py`: canonical entry snapshots, exact
  expectations, bounded listing/querying, deletion, clear, and lifecycle reports.
- `src/cacheness/storage/composition.py`: exact supported topology profiles,
  preflight capability validation, and participant ownership.
- `src/cacheness/storage/lifecycle.py`: sole storage sequencer; cache policy can
  consume its typed outcomes without copying its state machine.
- `src/cacheness/core.py`: already constructs `_cache_blob_store` and delegates
  basic put/get/delete/clear, providing the thin slice to finish rather than
  rebuild.
- `src/cacheness/decorators.py`: deterministic function argument/key namespace
  logic can be retained while its presence and ownership behavior changes.

### Established Patterns

- Cache and direct stores may use separate namespaces while sharing the same
  BlobStore implementation.
- Entry snapshots bind descriptor, generation expectation, and verified handler
  read; policy code should consume one snapshot per decision.
- Canonical corruption is fail-closed and non-destructive at the cache boundary.
- Optional projections and statistics are derived state and cannot gate reads,
  cleanup, or repair.
- Remote catalog/inventory work is page-bounded and may return typed retryable
  outcomes under contention.

### Integration Points

- Replace `core.py`'s repeated list/get/delete loops with bounded BlobStore
  catalog and lifecycle primitives while keeping TTL/keying/statistics above it.
- Replace decorator `cached_result is not None` with the presence-bearing lookup
  boundary and remove implicit `get_cache()` ownership.
- Collapse top-level exports and compatibility-heavy config construction around
  the selected public Phase 6 surface.
- Extend local and deterministic remote-candidate contract tests without
  pretending to close Phase 8's live-service gate.

</code_context>

<specifics>
## Specific Ideas

- A cache instance is always powered by a BlobStore engine, but it need not
  share a live namespace with a direct non-cache store.
- The API cleanup should exploit the pre-production cutover now; retaining
  wrappers would recreate overlapping lifecycle/configuration surfaces.
- Outcome classification is a diagnostic and policy vocabulary, not permission
  to turn storage errors into generic misses.

</specifics>

<deferred>
## Deferred Ideas

- General pluggable cache-policy interfaces — v2 EXTN-01.
- Distributed invalidation/coherence across cache processes or hosts — v2
  EXTN-05.
- Stored-schema/format migration and rebuild tooling — Phase 7.
- Real PostgreSQL/Amazon S3 support qualification, packaging/CI matrices,
  coverage gates, and performance budgets — Phase 8.
- Redesigning or merging `SqlCache` — outside this milestone.

</deferred>

---

*Phase: 6-unifiedcache-policy-composition*
*Context gathered: 2026-09-08*
