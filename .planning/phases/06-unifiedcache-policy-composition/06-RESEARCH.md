# Phase 6: UnifiedCache Policy Composition - Research

**Researched:** 2026-09-08
**Domain:** Python cache-policy composition over the existing BlobStore lifecycle authority
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Public Cache Surface

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

#### Presence and Outcome Semantics

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

#### TTL, Invalidation, and Size Eviction

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

#### Decorator Ownership and Lifecycle

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

### Deferred Ideas (OUT OF SCOPE)

- General pluggable cache-policy interfaces — v2 EXTN-01.
- Distributed invalidation/coherence across cache processes or hosts — v2
  EXTN-05.
- Stored-schema/format migration and rebuild tooling — Phase 7.
- Real PostgreSQL/Amazon S3 support qualification, packaging/CI matrices,
  coverage gates, and performance budgets — Phase 8.
- Redesigning or merging `SqlCache` — outside this milestone.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CACH-01 | `UnifiedCache` delegates every payload-plus-authoritative-catalog lifecycle operation to the supported BlobStore entry interface; derived adapters never authorize repair or deletion. | Use `BlobStore.open_entry()`, catalog pages, exact `EntryExpectation` deletion, and the existing `AuthorityLifecycleEngine`; remove every direct/list-driven lifecycle path. [VERIFIED: .planning/REQUIREMENTS.md:33-33] |
| CACH-02 | `UnifiedCache` exclusively owns cache keying, TTL, eviction, invalidation, statistics, and decorator-facing policy. | Isolate policy models and algorithms above BlobStore; statistics are a one-way observer, never an authority. [VERIFIED: .planning/REQUIREMENTS.md:34-34] |
| CACH-03 | Every invalidation path removes the complete stored entry through the storage lifecycle. | Converge TTL, size, predicate, decorator, key, and global deletion on exact-generation BlobStore deletion and one removal report. [VERIFIED: .planning/REQUIREMENTS.md:35-35] |
| CACH-04 | A cached `None` remains distinguishable from a miss. | Make the presence-bearing result the only lookup contract used by both direct callers and decorators. [VERIFIED: .planning/REQUIREMENTS.md:36-36] |
| CACH-05 | Statistics distinguish absent, expired, corrupt, conflict, and backend-error outcomes through one aggregate/result model. | Record the common result outcome once and derive totals/rates from immutable snapshots. [VERIFIED: .planning/REQUIREMENTS.md:37-37] |
| CACH-06 | Publish one coherent import, constructor, configuration, decorator, and result surface over BlobStore. | Perform the public export, config, constructor, decorator, examples, and contract-test cutover atomically; preserve explicit initialization and typed partial failures. [VERIFIED: .planning/REQUIREMENTS.md:38-38] |
</phase_requirements>

## Summary

Phase 6 should be planned as a policy-layer cutover, not as another storage rewrite. The current `UnifiedCache` already delegates its basic `put`, `get`, exact delete, and clear paths to a private `BlobStore`, while BlobStore already exposes the required single-snapshot read, exact-generation expectation, authenticated catalog query, and lifecycle deletion primitives. [VERIFIED: src/cacheness/core.py:281-363] [VERIFIED: src/cacheness/storage/blob_store.py:350-392] [VERIFIED: src/cacheness/storage/blob_store.py:535-631] The remaining work is to make every cache-policy path use those primitives, publish a presence-bearing outcome contract, and remove the global/alias-heavy surface in one deliberate pre-production break.

The central implementation rule is: policy decides *which canonical entry* should be read or removed, while BlobStore and the existing `AuthorityLifecycleEngine` remain the only code allowed to sequence payload plus authoritative-catalog state. BlobStore read snapshots already bind verified content to the entry generation, and delete already accepts an exact `EntryExpectation`; policy must retain and use that expectation rather than re-resolving by key. [VERIFIED: src/cacheness/storage/read_contract.py:87-114] [VERIFIED: src/cacheness/storage/blob_store.py:535-540] Statistics, external metadata, and convenience projections remain derived observers and cannot authorize cleanup.

The size-enforcement contract is resolved as bounded/resumable maintenance. Current catalog pages are stably ordered by key/generation and expose catalog values, but `CatalogEntry` does not expose the signed manifest's intrinsic `created_at` and `byte_size` even though those facts exist in the authenticated manifest. [VERIFIED: src/cacheness/storage/catalog.py:657-704] [VERIFIED: src/cacheness/storage/manifest.py:173-250] Add the narrowest BlobStore catalog-view enhancement needed to return authenticated descriptor facts plus the exact expectation. A finite quiescent inventory must converge across explicit bounded resumes; conflicts or authority revision churn return typed incomplete/retryable results. This contract makes no universal contender-success, perpetual-churn completion, exact LRU, or global-oldest promise. The enhancement stays inside the existing authority and does not create a second index, coordinator, queue, or lock layer.

**Primary recommendation:** Cut over to one explicit `UnifiedCache` whose presence-bearing results and structured removal reports drive all policy and decorator behavior, while every canonical read/write/delete/query remains a BlobStore operation over the shared lifecycle engine.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Cache key and function namespace policy | Python library policy layer (`UnifiedCache`) | Decorator adapter | Keying is cache semantics; the decorator supplies function identity but must not create storage ownership. [VERIFIED: src/cacheness/decorators.py:38-76] |
| Presence, verified payload, and generation snapshot | BlobStore | `AuthorityLifecycleEngine` | BlobStore already returns a presence-bearing `BlobEntry` from one lifecycle-engine snapshot. [VERIFIED: src/cacheness/storage/read_contract.py:87-114] [VERIFIED: src/cacheness/storage/lifecycle.py:608-678] |
| TTL classification | `UnifiedCache` policy | BlobStore exact delete | Policy compares authoritative snapshot time facts; storage performs generation-conditional removal. [VERIFIED: .planning/phases/06-unifiedcache-policy-composition/06-CONTEXT.md:69-73] |
| Predicate candidate selection | BlobStore catalog API | `UnifiedCache` policy | BlobStore owns validation, authority cursor, and authenticated catalog observation; policy defines cache predicates. [VERIFIED: src/cacheness/storage/blob_store.py:582-631] |
| Size candidate selection | `UnifiedCache` policy | BlobStore authenticated catalog view | Policy owns ordering/budget; BlobStore must expose only authenticated facts and exact expectations, without a parallel projection authority. [VERIFIED: .planning/phases/06-unifiedcache-policy-composition/06-CONTEXT.md:83-89] |
| Entry removal and cleanup debt | `AuthorityLifecycleEngine` through BlobStore | `UnifiedCache` report translation | The lifecycle engine owns state transitions, rollback/reconciliation, and typed progress; cache policy only records the result. [VERIFIED: docs/adr/0001-topology-specific-storage-guarantees.md:14-28] |
| Outcome statistics | `UnifiedCache` derived observer | Decorator | Statistics consume the same lookup/removal outcome and cannot gate or mutate canonical state. [VERIFIED: .planning/phases/06-unifiedcache-policy-composition/06-CONTEXT.md:61-65] |
| SQL pull-through caching | `SqlCache` | — | It remains a separate subsystem and is already covered by CACH-07. [VERIFIED: .planning/REQUIREMENTS.md:39-39] |

## Project Constraints (from AGENTS.md)

- Maintain Python `>=3.11`; the repository development interpreter is pinned separately and supported versions must be verified rather than inferred from the current interpreter. [VERIFIED: AGENTS.md:23-23]
- `BlobStore` owns storage lifecycle, `UnifiedCache` depends on it and owns cache policy, and `SqlCache` remains separate. [VERIFIED: AGENTS.md:15-18]
- Cover filesystem, memory, S3, JSON, SQLite, and PostgreSQL through declared supported topology contracts; do not overstate live-service qualification in this phase. [VERIFIED: AGENTS.md:18-18]
- Application payloads are trusted, but parsing, path containment, and integrity boundaries fail closed. [VERIFIED: AGENTS.md:19-19]
- Payload and authoritative metadata require atomic commit, rollback, or deterministic reconciliation; same-key concurrency must never create disagreement. [VERIFIED: AGENTS.md:20-21]
- Before changing cache/storage lifecycle, recovery, concurrency, topology, timeouts, or composition, ADR 0001 is mandatory and its stop conditions bind the plan. [VERIFIED: AGENTS.md:29-36]
- Use `uv`, pytest, Google-style public docstrings, package-relative internal imports, narrow exception translation with preserved causes, and Ruff's configured 88-column target for new code. [VERIFIED: AGENTS.md:49-63] [VERIFIED: AGENTS.md:133-158]
- Do not grow the already-large orchestration modules; extract focused policy/result helpers while leaving storage sequencing in the established storage modules. [VERIFIED: AGENTS.md:175-181]

## Standard Stack

### Core

| Library / component | Version | Purpose | Why Standard |
|---------------------|---------|---------|--------------|
| Python stdlib `dataclasses` | Python 3.11+ | Frozen lookup, statistics, and removal result records | `@dataclass(frozen=True)` provides generated value semantics and guarded field assignment without a new runtime dependency. [CITED: https://docs.python.org/3.11/library/dataclasses.html] |
| Python stdlib `enum` | Python 3.11+ | One finite lookup outcome vocabulary | `Enum` gives stable named members; string-valued members are suitable for documented results and tests. [CITED: https://docs.python.org/3.11/library/enum.html] |
| Existing `BlobStore` + `StoreTopology` | In-repo | All canonical payload/catalog lifecycle operations and supported composition preflight | They already resolve topology participants, initialize the authority, query catalog pages, and execute lifecycle operations. [VERIFIED: src/cacheness/storage/blob_store.py:145-183] [VERIFIED: src/cacheness/storage/composition.py:726-762] |
| Existing catalog contract | `STORE_FORMAT_VERSION = 2` | Typed portable predicates, stable cursor pages, and bounded candidate discovery | Supported field kinds are quoted verbatim as `{"string", "integer", "boolean"}` and operators as `{"eq", "lt", "lte", "gt", "gte", "in", "exists"}`. [VERIFIED: src/cacheness/storage/catalog.py:30-58] |
| pytest | `version = "8.4.1"`; `upload-time = "2025-06-18T05:48:03.955Z"` for the locked wheel | Contract, adversarial, and topology-candidate tests | Existing phase-relevant tests run under pytest; the targeted baseline was 60 passing tests on 2026-09-08. [VERIFIED: uv.lock:1607-1621] [VERIFIED: command `uv run --frozen pytest tests/test_unified_cache_lifecycle_authority.py tests/test_unified_cache_adversarial_lifecycle.py tests/test_decorators.py tests/test_blob_store_translation_seam.py -o addopts='' --disable-warnings`] |

### Supporting

| Library / component | Version | Purpose | When to Use |
|---------------------|---------|---------|-------------|
| `functools.wraps` | Python 3.11+ | Preserve decorated function metadata | Use on the sole public `cached` wrapper. [CITED: https://docs.python.org/3.11/library/functools.html] |
| `CacheReadFailureCategory` classifier | In-repo | Translate storage failures without erasing typed causes | Reuse for cache-result classification; current exact values are `"integrity"`, `"manifest_unsupported_version"`, `"payload_unsupported_version"`, `"unsupported_version"`, `"lifecycle_conflict"`, `"backend_failure"`, `"migration_required"`, and `"unclassified"`. [VERIFIED: src/cacheness/storage/read_contract.py:119-157] |
| `CatalogQuery` validation | In-repo | Fail closed before predicate deletion | Validate the complete query and configured bounds before the first authoritative query or delete. [VERIFIED: src/cacheness/storage/catalog.py:320-442] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Frozen dataclass results | Tuple/dictionary/sentinel | Recreates positional ambiguity, mutable legacy counter shapes, or a second presence convention; do not use. |
| Existing portable catalog | Direct backend SQL, filesystem walk, or S3 listing | Breaks topology neutrality, authority-owned cursors, and authenticated selection; do not use. |
| Deterministic bounded policy | Exact LRU with a write on every hit | Adds authority writes to the read path and increases contention; D-12 explicitly prefers deterministic oldest behavior when exact LRU needs another authority. [VERIFIED: .planning/phases/06-unifiedcache-policy-composition/06-CONTEXT.md:83-89] |
| Explicit cache injection | Module-global cache factory | Creates hidden lifecycle ownership and cross-test/process state; remove it. [VERIFIED: .planning/phases/06-unifiedcache-policy-composition/06-CONTEXT.md:35-38] |

**Installation:** No new external package is needed. Use the locked environment with `uv sync --frozen` when environment repair is necessary. [VERIFIED: pyproject.toml:70-74]

## Package Legitimacy Audit

No external package is recommended or installed by this phase, so the package-legitimacy gate is not applicable.

## Architecture Patterns

### System Architecture Diagram

```text
Application / explicit @cached(cache=...)
                  |
                  v
       UnifiedCache policy boundary
       - key + function namespace
       - TTL / eviction / invalidation
       - lookup outcome + removal report
       - derived statistics observer --------> immutable statistics snapshot
                  |
      put/open/query/delete exact expectation
                  v
              BlobStore
                  |
                  v
       AuthorityLifecycleEngine (one sequencer)
                  |
          +-------+------------------+
          |                          |
          v                          v
  one lifecycle authority      immutable payload backend
  (memory / SQLite / PG)       (memory / filesystem / S3)
          |
          +---- post-commit derived projection (never authorizes deletion)

Lookup branches:
  verified present -> hit (value may be None)
  authoritative absent -> absent
  present + TTL expired -> expired + exact-generation delete attempt
  integrity evidence -> corrupt, non-destructive
  typed contention -> conflict, preserve cause
  operational failure -> backend_error, preserve cause
```

The lifecycle profiles already declare exact authority/payload pairs. The quoted local profiles are `"memory", "memory"` with progress outcomes `{"success", "conflict"}` and `"sqlite", "filesystem"` with `{"success", "conflict", "retryable_timeout"}`; the remote candidate is `"postgresql", "s3"` with `{"success", "conflict", "retryable_serialization", "retryable_deadlock", "retryable_lock_timeout", "retryable_statement_timeout", "retryable_connection_timeout"}`. [VERIFIED: src/cacheness/storage/composition.py:127-190]

### Recommended Project Structure

```text
src/cacheness/
├── core.py                       # thin UnifiedCache orchestration/public methods
├── cache_policy.py               # outcome, stats, removal models + bounded policy logic [ASSUMED]
├── config.py                     # one nested CacheConfig vocabulary
├── decorators.py                 # explicit-cache cached wrapper only
├── __init__.py                   # canonical exports and honest optional exports
└── storage/
    ├── blob_store.py             # canonical storage facade; narrow catalog-view extension
    ├── catalog.py                # portable predicates/cursors/bounds
    └── lifecycle.py              # unchanged sole lifecycle sequencer
```

`cache_policy.py` is the recommended new module name, not an existing contract. [ASSUMED]

### Pattern 1: Presence-bearing lookup as the single policy seam

**What:** Return one frozen result containing the common outcome, optional value, and preserved typed cause. The locked vocabulary is quoted verbatim: `hit`, `absent`, `expired`, `corrupt`, `conflict`, and `backend_error`. [VERIFIED: .planning/phases/06-unifiedcache-policy-composition/06-CONTEXT.md:49-52]

**When to use:** Every direct lookup, decorated call, statistics update, and suppression decision.

**Example:**

```python
# Source pattern: https://docs.python.org/3.11/library/dataclasses.html
# Source vocabulary: 06-CONTEXT.md:49-52
from dataclasses import dataclass
from enum import Enum
from typing import Any


class CacheOutcome(str, Enum):  # recommended name [ASSUMED]
    HIT = "hit"
    ABSENT = "absent"
    EXPIRED = "expired"
    CORRUPT = "corrupt"
    CONFLICT = "conflict"
    BACKEND_ERROR = "backend_error"


@dataclass(frozen=True)
class CacheLookupResult:  # recommended name [ASSUMED]
    outcome: CacheOutcome
    value: Any = None
    cause: BaseException | None = None

    @property
    def is_hit(self) -> bool:
        return self.outcome is CacheOutcome.HIT
```

Do not add a second canonical raw-value `get` contract that again makes `None` ambiguous. If convenience access exists, it must operate on this result and raise or default explicitly for a non-hit.

### Pattern 2: One snapshot, one classification, exact cleanup

**What:** Call `BlobStore.open_entry()` once. Classify absence, expiry, corruption, conflict, or backend failure from that call/snapshot. For expiry, delete only with the expectation carried by the same entry and retain `expired` even if cleanup returns conflict. BlobStore's read path already takes one authority snapshot, performs a bounded internal retry, verifies the payload before the handler reader, and returns absence as `None`. [VERIFIED: src/cacheness/storage/lifecycle.py:608-678]

**When to use:** Cache reads and decorator hits/misses.

### Pattern 3: Preflight, page, decide, exact-delete, report

**What:** Validate a complete portable query and work limits before I/O; fetch one authority-owned page; decide using authenticated facts; delete each candidate with its exact expectation; accumulate a frozen removal report; resume only from the opaque cursor. Current catalog constants quote `DEFAULT_PAGE_SIZE = 100`, `MAX_PAGE_SIZE = 256`, and `MAX_WORK_CAP = 4096`. [VERIFIED: src/cacheness/storage/catalog.py:30-58]

**When to use:** Predicate invalidation, function clear, global clear, TTL maintenance, and size maintenance.

**Prescriptive rule:** Use the same internal bounded removal executor for all paths, but keep it a policy helper—not a new lifecycle coordinator. It invokes BlobStore once per canonical operation and never changes authority state itself.

### Pattern 4: Function identity as portable catalog policy metadata

**What:** Persist the deterministic function namespace as an authoritative cache catalog value at put time and query it for `wrapper.cache_clear()`. The existing decorator already computes a deterministic function identifier and normalized argument key. [VERIFIED: src/cacheness/decorators.py:38-76]

**When to use:** Decorated writes and one-function invalidation. The function clear predicate must not depend on reversing opaque hashed storage keys.

### Pattern 5: Derived, immutable statistics snapshots

**What:** Maintain best-effort process-local outcome observations and expose only a frozen snapshot with separate outcome counts plus computed total/rate properties. D-08 explicitly allows statistics loss, so do not add a lock layer or stronger storage coordination merely to make derived counters exact under close/concurrency races. [VERIFIED: .planning/phases/06-unifiedcache-policy-composition/06-CONTEXT.md:61-65]

**When to use:** Update exactly once after lookup classification and after explicit decorator suppression has recorded the underlying outcome. Never compute statistics by listing the canonical catalog.

### Anti-Patterns to Avoid

- **Second read for presence:** It races generation changes and violates D-06; the `BlobEntry` already carries presence and value. [VERIFIED: src/cacheness/storage/read_contract.py:87-114]
- **Key-only delete after selection:** Another writer may have promoted a new generation. Retain and pass the selected `EntryExpectation`. [VERIFIED: src/cacheness/storage/blob_store.py:535-540]
- **Unbounded `BlobStore.list()` policy loops:** That method is explicitly local-only; remote inventory uses paged catalog operations. [VERIFIED: src/cacheness/storage/blob_store.py:547-590]
- **Statistics as an eviction oracle:** A lossy derived counter cannot authorize deletion or prove canonical total bytes. [VERIFIED: .planning/phases/06-unifiedcache-policy-composition/06-CONTEXT.md:61-65] [VERIFIED: .planning/phases/06-unifiedcache-policy-composition/06-CONTEXT.md:83-86]
- **Constructor cleanup:** Construction must validate only; initialization and maintenance are deliberate lifecycle boundaries. Current `cleanup_on_init` defaults to `True`, so retaining it would hide destructive work in construction. [VERIFIED: src/cacheness/config.py:36-78]
- **Broad decorator suppression:** Current decorator catches broad exceptions around reads/writes. The new wrapper must suppress only explicitly configured outcome classes and still record the original result. [VERIFIED: src/cacheness/decorators.py:167-199]
- **Compatibility wrappers:** Keeping `get_cache`, `memoize`, `CacheContext`, alternate constructors, or class aliases reproduces the ownership/config ambiguity D-01 removes. [VERIFIED: src/cacheness/core.py:432-472] [VERIFIED: src/cacheness/decorators.py:259-362]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Payload/catalog atomicity | Cache-side rollback, queue, lock, repair hook, or sidecar | BlobStore + `AuthorityLifecycleEngine` | The engine already owns exact promotion, cleanup debt, and typed progress; another sequencer violates ADR 0001. [VERIFIED: docs/adr/0001-topology-specific-storage-guarantees.md:14-28] |
| Predicate language | Lambda/eval/raw-SQL predicates | `CatalogSchema`, `CatalogPredicate`, `CatalogQuery` | Existing validation is typed, bounded, portable, and fail-closed. [VERIFIED: src/cacheness/storage/catalog.py:246-442] |
| Pagination | Offset tokens or materialized full inventory | Authority-owned opaque cursors | Stable pagination semantics belong to the authority. [VERIFIED: src/cacheness/storage/catalog.py:657-704] |
| Integrity/signing | Cache-layer hashes or signature verification | BlobStore manifest/payload verification | A second verifier risks divergent canonical evidence. [VERIFIED: src/cacheness/storage/manifest.py:173-250] |
| Presence sentinel | `value is None`, magic objects, or exists-then-read | Presence-bearing BlobStore entry translated once | `None` is a valid stored value and exists-then-read races. [VERIFIED: src/cacheness/storage/read_contract.py:87-114] |
| Topology pairing | Cache-specific backend registry/constructor branches | `StoreTopology` and its supported profiles | It is the sole composition root and performs capability preflight. [VERIFIED: src/cacheness/storage/composition.py:726-762] |

**Key insight:** Cache policy is complex because selection, observation, and deletion occur under concurrency. The safe design is not a smarter cache-side coordinator; it is a thinner policy layer that carries authenticated snapshots and exact expectations through the existing lifecycle authority.

## Runtime State Inventory

| Category | Items Found | Action Required |
|----------|-------------|-----------------|
| Stored data | Filesystem inspection found both current and legacy stores under the configured root. The current source quotes `cache_dir: str = "./cache"`, `root = self.cache_dir / ".cacheness" / "blobstore"`, and `AUTHORITY_RELATIVE_PATH = Path(".cacheness") / "lifecycle-authority-v2.sqlite3"`. It also detects the exact legacy path expression `candidate / ".cacheness" / "lifecycle-authority-v1.sqlite3"` and classifies it as `"development-format-1"`. [VERIFIED: src/cacheness/config.py:36-45] [VERIFIED: src/cacheness/core.py:89-93] [VERIFIED: src/cacheness/storage/sqlite_lifecycle_authority.py:61-76] [VERIFIED: src/cacheness/storage/manifest.py:438-445] | Do not mutate or implicitly upgrade the observed stores. New cache catalog requirements must detect an unsupported existing layout and fail with the established migration/rebuild-required category; Phase 7 owns tooling. |
| Live service config | No checked-in Phase 6 requirement depends on live PostgreSQL/S3 configuration; real-service qualification is explicitly deferred. [VERIFIED: .planning/phases/06-unifiedcache-policy-composition/06-CONTEXT.md:241-246] | Use deterministic remote-candidate contracts only. Treat any caller-provided external metadata/statistics sink as derived and never as lifecycle authority. |
| OS-registered state | None found: the project is an in-process Python library with no service, daemon, launchd, systemd, or task-scheduler registration in the inspected repository. [VERIFIED: repository file inventory 2026-09-08] | No OS migration task. |
| Secrets/env vars | Existing local manifest-key artifacts were observed. Current creation quotes `self.cache_dir / "blob_manifest_hmac_key.bin"`; the current SQLite bootstrap names quote `"blob_manifest_hmac_key.bin"`, `"blob_manifest_hmac_key.bin.ready"`, and `"blob_manifest_hmac_key.bin.initializing.lock"`. [VERIFIED: src/cacheness/storage/blob_store.py:218-231] [VERIFIED: src/cacheness/storage/sqlite_lifecycle_authority.py:62-70] | Preserve them; do not log, rotate, rename, or recreate them during API cutover. No new environment-variable namespace is required. |
| Build artifacts / installed packages | Python bytecode and installed project-distribution metadata were observed in the worktree environment. [VERIFIED: filesystem inspection 2026-09-08] | Re-run tests through `uv`; any release/install verification should rebuild the wheel/editable environment after old imports are removed. Do not treat stale artifacts as source contracts. |

**Canonical post-refactor question:** after source aliases are removed, search examples, tests, docs, installed metadata, and user-facing exports for old constructor/import names; runtime cache files themselves must remain untouched until explicit Phase 7 migration/rebuild work.

## Common Pitfalls

### Pitfall 1: Cached `None` still recomputes

**What goes wrong:** The decorator reports a hit but calls the wrapped function again.

**Why it happens:** Current code tests `cached_result is not None`, collapsing a valid value with absence. [VERIFIED: src/cacheness/decorators.py:167-199]

**How to avoid:** Branch exclusively on the shared presence-bearing outcome; add an invocation-count test whose cached function returns `None`.

**Warning signs:** Tests only assert returned value and statistics, not that the function ran once. The current test checks `None` values without a call-count assertion. [VERIFIED: tests/test_decorators.py:438-461]

### Pitfall 2: Cleanup deletes a replacement generation

**What goes wrong:** TTL or predicate selection observes generation A, a writer promotes B, and cleanup removes B by key.

**Why it happens:** Selection and deletion discard the expectation.

**How to avoid:** Every candidate carries an exact expectation from its authenticated snapshot/page; conflict is reported under conflicted/retryable, never retried as an unconditional delete.

**Warning signs:** Internal helpers accept only `key: str`, or perform `get_entry_info()` immediately before key-only deletion.

### Pitfall 3: “Bounded” code still scans the whole catalog

**What goes wrong:** A page API is wrapped in a loop that accumulates all entries, or one public call drains every page without an explicit work cap.

**Why it happens:** Page size is mistaken for total work bound.

**How to avoid:** Specify page size *and* maximum entries/deletes per call; return continuation/incomplete state in the structured report. Resume only from the opaque cursor.

**Warning signs:** `list(...)`, `list.extend(...)`, or `while cursor` without a decrementing budget in policy code.

### Pitfall 4: Eviction promises more than the authority proves

**What goes wrong:** A process-local counter or partial page is called the canonical total/oldest set, causing arbitrary or insufficient eviction.

**Why it happens:** Current `CatalogEntry` omits manifest `created_at` and `byte_size`, and page order is key/generation rather than global age. [VERIFIED: src/cacheness/storage/catalog.py:657-704] [VERIFIED: src/cacheness/storage/manifest.py:173-250]

**How to avoid:** Expose authenticated descriptor facts through BlobStore, document the V1 ordering, and report pending work until the bounded scan can prove completion. Do not introduce a derived aggregate authority.

**Warning signs:** Size enforcement reads `_stats`, filesystem sizes, backend SQL directly, or claims exact LRU.

### Pitfall 5: Typed failure becomes a generic miss

**What goes wrong:** Conflict or backend outage triggers application recomputation and a competing write, masking the real fault.

**Why it happens:** Current `get()` returns `None` for absence, expiry, and integrity failure, while some conflicts are caught and converted to `False`. [VERIFIED: src/cacheness/core.py:184-230] [VERIFIED: src/cacheness/core.py:312-344]

**How to avoid:** Map failures through the shared classifier, preserve `cause`, and let the decorator suppress only declared outcomes.

**Warning signs:** `except Exception: return None`, `return False` on lifecycle conflict, or statistics recording `absent` for an exception.

### Pitfall 6: Public cutover is only half-applied

**What goes wrong:** Old aliases, global factories, examples, or tests silently recreate overlapping ownership.

**Why it happens:** Current top-level exports include `UnifiedCache as cacheness` and `get_cache`, while decorators retain `memoize`, `cache_function`, `for_api`, and `CacheContext`. [VERIFIED: src/cacheness/__init__.py:32-33] [VERIFIED: src/cacheness/__init__.py:178-244] [VERIFIED: src/cacheness/decorators.py:259-362]

**How to avoid:** Make export/config/constructor/decorator/docs/tests one wave and add negative import/attribute assertions for removed names.

**Warning signs:** Compatibility tests are updated to route aliases to the new constructor instead of asserting their absence.

### Pitfall 7: Close revokes a committed write

**What goes wrong:** A post-commit derived projection/statistics or close failure is treated as if the canonical BlobStore commit failed.

**Why it happens:** Policy owns an external sink and collapses its failure with lifecycle state.

**How to avoid:** Preserve `CacheBlobCommittedPartialError` fields (`receipt`, `continuation_cursor`, and projection evidence) and make policy close terminal only after the owned BlobStore close succeeds. [VERIFIED: src/cacheness/error_handling.py:450-484]

**Warning signs:** Policy rollback after a receipt exists, or a repair prerequisite created solely by statistics/projection failure.

## Code Examples

Verified patterns from official sources and current storage contracts:

### Explicit decorator binding and one lookup

```python
# Source pattern: https://docs.python.org/3.11/library/functools.html
from functools import wraps


def cached(*, cache: UnifiedCache):  # sole public decorator [VERIFIED: 06-CONTEXT.md:28-38]
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = cache.lookup_function(func, args, kwargs)  # recommended name [ASSUMED]
            if result.outcome is CacheOutcome.HIT:
                return result.value  # valid even when None
            if result.outcome in cache.policy.recompute_outcomes:
                value = func(*args, **kwargs)
                cache.store_function_result(func, args, kwargs, value)
                return value
            raise result.cause

        def cache_clear():
            return cache.invalidate_function(func)

        wrapper.cache_clear = cache_clear
        return wrapper

    return decorate
```

The method/class names not fixed by CONTEXT are illustrative recommendations and are logged as assumptions; the semantic constraints are locked.

### Bounded exact removal executor

```python
# Source contract: src/cacheness/storage/blob_store.py:535-631
def remove_page(cache, query, *, cursor=None, limit=100):  # recommended helper [ASSUMED]
    page = cache.blob_store.query_catalog(
        query,
        schema=cache.catalog_schema,
        cursor=cursor,
        limit=limit,
        work_cap=limit,
    )  # current BlobStore performs complete preflight before authority dispatch
    outcomes = []
    for candidate in page.entries:
        outcomes.append(
            cache.blob_store.delete(
                candidate.key,
                expected=candidate.expectation,
            )
        )
    return cache.removal_report(outcomes, continuation_cursor=page.next_cursor)
```

The current `CatalogEntry` does not yet carry `expectation`; adding that authenticated view is a prerequisite, not permission for policy to construct expectations from untrusted values. [VERIFIED: src/cacheness/storage/catalog.py:657-704]

### Immutable statistics snapshot

```python
# Source pattern: https://docs.python.org/3.11/library/dataclasses.html
from dataclasses import dataclass


@dataclass(frozen=True)
class CacheStatistics:  # recommended name [ASSUMED]
    hit: int = 0
    absent: int = 0
    expired: int = 0
    corrupt: int = 0
    conflict: int = 0
    backend_error: int = 0

    @property
    def lookups(self) -> int:
        return sum((self.hit, self.absent, self.expired,
                    self.corrupt, self.conflict, self.backend_error))
```

All six exact value names appear verbatim in the locked D-05 quote above. [VERIFIED: .planning/phases/06-unifiedcache-policy-composition/06-CONTEXT.md:49-52]

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Cache coordinates payload paths and metadata rows | BlobStore entry lifecycle with immutable generations and exact expectations | Delivered by Phases 3–5 | Phase 6 must consume, not duplicate, the lifecycle engine. [VERIFIED: .planning/phases/05-payload-backends-and-supported-topology-qualification/05-VERIFICATION.md:18-42] |
| Boolean/raw-value cache lookup | Presence-bearing result with finite outcome classification | Phase 6 locked decision | Fixes cached `None`, typed failures, decorator semantics, and statistics through one boundary. [VERIFIED: .planning/phases/06-unifiedcache-policy-composition/06-CONTEXT.md:49-65] |
| Local unbounded list loops | Portable bounded catalog query + opaque cursor + exact delete | Phase 4 storage contract, consumed in Phase 6 | Makes predicate/clear/eviction topology-neutral without pretending remote live qualification. [VERIFIED: src/cacheness/storage/blob_store.py:547-631] |
| Global cache factory and aliases | Explicit instance injection and one canonical surface | Phase 6 cutover | Removes hidden lifecycle ownership and overlapping constructors. [VERIFIED: .planning/phases/06-unifiedcache-policy-composition/06-CONTEXT.md:28-45] |

**Deprecated/outdated:**

- `cacheness` class alias, `get_cache`, `reset_cache`, implicit global cache acquisition, alternate decorator names, `CacheContext`, and overlapping convenience constructors: remove rather than wrap. [VERIFIED: src/cacheness/__init__.py:32-33] [VERIFIED: src/cacheness/core.py:432-472] [VERIFIED: src/cacheness/decorators.py:259-362]
- Mutable dictionary statistics such as exact keys `"cache_hits"` and `"cache_misses"`: replace with the common frozen outcome aggregate. [VERIFIED: src/cacheness/core.py:73-79]
- Local-only `list()` loops for cleanup/eviction/stats: replace with bounded catalog work. [VERIFIED: src/cacheness/core.py:232-279] [VERIFIED: src/cacheness/core.py:365-409]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Name the extracted internal module `cache_policy.py`. | Recommended Project Structure | Low; planner may select a different focused module name. |
| A2 | Use class names `CacheOutcome`, `CacheLookupResult`, and `CacheStatistics`. | Patterns / Code Examples | Low; exact names are explicitly delegated to implementation discretion. |
| A3 | Use method names `lookup_function`, `store_function_result`, and `invalidate_function`. | Code Examples | Low; public/internal naming can change without changing the required semantics. |
| A4 | Extend the BlobStore catalog view so a candidate carries authenticated intrinsic facts and an exact expectation. | Summary / Removal pattern | Medium; an equally narrow existing-authority API may satisfy the same need, but policy must not fabricate authority evidence. |
| A5 | Model size enforcement as bounded/resumable maintenance that reports incomplete work until the configured invariant can be proven. | Summary / Pitfalls | Resolved: explicit resumes converge only for a finite quiescent inventory; conflicts or revision churn remain typed incomplete/retryable outcomes. |

## Open Questions (RESOLVED)

1. **What exact V1 size-limit completion promise is public?**
   - **Resolution:** Per D-12, size enforcement is bounded and resumable. One `put()` performs at most one bounded maintenance step and may return incomplete work. Repeated caller-driven resumes must eventually enforce the configured byte limit only when the authoritative inventory is finite and quiescent. A conflict, stale continuation, unavailable authoritative fact, or authority revision churn returns a typed incomplete/retryable result with continuation or restart guidance. The contract does not promise universal contender success, completion under perpetual churn, exact LRU, or global-oldest selection.

2. **Does an injected BlobStore remain caller-owned?**
   - **Resolution:** Per D-02 and D-14, an injected `BlobStore` remains caller-owned and `UnifiedCache.close()` does not close it. A `BlobStore` created by `UnifiedCache` from the supplied `StoreTopology` is cache-owned and is initialized/closed exactly once by the cache. Ownership follows constructor form; there is no ownership boolean or implicit transfer.

3. **Which non-hit outcomes may the decorator recompute by default?**
   - **Resolution:** Per D-07 and D-15, default decorator recomputation is exactly `absent` and `expired`. Recomputing after `corrupt`, `conflict`, or `backend_error` requires an explicit policy, and the original outcome plus typed cause remains recorded and inspectable.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| `uv` | Locked environment and test commands | ✓ | `0.12.9` | — [VERIFIED: command `uv --version`] |
| Project virtualenv Python | Phase implementation/tests | ✓ | `3.13.15` | System Python `3.12.1` is also available. [VERIFIED: command `.venv/bin/python --version`; command `python3 --version`] |
| pytest | Validation | ✓ | `8.4.1` | — [VERIFIED: command `uv run --frozen pytest --version`] |
| PostgreSQL service | Deterministic remote candidate only in Phase 6 | Not required | — | Use contract doubles; live qualification is Phase 8. [VERIFIED: .planning/phases/06-unifiedcache-policy-composition/06-CONTEXT.md:245-246] |
| Amazon S3 | Deterministic remote candidate only in Phase 6 | Not required | — | Use existing deterministic candidate/mocking boundary; live qualification is Phase 8. [VERIFIED: .planning/phases/06-unifiedcache-policy-composition/06-CONTEXT.md:245-246] |

**Missing dependencies with no fallback:** None for Phase 6.

**Missing dependencies with fallback:** Python 3.11 itself was not observed locally; use the current 3.13 environment for Phase 6 and leave the full supported-version matrix to Phase 8 as explicitly deferred. [VERIFIED: .planning/phases/06-unifiedcache-policy-composition/06-CONTEXT.md:245-246]

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest, quoted lock value `version = "8.4.1"` [VERIFIED: uv.lock:1607-1621] |
| Config file | `pyproject.toml` [VERIFIED: pyproject.toml:84-116] |
| Quick run command | `uv run --frozen pytest <changed-test-file> -q -o log_cli=false` |
| Full suite command | `uv run --frozen pytest -q -o log_cli=false` |

The phase-relevant starting baseline is 60 passes for the four current UnifiedCache/decorator/translation test files; this is not a claim that the repository-wide suite is green. [VERIFIED: command recorded under Standard Stack]

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CACH-01 | Every cache path reaches canonical state only through BlobStore/shared engine across local profiles and deterministic remote candidate | contract/adversarial | `uv run --frozen pytest tests/contracts/test_phase6_topology_policy.py -q -o log_cli=false` | ❌ Wave 0 |
| CACH-02 | Key/TTL/eviction/invalidation/stats/decorator policy stays in UnifiedCache and never authorizes from derived state | unit/architecture | `uv run --frozen pytest tests/test_phase6_policy_contract.py -q -o log_cli=false` | ❌ Wave 0 |
| CACH-03 | TTL, size, predicate, decorator, key, and global paths use exact deletion and one structured report under conflict/retryable outcomes | adversarial | `uv run --frozen pytest tests/test_phase6_removal_contract.py -q -o log_cli=false` | ❌ Wave 0 |
| CACH-04 | Direct and decorated cached `None` are hits; BlobStore read occurs once; function invocation count stays one | unit | `uv run --frozen pytest tests/test_phase6_lookup_contract.py -q -o log_cli=false` | ❌ Wave 0 |
| CACH-05 | Immutable stats separately record exact locked outcomes and derive totals/rates without catalog scans | unit | `uv run --frozen pytest tests/test_phase6_statistics.py -q -o log_cli=false` | ❌ Wave 0 |
| CACH-06 | One import/config/constructor/decorator/result surface; removed aliases absent; optional failures and committed-partial evidence preserved | public contract | `uv run --frozen pytest tests/test_phase6_public_api_contract.py tests/test_phase6_decorator_contract.py -q -o log_cli=false` | ❌ Wave 0 |
| CACH-07 regression | `SqlCache` remains separately importable and representative behavior is unchanged | regression | `uv run --frozen pytest tests/test_sql_cache.py -q -o log_cli=false` | ✅ existing |

### Required Adversarial Cases

- Cached `None`: two identical decorated calls execute the function exactly once and record one absent then one hit.
- Single-snapshot rule: spy on BlobStore and assert one `open_entry()` per lookup, including expiry/corruption paths.
- Replacement race: selection observes generation A, test promotes B, exact delete conflicts, B remains readable, and the report increments conflicted/retryable.
- Malformed predicate: validation raises before query/delete call counts change.
- Bounded pagination: more entries than one page, configured work cap stops the call, returned continuation resumes without duplicate deletion/materialization.
- Function isolation: clearing one decorated function removes only its namespace and returns the actual removal report.
- Derived-state failure: statistics/projection failure never authorizes deletion or changes a successful BlobStore receipt; typed committed-partial evidence survives.
- Close race: failed/retryable close does not falsely advertise a terminally closed canonical store and cannot revoke an earlier commit.
- Public negative surface: removed aliases/factories/decorators are not exported and no wrapper silently constructs a cache.
- Remote candidate: typed retryable outcomes are translated without promising live PostgreSQL/S3 qualification or universal contender success.

### Sampling Rate

- **Per task commit:** run the mapped test file plus the nearest existing regression file.
- **Per wave merge:** `uv run --frozen pytest tests/test_unified_cache_lifecycle_authority.py tests/test_unified_cache_adversarial_lifecycle.py tests/test_blob_store_translation_seam.py tests/test_phase6_lookup_contract.py tests/test_phase6_removal_contract.py tests/test_phase6_statistics.py tests/test_phase6_public_api_contract.py tests/test_phase6_decorator_contract.py tests/contracts/test_phase6_topology_policy.py -q -o log_cli=false`
- **Phase gate:** run the full suite with `uv run --frozen pytest -q -o log_cli=false`; compare any unrelated pre-existing failures explicitly, but require all Phase 6 and Phase 3–5 lifecycle contract tests green before `$gsd-verify-work`.

### Wave 0 Gaps

- [ ] `tests/test_phase6_lookup_contract.py` — presence, outcome translation, one-read, cached-None cases for CACH-04/CACH-05.
- [ ] `tests/test_phase6_removal_contract.py` — exact deletion/report/pagination/race cases for CACH-03.
- [ ] `tests/test_phase6_statistics.py` — immutable per-outcome aggregate for CACH-05.
- [ ] `tests/test_phase6_decorator_contract.py` — explicit ownership, `wraps`, suppression, function clear for CACH-02/CACH-04/CACH-06.
- [ ] `tests/test_phase6_public_api_contract.py` — positive canonical exports and negative legacy surface for CACH-06.
- [ ] `tests/test_phase6_policy_contract.py` — TTL/size/invalidation ownership and fail-closed config preflight for CACH-02.
- [ ] `tests/contracts/test_phase6_topology_policy.py` — local supported profiles and deterministic remote-candidate semantics for CACH-01.

No new test framework or shared external-service fixture is required.

## Security Domain

OWASP ASVS 5.0.0 is the latest stable ASVS release listed by the official project; for this in-process storage library, only the categories touching input validation, integrity, and cryptography materially apply. [CITED: https://github.com/OWASP/ASVS]

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No user-authentication boundary in this library phase. |
| V3 Session Management | no | No web session boundary. |
| V4 Access Control | no at policy API | Deployment/backend credentials and filesystem permissions remain storage/environment concerns; cache policy must not bypass them. |
| V5 Validation, Sanitization and Encoding | yes | Existing typed `CatalogSchema`/`CatalogQuery` validation, strict config preflight, safe parsing, and fail-before-delete behavior. [VERIFIED: src/cacheness/storage/catalog.py:246-442] |
| V6 Stored Cryptography | yes | Reuse BlobStore manifest authenticity and payload integrity; never add cache-layer cryptography. [VERIFIED: src/cacheness/storage/manifest.py:173-250] |

### Known Threat Patterns for Python cache policy

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malformed or overbroad predicate causes mass deletion | Tampering | Validate the entire portable query and work bounds before the first query/delete; no `eval`, raw SQL, or backend bypass. |
| Stale selection deletes a replacement generation | Tampering | Carry exact `EntryExpectation`; conflict fails closed and is reported. |
| Corrupt evidence is translated to absence and destructively cleaned | Tampering / Repudiation | Classify `corrupt`, preserve evidence, and make canonical corruption non-destructive. |
| Unbounded catalog traversal exhausts memory/time | Denial of Service | Enforce page size plus total per-call work cap; return opaque continuation. |
| Broad decorator suppression hides outages/contention | Repudiation | Preserve typed cause and record actual outcome before any explicit fallback. |
| Result/report leaks signing keys, credentials, or payloads | Information Disclosure | Reports contain identifiers/counts/typed causes, never secret bytes or credential values. |
| Close/maintenance race changes canonical state after a valid commit | Denial of Service / Tampering | BlobStore/shared engine remains sole lifecycle sequencer; statistics/projections cannot revoke or repair. |

## Sources

### Primary (HIGH confidence)

- `docs/adr/0001-topology-specific-storage-guarantees.md` — lifecycle authority, progress, ACID boundary, and stop conditions.
- `.planning/phases/06-unifiedcache-policy-composition/06-CONTEXT.md` — locked public, outcome, TTL, invalidation, eviction, decorator, and close decisions.
- `src/cacheness/storage/blob_store.py`, `lifecycle.py`, `catalog.py`, `read_contract.py`, `manifest.py`, and `composition.py` — current canonical storage/query/topology contracts opened directly this session.
- `src/cacheness/core.py`, `decorators.py`, `config.py`, and `__init__.py` — current policy/public seams opened directly this session.
- Phase-relevant pytest execution — 60 tests passed in 1.44 seconds on 2026-09-08.

### Secondary (MEDIUM confidence)

- https://docs.python.org/3.11/library/dataclasses.html — frozen dataclass semantics.
- https://docs.python.org/3.11/library/enum.html — finite enum patterns.
- https://docs.python.org/3.11/library/functools.html — `wraps` behavior.
- https://github.com/OWASP/ASVS — current stable ASVS release and security-control taxonomy.

### Tertiary (LOW confidence)

- None. Proposed names remain isolated in the Assumptions Log; the size-enforcement completion model is resolved above.

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — no new package; current Python/storage/test contracts were inspected directly, and stdlib patterns were checked against official Python documentation.
- Architecture: HIGH — locked CONTEXT/ADR decisions align with directly inspected BlobStore, catalog, lifecycle, and current cache seams.
- Pitfalls: HIGH — each material pitfall is grounded in current source/tests or a locked concurrency/integrity rule.
- Size-enforcement completion model: HIGH — bounded explicit-resume completion for a finite quiescent inventory and typed incomplete/retryable behavior under conflicts or revision churn are fixed by the resolved planning contract.

**Research date:** 2026-09-08
**Valid until:** 2026-10-08 for stable in-repo contracts; re-check immediately if Phases 3–5 storage contracts change.
