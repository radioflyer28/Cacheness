# Phase 4: Metadata Composition and Topology Contracts - Research

**Researched:** 2026-09-07
**Domain:** authenticated catalog metadata, portable queries, backend composition, and topology capability contracts
**Confidence:** HIGH for repository architecture; MEDIUM for the recommended new public vocabulary

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

### Catalog Schema

- **D-01:** The existing application metadata mapping remains the default and compatibility surface. Callers may optionally attach a declared catalog schema; a schema is not required merely to store and reopen metadata.
- **D-02:** When a schema is present, undeclared metadata fields are preserved and round-trip as opaque values. Only declared fields receive validation and portable query/index guarantees. They must not be silently dropped or retroactively rejected.
- **D-03:** Phase 4 provides a small Cacheness-native declarative schema. Field definitions cover name, supported value type, required/default behavior, validation, and query/index intent. The native contract may expose an adapter seam for external model libraries later, but Pydantic, dataclasses, SQLAlchemy models, or another framework do not define the canonical Phase 4 API. — **Reversibility:** costly — Applications and backend adapters will persist and consume these declarations, so replacing the public schema vocabulary would require compatibility adapters and schema migration.
- **D-04:** Schema evolution is backward-readable. New writes and metadata updates validate against the active schema; existing entries with absent newly declared fields remain readable using explicit missing/default semantics. Incompatible schema changes require an explicit offline migration. Reads never rewrite metadata as an incidental schema upgrade. — **Reversibility:** costly — Automatic or strict reopen-time migration would change read side effects and stored-catalog compatibility.

### Portable Query and Index Contract

- **D-05:** The initial portable query language supports typed equality, comparison/range, membership, and existence predicates over declared queryable fields. Multiple predicates compose with AND. Nested OR/NOT expressions and backend-native query strings are outside the portable contract.
- **D-06:** Portable pagination uses a stable opaque cursor with deterministic canonical ordering by entry key and generation identity plus bounded page sizes. Arbitrary field sorting is capability-specific and is not implied by the portable query contract. Offset pagination is not the resumability contract.
- **D-07:** Queryability and index intent are explicit parts of each declared field. Backends do not silently index every field or choose indexes as an undocumented semantic behavior. Secondary indexes are rebuildable acceleration state; they never become canonical read, delete, cleanup, or repair authority.
- **D-08:** A successful portable query returns results complete for the canonical catalog snapshot promised by that backend. It may use an index maintained in the same catalog transaction or a canonical scan. A stale or independently updated external index cannot silently answer a portable query with incomplete results. — **Reversibility:** costly — Callers may rely on catalog queries for inventory and administrative workflows, so weakening completeness later would invalidate published behavior.

### Backend Composition and Capabilities

- **D-09:** One typed store configuration is the primary composition root. It selects the payload backend, catalog authority, optional projections, and minimum required capabilities as one validated topology. Existing public constructors and configuration names remain compatibility adapters into that same path rather than parallel selection logic. — **Reversibility:** costly — This becomes the shared construction contract for direct `BlobStore` use and later `UnifiedCache` composition.
- **D-10:** A caller-injected backend instance remains the exact selected instance. Registered backend names resolve through the same construction path as built-ins. Supplying both an instance and a name for one role is an error; no selector silently wins and options are not merged into caller-owned instances.
- **D-11:** Every composed store exposes the actual semantic capabilities of its active payload/authority/projection pairing. Callers may request minimum guarantees; unmet requirements and inherently invalid pairings fail during construction. Callers are not required to choose a named topology tier, and Cacheness does not infer a stronger promise for them to depend upon implicitly.
- **D-12:** Resource ownership is explicit. Backends created by the composition root are store-owned and closed with the store. Injected instances are caller-owned by default and remain open, with an explicit option to transfer ownership. Hidden reference counting or resource sharing is not introduced. — **Reversibility:** costly — Close behavior is observable and shared injected resources depend on it.

### Derived Projections and External ORM Integration

- **D-13:** Phase 4 external ORM models and external indexes are derived-only integrations. Authoritative custom fields remain in the native catalog transaction with the blob descriptor. An external model cannot replace the catalog authority or participate in lifecycle sequencing in this phase.
- **D-14:** Derived projections synchronize by idempotently pulling bounded pages of committed canonical catalog state from a checkpoint. Notifications may prompt a refresh but are not the correctness source. Phase 4 does not require a background worker, synchronous-callback-only delivery, or exactly-once event protocol.
- **D-15:** Projection failure cannot roll back or revoke an already committed blob. The committed receipt exposes projection status. Ordinary best-effort refresh may warn; when a caller explicitly requests projection refresh, failure produces a typed committed-partial outcome that retains the canonical receipt and attribution of remaining derived work. — **Reversibility:** costly — This is the public failure boundary that prevents applications from treating a committed generation as absent.
- **D-16:** Projection rebuild is an explicit, capability-qualified operation. It builds isolated derived state from canonical catalog state and publishes that state only when complete. Each adapter reports whether it can catch up online or requires offline maintenance; Phase 4 promises neither mode universally. Normal reads and writes do not perform hidden projection repair.

### the agent's Discretion

- Exact public class, method, and module names for schema fields, query predicates,
  cursors, composition specifications, capability reports, and projection reports.
- The finite portable scalar/type set, missing-value representation, and validation
  error types, provided validation does not silently coerce values into a different
  catalog meaning.
- Default and maximum query page sizes and internal cursor encoding, provided the
  cursor is opaque, bounded, deterministic, and rejects incompatible versions.
- Internal schema fingerprint/version encoding and projection checkpoint storage,
  subject to authenticated canonical catalog state and the ADR stop conditions.

### Deferred Ideas (OUT OF SCOPE)

- A transactional external ORM extension that joins the native authority database
  transaction is not part of Phase 4. Reconsider only with a concrete use case that
  cannot be represented by native authoritative fields plus a derived projection.
- Full PostgreSQL authority and filesystem/memory/S3 payload-pair qualification belongs
  to Phase 5.
- Complete `UnifiedCache` policy delegation, statistics, invalidation, and compatibility
  acceptance belongs to Phase 6.
- Stored-format and schema migration execution belongs to Phase 7; Phase 4 defines the
  compatibility boundary and explicit migration requirement only.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BACK-02 | JSON, memory, SQLite, and PostgreSQL metadata implementations have explicit authority/projection roles through one composition contract. Narrow transactional catalog adapters before expansion; a derived JSON view does not become an independent lifecycle authority. | Role table, one-root construction, and projection pull protocol below. |
| BACK-03 | Caller-injected and registered backend implementations remain selected rather than being silently replaced by configuration defaults. | Exact-instance and ownership rules plus compatibility tests below. |
| BACK-06 | Backends expose durability, process/host sharing, compare-and-swap, streaming, and listing capabilities, and configurations cannot claim guarantees their topology cannot provide. | Semantic participant and composed-capability model below. |
| BACK-07 | Direct `BlobStore` users can store, validate, query, and update application-defined catalog metadata without implementing a lifecycle backend; supported fields/operators and transactional limits are explicit. Extend the existing mapping/entry interface rather than replacing the engine. Authoritative metadata commits with the blob descriptor; external indexes or ORM links are explicitly derived unless they join that same transaction, with consistency and partial-failure behavior stated. | Native schema, typed query, same-transaction authority extension, receipt, and failure matrix below. |
</phase_requirements>

## Summary

Phase 4 should extend the Phase 3 authority visibility switch, not add another coordinator. The existing `AuthorityLifecycleEngine` already stages immutable payloads, signs authenticated `user_metadata`, and promotes one descriptor through the selected `LifecycleAuthority`; `BlobStore` already exposes `put_entry`, `get_entry_info`, `open_entry`, `get_metadata`, `update_metadata`, and `list`. [VERIFIED: src/cacheness/storage/blob_store.py:331-419] SQLite promotion updates the entry, lineage, mutation state, cleanup debt, authority revision, and projection-dirty bit within the existing `_transaction()` boundary. [VERIFIED: src/cacheness/storage/sqlite_lifecycle_authority.py:1222-1298] Authoritative declared catalog values and any semantic index must join that exact promotion transaction (and the memory authority's existing lock transition), or they become a second visibility switch. [VERIFIED: docs/adr/0001-topology-specific-storage-guarantees.md:36-44,83-100]

The current composition surface is fragmented: `BlobStore` overloads `backend` to choose both an authority and a compatibility projection, the backend registry has a different path, and `create_metadata_backend()` is a third hard-coded factory. [VERIFIED: src/cacheness/storage/blob_store.py:215-244; src/cacheness/storage/backends/__init__.py:99-235; src/cacheness/metadata.py:3275-3357] Worse, built-in JSON/SQLite/memory classes inherit the `MetadataBackend` ABC in `metadata.py`, while the registry validates against a different `MetadataBackend` ABC in `storage/backends/base.py`. [VERIFIED: src/cacheness/metadata.py:269-382; src/cacheness/storage/backends/base.py:12-159; src/cacheness/storage/backends/__init__.py:127-170] Plan a role-specific, structural capability seam and route every old name/constructor through one typed root.

The portable query must be a bounded authority query whose successful result is complete for one revision-bound canonical snapshot. SQLite supports efficient keyset paging using row-value comparison and guarantees that a read transaction sees an unchanging snapshot; because a cursor is resumed across calls, bind it to the authority revision and return a typed stale-cursor/restart outcome if the revision changed. [CITED: https://www.sqlite.org/rowvalue.html] [CITED: https://www.sqlite.org/isolation.html] External ORM/JSON/PostgreSQL metadata implementations remain lagging, rebuildable projections and may never authorize reads, deletes, cleanup, reconciliation, or query completeness. [VERIFIED: docs/adr/0001-topology-specific-storage-guarantees.md:139-171]

**Primary recommendation:** add a small native catalog value/query contract to the existing authority transaction, then compose payload, authority, projections, ownership, and minimum capabilities through one validated root; preserve every existing constructor as a thin adapter to that root. [ASSUMED]

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Schema declaration and validation | BlobStore storage API | Catalog authority | BlobStore validates before payload side effects; the authority records schema identity and declared values at promotion. [ASSUMED] |
| Canonical catalog commit | Catalog authority | BlobStore lifecycle engine | The authority transaction is the sole visibility switch; the lifecycle engine retains sequencing. [VERIFIED: docs/adr/0001-topology-specific-storage-guarantees.md:92-97,141-147] |
| Portable query and cursor | Catalog authority | BlobStore storage API | Authority supplies a complete bounded page; BlobStore authenticates public results and hides adapter details. [ASSUMED] |
| Payload bytes | Payload backend / guarded handler I/O | BlobStore lifecycle engine | Payloads are immutable generations published before authority promotion; cross-resource ACID is not promised. [VERIFIED: docs/adr/0001-topology-specific-storage-guarantees.md:36-40,88-97] |
| Derived ORM/index/JSON views | Projection adapter | External database or file | They pull committed canonical pages, checkpoint, and rebuild; they are never authority. [ASSUMED] |
| Topology selection and ownership | Composition root | Compatibility constructors | One root validates active instances and retains explicit close responsibility. [ASSUMED] |
| Cache TTL/eviction/statistics | UnifiedCache policy layer | BlobStore entry interface | Phase 4 must not absorb Phase 6 cache-policy migration. [VERIFIED: AGENTS.md:15-22] |

## Project Constraints (from AGENTS.md)

- Preserve supported public APIs; migration or rebuild must be explicit and documented. `BlobStore` owns storage lifecycle, `UnifiedCache` depends on it and owns cache policy, and `SqlCache` stays separate. [VERIFIED: AGENTS.md:13-22]
- Cover the advertised filesystem, memory, S3, JSON, SQLite, and PostgreSQL families without pretending every Cartesian pairing is qualified in this phase. [VERIFIED: AGENTS.md:15-22] [VERIFIED: .planning/REQUIREMENTS.md:24-29]
- Treat application payloads as trusted, but enforce safe parsing, path containment, and fail-closed integrity; same-key operations may conflict or time out but may not corrupt or split payload/metadata. [VERIFIED: AGENTS.md:18-22]
- Keep Python support at `>=3.11`; the checked development environment is pinned to 3.13, so public types and tests cannot rely only on the ambient interpreter. [VERIFIED: AGENTS.md:22,45-52]
- Use snake_case modules/functions, PascalCase classes, package-relative internal imports, domain exceptions with preserved causes, focused helpers, dataclass configuration groupings, and `__init__.py` re-exports for public conveniences. [VERIFIED: AGENTS.md:116-176,196-284]
- Run `uv run ruff check src tests`; do not add to the existing lint baseline or suppress findings without a local reason. [VERIFIED: AGENTS.md:146-157]
- The mandatory ADR forbids a second lifecycle owner, adapter-level lifecycle duplication, derived views as authority, unsupported cross-resource ACID claims, and benchmark-derived correctness deadlines. [VERIFIED: docs/adr/0001-topology-specific-storage-guarantees.md:139-171]

## Standard Stack

### Core

| Library / facility | Version | Purpose | Why Standard |
|--------------------|---------|---------|--------------|
| Python standard library | `>=3.11` | frozen value objects, protocols, enums, hashing/HMAC, JSON, SQLite | No new runtime dependency is needed; the repository already requires Python `>=3.11`. [VERIFIED: pyproject.toml:9] |
| `sqlite3` | runtime bundled; local probe `3.43.1` | qualified local catalog authority and atomic catalog rows/indexes | The existing SQLite authority uses bounded `BEGIN IMMEDIATE` transactions and is the Phase 3 canonical local authority. [VERIFIED: src/cacheness/storage/sqlite_lifecycle_authority.py:1030-1105] |
| Existing manifest/HMAC code | manifest schema `1`, payload format `1`, signature algorithm `"hmac-sha256"`, digest `"sha256"` | authenticate descriptor plus application metadata and cursor envelopes | Reuse the established integrity boundary and key provider; do not introduce an independent cursor secret. [VERIFIED: src/cacheness/storage/manifest.py:22-32] |
| pytest | `>=8.4.1` | authority-contract, fault-injection, composition, and compatibility tests | Declared dev framework and existing tests already exercise transition rollback/uncertain commit. [VERIFIED: pyproject.toml:71-91; tests/test_lifecycle_authority_contract.py:311-508] |

### Supporting

| Library / facility | Version | Purpose | When to Use |
|--------------------|---------|---------|-------------|
| SQLAlchemy | optional `>=2.0.0` | legacy/custom ORM derived projections only | Use behind projection adapters; do not expose its models/sessions as the native catalog API. [VERIFIED: pyproject.toml:23,40,56; src/cacheness/custom_metadata.py:100-274] |
| psycopg | optional `>=3.1.0` | existing PostgreSQL metadata/projection integration | Keep derived in Phase 4; PostgreSQL lifecycle authority qualification is Phase 5. [VERIFIED: pyproject.toml:40; .planning/phases/04-metadata-composition-and-topology-contracts/04-CONTEXT.md:166-172] |
| Existing safe query validation | repository code | bound identifiers, depth, and signed-64 integers | Reuse as input-validation substrate, then add declared-field/type/operator checks. [VERIFIED: src/cacheness/query_validation.py:11-77] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Native Cacheness schema | Pydantic/dataclasses/SQLAlchemy models | Locked out as the canonical API; adapters may be added later. |
| Normalized typed catalog rows | SQLite JSON expression indexes | JSON expression indexes are backend-specific and require query expressions to match the indexed expression; normalized typed rows make the portable type contract explicit. [CITED: https://www.sqlite.org/expridx.html] |
| Revision-bound keyset cursor | Long-lived read transaction or offset | A long-lived connection burdens callers; offset work grows with the offset. Keyset plus revision makes resumability explicit. [CITED: https://www.sqlite.org/rowvalue.html] |
| Pull/checkpoint projection | Synchronous callback/event authority | Callbacks cannot be the correctness source and would create a second commit dependency. |

**Installation:** No new package should be installed. [ASSUMED]

## Package Legitimacy Audit

Not applicable: the recommended design uses the Python standard library and existing declared optional dependencies; Phase 4 should add no external package. [ASSUMED]

## Architecture Patterns

### System Architecture Diagram

```text
put_entry / update_metadata
          |
          v
BlobStore schema validation ---- invalid ----> typed pre-commit failure
          |
          v
existing AuthorityLifecycleEngine (the only coordinator)
          |
          +--> publish immutable payload generation
          |
          v
LifecycleAuthority promotion transaction
  descriptor + schema identity + declared values + semantic indexes
          |
          +--> committed BlobEntryInfo / projection status
          |
          +--> bounded canonical catalog pages --> projection pull/checkpoint
          |                                      --> isolated rebuild/publish
          v
query(schema, predicates, cursor, limit)
  validate AST -> authority revision snapshot -> keyset page -> authenticate results

Composition root
  payload role + authority role + projection roles + ownership + minimum caps
          |
          +--> reject ambiguity/unsupported pairing before mutation
          +--> expose actual composed capability report
```

This flow preserves authority promotion as the only visibility point and keeps external effects in reconciliation/projection space rather than one fictional cross-resource transaction. [VERIFIED: docs/adr/0001-topology-specific-storage-guarantees.md:36-44,83-100]

### Recommended Project Structure

```text
src/cacheness/storage/
├── catalog.py                 # native schema, predicates, page/cursor/result values [ASSUMED]
├── composition.py             # role specs, ownership, registries, capability report [ASSUMED]
├── lifecycle_authority.py     # narrow catalog authority protocol extensions
├── lifecycle.py               # existing sole lifecycle coordinator
├── blob_store.py              # compatible public entry + query facade
├── memory_lifecycle_authority.py
├── sqlite_lifecycle_authority.py
└── projections.py             # pull/checkpoint/rebuild adapter contract [ASSUMED]
```

Names are within the agent's discretion; the important boundary is that `catalog.py` contains declarative/query values, authority implementations own transactions, and `lifecycle.py` remains the only sequencing engine. [ASSUMED]

### Pattern 1: Conservative Native Schema

Use immutable native declarations with: schema identity/version; field name; one of string, signed-64 integer, boolean, or explicit nullability; required/default behavior; serializable validation constraints; queryable flag; index intent. Keep missing distinct from stored `null`, reject coercion (`True` is not integer), and preserve undeclared values as opaque manifest metadata. [ASSUMED]

The finite set aligns with the current canonical manifest, which accepts `None`, `bool`, signed-64 `int`, `str`, lists, and mappings but explicitly rejects floating-point values. [VERIFIED: src/cacheness/storage/manifest.py:92-163] Recommended v1 declared fields are scalar only; opaque undeclared metadata may retain the existing bounded nested structures. [ASSUMED]

Use only serializable built-in constraints in the persisted schema fingerprint (for example integer bounds, string length, membership choices). Do not put arbitrary Python callables into canonical schema identity because other processes cannot deterministically reconstruct them. [ASSUMED]

Default semantics should be explicit: materialize defaults on new writes, expose a missing sentinel for old entries, and make existence predicates inspect stored presence. A read must not rewrite the descriptor or silently claim that an old entry physically contains a default. [ASSUMED]

### Pattern 2: Narrow Catalog Authority Seam

Add semantic operations such as `catalog_page(query, cursor, limit)` and schema/capability inspection to the authority, plus declared values carried as part of the existing mutation/promotion data. Do not expose `sqlite3.Connection` or reproduce `prepare/record/promote/cleanup` on a catalog adapter. [ASSUMED]

SQLite should store typed catalog values and rebuildable index state in authority-owned tables keyed by `(entry_key, generation, field_name)` and update them inside `promote_mutation()`'s existing transaction. [ASSUMED] The current promotion transaction's exact visibility state includes `entries`, `entry_lineage`, mutation state, cleanup debt, and `authority_state.revision`; placing catalog writes afterward would create a second visibility switch. [VERIFIED: src/cacheness/storage/sqlite_lifecycle_authority.py:1222-1298]

The signed manifest's `user_metadata` remains the canonical value source. Typed rows are query acceleration and must be rebuildable/corroborated from that signed mapping; a query result is exposed only after the selected `EntrySnapshot.manifest` authenticates and agrees. [ASSUMED] This preserves the existing exact v1 top-level shape while preventing a same-database index from becoming authorization evidence. [VERIFIED: src/cacheness/storage/manifest.py:34-53,238-257,388-395]

Memory authority must perform the equivalent immutable snapshot/catalog update inside its existing single `_transition()` lock. [VERIFIED: src/cacheness/storage/memory_lifecycle_authority.py:41-80]

### Pattern 3: Revision-Bound Portable Query

Represent query input as value objects, never raw SQL: a bounded tuple of AND predicates, each containing declared field, finite operator, and already type-checked operand. [ASSUMED] Exact portable operators should be equality, `<`, `<=`, `>`, `>=`, inclusive range, membership, and existence. [ASSUMED] Reject unqueryable/unknown fields, heterogeneous membership lists, boolean-as-integer, excess predicates/items, invalid cursor, and excess page size before touching the backend. [ASSUMED]

Return a bounded page ordered by `(key, generation)` and an opaque authenticated cursor containing format version, store identity, schema fingerprint, query fingerprint, authority revision, last key, and last generation. [ASSUMED] SQLite row-value comparisons directly support the keyset shape `(key, generation) > (?, ?)` and avoid offset-proportional scanning. [CITED: https://www.sqlite.org/rowvalue.html]

At resume, reject a cursor if store/schema/query/version differs. If the authority revision differs, return a typed retryable stale-snapshot result instructing restart; never label a page sequence complete after mixing revisions. [ASSUMED] SQLite can hold a stable snapshot within one read transaction, but a cursor resumed later is normally a new transaction, so the explicit revision check is the portable contract. [CITED: https://www.sqlite.org/isolation.html] [CITED: https://www.sqlite.org/lang_transaction.html]

An index may answer only when transactionally current for the same authority revision and schema fingerprint. Otherwise scan the authenticated canonical descriptor set or return a typed capability/rebuild-needed failure; stale derived indexes may not return a successful incomplete page. [ASSUMED]

### Pattern 4: One Typed Composition Root

Define a role specification with exactly one of `name` or `instance`; options apply only to named construction. Track `owned=True` for constructed resources, `owned=False` for injected resources, and accept an explicit transfer flag. [ASSUMED] Resolve built-ins and registered names through the same role registry, then validate the actual constructed payload/authority/projection capabilities before the store can initialize or stage bytes. [ASSUMED]

Capability reports should be semantic and scoped, not backend-name booleans: durability scope, sharing scope (`process`, `host`, `multi_host`), exact CAS, transaction scope, immutable generations, streaming, listing, portable query, indexed paging, projection catch-up/rebuild mode, and initialization/migration state. [ASSUMED] The current `AuthorityCapabilities` exact fields are `durable`, `multiprocess`, `transactional`, `exact_cas`, `indexed_paging`, and `projection`; it lacks payload streaming/listing and host-sharing scope. [VERIFIED: src/cacheness/storage/lifecycle_authority.py:267-276]

Keep current names and constructor arguments as compatibility adapters. In particular, `backend="memory"` must still map to the explicitly ephemeral memory topology and existing `backend="json"` behavior must remain SQLite authority plus JSON projection unless the compatibility baseline says otherwise. [VERIFIED: src/cacheness/storage/blob_store.py:215-244] Do not infer PostgreSQL authority in Phase 4: the existing PostgreSQL class is a metadata/projection backend, not a `LifecycleAuthority`. [VERIFIED: src/cacheness/storage/backends/postgresql_backend.py:88-490]

### Pattern 5: Pull-Based Derived Projection

Projection adapters accept bounded canonical pages plus an idempotency identity `(store, schema fingerprint, authority revision, key, generation)` and persist a checkpoint after applying the page. [ASSUMED] Notifications may only prompt a pull. A retry must converge without duplicate or stale ownership. [ASSUMED]

Ordinary best-effort refresh logs/records derived debt without revoking the commit. Explicit refresh raises a typed committed-partial exception/report carrying the exact `BlobEntryInfo` receipt and remaining projection adapter/checkpoint work. [ASSUMED] The existing receipt exact fields are `"key"`, `"generation"`, `"locator"`, `"expectation"`, `"metadata"`, and optional `"previous_locator"`; retain these fields and add projection status compatibly rather than replacing the type. [VERIFIED: src/cacheness/storage/read_contract.py:30-42]

Rebuild into isolated state, validate completion against a canonical snapshot/checkpoint, then atomically publish only if the adapter advertises that operation. Preserve old derived state on failure. Report online/offline mode; do not hide repair in reads/writes. [ASSUMED]

### Anti-Patterns to Avoid

- **Second lifecycle coordinator:** never let a metadata adapter repeat prepare/promote/delete/cleanup. Extend the existing authority's narrow transaction/query semantics. [VERIFIED: docs/adr/0001-topology-specific-storage-guarantees.md:141-147]
- **Projection as authority:** legacy SQLite/PostgreSQL projection CAS, JSON files, ORM links, or object listings cannot authorize canonical read/delete/cleanup/repair. [VERIFIED: docs/adr/0001-topology-specific-storage-guarantees.md:92-97,148-152]
- **Two-step authoritative metadata:** do not commit descriptor then separately write authoritative fields/index rows. Join the authority promotion transaction. [ASSUMED]
- **Name-based guarantees:** `"postgresql"`, `"sqlite"`, or `"s3"` is not evidence of topology semantics; validate active instances and their pairing. [VERIFIED: docs/adr/0001-topology-specific-storage-guarantees.md:70-81]
- **Silent selector precedence:** do not allow config defaults to overwrite injected instances or merge factory options into caller-owned objects. [VERIFIED: src/cacheness/core.py:181-284]
- **Unbounded/raw query:** no SQL strings, OR/NOT tree, offset resumability, unbounded `IN`, or unbounded page/list implementation in the new API. [ASSUMED]
- **Manifest-v1 field injection:** do not add a top-level field without a versioned reader; v1 requires the exact canonical field set and rejects missing/extra fields. [VERIFIED: src/cacheness/storage/manifest.py:34-53,388-395]
- **Implicit schema migration:** normal initialize/read cannot upgrade an established authority schema. [VERIFIED: src/cacheness/storage/blob_store.py:313-329; src/cacheness/storage/sqlite_lifecycle_authority.py:802-854]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Lifecycle sequencing | catalog adapter state machine | existing `AuthorityLifecycleEngine` + `LifecycleAuthority` transaction | Phase 3 already owns prepare/verify/promote/recovery. [VERIFIED: src/cacheness/storage/lifecycle.py:180-325] |
| SQLite locking/atomicity | extra filesystem lock or queue | existing bounded `_transaction()` with `BEGIN IMMEDIATE` | The ADR forbids patches that add competing coordination mechanisms. [VERIFIED: src/cacheness/storage/sqlite_lifecycle_authority.py:1030-1105; docs/adr/0001-topology-specific-storage-guarantees.md:153-171] |
| Cursor integrity | unsigned/base64 tuple | canonical bounded JSON + existing HMAC key provider | A cursor must reject tampering and incompatible context. [ASSUMED] |
| Query language | raw backend query strings | finite typed predicate AST compiled with bound parameters | Prevents injection and backend-semantic drift. [ASSUMED] |
| External projection delivery | exactly-once event bus/background daemon | idempotent bounded pull + checkpoint | Notifications are hints; checkpoints are correctness evidence. |
| Resource sharing | hidden refcounts/global pool ownership | explicit owned/caller-owned/transfer rules | Close behavior is observable and locked. |
| Schema/model framework | Pydantic/ORM canonical model | Cacheness-native declarations | Locked public contract; external models remain adapters. |

**Key insight:** Phase 4 is a seam-narrowing phase. Most hard primitives already exist; correctness depends on joining catalog semantics to the current authority transaction and deleting selector ambiguity, not on adding a more powerful coordinator. [ASSUMED]

## Runtime State Inventory

This phase changes stored catalog/schema layout and composition semantics, so migration-sensitive runtime state was audited even though Phase 7 performs migrations. [ASSUMED]

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | Existing SQLite authority is at `".cacheness/lifecycle-authority-v1.sqlite3"` with `SCHEMA_VERSION = 1`; v1 manifests have exact `schema_version = 1`/field shape. [VERIFIED: src/cacheness/storage/sqlite_lifecycle_authority.py:53-66; src/cacheness/storage/manifest.py:22-53] | Preserve no-schema reopen unchanged. Enabling persisted catalog schema/index layout on an established populated v1 store must return explicit migration-required evidence; do not mutate it during ordinary open. Phase 7 owns execution. [ASSUMED] |
| Live service config | No daemon/service-backed configuration is required for the qualified Phase 4 memory and SQLite/local implementations; PostgreSQL/custom ORM settings can exist in caller code and external databases. [VERIFIED: AGENTS.md:103-108; src/cacheness/storage/backends/postgresql_backend.py:88-245] | Treat external ORM/PostgreSQL structures as derived projections; do not silently create/upgrade them during BlobStore reads. [ASSUMED] |
| OS-registered state | None required by BlobStore; the library runs in the host process and has no worker/service registration contract. [VERIFIED: AGENTS.md:103-108] | None. |
| Secrets/env vars | Existing manifest HMAC key file is part of store identity/integrity; no environment-variable parser exists in `src/cacheness`. [VERIFIED: src/cacheness/storage/blob_store.py:197-200; AGENTS.md:93-101] | Reuse the store key for authenticated cursor envelopes through a narrow signer API; do not rename or regenerate it. [ASSUMED] |
| Build artifacts / installed packages | No new package is proposed. Existing optional SQLAlchemy/psycopg integrations remain packaging-controlled projections. [VERIFIED: pyproject.toml:23-63] | No reinstall beyond normal test extras; test base/no-extra behavior separately. [ASSUMED] |

## Common Pitfalls

### Pitfall 1: Catalog Rows Commit After Descriptor Promotion

**What goes wrong:** readers can observe a committed generation whose query fields are missing or stale, so direct read and portable query disagree. [ASSUMED]

**Why it happens:** current metadata projection hooks run after BlobStore commit, which is correct for derived state but not authoritative fields. [VERIFIED: src/cacheness/storage/blob_store.py:344-377; src/cacheness/core.py:1218-1334]

**How to avoid:** place authoritative catalog values and transaction-maintained indexes inside the existing promotion transaction; fault-inject every boundary and assert all participating rows roll back together. [ASSUMED]

**Warning signs:** a catalog adapter has `commit()`, a post-promotion hook can make a write appear failed without a receipt, or query tests read legacy projection tables. [ASSUMED]

### Pitfall 2: A Derived Index Silently Omits Results

**What goes wrong:** an administrative inventory query succeeds but misses canonical entries after projection lag/corruption. [ASSUMED]

**How to avoid:** prove index revision/schema identity matches the authority snapshot or perform a canonical scan; otherwise fail with a typed rebuild/capability outcome. Never use projection results to decide delete/repair. [ASSUMED]

### Pitfall 3: Cursor Mixes Revisions

**What goes wrong:** inserts/deletes between pages produce duplicates or omissions that are still labeled a complete snapshot. [ASSUMED]

**How to avoid:** bind cursors to store/schema/query/authority revision and reject resume after revision changes. SQLite snapshot isolation only lasts for the read transaction that holds it. [CITED: https://www.sqlite.org/isolation.html]

### Pitfall 4: Schema Defaults Rewrite History

**What goes wrong:** old entries are mutated during reads or a missing field becomes indistinguishable from an explicitly stored null/default. [ASSUMED]

**How to avoid:** model `MISSING` explicitly, define stored versus effective values, validate only new write/update, and reserve migrations for explicit offline work. [ASSUMED]

### Pitfall 5: Capability Booleans Overclaim a Pairing

**What goes wrong:** an authority reports durable/CAS while the payload backend is process-local or not shared across hosts; callers infer impossible topology guarantees. [ASSUMED]

**How to avoid:** compute the public report from every active participant and state its scope. The ADR's exact topologies distinguish one-process memory, one-host SQLite/local filesystem, and future multi-host PostgreSQL/shared blobs. [VERIFIED: docs/adr/0001-topology-specific-storage-guarantees.md:70-81]

### Pitfall 6: Registry Nominal-Type Trap

**What goes wrong:** a valid custom backend is rejected or a builtin bypasses validation because two unrelated `MetadataBackend` ABCs share a name. [VERIFIED: src/cacheness/metadata.py:269-382; src/cacheness/storage/backends/base.py:12-159; src/cacheness/storage/backends/__init__.py:99-170]

**How to avoid:** register explicit role descriptors with structural protocols/capabilities; keep legacy registration functions as adapters and test builtins and third-party subclasses through identical resolution. [ASSUMED]

### Pitfall 7: Compatibility Surface Changes Shape

**What goes wrong:** no-schema `get_metadata()` or `list(metadata_filter=...)` changes its dictionary/return behavior, or new receipt fields break equality/serialization. [ASSUMED]

**How to avoid:** preserve current mapping and entry methods exactly; add typed query alongside them and implement legacy equality filtering through the canonical scan/query path. Existing `BlobEntryInfo` exact fields are `"key"`, `"generation"`, `"locator"`, `"expectation"`, `"metadata"`, `"previous_locator"`. [VERIFIED: src/cacheness/storage/read_contract.py:30-42]

### Pitfall 8: Projection Failure Is Reported as Rollback

**What goes wrong:** the application retries a committed write and creates another generation because an external ORM/index refresh failed. [ASSUMED]

**How to avoid:** include commit receipt in the typed partial outcome and separate `committed=True` from remaining derived work; best-effort paths warn instead. [ASSUMED]

## Finite Failure Matrix

| Boundary | Required outcome |
|----------|------------------|
| Invalid declared type, required/default constraint, unknown query field/operator, heterogeneous/unbounded membership, or excess page size | Typed validation error before handler staging or backend query. [ASSUMED] |
| Undeclared metadata with schema | Preserve and authenticate as opaque metadata; do not query/index or reject it. |
| Metadata patch races another generation | Exact lifecycle conflict; old committed generation remains authoritative. [VERIFIED: src/cacheness/storage/lifecycle.py:328-382] |
| SQLite fault before commit while descriptor/catalog/index rows change | Roll back every participating row and keep previous generation. [VERIFIED: src/cacheness/storage/sqlite_lifecycle_authority.py:1030-1105] |
| Commit acknowledgement uncertain | Reopen/classify the exact operation and return committed receipt or typed uncertain/backend outcome; never blind retry. [VERIFIED: src/cacheness/storage/sqlite_lifecycle_authority.py:1291-1298] |
| Cursor malformed/tampered/wrong version/store/schema/query | Typed invalid-cursor failure with no query side effects. [ASSUMED] |
| Authority revision changed during paginated snapshot | Typed retryable stale-cursor/restart; no successful incomplete result. [ASSUMED] |
| Same-transaction index absent/stale/corrupt | Canonical scan or typed index/rebuild failure; never stale success. [ASSUMED] |
| Projection unavailable after canonical commit | Receipt remains committed; best effort warns, explicit refresh returns typed committed-partial with remaining work. |
| Projection retry/duplicate page | Idempotent convergence and monotonic checkpoint. [ASSUMED] |
| Rebuild interruption | Old projection remains published; isolated candidate is resumable or discardable. [ASSUMED] |
| Name + instance, options + instance, missing role capability, invalid pairing | Construction error before initialization or payload mutation. |
| Close of constructed vs injected resource | Constructed/ownership-transferred closes exactly once; caller-owned injected remains open on close and construction failure. [VERIFIED: src/cacheness/storage/blob_store.py:134-154,286-301,454-484] |
| Established schema/layout incompatible | `"blob_migration_required"`; ordinary reads do not rewrite. [VERIFIED: src/cacheness/error_handling.py:52-64; src/cacheness/storage/sqlite_lifecycle_authority.py:802-854] |

## Code Examples

These are planning skeletons, not locked names. All new class names and enum values below are recommendations and therefore `[ASSUMED]`.

### Native Schema and Typed Query

```python
# [ASSUMED] Recommended public shape; names are discretionary.
schema = CatalogSchema(
    name="documents",
    version=1,
    fields=(
        CatalogField("tenant", CatalogType.STRING, required=True, queryable=True, indexed=True),
        CatalogField("priority", CatalogType.INTEGER, queryable=True),
        CatalogField("archived", CatalogType.BOOLEAN, default=False, queryable=True),
    ),
)

store = BlobStore(config=StoreConfig(catalog_schema=schema, ...))
receipt = store.put_entry(payload, key="doc-1", metadata={"tenant": "acme", "priority": 4})

page = store.query_catalog(
    CatalogQuery.all_of(
        Field("tenant").eq("acme"),
        Field("priority").between(3, 8),
        Field("archived").exists(),
    ),
    limit=100,
)
next_page = store.query_catalog(cursor=page.next_cursor, limit=100)
```

The current manifest explicitly permits `None`, `bool`, signed-64 `int`, and `str` and rejects floats; those exact existing value categories justify the recommended initial declared scalar envelope. [VERIFIED: src/cacheness/storage/manifest.py:92-163]

### Composition and Ownership

```python
# [ASSUMED] Exactly one of name/instance is permitted per role.
config = StoreConfig(
    payload=BackendRef(name="filesystem", options={"root": root}),
    authority=BackendRef(instance=authority, ownership=Ownership.CALLER),
    projections=(BackendRef(name="json", options={"path": export_path}),),
    require=RequiredCapabilities(durability="host", exact_cas=True, portable_query=True),
)
store = BlobStore(config=config)
assert store.components.authority is authority
```

### Narrow Projection Pull

```python
# [ASSUMED] Projection never receives lifecycle transaction handles.
report = projection.refresh(
    source=store.catalog_source(),
    checkpoint=projection.load_checkpoint(),
    page_size=128,
)
if report.remaining_work:
    raise ProjectionRefreshPartial(receipt=receipt, report=report)
```

### SQLite Keyset Page

```sql
-- Source: https://www.sqlite.org/rowvalue.html
SELECT key, generation, manifest, revision
FROM entries
WHERE (key, generation) > (?, ?)
ORDER BY key, generation
LIMIT ?;
```

Use bound parameters only. The authority must additionally check the cursor's authority revision/schema/query identity; the SQL fragment alone does not establish snapshot completeness. [ASSUMED]

## State of the Art

| Old / current approach | Phase 4 approach | Impact |
|------------------------|------------------|--------|
| `backend` chooses authority and compatibility projection through special cases. [VERIFIED: src/cacheness/storage/blob_store.py:215-244] | Typed role specs resolved by one composition root. [ASSUMED] | Exact injection, registry parity, and truthful ownership/capabilities. |
| Broad `LifecycleAuthority.list_entries()` returns an unbounded tuple. [VERIFIED: src/cacheness/storage/lifecycle_authority.py:279-366] | Bounded revision-bound catalog page, keyset cursor, finite typed predicates. [ASSUMED] | Portable complete queries and projection pull. |
| `list(metadata_filter=dict)` scans signed metadata and applies equality only. [VERIFIED: src/cacheness/storage/lifecycle.py:543-555] | Preserve legacy equality while adding declared equality/range/membership/existence query. | Compatibility plus explicit query guarantees. |
| JSON/SQLite/PostgreSQL/custom ORM metadata paths can look authority-like. [VERIFIED: src/cacheness/metadata.py:269-382; src/cacheness/storage/backends/postgresql_backend.py:336-490] | Explicit projection role unless it is the selected `LifecycleAuthority` and joins its transaction. | Removes ambiguous sources of truth. |
| Projection exporter reads SQLite-private tables/backup. [VERIFIED: src/cacheness/storage/manifest_repository.py:33-175] | Projection pulls bounded semantic catalog pages/checkpoints. [ASSUMED] | Adapter independence and future PostgreSQL authority compatibility. |
| Capability checks cover authority booleans only. [VERIFIED: src/cacheness/storage/blob_store.py:246-276] | Composed report covers payload + authority + projections and explicit scope. [ASSUMED] | No guarantee inflation from names. |

**Deprecated/outdated:** treat `create_metadata_backend()` and BlobStore's private selectors as compatibility adapters, not independent construction roots. [ASSUMED] Keep legacy functions callable, but route them through the one role registry/root where scope permits. [ASSUMED]

## Migration and Compatibility Plan Constraints

1. Preserve v1 top-level manifest shape. The exact v1 fields are `"byte_size"`, `"created_at"`, `"digest"`, `"digest_algorithm"`, `"generation"`, `"handler_metadata"`, `"handler_type"`, `"key"`, `"locator"`, `"payload_format"`, `"payload_format_version"`, `"schema_version"`, `"signature"`, `"signature_algorithm"`, `"state"`, and `"user_metadata"`; v1 rejects unknown or missing top-level fields. [VERIFIED: src/cacheness/storage/manifest.py:34-53,388-395]
2. Store schema identity/catalog material in an authority-owned, independently versioned extension/envelope without changing handler-owned payload bytes. [ASSUMED]
3. Fresh/empty stores may initialize the new acceleration extension explicitly. Existing established v1 stores must remain readable and may attach an active schema in canonical-scan mode without rewriting entries. Attaching a persisted transaction-maintained schema/index layout to populated v1 data must fail with typed migration-required evidence until Phase 7 performs an offline migration/rebuild. [ASSUMED]
4. Existing entries missing newly declared fields remain readable and are not rewritten. Query semantics must specify missing vs effective default. [ASSUMED]
5. Preserve `BlobEntryInfo`, no-schema metadata dictionary, constructor names, backend registry functions, and direct list equality behavior. [ASSUMED]
6. The current immutable metadata patch field set includes `"schema_version"`, `"key"`, `"cache_key"`, `"generation"`, `"state"`, `"locator"`, `"actual_path"`, `"handler_type"`, `"data_type"`, `"payload_format"`, `"payload_format_version"`, `"storage_format"`, `"digest_algorithm"`, `"digest"`, `"byte_size"`, `"file_size"`, `"created_at"`, `"handler_metadata"`, `"user_metadata"`, `"signature_algorithm"`, and `"signature"`; any new schema identity exposed in public metadata must also be structurally immutable. [VERIFIED: src/cacheness/storage/blob_store.py:65-72]
7. Do not revive `_put_legacy` or let `UnifiedCache`'s projection/custom-metadata hooks become a second canonical write. Complete UnifiedCache convergence remains Phase 6. [VERIFIED: src/cacheness/core.py:1218-1336]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Initial declared scalar set is string, signed-64 integer, boolean, nullable, with missing separate; no float/nested declared field. | Native Schema | Public API/schema migration cost. |
| A2 | Defaults materialize only on new writes; old reads expose stored missing separately from effective value. | Native Schema | Query/default semantics could surprise users. |
| A3 | Typed authoritative values use normalized rows keyed by key/generation/field and join promotion. | Authority Seam | SQLite layout and migration scope. |
| A4 | Cursor is HMAC-authenticated and binds store/schema/query/revision/last identity. | Query | Retry and key-management semantics. |
| A5 | Any authority revision change invalidates a resumed portable cursor. | Query | High-write workloads restart more often; alternative is durable query snapshot state. |
| A6 | Existing populated v1 stores can attach schema in canonical-scan mode, but cannot attach persisted transaction-maintained index layout until explicit Phase 7 migration. | Migration | Index acceleration may be delayed on old stores, but functionality remains available without hidden DDL. |
| A7 | Public receipt gains backward-compatible projection status, or an adjacent report wraps the exact receipt. | Projection | Dataclass equality/serialization compatibility. |
| A8 | Role registry uses structural protocols/descriptors rather than one existing nominal MetadataBackend ABC. | Composition | Extension API migration. |

## Open Questions

1. **Persisted layout without Phase 7 migration**
   - What we know: established authority schema version `1` rejects incompatible layouts and ordinary initialization promises no upgrades. [VERIFIED: src/cacheness/storage/sqlite_lifecycle_authority.py:53-66,802-854; src/cacheness/storage/blob_store.py:313-329]
   - What's unclear: whether Phase 4 may create a separately versioned optional catalog side-table set on a populated v1 store without calling that a migration.
   - Recommendation: treat persisted transaction-maintained index layout as an offline migration boundary; allow active schema plus authenticated canonical scans on existing stores, and preserve old no-schema reads. [ASSUMED]

2. **Default query semantics**
   - What we know: old missing fields must remain readable with explicit missing/default semantics.
   - What's unclear: whether equality with a default should match absent old rows.
   - Recommendation: portable predicates operate on stored values; expose an explicit effective-value projection if needed, but do not make `exists()` true for absent data. [ASSUMED]

3. **Receipt compatibility shape**
   - What we know: projection status must be visible and existing `BlobEntryInfo` is a frozen exact-generation receipt. [VERIFIED: src/cacheness/storage/read_contract.py:30-42]
   - What's unclear: add a defaulted field or wrap it in a new result type.
   - Recommendation: keep `put_entry()` returning `BlobEntryInfo` with a defaulted immutable projection summary; explicit refresh uses a separate report/exception carrying that receipt. [ASSUMED]

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| `uv` | locked test/runtime commands | ✓ | `0.12.9` (local probe) | — |
| Python | core implementation | ✓ | ambient `3.12.1`; repo dev pin `3.13`; supported `>=3.11` | Use `uv` managed interpreters for matrix. [VERIFIED: pyproject.toml:9; .python-version:1] |
| stdlib SQLite | local authority/query tests | ✓ | `3.43.1` via ambient Python probe | Memory authority for unit parity only; not durable fallback. |
| SQLAlchemy/psycopg/PostgreSQL service | derived PostgreSQL/ORM projection tests | Optional | Declared in extras/lock; live service not required for Phase 4 core | Mock adapter contract; real service qualification is Phase 5. [VERIFIED: pyproject.toml:23-63] |

**Missing dependencies with no fallback:** None for core Phase 4 implementation. [ASSUMED]

**Missing dependencies with fallback:** a live PostgreSQL service is not needed to qualify Phase 4's derived-only projection contract; use mocked adapter behavior and defer real-service/topology qualification to Phase 5. [ASSUMED]

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest `>=8.4.1` [VERIFIED: pyproject.toml:71-91] |
| Config file | `pyproject.toml` |
| Quick run command | `uv run --frozen pytest -q -o log_cli=false tests/test_catalog_schema.py tests/test_catalog_query_contract.py tests/test_blob_store_composition.py` [ASSUMED] |
| Full suite command | `uv run --frozen pytest -q -o log_cli=false` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BACK-02 | explicit authority/projection role for JSON, memory, SQLite, PostgreSQL; no projection authorizes lifecycle/query completeness | contract/integration | `uv run --frozen pytest -q tests/test_metadata_role_contract.py` | ❌ Wave 0 [ASSUMED] |
| BACK-03 | exact injected instance, registered-name parity, ambiguous selector failure, ownership on close/init failure | unit/contract | `uv run --frozen pytest -q tests/test_blob_store_composition.py` | ❌ Wave 0 [ASSUMED] |
| BACK-06 | actual paired capabilities and construction-time minimum rejection | unit/contract | `uv run --frozen pytest -q tests/test_topology_capabilities.py` | ❌ Wave 0 [ASSUMED] |
| BACK-07 | schema validation, evolution, atomic metadata update, typed predicates, revision cursor, projection partial/rebuild | unit/integration/fault | `uv run --frozen pytest -q tests/test_catalog_schema.py tests/test_catalog_query_contract.py tests/test_catalog_projection.py` | ❌ Wave 0 [ASSUMED] |

### Required Contract Suites

- Parameterize the existing common authority contract for memory and SQLite; add catalog mutation/page parity and prove memory's one-lock transition and SQLite's one transaction produce identical semantic results. Existing transition parity is in `tests/test_lifecycle_authority_contract.py`. [VERIFIED: tests/test_lifecycle_authority_contract.py:311-508]
- Extend SQLite fault boundaries so a failure before/after every catalog row/index update rolls back descriptor + catalog together; retain uncertain-commit reopen classification. [ASSUMED]
- Test query validation before backend dispatch, every finite operator/type, missing/null/default, AND semantics, limits, deterministic `(key,generation)` order, tampered cursor, store/schema/query mismatch, and revision change between pages. [ASSUMED]
- Corrupt/delete/stale every semantic index and prove either complete canonical scan or typed failure, never incomplete success. [ASSUMED]
- Test projection duplicate pages, interrupted checkpoints, failure after canonical commit, explicit partial receipt, and isolated rebuild publication. [ASSUMED]
- Reuse current projection race tests as derived behavior only; do not promote legacy projection CAS to authority acceptance. [VERIFIED: tests/test_projection_mutation_contract.py:63-275; tests/test_projection_sql_atomicity.py:120-275]
- Keep no-schema read shape, `None` vs absence, stale metadata patch, close ownership, JSON corruption irrelevance, and legacy list equality regressions. [VERIFIED: tests/test_blob_store_read_contract.py:691-987,1195-1280; tests/test_phase3_local_workflows.py:116-130,214-268]
- Add Python 3.11 and 3.13 matrix evidence for public dataclasses/enums/cursor serialization. [ASSUMED]

### Sampling Rate

- **Per task commit:** targeted new file plus closest existing authority/read-contract file, under 30 seconds. [ASSUMED]
- **Per wave merge:** `uv run --frozen pytest -q -o log_cli=false` and `uv run ruff check src tests`. [VERIFIED: AGENTS.md:146-157]
- **Phase gate:** full suite green; all new role/capability/query/fault matrices green; no known baseline regression attributed to Phase 4. [ASSUMED]

### Wave 0 Gaps

- [ ] `tests/test_catalog_schema.py` — schema validation/evolution/missing/default. [ASSUMED]
- [ ] `tests/test_catalog_query_contract.py` — portable predicate and cursor contract across memory/SQLite. [ASSUMED]
- [ ] `tests/test_blob_store_composition.py` — one-root injection/registry/ownership behavior. [ASSUMED]
- [ ] `tests/test_topology_capabilities.py` — participant + composed semantic reports. [ASSUMED]
- [ ] `tests/test_catalog_projection.py` — pull/checkpoint/partial/rebuild behavior. [ASSUMED]
- [ ] Shared fixtures for schema, bounded query pages, fault injector, fake registered roles, and caller-owned resources. [ASSUMED]

## Security Domain

Security enforcement and ASVS Level 1 are enabled in `.planning/config.json`. [VERIFIED: .planning/config.json:45-48]

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | Library does not authenticate principals; store key authenticates data, not users. [ASSUMED] |
| V3 Session Management | no | No application session lifecycle. [ASSUMED] |
| V4 Access Control | limited | Caller owns filesystem/database credentials; composition must not convert a derived adapter into authority. [ASSUMED] |
| V5 Input Validation | yes | Native finite schema/query values, identifier/size/depth bounds, cursor authentication, and bound SQL parameters. [ASSUMED] |
| V6 Cryptography | yes | Reuse existing HMAC-SHA256 manifest key/signing primitive; no custom cipher or independent secret format. Exact existing algorithm is `"hmac-sha256"`. [VERIFIED: src/cacheness/storage/manifest.py:22-32] |

### Known Threat Patterns for This Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Raw field/operator reaches SQL | Tampering | Parse finite AST, allow only declared fields/operators, bind every operand; never interpolate identifiers from the request. [ASSUMED] |
| Cursor tampering/replay across store/schema/query | Tampering / spoofing | Bound canonical envelope, HMAC, version/store/schema/query/revision checks. [ASSUMED] |
| Boolean/integer/null/missing type confusion | Tampering | Exact type checks before persistence/query; separate `MISSING`; no coercion. [ASSUMED] |
| Oversized predicate, `IN`, cursor, string, or page | Denial of service | Explicit caps aligned with manifest limits; reject before backend access. Current manifest exact bounds are `MAX_MANIFEST_BYTES = 1_048_576`, `MAX_NESTING_DEPTH = 16`, `MAX_COLLECTION_ITEMS = 4_096`, `MAX_TOTAL_NODES = 16_384`, and `MAX_STRING_UTF8_BYTES = 262_144`. [VERIFIED: src/cacheness/storage/manifest.py:22-32] |
| Derived index/ORM impersonates authority | Elevation of privilege / tampering | Authority-only query/read/delete/recovery; projections receive bounded snapshots and no lifecycle transaction handle. [ASSUMED] |
| Malformed catalog/index row | Tampering | Digest/schema/type corroboration and fail closed; never skip malformed canonical rows and call result complete. [ASSUMED] |
| Injected shared backend closed unexpectedly | Denial of service | Explicit ownership; caller-owned default; close exactly once only for owned/transfer. [ASSUMED] |

## Sources

### Primary (HIGH confidence repository evidence)

- `docs/adr/0001-topology-specific-storage-guarantees.md` — sole authority, topology matrix, guarantee vocabulary, stop conditions.
- `src/cacheness/storage/lifecycle_authority.py` — current authority semantic protocol, snapshot and capability types.
- `src/cacheness/storage/sqlite_lifecycle_authority.py` and `memory_lifecycle_authority.py` — actual transaction/lock visibility boundaries.
- `src/cacheness/storage/lifecycle.py`, `blob_store.py`, `read_contract.py`, and `manifest.py` — Phase 3 entry/lifecycle/integrity compatibility surface.
- `src/cacheness/metadata.py`, `storage/backends/`, `config.py`, `core.py`, and `custom_metadata.py` — fragmented factories, projections, duplicate ABCs, legacy facade seams.
- Existing authority/read/projection/query/close tests cited above — executable contract evidence.

### Secondary (MEDIUM confidence official documentation)

- [SQLite isolation](https://www.sqlite.org/isolation.html) — committed visibility and snapshot isolation.
- [SQLite transactions](https://www.sqlite.org/lang_transaction.html) — read/write transaction behavior and `BEGIN IMMEDIATE`.
- [SQLite row values](https://www.sqlite.org/rowvalue.html) — keyset comparison and offset cost.
- [SQLite JSON functions](https://www.sqlite.org/json1.html) — JSON type/missing behavior.
- [SQLite expression indexes](https://www.sqlite.org/expridx.html) — expression-index matching constraints.
- [Python typing Protocol](https://docs.python.org/3/library/typing.html#typing.Protocol) and [dataclasses](https://docs.python.org/3/library/dataclasses.html) — structural implementation contracts and frozen value objects; neither defines the canonical user schema.

### Tertiary (LOW confidence)

- None. Unverified recommendations are marked `[ASSUMED]` and enumerated in the Assumptions Log.

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — no new dependency; versions and runtime facilities verified from project manifest/local probes.
- Architecture: HIGH for existing seams and forbidden boundaries; MEDIUM for recommended names/layout.
- Pitfalls: HIGH where grounded in current double factories/ABCs/transaction hooks; MEDIUM for future cursor/default policy.
- Validation: HIGH for reusable suites; MEDIUM for proposed Wave 0 filenames and exact API assertions.

**Research date:** 2026-09-07
**Valid until:** 2026-10-07 for repository architecture; re-check after any Phase 4 composition or authority-schema commit.
