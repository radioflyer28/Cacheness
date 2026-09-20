# Phase 4: Metadata Composition and Topology Contracts - Research

**Researched:** 2026-09-07
**Domain:** authenticated catalog metadata, portable queries, backend composition, and topology capability contracts
**Confidence:** HIGH for repository architecture; MEDIUM for the recommended new public vocabulary

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

### Catalog Schema

- **D-01:** An application metadata mapping remains the default surface. Callers may optionally attach a declared catalog schema; a schema is not required merely to store and reopen current-layout metadata. This preserves the useful data model, not the pre-Phase-4 implementation or constructor compatibility surface.
- **D-02:** When a schema is present, undeclared metadata fields are preserved and round-trip as opaque values. Only declared fields receive validation and portable query/index guarantees. They must not be silently dropped or retroactively rejected.
- **D-03:** Phase 4 provides a small Cacheness-native declarative schema. Field definitions cover name, supported value type, required/default behavior, validation, and query/index intent. The native contract may expose an adapter seam for external model libraries later, but Pydantic, dataclasses, SQLAlchemy models, or another framework do not define the canonical Phase 4 API. — **Reversibility:** costly — Applications and backend adapters will persist and consume these declarations, so replacing the public schema vocabulary would require compatibility adapters and schema migration.
- **D-04:** Within the new supported catalog format, additive schema evolution is readable using explicit missing/default semantics; incompatible future changes require explicit offline migration. Pre-Phase-4 development layouts are not required to reopen through runtime compatibility code and may return typed migration/rebuild-required evidence. Reads never rewrite metadata as an incidental schema upgrade. — **Reversibility:** costly — Future published schemas will depend on this explicit version/migration boundary even though the current pre-production cutover is allowed to break old layouts.

### Portable Query and Index Contract

- **D-05:** The initial portable query language supports typed equality, comparison/range, membership, and existence predicates over declared queryable fields. Multiple predicates compose with AND. Nested OR/NOT expressions and backend-native query strings are outside the portable contract.
- **D-06:** Portable pagination uses a stable opaque cursor with deterministic canonical ordering by entry key and generation identity plus bounded page sizes. Arbitrary field sorting is capability-specific and is not implied by the portable query contract. Offset pagination is not the resumability contract.
- **D-07:** Queryability and index intent are explicit parts of each declared field. Backends do not silently index every field or choose indexes as an undocumented semantic behavior. Secondary indexes are rebuildable acceleration state; they never become canonical read, delete, cleanup, or repair authority.
- **D-08:** A successful portable query returns results complete for the canonical catalog snapshot promised by that backend. It may use an index maintained in the same catalog transaction or a canonical scan. A stale or independently updated external index cannot silently answer a portable query with incomplete results. — **Reversibility:** costly — Callers may rely on catalog queries for inventory and administrative workflows, so weakening completeness later would invalidate published behavior.

### Backend Composition and Capabilities

- **D-09:** One typed store configuration is the only primary composition root. It selects the payload backend, catalog authority, optional projections, and minimum required capabilities as one validated topology. Overlapping pre-production constructors, backend overloads, duplicate factories, and legacy configuration names may be removed rather than preserved as adapters. — **Reversibility:** costly — This becomes the shared construction contract for direct `BlobStore` use and later `UnifiedCache` composition.
- **D-10:** A caller-injected backend instance remains the exact selected instance. Registered backend names resolve through the same construction path as built-ins. Supplying both an instance and a name for one role is an error; no selector silently wins and options are not merged into caller-owned instances.
- **D-11:** Every composed store exposes the actual semantic capabilities of its active payload/authority/projection pairing. Callers may request minimum guarantees; unmet requirements and inherently invalid pairings fail during construction. Callers are not required to choose a named topology tier, and Cacheness does not infer a stronger promise for them to depend upon implicitly.
- **D-12:** Resource ownership is explicit. Backends created by the composition root are store-owned and closed with the store. Injected instances are caller-owned by default and remain open, with an explicit option to transfer ownership. Hidden reference counting or resource sharing is not introduced. — **Reversibility:** costly — Close behavior is observable and shared injected resources depend on it.

### Derived Projections and External ORM Integration

- **D-13:** Phase 4 external ORM models and external indexes are derived-only integrations. Authoritative custom fields remain in the native catalog transaction with the blob descriptor. An external model cannot replace the catalog authority or participate in lifecycle sequencing in this phase.
- **D-14:** Derived projections synchronize by idempotently pulling bounded pages of committed canonical catalog state from a checkpoint. Notifications may prompt a refresh but are not the correctness source. Phase 4 does not require a background worker, synchronous-callback-only delivery, or exactly-once event protocol.
- **D-15:** Projection failure cannot roll back or revoke an already committed blob. The committed receipt exposes projection status. Ordinary best-effort refresh may warn; when a caller explicitly requests projection refresh, failure produces a typed committed-partial outcome that retains the canonical receipt and attribution of remaining derived work. — **Reversibility:** costly — This is the public failure boundary that prevents applications from treating a committed generation as absent.
- **D-16:** Projection rebuild is an explicit, capability-qualified operation. It builds isolated derived state from canonical catalog state and publishes that state only when complete. Each adapter reports whether it can catch up online or requires offline maintenance; Phase 4 promises neither mode universally. Normal reads and writes do not perform hidden projection repair.

### Pre-Production Compatibility Reset

- **D-17:** Cacheness has no production deployment to preserve. Phase 4 should delete superseded selection, metadata-backend, and catalog-layout paths when the new composition/catalog contract replaces them, rather than maintaining dual old/new behavior. Historical characterization tests remain evidence but may be retired or rewritten when they assert removed APIs. — **Source:** explicit user direction during Phase 4 planning, 2026-09-07. — **Reversibility:** one-way — Reintroducing the removed pre-production surface later would create a new compatibility contract and duplicate composition paths.
- **D-18:** Dropping current backward compatibility does not drop migration infrastructure. Persisted formats and schemas remain explicitly versioned; unsupported layouts fail without mutation; Phase 7 still delivers non-mutating inventory, offline migration, resumable copy-verify-switch, and confirmed rebuild tooling for future released versions. — **Source:** explicit user direction during Phase 4 planning, 2026-09-07.

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
- Stored-format and schema migration execution belongs to Phase 7. The tooling remains
  required for future released versions, but the supported source-version window may
  exclude pre-Phase-4 development formats.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BACK-02 | JSON, memory, SQLite, and PostgreSQL metadata implementations have explicit authority/projection roles through one composition contract. Narrow transactional catalog adapters before expansion; a derived JSON view does not become an independent lifecycle authority. | Role table, one-root construction, and projection pull protocol below. |
| BACK-03 | Caller-injected and registered backend implementations remain selected rather than being silently replaced by configuration defaults. | Exact-instance, one-registry construction, ownership, and superseded-selector removal tests below. |
| BACK-06 | Backends expose durability, process/host sharing, compare-and-swap, streaming, and listing capabilities, and configurations cannot claim guarantees their topology cannot provide. | Semantic participant and composed-capability model below. |
| BACK-07 | Direct `BlobStore` users can store, validate, query, and update application-defined catalog metadata without implementing a lifecycle backend; supported fields/operators and transactional limits are explicit. Extend the existing mapping/entry interface rather than replacing the engine. Authoritative metadata commits with the blob descriptor; external indexes or ORM links are explicitly derived unless they join that same transaction, with consistency and partial-failure behavior stated. | Native schema, typed query, same-transaction authority extension, receipt, and failure matrix below. |
</phase_requirements>

## Summary

Phase 4 should extend the Phase 3 authority visibility switch, not add another coordinator. The existing `AuthorityLifecycleEngine` already stages immutable payloads, signs authenticated `user_metadata`, and promotes one descriptor through the selected `LifecycleAuthority`; `BlobStore` already exposes `put_entry`, `get_entry_info`, `open_entry`, `get_metadata`, `update_metadata`, and `list`. [VERIFIED: src/cacheness/storage/blob_store.py:331-419] SQLite promotion updates the entry, lineage, mutation state, cleanup debt, authority revision, and projection-dirty bit within the existing `_transaction()` boundary. [VERIFIED: src/cacheness/storage/sqlite_lifecycle_authority.py:1222-1298] Authoritative declared catalog values and any semantic index must join that exact promotion transaction (and the memory authority's existing lock transition), or they become a second visibility switch. [VERIFIED: docs/adr/0001-topology-specific-storage-guarantees.md:36-44,83-100]

The current composition surface is fragmented: `BlobStore` overloads `backend` to choose both an authority and a projection, the backend registry has a different path, and `create_metadata_backend()` is a third hard-coded factory. [VERIFIED: src/cacheness/storage/blob_store.py:215-244; src/cacheness/storage/backends/__init__.py:99-235; src/cacheness/metadata.py:3275-3357] Worse, built-in JSON/SQLite/memory classes inherit the `MetadataBackend` ABC in `metadata.py`, while the registry validates against a different `MetadataBackend` ABC in `storage/backends/base.py`. [VERIFIED: src/cacheness/metadata.py:269-382; src/cacheness/storage/backends/base.py:12-159; src/cacheness/storage/backends/__init__.py:127-170] Plan a role-specific structural contract, one registry, and one typed root, then delete the overlapping selectors/factories and tests that exist only to preserve them. [VERIFIED: .planning/phases/04-metadata-composition-and-topology-contracts/04-CONTEXT.md:43-58]

The portable query must be a bounded authority query whose successful result is complete for one revision-bound canonical snapshot. SQLite supports efficient keyset paging using row-value comparison and guarantees that a read transaction sees an unchanging snapshot; because a cursor is resumed across calls, bind it to the authority revision and return a typed stale-cursor/restart outcome if the revision changed. [CITED: https://www.sqlite.org/rowvalue.html] [CITED: https://www.sqlite.org/isolation.html] External ORM/JSON/PostgreSQL metadata implementations remain lagging, rebuildable projections and may never authorize reads, deletes, cleanup, reconciliation, or query completeness. [VERIFIED: docs/adr/0001-topology-specific-storage-guarantees.md:139-171]

**Primary recommendation:** define the new supported catalog format and a single typed `BlobStore` composition API first, extend the existing authority transaction behind it, and remove pre-production constructors, backend overloads, duplicate factories, runtime metadata hooks, list-filter APIs, and stored-layout shims that would create a second path. Unsupported development layouts fail unchanged with typed migration/rebuild-required evidence. [VERIFIED: .planning/phases/04-metadata-composition-and-topology-contracts/04-CONTEXT.md:43-58,136-152]

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Schema declaration and validation | BlobStore storage API | Catalog authority | BlobStore validates before payload side effects; the authority records schema identity and declared values at promotion. [ASSUMED] |
| Canonical catalog commit | Catalog authority | BlobStore lifecycle engine | The authority transaction is the sole visibility switch; the lifecycle engine retains sequencing. [VERIFIED: docs/adr/0001-topology-specific-storage-guarantees.md:92-97,141-147] |
| Portable query and cursor | Catalog authority | BlobStore storage API | Authority supplies a complete bounded page; BlobStore authenticates public results and hides adapter details. [ASSUMED] |
| Payload bytes | Payload backend / guarded handler I/O | BlobStore lifecycle engine | Payloads are immutable generations published before authority promotion; cross-resource ACID is not promised. [VERIFIED: docs/adr/0001-topology-specific-storage-guarantees.md:36-40,88-97] |
| Derived ORM/index/JSON views | Projection adapter | External database or file | They pull committed canonical pages, checkpoint, and rebuild; they are never authority. [ASSUMED] |
| Topology selection and ownership | Composition root | Role registry | One root validates active instances and retains explicit close responsibility; no old constructor/factory tier remains. [ASSUMED] |
| Cache TTL/eviction/statistics | UnifiedCache policy layer | BlobStore entry interface | Phase 4 must not absorb Phase 6 cache-policy migration or attempt the final coherent cache API. [VERIFIED: AGENTS.md:15-23; .planning/ROADMAP.md:266-271] |

## Planning Complexity and Recommended Decomposition

**Complexity: HIGH, but materially smaller than the compatibility-preserving design.** The atomic catalog/query/capability/projection work is still cross-cutting and must respect one visibility switch, but the reset removes the dual-path state space: no old constructor adapter mesh, old receipt/dictionary shape matrix, legacy list-filter branch, or pre-Phase-4 canonical-scan bridge. [VERIFIED: .planning/PROJECT.md:38-71; .planning/phases/04-metadata-composition-and-topology-contracts/04-CONTEXT.md:55-58]

Plan in five dependency-ordered slices, with Wave 0 tests first. [ASSUMED]

1. **Clean contract and version boundary:** native schema/query/result/error values, new manifest/catalog version, unsupported-layout detector, and negative API/source assertions. [ASSUMED]
2. **Single composition root:** role registry, exact instance/name selection, ownership, participant/composed capabilities, and deletion of duplicate selectors/factories/ABCs. [ASSUMED]
3. **Authority integration:** memory and SQLite catalog state join their existing atomic transitions; portable bounded query/cursor semantics; fault matrix. [ASSUMED]
4. **Derived projection contract:** bounded pull/checkpoint, explicit refresh partials, isolated rebuild, JSON/ORM/PostgreSQL-derived adapters, and removal of runtime session hooks. [ASSUMED]
5. **Public cutover and documentation:** expose only the clean BlobStore catalog/composition surface, retire superseded exports/tests, rebaseline full suite, and record future Phase 7 version/tooling obligations. [ASSUMED]

Do not parallelize slices 2 and 3 against the same authority/configuration files; composition types and the new persisted format must settle before backend integration. Projection adapter work can begin after the page/result contract is fixed, but publication into `BlobStore` waits for authority integration. [ASSUMED]

## Project Constraints (from AGENTS.md)

- Current development-only APIs and stored layouts may be replaced instead of receiving compatibility adapters. Unsupported current layouts must fail explicitly; explicit schema/format versions and offline migration/rebuild infrastructure remain mandatory for future releases. [VERIFIED: AGENTS.md:13-23]
- `BlobStore` owns storage lifecycle, `UnifiedCache` depends on it and owns cache policy, and `SqlCache` stays separate. [VERIFIED: AGENTS.md:17-17]
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
| `sqlite3` | runtime bundled; local probe `3.43.1` | qualified local catalog authority and bounded canonical descriptor scans | The existing SQLite authority uses bounded `BEGIN IMMEDIATE` transactions and is the Phase 3 canonical local authority. [VERIFIED: src/cacheness/storage/sqlite_lifecycle_authority.py:1030-1105] |
| Existing manifest/HMAC code | manifest schema `1`, payload format `1`, signature algorithm `"hmac-sha256"`, digest `"sha256"` | authenticate descriptor plus application metadata and cursor envelopes | Reuse the established integrity boundary and key provider; do not introduce an independent cursor secret. [VERIFIED: src/cacheness/storage/manifest.py:22-32] |
| pytest | `>=8.4.1` | authority-contract, fault-injection, composition, cutover, and version-rejection tests | Declared dev framework and existing tests already exercise transition rollback/uncertain commit. Historical compatibility tests are evidence, not mandatory API-retention gates. [VERIFIED: pyproject.toml:71-91; tests/test_lifecycle_authority_contract.py:311-508; .planning/phases/04-metadata-composition-and-topology-contracts/04-VALIDATION.md:62-67] |

### Supporting

| Library / facility | Version | Purpose | When to Use |
|--------------------|---------|---------|-------------|
| SQLAlchemy | optional `>=2.0.0` | external ORM derived projections only | Replace the old session/model hooks with projection adapters; do not expose models/sessions as the native catalog API. [VERIFIED: pyproject.toml:23,40,56; src/cacheness/custom_metadata.py:100-274] |
| psycopg | optional `>=3.1.0` | existing PostgreSQL metadata/projection integration | Keep derived in Phase 4; PostgreSQL lifecycle authority qualification is Phase 5. [VERIFIED: pyproject.toml:40; .planning/phases/04-metadata-composition-and-topology-contracts/04-CONTEXT.md:170-174] |
| Existing safe query validation | repository code | bound identifiers, depth, and signed-64 integers | Reuse as input-validation substrate, then add declared-field/type/operator checks. [VERIFIED: src/cacheness/query_validation.py:11-77] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Native Cacheness schema | Pydantic/dataclasses/SQLAlchemy models | Locked out as the canonical API; adapters may be added later. |
| Canonical signed descriptor scans | Normalized rows or SQLite JSON expression indexes | Phase 4 needs one correctness source, not a second consistency surface. Preserve declared index intent in the schema, expose acceleration as unsupported, and add derived acceleration only after measured evidence identifies a need. [ASSUMED; resolved during planning] |
| Revision-bound keyset cursor | Long-lived read transaction or offset | A long-lived connection burdens callers; offset work grows with the offset. Keyset plus revision makes resumability explicit. [CITED: https://www.sqlite.org/rowvalue.html] |
| Pull/checkpoint projection | Synchronous callback/event authority | Callbacks cannot be the correctness source and would create a second commit dependency. |
| One clean composition API | Adapters for every old constructor/factory/backend overload | Pre-production reset makes deletion safer and smaller; retaining shims would preserve the ambiguity Phase 4 exists to remove. [VERIFIED: .planning/phases/04-metadata-composition-and-topology-contracts/04-CONTEXT.md:43-58] |

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
          +--> committed CatalogCommitResult / projection status [ASSUMED]
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
├── blob_store.py              # clean public entry + catalog-query facade
├── memory_lifecycle_authority.py
├── sqlite_lifecycle_authority.py
└── projections.py             # pull/checkpoint/rebuild adapter contract [ASSUMED]
```

Names are within the agent's discretion; the important boundary is that `catalog.py` contains declarative/query values, authority implementations own transactions, and `lifecycle.py` remains the only sequencing engine. [ASSUMED]

### Pattern 1: Conservative Native Schema

Use immutable native declarations with: schema identity/version; field name; one of string, signed-64 integer, boolean, or explicit nullability; required/default behavior; serializable validation constraints; queryable flag; index intent. Keep missing distinct from stored `null`, reject coercion (`True` is not integer), and preserve undeclared values as opaque manifest metadata. [ASSUMED]

The finite set aligns with the current canonical manifest, which accepts `None`, `bool`, signed-64 `int`, `str`, lists, and mappings but explicitly rejects floating-point values. [VERIFIED: src/cacheness/storage/manifest.py:92-163] Recommended initial declared fields are scalar only; opaque undeclared metadata may retain bounded nested structures in the new format. [ASSUMED]

Use only serializable built-in constraints in the persisted schema fingerprint (for example integer bounds, string length, membership choices). Do not put arbitrary Python callables into canonical schema identity because other processes cannot deterministically reconstruct them. [ASSUMED]

Default semantics should be explicit: materialize defaults on new-format writes, expose a missing sentinel for entries created under additive future schema versions, and make existence predicates inspect stored presence. A read must not rewrite the descriptor or silently claim that an entry physically contains a default. [ASSUMED] Pre-Phase-4 layouts do not participate in these evolution semantics; reject them unchanged. [VERIFIED: .planning/phases/04-metadata-composition-and-topology-contracts/04-CONTEXT.md:29-32,55-58]

### Pattern 2: Narrow Catalog Authority Seam

Add semantic operations such as `catalog_page(query, cursor, limit)` and schema/capability inspection to the authority, plus declared values carried as part of the existing mutation/promotion data. Do not expose `sqlite3.Connection` or reproduce `prepare/record/promote/cleanup` on a catalog adapter. [ASSUMED]

SQLite should keep schema identity, declared values, stored presence, and index intent in the canonical signed descriptor state already promoted by the authority transaction. Portable Phase 4 queries use bounded canonical scans and authenticate results before exposure. Normalized rows and physical secondary indexes are deferred because they would duplicate authoritative state without a Phase 4 consumer. [ASSUMED; resolved during planning] The current promotion transaction's exact visibility state includes `entries`, `entry_lineage`, mutation state, cleanup debt, and `authority_state.revision`; a future accelerator must remain derived and must not become a second visibility switch. [VERIFIED: src/cacheness/storage/sqlite_lifecycle_authority.py:1222-1298]

The new supported manifest/catalog format should make authoritative application metadata and schema identity explicit authenticated state. Typed rows are query acceleration and must be rebuildable/corroborated from that authenticated descriptor; a query result is exposed only after the selected `EntrySnapshot.manifest` authenticates and agrees. [ASSUMED] The current exact v1 shape is useful evidence for a bounded/versioned parser, but the pre-production reset permits replacing it rather than designing the new catalog around its top-level field constraint. [VERIFIED: src/cacheness/storage/manifest.py:34-53,238-257,388-395; .planning/phases/04-metadata-composition-and-topology-contracts/04-CONTEXT.md:55-58]

Memory authority must perform the equivalent immutable snapshot/catalog update inside its existing single `_transition()` lock. [VERIFIED: src/cacheness/storage/memory_lifecycle_authority.py:41-80]

### Pattern 3: Revision-Bound Portable Query

Represent query input as value objects, never raw SQL: a bounded tuple of AND predicates, each containing declared field, finite operator, and already type-checked operand. [ASSUMED] Exact portable operators should be equality, `<`, `<=`, `>`, `>=`, inclusive range, membership, and existence. [ASSUMED] Reject unqueryable/unknown fields, heterogeneous membership lists, boolean-as-integer, excess predicates/items, invalid cursor, and excess page size before touching the backend. [ASSUMED]

Return a bounded page ordered by `(key, generation)` and an opaque authenticated cursor containing format version, store identity, schema fingerprint, query fingerprint, authority revision, last key, and last generation. [ASSUMED] SQLite row-value comparisons directly support the keyset shape `(key, generation) > (?, ?)` and avoid offset-proportional scanning. [CITED: https://www.sqlite.org/rowvalue.html]

At resume, reject a cursor if store/schema/query/version differs. If the authority revision differs, return a typed retryable stale-snapshot result instructing restart; never label a page sequence complete after mixing revisions. [ASSUMED] SQLite can hold a stable snapshot within one read transaction, but a cursor resumed later is normally a new transaction, so the explicit revision check is the portable contract. [CITED: https://www.sqlite.org/isolation.html] [CITED: https://www.sqlite.org/lang_transaction.html]

An index may answer only when transactionally current for the same authority revision and schema fingerprint. Otherwise scan the authenticated canonical descriptor set or return a typed capability/rebuild-needed failure; stale derived indexes may not return a successful incomplete page. [ASSUMED]

### Pattern 4: One Typed Composition Root

Define a role specification with exactly one of `name` or `instance`; options apply only to named construction. Track `owned=True` for constructed resources, `owned=False` for injected resources, and accept an explicit transfer flag. [ASSUMED] Resolve built-ins and registered names through the same role registry, then validate the actual constructed payload/authority/projection capabilities before the store can initialize or stage bytes. [ASSUMED]

Capability reports should be semantic and scoped, not backend-name booleans: durability scope, sharing scope (`process`, `host`, `multi_host`), exact CAS, transaction scope, immutable generations, streaming, listing, portable query, indexed paging, projection catch-up/rebuild mode, and initialization/migration state. [ASSUMED] The current `AuthorityCapabilities` exact fields are `durable`, `multiprocess`, `transactional`, `exact_cas`, `indexed_paging`, and `projection`; it lacks payload streaming/listing and host-sharing scope. [VERIFIED: src/cacheness/storage/lifecycle_authority.py:267-276]

Delete the overloaded `backend=` selector and the private `_select_projection_backend()` / `_create_lifecycle_authority()` construction path after the typed root owns all supported construction. [ASSUMED] Register distinct payload, catalog-authority, and projection roles so `memory`, `json`, `sqlite`, and `postgresql` cannot silently change meaning by call site. [ASSUMED] Do not infer PostgreSQL authority in Phase 4: the existing PostgreSQL class is a metadata/projection backend, not a `LifecycleAuthority`. [VERIFIED: src/cacheness/storage/backends/postgresql_backend.py:88-490] The clean API may reuse useful class names, but it must not keep the old overload merely to accept historical calls. [VERIFIED: .planning/phases/04-metadata-composition-and-topology-contracts/04-CONTEXT.md:43-58,136-152]

### Pattern 5: Pull-Based Derived Projection

Projection adapters accept bounded canonical pages plus an idempotency identity `(store, schema fingerprint, authority revision, key, generation)` and persist a checkpoint after applying the page. [ASSUMED] Notifications may only prompt a pull. A retry must converge without duplicate or stale ownership. [ASSUMED]

Ordinary best-effort refresh logs/records derived debt without revoking the commit. Explicit refresh raises a typed committed-partial exception/report carrying the new canonical commit receipt and remaining projection adapter/checkpoint work. [ASSUMED] The current `BlobEntryInfo` fields—`"key"`, `"generation"`, `"locator"`, `"expectation"`, `"metadata"`, and optional `"previous_locator"`—are evidence for the minimum semantic receipt data, not a shape-compatibility requirement; replace the type if a clean projection-status/commit result model is clearer. [VERIFIED: src/cacheness/storage/read_contract.py:30-42; .planning/phases/04-metadata-composition-and-topology-contracts/04-CONTEXT.md:52-58]

Rebuild into isolated state, validate completion against a canonical snapshot/checkpoint, then atomically publish only if the adapter advertises that operation. Preserve old derived state on failure. Report online/offline mode; do not hide repair in reads/writes. [ASSUMED]

### Anti-Patterns to Avoid

- **Second lifecycle coordinator:** never let a metadata adapter repeat prepare/promote/delete/cleanup. Extend the existing authority's narrow transaction/query semantics. [VERIFIED: docs/adr/0001-topology-specific-storage-guarantees.md:141-147]
- **Projection as authority:** legacy SQLite/PostgreSQL projection CAS, JSON files, ORM links, or object listings cannot authorize canonical read/delete/cleanup/repair. [VERIFIED: docs/adr/0001-topology-specific-storage-guarantees.md:92-97,148-152]
- **Two-step authoritative metadata:** do not commit descriptor then separately write authoritative fields/index rows. Join the authority promotion transaction. [ASSUMED]
- **Name-based guarantees:** `"postgresql"`, `"sqlite"`, or `"s3"` is not evidence of topology semantics; validate active instances and their pairing. [VERIFIED: docs/adr/0001-topology-specific-storage-guarantees.md:70-81]
- **Compatibility shim mesh:** do not route old constructors, `backend=` overloads, duplicate factories, runtime metadata sessions, or dictionary list filters into the new root. Delete/retire them so one supported path remains. [VERIFIED: .planning/phases/04-metadata-composition-and-topology-contracts/04-CONTEXT.md:43-58,136-152]
- **Silent selector precedence:** within the new role spec, do not allow defaults to overwrite injected instances or merge factory options into caller-owned objects. [VERIFIED: src/cacheness/core.py:181-284]
- **Unbounded/raw query:** no SQL strings, OR/NOT tree, offset resumability, unbounded `IN`, or unbounded page/list implementation in the new API. [ASSUMED]
- **Unversioned format replacement:** a pre-production layout may be replaced, but the new manifest/catalog format must have an explicit version and reject unsupported layouts without mutation. The current v1 parser demonstrates the fail-closed pattern. [VERIFIED: src/cacheness/storage/manifest.py:34-53,388-395; AGENTS.md:13-23]
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
| Schema/model framework | Pydantic/ORM canonical model | Cacheness-native declarations | Locked public contract; external models remain projections/adapters. |

**Key insight:** Phase 4 is a seam-narrowing phase. Most hard primitives already exist; correctness depends on joining catalog semantics to the current authority transaction and deleting selector ambiguity, not on adding a more powerful coordinator. [ASSUMED]

## Runtime State Inventory

This phase changes stored catalog/schema layout and composition semantics, so migration-sensitive runtime state was audited even though Phase 7 performs migrations. [ASSUMED]

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | Existing development SQLite authority is at `".cacheness/lifecycle-authority-v1.sqlite3"` with `SCHEMA_VERSION = 1`; current manifests have exact `schema_version = 1`/field shape. [VERIFIED: src/cacheness/storage/sqlite_lifecycle_authority.py:53-66; src/cacheness/storage/manifest.py:22-53] | Replace with the new explicitly versioned supported catalog format. Detect this pre-Phase-4 layout and return typed migration/rebuild-required evidence without opening it through canonical-scan or runtime-compatibility paths. Phase 7 tooling may exclude it from supported source versions. [VERIFIED: .planning/phases/04-metadata-composition-and-topology-contracts/04-CONTEXT.md:55-58,177-179] |
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

**What goes wrong:** entries written under an earlier supported additive schema version are mutated during reads or a missing field becomes indistinguishable from an explicitly stored null/default. [ASSUMED]

**How to avoid:** model `MISSING` explicitly, define stored versus effective values, validate only new write/update, and reserve migrations for explicit offline work. [ASSUMED]

### Pitfall 5: Capability Booleans Overclaim a Pairing

**What goes wrong:** an authority reports durable/CAS while the payload backend is process-local or not shared across hosts; callers infer impossible topology guarantees. [ASSUMED]

**How to avoid:** compute the public report from every active participant and state its scope. The ADR's exact topologies distinguish one-process memory, one-host SQLite/local filesystem, and future multi-host PostgreSQL/shared blobs. [VERIFIED: docs/adr/0001-topology-specific-storage-guarantees.md:70-81]

### Pitfall 6: Registry Nominal-Type Trap

**What goes wrong:** a valid custom backend is rejected or a builtin bypasses validation because two unrelated `MetadataBackend` ABCs share a name. [VERIFIED: src/cacheness/metadata.py:269-382; src/cacheness/storage/backends/base.py:12-159; src/cacheness/storage/backends/__init__.py:99-170]

**How to avoid:** define one explicit role registry with structural protocols/capabilities; delete or consolidate the two nominal ABC/factory systems and test builtins plus registered third-party implementations through the same construction call. [ASSUMED]

### Pitfall 7: Historical Compatibility Tests Freeze Superseded APIs

**What goes wrong:** characterization tests for `get_metadata()`, `list(metadata_filter=...)`, the overloaded constructor, registry factories, or `BlobEntryInfo` shape force shims into the new design even though no production caller depends on them. [VERIFIED: .planning/phases/04-metadata-composition-and-topology-contracts/04-CONTEXT.md:55-58]

**How to avoid:** retain only semantic invariants deliberately reaffirmed—same-generation receipts, exact expectations, committed-partial outcomes, one authority—and rewrite or retire tests that assert removed shapes. Add negative source/API tests proving superseded selectors and list-filter paths are absent. [VERIFIED: .planning/phases/04-metadata-composition-and-topology-contracts/04-VALIDATION.md:50-67]

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
| Pre-Phase-4 or future unsupported schema/layout | Typed migration/rebuild-required rejection; no canonical scan, runtime shim, implicit DDL, deletion, or rewrite. Current reason value is `"blob_migration_required"`, but the clean API may replace the exception shape while retaining explicit typed evidence. [VERIFIED: src/cacheness/error_handling.py:52-64; .planning/phases/04-metadata-composition-and-topology-contracts/04-CONTEXT.md:55-58] |

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
| `backend` chooses authority and projection through special cases. [VERIFIED: src/cacheness/storage/blob_store.py:215-244] | Delete overload; typed role specs resolved by one composition root. [ASSUMED] | Exact injection, one selector path, truthful ownership/capabilities. |
| Broad `LifecycleAuthority.list_entries()` returns an unbounded tuple. [VERIFIED: src/cacheness/storage/lifecycle_authority.py:279-366] | Bounded revision-bound catalog page, keyset cursor, finite typed predicates. [ASSUMED] | Portable complete queries and projection pull. |
| `list(metadata_filter=dict)` scans signed metadata and applies equality only. [VERIFIED: src/cacheness/storage/lifecycle.py:543-555] | Remove it; expose only the typed bounded equality/range/membership/existence query contract. [ASSUMED] | One query language and no unbounded/list-filter compatibility branch. |
| JSON/SQLite/PostgreSQL/custom ORM metadata paths can look authority-like. [VERIFIED: src/cacheness/metadata.py:269-382; src/cacheness/storage/backends/postgresql_backend.py:336-490] | Explicit projection role unless it is the selected `LifecycleAuthority` and joins its transaction. | Removes ambiguous sources of truth. |
| Projection exporter reads SQLite-private tables/backup. [VERIFIED: src/cacheness/storage/manifest_repository.py:33-175] | Projection pulls bounded semantic catalog pages/checkpoints. [ASSUMED] | Adapter independence and future PostgreSQL authority compatibility. |
| Capability checks cover authority booleans only. [VERIFIED: src/cacheness/storage/blob_store.py:246-276] | Composed report covers payload + authority + projections and explicit scope. [ASSUMED] | No guarantee inflation from names. |

**Remove during cutover:** `create_metadata_backend()`, the duplicate metadata registry/ABCs where superseded, BlobStore's private authority/projection selectors, the overloaded `backend=` argument, the dictionary `list(metadata_filter=...)` query, and legacy custom-metadata/SQLAlchemy session hooks. [ASSUMED] Do not keep callable shims solely because historical tests mention them. [VERIFIED: .planning/phases/04-metadata-composition-and-topology-contracts/04-CONTEXT.md:55-58,136-152]

## Versioning and Cutover Plan Constraints

1. Define a new explicit manifest/catalog schema version as the first supported release contract. The current development v1 exact fields are `"byte_size"`, `"created_at"`, `"digest"`, `"digest_algorithm"`, `"generation"`, `"handler_metadata"`, `"handler_type"`, `"key"`, `"locator"`, `"payload_format"`, `"payload_format_version"`, `"schema_version"`, `"signature"`, `"signature_algorithm"`, `"state"`, and `"user_metadata"`; use them as implementation evidence, not as a compatibility constraint. [VERIFIED: src/cacheness/storage/manifest.py:34-53,388-395]
2. Put authenticated schema identity and authoritative application catalog data in the new descriptor/authority contract; keep handler payload bytes independently versioned. [ASSUMED]
3. Detect pre-Phase-4 layouts before mutation and return a typed migration/rebuild-required result. Do not reopen them through canonical-scan mode, auto-create side tables, silently delete them, or add a runtime adapter. [VERIFIED: AGENTS.md:13-23; .planning/phases/04-metadata-composition-and-topology-contracts/04-CONTEXT.md:55-58]
4. Within the new supported format, additive future schema versions remain readable with explicit missing/default semantics; incompatible released versions require stopped-worker offline migration. Reads never rewrite. [VERIFIED: .planning/phases/04-metadata-composition-and-topology-contracts/04-CONTEXT.md:29-32]
5. Preserve semantic invariants, not development shapes: a commit result must identify the exact generation/expectation and projection status, but the `BlobEntryInfo` class/fields, old metadata dictionary API, constructor names, registry functions, and direct list equality behavior may be replaced. [VERIFIED: .planning/phases/04-metadata-composition-and-topology-contracts/04-CONTEXT.md:43-58,113-152]
6. The new descriptor must mark schema/catalog identity structurally immutable and authenticated. The current immutable patch set is historical evidence only. [VERIFIED: src/cacheness/storage/blob_store.py:65-72]
7. Delete `_put_legacy` and replace UnifiedCache's obsolete projection/custom-metadata entry points where Phase 4 owns the underlying storage seam; do not complete cache policy migration before Phase 6. [ASSUMED]
8. Preserve Phase 7's future-facing infrastructure requirements: non-mutating inventory, stopped-worker offline migration, resumable copy-verify-switch, signing-material preservation, and confirmed rebuild. Phase 7 need not support this pre-production source layout. [VERIFIED: .planning/REQUIREMENTS.md:52-60]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Initial declared scalar set is string, signed-64 integer, boolean, nullable, with missing separate; no float/nested declared field. | Native Schema | Public API/schema migration cost. |
| A2 | Defaults materialize only on new-format writes; reads of earlier supported additive schema versions expose stored missing separately from effective value. | Native Schema | Query/default semantics could surprise users. |
| A3 | Typed catalog values, presence, schema identity, and index intent remain in canonical signed descriptors; Phase 4 queries use bounded authenticated scans and advertise physical acceleration as unsupported. | Authority Seam | Avoids a second consistency surface; acceleration can be added later as a derived rebuildable projection. |
| A4 | Cursor is HMAC-authenticated and binds store/schema/query/revision/last identity. | Query | Retry and key-management semantics. |
| A5 | Any authority revision change invalidates a resumed portable cursor. | Query | High-write workloads restart more often; alternative is durable query snapshot state. |
| A6 | The new supported manifest/catalog format uses an identifier distinguishable from the development v1 layout. | Cutover | Reusing an ambiguous version could accidentally adopt unsupported data. |
| A7 | A new commit-result type carries exact generation/expectation and projection status; `BlobEntryInfo` need not survive. | Projection | Public API naming remains a planning decision. |
| A8 | Role registry uses structural protocols/descriptors rather than one existing nominal MetadataBackend ABC. | Composition | Extension API migration. |

## Resolved Planning Decisions

1. **First supported catalog/manifest version number — resolved:** use
   `STORE_FORMAT_VERSION = 2` as an explicit collision-avoidance marker for the
   unsupported development format-1 layout. Store epoch, manifest-schema
   version, SQLite `user_version`, and handler payload-format versions remain
   independently meaningful and do not all become `2` merely for symmetry.
   Existing format-1 evidence is classified before mutation and returns typed
   migration/rebuild-required evidence. [ASSUMED; approved during planning]

2. **Default query semantics — resolved:** materialize declared defaults on new
   writes. Portable predicates operate on stored presence and stored values, so
   an absent field does not equal its schema default and `exists()` remains
   false. Reads may expose stored-missing separately from the effective default
   for future additive schema revisions; Phase 4 does not add a second
   effective-value query language. [ASSUMED; approved during planning]

3. **New commit-result shape — resolved:** define one frozen `BlobReceipt`
   semantic result containing the canonical generation/expectation and the
   projection-delivery summary. Committed-partial failures retain that same
   receipt plus remaining work. Do not preserve or alias the development
   `BlobEntryInfo` return shape. [ASSUMED; approved during planning]

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
| BACK-03 | exact injected instance, registered-name parity, ambiguous selector failure, ownership on close/init failure, absence of superseded selectors/factories | unit/contract | `uv run --frozen pytest -q tests/test_blob_store_composition.py` | ❌ Wave 0 [ASSUMED] |
| BACK-06 | actual paired capabilities and construction-time minimum rejection | unit/contract | `uv run --frozen pytest -q tests/test_topology_capabilities.py` | ❌ Wave 0 [ASSUMED] |
| BACK-07 | schema validation, evolution, atomic metadata update, typed predicates, revision cursor, projection partial/rebuild | unit/integration/fault | `uv run --frozen pytest -q tests/test_catalog_schema.py tests/test_catalog_query_contract.py tests/test_catalog_projection.py` | ❌ Wave 0 [ASSUMED] |

### Required Contract Suites

- Parameterize the existing common authority contract for memory and SQLite; add catalog mutation/page parity and prove memory's one-lock transition and SQLite's one transaction produce identical semantic results. Existing transition parity is in `tests/test_lifecycle_authority_contract.py`. [VERIFIED: tests/test_lifecycle_authority_contract.py:311-508]
- Extend SQLite fault boundaries so a failure before/after every catalog row/index update rolls back descriptor + catalog together; retain uncertain-commit reopen classification. [ASSUMED]
- Test query validation before backend dispatch, every finite operator/type, missing/null/default, AND semantics, limits, deterministic `(key,generation)` order, tampered cursor, store/schema/query mismatch, and revision change between pages. [ASSUMED]
- Corrupt/delete/stale every semantic index and prove either complete canonical scan or typed failure, never incomplete success. [ASSUMED]
- Test projection duplicate pages, interrupted checkpoints, failure after canonical commit, explicit partial receipt, and isolated rebuild publication. [ASSUMED]
- Reuse current projection race tests as derived behavior only; do not promote legacy projection CAS to authority acceptance. [VERIFIED: tests/test_projection_mutation_contract.py:63-275; tests/test_projection_sql_atomicity.py:120-275]
- Replace historical shape tests with: current-new-layout no-schema mapping round trip, stored `None` versus absence, stale-patch conflict, exact ownership, projection-corruption irrelevance, non-mutating old-layout rejection, and explicit absence of the overloaded constructor/factories/runtime session/list-filter APIs. Historical tests may be rewritten or retired when they assert removed development surfaces. [VERIFIED: .planning/phases/04-metadata-composition-and-topology-contracts/04-VALIDATION.md:50-67]
- Add Python 3.11 and 3.13 matrix evidence for public dataclasses/enums/cursor serialization. [ASSUMED]

### Sampling Rate

- **Per task commit:** targeted new file plus closest existing authority/read-contract file, under 30 seconds. [ASSUMED]
- **Per wave merge:** `uv run --frozen pytest -q -o log_cli=false` and `uv run ruff check src tests`. [VERIFIED: AGENTS.md:146-157]
- **Phase gate:** after rewriting or retiring obsolete characterization tests, the full suite is green; all new role/capability/query/fault/cutover matrices are green; no known baseline regression is attributed to Phase 4. [ASSUMED]

### Wave 0 Gaps

- [ ] `tests/test_catalog_schema.py` — schema validation/evolution/missing/default, new-format reopen, and typed non-mutating rejection of the development layout. [ASSUMED]
- [ ] `tests/test_catalog_query_contract.py` — portable predicate and cursor contract across memory/SQLite. [ASSUMED]
- [ ] `tests/test_metadata_role_contract.py` — explicit authority/projection roles and no projection authorization. [ASSUMED]
- [ ] `tests/test_blob_store_composition.py` — one-root injection/registry/ownership behavior plus absence of superseded selectors/factories/API overloads. [ASSUMED]
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
- `src/cacheness/storage/lifecycle.py`, `blob_store.py`, `read_contract.py`, and `manifest.py` — Phase 3 entry/lifecycle/integrity semantics and development shapes to reassess during cutover.
- `src/cacheness/metadata.py`, `storage/backends/`, `config.py`, `core.py`, and `custom_metadata.py` — fragmented factories, projections, duplicate ABCs, and superseded facade seams targeted for consolidation/removal.
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
