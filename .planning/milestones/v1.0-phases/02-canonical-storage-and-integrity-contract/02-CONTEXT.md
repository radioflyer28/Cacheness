# Phase 2: Canonical Storage and Integrity Contract - Context

**Gathered:** 2026-08-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Define and implement the canonical record and fail-closed read contract for direct `BlobStore` callers. New writes receive one explicit manifest with independently versioned metadata and payload formats; reads expose only committed records and authenticate and verify them before deserialization. This phase defines the typed seam that `UnifiedCache` will consume later, but does not build the general CAS/recovery engine, backend-composition matrix, cache-policy delegation, or migration runner owned by later phases.

</domain>

<decisions>
## Implementation Decisions

### Canonical Manifest and Versioning
- **D-01:** Every new `BlobStore` write produces one canonical manifest-v1 record. It carries explicit manifest schema version, payload format identifier and version, logical key, opaque generation identifier, lifecycle state, physical locator, handler/type identifier, digest algorithm and value, byte size, and creation metadata. — **Reversibility:** one-way — Once persisted, changing required fields or their meaning requires a format migration and compatibility reader.
- **D-02:** The manifest is backend-neutral and deterministic. Metadata backends may encode it differently internally, but callers and lifecycle code operate on one typed model rather than backend-shaped dictionaries. — **Reversibility:** costly — Backend adapters and public error behavior will depend on this model.
- **D-03:** Native handlers continue to own payload containers and serialization. The manifest records the handler-owned format and version; Cacheness does not invent a new payload container or header.
- **D-04:** Unknown future manifest schemas or payload format versions fail with a typed unsupported-version error. They are never guessed, silently rewritten, treated as ordinary misses, or deleted.

### Direct Read Outcomes and Committed Visibility
- **D-05:** Preserve the compatible direct-read absence behavior: an actually missing key remains a normal miss (`None` where the existing API returns `None`). Corrupt data, authenticity failure, unsupported version, lifecycle conflict, and backend failure are distinct typed exceptions with stable reason codes; they must not collapse into absence. — **Reversibility:** costly — Direct callers will build recovery and observability around these distinctions.
- **D-06:** Normal reads expose only manifests explicitly marked committed. Any prepared, replacing, tombstoned, conflicted, or otherwise non-committed record is rejected before payload access with a typed lifecycle/conflict error. Phase 2 defines this visibility rule without claiming the Phase 3 CAS or recovery mechanism.
- **D-07:** The read order is fail-closed: bounded parse and version validation, manifest authenticity, critical-field validation, payload snapshot, cryptographic digest and size verification, then handler deserialization. No handler runs before every applicable check succeeds.

### Integrity and Signing
- **D-08:** Canonical payload integrity uses a cryptographic digest with an explicit algorithm identifier; SHA-256 is the v1 default. Existing XXH3 cache/content hashes may remain for non-security keying or compatibility, but cannot satisfy the security integrity contract.
- **D-09:** Manifest signing uses deterministic canonical bytes and an explicit algorithm identifier; HMAC-SHA256 is the v1 built-in. A signature binds schema/payload versions, logical key, generation, lifecycle state, locator, handler/type, payload format, digest algorithm/value, and byte size. Mutable cache-policy fields are not allowed to weaken this storage authenticity boundary. — **Reversibility:** one-way — Changing canonicalization or the signed field set requires versioned verification and migration support.
- **D-10:** When signing is required, missing keys, invalid key material, unsafe key permissions, unsupported signer configuration, absent signatures, and invalid signatures fail closed at the earliest reliable boundary. Required signing cannot silently downgrade to unsigned storage or reads.
- **D-11:** Integrity/authenticity proves bytes and provenance; it does not make pickle or dill safe for hostile payloads. Phase 1's trusted-application-payload boundary remains unchanged.

### Compatibility and Downstream Seams
- **D-12:** New writes use the canonical v1 manifest only. Exact Phase 1 legacy adapters remain read-only compatibility paths and attach explicit legacy format identities in memory; they never rewrite or delete data as a side effect of reading.
- **D-13:** Legacy or incompatible records that require conversion expose an inspectable typed migration-required/unsupported-version outcome. Actual inventory, copy-verify-switch migration, resumption, and rebuild confirmation remain Phase 7 work.
- **D-14:** `BlobStore` owns the typed integrity and lifecycle outcomes. Phase 2 provides a narrow translation seam so `UnifiedCache` can later turn selected integrity failures into separately counted cache misses, but Phase 2 does not rewire cache policy or change `UnifiedCache` miss behavior.

### the agent's Discretion
- Exact Python type/module names for the manifest model, enums, codecs, and result/error helpers.
- Deterministic canonical encoding details, provided tests prove byte stability, bounded parsing, and signed-field completeness.
- Whether compatible absence is represented internally by a result object before the public `BlobStore.get()` adapter returns `None`.
- The opaque generation identifier format, provided it is stable, signed, and does not claim Phase 3 compare-and-swap semantics.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Product and Requirements
- `.planning/PROJECT.md` — Establishes `BlobStore` as canonical lifecycle owner and `UnifiedCache` as policy layered above it.
- `.planning/ROADMAP.md` § Phase 2 — Defines the canonical manifest, committed-read, integrity, signing, unknown-version, and downstream seam success criteria.
- `.planning/REQUIREMENTS.md` — Phase 2 requirements: STOR-01, STOR-02, STOR-08, SECU-03, SECU-04, SECU-05, SECU-08, MIGR-02, and MIGR-07.

### Prior Locked Decisions
- `.planning/phases/01-compatibility-and-security-baseline/01-CONTEXT.md` — Preserves public compatibility, native handler format ownership, trusted-object safety, and deferred lifecycle boundaries.
- `.planning/phases/01-compatibility-and-security-baseline/01-VERIFICATION.md` — Records the verified Phase 1 boundary and explicitly deferred Phase 3/4/6 contracts.
- `docs/SECURITY.md` — Canonical serializer trust boundary and the distinction between authenticity/integrity and safe deserialization.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/cacheness/storage/blob_store.py`: direct storage facade and the correct ownership point for canonical manifests and typed read outcomes.
- `src/cacheness/security.py`: existing HMAC signer/key-management behavior to version and harden rather than duplicate.
- `src/cacheness/storage/guarded_handler_io.py` and `src/cacheness/storage/path_security.py`: private snapshots and contained I/O that can enforce verify-before-deserialize ordering.
- `src/cacheness/error_handling.py`: typed error hierarchy and stable `CacheReason` pattern established in Phase 1.
- `src/cacheness/handlers.py` and `src/cacheness/interfaces.py`: handler/type/format identities whose native payload formats remain authoritative.

### Established Patterns
- Metadata backends currently persist backend-shaped entry dictionaries; Phase 2 should introduce one typed manifest above those adapters without claiming Phase 4 backend parity.
- Phase 1 candidate publication and bounded clear recovery already distinguish pre-commit work from authoritative metadata, but remain narrow prerequisites rather than the general Phase 3 lifecycle engine.
- Guarded reads already take one private snapshot before integrity/signature checks and handler invocation; Phase 2 can make the canonical ordering and cryptographic contract explicit.

### Integration Points
- `BlobStore.put/get/get_metadata/list` must translate between the typed manifest model and compatible public results.
- `JsonBackend`, `SqliteBackend`, and `InMemoryBackend` are the local metadata persistence points exercised in this phase; PostgreSQL/general composition remains Phase 4/5 scope.
- `UnifiedCache` needs only the typed translation seam now; full delegation through `BlobStore` remains Phase 6.

</code_context>

<specifics>
## Specific Ideas

- The long-term architecture is a blob store that the cache system uses, not two overlapping payload/metadata lifecycles.
- NumPy, Blosc2, PyArrow, pickle, and other handlers own their native serialization/container operations; the canonical manifest describes those payloads rather than wrapping them in another Cacheness format.

</specifics>

<deferred>
## Deferred Ideas

- General old-or-new generation CAS, overwrite/delete races, crash recovery, idempotent close, and reconciliation — Phase 3.
- Caller-injected metadata retention and truthful backend capability/topology contracts — Phase 4.
- Full filesystem/memory/S3 by JSON/memory/SQLite/PostgreSQL matrix — Phase 5.
- `UnifiedCache` delegation, cache-policy miss translation, statistics, and invalidation convergence — Phase 6.
- Inventory, resumable copy-verify-switch migration, and confirmed rebuild — Phase 7.

</deferred>

---

*Phase: 2-canonical-storage-and-integrity-contract*
*Context gathered: 2026-08-30*
