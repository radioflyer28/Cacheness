# Phase 2: Canonical Storage and Integrity Contract - Research

**Researched:** 2026-08-30
**Domain:** Versioned storage manifests, authenticated metadata, payload integrity, and fail-closed direct reads
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

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

### Deferred Ideas (OUT OF SCOPE)
- General old-or-new generation CAS, overwrite/delete races, crash recovery, idempotent close, and reconciliation — Phase 3.
- Caller-injected metadata retention and truthful backend capability/topology contracts — Phase 4.
- Full filesystem/memory/S3 by JSON/memory/SQLite/PostgreSQL matrix — Phase 5.
- `UnifiedCache` delegation, cache-policy miss translation, statistics, and invalidation convergence — Phase 6.
- Inventory, resumable copy-verify-switch migration, and confirmed rebuild — Phase 7.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| STOR-01 | Every stored entry uses one versioned canonical manifest across all supported backends. | Typed `BlobManifestV1`, canonical codec, and a narrow JSON/memory/SQLite manifest repository are specified below. |
| STOR-02 | Normal reads expose only committed entry generations. | The read pipeline authenticates the manifest and rejects every state other than committed before opening a payload. |
| STOR-08 | Direct `BlobStore` operations distinguish missing, corrupt, conflict, and backend failures through typed results or exceptions. | A stable public error taxonomy and operation matrix are specified below. |
| SECU-03 | Configured manifest authenticity and payload integrity are verified before deserialization. | The exact ordered read pipeline and same-snapshot verification pattern are specified below. |
| SECU-04 | Required signing fails closed when keys, signatures, permissions, or signer configuration are missing or invalid. | A strict key-provider contract replaces the existing auto-regeneration/fallback behavior for canonical manifests. |
| SECU-05 | Signed manifests bind critical locator, handler/type, format, and lifecycle-generation fields. | The v1 signature covers the entire canonical manifest excluding only the signature value. |
| SECU-08 | `BlobStore` raises a typed integrity exception for corrupt or invalidly signed entries, while `UnifiedCache` may translate it into a separately recorded cache miss. | BlobStore-owned exception classes plus a pure translation classifier form the Phase 6 seam. |
| MIGR-02 | Stored metadata schemas and payload formats have explicit, independently versioned identifiers. | Manifest schema version and handler-owned payload format version are independent required fields. |
| MIGR-07 | Unknown future formats fail explicitly rather than being guessed, rewritten, or silently deleted. | Strict schema/format dispatch and exact read-only legacy identities are specified below. |
</phase_requirements>

## Summary

Phase 2 should introduce a typed manifest boundary above the existing metadata backends, not add more fields to the current backend-shaped dictionaries. Today `BlobStore.put()` persists `cache_key`, `data_type`, `file_size`, `created_at`, and a nested metadata dictionary; `get()` treats both a missing metadata record and a missing payload as `None`, then invokes the handler before any cryptographic verification. [VERIFIED: src/cacheness/storage/blob_store.py:180-319] The current SQLite implementation cannot preserve an arbitrary canonical manifest in that dictionary: its authoritative column set is exactly `"cache_key"`, `"description"`, `"data_type"`, `"prefix"`, `"created_at"`, `"accessed_at"`, `"file_size"`, `"file_hash"`, `"entry_signature"`, `"object_type"`, `"storage_format"`, `"serializer"`, `"compression_codec"`, `"actual_path"`, and `"cache_key_params"`, and `put_entry()` writes only those columns. [VERIFIED: src/cacheness/metadata.py:76-92,1595-1680]

The safest Phase 2 architecture is therefore a frozen `BlobManifestV1` model plus a strict codec and a narrow `CanonicalManifestRepository` used only by direct `BlobStore`. JSON and memory adapters can store canonical manifest documents, while SQLite should use a dedicated manifest table rather than changing the legacy `cache_entries` discriminator. [ASSUMED] That preserves the exact Phase 1 legacy readers and postpones generalized metadata backend composition, CAS, topology capability claims, and PostgreSQL parity to Phase 4.

Reads must be one explicit pipeline: fetch raw record; bounded-decode enough to identify schema; reject unknown schema; authenticate deterministic manifest bytes; validate critical fields, logical key, committed state, contained locator, handler identity, and payload version; open one private snapshot; stream SHA-256 and count bytes from that snapshot; deserialize from the same snapshot; update access bookkeeping only after success. The existing guarded snapshot already copies the managed file once and requires hashing, signature checks, and handler access to finish inside its context. [VERIFIED: src/cacheness/storage/guarded_handler_io.py:336-367]

**Primary recommendation:** Make one immutable, fully signed `BlobManifestV1` the only authoritative record for new direct `BlobStore` writes; keep exact legacy layouts as read-only adapters, and make every non-absence failure typed and non-mutating.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Canonical record model and codec | API / Backend | Database / Storage | Lifecycle code consumes one type; persistence adapters only encode/decode it. [ASSUMED] |
| Manifest authenticity and payload integrity | API / Backend | Database / Storage | `BlobStore` decides validation order; storage supplies raw metadata and one contained payload snapshot. [ASSUMED] |
| Native payload serialization | Database / Storage | API / Backend | Existing handlers own NPZ, Parquet, Blosc2, pickle, and dill containers; the manifest describes them. [VERIFIED: src/cacheness/handlers.py:573-634,705-728,790-824,973-1072] |
| Direct-read outcome taxonomy | API / Backend | — | Direct callers need stable absence/integrity/conflict/backend distinctions. [ASSUMED] |
| Legacy format recognition | API / Backend | Database / Storage | Exact adapters recognize historical metadata/payload identities without changing them. [VERIFIED: tests/fixtures/compat/validate_corpus.py:128-243] |
| Cache miss translation | API / Backend | — | Phase 2 defines a pure classifier; `UnifiedCache` wiring and statistics remain Phase 6. [ASSUMED] |

## Project Constraints (from AGENTS.md)

- Preserve supported public APIs and permit stored-data conversion or rebuild only through an explicit documented path. [VERIFIED: AGENTS.md:13-16]
- `BlobStore` owns storage lifecycle, `UnifiedCache` owns cache policy, and `SqlCache` remains separate. [VERIFIED: AGENTS.md:15-17]
- Treat application payloads as trusted while enforcing safe parsing, path containment, and fail-closed integrity boundaries. [VERIFIED: AGENTS.md:18-18]
- The long-term lifecycle must cover filesystem, memory, S3, JSON, SQLite, and PostgreSQL, but this phase must not claim the later backend matrix. [VERIFIED: AGENTS.md:17-17; .planning/ROADMAP.md:90-120]
- Reliability requires atomic commit, rollback, or deterministic reconciliation; same-key operations must not corrupt payloads or split payload/metadata truth. Those mechanisms remain Phase 3 scope here. [VERIFIED: AGENTS.md:19-20; .planning/ROADMAP.md:70-88]
- Correctness precedes performance during migration; final measured budgets are deferred. [VERIFIED: AGENTS.md:21-21]
- Maintain Python `>=3.11`; `pyproject.toml` states the value verbatim as `requires-python = ">=3.11"`. [VERIFIED: pyproject.toml:9-9]
- Use domain-specific exceptions, preserve causes with `raise ... from ...`, avoid new broad `Exception` catches, keep public exports in package `__init__.py`, and add focused tests named `test_<subject>.py`. [VERIFIED: AGENTS.md:130-178]

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python `dataclasses`, `enum`, `typing` | 3.11+ stdlib | Frozen typed manifest and error/result types | Already supported; no dependency or import-surface expansion. [VERIFIED: pyproject.toml:9-13] |
| Python `json` | 3.11+ stdlib | Restricted deterministic manifest encoding | `sort_keys=True`, compact separators, `ensure_ascii=False`, `allow_nan=False`, and explicit UTF-8 produce testable bytes for the restricted v1 data model. [CITED: https://docs.python.org/3.11/library/json.html] |
| Python `hashlib` | 3.11+ stdlib | Streaming SHA-256 payload digest | SHA-256 is guaranteed and incremental `update()` avoids loading whole payloads. [CITED: https://docs.python.org/3.11/library/hashlib.html] |
| Python `hmac` | 3.11+ stdlib | HMAC-SHA256 signing and constant-time verification | Explicit `digestmod` and `compare_digest()` are the standard primitives. [CITED: https://docs.python.org/3.11/library/hmac.html] |
| Existing `GuardedHandlerIO` | repository | Contained staged writes and one private read snapshot | It already publishes a guarded artifact and exposes a snapshot without deserializing. [VERIFIED: src/cacheness/storage/guarded_handler_io.py:301-367] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Python `secrets` | 3.11+ stdlib | New-store signing keys and opaque generation IDs | Generate fresh random material only when initialization is provably safe; never replace a missing required key for an existing signed store. [ASSUMED] |
| Python `os`, `stat` | 3.11+ stdlib | No-follow key-file inspection and POSIX owner/mode checks | Required file-backed signer initialization. Windows `chmod` does not establish POSIX ownership/mode guarantees, so use an explicit caller-supplied key/provider or reject unsupported required-mode configuration. [CITED: https://docs.python.org/3.11/library/os.html] |
| Existing handler registry | repository | Type-to-handler resolution | Resolve only after authenticated format/type validation. [VERIFIED: src/cacheness/storage/blob_store.py:298-314] |

No new package is required, so there is no installation command or package-legitimacy gate for this phase. [VERIFIED: pyproject.toml:9-13]

## Architecture Patterns

### System Architecture Diagram

```text
BlobStore.put(value, key, user_metadata)
  -> handler selection and private serialization
  -> immutable payload publication (existing guarded I/O)
  -> SHA-256 + exact byte count
  -> BlobManifestV1(state=committed, format/version, locator, generation, ...)
  -> canonical bytes -> HMAC-SHA256
  -> CanonicalManifestRepository.put(raw canonical record)

BlobStore.get(key)
  -> repository.get_raw(key)
       -> absent -------------------------------> None
       -> backend failure ----------------------> BlobBackendError
       -> raw record
  -> bounded decode + manifest-version dispatch
       -> exact legacy identity ----------------> read-only legacy adapter
       -> unknown version ----------------------> BlobUnsupportedVersionError
  -> manifest authentication
       -> invalid/missing required signature ---> BlobIntegrityError
  -> critical fields + key + committed state + contained locator + format support
       -> non-committed ------------------------> BlobLifecycleConflictError
  -> one private payload snapshot
  -> streaming SHA-256 + byte-count comparison
       -> missing/tampered ---------------------> BlobIntegrityError
  -> handler.get(snapshot)
  -> success-only access bookkeeping
  -> value
```

### Recommended Project Structure

```text
src/cacheness/
├── error_handling.py                 # extend stable public BlobStore reasons/errors
└── storage/
    ├── manifest.py                   # BlobManifestV1, strict codec, version dispatch
    ├── manifest_repository.py        # narrow JSON/memory/SQLite raw-record adapters
    ├── integrity.py                  # SHA-256 snapshot verifier, strict manifest signer
    ├── read_contract.py              # internal outcome + pure cache translation classifier
    ├── blob_store.py                 # ordered orchestration and compatible public adapters
    └── __init__.py                   # supported exports
tests/
├── test_blob_manifest.py
├── test_blob_manifest_backends.py
├── test_blob_store_read_contract.py
├── test_blob_store_integrity.py
├── test_blob_store_legacy_contract.py
└── test_blob_store_translation_seam.py
```

### Pattern 1: Frozen Manifest, Strict Codec, Complete Signature Projection

Use a frozen typed manifest internally. The recommended v1 fields are `manifest_version`, `payload_format`, `payload_format_version`, `logical_key`, `generation`, `lifecycle_state`, `locator`, `handler_type`, `digest_algorithm`, `digest`, `byte_size`, `created_at`, `handler_metadata`, `user_metadata`, `signature_algorithm`, and `signature`. [ASSUMED] These names are implementation recommendations, not pre-existing wire values; freeze them with golden bytes once chosen.

Encode only `str`, `int`, `bool`, `None`, lists, and string-keyed maps; reject floats and non-JSON objects at the manifest boundary. Use sorted keys, compact separators, no non-finite numbers, explicit UTF-8, and a golden canonical byte fixture. [CITED: https://docs.python.org/3.11/library/json.html] Sign every canonical field except `signature` itself, rather than retaining a user-configurable signed subset. This is necessary because the current signer’s exact default list is `"cache_key"`, `"data_type"`, `"prefix"`, `"file_size"`, `"file_hash"`, `"object_type"`, `"storage_format"`, `"serializer"`, `"compression_codec"`, `"actual_path"`, and `"created_at"`, while custom fields can replace that list. [VERIFIED: src/cacheness/security.py:82-103]

Bound the metadata envelope independently of payload size: add a maximum raw manifest byte count plus structural limits on depth, collection sizes, string sizes, and integer ranges. [ASSUMED] Do **not** impose a general payload filesize policy in this phase. `byte_size` is an integrity claim verified against the snapshot; it is not an acceptance ceiling.

### Pattern 2: A Narrow Manifest Repository Above Legacy Metadata Backends

Do not call `MetadataBackend.put_entry()` with a nested manifest and assume parity. SQLite explicitly extracts a small set of nested fields and then writes only fixed columns, so other canonical fields disappear. [VERIFIED: src/cacheness/metadata.py:1595-1680]

Define a small Phase 2 repository protocol: `get_raw(key) -> bytes | None`, `put_raw(key, bytes)`, `remove(key)`, and `list_keys()`. [ASSUMED] JSON and in-memory repositories store the exact canonical bytes or an exactly reversible transport encoding. SQLite uses a dedicated canonical-manifest table keyed by logical key and stores the canonical bytes as a BLOB. [ASSUMED] Keep the existing `cache_entries` table untouched so the exact 0.3.9 and 0.3.14 layout discriminators remain readable; generalize this repository into the Phase 4 backend contract later.

Phase 2 may branch on the three current local backend types at construction, but it must reject unsupported metadata repositories with a typed configuration/backend error rather than claim PostgreSQL or arbitrary injected-backend parity. [ASSUMED]

### Pattern 3: Authenticate Before Trusting, Verify One Snapshot Before Deserializing

The codec may read only the minimal framing needed to select manifest schema and signing algorithm before authentication; no locator, handler, lifecycle, or size claim may drive I/O until HMAC succeeds. [ASSUMED] After authentication, validate that the manifest logical key equals the lookup key, state is exactly committed, digest/algorithm syntax is valid, size is a non-negative integer, locator is contained, and the handler supports the declared payload format/version. Then create one snapshot, stream its SHA-256 and byte count, compare both, and call the handler with that same live snapshot. Python supports incremental SHA-256 and constant-time HMAC comparison. [CITED: https://docs.python.org/3.11/library/hashlib.html] [CITED: https://docs.python.org/3.11/library/hmac.html]

The current guarded reader is the correct I/O primitive: it says the managed file is opened once for the copy and that callers must complete hashing, signature checks, and `handler.get` while the context is active. [VERIFIED: src/cacheness/storage/guarded_handler_io.py:336-367]

### Pattern 4: Explicit Handler-Owned Format Identity

Extend the handler write-result contract with a payload format identifier and payload format version, and add a support check that does not deserialize. [ASSUMED] The existing exact write-result keys are `"storage_format"`, `"file_size"`, `"actual_path"`, and `"metadata"`; no format-version field exists. [VERIFIED: src/cacheness/interfaces.py:39-53,121-132] Existing handlers already expose native format identifiers such as exact `"npz"` for new array writes and exact `"blosc2"` for the bounded legacy array read. [VERIFIED: src/cacheness/handlers.py:573-634,705-728] Object writes currently choose exact values `"pickle"`, `"dill"`, `"compressed_pickle"`, or `"compressed_dill"`. [VERIFIED: src/cacheness/handlers.py:973-1072]

Version the serialization contract Cacheness depends on, not the installed library version. [ASSUMED] A future NumPy, Blosc2, PyArrow, or pickle library upgrade must not silently change a stored payload identity; support tables and tests decide whether the existing payload version remains readable.

### Pattern 5: Stable Direct-Read Outcome Taxonomy

Add BlobStore-specific subclasses beneath compatible existing public bases. The repository already exposes `CacheStorageError`, `CacheIntegrityError`, and `CacheMetadataError`. [VERIFIED: src/cacheness/error_handling.py:63-89] Recommended types are `BlobIntegrityError`, `BlobUnsupportedVersionError`, `BlobLifecycleConflictError`, `BlobBackendError`, and `BlobMigrationRequiredError`, with stable reason values for invalid/missing signature, manifest invalid, payload missing, digest mismatch, size mismatch, non-committed state, manifest/payload version unsupported, backend failure, and migration required. [ASSUMED]

Preserve `None` only for a genuinely absent authoritative record. A metadata record whose payload is absent is corrupt, not absent; today `BlobStore.get()` returns `None` for both exact cases. [VERIFIED: src/cacheness/storage/blob_store.py:277-297] Preserve causes when translating I/O, codec, signer, or handler failures, and never delete or rewrite evidence during a read.

### Pattern 6: Public Operation Integrity Matrix

| Operation | Required validation | Compatible result |
|-----------|---------------------|-------------------|
| `get` | Full manifest authentication, critical fields/state, snapshot digest/size, deserialize | `None` only if raw record absent; otherwise value or typed error. [ASSUMED] |
| `get_metadata` | Bounded decode, version, authentication, critical fields/state; no payload deserialize | Compatible dictionary view or typed error. [ASSUMED] |
| `exists` | Authenticated committed manifest plus payload presence and digest/size | Boolean only for valid present/absent; typed corrupt/backend/conflict failures. [ASSUMED] |
| `list` | Authenticate every returned record; all-or-error rather than silently omitting corrupt records | Existing list shape after validation. [ASSUMED] |
| `update_metadata` | Authenticate current manifest; patch only `user_metadata`; re-canonicalize and re-sign | Existing bool for actual absence; typed error otherwise. [ASSUMED] |
| `delete` / `clear` | Authenticate before trusting locator; no read-time repair | Preserve evidence on corrupt/conflicted records for later reconciliation. [ASSUMED] |

Current `update_metadata()` merges arbitrary caller keys into the nested dictionary, including fields used for the locator and storage format. [VERIFIED: src/cacheness/storage/blob_store.py:338-371] Canonical v1 must separate immutable structural fields, handler metadata, and mutable user metadata; only the last is patchable through this API. [ASSUMED]

### Pattern 7: Pure UnifiedCache Translation Seam

Expose a pure classifier over typed BlobStore outcomes, such as `classify_cache_read_failure(error) -> translation category | None`, with no dependency on `UnifiedCache`. [ASSUMED] Phase 2 tests should prove integrity errors are distinguishable from absence and that the classifier is exhaustive. Do not change `core.py` read behavior, miss counters, eviction, TTL, or decorators; Phase 6 owns those effects.

### Anti-Patterns to Avoid

- **A custom payload header:** native libraries and handlers already own payload containers; add no Cacheness bytes before NPZ, Parquet, Blosc2, pickle, or dill payloads.
- **Nested-dict manifest storage through current SQLite:** unrecognized fields are discarded. [VERIFIED: src/cacheness/metadata.py:1595-1680]
- **Digest before manifest authentication:** it lets unauthenticated locator/size/algorithm values drive payload work and violates D-07.
- **User-configurable critical signed fields:** the canonical v1 projection is fixed and complete; mutable policy cannot omit structural fields.
- **Auto-regenerating required keys:** the current signer generates a new key for invalid length or load error and falls back to in-memory key after persistence failure. [VERIFIED: src/cacheness/security.py:108-160]
- **Treating missing payload as miss:** it destroys the required absent-versus-corrupt distinction. [VERIFIED: src/cacheness/storage/blob_store.py:288-297]
- **Deserializing to discover format support:** resolve exact handler/type/format/version before opening the payload.
- **Read-side cleanup:** no signature failure, unknown version, corrupt payload, or legacy read may delete, rewrite, rotate, repair, or update access time.
- **Implementing lifecycle state transitions:** Phase 2 accepts only committed visibility; Phase 3 owns preparation, replacement, tombstones, CAS, and recovery.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Payload containers | Cacheness header or container | Existing handler/native NPZ, Parquet, Blosc2, pickle, dill formats | Avoid duplicating format parsing and compression behavior. [VERIFIED: src/cacheness/handlers.py:573-634,790-824,973-1072] |
| Cryptographic primitives | Custom digest, MAC, or equality function | stdlib `hashlib.sha256`, `hmac.new`, `hmac.compare_digest` | Standard, Python 3.11-supported primitives. [CITED: https://docs.python.org/3.11/library/hashlib.html] [CITED: https://docs.python.org/3.11/library/hmac.html] |
| Payload snapshot containment | Second unguarded open or path-based hashing | Existing `GuardedHandlerIO.open_snapshot` | Prevents check/use divergence and preserves containment. [VERIFIED: src/cacheness/storage/guarded_handler_io.py:336-367] |
| General CAS/recovery | Revision protocol, journal, repair engine | Phase 3 lifecycle engine | Out of scope by locked decision. |
| General backend capability system | Backend matrices and topology promises | Phase 4 contract | Out of scope by locked decision. |
| Migration execution | Opportunistic rewrite on read | Phase 7 inventory and copy-verify-switch | Protects the only valid copy. |

**Key insight:** Phase 2’s job is to define and enforce truth at the record/read boundary. It should create the typed primitives later phases compose, not prematurely implement lifecycle, backend, cache-policy, or migration orchestration.

## Runtime State Inventory

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | The repository contains the immutable eight-fixture compatibility corpus. Its exact fixture IDs are `"array-raw-v035-compress"`, `"array-raw-v037-compress2"`, `"json-split-unsigned-v037"`, `"json-split-signed-v038"`, `"sqlite-metadata-json-v039"`, `"decorator-key-v0313"`, `"json-nested-v0314"`, and `"sqlite-columns-v0314"`. [VERIFIED: tests/fixtures/compat/validate_corpus.py:130-243] An ignored local `cache_metadata.db` was inspected read-only in this session and had zero cache entries. [VERIFIED: read-only SQLite probe, 2026-08-30] | Preserve corpus bytes; add canonical-v1 controls separately. Do not migrate the ignored database. User-configured stores outside the repository require Phase 7 inventory. |
| Live service config | None found for Phase 2’s direct local JSON/memory/SQLite scope; PostgreSQL and S3 configuration belong to later backend phases. [VERIFIED: AGENTS.md:80-85; .planning/ROADMAP.md:90-120] | No live-service mutation. Reject unsupported Phase 2 repository configurations explicitly. |
| OS-registered state | None — Cacheness is an in-process library with no web server, worker, or service registration documented in project instructions. [VERIFIED: AGENTS.md:104-108] | None. |
| Secrets/env vars | No `cache_signing_key.bin` exists in the repository; current configuration’s exact filename is `"cache_signing_key.bin"`, and current defaults are `enable_entry_signing: bool = True`, `use_in_memory_key: bool = False`, and `allow_unsigned_entries: bool = True`. [VERIFIED: src/cacheness/config.py:311-326; repository file search, 2026-08-30] | Do not rotate or create keys during reads. Add strict required-mode initialization and documentation; existing external keys remain compatibility inputs, not auto-migrated data. |
| Build artifacts | `.venv/lib/python3.13/site-packages/cacheness-0.3.14.dist-info` and `cacheness.pth` exist; the `.pth` contains `/Users/akriz/code/cacheness/src`. [VERIFIED: filesystem inspection, 2026-08-30] | Editable source is already active; no data migration. Refresh the environment only if dependency metadata changes, which this research does not recommend. |

**Canonical post-edit question:** after repository files are updated, legacy metadata files/databases and signing keys still exist in user-chosen cache roots. Phase 2 must classify them without mutation; Phase 7 performs any conversion. [ASSUMED]

## Common Pitfalls

### Pitfall 1: SQLite Appears to Accept a Manifest but Silently Loses It
**What goes wrong:** JSON/memory tests pass, while SQLite reads reconstruct an incomplete record.
**Why it happens:** SQLite extracts recognized nested keys and persists a fixed column set. [VERIFIED: src/cacheness/metadata.py:1595-1680]
**How to avoid:** Add backend-contract tests that round-trip the identical typed manifest and canonical signed bytes through JSON, memory, and the dedicated SQLite manifest table.
**Warning signs:** `payload_format_version`, `generation`, `lifecycle_state`, or user metadata are missing after reopen.

### Pitfall 2: Required Signing Quietly Becomes a New Key
**What goes wrong:** Existing records become unverifiable but initialization proceeds.
**Why it happens:** Current key loading generates a replacement on invalid length/load error and persistence failure may retain an in-memory key. [VERIFIED: src/cacheness/security.py:108-160]
**How to avoid:** Separate “initialize a provably empty new store” from “open an existing signed store”; required reopen fails for missing/invalid key material. Create with exclusive/no-follow semantics and attest regular-file ownership/mode before use. [ASSUMED]
**Warning signs:** startup logs mention generating a key after an error, or restarts cannot verify prior records.

### Pitfall 3: The Manifest Is Signed but Critical Fields Remain Mutable
**What goes wrong:** an attacker changes locator, handler, lifecycle, size, or format without invalidating the MAC.
**Why it happens:** the existing signer permits a custom field list. [VERIFIED: src/cacheness/security.py:85-103]
**How to avoid:** v1 signs every canonical field except the signature value; mutation creates and signs a new manifest record.
**Warning signs:** tests construct a “signed fields” configuration for canonical v1.

### Pitfall 4: Verification Uses Different Payload Opens
**What goes wrong:** digest covers one file state while deserialization consumes another.
**Why it happens:** code hashes a managed path, closes it, then asks the handler to reopen it.
**How to avoid:** hash and deserialize the same guarded private snapshot within one context. [VERIFIED: src/cacheness/storage/guarded_handler_io.py:336-367]
**Warning signs:** handler receives a managed-root locator or a second snapshot.

### Pitfall 5: Unknown Version Falls Through to a Familiar Handler
**What goes wrong:** future bytes are interpreted as an older NPZ/pickle/Parquet contract.
**Why it happens:** current handler selection uses only data type and storage format; the write-result protocol lacks a format version. [VERIFIED: src/cacheness/interfaces.py:39-53,121-132]
**How to avoid:** exact `(handler_type, payload_format, payload_format_version)` dispatch and typed unsupported-version failure before snapshot access.
**Warning signs:** default versions are supplied when persisted version is absent on a purported canonical record.

### Pitfall 6: Compatibility Adapter Becomes an Implicit Migrator
**What goes wrong:** a read rewrites, deletes, signs, or upgrades the only old copy.
**Why it happens:** current security configuration includes exact `delete_invalid_signatures: bool = True`. [VERIFIED: src/cacheness/config.py:321-326]
**How to avoid:** exact legacy detection returns an in-memory legacy identity; access is read-only, and conversion-needed becomes a typed inspectable outcome.
**Warning signs:** corpus fixture digests or mtimes change after tests.

## Code Examples

### Restricted Canonical Encoding

```python
# Source: Python 3.11 json documentation
encoded = json.dumps(
    unsigned_manifest,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
).encode("utf-8")
```

The implementation must validate the restricted manifest value types before this call; `json.dumps()` alone is not the schema validator. [CITED: https://docs.python.org/3.11/library/json.html]

### HMAC-SHA256 and Constant-Time Verification

```python
# Source: Python 3.11 hmac documentation
expected = hmac.new(key, canonical_unsigned_bytes, hashlib.sha256).digest()
if not hmac.compare_digest(expected, supplied_signature):
    raise BlobIntegrityError(...)
```

Use same-type byte strings in `compare_digest`; never compare hex text with ordinary equality. [CITED: https://docs.python.org/3.11/library/hmac.html]

### Same-Snapshot Payload Verification

```python
# Pattern based on the repository's GuardedHandlerIO contract.
with guarded_handler_io.open_snapshot(manifest.locator, handler_metadata) as snapshot:
    digest = hashlib.sha256()
    byte_size = 0
    with snapshot.path.open("rb") as payload:
        for chunk in iter(lambda: payload.read(1024 * 1024), b""):
            digest.update(chunk)
            byte_size += len(chunk)
    verify_digest_and_size(manifest, digest.hexdigest(), byte_size)
    value = handler.get(snapshot.path, snapshot.metadata)
```

The 1 MiB chunk in this example is an implementation buffer, not a payload-size policy. [ASSUMED] The guarded snapshot contract is verified in repository source. [VERIFIED: src/cacheness/storage/guarded_handler_io.py:336-367]

## State of the Art

| Old Approach | Current Recommendation | Impact |
|--------------|------------------------|--------|
| Backend-shaped entry dictionaries | Frozen backend-neutral manifest with strict repository codecs | Backend storage no longer defines lifecycle meaning. [ASSUMED] |
| XXH3 file hash | SHA-256 with explicit algorithm ID for security integrity | XXH3 remains only compatibility/keying. [ASSUMED] |
| Configurable subset signing | Fixed complete v1 signed projection | Critical fields cannot be omitted. [ASSUMED] |
| String concatenation with `str(value)` | Restricted canonical JSON bytes | Types and delimiters are unambiguous. The current payload uses sorted `field:value` pieces joined by `|`. [VERIFIED: src/cacheness/security.py:162-191] |
| Auto-generated replacement/fallback key | Strict required-key provider | Existing signed stores fail closed instead of becoming unreadable. [ASSUMED] |
| Missing metadata or payload both return `None` | Only absent record returns `None`; payload absence is typed corruption | Direct callers can recover and observe correctly. [VERIFIED: src/cacheness/storage/blob_store.py:277-297] |

**Deprecated/outdated:**
- `CacheEntrySigner`’s field-subset/string-concatenation scheme remains a compatibility verifier; do not use it as manifest-v1 authenticity. [VERIFIED: src/cacheness/security.py:75-103,162-191]
- `file_hash` is exactly documented in the current SQLite model as an `"XXH3_64 hash (16 hex chars)"`; it cannot satisfy D-08. [VERIFIED: src/cacheness/metadata.py:167-174]
- Read-time invalid-signature deletion must not apply to canonical BlobStore reads. The existing exact configuration value is `delete_invalid_signatures: bool = True`. [VERIFIED: src/cacheness/config.py:321-326]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | A dedicated SQLite canonical-manifest table is preferable to altering legacy `cache_entries`. | Architecture Pattern 2 | Planner may choose another lossless adapter, but must preserve exact legacy detection and round-trip bytes. |
| A2 | Recommended Python class, module, field, and reason-code names are not yet wire contracts. | Project Structure / Patterns 1 and 5 | Names must be frozen in implementation tests and public exports. |
| A3 | Canonical v1 metadata accepts only a bounded JSON-compatible restricted type set. | Pattern 1 | Some callers may currently supply non-JSON custom metadata; characterize and document the compatibility consequence. |
| A4 | A pure classifier is sufficient for the Phase 6 translation seam. | Pattern 7 | Later policy integration may require more context, but no runtime cache behavior should be added now. |
| A5 | Required file-key mode should be unsupported where ownership/mode cannot be truthfully attested; explicit supplied-key/provider remains available. | Standard Stack / Pitfall 2 | Cross-platform API details need confirmation and tests on supported operating systems. |
| A6 | `exists()` should verify full payload digest/size, not merely metadata and file presence. | Operation Integrity Matrix | This is stricter and costlier; the planner should lock the semantic because STOR-08 requires corruption distinction. |
| A7 | Lifecycle code owns the typed codec/authentication/outcome contract while persistence adapters provide raw records and contained snapshots. | Responsibility Map / Summary | A different seam could leak backend-shaped values into lifecycle code and defeat D-02. |
| A8 | The recommended v1 manifest field set, complete-signature projection, and frozen golden encoding are the correct new wire contract. | Pattern 1 / State of the Art | Field or encoding changes after release require a new manifest version. |
| A9 | Bounded parsing needs independent byte, depth, count, string, and integer limits, but Phase 2 should not impose a payload filesize ceiling. | Pattern 1 / Open Question 1 | Bounds that are too tight break metadata compatibility; absent bounds expose resource exhaustion. |
| A10 | A minimal raw-record repository protocol may branch over JSON/memory/SQLite now and reject unsupported backends until Phase 4. | Pattern 2 | Over-generalizing steals Phase 4; under-scoping could prevent required local-backend coverage. |
| A11 | Minimal framing/version/algorithm selection precedes authentication; all other critical-field use follows successful authentication. | Pattern 3 / Security Domain | Parsing too much unauthenticated data or verifying in another order weakens D-07. |
| A12 | Handlers need explicit format versions and non-deserializing support checks, versioned by the Cacheness serialization contract rather than dependency release number. | Pattern 4 | Ambiguous format identities make MIGR-02 and MIGR-07 unenforceable. |
| A13 | The recommended BlobStore exception subclasses and stable reason taxonomy preserve existing base-class catches while distinguishing required outcomes. | Pattern 5 | Public names/reasons become compatibility surface once shipped. |
| A14 | `get`, `get_metadata`, `exists`, `list`, `update_metadata`, `delete`, and `clear` should follow the operation integrity matrix, with structural/handler/user metadata separated and reads non-mutating. | Pattern 6 / Runtime Inventory / Security Domain | Compatibility or performance may require explicit adapters, but silent corruption/mutation is forbidden. |
| A15 | Required-key initialization, adversarial/golden tests, and the same-snapshot hashing example are the recommended enforcement details. | Pitfalls / Code Examples / Validation Architecture | Platform semantics, buffering, and empty-store detection need focused tests before API freeze. |

## Open Questions

1. **What exact bounded-manifest limits should become v1 wire-policy constants?**
   - What we know: D-07 requires bounded parsing, and limits must be separate from payload filesize.
   - What's unclear: Current public custom-metadata size/depth usage has no checked-in contract. [ASSUMED]
   - Recommendation: characterize existing tests/examples, then freeze generous byte/depth/collection/string bounds with boundary tests; do not add a payload filesize ceiling.

2. **What is the exact direct `exists()` integrity promise?**
   - What we know: it must not report corrupt content as healthy, and STOR-08 distinguishes corrupt from absent.
   - What's unclear: full SHA-256 verification makes `exists()` O(payload bytes).
   - Recommendation: define `exists()` as “valid committed readable record” and verify digest/size; a later explicit lightweight metadata-presence API can have weaker semantics. [ASSUMED]

3. **How should callers supply required signing keys on non-POSIX systems?**
   - What we know: Python documents that Windows `chmod()` can only set/clear the read-only flag, not establish POSIX owner/mode guarantees. [CITED: https://docs.python.org/3.11/library/os.html]
   - What's unclear: no current cross-platform secret-provider interface exists.
   - Recommendation: accept exact 32-byte application-provided key material or a narrow key-provider protocol; reject required file mode when its permission policy cannot be attested. [ASSUMED]

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| Python | Manifest/integrity implementation | ✓ | 3.13.3 in `.venv`; project supports `>=3.11` | Verify on 3.11 and 3.12 later; no syntax above 3.11. [VERIFIED: environment probe; pyproject.toml:9-9] |
| pytest | Validation | ✓ | 8.4.1 | `.venv/bin/pytest` if sandboxed `uv` cache is unavailable. [VERIFIED: environment probe] |
| Ruff | Lint of changed files | ✓ | 0.12.9 | `.venv/bin/ruff`. [VERIFIED: environment probe] |
| SQLAlchemy | SQLite manifest adapter tests | ✓ | 2.0.43 | Skip only via existing optional-dependency policy; SQLite contract coverage is required for this phase. [VERIFIED: environment probe] |
| NumPy / Blosc2 / pandas / polars / pyarrow | Representative handler format tests | ✓ | 2.3.2 / 3.7.0 / 2.3.1 / 1.32.3 / 21.0.0 | Exercise a minimal built-in subset per test and preserve optional guards. [VERIFIED: environment probe] |

**Missing dependencies with no fallback:** None for implementation; Python 3.11/3.12 runtimes are not the active local interpreter and require later compatibility execution. [VERIFIED: environment probe]

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest; repository discovery is `testpaths = ["tests"]`, `python_files = ["test_*.py"]`, `python_classes = ["Test*"]`, `python_functions = ["test_*"]`. [VERIFIED: pyproject.toml:82-99] |
| Config file | `pyproject.toml` |
| Quick run command | `uv run pytest -q -o log_cli=false <target> -x` |
| Full suite command | `uv run pytest -q -o log_cli=false` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| STOR-01 | Identical typed manifest semantics and signed bytes across JSON, memory, SQLite | contract | `uv run pytest -q -o log_cli=false tests/test_blob_manifest_backends.py -x` | ❌ Wave 0 |
| STOR-02 | Only committed authentic manifests reach payload snapshot | unit/ordering | `uv run pytest -q -o log_cli=false tests/test_blob_store_read_contract.py -x` | ❌ Wave 0 |
| STOR-08 | Missing/corrupt/conflict/backend outcomes remain distinct for every direct operation | unit/contract | `uv run pytest -q -o log_cli=false tests/test_blob_store_read_contract.py -x` | ❌ Wave 0 |
| SECU-03 | Signature then digest/size then handler order, using one snapshot | unit/integration | `uv run pytest -q -o log_cli=false tests/test_blob_store_integrity.py -x` | ❌ Wave 0 |
| SECU-04 | Missing/invalid/unsafe key, absent/invalid signature, unsupported signer all fail closed | unit/platform | `uv run pytest -q -o log_cli=false tests/test_blob_store_integrity.py -x` | ❌ Wave 0 |
| SECU-05 | Mutation of every critical manifest field invalidates signature | parameterized unit | `uv run pytest -q -o log_cli=false tests/test_blob_manifest.py -x` | ❌ Wave 0 |
| SECU-08 | Blob integrity exception is public; pure cache classifier recognizes it without rewiring | unit | `uv run pytest -q -o log_cli=false tests/test_blob_store_translation_seam.py -x` | ❌ Wave 0 |
| MIGR-02 | Manifest and payload versions are independent and exact | golden/contract | `uv run pytest -q -o log_cli=false tests/test_blob_manifest.py -x` | ❌ Wave 0 |
| MIGR-07 | Unknown manifest/payload versions and exact legacy conversion-needed outcomes are typed and non-mutating | compatibility | `uv run pytest -q -o log_cli=false tests/test_blob_store_legacy_contract.py -x` | ❌ Wave 0 |

### Required Adversarial Cases

- Golden canonical bytes across insertion order, Unicode, empty/optional metadata, and restart/backend reopen. [ASSUMED]
- Bounded parser rejects oversized raw document, excessive depth/count/string, invalid UTF-8, duplicate keys, non-string keys, floats/non-finite numbers, booleans where integer is expected, negative/oversized size, and invalid digest/signature syntax before payload access. [ASSUMED]
- Spy assertions prove no handler lookup/invocation, snapshot, access update, delete, rewrite, or signing-key creation on an earlier failure. [ASSUMED]
- Tamper each signed critical field independently and tamper payload bytes without changing length, truncate, extend, remove, and replace during/after snapshot. [ASSUMED]
- JSON, memory, and SQLite return the same manifest model after restart where applicable; SQLite must preserve user and handler metadata that current `put_entry()` loses. [VERIFIED: src/cacheness/metadata.py:1595-1680]
- Hash all eight Phase 1 fixture trees before and after legacy reads; assert byte and mtime stability and exact legacy identity. [VERIFIED: tests/fixtures/compat/validate_corpus.py:128-243]
- Run representative NPZ, Parquet, pickle/dill, and Blosc2 handler paths without adding a Cacheness header. [VERIFIED: src/cacheness/handlers.py:573-634,790-824,973-1072]

### Sampling Rate

- **Per task commit:** targeted test file plus `uv run ruff check <changed source/test paths>`
- **Per wave merge:** all Phase 2 test files plus Phase 1 compatibility/path/security regressions
- **Phase gate:** `uv run pytest -q -o log_cli=false` green, compatibility corpus validator green, and Ruff clean on changed files before `$gsd-verify-work`

### Wave 0 Gaps

- [ ] `tests/test_blob_manifest.py` — golden codec, bounds, versions, complete signed projection
- [ ] `tests/test_blob_manifest_backends.py` — JSON/memory/SQLite parity and reopen
- [ ] `tests/test_blob_store_read_contract.py` — ordered outcomes for every public operation
- [ ] `tests/test_blob_store_integrity.py` — digest/signature/key/tamper matrix and same snapshot
- [ ] `tests/test_blob_store_legacy_contract.py` — exact eight-fixture read-only outcomes
- [ ] `tests/test_blob_store_translation_seam.py` — pure Phase 6 classifier boundary

## Security Domain

### Applicable ASVS 5.0 Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V1 Encoding and Sanitization | yes | Strict bounded typed manifest decoder; no metadata-driven deserialization or guessing. [CITED: https://github.com/OWASP/ASVS/blob/master/5.0/docs_en/OWASP_Application_Security_Verification_Standard_5.0.0_en.flat.json] |
| V5 File Handling | yes | Existing contained no-follow managed I/O and one private snapshot. [VERIFIED: src/cacheness/storage/guarded_handler_io.py:336-367] |
| V11 Cryptography | yes | SHA-256 and HMAC-SHA256 via Python stdlib; constant-time verification; algorithm IDs are authenticated. [CITED: https://docs.python.org/3.11/library/hmac.html] |
| V13 Configuration | yes | Required signer configuration and key validation fail closed; secrets are not logged or auto-replaced. [CITED: https://github.com/OWASP/CheatSheetSeries/blob/master/cheatsheets/Key_Management_Cheat_Sheet.md] |
| Authentication / Session / Access Control | no | This phase is an in-process storage library boundary, not an identity/session authorization system. [VERIFIED: AGENTS.md:7-11,43-44] |

OWASP ASVS’s current stable release is 5.0.0, and its official repository exposes the machine-readable requirements used for the category mapping. [CITED: https://github.com/OWASP/ASVS]

### Known Threat Patterns for This Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Manifest tampering redirects locator or handler | Tampering | Authenticate canonical bytes before using any critical field; sign the complete projection. [ASSUMED] |
| Payload replacement after metadata validation | Tampering | One private snapshot, then SHA-256 + size, then handler on same snapshot. [VERIFIED: src/cacheness/storage/guarded_handler_io.py:336-367] |
| Key substitution or silent loss | Spoofing/Tampering | No-follow regular-file inspection, exact key validation, owner/mode attestation where supported, no auto-regeneration for existing signed stores. [CITED: https://docs.python.org/3.11/library/os.html] |
| Malformed/oversized manifest exhausts parser | Denial of Service | Bound raw bytes and structure before constructing the typed model; reject duplicate/unknown required fields. [ASSUMED] |
| Unknown future version is interpreted as current | Tampering | Exact version dispatch and typed unsupported-version failure without payload access. [ASSUMED] |
| Signed pickle/dill is treated as safe hostile input | Elevation of Privilege | Preserve trusted-application-payload boundary; authenticity does not make unsafe serializers sandboxed. [VERIFIED: AGENTS.md:15-18] |
| Failure path destroys forensic evidence | Repudiation | Reads are non-mutating; no automatic deletion, rewrite, rotation, or access update on failure. [ASSUMED] |

## Sources

### Primary (HIGH confidence)

- `src/cacheness/storage/blob_store.py` — current direct lifecycle and absence conflation.
- `src/cacheness/storage/guarded_handler_io.py` — safe staging and same-snapshot read seam.
- `src/cacheness/security.py` and `src/cacheness/config.py` — current signer, key fallback, signed-field, and deletion behavior.
- `src/cacheness/metadata.py` — exact SQLite schema and field-loss boundary.
- `src/cacheness/interfaces.py` and `src/cacheness/handlers.py` — handler write contract and native payload formats.
- `tests/fixtures/compat/validate_corpus.py` — normative eight-fixture compatibility matrix.
- `AGENTS.md`, `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md`, and Phase 2 `CONTEXT.md` — project and phase constraints.

### Secondary (MEDIUM confidence)

- https://docs.python.org/3.11/library/json.html — deterministic restricted JSON encoding controls.
- https://docs.python.org/3.11/library/hashlib.html — SHA-256 and incremental hashing.
- https://docs.python.org/3.11/library/hmac.html — HMAC and constant-time verification.
- https://docs.python.org/3.11/library/os.html — no-follow stat and platform permission limitations.
- https://github.com/OWASP/ASVS — official ASVS release/category reference.
- https://github.com/OWASP/CheatSheetSeries/blob/master/cheatsheets/Key_Management_Cheat_Sheet.md — key-management guidance.

### Tertiary (LOW confidence)

- None used as authority; implementation recommendations are explicitly tagged `[ASSUMED]` and recorded above.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — existing Python standard library and repository primitives; official Python docs checked.
- Architecture: HIGH — derived from opened live lifecycle, signer, handler, guarded I/O, and metadata implementations.
- Pitfalls: HIGH — each critical failure mode is visible in source and mapped to an adversarial test.
- SQLite persistence recommendation: MEDIUM — field loss is verified; the dedicated-table design is prescriptive but not yet implemented.

**Research date:** 2026-08-30
**Valid until:** 2026-09-29 (stable stdlib and in-repo contracts; refresh after Phase 2 implementation changes)
