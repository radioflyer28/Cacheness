---
phase: 02-canonical-storage-and-integrity-contract
verified: 2026-08-30T16:30:02Z
status: passed
score: 13/13 must-haves verified
behavior_unverified: 0
overrides_applied: 0
decision_coverage:
  honored: 14
  total: 14
  not_honored: []
human_verification: []
---

# Phase 2: Canonical Storage and Integrity Contract Verification Report

**Phase Goal:** Direct `BlobStore` users interact with one versioned,
backend-neutral record and deterministic fail-closed read contract.
**Verified:** 2026-08-30T16:30:02Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | A direct `BlobStore` caller can put and get an ordinary object through one signed, committed canonical manifest. | ✓ VERIFIED | `BlobStore.put()` writes a handler-owned candidate, computes SHA-256 and size, constructs/signs `BlobManifestV1`, and publishes exact manifest bytes. `BlobStore.get()` loads the same record and returns the object only after the authenticated snapshot pipeline. `test_tracer_direct_blob_store_uses_one_authenticated_committed_manifest` passes for JSON and SQLite. |
| 2 | Every admitted Phase 2 backend preserves the same backend-neutral canonical record and exact bytes. | ✓ VERIFIED | `create_manifest_repository()` admits exact JSON, memory, and SQLite identities. JSON/memory use reversible base64 transport; SQLite uses `cacheness_manifest_records_v1` with a BLOB. Exact-byte CRUD, reopen, isolation, and transactional rollback tests pass across all three. The broader advertised backend matrix remains explicitly owned by Phases 4-5 and is not falsely claimed here. |
| 3 | Manifest schema and handler-owned payload contracts are explicit and independently versioned. | ✓ VERIFIED | `BlobManifestV1` has separate `schema_version`, `payload_format`, and `payload_format_version` fields. `CacheHandler` and `GuardedWriteResult` expose exact payload identity; `HandlerRegistry.resolve_payload_contract()` resolves declarations without payload I/O. Independent-version and support-resolution tests pass. |
| 4 | New canonical writes retain native handler payload formats and add no Cacheness payload wrapper/header. | ✓ VERIFIED | Array writes call `numpy.savez*`; dataframe writes call native Parquet APIs; object writes use pickle/dill or their configured Blosc compression; TensorFlow writes call `blosc2.save_tensor`. Native NPZ, Parquet, pickle, dill, and legacy Blosc2 dispatch tests pass. The historical custom-framed Blosc2 array reader is bounded and read-only legacy compatibility only; it is not used by new canonical writes. |
| 5 | Normal direct reads expose only authenticated committed generations across `get`, `get_metadata`, `exists`, and `list`. | ✓ VERIFIED | `_load_authenticated_manifest()` verifies the HMAC, constructs the typed manifest, checks lookup-key equality and `state == "committed"`, then optionally resolves the handler and locator. Direct-operation tests prove prepared/conflicted records raise and are not silently omitted. |
| 6 | Only a genuinely absent repository record is a normal miss; malformed, unauthenticated, unsupported, conflicted, migration-required, and backend failures remain typed and distinct. | ✓ VERIFIED | Repository `get_raw()` returns `None` only when no record/projection exists. `error_handling.py` defines stable BlobStore error types/reasons and `read_contract.py` classifies them without collapsing absence. Backend, partial-publication, direct-surface, and public-export tests pass. |
| 7 | Read ordering is bounded decode/version → HMAC authentication → critical validation → one private snapshot → SHA-256/size → handler. | ✓ VERIFIED | `_load_authenticated_manifest()` performs bounded framing/version decode, `verify_hmac_sha256`, semantic construction, committed/key checks, handler-contract resolution, and contained locator resolution before `open_snapshot()`. `test_read_authenticates_validates_snapshots_hashes_then_deserializes` asserts the runtime event sequence `authenticate, validate, snapshot, digest, handler` and exactly one snapshot. |
| 8 | Required signing fails closed for absent, malformed, unsafe, unsupported, or replaced key/signature state, without ephemeral downgrade. | ✓ VERIFIED | `ManifestKeyProvider` requires exact 32-byte material, uses no-follow POSIX file attestation, creates only by explicit exclusive first-store initialization, and never creates during reads/reopen. HMAC uses SHA-256 and `compare_digest`. Key race, partial write, symlink/mode, bad length/signature, and missing-key reopen tests pass. |
| 9 | Signatures bind every critical locator, handler/type, payload-format, digest, size, and lifecycle-generation field. | ✓ VERIFIED | `BlobManifestV1.signing_bytes()` canonicalizes the complete record except only `signature`. The per-field mutation test covers schema, key, generation, state, locator, handler type, payload format/version, digest algorithm/value, size, creation metadata, handler/user metadata, and signature algorithm. |
| 10 | Payload integrity and size are verified from the same guarded private snapshot before trusted deserialization. | ✓ VERIFIED | `GuardedHandlerIO.open_snapshot()` opens the managed file once into a mode-0600 private snapshot. `BlobStore.get()` hashes and size-checks that snapshot path and passes the same live snapshot path/metadata to the handler before context cleanup. Tamper, truncation, extension, missing-payload, and handler-forbidden tests pass. |
| 11 | Failed reads and exact legacy/future-format inspection do not rewrite, quarantine, delete, rotate keys, or update access evidence. | ✓ VERIFIED | Tamper tests compare manifest bytes, payload bytes, mtimes, key bytes, and backend entries before/after. All eight compatibility fixture trees are hash/mtime checked across direct read surfaces. Unknown lookalikes, malformed signatures, symlinked evidence, extra sidecars, and unknown future versions fail typed without mutation. |
| 12 | Metadata and destructive operations cannot bypass the authenticated canonical boundary. | ✓ VERIFIED | `update_metadata()` rejects structural fields, patches bounded `user_metadata`, and re-signs the full record. `delete()` authenticates the locator before removal. `clear()` authenticates the complete selected manifest set before invoking recovery, so one invalid record produces zero mutation. Operation-matrix, containment, and clear-recovery tests pass. |
| 13 | `UnifiedCache` has a defined pure seam for later selected integrity-to-miss translation, without Phase 2 cache-policy wiring. | ✓ VERIFIED | `classify_cache_read_failure()` is a closed type-only transform over public BlobStore errors, returns `None` for absence, has no `UnifiedCache`/`cacheness.core` dependency, and does not mutate error context. Exhaustive taxonomy and purity tests pass. |

**Score:** 13/13 truths verified (0 present-but-behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/cacheness/storage/manifest.py` | Frozen schema-1 model, bounded canonical codec, exact version dispatch | ✓ VERIFIED | Substantive 400+ line implementation; deterministic canonical bytes, exact fields, independent versions, aggregate limits, and signed projection are wired into `BlobStore`. |
| `src/cacheness/storage/manifest_repository.py` | Lossless JSON, memory, and isolated SQLite canonical repositories | ✓ VERIFIED | All adapters implement raw CRUD; exact concrete selection is wired in `BlobStore.__init__`; SQLite publication is transactional. |
| `src/cacheness/storage/integrity.py` | Strict HMAC key/provider and SHA-256/size primitives | ✓ VERIFIED | Imported and called by canonical put/read/update paths; constant-time signature verification and no-follow key attestation are substantive. |
| `src/cacheness/storage/blob_store.py` | Canonical committed/integrity semantics across direct operations | ✓ VERIFIED | Production facade wires manifests, repositories, strict signer, guarded snapshots, handlers, typed outcomes, legacy recognition, and mutation preflight. |
| `src/cacheness/interfaces.py` | Explicit payload format/version contracts | ✓ VERIFIED | Public handler properties and guarded write-result fields are implemented and consumed by built-in/custom handlers. |
| `src/cacheness/handlers.py` | Stable handler-owned native formats and non-I/O support checks | ✓ VERIFIED | Built-ins declare native contracts; registry exact-resolution and native library operations are exercised. |
| `src/cacheness/storage/read_contract.py` | Pure future cache failure classifier | ✓ VERIFIED | Exhaustive public-type classifier, imported/exported without cache-policy dependency. |
| `src/cacheness/storage/legacy_manifest.py` | Exact bounded read-only legacy identity recognition | ✓ VERIFIED | Fixed-name, hash-pinned, no-follow recognition covers eight fixtures and immutable SQLite inspection. |
| `src/cacheness/error_handling.py` / `src/cacheness/storage/__init__.py` | Stable public direct-storage types/reasons and exports | ✓ VERIFIED | Exact reason/type/inheritance and barrel-export tests pass. |
| Six Phase 2 contract test files | Behavioral proof for all phase requirements and prohibitions | ✓ VERIFIED | 170 collected Phase 2 cases pass; no active skipped test in the verification environment. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `blob_store.py` | `manifest_repository.py` | `create_manifest_repository`, `get_raw`, `put_raw`, `remove`, `list_keys` | ✓ WIRED | Every direct operation consumes the raw canonical repository rather than a backend-shaped dictionary. |
| `blob_store.py` | `manifest.py` | decode, authenticate projection, `BlobManifestV1.from_mapping`, canonical publication | ✓ WIRED | Record bytes are typed only after HMAC authentication; new records are canonicalized before repository publication. |
| `blob_store.py` | `integrity.py` | `sign_hmac_sha256`, `verify_hmac_sha256`, `sha256_and_size` | ✓ WIRED | The mechanical plan checker expected a `verify_(manifest|payload)` name and reported this link false; direct source inspection and passing event-order/tamper tests verify the actual explicit calls. |
| `blob_store.py` | `guarded_handler_io.py` | `open_snapshot` | ✓ WIRED | One contained private snapshot is verified and remains live through handler deserialization. |
| `blob_store.py` | `handlers.py` | `resolve_payload_contract` / `get_handler_by_type`, then `handler.get` | ✓ WIRED | Handler contract is resolved after authentication and before snapshot; handler invocation follows digest/size checks. |
| `handlers.py` | `interfaces.py` | `payload_format`, `payload_format_version`, `supports_payload_contract` | ✓ WIRED | Built-in handlers implement the contract and return it in guarded write results. |
| `blob_store.py` | `legacy_manifest.py` | `recognize_legacy_fixture_tree` | ✓ WIRED | Exact recognizer runs before key/backend initialization and attaches an in-memory migration-required identity only. |
| `read_contract.py` | public BlobStore errors | `classify_cache_read_failure` | ✓ WIRED | Pure classifier exhaustively preserves integrity, version, lifecycle, backend, and migration categories. |

### Storage Data-Flow Trace (Level 4)

| Flow | Source | Critical transformations | Sink | Status |
|------|--------|--------------------------|------|--------|
| Canonical write | Caller value and metadata | handler-native candidate → SHA-256/size → committed manifest → HMAC → canonical JSON | exact local manifest repository plus native payload | ✓ FLOWING |
| Canonical read | exact repository bytes | bounded decode/version → HMAC → typed critical validation → contained private snapshot → SHA-256/size | exact declared handler → caller value | ✓ FLOWING |
| Metadata read | exact repository bytes | HMAC → committed/key/locator validation | compatible metadata view built only from signed manifest | ✓ FLOWING |
| Legacy inspection | fixed named fixture evidence | no-follow open → pinned SHA-256 → exact schema/signature validation | in-memory read-only identity / migration-required result | ✓ FLOWING |

This is a library/foundation phase; no rendered user-interface data flow applies.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Complete Phase 2 canonical/integrity contract | `uv run pytest -q -o log_cli=false` over the six Phase 2 test files | 170 collected cases; command exited 0 | ✓ PASS |
| Phase 1 compatibility, containment, integrity, and clear-recovery regressions | `uv run pytest -q -o log_cli=false tests/test_stored_compatibility.py tests/test_filesystem_containment.py tests/test_cache_integrity.py tests/test_clear_recovery.py -x` | 290 passed; 1 expected Windows-junction skip | ✓ PASS |
| Immutable eight-fixture corpus | `uv run python tests/fixtures/compat/validate_corpus.py --expected-through sqlite-columns-v0314` | `compatibility corpus validated through sqlite-columns-v0314` | ✓ PASS |
| Phase-created code quality | Exact targeted `uv run ruff check` from Plan 07 | `All checks passed!` | ✓ PASS |

### Probe Execution

No shell probe is declared or implied for this Python library phase. The independent
compatibility-corpus validator above is the required non-pytest executable check and
passed in the verifier's process.

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|--------------|-------------|--------|----------|
| STOR-01 | 01, 02, 03, 07 | One versioned canonical manifest | ✓ SATISFIED | Exact schema-1 manifest, deterministic bytes, JSON/memory/SQLite parity, native payload identity, end-to-end tracer. |
| STOR-02 | 01, 05, 06, 07 | Reads expose only committed generations | ✓ SATISFIED | Shared authenticated loader rejects every non-committed state before payload access across direct read surfaces. |
| STOR-08 | 01, 02, 04, 05, 06, 07 | Typed missing/corrupt/conflict/backend outcomes | ✓ SATISFIED | Public error hierarchy, repository/direct operation matrices, preserved causes, and absence-only `None`. |
| SECU-03 | 01, 03, 05, 06, 07 | Authenticity/integrity before deserialization | ✓ SATISFIED | Passing ordered event test and adversarial no-handler cases. |
| SECU-04 | 01, 05, 06, 07 | Required signing fails closed | ✓ SATISFIED | Exact key/provider/signature matrix and missing-key reopen non-mutation test. |
| SECU-05 | 01, 05, 06, 07 | Critical manifest fields are signed | ✓ SATISFIED | Complete projection plus independent per-field mutation assertions. |
| SECU-08 | 04, 05, 06, 07 | Typed integrity error and later cache seam | ✓ SATISFIED | BlobStore integrity base/types and pure classifier; no Phase 2 cache-policy mutation. |
| MIGR-02 | 01, 03, 07 | Independent metadata/payload version identifiers | ✓ SATISFIED | Explicit manifest schema and exact handler payload format/version contracts. |
| MIGR-07 | 01, 03, 04, 05, 07 | Unknown future formats fail explicitly | ✓ SATISFIED | Unknown schema/payload identity/version and exact legacy outcomes are typed and non-mutating. |

No Phase 2 requirement is orphaned: all nine roadmap requirement IDs appear in plan
frontmatter and have executable evidence.

### Prohibition Verification

| Prohibition | Status | Enforcement evidence |
|-------------|--------|----------------------|
| No Cacheness wrapper/header around native handler payload formats | ✓ VERIFIED | Direct source calls to NumPy/Parquet/pickle/dill/Blosc2-native APIs plus native container signature/reader tests. New arrays always write NPZ. Historical custom-framed Blosc2 arrays remain a bounded read-only legacy compatibility input, never a canonical write format. |
| Read paths do not mutate/rewrite/quarantine/delete legacy, corrupt, or future evidence | ✓ VERIFIED | Canonical tamper and all eight legacy fixtures assert pre/post bytes, hashes, mtimes, key material, access projections, and no sidecars. |
| Required signing does not downgrade, invent ephemeral replacement keys, or accept unsigned manifests | ✓ VERIFIED | Strict key provider and reopen tests assert no replacement file; absent/wrong signatures fail before locator, snapshot, or handler events. |

### Test Quality Audit

| Test File | Linked Requirements | Active Cases | Active Skips | Circular | Strongest Assertion | Verdict |
|-----------|---------------------|-------------:|-------------:|----------|---------------------|---------|
| `tests/test_blob_manifest.py` | STOR-01, SECU-05, MIGR-02, MIGR-07 | 39 | 0 | No | Exact bytes, boundaries, per-field HMAC mutation, native-library reads | ✓ STRONG |
| `tests/test_blob_manifest_backends.py` | STOR-01, STOR-08 | 17 | 0 | No | Exact-byte CRUD/reopen, transactional rollback, cause identity | ✓ STRONG |
| `tests/test_blob_store_read_contract.py` | STOR-02, STOR-08, SECU-03/04/05/08 | 47 | 0 | No | End-to-end state transitions, event order, zero-mutation failure matrices | ✓ STRONG |
| `tests/test_blob_store_integrity.py` | SECU-03, SECU-04, SECU-05 | 16 | 0 | No | Behavioral auth/snapshot ordering, key provenance, tamper evidence invariance | ✓ STRONG |
| `tests/test_blob_store_legacy_contract.py` | MIGR-07, STOR-08 | 26 | 0 | No | Independent pinned corpus, recursive digest/mtime invariance, historical HMAC | ✓ STRONG |
| `tests/test_blob_store_translation_seam.py` | SECU-08, STOR-08 | 25 | 0 | No | Exact public type/reason/category and source-level purity | ✓ STRONG |

Conditional platform/dependency skips are not active in this environment. The only
regression skip is the expected Windows-junction fixture on POSIX; other active tests
cover filesystem containment. Test writes are adversarial fixture setup, not expected
values generated by the production system under test. No circular oracle was found.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | No unreferenced `TBD`, `FIXME`, or `XXX`; no placeholder/stub implementation in the Phase 2 scope | — | None |

### Decision Coverage

All 14 trackable decisions from `02-CONTEXT.md` are honored by shipped artifacts.
The automated decision-coverage gate returned `honored: 14`, `total: 14`, and no
unhonored decisions.

### Human Verification Required

N/A — Infrastructure/foundation library phase with no user-facing visual or interactive
elements. Every state transition, ordering invariant, failure category, and prohibition
has passing automated behavioral evidence.

### Disconfirmation Pass

- **Potential partial scope:** Phase 2 deliberately admits only exact local JSON,
  memory, and SQLite manifest repositories. PostgreSQL/general metadata composition and
  S3/full backend-matrix parity are explicitly assigned to Phases 4-5. This does not
  weaken the Phase 2 backend-neutral manifest contract: every backend admitted here
  preserves the same bytes and semantics, and unsupported identities fail before
  mutation.
- **Potentially misleading evidence:** The historical Blosc2 array fixture contains the
  old bounded Cacheness shape/dtype framing; it is not evidence that this framing is a
  native Blosc2 container. Verification therefore relies on source and new-write tests
  for the no-wrapper truth, and classifies that fixture only as an exact read-only
  compatibility path required by D-12.
- **Error-path check:** Non-POSIX file-key attestation cannot be executed on this POSIX
  host, but production returns the explicit typed unsupported configuration before key
  or payload use. The supported POSIX path has active symlink, permissions, ownership,
  missing-key, race, partial-write, and reopen coverage. This is a platform capability
  branch, not an unverified phase behavior.

### Gaps Summary

No blocking or warning gap remains. All roadmap truths, plan artifacts, critical links,
nine requirements, three prohibitions, and behavior-dependent invariants are verified.

---

_Verified: 2026-08-30T16:30:02Z_
_Verifier: the agent (gsd-verifier)_
