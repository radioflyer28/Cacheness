# Phase 17: Key Rotation API - Context

**Gathered:** 2026-04-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Manual key rotation API — `rotate_key()` loads a new signing key, re-derives HKDF namespace keys, and re-signs all existing entries with the new key. Covers both cache mode (UnifiedCache) and storage mode (BlobStore). Includes v2→v3 migration re-signing deferred from Phase 16.

</domain>

<decisions>
## Implementation Decisions

### API Surface
- **D-01:** `rotate_key()` lives on **both** `UnifiedCache` and `BlobStore`. Both modes have independent signer wiring and need rotation support.
- **D-02:** Accepts `new_key_file: str | Path` parameter — user provides path to a pre-generated key file. No auto-generation.
- **D-03:** Returns a structured `RotationResult` dataclass with counts: `total`, `re_signed`, `failed`, `skipped` (and optionally `failures` list with details).
- **D-04:** Also handles v2→v3 migration (Phase 16 deferred item D-04). All entries re-signed as v3 with new HKDF-derived key. One API for both rotation and migration.

### Atomicity & Failure Handling
- **D-05:** Best-effort with progress report. If re-signing entry N fails (corrupt entry, I/O error), log it, skip, continue to next. All failures reported in `RotationResult.failures`.
- **D-06:** Hold `self._lock` for the entire rotation operation. Other threads block on `put()`/`get()` until done. Simple and safe.
- **D-07:** Forward-compatible crash recovery. New key file is loaded first. If crash mid-rotation, some entries have old signatures, some new. The existing signature version routing (`verify_entry` routes v1/v2→master_key, v3→derived_key) handles mixed state. Log warning on next init if rotation appears incomplete.

### Old-Key Entry Behavior
- **D-08:** Entries signed with old key still verify correctly via existing version routing in `verify_entry()` and `verify_namespace()`. No special dual-key verification needed — the signer already handles v1/v2 vs v3 routing.
- **D-09:** `rotate_key()` is idempotent — re-running picks up entries that weren't re-signed (entries still on old version get re-signed, already-migrated entries are skipped or re-signed again with same result).

### BlobStore (Storage Mode)
- **D-10:** `BlobStore.rotate_key()` with same semantics as `UnifiedCache.rotate_key()`. Iterate entries via metadata backend, re-sign each with new derived key.
- **D-11:** BlobStore has its own signer (`self.signer`) wired via `_init_signer()` — rotation replaces the signer instance with one using the new key.

### Namespace Signatures
- **D-12:** `rotate_key()` also re-signs namespace registry rows (ns1→ns2 with new derived key).

### Agent's Discretion
- `RotationResult` dataclass location (in `security.py`, `interfaces.py`, or `core.py`)
- Whether to add a `rotation_incomplete` flag to detect crash-interrupted rotations
- Exact log message wording
- Test file structure (new `test_key_rotation_api.py` or extend `test_key_rotation.py`)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Security Architecture
- `src/cacheness/security.py` — `CacheEntrySigner`, `_hkdf_sha256()`, `sign_entry()`, `verify_entry()`, `sign_namespace()`, `verify_namespace()`, `create_cache_signer()`, signature version routing (v1/v2→master_key, v3→derived_key)
- `src/cacheness/config.py` — `SecurityConfig` dataclass with `use_hkdf_derivation`, `key_fallback_policy`, `signing_key_file`
- `src/cacheness/core.py` — `_init_entry_signer()` (~L330), `_sign_current_namespace()` (~L360), `_extract_signable_fields()`, re-sign pattern in update operations (~L2391-2427)
- `src/cacheness/storage/blob_store.py` — `_init_signer()`, entry signing in `put()`, verification in `get()`
- `src/cacheness/interfaces.py` — `SignableFields` TypedDict, `EntrySummary` TypedDict
- `src/cacheness/error_handling.py` — `CacheSecurityError`

### Metadata Iteration
- `src/cacheness/metadata/base.py` — `iter_entry_summaries()` (~L169), `get_entry()`, `put_entry()`
- `src/cacheness/core.py` — existing usage of `iter_entry_summaries()` for batch ops (~L489, L2607, L2685)

### Phase 15-16 Context (predecessors)
- `.planning/phases/15-configurable-key-fallback/15-CONTEXT.md` — `key_fallback_policy` decisions
- `.planning/phases/16-hkdf-key-derivation/16-CONTEXT.md` — HKDF decisions, D-04 (deferred re-sign utility)

### Existing Tests
- `tests/test_key_rotation.py` — key deletion/regeneration scenarios, strict rejection mode
- `tests/test_hkdf_derivation.py` — HKDF function tests, signature versioning, config integration
- `tests/test_namespace_signing.py` — namespace signing/verification, cross-namespace isolation
- `tests/test_cache_signing.py` — in-memory key tests

### Requirement
- SEC-04 in `.planning/REQUIREMENTS.md`

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `CacheEntrySigner` — already stores `master_key` + `derived_key` with HKDF derivation
- `_hkdf_sha256(ikm, info, length, salt)` — stdlib HKDF function ready for re-derivation
- `iter_entry_summaries()` — lightweight metadata iteration for batch operations
- `_extract_signable_fields()` — builds `SignableFields` dict from entry data
- `sign_entry()` / `sign_namespace()` — already handle v2 vs v3 version selection
- `parse_versioned_signature()` — extracts version from stored signature string
- Re-sign pattern in `core.py` update operations (~L2391-2427) — can be extracted/reused

### Established Patterns
- Signature format: `v{N}:{hex}` for entries, `ns{N}:{hex}` for namespaces
- HKDF info parameter: `f"cacheness-ns-v1:{namespace_id}".encode("utf-8")`
- `self._lock` for thread-safe operations (threading.RLock)
- `RotationResult`-style return objects: see `put_batch()` returning counts

### Integration Points
- `UnifiedCache.rotate_key(new_key_file)` — new method on core class
- `BlobStore.rotate_key(new_key_file)` — new method on storage mode class
- `CacheEntrySigner` — may need method to re-initialize with new key (or create new instance)
- `metadata_backend.get_entry()` + `metadata_backend.put_entry()` — read/update entries during re-sign
- `self.signer` replacement — swap signer instance after loading new key

</code_context>

<specifics>
## Specific Ideas

- The signer already handles mixed v2/v3 state in `verify_entry()` — rotation creates a new signer with the new key, then iterates all entries to re-sign them as v3.
- For each entry during re-sign: extract signable fields via `_extract_signable_fields()`, call `signer.sign_entry()` to get new v3 signature, update metadata via `put_entry()`.
- Namespace re-signing follows same pattern: iterate namespaces, call `sign_namespace()`, update registry.
- After rotation completes, `self.signer` points to the new signer — all subsequent operations use the new key.
- If crash mid-rotation: next `__init__` creates signer with the new key (from file). Old v2 entries still verify via version routing. The cache works but with mixed signatures until `rotate_key()` is re-run.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 17-key-rotation-api*
*Context gathered: 2026-04-03*
