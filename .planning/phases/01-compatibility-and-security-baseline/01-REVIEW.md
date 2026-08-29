---
phase: 01-compatibility-and-security-baseline
reviewed: 2026-08-29T23:35:35Z
depth: standard
files_reviewed: 54
files_reviewed_list:
  - README.md
  - docs/SECURITY.md
  - src/cacheness/__init__.py
  - src/cacheness/config.py
  - src/cacheness/core.py
  - src/cacheness/decorators.py
  - src/cacheness/error_handling.py
  - src/cacheness/handlers.py
  - src/cacheness/interfaces.py
  - src/cacheness/metadata.py
  - src/cacheness/query_validation.py
  - src/cacheness/security.py
  - src/cacheness/sql_cache.py
  - src/cacheness/storage/backends/blob_backends.py
  - src/cacheness/storage/blob_store.py
  - src/cacheness/storage/guarded_handler_io.py
  - src/cacheness/storage/path_security.py
  - tests/fixtures/compat/README.md
  - tests/fixtures/compat/array-raw-v035-compress/payload.b2nd
  - tests/fixtures/compat/array-raw-v037-compress2/payload.b2nd
  - tests/fixtures/compat/decorator-key-v0313/metadata.json
  - tests/fixtures/compat/decorator-key-v0313/payload.npz
  - tests/fixtures/compat/decorator-key-v0313/provenance.json
  - tests/fixtures/compat/json-nested-v0314/metadata.json
  - tests/fixtures/compat/json-nested-v0314/payload.npz
  - tests/fixtures/compat/json-nested-v0314/provenance.json
  - tests/fixtures/compat/json-split-signed-v038/metadata.json
  - tests/fixtures/compat/json-split-signed-v038/payload.npz
  - tests/fixtures/compat/json-split-signed-v038/provenance.json
  - tests/fixtures/compat/json-split-unsigned-v037/metadata.json
  - tests/fixtures/compat/json-split-unsigned-v037/payload.npz
  - tests/fixtures/compat/json-split-unsigned-v037/provenance.json
  - tests/fixtures/compat/manifest.json
  - tests/fixtures/compat/sqlite-columns-v0314/metadata.sqlite3
  - tests/fixtures/compat/sqlite-columns-v0314/payload.npz
  - tests/fixtures/compat/sqlite-columns-v0314/provenance.json
  - tests/fixtures/compat/sqlite-metadata-json-v039/metadata.sqlite3
  - tests/fixtures/compat/sqlite-metadata-json-v039/payload.npz
  - tests/fixtures/compat/sqlite-metadata-json-v039/provenance.json
  - tests/fixtures/compat/validate_corpus.py
  - tests/test_blob_backend_registry.py
  - tests/test_config_validation.py
  - tests/test_core.py
  - tests/test_directory_sharding.py
  - tests/test_filesystem_containment.py
  - tests/test_legacy_array_security.py
  - tests/test_phase1_quality_gates.py
  - tests/test_public_api_contract.py
  - tests/test_query_meta.py
  - tests/test_query_meta_security.py
  - tests/test_security_documentation.py
  - tests/test_sql_cache.py
  - tests/test_sql_cache_failure_contract.py
  - tests/test_stored_compatibility.py
findings:
  critical: 5
  warning: 0
  info: 0
  total: 5
status: issues_found
---

# Phase 1: Code Review Report

**Reviewed:** 2026-08-29T23:35:35Z
**Depth:** standard
**Files Reviewed:** 54
**Status:** issues_found

## Summary

The Phase 1 compatibility corpus validates successfully and the focused quality, containment, and metadata-query suites pass, but adversarial review found five release-blocking defects. Two independently bypass the promised executable-serializer authenticity/integrity gate, one returns type-confused metadata query results, one leaves blob bytes behind after a successful clear, and one permits handler output outside the private staging directory to cross the managed publication boundary.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: Missing digests silently disable required integrity verification

**Classification:** BLOCKER

**File:** `src/cacheness/core.py:990-1004,1244-1248`

**Issue:** When integrity verification is enabled, `_calculate_file_hash()` is allowed to return `None`; `put()` persists that value and still signs/commits the entry. On read, `get()` performs the digest comparison only when `stored_hash is not None`, so the supposedly mandatory gate is skipped. This reaches executable `ObjectHandler` deserialization even under the complete trusted-object-array policy. A direct reproduction that replaced `_calculate_file_hash` with a failing/`None` result stored an object array with `file_hash=None` and then successfully deserialized it. This contradicts the Phase 1 requirement that object arrays require payload integrity verification.

**Fix:** Treat an unavailable digest as a write failure whenever `verify_cache_integrity` is enabled, and reject existing entries with a missing or malformed digest before signature verification or handler dispatch. Do not represent “verification requested but unavailable” with the same `None` value as “verification disabled.”

### CR-02: Custom signed-field sets can authenticate tampered executable payloads

**Classification:** BLOCKER

**File:** `src/cacheness/config.py:590-615`

**Issue:** The trusted-object-array validator checks that signing is enabled, integrity checking is enabled, and unsigned entries are rejected, but it does not require the signature to cover `file_hash`, `data_type`, `actual_path`, `storage_format`, `serializer`, or `compression_codec`. `SecurityConfig(custom_signed_fields=["cache_key"])` therefore passes the strict opt-in validation. An attacker who can alter cache payload and metadata can replace an object-array payload, update the unsigned `file_hash` to the replacement digest, retain the original valid cache-key-only HMAC, and cause the replacement pickle to deserialize. This was reproduced by copying a second valid object-array payload over the first and changing only its stored hash; `get(first_key)` returned the second payload.

**Fix:** When `allow_trusted_object_arrays=True`, reject custom signer configurations unless they include every field that controls payload identity and deserialization, at minimum `cache_key`, `file_hash`, `data_type`, `actual_path`, `storage_format`, `serializer`, and `compression_codec`. Prefer a fixed mandatory signed-field core that custom fields may extend but never remove.

### CR-03: Numeric metadata filters include nonnumeric values

**Classification:** BLOCKER

**File:** `src/cacheness/core.py:634-642`

**Issue:** Numeric `query_meta` filters strip everything through the first colon and cast the remainder to SQLite `FLOAT`, but never check the stored type prefix. SQLite casts nonnumeric strings such as `str:oops` to `0`. Consequently `query_meta(score=-1)` matches a record whose stored score is the string `"oops"`, while excluding a numeric `-2` record. This violates the documented raw numeric-versus-string semantics and can return materially incorrect cache metadata.

**Fix:** Add a bound type predicate before casting, accepting only the exact numeric encodings (`int:` and `float:`), or store/query JSON values with an explicit type discriminator. Keep the caller path and prefix values bound rather than interpolated.

### CR-04: BlobStore.clear reports deletion while retaining every payload

**Classification:** BLOCKER

**File:** `src/cacheness/storage/blob_store.py:365-374`

**Issue:** `clear()` preflights entries and then calls only `self.backend.clear_all()`. Metadata backends remove records, not payload files. The method returns the number of “blobs removed” even though all managed payload bytes remain on disk and become unreachable orphans. A direct reproduction returned `1`, removed the metadata, and left the payload path present. This is a data-retention and storage-correctness failure, and the Phase 1 plan explicitly required clear to route deletion through guarded operations after complete preflight.

**Fix:** After the full preflight pass, delete each validated locator through `guarded_handler_io.file_ops`, then clear metadata using a failure policy that does not falsely report success or silently orphan bytes. Add a safe-entry regression asserting both metadata and payload removal.

### CR-05: Lexical stage containment accepts `..` artifacts outside the private directory

**Classification:** BLOCKER

**File:** `src/cacheness/storage/guarded_handler_io.py:82-115`

**Issue:** `_staged_artifact()` uses lexical `artifact.relative_to(stage_root)` without resolving or rejecting `..`. A returned path such as `<stage>/../payload.npz` produces the relative path `../payload.npz`; the component loop accepts the ordinary parent directory and regular external file, and `_safe_suffix()` accepts its `payload` basename. Bytes outside the private handler stage can therefore be published through the managed boundary despite the stated containment contract. The symlink checks do not address parent traversal or hard-linked external files.

**Fix:** Reject `..` components before any filesystem access, resolve the candidate strictly and require it to be a descendant of the resolved stage root, then open it with a no-follow descriptor and verify that same descriptor is a regular file before copying. Add regressions for relative and absolute parent traversal and for a path swapped after validation.

---

_Reviewed: 2026-08-29T23:35:35Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
