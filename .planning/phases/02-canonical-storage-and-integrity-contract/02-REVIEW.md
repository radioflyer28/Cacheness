---
phase: 02-canonical-storage-and-integrity-contract
reviewed: 2026-08-30T15:20:18Z
depth: standard
files_reviewed: 20
files_reviewed_list:
  - src/cacheness/error_handling.py
  - src/cacheness/handlers.py
  - src/cacheness/interfaces.py
  - src/cacheness/security.py
  - src/cacheness/storage/__init__.py
  - src/cacheness/storage/blob_store.py
  - src/cacheness/storage/integrity.py
  - src/cacheness/storage/legacy_manifest.py
  - src/cacheness/storage/manifest.py
  - src/cacheness/storage/manifest_repository.py
  - src/cacheness/storage/read_contract.py
  - tests/test_blob_manifest.py
  - tests/test_blob_manifest_backends.py
  - tests/test_blob_store_integrity.py
  - tests/test_blob_store_legacy_contract.py
  - tests/test_blob_store_read_contract.py
  - tests/test_blob_store_translation_seam.py
  - tests/test_clear_recovery.py
  - tests/test_filesystem_containment.py
  - tests/test_public_api_contract.py
findings:
  critical: 10
  warning: 2
  info: 0
  total: 12
status: issues_found
---

# Phase 02: Code Review Report

**Reviewed:** 2026-08-30T15:20:18Z
**Depth:** standard
**Files Reviewed:** 20
**Status:** issues_found

## Summary

The Phase 02-focused tests (117 cases) and targeted Ruff gate pass, and native NPZ, pickle/dill, and Parquet payloads remain handler-owned without a Cacheness wrapper. Adversarial review nevertheless found ten shipping blockers: custom handlers can write unreadable records, outgoing manifest bounds disagree with read bounds, present metadata can collapse into absence, some backend faults escape the typed taxonomy, metadata updates can re-sign unsafe locators, key creation is racy, critical fields are validated before authentication, legacy evidence recognition is neither contained nor authentic, and SQLite publication is split across transactions. Two resource/observability defects are also present.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: Custom handlers write payload identities that their own read contract rejects

**Classification:** BLOCKER
**File:** `src/cacheness/storage/blob_store.py:328-330`
**Issue:** `CacheHandler.payload_format` explicitly promises a `data_type` fallback for custom handlers, but `BlobStore.put()` falls back to the handler result's `storage_format`. A registered custom handler that returns `storage_format="text"` and has `data_type="custom"` writes a manifest with `payload_format="text"`; the default `supports_payload_contract()` then requires `"custom"`, so the immediately following `get()` raises `CacheBlobPayloadUnsupportedVersionError`. The version fallback is likewise hardcoded to `1` instead of the handler's declared version.
**Fix:** Derive missing identity fields from the handler contract:
```python
payload_format = result.get("payload_format", handler.payload_format)
payload_format_version = result.get(
    "payload_format_version", handler.payload_format_version
)
```
Add a round-trip test using a public `HandlerRegistry` and a custom handler that returns only the historically required result fields.

### CR-02: Write-side manifests can exceed the total-node bound and become unreadable

**Classification:** BLOCKER
**File:** `src/cacheness/storage/manifest.py:242-243`
**Issue:** Model construction validates `handler_metadata` and `user_metadata` with separate node counters, while decoding validates the entire document with one counter at line 352. A manifest with two individually valid metadata maps can therefore serialize successfully but fail `from_canonical_bytes()` with `MANIFEST_BOUNDS`. An adversarial probe produced a 77,892-byte outgoing record that was accepted by `canonical_bytes()` and rejected immediately on read for exceeding `MAX_TOTAL_NODES`.
**Fix:** Validate the complete outgoing mapping with one shared node counter before returning canonical bytes (including top-level keys/values), and use that same routine for decoding. Add a write-then-read boundary test where the combined metadata maps cross the limit while each map remains below it.

### CR-03: A present backend record without raw canonical bytes is reported as absence

**Classification:** BLOCKER
**File:** `src/cacheness/storage/manifest_repository.py:80-85`
**Issue:** JSON and in-memory repositories return `None` when an entry exists but lacks `canonical_manifest_v1`. SQLite similarly returns `None` when its sidecar row is absent at lines 195-196 without checking `cache_entries`. `BlobStore` defines `None` as a genuinely absent key, so partial writes, old stores without the exact fixture provenance file, and damaged sidecars silently become cache misses instead of migration-required, lifecycle-conflict, malformed, or backend failures.
**Fix:** Reserve `None` for absence in both the canonical repository and compatibility metadata. If compatibility metadata exists without a canonical record, raise a typed migration/lifecycle/backend error. For SQLite, check the paired metadata row when the BLOB row is missing. Add JSON, memory, and SQLite tests for “metadata present / canonical bytes absent.”

### CR-04: `list()` and `clear()` bypass typed backend-failure translation

**Classification:** BLOCKER
**File:** `src/cacheness/storage/blob_store.py:594`
**Issue:** `list()` calls `self.backend.list_entries()` directly; `_preflight_clear_manifests()` does the same at line 1021. An `OSError` from that backend projection escapes unchanged, despite the public Phase 02 contract requiring `CacheBlobBackendError` on every direct surface. Existing tests inject failures only through `manifest_repository.get_raw`, so they miss this path.
**Fix:** Put all required projection access behind a repository method that translates the backend's documented operational exceptions, or catch the narrow backend exceptions here and raise `CacheBlobBackendError` with the original cause. Test both projection failure sites for all three local backends.

### CR-05: `update_metadata()` re-signs an unsafe locator without containment validation

**Classification:** BLOCKER
**File:** `src/cacheness/storage/blob_store.py:487-500`
**Issue:** Every other locator-consuming operation requests `require_locator=True`, but metadata update authenticates without it. A correctly signed manifest whose locator is outside the current store can be updated successfully and re-signed, preserving the unsafe locator as current canonical evidence. This contradicts the claimed authenticated critical-field gate and allows mutation before containment validation.
**Fix:** Call `_load_authenticated_manifest(..., require_locator=True)` before constructing the replacement record. Add a test with an authenticated outside-root locator and assert no repository write occurs.

### CR-06: Concurrent first-writer key initialization returns `None`

**Classification:** BLOCKER
**File:** `src/cacheness/storage/integrity.py:108-115`
**Issue:** If another writer creates the HMAC key after `get_key()` reports it missing but before this writer's `O_EXCL` open, `initialize_new_store()` catches `FileExistsError` and returns `None` despite its `bytes` contract. Signing then fails key validation instead of loading the winner's key. This breaks the project same-store concurrency requirement on a normal first-write race.
**Fix:** On `FileExistsError`, call and return `_read_existing_key()` so the winning file is fully attested. Add a deterministic race test that inserts a valid key between the initial read and exclusive create. Also remove a newly created partial file when write/fsync fails, or make incomplete initialization recoverable without manual deletion.

### CR-07: Semantic critical-field validation occurs before HMAC authentication

**Classification:** BLOCKER
**File:** `src/cacheness/storage/blob_store.py:910-928`
**Issue:** `from_canonical_bytes()` constructs `BlobManifestV1`, whose `__post_init__` validates lifecycle state, digest algorithm/value, signature syntax, size, and metadata structure before `verify_hmac_sha256()` runs. Consequently, changing a signed digest to a non-hex value is classified as malformed, while changing it to another hex value is classified as unauthenticated. This violates the specified order of bounded parse/version dispatch, authenticity, then critical-field validation and makes typed outcomes depend on attacker-chosen field syntax.
**Fix:** Split decoding into a bounded/version-dispatched raw record and a post-authentication semantic model. Build the deterministic signing projection from the bounded raw record, verify HMAC, then construct/validate `BlobManifestV1`. Keep only the minimal syntax needed to locate the signature and dispatch the schema before authentication.

### CR-08: Legacy “exact” recognition follows symlinks outside the fixture root

**Classification:** BLOCKER
**File:** `src/cacheness/storage/legacy_manifest.py:198-203`
**Issue:** `_require_exact_files()` uses `Path.is_file()`, which follows symlinks. Metadata, SQLite, provenance, or payload names can therefore reference files outside the supplied legacy root and still participate in identity recognition. This bypasses the project's path-containment boundary in the new compatibility seam.
**Fix:** Inspect every fixed name with `lstat`, reject symlinks and non-regular files, resolve/open it relative to a retained root descriptor with no-follow semantics, and prove the descriptor remains below the root. Add symlink cases for every evidence kind.

### CR-09: Signed legacy evidence is never authenticated and fixture identity is spoofable

**Classification:** BLOCKER
**File:** `src/cacheness/storage/legacy_manifest.py:219-229`
**Issue:** The signed v0.3.8 recognizer checks only that signatures look like 64 lowercase hex characters; it never invokes the newly added `verify_legacy_v038_entry()`. More generally, recognition trusts self-declared `fixture_id`/`source_version` and metadata shape but does not verify the provenance file hashes or reject extra evidence. A wrong-but-well-formed signature or arbitrary payload is therefore attached to an “exact” in-memory legacy identity and reported as migration-ready evidence.
**Fix:** Require trusted legacy key material when authenticating the signed layout and verify every entry using the exact historical verifier. Verify the bounded provenance digest allowlist (or rename this API to shape classification and never claim exact evidence identity), reject unrecognized extra files/sidecars, and test a one-nibble valid-hex signature mutation plus payload/metadata digest mutations.

### CR-10: SQLite manifest publication is split across independent transactions

**Classification:** BLOCKER
**File:** `src/cacheness/storage/manifest_repository.py:209-223`
**Issue:** SQLite `put_raw()` first commits `cache_entries` through `backend.put_entry()` and only then opens another transaction to upsert canonical bytes. If the second operation fails, `BlobStore.put()` deletes the candidate payload but leaves a committed compatibility row pointing to it; the canonical row is old or absent. This violates atomic/rollback semantics and feeds directly into CR-03's false-absence behavior.
**Fix:** Publish the compatibility projection and canonical BLOB in one SQLite transaction, or snapshot and reliably roll back the compatibility row if the BLOB write fails. Add fault injection between the two writes for both fresh puts and overwrites, asserting exact pre-operation state after failure.

## Warnings

### WR-01: Constructor failures leak the managed-root descriptor

**Classification:** WARNING
**File:** `src/cacheness/storage/blob_store.py:180-220`
**Issue:** `GuardedHandlerIO` opens a managed-root descriptor before legacy recognition and repository selection. If malformed provenance raises or `create_manifest_repository()` rejects an injected backend, construction exits without closing that descriptor. Repeated invalid-store probes can exhaust process descriptors.
**Fix:** Stage constructor resources under `contextlib.ExitStack` (or a guarded `try/except`) and close `guarded_handler_io` on every initialization failure before ownership transfers to the completed object. Add an fd-count or close-spy test for malformed legacy and unsupported backend paths.

### WR-02: A successful first write emits a public error log

**Classification:** WARNING
**File:** `src/cacheness/storage/blob_store.py:788-800`
**Issue:** First-store initialization calls `get_key()`, constructs/logs `ManifestKeyError` for the expected missing file, catches it, and successfully creates the key. Because every `CacheError` logs at construction, normal first writes emit “Cache error: Unable to read canonical manifest key,” creating false security alerts.
**Fix:** Give the provider an explicit atomic `get_or_initialize_new_store()` path that treats `ENOENT` as expected internal control flow without constructing a logged public failure; log only when initialization ultimately fails.

---

_Reviewed: 2026-08-30T15:20:18Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
