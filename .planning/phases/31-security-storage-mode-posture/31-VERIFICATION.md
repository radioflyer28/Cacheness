---
phase: 31-security-storage-mode-posture
verified: 2026-06-15T14:16:40Z
status: passed
score: 7/7 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 6/7
  gaps_closed:
    - "SEC-03: Hard interruption after staged-key metadata/ciphertext persistence but before active-key replacement leaves entries readable under fresh UnifiedCache and BlobStore instances."
  gaps_remaining: []
  regressions: []
---

# Phase 31: Security & Storage-Mode Posture Verification Report

**Phase Goal:** Strengthen signing, encryption, rotation, and storage-mode durability guarantees without silently changing public API semantics.
**Verified:** 2026-06-15T14:16:40Z
**Status:** passed
**Re-verification:** Yes - after 31-08 SEC-03 gap closure

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | SEC-01: Signature downgrade hardening and unsigned-entry risk documentation are tested and documented. | VERIFIED | `SecurityConfig.minimum_signature_version` defaults to 1 and validates input; `CacheEntrySigner.verify_entry()` rejects versions below the configured minimum; UnifiedCache and BlobStore pass the configured value into signer construction. Tests in `tests/test_cache_signing.py` and `tests/test_key_rotation_api.py` cover valid v2 rejection under minimum v3 and v3-to-v2 downgrade mutation. `docs/SECURITY.md` recommends strict v3 and documents `allow_unsigned_entries=True` stripping risk. |
| 2 | SEC-02: Encrypted reads use backend-routed reads and prefer in-memory handler paths over plaintext temp files. | VERIFIED | `BlobStore._read_encrypted_blob()` reads ciphertext through `self.blob_backend.read_blob(backend_path)`, then calls `handler.get_bytes(plaintext, metadata)` before the `mkstemp` fallback. Tests cover encrypted memory backend reads, byte-path preference, and temp cleanup. |
| 3 | SEC-03: Interrupted key rotation leaves the original key and entries usable, including the hard-interruption window from the prior gap. | VERIFIED | 31-08 adds `UnifiedCache._init_interrupted_rotation_fallback()`, `VerificationMixin._verify_with_rotation_staged_signer()`, and BlobStore staged signer/encryption fallback. Fresh reopened readers require active key plus `<keyfile>.new`, verify/decrypt active-key-first, then use staged fallback without replacing keys, deleting staged key files, rewriting metadata, or deleting entries. Focused hard-interruption run: 4 passed. |
| 4 | SEC-04: UnifiedCache and BlobStore signing converge on a shared canonical field strategy with compatibility handling. | VERIFIED | `src/cacheness/signing_fields.py` provides shared canonical extraction; `_verification_mixin.py` and `blob_store.py` both use it. BlobStore verification still tries legacy flattened signatures after canonical verification, and the 31-08 staged signer path preserves that order for both active and staged signers. |
| 5 | STRG-01: Storage-mode destructive APIs use explicit warning-first protection without restoring implicit cache eviction behavior. | VERIFIED | `UnifiedCache._warn_storage_mode_destructive_api()` emits warning log plus `RuntimeWarning`; storage-mode cleanup, clear, namespace clear, and forced size cleanup call it. Tests assert warning-first behavior and no implicit TTL/size eviction in storage mode. |
| 6 | STRG-02: Storage-mode durability contract and opt-in fsync policy are explicit, wired, and default-off. | VERIFIED | `CacheStorageConfig.fsync_on_write` defaults to False and validates boolean type. The policy is threaded into JSON metadata, filesystem blob backend, and `WriteIntentJournal`. Docs distinguish atomic rename from power-loss durability and state tests only verify hook invocation. |
| 7 | STRG-03: Write intents are recorded before blob writes and stale cleanup is conservative. | VERIFIED | Cache-mode `put()` and storage-mode `_storage_mode_put()` record planned blob intents before `_write_blob()`. `WriteIntentJournal.cleanup_stale_intents()` tolerates missing planned blobs, resolves relative paths under `cache_dir`, and preserves committed metadata. Tests cover cache and storage pre-blob intent boundaries. |

**Score:** 7/7 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `src/cacheness/core.py` | UnifiedCache rotation, staged fallback sharing, storage warnings, write-intent ordering | VERIFIED | Initializes staged signer before namespace verification, shares staged signer/encryption key with internal BlobStore, leaves public rotation API unchanged, and retains previously verified storage behavior. |
| `src/cacheness/_verification_mixin.py` | UnifiedCache staged signature fallback before invalid-signature deletion | VERIFIED | `_verify_with_rotation_staged_signer()` is called after active signature failure and before integrity hooks or `delete_invalid_signatures` deletion. |
| `src/cacheness/storage/blob_store.py` | BlobStore staged signature and encrypted-read fallback | VERIFIED | Standalone stores initialize fallback from `<keyfile>.new`; signature verification tries active canonical/legacy, then staged canonical/legacy; encrypted reads retry staged key only after active decrypt failure. |
| `tests/test_key_rotation_api.py` | Hard-interruption signature regressions for UnifiedCache and BlobStore | VERIFIED | Tests use `HardInterruption(BaseException)` to bypass rollback and assert fresh reads return entries while metadata remains. |
| `tests/test_encryption_at_rest.py` | Hard-interruption encrypted UnifiedCache regression | VERIFIED | Encrypted fresh reader returns the staged-rotated value and preserves metadata under strict signing policy. |
| `tests/test_blob_store.py` | Hard-interruption encrypted BlobStore regression plus SEC-02 coverage | VERIFIED | Encrypted fresh BlobStore reader returns staged-rotated value; memory backend encrypted reads still avoid filesystem reads. |
| `docs/SECURITY.md` | Strict signature and unsigned-entry risk documentation | VERIFIED | Documents `minimum_signature_version=3` guidance and metadata-write signature stripping risk. |
| `docs/TRANSACTION_GUARANTEES.md` | Storage-mode destructive warning and durability contract | VERIFIED | Documents warning-first destructive APIs, fsync defaults, local-only scope, and power-loss caveat. |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `core.py` | `_verification_mixin.py` | `self._rotation_staged_signer` | WIRED | `gsd-tools query verify.key-links 31-08-PLAN.md` verified the pattern; source confirms signer exists before namespace and entry verification paths. |
| `core.py` | `blob_store.py` | shared staged signer and staged encryption key after BlobStore construction | WIRED | `_init_blob_store()` assigns `_rotation_staged_signer` and `_rotation_staged_encryption_key` to the internal BlobStore. |
| `blob_store.py` | `security.py` | `staged_key_file_path()` and `create_cache_signer()` | WIRED | Standalone BlobStore fallback uses the staged key helper and signer factory. |
| `blob_store.py` | blob backend | encrypted ciphertext reads | WIRED | `_read_encrypted_blob()` calls `self.blob_backend.read_blob(backend_path)`. |
| `config.py` | local writers/journal | `fsync_on_write` propagation | WIRED | Config reaches JSON backend, filesystem blob backend, BlobStore construction, and write-intent journal. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|---|---|---|---|---|
| `UnifiedCache` staged fallback | `_rotation_staged_signer` / `_rotation_staged_encryption_key` | `staged_key_file_path(active_key_path)` plus 32-byte `<keyfile>.new` | Yes | FLOWING |
| `VerificationMixin._verify_entry()` | `stored_signature` and signable fields | metadata entry -> canonical field extraction -> active signer -> staged signer fallback | Yes | FLOWING |
| `BlobStore._verify_entry_signature()` | `stored_signature` and metadata | backend metadata -> canonical/legacy extraction -> active signer -> staged signer fallback | Yes | FLOWING |
| `BlobStore._read_encrypted_blob()` | ciphertext and plaintext | blob backend ciphertext -> active decrypt -> staged decrypt fallback -> handler bytes/temp path | Yes | FLOWING |
| `CacheEntrySigner.verify_entry()` | signature version policy | `SecurityConfig.minimum_signature_version` -> signer factory -> signer instance | Yes | FLOWING |
| `record_intent()` | planned blob path | cache/storage put paths before `_write_blob()` | Yes | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| SEC-03 hard-interruption window | `uv run --python 3.12 pytest -o addopts='' -p no:cacheprovider --basetemp .uv-cache\pytest-31-verify-hard tests/test_key_rotation_api.py tests/test_encryption_at_rest.py tests/test_blob_store.py -k "hard_interruption_after_staged" -x -q --ignore=tests/test_tensorflow_handler.py` | 4 passed, 85 deselected | PASS |
| Phase 31 focused regression sweep | `uv run --python 3.12 pytest -o addopts='' -p no:cacheprovider --basetemp .uv-cache\pytest-31-verify-sweep tests/test_cache_signing.py tests/test_blob_store.py tests/test_key_rotation.py tests/test_key_rotation_api.py tests/test_storage_mode.py tests/test_atomic_writes.py tests/test_write_intent.py tests/test_encryption_at_rest.py tests/test_handler_bytes_protocol.py -x -q --ignore=tests/test_tensorflow_handler.py` | 203 passed, 3 skipped | PASS |

### Probe Execution

| Probe | Command | Result | Status |
|---|---|---|---|
| Conventional probes | `Get-ChildItem scripts -Recurse -Filter 'probe-*.sh' -File` | No probe scripts found | SKIPPED |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|---|---|---|---|---|
| SEC-01 | 31-01 | Minimum accepted signature version and unsigned-entry risk docs | SATISFIED | Code, docs, and tests verify configurable minimum and downgrade rejection. |
| SEC-02 | 31-02 | Encrypted backend-routed reads without avoidable plaintext temp files | SATISFIED | Code routes ciphertext through backend and tries bytes path first; tests cover memory backend and temp cleanup. |
| SEC-03 | 31-03, 31-08 | Interrupted key rotation leaves original key and entries usable | SATISFIED | 31-08 hard-interruption regressions prove fresh UnifiedCache and BlobStore reads work while active old key remains and `<keyfile>.new` remains. |
| SEC-04 | 31-04 | Shared canonical signing with compatibility path | SATISFIED | Shared helper and explicit legacy fallback are present and tested. |
| STRG-01 | 31-05 | Storage-mode destructive API warning policy | SATISFIED | Warning helper is wired to destructive APIs and tests assert warning-first behavior. |
| STRG-02 | 31-06 | Durability contract and opt-in fsync policy | SATISFIED | Config, local fsync helpers, JSON/blob/intent wiring, and docs are present and tested. |
| STRG-03 | 31-07 | Write intents before blob writes | SATISFIED | Cache and storage put paths record planned blob intent before `_write_blob()` and tests cover cleanup guards. |

No orphaned Phase 31 requirements were found in `.planning/REQUIREMENTS.md`.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---:|---|---|---|
| `tests/test_blob_store.py` and `tests/test_encryption_at_rest.py` | multiple | Empty-list assertions such as `direct_blob_writes == []` | INFO | Test assertions and local accumulators, not stubs. |
| `src/cacheness/core.py`, `_verification_mixin.py`, `blob_store.py` | multiple | Empty dict/list initializers | INFO | Runtime accumulators for rollback, reporting, or local state, not hardcoded user-visible data. |

No unreferenced `TBD`, `FIXME`, or `XXX` debt markers were found in the 31-08 source/test files. No placeholder or console-log-only implementations were found.

### Human Verification Required

None.

### Gaps Summary

No blocking gaps remain. The prior SEC-03 gap is closed: a hard interruption after staged metadata/signature/ciphertext persistence but before active-key replacement leaves the old active key unchanged, leaves `<keyfile>.new` present, and fresh `UnifiedCache` and `BlobStore` readers can still return signed and encrypted entries without default deletion.

---

_Verified: 2026-06-15T14:16:40Z_
_Verifier: the agent (gsd-verifier)_
