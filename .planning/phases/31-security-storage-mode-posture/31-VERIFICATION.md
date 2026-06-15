---
phase: 31-security-storage-mode-posture
verified: 2026-06-15T03:47:33Z
status: gaps_found
score: 6/7 must-haves verified
overrides_applied: 0
gaps:
  - truth: "SEC-03: User can rotate keys with a two-phase process that leaves the original key and entries usable if rotation is interrupted."
    status: failed
    reason: "Rotation re-signs and persists entries with the staged new key before publishing the staged key file. A process interruption after metadata commit but before key-file replacement leaves the active key unchanged while entries require the staged key; default reads reject and delete the entry."
    artifacts:
      - path: "src/cacheness/core.py"
        issue: "UnifiedCache.rotate_key writes staged-key signatures to metadata before os.replace(staged_key_path, dest), so uncaught process interruption can leave metadata ahead of the active key."
      - path: "src/cacheness/storage/blob_store.py"
        issue: "BlobStore.rotate_key has the same metadata-before-key-publication crash window."
      - path: "tests/test_key_rotation_api.py"
        issue: "Interrupted-rotation tests cover controlled exceptions that trigger rollback, not hard interruption after staged-key metadata is committed."
    missing:
      - "Make rotation crash-safe across the metadata-before-key-publication window: either publish metadata atomically with key replacement, make startup recover/complete from <keyfile>.new, or keep committed entries verifiable by the old key until the key publish is durable."
      - "Add a regression that simulates process interruption after one entry is signed with the staged key and before os.replace(staged_key_path, dest); reopening with the active old key must still read the entry under default security policy."
---

# Phase 31: Security & Storage-Mode Posture Verification Report

**Phase Goal:** Strengthen signing, encryption, rotation, and storage-mode durability guarantees without silently changing public API semantics.
**Verified:** 2026-06-15T03:47:33Z
**Status:** gaps_found
**Re-verification:** No - initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | SEC-01: Minimum signature versions reject downgrades and unsigned-entry risk is documented. | VERIFIED | `SecurityConfig.minimum_signature_version` defaults to 1 and validates input; `CacheEntrySigner.verify_entry()` rejects versions below the configured minimum; UnifiedCache and BlobStore pass the configured value into signer construction. Tests cover valid v2 rejection under minimum v3 and v3-to-v2 downgrade mutation. Docs recommend `minimum_signature_version=3` and warn that `allow_unsigned_entries=True` lets metadata writers strip signatures. |
| 2 | SEC-02: Encrypted reads use backend-routed reads and prefer in-memory handler paths over plaintext temp files. | VERIFIED | `BlobStore._read_encrypted_blob()` calls `self.blob_backend.read_blob(backend_path)`, then tries `handler.get_bytes(plaintext, metadata)` before the `mkstemp` fallback; fallback temp files are under `cache_dir`, chmod 0600 on POSIX, and unlinked in `finally`. Tests cover encrypted memory backend reads, byte path preference, and fallback cleanup. |
| 3 | SEC-03: Interrupted key rotation leaves the original key and entries usable. | FAILED | `src/cacheness/core.py:575-582` and `src/cacheness/storage/blob_store.py:468-472` persist staged-key signatures before `os.replace(staged_key_path, dest)` at `core.py:598` / `blob_store.py:475`. A probe simulating interruption in that window reopened with the old active key: `manual_verifies_with_active_old_key=False`, `read_after_simulated_crash_default_policy=None`, and the entry was deleted. |
| 4 | SEC-04: UnifiedCache and BlobStore share canonical signing fields with legacy compatibility. | VERIFIED | `src/cacheness/signing_fields.py` provides `extract_signable_fields()` and `extract_legacy_blobstore_signable_fields()`; `_verification_mixin.py` delegates to the shared helper; BlobStore new writes and rotation use canonical fields and verification tries canonical first, then explicit legacy fallback. Tests cover canonical field parity and old flattened BlobStore signatures. |
| 5 | STRG-01: Storage-mode destructive APIs use explicit warning-first protection without restoring implicit cache eviction behavior. | VERIFIED | `UnifiedCache._warn_storage_mode_destructive_api()` emits logger warning plus `RuntimeWarning`; `clear_all()`, `clear_all_namespaces()`, explicit `cleanup_expired()`, and size enforcement call it. Tests assert warning-first behavior and no implicit TTL/size eviction in storage mode. |
| 6 | STRG-02: Storage-mode durability contract and opt-in fsync policy are explicit, wired, and default-off. | VERIFIED | `CacheStorageConfig.fsync_on_write` defaults to False and validates boolean type. The policy is threaded into JSON metadata, filesystem blob backend, and `WriteIntentJournal`, using `_durability.flush_and_fsync()` and best-effort parent directory fsync. Docs distinguish atomic rename from power-loss durability and state tests do not prove real power-loss survival. |
| 7 | STRG-03: Write intents are recorded before blob writes and stale cleanup is conservative. | VERIFIED | `UnifiedCache.put()` and `_storage_mode_put()` call `record_intent()` before `_blob_store._write_blob()`. `WriteIntentJournal.cleanup_stale_intents()` resolves relative paths under `cache_dir`, removes only the intent for committed keys, and tolerates missing planned blobs. Tests assert intent payload exists at the `_write_blob` boundary for cache and storage mode. |

**Score:** 6/7 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `src/cacheness/config.py` | `minimum_signature_version` and `fsync_on_write` config | VERIFIED | Both fields exist, default compatibly, and are validated. |
| `src/cacheness/security.py` | Minimum-version enforcement and staged key helpers | PARTIAL | Minimum-version enforcement exists. Staged key helpers exist, but rotation consumers still have a hard-interruption metadata/key ordering gap. |
| `src/cacheness/signing_fields.py` | Shared canonical signing helper | VERIFIED | Imported by UnifiedCache verification and BlobStore. |
| `src/cacheness/storage/blob_store.py` | Backend-routed encrypted reads, canonical signing, BlobStore rotation | PARTIAL | SEC-02 and SEC-04 are wired. SEC-03 rotation has the same crash window as UnifiedCache. |
| `src/cacheness/core.py` | UnifiedCache rotation, storage warnings, write-intent ordering | PARTIAL | Storage warnings and write-intent ordering are verified. Rotation persists staged-key metadata before key publication. |
| `src/cacheness/_storage_mode_mixin.py` | Storage-mode pre-blob intent and no-delete reads | VERIFIED | Records intent before blob write and calls `_verify_entry(..., storage_mode=True)`. |
| `src/cacheness/write_intent.py` | Conservative cleanup and fsync support | VERIFIED | Handles committed metadata guard, missing blobs, relative path resolution, and opt-in fsync. |
| `src/cacheness/_durability.py` | Local fsync helpers | VERIFIED | Provides file fsync and best-effort parent-dir fsync. |
| `docs/SECURITY.md` | Unsigned-entry risk and strict signature guidance | VERIFIED | Includes strict v3 guidance and `allow_unsigned_entries=True` stripping risk. |
| `docs/TRANSACTION_GUARANTEES.md` | Storage-mode destructive warnings and durability contract | VERIFIED | Documents warning-first destructive APIs and local-only `fsync_on_write`. |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `config.py` | `security.py` | signer construction | VERIFIED | `minimum_signature_version` is passed into `create_cache_signer()` from UnifiedCache and BlobStore. |
| `blob_store.py` | `BlobBackend.read_blob` | encrypted ciphertext reads | VERIFIED | `_read_encrypted_blob()` reads ciphertext via `self.blob_backend.read_blob(backend_path)`. |
| `blob_store.py` | handler `get_bytes` | plaintext in-memory deserialize | VERIFIED | `handler.get_bytes()` is attempted before temp fallback. |
| `core.py` / `blob_store.py` | `security.py` staged keys | key staging | PARTIAL | Staged key construction is wired, but metadata commits are not interruption-safe before key publication. |
| `_verification_mixin.py` / `blob_store.py` | `signing_fields.py` | canonical helper import | VERIFIED | Both use shared canonical extraction; BlobStore has explicit legacy fallback. |
| `core.py` / `_storage_mode_mixin.py` | `write_intent.py` | pre-blob intent | VERIFIED | Intent is recorded before `_write_blob()` in both put paths. |
| `config.py` | local file writers | `fsync_on_write` propagation | VERIFIED | Policy reaches JSON backend, filesystem blob backend, BlobStore construction, and write-intent journal. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|---|---|---|---|---|
| `CacheEntrySigner.verify_entry()` | `minimum_signature_version` | `SecurityConfig` -> signer factory -> signer instance | Yes | FLOWING |
| `BlobStore._read_encrypted_blob()` | ciphertext / plaintext | `blob_backend.read_blob()` -> decrypt -> `handler.get_bytes()` | Yes | FLOWING |
| `rotate_key()` | new signatures | staged key signer -> metadata backend `put_entry()` before active key replace | No for interruption safety | HOLLOW for SEC-03 crash window |
| `extract_signable_fields()` | canonical field set | shared helper imported by UnifiedCache and BlobStore | Yes | FLOWING |
| `fsync_on_write` | local durability policy | config -> backend/journal constructors -> fsync helpers | Yes | FLOWING |
| `record_intent()` | planned blob path | cache/storage put paths before blob write | Yes | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| SEC-03 hard-interruption window | Inline `uv run --python 3.12 python -` probe under `.uv-cache`: create signed entry, stage new key, re-sign metadata with staged key, do not replace active key, reopen cache. | `old_key_unchanged=True`, `staged_exists=True`, `manual_verifies_with_active_old_key=False`, `read_after_simulated_crash_default_policy=None`, `entry_still_exists_after_read=False`. | FAIL |
| Phase 31 controlled regression sweep | Orchestrator-provided controlled command with repo-local basetemp and addopts cleared across Phase 31 test files. | `381 passed, 3 skipped`. | PASS, but not sufficient for SEC-03 hard-interruption window |

### Probe Execution

| Probe | Command | Result | Status |
|---|---|---|---|
| Conventional probes | `Get-ChildItem scripts -Recurse -Filter 'probe-*.sh'` | No probe scripts found. | SKIPPED |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|---|---|---|---|---|
| SEC-01 | 31-01 | Minimum accepted signature version and unsigned-entry risk docs | SATISFIED | Code, docs, and tests verify configurable minimum and downgrade rejection. |
| SEC-02 | 31-02 | Encrypted backend-routed reads without avoidable plaintext temp files | SATISFIED | Code routes ciphertext through backend and tries bytes path first; tests cover memory backend and temp cleanup. |
| SEC-03 | 31-03 | Interrupted key rotation leaves original key and entries usable | BLOCKED | Hard-interruption probe shows active old key cannot verify staged-key metadata and default read deletes the entry. |
| SEC-04 | 31-04 | Shared canonical signing with compatibility path | SATISFIED | Shared helper and explicit legacy fallback are present and tested. |
| STRG-01 | 31-05 | Storage-mode destructive API warning policy | SATISFIED | Warning helper is wired to destructive APIs and tests assert warning-first behavior. |
| STRG-02 | 31-06 | Durability contract and opt-in fsync policy | SATISFIED | Config, local fsync helpers, JSON/blob/intent wiring, and docs are present and tested. |
| STRG-03 | 31-07 | Write intents before blob writes | SATISFIED | Cache and storage put paths record planned blob intent before `_write_blob()` and tests cover cleanup guards. |

No orphaned Phase 31 requirements were found in `.planning/REQUIREMENTS.md`.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---:|---|---|---|
| `src/cacheness/core.py` | 202, 279 | "not available" optional-backend log text | INFO | Ordinary optional dependency fallback; not a stub. |
| `src/cacheness/config.py` | 767 | "not available" handler-priority message | INFO | Existing config message; not a stub. |
| `tests/test_handler_bytes_protocol.py` | 187, 198, 213, 316 | optional dependency skip messages | INFO | Test skip gates; not a stub. |
| `src/cacheness/metadata/json_backend.py` | 649 | `return []` for no JSON migrations | INFO | Valid empty migration list. |
| `src/cacheness/storage/backends/blob_backends.py` | 155 | `return {}` default write metadata | INFO | Valid default implementation. |

No unreferenced `TBD`, `FIXME`, or `XXX` debt markers were found in the scanned Phase 31 files.

### Human Verification Required

None.

### Gaps Summary

Phase 31 is blocked by SEC-03. The implementation handles controlled Python exceptions with rollback, but it does not make the rotation durable across a real interruption after staged-key metadata is committed and before the staged key replaces the active key. This directly violates the roadmap success criterion and requirement text for interrupted rotation.

---

_Verified: 2026-06-15T03:47:33Z_
_Verifier: the agent (gsd-verifier)_
