# Phase 31: Security & Storage-Mode Posture - Research

**Researched:** 2026-06-15
**Domain:** Python security, encrypted blob I/O, key rotation, storage-mode durability
**Confidence:** HIGH for codebase seams; MEDIUM for exact final task split because several Phase 31-adjacent fixes already exist and need verify-first checks.

## User Constraints (from CONTEXT.md)

### Locked Decisions

Phase 31 implements the security posture wave and the storage-mode durability seeds:

- TASK-13 / S1: harden signature verification against version downgrade and document unsigned-entry risks.
- TASK-14 / S2/S3: route encrypted reads through the blob backend and prefer in-memory handler reads over plaintext temp files.
- TASK-15 / S4: make key rotation two-phase enough that interrupted rotation leaves the old key and entries usable.
- SEED-004 / U4: converge `UnifiedCache` and `BlobStore` on a shared canonical signing field strategy with compatibility handling.
- SEED-001 / STRG-01: choose and implement the storage-mode policy for public destructive cache APIs.
- SEED-005 / R16: document storage-mode durability limits and provide an opt-in fsync policy where feasible.
- SEED-006 / R8: record write intents before blob writes where needed so the journal covers the full uncommitted-blob window.

This phase covers requirements `SEC-01`, `SEC-02`, `SEC-03`, `SEC-04`, `STRG-01`, `STRG-02`, and `STRG-03`.

The user's conditional selection was "1,1,1 if that's what's recommended in the code_review_actions.md and code_review_findings.md docs." Interpreted against those docs:

- Storage-mode destructive APIs: use the review-recommended loud warning policy, not a default hard raise. `CODE_REVIEW_FINDINGS.md` recommends "at least a loud warning log"; `CODE_REVIEW_ACTIONS.md` says raise-vs-warn strictness needed an owner decision.
- Fsync/durability: document the contract and add an opt-in `fsync_on_write` path where feasible.
- Signing convergence: use a shared canonical signing field strategy for new writes with a compatibility path for existing signed entries.

#### Implementation Decisions

- D-01 through D-08: implement configurable `minimum_signature_version`, keep default compatibility at `1`, reject stored signatures below the configured minimum, recommend `3` for new deployments, do not flip `allow_unsigned_entries`, and document unsigned-entry stripping risk.
- D-09 through D-14: encrypted BlobStore reads must use `blob_backend.read_blob()`, support non-filesystem backends, prefer `handler.get_bytes(plaintext, metadata)`, and only use a hardened temp-file fallback when `get_bytes()` raises `NotImplementedError`.
- D-15 through D-22: implement two-phase rotation for `UnifiedCache.rotate_key()` and `BlobStore.rotate_key()` with `<keyfile>.new`, old-key verification during the pass, atomic local blob re-encryption, final key replacement only after success, and startup logging for leftover `.new` files.
- D-23 through D-28: use one shared canonical signable-field extraction strategy for `UnifiedCache` and `BlobStore`, with an explicit compatibility path for old BlobStore signing shapes.
- D-29 through D-34: storage-mode destructive APIs use a warning-first policy, covering cleanup/eviction/clear entry points, without a default hard raise.
- D-35 through D-40: document storage-mode durability limits and add opt-in `fsync_on_write=False` where feasible for filesystem blobs, JSON metadata, and write-intent files.
- D-41 through D-45: record write intents with planned blob paths before blob writes where needed, covering cache and storage mode, and preserve the Phase 28 invariant that committed metadata protects blobs from stale-intent cleanup.
- D-46 through D-53: do not implement Phase 32 polish, do not reopen Phase 29 or Phase 30 except for required interactions, do not change public APIs unless justified by `minimum_signature_version` or `fsync_on_write`, use `uv`, include `--ignore=tests/test_tensorflow_handler.py` on Windows pytest commands, preserve service skip gates, and ignore beads.

### the agent's Discretion

- Whether to split SEC-04 signing convergence into its own plan or pair it with SEC-01, provided compatibility and minimum-version interactions are explicit.
- Whether `fsync_on_write` belongs under an existing storage/config dataclass or a narrow backend-local option, provided the user-visible config surface is documented.
- Whether storage-mode destructive API warnings use `logging.warning`, `warnings.warn`, or both, provided tests can assert the loud signal and docs explain it.
- Whether key rotation's `<blob>.rotating` behavior reuses existing atomic blob backend helpers or a rotation-specific helper, provided it does not truncate the committed blob in place.

### Deferred Ideas (OUT OF SCOPE)

- Flipping `allow_unsigned_entries` to `False` by default remains out of scope and belongs to future/breaking-change work.
- Strict signing migration that rejects all existing old-shape BlobStore signatures by default is out of scope.
- Always-on fsync by default is out of scope because the review docs call out performance tradeoffs.
- Full interrupted key-rotation resume is out of scope; Phase 31 must warn on `<keyfile>.new` and preserve the old key, not implement resumable rotation.
- JSON backend write batching/debouncing remains a future performance-policy item.
- Phase 32 polish items remain deferred to Phase 32.

## Summary

Phase 31 should be planned as a codebase hardening phase with no new external packages. [VERIFIED: pyproject.toml; 31-CONTEXT.md] The main work is inside existing Cacheness modules: `SecurityConfig`, `CacheEntrySigner.verify_entry()`, `VerificationMixin`, `BlobStore`, `UnifiedCache.rotate_key()`, storage-mode public destructive APIs, local filesystem durability hooks, and write-intent preservation. [VERIFIED: src/cacheness/config.py; src/cacheness/security.py; src/cacheness/_verification_mixin.py; src/cacheness/storage/blob_store.py; src/cacheness/core.py; src/cacheness/write_intent.py]

Several review findings are already partly remediated in the current tree, so the planner must include verify-first checks before writing tasks. [VERIFIED: src/cacheness/core.py; src/cacheness/write_intent.py; src/cacheness/storage/backends/blob_backends.py; src/cacheness/metadata/json_backend.py] Current code already records write intents before blob writes in both normal `put()` and `_storage_mode_put()`, resolves relative intent paths against `cache_dir`, checks committed metadata before deleting stale-intent blobs, runs stale-intent cleanup outside the `cleanup_on_init` guard, uses unique filesystem temp names, and preserves corrupt JSON metadata files. [VERIFIED: src/cacheness/core.py; src/cacheness/_storage_mode_mixin.py; src/cacheness/write_intent.py; src/cacheness/storage/backends/blob_backends.py; src/cacheness/metadata/json_backend.py]

The highest-risk unimplemented seams remain signature downgrade enforcement, BlobStore encrypted reads bypassing `blob_backend.read_blob()`, plaintext temp files on encrypted read, non-two-phase key rotation that writes the new key before re-sign/re-encrypt completes, divergent `UnifiedCache` vs `BlobStore` signing shapes, missing storage-mode destructive warnings, and missing opt-in fsync plumbing. [VERIFIED: src/cacheness/security.py; src/cacheness/storage/blob_store.py; src/cacheness/core.py; src/cacheness/config.py; docs/CODE_REVIEW_FINDINGS.md]

**Primary recommendation:** Plan four dependent waves: signing/downgrade plus canonical helper, encrypted reads, two-phase rotation, and storage-mode posture/durability, with a Wave 0 verify-first pass to adjust stale review assumptions. [VERIFIED: 31-CONTEXT.md; docs/CODE_REVIEW_ACTIONS.md; codebase grep]

## Project Constraints (from AGENTS.md)

- `AGENTS.md` redirects all actionable instructions to `.github/copilot-instructions.md`. [VERIFIED: AGENTS.md]
- Use `uv` for Python commands; do not use `python`, `pip`, or `python -m pytest` directly. [VERIFIED: .github/copilot-instructions.md]
- On Windows, every pytest command must include `--ignore=tests/test_tensorflow_handler.py`. [VERIFIED: .github/copilot-instructions.md]
- For this user-requested research run, ignore beads despite the project default requiring beads. [VERIFIED: user prompt; .github/copilot-instructions.md]
- Do not create or switch branches for this research task. [VERIFIED: user prompt]
- Do not edit code; create only `.planning/phases/31-security-storage-mode-posture/31-RESEARCH.md`. [VERIFIED: user prompt]
- Do not touch unrelated dirty files `.planning/config.json`, `tests/test_backend_parity.py`, or `tests/test_blob_namespace.py`. [VERIFIED: user prompt; git status]
- Test imports should use `from cacheness.core import UnifiedCache`, not package-root import. [VERIFIED: .github/copilot-instructions.md]

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SEC-01 | User can configure a minimum accepted signature version and reject downgraded signatures while receiving clear documentation for unsigned-entry risks. [VERIFIED: .planning/REQUIREMENTS.md] | Add `SecurityConfig.minimum_signature_version`, pass it to signer verification, reject parsed versions below the minimum, document `allow_unsigned_entries` stripping risk. [VERIFIED: src/cacheness/config.py; src/cacheness/security.py; docs/SECURITY.md] |
| SEC-02 | User can read encrypted blobs through the blob backend abstraction without avoidable plaintext temp files on disk. [VERIFIED: .planning/REQUIREMENTS.md] | Replace encrypted direct `Path.read_bytes()` reads with `blob_backend.read_blob()`, try `handler.get_bytes()` first, then hardened temp fallback. [VERIFIED: src/cacheness/storage/blob_store.py; src/cacheness/interfaces.py; tests/test_handler_bytes_protocol.py] |
| SEC-03 | User can rotate keys with a two-phase process that leaves the original key and entries usable if rotation is interrupted. [VERIFIED: .planning/REQUIREMENTS.md] | Rewrite both rotation paths around `<keyfile>.new`, old-key verification, atomic `.rotating` blob writes, and final key replacement. [VERIFIED: src/cacheness/core.py; src/cacheness/storage/blob_store.py; tests/test_key_rotation_api.py] |
| SEC-04 | User can rely on UnifiedCache and BlobStore signing the same canonical field set, with a documented compatibility path for existing signed entries. [VERIFIED: .planning/REQUIREMENTS.md] | Move `_extract_signable_fields()` logic to a shared helper, use it for new BlobStore signatures, and test old flattened BlobStore signatures remain readable. [VERIFIED: src/cacheness/_verification_mixin.py; src/cacheness/storage/blob_store.py; .planning/seeds/SEED-004-unify-signing-schemes.md] |
| STRG-01 | User can enable storage mode and be protected from accidental cache-eviction APIs through an explicit raise-or-warning policy. [VERIFIED: .planning/REQUIREMENTS.md] | Implement warning-first storage-mode policy on public destructive APIs such as `cleanup_expired()`, size enforcement paths, `clear_all()`, and namespace clearing. [VERIFIED: 31-CONTEXT.md; src/cacheness/core.py; .planning/seeds/SEED-001-storage-mode-api-hardening.md] |
| STRG-02 | User can understand and configure the storage-mode durability contract, including whether fsync is performed for JSON saves, blob writes, and intent files. [VERIFIED: .planning/REQUIREMENTS.md] | Add opt-in `fsync_on_write=False`, wire local fsync where file descriptors are available, and update transaction docs to distinguish atomic rename from power-loss durability. [VERIFIED: docs/TRANSACTION_GUARANTEES.md; src/cacheness/metadata/json_backend.py; src/cacheness/storage/backends/blob_backends.py; src/cacheness/write_intent.py] |
| STRG-03 | User can rely on write intents being recorded before blob writes where needed so crash recovery covers the full uncommitted-blob window. [VERIFIED: .planning/REQUIREMENTS.md] | Current code already records pre-blob intents in cache and storage mode; planner should preserve and extend tests while adding fsync and warning work. [VERIFIED: src/cacheness/core.py; src/cacheness/_storage_mode_mixin.py; tests/test_write_intent.py] |

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|--------------|----------------|-----------|
| Signature downgrade policy | API / Backend | Database / Storage | Signature creation and verification are backend integrity logic, while metadata stores the versioned signature string. [VERIFIED: src/cacheness/security.py; src/cacheness/_verification_mixin.py] |
| Canonical signing shape | API / Backend | Database / Storage | `UnifiedCache` and `BlobStore` must prepare the same signable payload before persistence. [VERIFIED: src/cacheness/_verification_mixin.py; src/cacheness/storage/blob_store.py] |
| Encrypted blob reads | Database / Storage | API / Backend | Blob bytes come from the configured blob backend; handlers deserialize after decryption. [VERIFIED: src/cacheness/storage/blob_store.py; src/cacheness/storage/backends/blob_backends.py; src/cacheness/interfaces.py] |
| Two-phase key rotation | API / Backend | Database / Storage | Rotation coordinates metadata re-signing, blob re-encryption, signer state, and key-file replacement under the cache lock. [VERIFIED: src/cacheness/core.py; src/cacheness/storage/blob_store.py] |
| Storage-mode destructive warning policy | API / Backend | Database / Storage | Public cache APIs decide when to warn before invoking metadata cleanup or blob deletion. [VERIFIED: src/cacheness/core.py; 31-CONTEXT.md] |
| `fsync_on_write` durability option | Database / Storage | API / Backend | Local backends own file descriptors and atomic rename behavior; config exposes the opt-in policy. [VERIFIED: src/cacheness/config.py; src/cacheness/metadata/json_backend.py; src/cacheness/storage/backends/blob_backends.py; src/cacheness/write_intent.py] |
| Write-intent crash coverage | Database / Storage | API / Backend | Intent files describe planned blob writes; cache/storage put paths decide when to record and clear them. [VERIFIED: src/cacheness/write_intent.py; src/cacheness/core.py; src/cacheness/_storage_mode_mixin.py] |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python stdlib `hmac`, `hashlib`, `os.replace`, `tempfile`, `logging`, `warnings`, `pathlib` | Python >=3.11 project requirement | HMAC validation, atomic local file replacement, secure temp files, warning signals, path handling | Existing code already uses these primitives; no new package is required. [VERIFIED: pyproject.toml; src/cacheness/security.py; src/cacheness/storage/backends/blob_backends.py] |
| `cryptography` | Existing optional dependency | AES-256-GCM encryption/decryption | Existing encryption tests import it conditionally and current code uses Cacheness encryption helpers. [VERIFIED: tests/test_encryption_at_rest.py; src/cacheness/encryption.py] |
| `xxhash` | Existing dependency | File/blob hash and intent filename hashing | Existing integrity and write-intent code already depends on it. [VERIFIED: src/cacheness/write_intent.py; src/cacheness/storage/blob_store.py] |
| `pytest` | 8.4.1 observed | Automated verification | Project test config and validation strategy use pytest through `uv`. [VERIFIED: uv run pytest --version; pyproject.toml; 31-VALIDATION.md] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `ruff` | >=0.12.8 in project metadata | Formatting and lint validation | Use only during execution after Python file edits. [VERIFIED: pyproject.toml; .github/copilot-instructions.md] |
| `ty` | >=0.0.15 in project metadata | Type checking touched Python files | Use only during execution after Python file edits; document unrelated diagnostics. [VERIFIED: pyproject.toml; .github/copilot-instructions.md] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| New signing package | None | Existing HMAC/HKDF design is already implemented; Phase 31 is policy and compatibility hardening, not a crypto replacement. [VERIFIED: src/cacheness/security.py; docs/CODE_REVIEW_FINDINGS.md] |
| New durability library | None | Local `os.fsync`, directory fsync where supported, and backend-level helpers are enough for the opt-in scope. [VERIFIED: .planning/seeds/SEED-005-fsync-policy-storage-mode-durability.md; src/cacheness/storage/backends/blob_backends.py] |

**Installation:** No new package installation is recommended. [VERIFIED: pyproject.toml; 31-CONTEXT.md]

## Package Legitimacy Audit

No external package is installed by this phase. [VERIFIED: 31-CONTEXT.md; pyproject.toml]

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| N/A | N/A | N/A | N/A | N/A | N/A | No package changes. [VERIFIED: pyproject.toml] |

**Packages removed due to [SLOP] verdict:** none. [VERIFIED: no external package recommendation]
**Packages flagged as suspicious [SUS]:** none. [VERIFIED: no external package recommendation]

## Current Code Shape and Implementation Seams

### SEC-01: Minimum Signature Version and Unsigned-Entry Risk

- `SecurityConfig` currently has `enable_entry_signing`, `signing_key_file`, `use_in_memory_key`, `allow_unsigned_entries`, `delete_invalid_signatures`, `key_fallback_policy`, `use_hkdf_derivation`, and encryption fields, but no `minimum_signature_version`. [VERIFIED: src/cacheness/config.py]
- `CacheEntrySigner.parse_versioned_signature()` treats `vN:<hex>` as version `N` and bare hex as version `1`. [VERIFIED: src/cacheness/security.py]
- `CacheEntrySigner.verify_entry()` chooses the HKDF-derived key for version `>=3` and the master key for older versions, but does not enforce a minimum accepted version. [VERIFIED: src/cacheness/security.py]
- `VerificationMixin._verify_entry()` calls `self.signer.verify_entry(verify_data, stored_signature)` and handles invalid or unsigned entries according to `delete_invalid_signatures`, `allow_unsigned_entries`, and `storage_mode`. [VERIFIED: src/cacheness/_verification_mixin.py]
- `BlobStore.get()` also verifies signatures directly through `self.signer.verify_entry(signable, stored_signature)` and must receive the same minimum-version behavior. [VERIFIED: src/cacheness/storage/blob_store.py]
- Recommended seam: store `minimum_signature_version` on `SecurityConfig`, thread it into `CacheEntrySigner` construction or pass it through verifier calls, and make all `verify_entry()` paths reject parsed versions below the configured minimum. [VERIFIED: src/cacheness/config.py; src/cacheness/security.py; src/cacheness/_verification_mixin.py; src/cacheness/storage/blob_store.py]
- Docs seam: `docs/SECURITY.md` currently describes `allow_unsigned_entries` but does not explicitly state that an attacker with metadata write access can strip signatures when `allow_unsigned_entries=True`. [VERIFIED: docs/SECURITY.md]

### SEC-02: Backend-Routed Encrypted Reads and No Avoidable Plaintext Temp Files

- `BlobStore.get()` resolves `actual_path`, checks backend existence, and when encrypted calls `actual_path.read_bytes()` directly, then writes plaintext to `tempfile.NamedTemporaryFile()` for handler reads. [VERIFIED: src/cacheness/storage/blob_store.py]
- `BlobStore._read_blob()` repeats the direct `path.read_bytes()` and plaintext temp-file pattern for delegated `UnifiedCache` reads. [VERIFIED: src/cacheness/storage/blob_store.py]
- Handlers already expose `get_bytes()` with a default `NotImplementedError`, and `BytesHandler`, `ObjectHandler`, and `ArrayHandler` have byte-path coverage. [VERIFIED: src/cacheness/interfaces.py; tests/test_handler_bytes_protocol.py]
- Recommended seam: in both `BlobStore.get()` and `_read_blob()`, read ciphertext with `self.blob_backend.read_blob(actual_path_str_or_resolved_backend_path)`, decrypt in memory, try `handler.get_bytes(plaintext, handler_metadata)`, and only fall back to a temp file if `NotImplementedError` is raised. [VERIFIED: src/cacheness/storage/blob_store.py; src/cacheness/interfaces.py]
- Temp fallback should use `tempfile.mkstemp(dir=self.cache_dir or local blob parent)`, close the fd, write bytes, `os.chmod(path, 0o600)` where supported, then unlink in `finally`. [VERIFIED: 31-CONTEXT.md; src/cacheness/storage/blob_store.py]

### SEC-03: Two-Phase Key Rotation

- `UnifiedCache.rotate_key()` currently reads the new key, immediately writes it to the active key file, constructs a new signer, then re-signs metadata and re-encrypts blobs. [VERIFIED: src/cacheness/core.py]
- `BlobStore.rotate_key()` has the same immediate active-key overwrite pattern. [VERIFIED: src/cacheness/storage/blob_store.py]
- Both rotation paths currently re-encrypt local blobs by writing ciphertext directly back to the final blob path. [VERIFIED: src/cacheness/core.py; src/cacheness/storage/blob_store.py]
- Recommended seam: add a shared helper or duplicated small routine that writes `<keyfile>.new`, creates a new signer from that staged key without replacing the active key, verifies each old entry with the old signer before mutation, writes local re-encryption to `<blob>.rotating` followed by `os.replace`, updates metadata and signatures, and only replaces the active key file after all entries and encrypted blobs complete. [VERIFIED: 31-CONTEXT.md; src/cacheness/core.py; src/cacheness/storage/blob_store.py]
- Startup seam: signer initialization should detect `<keyfile>.new` next to the active key file and log an error that rotation was interrupted; resume is out of scope. [VERIFIED: 31-CONTEXT.md; src/cacheness/security.py]

### SEC-04: Shared Canonical Signing

- `VerificationMixin._extract_signable_fields()` normalizes `created_at` and returns a canonical superset including `cache_key`, `data_type`, `file_size`, `actual_path`, `file_hash`, `object_type`, `storage_format`, `serializer`, and `compression_codec`. [VERIFIED: src/cacheness/_verification_mixin.py]
- `UnifiedCache` signs through `_sign_entry_if_enabled()`, which uses `_extract_signable_fields()`. [VERIFIED: src/cacheness/_verification_mixin.py; src/cacheness/core.py]
- `BlobStore.put()`, `BlobStore.get()`, and `BlobStore.rotate_key()` use a flattened signing payload such as `{**entry_data, **custom_metadata}` or `{**full_entry, **nested_meta, "cache_key": cache_key}`. [VERIFIED: src/cacheness/storage/blob_store.py]
- Recommended seam: move canonical extraction to a shared module such as `src/cacheness/signing_fields.py`, import it from `VerificationMixin` and `BlobStore`, and make new `BlobStore` writes use the same canonical payload as `UnifiedCache`. [VERIFIED: .planning/seeds/SEED-004-unify-signing-schemes.md; src/cacheness/_verification_mixin.py; src/cacheness/storage/blob_store.py]
- Compatibility seam: verification should first try the canonical payload, then explicitly try the legacy flattened BlobStore payload for existing signatures unless `minimum_signature_version` intentionally rejects the signature version. [VERIFIED: 31-CONTEXT.md; src/cacheness/storage/blob_store.py]

### STRG-01: Storage-Mode Destructive API Warnings

- `CacheConfig(storage_mode=True)` disables TTL, size limits, cleanup on init, stats, and auto cleanup, but public destructive methods remain callable. [VERIFIED: src/cacheness/config.py; src/cacheness/core.py]
- `UnifiedCache.clear_all()`, `clear()`, `clear_all_namespaces()`, and `cleanup_expired(ttl_seconds=...)` can delete metadata and blob files when called explicitly. [VERIFIED: src/cacheness/core.py]
- `_enforce_size_limit()` returns early when max size is `None`; storage mode sets max size to `None`, so implicit size eviction is disabled. [VERIFIED: src/cacheness/config.py; src/cacheness/core.py]
- Recommended seam: add a helper such as `_warn_storage_mode_destructive_api(api_name, detail)` in `UnifiedCache` and call it at the start of destructive public APIs when `self.config.storage_mode` is true. [VERIFIED: 31-CONTEXT.md; src/cacheness/core.py]
- Warning mechanism should be either `logging.warning`, `warnings.warn`, or both; choose the one the tests assert consistently. [VERIFIED: 31-CONTEXT.md]

### STRG-02: Durability Contract and `fsync_on_write`

- `docs/TRANSACTION_GUARANTEES.md` already documents that Cacheness does not explicitly fsync and trades power-loss durability for speed. [VERIFIED: docs/TRANSACTION_GUARANTEES.md]
- `CacheStorageConfig` and `SecurityConfig` currently do not expose `fsync_on_write`. [VERIFIED: src/cacheness/config.py]
- `FilesystemBlobBackend.write_blob()` and `write_blob_stream()` write temp files and `os.replace()` them, but do not fsync the temp file or containing directory. [VERIFIED: src/cacheness/storage/backends/blob_backends.py]
- `JsonBackend._save_to_disk()` writes a temp JSON file and moves it into place, but does not fsync the temp file or directory. [VERIFIED: src/cacheness/metadata/json_backend.py]
- `WriteIntentJournal.record_intent()` writes JSON text without fsync. [VERIFIED: src/cacheness/write_intent.py]
- Recommended seam: add `fsync_on_write: bool = False` to the narrowest config object the planner chooses, pass it into local write layers, and implement best-effort fsync only where a local file descriptor and local directory are available. [VERIFIED: 31-CONTEXT.md; .planning/seeds/SEED-005-fsync-policy-storage-mode-durability.md; src/cacheness/config.py]
- Docs must state that atomic rename is crash-consistency against partial files, not a guarantee that bytes reached stable storage before power loss. [VERIFIED: docs/TRANSACTION_GUARANTEES.md; docs/CODE_REVIEW_FINDINGS.md]

### STRG-03: Pre-Blob Write Intent Coverage

- `UnifiedCache.put()` currently records an intent with the planned relative blob path before `_blob_store._write_blob()`. [VERIFIED: src/cacheness/core.py]
- `_storage_mode_put()` currently records an intent with the planned relative blob path before `_blob_store._write_blob()`. [VERIFIED: src/cacheness/_storage_mode_mixin.py]
- `WriteIntentJournal.cleanup_stale_intents()` resolves relative paths against `cache_dir`, checks `entry_exists` before deleting an intended blob, and only unlinks a blob if the path exists. [VERIFIED: src/cacheness/write_intent.py]
- `tests/test_write_intent.py` already contains storage-mode tests for preserving committed entries, removing uncommitted orphan blobs, and asserting pre-blob intent creation before storage-mode blob writes. [VERIFIED: tests/test_write_intent.py]
- Recommended seam: do not reimplement intent ordering from scratch; preserve current behavior while adding missing cache-mode failure-before-blob-created tests and optional intent-file fsync under STRG-02. [VERIFIED: src/cacheness/core.py; src/cacheness/_storage_mode_mixin.py; tests/test_write_intent.py]

## Architecture Patterns

### System Architecture Diagram

```text
User API call
  |
  v
UnifiedCache / BlobStore public API
  |
  +--> storage_mode? ---- yes --> warn on destructive public API, no implicit TTL/eviction
  |                         |
  |                         no
  v
Signable metadata construction
  |
  +--> canonical payload helper --> signer.sign_entry / signer.verify_entry
  |                                  |
  |                                  +--> reject if stored_version < minimum_signature_version
  v
Blob write/read path
  |
  +--> put: record write intent before local/remote blob write
  |         |
  |         +--> local backend atomic temp + replace (+ optional fsync)
  |         +--> metadata commit (+ optional JSON fsync)
  |         +--> clear intent
  |
  +--> encrypted get: blob_backend.read_blob -> decrypt in memory
            |
            +--> handler.get_bytes succeeds -> no temp plaintext
            +--> NotImplementedError -> secure local temp fallback -> unlink finally

rotate_key()
  |
  +--> write <keyfile>.new
  +--> old signer verifies existing entries
  +--> new signer re-signs metadata
  +--> local encrypted blob writes use <blob>.rotating + os.replace
  +--> replace active key only after all work succeeds
```

### Recommended Project Structure

```text
src/cacheness/
|-- config.py                  # minimum_signature_version and fsync_on_write config surface
|-- security.py                # signature version enforcement and staged key handling
|-- signing_fields.py          # new shared canonical signing helper
|-- _verification_mixin.py     # UnifiedCache verification delegates to shared helper
|-- core.py                    # UnifiedCache rotation, warnings, write-intent preservation
|-- _storage_mode_mixin.py     # storage-mode put/get intent preservation
|-- write_intent.py            # optional intent fsync
|-- storage/
|   |-- blob_store.py          # encrypted read path, BlobStore signing, BlobStore rotation
|   `-- backends/blob_backends.py # local blob fsync and atomic write helpers
`-- metadata/json_backend.py   # JSON metadata fsync
```

### Pattern 1: Canonical Signable Payload Helper

**What:** one function produces a normalized signable field superset for both APIs. [VERIFIED: src/cacheness/_verification_mixin.py; src/cacheness/storage/blob_store.py]
**When to use:** every signature generation or verification path for entry metadata. [VERIFIED: .planning/seeds/SEED-004-unify-signing-schemes.md]

```python
# Source: src/cacheness/_verification_mixin.py
signable = extract_signable_fields(cache_key, entry_data, metadata)
signature = signer.sign_entry(signable)
```

### Pattern 2: Encrypted Read Zero-Disk Fast Path

**What:** decrypt ciphertext to bytes, deserialize from `handler.get_bytes()`, and only use a secure temp file when the handler explicitly lacks byte support. [VERIFIED: src/cacheness/storage/blob_store.py; src/cacheness/interfaces.py]
**When to use:** encrypted `BlobStore.get()` and low-level `_read_blob()` paths. [VERIFIED: 31-CONTEXT.md]

```python
# Source: tests/test_handler_bytes_protocol.py and src/cacheness/interfaces.py
try:
    return handler.get_bytes(plaintext, handler_metadata)
except NotImplementedError:
    # secure temp-file fallback only
    ...
```

### Pattern 3: Two-Phase Rotation Commit

**What:** stage new key material separately, mutate entries only after old-key verification, then replace active key last. [VERIFIED: 31-CONTEXT.md; src/cacheness/core.py]
**When to use:** both `UnifiedCache.rotate_key()` and `BlobStore.rotate_key()`. [VERIFIED: 31-CONTEXT.md]

```python
# Source: 31-CONTEXT.md
staged_key = key_file.with_name(key_file.name + ".new")
staged_key.write_bytes(new_key_bytes)
# re-sign/re-encrypt using old + new signer objects
os.replace(staged_key, key_file)
```

### Anti-Patterns to Avoid

- **Minimum-version check only in `VerificationMixin`:** `BlobStore.get()` calls the signer directly, so SEC-01 would remain bypassable there. [VERIFIED: src/cacheness/storage/blob_store.py]
- **Canonical signing without compatibility:** old flattened BlobStore signatures would fail unless a compatibility verifier is retained. [VERIFIED: .planning/seeds/SEED-004-unify-signing-schemes.md; src/cacheness/storage/blob_store.py]
- **Direct `Path.read_bytes()` for encrypted blobs:** this bypasses non-filesystem backends and backend controls. [VERIFIED: src/cacheness/storage/blob_store.py; docs/CODE_REVIEW_FINDINGS.md]
- **Writing the active key before rotation finishes:** an interrupted rotation can leave old entries unverifiable with the new active key. [VERIFIED: src/cacheness/core.py; src/cacheness/storage/blob_store.py; docs/CODE_REVIEW_FINDINGS.md]
- **Claiming fsync proves power-loss safety in tests:** tests can verify fsync hooks are invoked, not simulate real power loss. [VERIFIED: 31-VALIDATION.md; docs/TRANSACTION_GUARANTEES.md]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Cryptographic comparison | Custom string equality | `hmac.compare_digest` | Existing signer already uses constant-time comparison. [VERIFIED: src/cacheness/security.py] |
| Local atomic file replacement | Manual truncate/rewrite | temp file plus `os.replace` | Existing filesystem backend uses this pattern for blob writes. [VERIFIED: src/cacheness/storage/backends/blob_backends.py] |
| Secure temp-file creation | Predictable temp filenames | `tempfile.mkstemp` | Existing backend uses unique temp files; encrypted plaintext fallback should match that pattern. [VERIFIED: src/cacheness/storage/backends/blob_backends.py; 31-CONTEXT.md] |
| Handler byte deserialization | Custom pickle/numpy parsing in BlobStore | `handler.get_bytes()` | Handler protocol already owns byte deserialization. [VERIFIED: src/cacheness/interfaces.py; tests/test_handler_bytes_protocol.py] |
| Crash recovery journal | New journal format | Existing `WriteIntentJournal` | Current journal already covers pre-blob intent ordering and conservative cleanup. [VERIFIED: src/cacheness/write_intent.py; tests/test_write_intent.py] |

**Key insight:** Phase 31 should tighten existing primitives rather than replace them; the codebase already has the right layers, but policy checks and cross-layer consistency are incomplete. [VERIFIED: docs/CODE_REVIEW_FINDINGS.md; codebase grep]

## Common Pitfalls

### Pitfall 1: Validation Doc References a Missing Test File

**What goes wrong:** Commands that include `tests/test_security.py` fail at collection because that file is absent. [VERIFIED: rg --files tests]
**Why it happens:** Phase validation inherited task names from review docs, while the current test suite uses `tests/test_cache_signing.py`, `tests/test_encryption_at_rest.py`, `tests/test_key_rotation.py`, and `tests/test_key_rotation_api.py`. [VERIFIED: tests inventory grep]
**How to avoid:** Replace `tests/test_security.py` with the existing security-related test files in execution plans. [VERIFIED: tests inventory grep]
**Warning signs:** `pytest` reports "file or directory not found: tests/test_security.py". [VERIFIED: rg command output]

### Pitfall 2: Review Findings May Be Stale

**What goes wrong:** The planner creates tasks for fixes already present, duplicating work or regressing earlier phases. [VERIFIED: current code vs docs/CODE_REVIEW_ACTIONS.md]
**Why it happens:** The current code already contains several Phase 28/30-style remediations that the original review listed as future work. [VERIFIED: src/cacheness/core.py; src/cacheness/write_intent.py; src/cacheness/storage/backends/blob_backends.py; src/cacheness/metadata/json_backend.py]
**How to avoid:** Start Wave 0 with verify-first checks for write-intent ordering, stale-intent cleanup, unique temp files, JSON save errors, and corrupt JSON preservation. [VERIFIED: source grep]
**Warning signs:** Existing tests in `tests/test_write_intent.py` already assert the intended STRG-03 behavior. [VERIFIED: tests/test_write_intent.py]

### Pitfall 3: Minimum Signature Version Is Not Global Unless the Signer Owns It

**What goes wrong:** `UnifiedCache` may reject downgraded signatures while `BlobStore` still accepts them. [VERIFIED: src/cacheness/_verification_mixin.py; src/cacheness/storage/blob_store.py]
**Why it happens:** `BlobStore.get()` verifies signatures directly through the signer, bypassing `VerificationMixin`. [VERIFIED: src/cacheness/storage/blob_store.py]
**How to avoid:** Put minimum-version enforcement inside `CacheEntrySigner.verify_entry()` or centralize through a shared verification helper used by both APIs. [VERIFIED: src/cacheness/security.py; src/cacheness/storage/blob_store.py]
**Warning signs:** SEC-01 tests pass for `UnifiedCache` but not for `BlobStore`. [VERIFIED: current architecture]

### Pitfall 4: Canonical Signing Can Break Existing BlobStore Entries

**What goes wrong:** Changing BlobStore signing to canonical fields makes old flattened signatures fail. [VERIFIED: src/cacheness/storage/blob_store.py; .planning/seeds/SEED-004-unify-signing-schemes.md]
**Why it happens:** Existing BlobStore entries were signed over a different payload shape. [VERIFIED: src/cacheness/storage/blob_store.py]
**How to avoid:** Verification must try canonical first and legacy flattened second, or use a versioned compatibility transition documented and tested. [VERIFIED: 31-CONTEXT.md]
**Warning signs:** `tests/test_blob_store.py` signed round-trip or tamper tests start failing after helper extraction. [VERIFIED: tests/test_blob_store.py]

### Pitfall 5: Encrypted In-Memory Backends Need Backend Paths, Not Filesystem Paths

**What goes wrong:** `memory://` or S3-style encrypted reads fail because code tries to construct and read a local `Path`. [VERIFIED: src/cacheness/storage/blob_store.py; docs/CODE_REVIEW_FINDINGS.md]
**Why it happens:** The encrypted branch currently uses direct filesystem reads. [VERIFIED: src/cacheness/storage/blob_store.py]
**How to avoid:** Preserve the stored backend path string and route reads through `blob_backend.read_blob()`. [VERIFIED: src/cacheness/storage/backends/blob_backends.py]
**Warning signs:** Encrypted `InMemoryBlobBackend` round-trip fails before the change. [VERIFIED: 31-CONTEXT.md; tests/test_blob_store.py]

## Verify-First Checks for Planner

| Check | Purpose | Command / Method |
|-------|---------|------------------|
| Missing test file audit | Avoid planning commands with absent `tests/test_security.py`. [VERIFIED: rg output] | `rg --files tests | rg "security|cache_signing|encryption|key_rotation"` |
| STRG-03 already-present audit | Confirm pre-blob intent behavior before creating implementation work. [VERIFIED: src/cacheness/core.py; src/cacheness/_storage_mode_mixin.py] | Inspect `record_intent` before `_write_blob` in both put paths. |
| Encrypted read failure baseline | Prove SEC-02 still fails before changing it. [VERIFIED: src/cacheness/storage/blob_store.py] | Add/expect failing test for encrypted `InMemoryBlobBackend` and temp-file-free byte path. |
| Rotation crash baseline | Prove active key is overwritten before re-sign/re-encrypt completes. [VERIFIED: src/cacheness/core.py; src/cacheness/storage/blob_store.py] | Fault inject after `dest.write_bytes(new_key_bytes)` and assert old key file changed today. |
| Storage-mode destructive API baseline | Prove warning is absent today. [VERIFIED: src/cacheness/core.py] | Use `caplog` or `pytest.warns` around `storage_mode=True` `clear_all()` and `cleanup_expired(ttl_seconds=...)`. |
| fsync wiring baseline | Prove no fsync calls exist today. [VERIFIED: rg fsync] | `rg -n "fsync|fdatasync" src/cacheness` should be empty or only comments after planner confirms. |

## Dependency Ordering

1. **Wave 0: verify-first and test command correction.** Update plans to use existing test files, not `tests/test_security.py`. [VERIFIED: tests inventory grep; 31-VALIDATION.md]
2. **Wave 1: SEC-01 + SEC-04 shared signing foundation.** Minimum-version enforcement and canonical signing interact, so centralize signing payloads before or alongside downgrade tests. [VERIFIED: src/cacheness/security.py; src/cacheness/_verification_mixin.py; src/cacheness/storage/blob_store.py]
3. **Wave 2: SEC-02 encrypted read path.** This is mostly isolated to `BlobStore`, but it uses handler byte protocol and backend abstractions. [VERIFIED: src/cacheness/storage/blob_store.py; tests/test_handler_bytes_protocol.py]
4. **Wave 3: SEC-03 two-phase rotation.** Rotation depends on the final signing strategy to avoid re-signing entries twice. [VERIFIED: src/cacheness/core.py; src/cacheness/storage/blob_store.py]
5. **Wave 4: STRG-01/STRG-02/STRG-03 storage-mode posture.** Warning policy and fsync can land after security semantics are stable; STRG-03 should be mostly tests/preservation unless fsync touches intents. [VERIFIED: src/cacheness/core.py; src/cacheness/write_intent.py; 31-CONTEXT.md]

## Runtime State Inventory

Phase 31 is a behavior hardening phase, not a rename/refactor/migration phase, but key files and cache metadata are runtime state affected by rotation and compatibility. [VERIFIED: 31-CONTEXT.md; src/cacheness/security.py]

| Category | Items Found | Action Required |
|----------|-------------|-----------------|
| Stored data | Existing metadata entries can contain legacy `v1`, `v2`, `v3`, or flattened BlobStore signatures. [VERIFIED: src/cacheness/security.py; src/cacheness/storage/blob_store.py; tests/test_key_rotation_api.py] | Compatibility verifier; no bulk migration by default. [VERIFIED: 31-CONTEXT.md] |
| Live service config | No external service config is required by Phase 31; S3/PostgreSQL paths must keep existing skip gates. [VERIFIED: 31-CONTEXT.md; .github/copilot-instructions.md] | No service reconfiguration. [VERIFIED: 31-CONTEXT.md] |
| OS-registered state | None identified for this phase. [VERIFIED: codebase grep scope; phase docs] | None. [VERIFIED: phase docs] |
| Secrets/env vars | Signing/encryption key files under cache directories are active runtime state; interrupted rotation may leave `<keyfile>.new`. [VERIFIED: src/cacheness/security.py; src/cacheness/core.py; src/cacheness/storage/blob_store.py] | Stage new key separately and log leftover `.new` at startup. [VERIFIED: 31-CONTEXT.md] |
| Build artifacts | No build artifact migration is needed. [VERIFIED: phase scope] | None. [VERIFIED: phase scope] |

## Code Examples

### Signature Minimum Version Enforcement

```python
# Source: src/cacheness/security.py
version, hex_sig = self.parse_versioned_signature(stored_signature)
if version < self.minimum_signature_version:
    logger.warning("Rejected signature below configured minimum version")
    return False
```

### Legacy BlobStore Compatibility Verification

```python
# Source: src/cacheness/storage/blob_store.py
canonical = extract_signable_fields(cache_key, entry, nested_meta)
if signer.verify_entry(canonical, stored_signature):
    return True
legacy = {**entry, **nested_meta, "cache_key": cache_key}
return signer.verify_entry(legacy, stored_signature)
```

### Local fsync Helper

```python
# Source: STRG-02 research based on stdlib os usage in blob_backends/json_backend
def fsync_file_and_parent(file_obj, final_path):
    file_obj.flush()
    os.fsync(file_obj.fileno())
    if os.name != "nt":
        dir_fd = os.open(str(Path(final_path).parent), os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Direct filesystem reads for encrypted blobs | Backend-routed ciphertext reads plus byte deserialization | Phase 31 target | Required for `memory://` and S3-style backends. [VERIFIED: 31-CONTEXT.md; src/cacheness/storage/blob_store.py] |
| Active key overwrite before rotation | Staged `<keyfile>.new` and replace active key last | Phase 31 target | Prevents interrupted rotation from marooning old entries. [VERIFIED: 31-CONTEXT.md; src/cacheness/core.py] |
| Divergent signing payloads | Shared canonical signable helper plus legacy verifier | Phase 31 target | Prevents new drift without breaking old BlobStore signatures. [VERIFIED: .planning/seeds/SEED-004-unify-signing-schemes.md] |
| No explicit fsync | Opt-in local fsync where feasible | Phase 31 target | Makes storage-mode durability contract configurable without default performance cost. [VERIFIED: .planning/seeds/SEED-005-fsync-policy-storage-mode-durability.md] |

**Deprecated/outdated:**
- Documentation that says deleting the key file is the key-rotation path is outdated for Phase 31 because an explicit rotation API exists and needs two-phase safety. [VERIFIED: docs/SECURITY.md; tests/test_key_rotation_api.py]
- Documentation that says crash recovery after rotation is safe is outdated relative to current code because the active key is overwritten before entry mutation completes. [VERIFIED: docs/SECURITY.md; src/cacheness/core.py; src/cacheness/storage/blob_store.py]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `fsync_on_write` should likely live on storage config rather than security config. [ASSUMED] | STRG-02 | Planner may choose a different config surface; tests/docs must follow the chosen location. |
| A2 | Directory fsync can be skipped or best-effort on Windows if unsupported. [ASSUMED] | Code Examples | Implementation may need platform-specific handling to avoid Windows failures. |

## Open Questions

1. **Where should `fsync_on_write` live?** [VERIFIED: 31-CONTEXT.md]
   - What we know: user accepted an opt-in knob defaulting to `False`. [VERIFIED: 31-CONTEXT.md]
   - What's unclear: whether it belongs under `CacheStorageConfig`, a narrower backend option, or both. [VERIFIED: 31-CONTEXT.md]
   - Recommendation: put it under `CacheStorageConfig` so JSON metadata, blob backend, and write-intent journal can share one policy. [ASSUMED]

2. **Should storage-mode warnings use logs, Python warnings, or both?** [VERIFIED: 31-CONTEXT.md]
   - What we know: warning-first is locked, and hard-raise is not default. [VERIFIED: 31-CONTEXT.md]
   - What's unclear: exact warning mechanism. [VERIFIED: 31-CONTEXT.md]
   - Recommendation: use `logging.warning` for operational visibility and `warnings.warn(..., RuntimeWarning)` only if tests/docs need a Python warnings contract. [ASSUMED]

3. **Should legacy BlobStore signatures be opportunistically re-signed?** [VERIFIED: .planning/seeds/SEED-004-unify-signing-schemes.md]
   - What we know: compatibility is required and strict migration is out of scope. [VERIFIED: 31-CONTEXT.md]
   - What's unclear: whether read-time re-sign is acceptable in a `get()` path. [VERIFIED: 31-CONTEXT.md]
   - Recommendation: prefer verify-only compatibility in Phase 31; re-sign on explicit mutation/rotation only. [ASSUMED]

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| `uv` | All Python/test commands | yes | 0.11.19 | None needed. [VERIFIED: uv --version] |
| Python via `uv run --python 3.12` | Test execution | partially | Command hit uv cache initialization error in default user cache path | Set a writable `UV_CACHE_DIR` if executor sees the same failure. [VERIFIED: command output] |
| pytest via `uv run --python 3.12 pytest --version` | Test execution | yes | 8.4.1 | None needed. [VERIFIED: command output] |
| pytest-xdist | Default pytest addopts | configured | >=3.8.0 in project metadata | Use `-p no:xdist` only for sequential debugging. [VERIFIED: pyproject.toml; .github/copilot-instructions.md] |

**Missing dependencies with no fallback:** none identified. [VERIFIED: pyproject.toml; command output]

**Missing dependencies with fallback:** Python subcommand cache initialization may need `UV_CACHE_DIR` set to a writable directory if the default user cache collision recurs. [VERIFIED: command output]

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.4.1 via `uv run --python 3.12 pytest`. [VERIFIED: command output; 31-VALIDATION.md] |
| Config file | `pyproject.toml`. [VERIFIED: pyproject.toml] |
| Quick run command | `uv run --python 3.12 pytest tests/test_cache_signing.py tests/test_blob_store.py tests/test_key_rotation.py tests/test_key_rotation_api.py tests/test_storage_mode.py tests/test_atomic_writes.py tests/test_write_intent.py tests/test_encryption_at_rest.py tests/test_handler_bytes_protocol.py -x -q --ignore=tests/test_tensorflow_handler.py` [VERIFIED: tests inventory grep; 31-VALIDATION.md corrected for missing test file] |
| Full suite command | `uv run --python 3.12 pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py` [VERIFIED: .github/copilot-instructions.md; 31-VALIDATION.md] |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| SEC-01 | Minimum signature version rejects downgraded signatures and docs explain unsigned-entry risk. [VERIFIED: 31-CONTEXT.md] | unit/security/docs | `uv run --python 3.12 pytest tests/test_cache_signing.py tests/test_key_rotation_api.py -x -q --ignore=tests/test_tensorflow_handler.py` | yes [VERIFIED: rg --files tests] |
| SEC-02 | Encrypted reads route through blob backend and prefer byte handlers. [VERIFIED: 31-CONTEXT.md] | integration/security | `uv run --python 3.12 pytest tests/test_blob_store.py tests/test_encryption_at_rest.py tests/test_handler_bytes_protocol.py -x -q --ignore=tests/test_tensorflow_handler.py` | yes [VERIFIED: rg --files tests] |
| SEC-03 | Interrupted rotation leaves old key and old entries usable. [VERIFIED: 31-CONTEXT.md] | fault-injection/security | `uv run --python 3.12 pytest tests/test_key_rotation.py tests/test_key_rotation_api.py tests/test_encryption_at_rest.py -x -q --ignore=tests/test_tensorflow_handler.py` | yes [VERIFIED: rg --files tests] |
| SEC-04 | UnifiedCache and BlobStore share canonical signing for new writes and accept old-shape signatures. [VERIFIED: 31-CONTEXT.md] | parity/compatibility | `uv run --python 3.12 pytest tests/test_cache_signing.py tests/test_blob_store.py tests/test_namespace_signing.py tests/test_cross_system_compatibility.py -x -q --ignore=tests/test_tensorflow_handler.py` | yes [VERIFIED: rg --files tests] |
| STRG-01 | Storage-mode destructive APIs emit loud warnings and implicit eviction remains disabled. [VERIFIED: 31-CONTEXT.md] | storage-mode regression | `uv run --python 3.12 pytest tests/test_storage_mode.py tests/test_blob_store.py -x -q --ignore=tests/test_tensorflow_handler.py` | yes [VERIFIED: rg --files tests] |
| STRG-02 | Docs describe durability limits and opt-in fsync hooks are invoked. [VERIFIED: 31-CONTEXT.md] | docs/unit | `uv run --python 3.12 pytest tests/test_config_options.py tests/test_config_validation.py tests/test_atomic_writes.py tests/test_write_intent.py tests/test_blob_store.py tests/test_metadata.py -x -q --ignore=tests/test_tensorflow_handler.py` | yes [VERIFIED: rg --files tests] |
| STRG-03 | Write intents cover pre-blob window and cleanup tolerates missing blobs/committed metadata. [VERIFIED: 31-CONTEXT.md] | fault-injection/storage-mode | `uv run --python 3.12 pytest tests/test_write_intent.py tests/test_atomic_writes.py tests/test_storage_mode.py -x -q --ignore=tests/test_tensorflow_handler.py` | yes [VERIFIED: rg --files tests] |

### Sampling Rate

- **Per task commit:** run the task-specific command above. [VERIFIED: 31-VALIDATION.md]
- **Per wave merge:** run the focused Phase 31 quick command. [VERIFIED: 31-VALIDATION.md]
- **Phase gate:** full suite green before verification, or document unrelated baseline failure with evidence. [VERIFIED: 31-VALIDATION.md; .github/copilot-instructions.md]

### Wave 0 Gaps

- Replace `tests/test_security.py` references in plans with `tests/test_cache_signing.py` and/or `tests/test_encryption_at_rest.py` because `tests/test_security.py` is absent. [VERIFIED: rg --files tests; 31-VALIDATION.md]
- Add targeted SEC-01 downgrade tests if not already present; current greps did not find `minimum_signature_version`. [VERIFIED: rg output]
- Add targeted SEC-02 encrypted `InMemoryBlobBackend` and no-stale-decrypt-temp tests. [VERIFIED: tests/test_blob_store.py; tests/test_encryption_at_rest.py]
- Add targeted SEC-03 interrupted-rotation fault tests. [VERIFIED: tests/test_key_rotation_api.py]
- Add STRG-01 warning assertions. [VERIFIED: tests/test_storage_mode.py]
- Add STRG-02 fsync hook tests after config location is chosen. [VERIFIED: tests/test_config_options.py; tests/test_write_intent.py]

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | no | No user authentication is introduced by Phase 31. [VERIFIED: phase scope] |
| V3 Session Management | no | No sessions are introduced by Phase 31. [VERIFIED: phase scope] |
| V4 Access Control | yes | Storage-mode destructive APIs must warn before durable deletion; no auth layer is added. [VERIFIED: 31-CONTEXT.md; src/cacheness/core.py] |
| V5 Input Validation | yes | Validate signature versions and reject versions below configured minimum. [VERIFIED: src/cacheness/security.py; 31-CONTEXT.md] |
| V6 Cryptography | yes | Use existing HMAC/HKDF and AES-GCM helpers; do not hand-roll crypto. [VERIFIED: src/cacheness/security.py; src/cacheness/encryption.py] |
| V8 Data Protection | yes | Encrypted reads should avoid plaintext temp files when handlers support byte reads; fsync docs must not overclaim durability. [VERIFIED: src/cacheness/storage/blob_store.py; docs/TRANSACTION_GUARANTEES.md] |
| V10 Malicious Code | yes | Unsigned-entry risk matters because pickle deserialization can run attacker-controlled payloads if metadata/blob integrity is bypassed. [VERIFIED: docs/CODE_REVIEW_FINDINGS.md; docs/SECURITY.md] |

### Known Threat Patterns for Cacheness

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Metadata attacker strips `entry_signature` while `allow_unsigned_entries=True`. [VERIFIED: docs/CODE_REVIEW_FINDINGS.md; docs/SECURITY.md] | Tampering / Elevation of Privilege | Document risk and recommend `allow_unsigned_entries=False` for attacker-writable metadata stores. [VERIFIED: 31-CONTEXT.md] |
| Signature version downgrade from `v3:` to `v2:`. [VERIFIED: docs/CODE_REVIEW_FINDINGS.md; src/cacheness/security.py] | Tampering | Enforce `minimum_signature_version`. [VERIFIED: 31-CONTEXT.md] |
| Plaintext leaks from encrypted read temp files after crash. [VERIFIED: docs/CODE_REVIEW_FINDINGS.md; src/cacheness/storage/blob_store.py] | Information Disclosure | Prefer `handler.get_bytes()` and harden temp fallback. [VERIFIED: 31-CONTEXT.md; tests/test_handler_bytes_protocol.py] |
| Interrupted rotation maroons old-key entries. [VERIFIED: docs/CODE_REVIEW_FINDINGS.md; src/cacheness/core.py] | Denial of Service / Tampering | Stage new key and replace active key last. [VERIFIED: 31-CONTEXT.md] |
| Storage-mode user accidentally calls cache eviction API. [VERIFIED: .planning/seeds/SEED-001-storage-mode-api-hardening.md; src/cacheness/core.py] | Denial of Service | Warning-first destructive API policy. [VERIFIED: 31-CONTEXT.md] |

## Sources

### Primary (HIGH confidence)

- `.planning/phases/31-security-storage-mode-posture/31-CONTEXT.md` - locked decisions, scope, discretion, deferred work. [VERIFIED: local read]
- `.planning/phases/31-security-storage-mode-posture/31-VALIDATION.md` - validation strategy and per-requirement checks. [VERIFIED: local read]
- `.planning/ROADMAP.md` and `.planning/REQUIREMENTS.md` - Phase 31 goal and requirement wording. [VERIFIED: local read]
- `src/cacheness/config.py` - `SecurityConfig`, storage-mode config overrides. [VERIFIED: local read]
- `src/cacheness/security.py` - signature parsing and verification behavior. [VERIFIED: local read]
- `src/cacheness/_verification_mixin.py` - canonical UnifiedCache signing fields and verification behavior. [VERIFIED: local read]
- `src/cacheness/storage/blob_store.py` - BlobStore signing, encrypted reads, rotation, temp plaintext fallback. [VERIFIED: local read]
- `src/cacheness/core.py` and `src/cacheness/_storage_mode_mixin.py` - put paths, cleanup APIs, rotation, storage-mode behavior. [VERIFIED: local read]
- `src/cacheness/write_intent.py`, `src/cacheness/storage/backends/blob_backends.py`, `src/cacheness/metadata/json_backend.py` - intent cleanup, local atomic blob writes, JSON persistence. [VERIFIED: local read]
- Relevant tests listed in the user prompt and discovered test inventory. [VERIFIED: local read and `rg --files tests`]

### Secondary (MEDIUM confidence)

- `docs/CODE_REVIEW_FINDINGS.md`, `docs/CODE_REVIEW_ACTIONS.md`, and seeds SEED-001/004/005/006 - source review findings and accepted directions, with some stale implementation details relative to current code. [CITED: local review docs]
- `docs/SECURITY.md` and `docs/TRANSACTION_GUARANTEES.md` - current documentation targets. [CITED: local docs]

### Tertiary (LOW confidence)

- No web-only findings were used. [VERIFIED: no web research needed]
- The configured `gsd-tools query research-plan`, `research-store`, and `classify-confidence` seams are unavailable in this local GSD version, so confidence is assigned from local source verification rather than those seam commands. [VERIFIED: gsd-tools command output]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - no new dependencies; all stack claims came from `pyproject.toml` and command output. [VERIFIED: pyproject.toml; command output]
- Architecture: HIGH - implementation seams were read directly in source files. [VERIFIED: local source reads]
- Pitfalls: HIGH - missing test file, direct encrypted reads, and immediate key overwrite were confirmed by grep/source reads. [VERIFIED: rg output; source reads]
- Storage-mode warning mechanism: MEDIUM - warning-first is locked, but exact mechanism is discretionary. [VERIFIED: 31-CONTEXT.md]
- `fsync_on_write` config placement: MEDIUM - opt-in is locked, but placement is discretionary. [VERIFIED: 31-CONTEXT.md]

**Research date:** 2026-06-15
**Valid until:** 2026-07-15 for current codebase planning, or earlier if Phase 31 code changes land before planning. [ASSUMED]

## RESEARCH COMPLETE
