# Phase 31: Security & Storage-Mode Posture - Context

**Gathered:** 2026-06-14
**Status:** Ready for planning
**Source:** User requested Phase 31 planning with selections sourced from `docs/CODE_REVIEW_FINDINGS.md` and `docs/CODE_REVIEW_ACTIONS.md`.

<domain>

## Phase Boundary

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

</domain>

<decisions>

## Implementation Decisions

### Signature Downgrade and Unsigned-Entry Risk

- **D-01:** Implement TASK-13 for `SecurityConfig`, `verify_entry()`, and verification callers: users must be able to configure a minimum accepted signature version.
- **D-02:** Add `minimum_signature_version: int = 1` to `SecurityConfig` without changing the existing default acceptance behavior for old caches.
- **D-03:** `verify_entry()` must reject a signature whose parsed version is lower than `minimum_signature_version` by returning `False`.
- **D-04:** New deployments should be documented as recommended to use `minimum_signature_version=3`.
- **D-05:** Do not flip `allow_unsigned_entries` in this phase. The review docs call that a breaking change and keep it as future/backlog work.
- **D-06:** `docs/SECURITY.md` must explicitly state that `allow_unsigned_entries=True` lets an attacker with metadata write access strip signatures and bypass signature verification.
- **D-07:** `docs/SECURITY.md` must recommend `allow_unsigned_entries=False` for threat models where the metadata store is attacker-writable.
- **D-08:** SEC-01 verification must include a downgrade regression: sign at v3, set `minimum_signature_version=3`, rewrite the stored signature prefix to v2, and assert verification fails.

### Encrypted Read Path

- **D-09:** Implement TASK-14 for encrypted reads in `BlobStore`: encrypted ciphertext must be read through `self.blob_backend.read_blob(actual_path_str)`, not direct filesystem reads.
- **D-10:** The encrypted read path must support non-filesystem backends such as `memory://` and should preserve S3/backend abstraction behavior.
- **D-11:** After decrypting, the read path must first try `handler.get_bytes(plaintext, metadata)` so handlers with byte support avoid plaintext temp files.
- **D-12:** Temp-file fallback is allowed only when `handler.get_bytes()` raises `NotImplementedError`.
- **D-13:** Any temp-file fallback for decrypted plaintext must use `tempfile.mkstemp` in the cache directory, set POSIX permissions to `0600` where applicable, and unlink in `finally`.
- **D-14:** SEC-02 verification must prove encrypted `InMemoryBlobBackend` entries round-trip and encrypted pickle reads leave no stale decrypt temp files.

### Two-Phase Key Rotation

- **D-15:** Implement TASK-15 for both `UnifiedCache.rotate_key()` and `BlobStore.rotate_key()`.
- **D-16:** Key rotation must write the new key to `<keyfile>.new` first and must not overwrite the active key file until all entries are successfully re-signed and re-encrypted.
- **D-17:** Rotation must construct a second signer/encryptor from the new key while keeping the old key available for verification during the rotation pass.
- **D-18:** Each entry must be verified with the old key before being signed with the new key.
- **D-19:** Re-encrypted local blobs must be written through an atomic temp path such as `<blob>.rotating` plus `os.replace`, not in-place truncating writes.
- **D-20:** Only after all entries succeed may rotation replace `<keyfile>.new` over the active key file.
- **D-21:** Startup must detect a leftover `<keyfile>.new` and log an error telling the user rotation was interrupted; full resume is out of scope.
- **D-22:** SEC-03 verification must monkeypatch a mid-rotation failure and assert the original key file is unchanged and old entries still verify with the old key.

### Shared Canonical Signing Strategy

- **D-23:** Implement SEC-04 / SEED-004 by making `UnifiedCache` and `BlobStore` share a canonical signable-field extraction strategy.
- **D-24:** Prefer moving `_extract_signable_fields()` or its logic from `_verification_mixin.py` to a shared module/helper used by both APIs.
- **D-25:** New signatures produced by `UnifiedCache` and `BlobStore` must cover the same canonical field set for equivalent metadata.
- **D-26:** Existing signed entries must retain a compatibility path. Do not require users to strict-migrate or bulk re-sign existing caches in this phase.
- **D-27:** The compatibility path may accept old BlobStore signing shapes during verification, re-sign on mutation/read if the planner judges that safe, or use a documented signature version transition. It must be explicit and tested.
- **D-28:** Coordinate SEC-04 with SEC-01 so minimum signature version enforcement does not accidentally reject existing compatible entries unless the user configured that stricter policy.

### Storage-Mode Destructive API Policy

- **D-29:** Implement STRG-01 with a warning-first policy for storage-mode destructive cache APIs, sourced from the review docs' "recommend at least a loud warning log" guidance.
- **D-30:** Storage-mode destructive APIs must emit a loud warning when an explicit cache-eviction/destructive method could delete durable entries.
- **D-31:** The default Phase 31 behavior should not hard-raise for these APIs unless an existing API already does so or the implementation exposes an explicit stricter opt-in. This avoids silently changing public API semantics.
- **D-32:** The warning policy must cover at least public cleanup/eviction entry points that can delete storage-mode data, including `cleanup_expired(ttl_seconds=...)`, size-limit cleanup paths, and `clear_all()` or namespace-clearing APIs as applicable.
- **D-33:** Tests must assert the warning is emitted for storage-mode destructive calls and that no implicit TTL/eviction behavior is reintroduced under storage mode.
- **D-34:** Hard-refusal remains a future stricter option; if an opt-in strict mode is added, it must be documented as opt-in and not the default.

### Storage-Mode Durability and Fsync

- **D-35:** Implement STRG-02 with documentation plus an opt-in `fsync_on_write` policy where feasible, sourced from R16 and SEED-005.
- **D-36:** `docs/TRANSACTION_GUARANTEES.md` must clearly document current storage-mode durability limits for JSON saves, blob writes, and write-intent files, including power-loss caveats.
- **D-37:** If adding a config knob, prefer `fsync_on_write` defaulting to `False` so cache-mode and existing storage-mode performance are not silently changed.
- **D-38:** The fsync policy should cover local filesystem blob writes, JSON metadata saves, and write-intent files where local file descriptors are available.
- **D-39:** Do not force always-fsync behavior by default. The review docs say documentation-only is acceptable as a first step and durability changes need performance discussion.
- **D-40:** STRG-02 verification must include docs checks and targeted tests or helper-level tests showing opt-in fsync paths are invoked without requiring power-loss simulation.

### Pre-Blob Write Intent Coverage

- **D-41:** Implement STRG-03 / SEED-006 by recording write intents with the planned blob path before invoking handler serialization/encryption/blob write where needed.
- **D-42:** This must cover both cache-mode `put()` and storage-mode `_storage_mode_put()` unless the implementation proves one path no longer records intents.
- **D-43:** The intent cleanup logic must tolerate an intent that references a blob that was never created because the crash/failure happened before the blob write completed.
- **D-44:** Preserve Phase 28's safety invariant: stale-intent cleanup must not delete a blob when committed metadata for that key exists.
- **D-45:** STRG-03 verification must include a failure-before-blob-created or handler-raises case where a pre-recorded intent is left behind and later cleanup removes only the intent/no-op target safely.

### Scope and Risk Fences

- **D-46:** Do not implement Phase 32 polish items in Phase 31, including metadata dict mutation, sanitized key collision hardening, hot-path double-read cleanup, package version metadata, absolute blob id rejection, SQLite PRAGMA lifecycle, S3 delete failure reporting, or root `UnifiedCache` re-export.
- **D-47:** Do not re-open Phase 29 TTL/eviction semantics except where needed to warn for storage-mode destructive API calls.
- **D-48:** Do not re-open Phase 30 same-key overwrite or blob enumeration behavior except where required for rotation or fsync implementation details.
- **D-49:** Do not change public API signatures unless the plan explicitly justifies it from the review action docs or an opt-in config field is required for `minimum_signature_version` / `fsync_on_write`.
- **D-50:** Use `uv` for all commands. On Windows, always include `--ignore=tests/test_tensorflow_handler.py`.
- **D-51:** For changed Python files, quality gates are `ruff format`, `ruff check --fix`, `ruff check`, and `ty check` on touched files. Existing unrelated `ty` diagnostics should be documented rather than hidden.
- **D-52:** Existing PostgreSQL/S3 service availability and skip gates must be preserved. Phase 31 should not require local Docker or cloud services beyond existing test gates.
- **D-53:** Ignore beads for this workflow per the user's instruction.

### the agent's Discretion

- Whether to split SEC-04 signing convergence into its own plan or pair it with SEC-01, provided compatibility and minimum-version interactions are explicit.
- Whether `fsync_on_write` belongs under an existing storage/config dataclass or a narrow backend-local option, provided the user-visible config surface is documented.
- Whether storage-mode destructive API warnings use `logging.warning`, `warnings.warn`, or both, provided tests can assert the loud signal and docs explain it.
- Whether key rotation's `<blob>.rotating` behavior reuses existing atomic blob backend helpers or a rotation-specific helper, provided it does not truncate the committed blob in place.

</decisions>

<canonical_refs>

## Canonical References

Downstream agents MUST read these before planning or implementing.

### Phase Definition

- `.planning/ROADMAP.md` - Phase 31 goal, requirements, success criteria, and source mapping to TASK-13 through TASK-15 plus seeds.
- `.planning/REQUIREMENTS.md` - `SEC-01` through `SEC-04` and `STRG-01` through `STRG-03` requirement wording.

### Source Review Docs and Seeds

- `docs/CODE_REVIEW_FINDINGS.md` - Findings S1, S2, S3, S4, U4, R8, R16, storage-mode impact analysis, and the storage-mode destructive API open question.
- `docs/CODE_REVIEW_ACTIONS.md` - TASK-13 through TASK-15 files, change directions, acceptance criteria, tiered test commands, and out-of-scope decision items now promoted into Phase 31.
- `.planning/seeds/SEED-001-storage-mode-api-hardening.md` - STRG-01 warning-or-refuse decision context.
- `.planning/seeds/SEED-004-unify-signing-schemes.md` - SEC-04 shared signing strategy and compatibility context.
- `.planning/seeds/SEED-005-fsync-policy-storage-mode-durability.md` - STRG-02 fsync/durability policy context.
- `.planning/seeds/SEED-006-record-write-intent-before-blob-write.md` - STRG-03 pre-blob intent recording context.

### Code Areas

- `src/cacheness/config.py` - `SecurityConfig`, storage-mode config guards, and any `fsync_on_write` config placement.
- `src/cacheness/security.py` - signature parsing/version verification, key file writing, startup key behavior, signer/encryptor construction, and key rotation helpers.
- `src/cacheness/_verification_mixin.py` - current `_extract_signable_fields()`, verification behavior, delete-on-invalid behavior, and storage-mode delete guards.
- `src/cacheness/storage/blob_store.py` - encrypted read branches, `BlobStore.put()` signing, `BlobStore.rotate_key()`, temp plaintext fallback, and metadata read/write flow.
- `src/cacheness/core.py` - cache-mode `put()`, write-intent ordering, `rotate_key()`, public cleanup APIs, and storage-mode routing.
- `src/cacheness/_storage_mode_mixin.py` - storage-mode `put/get` behavior, write-intent ordering, and durability invariants.
- `src/cacheness/write_intent.py` - write-intent file creation, cleanup behavior, local fsync opportunities, and pre-blob-intent tolerance.
- `src/cacheness/storage/backends/blob_backends.py` - filesystem blob write atomicity, local fsync opportunities, and backend read/write abstractions.
- `src/cacheness/metadata/json_backend.py` - JSON metadata save behavior and local fsync opportunities.
- `docs/SECURITY.md` - unsigned-entry and signature-version documentation target.
- `docs/TRANSACTION_GUARANTEES.md` - storage-mode durability/fsync documentation target.

### Testing Guidance

- `.planning/codebase/TESTING.md` - Test commands, Windows TensorFlow exclusion, backend skip patterns, and quality gates.
- `tests/test_security.py` - signature verification, downgrade, unsigned-entry docs-adjacent behavior, key rotation, encryption/security coverage.
- `tests/test_blob_store.py` - BlobStore encrypted reads, backend-routed reads, signing, metadata, and rotation behavior.
- `tests/test_core.py` - `UnifiedCache` put/get, cleanup, rotation, and end-to-end behavior.
- `tests/test_storage_mode.py` - storage-mode durable behavior, destructive API warnings, and write-intent behavior.
- `tests/test_atomic_writes.py` - write-intent journal and crash-recovery regression patterns.
- `tests/test_encryption_at_rest.py` - encryption round-trip coverage if present/applicable.
- `tests/test_cache_integrity.py` and `tests/test_cache_integrity_verification.py` - integrity verification compatibility if signing changes touch read/verify paths.

</canonical_refs>

<specifics>

## Specific Ideas

- TASK-13 tier-1 hint: `uv run pytest tests/test_security.py -x -q --ignore=tests/test_tensorflow_handler.py`.
- TASK-14 tier-1 hint: `uv run pytest tests/test_blob_store.py tests/test_security.py tests/test_core.py -x -q --ignore=tests/test_tensorflow_handler.py`, plus encryption tests if present.
- TASK-15 tier-1 hint: `uv run pytest tests/test_security.py tests/test_core.py -x -q --ignore=tests/test_tensorflow_handler.py`.
- Storage-mode warning tests should include `tests/test_storage_mode.py` in Tier 1 because `CODE_REVIEW_ACTIONS.md` says any storage-mode deletion/cleanup path must include it.
- SEC-04 tests should create or mutate equivalent entries through `UnifiedCache` and `BlobStore`, verify both use the shared canonical signing fields for new writes, and include old-shape compatibility coverage.
- STRG-02 docs should distinguish atomic rename crash consistency from power-loss durability; tests should not pretend to simulate power loss.
- STRG-03 tests should verify stale cleanup handles both "intent exists and blob missing" and "intent exists and metadata already committed" safely.

</specifics>

<deferred>

## Deferred Ideas

- Flipping `allow_unsigned_entries` to `False` by default remains out of scope and belongs to future/breaking-change work.
- Strict signing migration that rejects all existing old-shape BlobStore signatures by default is out of scope.
- Always-on fsync by default is out of scope because the review docs call out performance tradeoffs.
- Full interrupted key-rotation resume is out of scope; Phase 31 must warn on `<keyfile>.new` and preserve the old key, not implement resumable rotation.
- JSON backend write batching/debouncing remains a future performance-policy item.
- Phase 32 polish items remain deferred to Phase 32.

</deferred>

---

*Phase: 31-security-storage-mode-posture*
*Context gathered: 2026-06-14 from code review findings/actions and seeds*
