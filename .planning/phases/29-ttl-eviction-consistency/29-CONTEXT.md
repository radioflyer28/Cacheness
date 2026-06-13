# Phase 29: TTL & Eviction Consistency - Context

**Gathered:** 2026-06-13
**Status:** Ready for planning
**Source:** User requested context synthesis from `docs/CODE_REVIEW_FINDINGS.md` and `docs/CODE_REVIEW_ACTIONS.md`

<domain>

## Phase Boundary

Phase 29 implements code-review Wave 2, scoped to TASK-5 through TASK-8:

- TASK-5 / R5: honor stored per-entry TTL fields end-to-end.
- TASK-6 / R6: make init-time expired cleanup delete blob files through the public cleanup path.
- TASK-7 / R9/R10: preserve access counters, provenance timestamps, expiry semantics, and signatures across overwrite and metadata-only updates.
- TASK-8 / R13: delete remote blobs during size-limit eviction through the configured blob backend.

This phase covers requirements `TTL-01`, `TTL-02`, `TTL-03`, and `TTL-04` only.

</domain>

<decisions>

## Implementation Decisions

### TTL Semantics

- **D-01:** Implement the approved TASK-5 direction: stored `expires_at` wins when present; caller/config TTL applies only when an entry has no stored expiry.
- **D-02:** `_is_expired()` must parse `entry["expires_at"]` when set. String values use `datetime.fromisoformat`; naive datetimes are treated as UTC.
- **D-03:** When `expires_at` is present, `_is_expired()` ignores the `ttl_seconds` argument for that entry.
- **D-04:** Backend `cleanup_expired(ttl_seconds)` implementations for JSON, SQLite, and PostgreSQL must delete entries where `expires_at < now` when `expires_at` is set, or where `expires_at` is null and `created_at < cutoff`.
- **D-05:** Do not remove the v3 TTL columns. Phase 29 chooses to honor them, not to drop them.
- **D-06:** Storage mode must remain protected from cache expiry behavior. `_storage_mode_get` must not call `_is_expired()`, and a storage-mode read of an entry with past `expires_at` must still return the data.

### Expired Cleanup Path

- **D-07:** Implement TASK-6 by changing `UnifiedCache._cleanup_expired()` to call the public `self.cleanup_expired(ttl_seconds)` path instead of `self.metadata_backend.cleanup_expired(ttl_seconds)`.
- **D-08:** Init-time expired cleanup must remove both metadata and blob files and must preserve the public cleanup path's hook behavior, including `on_evict`.
- **D-09:** Confirm `_cleanup_expired()` is called after the instance lock exists and while that lock is not already held. Existing RLock behavior may be relied on only after reading current `UnifiedCache.__init__`.
- **D-10:** The read-path expired-entry issue from R6 is not mandatory Phase 29 scope. It may be added only if it is a small, well-tested extension that does not blur TASK-6's init-cleanup acceptance.

### Metadata Counters, Timestamps, TTL, and Signatures

- **D-11:** Implement TASK-7 R9 for SQLite and PostgreSQL: same-key overwrites must not reset existing `access_count`.
- **D-12:** SQLite `put_entry` should move away from `INSERT OR REPLACE` for overwrites and use conflict-update semantics that omit `access_count` from the update set so existing access counts survive. New inserts still initialize `access_count` from input/default values.
- **D-13:** PostgreSQL `_upsert_entry` must apply the same access-count preservation rule as SQLite.
- **D-14:** Implement TASK-7 R10 for JSON, SQLite, and PostgreSQL: `update_entry_metadata()` must not reset `created_at`.
- **D-15:** Metadata-only updates must not alter `expires_at` or `ttl_seconds` unless the caller explicitly updates those fields through an existing supported path.
- **D-16:** Because `created_at` is signed, update paths must preserve or recompute signatures using the preserved `created_at`. Read `_update_mixin.py` before implementing metadata update changes.
- **D-17:** `update_data()` is the content-update path and may continue to refresh content metadata according to existing semantics, but metadata-only updates must not masquerade as content rewrites.
- **D-18:** Storage mode amplifies R10: users rely on `created_at` as durable provenance. Do not introduce storage-mode behavior that resets provenance timestamps during metadata-only operations.

### Remote Blob Eviction

- **D-19:** Implement TASK-8 in `UnifiedCache._enforce_size_limit()`: evicted entries whose `actual_path` contains `://` must be deleted through `self._blob_store.blob_backend.delete_blob(actual_path)`.
- **D-20:** Local filesystem blob deletion may keep the existing local unlink behavior, but remote URI deletion must no longer be skipped.
- **D-21:** Remote deletion failures should be caught and logged with `logger.warning`; they must not crash size-limit enforcement unless the existing local path behavior already does so.
- **D-22:** Use `InMemoryBlobBackend` and its `memory://` URIs as the fast local regression for remote-style deletion. S3 behavior is the production rationale, but tests do not need real S3 if the backend-abstraction contract is covered.
- **D-23:** Storage mode size limits are forced off, so TASK-8 must not add storage-mode eviction behavior.

### Planning and Test Packaging

- **D-24:** Preserve Phase 29 as Wave 2 consistency work. Do not silently include Phase 30 parity work, Phase 31 security/storage-mode policy work, or Phase 32 polish items.
- **D-25:** Plans should keep TASK-5 through TASK-8 traceable. The preferred granularity is four atomic plans unless the planner finds a strong reason to combine adjacent low-risk changes.
- **D-26:** Verify-first checks should be used for each task where the current bug is readily reproducible.
- **D-27:** Use `uv` for all commands. On Windows, always include `--ignore=tests/test_tensorflow_handler.py`.
- **D-28:** For changed Python files, quality gates are `ruff format`, `ruff check --fix`, `ruff check`, and `ty check` on touched files. Existing baseline `ty` diagnostics should be documented rather than hidden if unrelated.

### the agent's Discretion

- Whether to add read-path expired-entry deletion on `get()` as part of TASK-6, provided TASK-6's required init-cleanup behavior remains clear and independently verified.
- Whether to place PostgreSQL-specific coverage in backend unit tests, parity tests, or existing Docker-skipping PostgreSQL tests, as long as PostgreSQL semantics are planned explicitly.
- Whether to use direct backend tests or full `UnifiedCache` integration tests for access-count preservation, as long as at least one integration path proves cache-mode behavior.
- Whether to add a small helper for parsing `expires_at` if it reduces duplication across `_is_expired()` and backend cleanup code.

</decisions>

<canonical_refs>

## Canonical References

Downstream agents MUST read these before planning or implementing.

### Phase Definition

- `.planning/ROADMAP.md` - Phase 29 goal, requirements, success criteria, and source mapping to TASK-5 through TASK-8.
- `.planning/REQUIREMENTS.md` - `TTL-01` through `TTL-04` requirement wording.

### Source Review Docs

- `docs/CODE_REVIEW_FINDINGS.md` - Findings R5, R6, R9, R10, R13, storage-mode impact analysis, PostgreSQL/S3 secondary scope, and recommended Wave 2 fix order.
- `docs/CODE_REVIEW_ACTIONS.md` - TASK-5 through TASK-8 files, concrete change directions, acceptance criteria, and tiered test commands.

### Code Areas

- `src/cacheness/core.py` - `_is_expired`, `_cleanup_expired`, `cleanup_expired`, `_enforce_size_limit`, and cache-mode put/get behavior.
- `src/cacheness/metadata/json_backend.py` - JSON `cleanup_expired`, `update_entry_metadata`, TTL fields, and access-count metadata.
- `src/cacheness/metadata/sqlite_backend.py` - SQLite `put_entry`, `cleanup_expired`, `update_entry_metadata`, v3 TTL/access-count fields, and schema indexes.
- `src/cacheness/storage/backends/postgresql_backend.py` - PostgreSQL `_upsert_entry`, `cleanup_expired`, `update_entry_metadata`, and S3-path metadata fields.
- `src/cacheness/_update_mixin.py` - `update_metadata`, `update_data`, signing, and created-at preservation risk.
- `src/cacheness/storage/backends/blob_backends.py` - `BlobBackend.delete_blob` and `InMemoryBlobBackend` `memory://` URI behavior.
- `src/cacheness/storage/backends/s3_backend.py` - Production remote deletion rationale for S3 blob URIs.
- `src/cacheness/config.py` - Storage-mode guards that disable TTL, size limits, stats, and cleanup behaviors.

### Testing Guidance

- `.planning/codebase/TESTING.md` - Test commands, backend parity patterns, Docker skip behavior, and Windows TensorFlow exclusion.
- `tests/test_core.py` - Core expiry, cleanup, eviction, and blob deletion integration tests.
- `tests/test_metadata.py` - Metadata backend CRUD and persistence behavior.
- `tests/test_backend_parity.py` - Cross-backend behavioral equivalence.
- `tests/test_update_operations.py` - `update_data()` and metadata-update coverage.
- `tests/test_security.py`, `tests/test_namespace_signing.py`, `tests/test_cache_signing.py` - Signature verification coverage.
- `tests/test_storage_mode.py` - Storage-mode non-expiry and no-eviction guarantees.
- `tests/test_blob_store.py` - Blob backend and in-memory blob behavior.
- `tests/test_s3_blob_backend.py`, `tests/test_s3_orphan_cleanup.py` - S3 and remote delete/rollback patterns.
- `tests/test_sqlite_schema_versioning.py`, `tests/test_json_schema_versioning.py`, `tests/test_pg_schema_versioning.py`, `tests/test_postgresql_backend.py` - Backend-specific TTL/access-count schema behavior.

</canonical_refs>

<specifics>

## Specific Ideas

- TASK-5 parity tests should include:
  - `expires_at` in the past and recent `created_at`: cleanup with large fallback TTL deletes the entry.
  - `expires_at` in the future and old `created_at`: cleanup with small fallback TTL keeps the entry.
  - storage-mode `get()` returns data even when stored `expires_at` is in the past.
- TASK-6 regression should create an expired entry with a blob, reinitialize with init cleanup enabled, and assert metadata and blob are both gone.
- TASK-7 regressions should prove:
  - repeated `get()` increments `access_count`;
  - same-key overwrite preserves prior `access_count` for SQLite and PostgreSQL semantics;
  - `update_metadata(key, description="new")` leaves `created_at` unchanged and `get()` still verifies the entry signature.
- TASK-8 regression should configure an in-memory blob backend, force size-limit eviction, and assert the evicted `memory://` blob no longer exists in the backend store.
- Tier-1 command hints from `docs/CODE_REVIEW_ACTIONS.md`:
  - TASK-5: `uv run pytest tests/test_metadata.py tests/test_core.py tests/test_backend_parity.py tests/test_storage_mode.py -x -q --ignore=tests/test_tensorflow_handler.py`
  - TASK-6: `uv run pytest tests/test_core.py -x -q --ignore=tests/test_tensorflow_handler.py`
  - TASK-7: `uv run pytest tests/test_metadata.py tests/test_update_operations.py tests/test_backend_parity.py tests/test_security.py -x -q --ignore=tests/test_tensorflow_handler.py`
  - TASK-8: `uv run pytest tests/test_core.py tests/test_blob_store.py -x -q --ignore=tests/test_tensorflow_handler.py`

</specifics>

<deferred>

## Deferred Ideas

- Phase 30: PAR-01 through PAR-04, including unique temp blob names, SQLite/PostgreSQL user metadata parity, non-destructive overwrite, and backend-driven blob enumeration.
- Phase 31: SEC-01 through SEC-04 and STRG-01 through STRG-03, including storage-mode destructive API policy and fsync/durability contract.
- Phase 32: POL-01 through POL-08 release polish and small fixes.
- R11, R12, R7, U2, S1 through S4, and POL items are out of Phase 29 unless needed only as tiny compatibility adjustments to land TASK-5 through TASK-8 safely.

</deferred>

---

*Phase: 29-ttl-eviction-consistency*
*Context gathered: 2026-06-13 from code review findings/actions*
