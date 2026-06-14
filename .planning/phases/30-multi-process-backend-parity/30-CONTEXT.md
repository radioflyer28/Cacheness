# Phase 30: Multi-Process & Backend Parity - Context

**Gathered:** 2026-06-14
**Status:** Ready for planning
**Source:** User requested context synthesis from `docs/CODE_REVIEW_FINDINGS.md` and `docs/CODE_REVIEW_ACTIONS.md`

<domain>

## Phase Boundary

Phase 30 implements code-review Wave 3, scoped to TASK-9 through TASK-12:

- TASK-9 / R11: use unique temp names for filesystem blob writes to avoid cross-process write races.
- TASK-10 / U2: preserve user metadata on SQLite and PostgreSQL with parity against JSON.
- TASK-11 / R7: make same-key overwrite failure non-destructive in cache mode and storage mode.
- TASK-12 / R12: make filesystem blob enumeration backend-driven enough for integrity checks and namespace cleanup to see custom-handler blobs.

This phase covers requirements `PAR-01`, `PAR-02`, `PAR-03`, and `PAR-04` only.

</domain>

<decisions>

## Implementation Decisions

### Multi-Process Blob Writes

- **D-01:** Implement TASK-9 for `FilesystemBlobBackend.write_blob`: deterministic `<final>.tmp` temp paths must be replaced with unique temp file names.
- **D-02:** Unique temp files should use a `tempfile.mkstemp(dir=blob_path.parent, suffix=".tmp")` style approach, write through the returned file descriptor, and publish with `os.replace(tmp, blob_path)`.
- **D-03:** Failed writes must clean up their unique temp file in an exception path without deleting the previously committed final blob.
- **D-04:** PAR-01 verification must prove repeated same-blob writes leave the final content equal to the last successful write. If practical, include a concurrency-style regression; at minimum, cover the sequential same-blob overwrite acceptance from TASK-9.
- **D-05:** Reuse existing JSON backend atomic-save patterns as a reference for unique temp file behavior, but do not change JSON metadata save semantics in this phase unless a tiny shared helper is clearly lower risk.

### SQLite/PostgreSQL User Metadata Parity

- **D-06:** Implement TASK-10 for SQLite `put_entry` and PostgreSQL `_upsert_entry`: leftover nested `metadata` keys must not be silently discarded after known fields are popped.
- **D-07:** If leftover user metadata exists and `metadata_dict_value` is absent, serialize the leftovers into `metadata_dict` using the project's JSON utility patterns.
- **D-08:** If `metadata_dict_value` already exists, merge leftover user metadata into it while preserving core-provided `metadata_dict_value` keys on conflicts.
- **D-09:** SQLite and PostgreSQL read paths must expose deserialized `metadata_dict` values in the nested `metadata` dict in the same observable shape JSON provides.
- **D-10:** PAR-02 verification must prove `BlobStore.put(key, data, metadata={"experiment": "x42"})`, `get_metadata(key)["experiment"] == "x42"`, and `list_keys(metadata_filter={"experiment": "x42"}) == [key]` across JSON and SQLite. PostgreSQL coverage should be planned explicitly behind the existing availability/skip gates.

### Non-Destructive Same-Key Overwrite Failure

- **D-11:** Implement TASK-11 for both cache mode `put()` and storage mode `_storage_mode_put()`. The storage-mode variant is mandatory because R7 is durable-data loss there.
- **D-12:** A failed same-key overwrite must leave the previous committed value readable and its blob file present.
- **D-13:** Use the minimal-risk snapshot strategy from `docs/CODE_REVIEW_ACTIONS.md`: before writing a new blob that resolves to the same local path as an existing entry, move the old blob to `<path>.prev`.
- **D-14:** On successful metadata commit and cleanup commit, delete `<path>.prev`; on rollback, restore `<path>.prev` to the original path.
- **D-15:** Extend `_PutCleanup` with explicit previous-blob tracking and rollback restoration instead of scattering ad hoc cleanup across `put()` and `_storage_mode_put()`.
- **D-16:** Skip previous-blob snapshot/restore handling for remote URI paths and inline entries; this phase's approved TASK-11 fix is for same-local-path overwrite rollback.
- **D-17:** PAR-03 verification must cover cache mode and `storage_mode=True`: put key with value A, force the next metadata write to raise, attempt value B, assert the exception occurs, then assert `get(key)` returns A and the old blob still exists.

### Backend-Driven Blob Enumeration

- **D-18:** Implement TASK-12 in `FilesystemBlobBackend.list_blobs`: replace hardcoded extension whitelist enumeration with directory enumeration under namespace blob directories.
- **D-19:** Enumeration must include custom-handler files and inline fallback extensions such as `.custom` or `.bin`.
- **D-20:** Enumeration must exclude reserved files and directories, including `.intents/`, metadata JSON/DB files, signing/key files, and temp files such as `*.tmp`.
- **D-21:** Prefer a shared `RESERVED_FILE_PATTERNS` or similar module-level constant if it reduces duplication with Phase 28 clear-all cleanup rules, but keep the Phase 30 change scoped to enumeration/integrity behavior.
- **D-22:** PAR-04 verification must create a fake custom-extension blob in a namespace directory and assert `verify_integrity()` reports it as orphaned; repair behavior may be included if already covered by the existing integrity test pattern.

### Scope and Risk Fences

- **D-23:** Do not re-open Phase 29 TTL/eviction behavior except for test compatibility with Phase 30 changes. TTL-01 through TTL-04 are already verified complete.
- **D-24:** Do not implement Phase 31 security/storage-mode policy decisions in Phase 30. The storage-mode destructive API policy, fsync/durability contract, signature downgrade policy, encryption read path, and key rotation work remain Phase 31 scope.
- **D-25:** Do not implement Phase 32 small polish items in Phase 30, including metadata dict mutation, sanitized key collision hardening, hot-path double-read cleanup, package version metadata, absolute blob id rejection, SQLite PRAGMA lifecycle, S3 delete failure reporting, or root `UnifiedCache` re-export.
- **D-26:** Use verify-first checks for each TASK-9 through TASK-12 item where the current bug is readily reproducible.
- **D-27:** Plans should keep TASK-9 through TASK-12 traceable. Preferred granularity is four atomic plans unless the planner finds a strong reason to combine adjacent low-risk test-only or helper-only work.
- **D-28:** Use `uv` for all commands. On Windows, always include `--ignore=tests/test_tensorflow_handler.py`.
- **D-29:** For changed Python files, quality gates are `ruff format`, `ruff check --fix`, `ruff check`, and `ty check` on touched files. Existing baseline `ty` diagnostics should be documented rather than hidden if unrelated.
- **D-30:** Preserve existing PostgreSQL Docker/service availability behavior. Plan PostgreSQL parity tests explicitly, but do not require a local PostgreSQL service when the existing test suite would skip it.

### the agent's Discretion

- Whether TASK-9 should add only sequential same-blob regression coverage or a true multi-thread/multi-process stress test, provided the deterministic temp-name race is prevented in source.
- Whether TASK-10 user metadata parity tests live in `tests/test_blob_store.py`, `tests/test_backend_parity.py`, or backend-specific files, provided JSON and SQLite behavior are compared directly and PostgreSQL semantics are represented.
- Whether the previous-blob restore helper for TASK-11 lives in `_PutCleanup`, a small adjacent helper, or existing put-path code, provided both cache mode and storage mode use the same rollback semantics.
- Whether TASK-12 extracts reserved-file matching into a shared constant now or leaves a narrowly-scoped exclusion helper in `blob_backends.py`.

</decisions>

<canonical_refs>

## Canonical References

Downstream agents MUST read these before planning or implementing.

### Phase Definition

- `.planning/ROADMAP.md` - Phase 30 goal, requirements, success criteria, and source mapping to TASK-9 through TASK-12.
- `.planning/REQUIREMENTS.md` - `PAR-01` through `PAR-04` requirement wording.

### Source Review Docs

- `docs/CODE_REVIEW_FINDINGS.md` - Findings R7, R11, R12, U2, storage-mode impact analysis, and Wave 3 ordering.
- `docs/CODE_REVIEW_ACTIONS.md` - TASK-9 through TASK-12 files, concrete change directions, acceptance criteria, and tiered test commands.

### Code Areas

- `src/cacheness/storage/backends/blob_backends.py` - `FilesystemBlobBackend.write_blob`, `FilesystemBlobBackend.list_blobs`, filesystem namespace layout, temp-file handling, and in-memory/blob backend contracts.
- `src/cacheness/storage/blob_store.py` - `BlobStore.put`, `get_metadata`, `list_keys`, `verify_integrity`, repair behavior, and metadata filter behavior.
- `src/cacheness/metadata/sqlite_backend.py` - SQLite `put_entry`, known-field popping, `metadata_dict` storage/readback, `list_entries`, custom metadata and filter behavior.
- `src/cacheness/storage/backends/postgresql_backend.py` - PostgreSQL `_upsert_entry`, `metadata_dict` storage/readback, list/filter behavior, and existing Docker skip gates.
- `src/cacheness/core.py` - Cache-mode `put()`, blob write flow, metadata commit flow, old blob cleanup, and rollback path.
- `src/cacheness/_storage_mode_mixin.py` - Storage-mode `_storage_mode_put()` and durable same-key overwrite behavior.
- `src/cacheness/_put_cleanup.py` - Rollback/commit cleanup state, blob unlinking, and the right home for previous-blob restore handling.
- `src/cacheness/json_utils.py` - JSON serialization helpers to use for `metadata_dict` merge/serialization.
- `src/cacheness/config.py` - Storage-mode config guards and backend configuration.

### Testing Guidance

- `.planning/codebase/TESTING.md` - Test commands, backend parity patterns, Docker skip behavior, and Windows TensorFlow exclusion.
- `tests/test_blob_store.py` - BlobStore metadata, blob write/read, integrity, repair, and `list_keys(metadata_filter=...)` coverage.
- `tests/test_blob_namespace.py` - Namespace blob layout and filesystem blob backend behavior.
- `tests/test_backend_parity.py` - JSON/SQLite/PostgreSQL parity conventions and availability skip patterns.
- `tests/test_metadata.py` - Metadata backend CRUD, `metadata_dict`, and backend behavior tests.
- `tests/test_core.py` - Cache-mode `put()` behavior and overwrite failure regressions.
- `tests/test_fault_injection.py` - Failure injection and rollback patterns for metadata/blob write failures.
- `tests/test_cache_integrity.py` - Cache integrity regression tests.
- `tests/test_cache_integrity_verification.py` - Orphan/dangling blob detection and repair behavior.
- `tests/test_storage_mode.py` - Storage-mode put/get and durability behavior.
- `tests/test_postgresql_backend.py` - PostgreSQL backend semantics behind existing availability gates.

</canonical_refs>

<specifics>

## Specific Ideas

- TASK-9 should verify the implementation no longer uses `<final>.tmp` as a deterministic temp name and that repeated writes to the same `blob_id` leave the last content readable.
- TASK-10 parity tests should use the public `BlobStore` API rather than only backend internals because the user-visible failure is `get_metadata()` and `list_keys(metadata_filter=...)`.
- TASK-11 tests should monkeypatch `metadata_backend.put_entry` to raise on a same-key overwrite after value A was committed. After the failed value B write, reads must return A in both cache mode and storage mode.
- TASK-11 should explicitly assert the old blob file still exists after rollback and no dangling metadata points to a missing local blob.
- TASK-12 tests should create a custom-extension file under the namespace blob directory without metadata and verify `verify_integrity()` detects it as orphaned.
- Tier-1 command hints from `docs/CODE_REVIEW_ACTIONS.md`:
  - TASK-9: `uv run pytest tests/test_blob_store.py tests/test_blob_namespace.py -x -q --ignore=tests/test_tensorflow_handler.py`
  - TASK-10: `uv run pytest tests/test_blob_store.py tests/test_metadata.py tests/test_backend_parity.py -x -q --ignore=tests/test_tensorflow_handler.py`
  - TASK-11: `uv run pytest tests/test_core.py tests/test_fault_injection.py tests/test_cache_integrity.py tests/test_storage_mode.py -x -q --ignore=tests/test_tensorflow_handler.py`
  - TASK-12: `uv run pytest tests/test_blob_store.py tests/test_cache_integrity_verification.py -x -q --ignore=tests/test_tensorflow_handler.py`

</specifics>

<deferred>

## Deferred Ideas

- Phase 31: SEC-01 through SEC-04 and STRG-01 through STRG-03, including storage-mode destructive API policy, fsync/durability contract, signature downgrade hardening, encryption read path, and two-phase key rotation.
- Phase 32: POL-01 through POL-08 release polish and small fixes.
- Phase 28/29 work is complete or separately tracked; do not fold silent-data-loss or TTL/eviction consistency work into Phase 30 except where directly required for PAR-01 through PAR-04.
- True S3 integration changes are not required for Phase 30 unless needed by existing blob enumeration abstractions. Remote deletion behavior was Phase 29 scope.

</deferred>

---

*Phase: 30-multi-process-backend-parity*
*Context gathered: 2026-06-14 from code review findings/actions*
