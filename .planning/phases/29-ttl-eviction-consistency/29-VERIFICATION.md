---
phase: 29-ttl-eviction-consistency
verified: 2026-06-14T02:02:17Z
status: passed
score: 4/4 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 3/4
  gaps_closed:
    - "Eviction routes remote blob deletion through the blob backend and verifies S3/memory-style URI cleanup."
  gaps_remaining: []
  regressions: []
---

# Phase 29: ttl-eviction-consistency Verification Report

**Phase Goal:** Make TTL and eviction behavior coherent end-to-end across metadata backends, blob files, stats, and remote blob storage.
**Verified:** 2026-06-14T02:02:17Z
**Status:** passed
**Re-verification:** Yes - after final TTL-04 stabilization commit `093cda0`.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | Stored `expires_at` controls per-entry expiry when present; global TTL applies only as fallback. | VERIFIED | `UnifiedCache._is_expired()` reads stored `entry["expires_at"]` before fallback TTL. Public cleanup scans stored `expires_at` before created-at fallback. JSON, SQLite, and PostgreSQL cleanup paths all include stored-expiry predicates. Targeted TTL-01 regression checks passed. |
| 2 | Init-time cleanup uses the public cleanup path and removes both metadata and blob files. | VERIFIED | `UnifiedCache._cleanup_expired()` delegates to `self.cleanup_expired(ttl_seconds)`, and the public path deletes blob files and invokes eviction hooks before metadata cleanup. Targeted TTL-02 regression passed. |
| 3 | Metadata-only updates preserve `created_at`, signatures, and access-count semantics. | VERIFIED | SQLite conflict-update and PostgreSQL existing-row update paths omit `access_count`; metadata-only update paths only mutate supplied fields. Signed-entry metadata-only update regression passed. |
| 4 | Eviction routes remote blob deletion through the blob backend and verifies S3/memory-style URI cleanup. | VERIFIED | `_enforce_size_limit()` passes URI `actual_path` values through `self._blob_store.blob_backend.delete_blob(actual_path)`. `test_size_limit_eviction_deletes_memory_uri_blob` now uses deterministic bytes and derives `max_cache_size` from observed metadata size; focused TTL-04 slice passed, proving memory URI deletion and retained newest entry behavior. |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `src/cacheness/core.py` | Stored expiry read path, public cleanup routing, init cleanup delegation, URI size-eviction deletion. | VERIFIED | Exists, substantive, wired. Manual source checks verified `_is_expired`, `_cleanup_expired`, `cleanup_expired`, and `_enforce_size_limit`; targeted tests passed. |
| `src/cacheness/_update_mixin.py` | Content updates refresh timestamps and re-sign entries when needed. | VERIFIED | Exists and substantive; metadata-only preservation remains covered by backend/update/signing tests. |
| `src/cacheness/metadata/json_backend.py` | Stored expiry cleanup, metadata-only preservation, JSON size cleanup returning removed URI paths. | VERIFIED | Exists and substantive. `cleanup_by_size()` returns removed entries with `actual_path`; JSON stored-expiry and metadata-update tests passed. |
| `src/cacheness/metadata/sqlite_backend.py` | Stored expiry cleanup and access-count-preserving overwrite. | VERIFIED | Exists and substantive. `cleanup_expired()` includes stored-expiry predicates; conflict update omits `access_count`. |
| `src/cacheness/storage/backends/postgresql_backend.py` | Stored expiry cleanup and access-count-preserving overwrite. | VERIFIED | Exists and substantive. PostgreSQL cleanup mirrors stored-expiry semantics; existing-row update omits `access_count`. |
| `tests/test_core.py` | Core regressions for TTL-01, TTL-02, and TTL-04. | VERIFIED | Required tests exist and targeted slices passed, including the previously failing memory URI size-eviction regression. |
| `tests/test_metadata.py` | JSON/SQLite stored expiry cleanup regression. | VERIFIED | Stored-expiry cleanup regression exists and is wired to backend cleanup behavior. |
| `tests/test_backend_parity.py` | SQLite access-count and cleanup parity regressions. | VERIFIED | Cleanup parity and overwrite/access-count coverage remain present. |
| `tests/test_update_operations.py` | Metadata-only preservation regressions. | VERIFIED | JSON and SQLite metadata-only preservation regression passed. |
| `tests/test_storage_mode.py` | Storage mode ignores stored TTL and size enforcement remains disabled. | VERIFIED | Storage-mode TTL bypass and no-size-enforcement regressions passed. |
| `tests/test_cache_signing.py` | Signed-entry metadata-only update regression. | VERIFIED | Signed-entry read-after-metadata-update regression passed. |
| `tests/test_postgresql_backend.py` | PostgreSQL TTL/access-count regressions. | VERIFIED | PostgreSQL stored-expiry and access-count tests remain present behind existing availability gates. |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `UnifiedCache._is_expired` | `metadata_backend.get_entry` | Stored `expires_at` checked before fallback TTL | VERIFIED | Manual source check shows stored expiry returns before fallback TTL calculation. |
| `UnifiedCache.cleanup_expired` | `metadata_backend.cleanup_expired` | Public scan deletes blobs/hooks, backend removes metadata | VERIFIED | Public cleanup scans `iter_entry_summaries()`, deletes local blob files, invokes `on_evict`, then calls backend cleanup. |
| `UnifiedCache._cleanup_expired` | `UnifiedCache.cleanup_expired` | Direct method call | VERIFIED | Init-time cleanup delegates to the public cleanup path. |
| `UnifiedCache.get` | `_storage_mode_get` | Storage-mode guard before expiry checks | VERIFIED | `get()` returns through `_storage_mode_get(cache_key)` before `_is_expired()` when storage mode is enabled. |
| `SqliteBackend.put_entry` | Existing row `access_count` | `ON CONFLICT` update omits `access_count` | VERIFIED | Manual source check confirms overwrite values update content/provenance fields without resetting `access_count`. |
| `PostgresBackend._upsert_entry` | Existing row `access_count` | Existing-row update omits `access_count` | VERIFIED | Manual source check confirms update values do not include `access_count`; insert initializes it. |
| `UnifiedCache._enforce_size_limit` | `self._blob_store.blob_backend.delete_blob` | URI `actual_path` branch | VERIFIED | Manual source check confirms URI paths call backend deletion and warning-path exceptions are swallowed with logs. TTL-04 focused slice passed. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|---|---|---|---|---|
| `UnifiedCache._is_expired` | `entry["expires_at"]`, `entry["created_at"]` | `metadata_backend.get_entry(cache_key)` | Yes | VERIFIED |
| `UnifiedCache.cleanup_expired` | `expires_at`, `created_at`, `actual_path` | `metadata_backend.iter_entry_summaries()` | Yes | VERIFIED |
| `UnifiedCache._cleanup_expired` | `ttl_seconds` | `config.metadata.default_ttl_seconds` into public cleanup | Yes | VERIFIED |
| `JsonBackend.cleanup_expired` | JSON metadata entries | `_metadata["entries"]` | Yes | VERIFIED |
| `SqliteBackend.cleanup_expired` | ORM metadata rows | SQLAlchemy delete filter | Yes | VERIFIED |
| `PostgresBackend.cleanup_expired` | ORM metadata rows | SQLAlchemy count/delete filter | Yes | VERIFIED |
| `UnifiedCache._enforce_size_limit` | `removed_entries.actual_path` | `metadata_backend.cleanup_by_size(target_size_bytes)` | Yes | VERIFIED - focused TTL-04 regression populated real memory URI entries, lowered the size limit from observed metadata stats, then verified old metadata and backend blob removal. |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| TTL-01 through TTL-03 representative regression slice | `uv run --python 3.12 pytest --override-ini testpaths= -o addopts="" tests/test_core.py::TestCacheness::test_is_expired_honors_stored_expires_at_before_fallback_ttl tests/test_core.py::TestCacheness::test_cleanup_expired_honors_stored_expires_at_for_blob_cleanup tests/test_core.py::TestCacheness::test_init_cleanup_expired_deletes_blob_files_and_invokes_hook tests/test_update_operations.py::TestUpdateEntryMetadataBackends::test_metadata_only_update_preserves_provenance_and_ttl_fields tests/test_cache_signing.py::TestDeleteInvalidSignatures::test_signed_entry_survives_metadata_only_update tests/test_storage_mode.py::TestNoTTLExpiration::test_get_ignores_stored_past_expires_at -q --ignore=tests/test_tensorflow_handler.py --basetemp ".pytest-tmp-phase29-final-ttl0103" -p no:cacheprovider -p no:xdist` | 7 passed in 1.46s. | PASS |
| TTL-04 focused three-test slice | `uv run --python 3.12 pytest --override-ini testpaths= -o addopts="" tests/test_core.py::TestCacheness::test_size_limit_eviction_deletes_memory_uri_blob tests/test_core.py::TestCacheness::test_size_limit_remote_delete_failure_logs_warning tests/test_core.py::TestCacheness::test_storage_mode_size_enforcement_remains_disabled -q --ignore=tests/test_tensorflow_handler.py --basetemp ".pytest-tmp-phase29-final-ttl04" -p no:cacheprovider -p no:xdist` | 3 passed in 1.05s. Warning-path test logged the expected backend deletion warning. | PASS |

### Probe Execution

| Probe | Command | Result | Status |
|---|---|---|---|
| Phase probes | `rg -n "probe-[^\\s]+\\.sh|scripts/.*/tests/probe-.*\\.sh|PASS markers|stage markers|probe" .planning/phases/29-ttl-eviction-consistency scripts` | No phase-owned shell probes are declared. | SKIPPED |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|---|---|---|---|---|
| TTL-01 | 29-01-PLAN.md | User can set per-entry TTL and have stored `expires_at` honored by reads and cleanup across JSON, SQLite, and PostgreSQL semantics. | SATISFIED | Source checks plus targeted read, cleanup, backend, and storage-mode regressions verify stored-expiry precedence and fallback behavior. |
| TTL-02 | 29-02-PLAN.md | User can rely on init-time expired-entry cleanup deleting both metadata and blob files through the same path as public cleanup. | SATISFIED | `_cleanup_expired()` delegates to public cleanup; init cleanup blob/hook regression passed. |
| TTL-03 | 29-03-PLAN.md | User can overwrite or update metadata without unexpectedly resetting access counters, provenance timestamps, expiry semantics, or signatures. | SATISFIED | SQLite/PostgreSQL source checks plus JSON/SQLite metadata-only and signing spot-checks verify preservation behavior. |
| TTL-04 | 29-04-PLAN.md | User can rely on size/eviction cleanup deleting remote blobs through the configured blob backend instead of leaking S3 or memory-backed objects. | SATISFIED | `_enforce_size_limit()` URI branch is wired to `blob_backend.delete_blob`; focused memory URI regression now passes and proves deletion of the evicted URI blob. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---|---|---|---|
| None | - | - | - | No unreferenced `TBD`, `FIXME`, or `XXX` markers found in phase-relevant source/test files. Generic empty utility returns in backend helpers are documented non-stub behavior. |

### Human Verification Required

None.

### Gaps Summary

No blocking gaps remain. The previous TTL-04 gap is closed in current commit `093cda0`: the memory URI size-eviction regression now exceeds the limit deterministically, exercises `cleanup_by_size()`, deletes the evicted `memory://` blob through the configured blob backend, and leaves the newer entry intact.

---

_Verified: 2026-06-14T02:02:17Z_
_Verifier: the agent (gsd-verifier)_
