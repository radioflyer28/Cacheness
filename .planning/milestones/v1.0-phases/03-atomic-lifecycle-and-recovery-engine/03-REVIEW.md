---
phase: 03-atomic-lifecycle-and-recovery-engine
reviewed: 2026-09-06T17:38:58Z
depth: standard
files_reviewed: 64
files_reviewed_list:
  - benchmarks/lifecycle_authority_baseline.json
  - benchmarks/lifecycle_authority_benchmark.py
  - pyproject.toml
  - src/cacheness/__init__.py
  - src/cacheness/config.py
  - src/cacheness/core.py
  - src/cacheness/error_handling.py
  - src/cacheness/metadata.py
  - src/cacheness/serialization.py
  - src/cacheness/storage/__init__.py
  - src/cacheness/storage/backends/postgresql_backend.py
  - src/cacheness/storage/blob_store.py
  - src/cacheness/storage/coordination.py
  - src/cacheness/storage/guarded_handler_io.py
  - src/cacheness/storage/integrity.py
  - src/cacheness/storage/lifecycle.py
  - src/cacheness/storage/lifecycle_authority.py
  - src/cacheness/storage/manifest_repository.py
  - src/cacheness/storage/memory_lifecycle_authority.py
  - src/cacheness/storage/path_security.py
  - src/cacheness/storage/reconciliation.py
  - src/cacheness/storage/sqlite_lifecycle_authority.py
  - tests/_lifecycle_test_support.py
  - tests/fixtures/phase3_ruff_baseline.json
  - tests/test_blob_manifest.py
  - tests/test_blob_manifest_backends.py
  - tests/test_blob_store_atomic_lifecycle.py
  - tests/test_blob_store_close_contract.py
  - tests/test_blob_store_concurrency.py
  - tests/test_blob_store_integrity.py
  - tests/test_blob_store_read_contract.py
  - tests/test_blob_store_reconciliation.py
  - tests/test_cache_integrity.py
  - tests/test_cached_custom_metadata.py
  - tests/test_cached_query_meta.py
  - tests/test_clear_recovery.py
  - tests/test_config_validation.py
  - tests/test_filesystem_containment.py
  - tests/test_full_suite_environment.py
  - tests/test_lifecycle_authority_contract.py
  - tests/test_manifest_repository_cas.py
  - tests/test_phase1_quality_gates.py
  - tests/test_phase3_gap_acceptance.py
  - tests/test_phase3_postreview_concurrency.py
  - tests/test_phase3_release_evidence.py
  - tests/test_phase3_ruff_delta.py
  - tests/test_phase3_scheduler_retirement.py
  - tests/test_phase3_windows_contract.py
  - tests/test_phase3_windows_qualification_attestation.py
  - tests/test_projection_mutation_contract.py
  - tests/test_projection_sql_atomicity.py
  - tests/test_public_api_contract.py
  - tests/test_serialization.py
  - tests/test_sqlite_authority_admission.py
  - tests/test_sqlite_bootstrap_concurrency.py
  - tests/test_sqlite_concurrency.py
  - tests/test_sqlite_concurrency_temp.py
  - tests/test_sqlite_lifecycle_authority.py
  - tests/test_sqlite_metadata_bootstrap_atomicity.py
  - tests/test_unified_cache_adversarial_lifecycle.py
  - tests/test_unified_cache_lifecycle_authority.py
  - tools/capture_phase3_windows_qualification.py
  - tools/verify_phase3_ruff_delta.py
  - verify_platform.py
findings:
  critical: 4
  warning: 1
  info: 0
  total: 5
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-09-06T17:38:58Z
**Depth:** standard
**Files Reviewed:** 64
**Status:** issues_found

## Summary

The ADR-driven Plan 03-20 changes correctly separate SQLite safety and recovery
from progress and benchmark policy. The 5-second default is caller-owned runtime
policy, the historical 0.187-second observation is not used as a correctness
deadline, and the revised concurrency tests accept the documented typed,
retryable timeout without weakening integrity assertions. This review does not
treat the absence of cross-resource ACID or universal contender success as a
defect.

The phase still has four shipping blockers outside that deliberate guarantee
boundary. The in-memory authority's reconciliation cursors address mutable
collections, causing two independently reproduced public reconciliation
crashes. The SQLite metadata backend presents malformed committed query
parameters as valid metadata with the field silently removed. Finally, a
concurrent reader can misclassify the first writer's visible-but-uninitialized
SQLite authority leaf as an incompatible store. SQLite authority error
classification also conflates operational failures with migration-required
evidence.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: Aborting an in-memory mutation leaves a dangling reconciliation row

**Classification:** BLOCKER

**File:** `src/cacheness/storage/memory_lifecycle_authority.py:208-266`

**Issue:** `prepare_mutation()` appends every operation ID to
`_mutation_order`, but `abort_mutation(..., candidate_persisted=False)` deletes
the corresponding `_mutations` entry without removing or terminally retaining
the ordered record. `reconciliation_snapshot()` still publishes the larger
high-water mark, and `page_reconciliation_work()` later indexes
`self._mutations[operation_id]`. A prepare followed by the normal
pre-publication abort deterministically makes `BlobStore.reconcile()` fail with
`KeyError`. This is a correctness failure in the declared same-process memory
topology, not an accepted contention timeout.

**Fix:** Keep the reconciliation sequence append-only for the life of a
snapshot. Mark a non-persisted abort with a terminal state (or store immutable
sequence records separately) and have paging skip terminal records while still
advancing through their stable IDs. Add a regression that aborts before payload
publication and then requires both dry-run and apply reconciliation to finish
without findings or raw exceptions.

### CR-02: Retiring in-memory cleanup debt invalidates signed resume cursors

**Classification:** BLOCKER

**File:** `src/cacheness/storage/memory_lifecycle_authority.py:413-443`

**Issue:** A reconciliation snapshot records `debt_high_water` as the current
list length and treats one-based list positions as durable row IDs. Applying a
page removes debt from `_debts`; resuming then indexes the shortened list using
the old cursor. With three pending debts and `max_reconcile_actions=1`, the
first public `reconcile(apply=True)` returns a resume token, and the second call
deterministically raises `IndexError`. Other cardinalities can skip a debt
instead. The authenticated token cannot make a mutable list position stable.

**Fix:** Assign cleanup debts monotonic immutable IDs in the memory authority,
retain terminal state until every active snapshot has advanced past it, and
page by `id > cursor AND id <= high_water` as the SQLite authority does. Add a
multi-page apply test that uses an action budget smaller than the debt count and
proves every debt is retired exactly once across resumes.

### CR-03: SQLite point reads silently erase corrupt committed key parameters

**Classification:** BLOCKER

**File:** `src/cacheness/metadata.py:2306-2311`

**Issue:** `SqliteBackend.get_entry()` catches decoding failures for a non-null
`cache_key_params` value and returns an otherwise plausible entry with the
field omitted. The same permissive substitution remains in `_list_entries()`
at lines 2948-2955. A live row changed to the invalid JSON text `"{"` was
returned successfully by `get_entry()` with only `actual_path` metadata. This
contradicts the ADR's fail-closed corrupt-metadata invariant and is inconsistent
with `query_entries_by_key_params()`, which now strictly validates the same
persisted field. It can hide integrity loss from callers, signatures that do
not cover this optional field, and direct metadata-backend consumers.

**Fix:** Route every non-null value through one strict bounded decoder, require
a mapping result, and raise a key-attributed `CacheIntegrityError` with
`METADATA_CORRUPT` rather than omitting the field. Reuse
`_decode_cache_key_params()` for listing and an equivalent ORM-value wrapper
for point reads. Add invalid UTF-8/JSON, duplicate-key, scalar, depth, and size
regressions for `get_entry()` and `list_entries()`.

### CR-04: Readers misclassify the first writer's transient SQLite leaf as incompatible

**Classification:** BLOCKER

**File:** `src/cacheness/storage/sqlite_lifecycle_authority.py:541-554`

**Issue:** The first mutator publishes an empty regular database leaf with
`O_CREAT | O_EXCL` and closes it before opening SQLite and committing the
schema/application identity. During that interval, `_classify_for_open()` calls
the object an `authority`; a concurrent read therefore opens it and immediately
validates `PRAGMA application_id`. The still-zero value raises
`CacheBlobMigrationRequiredError("Lifecycle authority application ID is
incompatible")`. A clean detached full-suite run observed exactly this failure
in `test_query_meta_concurrent_access`, while 10 immediate focused reruns passed,
which is consistent with the narrow first-use window. This is neither a corrupt
store nor the ADR-approved typed BUSY/LOCKED timeout: it is a transient bootstrap
state incorrectly presented as durable incompatible evidence.

**Fix:** Make read-side bootstrap classification understand the exact pristine
SQLite state without introducing another authority or process-local correctness
gate. For example, validate under SQLite's own bounded coordination and treat
an exact empty schema/application-ID-zero leaf in the recognized pristine
namespace as initialization in progress (bounded retry or absence), while all
other identity/schema mismatches remain fail-closed migration errors. Add a
deterministic process test that pauses the first mutator after leaf creation,
runs reads from independent authority instances, and requires only absence or a
contextual typed retryable outcome until the initialized authority becomes
visible.

## Warnings

### WR-01: SQLite authority maps operational database failures to migration-required

**Classification:** WARNING

**File:** `src/cacheness/storage/sqlite_lifecycle_authority.py:564-592`

**Issue:** After handling BUSY/LOCKED, `_translate_sqlite_error()` maps every
remaining `sqlite3.DatabaseError` to `CacheBlobMigrationRequiredError`.
`sqlite3.OperationalError` is a `DatabaseError`, so valid stores suffering
`SQLITE_IOERR`, `SQLITE_FULL`, `SQLITE_READONLY`, `SQLITE_CANTOPEN`, or an
interrupted operation are mislabeled as incompatible data requiring migration.
The failure remains closed, but the recovery instruction and typed outcome are
wrong and can prompt an unnecessary rebuild of valid state.

**Fix:** Classify by SQLite primary result code. Reserve migration/integrity
errors for `SQLITE_CORRUPT`, `SQLITE_NOTADB`, incompatible schema/application
identity, and explicitly validated malformed rows. Translate I/O, capacity,
permissions, and open failures to a contextual `CacheBlobBackendError` while
preserving the original SQLite cause; retain BUSY/LOCKED as the retryable
lifecycle timeout.

---

_Reviewed: 2026-09-06T17:38:58Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
