---
phase: 07-explicit-migration-and-rebuild-cutover
reviewed: 2026-09-11T20:28:13Z
depth: standard
files_reviewed: 31
files_reviewed_list:
  - docs/BACKEND_SELECTION.md
  - docs/STORAGE_INITIALIZATION.md
  - docs/STORAGE_MIGRATION.md
  - src/cacheness/error_handling.py
  - src/cacheness/handlers.py
  - src/cacheness/interfaces.py
  - src/cacheness/storage/__init__.py
  - src/cacheness/storage/backends/postgresql_lifecycle_authority.py
  - src/cacheness/storage/backends/s3_backend.py
  - src/cacheness/storage/blob_store.py
  - src/cacheness/storage/lifecycle.py
  - src/cacheness/storage/lifecycle_authority.py
  - src/cacheness/storage/memory_lifecycle_authority.py
  - src/cacheness/storage/migration.py
  - src/cacheness/storage/migration_authority.py
  - src/cacheness/storage/migration_evidence.py
  - src/cacheness/storage/sqlite_lifecycle_authority.py
  - tests/contracts/test_lifecycle_authority.py
  - tests/contracts/test_postgresql_lifecycle_authority.py
  - tests/test_blob_store_atomic_lifecycle.py
  - tests/test_handler_registration.py
  - tests/test_migration_cutover.py
  - tests/test_migration_inspection.py
  - tests/test_migration_plan_contract.py
  - tests/test_migration_public_contract.py
  - tests/test_migration_remote_contract.py
  - tests/test_migration_run_evidence.py
  - tests/test_phase7_contract_verifier.py
  - tests/test_rebuild_workflow.py
  - tests/test_stored_compatibility.py
  - tools/verify_phase7_contracts.py
findings:
  critical: 3
  warning: 2
  info: 0
  total: 5
status: issues_found
---

# Phase 7: Code Review Report

**Reviewed:** 2026-09-11T20:28:13Z
**Depth:** standard
**Files Reviewed:** 31
**Status:** issues_found

## Summary

The reviewed Phase 7 surface still has three shipping blockers: an executable
same-version format migration can be misclassified as an identity copy, normal
migration abort does not persist the S3 participant's typed operational
failures as cleanup debt, and rebuild cleanup debt is placed in a terminal
state with no settlement path. Two public-result/handler-registration defects
also make outcomes misleading or registrations ineffective.

These findings stay inside the approved ADR 0001 boundary. They do not require
cross-resource ACID, listing-based adoption, obstore adoption, or another lock,
queue, lease, journal, coordinator, sidecar, or authority. The accepted
post-publication/pre-checkpoint orphan limit and Phase 8-owned live-platform,
performance, and Python-matrix qualification are not findings.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01 [BLOCKER]: A format change is silently bypassed when its numeric version is unchanged

**File:** `src/cacheness/storage/migration.py:1966-1995`

**Issue:** `_configured_destination_contract()` substitutes the source
`(payload_format, payload_format_version)` whenever the destination handler has
the same numeric version and can read the source. A handler whose current
native contract is `("mcap-v2", 1)` and which declares a directed edge from
`("mcap-v1", 1)` therefore gets a target of `("mcap-v1", 1)`. Inspection and
staging treat the entry as an identity-compatible byte copy, so the directed
handler transformation is never called and the candidate manifest continues
to advertise the old format. This violates D-03/D-10 and Plan 07-16's exact
destination-contract requirement; payload format is an independently relevant
part of the contract even when its version integer happens to match.

**Fix:** Always derive the target identity from the destination handler's
declared current contract. Use `supports_payload_contract()` only to decide
whether the source is readable, never to redefine the destination.

```python
target_identity = (payload_format, payload_version)
destination_versions = StoreVersionDimensions(
    payload_format_version=payload_version,
)
```

Add a migration regression with different source/target format names but the
same version number and assert that the exact directed transform runs and the
candidate manifest contains the destination format.

### CR-02 [BLOCKER]: S3 abort failures escape before cleanup debt is checkpointed

**Files:** `src/cacheness/storage/migration.py:3943-3967`, `src/cacheness/storage/backends/s3_backend.py:658-815`, `src/cacheness/error_handling.py:443`

**Issue:** `OfflineMigrationService.abort()` converts only `OSError` and
`ValueError` into durable cleanup debt. The S3 participant deliberately
normalizes failed `HEAD`, `GET`, delete, and absence-proof operations to
`CacheBlobBackendError`, which derives from `CacheStorageError`, not
`OSError`. An expected remote outage during either `open_snapshot()` or
`delete_migration_payload()` consequently escapes the abort loop without the
required debt checkpoint or typed partial `AbortReceipt`. The authority still
attributes the candidate, so this is fixable without stronger coordination,
but the implemented remote path does not satisfy Plan 07-12's requirement that
bounded deletion failures persist as retryable cleanup debt.

**Fix:** Catch the participant's typed operational error explicitly and record
the existing retirement digest in the current evidence, while continuing to
fail closed for ownership/integrity conflicts.

```python
except (OSError, ValueError, CacheBlobBackendError):
    cleanup_debt.append(retirement)
```

Add a deterministic S3-adapter abort test that injects both snapshot and delete
failures and verifies `STAGING` evidence contains the exact attributed debt and
that a later retry settles it without listing or adoption.

### CR-03 [BLOCKER]: Rebuild cleanup debt is called retryable but is written into an unrecoverable terminal state

**Files:** `src/cacheness/storage/migration.py:2705-2780`, `src/cacheness/storage/migration.py:4219-4243`, `src/cacheness/storage/migration_evidence.py:225-235`, `tests/test_rebuild_workflow.py:577-625`

**Issue:** `_abort_rebuild_after_failure()` records exact receipt-backed debt
and then unconditionally checkpoints `ABORTED` once every receipt is either
retired or merely represented by debt. `ABORTED` has no legal transition, and
the rebuild branch of `resume()` rejects it. No other reviewed path parses or
settles `rebuild:<operation_id>:<key>:<generation>:<locator>` debt. The existing
test labels the debt retryable while asserting the terminal state, but never
proves a retry. As a result, an operationally failed exact delete can leave a
destination entry present forever with evidence that cannot be advanced or
cleared. This contradicts ADR 0001's durable cleanup-debt invariant and Plan
07-15's promised retryable receipt-bound cleanup.

**Fix:** Reuse the existing authenticated rebuild receipts and authority as the
only inputs to an explicit retry path. Keep the run in an existing resumable
rebuild cleanup condition while debt remains (or permit only the narrow
receipt-bound retry from `ABORTED`), remove a debt item only after exact
deletion/proven absence, and enter terminal `ABORTED` only after every item is
settled. Do not add another intent layer or coordination mechanism. Extend the
failure test to restore the participant, retry exact cleanup, and prove both
the payload state and authenticated evidence converge.

## Warnings

### WR-01 [WARNING]: Partial abort receipts report failed candidates as deleted

**File:** `src/cacheness/storage/migration.py:3940-3989`

**Issue:** When any candidate deletion fails, `abort()` correctly returns a
nonterminal `STAGING` receipt but sets `deleted_entries=len(candidates)`.
Candidates represented by `cleanup_debt` were not deleted (and may still be
present), so the public field overstates completed cleanup. Operator automation
can interpret a partial failure as full physical retirement even though the
receipt's state says otherwise.

**Fix:** Track the number of candidates actually deleted or proven absent and
return that count. If callers also need attempted/pending counts, expose them
as separate explicit fields rather than folding them into `deleted_entries`.
Add assertions for the count in the existing partial-abort failure test.

### WR-02 [WARNING]: The documented custom handler name is discarded after duplicate validation

**File:** `src/cacheness/handlers.py:1551-1591`

**Issue:** `register_handler(..., name=...)` checks the supplied name against
existing handlers, but stores only the handler object. Lookup, listing, and
unregistration continue to use `handler.data_type`. Supplying a unique name can
therefore bypass the duplicate-`data_type` check, after which the new handler is
unreachable by that name and type-based payload resolution selects whichever
duplicate appears first. This is particularly hazardous for store-local custom
format/migration handlers because the registration appears successful while
the intended serializer or transformation edge is not the one resolved.

**Fix:** Either remove the unsupported `name` parameter from the public API and
always reject duplicate `data_type`, or persist a registration record containing
the alias and use it consistently in duplicate checks, lookup, listing, and
unregistration. Add tests for alias lookup/removal and for duplicate data types
registered under different aliases.

---

_Reviewed: 2026-09-11T20:28:13Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
