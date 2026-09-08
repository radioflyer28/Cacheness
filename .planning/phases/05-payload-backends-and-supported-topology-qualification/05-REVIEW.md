---
phase: 05-payload-backends-and-supported-topology-qualification
reviewed: 2026-09-08T15:29:45Z
depth: standard
files_reviewed: 32
files_reviewed_list:
  - docs/CATALOG_AND_TOPOLOGY.md
  - docs/STORAGE_INITIALIZATION.md
  - pyproject.toml
  - src/cacheness/storage/__init__.py
  - src/cacheness/storage/backends/__init__.py
  - src/cacheness/storage/backends/blob_backends.py
  - src/cacheness/storage/backends/postgresql_lifecycle_authority.py
  - src/cacheness/storage/blob_store.py
  - src/cacheness/storage/composition.py
  - src/cacheness/storage/lifecycle.py
  - src/cacheness/storage/reconciliation.py
  - tests/contracts/test_lifecycle_authority.py
  - tests/contracts/test_payload_generation_io.py
  - tests/contracts/test_postgresql_lifecycle_authority.py
  - tests/contracts/test_topology_lifecycle.py
  - tests/integration/test_postgresql_authority.py
  - tests/integration/test_remote_topology.py
  - tests/integration/test_s3_generation.py
  - tests/qualification/conftest.py
  - tests/qualification/test_live_evidence.py
  - tests/test_blob_store_composition.py
  - tests/test_catalog_projection.py
  - tests/test_catalog_query_contract.py
  - tests/test_lifecycle_authority_contract.py
  - tests/test_metadata_backend_registry.py
  - tests/test_payload_faults.py
  - tests/test_phase5_contract_verifier.py
  - tests/test_postgresql_backend.py
  - tests/test_sqlite_bootstrap_concurrency.py
  - tests/test_supported_topologies.py
  - tools/run_phase5_qualification.py
  - tools/verify_phase5_contracts.py
findings:
  critical: 6
  warning: 1
  info: 0
  total: 7
status: issues_found
---

# Phase 5: Code Review Report

**Reviewed:** 2026-09-08T15:29:45Z
**Depth:** standard
**Files Reviewed:** 32
**Status:** issues_found

## Summary

Plans 01–09 establish the intended single-authority topology, but six correctness
defects remain in the PostgreSQL initialization/replay path, reconciliation, and
live-evidence boundary. The remote live-service qualification in Plan 10 remains
unavailable; this review does not treat unavailable PostgreSQL/AWS evidence as a
pass. None of the fixes below requires another coordinator, lock, queue, or
authority.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01 [BLOCKER]: The public remote initialization path validates a schema before creating it

**File:** `/Users/akriz/code/cacheness/src/cacheness/storage/blob_store.py:332-344`

**Issue:** `BlobStore.initialize()` calls `preflight_mutation()` before the
authority's explicit `initialize()` method. For PostgreSQL, preflight calls
`open()`, and `open()` performs read-only validation of an already-existing schema
(`postgresql_lifecycle_authority.py:414-420,532-534`). A fresh supported
PostgreSQL/S3 store therefore fails with migration-required before the explicit
initializer can create its current-version schema. Ordinary `put()` has the same
ordering at `lifecycle.py:220-223`: it preflights before delegating to
`store.initialize()`. The live fixture conceals the defect by directly invoking
`authority.initialize()` before constructing and initializing the public store
(`tests/qualification/conftest.py:510-517`). As a result, the advertised public
initialization boundary cannot provision the qualified remote topology.

**Fix:** At the explicit public initialization boundary, invoke the selected
authority's `initialize()` first and then perform validation/preflight. Change
ordinary mutation startup to delegate to `store.initialize()` before any redundant
preflight (or remove that redundant call). Keep SQLite path-security checks inside
its authority initialization path. Add a test that constructs a fresh PostgreSQL
authority and provisions it solely through `BlobStore.initialize()`.

### CR-02 [BLOCKER]: Replaying an older promoted operation returns the current replacement entry

**File:** `/Users/akriz/code/cacheness/src/cacheness/storage/backends/postgresql_lifecycle_authority.py:743-770`

**Issue:** `_promoted_result()` joins a promoted mutation to `entries` by key only.
After operation A promotes a generation and operation B overwrites that key,
replaying A (or classifying A after an ambiguous commit) returns B's current entry
as A's `PromotionResult`. Both the already-promoted branch at lines 799-806 and the
uncertain-commit classifier at lines 772-796 use this query. This corrupts
operation-idempotent result identity and can give a caller a receipt for a
generation its operation did not publish.

**Fix:** Persist A's exact promotion result in the existing PostgreSQL authority
transaction (at least promoted lineage and authority revision; generation,
locator, and manifest are already in the mutation row), then reconstruct replay
results from that operation-owned record. Do not join replay results to the
mutable current `entries` row. This is a schema change, so bump the exact schema
version and preserve fail-closed offline migration/rebuild handling. Add a replay
test that promotes A, overwrites with B, and then replays/classifies A.

### CR-03 [BLOCKER]: Reconciliation advances past prepared mutations omitted by its byte bound

**File:** `/Users/akriz/code/cacheness/src/cacheness/storage/backends/postgresql_lifecycle_authority.py:1557-1620`

**Issue:** The adapter fetches a page of prepared mutations, stops appending work
when the cumulative manifest-byte bound is reached at lines 1586-1587, but returns
`mutation_rows[-1]` as the next cursor at line 1618. The reconciler advances to
that cursor after processing the returned work (`reconciliation.py:252-254`). Any
fetched rows after the byte break are therefore skipped permanently, and a resume
token can eventually report completion while durable prepared-mutation evidence
was never inspected or reconciled.

**Fix:** Return the row ID of the last mutation actually emitted, not the final row
fetched. Apply the same rule to every independently truncated source cursor. Add a
regression with multiple individually valid manifests whose cumulative size
exceeds the page byte bound, and assert every row is returned across resumed pages.

### CR-04 [BLOCKER]: S3 inventory labels every normal committed generation as unattributed residue

**File:** `/Users/akriz/code/cacheness/src/cacheness/storage/reconciliation.py:198-259,320-367`

**Issue:** `attributed_locators` contains only locators seen in the current page of
prepared mutations or cleanup debt. `_inventory_findings()` then reports every
other S3 object as `unattributed_payload_inventory`. Committed entry locators are
never added, and attribution is not carried across resumed inventory pages. Thus a
healthy remote store's ordinary committed generations are reported as blocked
residue. The live test explicitly ignores report contents and checks only that
reconciliation did not revoke reads (`tests/integration/test_remote_topology.py:134-138`),
so it does not detect the false classification.

**Fix:** Report an inventory object as unattributed only after bounded,
revision-consistent evidence from the existing PostgreSQL authority proves that
no committed entry, prepared mutation, or cleanup debt owns the locator. If that
proof cannot be obtained within the work budget, report an indeterminate bounded
finding rather than orphan residue. Keep PostgreSQL as the sole authority and S3
inventory as evidence only. Add a test with both a committed generation and a
genuinely unknown object, including continuation across inventory pages.

### CR-05 [BLOCKER]: Live qualification cleanup can delete its authorization marker before cleanup completes

**File:** `/Users/akriz/code/cacheness/tests/qualification/conftest.py:206-269`

**Issue:** `_bounded_delete_prefix()` includes the ownership marker in the normal
bulk deletion list. If a later list/delete page fails or a bound is reached after
the marker's page was deleted, `cleanup_s3_run()` returns `RESIDUE`. A retry then
finds residue without the marker and correctly refuses to touch it at lines
342-352, making the test-owned prefix permanently uncleanable through the guarded
cleanup API. This creates a real external-resource leak in the live qualification
harness and breaks its claimed bounded, idempotent cleanup behavior.

**Fix:** Exclude the exact ownership marker from paginated bulk deletion and delete
it last, only after all other objects and multipart uploads are proven absent. Add
a deterministic multi-page test that fails after the first deletion batch and
proves a second cleanup attempt can safely finish.

### CR-06 [BLOCKER]: Evidence validation accepts contradictory claims as QUALIFIED

**File:** `/Users/akriz/code/cacheness/tools/run_phase5_qualification.py:247-304`

**Issue:** `validate_evidence()` validates each enum and field shape independently
but never validates their relationship. A hand-edited or malformed artifact can
therefore claim `status="QUALIFIED"` while also declaring `result="failed"`,
`cleanup_status="RESIDUE"`, nonempty `missing_configuration`, or no AWS identity.
The read-only verifier then returns that status as valid at
`tools/verify_phase5_contracts.py:383-394`. This defeats the release-truthfulness
boundary and can turn unavailable or failed live-service evidence into an apparent
BACK-05 pass even though the normal runner happens to emit consistent fields.

**Fix:** Enforce cross-field invariants in `validate_evidence()`: `QUALIFIED` must
require `result="passed"`, `cleanup_status="CLEAN"`, no missing configuration, and
the expected real Amazon S3 identity; `UNAVAILABLE` must require `result="not_run"`
and `cleanup_status="NOT_ATTEMPTED"`; all other failed/residue combinations must be
`NOT_QUALIFIED`. Add forged-artifact tests for every contradictory combination and
assert `read_live_evidence_status()` returns `invalid`.

## Warnings

### WR-01 [WARNING]: The public metadata-role helper contradicts the qualified PostgreSQL role

**File:** `/Users/akriz/code/cacheness/src/cacheness/storage/composition.py:836-849`

**Issue:** `resolve_metadata_role("postgresql")` still returns `projection`, while
the Phase 5 registry and topology catalog define PostgreSQL as the canonical
lifecycle authority and explicitly register no PostgreSQL projection. The test
suite currently codifies both contradictory claims:
`tests/test_postgresql_backend.py:15-21` expects projection, while
`tests/test_metadata_backend_registry.py:66-75` expects authority and no
projection. Callers of the exported helper receive a role classification that is
false for the supported topology.

**Fix:** Remove the stale Phase 4 helper/export under the intentional
pre-production compatibility reset, or make it delegate to the role registry and
return PostgreSQL's authority role. Reserve a distinct name for any future
PostgreSQL-derived projection. Update the contradictory test accordingly.

---

_Reviewed: 2026-09-08T15:29:45Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
