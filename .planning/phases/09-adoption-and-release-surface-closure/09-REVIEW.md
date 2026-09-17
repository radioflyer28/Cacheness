---
phase: 09-adoption-and-release-surface-closure
reviewed: 2026-09-17T04:50:06Z
depth: standard
files_reviewed: 45
files_reviewed_list:
  - .codex/skills/spike-findings-cacheness/SKILL.md
  - .codex/skills/spike-findings-cacheness/references/handler-integration.md
  - .github/workflows/quality.yml
  - AGENTS.md
  - README.md
  - docs/API_REFERENCE.md
  - docs/BLOB_STORE.md
  - docs/CACHE_POLICY.md
  - docs/CATALOG_AND_TOPOLOGY.md
  - docs/PLUGIN_DEVELOPMENT.md
  - docs/README.md
  - docs/RELEASE_QUALIFICATION.md
  - docs/SECURITY.md
  - docs/STORAGE_INITIALIZATION.md
  - docs/STORAGE_MIGRATION.md
  - examples/README.md
  - examples/custom_mcap_format.py
  - examples/durable_catalog_store.py
  - examples/memory_blob_store.py
  - examples/unified_cache.py
  - pyproject.toml
  - src/cacheness/error_handling.py
  - src/cacheness/handlers.py
  - src/cacheness/interfaces.py
  - src/cacheness/storage/__init__.py
  - src/cacheness/storage/handlers/__init__.py
  - tests/packaging/test_wheel_matrix.py
  - tests/test_error_handling.py
  - tests/test_handler_registration.py
  - tests/test_handlers.py
  - tests/test_interfaces.py
  - tests/test_migration_public_contract.py
  - tests/test_phase5_contract_verifier.py
  - tests/test_phase7_contract_verifier.py
  - tests/test_phase9_documentation.py
  - tests/test_phase9_evidence_metadata.py
  - tests/test_phase9_examples.py
  - tests/test_phase9_quality_workflow.py
  - tests/test_public_api_contract.py
  - tests/test_security_documentation.py
  - tests/test_stored_compatibility.py
  - tools/run_phase8_packaging.py
  - tools/verify_phase071_contracts.py
  - tools/verify_phase5_contracts.py
  - tools/verify_phase7_contracts.py
findings:
  critical: 1
  warning: 6
  info: 0
  total: 7
status: issues_found
---

# Phase 09: Code Review Report

**Reviewed:** 2026-09-17T04:50:06Z
**Depth:** standard
**Files Reviewed:** 45
**Status:** issues_found

## Summary

The alias-free `FormatHandler` cutover and the four executable examples pass the
targeted Phase 9 suite, but the submitted surface is not ready to close. The
README still duplicates detailed qualification facts in direct conflict with
locked decision D-04, and an inherited verifier now enforces that duplication.
The review also found an incorrect documented compare-and-swap outcome, two
incompatible `FormatHandlerError` base classes, dangling links created by the
documentation deletion, weakened documentation-verifier assertions, an
unbounded subprocess harness, and a materially stale agent architecture map.

No finding asks for cross-resource ACID, another lock, or another lifecycle
coordinator. The fixes are documentation, exception-surface, and verification
repairs within ADR 0001's existing topology-specific boundary.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: README duplicates detailed qualification contracts despite D-04's single owner

**Classification:** BLOCKER

**File:** `/Users/akriz/code/cacheness/README.md:112-115`

**Issue:** Locked decision D-04 says the README is a concise gateway and one
linked guarantees page owns detailed topology, payload-bound, and evidence
claims. The README nevertheless repeats the exact 128 MiB limit, transport-
evidence semantics, and the `actual_path` projection detail. Worse,
`tools/verify_phase071_contracts.py:665-678` says details are no longer required
on every page while still requiring `128 MiB`, `opaque transport evidence`, and
`Phase 8` in the README. That makes the regression gate force the specification
violation and invites future edits to duplicate the qualification contract
again.

**Fix:** Remove the detailed paragraph from the README and leave its existing
link to `docs/RELEASE_QUALIFICATION.md`. Remove the README-specific detail terms
from `audit_documentation()` and keep those assertions only against the release
qualification owner. Add a Phase 9 test that rejects the detailed payload and
transport phrases outside their designated reference owner.

## Warnings

### WR-01: Two incompatible classes now claim to be the canonical `FormatHandlerError`

**Classification:** WARNING

**File:** `/Users/akriz/code/cacheness/src/cacheness/error_handling.py:111-114`

**Issue:** `error_handling.FormatHandlerError` inherits `CacheError` and accepts a
context mapping, while `interfaces.FormatHandlerError` at
`src/cacheness/interfaces.py:340-356` inherits plain `Exception` and accepts
`handler_type`/`data_type`. `CacheWriteError`, `CacheReadError`, and the storage
barrel use the latter, so catching the equally named error from
`cacheness.error_handling` does not catch actual handler failures. The updated
tests exercise the two hierarchies independently and therefore conceal the
identity split introduced into the renamed public vocabulary.

**Fix:** Define one canonical `FormatHandlerError` and make the focused handler
exceptions inherit it. If handler-specific fields are retained, extend the
canonical constructor explicitly. Add an identity/inheritance test spanning
`cacheness.error_handling`, `cacheness.interfaces`, and `cacheness.storage`.

### WR-02: The BlobStore guide documents the wrong stale-update outcome

**Classification:** WARNING

**File:** `/Users/akriz/code/cacheness/docs/BLOB_STORE.md:73-76`

**Issue:** The guide says a stale `BlobReceipt.expectation` makes
`update_catalog()` return `None`. The actual authority contract raises
`CacheBlobLifecycleConflictError`, as exercised in
`tests/test_catalog_query_contract.py:543-549` and correctly stated in
`docs/CATALOG_AND_TOPOLOGY.md:113-117`. An adopter following this guide will not
handle the typed contention outcome and may incorrectly treat the exception as
an unexpected storage failure.

**Fix:** Document the typed lifecycle-conflict exception for a stale expectation
and reserve `None` for the genuinely absent-entry result. Add a behavioral
documentation contract that binds this sentence to the existing stale-
expectation test.

### WR-03: Deleting the old guides left live documentation links dangling

**Classification:** WARNING

**File:** `/Users/akriz/code/cacheness/tests/test_phase9_documentation.py:189-212`

**Issue:** The cleanup test proves that `CONFIGURATION.md` and
`BACKEND_SELECTION.md` are absent and unlinked from the new index, but it never
checks the rest of the documentation tree. The deletion therefore leaves
broken links in `docs/PERFORMANCE.md:172`, `docs/WINDOWS_COMPATIBILITY.md:287`,
`docs/TENSORFLOW_TENSOR_GUIDE.md:650`, `docs/CROSS_PLATFORM_GUIDE.md:444`,
`docs/DILL_INTEGRATION.md:950`, and `docs/PANDAS_COMPATIBILITY.md:264`. Users who
reach those still-present guides hit missing pages.

**Fix:** Repair each link to the current task/reference owner or remove the stale
guide if its primary purpose is a retired API. Extend the Phase 9 test to parse
all repository Markdown links and require every relative target to exist (with
an explicit allowlist only for intentional external/generated targets).

### WR-04: The relocated Phase 5 documentation gate is no longer fail-closed

**Classification:** WARNING

**File:** `/Users/akriz/code/cacheness/tools/verify_phase5_contracts.py:194-229`

**Issue:** The former marker-bounded contract was replaced by whole-document
substring checks such as merely `read-only` and `stopped-worker`. Unrelated prose
can satisfy those tokens after the actual migration rule disappears. The code
also says mutable evidence status is permitted only in the qualification owner,
but checks only the catalog and coverage files; `STORAGE_INITIALIZATION.md` and
`STORAGE_MIGRATION.md` could acquire `QUALIFIED` or `NOT_QUALIFIED` claims without
failing. `tests/test_phase5_contract_verifier.py:179-200` repeats both gaps.

**Fix:** Give each relocated contract a unique marker-bounded section or assert
the complete semantic phrases, and scan every non-owner document read by the
verifier for mutable evidence-status vocabulary. Add mutation tests that remove
the actual requirement while leaving generic words elsewhere and prove the
verifier fails.

### WR-05: The exact-example harness has no per-process timeout

**Classification:** WARNING

**File:** `/Users/akriz/code/cacheness/tests/test_phase9_examples.py:63-70`

**Issue:** Each example is launched with `subprocess.run()` without a timeout.
A deadlock, blocked import, or accidental wait can consume the entire 45-minute
workflow job and prevent the remaining published examples from being checked.
That contradicts Phase 9's requirement for a bounded example harness even
though the current examples finish normally.

**Fix:** Pass a short explicit timeout appropriate for these local examples and
convert `subprocess.TimeoutExpired` into an assertion that names the example and
captured output. Add a unit test with a deliberately blocking child to freeze
the bound.

### WR-06: AGENTS.md still describes the retired pre-refactor lifecycle

**Classification:** WARNING

**File:** `/Users/akriz/code/cacheness/AGENTS.md:224-241`

**Issue:** The agent architecture map says the primary `UnifiedCache` path writes
handler files directly, that `BlobStore` repeats a separate lifecycle, and that
no high-level coordinator injects blob backends. Current `UnifiedCache` instead
constructs or accepts a `BlobStore` and commits through
`BlobStore.put_entry()`. The same stale block later says `UnifiedCache` does not
instantiate `BlobStore` (`AGENTS.md:265-269`). This is precisely the obsolete
split-lifecycle model that ADR 0001 and the recent refactor removed; leaving it
as agent guidance can steer future work back toward parallel coordination
surfaces.

**Fix:** Regenerate or manually refresh the architecture block from the current
`UnifiedCache`/`BlobStore` composition. State that `BlobStore` is the one storage
lifecycle engine, `UnifiedCache` is policy over it, and obstore participants are
selected through `StoreTopology`; preserve the topology-specific nonclaims from
ADR 0001.

---

_Reviewed: 2026-09-17T04:50:06Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
