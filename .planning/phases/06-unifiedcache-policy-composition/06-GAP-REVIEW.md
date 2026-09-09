---
phase: 06-unifiedcache-policy-composition
reviewed: 2026-09-09T08:28:22Z
depth: standard
files_reviewed: 15
files_reviewed_list:
  - tests/test_blob_manifest.py
  - tests/test_filesystem_containment.py
  - tests/test_cache_signing.py
  - tests/test_legacy_array_security.py
  - tests/test_public_api_contract.py
  - tests/test_query_meta.py
  - tests/test_store_cache_key_params_config.py
  - tests/test_query_meta_security.py
  - tests/test_phase1_quality_gates.py
  - tools/run_phase6_local_suite.py
  - tests/test_phase6_suite_isolation.py
  - tests/test_full_suite_environment.py
  - tests/test_phase3_gap_acceptance.py
  - tools/verify_phase6_contracts.py
  - tests/test_phase6_contract_verifier.py
findings:
  critical: 5
  warning: 0
  info: 0
  total: 5
status: issues_found
---

# Phase 6: Gap-Change Code Review Report

**Reviewed:** 2026-09-09T08:28:22Z
**Depth:** standard
**Files Reviewed:** 15
**Status:** issues_found

## Summary

The migrated catalog, containment, array-security, public-surface, parameter
round-trip, and retained Phase 3 acceptance tests preserve substantive current
behavior. The canonical all-extras focused scope passed, the complete Phase 6
non-live runner exited zero, and scoped Ruff passed for all fifteen reviewed
Python paths. The runner currently selects the repository root and excludes
only the three regular Phase 8 live modules. Its output correctly leaves live
PostgreSQL/Amazon S3 and native Windows UNAVAILABLE/NOT_QUALIFIED; this review
does not upgrade those Phase 8 qualifications.

The requested `tests/test_public_api.py` path does not exist. The Plan 09-11
inventory, implementation commit, and verifier all identify
`tests/test_public_api_contract.py`; that canonical file was reviewed in its
place. `06-VALIDATION.md` was reviewed as evidence context but is excluded from
the source-file count because it is a planning artifact.

Five blockers remain. The signing migration is behaviorally vacuous for the
configuration it claims to cover, the ordered probes never exercise the
reverse Phase 6/SqlCache direction, the local runner can translate a live-path
symlink into an unrelated in-repository exclusion, and the fixed verifier has
two independent false-green paths. None of the fixes below requires or
recommends a lifecycle lock, queue, coordinator, readiness mechanism,
compatibility restoration, cross-resource ACID claim, or timing guarantee.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: The migrated signing suite no longer exercises signing behavior

**Classification:** BLOCKER

**File:** `tests/test_cache_signing.py:24-40,84-144`

**Issue:** `_signed_cache()` varies `enable_entry_signing`,
`delete_invalid_signatures`, and `use_in_memory_key`, but the tests only perform
an ordinary dictionary put/lookup and inspect the dataclass values. No current
`UnifiedCache` or `BlobStore` path consumes `delete_invalid_signatures` or
`use_in_memory_key`, and dictionary storage does not use the trusted-object
array gate that still reads `enable_entry_signing`. Consequently both
`delete_invalid_signatures` variants and the “in-memory key” case pass even if
all legacy entry-signer behavior is absent. The migration also deleted the
former persistent-key-file and second-instance/in-memory-key assertions, so the
claimed retention of substantive signing coverage is false.

**Fix:** Replace the vacuous configuration variants with current canonical
manifest-authenticity tests that mutate a committed signed descriptor and
assert the typed, non-destructive integrity outcome. Remove runtime-inert
legacy signing options from the pre-production configuration/public tests (or,
only if they are intentionally current policy, connect them to their actual
owning boundary and restore behavioral persistence/reopen assertions). Do not
restore a compatibility signer alongside BlobStore manifest authority.

### CR-02: The order-isolation matrix never runs SqlCache before the Phase 6 public contract

**Classification:** BLOCKER

**File:** `tests/test_phase6_suite_isolation.py:110-138`

**Issue:** The two orders are
`public_api_contract -> phase6_public_api_contract -> sql_cache` and
`phase6_public_api_contract -> sql_cache -> public_api_contract`. In both,
`test_phase6_public_api_contract.py` precedes `test_sql_cache.py`. The second
tuple rotates the first item instead of reversing the Phase 6 public/SqlCache
pair. State leaked by SqlCache into the Phase 6 public tests can therefore go
undetected even while the test, verifier, and validation artifact report
bidirectional order isolation.

**Fix:** Make the second tuple a genuine reverse, for example
`test_sql_cache.py -> test_phase6_public_api_contract.py ->
test_public_api_contract.py`, and assert in the test that each relevant pair
appears in both relative orders before spawning pytest.

### CR-03: A symlinked live module can turn the exact-three runner into an unrelated exclusion

**Classification:** BLOCKER

**File:** `tools/run_phase6_local_suite.py:83-108`

**Issue:** Validation resolves each configured live path and accepts any
resolved regular file below the repository root. `build_pytest_argv()` then
computes `--ignore` from that resolved target. If a named live module is
replaced by a symlink to another in-repository test, the runner accepts it and
ignores the unrelated target rather than the literal live-module path, while
lines 116-120 still print the original three live names. This defeats the exact
selection boundary and can produce false local evidence without escaping the
repository root. The isolation tests cover missing paths and broad ignores but
not this substitution.

**Fix:** Reject symlinks/non-regular lexical entries with an `lstat`-style
check and build each `--ignore` from the validated literal relative path, not
from a resolved target. Add a test whose live-module path symlinks to an
unrelated in-repository test and require validation to fail before pytest.

### CR-04: Canonical inventory completeness is checked against an oracle derived from itself

**Classification:** BLOCKER

**File:** `tools/verify_phase6_contracts.py:69-83,542-554`

**Issue:** `_EXPECTED_CANONICAL_CUTOVER_NODES` is initialized as
`frozenset(CANONICAL_CUTOVER_NODES)`. The self-test mutates the tuple only after
module import, so it leaves the derived expected set intact and appears to
prove omission detection. A real source edit that deletes a node from the
literal tuple initializes both values from the shortened inventory; manifest
validation, AST auditing, and pytest execution all shrink with it. This was
reproduced by executing an in-memory copy of the module with
`tests/test_phase3_gap_acceptance.py` removed: it loaded 11 nodes and
`_validate_cutover_inventory()` returned `()`.

**Fix:** Define the expected Plan 09-11 inventory independently (for example as
a separate immutable literal or plan-owned manifest) and derive the execution
tuple from that single fixed oracle, not the reverse. Change the mutation test
to load modified source or independently patch the expected/actual inputs so
it models module initialization after a source omission.

### CR-05: The compatibility AST audit misses aliases of a known UnifiedCache

**Classification:** BLOCKER

**File:** `tools/verify_phase6_contracts.py:269-329`

**Issue:** `_CutoverVisitor` recognizes only the hard-coded names `cache` and
`unified_cache`, plus variables assigned directly from a `UnifiedCache(...)`
call. It does not propagate aliases or recognize cache-returning helpers. The
positive mutation self-test uses the privileged name `cache`, masking the gap.
The following prohibited migrated source was reproduced as a clean audit:

```python
cache = UnifiedCache(config, store=store)
alias = cache
alias.get("entry")
```

`audit_cutover_source(...)` returned `()`. Thus a removed cache surface can be
reintroduced in the fixed inventory while the verifier prints green.

**Fix:** Propagate assignments from known cache values (including annotated or
known cache-returning fixtures/helpers), or enforce a source shape that rejects
ambiguous retired-surface calls unless the receiver is proven to be a current
non-cache type. Add alias and helper-return mutation tests and require both to
produce a canonical-cutover diagnostic.

---

_Reviewed: 2026-09-09T08:28:22Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
