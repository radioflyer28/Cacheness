---
phase: 06-unifiedcache-policy-composition
verified: 2026-09-09T06:34:02Z
status: gaps_found
score: 7/8 must-haves verified
behavior_unverified: 0
overrides_applied: 0
decision_coverage:
  honored: 17
  total: 17
  not_honored: []
gaps:
  - truth: "The Phase 6 pre-production API/configuration cutover is complete across the current runnable repository test corpus, and the fully provisioned local suite is green outside the three explicit Phase 8 live-qualification files."
    status: failed
    reason: "The fixed Phase 6 verifier is green, but the broader fully provisioned local suite fails at runtime. Static AST inspection finds 37 removed flat CacheConfig keyword uses in eight test files (one is the deliberate negative assertion in test_phase6_public_api_contract.py) and 20 UnifiedCache constructions without store= in six files (one is a deliberate negative assertion). First observed local failures include tests/test_blob_manifest.py and tests/test_cache_signing.py. These are stale test consumers of the intentional cutover, not evidence that compatibility APIs should be restored."
    artifacts:
      - path: "tests/test_blob_manifest.py"
        issue: "Three handler-format tests still pass cache_dir directly to CacheConfig."
      - path: "tests/test_cache_signing.py"
        issue: "Fixtures and assertions still require flat security/cache_dir options, implicit UnifiedCache construction, raw put keys, and get(); one test explicitly demands the removed backward-compatibility parameter."
      - path: "tests/test_query_meta.py"
        issue: "The module still constructs the retired metadata-selected UnifiedCache/query_meta surface instead of testing canonical BlobStore catalog queries or being retired as compatibility-only evidence."
      - path: "tests/test_query_meta_security.py"
        issue: "The cache fixture still uses flat metadata selection and an implicit cache store."
      - path: "tests/test_store_cache_key_params_config.py"
        issue: "Most tests still use flat CacheConfig metadata/cache_dir fields and implicit UnifiedCache construction."
      - path: "tests/test_filesystem_containment.py"
        issue: "Eight handler-I/O tests still pass flat cache_dir configuration."
      - path: "tests/test_legacy_array_security.py"
        issue: "One UnifiedCache fixture still uses flat backend configuration and omits explicit store composition."
    missing:
      - "Migrate retained behavioral tests to CacheStorageConfig/CacheMetadataConfig/SecurityConfig, explicit StoreTopology or caller-owned BlobStore, and the typed CachePutResult/CacheLookupResult surfaces."
      - "Delete or archive compatibility-only assertions for flat fields, implicit construction, raw get/put results, and retired query_meta rather than implementing wrappers or shims. Preserve equivalent security, format, containment, and catalog behavior coverage through canonical boundaries."
  - truth: "The fixed Phase 6/SqlCache regressions remain green when run in the complete local suite, with Phase 8 live tests selected out cleanly."
    status: partial
    reason: "The fixed verifier independently passes CACH-01..CACH-07, strict projection rejection, and retained Phase 3-5 lifecycle contracts, while the complete-suite run later reports failures in Phase 6 public API and SqlCache tests that pass in the fixed process. The run also exposed live remote tests without their qualification fixture. This is suite-order/global-state and test-selection debt; it is not a reason to merge SqlCache into UnifiedCache or weaken the live-evidence boundary."
    artifacts:
      - path: "tests/test_phase6_public_api_contract.py"
        issue: "Passes in the fixed verifier but fails in whole-suite execution, indicating leaked import/optional-capability state or ordering dependence."
      - path: "tests/test_sql_cache.py"
        issue: "Passes as the fixed CACH-07 regression but fails in whole-suite execution, so representative isolation is not preserved."
      - path: "tests/integration/test_remote_topology.py"
        issue: "Collected without live_qualification_resources in the local run; the live boundary is not selected consistently."
    missing:
      - "Find and reset test-owned module/import, optional-dependency, registry, environment, and backend state so the fixed Phase 6 and SqlCache nodes pass in suite order as well as alone."
      - "Define one exact local-suite selection that excludes only the three explicit Phase 8 live-qualification modules and does not collect tests whose live fixture is unavailable."
      - "Rerun the fully provisioned local suite once, the fixed verifier, and scoped Ruff. Keep PostgreSQL/Amazon-S3 and native-Windows status UNAVAILABLE/NOT_QUALIFIED for Phase 8/backlog."
---

# Phase 6: UnifiedCache Policy Composition Verification Report

**Phase Goal:** Cache users receive one coherent policy API while all payload-plus-metadata lifecycle work is delegated to `BlobStore`.
**Verified:** 2026-09-09T06:34:02Z
**Status:** gaps_found
**Re-verification:** No — initial goal-backward verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
| --- | --- | --- | --- |
| 1 | One documented cache import, constructor, nested configuration, decorator, and result surface remains; removed pre-production routes do not delegate. | ✓ VERIFIED | `cacheness.__all__`, `UnifiedCache(config, *, store=...)`, `cached(cache=...)`, the negative public-surface tests, examples, and fixed CACH-06 verifier pass. |
| 2 | Cache payload/catalog lifecycle reaches one selected `BlobStore`; `UnifiedCache` owns only keying, TTL, eviction, invalidation, statistics, and decorator policy. | ✓ VERIFIED | `core.py` binds `self._cache_blob_store = self.store`; put/open/query/exact-delete calls target that store. Fixed CACH-01/CACH-02 and retained Phase 3-5 contracts pass. |
| 3 | TTL, size, predicate, decorator, single-key, and global removal use bounded selection and the same exact-generation lifecycle primitive. | ✓ VERIFIED | `invalidate_where`, `_cleanup_expired`, `invalidate_function`, `clear_all`, and size maintenance converge on `_remove_exact_candidates` -> `BlobStore.delete(expected=...)`; fixed CACH-03 passes. |
| 4 | Cached `None` is a hit and decorator clear reports actual bounded removal truth. | ✓ VERIFIED | `lookup()` separates `snapshot is None` from `snapshot.read() is None`; decorator consumes `CacheOutcome.HIT`; fixed CACH-04 tests pass. |
| 5 | Statistics expose one immutable six-outcome aggregate, independent of catalog authority. | ✓ VERIFIED | Frozen `CacheStatistics` and `_CacheOutcomeRecorder`; statistics tests prove all outcomes and no catalog observation. Fixed CACH-05 passes. |
| 6 | Optional capability/error, committed-partial/close, non-destructive corruption, and strict malformed projection contracts are preserved. | ✓ VERIFIED | Phase 6 policy/topology/public tests plus exact hostile projection node pass; typed causes and receipts are retained. |
| 7 | `SqlCache` remains separate with representative regression coverage. | ✓ VERIFIED | The fixed verifier runs `tests/test_sql_cache.py` under the fully provisioned dependency groups and reports `CACH-07 SqlCache regression: PASS`. |
| 8 | The cutover is complete across current runnable tests and the non-live fully provisioned local suite is green. | ✗ FAILED | Broader runtime suite fails on stale flat `CacheConfig`/implicit-cache tests and later suite-order/selection failures. Plan 06-08 explicitly forbids a phase-completion claim while this gate is red. |

**Score:** 7/8 truths verified (0 present-but-behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
| --- | --- | --- | --- |
| `src/cacheness/cache_policy.py` | Immutable lookup, put, removal, maintenance, and statistics contracts | ✓ VERIFIED | Substantive (423 lines), imported and used by `core.py`/decorators; fixed behavioral tests pass. |
| `src/cacheness/core.py` | One-store cache policy facade | ✓ VERIFIED | Substantive (1,152 lines), wired to BlobStore/catalog and exercised by all Phase 6 contract suites. |
| `src/cacheness/decorators.py` | Explicit-cache, outcome-aware decorator | ✓ VERIFIED | Substantive (108 lines); calls `lookup_call`, `put_call`, and `invalidate_function`. |
| `src/cacheness/config.py` | Nested ownership-aligned cache configuration | ✓ VERIFIED | Canonical constructor accepts nested components only; scoped validation and public-contract tests pass. |
| `src/cacheness/__init__.py` | Canonical exported cache/storage/SqlCache surfaces | ✓ VERIFIED | Explicit `__all__`; retired aliases/factories absent. |
| `docs/CACHE_POLICY.md` and examples | Executable public contract without guarantee inflation | ✓ VERIFIED | Imports are canonical; API example uses `@cached(cache=cache)` and actual removal reports. Plan-07 RE2 checker limitation was manually resolved from source and executable example tests. |
| `tools/verify_phase6_contracts.py` | Fixed fail-closed verifier | ✓ VERIFIED | Fixed manifest, static architecture checks, requirement-linked diagnostics, and retained regressions; exits 0 independently. |
| Current non-live repository test corpus | All retained tests consume the canonical cutover and pass together | ✗ FAILED | Stale and order-dependent tests remain outside the fixed manifest. |

### Key Link Verification

| From | To | Via | Status | Details |
| --- | --- | --- | --- | --- |
| `src/cacheness/core.py` | `storage/blob_store.py` | one selected store, `open_entry`, `put_entry`, `query_catalog`, `delete(expected=...)` | ✓ WIRED | No policy-side payload/catalog resource deletion found. |
| `src/cacheness/decorators.py` | `src/cacheness/core.py` | lookup/put/function invalidation policy methods | ✓ WIRED | Explicit cache injection; no singleton, atexit, or weakref owner. |
| `src/cacheness/core.py` | `storage/catalog.py` | authenticated schema/query/page and function namespace | ✓ WIRED | Bounded page/work arguments and exact expectations are enforced. |
| `examples/api_request_caching.py` | `src/cacheness/decorators.py` | `@cached(cache=cache)` plus explicit recompute outcomes | ✓ WIRED | Manual source check resolves the plan checker’s unsupported-RE2-pattern result. |
| `tools/verify_phase6_contracts.py` | Phase 6 and retained lifecycle tests | fixed node manifest | ✓ WIRED | Includes CACH-07, strict projection node, and Phase 3-5 regressions. |

### Data-Flow Trace (Level 4)

| Artifact | Data/decision | Source | Produces real canonical data | Status |
| --- | --- | --- | --- | --- |
| `UnifiedCache.lookup` | presence, value, generation | one `BlobStore.open_entry` snapshot | Yes | ✓ FLOWING |
| `UnifiedCache.put` | receipt and policy maintenance | `BlobStore.put_entry`, then one bounded maintenance step | Yes; receipt remains canonical | ✓ FLOWING |
| Removal paths | attempted/removed/conflict/failure | authenticated catalog/snapshot expectations -> `BlobStore.delete(expected=...)` | Yes | ✓ FLOWING |
| `CacheStatistics` | six outcome counters | lookup outcome recorder only | Derived observer; never authority | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| --- | --- | --- | --- |
| Fixed Phase 6, CACH-07, strict projection, retained lifecycle contract | `uv run --group dev --group recommended --group sql --group dataframes python tools/verify_phase6_contracts.py --repo-root .` | CACH-01..07 PASS; strict projection PASS; retained Phase 3-5 PASS; remote explicitly mocked-candidate | ✓ PASS |
| Complete local dependency suite outside explicit live evidence | Fully provisioned `pytest` suite with live integration files selected out | Runtime failures begin in stale `test_blob_manifest.py`/`test_cache_signing.py` and continue across legacy consumers/order-sensitive tests | ✗ FAIL |
| Scoped phase lint | `uv run --group dev --group recommended --group sql --group dataframes ruff check` over Phase 6 source/tool/contract paths | `All checks passed!` | ✓ PASS |

### Probe Execution

No shell probe is declared. `tools/verify_phase6_contracts.py` is the phase’s executable fixed verifier and was run independently above.

### Requirements Coverage

| Requirement | Source | Status | Evidence |
| --- | --- | --- | --- |
| CACH-01 | ROADMAP / Plans 02, 05, 08 | ✓ SATISFIED | One BlobStore/engine; fixed topology and architecture checks pass. |
| CACH-02 | ROADMAP / Plans 01-05, 08 | ✓ SATISFIED | Policy remains in UnifiedCache; lifecycle remains in BlobStore. |
| CACH-03 | ROADMAP / Plans 02-04, 08 | ✓ SATISFIED | All named removal paths share bounded exact deletion. |
| CACH-04 | ROADMAP / Plans 01, 04 | ✓ SATISFIED | Present `None` and decorated `None` are behavioral hits. |
| CACH-05 | ROADMAP / Plans 01, 08 | ✓ SATISFIED | Immutable six-outcome statistics tests pass. |
| CACH-06 | ROADMAP / Plans 01, 04-08 | ✓ SATISFIED | Canonical API/config/result surface and explicit ownership tests pass. |
| CACH-07 | Fixed regression label / Phase 1 requirement | ✓ SATISFIED | `tests/test_sql_cache.py` passes in the fixed fully provisioned verifier; whole-suite isolation remains a separate gap. |

No additional Phase 6 requirement is orphaned from the plans.

### Decision Coverage

| Decisions | Status | Evidence |
| --- | --- | --- |
| D-01 through D-04 | ✓ VERIFIED | Canonical export/construction/nested config and optional-capability contracts. |
| D-05 through D-08 | ✓ VERIFIED | Presence-bearing results, typed classification, one lookup, immutable statistics. |
| D-09 through D-13 | ✓ VERIFIED | TTL and every removal/eviction path are exact, bounded, resumable, and fail closed. |
| D-14 through D-17 | ✓ VERIFIED | Explicit decorator ownership, deterministic function policy, truthful clear, receipt-preserving close/partial semantics. |

The decision-coverage gate returned `17/17 honored`; direct code and behavioral evidence above independently supports that heuristic result.

### Test Quality Audit

| Test Scope | Linked Req | Active/Skipped | Circular | Assertion Level | Verdict |
| --- | --- | --- | --- | --- | --- |
| Phase 6 lookup/removal/statistics/policy/decorator/topology/public suites | CACH-01..06 | Active; no requirement-only disabled test found | None found | Behavioral, value, and exact-call assertions | PASS |
| `tests/test_sql_cache.py` | CACH-07 | Active with provisioned SQL/dataframe groups; dependency skip guards remain for unsupported installs | None found | Behavioral/value | PASS in fixed process; suite-isolation gap |
| Broader current repository suite | Phase completion gate | Active | None established | Mixed | BLOCKER: stale cutover consumers fail before retained behavior can be evaluated |

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
| --- | --- | --- | --- |
| Phase 6 modified source/test/tool scope | `TBD` / `FIXME` / `XXX` debt markers | None | No unreferenced blocker marker found. |
| Current tests (eight files) | Removed flat `CacheConfig` keywords | BLOCKER | 37 calls found; one is a deliberate negative assertion, the rest leave the suite on the retired surface. |
| Current tests (six files) | `UnifiedCache(...)` without explicit `store=` | BLOCKER | 20 calls found; one is a deliberate negative assertion, the rest contradict explicit composition. |

### Human Verification Required

N/A — infrastructure/core-library phase with no user-facing UI. All behavior-dependent Phase 6 truths have executable tests; the remaining failures require test migration and isolation, not subjective UAT.

### Deferred Items

PostgreSQL/Amazon-S3 real-service qualification and native Windows remain `UNAVAILABLE`/`NOT_QUALIFIED` in Phase 8/backlog and do not affect this Phase 6 verdict. The stale canonical-cutover tests are **not deferred**: Plan 06-08 explicitly makes a green current-environment suite a Phase 6 completion gate. Phase 8’s permission to retain historical compatibility evidence does not authorize keeping incompatible tests in the default runnable gate.

### Gaps Summary

The product implementation is strongly evidenced: the independent fixed verifier passes CACH-01 through CACH-07, strict malformed-projection rejection, and retained Phase 3-5 lifecycle contracts; every D-01..D-17 decision has behavioral or structural evidence; scoped Ruff passes. The phase still cannot be marked complete because its repository-level cutover is incomplete.

Close the gap only in tests and test harnesses unless a migrated canonical test demonstrates an independent implementation defect. Convert retained behavior tests to nested configuration, explicit store composition, and typed results; retire compatibility-only expectations; isolate global/import/optional dependency state; and select Phase 8 live suites explicitly. Do **not** add locks, queues, coordinators, readiness mechanisms, timing guarantees, cross-resource ACID, compatibility shims, implicit constructors, or alternate cache authorities.

---

_Verified: 2026-09-09T06:34:02Z_
_Verifier: the agent (gsd-verifier)_
