---
phase: 06
slug: unifiedcache-policy-composition
status: validated
nyquist_compliant: true
wave_0_complete: true
executed: 2026-09-09T08:10:40Z
---

# Phase 6 — Validation Strategy and Executed Evidence

> This ledger records the final local commands actually run on 2026-09-09. It
> deliberately distinguishes deterministic local evidence from native-platform
> and real-service qualification. `PASS` below never credits a mock, skip, or
> local collection result as PostgreSQL, Amazon S3, or Windows qualification.

## Environment and Boundaries

| Item | Observed value | Disposition |
| --- | --- | --- |
| Python | CPython 3.13.15 | Current host only; Python-version qualification remains Phase 8. |
| Local dependency environment | `uv run --isolated --all-extras --group dev --frozen` | Locked all-extras local environment; no package or lockfile mutation. |
| PostgreSQL/Amazon S3 | No live resources or credentials used | `UNAVAILABLE`/`NOT_QUALIFIED`; deterministic candidate evidence is not BACK-05 qualification. |
| Native Windows | Not run on this macOS host | `UNAVAILABLE`/`NOT_QUALIFIED`; no native-Windows claim. |
| Packaging, coverage, performance | Not run as Phase 6 acceptance | Explicitly deferred to Phase 8. |

## Executed Commands

| Command | Actual result | Status |
| --- | --- | --- |
| `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase6_contract_verifier.py -o log_cli=false` | `18 passed` | ✅ green |
| `uv run --isolated --all-extras --group dev --frozen python tools/run_phase6_local_suite.py --repo-root .` | Exit `0`; exact three live-path exclusions; `1232` collected local nodes, `1223 passed`, `9 skipped`, `0 failed`. | ✅ green |
| `uv run --isolated --all-extras --group dev --frozen python tools/verify_phase6_contracts.py --repo-root .` | Exit `0`; CACH-01 through CACH-06, strict projection, canonical cutover/suite-order, and retained Phase 3–5 groups all passed. | ✅ green |
| `uv run --isolated --all-extras --group dev --frozen ruff check` over the 15 Plan 09–11 Python paths named in `06-11-PLAN.md` | `All checks passed!` | ✅ green |

The local runner printed only these exclusions:

- `tests/integration/test_postgresql_authority.py`
- `tests/integration/test_s3_generation.py`
- `tests/integration/test_remote_topology.py`

The passing runner and verifier are recorded at `2026-09-09T08:10:40Z`.
Their concrete artifacts are [the root-safe runner](../../../tools/run_phase6_local_suite.py),
[the fixed verifier](../../../tools/verify_phase6_contracts.py), and
[its adversarial contract tests](../../../tests/test_phase6_contract_verifier.py).

## Per-Task Verification Map

### Requirement Evidence

| Requirement | Fixed evidence | Current result | Status |
| --- | --- | --- | --- |
| CACH-01 | `tests/contracts/test_phase6_topology_policy.py` plus verifier one-store/one-engine AST checks | CACH-01 printed `PASS` | ✅ green |
| CACH-02 | `tests/test_phase6_policy_contract.py` plus verifier direct-mutation/projection-authority checks | CACH-02 printed `PASS` | ✅ green |
| CACH-03 | `tests/test_phase6_removal_contract.py` plus bounded query/direct-delete AST checks | CACH-03 printed `PASS` | ✅ green |
| CACH-04 | `tests/test_phase6_lookup_contract.py`, `tests/test_phase6_decorator_contract.py` | CACH-04 printed `PASS` | ✅ green |
| CACH-05 | `tests/test_phase6_statistics.py` | CACH-05 printed `PASS` | ✅ green |
| CACH-06 | `tests/test_phase6_public_api_contract.py`, `tests/test_phase6_decorator_contract.py`, contract-text check | CACH-06 printed `PASS` | ✅ green |
| CACH-07 (historical) | Historical `tests/test_sql_cache.py` ordered-isolation probe | Original local-suite result was `CACH-07 SqlCache regression: PASS`; the deleted surface is now covered by Phase 10's direct-removal/negative contract | ⚪ superseded — Phase 10 direct-removal/negative contract |

## Strict Derived-Projection Evidence

| Criterion | Exact node | Observed result | Status |
| --- | --- | --- | --- |
| ROADMAP Phase 6 SC-06 — malformed direct projection observations fail closed | `tests/test_catalog_projection.py::test_json_projection_rejects_incompatible_derived_documents` | `1 passed`; the named test writes `{"format_version": 999}` and requires `load_projection_checkpoint()` to raise `JsonProjectionError` matching `incompatible shape`. | PASS |

## Decision Coverage

| Decision | Concrete fixed test or source assertion | Status |
| --- | --- | --- |
| D-01 | `test_removed_development_public_names_do_not_delegate`; verifier retired-route AST audit | PASS |
| D-02 | `test_local_profiles_share_one_explicit_cache_lifecycle`; static one-`BlobStore` composition check | PASS |
| D-03 | `test_config_requires_an_explicit_blobstore_composition_before_io`; nested-config API check | PASS |
| D-04 | `test_optional_yaml_capability_fails_only_when_requested` | PASS |
| D-05 | `test_explicit_memory_composition_returns_hit_for_cached_none` | PASS |
| D-06 | `test_lookup_observes_blob_store_once_for_absent_and_present_none`; decorator cached-`None` test | PASS |
| D-07 | `test_lookup_classifies_declared_storage_failures_without_cleanup`; typed decorator failure test | PASS |
| D-08 | `test_statistics_count_every_outcome_once_and_are_order_independent`; no-catalog-observation test | PASS |
| D-09 | `test_expired_lookup_reports_the_exact_lifecycle_removal` | PASS |
| D-10 | `test_single_key_invalidation_returns_a_truthful_report`; verifier direct-resource-delete AST audit | PASS |
| D-11 | `test_function_clear_is_exact_truthful_and_namespace_scoped` | PASS |
| D-12 | `test_size_maintenance_is_page_bounded_and_requires_explicit_resumes` | PASS |
| D-13 | `test_unsupported_predicate_fails_before_catalog_io`; resumable predicate removal test | PASS |
| D-14 | `test_explicit_decorator_module_has_no_implicit_lifecycle_owner`; hidden-global/atexit/weakref audit | PASS |
| D-15 | `test_default_decorator_recomputes_only_normal_miss_outcomes`; pre-call key mutation test | PASS |
| D-16 | `test_function_clear_preserves_a_concurrently_replaced_generation` | PASS |
| D-17 | `test_put_retains_canonical_receipt_when_maintenance_is_incomplete`; retained close-after-commit regression | PASS |

## Edge and Prohibition Coverage

| Scope | Evidence | Status |
| --- | --- | --- |
| CACH-01 empty/adjacency/order predicates | Unsupported-topology preflight and stable authority/payload order tests | PASS |
| CACH-02 classification predicate | Declared-storage-failure classification test | PASS |
| CACH-03 empty/order/resume predicates | Empty/global invalidation plus bounded/resumable predicate removal tests | PASS |
| CACH-04 result classification predicate | One-read absent/present-`None` lookup test | PASS |
| CACH-05 classification predicate | Immutable, order-independent statistics test | PASS |
| CACH-06 empty/ordering predicate | Public-surface ordering and removed-surface tests | PASS |
| `[FLAGGED/UNVERIFIED]` no second engine/coordinator, lock, queue, sidecar, readiness registry, or projection authority | `tests/test_phase6_contract_verifier.py` mutation fixtures plus AST audit of canonical modules | PASS |
| `[FLAGGED/UNVERIFIED]` no direct payload/catalog/projection deletion from cache policy | Verifier direct-resource-delete mutation fixture and canonical-module AST audit | PASS |
| `[FLAGGED/UNVERIFIED]` no global cache, compatibility route, or ambiguous ownership boolean | Verifier hidden-global and retired-route fixtures; constructor-form contract tests | PASS |
| `[FLAGGED/UNVERIFIED]` no unbounded policy traversal | Verifier bounded-query fixture; bounded size-maintenance test | PASS |
| `[FLAGGED/UNVERIFIED]` no guarantee inflation | Contract-text self-tests and `docs/CACHE_POLICY.md` audit reject cross-resource ACID, universal contender success, global-oldest, and benchmark-deadline promises | PASS |
| Mocked PostgreSQL/S3 candidate | `test_deterministic_remote_candidate_uses_the_same_policy_call_graph` | Candidate-only; not BACK-05 qualification |

## Retained Storage Regression Evidence

`tools/verify_phase6_contracts.py` now contains the literal Plan 09–11
canonical-cutover inventory, audits each migrated source AST, and explicitly
names retained Phase 3–5 local lifecycle, integrity, recovery, authority, and
topology nodes. The final fixed-verifier run passed that finite group. The
historical separate SqlCache regression is superseded by the Phase 10
direct-removal/negative contract; no live service result is credited here.

## Open Gates

1. **BACK-05 live remote qualification:** PostgreSQL and Amazon S3 were not
   contacted. Mocked/deterministic remote-candidate coverage is not release
   evidence. Phase 8 retains the non-substitutable live gate.
2. **Native Windows:** no native Windows execution occurred. Existing skipped
   Windows evidence is not counted as Phase 6 qualification.

## Validation Sign-Off

- [x] Every Phase 6 plan task has an automated command.
- [x] Sampling continuity is preserved by Task 1 self-tests and Task 2 fixed checks.
- [x] Wave 0 Phase 6 contract files exist and are exercised by the fixed manifest.
- [x] No watch-mode flags were used.
- [x] Scoped Ruff passed for Phase 6 production/test/tool files.
- [x] Current-environment non-live local suite is green under the locked all-extras environment.
- [x] Fixed verifier is fully green for CACH-01 through CACH-06; the historical CACH-07 collection is explicitly superseded by Phase 10.
- [x] `nyquist_compliant: true` for the finite Phase 6 local gate.

**Approval:** local Phase 6 evidence is complete. PostgreSQL/Amazon-S3 and
native-Windows qualification remain explicitly open for Phase 8/backlog.

## Canonical Validation Normalization (Phase 11)

Phase 6 owns only CACH-01 through CACH-06. Its former CACH-07/`SqlCache`
selector is retained above as historical evidence with an explicit supersession
mapping to the Phase 10 direct-removal/negative contract; it is not a runnable current
requirement. PostgreSQL and Amazon S3 remain `NOT_QUALIFIED`, controlled-Linux
performance remains `NOT_QUALIFIED`, Windows remains `NOT_QUALIFIED`, and
immutable publication remains `NOT_PUBLISHED`.
