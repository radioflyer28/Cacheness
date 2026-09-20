---
phase: 06
fixed_at: 2026-09-09T06:20:42Z
review_path: .planning/phases/06-unifiedcache-policy-composition/06-REVIEW.md
iteration: 3
findings_in_scope: 2
fixed: 2
skipped: 0
status: all_fixed
---

# Phase 6: Code Review Fix Report

**Fixed at:** 2026-09-09T06:20:42Z
**Source review:** `.planning/phases/06-unifiedcache-policy-composition/06-REVIEW.md`  
**Current iteration:** 3

## Cumulative Summary

| Iteration | Findings in scope | Fixed | Skipped |
| --- | ---: | ---: | ---: |
| 1 | 7 | 7 | 0 |
| 2 | 4 | 4 | 0 |
| 3 | 2 | 2 | 0 |

Iteration 1 resolved the prior policy/configuration and warning findings.
Iteration 2 resolves the four critical findings in the current review without
adding a lifecycle lock, queue, coordinator, background loop, timing guarantee,
or a second lifecycle authority.

Iteration 3 resolves the remaining two warnings without changing storage
lifecycle authority, concurrency behavior, or topology guarantees.

## Fixed Issues — Iteration 3

### WR-01: Requirement labels can report PASS while architecture verification failed

**Status:** Fixed — requires human verification
**Files modified:** `tools/verify_phase6_contracts.py`, `tests/test_phase6_contract_verifier.py`
**Commit:** `23ad759`

Static verifier diagnostics now map explicitly to the Phase 6 requirement
labels they invalidate. Unreadable static evidence marks every CACH label as
incomplete rather than rendering a partial PASS. The new self-test injects a
second lifecycle engine diagnostic and proves `CACH-01` renders `see
diagnostics`, never `PASS`.

### WR-02: Canonical persisted configuration still advertises runtime-inert options

**Files modified:** `src/cacheness/config.py`, `tests/test_config_validation.py`, `tests/test_phase6_public_api_contract.py`
**Commit:** `409c704`

The canonical nested configuration no longer accepts `create_cache_dir`,
`temp_dir`, `enable_metadata`, or the inert memory-cache options. Their
validation and logging paths were removed with the fields; no alias,
compatibility property, or silent loader conversion remains. Direct tests now
validate a live compression field and assert every removed option is rejected.

## Fixed Issues — Iteration 2

### CR-01: Bounded TTL cleanup still returns a cursor invalidated by its own deletion

**Status:** Fixed — requires human verification
**Files modified:** `src/cacheness/core.py`, `tests/test_phase6_removal_contract.py`
**Commit:** `da29f78` (see execution incident below)

`_cleanup_expired()` now consumes the cache-policy restart token as a fresh
scan, translates stale authority cursors into typed restart evidence, and never
returns a post-delete authority cursor. The new three-page regression follows
each continuation to completion.

### CR-02: Close-after-commit still ignores facade closure for an injected store

**Status:** Fixed — requires human verification
**Files modified:** `src/cacheness/core.py`, `tests/test_phase6_policy_contract.py`
**Commit:** `164bf63`

The internal post-commit maintenance boundary detects an already-closed cache
facade before policy I/O and returns typed, retryable closed maintenance. The
receipt remains valid and the caller-owned injected `BlobStore` stays usable.

### CR-03: Direct BlobStore construction discards configured handler policy

**Files modified:** `src/cacheness/storage/blob_store.py`, `tests/contracts/test_phase6_topology_policy.py`
**Commit:** `a6a087e`

Direct stores now construct `HandlerRegistry(self.config)`. Direct and injected
composition tests prove that disabled handlers remain unavailable and the
caller-selected handler priority is retained by `UnifiedCache`.

### CR-04: The public cutover leaves current test modules uncollectable

**Files modified:** fifteen current test modules, including the fourteen
reviewed stale modules and `tests/test_pandas_compatibility.py` revealed once
the optional dataframes group was installed
**Commit:** `f8d56c6`

All modules now use nested `CacheConfig`, explicit `StoreTopology`/`BlobStore`,
`UnifiedCache`, typed policy results, and the explicit `cached(cache=...)`
decorator. Compatibility-only assertions for removed aliases, flat/blob config,
factories, and legacy metadata APIs were removed; the retained coverage exercises
key serialization, handler selection/priority, integrity outcomes, Pandas
round-trips, policy invalidation, and topology ownership.

## Prior Iteration — Archived Summary

Iteration 1 fixed seven findings: stale policy removal continuation (`6be21e1`),
injected handler-registry ownership (`bf2f5ee`), post-commit maintenance receipt
preservation (`877784d`, `a21843d`), policy config serialization (`29978fb`),
strict-projection verifier rendering (`335265e`), canonical decorator/key
regressions (`dc3e38f`), and retirement of obsolete nested size/TTL settings
(`2b3e4f1`).

## Verification — Iteration 2

All verification ran in the **main checkout** at
`/Users/akriz/code/cacheness`; `.planning/config.json` sets
`workflow.use_worktrees=false`.

- Mandatory Tier 1 re-reads and Python AST parsing passed for every edited
  source and test module.
- Focused tests passed: the four direct critical-finding scopes passed, the
  fourteen migrated modules passed (`53 passed`), and the newly exposed Pandas
  compatibility scope passed (`2 passed`) with the complete local groups.
- Targeted Ruff checks passed for every touched source/test path.
- Complete local-group collection passed with
  `uv run --group dev --group recommended --group sql --group dataframes pytest --collect-only -q`.
  It reports only the existing non-fatal dataclass collection warning.
- `uv run --group dev --group recommended --group sql --group dataframes python tools/verify_phase6_contracts.py --repo-root .`
  passed CACH-01 through CACH-07, canonical decorator/key regressions, strict
  projection rejection, and retained Phase 3-5 lifecycle contracts. Remote
  evidence remains a mocked candidate; BACK-05 is still Phase 8 work.
- Repository-wide `ruff check src tests` reports 33 pre-existing findings in
  untouched paths (for example `file_hashing.py`, compatibility handler barrels,
  and older tests). No finding is in an Iteration 2 edited file.

## Verification — Iteration 3

All verification ran in the **main checkout** at
`/Users/akriz/code/cacheness`; `.planning/config.json` sets
`workflow.use_worktrees=false`.

- Mandatory Tier 1 re-reads, `git diff --check`, and Python AST parsing passed
  for all five edited source and test modules.
- Focused warning tests passed: `15 passed` for the verifier self-test module,
  and `37 passed` for the configuration-validation and public API modules.
- The 27-module Phase 6 reviewed test scope passed. It emitted only the
  existing non-fatal dataclass collection warning in
  `tests/test_cache_key_consistency.py`.
- Complete local-group collection passed with
  `uv run --group dev --group recommended --group sql --group dataframes pytest --collect-only -q`.
  It emitted the same existing non-fatal dataclass collection warning.
- The fully provisioned Phase 6 verifier passed CACH-01 through CACH-07,
  canonical decorator/key regressions, strict-projection rejection, and retained
  Phase 3-5 lifecycle contracts. Remote evidence remains a mocked candidate;
  BACK-05 remains Phase 8 work.
- Scoped Ruff passed for all Phase 6 production, verifier, and contract-test
  files, including every Iteration 3 edited path.

## Execution Incident

The first commit helper invocation used a shared Git index that already
contained unrelated staged planning artifacts. Its resulting CR-01 commit,
`da29f78`, contains 66 files rather than only the two CR-01 paths. No unrelated
file was reverted or altered to repair that history. Per coordinator direction,
the commit is preserved pending explicit user authorization for any history
rewrite. Every subsequent Iteration 2 commit used exact-path staging and
file-only commits.

---

_Fixed: 2026-09-09T06:20:42Z_
_Fixer: the agent (gsd-code-fixer)_  
_Iteration: 3_
