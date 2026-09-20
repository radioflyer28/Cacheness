---
phase: 03-atomic-lifecycle-and-recovery-engine
plan: "20"
subsystem: sqlite-lifecycle-authority
tags: [sqlite, blobstore, recovery, concurrency, benchmark]
status: complete
qualification_commit: 1734757949a100d98a6d23eaab264cb274058ce6
correctness: PASS
recovery: PASS
progress: PASS
performance: PASS
repository_quality: PASS
full_suite: PASS
qualification_worktree: CLEAN
protected_sidecars: IDENTICAL
dependency_graph:
  requires: ["03-18 SQLite lifecycle authority", "ADR 0001"]
  provides: ["SQLite/CAS lifecycle boundary", "typed retryable contention outcome", "separate performance evidence"]
  affects: ["BlobStore lifecycle", "reconciliation", "lifecycle benchmark"]
tech_stack:
  added: []
  patterns: ["SQLite is the sole lifecycle truth", "immutable external effects reconcile from durable intent"]
key_files:
  modified:
    - src/cacheness/config.py
    - src/cacheness/storage/sqlite_lifecycle_authority.py
    - benchmarks/lifecycle_authority_benchmark.py
    - benchmarks/lifecycle_authority_baseline.json
    - tests/test_blob_store_atomic_lifecycle.py
    - tests/test_phase3_postreview_concurrency.py
decisions:
  - "Authority contention may succeed, conflict exactly, or raise a contextual retryable timeout."
  - "The 5.0-second runtime default is caller policy; 0.187 seconds is historical benchmark evidence only."
  - "Native Windows remains UNAVAILABLE/NOT_QUALIFIED."
metrics:
  duration: "~32m"
  completed: 2026-09-06
actuals:
  tokens: 73619
  tasks: 4
  commits: 6
---

# Phase 03 Plan 20: ADR-Scoped SQLite Lifecycle Summary

SQLite remains the one lifecycle authority for local filesystem BlobStore data: durable intent precedes immutable payload effects, promotion changes visibility, and reconciliation resolves exact residue without a cross-resource ACID claim.

## Outcomes

- Set `LifecycleLimits.authority_busy_timeout_seconds` to the caller-configurable 5.0-second operational default and retained a contextual, caused `CacheBlobLifecycleTimeoutError` with `retryable: true` for exhausted SQLite `BUSY`/`LOCKED` contention.
- Removed the authority-wide writer-admission ticket/gate/registry and fork reset, bootstrap readiness event, staged connection wrapper, exhaustive timeout-stage taxonomy, test observer, and the interrupted Plan 03-19 scheduler-dispatch relabel patch. SQLite `BEGIN IMMEDIATE`, short transactions, exact lineage CAS, schema/resource validation, rollback, and deterministic recovery remain.
- Added process-interruption coverage for an in-progress durable exclusive stream and the interval after file fsync but before parent-directory fsync. Both cases preserve the old committed generation and reconcile only the prepared mutation's exact candidate locator.
- Reclassified concurrency evidence around integrity, recovery, and progress. Tests accept success, exact conflict, or fully contextual retryable timeout; no test requires universal completion under the historical 0.187-second observation.
- Decoupled the benchmark baseline from runtime configuration. Historic contention distributions, provenance, and the 0.187-second capture remain auditable performance evidence, while newly captured baselines need not recreate retired queue-stage instrumentation.

## Commits

- `4d629a2` `test(03-20): define operational authority timeout default`
- `6658e52` `feat(03-20): simplify SQLite lifecycle contention boundary`
- `b1988f8` `test(03-20): classify SQLite contention outcomes`
- `438250b` `feat(03-20): decouple benchmark timeout policy`
- `47f2cfe` `test(03-20): make reconciliation timeout fixture explicit`
- `1734757` `chore(03-20): clear owned config lint debt`

## Qualification Evidence

The final detached worktree was created at `1734757949a100d98a6d23eaab264cb274058ce6`, started clean, and remained clean after all commands.

| Group | Command/result |
| --- | --- |
| Correctness | `pytest` over lifecycle contract, BlobStore read/atomic lifecycle, projection mutation/SQL atomicity, and filesystem containment: PASS. Two platform-permission containment skips were expected. |
| Recovery | `pytest` over SQLite authority, BlobStore reconciliation/close, metadata bootstrap, and interruption-focused postreview cases: PASS (25 selected tests). |
| Progress | `pytest` over BlobStore concurrency, SQLite admission, and both SQLite concurrency modules: PASS (18 selected tests). |
| Performance | `python benchmarks/lifecycle_authority_benchmark.py --verify-baseline benchmarks/lifecycle_authority_baseline.json`: PASS. Its intentional held-writer samples emitted typed retryable timeouts under the explicit 0.075-second benchmark-local limit. |
| Repository quality | Phase 3 Ruff-delta verifier and direct Ruff over all Plan 03-20 owned Python files: PASS. |
| Full suite | `uv run --isolated --python 3.11 --all-extras --group dev --frozen pytest -q -o log_cli=false`: PASS. Expected platform/optional-dependency skips include native Windows, live PostgreSQL, and TensorFlow. |

`03-VERIFICATION.md` was intentionally not changed; it remains non-passing pending a fresh verifier.

## Protected Sidecar Fingerprints

The original workspace was never copied into the detached worktree. Before and after qualification fingerprints were identical.

| Path | Presence/type | Device/inode | Size | mtime_ns | SHA-256 |
| --- | --- | --- | ---: | ---: | --- |
| `tests/fixtures/compat/sqlite-columns-v0314/metadata.sqlite3-wal` | present/regular | `16777235/113687243` | 0 | `1788692452094166062` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `tests/fixtures/compat/sqlite-columns-v0314/metadata.sqlite3-shm` | present/regular | `16777235/113687244` | 32768 | `1788692452094316312` | `fd4c9fda9cd3f9ae7c962b0ddf37232294d55580e1aa165aa06129b8549389eb` |

## Deviations from Plan

### Auto-fixed Issues

1. **[Rule 1 - Bug] Removed an obsolete event set after authority queue removal**
   - **Found during:** Task 1
   - **Fix:** Removed the dangling bootstrap readiness-event use with the retired event.
   - **Commit:** `6658e52`

2. **[Rule 1 - Bug] Corrected test worker variable shadowing**
   - **Found during:** Task 2
   - **Fix:** Avoided shadowing the nested worker callable while constructing test threads.
   - **Commit:** `b1988f8`

3. **[Rule 1 - Regression] Made reconciliation's mocked expiration policy explicit**
   - **Found during:** Task 4 recovery qualification
   - **Issue:** The test intended to expire a reconciliation clock after two samples but accidentally relied on the prior benchmark-derived default.
   - **Fix:** Set its local authority timeout to `0.1`, retaining the deterministic boundary test without coupling it to the production default.
   - **Commit:** `47f2cfe`

4. **[Rule 3 - Blocking quality gate] Removed dead config locals in a Plan-owned file**
   - **Found during:** Task 4 direct Ruff qualification
   - **Fix:** Removed two unused backend-category bindings without changing configuration validation behavior.
   - **Commit:** `1734757`

5. **[Rule 3 - Verification command resolution] Used the repository's containment regression filename**
   - **Found during:** Task 4 qualification setup
   - **Fix:** Replaced a nonexistent shorthand `test_path_security.py` selection with `tests/test_filesystem_containment.py`; the corrected integrity group passed.

## TDD Gate Compliance

Task 1 completed RED (`4d629a2`) before GREEN (`6658e52`). Task 2 was a guarantee-class test rewrite after the real implementation was present, so its test-only atomic commit is `b1988f8`.

## Known Stubs

None. The modified lifecycle, benchmark, and test paths contain no plan-blocking placeholders or empty data wiring.

## Self-Check: PASSED

- All six implementation commits are present in Git history.
- The summary exists at the planned path.
- The detached worktree was removed and pruned after a clean status check.
