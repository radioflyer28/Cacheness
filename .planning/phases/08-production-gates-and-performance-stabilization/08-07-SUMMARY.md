---
phase: 08-production-gates-and-performance-stabilization
plan: 07
subsystem: performance-qualification
tags: [benchmarks, distributions, sha256, xxh3, controlled-runner, github-actions]
requires:
  - phase: 07.1-obstore-payload-participant-unification
    provides: Current public BlobStore/UnifiedCache composition and guarded obstore payload participant
  - phase: 08-production-gates-and-performance-stabilization
    provides: Exact source/revision evidence vocabulary and structural scale evidence
provides:
  - Deterministic representative handler, BlobStore, and UnifiedCache workload definitions
  - Stdlib raw-sample distribution and SHA-256/XXH3 comparison harness with immutable baseline checks
  - Exact-SHA controlled-Linux workflow contract with pinned actions and bounded artifacts
affects: [phase-08-controlled-performance, release-evidence, QUAL-03, QUAL-06]
actuals:
  tokens: 13825
  tasks: 3
  commits: 8
tech-stack:
  added: []
  patterns:
    - Stdlib percentiles, subprocess isolation, and raw-sample retention instead of a benchmark dependency
    - Relative reviewed median/tail envelopes kept outside runtime policy
    - Exact-SHA controlled-runner workflow with source and environment binding
key-files:
  created:
    - benchmarks/phase8_workloads.py
    - benchmarks/phase8_benchmarks.py
    - tests/performance/test_phase8_benchmarks.py
    - .github/workflows/performance.yml
  modified: []
key-decisions:
  - "The shipped ArrayHandler writes native NPZ only; read-only legacy Blosc2 is not benchmarked as a fabricated current write path."
  - "Controlled performance compares raw p50/p99 distributions only on cacheness-perf-linux-x64; diagnostics and remote/macOS timings do not enter the release envelope."
  - "SHA-256 remains the canonical persisted digest; XXH3 is comparative throughput evidence only."
patterns-established:
  - "Benchmark callbacks time one named boundary after fixture/setup work has completed."
  - "Baseline writes require explicit capture or justified recalibration and use atomic replacement."
requirements-completed: [QUAL-03, QUAL-06]
coverage:
  - id: D1
    description: Representative deterministic generic, NumPy NPZ, and dataframe workload inventory is separated across handler, BlobStore, and UnifiedCache boundaries.
    requirement: QUAL-06
    verification:
      - kind: unit
        ref: uv run pytest -q -o log_cli=false tests/performance/test_phase8_benchmarks.py -k workloads -x
        status: pass
    human_judgment: false
  - id: D2
    description: Raw latency samples, deterministic percentile derivations, SHA-256/XXH3 comparisons, and immutable baseline verification remain evidence-only.
    requirement: QUAL-06
    verification:
      - kind: unit
        ref: uv run pytest -q -o log_cli=false tests/performance/test_phase8_benchmarks.py -k 'distribution or hash or baseline' -x
        status: pass
    human_judgment: false
  - id: D3
    description: The manual controlled-Linux workflow requires an exact detached candidate SHA, a named runner, pinned actions, and a fixed envelope artifact.
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: uv run pytest -q -o log_cli=false tests/performance/test_phase8_benchmarks.py -k workflow -x
        status: pass
    human_judgment: false
duration: 11m 20s
completed: 2026-09-14
status: complete
---

# Phase 08 Plan 07: Controlled Performance Harness Summary

**A canonical stdlib performance harness now measures isolated handler, BlobStore, and UnifiedCache distributions while preserving SHA-256 integrity semantics and restricting release enforcement to one exact-SHA Linux runner.**

## Performance

- **Duration:** 11m 20s
- **Started:** 2026-09-14T00:35:13Z
- **Completed:** 2026-09-14T00:46:33Z
- **Tasks:** 3/3
- **Files modified:** 4

## Accomplishments

- Added deterministic 4 KiB generic-object, 16/128 MiB NPZ-array, and 100k-row pandas/Polars Parquet workload definitions, with cold/warm labels and one reviewed topology per tier.
- Added stdlib-based raw sample distributions, calibrated subprocess workers, environment/source identity, SHA-256-versus-XXH3 throughput and lifecycle-share observations, plus non-mutating baseline verification.
- Added a manually dispatched, protected controlled-performance workflow that checks out one exact candidate SHA detached, validates its Linux runner identity, and uploads a bounded fixed-name envelope.

## Task Commits

1. **Task 1: Define representative format tiers and separated layer workloads**
   - `0c3797f` (`test`) — failing workload contracts
   - `e32738a` (`test`) — clarified byte versus row units
   - `06a0b27` (`feat`) — public-composition workload implementation
2. **Task 2: Implement stdlib distributions, hash comparison, and immutable baseline verification**
   - `e82d21e` (`test`) — failing distribution/hash/baseline contracts
   - `c209e2d` (`feat`) — distribution and immutable-baseline harness
3. **Task 3: Restrict the blocking performance workflow to controlled Linux**
   - `58dc01d` (`test`) — failing workflow contract
   - `d193472` (`feat`) — pinned exact-SHA controlled workflow
4. **Post-task evidence hardening**
   - `353be85` (`fix`) — persisted recalibration rationale

## Files Created/Modified

- `benchmarks/phase8_workloads.py` — deterministic tier descriptors and current public-layer callbacks.
- `benchmarks/phase8_benchmarks.py` — distribution/hash evidence CLI, worker isolation, and baseline capture/verify logic.
- `tests/performance/test_phase8_benchmarks.py` — workload, distribution, hash, baseline, and workflow contract tests.
- `.github/workflows/performance.yml` — manual protected controlled-Linux release performance job.

## Decisions Made

- Kept the 128 MiB persistent array tier on the supported SQLite/filesystem topology and smaller tiers on memory/memory; no handler-by-topology cross-product is generated.
- Kept canonical persisted identity as SHA-256 plus size. XXH3 is emitted only as raw comparative evidence and cannot affect manifest behavior.
- Treat controlled performance envelopes as release evidence rather than runtime deadlines; PostgreSQL/S3 and macOS timing remain diagnostics.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Correctness] Removed a fabricated Blosc2 current-write benchmark path.**

- **Found during:** Task 1
- **Issue:** `ArrayHandler` documents and enforces native NPZ for all new writes; Blosc2 `.b2nd` support is read-only legacy compatibility. Benchmarking a Blosc2 write through another path would have bypassed the shipped public composition.
- **Fix:** Defined the canonical NumPy workload tiers against the current NPZ writer and documented why legacy Blosc2 is excluded.
- **Files modified:** `benchmarks/phase8_workloads.py`, `tests/performance/test_phase8_benchmarks.py`
- **Verification:** `uv run pytest -q -o log_cli=false tests/performance/test_phase8_benchmarks.py -k workloads -x`
- **Committed in:** `06a0b27`

**2. [Rule 1 - Bug] Returned an empty validation result for a successful baseline comparison.**

- **Found during:** Task 2
- **Issue:** The initial verifier returned internal compared-workload names, contradicting its validation-only contract.
- **Fix:** Successful verification returns no failures while regression paths retain typed exceptions.
- **Files modified:** `benchmarks/phase8_benchmarks.py`
- **Verification:** Distribution/hash/baseline contract suite passed.
- **Committed in:** `c209e2d`

**3. [Rule 2 - Missing critical functionality] Persisted recalibration justification in the replacement baseline.**

- **Found during:** Final verification
- **Issue:** Recalibration required a justification at invocation time but would not retain it for later review.
- **Fix:** Wrote and validated a `baseline_change` record containing the mode and required recalibration rationale before atomic replacement.
- **Files modified:** `benchmarks/phase8_benchmarks.py`, `tests/performance/test_phase8_benchmarks.py`
- **Verification:** Full benchmark contract suite passed.
- **Committed in:** `353be85`

---

**Total deviations:** 3 auto-fixed (1 Rule 1, 2 Rule 2).
**Impact on plan:** The fixes retain the actual shipped storage boundary and auditable release evidence without changing production payloads, digest semantics, runtime deadlines, topology guarantees, or lifecycle coordination.

## Known Stubs

None.

## Threat Flags

None. This plan adds benchmark/test/workflow evidence only; it introduces no production network endpoint, authentication boundary, file-access interface, or schema change.

## Verification

- `uv run pytest -q -o log_cli=false tests/performance/test_phase8_benchmarks.py -k workloads -x` — 3 passed.
- `uv run pytest -q -o log_cli=false tests/performance/test_phase8_benchmarks.py -k 'distribution or hash or baseline' -x` — 4 passed.
- `uv run pytest -q -o log_cli=false tests/performance/test_phase8_benchmarks.py -k workflow -x` — 2 passed.
- `uv run pytest -q -o log_cli=false tests/performance/test_phase8_benchmarks.py -x` — 9 passed.
- `uv run ruff check benchmarks/phase8_workloads.py benchmarks/phase8_benchmarks.py tests/performance/test_phase8_benchmarks.py` — passed.
- Reduced diagnostic command wrote `/private/tmp/cacheness-phase8-diagnostic-final.json` without creating or modifying a baseline.
- Diff from `95cd083` contains only the four planned benchmark, test, and workflow files; no production digest, lifecycle, or cache-policy source changed.

## User Setup Required

The controlled release baseline remains an external Plan 08-11 prerequisite. A repository administrator must provide the protected `controlled-performance` environment and a Linux x86-64 runner labelled `cacheness-perf-linux-x64`; ordinary local diagnostics do not qualify a release.

## Next Phase Readiness

Plan 08-11 can capture and verify `benchmarks/phase8_baseline.json` only on the named controlled runner. Its baseline must bind the exact commit, recomputed source digest, runner identity, and environment fingerprint before any controlled envelope can qualify.

## Self-Check: PASSED

- All four planned artifacts and all eight task/fix commits exist in repository history.
- No task commit deleted a tracked file.

---
*Phase: 08-production-gates-and-performance-stabilization*
*Completed: 2026-09-14*
