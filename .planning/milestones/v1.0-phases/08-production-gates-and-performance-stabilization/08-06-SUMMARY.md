---
phase: 08-production-gates-and-performance-stabilization
plan: 06
subsystem: structural-performance-testing
tags: [pytest, scale, memory, rss, evidence, lifecycle]
requires:
  - phase: 07.1-obstore-payload-participant-unification
    provides: One bounded obstore payload participant below BlobStore lifecycle authority
  - phase: 08-production-gates-and-performance-stabilization
    provides: Exact-commit deterministic evidence envelopes and fixed lifecycle contracts
provides:
  - Independent authority and payload-participant call formulas for bounded aggregate operations
  - Fresh-child, normalized peak-RSS observations with fail-closed evidence validation
  - A bounded structural evidence envelope that preserves raw counts without timing claims
affects: [QUAL-07, phase-08-release-evidence, controlled-performance]
actuals:
  tokens: 10840
  tasks: 2
  commits: 5
tech-stack:
  added: []
  patterns:
    - Transparent method-class counting wrappers around existing authority and participant protocol seams
    - Spawned child-process RSS probes compared by workload growth rather than elapsed time
    - Exact allow-listed structural evidence observations with raw counter and RSS facts
key-files:
  created:
    - tools/run_phase8_scale_gates.py
    - tests/performance/test_complexity_contracts.py
    - tests/performance/test_memory_bounds.py
  modified:
    - tools/phase8_evidence.py
key-decisions:
  - "Catalog, reconciliation, statistics, invalidation, clear, and maintenance use independent call classes, so aggregate totals cannot conceal N+1 access."
  - "Peak RSS evidence is collected in fresh children; Linux requires both resource and /proc facts, while unavailable POSIX support fails closed."
  - "Structural evidence keeps performance NOT_QUALIFIED because resource bounds do not replace controlled timing distributions."
patterns-established:
  - "Scale tests seed outside measurement and exercise 10, 100, 1,000, and 10,000 entry tiers with below- and above-default page sizes."
  - "Evidence producers reject absent, crashed, unit-mismatched, or unbounded child results instead of substituting host timing."
requirements-completed: [QUAL-03, QUAL-07]
coverage:
  - id: D1
    description: Fixed-scale call formulas expose authority pages, reads/writes, and participant head/open/delete/list accesses separately for bounded catalog, reconciliation, statistics, invalidation, clear, and maintenance operations.
    requirement: QUAL-07
    verification:
      - kind: unit
        ref: uv run pytest -q -o log_cli=false tests/performance/test_complexity_contracts.py -x
        status: pass
    human_judgment: false
  - id: D2
    description: Isolated child-process RSS probes preserve raw byte evidence, reject malformed observations, and keep direct conditional upload capped rather than claiming streaming behavior.
    requirement: QUAL-07
    verification:
      - kind: unit
        ref: uv run pytest -q -o log_cli=false tests/performance/test_memory_bounds.py tests/performance/test_complexity_contracts.py -x
        status: pass
      - kind: unit
        ref: uv run pytest -q -o log_cli=false tests/test_phase8_release_tracer.py -x
        status: pass
    human_judgment: false
duration: 12m 10s
completed: 2026-09-14
status: complete
---

# Phase 08 Plan 06: Structural Scale and Memory Gates Summary

**Bounded authority/participant call formulas and isolated RSS evidence now expose N+1 and cardinality-proportional memory regressions without using machine speed as correctness.**

## Performance

- **Duration:** 12m 10s
- **Started:** 2026-09-14T00:16:42Z
- **Completed:** 2026-09-14T00:28:52Z
- **Tasks:** 2/2
- **Files modified:** 4

## Accomplishments

- Added transparent wrappers with separate authority page/read/write and participant head/open/delete/list counters, then encoded fixed formulas across the required scale tiers.
- Added fresh-child RSS collection with Linux resource and `/proc` evidence, explicit byte units, timeout/error rejection, and structural growth checks that remain separate from timing.
- Added strict structural evidence records carrying raw call and RSS facts while retaining `performance: NOT_QUALIFIED`; direct conditional uploads remain capped and non-streaming.

## Task Commits

1. **Task 1: Encode per-operation authority and participant call formulas** - `69b771d` (test RED), `d658605` (feat GREEN)
2. **Task 2: Prove bounded peak-memory behavior separately from timing** - `525e0c7` (test RED), `4481a63` (feat GREEN), `681b215` (portability fix)

## Files Created/Modified

- `tools/run_phase8_scale_gates.py` - Counting wrappers, formula gates, fresh-child peak-RSS probes, and structural-evidence writer.
- `tests/performance/test_complexity_contracts.py` - Fixed-tier formula and wrapper-transparency contracts.
- `tests/performance/test_memory_bounds.py` - Isolated RSS, malformed-child, peak-growth, and direct-upload-bound contracts.
- `tools/phase8_evidence.py` - Strict bounded schema for structural evidence observations.

## Decisions Made

- Kept call classes independent so a single aggregate metric cannot obscure participant reads or authority N+1 scans.
- Required byte-normalized child observations and explicit Linux `/proc` evidence; unsupported POSIX RSS support produces no structural claim.
- Treated peak-memory proof as structural qualification only; measured timing remains the later controlled-performance evidence class.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Made the spawned memory child import the runner through its normal module path.**

- **Found during:** Task 2
- **Issue:** A dynamically loaded test module could not be imported by Python's spawn child, producing a missing result rather than a measurement.
- **Fix:** Loaded the runner from the tools path as an importable module before starting the child.
- **Files modified:** `tests/performance/test_memory_bounds.py`, `tests/performance/test_complexity_contracts.py`
- **Verification:** All child-process fixed-tier tests pass.
- **Committed in:** `4481a63`

**2. [Rule 2 - Missing critical functionality] Added strict structural-evidence validation.**

- **Found during:** Task 2
- **Issue:** The shared Phase 8 envelope accepted no raw structural observation schema, so required count/RSS facts would be dropped or rejected.
- **Fix:** Added finite allow-lists and numeric bounds for environment, call classes, scale/page facts, workload bytes, and peak RSS.
- **Files modified:** `tools/phase8_evidence.py`, `tools/run_phase8_scale_gates.py`, `tests/performance/test_memory_bounds.py`
- **Verification:** Structural envelope and existing release-tracer tests pass.
- **Committed in:** `4481a63`

**3. [Rule 1 - Bug] Failed closed where POSIX resource accounting is unavailable.**

- **Found during:** Task 2 final portability review
- **Issue:** Importing the POSIX-only `resource` module would prevent an unsupported host from reporting unavailable RSS evidence cleanly.
- **Fix:** Guarded the import and raised a typed probe error before any byte claim.
- **Files modified:** `tools/run_phase8_scale_gates.py`, `tests/performance/test_memory_bounds.py`
- **Verification:** Focused structural/memory and release-tracer suites pass.
- **Committed in:** `681b215`

---

**Total deviations:** 3 auto-fixed (2 Rule 1, 1 Rule 2).
**Impact on plan:** The fixes make the test/evidence boundary trustworthy without changing production caching, payload publication, lifecycle authority, topology guarantees, or timeout semantics.

## Known Stubs

None.

## Threat Flags

None. The plan adds only test and evidence-tooling surfaces; it introduces no production endpoint, authentication path, file-access boundary, or schema change.

## Verification

- `uv run pytest -q -o log_cli=false tests/performance/test_memory_bounds.py tests/performance/test_complexity_contracts.py -x` — 52 passed.
- `uv run pytest -q -o log_cli=false tests/test_phase8_release_tracer.py -x` — 16 passed.
- `uv run ruff check tools/run_phase8_scale_gates.py tools/phase8_evidence.py tests/performance/test_memory_bounds.py tests/performance/test_complexity_contracts.py` — passed.

## User Setup Required

None - no external service configuration is required. Linux-specific measured RSS remains an evidence fact; it is not a runtime deadline, cross-topology promise, or S3 streaming claim.

## Next Phase Readiness

Later controlled-performance work can consume strict structural facts while separately measuring timing distributions. No scale result authorizes new production caching, coordination, lifecycle state, or stronger progress guarantees.

## Self-Check: PASSED

- `tools/run_phase8_scale_gates.py`, both performance test files, and all five task commits exist in repository history.
- No tracked file deletion occurred in any task commit.

---
*Phase: 08-production-gates-and-performance-stabilization*
*Completed: 2026-09-14*
