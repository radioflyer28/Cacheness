---
phase: 08-production-gates-and-performance-stabilization
plan: 02
subsystem: packaging-qualification
tags: [wheel, uv, isolated-environment, public-api, evidence]
requires:
  - phase: 08-production-gates-and-performance-stabilization
    provides: Exact-commit deterministic evidence envelopes with fail-closed parsing
provides:
  - One hashed wheel and source-free base public-surface qualification
  - Literal isolated optional-group matrix with representative public round trips
  - Canonical packaging evidence that preserves non-live service boundaries
affects: [phase-08-release-verifier, platform-gates, quality-workflow]
actuals:
  tokens: 10155
  tasks: 2
  commits: 6
tech-stack:
  added: []
  patterns:
    - Literal public export and optional-extra inventories instead of adaptive discovery
    - One temporary isolated uv environment per wheel requirement
    - Class-scoped canonical evidence with bounded, sanitized payload fields
key-files:
  created:
    - tools/run_phase8_packaging.py
    - tests/packaging/test_wheel_matrix.py
  modified:
    - tools/phase8_evidence.py
key-decisions:
  - "Wheel probes enumerate the supported public barrels literally and reject checkout imports instead of adapting to a wheel's current exports."
  - "Each advertised extra gets an independent wheel requirement; S3, PostgreSQL, and cloud probes remain memory-backed and explicitly non-live."
  - "TensorFlow evidence is UNAVAILABLE outside the reviewed stable-minor set, never a skip-based packaging pass."
patterns-established:
  - "Packaging evidence validates exact optional-group order, compatibility labels, and service non-claims before release tooling consumes it."
  - "Qualification commands bind both wheel SHA-256 and reviewed Git-source identity."
requirements-completed: [QUAL-01, QUAL-02, QUAL-03]
coverage:
  - id: D1
    description: One isolated built wheel proves every literal base public export and public generic/NumPy/cache round trips.
    requirement: QUAL-01
    verification:
      - kind: integration
        ref: uv run pytest -q -o log_cli=false tests/packaging/test_wheel_matrix.py -k base -x
        status: pass
    human_judgment: false
  - id: D2
    description: Every reviewed optional group is installed from the same wheel in a fresh environment with format-specific or non-live public behavior.
    requirement: QUAL-02
    verification:
      - kind: integration
        ref: uv run pytest -q -o log_cli=false tests/packaging/test_wheel_matrix.py -x
        status: pass
    human_judgment: false
  - id: D3
    description: Packaging evidence is canonical, exact-commit, sanitized, and unable to turn unavailable TensorFlow or live services into a release claim.
    requirement: QUAL-03
    verification:
      - kind: e2e
        ref: uv run python tools/run_phase8_packaging.py --output /private/tmp/phase8-packaging-evidence.json
        status: pass
    human_judgment: false
duration: 17m
completed: 2026-09-13
status: complete
---

# Phase 08 Plan 02: Isolated Wheel Qualification Summary

**A single hashed Cacheness wheel now proves its literal public surface and each advertised extra in separate source-free environments, with canonical evidence that does not overclaim live services or incompatible TensorFlow.**

## Performance

- **Duration:** 17m
- **Started:** 2026-09-13T23:23:05Z
- **Completed:** 2026-09-13T23:39:53Z
- **Tasks:** 2/2
- **Files modified:** 3

## Accomplishments

- Added a one-wheel `uv --isolated --no-project` runner that strips inherited Python path state, rejects source imports, hashes the artifact, and exercises literal public barrels through public `BlobStore` and `UnifiedCache` composition.
- Added independent base, recommended/Blosc2, dataframe/Parquet, TensorFlow-compatible, S3, PostgreSQL, and cloud group probes; the service-labelled groups stay memory-backed and report no live-service qualification.
- Extended the Phase 8 evidence boundary only for the exact bounded packaging payload, including wheel SHA-256, source identity, probe labels, compatibility state, and explicit non-live group order.

## Task Commits

1. **Task 1: Prove the base wheel public surface and required formats** - `c3d30b6` (test), `745145d` (feat)
2. **Task 2: Qualify every advertised optional group independently** - `d270442` (test), `96f0633` (test), `37f1037` (feat), `c09c186` (fix)

## Files Created/Modified

- `tools/run_phase8_packaging.py` - Builds and hashes one wheel, runs source-free base/extra probes, and writes class-scoped packaging evidence.
- `tests/packaging/test_wheel_matrix.py` - TDD contracts and real isolated wheel-matrix integration coverage.
- `tools/phase8_evidence.py` - Exact allow-list validation for the sanctioned packaging evidence producer.

## Decisions Made

- Public exports are deliberately hardcoded in the probe rather than derived from an installed wheel, so an accidental public-surface contraction fails qualification.
- S3, PostgreSQL, and cloud extras prove installation plus public memory-backed composition only; real-service evidence remains a later dedicated class.
- TensorFlow runs only on reviewed stable minors. On this Python 3.13 host, qualification produces a canonical `UNAVAILABLE` packaging envelope rather than counting a skipped tensor probe as success.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Corrected the native NPZ metadata assertion**
- **Found during:** Task 1
- **Issue:** The first probe expected `payload_format` at the top level of public BlobStore metadata, but the actual public snapshot places its storage format in nested metadata.
- **Fix:** Assert the public `data_type` and nested `storage_format == "npz"` instead.
- **Files modified:** `tools/run_phase8_packaging.py`
- **Verification:** Base isolated wheel test passes.
- **Committed in:** `745145d`

**2. [Rule 2 - Missing Critical Functionality] Added the narrowly scoped packaging evidence schema**
- **Found during:** Task 2
- **Issue:** The Phase 8 evidence validator intentionally allowed only the deterministic producer to emit `PASS`, which blocked the plan's required sanitized packaging envelope.
- **Fix:** Added a packaging-only exact allow-list with wheel digest, reviewed groups, compatibility labels, bounded probe names, and non-live service declarations.
- **Files modified:** `tools/phase8_evidence.py`
- **Verification:** Packaging evidence contract and end-to-end envelope validation pass.
- **Committed in:** `37f1037`

**3. [Rule 1 - Bug] Preserved reviewed non-live group ordering**
- **Found during:** Task 2 end-to-end evidence verification
- **Issue:** Iterating an unordered set emitted a valid-looking but noncanonical service-group order, which the exact validator rejected.
- **Fix:** Replaced the set-derived output with the reviewed tuple order and added a regression test.
- **Files modified:** `tools/run_phase8_packaging.py`, `tests/packaging/test_wheel_matrix.py`
- **Verification:** Full matrix test, scoped Ruff, and canonical envelope validation pass.
- **Committed in:** `c09c186`

---

**Total deviations:** 3 auto-fixed (2 Rule 1, 1 Rule 2)
**Impact on plan:** All changes were necessary for truthful, canonical packaging qualification; no dependency, handler, storage lifecycle, or service architecture changed.

## Known Stubs

None.

## Issues Encountered

TensorFlow's handler remains deliberately disabled on the current Python 3.13 host because the project records a system-level mutex freeze. The runner records this as `UNAVAILABLE` outside reviewed stable minors, never as a skipped or passing probe.

## User Setup Required

None - local packaging qualification requires no external service configuration. Real PostgreSQL and Amazon S3 evidence remain explicitly separate.

## Next Phase Readiness

- Later Phase 8 release tooling can consume the exact packaging evidence class without treating it as live-service evidence.
- Platform qualification owns the compatible TensorFlow interpreter matrix; live PostgreSQL and AWS S3 remain dedicated evidence producers.

## Post-Completion Supersession (2026-09-14)

D-23 removes native TensorFlow support from the release. This summary remains an
accurate historical record of Plan 08-02's implementation and commits, but its
TensorFlow extra/probe/compatibility claims are no longer current release guidance.
Plan 08-13 removes that dependency group, lock graph, runtime surface, and evidence
slot before the remaining Phase 8 gates. The base, NumPy/Blosc2, dataframe/Parquet,
S3, PostgreSQL, and cloud qualification work remains applicable.

## Self-Check: PASSED

- Required runner, matrix tests, and evidence validator changes exist on disk.
- All six Task 1/Task 2 commits are present in Git history.

---
*Phase: 08-production-gates-and-performance-stabilization*
*Completed: 2026-09-13*
