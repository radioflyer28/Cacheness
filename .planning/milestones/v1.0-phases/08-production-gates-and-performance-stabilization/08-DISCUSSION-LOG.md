# Phase 8: Production Gates and Performance Stabilization - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-13
**Phase:** 08-production-gates-and-performance-stabilization
**Areas discussed:** Phase boundary adjustment, Release matrix, Live-service qualification, Quality gates, Performance and scale

---

## Phase boundary adjustment

| Option | Description | Selected |
|--------|-------------|----------|
| Insert Phase 07.1 before Phase 8 | Replace built-in payload mechanics first, then qualify the final architecture. | ✓ |
| Include adoption inside Phase 8 | Mix architecture replacement with release qualification and performance baselining. | |
| Defer beyond V1 | Qualify custom payload mechanics that may later be removed. | |

**User's choice:** Insert Phase 07.1 before Phase 8.
**Notes:** Phase 07.1 subsequently completed and verified. It delivered one obstore participant for filesystem, memory, and S3 while retaining the lifecycle authority and path-based handler seam. Phase 8 therefore qualifies that shipped architecture.

---

## Release matrix

### Supported Python versions

| Option | Description | Selected |
|--------|-------------|----------|
| Every supported minor | Gate every stable Python minor from 3.11 through the latest compatible stable release. | ✓ |
| Oldest and newest only | Use boundary versions as substitutes for intermediate stable minors. | |
| Narrow declared range | Reduce the advertised Python range. | |

**User's choice:** Every supported stable minor; pre-release Python is advisory.

### Operating systems

| Option | Description | Selected |
|--------|-------------|----------|
| Linux full, macOS boundary smoke | Full supported-Python Linux matrix plus oldest/newest macOS smoke. | ✓ |
| Linux and macOS full | Duplicate the complete version matrix across both operating systems. | |
| Linux only | Make no macOS release claim. | |

**User's choice:** Linux full matrix and macOS boundary smoke.
**Notes:** Windows remains explicitly unqualified until Phase 999.1 because no eligible Windows environment is available.

### Base installation and optional groups

| Option | Description | Selected |
|--------|-------------|----------|
| Objects plus NumPy | Keep NumPy core and prove retained generic-object and native NumPy formats. | ✓ |
| Generic objects only | Move NumPy entirely behind an extra. | |
| Broad batteries-included base | Put dataframe and other optional ecosystems into the base package. | |

**User's choice:** Keep NumPy core. Arrays use retained NPZ/Blosc2 behavior as applicable; dataframe extras use retained Parquet handlers.

| Option | Description | Selected |
|--------|-------------|----------|
| Install plus feature round trip | Clean install, public import, and one representative retained behavior per advertised extra. | ✓ |
| Install and import only | Do not prove that the installed feature operates. | |
| Consolidate extras first | Redesign dependency groups before qualification. | |

**User's choice:** Install plus representative feature round trip.
**Notes:** These handlers already exist; Phase 8 qualifies rather than invents them.

---

## Live-service qualification

### Execution cadence

| Option | Description | Selected |
|--------|-------------|----------|
| Protected release plus scheduled runs | Deterministic PR contracts; real services on release candidates and a drift schedule. | ✓ |
| Every trusted PR and release | Spend live-service resources on every eligible change. | |
| Manual release qualification only | Depend on an operator to launch every live run. | |

**User's choice:** Protected release-candidate and scheduled real-service runs.

### Evidence binding

| Option | Description | Selected |
|--------|-------------|----------|
| Exact release commit | Relevant source/test/tool/contract changes invalidate earlier evidence. | ✓ |
| Relevant-tree digest | Qualify a derived subset identity instead of the release revision. | |
| Release branch result | Allow later branch changes to inherit a prior result. | |

**User's choice:** Bind evidence to the exact release commit.

### Unavailable services

| Option | Description | Selected |
|--------|-------------|----------|
| Block the release | Advertised remote support requires non-substitutable clean evidence. | ✓ |
| Ship with topology unqualified | Release while retaining the remote support claim. | |
| Remove remote support from V1 | Narrow the V1 product topology. | |

**User's choice:** Block the release.

### Evidence retention

| Option | Description | Selected |
|--------|-------------|----------|
| Release record plus bounded diagnostics | Keep successful release evidence for the release lifetime and failures for a bounded window. | ✓ |
| Retain every run indefinitely | Preserve all scheduled and failed artifacts forever. | |
| Commit evidence into the repository | Store environment-generated run artifacts in source history. | |

**User's choice:** Release record plus bounded diagnostics.

---

## Quality gates

### Coverage

| Option | Description | Selected |
|--------|-------------|----------|
| Fixed high percentages | Choose statement and branch targets before measuring current risk and baseline. | |
| Measured-baseline ratchet | Measure, close named meaningful gaps, and hold or improve both critical and repository coverage. | ✓ |
| Critical modules only | Permit total repository coverage to regress. | |

**User's choice:** A critical-module gate plus a repository-wide statement/branch ratchet, derived from measured baselines rather than arbitrary upfront percentages.
**Notes:** Contract coverage remains necessary; a percentage is not a substitute.

### Ruff and formatting

| Option | Description | Selected |
|--------|-------------|----------|
| Direct scoped clean gates | Gate changed files and complete lifecycle/cache-policy/qualification/packaging scopes. | ✓ |
| Clean the entire repository now | Expand Phase 8 into unrelated legacy lint cleanup. | |
| Ruff remains advisory | Establish no release-blocking code-quality scope. | |

**User's choice:** Direct scoped clean lint and formatting gates.

| Option | Description | Selected |
|--------|-------------|----------|
| Fingerprint retained findings | Maintain a custom file/location lint-debt ledger. | |
| No custom fingerprint | Directly gate changed and critical scopes; leave untouched unrelated legacy scope advisory. | ✓ |

**User's choice:** No lint fingerprinting.
**Notes:** Linting and formatting do not justify custom lifecycle-like state machinery.

---

## Performance and scale

### Representative workloads and hash evidence

| Option | Description | Selected |
|--------|-------------|----------|
| Representative format tiers | Small objects, medium/large NumPy native formats, and Parquet dataframes; separate layer costs. | ✓ |
| Every built-in handler everywhere | Build a handler-by-topology Cartesian benchmark matrix. | |
| Storage bytes only | Omit handler and cache-policy costs. | |

**User's choice:** Representative format tiers with serialization, BlobStore lifecycle, and cache-policy overhead measured separately.
**Notes:** Include SHA-256 versus XXH3 raw throughput and end-to-end share. The result informs SEED-003 but does not change the current digest.

### Regression policy

| Option | Description | Selected |
|--------|-------------|----------|
| Distribution-based relative envelope | Compare medians and tails on a controlled runner with reviewed relative bounds. | ✓ |
| Fixed absolute limits | Treat machine-specific numbers as universal budgets. | |
| Advisory measurements initially | Do not establish a local release-blocking performance gate. | |

**User's choice:** Distribution-based relative envelopes on a named controlled Linux runner.

### Structural scale guarantees

| Option | Description | Selected |
|--------|-------------|----------|
| Structural complexity budgets | State page/work and call-count formulas, then test fixed scale tiers. | ✓ |
| Fixed absolute caps | Use machine-dependent latency or memory numbers alone. | |
| Empirical trend only | Record results without executable bounds. | |

**User's choice:** Structural complexity and backend-call budgets, independently of machine speed.

### Remote performance

| Option | Description | Selected |
|--------|-------------|----------|
| Correctness blocks; latency is diagnostic | Live integrity/recovery qualifies support while remote latency is recorded. | ✓ |
| Remote latency also blocks | Make internet/service latency part of the release correctness gate. | |
| Only remote performance matters | Omit controlled local regression budgets. | |

**User's choice:** Live correctness blocks; remote latency remains diagnostic.

---

## the agent's Discretion

- Exact CI job decomposition and clean-environment tooling.
- Bounded diagnostic-artifact retention duration.
- Representative fixture sizes, sample counts, statistical method, and controlled Linux runner identity.
- Exact coverage thresholds after baseline measurement and the named meaningful gaps to close.

## Deferred Ideas

- Investigate Narwhals as a future dataframe-handler compatibility layer across pandas, PyArrow, and Polars while retaining Parquet formats.
- Revisit a versioned XXH3 canonical payload digest through SEED-003 after Phase 8 benchmarking.
- Native Windows lifecycle qualification remains Phase 999.1.

---

## 2026-09-15 local-readiness scope cutover

The user asked whether Plans 08-11 and 08-12 were necessary before beginning to
use Cacheness locally. The agreed answer was no: those plans prove real
PostgreSQL/Amazon-S3 and immutable GitHub release publication, not the already
implemented deterministic local storage/cache boundary.

**User's choice:** Finish Phase 8 on verified local readiness now. Preserve the
existing live qualification and publication tooling, but supersede Plans 08-11
and 08-12 for this milestone and move their exact evidence standard to SEED-007.

**Locked nonclaims:** BACK-05 remains `DEFERRED`/`NOT_QUALIFIED`; immutable
publication remains `DEFERRED`/`NOT_PUBLISHED`; QUAL-06 remains deferred under
SEED-006; Windows remains unqualified. No mock, emulator, local run,
configuration-only preflight, or stale artifact may substitute for those future
evidence classes. This choice is recorded as D-24 in `08-CONTEXT.md`.
