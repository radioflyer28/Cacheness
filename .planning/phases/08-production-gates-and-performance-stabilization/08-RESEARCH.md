# Phase 08: Production Gates and Performance Stabilization - Research

**Researched:** 2026-09-13
**Domain:** Python package release qualification, live-service evidence, coverage ratchets, and controlled performance gates
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

### Phase boundary and architectural guardrails
- **D-01:** Phase 07.1 owns obstore adoption; Phase 8 qualifies and benchmarks the resulting final architecture. Do not reintroduce legacy filesystem/S3 implementations, a runtime selector, dual reads/writes, or a boto3 production fallback.
- **D-02:** `BlobStore` and `AuthorityLifecycleEngine` remain the sole lifecycle authority. Obstore remains a payload byte/object participant, and `UnifiedCache` remains a cache-policy layer over `BlobStore`.
- **D-03:** Apply ADR 0001 to every contention, fault, performance, and live-service result. Separate integrity, recovery, progress, and performance claims. A typed contention outcome or benchmark regression is not evidence that another lock, queue, lease, sidecar, or lifecycle source of truth is needed.
- **D-04:** Qualification must preserve the accepted topology-specific limits, including the absence of cross-resource ACID and the bounded Phase 7/07.1 orphan-reclamation limitation. Tests may not silently strengthen these contracts.

### Python, platform, and installation matrix
- **D-05:** Gate releases on every supported stable Python minor from 3.11 through the latest compatible stable release. Pre-release Python runs are advisory.
- **D-06:** Run the full supported-Python matrix on Linux. Run boundary smoke tests on macOS using the oldest and newest supported Python minors. Windows remains `UNAVAILABLE`/`NOT_QUALIFIED` until Phase 999.1 supplies eligible native evidence.
- **D-07:** NumPy remains a core dependency. A clean base wheel must prove guaranteed public imports, a generic-object round trip, and retained NumPy handler round trips using pickle and native NPZ/Blosc2 formats as applicable.
- **D-08:** Dataframe support remains optional and uses the retained Parquet handlers. Every advertised optional dependency group must install in a clean isolated environment, import its guaranteed public surface, and complete a representative existing handler or backend round trip. This is qualification of retained behavior, not a new format or handler redesign.

### Live PostgreSQL and Amazon S3 qualification
- **D-09:** Run deterministic backend and topology contracts on every pull request. Run real PostgreSQL and real Amazon S3 qualification on protected release candidates and on a schedule for service/API drift detection.
- **D-10:** Live evidence qualifies the exact release commit. Changes to relevant production code, tests, qualification tooling, or contract definitions invalidate earlier evidence for that release candidate.
- **D-11:** If either live service is unavailable or the fixed suite cannot produce clean qualifying evidence for the exact release commit, block the release. PostgreSQL/Amazon-S3 remains an advertised V1 topology and cannot inherit a pass from mocks, skips, an earlier commit, or a substitute service.
- **D-12:** Preserve the frozen Phase 5 live-service contract and fail-closed runner, adapting only what the completed obstore cutover makes necessary. Successful qualification must include sanitized evidence and exact bounded cleanup. Phase 07.1's explicit S3 constraints remain in force: explicit bucket and region, standard AWS credentials/IAM and bucket policy, no production endpoint override, and no `ExpectedBucketOwner` claim.
- **D-13:** Attach sanitized `QUALIFIED` evidence to the release record for its lifetime. Retain redacted failed, unavailable, and scheduled diagnostic artifacts for a bounded operational window; they are diagnostics, not substitute passes.

### Coverage, lint, and formatting gates
- **D-14:** Measure current statement and branch coverage before choosing thresholds. Close named meaningful gaps, then ratchet upward or hold the measured baseline; do not impose arbitrary percentages before measurement.
- **D-15:** Lifecycle and cache-policy modules receive their own statement/branch non-regression gate plus explicit named-gap coverage. Total repository statement and branch coverage also may not decrease. A percentage alone cannot substitute for executable lifecycle, integrity, recovery, and policy contracts.
- **D-16:** Changed Python files and the complete lifecycle, cache-policy, qualification, and packaging scopes must pass Ruff lint and formatting checks. Unrelated untouched legacy files may remain advisory.
- **D-17:** Do not build a custom lint-finding fingerprint or debt ledger. Enforce direct changed-file and critical-scope checks.

### Performance, hashing, memory, and scale
- **D-18:** Benchmark representative format tiers: small generic objects; medium and large NumPy arrays through NPZ and Blosc2 as applicable; and Parquet dataframes in the dataframe environment. Measure handler serialization separately from `BlobStore` lifecycle cost and `UnifiedCache` policy overhead. Do not create a handler-by-topology Cartesian matrix.
- **D-19:** Benchmark canonical SHA-256 against XXH3 across representative blob sizes. Report raw hashing throughput and each digest's share of end-to-end lifecycle time. These measurements inform SEED-003; Phase 8 does not change the persisted SHA-256-plus-size integrity contract.
- **D-20:** On a named controlled Linux runner, compare median and tail distributions with a checked-in baseline. Release blocking applies only to reviewed regressions outside a statistically defensible relative envelope. Baseline changes require explicit justification and must not become runtime deadlines or stronger public progress semantics.
- **D-21:** Declare expected page/work complexity and maximum backend-call formulas for inventory, reconciliation, statistics, clear, and aggregate operations. Exercise fixed scale tiers to detect accidental N+1 behavior and unbounded memory independently of machine speed.
- **D-22:** Controlled local performance gates may block release. Live PostgreSQL/Amazon-S3 qualification blocks on integrity and recovery behavior; remote latency distributions are diagnostic only.

### the agent's Discretion
- Select the exact clean-environment tooling, CI job decomposition, artifact retention duration, representative fixture sizes, sample counts, warmups, statistical envelope calculation, and controlled Linux runner identity, provided the choices are explicit and reproducible.
- Select named coverage gaps after measuring the current baseline and inspecting risk; prioritize lifecycle, integrity-before-deserialization, cache invalidation/policy, qualification, packaging, and error-boundary branches.
- Consolidate or replace stale benchmark scripts when necessary, while retaining relevant historical evidence and making the new release benchmark entry points unambiguous.

### Deferred Ideas (OUT OF SCOPE)
- Investigate Narwhals as a future dataframe-handler compatibility layer across pandas, PyArrow, and Polars while retaining Parquet as the handler-owned format. This is a future handler/extensibility milestone, not Phase 8 qualification work.
- SEED-003 may revisit a versioned XXH3 canonical payload digest after Phase 8 supplies comparative throughput and end-to-end cost evidence.
- Native Windows lifecycle qualification remains Phase 999.1 because no eligible Windows environment is available.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BACK-05 | PostgreSQL and AWS S3 behavior is verified with real-service integration coverage; compatible S3 services are supported only where explicitly verified. | Preserve and rebind the Phase 5 fail-closed runner to the Phase 07.1 obstore participant; run it on protected release candidates and schedule, with exact-revision and clean-cleanup evidence. [VERIFIED: `.planning/REQUIREMENTS.md:27`; `tools/run_phase5_qualification.py:27-55,291-368,395-491`] |
| QUAL-01 | A clean minimal wheel installation imports every guaranteed public symbol and completes a memory-backed round trip. | Extend the existing base-wheel probe to import the complete public barrels and exercise generic plus NumPy formats in a genuinely isolated environment. [VERIFIED: `.planning/REQUIREMENTS.md:64`; `tests/test_full_suite_environment.py:122-174`] |
| QUAL-02 | Each advertised optional dependency group installs and imports independently. | Generate one wheel, create one fresh environment per literal optional group, and require an independent representative probe; do not use one all-extras environment as proof. [VERIFIED: `.planning/REQUIREMENTS.md:65`; `pyproject.toml:18-45`] |
| QUAL-03 | CI covers supported Python versions, backend contracts, lint policy, coverage, packaging, PostgreSQL, and AWS S3 integration. | Add deterministic PR, Linux Python matrix, macOS boundary smoke, protected/scheduled live, and controlled-performance workflows; the repository currently has no workflow files. [VERIFIED: `.planning/REQUIREMENTS.md:66`; repository file inventory, 2026-09-13] |
| QUAL-04 | Carry forward the finite Phase 3 integrity/recovery regressions and cover named commit boundaries for each new supported topology with deterministic fault/crash tests. Shared-worker fixtures initialize first; success/conflict/typed retryable outcomes are distinguished from corruption. No universal scheduling guarantee or automatic repeated race-fix loop is required. | Keep the completed Phase 07.1 fixed verifier and focused fault tests in the deterministic gate; do not translate typed contention into a new scheduler. [VERIFIED: `.planning/REQUIREMENTS.md:67`; `.planning/phases/07.1-obstore-payload-participant-unification/07.1-VERIFICATION.md:160-180`] |
| QUAL-05 | Lifecycle and cache-policy code meets targeted statement and branch coverage thresholds established by the project. | Check in the measured branch-aware baseline, close named gaps, then gate both repository total and an explicit critical-module aggregate against non-regression. [VERIFIED: `.planning/REQUIREMENTS.md:68`; local branch-coverage measurement, 2026-09-13] |
| QUAL-06 | Checked-in benchmarks establish final performance budgets and distributions for named workloads/environments after lifecycle behavior stabilizes. Benchmark thresholds do not become runtime deadlines or strengthen public progress/atomicity promises. | Replace the stale canonical entry point with controlled Linux pyperf distributions, layer-separated workloads, hashes, memory evidence, and reviewed relative envelopes. [VERIFIED: `.planning/REQUIREMENTS.md:69`; `benchmarks/lifecycle_authority_benchmark.py:1-9,43-70,562-571`] |
| QUAL-07 | Supported inventory and aggregate operations avoid unbounded memory use and accidental N+1 backend calls. | Carry forward the completed contracts and add explicit formula/scale assertions for inventory, reconciliation, statistics, clear, and aggregate/policy-maintenance operations. [VERIFIED: `.planning/REQUIREMENTS.md:70`; `src/cacheness/storage/reconciliation.py:176-205`; `src/cacheness/storage/lifecycle.py:933-1047`] |
</phase_requirements>

## Summary

Phase 8 should be planned as an evidence pipeline around the architecture that already exists. `BlobStore` and `AuthorityLifecycleEngine` remain the only lifecycle authority; `ObstoreGenerationIO` remains a mechanics-only payload participant; `UnifiedCache` observes that storage lifecycle and adds policy. The ADR explicitly distinguishes integrity, deterministic recovery, bounded/typed progress, measured performance, and the ACID boundary of a single transactional resource. Nothing in a benchmark, coverage report, or live-service run authorizes a new lock, queue, lease, retry loop, metadata mirror, or coordinator. [VERIFIED: `docs/adr/0001-topology-specific-storage-guarantees.md:30-68`; `.planning/phases/07.1-obstore-payload-participant-unification/07.1-VERIFICATION.md:64-79,127-152`]

The major planning work is therefore release orchestration: a branch-aware coverage ratchet, direct Ruff scope gates, clean-wheel feature probes, a supported-version/platform matrix, exact-commit live evidence, and a replacement performance suite on a controlled Linux runner. The current deterministic suite is green when the three live modules are excluded, but the measured baseline is only 75.29% statement and 58.82% branch coverage, there are no checked-in CI workflow files, all six TensorFlow behavior tests are unconditionally skipped, and the checked-in lifecycle benchmark rejects its own baseline because production SQLite schema version `9` does not match baseline version `1`. [VERIFIED: local test/coverage/benchmark probes, 2026-09-13; `tests/test_tensorflow_handler.py:7-24,36-39,56-59,94-97,122-125,170-173,188-191`; `src/cacheness/storage/sqlite_lifecycle_authority.py:90-97`; `benchmarks/lifecycle_authority_baseline.json:1-31`; `benchmarks/lifecycle_authority_benchmark.py:562-567`]

**Primary recommendation:** Build one fail-closed Phase 8 release manifest that references independent deterministic, packaging, coverage, controlled-performance, platform, and live-service evidence for the exact commit; keep the evidence classes and their blocking rules separate. [VERIFIED: `08-CONTEXT.md` D-03, D-09 through D-13, D-20 through D-22]

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Deterministic contract execution | CI / release automation | Python test suite | CI selects the fixed tests; production code remains unchanged. [VERIFIED: `tools/verify_phase071_contracts.py`; `.planning/phases/07.1-obstore-payload-participant-unification/07.1-VERIFICATION.md:160-180`] |
| Clean-wheel and optional-group qualification | Packaging boundary | Isolated Python process | The built wheel and its metadata are the subject; probes execute imports and representative round trips without the source environment. [VERIFIED: `tests/test_full_suite_environment.py:122-174`; `pyproject.toml:1-45`] |
| Coverage and lint ratchets | CI / quality tooling | Test suite | Coverage.py and Ruff evaluate existing code/tests; they must not become runtime dependencies. [VERIFIED: `pyproject.toml:68-73,116-146`] |
| Controlled performance and memory evidence | Benchmark harness | Named Linux runner | The harness defines workloads and layers; the runner supplies stable machine identity and noise controls. [CITED: https://pyperf.readthedocs.io/en/latest/] |
| PostgreSQL authority qualification | External PostgreSQL service | `PostgresqlLifecycleAuthority` via `BlobStore` | PostgreSQL remains the lifecycle authority while S3 remains outside its transaction. [VERIFIED: `docs/adr/0001-topology-specific-storage-guarantees.md:12-16,36-44`; `tests/integration/test_postgresql_authority.py`] |
| Amazon S3 payload qualification | External Amazon S3 | `ObstoreGenerationIO` via topology composition | S3 supplies payload mechanics and exact object observations; it does not select lifecycle visibility. [VERIFIED: `.planning/phases/07.1-obstore-payload-participant-unification/07.1-VERIFICATION.md:99-109,127-152`; `tests/integration/test_s3_generation.py`] |
| Exact-release qualification decision | Protected release workflow | Immutable release record | A pass is valid only for the exact revision and sanitized evidence; finite workflow artifacts alone cannot satisfy lifetime retention. [VERIFIED: `tools/run_phase5_qualification.py:291-368`; CITED: https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases] |
| Runtime cache semantics | `UnifiedCache` policy layer | `BlobStore` lifecycle | Statistics are storage-free derived outcomes; invalidation routes exact deletes through `BlobStore`. [VERIFIED: `src/cacheness/core.py:933-978,1026-1125`] |

## Project Constraints (from AGENTS.md)

- Preserve the pre-production cutover model: unsupported current layouts fail explicitly; future compatibility uses explicit versions and offline migration/rebuild tooling. [VERIFIED: `AGENTS.md` Project Constraints]
- Keep the ownership split verbatim: “`BlobStore` owns storage lifecycle; `UnifiedCache` depends on it and owns cache policy; `SqlCache` remains separate.” [VERIFIED: `AGENTS.md` Project Constraints]
- Keep application payloads trusted while enforcing safe parsing, path containment, fail-closed integrity, atomic/rollback/reconciliation semantics, and same-key non-corruption. [VERIFIED: `AGENTS.md` Project Constraints]
- Maintain Python `>=3.11` support and use `uv` plus `uv.lock`; verify supported minors instead of relying on the current development interpreter. [VERIFIED: `AGENTS.md` Technology Stack and Platform Requirements; `pyproject.toml:9`; `.python-version:1`]
- Use lowercase `snake_case.py`, `test_<subject>.py`, public barrel exports, focused interfaces/dataclasses, package-relative internal imports, narrow domain exceptions with `raise ... from e`, Google-style public docstrings, and contextual logging. [VERIFIED: `AGENTS.md` Conventions]
- Run Ruff on `src` and `tests`, target Python 3.11 and line length 88, avoid increasing the legacy baseline, and do not assume commented-out lint groups are active. [VERIFIED: `AGENTS.md` Code Style; `pyproject.toml:131-146`]
- Treat the architecture narrative embedded in `AGENTS.md` as historical where it conflicts with the completed Phase 07.1 source and verification; the actionable guardrail and ADR remain binding. [VERIFIED: `.planning/phases/07.1-obstore-payload-participant-unification/07.1-VERIFICATION.md:64-79`; `AGENTS.md` Storage Lifecycle Design Guardrail]

## Current Baselines and Blocking Gaps

### Coverage Baseline

The branch-aware deterministic measurement used the locked all-extras/dev environment, excluded only the three live Phase 8 modules, and completed successfully with nine platform/feature skips. Repository totals were 15,005 statements / 11,297 covered (75.29%) and 5,444 branches / 3,202 covered (58.82%); Coverage.py's combined display was 71%. [VERIFIED: `/tmp/cacheness-phase8-coverage.json` local measurement, 2026-09-13]

The following critical-scope aggregate is a research baseline, not the final threshold: 5,621 statements / 4,347 covered (77.33%) and 2,016 branches / 1,192 covered (59.13%). The planner should close named gaps first, rerun on the CI-controlled environment, and only then check in the gate baseline. [VERIFIED: `/tmp/cacheness-phase8-coverage.json` local measurement, 2026-09-13; `08-CONTEXT.md` D-14 and D-15]

**Mandatory ordering before baseline capture:** Plan 04 must first add deterministic, named `PostgresqlLifecycleAuthority` contracts for all four highest-risk families: DB-API error classification, exact operation/proof replay, bounded inventory/reconciliation pagination, and rollback of failed transactions. Only after those selectors pass may Plan 05 measure and capture repository/critical floors. The current authority maps SQLSTATE and driver classes at the boundary, executes each semantic transition inside `connection.transaction()`, reconstructs exact persisted mutations, and applies explicit page/work caps; the existing tests cover pieces of each family but not a Phase 8-owned exhaustive selector inventory. [VERIFIED: `src/cacheness/storage/backends/postgresql_lifecycle_authority.py:247-345,1577-1695`; `tests/contracts/test_postgresql_lifecycle_authority.py:188-230,682-779,850-927,1181-1226`]

Plan 04 therefore owns the test-first closure and Plan 05 owns the later measurement. The pre-gap percentages below are diagnostic context only: they are not acceptance floors and must not be copied into the checked baseline. [VERIFIED: `08-CONTEXT.md` D-14 and D-15]

| Critical module | Statement | Branch | Planning implication |
|---|---:|---:|---|
| `src/cacheness/cache_policy.py` | 85.05% | 58.57% | Add explicit validation, continuation, and immutable-result branch tests. [VERIFIED: local coverage JSON, 2026-09-13] |
| `src/cacheness/core.py` | 84.35% | 66.44% | Cover typed lookup outcomes, stale-cursor restart, and removal accounting. [VERIFIED: local coverage JSON, 2026-09-13] |
| `src/cacheness/storage/blob_store.py` | 88.13% | 71.43% | Prioritize close/ownership and typed composition failures, not happy-path inflation. [VERIFIED: local coverage JSON, 2026-09-13] |
| `src/cacheness/storage/lifecycle.py` | 75.16% | 62.16% | Prioritize ambiguous publication, exact cleanup, and bounded clear branches. [VERIFIED: local coverage JSON, 2026-09-13] |
| `src/cacheness/storage/lifecycle_authority.py` | 88.33% | 58.70% | Cover protocol validation and typed boundary branches. [VERIFIED: local coverage JSON, 2026-09-13] |
| `src/cacheness/storage/memory_lifecycle_authority.py` | 85.44% | 63.40% | Preserve parity for finite paging/debt/recovery branches. [VERIFIED: local coverage JSON, 2026-09-13] |
| `src/cacheness/storage/sqlite_lifecycle_authority.py` | 76.16% | 59.24% | Target schema/replay/CAS/error branches without raising runtime guarantees. [VERIFIED: local coverage JSON, 2026-09-13] |
| `src/cacheness/storage/backends/postgresql_lifecycle_authority.py` | 55.37% | 36.89% | Largest deterministic gap; exercise DB-API failure mapping, replay, page bounds, and transactional rollback before relying on live coverage. [VERIFIED: local coverage JSON, 2026-09-13] |
| `src/cacheness/storage/obstore_generation_io.py` | 80.06% | 69.30% | Cover exact conditional/head/delete outcomes and bounded inventory failures. [VERIFIED: local coverage JSON, 2026-09-13] |
| `src/cacheness/storage/reconciliation.py` | 90.03% | 80.00% | Hold the strong baseline and add scale/call formulas rather than percentage chasing. [VERIFIED: local coverage JSON, 2026-09-13] |
| `src/cacheness/storage/composition.py` | 88.89% | 73.53% | Cover invalid profiles, injected-provider ownership, and factory capability boundaries. [VERIFIED: local coverage JSON, 2026-09-13] |

Named gaps should be executable behaviors: integrity before handler deserialization, prepared/verified/promotion response loss, cleanup-debt recovery, stale/foreign continuation rejection, every cache invalidation mode, fail-closed live evidence, clean-wheel public barrels, every optional-group round trip, and PostgreSQL error classification. Percentage movement without those selectors does not satisfy D-15. [VERIFIED: `08-CONTEXT.md` D-14 and D-15; `.planning/phases/07.1-obstore-payload-participant-unification/07.1-VERIFICATION.md:81-87,207-223`]

### Packaging and Platform Gap

The wheel test currently builds once, probes base, `s3`, and `cloud`, and performs a memory round trip only for base. It does not independently prove `recommended`, `dataframes`, `tensorflow`, or `postgresql`, nor a representative round trip for every group. [VERIFIED: `tests/test_full_suite_environment.py:122-174`; `pyproject.toml:18-45`]

The literal advertised optional groups are: `recommended`, `dataframes`, `tensorflow`, `s3`, `postgresql`, and `cloud`. The relevant source quote is: `recommended = [...]`, `dataframes = [...]`, `tensorflow = ["tensorflow>=2.0.0"]`, `s3 = []`, `postgresql = ["psycopg[binary]>=3.1.0", "sqlalchemy>=2.0.0"]`, and `cloud = ["psycopg[binary]>=3.1.0", "sqlalchemy>=2.0.0"]`. [VERIFIED: `pyproject.toml:18-45`]

Python 3.11, 3.12, 3.13, and 3.14 are stable supported CPython branches as of the research date; 3.15 is prerelease. [CITED: https://devguide.python.org/versions/] The pinned core `obstore==0.11.1` provides CPython 3.11+ ABI wheels, including 3.14/free-threaded artifacts. [CITED: https://pypi.org/project/obstore/0.11.1/] TensorFlow 2.21.0 advertises and publishes wheels only through CPython 3.13, with no 3.14 wheel. [CITED: https://pypi.org/project/tensorflow/] Therefore, define a feature-aware matrix: run core plus all installable non-TensorFlow groups on Linux 3.11-3.14, run the TensorFlow probe on Linux 3.11-3.13, and record TensorFlow-on-3.14 as dependency-incompatible rather than silently skipping it. This preserves base 3.14 support without pretending the TensorFlow extra is qualified there. [ASSUMED]

The six existing TensorFlow behavior tests set `SKIP_TENSORFLOW_TESTS = True` with the exact reason `"TensorFlow causes system freezes with mutex lock issues"`; they cannot be the QUAL-02 proof. Use a short subprocess smoke on a supported Linux runner with a hard harness timeout and a representative `BlobStore` round trip, then either remove the unconditional skip or make the broader handler suite platform-capability driven. [VERIFIED: `tests/test_tensorflow_handler.py:7-24,36-39,56-59,94-97,122-125,170-173,188-191`]

### Performance Baseline Gap

The checked benchmark records schema version `3`, five repetitions, one warmup, clear sizes `(16, 64)`, 16 reconciliation items, eight contention workers, and custom percentile math. The exact source values are: `BENCHMARK_SCHEMA_VERSION = 3`, `REPETITIONS = 5`, `WARMUPS = 1`, `CLEAR_TARGET_COUNTS = (16, 64)`, `RECONCILIATION_ITEMS = 16`, and `CONTENTION_WORKERS = 8`. [VERIFIED: `benchmarks/lifecycle_authority_benchmark.py:43-57,77-88`]

That harness is no longer executable end-to-end: it constructs `BlobStore(..., backend="json")`, a removed pre-Phase-07.1 API, and validates the baseline against current SQLite authority identity. [VERIFIED: `benchmarks/lifecycle_authority_benchmark.py:257-290,562-567`] The baseline records `"user_version": 1`, while production defines `SQLITE_USER_VERSION = 9` and `SCHEMA_VERSION = SQLITE_USER_VERSION`; the local `--verify-baseline` probe failed with `baseline authority schema does not match production`. [VERIFIED: `benchmarks/lifecycle_authority_baseline.json:1-31`; `src/cacheness/storage/sqlite_lifecycle_authority.py:90-97`; local benchmark probe, 2026-09-13]

Keep the old JSON as explicitly labeled historical evidence, but make a new Phase 8 benchmark suite and baseline the sole release entry point. Do not mutate the old evidence into a misleading current pass. [ASSUMED]

## Standard Stack

### Core

| Library / service | Version | Purpose | Why standard here |
|---|---|---|---|
| CPython | Stable 3.11-3.14; 3.15 advisory | Supported-runtime matrix | The project declares `requires-python = ">=3.11"`; the PSF currently classifies 3.11-3.14 as supported stable and 3.15 as prerelease. [VERIFIED: `pyproject.toml:9`; CITED: https://devguide.python.org/versions/] |
| uv | Pin the workflow bootstrap; local probe 0.12.12 | Locked environment, wheel build, clean probes | The repository uses `uv.lock`; uv documents wheel builds and isolated project execution. [VERIFIED: `uv.lock`; local environment probe, 2026-09-13; CITED: https://docs.astral.sh/uv/concepts/projects/build/] |
| pytest | 8.4.1 in local project environment | Deterministic, package, and live contracts | Existing marker and strict-marker configuration already separates live evidence. [VERIFIED: local package metadata, 2026-09-13; `pyproject.toml:82-102`] |
| pytest-cov / Coverage.py | 6.2.1 / 7.10.3 locally | Statement, branch, JSON/XML evidence | Coverage.py supports combined data and multiple report forms; branch mode is not yet enabled in repository config. [VERIFIED: local package metadata, 2026-09-13; `pyproject.toml:116-129`; CITED: https://coverage.readthedocs.io/en/latest/commands/cmd_reporting.html] |
| Ruff | 0.12.9 locally | Direct lint and format checks | Ruff has separate lint and formatter check modes; repository policy is Python 3.11 / line length 88. [VERIFIED: local package metadata, 2026-09-13; `pyproject.toml:131-146`; CITED: https://docs.astral.sh/ruff/linter/; CITED: https://docs.astral.sh/ruff/formatter/] |
| GitHub Actions | Hosted Linux/macOS plus protected environments; controlled self-hosted performance runner | CI and release evidence orchestration | GitHub recommends explicit `setup-python` versions; workflow dispatch supports typed inputs and protected environment jobs can own live secrets. [CITED: https://docs.github.com/en/actions/tutorials/build-and-test-code/python; CITED: https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow] |

### Supporting

| Library | Version | Purpose | When to Use |
|---|---|---|---|
| `pyperf` [WARNING: flagged as suspicious — verify before using.] | 2.10.0, released 2026-02-07 | Calibrated worker-based distributions, machine metadata, comparison significance, JSON, and memory modes | Use only after the required human package checkpoint, for controlled local performance evidence. [CITED: https://pyperf.readthedocs.io/en/latest/; CITED: https://pypi.org/project/pyperf/] |
| GitHub workflow artifacts | Service feature | Short-lived redacted diagnostics | Use `retention-days: 30` for failed, unavailable, and scheduled diagnostics. The service defaults to 90 days and permits 1-90 days publicly or 1-400 days privately. [CITED: https://docs.github.com/en/organizations/managing-organization-settings/configuring-the-retention-period-for-github-actions-artifacts-and-logs-in-your-organization] |
| GitHub immutable release assets | Service feature | Lifetime exact-release evidence | Attach sanitized `QUALIFIED` JSON before publishing the release; immutable releases lock tag and assets and create an attestation. [CITED: https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `pyperf` | Repair the custom percentile runner | Avoids a package, but preserves hand-rolled calibration/statistics and makes D-20's “statistically defensible” requirement harder to review. [VERIFIED: `benchmarks/lifecycle_authority_benchmark.py:43-88`; CITED: https://pyperf.readthedocs.io/en/latest/] |
| Direct uv/pytest matrix | Add tox/nox | An extra environment orchestrator adds little because GitHub's matrix and uv already select Python and locked dependencies. [ASSUMED] |
| Release assets for qualifying evidence | Actions artifacts only | Actions artifacts have finite retention and cannot satisfy D-13's release-lifetime requirement. [CITED: https://docs.github.com/en/organizations/managing-organization-settings/configuring-the-retention-period-for-github-actions-artifacts-and-logs-in-your-organization; CITED: https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases] |

**Installation after human legitimacy checkpoint:**

```bash
uv add --group dev "pyperf==2.10.0"
```

[CITED: https://pypi.org/project/pyperf/]

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| `pyperf` | PyPI | Released 2026-02-07 | Unknown to legitimacy seam | `github.com/psf/pyperf` with PyPI Trusted Publishing attestation | SUS | Flagged — planner must add `checkpoint:human-verify` before install. [VERIFIED: package-legitimacy seam, 2026-09-13; CITED: https://pypi.org/project/pyperf/] |

**Packages removed due to [SLOP] verdict:** none. [VERIFIED: package-legitimacy seam, 2026-09-13]

**Packages flagged as suspicious [SUS]:** `pyperf`; the seam could not resolve age/download/repository signals in this sandbox even though official PyPI provenance identifies the PSF source and Trusted Publishing. The protocol still requires a human checkpoint. [VERIFIED: package-legitimacy seam, 2026-09-13; CITED: https://pypi.org/project/pyperf/]

## Architecture Patterns

### System Architecture Diagram

```text
                         pull request / release candidate / schedule
                                           |
                                           v
                              Phase 8 workflow dispatcher
                                           |
              +----------------------------+--------------------------+
              |                            |                          |
              v                            v                          v
    deterministic evidence       packaging/platform evidence     live evidence
    - fixed contract suite       - build one wheel               - protected secrets
    - coverage JSON              - fresh env per extra           - real PostgreSQL
    - Ruff direct scopes         - Linux Python matrix           - real Amazon S3
    - call-count/scale tests      - macOS boundary smoke          - exact cleanup
              |                            |                          |
              +----------------------------+--------------------------+
                                           |
                                           v
                                exact-commit release manifest
                                           ^
                                           |
                   controlled Linux performance + memory evidence
                   - handler serialization
                   - BlobStore lifecycle
                   - UnifiedCache policy
                   - SHA-256 vs XXH3
                   - structural call formulas
                                           |
                         +-----------------+-----------------+
                         |                                   |
             short-lived diagnostic artifacts      immutable release asset
             (failed/unavailable/scheduled)         (sanitized QUALIFIED only)
```

This flow keeps mocks, local performance, remote latency, and live integrity evidence in separate classes. It also makes exact revision, environment, workload, and cleanup part of the release decision instead of treating “CI green” as one undifferentiated fact. [VERIFIED: `08-CONTEXT.md` D-03, D-09 through D-13, D-18 through D-22]

### Recommended Project Structure

```text
.github/workflows/
├── quality.yml                 # deterministic PR + Linux/macOS Python matrix
├── live_qualification.yml      # protected RC dispatch + scheduled service drift
└── performance.yml             # controlled Linux runner only
benchmarks/
├── phase8_benchmarks.py        # canonical pyperf entry point
├── phase8_workloads.py         # deterministic fixture/layer definitions
├── phase8_baseline.json        # controlled-runner baseline and environment identity
└── historical/                 # retained, explicitly non-gating legacy evidence
tools/
├── run_phase8_local_gates.py   # one deterministic developer/CI entry point
├── run_phase8_qualification.py # adapted Phase 5 fail-closed runner
├── verify_phase8_coverage.py   # coverage JSON scope/ratchet check
└── verify_phase8_release.py    # exact-commit evidence aggregator
tests/
├── qualification/              # evidence schema/runner/redaction/cleanup contracts
├── packaging/                  # base and one-fresh-env-per-extra probes
├── performance/                # harness schema/formula/baseline self-tests
└── test_phase8_quality_gates.py
```

These are proposed paths, not claims about existing files. [ASSUMED] If the planner keeps existing test locations, preserve the same responsibility split and avoid duplicating selectors in workflow YAML. [ASSUMED]

### Pattern 1: Evidence-Class Separation

**What:** Give each evidence class its own command, schema, blocking rule, and artifact. Deterministic tests and packaging block PRs; controlled local performance blocks release; live remote integrity/recovery blocks release; remote latency and prerelease Python remain advisory. [VERIFIED: `08-CONTEXT.md` D-05, D-09 through D-13, D-20 through D-22]

**When to use:** Every workflow and release aggregation task.

**Implementation guidance:** Centralize selectors and evidence schemas in Python modules that workflow jobs call. YAML should orchestrate commands, not reimplement skip detection, revision validation, redaction, or threshold math. [ASSUMED]

### Pattern 2: Exact-Revision Evidence Manifest

**What:** Extend, do not weaken, the Phase 5 runner's allow-listed evidence. It already requires a 40-character Git revision, a sanitized run namespace, standard Amazon S3 identity, all live tests passing, no skips/deselection, and `CLEAN` cleanup before `QUALIFIED`. [VERIFIED: `tools/run_phase5_qualification.py:60-103,291-368,395-491`]

**When to use:** Protected release-candidate and scheduled live jobs.

**Implementation guidance:** Adapt the schema/run name to Phase 8, add the obstore version and the Phase 8 gate/contract sources to the safe allow-list and source fingerprint, retain boto3 only in test-runner credential/cleanup tooling, and keep production S3 calls through `ObstoreGenerationIO.for_s3`. [VERIFIED: `tools/run_phase5_qualification.py:47-55,215-235`; `tests/integration/test_s3_generation.py`; `tests/test_full_suite_environment.py:98-119`]

Create one aggregate release manifest that contains digests/references to deterministic, package, platform, coverage, performance, and live evidence. It must recompute relevant-source identity at release time; do not accept an artifact merely because its filename or workflow run is recent. The fixed-verifier implementation should copy the Phase 5 separation between fixed local contract proof and separately sanitized live proof, and the release test should copy the Phase 3 practice of inspecting exact Git/tag/artifact state rather than trusting prose. [VERIFIED: `tools/verify_phase5_contracts.py:1-10,59-88`; `tests/test_phase3_release_evidence.py:22-94`]

**Exact-SHA dispatch/collection contract:** the release operator supplies one explicit 40-character candidate SHA; the protected workflow checks out that input in detached state and proves `HEAD` equals it; the dispatcher records the resulting workflow run ID; it waits for that exact run to reach a successful terminal conclusion; and it downloads the named artifact using both run ID and artifact name. It must then validate the artifact envelope revision, relevant-source digest, terminal state, and cleanup status. A “latest run” or filename-only lookup is forbidden. `gh workflow run` supports dispatch inputs, `gh run view` exposes `databaseId`, `headSha`, status, and conclusion, `gh run watch <run-id> --exit-status` waits on the selected run, and `gh run download <run-id> -n <name>` selects a particular run artifact. [CITED: https://cli.github.com/manual/gh_workflow_run; CITED: https://cli.github.com/manual/gh_run_view; CITED: https://cli.github.com/manual/gh_run_watch; CITED: https://cli.github.com/manual/gh_run_download]

**Immutable publication contract:** create the release as a draft, attach the exact sanitized aggregate and required qualifying evidence, verify the tag resolves to the candidate SHA and that the release asset set is exact, verify every API asset has state `uploaded` and a matching `sha256:` digest, then publish the draft and run `gh release verify <tag>` plus `gh release verify-asset <tag> <path>`. The final verifier must require the published release to report immutable state and must reject missing, extra, starter, digest-mismatched, or diagnostic assets. This transition is externally authorized and cannot be faked by deterministic tests. [CITED: https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases; CITED: https://docs.github.com/en/rest/releases/assets; CITED: https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/secure-your-dependencies/verify-release-integrity]

### Pattern 3: Measure → Close Gaps → Ratchet

**What:** Enable branch coverage, save JSON/XML evidence, assert named behavior selectors, then compare repository-total and critical-scope statement/branch rates against the checked-in baseline. [VERIFIED: `08-CONTEXT.md` D-14 and D-15; CITED: https://coverage.readthedocs.io/en/latest/commands/cmd_reporting.html]

**When to use:** Deterministic Linux gate on the nominated coverage interpreter, after the all-supported-version test matrix.

**Implementation guidance:** Keep the critical-module list explicit and reviewable. A baseline update must contain before/after rates, newly closed selectors, and a reason; never auto-rewrite the baseline in a normal CI run. [ASSUMED]

### Pattern 4: Controlled Relative Performance Envelope

**What:** Use pyperf worker processes, calibration, warmups, raw JSON, machine metadata, stability checks, and statistical comparison. Compare medians and tails on one named runner; separately record maximum memory and backend-call formulas. [CITED: https://pyperf.readthedocs.io/en/latest/]

**When to use:** Phase 8 baseline capture and later release comparisons on the same controlled Linux identity.

**Implementation guidance:** Use 20 measured worker processes after pyperf calibration for release workloads, reject unstable runs before comparison, and block only when the comparison is statistically significant and exceeds a reviewed 20% relative slowdown envelope. Treat these exact defaults as initial research recommendations to validate on the controlled runner, not as correctness or runtime deadline constants. [ASSUMED]

### Pattern 5: Layer-Separated Workloads

**What:** For each representative format, measure three distinct boundaries: handler staging/serialization alone, public `BlobStore` put/get lifecycle, and `UnifiedCache` put/lookup policy. Measure SHA-256 and XXH3 both in raw buffers and as percentage of the matching end-to-end lifecycle. [VERIFIED: `08-CONTEXT.md` D-18 and D-19]

**When to use:** Performance suite only; do not expand to every topology.

**Recommended fixtures:** [ASSUMED]

| Tier | Fixture | Formats / boundary |
|---|---|---|
| Small | 4 KiB nested generic object | pickle handler, memory payload + memory authority |
| Medium array | 16 MiB deterministic contiguous NumPy array | native NPZ and Blosc2, then BlobStore and UnifiedCache |
| Large array | 128 MiB deterministic contiguous NumPy array | native NPZ and Blosc2, then local filesystem payload + SQLite authority |
| Dataframe | 100k rows with numeric/string/datetime columns | pandas and Polars retained Parquet handlers in dataframe environment |
| Hash sweep | 4 KiB, 1 MiB, 16 MiB, 128 MiB buffers | `hashlib.sha256` vs project XXH3 path; raw throughput and lifecycle share |

Keep generated fixtures deterministic and outside timed setup, include cold and warm read labels explicitly, and do not compare formats as if compression ratio, serialization cost, and lifecycle overhead were one metric. [ASSUMED]

### Pattern 6: Structural Complexity Contracts

**What:** Instrument the authority and payload participant at their existing interfaces and assert call ceilings as formulas of requested page size/work cap, returned candidates, and lifecycle actions. Timing is not a substitute for call counts or peak memory. [VERIFIED: `08-CONTEXT.md` D-21; `src/cacheness/storage/blob_store.py:676-749`; `src/cacheness/storage/reconciliation.py:176-205,321-408`; `src/cacheness/storage/lifecycle.py:933-1047`]

**When to use:** Deterministic scale tests at 10, 100, 1,000, and 10,000 authority entries, with page sizes below and above the default. The exact scale tiers are a research recommendation and should be reduced only if CI duration evidence requires it. [ASSUMED]

The contracts should distinguish operations:

| Operation | Complexity/call contract to encode |
|---|---|
| Catalog inventory/query | One authority `catalog_page` per public page request; at most `work_cap` authenticated descriptors examined and at most `limit` returned; no payload open/list calls. [VERIFIED: `src/cacheness/storage/blob_store.py:676-749`; `src/cacheness/storage/catalog.py:419-432,747-765`] |
| Reconciliation | One bounded authority work page plus at most one bounded payload inventory page per invocation; rows ≤ `operation_page_size`, applied actions ≤ `max_reconcile_actions`, record bytes ≤ `max_operation_record_bytes`; payload reads are forbidden for classification. [VERIFIED: `src/cacheness/storage/reconciliation.py:176-205,207-285,321-408`; `src/cacheness/config.py:336-387`] |
| Statistics | O(1) in stored entry count and zero storage calls; it snapshots the derived outcome recorder. [VERIFIED: `src/cacheness/core.py:933-978`] |
| UnifiedCache invalidation/clear page | One catalog page plus at most one exact `BlobStore.delete(expected=...)` per selected candidate; never materialize the full catalog. [VERIFIED: `src/cacheness/core.py:1040-1125`; `src/cacheness/cache_policy.py:300-320`] |
| BlobStore clear | One finite authority snapshot, bounded authority pages, at most `max_reconcile_actions` attempted targets per invocation, exact current-entry revalidation per target, and no payload listing. [VERIFIED: `src/cacheness/storage/lifecycle.py:933-1047`; `src/cacheness/config.py:336-387`] |
| Cache size aggregate/maintenance | One catalog page per caller-driven step, at most `catalog_page_size` returned / `maintenance_work_cap` examined, bounded signed continuation, and at most one exact eviction candidate per eviction step. [VERIFIED: `src/cacheness/core.py:492-643,669-680`; `src/cacheness/config.py:66-125`] |

For clear/reconciliation, assert separate authority-read, authority-write, participant-head/open/delete, and inventory-list counters. A single “backend calls” total can hide a new N+1 in one participant behind fewer calls in another. [ASSUMED]

### Anti-Patterns to Avoid

- **One omnibus green job:** It cannot show whether a pass came from deterministic mocks, live services, packaging, or controlled performance. Preserve evidence classes. [VERIFIED: `08-CONTEXT.md` D-09 through D-13, D-22]
- **All-extras environment as QUAL-02 proof:** Transitive dependencies can hide a broken individual group. Use one new environment per literal group. [VERIFIED: `pyproject.toml:18-45`; `tests/test_full_suite_environment.py:122-174`]
- **Repository-wide Ruff blocking before debt cleanup:** D-16 explicitly allows untouched legacy debt; block changed files plus the complete critical scopes and report the rest advisory. [VERIFIED: `08-CONTEXT.md` D-16 and D-17]
- **Auto-updating baselines:** This turns regressions into passes and erases review history. Require explicit recalibration with justification. [VERIFIED: `08-CONTEXT.md` D-20; `benchmarks/lifecycle_authority_benchmark.py:1-9`]
- **Converting benchmark envelopes into timeouts:** Performance evidence is not a progress guarantee. [VERIFIED: `docs/adr/0001-topology-specific-storage-guarantees.md:46-68`; `08-CONTEXT.md` D-20]
- **Calling boto3 in production qualification paths:** boto3 is permitted only for runner credential discovery and test resource cleanup; payload operations remain obstore. [VERIFIED: `tests/test_full_suite_environment.py:98-119`; `.planning/phases/07.1-obstore-payload-participant-unification/07.1-VERIFICATION.md:64-79`]
- **Using S3 listings/ETags as authority or canonical digest:** The authority owns membership; exact head evidence is corroboration; persisted SHA-256 plus size stays canonical. [VERIFIED: `.planning/phases/07.1-obstore-payload-participant-unification/07.1-VERIFICATION.md:143-152`; `08-CONTEXT.md` D-19]
- **Running secrets on untrusted pull requests:** Live jobs belong behind a protected environment and fixed trusted revision; PRs run only deterministic contracts. [CITED: https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Benchmark calibration and significance | More custom percentile/sample heuristics | `pyperf` after human legitimacy checkpoint | It supplies calibrated loops, multiple workers, stability diagnostics, metadata, JSON, percentiles, significance comparison, and memory modes. [CITED: https://pyperf.readthedocs.io/en/latest/] |
| Lint debt tracking | Finding hashes/fingerprints or bespoke debt ledger | Direct `ruff check` and `ruff format --check` on changed and critical scopes | D-17 forbids a custom fingerprint; direct scopes are reviewable and deterministic. [VERIFIED: `08-CONTEXT.md` D-16 and D-17; CITED: https://docs.astral.sh/ruff/linter/; CITED: https://docs.astral.sh/ruff/formatter/] |
| Live-service emulation | LocalStack, MinIO, Moto, skipped tests, or prior-run inheritance | Existing fixed PostgreSQL/Amazon-S3 runner and exact cleanup fixtures | The runner already rejects emulator-oriented source and incomplete/skip outcomes. [VERIFIED: `tools/run_phase5_qualification.py:95-103,395-491`] |
| Lifecycle coordination | Qualification-only locks, queues, leases, or retry coordinators | Existing `BlobStore` + `AuthorityLifecycleEngine` and typed outcomes | Another mechanism violates the single-authority stop condition and can create new correctness seams. [VERIFIED: `docs/adr/0001-topology-specific-storage-guarantees.md:18-44,125-151`] |
| Clean environment management | Ad hoc `venv` shell scripts with inherited site packages | Wheel build plus `uv run --isolated --no-project --with` | The repository already uses this pattern and uv documents isolated project execution/building. [VERIFIED: `tests/test_full_suite_environment.py:122-174`; CITED: https://docs.astral.sh/uv/concepts/projects/build/] |
| Lifetime qualifying evidence | “Keep Actions artifact forever” convention | Immutable release asset attached before publish | Actions retention is finite; immutable releases lock the tag and attached assets and add an attestation. [CITED: https://docs.github.com/en/organizations/managing-organization-settings/configuring-the-retention-period-for-github-actions-artifacts-and-logs-in-your-organization; CITED: https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases] |

**Key insight:** This phase should standardize evidence generation and decision rules, not storage behavior. Every custom mechanism added to production code increases the chance that qualification changes the system it was supposed to measure. [VERIFIED: `08-CONTEXT.md` Phase Boundary and D-01 through D-04]

## Common Pitfalls

### Pitfall 1: Treating Python 3.14 as an All-Extras Job

**What goes wrong:** Dependency resolution fails on the TensorFlow extra or the job silently excludes it, making the support claim ambiguous. [CITED: https://pypi.org/project/tensorflow/]

**Why it happens:** Project core metadata is open-ended at Python `>=3.11`, while current TensorFlow wheels stop at 3.13. [VERIFIED: `pyproject.toml:9,34-36`; CITED: https://pypi.org/project/tensorflow/]

**How to avoid:** Publish a feature-aware compatibility table, gate core/non-TensorFlow groups on 3.11-3.14, and gate TensorFlow on 3.11-3.13. A missing dependency wheel is not a test skip. [ASSUMED]

**Warning signs:** `uv sync --all-extras` fails only on 3.14, or a “full” matrix shows TensorFlow selectors as skipped. [ASSUMED]

### Pitfall 2: Reusing the Stale Lifecycle Baseline

**What goes wrong:** The gate fails before measurement or benchmarks a removed constructor rather than the released composition. [VERIFIED: `benchmarks/lifecycle_authority_benchmark.py:257-290,562-567`]

**Why it happens:** The baseline predates authority schema version 9 and the Phase 07.1 constructor boundary. [VERIFIED: `benchmarks/lifecycle_authority_baseline.json:1-31`; `src/cacheness/storage/sqlite_lifecycle_authority.py:90-97`]

**How to avoid:** Freeze old data under `historical/`, write one new canonical entry point against current public composition, self-test its schema, then capture on the controlled runner. [ASSUMED]

**Warning signs:** `backend="json"`, `user_version: 1`, macOS source environment, or release commands pointing at `lifecycle_authority_baseline.json`. [VERIFIED: `benchmarks/lifecycle_authority_benchmark.py:263-276`; `benchmarks/lifecycle_authority_baseline.json:1-31`]

### Pitfall 3: A Coverage Percentage with No Behavioral Contract

**What goes wrong:** Tests execute easy lines while integrity, recovery, continuation, and error branches remain uncovered. [VERIFIED: local critical-scope branch baseline, 2026-09-13]

**Why it happens:** Coverage.py measures execution, not whether the selected assertions prove the lifecycle promise. [CITED: https://coverage.readthedocs.io/en/latest/commands/cmd_reporting.html]

**How to avoid:** Require named selectors alongside total and critical-scope statement/branch ratchets. [VERIFIED: `08-CONTEXT.md` D-14 and D-15]

**Warning signs:** Aggregate coverage rises while PostgreSQL authority branch coverage remains 36.89%, or a PR deletes a fault selector without failing a manifest check. [VERIFIED: local coverage JSON, 2026-09-13]

### Pitfall 4: Assuming Scheduled Live Runs Are Guaranteed

**What goes wrong:** Service drift goes undetected because a scheduled job is delayed or dropped. [CITED: https://docs.github.com/en/actions/tutorials/manage-your-work/schedule-issue-creation]

**Why it happens:** GitHub schedules run from the default branch and can be delayed/dropped under high load. [CITED: https://docs.github.com/en/actions/tutorials/manage-your-work/schedule-issue-creation]

**How to avoid:** Schedule off the top of the hour, monitor freshness, retain bounded diagnostics, and always rerun manually for the exact release candidate. A schedule never substitutes for RC evidence. [ASSUMED]

**Warning signs:** Latest live artifact revision differs from the release SHA, or artifact age exceeds the declared drift interval. [VERIFIED: `08-CONTEXT.md` D-10 and D-11]

### Pitfall 5: Evidence Invalidated by New Gate Files

**What goes wrong:** Production/tests/qualification sources change but an older artifact still appears valid. [VERIFIED: `08-CONTEXT.md` D-10]

**Why it happens:** The current source fingerprint does not include future Phase 8 workflow, packaging, coverage, or benchmark contract files. [VERIFIED: `tools/run_phase5_qualification.py:47-55`]

**How to avoid:** Define one reviewed relevant-source inventory used by both live evidence and the aggregate release verifier; bind every artifact to the actual 40-character release commit and source digest. [ASSUMED]

**Warning signs:** A workflow/test edit leaves release evidence unchanged, or an artifact is accepted by filename alone. [ASSUMED]

### Pitfall 6: Remote Latency Becomes a Release Performance Gate

**What goes wrong:** Transient Internet/service variability blocks release or motivates stronger availability machinery. [VERIFIED: `08-CONTEXT.md` D-03 and D-22]

**Why it happens:** Latency, progress, integrity, and recovery are reported in one result. [VERIFIED: `docs/adr/0001-topology-specific-storage-guarantees.md:46-68`]

**How to avoid:** Live PostgreSQL/S3 blocks on behavior and cleanup only; publish remote timings as diagnostic distributions. Controlled local Linux performance is the only blocking timing gate. [VERIFIED: `08-CONTEXT.md` D-20 and D-22]

**Warning signs:** Live runner contains p95/p99 thresholds or changes a runtime timeout after a remote variance spike. [ASSUMED]

### Pitfall 7: Ruff Scope Drift

**What goes wrong:** Either all legacy findings suddenly block Phase 8, or critical unchanged lifecycle files escape checks because only the Git diff is linted. [VERIFIED: `AGENTS.md` Code Style; `08-CONTEXT.md` D-16]

**Why it happens:** Changed-file and critical-scope policies are different sets. [VERIFIED: `08-CONTEXT.md` D-16]

**How to avoid:** Compute changed Python paths from the merge base, union them with a fixed critical-scope list, then pass those paths directly to both Ruff commands. Report untouched files separately and advisory. Do not persist a finding ledger. [ASSUMED]

**Warning signs:** `ruff check .` is the only gate, or a custom JSON fingerprint file appears. [VERIFIED: `AGENTS.md` Code Style; `08-CONTEXT.md` D-17]

### Pitfall 8: Packaging Probe Reaches into Handler Internals

**What goes wrong:** A probe passes against a direct legacy handler path but fails through actual `BlobStore` publication/read guards. [VERIFIED: `tests/test_tensorflow_handler.py:60-92`; `.planning/phases/07.1-obstore-payload-participant-unification/07.1-VERIFICATION.md:99-109,143-152`]

**Why it happens:** Existing TensorFlow tests call `handler.put()`/`handler.get()` on paths directly, while Phase 07.1 moved payload mechanics under guarded obstore participation. [VERIFIED: `tests/test_tensorflow_handler.py:60-92`; `.planning/phases/07.1-obstore-payload-participant-unification/07.1-VERIFICATION.md:99-109`]

**How to avoid:** Optional-group proof should import the public surface and execute a public `BlobStore` or `UnifiedCache` round trip using the retained type/format, not only instantiate a handler. [ASSUMED]

**Warning signs:** Probe imports `cacheness.handlers` only or writes arbitrary file paths itself. [ASSUMED]

## Code Examples

Verified patterns from official sources and current project contracts:

### Supported-Python Matrix with Advisory Prerelease

The literal stable values are sourced from the PSF support table; 3.15 is prerelease. [CITED: https://devguide.python.org/versions/]

```yaml
strategy:
  fail-fast: false
  matrix:
    python: ["3.11", "3.12", "3.13", "3.14"]
steps:
  - uses: actions/checkout@v6
  - uses: actions/setup-python@v5
    with:
      python-version: ${{ matrix.python }}
  - run: uv run --isolated --python ${{ matrix.python }} --frozen pytest -q -o log_cli=false
```

[CITED: https://docs.github.com/en/actions/tutorials/build-and-test-code/python]

Put 3.15 in a separate advisory job with `continue-on-error: true`; do not make stable jobs advisory. [VERIFIED: `08-CONTEXT.md` D-05]

### Branch-Aware Coverage Evidence

```bash
uv run --isolated --all-extras --group dev --frozen pytest \
  -q -o log_cli=false \
  --cov=cacheness --cov-branch \
  --cov-report=json:build/coverage.json \
  --cov-report=xml:build/coverage.xml
uv run --isolated --group dev --frozen python tools/verify_phase8_coverage.py \
  --report build/coverage.json --baseline benchmarks/phase8_coverage_baseline.json
```

The output paths and verifier names above are proposed. [ASSUMED] Coverage.py supports multiple report types from one measured dataset. [CITED: https://coverage.readthedocs.io/en/latest/commands/cmd_reporting.html]

### Direct Ruff Scope Gate

```bash
uv run --isolated --group dev --frozen ruff check ${PHASE8_PYTHON_PATHS}
uv run --isolated --group dev --frozen ruff format --check ${PHASE8_PYTHON_PATHS}
```

`PHASE8_PYTHON_PATHS` must be a safely constructed argv list containing changed Python files plus fixed critical scopes; do not interpolate untrusted filenames into a shell string in implementation. [ASSUMED] Ruff documents separate lint and formatter check modes. [CITED: https://docs.astral.sh/ruff/linter/; CITED: https://docs.astral.sh/ruff/formatter/]

### pyperf Benchmark Entry Point

```python
import pyperf

runner = pyperf.Runner()
runner.bench_func("blobstore_put_small_generic", benchmark_blobstore_put_small_generic)
```

[CITED: https://pyperf.readthedocs.io/en/latest/]

The real implementation should create deterministic fixtures outside the timed callable and attach layer, format, size, Git revision, and controlled-runner identity as metadata. [ASSUMED]

### Fail-Closed Release Decision

```python
if evidence_revision != release_revision:
    raise ReleaseQualificationError("evidence revision does not match release revision")
if live_status != "QUALIFIED" or cleanup_status != "CLEAN":
    raise ReleaseQualificationError("live service evidence does not qualify release")
```

The discrete values `"QUALIFIED"` and `"CLEAN"` are quoted verbatim from `_STATUS_VALUES = frozenset({"QUALIFIED", "UNAVAILABLE", "NOT_QUALIFIED"})` and `_CLEANUP_VALUES = frozenset({"NOT_ATTEMPTED", "CLEAN", "RESIDUE", "ERROR"})`. [VERIFIED: `tools/run_phase5_qualification.py:75-85`] The proposed exception type is illustrative and therefore [ASSUMED].

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Backend-specific filesystem/memory/S3 payload mechanics | One `ObstoreGenerationIO` participant beneath the lifecycle authority | Phase 07.1 | Phase 8 must benchmark and qualify the shared participant, not resurrect a legacy comparison path. [VERIFIED: `.planning/phases/07.1-obstore-payload-participant-unification/07.1-VERIFICATION.md:64-79,99-109`] |
| boto3 production S3 dependency | obstore core dependency; boto3/Moto test tooling only | Phase 07.1 | Live runner may use boto3 for credential preflight/cleanup, but production payload calls must remain obstore. [VERIFIED: `pyproject.toml:10-16,37,68-73`; `tests/test_full_suite_environment.py:98-119`] |
| One local Python/macOS benchmark record | Controlled named Linux distributions with exact revision/environment and relative statistical envelope | Phase 8 target | Existing evidence is historical; capture a new baseline only after the harness and machine are fixed. [VERIFIED: `benchmarks/lifecycle_authority_baseline.json:15-31`; `08-CONTEXT.md` D-20] |
| Line-oriented/no-threshold coverage config | Statement plus branch, named gaps, total and critical-scope ratchets | Phase 8 target | Add `branch = true` and a checked baseline after the named gaps are closed. [VERIFIED: `pyproject.toml:116-129`; `08-CONTEXT.md` D-14 and D-15] |
| Selected base/S3/cloud install smoke | One built wheel and one fresh environment for every advertised optional group | Phase 8 target | QUAL-02 cannot pass through transitive all-extras availability. [VERIFIED: `tests/test_full_suite_environment.py:122-174`; `pyproject.toml:18-45`] |
| Finite Actions workflow artifact retention | Qualified JSON attached to an immutable release; diagnostics retained 30 days | Phase 8 target | Release evidence survives normal artifact expiry while failed/unavailable logs remain bounded. [CITED: https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases; CITED: https://docs.github.com/en/organizations/managing-organization-settings/configuring-the-retention-period-for-github-actions-artifacts-and-logs-in-your-organization] |

**Deprecated/outdated:**

- `benchmarks/lifecycle_authority_benchmark.py` as the release entry point: it calls removed constructor syntax and rejects the checked baseline. Retain it only as labeled historical evidence or extract still-valid workload concepts into the new suite. [VERIFIED: `benchmarks/lifecycle_authority_benchmark.py:257-290,562-567`; local benchmark probe, 2026-09-13]
- The existing all-TensorFlow skip as qualification evidence: six behavior tests are deliberately disabled, so a new Linux subprocess round trip is required. [VERIFIED: `tests/test_tensorflow_handler.py:7-24`]
- The Phase 5 artifact pathname/schema as a final release gate: reuse its validation logic, but the current default writes under Phase 5 and fingerprints only the Phase 5 source set. [VERIFIED: `tools/run_phase5_qualification.py:27-55`]

## Deterministic Work vs Environmental/Human Prerequisites

### Deterministic implementation work

1. Add Phase 8 gate definitions, fixed selectors, self-tests, branch coverage config, coverage verifier, package-matrix runner, benchmark harness/schema tests, and workflow files. [ASSUMED]
2. Adapt the existing qualification runner/evidence schema to Phase 8 and obstore service metadata while retaining redaction, source binding, no-skip, standard-AWS, and exact-cleanup behavior. [VERIFIED: `tools/run_phase5_qualification.py:27-103,291-491`]
3. Add independent wheel probes for the literal six optional groups and a complete base public import plus generic/NumPy round trip. [VERIFIED: `pyproject.toml:18-45`; `08-CONTEXT.md` D-07 and D-08]
4. Close named coverage gaps, run the deterministic suite, then record total and critical-scope statement/branch baselines. [VERIFIED: `08-CONTEXT.md` D-14 and D-15]
5. Add call-count and peak-memory contracts at fixed scale tiers without changing production lifecycle coordination. [VERIFIED: `08-CONTEXT.md` D-21; `docs/adr/0001-topology-specific-storage-guarantees.md:125-151`]
6. Replace the canonical benchmark entry point, preserve historical evidence, and document baseline recalibration review rules. [VERIFIED: `08-CONTEXT.md` the agent's Discretion; `benchmarks/lifecycle_authority_benchmark.py:1-9`]

### Environmental/human prerequisites

1. Approve `pyperf==2.10.0` after reviewing the official PSF repository/PyPI attestation; the automated legitimacy seam returned `SUS`. [VERIFIED: package-legitimacy seam, 2026-09-13; CITED: https://pypi.org/project/pyperf/]
2. Provision the physical machine behind the settled workflow label `cacheness-perf-linux-x64`, then record stable CPU, governor, filesystem, Python, uv, SQLite, workload-isolation, and runner-image facts in the baseline fingerprint. The current machine is macOS ARM64 and is not eligible for D-20; no controlled-runner evidence currently exists. [VERIFIED: `08-11-PLAN.md` Task 1 precondition; local environment probe, 2026-09-13; `08-CONTEXT.md` D-20]
3. Configure the protected GitHub release environment with real PostgreSQL and Amazon S3 secrets, IAM/bucket policy, explicit bucket/region, and no endpoint override. The owning cloud/repository administrator must run the Phase 8 runner preflight before dispatch; all four required qualification variables are currently unset locally, so the present state is `UNAVAILABLE`, not evidence. [VERIFIED: `tools/run_phase5_qualification.py:34-39,95-99`; local environment probe, 2026-09-13]
4. Enable immutable GitHub releases and give a named release operator permission to create a draft, upload assets, verify digests, and publish it. The checkpoint must inspect actual repository/organization settings and `gh auth status`; repository settings, reviewer identity, credentials, and approval are external facts and are not claimed here. [CITED: https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases]
5. Dispatch the protected workflow for the explicit candidate SHA, record its exact run ID, wait for that run, download the named artifact by run ID, then validate the envelope before aggregation. Run the live suite against both services only after all relevant code/test/tool/workflow changes; no local substitute can close BACK-05. [VERIFIED: `08-CONTEXT.md` D-10 through D-12; CITED: https://cli.github.com/manual/gh_run_download]
6. Capture and review the first controlled Linux performance baseline; statistical thresholds cannot be finalized from this macOS research run. [VERIFIED: `08-CONTEXT.md` D-14 and D-20]
7. Provide/confirm GitHub-hosted macOS capacity for Python 3.11 and 3.14 boundary smoke. Native Windows remains explicitly deferred. [VERIFIED: `08-CONTEXT.md` D-06]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A4 | Start with 20 pyperf worker processes and a statistically significant 20% relative slowdown envelope, then validate noise on the controlled runner. | Controlled Relative Performance Envelope | Actual variance may require a different sample count/envelope before baseline lock. |
| A5 | Use 4 KiB, 16 MiB, and 128 MiB object tiers and a 100k-row dataframe fixture. | Layer-Separated Workloads | Runtime or memory cost may exceed CI budgets or underrepresent real workloads. |
| A6 | Use 10/100/1,000/10,000 entry scale tiers for structural call/memory tests. | Structural Complexity Contracts | Seeding cost may be excessive; lower tiers might still prove formulas if instrumentation is exact. |
| A7 | Split gate code among the proposed workflow/tool/test paths. | Recommended Project Structure | Existing verifier conventions may favor fewer files; responsibilities must remain equivalent. |
| A9 | Add pyperf rather than extending the custom harness. | Standard Stack | Human legitimacy checkpoint may reject the dependency, requiring a reviewed in-repo statistics alternative. |
| A11 | Build the Ruff argv from the merge-base changed-file set union a fixed critical scope. | Ruff Scope Drift | Repository workflow conventions may provide a different safe changed-file source. |
| A12 | Count authority reads/writes and participant head/open/delete/list separately at fixed scale tiers. | Structural Complexity Contracts | Existing spy interfaces or seeding cost may require equivalent counters/tiers. |
| A14 | Optional-group qualification should use public `BlobStore`/`UnifiedCache` round trips rather than handler-internal file calls. | Packaging Pitfall | A public route may not expose every handler format selector without a small test-only fixture seam. |

The removed assumptions (TensorFlow matrix, 30-day diagnostic retention, exact controlled-runner label, GitHub Actions/immutable release system, centralized exact-commit aggregation, protected environment, and validation layout) are settled Phase 8 design choices in Plans 01-12. Their external availability is handled by the checkpoints below; it is not assumed.

## Open Questions — RESOLVED FOR PLANNING

No design question remains open. The five prior questions are resolved into deterministic implementation contracts plus external execution prerequisites. A prerequisite that is absent must remain `UNAVAILABLE` or block publication; it must never be converted into a passing claim.

1. **RESOLVED — controlled Linux runner identity**
   - **Settled design:** `.github/workflows/performance.yml` targets the exact logical label `cacheness-perf-linux-x64`; every baseline/evidence envelope carries the full physical fingerprint and rejects label, OS/architecture, CPU/governor, filesystem, Python/uv/SQLite, or source-SHA drift. [VERIFIED: `08-07-PLAN.md` Task 3; `08-11-PLAN.md` Task 1]
   - **External prerequisite/owner:** a repository/infrastructure maintainer must provision and register the physical runner. No machine identity is asserted by this research.
   - **Preflight/checkpoint:** before baseline capture, Plan 11 Task 1 verifies the exact label, Linux x86-64 identity, stable fingerprint/noise, clean worktree, and candidate SHA. If unavailable or unstable, QUAL-06 stays unqualified and there is no local/macOS fallback. [VERIFIED: `08-CONTEXT.md` D-20]

2. **RESOLVED — TensorFlow support messaging**
   - **Settled design:** advertise and gate base/core plus installable non-TensorFlow groups on stable Python 3.11-3.14; qualify the TensorFlow extra on Python 3.11-3.13; explicitly record TensorFlow-on-3.14 as dependency-incompatible and never silently skip it. [VERIFIED: `08-02-PLAN.md` and `08-03-PLAN.md`; CITED: https://devguide.python.org/versions/; CITED: https://pypi.org/project/tensorflow/]
   - **Owner:** Plans 02-03 own the compatibility manifest, fresh-environment wheel probes, platform workflow, and public support table.
   - **Preflight/checkpoint:** each matrix row builds/installs from the exact lock/source in a fresh environment and either passes its representative public round trip or records a nonqualifying incompatibility. No human decision is needed unless upstream availability changes.

3. **RESOLVED — coverage floors**
   - **Settled design:** do not invent a threshold from the pre-gap research numbers. Plan 04 first adds named deterministic selectors, including the four PostgreSQL families (DB-API error classification, replay, bounded pagination, transactional rollback); Plan 05 then captures the actual post-gap repository-total and critical-scope statement/branch counts and rates as immutable non-regression floors. [VERIFIED: `08-CONTEXT.md` D-14 and D-15; `src/cacheness/storage/backends/postgresql_lifecycle_authority.py:247-345,1577-1695`]
   - **Owner:** Plan 04 owns test-first gap closure; Plan 05 owns baseline capture and the read-only verifier.
   - **Preflight/checkpoint:** the named PostgreSQL and cache/lifecycle suites must exist, collect without skip/deselection, and pass before capture. Baseline capture requires an explicit justification and review; ordinary verification cannot rewrite it. The currently measured 75.29%/58.82% total and 77.33%/59.13% critical rates remain diagnostics only.

4. **RESOLVED — immutable-release approval and publication**
   - **Settled design:** the final transition is draft release → attach exact sanitized qualifying assets → verify tag SHA, exact asset-name set, each asset state and SHA-256 digest → publish → verify immutable release and each local asset. Extra diagnostic artifacts are forbidden from the qualifying release asset set. [CITED: https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases; CITED: https://docs.github.com/en/rest/releases/assets; CITED: https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/secure-your-dependencies/verify-release-integrity]
   - **External prerequisite/owner:** a repository administrator enables immutable releases and names an authorized release operator/reviewer. This research does not claim the setting or identity exists.
   - **Preflight/checkpoint:** before Plan 12 Task 1, verify repository/org immutable-release policy, `gh auth status`, release permissions, exact tag-to-SHA resolution, and complete same-SHA evidence. Plan 12 Task 2 binds operator/reviewer approval to the exact prepublication report digest; Task 3 performs the irreversible publish and then requires published/non-draft immutable state plus the exact assets/digests in a final read-only verifier.

5. **RESOLVED — live PostgreSQL/Amazon-S3 environment**
   - **Settled design:** use a protected GitHub environment, the four existing configuration names, real PostgreSQL plus authoritative Amazon S3 with standard provider identity and no endpoint override, exact owner markers, bounded cleanup, and the frozen three-module live suite. [VERIFIED: `08-08-PLAN.md`; `tools/run_phase5_qualification.py:34-39,395-491`; `08-CONTEXT.md` D-09 through D-13]
   - **External prerequisite/owner:** a cloud/repository administrator provisions least-privilege database/schema and bucket/prefix access and installs the protected secrets. No endpoint, account, credential, or budget is invented here.
   - **Preflight/checkpoint:** run the Phase 8 runner’s configuration-only preflight; dispatch the explicit 40-character SHA; record and wait for its exact workflow run ID; confirm `headSha`; download the named artifact by run ID; and validate revision, relevant-source digest, `QUALIFIED`, complete frozen selection, and `CLEAN`. If any preflight, execution, or cleanup check fails, BACK-05 remains `UNAVAILABLE`/`NOT_QUALIFIED`. [CITED: https://cli.github.com/manual/gh_workflow_run; CITED: https://cli.github.com/manual/gh_run_view; CITED: https://cli.github.com/manual/gh_run_watch; CITED: https://cli.github.com/manual/gh_run_download]

### Required checkpoint mechanics

| Gate | Checkpoint type | Owner | Resume signal | On failure/absence |
|---|---|---|---|---|
| `pyperf` package legitimacy | `checkpoint:human-verify` | Maintainer | Explicit package approval or explicit rejection selecting the documented in-repo fallback | Do not install or silently continue |
| Physical controlled runner | `checkpoint:human-action` | Repository/infrastructure maintainer | Runner registered under `cacheness-perf-linux-x64` and preflight fingerprint attached | QUAL-06 remains unavailable |
| Protected live resources/secrets | `checkpoint:human-action` | Cloud/repository administrator | Configuration-only preflight succeeds without disclosing values | BACK-05 remains `UNAVAILABLE` |
| Exact candidate dispatch | `checkpoint:human-verify` | Release operator | Exact candidate SHA approved; resulting run ID recorded | Do not select another/latest run |
| Immutable release enablement/permissions | `checkpoint:human-action` | Repository administrator | Actual policy and authenticated permission preflights pass | Final release publication is blocked |
| Irreversible draft publication | `checkpoint:human-verify` | Release operator/reviewer | Exact tag/SHA and prepublication state/assets/digests report approved | Leave draft unpublished and report the failing class |

After the resume signal, deterministic automation continues and revalidates the external state. A human assertion alone does not fill any evidence class.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| macOS ARM64 host | Local smoke/research only | ✓ | Darwin 25.6.0 | GitHub-hosted macOS for release evidence. [VERIFIED: local environment probe, 2026-09-13] |
| CPython | Local deterministic work | Partial | `.venv` 3.11.16; system 3.12.1; uv-managed 3.12.10 and 3.13.15; no installed 3.14 found | CI `setup-python` matrix. [VERIFIED: local filesystem/environment probe, 2026-09-13] |
| uv | Locked environments/build | ✓ | 0.12.12 | None recommended. [VERIFIED: local environment probe, 2026-09-13] |
| pytest / pytest-cov / Coverage.py | Tests/coverage | ✓ in `.venv` | 8.4.1 / 6.2.1 / 7.10.3 | Locked CI environment. [VERIFIED: local package metadata, 2026-09-13] |
| Ruff | Lint/format | ✓ in `.venv` | 0.12.9 | Locked CI environment. [VERIFIED: local package metadata, 2026-09-13] |
| GitHub CLI | Exact-run collection and immutable release verification | ✓ | 2.98.0; `gh release verify` and `gh release verify-asset` available | GitHub REST API, with the same exact state/digest checks. [VERIFIED: local environment probe, 2026-09-13] |
| Docker CLI | Optional local service tests | ✓ binary | Daemon not probed | Deterministic fakes/CI service; never substitute for live release qualification. [VERIFIED: local environment probe, 2026-09-13] |
| PostgreSQL client/service | Live qualification | ✗ locally | — | No qualifying fallback; protected real service required. [VERIFIED: local environment probe, 2026-09-13; `08-CONTEXT.md` D-11] |
| AWS CLI | Operations support | ✓ binary | Not probed | Runner uses standard AWS credential chain; CLI itself is not proof. [VERIFIED: local environment probe, 2026-09-13] |
| Live qualification configuration | BACK-05 | ✗ locally | All four required variable names unset | No qualifying fallback. [VERIFIED: local environment probe, 2026-09-13; `tools/run_phase5_qualification.py:34-39`] |
| Controlled Linux performance runner | QUAL-06 | ✗ not identified | — | No release-gate fallback; local macOS numbers are diagnostic only. [VERIFIED: `08-CONTEXT.md` D-20] |
| GitHub Actions workflows | QUAL-03 | ✗ no files found | — | Wave 0 must create workflows. [VERIFIED: repository file inventory, 2026-09-13] |
| Network package registry access | Clean installs/current probe | ✗ in this sandbox | DNS unavailable | CI or approved networked environment. [VERIFIED: failed uv/pip registry probes, 2026-09-13] |
| `pyperf` | Controlled benchmarks | ✗ locally | Official latest 2.10.0 | Human checkpoint, then add locked dev dependency; otherwise retain reviewed custom runner. [VERIFIED: local package probe and legitimacy seam, 2026-09-13; CITED: https://pypi.org/project/pyperf/] |

**Missing dependencies with no fallback:** real PostgreSQL/Amazon S3 configuration for BACK-05, the controlled Linux performance runner for QUAL-06, and repository CI/protected-release configuration for QUAL-03. [VERIFIED: environment audit and `08-CONTEXT.md` D-09 through D-13, D-20]

**Missing dependencies with fallback:** local Python 3.14 and network installs can run in CI; local macOS is not the controlled performance source. [ASSUMED]

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.4.1 + pytest-cov 6.2.1 + Coverage.py 7.10.3 + Ruff 0.12.9 in the current local project environment. [VERIFIED: local package metadata, 2026-09-13] |
| Config file | `pyproject.toml`; strict markers and test discovery are configured, but branch coverage and thresholds are absent. [VERIFIED: `pyproject.toml:82-129`] |
| Quick run command | `uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false tests/qualification/test_live_evidence.py tests/test_full_suite_environment.py tests/test_phase6_statistics.py tests/test_phase6_removal_contract.py` [ASSUMED] |
| Full suite command | `uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false` [VERIFIED: `tests/test_full_suite_environment.py:15-24,79-96`] |

The full-suite command includes marked live modules and therefore must be used with the Phase 8 deterministic selector/exclusions on PRs; live modules run only through the fixed live runner. The existing Phase 6 command is the current non-live precedent. [VERIFIED: `tests/test_full_suite_environment.py:15-24,177-186`; `pyproject.toml:94-102`]

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BACK-05 | Exact-commit PostgreSQL + Amazon S3 behavior, no skips/emulator, exact cleanup | live integration + runner contract | `uv run ... python tools/run_phase8_qualification.py --output build/live.json` [ASSUMED] | ❌ Wave 0 adapter; existing Phase 5 runner/live modules are reusable. [VERIFIED: `tools/run_phase5_qualification.py`; `tests/integration/test_postgresql_authority.py`; `tests/integration/test_s3_generation.py`; `tests/integration/test_remote_topology.py`] |
| QUAL-01 | Base wheel public imports + generic and NumPy format round trips | packaging integration | `uv run ... pytest -q tests/packaging/test_wheel_matrix.py -k base` [ASSUMED] | ❌ Wave 0 extension; partial existing probe at `tests/test_full_suite_environment.py:122-174`. [VERIFIED] |
| QUAL-02 | Each literal optional group installs independently and performs representative round trip | packaging integration | `uv run ... pytest -q tests/packaging/test_wheel_matrix.py -k extras` [ASSUMED] | ❌ Wave 0; current test covers only base/S3/cloud import and base round trip. [VERIFIED: `tests/test_full_suite_environment.py:122-174`] |
| QUAL-03 | Linux stable matrix, macOS boundary, deterministic backend, lint/format, coverage, package, live and performance gates | CI contract/self-test | `uv run ... pytest -q tests/test_phase8_quality_gates.py` [ASSUMED] | ❌ Wave 0; no workflow files exist. [VERIFIED: repository file inventory, 2026-09-13] |
| QUAL-04 | Existing finite integrity/recovery/commit-boundary selectors stay present and green | deterministic fault/crash | `uv run ... python tools/verify_phase071_contracts.py --all` | ✅ Existing fixed verifier. [VERIFIED: `.planning/phases/07.1-obstore-payload-participant-unification/07.1-VERIFICATION.md:160-180`] |
| QUAL-05 | Named gaps + repository and critical statement/branch non-regression | deterministic coverage | `uv run ... pytest --cov=cacheness --cov-branch ... && uv run ... python tools/verify_phase8_coverage.py ...` [ASSUMED] | ❌ Wave 0 verifier/baseline; raw measurement completed in research. [VERIFIED: local coverage JSON, 2026-09-13] |
| QUAL-06 | Layered formats/hashes, distributions, memory, exact environment/revision, reviewed relative envelope | benchmark + harness contracts | `uv run ... python benchmarks/phase8_benchmarks.py -o build/phase8.json` [ASSUMED] | ❌ Wave 0 replacement; existing benchmark is stale. [VERIFIED: `benchmarks/lifecycle_authority_benchmark.py:257-290,562-567`] |
| QUAL-07 | Inventory/reconciliation/statistics/clear/aggregate call and memory formulas at scale | deterministic structural | `uv run ... pytest -q tests/performance/test_complexity_contracts.py` [ASSUMED] | ❌ Wave 0 formula consolidation; existing bounded behavior tests are reusable. [VERIFIED: `tests/test_catalog_query_contract.py`; `tests/test_blob_store_reconciliation.py`; `tests/test_phase6_statistics.py`; `tests/test_phase6_removal_contract.py`] |

### Sampling Rate

- **Per task commit:** Run the focused files changed plus `tests/test_phase8_quality_gates.py`; for lifecycle/cache-policy changes also run the exact Phase 07.1 selector subset and coverage quick scope. [ASSUMED]
- **Per wave merge:** Run the deterministic full suite on the wave's Python, package-matrix self-tests, direct Ruff scopes, and coverage verifier. [ASSUMED]
- **Phase gate:** Linux 3.11-3.14 compatible matrix green, TensorFlow 3.11-3.13 probe green, macOS 3.11/3.14 smoke green, exact-commit live evidence `QUALIFIED`/`CLEAN`, controlled performance within reviewed envelope, and immutable release evidence attached. [ASSUMED]

### Wave 0 Gaps

- [ ] `.github/workflows/quality.yml` — stable/advisory/platform/deterministic gate orchestration. [ASSUMED]
- [ ] `.github/workflows/live_qualification.yml` — protected dispatch and off-hour schedule, short-lived diagnostics. [ASSUMED]
- [ ] `.github/workflows/performance.yml` — controlled-runner-only benchmark gate. [ASSUMED]
- [ ] `tools/run_phase8_qualification.py` and runner self-tests — adapt Phase 5 evidence without weakening it. [ASSUMED]
- [ ] `tests/test_phase8_lifecycle_coverage.py` — add literal Phase 8 selectors for PostgreSQL DB-API error classification, exact replay, bounded pagination, and transactional rollback before any coverage baseline capture. [VERIFIED: `08-CONTEXT.md` D-14 and D-15; `src/cacheness/storage/backends/postgresql_lifecycle_authority.py:247-345,1577-1695`]
- [ ] `tools/verify_phase8_coverage.py` plus checked baseline schema/self-tests — total and critical statement/branch ratchet. [ASSUMED]
- [ ] `tests/packaging/test_wheel_matrix.py` — base plus literal per-extra independent probes. [ASSUMED]
- [ ] `tests/performance/test_complexity_contracts.py` — call formulas and peak-memory tiers. [ASSUMED]
- [ ] `benchmarks/phase8_benchmarks.py`, workload definitions, schema tests, and controlled baseline path. [ASSUMED]
- [ ] `pyproject.toml` branch coverage setting and `pyperf` dev dependency after human package checkpoint. [ASSUMED]
- [ ] Exact fixed selectors for named integrity/recovery/policy/qualification/package gaps; percentages alone are insufficient. [VERIFIED: `08-CONTEXT.md` D-15]
- [ ] Exact-SHA workflow dispatch/run-ID wait/artifact-name download tests, and a final immutable publication verifier for exact tag SHA, published immutable state, exact asset set, upload states, and SHA-256 digests. [VERIFIED: `08-08-PLAN.md` Task 3; `08-10-PLAN.md`; `08-11-PLAN.md` Task 2; `08-12-PLAN.md`; CITED: https://cli.github.com/manual/gh_run_download; CITED: https://docs.github.com/en/rest/releases/assets]

## Security Domain

Security enforcement is enabled at ASVS level 1 in `.planning/config.json`. [VERIFIED: `.planning/config.json`]

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No application-user authentication surface | GitHub protected environments and standard cloud credentials are operational controls, not a library auth feature. [VERIFIED: project architecture; `08-CONTEXT.md` D-09 through D-12] |
| V3 Session Management | No | No browser/user session exists in this Python library. [VERIFIED: `AGENTS.md` Project and Technology Stack] |
| V4 Access Control | Yes, operational | Least-privilege PostgreSQL role, AWS IAM/bucket policy, protected release environment, and exact run-owner cleanup marker. [VERIFIED: `08-CONTEXT.md` D-12; `tests/qualification/conftest.py`] |
| V5 Input Validation | Yes | Existing exact evidence allow-lists, typed status/cleanup sets, bounded env/config validation, safe-text redaction, and catalog/work caps. [VERIFIED: `tools/run_phase5_qualification.py:60-103,200-212,291-368`; `src/cacheness/config.py:66-125,336-387`] |
| V6 Cryptography | Yes | Preserve standard HMAC/SHA-256/constant-time comparison and canonical SHA-256-plus-size integrity; XXH3 remains performance evidence only. [VERIFIED: `08-CONTEXT.md` D-19; `src/cacheness/core.py:492-583`; `.planning/phases/07.1-obstore-payload-participant-unification/07.1-VERIFICATION.md:143-152`] |
| V12 Files and Resources | Yes | Path/namespace containment, immutable generations, exact owned cleanup, bounded artifact size/retention, and no untrusted payload logging. [VERIFIED: `AGENTS.md` Project Constraints; `.planning/phases/07.1-obstore-payload-participant-unification/07.1-VERIFICATION.md:207-223`; `tests/qualification/conftest.py`] |

### Known Threat Patterns for Release Qualification

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Secret values or DSNs leak into evidence/logs | Information Disclosure | Keep exact allow-list and forbidden-value scan; upload only sanitized JSON; never echo environment/context. [VERIFIED: `tools/run_phase5_qualification.py:60-103,291-368,439-453`] |
| Forked PR executes with live cloud secrets | Elevation of Privilege / Information Disclosure | Deterministic PR workflow only; live workflow uses protected environment and trusted exact revision. [CITED: https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions] |
| Old artifact is attached to a new release | Spoofing / Tampering | Validate full Git revision and relevant-source digest at aggregation; immutable release locks tag/assets. [VERIFIED: `tools/run_phase5_qualification.py:291-368`; CITED: https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases] |
| Cleanup deletes non-test objects/rows | Tampering / Denial of Service | Retain exact owner marker, per-run namespace, page/object/byte/delete/multipart caps, and fail closed on marker mismatch. [VERIFIED: `tests/qualification/conftest.py`] |
| Benchmark changes canonical integrity digest | Tampering | Keep persisted SHA-256+size untouched; report XXH3 only as comparative data for SEED-003. [VERIFIED: `08-CONTEXT.md` D-19] |
| Payload bytes deserialize before authority/integrity verification | Tampering / Information Disclosure | Preserve authority-first selection and integrity-before-handler-read; add named coverage selector. [VERIFIED: `.planning/phases/07.1-obstore-payload-participant-unification/07.1-VERIFICATION.md:143-152,207-223`] |
| Dependency or action substitution | Tampering / Supply Chain | Lock `uv.lock`, pin new dependency exactly after legitimacy checkpoint, and pin third-party workflow actions to reviewed commit SHAs for release/live jobs. [ASSUMED] |

## Sources

### Primary (HIGH confidence)

- `docs/adr/0001-topology-specific-storage-guarantees.md` — single authority, guarantee vocabulary, topology limits, and stop conditions. [VERIFIED]
- `.planning/phases/08-production-gates-and-performance-stabilization/08-CONTEXT.md` — locked phase and gate decisions. [VERIFIED]
- `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md`, `.planning/STATE.md` — requirement ownership/status and current milestone boundary. [VERIFIED]
- `.planning/phases/07.1-obstore-payload-participant-unification/07.1-CONTEXT.md` and `07.1-VERIFICATION.md` — completed participant boundary and non-claims. [VERIFIED]
- `.codex/skills/spike-findings-cacheness/SKILL.md` plus `references/handler-integration.md`, `references/lifecycle-and-recovery.md`, and `references/backend-mechanics.md` — verified payload-participant implementation guidance. [VERIFIED]
- `pyproject.toml`, `src/cacheness/**`, `tests/**`, `tools/run_phase5_qualification.py`, and `benchmarks/**` — live repository source opened in this research session. [VERIFIED]
- Local deterministic coverage and benchmark/environment probes, 2026-09-13 — measured baseline and availability. [VERIFIED]

### Secondary (MEDIUM confidence)

- https://devguide.python.org/versions/ — current stable/prerelease CPython branches. [CITED]
- https://docs.github.com/en/actions/tutorials/build-and-test-code/python — explicit Python setup, matrix, pytest/coverage, and Ruff workflow patterns. [CITED]
- https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow — protected/manual workflow orchestration facts. [CITED]
- https://docs.github.com/en/actions/tutorials/manage-your-work/schedule-issue-creation — schedule delay/drop caveat. [CITED]
- https://docs.github.com/en/organizations/managing-organization-settings/configuring-the-retention-period-for-github-actions-artifacts-and-logs-in-your-organization — artifact retention bounds. [CITED]
- https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases — immutable tag/asset and attestation behavior. [CITED]
- https://docs.github.com/en/rest/releases/assets — release asset upload state and server-reported SHA-256 digest. [CITED]
- https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/secure-your-dependencies/verify-release-integrity — immutable release and local release-asset verification commands. [CITED]
- https://cli.github.com/manual/gh_workflow_run, https://cli.github.com/manual/gh_run_view, https://cli.github.com/manual/gh_run_watch, and https://cli.github.com/manual/gh_run_download — exact dispatch inputs, run identity/status, wait, and run-ID artifact download. [CITED]
- https://docs.astral.sh/uv/concepts/projects/build/ — wheel build pattern. [CITED]
- https://coverage.readthedocs.io/en/latest/commands/cmd_reporting.html — coverage reporting. [CITED]
- https://docs.astral.sh/ruff/linter/ and https://docs.astral.sh/ruff/formatter/ — direct lint/format modes. [CITED]
- https://pyperf.readthedocs.io/en/latest/ and https://pypi.org/project/pyperf/ — benchmark capabilities, version, and package provenance. [CITED]
- https://pypi.org/project/tensorflow/ and https://pypi.org/project/obstore/0.11.1/ — current interpreter/wheel compatibility. [CITED]

### Tertiary (LOW confidence)

- Workload sizes, sample count, scale tiers, and the initial 20% envelope remain research recommendations to validate on the controlled runner. The logical runner label and diagnostic retention are settled; the physical runner and credentials remain external prerequisites, not assumptions. [ASSUMED]

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — existing versions/config were read locally and current official documentation was checked; pyperf remains behind the required human legitimacy checkpoint. [VERIFIED: local source/environment probes and official sources listed above]
- Architecture: HIGH — locked CONTEXT, ADR 0001, Phase 07.1 verification, and live source agree on the evidence-only boundary. [VERIFIED: primary sources listed above]
- Coverage baseline: HIGH for this macOS run, MEDIUM as a release floor — final floors must be captured after gap closure in CI. [VERIFIED: local coverage JSON, 2026-09-13; `08-CONTEXT.md` D-14]
- Performance thresholds: MEDIUM — methodology is official and prescriptive, but sample/envelope values require the controlled Linux runner. [CITED: https://pyperf.readthedocs.io/en/latest/; ASSUMED]
- Environment/live readiness: HIGH — local tools/config were probed; external services and repository settings remain genuine prerequisites. [VERIFIED: local environment probe, 2026-09-13]
- Pitfalls: HIGH — most are directly visible in current source, baseline, or official service behavior. [VERIFIED: primary sources listed above]

**Research date:** 2026-09-13
**Valid until:** 2026-10-13 for repository architecture; re-check Python/dependency wheels, GitHub Actions behavior, and package versions immediately before release.
