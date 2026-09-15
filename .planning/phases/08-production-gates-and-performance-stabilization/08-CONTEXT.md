# Phase 8: Production Gates and Performance Stabilization - Context

**Gathered:** 2026-09-13
**Status:** Replanned for local-readiness closure (2026-09-15)

<domain>
## Phase Boundary

Qualify the post-Phase-07.1 Cacheness architecture for reliable local use. Phase 8 turns
the already-implemented `BlobStore` lifecycle, `UnifiedCache` policy layer,
obstore payload participant, supported handlers, packaging groups, and live
PostgreSQL/Amazon-S3 candidate into reproducible local evidence and preserved
future qualification machinery. It establishes measured coverage, memory, and
backend-call gates while retaining the implemented performance harness for future
controlled-Linux qualification under SEED-006 and the protected live/publication
pipeline under SEED-007, without changing lifecycle semantics or adding another
coordination mechanism.

This is qualification and stabilization, not a storage redesign. Phase 07.1
already replaced the built-in filesystem, memory, and S3 payload mechanics with
one obstore participant beneath the single lifecycle authority. Phase 8 must
test that final architecture as shipped.

</domain>

<decisions>
## Implementation Decisions

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
- **D-23:** Supersede only the current-milestone release-blocking portions of D-20 and D-22. Controlled-Linux performance qualification and QUAL-06 are `DEFERRED`/`NOT_QUALIFIED` under `.planning/seeds/SEED-006-qualify-controlled-linux-performance.md`; they do not block this milestone's release aggregate or publication. Preserve the implemented benchmark harness, representative workloads, SHA-256/XXH3 comparison, controlled-runner preflight, and performance workflow as future qualification capability. macOS measurements may remain diagnostic only and must not be presented as controlled-Linux evidence, Linux equivalence, a cross-platform budget, or a substitute pass. At the time of this decision every other class remained mandatory; D-24 subsequently supersedes that current-milestone premise for BACK-05 and publication without weakening their future standard.

### Local-readiness cutover
- **D-24:** Close Phase 8 on verified local readiness so Cacheness can be used locally now. Supersede Plans 08-11 and 08-12 for this milestone and defer their exact-SHA real PostgreSQL/Amazon-S3 qualification plus immutable GitHub release publication to `.planning/seeds/SEED-007-qualify-real-postgresql-s3-and-publish-release.md`. `BACK-05` remains `DEFERRED`/`NOT_QUALIFIED`; publication remains `DEFERRED`/`NOT_PUBLISHED`; neither state is a pass, a release claim, or permission to advertise the remote topology. Preserve the completed live runner, workflows, collector, aggregator, publication controller, exact-cleanup rules, and same-commit evidence requirements for SEED-007. Mocked S3, local emulators, configuration-only preflight, local deterministic evidence, and stale/earlier evidence may not substitute. Phase 8 closure must instead pass the deterministic local suite, base packaging/import round trip, measured coverage and Ruff gates, structural memory/call bounds, and finite integrity/recovery contracts, while reporting current-host platform scope and all unavailable platform/service claims honestly. D-24 supersedes only the current-milestone blocking/publication portions of D-09 through D-13 and D-23; it does not weaken their future SEED-007 qualification standard.

### the agent's Discretion
- Select the exact clean-environment tooling, CI job decomposition, artifact retention duration, representative fixture sizes, sample counts, warmups, statistical envelope calculation, and controlled Linux runner identity for the retained future capability, provided the choices are explicit and reproducible. D-23 prevents those future performance choices from blocking the current milestone.
- Select named coverage gaps after measuring the current baseline and inspecting risk; prioritize lifecycle, integrity-before-deserialization, cache invalidation/policy, qualification, packaging, and error-boundary branches.
- Consolidate or replace stale benchmark scripts when necessary, while retaining relevant historical evidence and making the new release benchmark entry points unambiguous.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project scope and release requirements
- `.planning/PROJECT.md` — current product architecture, constraints, and Phase 8 release boundary.
- `.planning/ROADMAP.md` § Phase 8 — goal, dependencies, requirements, and six success criteria.
- `.planning/REQUIREMENTS.md` — authoritative status and definitions for `BACK-05` and `QUAL-01` through `QUAL-07`.
- `docs/adr/0001-topology-specific-storage-guarantees.md` — mandatory guarantee vocabulary, single-authority design rules, topology limits, and stop conditions.

### Completed architecture being qualified
- `.planning/phases/07.1-obstore-payload-participant-unification/07.1-CONTEXT.md` — locked obstore participant, handler, S3 evidence, packaging, and Phase 8 handoff decisions.
- `.planning/phases/07.1-obstore-payload-participant-unification/07.1-VERIFICATION.md` — verified final architecture and explicit Phase 8 non-claims.
- `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md` — migration fixtures, stopped-worker cutover, and future-version compatibility boundary.
- `.planning/phases/06-unifiedcache-policy-composition/06-CONTEXT.md` — `UnifiedCache` policy-only ownership and fixed local-suite exclusions.
- `.planning/phases/05-payload-backends-and-supported-topology-qualification/05-CONTEXT.md` — topology contract and transferred live-service qualification decisions.

### Deferred measurement inputs
- `.planning/seeds/SEED-003-revisit-xxh3-for-canonical-blob-payload-hashing-after-measur.md` — future digest decision that Phase 8's SHA-256/XXH3 evidence must inform without implementing it.
- `.planning/seeds/SEED-006-qualify-controlled-linux-performance.md` — future controlled-Linux capture and QUAL-06 closure; current release evidence must preserve its explicit deferred nonclaim.
- `.planning/seeds/SEED-007-qualify-real-postgresql-s3-and-publish-release.md` — future real-service qualification and immutable publication; local readiness must preserve explicit `NOT_QUALIFIED`/`NOT_PUBLISHED` nonclaims.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tests/qualification/conftest.py`, `tests/qualification/test_live_evidence.py`, and the Phase 5 qualification runner provide fail-closed live-service configuration, isolated run namespaces, exact cleanup authorization, source-revision binding, evidence validation, and redaction contracts.
- `tests/integration/test_postgresql_authority.py` and `tests/integration/test_s3_generation.py` are the frozen non-substitutable live-service behavior suites; deterministic participant and topology coverage already exists under `tests/contracts/`.
- `tools/verify_phase071_contracts.py` supplies a fixed, fail-closed inventory for the architecture Phase 8 is qualifying and identifies which remote/platform/performance claims remain open.
- `benchmarks/lifecycle_authority_benchmark.py` and `benchmarks/lifecycle_authority_baseline.json` already model checked-in distributions, explicit environment/revision evidence, structural bounds, and deliberately separate runtime timeout policy from benchmark envelopes.
- `pyproject.toml` already declares Python 3.11+, the core NumPy/obstore dependencies, optional groups, pytest markers, coverage source configuration, and Ruff's Python/line-length policy.

### Established Patterns
- Deterministic PR contracts and live release evidence are separate evidence classes. Mocks and skips never qualify real PostgreSQL or Amazon S3.
- Built-in memory, filesystem, and S3 payload providers all materialize as `ObstoreGenerationIO`; qualification must exercise this shared participant rather than resurrect removed backend-specific paths.
- The lifecycle authority selects visibility and generation identity. Exact obstore `head()` evidence is signed, generation-bound corroboration only; listings and ETags are not lifecycle authority or canonical content hashes.
- Fixed selectors and schema-validated evidence fail closed when a promised test, source scope, cleanup outcome, or exact revision is missing.
- Existing benchmark material mixes current lifecycle evidence with older pre-refactor backend benchmarks. Planning should identify the canonical Phase 8 suite instead of treating every historical script as a release gate.

### Integration Points
- Packaging and isolated-extra checks attach at the wheel/build metadata and public barrels in `pyproject.toml`, `src/cacheness/__init__.py`, and `src/cacheness/storage/__init__.py`.
- Coverage and finite fault/concurrency gates attach to `src/cacheness/storage/lifecycle.py`, lifecycle authorities, `src/cacheness/storage/blob_store.py`, `src/cacheness/cache_policy.py`, and `src/cacheness/core.py` through their existing focused contract and integration tests.
- Live qualification composes the PostgreSQL lifecycle authority with the obstore S3 participant through the production topology factories; it must not reach around `BlobStore` or add a qualification-only lifecycle path.
- Performance measurement should instrument handler staging/serialization, `BlobStore` lifecycle, `UnifiedCache` policy, authority pagination, and participant calls as distinct layers.

</code_context>

<specifics>
## Specific Ideas

- NumPy arrays should use retained NPZ/Blosc2 handlers, and pandas/polars dataframe behavior should use retained Parquet handlers; Phase 8 verifies these rather than presenting them as new features.
- Hash benchmarking must include XXH3 because blob-sized hashing is performance-sensitive, but canonical SHA-256 remains unchanged until a separately versioned future decision evaluates the measured security and migration tradeoffs.
- Release evidence should say exactly what commit, environment, topology, workload, and evidence class it qualifies. `UNAVAILABLE` and `NOT_QUALIFIED` must remain visibly distinct from a pass.

</specifics>

<deferred>
## Deferred Ideas

- Investigate Narwhals as a future dataframe-handler compatibility layer across pandas, PyArrow, and Polars while retaining Parquet as the handler-owned format. This is a future handler/extensibility milestone, not Phase 8 qualification work.
- SEED-003 may revisit a versioned XXH3 canonical payload digest after Phase 8 supplies comparative throughput and end-to-end cost evidence.
- Native Windows lifecycle qualification remains Phase 999.1 because no eligible Windows environment is available.
- Controlled-Linux performance qualification and the final QUAL-06 budget/distribution claim are deferred to SEED-006. The Phase 8 release record must retain an explicit `DEFERRED`/`NOT_QUALIFIED` nonclaim rather than omit the class or substitute macOS measurements.
- Real PostgreSQL/Amazon-S3 qualification and immutable GitHub release publication are deferred to SEED-007. Phase 8 may close for local use only while `BACK-05` stays `DEFERRED`/`NOT_QUALIFIED` and publication stays `DEFERRED`/`NOT_PUBLISHED`; existing live/publication tooling remains intact.

</deferred>

---

*Phase: 08-production-gates-and-performance-stabilization*
*Context gathered: 2026-09-13*
