# Phase 08 Multi-Source Coverage Audit

| Source | ID | Feature / requirement | Plan | Status | Notes |
|---|---|---|---|---|---|
| GOAL | — | Reproducible release evidence across supported installations, Python versions, backends, failures, and operational scale | 01-15 | COVERED | Evidence tracer expands through isolated gates, a configuration-only protected-live preflight, exact-run collection, explicit performance deferral, and verified immutable publication. |
| REQ | BACK-05 | Real PostgreSQL and authoritative Amazon S3 integration qualification | 08, 10, 15, 11, 12 | COVERED | Runner/workflow are deterministic to validate locally; Plan 15 supplies the required no-side-effect configuration preflight, Plan 11 requires real service proof, and Plan 12 publishes/verifies it. |
| REQ | QUAL-01 | Clean base wheel imports guaranteed public surface and completes memory-backed round trips | 02, 09, 10 | COVERED | Base includes generic and retained NumPy formats. |
| REQ | QUAL-02 | Every advertised optional group installs/imports independently | 02, 09, 10 | COVERED | One wheel, one fresh environment per literal group. |
| REQ | QUAL-03 | CI covers runtimes, contracts, quality, packaging, PostgreSQL, and AWS | 01-03, 05-15 | COVERED | Evidence classes use separate workflows, fixed local prerequisite tests, exact run-ID collection, aggregation, and publication verification. |
| REQ | QUAL-04 | Finite integrity/recovery/commit-boundary regressions remain executable | 01, 04, 09, 10 | COVERED | Inherited fixed verifier plus named lifecycle/policy gaps. |
| REQ | QUAL-05 | Lifecycle/cache-policy statement and branch targets are measured and gated | 04, 05, 09, 10 | COVERED | Named gaps precede measured total/critical ratchets. |
| REQ | QUAL-06 | Checked performance distributions and reviewed budgets | 07, 13 | DEFERRED | Harness, workloads, hash comparison, workflow, and preflight remain available; controlled-Linux qualification is `NOT_QUALIFIED` under SEED-006 and is excluded from the current milestone by D-23. |
| REQ | QUAL-07 | Bounded memory and backend-call behavior | 06, 09, 10 | COVERED | Exact per-call-class formulas plus isolated peak RSS. |
| RESEARCH | R-01 | Evidence-class separation and exact-commit release manifest | 01, 10, 11, 12 | COVERED | Common strict envelope, exact-run collection, read-only aggregator, and immutable publication verifier. |
| RESEARCH | R-02 | One-wheel base and per-extra isolated probes | 02 | COVERED | Public round trips prevent import-only qualification. |
| RESEARCH | R-03 | Stable/advisory Python and Linux/macOS/Windows roles | 03, 09 | COVERED | Windows stays NOT_QUALIFIED. |
| RESEARCH | R-04 | Measure, close named gaps, then ratchet total and critical branch coverage | 04, 05 | COVERED | Baseline capture follows named behavior tests. |
| RESEARCH | R-05 | Direct changed-file plus critical-scope Ruff lint/format checks | 05, 09 | COVERED | No custom finding fingerprint/debt ledger. |
| RESEARCH | R-06 | Representative layer-separated format workloads | 07 | COVERED | Generic, NumPy NPZ/Blosc2, pandas/Polars Parquet; no topology cross-product. |
| RESEARCH | R-07 | SHA-256 versus XXH3 throughput and lifecycle share | 07 | COVERED | Measurement only; persisted digest stays SHA-256 plus size. |
| RESEARCH | R-08 | Controlled Linux relative performance envelope | 07, 13 | DEFERRED | Existing eligibility machinery is retained; actual capture moves to SEED-006 and macOS diagnostics do not substitute. |
| RESEARCH | R-09 | Explicit structural call formulas and fixed scale tiers | 06 | COVERED | Inventory/reconciliation/statistics/clear/policy maintenance. |
| RESEARCH | R-10 | Preserve Phase 5 fail-closed live runner and exact cleanup | 08, 15, 11 | COVERED | Adapted only to current obstore sources/evidence; preflight validates the reviewed owner-marker/cap policy without authorizing or performing cleanup. |
| RESEARCH | R-11 | Protected RC versus scheduled diagnostic live runs | 08 | COVERED | Scheduled evidence cannot qualify a release. |
| RESEARCH | R-12 | Fixed final selector/source/decision/threat verifier | 10, 14, 15, 12 | COVERED | Exact literal inventories, adversarial self-tests, and final remote-state verification. |
| RESEARCH | R-13 | Release-lifetime qualifying evidence and bounded diagnostics | 08, 10, 14, 11, 12 | COVERED | The publication controller is completed before exact-SHA collection; QUALIFIED evidence attaches to the published immutable release and diagnostics retain 30 days. |
| RESEARCH | R-14 | Package/tooling choice | 07 | COVERED | Existing stdlib benchmark/distribution patterns selected; no new dependency. |
| CONTEXT | D-01 | Qualify Phase 07.1 obstore architecture; no legacy/runtime fallback | 01, 07, 08, 10 | COVERED | Source audits and current public composition only. |
| CONTEXT | D-02 | BlobStore/lifecycle authority sole owner; UnifiedCache policy only | 01, 04, 10 | COVERED | Tests and AST audits preserve ownership. |
| CONTEXT | D-03 | Separate integrity, recovery, progress, and performance | 01, 04, 06, 07, 09, 10, 13, 15 | COVERED | Runner preflights emit eligibility/configuration rather than performance or service evidence. |
| CONTEXT | D-04 | Preserve topology limits and bounded orphan limitation | 01, 03, 04, 09, 10 | COVERED | Docs/tests cannot strengthen guarantees. |
| CONTEXT | D-05 | Gate stable Python 3.11 through latest compatible stable; prerelease advisory | 03, 09 | COVERED | Fixed matrix currently names 3.11-3.14. |
| CONTEXT | D-06 | Linux full matrix, macOS boundary, Windows nonclaim | 03, 09, 10 | COVERED | Native Windows remains Phase 999.1. |
| CONTEXT | D-07 | NumPy core and base generic/NumPy round trips | 02, 09 | COVERED | Base proves pickle/native NPZ; Blosc2 where installed. |
| CONTEXT | D-08 | Independent extras and retained Parquet handlers | 02, 03, 09 | COVERED | Dataframe/TensorFlow compatibility is explicit. |
| CONTEXT | D-09 | Deterministic PR and protected/scheduled live cadence | 08, 09, 15 | COVERED | Separate workflow surfaces plus a local configuration-only protected-live preflight. |
| CONTEXT | D-10 | Live evidence qualifies exact release commit | 01, 08, 10, 13, 15, 11, 12 | COVERED | Exact clean detached SHA proof precedes configuration preflight, capture, collection, aggregation, tag target, and publication. |
| CONTEXT | D-11 | Unavailable/incomplete live evidence blocks release | 01, 08, 10, 15, 11, 12 | COVERED | Missing/invalid preflight state and unavailable live proof remain blocking; no mock/skip/earlier-commit/draft substitution. |
| CONTEXT | D-12 | Preserve frozen runner, exact cleanup, and S3 constraints | 08, 15, 11, 12 | COVERED | Explicit bucket/region and standard AWS only; preflight performs no service operation, and only later CLEAN evidence may be published. |
| CONTEXT | D-13 | Lifetime QUALIFIED evidence; bounded diagnostics | 08, 10, 11, 12 | COVERED | 30-day diagnostics and verified immutable release attachment. |
| CONTEXT | D-14 | Measure before threshold and close named gaps | 04, 05 | COVERED | Research baseline is input, not automatic floor. |
| CONTEXT | D-15 | Critical plus total statement/branch gates and named behavior | 04, 05, 09 | COVERED | Four numerical dimensions plus selector inventory. |
| CONTEXT | D-16 | Changed plus critical Ruff lint/format scopes | 05, 09 | COVERED | Direct argv scopes. |
| CONTEXT | D-17 | No custom lint debt ledger | 05, 09 | COVERED | Explicitly prohibited and audited. |
| CONTEXT | D-18 | Representative tiers and separated handler/store/cache layers | 07 | COVERED | No handler/topology Cartesian matrix. |
| CONTEXT | D-19 | SHA-256 versus XXH3 without persisted semantic change | 07, 13, 14 | COVERED | The benchmark and preflight remain intact as future capability; Plan 14 prevents release tooling from treating diagnostics as current qualification. |
| CONTEXT | D-20 | Controlled Linux distributions and reviewed envelope | 07, 13 | SUPERSEDED IN PART | D-23 preserves the capability but defers current-milestone capture and release blocking to SEED-006. |
| CONTEXT | D-21 | Page/work complexity and backend-call formulas | 06, 09 | COVERED | Fixed tiers and independent call classes. |
| CONTEXT | D-22 | Controlled local performance blocks; remote latency diagnostic | 07, 08, 09, 10, 13 | SUPERSEDED IN PART | Remote latency remains diagnostic; D-23 removes only controlled performance from the current release-blocking aggregate. |
| CONTEXT | D-23 | Defer QUAL-06 without Linux/macOS substitution while preserving future capability | 14, 11, 12 | COVERED | Release tooling records explicit `DEFERRED`/`NOT_QUALIFIED`, excludes performance artifacts from current publication, and keeps every non-deferred class mandatory. |

## Exclusions (Not Gaps)

- Narwhals dataframe compatibility is a deferred handler/extensibility idea.
- Changing the canonical digest to XXH3 remains deferred to SEED-003; Phase 8 supplies evidence only.
- Native Windows lifecycle qualification remains Phase 999.1.
- Controlled-Linux performance qualification and QUAL-06 are explicitly deferred to SEED-006 under D-23; this is not an unplanned gap.
- New backend families, hostile pickle/dill sandboxing, async APIs, distributed coherence, and SqlCache redesign remain outside this phase.

## Audit Result

All current-milestone GOAL, REQ, in-scope RESEARCH, and CONTEXT items are covered, including the additive configuration-only live-preflight gap in Plan 08-15. QUAL-06/R-08 and the current release-blocking portions of D-20/D-22 are explicitly deferred by developer decision D-23 to SEED-006, not silently omitted.
