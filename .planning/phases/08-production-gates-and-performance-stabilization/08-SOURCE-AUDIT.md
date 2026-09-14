# Phase 08 Multi-Source Coverage Audit

| Source | ID | Feature / requirement | Plan | Status | Notes |
|---|---|---|---|---|---|
| GOAL | — | Reproducible release evidence across supported installations, Python versions, backends, failures, and operational scale | 01-13 | COVERED | Evidence tracer expands through isolated gates, the D-23 support-surface removal, exact-run collection, and verified immutable publication. |
| REQ | BACK-05 | Real PostgreSQL and authoritative Amazon S3 integration qualification | 08, 10, 11, 12 | COVERED | Runner/workflow are deterministic to validate locally; Plan 11 requires real service proof and Plan 12 publishes/verifies it. |
| REQ | QUAL-01 | Clean base wheel imports guaranteed public surface and completes memory-backed round trips | 02, 09, 10, 13 | COVERED | Base includes generic and retained NumPy formats; Plan 13 removes the obsolete native TensorFlow surface. |
| REQ | QUAL-02 | Every advertised optional group installs/imports independently | 02, 09, 10, 13 | COVERED | One wheel, one fresh environment per remaining literal group; TensorFlow is no longer advertised. |
| REQ | QUAL-03 | CI covers runtimes, contracts, quality, packaging, PostgreSQL, and AWS | 01-13 | COVERED | Evidence classes use separate workflows, exact run-ID collection, aggregation, and publication verification after D-23 removal. |
| REQ | QUAL-04 | Finite integrity/recovery/commit-boundary regressions remain executable | 01, 04, 09, 10, 13 | COVERED | Inherited fixed verifier plus named lifecycle/policy gaps and typed unsupported stored-handler rejection. |
| REQ | QUAL-05 | Lifecycle/cache-policy statement and branch targets are measured and gated | 04, 05, 09, 10 | COVERED | Named gaps precede measured total/critical ratchets. |
| REQ | QUAL-06 | Checked performance distributions and reviewed budgets | 07, 10, 11 | COVERED | Stdlib harness, controlled workflow, actual controlled-runner capture. |
| REQ | QUAL-07 | Bounded memory and backend-call behavior | 06, 09, 10 | COVERED | Exact per-call-class formulas plus isolated peak RSS. |
| RESEARCH | R-01 | Evidence-class separation and exact-commit release manifest | 01, 10, 11, 12 | COVERED | Common strict envelope, exact-run collection, read-only aggregator, and immutable publication verifier. |
| RESEARCH | R-02 | One-wheel base and per-extra isolated probes | 02, 13 | COVERED | Public round trips prevent import-only qualification; Plan 13 narrows the literal group inventory. |
| RESEARCH | R-03 | Stable/advisory Python and Linux/macOS/Windows roles | 03, 09, 13 | COVERED | Windows stays NOT_QUALIFIED and no TensorFlow feature-profile exception remains. |
| RESEARCH | R-15 | D-23 removes native TensorFlow while retaining unsupported-contract failure and generic custom handlers | 13 | COVERED | Runtime, packaging, lock, docs, matrix, and evidence absence is executable. |
| RESEARCH | R-04 | Measure, close named gaps, then ratchet total and critical branch coverage | 04, 05 | COVERED | Baseline capture follows named behavior tests. |
| RESEARCH | R-05 | Direct changed-file plus critical-scope Ruff lint/format checks | 05, 09 | COVERED | No custom finding fingerprint/debt ledger. |
| RESEARCH | R-06 | Representative layer-separated format workloads | 07 | COVERED | Generic, NumPy NPZ/Blosc2, pandas/Polars Parquet; no topology cross-product. |
| RESEARCH | R-07 | SHA-256 versus XXH3 throughput and lifecycle share | 07, 11 | COVERED | Measurement only; persisted digest stays SHA-256 plus size. |
| RESEARCH | R-08 | Controlled Linux relative performance envelope | 07, 11 | COVERED | Named runner prerequisite and reviewed capture. |
| RESEARCH | R-09 | Explicit structural call formulas and fixed scale tiers | 06 | COVERED | Inventory/reconciliation/statistics/clear/policy maintenance. |
| RESEARCH | R-10 | Preserve Phase 5 fail-closed live runner and exact cleanup | 08, 11 | COVERED | Adapted only to current obstore sources/evidence. |
| RESEARCH | R-11 | Protected RC versus scheduled diagnostic live runs | 08 | COVERED | Scheduled evidence cannot qualify a release. |
| RESEARCH | R-12 | Fixed final selector/source/decision/threat verifier | 10, 12 | COVERED | Exact literal inventories, adversarial self-tests, and final remote-state verification. |
| RESEARCH | R-13 | Release-lifetime qualifying evidence and bounded diagnostics | 08, 10, 11, 12 | COVERED | QUALIFIED evidence attaches to the published immutable release; diagnostics retain 30 days. |
| RESEARCH | R-14 | Package/tooling choice | 07 | COVERED | Existing stdlib benchmark/distribution patterns selected; no new dependency. |
| CONTEXT | D-01 | Qualify Phase 07.1 obstore architecture; no legacy/runtime fallback | 01, 07, 08, 10 | COVERED | Source audits and current public composition only. |
| CONTEXT | D-02 | BlobStore/lifecycle authority sole owner; UnifiedCache policy only | 01, 04, 10 | COVERED | Tests and AST audits preserve ownership. |
| CONTEXT | D-03 | Separate integrity, recovery, progress, and performance | 01, 04, 06, 07, 09, 10 | COVERED | Separate schemas, commands, reports, and blocking rules. |
| CONTEXT | D-04 | Preserve topology limits and bounded orphan limitation | 01, 03, 04, 09, 10 | COVERED | Docs/tests cannot strengthen guarantees. |
| CONTEXT | D-05 | Gate stable Python 3.11 through latest compatible stable; prerelease advisory | 03, 09 | COVERED | Fixed matrix currently names 3.11-3.14. |
| CONTEXT | D-06 | Linux full matrix, macOS boundary, Windows nonclaim | 03, 09, 10 | COVERED | Native Windows remains Phase 999.1. |
| CONTEXT | D-07 | NumPy core and base generic/NumPy round trips | 02, 09, 13 | COVERED | Base proves pickle/native NPZ; Plan 13 explicitly preserves Blosc2 and NumPy while removing TensorFlow. |
| CONTEXT | D-08 | Independent advertised extras and retained Parquet handlers | 02, 09, 13 | COVERED | Plans 02/03 are historical where they qualified TensorFlow; Plan 13 removes that group before final evidence. |
| CONTEXT | D-09 | Deterministic PR and protected/scheduled live cadence | 08, 09 | COVERED | Separate workflow surfaces. |
| CONTEXT | D-10 | Live evidence qualifies exact release commit | 01, 08, 10, 11, 12 | COVERED | SHA/source bind production, run-ID collection, aggregation, tag target, and final publication. |
| CONTEXT | D-11 | Unavailable/incomplete live evidence blocks release | 01, 08, 10, 11, 12 | COVERED | No mock/skip/earlier-commit/draft substitution. |
| CONTEXT | D-12 | Preserve frozen runner, exact cleanup, and S3 constraints | 08, 11, 12 | COVERED | Explicit bucket/region and standard AWS only; only CLEAN evidence may be published. |
| CONTEXT | D-13 | Lifetime QUALIFIED evidence; bounded diagnostics | 08, 10, 11, 12 | COVERED | 30-day diagnostics and verified immutable release attachment. |
| CONTEXT | D-14 | Measure before threshold and close named gaps | 04, 05 | COVERED | Research baseline is input, not automatic floor. |
| CONTEXT | D-15 | Critical plus total statement/branch gates and named behavior | 04, 05, 09 | COVERED | Four numerical dimensions plus selector inventory. |
| CONTEXT | D-16 | Changed plus critical Ruff lint/format scopes | 05, 09, 13 | COVERED | Plan 13 checks its changed Python files directly; Plan 05 retains the complete critical scope. |
| CONTEXT | D-17 | No custom lint debt ledger | 05, 09, 13 | COVERED | Explicitly prohibited and audited. |
| CONTEXT | D-18 | Representative tiers and separated handler/store/cache layers | 07 | COVERED | No handler/topology Cartesian matrix. |
| CONTEXT | D-19 | SHA-256 versus XXH3 without persisted semantic change | 07, 10, 11 | COVERED | Manifest source is audited unchanged. |
| CONTEXT | D-20 | Controlled Linux distributions and reviewed envelope | 07, 10, 11, 12 | COVERED | Actual baseline capture is prerequisite-gated and the exact envelope is published. |
| CONTEXT | D-21 | Page/work complexity and backend-call formulas | 06, 09 | COVERED | Fixed tiers and independent call classes. |
| CONTEXT | D-22 | Controlled local performance blocks; remote latency diagnostic | 07, 08, 09, 10, 11, 12 | COVERED | Workflow, collection, and release rules enforce separation. |
| CONTEXT | D-23 | Remove native TensorFlow support and qualification; retain custom-handler seam and typed unsupported stored contract | 13, 09, 10, 11, 12 | COVERED | Plan 13 performs and proves removal before all remaining quality, aggregation, and publication gates. |

## Exclusions (Not Gaps)

- Narwhals dataframe compatibility is a deferred handler/extensibility idea.
- Changing the canonical digest to XXH3 remains deferred to SEED-003; Phase 8 supplies evidence only.
- Native Windows lifecycle qualification remains Phase 999.1.
- Native TensorFlow support is removed from this release rather than deferred inside the shipped package. A future in-tree implementation would require a separately discussed phase; ordinary out-of-tree custom handlers remain supported.
- New backend families, hostile pickle/dill sandboxing, async APIs, distributed coherence, and SqlCache redesign remain outside this phase.

## Audit Result

All GOAL, REQ, in-scope RESEARCH, and CONTEXT items are covered. No item requires a phase split or developer-approved deferral.
