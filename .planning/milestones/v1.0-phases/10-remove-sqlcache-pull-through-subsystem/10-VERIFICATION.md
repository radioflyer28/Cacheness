---
phase: 10-remove-sqlcache-pull-through-subsystem
verified: 2026-09-17T18:58:39Z
status: passed
score: 20/20 must-haves verified
behavior_unverified: 0
overrides_applied: 1
unverified_prohibition_count: 13
unverified_prohibitions:
  - statement: "Removal must not become a storage-lifecycle, concurrency, recovery, topology, or BlobStore/UnifiedCache composition redesign."
    verdict: "NON-AUTHORITATIVE SATISFIED — the Phase 10 diff changes no lifecycle/composition implementation file."
  - statement: "Removal must not retain a compatibility shim, tombstone module, tailored exception, alias, or replacement query-cache layer."
    verdict: "NON-AUTHORITATIVE SATISFIED — source/import/export and isolated-wheel absence contracts pass."
  - statement: "Closed verifier repair must not reopen, reinterpret, or rewrite completed phase history."
    verdict: "NON-AUTHORITATIVE SATISFIED — only current verifier manifests changed; completed phase artifacts did not."
  - statement: "Packaging proof must not add a second build harness or accept a source-tree-only result."
    verdict: "NON-AUTHORITATIVE SATISFIED — one digest-bound WheelArtifact drives archive inspection, install, metadata, and round trips."
  - statement: "Documentation assertions must not enforce repository-wide historical erasure."
    verdict: "NON-AUTHORITATIVE SATISFIED — the current-surface scanner explicitly excludes dated/planning history."
  - statement: "Direct deletion must not be replaced by a tombstone, alias, package hook, tailored runtime error, or new query-cache abstraction."
    verdict: "NON-AUTHORITATIVE SATISFIED — natural ImportError/ModuleNotFoundError is tested in source and wheel environments."
  - statement: "Runtime deletion must not touch BlobStore, UnifiedCache, handler, authority, topology, or caller database code."
    verdict: "NON-AUTHORITATIVE SATISFIED — production diff is limited to the retired module, package barrel, and orphan error reasons."
  - statement: "Dependency pruning must not redesign retained SQLAlchemy/PostgreSQL groups or remove dataframe-handler dependencies."
    verdict: "NON-AUTHORITATIVE SATISFIED — retained dependency declarations and full regressions pass."
  - statement: "Obsolete assets must not be retained as archives, redirects, warning banners, or renamed examples."
    verdict: "NON-AUTHORITATIVE SATISFIED — all nine exact paths are absent and the current reference scan is green."
  - statement: "Guidance must not imply UnifiedCache or BlobStore reproduces query-gap detection, table upserts, or range-aware pull-through."
    verdict: "NON-AUTHORITATIVE SATISFIED — all three canonical notes explicitly state that no in-package replacement exists."
  - statement: "Documentation must not authorize discovery, mutation, export, or migration of caller-owned tables."
    verdict: "NON-AUTHORITATIVE SATISFIED — notes state tables are untouched/unsupported and maintenance-tool contracts reject table-shaped inputs."
  - statement: "Current-map refresh must not modify source lifecycle code or broaden ADR 0001 guarantees."
    verdict: "NON-AUTHORITATIVE SATISFIED — lifecycle source is unchanged and maps retain the ADR guardrail/nonclaims."
  - statement: "Final reference scans must not erase completed planning or dated audit history."
    verdict: "NON-AUTHORITATIVE SATISFIED — no earlier phase artifact or dated audit changed in the Phase 10 implementation range."
human_verification:
  - test: "Accept or reject the 13 explicitly flagged prohibition judgments listed in this report."
    expected: "Each must-NOT statement is accepted as satisfied by the cited diff, source, documentation, wheel, and regression evidence."
    why_human: "The PLAN files label these prohibitions flagged-unverified; autonomous verifier judgments are non-authoritative and cannot silently clear them."
decision_coverage:
  honored: 16
  total: 16
  not_honored: []
---

# Phase 10: Remove SqlCache Pull-Through Subsystem Verification Report

**Phase Goal:** Remove the unrelated table/range-oriented `SqlCache` product so the first supported Cacheness release has one coherent BlobStore foundation and one cache-policy layer built over it.
**Verified:** 2026-09-17T18:58:39Z
**Status:** passed
**Re-verification:** No — initial independent verification

## Goal Achievement

The implementation passes every executable goal, artifact, wiring, source, dependency, documentation, and distribution check. The plans' 13 explicit `flagged-unverified` prohibitions were accepted through Phase 10 UAT after review of the independent verifier's evidence; this acceptance covers repository-state guardrails only and introduces no experiential or external-system claim.

### Observable Truths

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | The runtime module, public names, builders, aliases, hooks, and dedicated errors are absent naturally. | ✓ VERIFIED | `src/cacheness/sql_cache.py` is absent; `src/cacheness/__init__.py:13-47` exports only retained APIs; isolated source and wheel import probes pass. |
| 2 | Canonical guidance routes object/function caching to UnifiedCache and direct persistence to BlobStore without claiming a range-aware replacement. | ✓ VERIFIED | Exact notes in `docs/API_REFERENCE.md:8-10`, `docs/STORAGE_MIGRATION.md:10-12`, and `docs/README.md:6-8`; documentation contracts pass. |
| 3 | Caller-owned SQL tables remain untouched and no cleanup/export/migration surface was added. | ✓ VERIFIED | `tests/test_phase10_sqlcache_removal.py:366-429` checks signatures, imports/calls, no scripts, and fail-fast table-shaped inputs; docs state untouched/unsupported. |
| 4 | Package version remains 0.3.14. | ✓ VERIFIED | `pyproject.toml:3`, `src/cacheness/__init__.py:27`, and the exact version test pass. |
| 5 | The `sql` dependency group/extra is absent. | ✓ VERIFIED | `pyproject.toml:18-70` has six retained extras and no `sql` group; installed wheel metadata rejects any `sql` extra. |
| 6 | DuckDB is absent from source-facing claims, manifest, lock, wheel metadata, examples, and CI/current docs. | ✓ VERIFIED | Manifest/lock contract passes; bounded `rg` found no positive current DuckDB edge outside negative tests/tools; installed metadata check passes. |
| 7 | SQLAlchemy and PostgreSQL authority/catalog behavior remain intact. | ✓ VERIFIED | SQLAlchemy and psycopg declarations remain at `pyproject.toml:24,37-43,53`; full frozen non-live suite passes PostgreSQL deterministic contracts. |
| 8 | Dataframe/Parquet handlers and their dependency ownership remain intact. | ✓ VERIFIED | pandas/Polars/PyArrow declarations remain; handler/Parquet tests are active and pass in the full suite. |
| 9 | Dedicated SqlCache tests, docs, and examples are physically deleted, not archived. | ✓ VERIFIED | The literal 13-path absence inventory at `tests/test_phase10_sqlcache_removal.py:23-42,219-229` passes. |
| 10 | Truthful dated audits and completed planning history remain preserved. | ✓ VERIFIED | Phase diff changes no earlier phase directory or dated audit; scanner excludes the named dated audit and planning history. |
| 11 | Mixed current docs retain supported dataframe/platform content while dropping obsolete ownership claims. | ✓ VERIFIED | `docs/PANDAS_API_AUDIT.md` retains Parquet/handler coverage; `docs/CROSS_PLATFORM_GUIDE.md` retains platform guidance; docs tests pass. |
| 12 | Removal is recorded concisely in existing canonical owners only; no standalone removal guide exists. | ✓ VERIFIED | The exact-owner scanner at `tests/test_phase10_sqlcache_removal.py:76-138,197-206,308-333` passes. |
| 13 | Source API and all installable wheel locations omit the retired module. | ✓ VERIFIED | Wheel inspection covers root plus `.data/purelib` and `.data/platlib` paths at `tools/run_phase8_packaging.py:229-285`; adversarial path tests and real wheel pass. |
| 14 | Current-facing references use a narrow explicit allowlist without erasing history. | ✓ VERIFIED | `CURRENT_REFERENCE_ROOTS`, exact note owners, marker counts, and historical exclusion are literal at `tests/test_phase10_sqlcache_removal.py:60-165`; scan passes. |
| 15 | Reusable public, isolation, quality, docs, and closed verifier contracts enforce the retained boundary. | ✓ VERIFIED | All declared artifacts/key links pass `verify.artifacts`/`verify.key-links`; focused contract suite exits 0. |
| 16 | A fresh source-free wheel proves retained BlobStore/UnifiedCache imports and local round trips plus pruned installed metadata. | ✓ VERIFIED | `tools/run_phase8_packaging.py:324-470` installs the one digest-bound wheel with `--isolated --no-project`, checks metadata/imports, and performs BlobStore and UnifiedCache round trips; real wheel test passes. |
| 17 | The supported product surface is one BlobStore lifecycle foundation plus UnifiedCache cache policy and store-local handlers. | ✓ VERIFIED | Public barrel and current maps agree; `src/cacheness/__init__.py:1-10` states the boundary; retained public-contract and wheel tests pass. |
| 18 | Documentation states Linux full-matrix qualification, macOS boundary smoke, and native Windows unqualified. | ✓ VERIFIED | `docs/CROSS_PLATFORM_GUIDE.md:5-11` and `docs/RELEASE_QUALIFICATION.md:22,37-40` state the exact tiered boundary. |
| 19 | Phase 10 did not redesign lifecycle, concurrency, recovery, topology, handler persistence, or ADR 0001. | ✓ VERIFIED | Git diff from the pre-phase context commit changes no `core.py`, handler/interface, storage lifecycle/composition/authority, or ADR file; only `sql_cache.py` is deleted from production storage-related scope. |
| 20 | Post-fix code review and current full non-live acceptance are clean. | ✓ VERIFIED | `10-REVIEW.md` is clean after `2dc3cbf`; verifier reran focused contracts, lock/Ruff, and full frozen non-live suite at current HEAD, all exit 0. |

**Score:** 20/20 truths verified (0 present-but-behavior-unverified)

### Required Artifacts

| Artifact group | Expected | Status | Details |
|---|---|---|---|
| `tests/test_phase10_sqlcache_removal.py` and retained public/isolation suites | Fail-closed source/API/reference/caller-data contracts | ✓ VERIFIED | Substantive, active, and wired; no skip/xfail markers. |
| Phase 4/6/07.1 verifier tool/test pairs | Closed retained matrices without deleted test nodes | ✓ VERIFIED | All three artifact/link queries pass and focused suite exits 0. |
| `tools/run_phase8_packaging.py`, `tests/packaging/test_wheel_matrix.py` | One-artifact archive/metadata/import/round-trip proof | ✓ VERIFIED | Digest-bound, source-free, and behaviorally executed against a fresh wheel. |
| `src/cacheness/__init__.py`, `src/cacheness/error_handling.py` | Retained barrel and shared error vocabulary | ✓ VERIFIED | SqlCache imports/exports and four orphan reasons are absent; retained exact-set contracts pass. |
| `pyproject.toml`, `uv.lock` | Exact pruned dependency graph | ✓ VERIFIED | `uv lock --check` exits 0; no DuckDB/`sql` group; retained dependency declarations present. |
| Canonical and mixed docs | Bounded removal guidance with useful retained material | ✓ VERIFIED | Exact-owner and relative-link documentation contracts pass. |
| `AGENTS.md`, `.planning/codebase/*.md` | Current BlobStore-first maps and qualification boundaries | ✓ VERIFIED | Current reference scan passes; lifecycle guardrail remains linked. |

Deleted files are verified by literal absence contracts rather than treated as missing required artifacts.

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| Phase 10 negative contract | package barrel and manifest/lock | isolated imports plus TOML/lock parsing | ✓ WIRED | Both Plan 10-01 links verified. |
| Phase 6/07.1 verifiers | retained tests / Phase 10 negative owner | fixed literal matrices | ✓ WIRED | Both Plan 10-02 links verified and execute in focused suite. |
| wheel matrix | packaging runner | one `WheelArtifact` path and SHA-256 | ✓ WIRED | Synthetic adversarial and real fresh-wheel cases pass. |
| docs tests | three canonical notes | exact owner and exact wording | ✓ WIRED | Documentation suite passes. |
| manifest | lock and installed metadata | `uv lock` convergence plus wheel metadata | ✓ WIRED | Lock check and fresh wheel pass. |
| AGENTS/current maps | ADR and manifest | explicit guardrail and dependency claims | ✓ WIRED | Plan 10-09 links verified; source claims match manifest. |

All 13 key links declared across the nine plans pass the GSD key-link verifier.

### Data-Flow Trace (Level 4)

| Artifact | Data | Source | Produces Real Data | Status |
|---|---|---|---|---|
| packaging runner | wheel members | freshly built digest-bound wheel ZIP | Yes | ✓ FLOWING |
| installed probe | exports, requirements, extras | source-free `importlib.metadata.distribution("cacheness")` | Yes | ✓ FLOWING |
| installed probe | BlobStore/UnifiedCache values | real isolated memory topology and public methods | Yes | ✓ FLOWING |
| removal contract | current references | explicit repository roots and file contents | Yes | ✓ FLOWING |
| maintenance boundary contract | constructor/tool behavior | real exported APIs plus fail-fast probe objects | Yes | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| Complete focused cutover/public/verifier/docs/fresh-wheel contract | frozen isolated pytest over the nine Phase 10 acceptance modules | exit 0 | ✓ PASS |
| Manifest/lock convergence | `uv lock --check` | resolved successfully | ✓ PASS |
| Changed surviving Python files meet scoped lint | frozen isolated Ruff over Phase 10 files | `All checks passed!` | ✓ PASS |
| Retained non-live behavior | full frozen isolated suite excluding live service markers | exit 0; 9 expected skips; one pre-existing collection warning; no failures | ✓ PASS |

### Probe Execution

No `scripts/**/tests/probe-*.sh` or phase-declared shell probe exists. The runnable fresh-wheel acceptance is part of the pytest packaging contract and was executed above.

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|---|---|---|---|---|
| CACH-07 | 10-01 through 10-09 | Remove SqlCache product and orphan dependencies while retaining BlobStore, UnifiedCache, SQL authorities/catalog, and dataframe handlers | ✓ SATISFIED | All 20 truths, artifact/link checks, fresh wheel, and full non-live suite pass. |

No Phase 10 requirement is orphaned from the plans. `REQUIREMENTS.md` still says Pending because phase completion bookkeeping has not yet run; implementation evidence satisfies the requirement.

### Decision Coverage

All 16 trackable decisions D-01 through D-16 are honored. `check.decision-coverage-verify` returned `honored: 16`, `total: 16`, and no unhonored decisions.

### Test Quality Audit

| Test group | Linked Req | Active | Skipped | Circular | Assertion Level | Verdict |
|---|---|---:|---:|---|---|---|
| Phase 10 removal contract | CACH-07 | yes | 0 | no | behavioral + exact value + negative import/source | strong |
| public/verifier/docs contracts | CACH-07 | yes | 0 | no | exact value/inventory + subprocess behavior | strong |
| wheel matrix | CACH-07 | yes | 0 | no | behavioral fresh build/install/round trip + adversarial archive paths | strong |
| full non-live regressions | CACH-07 retained surfaces | yes | 9 unrelated expected platform/TensorFlow skips | no | behavioral | strong |

**Disabled tests on CACH-07:** 0. **Circular patterns:** 0. **Insufficient assertions:** 0.

### Anti-Patterns Found

No `TBD`, `FIXME`, `XXX`, `TODO`, `HACK`, placeholder, disabled-test, or empty user-visible implementation pattern was found in the changed surviving Phase 10 files. The Phase 10 code review is clean after the wheel-path fix.

### Human Verification Required

#### 1. Accept the explicitly flagged prohibitions

**Test:** Review the 13 `unverified_prohibitions` judgments in frontmatter against the cited diff, code, docs, and tests.

**Expected:** Accept each must-NOT as satisfied: no lifecycle redesign, compatibility residue, history erasure, replacement query cache, caller-table authority, dependency over-pruning, duplicate build harness, or false qualification claim.

**Why human:** Each corresponding PLAN prohibition is explicitly prefixed `flagged-unverified`. Under autonomous verification these receive a non-authoritative LLM judgment and must remain visibly flagged for human acceptance.

### UAT Acceptance

Accepted on 2026-09-17 in `10-UAT.md`. The user selected option 1 after the scope was stated: all 13 items are objective repository-state prohibitions backed by the cited diff, source, documentation, wheel, and regression evidence. No implementation gap, experiential behavior, or external-system claim was auto-accepted.

### Gaps Summary

No implementation gap was found. Every roadmap criterion, CACH-07 behavior, D-01 through D-16 decision, artifact, key link, and behavioral gate is green. The 13 judgment-tier prohibitions have been accepted through UAT.

---

_Verified: 2026-09-17T18:58:39Z_
_Verifier: the agent (gsd-verifier)_
