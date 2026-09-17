# Phase 11: Clean Supplemental Documentation and Normalize Validation Evidence - Research

**Researched:** 2026-09-17
**Domain:** Python package-surface removal, documentation consolidation, and GSD/Nyquist evidence normalization
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

### Supplemental documentation
- **D-01:** Apply selective consolidation: move still-useful, verified material into canonical task/reference guides, repair only genuinely distinct guidance, and delete redundant or misleading supplemental documents.
- **D-02:** Consolidate unique verified platform material into `docs/RELEASE_QUALIFICATION.md`, then delete `docs/CROSS_PLATFORM_GUIDE.md` and `docs/WINDOWS_COMPATIBILITY.md`. Native Windows remains explicitly `UNAVAILABLE` / `NOT_QUALIFIED`.
- **D-03:** Evaluate `docs/PANDAS_COMPATIBILITY.md` and `docs/CUSTOM_METADATA.md` under the same selective-consolidation rule. Their content survives only where it adds verified value beyond current canonical format/catalog guidance.
- **D-04:** Prefer links to the four canonical executable examples. Any retained additional runnable snippet must execute directly or have an exact import/signature contract test.

### TensorFlow removal
- **D-05:** Remove dormant native TensorFlow support completely: handler implementation and exports, lazy-import machinery, configuration flags, optional dependency groups and lock entries, dedicated tests, packaging profiles, CI/qualification jobs, documentation, and current feature claims. Do not retain aliases, tombstones, dormant code, or a re-enable recipe. — **Reversibility:** costly — restoring support would require deliberately reintroducing the dependency, format handler, public/configuration surface, qualification matrix, and persisted-format support.
- **D-06:** Delete TensorFlow-specific documentation, including both the user guide and implementation-status document, rather than consolidating it into supported format documentation.
- **D-07:** Mark `.planning/seeds/SEED-005-remove-native-tensorflow-support.md` fulfilled and retain it as historical rationale with a Phase 11 resolution link so it cannot be promoted again.
- **D-08:** TensorFlow removal is a bounded package/handler cutover. It does not authorize changes to `BlobStore`, `UnifiedCache`, lifecycle authorities, obstore participation, or other retained handlers.

### Validation normalization
- **D-09:** Normalize Phases 1, 5, 6, 7, 8, and 9 using their existing evidence plus one bounded current non-live regression gate. Run phase-specific checks only where evidence is missing, stale, or contradicted.
- **D-10:** Update each existing `*-VALIDATION.md` in place to the current canonical schema (`status: validated`, accurate Nyquist fields, green or explicitly superseded task rows, commands, scope, and nonclaims). Do not create wrapper reports or a competing milestone-level source of validation truth.
- **D-11:** When an old row names a removed test, command, or feature, preserve a traceable supersession mapping to the current replacement evidence or explicit removal decision. Do not silently rewrite history or leave obsolete pending rows controlling discovery.
- **D-12:** Scoped validation is valid: real PostgreSQL/Amazon-S3 evidence, controlled-Linux performance, native Windows, and immutable publication remain visibly `NOT_QUALIFIED`, `NOT_PUBLISHED`, or deferred without blocking validation of the approved local/deterministic phase scope.

### Phase 3 evidence boundary
- **D-13:** Replace `03-VALIDATION.md` with a compact canonical validation record covering the approved SQLite/local-filesystem and single-process-memory scopes, ADR guarantee classes, qualified commit/results, controlling evidence chain, current regression confirmation, explicit nonclaims, and stop conditions. Preserve obsolete detail through Git history and dated evidence rather than embedding it as a live appendix.
- **D-14:** Phase 3's controlling historical evidence remains `03-25-SUMMARY.md` and `docs/phase3-direct-implementation-2026-09-06.md`, corroborated by later Phase 07.1/8 evidence and one finite current contract regression. Do not attempt to reproduce every original environment, timing, or detached-checkout detail.
- **D-15:** The current Phase 3 gate covers named lifecycle integrity, deterministic recovery, cache-over-store composition, and the frozen non-live repository suite once. Documented typed contention outcomes are valid; timing noise or universal contender success is not a reason to add coordination.
- **D-16:** If that gate exposes a genuine integrity or recovery defect, classify it against ADR 0001 and stop Phase 11. This phase may correct obsolete evidence references, but it may not repair production lifecycle code or launch an automatic review/fix cycle.
- **D-17:** State the original provenance plainly: Phase 3 used user-approved direct primary-agent implementation and exact-commit qualification rather than an independent verifier. Phase 11 confirms continued behavior but must not retroactively relabel that history.

### Completion and audit closure
- **D-18:** Final acceptance is a layered bounded gate: focused TensorFlow-removal/package contracts, exact documentation/reference scans, executable-example ownership, canonical validation discovery, finite Phase 3 contracts, scoped Ruff, fresh-wheel membership/metadata/import/round-trip inspection, and one frozen non-live regression suite.
- **D-19:** Refresh `.planning/v1.0-v1.0-MILESTONE-AUDIT.md` after the gates pass. Mark supplemental-documentation and validation-format debt resolved, record current Nyquist and package evidence, preserve all explicit deferrals, and issue the current milestone-completion verdict.
- **D-20:** The completion record must not convert unavailable external evidence into a pass or reopen `BACK-05`, `QUAL-06`, native Windows, or immutable publication.

### the agent's Discretion
- Choose the exact canonical destination or deletion outcome for unique pandas and custom-metadata material after checking it against current code and task guides.
- Choose the smallest exact test selectors and supersession mappings that satisfy the bounded validation gate without recreating obsolete suites.
- Choose the internal edit order for TensorFlow removal, manifest/lock convergence, documentation cleanup, validation normalization, and final audit refresh.

### Deferred Ideas (OUT OF SCOPE)
- Real PostgreSQL/Amazon-S3 qualification and immutable publication remain `SEED-007` / `BACK-05` work.
- Controlled-Linux performance qualification remains `SEED-006` / `QUAL-06` work.
- Native Windows lifecycle qualification remains Phase 999.1.
- Narwhals/dataframe expansion, XXH3 canonical-digest reconsideration, and broader format-handler developer tooling remain their existing future seeds/backlog work.
</user_constraints>

## Summary

Phase 11 should be planned as one direct product-surface cutover followed by evidence repair, not as a storage redesign. The TensorFlow surface is wider than its dormant handler: it spans runtime code, configuration, compatibility exports, two dependency-group declarations, lock metadata, package/profile tooling, CI, qualification evidence, dedicated and cross-cutting tests, current documentation, current codebase maps, `AGENTS.md`, and a dormant seed. The safe completion test is negative and source-free: the built wheel, installed metadata, imports, current docs, and qualification profiles must contain no TensorFlow capability, while retained handlers still pass their existing contracts. [VERIFIED: `src/cacheness/handlers.py:76-104,773-920,1294-1300,1323-1326,1357-1365,1603-1606`; `src/cacheness/config.py:211-248`; `src/cacheness/storage/handlers/__init__.py:1-97`; `pyproject.toml:18-44,47-70`; `.github/workflows/quality.yml:54-83`]

The documentation decision can be made prescriptively. Move only a short, tested pandas-format statement into `docs/API_REFERENCE.md`; delete `docs/PANDAS_COMPATIBILITY.md` after that move. Delete `docs/CUSTOM_METADATA.md` without consolidation because its current catalog operation is already owned by the API and BlobStore guides. Move only the truthful local-regression command and platform evidence boundary into `docs/RELEASE_QUALIFICATION.md`, then delete both platform supplements. Delete both TensorFlow documents outright. Keep the four executable examples as the only runnable example owners. [VERIFIED: `docs/API_REFERENCE.md:82-90`; `docs/BLOB_STORE.md:12-77`; `src/cacheness/handlers.py:327-410,495-557`; `tests/test_pandas_compatibility.py:12-58`; `tests/test_phase9_examples.py:17-23`; `docs/RELEASE_QUALIFICATION.md:1-103`]

Validation normalization is a schema migration of evidence, not a new verification campaign. Update the existing Phase 1/5/6/7/8/9 records in place, retain traceable supersession notes for removed SqlCache tests, and replace Phase 3's draft narrative with a compact record backed by its exact-commit ledger plus one finite current contract gate. If that focused gate reveals an integrity or recovery defect, stop the phase and classify it under ADR 0001; do not edit lifecycle code. The milestone audit is the final derived artifact and must be refreshed only after the package, documentation, validation-discovery, Phase 3, wheel, Ruff, and one frozen non-live suite are green. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:45-84`; `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-25-SUMMARY.md:1-31`; `docs/phase3-direct-implementation-2026-09-06.md:103-186`; `docs/adr/0001-topology-specific-storage-guarantees.md:1-236`]

**Primary recommendation:** implement the removal and documentation contracts first, normalize evidence second, run the bounded Phase 3 and final layered gates once, and regenerate the milestone audit last. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:76-84,208-213`]

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|---|---|---|---|
| TensorFlow handler/config/export removal | Package runtime boundary | Handler integration boundary | The removed class, lazy import, config flag, and compatibility export are package API surface; `BlobStore` remains untouched. [VERIFIED: `src/cacheness/handlers.py:76-104,773-920`; `src/cacheness/config.py:211-248`; `src/cacheness/storage/handlers/__init__.py:1-97`] |
| TensorFlow dependency and lock removal | Build / packaging | Installed-wheel verification | Extras are published package metadata; dependency groups support local workflows. [CITED: https://docs.astral.sh/uv/concepts/projects/dependencies/] |
| CI and qualification profile removal | Qualification tooling | Package tests | The current workflow and Phase 8 tools each encode a TensorFlow-specific profile/job. [VERIFIED: `.github/workflows/quality.yml:54-83`; `tools/run_phase8_platform_gates.py:20-31,192-220`; `tools/run_phase8_packaging.py:27-35,476-654`] |
| Supplemental-document consolidation | Documentation | Contract tests | Canonical guides own current claims; tests own exact inventory, references, and executable examples. [VERIFIED: `tests/test_phase9_documentation.py:239-274,336-349`; `tests/test_phase9_examples.py:17-23`] |
| Validation normalization | Planning evidence | Test/evidence commands | Existing `*-VALIDATION.md` files remain the canonical discovery targets; no wrapper truth source is allowed. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:48-58`] |
| Phase 3 regression confirmation | Test/evidence boundary | ADR classification | Production lifecycle code is explicitly outside Phase 11; a genuine defect stops the phase. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:60-72`; `docs/adr/0001-topology-specific-storage-guarantees.md:198-236`] |
| Milestone audit refresh | Derived planning artifact | Validation discovery | The audit consumes completed Phase 11 gates and must be written last. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:74-84`] |

## Project Constraints (from AGENTS.md)

- Use a direct pre-production cutover; do not add compatibility shims for the retired TensorFlow surface. Preserve explicit future migration/rebuild tooling boundaries. [VERIFIED: `AGENTS.md:12-16`]
- `BlobStore` remains the sole lifecycle coordinator; `UnifiedCache` remains policy-only. Do not make policy, projections, or removed handlers another lifecycle authority. [VERIFIED: `AGENTS.md:17-21,46-57`]
- Preserve safe parsing, path containment, signing, fail-closed integrity, immutable payload publication, exact cleanup, and topology-specific guarantees. Do not infer cross-resource ACID or universal availability. [VERIFIED: `AGENTS.md:22-29,91-103`]
- Keep Python `>=3.11`; use `uv` and the committed lockfile. Keep optional dependencies, guarded imports, published extras, tests, and documentation synchronized. [VERIFIED: `AGENTS.md:70-82`; `pyproject.toml:6-16`]
- The current `AGENTS.md` statement that the six extras include `tensorflow` is itself part of the removal surface and must be updated in the same cutover; otherwise future agents would be instructed to preserve a retired feature. The current verbatim extras are `"recommended"`, `"dataframes"`, `"tensorflow"`, `"s3"`, `"postgresql"`, and `"cloud"`. [VERIFIED: `AGENTS.md:70-82`; `pyproject.toml:18-44`]
- Read ADR 0001 before any lifecycle/topology/concurrency/recovery work. Stop rather than adding a lock, queue, bootstrap state, projection gate, or cross-resource commit protocol to strengthen a guarantee. [VERIFIED: `AGENTS.md:89-103`]
- Use `test_<subject>.py`, real temporary local stores, small external-boundary fakes, and explicit live-service markers. Run scoped Ruff on touched Python files. [VERIFIED: `AGENTS.md:75-87`]
- Final project validation uses `uv run --isolated --all-extras --group dev --frozen pytest ...`; manifest work requires `uv lock --check`; fresh-wheel qualification must inspect the artifact and installed metadata and exercise local `BlobStore`/`UnifiedCache` round trips. [VERIFIED: `AGENTS.md:117-124`]
- Preserve ignored user-owned files, unrelated planning configuration, and historical dated planning/audit material. [VERIFIED: `AGENTS.md:101-103,125-125`]

## Standard Stack

### Core

| Tool | Verified Version | Purpose | Why Standard |
|---|---:|---|---|
| Python | project pin `"3.13"`; package floor `">=3.11"` | Runtime and tests | These are the committed project constraints. [VERIFIED: `.python-version:1`; `pyproject.toml:6-16`] |
| uv | 0.12.12 | Lock convergence, isolated test environments, wheel build/install qualification | The repository mandates uv and has a currently fresh lockfile; `uv lock --check` resolved 119 packages on 2026-09-17. [VERIFIED: local `uv --version` and `uv lock --check` executed 2026-09-17] |
| pytest | 8.4.1 | Focused contracts and frozen non-live suite | Existing tests and marker configuration already define the qualification surface. [VERIFIED: local `.venv/bin/python -m pytest --version`; `pyproject.toml:92-120`] |
| Ruff | 0.12.9 | Scoped lint of touched Python | Project instructions require no increase to the Ruff baseline. [VERIFIED: local `.venv/bin/ruff --version`; `AGENTS.md:75-87`] |
| `uv_build` | `">=0.8.15,<0.9.0"` | Build source and wheel artifacts | This is the committed build backend and range. [VERIFIED: `pyproject.toml:72-74`] |

### Supporting

| Asset | Purpose | When to Use |
|---|---|---|
| `tests/packaging/test_wheel_matrix.py` | Source-free wheel membership, metadata, imports, and local round trips | Extend/invert for complete TensorFlow absence. [VERIFIED: `tests/packaging/test_wheel_matrix.py:288-430`] |
| `tests/test_phase9_documentation.py` | Current-document inventory, references, and link contracts | Enforce supplemental deletion, canonical destinations, and absence of current TensorFlow claims. [VERIFIED: `tests/test_phase9_documentation.py:239-274,336-349`] |
| `tests/test_phase9_examples.py` | Four canonical executable examples | Keep example ownership exact and avoid duplicate snippets. [VERIFIED: `tests/test_phase9_examples.py:17-23`] |
| Phase 8 qualification tools | Deterministic local/package evidence | Remove the TensorFlow profile rather than creating a Phase 11 parallel framework. [VERIFIED: `tools/phase8_evidence.py:117-125`; `tools/run_phase8_packaging.py:27-35`; `tools/run_phase8_platform_gates.py:20-31`] |
| GSD canonical `VALIDATION.md` schema | Validation discovery | Normalize existing files to `status: validated`, `nyquist_compliant: true`, and accurate task rows. [VERIFIED: `/Users/akriz/.codex/gsd-core/templates/VALIDATION.md:1-78`; `/Users/akriz/.codex/gsd-core/workflows/audit-milestone.md:167-178`] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|---|---|---|
| Direct removal | Deprecation alias/tombstone | Forbidden by D-05 and would preserve an unsupported public and persisted-format surface. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:39-46`] |
| Updating existing validation files | Wrapper reports | Forbidden by D-10; wrappers would create competing evidence truth. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:48-55`] |
| Existing packaging/docs contracts | New Phase 11 validation framework | Duplicates mature repository contracts and expands the surface without benefit. [VERIFIED: `tests/packaging/test_wheel_matrix.py:288-430`; `tests/test_phase9_documentation.py:239-274`] |
| One finite Phase 3 gate | Reproducing every Phase 3 timing/environment | Explicitly rejected by D-14 and risks reopening concurrency design. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:60-72`] |

**Installation:** none. This phase removes an optional dependency/profile and introduces no external package. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:39-46`]

**Lock procedure:** edit `pyproject.toml`, run plain `uv lock` so the existing resolution is preferred, then run `uv lock --check`; do not hand-edit `uv.lock` and do not use `--upgrade` unless a separate dependency update is authorized. [CITED: https://docs.astral.sh/uv/concepts/projects/sync/]

## Package Legitimacy Audit

Not applicable: Phase 11 installs no package. It removes TensorFlow from the existing optional dependency surface and therefore requires negative wheel/metadata verification rather than a package-legitimacy gate. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:39-46`]

## Architecture Patterns

### System Architecture Diagram

```text
Current source surfaces
  handlers/config/exports ─┐
  pyproject + uv.lock ─────┼──> direct TensorFlow cutover
  CI + qualification ─────┤          │
  docs + current maps ─────┘          v
                               negative source contracts
                               + source-free fresh wheel
                                         │
Supplemental docs ──> verify unique value ├──> canonical docs + exact scans
                         │               │
                         └──> delete stale/redundant docs
                                         │
Historical evidence ──> normalize existing VALIDATION.md files
                         │               │
Phase 3 ledger ──────────┴──> finite lifecycle contract gate
                                         │
                              defect? ─yes─> STOP under ADR 0001
                                         │ no
                                         v
                           one frozen non-live suite
                                         │
                                         v
                            refresh milestone audit last
```

This flow keeps runtime implementation, package truth, current documentation, evidence normalization, and the final derived audit in a one-way order; no validation artifact becomes a reason to mutate lifecycle code. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:39-84`]

### Recommended Project Structure

```text
src/cacheness/                         # retained runtime, minus TensorFlow surface
tests/
├── packaging/test_wheel_matrix.py     # source-free negative package contract
├── qualification/                     # retained local qualification contracts
├── test_phase9_documentation.py       # current docs/reference inventory
└── test_phase9_examples.py            # four executable example owners
docs/
├── API_REFERENCE.md                   # concise verified pandas format note
├── RELEASE_QUALIFICATION.md           # sole platform evidence owner
└── adr/0001-...md                     # unchanged lifecycle guardrail
.planning/phases/*/*-VALIDATION.md     # normalized in place
.planning/v1.0-v1.0-MILESTONE-AUDIT.md # refreshed last
```

The mapping above names existing paths and intended ownership; it does not authorize a new source tree or validation wrapper. [VERIFIED: `tests/test_phase9_documentation.py:239-274`; `tests/test_phase9_examples.py:17-23`; `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:48-58,76-84`]

### Pattern 1: Complete Negative Capability Cutover

**What:** remove the capability from runtime, exports, configuration, dependency metadata, lock state, profiles, CI, tests, docs, and current claims in one bounded change. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:39-46`]

**When to use:** for this pre-production TensorFlow removal only. Do not generalize it into a lifecycle migration. [VERIFIED: `AGENTS.md:12-29`]

**Implementation checklist:**

1. Remove `_get_tensorflow`, `TensorFlowTensorHandler`, its identity/member references, the config flag and valid-name entry, and the compatibility export. [VERIFIED: `src/cacheness/handlers.py:76-104,773-920,1294-1300,1323-1326,1357-1365,1603-1606`; `src/cacheness/config.py:211-248`; `src/cacheness/storage/handlers/__init__.py:1-97`]
2. Remove `[project.optional-dependencies].tensorflow`, `[dependency-groups].tensorflow`, and resulting lock entries using `uv lock`; remove the TensorFlow package/profile/CI paths from the existing evidence tools. The current published extras quote is `"recommended"`, `"dataframes"`, `"tensorflow"`, `"s3"`, `"postgresql"`, and `"cloud"`. [VERIFIED: `pyproject.toml:18-44,47-70`; `uv.lock:201-253,2671-2720`; `tools/phase8_evidence.py:117-125`; `tools/run_phase8_packaging.py:27-35`; `.github/workflows/quality.yml:54-83`]
3. Delete `tests/test_tensorflow_handler.py`, invert cross-cutting package/qualification assertions to prove absence, and repair workflow parsers that currently use the TensorFlow job as a structural delimiter. [VERIFIED: `tests/test_tensorflow_handler.py:1-430`; `tests/packaging/test_wheel_matrix.py:288-430`; `tests/qualification/test_phase8_quality_workflow.py:57-79`; `tests/test_phase9_quality_workflow.py:36-45`]
4. Delete both TensorFlow documents; update current package/docs/codebase maps and fulfill SEED-005 without rewriting historical phase artifacts. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:39-46,202-213`; `.planning/seeds/SEED-005-remove-native-tensorflow-support.md:1-8`]

### Pattern 2: Selective Documentation Consolidation

**What:** promote only verified unique content into the existing canonical owner, then delete the stale supplemental owner. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:31-37`]

**Prescriptive disposition:**

| Supplemental document | Disposition | Verified survivor |
|---|---|---|
| `docs/PANDAS_COMPATIBILITY.md` | Consolidate minimally, then delete | Add a concise API-reference statement that Series/DataFrame handlers use Parquet and that the current compatibility test preserves representative index/name/dtypes. [VERIFIED: `src/cacheness/handlers.py:327-410,495-557`; `tests/test_pandas_compatibility.py:12-58`] |
| `docs/CUSTOM_METADATA.md` | Delete without consolidation | Catalog metadata and supported update/query operations are already owned by the API and BlobStore guides. [VERIFIED: `docs/API_REFERENCE.md:82-90`; `docs/BLOB_STORE.md:12-77`] |
| `docs/CROSS_PLATFORM_GUIDE.md` | Move only exact local regression invocation/boundary into release qualification, then delete | `docs/RELEASE_QUALIFICATION.md` already owns the evidence matrix and nonclaims. [VERIFIED: `docs/RELEASE_QUALIFICATION.md:1-103`] |
| `docs/WINDOWS_COMPATIBILITY.md` | Delete after release-qualification wording is explicit | Windows must remain `UNAVAILABLE` / `NOT_QUALIFIED`; portable wheels are not native qualification. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:31-34,79-84`] |
| `docs/TENSORFLOW_TENSOR_GUIDE.md`, `docs/TENSORFLOW_HANDLER_STATUS.md` | Delete | D-06 forbids consolidation or a re-enable recipe. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:39-45`] |

### Pattern 3: In-Place Evidence Schema Normalization

**What:** update each existing validation file to the canonical schema, preserving scope, provenance, commands, nonclaims, and explicit supersession mappings. The discovery rules classify a phase as compliant only when the record says `status: validated`, `nyquist_compliant: true`, and task rows are green. [VERIFIED: `/Users/akriz/.codex/gsd-core/templates/VALIDATION.md:1-78`; `/Users/akriz/.codex/gsd-core/workflows/audit-milestone.md:167-178`]

**Known repair points:**

- Phase 1 references removed `tests/test_sql_cache.py` and `tests/test_sql_cache_failure_contract.py`; replace those controlling rows with a traceable supersession link to the Phase 10 removal decision/negative contract. [VERIFIED: `.planning/phases/01-compatibility-and-security-baseline/01-VALIDATION.md:25-47,89-98`; `.planning/phases/10-remove-sqlcache-pull-through-subsystem/10-CONTEXT.md:1-120`]
- Phase 6 references removed `tests/test_sql_cache.py`; map it to the same explicit removal evidence rather than silently inventing a passing historical test. [VERIFIED: `.planning/phases/06-unifiedcache-policy-composition/06-VALIDATION.md:48-62`; `.planning/phases/10-remove-sqlcache-pull-through-subsystem/10-CONTEXT.md:1-120`]
- Phase 3 is not incrementally patched: replace it with the compact D-13 record and retain obsolete detail through Git history and its dated ledger. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:60-72`]
- Phase 9's draft/pending task rows must become current green rows after the documentation/package contracts pass. [VERIFIED: `.planning/phases/09-adoption-and-release-surface-closure/09-VALIDATION.md:1-120`]

### Anti-Patterns to Avoid

- **Residual compatibility marker:** do not leave a handler name, export, config flag, empty extra, profile, or documentation tombstone. Any one of these keeps the retired surface discoverable. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:39-46`]
- **Hand-edited lockfile:** `uv.lock` is generated convergence evidence, not a text-cleanup target. Use `uv lock`, then `uv lock --check`. [CITED: https://docs.astral.sh/uv/concepts/projects/sync/]
- **Current-only grep across all history:** dated phase ledgers and audit artifacts preserve historical truth. Scope absence scans to runtime, packaging, CI, current docs/maps, and current tests; do not rewrite historical evidence. [VERIFIED: `AGENTS.md:101-103`; `.planning/phases/10-remove-sqlcache-pull-through-subsystem/10-CONTEXT.md:1-120`]
- **False validation through relabeling:** changing frontmatter without green/current or explicitly superseded rows is not normalization. [VERIFIED: `/Users/akriz/.codex/gsd-core/workflows/audit-milestone.md:167-178`]
- **Universal concurrency success as an acceptance criterion:** typed contention outcomes are allowed; timing noise is not authorization for new coordination. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:66-72`; `docs/adr/0001-topology-specific-storage-guarantees.md:198-236`]
- **Audit-first editing:** the milestone audit is derived output; updating it before package/docs/validation gates pass can record a verdict unsupported by evidence. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:76-84`]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---|---|---|---|
| Lockfile cleanup | Manual TOML/package-block surgery | `uv lock` + `uv lock --check` | uv preserves the existing resolution where possible and checks lock freshness. [CITED: https://docs.astral.sh/uv/concepts/projects/sync/] |
| Installed package truth | Source-tree import/grep alone | Existing source-free wheel matrix | Published extras become `Provides-Extra`; optional requirements become conditional `Requires-Dist`, so installed metadata must be inspected. [CITED: https://packaging.python.org/en/latest/specifications/core-metadata/] |
| Documentation inventory | A second docs registry | Extend `test_phase9_documentation.py` | The current contract already owns document/reference checks. [VERIFIED: `tests/test_phase9_documentation.py:239-274,336-349`] |
| Example framework | New snippets or runners | Four canonical example scripts and their tests | D-04 assigns runnable ownership to the existing examples. [VERIFIED: `tests/test_phase9_examples.py:17-23`; `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:31-37`] |
| Validation aggregator | Wrapper report or parallel ledger | Existing per-phase `*-VALIDATION.md` files | D-10 requires in-place canonical discovery. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:48-55`] |
| Lifecycle regression fix | New lock/queue/retry/coordinator | Stop and classify under ADR 0001 | Production lifecycle repair is outside Phase 11. [VERIFIED: `AGENTS.md:89-103`; `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:66-72`] |

**Key insight:** the difficult part is proving absence and preserving evidence semantics across several surfaces; repository-native wheel, docs, validation, and lifecycle contracts already handle those edge cases better than new Phase 11 machinery. [VERIFIED: `tests/packaging/test_wheel_matrix.py:288-430`; `tests/test_phase9_documentation.py:239-274`; `/Users/akriz/.codex/gsd-core/templates/VALIDATION.md:1-78`]

## Runtime State Inventory

This phase removes a persisted-format handler, so runtime state was audited even though the project is pre-production. [VERIFIED: `src/cacheness/handlers.py:773-920`; `AGENTS.md:12-16`]

| Category | Items Found | Action Required |
|---|---|---|
| Stored data | No tracked repository fixture or payload with the `.b2tr` suffix was found. The retired verbatim identities are `data_type = "tensorflow_tensor"`, `payload_format = "blosc2_tensor"`, and `file_suffix = "b2tr"`. [VERIFIED: repository `rg`/`find` audit on 2026-09-17; `src/cacheness/handlers.py:773-920`] | No data migration is planned. State explicitly that any untracked pre-production manifest using those exact identities becomes unreadable after the cutover; D-05 forbids compatibility/tombstone support. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:39-46`] |
| Live service config | No live-service UI/database configuration for TensorFlow was found; the only active service-like declaration is the tracked `tensorflow-compatible` CI job. [VERIFIED: repository current-surface scan on 2026-09-17; `.github/workflows/quality.yml:54-83`] | Remove the tracked CI job and dependent workflow assertions; no live-service migration. |
| OS-registered state | None found in repository or project instructions. [VERIFIED: repository current-surface scan on 2026-09-17; `AGENTS.md:1-129`] | None. Do not add OS cleanup work. |
| Secrets/env vars | No TensorFlow-specific secret or environment-variable name was found in `src/`, `tests/`, `tools/`, or `.github/`. [VERIFIED: repository current-surface scan on 2026-09-17] | None. Keep scans bounded to current surfaces. |
| Build artifacts / installed packages | Ignored `tests/__pycache__` entries contain stale test bytecode; `.venv` uses Python 3.11.16 and does not have TensorFlow importable. `.venv` and `__pycache__` are ignored. [VERIFIED: local filesystem/import audit on 2026-09-17; `.gitignore:1-120`] | Do not delete user/local environment state as product work. Build a fresh isolated source-free wheel and verify its membership and metadata after lock convergence. |

## Common Pitfalls

### Pitfall 1: Removing the class but retaining package truth

**What goes wrong:** the wheel still advertises `tensorflow`, the lock retains its graph, or qualification tooling still presents a supported profile. [VERIFIED: `uv.lock:201-253,2671-2720`; `tools/run_phase8_packaging.py:27-35,476-654`]

**Why it happens:** TensorFlow is encoded in independent runtime, dependency, generated-lock, CI, tooling, and test surfaces. [VERIFIED: `pyproject.toml:18-44,47-70`; `.github/workflows/quality.yml:54-83`; `tests/packaging/test_wheel_matrix.py:288-430`]

**How to avoid:** make the negative source scan and source-free installed-metadata check explicit acceptance criteria. The installed distribution must not emit `Provides-Extra: tensorflow` or a TensorFlow conditional `Requires-Dist`. [CITED: https://packaging.python.org/en/latest/specifications/core-metadata/]

**Warning signs:** `rg` still finds TensorFlow in current runtime/package/CI/docs/test-map surfaces, or the wheel metadata exposes the removed extra. [VERIFIED: repository current-surface scan on 2026-09-17]

### Pitfall 2: Deleting a CI job breaks structural tests

**What goes wrong:** Phase 9's workflow parser currently uses `tensorflow-compatible` as a delimiter, so deleting the job can break tests unrelated to TensorFlow behavior. [VERIFIED: `tests/test_phase9_quality_workflow.py:36-45`]

**How to avoid:** refactor the parser to anchor on retained job structure and update Phase 8 workflow contracts in the same change. [VERIFIED: `tests/qualification/test_phase8_quality_workflow.py:57-79,181-181`]

### Pitfall 3: Consolidating stale snippets into canonical docs

**What goes wrong:** removed APIs and unqualified platform/performance claims gain new authority merely because they were copied. [VERIFIED: `docs/PANDAS_COMPATIBILITY.md:1-240`; `docs/CUSTOM_METADATA.md:1-260`; `docs/WINDOWS_COMPATIBILITY.md:1-260`]

**How to avoid:** retain only behavior directly demonstrated by current source/tests, link to canonical executable examples, and delete the rest. [VERIFIED: `tests/test_pandas_compatibility.py:12-58`; `tests/test_phase9_examples.py:17-23`]

### Pitfall 4: Treating frontmatter edits as Nyquist compliance

**What goes wrong:** a file says `validated` while pending or dead task rows still control discovery. [VERIFIED: `/Users/akriz/.codex/gsd-core/workflows/audit-milestone.md:167-178`]

**How to avoid:** every task row must be green or explicitly superseded, and each record must include runnable commands, scope, nonclaims, and an audit trail. [VERIFIED: `/Users/akriz/.codex/gsd-core/templates/VALIDATION.md:1-78`; `/Users/akriz/.codex/gsd-core/workflows/validate-phase.md:131-149`]

### Pitfall 5: Silently rewriting removed SqlCache evidence

**What goes wrong:** Phase 1/6 appear to have always referenced a current test, erasing why their old commands disappeared. [VERIFIED: `.planning/phases/01-compatibility-and-security-baseline/01-VALIDATION.md:25-47`; `.planning/phases/06-unifiedcache-policy-composition/06-VALIDATION.md:48-62`]

**How to avoid:** add explicit `old evidence -> Phase 10 removal/negative contract` mappings and retain the original historical commit/path in prose. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:52-55`]

### Pitfall 6: Turning Phase 3 evidence repair into lifecycle implementation

**What goes wrong:** timing variability or a typed contender result prompts another lock, retry, or coordination loop. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:66-72`]

**How to avoid:** run the named finite gate once; if integrity/recovery is genuinely broken, stop and classify under ADR 0001. Progress/performance variation is not itself an integrity defect. [VERIFIED: `docs/adr/0001-topology-specific-storage-guarantees.md:198-236`]

### Pitfall 7: Overclaiming deferred qualification

**What goes wrong:** a green local/non-live suite is converted into real PostgreSQL/S3, Linux performance, Windows, or publication qualification. [VERIFIED: `.planning/v1.0-v1.0-MILESTONE-AUDIT.md:185-200`]

**How to avoid:** carry exact `NOT_QUALIFIED`, `NOT_PUBLISHED`, and deferred statements into each normalized validation and the final audit. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:56-58,79-84`]

## Code Examples

### Bounded Phase 3 Contract Gate

```bash
uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false \
  tests/test_phase3_gap_acceptance.py \
  tests/test_phase3_local_workflows.py \
  tests/test_blob_store_atomic_lifecycle.py \
  tests/test_blob_store_reconciliation.py \
  tests/test_unified_cache_lifecycle_authority.py \
  tests/test_integration.py \
  -x
```

Every listed path exists. Together they cover the current Phase 3 node inventory, local workflows, atomic lifecycle, reconciliation, cache-over-store authority, and integration surface. If this command exposes a genuine integrity or recovery defect, stop Phase 11. [VERIFIED: `tests/test_phase3_gap_acceptance.py:15-38,110-150`; `tests/test_phase3_local_workflows.py:1-420`; `tests/test_blob_store_atomic_lifecycle.py:1-420`; `tests/test_blob_store_reconciliation.py:1-420`; `tests/test_unified_cache_lifecycle_authority.py:1-360`; `tests/test_integration.py:1-420`]

### One Final Frozen Non-Live Gate

```bash
uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false \
  -m 'not (live_postgresql or live_aws_s3 or live_remote)'
```

The three quoted marker values are declared in the repository's pytest configuration and form the current external-service exclusion boundary. [VERIFIED: `pyproject.toml:92-120`]

### Lock Convergence

```bash
uv lock
uv lock --check
```

Plain `uv lock` updates the lock while preferring already locked versions when possible; `--check` verifies freshness. [CITED: https://docs.astral.sh/uv/concepts/projects/sync/]

### Current-Surface Absence Scan

```bash
rg -n -i 'tensorflow|tensorflow_tensor|blosc2_tensor|b2tr' \
  AGENTS.md pyproject.toml uv.lock src tests tools .github docs/README.md \
  docs/API_REFERENCE.md docs/BLOB_STORE.md docs/RELEASE_QUALIFICATION.md \
  .planning/codebase .planning/seeds/SEED-005-remove-native-tensorflow-support.md
```

This scan is intentionally bounded to current product surfaces and the fulfilled seed. Historical phase evidence is excluded so historical truth is not rewritten. [VERIFIED: `AGENTS.md:101-103`; `.planning/phases/10-remove-sqlcache-pull-through-subsystem/10-CONTEXT.md:1-120`]

## State of the Art

| Old Approach | Current Phase 11 Approach | Impact |
|---|---|---|
| Disabled registration with retained TensorFlow handler/dependency/profile | Complete direct removal with negative installed-wheel proof | Package truth matches supported product surface. [VERIFIED: `src/cacheness/handlers.py:1294-1300,1323-1326`; `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:39-46`] |
| Multiple platform supplemental guides | `docs/RELEASE_QUALIFICATION.md` as sole detailed platform evidence owner | Removes contradictory Windows/support language. [VERIFIED: `docs/RELEASE_QUALIFICATION.md:1-103`; `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:31-34`] |
| Draft/complete legacy validation statuses and pending rows | Canonical `validated` records with green/superseded rows and explicit nonclaims | GSD milestone discovery reflects current evidence without erasing history. [VERIFIED: `/Users/akriz/.codex/gsd-core/workflows/audit-milestone.md:167-178`; `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:48-58`] |
| Phase 3's obsolete live narrative as current control | Compact record pointing to exact-commit ledger plus one current finite regression | Preserves provenance and prevents accidental lifecycle authorization. [VERIFIED: `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-25-SUMMARY.md:1-31`; `docs/phase3-direct-implementation-2026-09-06.md:103-186`] |

**Deprecated/outdated:**

- Native TensorFlow handler/package/qualification support is removed, not deprecated. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:39-46`]
- The supplemental platform, pandas, custom-metadata, and TensorFlow documents cease to be current evidence owners after the prescribed consolidation/deletion. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:31-45,191-201`]
- Legacy validation rows naming removed SqlCache tests remain historical context only after explicit supersession mapping. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:48-58`]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|---|---|---|
| — | None. Recommendations are derived from locked decisions, opened repository sources, executed local probes, GSD schema sources, or official uv/Python Packaging documentation. | — | — |

## Open Questions

1. **What is the final milestone verdict?**
   - What we know: the current audit is `tech_debt`, and Phase 11 is designed to close its supplemental-documentation and validation-format debt while preserving explicit deferrals. [VERIFIED: `.planning/v1.0-v1.0-MILESTONE-AUDIT.md:1-35,142-200`]
   - What's unclear: the current verdict cannot be known until the final layered gate completes.
   - Recommendation: derive and record the verdict in the audit-refresh task only after every Phase 11 acceptance gate is green; do not preselect it in the plan. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:76-84`]

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|---|---|---:|---|---|
| uv | lock/build/isolated tests | ✓ | 0.12.12 | none; mandated by project [VERIFIED: local probe 2026-09-17; `AGENTS.md:70-73`] |
| Project Python pin | isolated execution | ✓ | 3.13.15 selected by uv | package floor remains 3.11 [VERIFIED: `uv lock --check` output 2026-09-17; `.python-version:1`; `pyproject.toml:6-16`] |
| pytest | validation gates | ✓ | 8.4.1 | none [VERIFIED: local `.venv/bin/python -m pytest --version` 2026-09-17] |
| Ruff | scoped lint | ✓ | 0.12.9 | none [VERIFIED: local `.venv/bin/ruff --version` 2026-09-17] |
| Git | evidence provenance/history | ✓ | 2.55.0 | none [VERIFIED: local `git --version` 2026-09-17] |
| TensorFlow | removed capability | ✗ | not importable in `.venv` | no fallback; absence is desired [VERIFIED: local import probe 2026-09-17] |
| Live PostgreSQL/S3, controlled Linux, native Windows | deferred qualification | not required | — | retain explicit nonclaims [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:56-58,79-84,217-222`] |

**Missing dependencies with no fallback:** none for the approved local Phase 11 scope. [VERIFIED: local environment audit 2026-09-17]

**Missing dependencies with fallback:** none; deliberately unavailable external qualification environments remain deferred rather than simulated. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:217-222`]

## Validation Architecture

### Test Framework

| Property | Value |
|---|---|
| Framework | pytest 8.4.1 [VERIFIED: local `.venv/bin/python -m pytest --version` 2026-09-17] |
| Config file | `pyproject.toml` [VERIFIED: `pyproject.toml:92-120`] |
| Quick run command | `uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false <touched selectors> -x` [VERIFIED: `AGENTS.md:117-124`] |
| Full suite command | `uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false -m 'not (live_postgresql or live_aws_s3 or live_remote)'` [VERIFIED: `pyproject.toml:92-120`; `AGENTS.md:117-124`] |

### Decisions to Test Map

| Decision | Behavior | Test Type | Automated Command / Evidence | File Exists? |
|---|---|---|---|---|
| D-05–D-08 | TensorFlow absent from runtime/config/exports and retained handlers unaffected | unit + source contract | focused handler/config tests plus bounded `rg` absence scan | Existing retained tests; ❌ Wave 0 negative contract extension [VERIFIED: `tests/test_tensorflow_handler.py:1-430`; `tests/packaging/test_wheel_matrix.py:288-430`] |
| D-05 | No TensorFlow extra, requirement, member, import, or profile in built installation | packaging integration | extend `tests/packaging/test_wheel_matrix.py`, then run its exact selector in isolated mode | ✅ existing file; ❌ new negative assertions [VERIFIED: `tests/packaging/test_wheel_matrix.py:288-430`] |
| D-05 | No TensorFlow CI/qualification profile; retained jobs parse correctly | contract | `tests/qualification/test_phase8_quality_workflow.py`, `tests/qualification/test_phase8_platform.py`, `tests/test_phase9_quality_workflow.py` | ✅ [VERIFIED: `tests/qualification/test_phase8_quality_workflow.py:57-79`; `tests/qualification/test_phase8_platform.py:96-140`; `tests/test_phase9_quality_workflow.py:36-45`] |
| D-01–D-04, D-06 | Supplemental docs removed, destinations current, references exact | documentation contract | `tests/test_phase9_documentation.py` and `tests/test_phase9_examples.py` | ✅ existing files; ❌ Wave 0 inventory/absence assertions [VERIFIED: `tests/test_phase9_documentation.py:239-274,336-349`; `tests/test_phase9_examples.py:17-23`] |
| D-07 | Seed cannot be promoted again | planning contract | assert seed frontmatter is fulfilled and contains Phase 11 resolution link | ❌ Wave 0 assertion [VERIFIED: `.planning/seeds/SEED-005-remove-native-tensorflow-support.md:1-8`] |
| D-09–D-12 | Target validations discover as compliant and retain supersession/nonclaims | schema/contract | parse target `*-VALIDATION.md` files using canonical frontmatter/task rules | ❌ Wave 0 canonical validation-discovery test [VERIFIED: `/Users/akriz/.codex/gsd-core/workflows/audit-milestone.md:167-178`] |
| D-13–D-17 | Compact Phase 3 record plus named behavior remains valid | focused regression | bounded six-file Phase 3 command in Code Examples | ✅ tests exist; validation record rewrite is implementation [VERIFIED: `tests/test_phase3_gap_acceptance.py:15-38,110-150`] |
| D-18 | Scoped Ruff, lock, wheel, docs, validation, Phase 3, and one non-live suite pass | layered phase gate | commands in this section and final plan task | ✅ infrastructure exists [VERIFIED: `AGENTS.md:117-124`; `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:76-78`] |
| D-19–D-20 | Audit refreshed last without promoting deferrals | artifact contract/manual semantic review | rerun canonical milestone discovery, then inspect explicit deferral table and verdict | ❌ Wave 0 audit assertions; semantic verdict remains derived [VERIFIED: `.planning/v1.0-v1.0-MILESTONE-AUDIT.md:167-200`] |

### Sampling Rate

- **Per task commit:** run the narrow touched selectors plus scoped Ruff for touched Python files; run `uv lock --check` after manifest/lock work. [VERIFIED: `AGENTS.md:75-87,117-124`]
- **Per wave merge:** run the affected package/docs/qualification/validation contract cluster, not the full suite. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:48-58,76-78`]
- **Phase 3 gate:** run the finite six-file command once after runtime/package removal stabilizes; stop on a genuine integrity/recovery defect. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:60-72`]
- **Phase gate:** build and inspect a fresh source-free wheel, then run the frozen non-live suite exactly once before refreshing the audit. [VERIFIED: `AGENTS.md:117-124`; `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:76-84`]

### Wave 0 Gaps

- [ ] Extend `tests/packaging/test_wheel_matrix.py` with negative membership, metadata, import/export, optional-extra, and local round-trip assertions for the removed surface. [VERIFIED: `tests/packaging/test_wheel_matrix.py:288-430`]
- [ ] Extend `tests/test_phase9_documentation.py` with the exact supplemental-document deletion set, bounded current-reference scan, canonical destination assertions, and no current TensorFlow claims. [VERIFIED: `tests/test_phase9_documentation.py:239-274,336-349`]
- [ ] Add/extend a planning contract for the fulfilled SEED-005 frontmatter/resolution link and canonical validation discovery of Phases 1, 3, 5, 6, 7, 8, and 9. [VERIFIED: `.planning/seeds/SEED-005-remove-native-tensorflow-support.md:1-8`; `/Users/akriz/.codex/gsd-core/workflows/audit-milestone.md:167-178`]
- [ ] Update workflow/profile contracts before deleting the TensorFlow job/profile so failures localize to the intended cutover. [VERIFIED: `tests/qualification/test_phase8_quality_workflow.py:57-79`; `tests/qualification/test_phase8_platform.py:96-140`; `tests/test_phase9_quality_workflow.py:36-45`]
- [ ] Add audit assertions that debt is resolved while `BACK-05`, `QUAL-06`, native Windows, and immutable publication remain deferred/nonqualified. [VERIFIED: `.planning/v1.0-v1.0-MILESTONE-AUDIT.md:185-200`; `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:79-84`]

## Recommended Plan Decomposition

1. **Wave 0 — encode negative and discovery contracts.** Add the package, docs, workflow/profile, seed, validation-discovery, and audit-boundary assertions above. [VERIFIED: Validation Architecture sources]
2. **Wave 1 — remove the runtime/package surface.** Remove handler/config/export code and the dedicated test; remove manifest groups, converge the lock, update package/qualification tools and CI, repair dependent tests, and update `AGENTS.md` plus current codebase maps. Do not touch `BlobStore`, `UnifiedCache`, authorities, obstore, or other handlers. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:39-46`]
3. **Wave 1 — consolidate/delete current documentation.** Apply the prescriptive disposition table, keep only the four executable examples, delete both TensorFlow documents, and fulfill SEED-005 with a Phase 11 resolution link. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:31-45`]
4. **Wave 2 — normalize Phase 1/5/6/7/8/9 evidence in place.** Use existing evidence plus explicit supersession mappings; retain every external/platform/performance nonclaim. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:48-58`]
5. **Wave 2 stop gate — confirm Phase 3.** Run the finite focused command. On green, replace `03-VALIDATION.md` with the compact scoped record; on genuine integrity/recovery failure, stop and report classification under ADR 0001. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:60-72`]
6. **Wave 3 — layered acceptance.** Run focused contracts, exact scans, example ownership, validation discovery, scoped Ruff, `uv lock --check`, fresh source-free wheel qualification, and one frozen non-live suite. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:76-78`; `AGENTS.md:117-124`]
7. **Wave 3 — refresh the milestone audit last.** Derive its current verdict, mark only the targeted debt resolved, preserve all deferrals, and rerun audit contracts. [VERIFIED: `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:79-84`]

## Security Domain

Security enforcement is enabled because `.planning/config.json` does not disable it. [VERIFIED: `.planning/config.json:1-120`]

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---|---|---|
| V2 Authentication | no | No authentication surface changes in this phase. [VERIFIED: phase boundary in `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:8-24`] |
| V3 Session Management | no | No session surface changes. [VERIFIED: same phase boundary] |
| V4 Access Control | no | No authorization boundary changes. [VERIFIED: same phase boundary] |
| V5 Input Validation | yes | Preserve existing handler registry validation, safe parsing, path containment, and fail-closed boundaries while removing only TensorFlow members. [VERIFIED: `AGENTS.md:22-24,46-57`; `src/cacheness/config.py:211-248`] |
| V6 Cryptography | yes, unchanged | Preserve signed manifest digest/size as canonical verification; do not reinterpret ETag/version evidence. [VERIFIED: `AGENTS.md:97-99`] |

### Known Threat Patterns for This Phase

| Pattern | STRIDE | Standard Mitigation |
|---|---|---|
| Stale handler/export/config identity leaves retired capability reachable | Elevation of privilege / Tampering | Exact current-surface absence scan plus import/export/config negative tests. [VERIFIED: `src/cacheness/handlers.py:773-920,1603-1606`; `src/cacheness/config.py:211-248`] |
| Wheel still advertises or installs removed dependency graph | Tampering / Supply-chain exposure | Source-free wheel membership and `Provides-Extra`/`Requires-Dist` inspection. [CITED: https://packaging.python.org/en/latest/specifications/core-metadata/] |
| Deleted docs leave dangling links or unsupported re-enable guidance | Spoofing / Information integrity | Explicit current-document inventory, link scan, and bounded TensorFlow-reference scan. [VERIFIED: `tests/test_phase9_documentation.py:239-274,336-349`] |
| Validation frontmatter promotes stale/missing evidence to pass | Repudiation | Canonical discovery parser, green/superseded rows, commands, scope, nonclaims, audit trail. [VERIFIED: `/Users/akriz/.codex/gsd-core/templates/VALIDATION.md:1-78`; `/Users/akriz/.codex/gsd-core/workflows/audit-milestone.md:167-178`] |
| Timing noise triggers unauthorized lifecycle changes | Tampering / Denial of service | ADR classification and hard stop; no locks, queues, retries, or coordination added in this phase. [VERIFIED: `AGENTS.md:89-103`; `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md:66-72`] |
| Removal accidentally alters retained handlers or staging boundary | Tampering | Existing handler contracts and full non-live suite; preserve store-local registry/private staging pattern. [VERIFIED: `AGENTS.md:46-57`; `.codex/skills/spike-findings-cacheness/references/handler-integration.md:1-220`] |

## Sources

### Primary (HIGH confidence)

- `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-CONTEXT.md` — locked scope, D-01 through D-20, discretion, deferrals, and target inventory.
- `AGENTS.md` and `docs/adr/0001-topology-specific-storage-guarantees.md` — project guardrails and lifecycle stop conditions.
- `src/cacheness/handlers.py`, `src/cacheness/config.py`, `src/cacheness/storage/handlers/__init__.py` — runtime TensorFlow surface and exact persisted identities.
- `pyproject.toml`, `uv.lock`, `.github/workflows/quality.yml`, `tools/phase8_evidence.py`, `tools/run_phase8_packaging.py`, `tools/run_phase8_platform_gates.py`, `tools/run_phase8_local_gates.py` — package, lock, CI, and qualification surface.
- `tests/packaging/test_wheel_matrix.py`, `tests/test_phase9_documentation.py`, `tests/test_phase9_examples.py`, and named qualification tests — reusable acceptance contracts.
- Existing Phase 1/3/5/6/7/8/9 validation files, `03-25-SUMMARY.md`, and `docs/phase3-direct-implementation-2026-09-06.md` — normalization inputs and Phase 3 provenance.
- `/Users/akriz/.codex/gsd-core/templates/VALIDATION.md` and GSD audit/validate workflows — canonical validation schema and discovery rules.

### Secondary (MEDIUM confidence)

- [uv Locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/) — lock update/check semantics.
- [uv Managing dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/) — published optional dependencies versus local dependency groups.
- [Python Packaging Core Metadata](https://packaging.python.org/en/latest/specifications/core-metadata/) — `Provides-Extra` and conditional `Requires-Dist` installed metadata.

### Tertiary (LOW confidence)

- None.

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — committed tool configuration and local installed versions were opened/probed; no new packages are proposed.
- Architecture: HIGH — locked context and project ADR/AGENTS boundaries are explicit.
- TensorFlow removal inventory: HIGH — current source, package, CI, tests, tooling, docs, maps, seed, and local runtime state were scanned.
- Documentation disposition: HIGH — recommendations are constrained to current source/test-backed facts and canonical ownership.
- Validation architecture: HIGH — canonical GSD schema/workflow sources and every target validation file were inspected.
- External tool semantics: MEDIUM — official uv and Python Packaging documentation was used and classified through the research confidence seam.

**Research date:** 2026-09-17
**Valid until:** 2026-10-17
