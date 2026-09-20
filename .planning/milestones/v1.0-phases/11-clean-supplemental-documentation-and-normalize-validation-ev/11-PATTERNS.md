# Phase 11: Clean Supplemental Documentation and Normalize Validation Evidence - Pattern Map

**Mapped:** 2026-09-17  
**Files analyzed:** 48 planned edits/deletions plus implied contract owners  
**Analogs found:** 40 / 48 (8 are intentionally deleted artifacts)

Phase 11 is a bounded pre-production surface cutover followed by evidence
maintenance. The implementation should copy Phase 10's direct-removal and
negative-contract patterns, Phase 9's explicit documentation/example ownership,
Phase 8's fixed evidence inventories, and the canonical GSD validation schema.
It must not copy any TensorFlow implementation into a replacement, create a
validation wrapper, or change BlobStore/lifecycle behavior.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `src/cacheness/handlers.py` | component/handler | file-I/O/transform | `src/cacheness/handlers.py` retained `ArrayHandler`/`ObjectHandler` | exact role; remove-only slice |
| `src/cacheness/config.py` | config/model | request-response | `src/cacheness/config.py` `HandlerConfig` validation | exact role; remove-only field/enum slice |
| `src/cacheness/storage/handlers/__init__.py` | package barrel/export | request-response | `src/cacheness/storage/__init__.py` explicit `__all__` | role match |
| `pyproject.toml` | config/packaging | build/package metadata | `tests/test_phase10_sqlcache_removal.py` manifest contract | role match |
| `uv.lock` | config/lockfile | build/package metadata | `pyproject.toml` plus `uv lock --check` | role match; generated artifact |
| `.github/workflows/quality.yml` | config/CI | batch/process | `tests/qualification/test_phase8_quality_workflow.py` | exact contract owner |
| `tools/phase8_evidence.py` | utility/evidence schema | transform/batch | `tools/phase8_evidence.py` fixed envelope constants | exact role; remove group entries |
| `tools/run_phase8_packaging.py` | utility/packaging harness | batch/process | `tests/packaging/test_wheel_matrix.py` | exact role and flow |
| `tools/run_phase8_platform_gates.py` | utility/qualification | batch/request-response | `tests/qualification/test_phase8_platform.py` | exact role and flow |
| `tools/run_phase8_local_gates.py` | utility/qualification | batch/process | `tools/run_phase8_local_gates.py` fixed gate runner | exact role; remove profile choices only |
| `tests/packaging/test_wheel_matrix.py` | test/packaging integration | batch/process | `tools/run_phase8_packaging.py` | exact role and flow |
| `tests/qualification/test_phase8_quality_workflow.py` | test/CI contract | static scan/request-response | `tests/test_phase9_quality_workflow.py` | role match |
| `tests/qualification/test_phase8_platform.py` | test/qualification contract | batch/request-response | `tools/run_phase8_platform_gates.py` | exact role and flow |
| `tests/qualification/test_phase8_release.py` | test/evidence contract | batch/transform | `tools/verify_phase8_release.py` fixed inventories | role match |
| `tests/test_phase9_quality_workflow.py` | test/CI contract | static scan/request-response | its `_job_block`/workflow assertions | exact role and flow |
| `tests/test_phase9_documentation.py` | test/documentation inventory | static scan/request-response | its explicit owners and link resolver | exact role and flow |
| `tests/test_phase9_evidence_metadata.py` | test/evidence artifact | static scan/request-response | its Phase 3/8 evidence assertions | role match |
| `docs/API_REFERENCE.md` | documentation/API guide | request-response | `docs/BLOB_STORE.md` task/reference guide | role + flow match |
| `docs/RELEASE_QUALIFICATION.md` | documentation/qualification matrix | batch/reporting | current release evidence matrix in same file | exact role and flow |
| `docs/README.md` | documentation/index | request-response/navigation | `examples/README.md` literal example index | role + flow match |
| `AGENTS.md` | agent/config guidance | request-response | `.planning/codebase/ARCHITECTURE.md` | role match |
| `.planning/codebase/ARCHITECTURE.md` | architecture map | request-response/transform | `AGENTS.md` current architecture block | role match |
| `.planning/codebase/CONCERNS.md` | risk map | request-response | `AGENTS.md` hard constraints/risks | role match |
| `.planning/codebase/STACK.md` | stack/config map | build/package metadata | `pyproject.toml` dependency inventory | role + flow match |
| `.planning/codebase/TESTING.md` | test map | batch/request-response | `tests/qualification/test_phase8_quality_workflow.py` | role + flow match |
| `.planning/seeds/SEED-005-remove-native-tensorflow-support.md` | planning state | request-response | `tests/test_phase9_evidence_metadata.py` seed-preservation assertions | role match |
| `.planning/phases/01-compatibility-and-security-baseline/01-VALIDATION.md` | validation evidence | batch/reporting | `.planning/phases/10-*/10-VALIDATION.md` | schema migration match |
| `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-VALIDATION.md` | validation evidence | batch/reporting | `03-25-SUMMARY.md` plus direct ledger | exact evidence-boundary match |
| `.planning/phases/05-payload-backends-and-supported-topology-qualification/05-VALIDATION.md` | validation evidence | batch/reporting | Phase 10 validation map | schema migration match |
| `.planning/phases/06-unifiedcache-policy-composition/06-VALIDATION.md` | validation evidence | batch/reporting | Phase 10 validation map | schema migration match |
| `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-VALIDATION.md` | validation evidence | batch/reporting | Phase 10 validation map | schema migration match |
| `.planning/phases/08-production-gates-and-performance-stabilization/08-VALIDATION.md` | validation evidence | batch/reporting | current Phase 8 local-readiness record | schema migration match |
| `.planning/phases/09-adoption-and-release-surface-closure/09-VALIDATION.md` | validation evidence | batch/reporting | Phase 10 validation map | schema migration match |
| `.planning/v1.0-v1.0-MILESTONE-AUDIT.md` | derived audit/report | batch/reporting | same audit's evidence/deferral tables | exact role; write last |
| `docs/PANDAS_COMPATIBILITY.md` | documentation (delete) | batch/transform | `docs/API_REFERENCE.md` | no analog; stale supplement |
| `docs/CUSTOM_METADATA.md` | documentation (delete) | request-response | `docs/BLOB_STORE.md` catalog section | no analog; redundant supplement |
| `docs/CROSS_PLATFORM_GUIDE.md` | documentation (delete) | batch/reporting | `docs/RELEASE_QUALIFICATION.md` | no analog; consolidated owner |
| `docs/WINDOWS_COMPATIBILITY.md` | documentation (delete) | batch/reporting | `docs/RELEASE_QUALIFICATION.md` | no analog; misleading supplement |
| `docs/TENSORFLOW_TENSOR_GUIDE.md` | documentation (delete) | file-I/O/transform | `docs/PLUGIN_DEVELOPMENT.md` retained handler guide | no analog; forbidden re-enable surface |
| `docs/TENSORFLOW_HANDLER_STATUS.md` | documentation (delete) | request-response/reporting | `docs/RELEASE_QUALIFICATION.md` | no analog; retired status document |
| `tests/test_tensorflow_handler.py` | test (delete) | file-I/O/transform | `tests/test_handlers.py` retained handler tests | no analog; dedicated retired capability test |

## Pattern Assignments

### Direct TensorFlow runtime cutover: `src/cacheness/handlers.py`

**Analog:** the retained handler implementations in the same module, plus
Phase 10's physical removal contract at
`tests/test_phase10_sqlcache_removal.py:219-276`.

Remove the lazy-import globals/helper, `TensorFlowTensorHandler`, commented
registration branches, enable-map member, and `tensorflow_tensor` built-in
identity. Keep the surrounding imports, retained handlers, store-local
registry, and existing `FormatHandler` boundary unchanged. The relevant stale
surface is visible at `src/cacheness/handlers.py:76-104,773-920,1294-1300,
1323-1326,1352-1365,1603-1606`.

The removal is physical and natural: do not add a disabled replacement,
`__getattr__`, alias, tombstone, or special error. Preserve narrow handler
errors and the existing private staging contract for retained handlers.

### Handler configuration and compatibility barrel:
`src/cacheness/config.py`, `src/cacheness/storage/handlers/__init__.py`

**Analogs:** `src/cacheness/config.py:211-252` for dataclass validation and
`src/cacheness/storage/handlers/__init__.py:28-97` for guarded optional exports.

Delete `enable_tensorflow_tensors` and its valid priority name from
`HandlerConfig`; leave retained enable flags and `ValueError` validation
unchanged. Delete the TensorFlow optional import probe and conditional
`__all__` extension. Keep the explicit core exports and pandas/polars guarded
imports. The package barrel must not create a compatibility path for the
removed handler.

### Manifest and lock convergence: `pyproject.toml`, `uv.lock`

**Analog:** `tests/test_phase10_sqlcache_removal.py:287-305` and the current
literal wheel inventory in `tests/packaging/test_wheel_matrix.py:288-353`.

`pyproject.toml` is authoritative. Remove the TensorFlow project extra and
dependency-group table, preserve the retained `recommended`, `dataframes`,
`s3`, `postgresql`, and `cloud` ordering/content, then run plain `uv lock` and
`uv lock --check`. Do not hand-edit generated package blocks or upgrade
unrelated dependencies. Invert the packaging assertions so installed metadata
has no `Provides-Extra: tensorflow` or TensorFlow conditional requirement while
retained local round trips still execute.

```python
project = tomllib.loads(project_path.read_text(encoding="utf-8"))
optional = project["project"]["optional-dependencies"]
assert tuple(optional) == ("recommended", "dataframes", "s3", "postgresql", "cloud")
assert "tensorflow" not in project_source.casefold()
assert "tensorflow" not in lock_source.casefold()
```

The tuple is a closed contract, not dynamically derived from the current
manifest after the edit.

### CI and qualification profile removal: workflow plus Phase 8 tools

**Analogs:** `tests/qualification/test_phase8_quality_workflow.py:20-53,57-79`,
`tools/run_phase8_local_gates.py:17-79,225-239`, and
`tools/run_phase8_platform_gates.py:101-112,192-255`.

Delete the `tensorflow-compatible` workflow job and both redundant feature
profile names/branches (`tensorflow` and `non_tensorflow`) from
`tools/run_phase8_platform_gates.py`,
`tools/run_phase8_packaging.py`, `tools/run_phase8_local_gates.py`, and
`tools/phase8_evidence.py`. Repair workflow tests to find the retained Linux
job using a retained-job boundary (the current Phase 9 parser at
`tests/test_phase9_quality_workflow.py:36-45` incorrectly uses the deleted job
as a delimiter). Remove TensorFlow from release/qualification fixture
inventories while preserving fixed evidence classes, exact SHA/source-digest
binding, deferred service/performance records, and no-live-credential rules.

The retained profile inventory is exactly `core` and remains literal and fail
closed; tests reject both removed names as CLI input and evidence-row values:

```python
FEATURE_PROFILES = frozenset({"core"})
assert feature_profile in FEATURE_PROFILES
```

Do not broaden a profile parser into discovery or alter any lifecycle gate to
compensate for profile deletion.

### Fresh wheel negative contract: `tests/packaging/test_wheel_matrix.py`

**Analog:** `tests/test_phase10_sqlcache_removal.py:232-276,287-326` and
`tests/packaging/test_wheel_matrix.py:104-150,168-180,288-353`.

Keep one built artifact, source-free isolated probes, explicit public exports,
member checks, metadata checks, and installed `BlobStore`/`UnifiedCache` round
trips. Replace the positive TensorFlow extra/round-trip expectations with
absence assertions. Include ZIP-member absence, no current TensorFlow import or
export, no optional extra/requirement metadata, and retained-extra behavior.

```python
artifact = runner.build_wheel(tmp_path / "dist")
result = runner.run_base_probe(artifact, workspace=tmp_path / "probe")
assert artifact.path.is_file()
assert len(artifact.sha256) == 64
assert "tensorflow" not in runner.optional_groups_from_pyproject(
    PROJECT_ROOT / "pyproject.toml"
)
```

The wheel path/digest remains bound from build through ZIP inspection and
isolated installation; source-tree imports alone are insufficient.

### Documentation inventory and executable ownership:
`tests/test_phase9_documentation.py`, `tests/test_phase9_quality_workflow.py`,
`tests/test_phase9_evidence_metadata.py`

**Analogs:** `tests/test_phase9_documentation.py:239-274,307-349`,
`tests/test_phase9_examples.py:15-23,95-113`, and
`tests/test_phase10_sqlcache_removal.py:168-205,219-229`.

Extend the existing documentation test rather than creating a second docs
registry. Assert the exact deleted-document set, surviving canonical files,
relative-link resolution, and a bounded current-surface TensorFlow scan.
Require pandas' concise verified Parquet/index/name/dtype statement in
`API_REFERENCE.md`, platform/nonclaim wording in `RELEASE_QUALIFICATION.md`,
and links to exactly the four existing executable examples. Do not scan dated
phase evidence as if it were current product truth.

```python
for document in sorted((PROJECT_ROOT / "docs").glob("*.md")):
    for target in markdown_link.findall(document.read_text(encoding="utf-8")):
        resolved = (document.parent / unquote(target.split("#", 1)[0])).resolve()
        if target and not target.startswith("#") and not re.match(r"^[a-z]+:", target):
            assert resolved.exists()
```

Retain the exact allowlist style for `tests/test_phase9_examples.py`: the
four literal names/markers remain the only runnable example owners. Repair the
Phase 9 workflow parser's delimiter in the same contract cluster.

### Canonical documentation destinations: `docs/API_REFERENCE.md`,
`docs/RELEASE_QUALIFICATION.md`, `docs/README.md`

**Analogs:** `docs/API_REFERENCE.md:75-99`, `docs/BLOB_STORE.md:12-77`, and
`docs/RELEASE_QUALIFICATION.md:1-35,42-57,76-103`.

Keep prose task-oriented and ownership-specific. `API_REFERENCE.md` receives
only the source/test-backed pandas format note; catalog metadata remains owned
by the existing API and BlobStore guides. `RELEASE_QUALIFICATION.md` becomes
the sole detailed owner for platform evidence, carrying the exact local
regression command and explicit `UNAVAILABLE`/`NOT_QUALIFIED`,
`DEFERRED`/`NOT_QUALIFIED`, and `NOT_PUBLISHED` boundaries. `docs/README.md`
must link only current guides and no deleted supplements.

Do not copy stale snippets from the supplements, introduce `cacheness(...)`,
or turn portable wheel tags into native Windows qualification. The four
canonical examples are links, not duplicated long snippets.

### Current maps and agent guidance: `AGENTS.md`, `.planning/codebase/*.md`

**Analogs:** `AGENTS.md:49-82,100-124` and
`.planning/codebase/ARCHITECTURE.md:17-36,87-111`.

Update current statements to remove TensorFlow from the optional integration
and extras inventory, and remove the retired handler from architecture,
concerns, stack, and testing maps. Preserve the BlobStore sole-authority,
private handler staging, fail-closed integrity, and topology-specific
nonclaims. Historical phase summaries/audits are not current-surface scan
targets and must remain truthful.

### In-place validation schema migration: target `*-VALIDATION.md` files

**Analog:** `/Users/akriz/.codex/gsd-core/templates/VALIDATION.md:1-78` and
the already validated Phase 10 record at
`.planning/phases/10-remove-sqlcache-pull-through-subsystem/10-VALIDATION.md:1-65,69-106`.

Update Phases 1, 5, 6, 7, 8, and 9 in place to the current frontmatter and
task-table contract: `status: validated`, `nyquist_compliant: true`, accurate
`wave_0_complete`, runnable commands, scope, nonclaims, and only green or
explicitly superseded rows. Preserve historical evidence, but add a visible
mapping when a row names removed SqlCache tests/commands; never silently turn
an old pending row into a current pass. Existing per-phase files remain the
only discovery truth; do not add an aggregator/wrapper report.

Use the current Phase 10 shape as the compact target:

```yaml
status: validated
nyquist_compliant: true
wave_0_complete: true
```

Then retain the task map, sampling commands, validation sign-off, and an
explicit deferred/nonclaim section. A command is evidence only for its stated
scope; missing live services, native Windows, controlled Linux, or publication
must remain visibly nonqualified.

### Compact Phase 3 record and stop gate:
`03-VALIDATION.md`

**Analogs:** `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-25-SUMMARY.md:10-38`,
`docs/phase3-direct-implementation-2026-09-06.md:103-186`, and the ADR
guardrail at `docs/adr/0001-topology-specific-storage-guarantees.md:35-70,198-236`.

Replace the draft narrative with a short canonical record covering only the
approved SQLite/local-filesystem and single-process-memory scopes. Name the
ADR guarantee classes, qualified commit/results, controlling historical chain
(`03-25-SUMMARY.md`, direct ledger, later Phase 07.1/8 evidence), one finite
current contract command, provenance (`direct_primary_agent`, not an
independent verifier), explicit nonclaims, and stop conditions.

The current gate is evidence-only and runs once:

```bash
uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false \
  tests/test_phase3_gap_acceptance.py tests/test_phase3_local_workflows.py \
  tests/test_blob_store_atomic_lifecycle.py tests/test_blob_store_reconciliation.py \
  tests/test_unified_cache_lifecycle_authority.py tests/test_integration.py -x
```

If it reveals a genuine integrity or recovery defect, stop and classify it
under ADR 0001. Typed contention outcomes, timing variation, or a request for
universal contender success do not authorize locks, queues, retries, a new
authority, or production edits. Obsolete detail stays in Git history and the
dated ledger, not in the live validation record.

### Seed and milestone audit: `SEED-005` and `.planning/v1.0-v1.0-MILESTONE-AUDIT.md`

**Analogs:** `.planning/seeds/SEED-005-remove-native-tensorflow-support.md:1-39`,
`tests/test_phase9_evidence_metadata.py:18-80`, and the audit's evidence and
deferral tables at `.planning/v1.0-v1.0-MILESTONE-AUDIT.md:112-128,167-200`.

Mark SEED-005 fulfilled and add a Phase 11 resolution link while retaining its
historical rationale; it must not remain promotable. Refresh the milestone
audit only after all package, docs, validation-discovery, Phase 3, wheel,
Ruff, and frozen non-live gates pass. Update debt/resolution and current
Nyquist verdicts from evidence, while preserving BACK-05, QUAL-06, native
Windows, and immutable-publication deferrals exactly. The audit is derived
output and is always the last write in the phase.

## Shared Patterns

### Direct pre-production removal

Phase 10's complete absence contract (`tests/test_phase10_sqlcache_removal.py:
219-326`) is the governing removal pattern: delete implementation, exports,
configuration, dependency metadata, CI/profile references, dedicated tests,
docs, and current claims together. Old imports fail naturally. No aliases,
tombstones, warning-only files, or re-enable recipes survive.

### Literal inventories and fail-closed checks

Keep explicit tuples/dicts for optional groups, public exports, wheel members,
qualification profiles, canonical docs, examples, and validation targets. Tests
must reject missing, renamed, or extra entries. Do not derive scope from
`git diff`, recursive replacement, runtime `__all__`, or broad history grep.

### Canonical ownership and historical truth

Current docs and maps describe only the supported product. Canonical guides own
current claims, the four examples own executable journeys, per-phase validation
files own validation discovery, and the milestone audit owns the final derived
verdict. Dated summaries, ledgers, audits, and Git history preserve provenance;
they are not silently rewritten to satisfy current scans.

### Evidence scope and nonclaims

A local/deterministic pass is scoped evidence. Carry explicit nonclaims for live
PostgreSQL/S3, controlled Linux performance, native Windows, and immutable
publication. `DEFERRED`, `NOT_QUALIFIED`, `UNAVAILABLE`, and `NOT_PUBLISHED` are
valid closed states, never synthetic passes.

### Audit-last ordering

The phase flow is: encode negative/discovery contracts -> remove package
surface -> consolidate/delete current docs -> normalize validation -> run the
finite Phase 3 stop gate -> run layered acceptance -> refresh the milestone
audit last. A premature audit can record a verdict unsupported by the final
wheel/docs/evidence state.

### Lifecycle stop boundary

This phase does not modify `BlobStore`, `UnifiedCache`, lifecycle authorities,
obstore participation, payload handlers that remain supported, or topology
guarantees. ADR 0001 separates integrity, recovery, progress, and performance;
if the Phase 3 gate exposes a real integrity/recovery defect, stop and report it
for separately authorized work.

## No Analog Found

These files are intentionally deleted rather than reimplemented or
consolidated into a replacement feature:

| File | Role | Data Flow | Reason |
|---|---|---|---|
| `tests/test_tensorflow_handler.py` | test | file-I/O/transform | Dedicated dormant TensorFlow contract is removed with the capability. |
| `docs/TENSORFLOW_TENSOR_GUIDE.md` | documentation | file-I/O/transform | D-06 forbids preserving a TensorFlow guide or re-enable recipe. |
| `docs/TENSORFLOW_HANDLER_STATUS.md` | documentation | request-response/reporting | Retired status documentation must not remain discoverable. |
| `docs/PANDAS_COMPATIBILITY.md` | documentation | batch/transform | Only verified concise Parquet behavior survives in `API_REFERENCE.md`; stale snippets are deleted. |
| `docs/CUSTOM_METADATA.md` | documentation | request-response | Catalog metadata and operations already have canonical API/BlobStore ownership. |
| `docs/CROSS_PLATFORM_GUIDE.md` | documentation | batch/reporting | Platform evidence is consolidated into `RELEASE_QUALIFICATION.md`. |
| `docs/WINDOWS_COMPATIBILITY.md` | documentation | batch/reporting | It is redundant/misleading once Windows nonqualification is owned by the release matrix. |
| Any new TensorFlow replacement module/alias | component/service | file-I/O/transform | No replacement format, compatibility shim, tombstone, or persisted-format migration is authorized. |

## Metadata

**Analog search scope:** `src/cacheness/`, `tests/`, `tools/`, `.github/`,
`docs/`, `examples/`, `AGENTS.md`, `.planning/codebase/`, `.planning/phases/`,
`.planning/seeds/`, `pyproject.toml`, and `uv.lock`.  
**Files scanned:** 48 planned/edit/delete surfaces plus canonical Phase 3,
Phase 8, Phase 9, and Phase 10 evidence owners.  
**Pattern extraction date:** 2026-09-17
