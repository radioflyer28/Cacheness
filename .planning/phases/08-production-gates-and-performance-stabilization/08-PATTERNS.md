# Phase 8: Production Gates and Performance Stabilization - Pattern Map

**Mapped:** 2026-09-13
**Files analyzed:** 17 proposed new/modified implementation, test, configuration, and workflow files
**Analogs found:** 14 / 17 (the three workflow surfaces have no direct repository analog)

**Revised 2026-09-14:** D-23 supersedes the TensorFlow-specific pattern assignments
implemented by completed Plans 08-02/08-03. Plan 08-13 uses a pre-production removal
pattern: delete the in-tree handler/config/export/docs surface, remove the optional
group and its lock graph, contract-test exact absence from final packaging/platform
evidence, and preserve the generic custom-handler seam. Historical `.planning`
summaries remain evidence of work performed and are excluded from production-surface
absence scans. No lifecycle or ADR 0001 pattern changes.

Phase 8 is release evidence and measurement work. Preserve the Phase 07.1 single
BlobStore/AuthorityLifecycleEngine authority and ADR 0001's separation of integrity,
recovery, bounded progress, and performance. These assignments authorize no new
lifecycle lock, queue, lease, retry coordinator, or source of truth.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| ".github/workflows/quality.yml" | config / workflow | event-driven | tools/verify_phase071_contracts.py main gate | role-match; no YAML analog |
| ".github/workflows/live_qualification.yml" | config / workflow | event-driven, request-response | tools/run_phase5_qualification.py | role-match; no YAML analog |
| ".github/workflows/performance.yml" | config / workflow | event-driven, batch | benchmarks/lifecycle_authority_benchmark.py | role-match; no YAML analog |
| tools/run_phase8_local_gates.py | utility / CLI orchestrator | request-response | tools/verify_phase071_contracts.py | strong match |
| tools/run_phase8_qualification.py | utility / evidence runner | request-response, file I/O | tools/run_phase5_qualification.py | exact match |
| tools/verify_phase8_coverage.py | utility / validator | transform, file I/O | tests/test_phase1_quality_gates.py + fixed selector verifier | role-match |
| tools/verify_phase8_release.py | utility / release aggregator | transform, file I/O | tools/verify_phase5_contracts.py + tests/test_phase3_release_evidence.py | role-match |
| tests/qualification/test_phase8_evidence.py (or extend test_live_evidence.py) | test | request-response, file I/O | tests/qualification/test_live_evidence.py | exact match |
| tests/packaging/test_wheel_matrix.py | test | batch, file I/O, subprocess | tests/test_full_suite_environment.py | exact match |
| tests/test_tensorflow_removal.py | test | public-surface and repository contract | tests/test_blob_manifest.py + tests/test_handler_registration.py | strong match |
| tests/performance/test_complexity_contracts.py | test | batch, CRUD/metrics | catalog/reconciliation/policy tests | exact match |
| tests/performance/test_phase8_benchmarks.py | test | batch, transform, file I/O | lifecycle benchmark + verifier tests | role-match |
| tests/test_phase8_quality_gates.py | test / verifier self-test | transform, request-response | tests/test_phase071_contract_verifier.py | exact match |
| benchmarks/phase8_benchmarks.py | benchmark / utility | batch, transform, file I/O | benchmarks/lifecycle_authority_benchmark.py | strong role match; analog stale |
| benchmarks/phase8_workloads.py | utility / fixture factory | batch, transform | benchmark seed helpers | role-match |
| benchmarks/phase8_baseline.json | config / evidence artifact | file I/O, transform | benchmarks/lifecycle_authority_baseline.json | exact artifact role; schema replaced |
| pyproject.toml | config | build/test/tool configuration | current tool sections | exact role match |
| uv.lock | config / lockfile | build transform | existing lockfile | exact role match |

The workflow paths are proposed by RESEARCH.md and have no existing YAML analog.
YAML should orchestrate Python commands; selectors, evidence validation, source
fingerprints, and threshold math belong in tested Python modules.

## Pattern Assignments

### tools/run_phase8_qualification.py (utility, request-response + file I/O)

Analog: tools/run_phase5_qualification.py

This is the closest implementation. Preserve standalone importability, three terminal
statuses, exact source revision, fixed test selection, no-skip rule, sanitized
allow-list, and bounded cleanup. Adapt schema/source inventory to Phase 8 and keep
production S3 calls in ObstoreGenerationIO; boto3 remains runner preflight/cleanup
only.

Imports/constants (analog lines 11-58):
    import argparse
    import base64
    from dataclasses import dataclass
    from datetime import UTC, datetime
    import hashlib
    import importlib.util
    import json
    import os
    from pathlib import Path
    import re
    import secrets
    import subprocess
    import sys
    from typing import Callable, Mapping, Sequence

    REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
    EVIDENCE_SCHEMA = "phase5-live-qualification-v1"  # replace with phase8 schema
    REQUIRED_CONFIGURATION = (...)
    LIVE_TEST_MODULES = (...)
    QUALIFICATION_SOURCE_PATHS = (...)

Revision/configuration validation (lines 119-159, 183-212):
    revision = completed.stdout.strip()
    return revision if re.fullmatch(r"[0-9a-f]{40}", revision) else "unavailable"
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *QUALIFICATION_SOURCE_PATHS],
        cwd=REPOSITORY_ROOT, capture_output=True, check=True, text=True, timeout=10,
    )
    return revision if not status.stdout.strip() else None
    if any(environment.get(name) for name in _DISALLOWED_S3_ENDPOINT_ENVIRONMENT):
        raise ValueError("invalid external configuration")

Evidence allow-list and terminal states (lines 60-89, 291-368):
    if set(evidence) != _ALLOWED_EVIDENCE_KEYS:
        raise ValueError("evidence violates the exact allow-list")
    if evidence.get("status") not in _STATUS_VALUES:
        raise ValueError("evidence status is invalid")
    if status == "QUALIFIED":
        if not complete_qualified_evidence or not re.fullmatch(r"[0-9a-f]{40}", evidence["revision"]):
            raise ValueError("qualified evidence contradicts its required service proof")
    elif status == "UNAVAILABLE":
        if result != "not_run" or cleanup_status != "NOT_ATTEMPTED":
            raise ValueError("unavailable evidence contradicts its terminal state")

Fixed child suite/no-skip completion (lines 395-491):
    return [sys.executable, "-m", "pytest", "-p", QUALIFICATION_FIXTURE_PLUGIN,
            "-q", "-ra", *LIVE_TEST_MODULES, "-m", LIVE_MARKER_EXPRESSION]
    if completed.returncode != 0 or any(marker in output for marker in incomplete_markers):
        return False
    return re.search(r"\b[1-9][0-9]* passed\b", output) is not None

Run/cleanup/source stability (lines 518-629):
    child_environment["CACHENESS_PHASE5_QUALIFICATION_RUN_ID"] = namespace
    try:
        completed = _run_fixed_suite(_qualification_arguments(), timeout, child_environment)
    finally:
        cleanup_status = cleanup(supplied_environment, namespace)
    qualified = is_complete_pass and cleanup_status == "CLEAN" and source_stable
    return 0 if qualified else 1

Do not turn remote latency into a gate or accept mocks, skips, an earlier commit, or
a substitute service as QUALIFIED.

### tests/qualification/test_phase8_evidence.py (test, request-response + file I/O)

Analog: tests/qualification/test_live_evidence.py

Reuse dynamic standalone loaders, temporary outputs, pytest.raises, parametrized
contradictory evidence, and fake subprocess/service callbacks. The tests guard the
evidence schema; they are not live-service tests.

Loader (analog lines 21-51):
    spec = importlib.util.spec_from_file_location("phase8_qualification_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

Unavailable/incomplete/unclean cases (lines 74-141):
    exit_code = runner.run_qualification(output=output, environment={})
    assert exit_code == 2
    evidence = runner.load_evidence(output)
    assert evidence["status"] == "UNAVAILABLE"
    runner.validate_evidence(evidence)

Use the parametrized matrix for failed/empty/skipped runs, residue, and successful
clean runs. Retain exact run-ID propagation, dirty-source prevention, endpoint
rejection, secret scans, malformed status/revision, and marker-owned bounded cleanup
cases from lines 144-535. If extending test_live_evidence.py, preserve all Phase 5
contracts and deliberately update only schema/name expectations.

### tests/qualification/conftest.py (fixture, request-response + file I/O)

Analog: itself, tests/qualification/conftest.py (the Phase 5 fixture is the strongest
reusable implementation).

Preserve exact namespace dataclasses and bounds (lines 23-88):
    _MAX_CLEANUP_PAGES = 10
    _MAX_CLEANUP_OBJECTS = 1_000
    _MAX_CLEANUP_BYTES = 64 * 1024 * 1024
    @dataclass(frozen=True)
    class QualificationNamespace:
        run_id: str
        schema: str
        prefix: str

Environment decoding returns names/errors only and rejects absent region/key material
(lines 91-119). S3 cleanup verifies marker ownership, prefix containment, page/object/
byte/upload limits, and returns RESIDUE on uncertainty (156-458). PostgreSQL cleanup
verifies the owner row before DROP SCHEMA ... CASCADE, rolling back/closing errors
(461-524). The session fixture closes the authority and fails if cleanup is not CLEAN
(639-657). Rename only run identifiers for Phase 8; do not broaden cleanup.

### tests/packaging/test_wheel_matrix.py (test, batch + file I/O + subprocess)

Analog: tests/test_full_suite_environment.py

Copy wheel build and isolated subprocess pattern (lines 122-174):
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(dist)],
                   cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    wheel = next(dist.glob("cacheness-*.whl"))
    command = ["uv", "run", "--isolated", "--no-project", "--with",
               requirement, "python", "-c", probe]
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)

Base probe imports public cacheness/cacheness.storage, creates explicit memory/memory
StoreTopology, initializes, and performs generic plus NumPy round-trips through
BlobStore/UnifiedCache. Iterate every remaining literal optional group in a fresh
environment; one all-extras process is not QUAL-02 proof. Preserve runtime-boto3
absence assertion from lines 98-119. Use retained Parquet, NPZ/Blosc2, and SQL
representatives. Per D-23, no TensorFlow group, compatibility row, or dormant probe
remains.

### tools/run_phase8_local_gates.py (utility, request-response)

Analogs: tools/verify_phase071_contracts.py main (722-805) and
tools/verify_phase5_contracts.py main (465-484).

Use literal reviewed inventories, static checks before pytest, fixed child timeout,
separate labels, and explicit non-claim reporting:
    static_errors = (*validate_fixed_manifest(root), *audit_runtime_source(root), ...)
    if static_errors:
        print("Phase contract audit failed:", file=sys.stderr)
        return 1
    if _run_pytest(nodes):
        return 1
    if arguments.all and _run_ruff():
        return 1
    print("Performance boundary: PASS (no performance threshold is evaluated here)")

Use Phase 07.1 literal plan/decision/threat registries (33-45, 278-355), selector
validation (417-440), and AST audit (536-662). Do not derive scope from git diff,
filesystem discovery, or mutable plan text.

### tests/test_phase8_quality_gates.py (test, transform + request-response)

Analogs: tests/test_phase071_contract_verifier.py and tests/test_phase6_contract_verifier.py.

Load standalone verifiers with spec_from_file_location; test omission and hostile
source mutation. Phase 07.1 tests fixed manifests and renamed/malformed/unowned
selectors (48-97), source inventory omission (77-97), and executable AST versus
comments/strings (114-152). Phase 6 covers fixed inventory drift and prohibited
architecture forms (35-82, 176-264). Apply this to Phase 8 gate definitions: missing
files/selectors, false QUALIFIED states, and workflow command drift must fail.

### tools/verify_phase8_coverage.py (utility, transform + file I/O)

Analogs: tests/test_phase1_quality_gates.py Ruff JSON parser (153-184) plus
tools/verify_phase071_contracts.py validate_selector (417-440).

Read Coverage.py JSON with explicit type/range checks, fixed critical-module paths,
named selectors, and non-mutating baseline comparison. Compare repository and critical
aggregate statement/branch floors; reject missing or forged fields. Never rewrite the
baseline during ordinary CI. Reuse subprocess result validation but do not create a
Ruff finding fingerprint/debt ledger.

### benchmarks/phase8_benchmarks.py (benchmark, batch + transform + file I/O)

Analog: benchmarks/lifecycle_authority_benchmark.py (same role, but stale
constructor/schema) and tests/integration/test_s3_generation.py for participant
boundaries.

Keep fixture setup outside timed calls, explicit repetitions/warmups, environment and
revision metadata, raw distributions, and explicit record/recalibrate/verify modes.
The analog documents evidence versus policy at lines 1-9, distribution helpers at
77-101, real transition seeding at 120-130, public BlobStore measurement at 257-314,
and warm/repeat collection at 317-327.

    def _warm(callback):
        for _ in range(WARMUPS):
            callback()
    _warm(_measure_transition)
    transition_samples = [_measure_transition() for _ in range(REPETITIONS)]

Replace stale schema/constructor usage and retain old data only under a non-gating
historical path. Use pyperf.Runner only after the required human legitimacy checkpoint.
Keep handler serialization, BlobStore lifecycle, UnifiedCache policy, raw SHA-256/XXH3
hashing, and memory/call-count evidence separate. An envelope is never a runtime
deadline.

### benchmarks/phase8_workloads.py (utility, batch + transform)

Analogs: benchmark _seed_entry/_put_pair helpers (lifecycle_authority_benchmark.py:
120-130,257-261) and native handler fixture in tests/integration/test_s3_generation.py:
19-30.

Build deterministic 4 KiB generic, 16 MiB/128 MiB NumPy, and 100k-row dataframe
fixtures once outside timed loops. Keep handler-owned suffixes and public composition
boundaries. Workload descriptors carry tier, format, size, and cold/warm label; do
not instantiate a handler-by-topology Cartesian matrix.

### benchmarks/phase8_baseline.json (config/evidence, file I/O)

Analog: benchmarks/lifecycle_authority_baseline.json and validation at
lifecycle_authority_benchmark.py:548-612.

Retain command/harness/source commit/environment/cardinalities/repetitions/warmups,
metrics, and derived envelopes, but bind them to current obstore/authority composition
and the controlled Linux runner. The old artifact records macOS arm64, user_version 1,
and legacy cardinalities (lines 2-31); production now uses schema 9, so it is not a
release baseline.

    for name in ("command", "harness", "source_commit", "environment", "cardinalities"):
        if not benchmark.get(name):
            raise BenchmarkVerificationError(f"baseline benchmark.{name} is required")
    ...
    if derived.get("release_envelopes") != expected_envelopes:
        raise BenchmarkVerificationError("baseline release envelope derivation is inconsistent")

Baseline replacement is explicit and atomically written with temporary file, os.replace,
and directory fsync (analog lines 642-660). Do not auto-update during normal verification.

### tests/performance/test_complexity_contracts.py (test, batch + CRUD/metrics)

Analogs: tests/test_catalog_query_contract.py, tests/test_blob_store_reconciliation.py,
tests/test_phase6_statistics.py, and tests/test_phase6_removal_contract.py.

For catalog validation, copy _AccessSpy and assert malformed queries perform zero
authority calls (test_catalog_query_contract.py:75-118). For reconciliation, seed
MutationSpec records under small LifecycleLimits, then assert row/action/byte/time
caps and signed continuation (test_blob_store_reconciliation.py:165-217,264-363).
For statistics, freeze the result and monkeypatch catalog/query/delete to fail—stats
must make zero storage calls (test_phase6_statistics.py:90-150). For policy removal,
count exact BlobStore.delete calls and preserve cursor restart semantics
(test_phase6_removal_contract.py:99-128,253-324).

Count authority reads/writes and participant head/open/delete/list separately. Encode
one bounded catalog_page per request; reconciliation limits from operation_page_size,
max_reconcile_actions, and max_operation_record_bytes; statistics O(1)/zero storage
calls; policy one page plus at most one exact delete per candidate; and BlobStore
clear page/action/byte bounds. Add peak RSS separately from these structural assertions.

### tests/performance/test_phase8_benchmarks.py (test, batch + transform)

Analogs: tests/test_phase071_contract_verifier.py and benchmark validation
lifecycle_authority_benchmark.py:548-640.

Use fake measurements and temporary JSON to test malformed-baseline rejection, raw
distributions, environment/revision, envelope derivation, no auto-update, and
performance-label versus timeout-policy separation. Do not run 128 MiB or live service
workloads in ordinary self-tests.

### .github/workflows/quality.yml (config/workflow, event-driven)

No direct analog: no .github/workflows/ exists. Use verify_phase071_contracts.py fixed
suites as the command analog. Define PR deterministic jobs with fail-fast false, Linux
stable matrix 3.11 through latest compatible stable, macOS boundary smokes, packaging
for only the advertised groups, coverage, direct Ruff scopes, and complexity contracts.
Put prerelease Python in a separate advisory/continue-on-error job. Invoke frozen
isolated uv commands and Python gate tools; keep logic out of YAML.

### .github/workflows/live_qualification.yml (config/workflow, event-driven + request-response)

No direct analog: use Phase 5 runner/fixture boundaries. Only protected release
candidate dispatch and off-hour schedule may receive real PostgreSQL/AWS secrets. Run
against the exact release SHA, upload redacted failed/unavailable diagnostics with
bounded retention, and attach only sanitized QUALIFIED evidence to the release record.
Never run on untrusted pull requests or inherit prior artifacts.

### .github/workflows/performance.yml (config/workflow, event-driven + batch)

No direct analog: use benchmark CLI modes/baseline validation. Restrict to the named
controlled Linux runner; run canonical Phase 8 benchmarks and compare against the
checked-in baseline. Keep pyperf stability and relative envelopes in Python. Remote
live latency is diagnostic, not this blocking local timing gate.

### tools/verify_phase8_release.py (utility, transform + file I/O)

Analogs: tools/verify_phase5_contracts.py read_live_evidence_status (384-395) and
tests/test_phase3_release_evidence.py release artifact checks (34-83).

Read artifacts without repairing/upgrading, validate schema/status, recompute relevant
source identity, and require exact release revision. Aggregate references or digests
for deterministic, packaging, platform, coverage, performance, and live classes. Fail
closed if live is not QUALIFIED, cleanup is not CLEAN, source identity is stale, or a
required class is absent. Keep diagnostics separate from the lifetime release asset.

    if evidence_revision != release_revision:
        raise ReleaseQualificationError("evidence revision does not match release revision")
    if live_status != "QUALIFIED" or cleanup_status != "CLEAN":
        raise ReleaseQualificationError("live service evidence does not qualify release")

The exception name is illustrative; preserve narrow domain errors and raise ... from e
for translated file/JSON failures.

### pyproject.toml and uv.lock (config, build/test transform)

Analog: pyproject.toml lines 9-16, 18-45, 68-76, 82-129, and 131-146.

Keep Python >=3.11, core NumPy/obstore, the remaining literal optional groups, strict
markers, coverage source/omit, and Ruff target/line-length policy. Per D-23, remove the
TensorFlow project/dev groups and all resulting TensorFlow-named lock packages rather
than retaining an empty or incompatible extra. Add branch coverage only with the Phase
8 verifier/baseline. Add pyperf to dev only after human legitimacy checkpoint, pin its
reviewed version, and update uv.lock normally. Do not add runtime coverage/benchmark
dependencies or boto3 to production groups.

## Shared Patterns

### Fixed inventories and selector validation

Sources: tools/verify_phase071_contracts.py:33-45,347-355,417-440; tests/test_phase071_contract_verifier.py:48-97.

Use separately reviewed literal inventories and execution tuples. Validate each
path::test_name before subprocess execution; reject missing, renamed, malformed,
duplicate, or escaping selectors. Do not use discovery or git diff as an oracle.

### Exact revision, evidence allow-list, and redaction

Sources: tools/run_phase5_qualification.py:60-89,291-384; tests/qualification/test_live_evidence.py:293-394.

Evidence is exact-shape JSON, safe text only, bound to a 40-character lowercase Git
revision and sanitized run namespace. QUALIFIED requires complete passing tests,
standard Amazon S3 identity, clean cleanup, and stable source. Missing services stay
UNAVAILABLE; dirty/incomplete/failed runs are NOT_QUALIFIED.

### Exact owned cleanup

Source: tests/qualification/conftest.py:156-458,498-524.

Require marker-authorized exact namespace, bounded pages/objects/bytes/uploads,
prefix containment, and RESIDUE on uncertainty. Delete marker/schema only after all
bounded checks pass.

### Authority-first lifecycle and integrity

Sources: src/cacheness/core.py:972-1024; src/cacheness/storage/lifecycle.py:950-1047;
tests/test_phase6_removal_contract.py:132-167.

Observe authoritative entries first, verify canonical integrity before handler reads, and
route deletion through exact BlobStore.delete(expected=...). Payload presence, listings,
and ETags never establish visibility or canonical digest. Preserve typed conflict,
timeout, and recovery outcomes.

### Bounded work and separate call classes

Sources: src/cacheness/storage/reconciliation.py:176-285,321-408;
src/cacheness/storage/lifecycle.py:933-1047; src/cacheness/core.py:1040-1125.

Test finite row/action/byte/time budgets and continuation. Count authority reads,
authority writes, participant head/open/delete, and inventory list separately so one
counter cannot hide an N+1 elsewhere. These are structural contracts independent of
machine speed and remote latency.

### Immutable baseline by default

Source: benchmarks/lifecycle_authority_benchmark.py:1-9,548-612,642-701.

Validate schema, cardinality, environment, distributions, and derived envelopes before
comparison. Normal verification is non-mutating; capture/recalibration is explicit,
atomically published, and review-justified.

## No Analog Found

| File | Role | Data Flow | Reason / guidance |
|---|---|---|---|
| .github/workflows/quality.yml | workflow config | event-driven | No workflow files; use fixed Python gates and explicit matrices. |
| .github/workflows/live_qualification.yml | workflow config | event-driven, request-response | No YAML analog; use Phase 5 runner/fixtures and protected secrets. |
| .github/workflows/performance.yml | workflow config | event-driven, batch | No YAML analog; use benchmark CLI/baseline evidence. |

tools/verify_phase8_release.py has no exact aggregate-manifest analog; combine the
read-only Phase 5 evidence reader with Phase 3 artifact checks.
tools/verify_phase8_coverage.py has no coverage-ratchet implementation; use fixed
JSON/selector patterns and direct Coverage.py fields without inventing a debt ledger.

## Metadata

Analog search scope: tools/, tests/qualification/, tests/integration/, tests/contracts/,
tests/, benchmarks/, src/cacheness/storage/, src/cacheness/core.py,
src/cacheness/cache_policy.py, and pyproject.toml.
Files scanned: 18 primary analogs plus current configuration/lifecycle modules.
Pattern extraction date: 2026-09-13
