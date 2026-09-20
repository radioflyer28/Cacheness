---
phase: 01-compatibility-and-security-baseline
verified: 2026-08-30T04:32:22Z
status: passed
score: 13/13 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 12/13
  gaps_closed:
    - "The finalized Phase 1 validation state is accepted by its executable quality gate and the full workspace suite is green."
  gaps_remaining: []
  regressions: []
decision_coverage:
  honored: 21
  total: 21
  not_honored: []
deferred:
  - truth: "General old-or-new generation atomicity, failed-write preservation, lifecycle idempotence, and reconciliation (STOR-03 through STOR-06)."
    addressed_in: "Phase 3"
    evidence: "Phase 3 goal and success criteria explicitly own the atomic lifecycle and recovery engine."
  - truth: "Caller-injected backend retention and honest topology capabilities (BACK-03 and BACK-06)."
    addressed_in: "Phase 4"
    evidence: "Phase 4 goal and success criteria explicitly own metadata composition and topology contracts."
  - truth: "Every cache invalidation path removes a complete entry through one storage lifecycle (CACH-03)."
    addressed_in: "Phase 6"
    evidence: "Phase 6 success criterion 3 explicitly owns TTL, eviction, predicate, decorator, single-key, and global invalidation convergence."
acknowledged_prohibitions:
  - statement: "The compatibility baseline must not elevate known defects, private internals, or accidental quirks into supported contracts."
    disposition: "acknowledged by prior Phase 1 human decision D-06"
    llm_judgment: "Evidence is consistent with compliance: characterization tests target documented/corrected public behavior and the roadmap retains downstream gaps."
  - statement: "SqlCache must not be merged into the object-storage lifecycle or present incomplete best-effort rows as a complete pull-through result."
    disposition: "acknowledged by prior Phase 1 human decisions D-20 and D-21"
    llm_judgment: "Evidence is consistent with compliance: SqlCache has no BlobStore/UnifiedCache dependency and explicit partial-result tests pass."
  - statement: "Security documentation must not claim that a valid signature, HMAC, or content digest makes hostile pickle or dill deserialization safe."
    disposition: "acknowledged by prior Phase 1 human decisions D-14 and D-16"
    llm_judgment: "Evidence is consistent with compliance: docs/SECURITY.md explicitly says those controls do not sandbox hostile deserialization, and its documentation test passes."
human_verification: []
---

# Phase 1: Compatibility and Security Baseline Verification Report

**Phase Goal:** Users have a frozen compatibility baseline and safe boundary behavior before lifecycle ownership changes.
**Verified:** 2026-08-30T04:32:22Z
**Status:** passed
**Re-verification:** Yes — after gap closure at `490ac04`

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Supported public imports, constructors, configuration names, registries, aliases, decorators, exceptions, and representative results have an executable compatibility baseline (MIGR-01). | ✓ VERIFIED | `tests/test_public_api_contract.py`, configuration round-trip tests, and the focused Phase 1 suite pass. `CacheReason` and typed boundary exceptions are exported; both SQLAlchemy adapter aliases remain present. |
| 2 | The eight selected current/legacy stored-format variants are immutable, independently validated, and readable only through exact compatibility adapters (MIGR-01 detail). | ✓ VERIFIED | `validate_corpus.py --expected-through sqlite-columns-v0314` passes; `tests/test_stored_compatibility.py` passes and exercises exact JSON, SQLite, signature, decorator-key, and raw-array adapters. |
| 3 | `SqlCache` remains a separate subsystem and representative strict and explicit-best-effort pull-through behavior works (CACH-07). | ✓ VERIFIED | `sql_cache.py` contains no BlobStore/UnifiedCache dependency. `tests/test_sql_cache.py` and `tests/test_sql_cache_failure_contract.py` pass behavioral workflows and failure/result assertions. |
| 4 | Filesystem reads, writes, deletes, listings, streams, sizes, and high-level handler I/O reject traversal, absolute, drive, UNC, rooted, and symlink/race escapes (SECU-01). | ✓ VERIFIED | `ManagedFileOps`, `GuardedHandlerIO`, BlobStore, and UnifiedCache are wired. `tests/test_filesystem_containment.py` passes the active host-independent, no-follow, staging, snapshot, prefix, preflight, and POSIX race matrix; the Windows-only junction case is appropriately skipped on this host. |
| 5 | Legacy array metadata uses bounded non-executing parsing; ordinary NumPy arrays use native NPZ with pickle disabled; trusted object arrays require explicit strict configuration (SECU-02). | ✓ VERIFIED | Production uses `np.load(..., allow_pickle=False)`, contains no executable legacy metadata evaluator, and validates all four trusted-object predicates. `tests/test_legacy_array_security.py` and the non-vacuous AST gates pass. |
| 6 | Metadata query fields and numeric values are validated before backend access and reach SQLite/SQLAlchemy only as validated paths and bound operands (SECU-06). | ✓ VERIFIED | `validate_query_fields`, `validate_query_numeric_filters`, `to_sqlite_json_path`, and `bindparam` wiring are present. Query/security tests pass, including signed-64 endpoints and out-of-domain rejection before session access. |
| 7 | Public documentation identifies the trusted-application-payload boundary and unsafe serializer configuration/risk (SECU-07). | ✓ VERIFIED | `docs/SECURITY.md` explicitly states pickle/dill execution risk, that HMAC/integrity is not a sandbox, NPZ disables pickle, and strict object-array requirements. `tests/test_security_documentation.py` passes. |
| 8 | Guarded publication opens the exact staged regular-file identity that was validated, and signed-64 query gap cases fail correctly (CR-01/CR-06). | ✓ VERIFIED | Descriptor/fallback identity checks bind `st_dev`, `st_ino`, type, and link count; targeted containment and query regressions pass. |
| 9 | JSON metadata publication does not revoke an acknowledged authoritative candidate after backup-retirement faults (CR-05/CR-R1). | ✓ VERIFIED | `_save_to_disk` fsyncs candidate/replacement and treats only post-authority backup retirement as logged debt. Parameterized first-write/cross-format fault tests pass. |
| 10 | BlobStore and UnifiedCache candidates remain private until metadata authority; pre-commit failures preserve the prior entry and post-authority cleanup cannot delete the new candidate (CR-02/CR-03). | ✓ VERIFIED | Both code paths use unique candidate IDs and explicit ownership boundaries. Candidate, metadata, signing, snapshot, overwrite, and cleanup-fault regressions pass. |
| 11 | The narrow local global-clear contract has bounded prepared/committed recovery, fail-closed poison handling, lifecycle/query admission, concurrent put linearization, and non-resurrecting JSON close (CR-04, CR-R2 through CR-R5). | ✓ VERIFIED | `ClearRecoveryCoordinator` is wired to both APIs. `tests/test_clear_recovery.py` passes journal-bound, rollback/roll-forward, poison, same/second-instance, subprocess, query-context, and stale-close behavioral tests. |
| 12 | Phase 1 closes only its narrow reviewed cases and does not claim STOR-03..06, CACH-03, BACK-03, or BACK-06. | ✓ VERIFIED | REQUIREMENTS.md keeps those IDs pending and maps them to Phases 3, 6, and 4 respectively; ROADMAP.md and `01-VALIDATION.md` preserve the same boundary. |
| 13 | The finalized validation state is accepted by the executable Phase 1 quality gate and the full workspace test suite is green. | ✓ VERIFIED | `test_validation_artifact_records_terminal_approval_and_gap_wave_history` now checks terminal frontmatter separately from retained history. The quality gate passes 7/7 and the full suite passes with 27 expected skips. |

**Score:** 13/13 truths verified (0 present, behavior-unverified)

### Deferred Items

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | STOR-03 through STOR-06 general atomic lifecycle/reconciliation | Phase 3 | Phase 3 goal and success criteria own old-or-new generations, failure recovery, idempotent lifecycle operations, and reconciliation. |
| 2 | BACK-03 and BACK-06 backend injection/topology guarantees | Phase 4 | Phase 4 owns one composition root and truthful backend capabilities. |
| 3 | CACH-03 complete invalidation convergence | Phase 6 | Phase 6 criterion 3 names the complete invalidation surface. |

### Required Artifacts

| Artifact Group | Expected | Status | Details |
|----------------|----------|--------|---------|
| Public API/errors/config (`__init__.py`, `error_handling.py`, `config.py`) | Stable compatibility surface | ✓ VERIFIED | Exists, substantive, exported, and exercised by public/config tests. |
| Compatibility corpus and adapters (`tests/fixtures/compat`, `metadata.py`, `security.py`, `decorators.py`) | Exact immutable fixtures and production readers | ✓ VERIFIED | Corpus validator passes; production-adapter semantic/invariance tests pass. |
| Filesystem boundary (`path_security.py`, `guarded_handler_io.py`) | Contained no-follow I/O | ✓ VERIFIED | Substantive descriptor/fallback code is used by payload backends and high-level stores. |
| Array/query boundaries (`handlers.py`, `query_validation.py`) | Safe parsing and bound queries | ✓ VERIFIED | Wired into the registry/core; negative and ordering tests pass. |
| SqlCache (`sql_cache.py`) | Independent strict/partial contract | ✓ VERIFIED | Substantive and independently imported/tested. |
| Clear lifecycle (`clear_recovery.py`, `blob_store.py`, `core.py`, `metadata.py`) | Narrow recoverable clear and candidate ownership | ✓ VERIFIED | Substantive, wired to both public coordinators, and behaviorally exercised. |
| Security documentation (`docs/SECURITY.md`, `README.md`) | Accurate trust boundary | ✓ VERIFIED | Documentation contract tests pass. |
| Validation gate (`01-VALIDATION.md`, `tests/test_phase1_quality_gates.py`) | Executable terminal approval consistency | ✓ VERIFIED | Terminal complete/approved state and retained draft/pending history are asserted independently; 7/7 gate tests pass. |

**Artifacts:** 55/55 verified. All declared files exist, are substantive, and the terminal validation/gate pair now behaves consistently.

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| Public API/config tests | package exports/config | direct import and serialization | ✓ WIRED | Automated verifier passed both links. |
| Blob filesystem backend | `ManagedFileOps` | all payload operations | ✓ WIRED | Automated verifier and source inspection confirm usage. |
| BlobStore and UnifiedCache | physical-name encoder + guarded handler I/O | opaque IDs, staged writes, live snapshots | ✓ WIRED | The automated verifier passed all single-file links; the one malformed comma-separated PLAN `from:` was manually confirmed in both source files. |
| Core query methods | query validation + SQLAlchemy | pre-session validation and bind parameters | ✓ WIRED | Source and behavioral call-order tests confirm. |
| Compatibility adapters | immutable fixtures | exact JSON/SQLite/signature/key variants | ✓ WIRED | Five adapter links pass automated checks and production reads. |
| BlobStore/UnifiedCache | clear coordinator | constructor recovery, admission, clear | ✓ WIRED | Both public coordinators construct and invoke the same recovery engine. |
| Validation artifact | quality gate | terminal state assertions | ✓ WIRED | The gate distinguishes frontmatter terminal state from historical gap-wave text and passes. |

**Wiring:** 39/39 links verified (38 by the PLAN link query and the malformed multi-file `from:` manually confirmed in both source files).

### Data-Flow Trace (Level 4)

| Artifact | Data | Source | Produces Real Data | Status |
|----------|------|--------|--------------------|--------|
| Guarded payload path | serialized bytes | caller value -> handler private stage -> `ManagedFileOps` candidate | Yes | ✓ FLOWING |
| Metadata authority | entry/candidate locator | high-level put -> backend `put_entry` -> read lookup | Yes | ✓ FLOWING |
| Query path | caller filters | validation -> bound SQLAlchemy expression -> backend session | Yes | ✓ FLOWING |
| Clear recovery | entries, payload locators, counters | backend snapshot -> bounded journal/tombstones -> rollback/roll-forward | Yes | ✓ FLOWING |
| Compatibility path | fixed historical data | immutable corpus -> exact adapter -> guarded snapshot -> handler result | Yes | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Phase 1 requirements and narrow gap closures | focused pytest over public API, stored compatibility, containment, arrays, queries, SqlCache, docs, clear recovery, and candidate integrity | Completed successfully; one expected Windows-only junction skip | ✓ PASS |
| Eight-fixture immutable corpus | `uv run python tests/fixtures/compat/validate_corpus.py --expected-through sqlite-columns-v0314` | `compatibility corpus validated through sqlite-columns-v0314` | ✓ PASS |
| Final validation quality gate | `uv run pytest -q -o log_cli=false tests/test_phase1_quality_gates.py -x` | 7 passed | ✓ PASS |
| Full workspace suite | `uv run pytest -q -o log_cli=false` | Passed with 27 expected optional/platform skips and one existing collection warning | ✓ PASS |

### Probe Execution

No `probe-*.sh` files are declared. The phase's explicit runnable compatibility-corpus validator was executed independently and passed as recorded above.

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|--------------|-------------|--------|----------|
| MIGR-01 | 01-01, 01-08 through 01-12 | Public and selected stored-data compatibility characterization | ✓ SATISFIED | Public contract, corpus, and production adapter tests pass. |
| CACH-07 | 01-06 | SqlCache remains separate with representative workflows | ✓ SATISFIED | Independent imports and strict/best-effort behavior tests pass. |
| SECU-01 | 01-02, 01-03, 01-13 through 01-15 | Filesystem containment and narrow reviewed lifecycle closures | ✓ SATISFIED | Containment, candidate, recovery, concurrency, query-admission, and stale-close tests pass. |
| SECU-02 | 01-04 | Typed safe metadata parser and no metadata-controlled eval | ✓ SATISFIED | Negative parser corpus and AST gate pass. |
| SECU-06 | 01-05, 01-13 | Validated/bound metadata query fields and signed-64 values | ✓ SATISFIED | Grammar, injection, semantic, and call-order tests pass. |
| SECU-07 | 01-07 | Trusted payload boundary documentation | ✓ SATISFIED | Documentation tests and direct text inspection pass. |

**Coverage:** 6/6 Phase 1 requirements satisfied. No Phase 1 requirement is orphaned from plan frontmatter.

### Decision Coverage

All 21 trackable CONTEXT.md decisions are honored by shipped artifacts. This gate is advisory and non-blocking.

### Test Quality Audit

| Test Group | Linked Req | Active Evidence | Skipped | Circular | Assertion Level | Verdict |
|------------|------------|-----------------|---------|----------|-----------------|---------|
| Public API + stored compatibility | MIGR-01 | Imports, signatures, exact values, immutable digests, production adapter workflows | 0 requirement-only skips | No | Value + behavioral | ✓ STRONG |
| SqlCache | CACH-07 | SQLite pull-through, rollback, strict failure, structured partial results | Dependency guards only; active environment ran them | No | Behavioral | ✓ STRONG |
| Filesystem + lifecycle | SECU-01 | Outside sentinels, descriptor identity, fault boundaries, reopen, concurrency, subprocess | Windows junction only on this non-Windows host | No | Behavioral | ✓ STRONG |
| Array parser | SECU-02 | Independent malformed corpus and non-vacuous AST sentinels | 0 | No | Value + behavioral | ✓ STRONG |
| Query validation | SECU-06 | Hostile grammar, signed-64 boundaries, pre-session spies, result semantics | 0 | No | Behavioral | ✓ STRONG |
| Security docs | SECU-07 | Exact required and forbidden claims | 0 | No | Value | ✓ SUFFICIENT |
| Phase quality gate | Phase closure | Ruff/AST checks plus terminal-state/history assertions | 0 | No | Value | ✓ STRONG |

**Disabled tests on requirements:** no requirement is proved only by a disabled test.  
**Circular patterns detected:** 0; compatibility expected values originate from pinned historical writers and an independent manifest/validator.  
**Insufficient assertions:** 0 for the six requirements; one incorrect assertion blocks terminal validation.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/cacheness/metadata.py` | 2082 teardown | Existing `SqliteBackend.__del__` shutdown `ImportError` | ℹ️ Info | Reproduced only during interpreter teardown; already tracked in protected CONCERNS.md and not introduced by this phase. |

No unreferenced `TBD`, `FIXME`, or `XXX` debt markers were found in Phase 1-changed source/test files. Phase-created Python files pass Ruff.

### Prohibition Review

The three judgment-tier prohibitions are backed by automated evidence and were already explicitly decided by the user during Phase 1 discussion. The recorded decisions provide the required human acknowledgment:

1. The baseline does not freeze defects/private quirks.
2. SqlCache remains separate and never labels explicit best-effort partial rows complete.
3. Documentation does not claim HMAC/digests make hostile pickle/dill safe.

### Human Verification

No further human verification is required. The PLAN-declared judgment-tier prohibitions were acknowledged through the approved Phase 1 context:

1. **Compatibility-scope prohibition** — D-06 requires corrected intended behavior and says known defects are not frozen.
2. **SqlCache boundary prohibition** — D-20/D-21 require strict completeness by default and explicit, fully reported best-effort results.
3. **Serializer-documentation prohibition** — D-14/D-16 require an explicit trusted-object path and native-handler ownership rather than invented safety or formats; the approved documentation states that HMAC/digests do not sandbox pickle/dill.

### Gaps Summary

**No gaps remain.** The former validation-state mismatch is closed with independent focused/full-suite evidence, and the three judgment-tier prohibitions are acknowledged by the prior user-approved Phase 1 decisions cited above.

The deferred lifecycle, invalidation, backend-selection, and topology contracts remain correctly assigned to later phases and were not counted as Phase 1 failures or achievements.

---

_Verified: 2026-08-30T04:32:22Z_  
_Verifier: the agent (gsd-verifier)_
