---
phase: 03-atomic-lifecycle-and-recovery-engine
verified: 2026-09-06T04:58:51Z
status: gaps_found
score: 3/8 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 3/8
  gaps_closed:
    - "Plan 03-14 closes the previously reported key-only stale teardown, live-tombstone projection, and mixed-pair defects in its covered schedules."
  gaps_remaining:
    - "UnifiedCache uses BlobStore's unadmitted private put seam."
    - "Pre-promotion projection replacement destroys M1 custom-metadata links when M2 later fails."
    - "Projection retry may adopt and overwrite a peer's pending candidate token."
    - "SQLite/PostgreSQL projection compare-and-mutate is not atomic across independent adapters."
    - "Fresh-root SQLite bootstrap races across authority instances/processes."
    - "Cached SQL metadata wrappers do not delegate custom-metadata APIs."
    - "Empty authority snapshots are misclassified as legacy and can delete an in-flight first generation."
    - "Same-key put accepts an uncontained persisted locator as a mutation token."
    - "PostgreSQL reconstructs cache_key_params outside nested metadata, breaking signature parity."
  regressions: []
gaps:
  - truth: "Repeated overwrite, delete, clear, and close operations converge safely while cleaning both payload and metadata state."
    status: failed
    reason: "CR-01: UnifiedCache.put calls unadmitted BlobStore._put_with_result, so close or clear can pass an in-flight facade mutation."
    artifacts:
      - path: src/cacheness/storage/blob_store.py
        issue: "put is admitted, but _put_with_result is not."
      - path: src/cacheness/core.py
        issue: "The facade calls the unadmitted seam directly."
    missing:
      - "Admit the result-returning put seam and split out one already-admitted implementation."
      - "A deterministic facade-put versus close/clear barrier regression."
  - truth: "Failures preserve the last valid generation and leave all residue detectable."
    status: failed
    reason: "CR-02: candidate projection publication deletes M1 custom links before M2 is signed, verified, or promoted; failed overwrite repair cannot reconstruct them."
    artifacts:
      - path: src/cacheness/core.py
        issue: "_prepare_authority_projection publishes the token-changing candidate before authority promotion."
      - path: src/cacheness/metadata.py
        issue: "SQLite deletes CacheMetadataLink rows on that token change."
      - path: src/cacheness/storage/backends/postgresql_backend.py
        issue: "PostgreSQL performs the same destructive transition."
    missing:
      - "Preserve M1 link ownership until exact M2 promotion succeeds."
      - "Failed-overwrite regressions starting from linked M1 at each failure boundary."
  - truth: "Forced same-key races have deterministic outcomes without globally serializing distinct keys."
    status: failed
    reason: "CR-03: a projection mismatch retry substitutes the currently visible locator, which may be another operation's pending candidate."
    artifacts:
      - path: src/cacheness/core.py
        issue: "_prepare_authority_projection re-reads and adopts a peer token without proving committed ownership."
    missing:
      - "Never adopt an observed peer token; mismatch must conflict unless it is proved to be the same committed snapshot."
      - "A two-pending-candidate barrier test."
  - truth: "A write exposes only an old or new complete generation."
    status: failed
    reason: "CR-04: SQL projection comparison and mutation are not cross-instance atomic. SQLite uses an instance-local lock around SELECT-then-write; PostgreSQL expected-absence locks no row."
    artifacts:
      - path: src/cacheness/metadata.py
        issue: "SQLite lacks BEGIN IMMEDIATE or one conditional statement spanning independent adapters."
      - path: src/cacheness/storage/backends/postgresql_backend.py
        issue: "FOR UPDATE cannot serialize two absent-row creators."
    missing:
      - "Database-native per-key CAS with deterministic mismatch classification."
      - "Independent SQLite and real PostgreSQL concurrency tests."
  - truth: "Repeated operations converge safely."
    status: failed
    reason: "CR-05: fresh-root bootstrap is protected only by one authority instance's lock; a losing instance reaches mkdir(exist_ok=False) and raises FileExistsError."
    artifacts:
      - path: src/cacheness/storage/sqlite_lifecycle_authority.py
        issue: "Missing-root creation does not catch and safely reclassify FileExistsError."
    missing:
      - "Bounded safe reclassification/join after a competing root creator wins."
      - "Unseeded fresh-root two-authority and two-process tests."
  - truth: "UnifiedCache remains a functional compatibility facade over BlobStore authority."
    status: failed
    reason: "CR-06: CachedMetadataBackend delegates projection mutation but not store_custom_metadata_if_current or a safe query/session seam."
    artifacts:
      - path: src/cacheness/metadata.py
        issue: "The wrapper exposes neither custom-metadata delegation nor SessionLocal/engine."
      - path: src/cacheness/core.py
        issue: "Public custom metadata put/query cannot reach the wrapped SQL backend."
    missing:
      - "Explicit wrapper custom-metadata store/query/session delegation."
      - "SQLite/PostgreSQL memory-cache parity tests through public APIs."
  - truth: "Repeated overwrite, delete, clear, and close operations converge safely."
    status: failed
    reason: "CR-07: clear_all treats an empty committed-key list as legacy permission and removes candidate state; invalidate and absent-snapshot retirement remain key-only."
    artifacts:
      - path: src/cacheness/core.py
        issue: "clear_all uses an empty-list heuristic; _retire_exact_authority_snapshot removes by key when snapshot is None."
    missing:
      - "Use explicit legacy classification and exact-token projection teardown."
      - "First-put versus clear/invalidate tests across independent facades."
  - truth: "A write exposes only an old or new complete generation."
    status: failed
    reason: "CR-08: same-key put reads actual_path as a CAS token without containment validation, allowing hostile persisted evidence to be overwritten."
    artifacts:
      - path: src/cacheness/core.py
        issue: "put uses _projection_locator_from_entry instead of fail-closed _entry_locator before mutation."
    missing:
      - "Validate top-level and nested locators before payload/projection mutation."
      - "Hostile same-key locator zero-mutation tests."
  - truth: "UnifiedCache remains a functional compatibility facade over BlobStore authority."
    status: failed
    reason: "CR-09: PostgreSQL restores cache_key_params at result top level while signing verification reads metadata.cache_key_params."
    artifacts:
      - path: src/cacheness/storage/backends/postgresql_backend.py
        issue: "_entry_to_dict writes result['cache_key_params'] instead of nested metadata."
      - path: src/cacheness/core.py
        issue: "Signature verification reads only nested metadata."
    missing:
      - "Restore nested cache_key_params, retaining a top-level alias only if required."
      - "End-to-end PostgreSQL signing/key-params parity coverage."
decision_coverage:
  honored: 32
  total: 32
  not_honored: []
---

# Phase 3: Atomic Lifecycle and Recovery Engine Verification Report

**Phase Goal:** Object lifecycle operations preserve an old or new complete generation and leave every incomplete outcome recoverable.
**Verified:** 2026-09-06T04:58:51Z
**Status:** gaps_found
**Re-verification:** Yes — after Plan 03-14

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
| --- | --- | --- | --- |
| 1 | A write or overwrite exposes either the previous complete generation or the new complete generation, never mixed payload and metadata state. | ✗ FAILED | CR-02/03/04/07/08/09 leave destructive pre-promotion changes, token theft, non-atomic SQL transitions, unsafe empty-state cleanup, hostile-token acceptance, and PostgreSQL signing mismatch. |
| 2 | Failures preserve the last valid generation and leave all residue detectable. | ✗ FAILED | A failed M2 can permanently remove M1 custom-metadata links before authority promotion. |
| 3 | Repeated overwrite, delete, clear, and close converge safely while cleaning payload and metadata. | ✗ FAILED | Unadmitted puts, fresh-root bootstrap races, and empty-snapshot fallback permit close/clear/bootstrap failure or committed authority with a deleted payload. |
| 4 | Operators can dry-run and resume reconciliation without guessing provenance. | ✓ VERIFIED | Canonical authority-indexed reconciliation and its focused behavioral coverage remain present; no fresh finding invalidates that engine itself. |
| 5 | Forced same-key races are deterministic without globally serializing distinct keys. | ✗ FAILED | SQL projection CAS is not cross-instance atomic and retry can overwrite a peer's pending token; tests omit both schedules. |
| 6 | Release limits derive from a checked-in measured baseline. | ✓ VERIFIED | Baseline and Ruff-delta artifacts remain present/wired; Plan 14 artifact query passed 9/9. |
| 7 | Darwin remains unavailable, not native Windows evidence, with a future gate mandatory. | ✓ VERIFIED | Fixed Python 3.11 command exited 2 with `UNAVAILABLE`; `native_evidence: false`, Phase 999.1 remains required. |
| 8 | UnifiedCache remains a functional compatibility facade over BlobStore authority. | ✗ FAILED | CR-01, CR-06, and CR-09 break admission, cached custom metadata, and PostgreSQL signing/key-parameter parity. |

**Score:** 3/8 truths verified (0 present-but-behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
| --- | --- | --- | --- |
| `src/cacheness/storage/lifecycle.py` | Transactional lifecycle engine | ⚠️ PARTIAL | Canonical transitions are substantive, but the pre-promotion hook enables destructive derived-state mutation. |
| `src/cacheness/storage/blob_store.py` | Canonical lifecycle composition/admission | ✗ FAILED | Public put is admitted; the private result seam called by UnifiedCache is not. |
| `src/cacheness/storage/sqlite_lifecycle_authority.py` | Durable multiprocess authority/bootstrap | ✗ FAILED | Fresh-root creation is not safely joined across independent instances/processes. |
| `src/cacheness/metadata.py` | Exact projection and cached-wrapper parity | ✗ FAILED | SQLite CAS is instance-locked; wrapper omits custom metadata; destructor emits shutdown errors. |
| `src/cacheness/storage/backends/postgresql_backend.py` | PostgreSQL parity | ✗ FAILED | Absence CAS is not serialized and key-parameter nesting disagrees with signing. |
| `src/cacheness/core.py` | Safe policy facade | ✗ FAILED | Admission, pre-promotion ownership, empty-snapshot, hostile-locator, and wrapper call chains are incomplete. |
| `tests/test_unified_cache_lifecycle_authority.py` | Deterministic facade races | ⚠️ PARTIAL | Covered schedules pass; admitted close/clear, linked failed overwrite, two candidates, and empty first-put cleanup are absent. |
| `tests/test_projection_mutation_contract.py` | Cross-adapter atomic contract | ⚠️ PARTIAL | SQL paths are sequential/mocked; no independent SQLite or real PostgreSQL contention and no public cached-wrapper workflow. |

The mechanical Plan 14 checks reported 9/9 artifacts present and 6/6 pattern links found. That establishes existence and nominal wiring, not the failed runtime invariants above.

### Key Link Verification

| From | To | Via | Status | Details |
| --- | --- | --- | --- | --- |
| `UnifiedCache.put` | BlobStore admission | private put result seam | ✗ NOT WIRED | `_put_with_result` bypasses `_ordinary_admitted`. |
| Lifecycle promotion | projection/link ownership | token transition | ✗ PARTIAL | Candidate replacement deletes old links before promotion. |
| Projection repository | SQLite/PostgreSQL concurrency | atomic CAS | ✗ PARTIAL | Instance/row locks do not cover independent SQLite adapters or PostgreSQL absence. |
| Cached wrapper | custom metadata | exact-current store/query | ✗ NOT WIRED | Required methods/session abstraction are absent. |
| PostgreSQL projection | signer | nested key params | ✗ NOT WIRED | Read-back nesting differs from verification. |
| BlobStore | LifecycleAuthority | manifests/reconciliation | ✓ WIRED | Direct lifecycle authority remains canonical. |

### Data-Flow Trace (Level 4)

No rendered UI data exists. Relevant flow is application → UnifiedCache → BlobStore admission/lifecycle → authority promotion → projection/custom links. The canonical authority segment is substantive; admission and projection/link branches fail as above.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| --- | --- | --- | --- |
| Plan 14 focused facade/projection tests | `uv run --isolated --python 3.11 --all-extras --group dev --frozen pytest -q tests/test_unified_cache_lifecycle_authority.py tests/test_projection_mutation_contract.py -o log_cli=false` | 25 passed | ✓ PASS, but omits CR-01–CR-07 schedules |
| Complete frozen Python 3.11 suite | `uv run --isolated --python 3.11 --all-extras --group dev --frozen pytest -q -o log_cli=false` | Passed with documented skips; emitted repeated destructor tracebacks | ⚠️ PASS WITH WR-01 OBSERVED |
| Fixed Windows command | `uv run --python 3.11 --frozen python verify_platform.py --phase3 --require-system Windows --require-python 3.11` | Exit 2, Darwin `UNAVAILABLE` JSON | ✓ EXPECTED UNAVAILABLE |

### Probe Execution

No phase-declared `probe-*.sh` scripts were found. Phase checks use pytest/tool commands.

### Requirements Coverage

| Requirement | Status | Evidence |
| --- | --- | --- |
| STOR-03 | ✗ BLOCKED | Pre-promotion link destruction, token theft, non-atomic SQL CAS, and unsafe first-put cleanup violate complete-generation visibility. |
| STOR-04 | ✗ BLOCKED | Failed overwrite can retain M1 authority/payload but lose M1 custom links. |
| STOR-05 | ✗ BLOCKED | Unadmitted put versus close/clear, bootstrap race, and empty-snapshot fallback do not converge. |
| STOR-06 | ⚠️ PARTIAL | Canonical reconciliation is intact, but cannot reconstruct deleted custom links. |
| STOR-07 | ✗ BLOCKED | Pending-candidate and independent SQL races lack deterministic semantics. |

No Phase 3 requirement is orphaned. Phases 4–6 broaden composition but do not explicitly defer defects introduced in this Phase 3 seam; none of the nine gaps is deferred.

### Fresh Review Finding Disposition

| Finding | Verdict | Independent evidence |
| --- | --- | --- |
| CR-01 admission bypass | CONFIRMED BLOCKER | Only public `put` is decorated; facade calls undecorated `_put_with_result`; close drains admission. |
| CR-02 failed overwrite link loss | CONFIRMED BLOCKER | Pre-promotion SQL token change deletes links and commits before `promote_mutation`. |
| CR-03 pending token theft | CONFIRMED BLOCKER | Mismatch retry adopts a fresh metadata locator without proving committed ownership. |
| CR-04 SQL atomicity | CONFIRMED BLOCKER | SQLite SELECT/write has only `self._lock`; PostgreSQL absence has no locked row. |
| CR-05 bootstrap | CONFIRMED BLOCKER | Instance-owned lock and uncaught root `mkdir(exist_ok=False)`. |
| CR-06 cached custom metadata | CONFIRMED BLOCKER | Wrapper has projection delegation but no custom store/session/query seam. |
| CR-07 empty-state heuristic | CONFIRMED BLOCKER | `clear_all` branches on empty `list`; absent retirement removes by key. |
| CR-08 hostile locator | CONFIRMED BLOCKER | Authority put uses `_projection_locator_from_entry`, bypassing containment validation. |
| CR-09 PostgreSQL key params | CONFIRMED BLOCKER | Read-back is top-level; signing verification reads nested metadata. |
| WR-01 destructor | CONFIRMED WARNING | Full suite independently emitted repeated `sys.meta_path is None` shutdown tracebacks. |
| WR-02 test coverage | CONFIRMED WARNING | 25 tests pass, but SQL is sequential/mocked and enumerated barriers are absent. |

### Test Quality Audit

| Test Group | Linked Req | Active | Skipped | Circular | Assertion Level | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Direct lifecycle/reconciliation | STOR-03..07 | Yes | Platform cases only | None observed | Behavioral | ✓ Strong for canonical engine |
| Unified facade lifecycle | STOR-03..07 | Yes | No | None | Behavioral/barrier | ✗ Missing CR-01/02/03/05/07 schedules |
| Projection adapters | STOR-03/05/07 | Yes | No | None | Sequential/mocked | ✗ Insufficient for cross-instance SQL and public wrapper/PostgreSQL behavior |
| Native Windows | D-32 | Target exists | Darwin skip | None | Fixed command | ✓ Honest; cannot qualify Windows |

Disabled requirement tests are limited to native Windows and correctly map to backlog Phase 999.1. No circular oracle was observed. The two insufficient test groups are blockers because they are the only claimed evidence for the affected paths.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| --- | --- | --- | --- | --- |
| `src/cacheness/metadata.py` | 2407-2409 | destructor calls import/logging close path without finalization guard | ⚠️ Warning | Repeated ignored exceptions after green processes; directly observed. |
| Changed scope | — | No unreferenced TBD/FIXME/XXX markers | — | No debt-marker blocker. |

### Decision Coverage

All 32 trackable decisions were recognized as honored by the non-blocking decision-coverage check. This heuristic does not override the concrete defects.

## Human Verification

N/A — infrastructure/foundation phase. The failures require deterministic automated regressions, not subjective manual testing.

## Gaps Summary

Phase 3 remains blocked. Plan 03-14 fixes the three prior projection defects in covered schedules but leaves nine adjacent lifecycle violations. Keep the facade put admitted; defer destructive projection/link ownership changes until promotion; make projection CAS database-native; classify canonical empty state explicitly; validate observed locators before mutation; and restore cached/PostgreSQL parity. Add the exact barrier and real-adapter tests before re-verifying.

D-32 remains unchanged: Darwin is `UNAVAILABLE`/`NOT_QUALIFIED`, `native_evidence` is false, and Phase 999.1 still requires native Windows Python 3.11 `PASS` at exit 0.

---

_Verified: 2026-09-06T04:58:51Z_  
_Verifier: gsd-verifier_
