---
phase: 03-atomic-lifecycle-and-recovery-engine
verified: 2026-09-06T17:50:59Z
status: gaps_found
score: 3/8
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 3/8
  gaps_closed:
    - "The nine pre-ADR facade admission, projection ownership/CAS, cached-wrapper, locator, and PostgreSQL parity gaps were closed by Plans 03-15 through 03-18."
  gaps_remaining:
    - "CR-01: non-persisted in-memory abort leaves a dangling reconciliation row."
    - "CR-02: retiring in-memory cleanup debt invalidates resume cursors."
    - "CR-03: SQLite point/list reads silently erase corrupt committed cache_key_params."
    - "CR-04: readers misclassify a transient first-writer SQLite leaf as incompatible."
  regressions: []
gaps:
  - truth: "A non-persisted aborted in-memory mutation remains safely reconcilable."
    status: failed
    reason: "CR-01: abort_mutation deletes the mutation mapping but retains its operation ID in the ordered reconciliation index; public dry-run raises KeyError."
    artifacts:
      - path: src/cacheness/storage/memory_lifecycle_authority.py
        issue: "_mutation_order retains an ID removed from _mutations."
    missing:
      - "Retain a stable terminal reconciliation row or remove it without invalidating captured high-water identity."
      - "Add public dry-run/apply coverage after a pre-publication abort."
  - truth: "Resumable in-memory cleanup retires every captured debt exactly once."
    status: failed
    reason: "CR-02: cleanup debts use mutable list positions as row IDs; applying one page shrinks the list, so a signed resume cursor skips work and eventually raises IndexError."
    artifacts:
      - path: src/cacheness/storage/memory_lifecycle_authority.py
        issue: "Debt high-water/cursor values address a shrinking _debts list."
    missing:
      - "Assign stable monotonic debt IDs and page by stable ID/high-water."
      - "Add a multi-page public apply/resume regression for the memory topology."
  - truth: "Corrupt committed SQLite metadata fails closed without mutating a valid authority generation."
    status: failed
    reason: "CR-03: get_entry and list_entries suppress malformed non-null cache_key_params. Under strict signing, public get treats the omitted field as an invalid signature and deletes the valid canonical authority entry."
    artifacts:
      - path: src/cacheness/metadata.py
        issue: "Malformed cache_key_params is silently omitted at the point/list read seams."
      - path: src/cacheness/core.py
        issue: "Derived projection corruption can trigger retirement of the valid authority snapshot."
    missing:
      - "Use the strict bounded decoder for live point/list observations and raise key-attributed METADATA_CORRUPT."
      - "Prove strict-signing public get/list leaves authority, payload, and corrupt evidence unchanged."
  - truth: "A reader racing first SQLite initialization receives only absence or a declared typed retryable outcome."
    status: failed
    reason: "CR-04: the first writer exposes an empty exclusive leaf before schema/application identity commits; a reader classifies it as durable authority and raises CacheBlobMigrationRequiredError."
    artifacts:
      - path: src/cacheness/storage/sqlite_lifecycle_authority.py
        issue: "Read-side classification cannot distinguish the exact pristine initialization leaf from durable incompatible evidence."
      - path: tests/test_sqlite_bootstrap_concurrency.py
        issue: "Competing mutators are covered, but reader-versus-initializer after leaf creation is not."
    missing:
      - "Recognize the exact pristine leaf under SQLite's bounded coordination without adding another authority or process-local correctness gate."
      - "Add a deterministic independent reader-versus-first-writer barrier test."
decision_coverage:
  honored: 32
  total: 32
  not_honored: []
---

# Phase 3: Atomic Lifecycle and Recovery Engine Verification Report

**Phase Goal:** Object lifecycle operations preserve an old or new complete generation and leave every incomplete outcome recoverable.
**Status:** gaps_found
**Score:** 3/8 must-haves verified
**Re-verification:** Yes — after ADR-driven Plan 03-20

## Verdict

Phase 3 is not complete. Four observed safety/recovery defects remain. This verdict follows [ADR 0001](../../../docs/adr/0001-topology-specific-storage-guarantees.md): it does not require cross-resource ACID, universal contender success, starvation freedom, or completion within the historical 0.187-second benchmark sample.

Plan 03-20 made the correct architectural correction. SQLite remains the sole local durable lifecycle authority; filesystem payloads are immutable external effects recovered from durable intent; BUSY/LOCKED may return contextual retryable timeout; the runtime default is caller-owned `5.0` seconds; benchmark evidence is separate; and the retired authority-wide FIFO/stage machinery should not return.

## Observable Truths

| # | Truth | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Writes expose only an old or new complete generation. | ✓ VERIFIED | Candidate publication, exact promotion, cleanup debt, and partial-stream/fsync interruption coverage remain present and wired. |
| 2 | Failures preserve the last valid generation and detectable residue. | ✗ FAILED | CR-01 and CR-03 leave unreconcilable evidence or permit projection corruption to delete valid canonical state. |
| 3 | Repeated overwrite/delete/clear/close converge safely. | ✗ FAILED | CR-02 breaks bounded memory cleanup resume. |
| 4 | Operators can dry-run and resume reconciliation. | ✗ FAILED | CR-01 raises `KeyError`; CR-02 raises `IndexError`. |
| 5 | Forced same-key races have deterministic topology-valid outcomes without global serialization. | ✗ FAILED | CR-04 returns false migration-required during valid first initialization. |
| 6 | Performance evidence does not define runtime failure semantics. | ✓ VERIFIED | Runtime default `5.0`; benchmark-local limit `0.075`; `0.187` retained only as historical provenance. |
| 7 | Native Windows remains honestly unqualified. | ✓ VERIFIED | Darwin evidence remains `UNAVAILABLE`/`NOT_QUALIFIED`. |
| 8 | UnifiedCache remains a safe compatibility facade over BlobStore authority. | ✗ FAILED | CR-03 strict-signing reproduction returned a miss and removed the valid authority generation. |

## Exact Gaps

### CR-01 — dangling in-memory mutation row

`abort_mutation(..., candidate_persisted=False)` removes `_mutations[operation_id]` but leaves the ID in `_mutation_order`. A direct public reconciliation probe deterministically raised `KeyError`. This violates STOR-04 and STOR-06 for the explicitly supported same-process memory topology.

### CR-02 — mutable in-memory debt cursor

Reconciliation snapshots and resume tokens store list positions while apply removes debts from that list. With three debts and an action budget of one, two resumes advanced and the next raised `IndexError`, leaving work pending. This violates STOR-05 and STOR-06.

### CR-03 — permissive SQLite point/list metadata decoding

`SqliteBackend.get_entry()` and `list_entries()` omit malformed non-null `cache_key_params` instead of raising `CacheIntegrityError`. With that field included in strict signing, a public `get()` returned `None` and deleted the otherwise valid authority entry. This violates the ADR's fail-closed corrupt-metadata invariant and STOR-04.

### CR-04 — transient SQLite bootstrap leaf misclassified

The first mutator closes an empty O_EXCL-created database leaf before committing application/schema identity. A reader can classify that exact transient state as a durable authority and raise migration-required. The independent clean full-suite run observed this at `test_query_meta_concurrent_access`; ten immediate focused reruns passed, confirming a narrow intermittent first-use window rather than an allowed BUSY/LOCKED timeout. This violates STOR-07.

## Warning

WR-01 is confirmed but non-blocking: `_translate_sqlite_error()` maps all remaining `sqlite3.DatabaseError` subclasses, including operational I/O/full/read-only/open failures, to migration-required. The failure remains closed, but the type and rebuild guidance are wrong. Correct it with SQLite primary-code classification while preserving BUSY/LOCKED as retryable lifecycle timeout.

## Artifact and Wiring Check

Plan 03-20's mechanical checks passed: 5/5 required artifacts exist and 5/5 declared key links are present. The implementation is substantive and nominally wired. The four gaps are runtime invariant failures that existence/pattern checks cannot detect.

## Behavioral Evidence

| Check | Result |
| --- | --- |
| Focused lifecycle/bootstrap command over `test_blob_store_atomic_lifecycle.py`, `test_phase3_postreview_concurrency.py`, and `test_sqlite_bootstrap_concurrency.py` | 31 passed |
| Clean detached Python 3.11 full suite | Failed once at `test_query_meta_concurrent_access` with `Lifecycle authority application ID is incompatible` |
| Immediate focused repetition of that test | 10/10 passed; confirms intermittency, not correctness |
| In-memory abort followed by public dry-run reconciliation | `KeyError` |
| Multi-page in-memory cleanup apply/resume | `IndexError` |
| Strict-signing SQLite projection corruption then public get | miss returned; valid authority entry deleted |
| Read from exact transient empty authority leaf | `CacheBlobMigrationRequiredError` |

No phase probe scripts exist. No unreferenced `TBD`, `FIXME`, or `XXX` marker was found in the reviewed implementation scope. The test gaps are precise: no memory abort/reconciliation regression, no memory multi-page debt regression, no strict-signing live point/list corruption preservation test, and no reader-versus-first-initializer barrier.

## Requirements Coverage

| Requirement | Status | Evidence |
| --- | --- | --- |
| STOR-03 | ✓ SATISFIED | Old-or-new complete generation visibility remains proven. |
| STOR-04 | ✗ BLOCKED | CR-01 and CR-03. |
| STOR-05 | ✗ BLOCKED | CR-02. |
| STOR-06 | ✗ BLOCKED | CR-01 and CR-02. |
| STOR-07 | ✗ BLOCKED | CR-04. |

No Phase 3 requirement is orphaned, and no gap is explicitly deferred to Phases 4–8. Plan 03-20's prohibitions are honored: no cross-resource ACID claim, benchmark-derived runtime deadline, universal-success test, second lifecycle authority, sidecar, lease, or restored FIFO scheduler was introduced.

## Human Verification

N/A — infrastructure/foundation phase. Every remaining gap has programmatic evidence and requires automated regression closure, not subjective UAT.

## Next Action

Create one focused gap-closure plan for CR-01 through CR-04, carrying WR-01 as a warning fix. Preserve the ADR simplification; do not reintroduce the race-chasing coordination machinery.

---

_Verified: 2026-09-06T17:50:59Z_
_Verifier: the agent (gsd-verifier)_
