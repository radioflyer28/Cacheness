---
phase: 05-payload-backends-and-supported-topology-qualification
plan: "09"
subsystem: topology-contracts
tags: [topology, boto3, psycopg, documentation, architecture-verifier, qualification]
requires:
  - phase: 05-08
    provides: real-service qualification suite definitions and sanitized evidence rules
provides:
  - exact three-profile support and API capability publication
  - fixed local contract and one-engine architecture verifier
  - immutable separation of local verification from remote service observation
affects: [05-10, BACK-05, phase-08-runtime-qualification]
actuals:
  tokens: 14298
  tasks: 2
  commits: 5
tech-stack:
  added: []
  patterns:
    - marker-bounded documentation tables compared directly with runtime topology profiles
    - AST-only fixed-inventory architecture audit that ignores comments and string-only text
    - read-only sanitized live-evidence reporting separate from local correctness checks
key-files:
  created:
    - tools/verify_phase5_contracts.py
    - tests/test_phase5_contract_verifier.py
  modified:
    - docs/CATALOG_AND_TOPOLOGY.md
    - docs/STORAGE_INITIALIZATION.md
    - .planning/phases/05-payload-backends-and-supported-topology-qualification/05-COVERAGE.md
key-decisions:
  - "Exactly memory/memory, SQLite/filesystem, and PostgreSQL/Amazon-S3 are published; no Cartesian pairing is implied."
  - "Runtime and documentation retain immutable qualification requirements; the sanitized release-evidence artifact remains the exclusive service-run observation."
  - "The local verifier runs fixed local contracts and collects real-service suites without executing or upgrading their live qualification."
patterns-established:
  - "Parse marker-bounded support tables and compare every D-19 field to immutable runtime profile records."
  - "Audit executable AST nodes for coordination and authority-boundary regressions instead of grepping prose."
requirements-completed: [BACK-01, BACK-04]
coverage:
  - id: D1
    description: Exact documentation and API coverage publish all three immutable topology contracts and every named integration/opt-out decision.
    requirement: BACK-01
    verification:
      - kind: unit
        ref: tests/test_phase5_contract_verifier.py
        status: pass
    human_judgment: false
  - id: D2
    description: One deterministic local command verifies the fixed Phase 5 contract inventory and rejects lifecycle/coordinator, S3-visibility, advisory-lock, unbounded-listing, ETag-authority, and inline-secret regressions.
    requirement: BACK-04
    verification:
      - kind: other
        ref: uv run --frozen --extra cloud python tools/verify_phase5_contracts.py
        status: pass
    human_judgment: false
  - id: D3
    description: Local success reports live evidence read-only and cannot convert absent external PostgreSQL/Amazon-S3 evidence into a remote support qualification.
    verification:
      - kind: unit
        ref: tests/test_phase5_contract_verifier.py#test_local_verifier_inventory_and_evidence_boundary_are_fixed
        status: pass
    human_judgment: false
metrics:
  duration: 11min
  completed: 2026-09-08
  tasks: 2
  files: 5
status: complete
---

# Phase 05 Plan 09: Topology Contract and Local Architecture Gate Summary

**Published the exact three supported topology profiles and a fixed local verifier that enforces their boundary without converting local green checks into remote-service qualification.**

## Performance

- **Duration:** 11 min
- **Started:** 2026-09-08T10:55:40-04:00
- **Completed:** 2026-09-08T11:05:41-04:00
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Published a marker-bounded D-19 matrix that matches the runtime catalog for memory/memory, SQLite/filesystem, and PostgreSQL/Amazon-S3 only. Each row names coordination, the transaction/immutable-effect boundary, declared progress results, service prerequisites, JSON projection, and immutable evidence identifiers.
- Replaced the planning coverage baseline with a complete boto3/Amazon-S3 and psycopg/PostgreSQL capability table. Every listed surface is `INTEGRATE` or reasoned `OPT-OUT`; versioning, object lock, replication, bucket management, eventing, compatible endpoints, advisory locks, 2PC, async APIs, implicit migration, and retry-until-success are explicit exclusions.
- Added `tools/verify_phase5_contracts.py`, which checks the bounded tables against runtime declarations, audits only the fixed Phase 5 production inventory with AST rules, runs local contract suites, collects the fixed live modules, and reads release evidence without editing it.

## Task Commits

1. **Task 1: Publish one exact topology/API capability source of truth**
   - `4da5e78` (`test`): red marker-bounded documentation contract.
   - `6ad97f4` (`docs`): green three-profile matrix, initialization guide, API coverage, and contract checks.
2. **Task 2: Build the exact local Phase 5 contract and architecture verifier**
   - `0ad2c70` (`test`): red AST/evidence-boundary verifier contract.
   - `f35d243` (`feat`): fixed-inventory local verifier.
   - `bc59517` (`fix`): completed the exact Plan 01-08 local test inventory.

## Files Created/Modified

- `docs/CATALOG_AND_TOPOLOGY.md` — exact profile matrix and the topology-specific non-ACID/progress/performance boundary.
- `docs/STORAGE_INITIALIZATION.md` — explicit SQLite and PostgreSQL initialization, read-only validation, and stopped-worker Phase 7 maintenance rules.
- `05-COVERAGE.md` — complete API integration/opt-out publication.
- `tools/verify_phase5_contracts.py` — reproducible local documentation, AST, contract-suite, collection, and evidence-read gate.
- `tests/test_phase5_contract_verifier.py` — table, capability, AST-positive/negative, inventory, and evidence non-mutation regression coverage.

## Decisions Made

- Remote PostgreSQL/Amazon-S3 service requirements are immutable profile inputs. The sanitized release-evidence artifact, not runtime records or documentation, holds the service-run observation.
- The verifier uses AST nodes and a fixed source inventory rather than repository-wide text scans, preventing comments, strings, and historical Phase 3 artifacts from affecting the result.
- Performance remains a distinct report class. The local contract gate deliberately evaluates no latency threshold; Phase 8 owns final budgets and matrix expansion.

## Verification

- `uv run --frozen --extra cloud pytest -q tests/test_phase5_contract_verifier.py tests/test_supported_topologies.py -x -o log_cli=false` — passed (32 tests).
- `uv run --frozen --extra cloud python tools/verify_phase5_contracts.py` — passed: topology declaration, one-engine architecture, and local integrity/recovery/progress contracts all passed; the read-only live-evidence report was `absent`.
- `uv run --frozen --extra cloud ruff check tools/verify_phase5_contracts.py tests/test_phase5_contract_verifier.py` — passed.
- `uv run --frozen --extra cloud python -m py_compile tools/verify_phase5_contracts.py` — passed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Loaded the qualification runner through normal import registration.**
- **Found during:** Task 2 verifier self-test.
- **Issue:** Loading the standalone qualification runner without registering its module in `sys.modules` broke dataclass annotation resolution under Python 3.13.
- **Fix:** Registered the module before execution and removed it again if loading fails.
- **Files modified:** `tools/verify_phase5_contracts.py`
- **Verification:** Focused verifier tests passed.
- **Committed in:** `f35d243`

**2. [Rule 2 - Missing critical verification] Added omitted Plan 01 and Plan 03 contract modules to the fixed inventory.**
- **Found during:** Task 2 final inventory audit.
- **Issue:** The first verifier inventory omitted `tests/test_topology_capabilities.py` and `tests/test_s3_blob_backend.py`, despite those being exact Phase 5 plan verification modules.
- **Fix:** Added both modules and made the self-test assert the complete inventory exactly.
- **Files modified:** `tools/verify_phase5_contracts.py`, `tests/test_phase5_contract_verifier.py`
- **Verification:** The complete local verifier passed.
- **Committed in:** `bc59517`

**Total deviations:** 2 auto-fixed correctness/completeness fixes. No lifecycle coordination, compatibility surface, or support claim was added.

## Known Stubs

None.

## User Setup Required

No setup is required for this local plan. The separate remote qualification gate still requires externally supplied real PostgreSQL, Amazon S3, standard AWS credentials, and a shared manifest key. The local evidence report is currently `absent`; therefore this plan does not qualify the remote profile and BACK-05 remains open for Plan 05-10.

## Next Phase Readiness

- Plan 05-10 can run the fixed real-service suite and then rerun this local verifier without changing the immutable topology contract.
- Phase 8 can reuse the exact contract and live-suite inventory while adding its separate platform/install/performance evidence.

## Self-Check: PASSED

- Confirmed all five owned artifacts exist and the five task commits are present in repository history.
- Confirmed the realized plan diff has no tracked-file deletion and the owned artifact scan found no stubs.

---
*Phase: 05-payload-backends-and-supported-topology-qualification*
*Completed: 2026-09-08*
