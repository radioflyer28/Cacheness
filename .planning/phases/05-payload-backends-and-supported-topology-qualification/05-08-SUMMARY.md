---
phase: 05-payload-backends-and-supported-topology-qualification
plan: "08"
subsystem: testing
tags: [postgresql, amazon-s3, live-qualification, blobstore, recovery, integration]
requires:
  - phase: 05-06
    provides: PostgreSQL/S3 composition through the one BlobStore lifecycle engine
  - phase: 05-07
    provides: fail-closed live-service fixture ownership and evidence runner
provides:
  - real PostgreSQL authority lifecycle and contention test matrix
  - real Amazon S3 immutable-generation and bounded-inventory test matrix
  - independent remote BlobStore-client recovery and cleanup-debt test matrix
affects: [05-10, phase-08-runtime-qualification, BACK-05]
actuals:
  tokens: 6253
  tasks: 2
  commits: 2
tech-stack:
  added: []
  patterns:
    - fixed real-service markers with collection separate from qualification status
    - independently constructed remote clients sharing only externally supplied service identity and signer bytes
    - bounded database and S3 test work with exact run-owned cleanup
key-files:
  created:
    - tests/integration/test_postgresql_authority.py
    - tests/integration/test_s3_generation.py
    - tests/integration/test_remote_topology.py
  modified: []
key-decisions:
  - "Live suites use the Plan 05-07 exact-run fixture and fail rather than skip when configuration is absent."
  - "Contention assertions accept success, exact conflict, or typed retryable lifecycle progress; they never require universal contender success."
  - "Real S3 response-loss follow-up observes one accepted exact object; unsafe network-failure induction remains deterministic contract coverage."
patterns-established:
  - "Two remote BlobStore clients must construct separate S3 clients, authority instances, and signer providers before sharing only PostgreSQL/S3 state."
  - "Live cleanup/recovery verifies canonical authority state and debt, never a cross-resource transaction or inventory-derived membership."
requirements-completed: [BACK-04, BACK-05]
coverage:
  - id: D1
    description: Real PostgreSQL authority suite specifies explicit initialization, exact CAS, debt, cross-connection contention, typed timeout, and bounded reconciliation work.
    requirement: BACK-05
    verification:
      - kind: integration
        ref: tests/integration/test_postgresql_authority.py (3 marked cases; collected)
        status: pass
    human_judgment: true
    rationale: Real PostgreSQL execution is intentionally pending externally supplied service configuration and Plan 05-10 qualification evidence.
  - id: D2
    description: Real Amazon S3 suite specifies conditional small/multipart generations, exact ambiguity observation, bounded continuation pages, deletion proof, and denied credential handling.
    requirement: BACK-05
    verification:
      - kind: integration
        ref: tests/integration/test_s3_generation.py (4 marked cases; collected)
        status: pass
    human_judgment: true
    rationale: Real AWS execution is intentionally pending externally supplied bucket and credentials and cannot be inferred from collection.
  - id: D3
    description: Independent PostgreSQL/S3 BlobStore clients specify cross-client reads, paged catalog visibility, bounded contention, interrupted candidates, and cleanup-debt reconciliation.
    requirement: BACK-04
    verification:
      - kind: integration
        ref: tests/integration/test_remote_topology.py (3 marked cases; collected)
        status: pass
    human_judgment: true
    rationale: The multi-client workflow requires the same real service qualification run as D1 and D2.
duration: 10min
completed: 2026-09-08
status: complete
---

# Phase 05 Plan 08: Real-Service Qualification Suites Summary

**Ten fixed live cases now exercise PostgreSQL authority semantics, Amazon S3 immutable generations, and independently constructed BlobStore clients without treating collection or an ordinary skip as remote-topology qualification.**

## Performance

- **Duration:** 10 min
- **Started:** 2026-09-08T14:39:08Z
- **Completed:** 2026-09-08T14:49:36Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Added three real PostgreSQL cases for explicit initialization/reopen, foreign-version rejection, exact promotion/CAS, candidate and previous-generation cleanup debt, a real cross-connection row lock, and bounded reconciliation work.
- Added four real Amazon S3 cases for conditional small and multipart immutable publication, SHA-256/size snapshot evidence, exact-key ambiguity classification, bounded continuation inventory, deletion proof, and denied-credential behavior.
- Added three independent remote-client cases. They construct separate authority connections, S3 clients, store instances, and signer providers; they then prove authority-visible reads, bounded progress outcomes, durable candidate intent/debt, and idempotent cleanup recovery.

## Task Commits

Each task was committed atomically:

1. **Task 1: Exercise PostgreSQL authority and Amazon S3 generation semantics against real services** - `5dfd7ee` (`test`)
2. **Task 2: Exercise two independent remote BlobStore clients and recovery boundaries** - `e20f37d` (`test`)

## Files Created/Modified

- `tests/integration/test_postgresql_authority.py` - marker-only real PostgreSQL transaction, CAS, timeout, and paging matrix.
- `tests/integration/test_s3_generation.py` - marker-only real Amazon S3 conditional publication, verification, bounds, and credential matrix.
- `tests/integration/test_remote_topology.py` - marker-only independent-client topology, recovery, and cleanup-debt matrix.

## Decisions Made

- The live runner’s existing external fixture is the sole service/cleanup owner; the suites do not create endpoint overrides, service fakes, or unscoped resources.
- Database test setup uses a real row lock solely to observe the authority’s typed timeout mapping. It introduces no advisory-lock or queue behavior into production.
- A response-loss path observes one exact accepted S3 generation against its signed digest and size; failure injection stays in contract tests because it cannot safely be forced on the real service.

## Verification

- `uv run --frozen --extra cloud pytest --collect-only -q tests/integration/test_postgresql_authority.py tests/integration/test_s3_generation.py tests/integration/test_remote_topology.py -m 'live_postgresql or live_aws_s3 or live_remote' -o log_cli=false` - passed, 10 fixed live cases collected.
- `uv run --frozen --extra cloud ruff check tests/integration/test_postgresql_authority.py tests/integration/test_s3_generation.py tests/integration/test_remote_topology.py` - passed.
- `uv run --frozen --extra cloud python -m py_compile ...` - passed for all three files.
- `git diff --check HEAD~2 HEAD` - passed; task commits delete no tracked files.
- The qualification runner’s static source gate accepted all three modules; source contains no emulator, local-compatible service, or endpoint override path.
- With live configuration intentionally absent, `tools/run_phase5_qualification.py` returned exit 2 and temporary sanitized `UNAVAILABLE` evidence containing only the three missing configuration names. This is a deliberate non-passing result; no release evidence was created in the repository.

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None.

## User Setup Required

Plan 05-10 still needs externally supplied `CACHENESS_TEST_POSTGRES_DSN`, `CACHENESS_TEST_S3_BUCKET`, `CACHENESS_TEST_MANIFEST_KEY_B64`, and standard AWS credentials. Until the fixed runner returns `QUALIFIED`, BACK-05 remains open; no service substitute or skipped case changes that.

## Next Phase Readiness

- Plan 05-10 can run the complete fixed ten-case suite through the existing fail-closed qualification command.
- The suite leaves lifecycle ownership unchanged: PostgreSQL promotion remains visibility, while S3 effects reconcile through durable intent and cleanup debt rather than cross-resource ACID.

## Self-Check: PASSED

- Confirmed all three integration modules exist and task commits `5dfd7ee` and `e20f37d` are present in repository history.
- Confirmed no stubs or new non-summary untracked files are attributable to this plan.

---
*Phase: 05-payload-backends-and-supported-topology-qualification*
*Completed: 2026-09-08*
