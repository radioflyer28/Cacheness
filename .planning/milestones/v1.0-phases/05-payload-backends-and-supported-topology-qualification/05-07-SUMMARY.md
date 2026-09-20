---
phase: 05-payload-backends-and-supported-topology-qualification
plan: "07"
subsystem: live-qualification
tags: [postgresql, amazon-s3, qualification, evidence, cleanup, credentials]
requires:
  - phase: 05-03
    provides: bounded Amazon S3 generation I/O contracts
  - phase: 05-05
    provides: explicit PostgreSQL lifecycle authority initialization
  - phase: 05-06
    provides: PostgreSQL/Amazon-S3 remote topology composition
provides:
  - fail-closed live-service runner with sanitized terminal evidence
  - exact run-owned PostgreSQL schema and S3-prefix fixture lifecycle
  - memory-only shared manifest signer providers for independent remote clients
affects: [05-08, 05-10, phase-08-runtime-qualification]
actuals:
  tokens: 13126
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - exact allow-listed release evidence
    - marker-authorized bounded external cleanup
    - distinct in-memory signer providers from shared supplied key bytes
key-files:
  created:
    - tools/run_phase5_qualification.py
    - tests/qualification/conftest.py
    - tests/qualification/test_live_evidence.py
  modified:
    - pyproject.toml
key-decisions:
  - "Only a complete fixed live suite with clean exact-run cleanup can emit QUALIFIED and exit zero."
  - "Missing service configuration remains UNAVAILABLE and exits two; it never produces a skipped success claim."
  - "Cleanup requires an exact PostgreSQL/S3 ownership marker and returns residue instead of crossing a run namespace."
patterns-established:
  - "Live qualification uses external configuration names, never repository-persisted credentials or endpoint overrides."
  - "Evidence includes sanitized provenance and terminal classes only; test output, DSNs, payloads, inventory, and signing material are excluded."
requirements-completed: []
requirements-progressed: [BACK-05]
coverage:
  - id: D1
    description: "The canonical runner emits strict UNAVAILABLE, NOT_QUALIFIED, or QUALIFIED evidence and fails closed on absent, skipped, deselected, or incomplete live execution."
    requirement: BACK-05
    verification:
      - kind: unit
        ref: "tests/qualification/test_live_evidence.py"
        status: pass
    human_judgment: false
  - id: D2
    description: "Live fixture configuration derives one marked PostgreSQL schema and S3 prefix, keeps signing material in memory, and refuses cleanup beyond an owned prefix."
    requirement: BACK-05
    verification:
      - kind: unit
        ref: "tests/qualification/test_live_evidence.py"
        status: pass
    human_judgment: false
duration: 20min
completed: 2026-09-08
status: complete
---

# Phase 05 Plan 07: Fail-Closed Live Qualification Harness Summary

**A fixed real-service qualification command now creates sanitized non-passing evidence by default and can qualify only a complete, genuine PostgreSQL/Amazon-S3 run with marker-authorized cleanup.**

## Performance

- **Duration:** 20 min
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Added strict `live_postgresql`, `live_aws_s3`, and `live_remote` markers plus `uv run --frozen --extra cloud python tools/run_phase5_qualification.py` as the canonical qualification gate.
- Recorded only an exact allow-list of revision, UTC time, package/runtime versions, AWS region/service metadata, a hashed run namespace, result class, and cleanup status. Missing required configuration records variable names only and exits 2.
- Rejected skipped, deselected, empty, failed, emulator-oriented, compatible-endpoint, timeout, or cleanup-leaking suites as `NOT_QUALIFIED` with exit 1.
- Built externally configured fixtures that create one quoted PostgreSQL schema and one S3 prefix from a cryptographically random run identifier, require matching ownership markers before deletion, and bound pages, objects, bytes, batches, and multipart abort work.
- Decoded the shared manifest key only in memory and created two independent signer-provider objects for separate remote clients.

## Task Commits

1. **Task 1: Emit sanitized UNAVAILABLE evidence and a non-zero result with no live configuration**
   - `a477a56` (`test`): added the red terminal-status and evidence-redaction contract.
   - `35cd9e2` (`feat`): implemented the fixed fail-closed runner and strict markers.
2. **Task 2: Create exact-run live fixtures with bounded idempotent cleanup**
   - `85529c0` (`test`): added the red owned-namespace and signer contract.
   - `42f0f29` (`feat`): implemented marked PostgreSQL/S3 fixtures, cleanup bounds, endpoint rejection, and runner-to-fixture namespace propagation.

## Files Created/Modified

- `pyproject.toml` — declares strict real-service markers.
- `tools/run_phase5_qualification.py` — runs only the fixed suite, writes redacted evidence, and enforces terminal exit codes.
- `tests/qualification/conftest.py` — owns external configuration parsing, resource creation, signer providers, and bounded cleanup.
- `tests/qualification/test_live_evidence.py` — tests evidence shape, redaction, non-passing paths, endpoint rejection, namespace propagation, and cleanup refusal.

## Decisions Made

- An Amazon-S3-compatible endpoint cannot qualify: endpoint-override environment values and non-Amazon S3 client endpoints fail before tests run.
- PostgreSQL may be an externally supplied real service, including Docker-hosted PostgreSQL transaction evidence; it never substitutes for real Amazon S3.
- A missing marker is clean only when the exact prefix is also empty; otherwise it is residue and blocks qualification.

## Verification

- `uv run --frozen --extra cloud pytest -q tests/qualification/test_live_evidence.py -x -o log_cli=false` — passed (11 tests).
- `uv run --frozen --extra cloud ruff check tools/run_phase5_qualification.py tests/qualification/conftest.py tests/qualification/test_live_evidence.py` — passed.
- `uv run --frozen --extra cloud python -m py_compile tools/run_phase5_qualification.py tests/qualification/conftest.py` — passed.
- The canonical command was run with all three required variables scrubbed. It exited 2 and wrote a schema-valid `UNAVAILABLE` record containing only required variable names, version/provenance metadata, and a hashed namespace.
- Static review confirms every S3 list/delete/abort call uses the exact generated prefix with fixed work bounds, and PostgreSQL DDL uses `psycopg.sql.Identifier` after a marker check.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Runner did not pass its generated run namespace to the live subprocess.**
- **Found during:** Task 2
- **Issue:** Fixtures need the exact runner namespace to make timeout backstop cleanup attributable; without propagation, they could generate an unrelated resource name.
- **Fix:** The runner now supplies only `CACHENESS_PHASE5_QUALIFICATION_RUN_ID` to the child environment and tests verify the exact identifier.
- **Files modified:** `tools/run_phase5_qualification.py`, `tests/qualification/test_live_evidence.py`
- **Verification:** Focused evidence tests passed.
- **Committed in:** `42f0f29`

**2. [Rule 2 - Security] Closed endpoint-override and over-broad secret-scan loopholes.**
- **Found during:** Task 2
- **Issue:** A standard SDK client could inherit an S3-compatible endpoint override, and treating every environment value as a secret fragment would reject non-secret AWS region evidence.
- **Fix:** Reject configured endpoint overrides and non-Amazon S3 endpoint URLs; scan only secret-shaped supplied values while preserving safe region metadata.
- **Files modified:** `tools/run_phase5_qualification.py`, `tests/qualification/conftest.py`, `tests/qualification/test_live_evidence.py`
- **Verification:** Endpoint-rejection and evidence tests passed.
- **Committed in:** `42f0f29`

**3. [Rule 1 - Bug] Aligned runner key validation with the signer provider's exact HMAC key length.**
- **Found during:** Task 2
- **Issue:** The runner accepted keys longer than the provider can safely use, deferring the failure into the live subprocess.
- **Fix:** Require exactly 32 decoded bytes at both boundaries.
- **Files modified:** `tools/run_phase5_qualification.py`, `tests/qualification/test_live_evidence.py`
- **Verification:** Focused evidence tests passed.
- **Committed in:** `42f0f29`

**Total deviations:** 3 correctness/security fixes, all contained to the qualification harness.

## Known Stubs

None.

## User Setup Required

Phase 8 must provide `CACHENESS_TEST_POSTGRES_DSN`, `CACHENESS_TEST_S3_BUCKET`, and `CACHENESS_TEST_MANIFEST_KEY_B64` through the external environment, plus standard AWS credentials. No credential, DSN, payload, or signing key is stored in this repository. Until that exact run returns `QUALIFIED`, BACK-05 remains an open release-evidence gate; superseded Plan 05-10 preserves the transferred command and acceptance contract.

## Next Phase Readiness

- Plan 05-08 can import the fixture module to exercise genuine PostgreSQL, Amazon S3, and two-client remote topology behavior.
- Phase 8 inherits the deterministic release gate from superseded Plan 05-10; `UNAVAILABLE` and `NOT_QUALIFIED` are intentional non-passing outputs rather than reasons to weaken topology claims.

## Self-Check: PASSED

- Confirmed the three qualification implementation/test files and this summary exist.
- Confirmed task commits `a477a56`, `35cd9e2`, `85529c0`, and `42f0f29` exist in repository history.

---
*Phase: 05-payload-backends-and-supported-topology-qualification*
*Completed: 2026-09-08*
