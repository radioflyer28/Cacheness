---
phase: 05
fixed_at: 2026-09-08T16:30:11Z
review_path: .planning/phases/05-payload-backends-and-supported-topology-qualification/05-REVIEW.md
iteration: 3
findings_in_scope: 12
findings_this_iteration: 1
fixed: 12
fixed_this_iteration: 1
skipped: 0
status: all_fixed
---

# Phase 05: Code Review Fix Report

**Fixed at:** 2026-09-08T16:30:11Z
**Source review:** `.planning/phases/05-payload-backends-and-supported-topology-qualification/05-REVIEW.md`
**Iteration:** 3 (cumulative)

**Summary:**

- Findings in scope: 12
- Fixed: 12
- Skipped: 0

Iteration 1 closed seven findings. Iteration 2 closed four findings from the
updated review. The capped re-review found one remaining release-evidence
provenance gap, which the primary agent closed directly without expanding the
storage lifecycle authority.

## Fixed Issues

### CR-01: The public remote initialization path validates a schema before creating it

**Files modified:** `src/cacheness/storage/blob_store.py`, `src/cacheness/storage/lifecycle.py`, `tests/contracts/test_postgresql_lifecycle_authority.py`
**Commit:** `05f8529`
**Applied fix:** The explicit BlobStore initialization boundary initializes a PostgreSQL authority before read-only preflight; ordinary mutation startup delegates to that boundary without a redundant preflight.

### CR-02: Replaying an older promoted operation returns the current replacement entry

**Files modified:** `src/cacheness/storage/backends/postgresql_lifecycle_authority.py`, `tests/contracts/test_postgresql_lifecycle_authority.py`, `tests/integration/test_postgresql_authority.py`, `docs/STORAGE_INITIALIZATION.md`
**Commit:** `cefaf3b`
**Applied fix:** Schema v3 persists promotion lineage and revision in the operation record, so idempotent replay reconstructs the operation-owned receipt rather than joining the mutable current entry.

### CR-03: Reconciliation advances past prepared mutations omitted by its byte bound

**Files modified:** `src/cacheness/storage/backends/postgresql_lifecycle_authority.py`, `tests/contracts/test_postgresql_lifecycle_authority.py`
**Commit:** `d165eed`
**Applied fix:** Mutation and cleanup-debt cursors advance only through rows emitted in the bounded reconciliation page.

### CR-04: S3 inventory labels every normal committed generation as unattributed residue

**Files modified:** `src/cacheness/storage/reconciliation.py`, `src/cacheness/storage/backends/postgresql_lifecycle_authority.py`, `tests/contracts/test_topology_lifecycle.py`, `tests/contracts/test_postgresql_lifecycle_authority.py`
**Commit:** `d61b296`
**Applied fix:** Each bounded S3 inventory page receives a single-statement, revision-consistent PostgreSQL attribution query over committed entries, prepared mutations, and pending cleanup debt. A changed or unavailable authority snapshot yields an indeterminate report rather than an orphan claim.

### CR-05: Live qualification cleanup can delete its authorization marker before cleanup completes

**Files modified:** `tests/qualification/conftest.py`, `tests/qualification/test_live_evidence.py`
**Commit:** `aecdb9a`
**Applied fix:** Bulk deletion excludes the exact owner marker; cleanup proves that only the marker and no multipart uploads remain before deleting that marker last. A deterministic later-page failure proves retry remains safe.

### CR-06: Evidence validation accepts contradictory claims as QUALIFIED

**Files modified:** `tools/run_phase5_qualification.py`, `tests/qualification/test_live_evidence.py`
**Commit:** `cbbc434`
**Applied fix:** Evidence validation now enforces terminal-state relationships, including the required standard Amazon S3 identity, and forged contradictory artifacts are rejected by both runner and read-only verifier.

### WR-01: The public metadata-role helper contradicts the qualified PostgreSQL role

**Files modified:** `src/cacheness/storage/composition.py`, `tests/test_postgresql_backend.py`, `tests/test_metadata_role_contract.py`, `tests/test_metadata.py`
**Commit:** `40a50dd`
**Applied fix:** The public helper now classifies PostgreSQL as the canonical authority role; focused role-contract tests no longer preserve the obsolete projection assertion.

### CR-01 (iteration 2): PostgreSQL promotion confuses payload verification and manifest digests

**Files modified:** `src/cacheness/storage/backends/postgresql_lifecycle_authority.py`, `tests/contracts/test_postgresql_lifecycle_authority.py`, `tests/integration/test_postgresql_authority.py`
**Commit:** `cf2799c`
**Applied fix:** Promotion receipt reconstruction now derives the manifest digest from the stored manifest rather than reusing the separately verified payload digest. Contract and live-capable integration coverage exercise distinct payload and manifest bytes.

### CR-02 (iteration 2): Frozen qualification tests do not load qualification fixtures

**Files modified:** `tools/run_phase5_qualification.py`, `tools/verify_phase5_contracts.py`, `tests/qualification/test_live_evidence.py`
**Commit:** `6fb5c41`
**Applied fix:** The runner and the read-only verifier explicitly load the qualification fixture plugin. A subprocess test proves the no-credentials outcome is the intended unavailable configuration result, not a missing-fixture error.

### CR-03 (iteration 2): Qualified evidence is not bound to an exact clean source revision

**Files modified:** `tools/run_phase5_qualification.py`, `tests/qualification/test_live_evidence.py`
**Commit:** `8975d0f`
**Applied fix:** QUALIFIED evidence requires a lower-case, full Git revision and is issued only when all qualification-relevant paths are clean before the external run and unchanged after cleanup. The default no-configuration boundary remains UNAVAILABLE.

### WR-01 (iteration 2): Reconciliation treats malformed cleanup debt as safe work

**Files modified:** `src/cacheness/storage/backends/postgresql_lifecycle_authority.py`, `src/cacheness/storage/reconciliation.py`, `tests/contracts/test_postgresql_lifecycle_authority.py`, `tests/test_blob_store_reconciliation.py`
**Commit:** `690f335`
**Applied fix:** The PostgreSQL authority surfaces the earliest non-pending cleanup debt without advancing the pending cursor beyond it. Reconciliation reports unsupported debt states as blocked, report-only evidence and never performs payload cleanup from malformed state.

### CR-01 (iteration 3): Qualification provenance omits imported production code

**Files modified:** `tools/run_phase5_qualification.py`, `tests/qualification/test_live_evidence.py`
**Commit:** `14fd480`
**Applied fix:** Qualification now requires the complete `src/cacheness` and `tests` trees, together with its fixed tools, configuration, and topology documents, to be clean before and after the live run. The regression proves a dirty imported module outside `storage` prevents an attributable qualification result.

## Verification

All verification ran in the main checkout because `workflow.use_worktrees` is `false`.

- Focused PostgreSQL authority and remote topology tests passed after CR-04.
- Qualification harness, evidence, and verifier tests passed after CR-05 and CR-06.
- Focused metadata-role tests passed after WR-01.
- `uv run python tools/verify_phase5_contracts.py` passed its topology, one-engine, local integrity/recovery/progress, performance-boundary, and live-evidence read-only checks.
- `uv run pytest -q tests/contracts/test_postgresql_lifecycle_authority.py tests/test_blob_store_reconciliation.py tests/qualification/test_live_evidence.py tests/test_phase5_contract_verifier.py -o log_cli=false` passed: 70 tests.
- Targeted Ruff checks and `git diff --check` passed for the iteration-2 source and test files.
- `uv run python tools/run_phase5_qualification.py --output /private/tmp/phase5-unavailable-evidence.json` correctly returned exit code 2 and wrote UNAVAILABLE evidence because no live credentials/configuration were supplied. Plan 10 therefore remains unavailable.
- The iteration-3 qualification evidence regressions passed: 44 tests across `tests/qualification/test_live_evidence.py` and `tests/test_phase5_contract_verifier.py`; targeted Ruff and `git diff --check` also passed.

---

_Fixed: 2026-09-08T16:30:11Z_
_Fixer: the agent (iterations 1–2), primary agent (iteration 3)_
_Iteration: 3 (cumulative)_
