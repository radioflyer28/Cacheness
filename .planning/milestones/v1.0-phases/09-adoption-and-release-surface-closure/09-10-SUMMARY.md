---
phase: 09-adoption-and-release-surface-closure
plan: 10
subsystem: planning-evidence
tags: [evidence, verification, nonclaims, narwhals, parquet]
requires:
  - phase: 03-atomic-lifecycle-and-recovery-engine
    provides: "Direct scoped-local qualification ledger and historical verifier snapshot"
  - phase: 08-production-gates-and-performance-stabilization
    provides: "LOCAL_READY record, verification report, and deferred qualification boundaries"
  - phase: 09-adoption-and-release-surface-closure
    provides: "Current BlobStore-first adoption and format-handler documentation"
provides:
  - "Current Phase 3/8 evidence dispositions that cite checked-in closure records without promoting nonclaims"
  - "One dormant Narwhals investigation seed with handler-owned Parquet and no dependency decision"
  - "Fail-closed tests for evidence provenance, deferred status, and seed dependency boundaries"
affects: [phase-10-sqlcache-removal, milestone-verification, future-handler-developer-kit]
actuals:
  tokens: 3943
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - "Historical verification is explicitly superseded by named checked-in closure evidence rather than new lifecycle work."
    - "A dormant seed records a future dependency investigation without changing package, lockfile, or handler behavior."
key-files:
  created:
    - tests/test_phase9_evidence_metadata.py
    - .planning/seeds/SEED-008-investigate-narwhals-dataframe-format-handlers.md
  modified:
    - .planning/phases/03-atomic-lifecycle-and-recovery-engine/03-VERIFICATION.md
    - .planning/phases/08-production-gates-and-performance-stabilization/08-VALIDATION.md
key-decisions:
  - "Phase 3's pre-03-21 3/8 verdict is historical; the named direct ledger, scoped topology, and later regression reports control the current disposition."
  - "Phase 8 local readiness completes only Wave 0/Nyquist metadata and preserves every remote, platform, performance, and publication nonclaim."
  - "Narwhals remains a future optional compatibility investigation; Parquet and persisted identities remain owned by native format handlers."
patterns-established:
  - "Evidence tests assert both a positive checked-in provenance chain and the exact deferred/nonqualified boundary."
  - "Future extension seeds distinguish evaluation questions from an adoption, package, or runtime decision."
requirements-completed: [CACH-06]
coverage:
  - id: D1
    description: "Phase 3 and Phase 8 metadata cite current local closure evidence while retaining ADR 0001 scope and all external nonclaims."
    requirement: CACH-06
    verification:
      - kind: unit
        ref: tests/test_phase9_evidence_metadata.py#test_phase3_verification_is_historical_with_named_local_closure_evidence
        status: pass
      - kind: unit
        ref: tests/test_phase9_evidence_metadata.py#test_phase8_validation_records_completed_local_evidence_without_promoting_nonclaims
        status: pass
    human_judgment: false
  - id: D2
    description: "Exactly one dormant Narwhals investigation captures dataframe compatibility questions without adding a dependency or changing format behavior."
    requirement: CACH-06
    verification:
      - kind: unit
        ref: tests/test_phase9_evidence_metadata.py#test_one_dormant_narwhals_investigation_preserves_handler_owned_parquet
        status: pass
      - kind: other
        ref: "seed/dependency probe from 09-10-PLAN.md"
        status: pass
    human_judgment: false
duration: 15m
completed: 2026-09-17
status: complete
---

# Phase 09 Plan 10: Evidence and deferred-extension closure Summary

**Current local evidence dispositions and one dormant Narwhals investigation, with no lifecycle, dependency, or qualification expansion.**

## Performance

- **Duration:** 15m
- **Started:** 2026-09-17T04:23:00Z
- **Completed:** 2026-09-17T04:38:33Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Reframed the old Phase 3 `gaps_found` / `3/8` verifier snapshot as historical,
  naming Plans 03-21 through 03-25, qualified tree `5282dca`, the direct ledger,
  and later Phase 07.1/08 regression records as the scoped local closure chain.
- Marked Phase 8 validation metadata complete for the checked-in `LOCAL_READY`
  Wave 0/Nyquist boundary while preserving `BACK-05`, `QUAL-06`, controlled-Linux,
  Windows, live-service, and publication nonclaims.
- Captured exactly one dormant Narwhals investigation for pandas, PyArrow, and
  Polars compatibility; native format handlers retain Parquet and persisted
  identity ownership, and no dependency or adapter was added.

## Task Commits

Each TDD task recorded a red contract before its evidence-only implementation:

1. **Task 1: Refresh stale Phase 3 and Phase 8 metadata from named evidence**
   - `cacdf0a` — failing evidence-metadata contract
   - `f8708b7` — scoped Phase 3/8 evidence disposition refresh
2. **Task 2: Capture exactly one dormant Narwhals investigation seed**
   - `c9e1ab2` — failing seed/dependency contract
   - `1d70869` — dormant Narwhals investigation seed

## Files Created/Modified

- `tests/test_phase9_evidence_metadata.py` — asserts provenance, nonclaims,
  exactly one dormant seed, and dependency absence.
- `03-VERIFICATION.md` — clearly separates the historic pre-03-21 verdict from
  current scoped local closure evidence.
- `08-VALIDATION.md` — records completed local-readiness/Wave 0/Nyquist metadata
  without qualifying deferred work.
- `SEED-008-investigate-narwhals-dataframe-format-handlers.md` — frames a future
  dataframe compatibility decision with explicit evidence and non-goals.

## Decisions Made

- The older Phase 3 report remains available as a historical snapshot but is no
  longer a current request for lifecycle coordination or race fixes.
- Local readiness remains an evidence class, not authority to qualify real
  services, other platforms, controlled performance, or publication.
- A potential dataframe compatibility layer must be evaluated separately from
  the handler-owned Parquet/native identity contract and package surface.

## Verification

Passed:

```text
uv run pytest -q -o log_cli=false tests/test_phase9_evidence_metadata.py -x
uv run python -c "from pathlib import Path; seeds=list(Path('.planning/seeds').glob('*narwhals*')); assert len(seeds)==1; assert 'narwhals' not in Path('pyproject.toml').read_text().lower(); assert 'narwhals' not in Path('uv.lock').read_text().lower()"
uv run pytest -q -o log_cli=false tests/test_phase9_examples.py tests/test_phase9_documentation.py tests/test_phase9_quality_workflow.py tests/test_phase9_evidence_metadata.py tests/test_interfaces.py tests/test_handler_registration.py tests/test_guarded_handler_io.py tests/test_public_api_contract.py tests/test_security_documentation.py tests/packaging/test_wheel_matrix.py tests/qualification/test_phase8_quality_workflow.py -x
uv run ruff check tests/test_phase9_evidence_metadata.py
uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false -m "not (live_postgresql or live_aws_s3 or live_remote)" -x
```

The final canonical non-live suite passed. Its documented Windows containment,
native Windows, and TensorFlow safety skips remain intentional and do not make
any new qualification claim.

## Deviations from Plan

None - plan-owned work executed exactly as written.

## Issues Encountered

The first final-suite runs exposed two inherited documentation-verifier paths
that still referenced Phase 9's deliberately retired documentation or an older
controlled-Linux phrase. They were corrected separately at phase level in
`1199207` and `64e6eda`, then the final non-live suite passed. Those external
realignments did not alter this plan's evidence or seed scope.

`requirements.mark-complete CACH-06` again reported the identifier as absent
from the machine-readable requirements tracker, so it made no requirements-file
change. The same tracker limitation was recorded by Plan 09-09; this plan does
not alter requirements text to work around it.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 9's adoption, evidence, and deferred-extension surfaces are complete
  and ready for independent phase verification.
- Phase 10 may remove the separate `SqlCache` subsystem without reopening this
  plan's BlobStore, format-handler, or qualification boundaries.

## Self-Check: PASSED

- All four plan-owned files exist and match the recorded scope.
- Task commits `cacdf0a`, `f8708b7`, `c9e1ab2`, and `1d70869` exist in Git
  history.
- No plan-owned stub or new security-relevant surface was introduced.

---
*Phase: 09-adoption-and-release-surface-closure*
*Completed: 2026-09-17*
