---
phase: 09-adoption-and-release-surface-closure
plan: 02
subsystem: package-qualification-and-format-guidance
tags: [wheel, public-api, format-handlers, packaging, obstore]
requires:
  - phase: 09-01
    provides: Alias-free FormatHandler and FormatHandlerError storage contracts
provides:
  - Source-free wheel probe with quiet base imports and retired-name rejection
  - Durable local reopen regression that freezes handler and version identities
  - Current FormatHandler guidance that retains one BlobStore lifecycle authority
affects: [09-03, 09-06, 09-07, 09-08, 09-09, phase-10]
actuals:
  tokens: 3741
  tasks: 2
  commits: 3
tech-stack:
  added: []
  patterns:
    - Literal wheel-export inventories reject retired public names instead of deriving the contract from runtime exports.
    - Persisted handler/payload identities are validated through authenticated authority manifests before and after a durable reopen.
key-files:
  created: []
  modified:
    - tools/run_phase8_packaging.py
    - tests/packaging/test_wheel_matrix.py
    - tests/test_stored_compatibility.py
    - tests/test_public_api_contract.py
    - pyproject.toml
    - AGENTS.md
    - .codex/skills/spike-findings-cacheness/SKILL.md
    - .codex/skills/spike-findings-cacheness/references/handler-integration.md
    - tests/test_handlers.py
key-decisions:
  - "The built-wheel probe captures output only while importing Cacheness, so quiet base import is enforced without suppressing normal application output."
  - "Durable identity continuity is verified from the authenticated authority manifest, while the public entry-inspection projection remains intentionally metadata-only."
  - "FormatHandler remains the sole generic extension noun; custom formats continue through store.handlers.register_handler(...)."
patterns-established:
  - "Optional format integrations remain silent at base import and provide feedback only when a caller requests that capability."
  - "Private staging and snapshots remain below BlobStore's one lifecycle authority; handler guidance does not add another I/O or coordination seam."
requirements-completed: [CACH-06]
coverage:
  - id: D1
    description: Source-free package qualification enforces a quiet alias-free storage barrel while retaining existing base and dataframe probes.
    requirement: CACH-06
    verification:
      - kind: integration
        ref: tests/packaging/test_wheel_matrix.py
        status: pass
    human_judgment: false
  - id: D2
    description: Filesystem-plus-SQLite close/reopen preserves the stored handler type, payload format/version, and every store-version dimension without rewrite.
    requirement: CACH-06
    verification:
      - kind: integration
        ref: tests/test_stored_compatibility.py#test_current_store_reopen_preserves_handler_and_version_identity
        status: pass
    human_judgment: false
  - id: D3
    description: Active guidance uses FormatHandler with request-bound optional integrations and one custom registration seam below BlobStore authority.
    requirement: CACH-06
    verification:
      - kind: unit
        ref: tests/test_handlers.py and guidance terminology assertion
        status: pass
    human_judgment: false
duration: 13min
completed: 2026-09-17
status: complete
---

# Phase 9 Plan 02: Package Qualification and Format Guidance Summary

**The local wheel now proves a quiet, alias-free `FormatHandler` storage surface while durable manifests retain their existing handler and payload identities across reopen.**

## Performance

- **Duration:** 13 min
- **Started:** 2026-09-17T02:50:00Z
- **Completed:** 2026-09-17T03:03:31Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments

- Extended the existing source-free wheel runner with literal `FormatHandler`/`FormatHandlerError` exports, retired-name rejection, and output capture scoped to package imports.
- Added a durable filesystem-plus-SQLite reopen regression that compares authenticated manifest handler/payload identities and all version dimensions without a migration, rebuild, or file rewrite.
- Updated package identity to BlobStore-first without changing its version, Python floor, dependencies, extras, or lockfile.
- Updated current agent and spike guidance to retain private handler staging, immutable publication, and one BlobStore lifecycle authority; optional format integrations are request-bound.

## Task Commits

1. **Task 1: Prove the alias-free wheel surface and identity-preserving durable reopen**
   - `3584869` — `test(09-02): add wheel surface and durable identity regressions`
   - `d06f246` — `feat(09-02): qualify alias-free packaged storage surface`
2. **Task 2: Promote format-handler identity in current project and spike guidance**
   - `361ff8e` — `docs(09-02): align guidance with FormatHandler boundary`

## Verification

- `uv run pytest -q -o log_cli=false tests/packaging/test_wheel_matrix.py tests/test_stored_compatibility.py tests/test_public_api_contract.py tests/test_guarded_handler_io.py tests/test_handlers.py -x` — passed.
- Guidance terminology assertion confirmed `FormatHandler`, preserved `store.handlers.register_handler(...)`, and no `CacheHandler` term in active guidance or handler tests.
- `git diff --exit-code -- uv.lock` — passed; no lockfile change.

## Decisions Made

- Import-output capture is intentionally narrow: it detects package-generated optional-capability noise without treating normal application output as a package violation.
- The durable regression inspects the authenticated authority manifest because `BlobEntry` intentionally presents only the safe metadata projection.
- No lifecycle, backend, optional-extra, remote-qualification, or SqlCache scope changed.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The first durable assertion assumed that public `BlobEntry` exposed its underlying manifest. The projection is intentionally metadata-only, so the regression now reads the authenticated authority manifest directly; this tests the planned persisted-identity contract without widening the public API.
- The final verification initially could not open uv's external cache inside the filesystem sandbox. Re-running the unchanged command with authorized cache access passed.

## Known Stubs

None.

## Next Phase Readiness

- Plan 09-03 can build the four disposable canonical example journeys against the quiet, alias-free installed surface.
- The protected lifecycle boundary remains unchanged: `BlobStore` owns publication and recovery, while format handlers remain private path-based participants.

## Self-Check: PASSED

- The summary exists; task commits `3584869`, `d06f246`, and `361ff8e` are present in Git history.

*Phase: 09-adoption-and-release-surface-closure*
*Completed: 2026-09-17*
