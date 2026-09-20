---
phase: 09-adoption-and-release-surface-closure
plan: 08
subsystem: qualification-documentation
tags: [documentation, security, qualification, topology, payload-boundaries]
requires:
  - phase: 09-adoption-and-release-surface-closure
    provides: Local-ready README and task-first documentation
provides:
  - A single detailed qualification matrix with explicit local-only evidence boundaries
  - Current trusted-payload, containment, integrity, and fail-closed security guidance
  - Static topology/catalog reference that defers status claims to the matrix
affects: [09-09, 09-10, phase-10]
actuals:
  tokens: 14274
  tasks: 1
  commits: 2
tech-stack:
  added: []
  patterns:
    - Detailed topology, platform, payload, and performance claims have one documentation owner.
    - Catalog topology profiles declare runtime requirements without reporting mutable evidence status.
key-files:
  created: []
  modified:
    - README.md
    - docs/RELEASE_QUALIFICATION.md
    - docs/SECURITY.md
    - docs/CATALOG_AND_TOPOLOGY.md
    - tests/test_phase9_documentation.py
    - tests/test_security_documentation.py
key-decisions:
  - Release qualification is the sole detailed evidence matrix; guides link to it rather than repeat status.
  - Canonical SHA-256 plus size remains integrity authority while ETag/version is opaque corroborating evidence only.
  - Local readiness retains explicit nonclaims for S3/PostgreSQL, Windows, controlled Linux performance, and immutable publication.
requirements-completed: [CACH-06]
coverage:
  - id: D1
    description: One qualification guide distinguishes deterministic, package, platform, quality, structural, performance, live-service, and publication evidence.
    requirement: CACH-06
    verification:
      - kind: integration
        ref: tests/test_phase9_documentation.py#test_qualification_guide_is_the_one_detailed_owner_of_current_nonclaims
        status: pass
      - kind: integration
        ref: tests/qualification/test_phase8_quality_workflow.py#test_documentation_states_all_evidence_classes_and_nonclaims
        status: pass
    human_judgment: false
  - id: D2
    description: Security guidance keeps trusted executable-payload warnings and fail-closed parsing/containment boundaries visible.
    requirement: CACH-06
    verification:
      - kind: integration
        ref: tests/test_security_documentation.py
        status: pass
duration: 24min
completed: 2026-09-17
status: complete
---

# Phase 9 Plan 08: Qualification and Security Boundaries Summary

**Cacheness now has one tested qualification owner that keeps local readiness separate from remote services, platform support, performance, and immutable publication.**

## Accomplishments

- Replaced the Phase 8-era qualification prose with one evidence matrix covering deterministic/local, packaging, platform, quality, structural, controlled-performance, live-service, and publication evidence.
- Preserved explicit NOT_QUALIFIED, NOT_PUBLISHED, and DEFERRED boundaries for S3/PostgreSQL, Windows, controlled Linux performance, and publication.
- Documented the 128 MiB direct-create and private-staging ceiling, canonical SHA-256-plus-size integrity, opaque ETag/version corroboration, catalog field/query limits, authority transaction scope, recovery, and typed contention outcomes.
- Rewrote the security guide around trusted executable payloads, safe parsing, containment, guarded private staging, and fail-closed verification.
- Narrowed the topology reference to static composition contracts and linked it to the single qualification owner.

## Task Commits

1. **Task 1: Publish one truthful qualification owner and preserve security boundaries**
   - 01ad4c9 — test(09-08): add qualification documentation contracts
   - d3d0be5 — docs(09-08): consolidate qualification boundaries

## Verification

- uv run pytest -q -o log_cli=false tests/test_phase9_documentation.py tests/test_security_documentation.py tests/qualification/test_phase8_quality_workflow.py -x — passed (16 tests).
- git diff --check — passed.

## Decisions Made

- Detailed operational nonclaims are centralized in RELEASE_QUALIFICATION.md; onboarding and topology references remain intentionally narrower.
- Canonical integrity remains SHA-256 plus size. ETag/version stays signed, generation-bound, opaque transport corroboration and cannot select visibility or replace rehashing.
- Local evidence remains evidence-specific: it does not turn mocks, the current host, or retained machinery into remote, Windows, controlled-performance, or release qualification.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Documentation regression] Restored the defined trusted-payload term in the README handoff**
- **Found during:** Required security documentation verification.
- **Issue:** README used “trusted input” while its contract test and the security model use the more precise “trusted application payload” term.
- **Fix:** Restored that exact boundary term without adding serializer or qualification claims.
- **Files modified:** README.md
- **Commit:** d3d0be5

## Deferred Issues

- tests/test_phase5_contract_verifier.py::test_documentation_declares_requirements_not_an_observed_remote_status fails before its assertions about this plan because docs/STORAGE_INITIALIZATION.md lacks the required phase5-initialization-contract markers. It is outside this plan's files and should be addressed by the phase-level gap cycle; the required Plan 09-08 verification passed.

## Known Stubs

None.

## Self-Check: PASSED

- docs/RELEASE_QUALIFICATION.md, docs/SECURITY.md, and docs/CATALOG_AND_TOPOLOGY.md exist.
- Task commits 01ad4c9 and d3d0be5 exist in Git history.
