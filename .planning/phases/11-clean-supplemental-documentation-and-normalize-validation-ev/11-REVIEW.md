---
phase: 11-clean-supplemental-documentation-and-normalize-validation-ev
reviewed: 2026-09-19T22:51:20Z
depth: standard
files_reviewed: 23
files_reviewed_list:
  - .github/workflows/quality.yml
  - AGENTS.md
  - docs/API_REFERENCE.md
  - docs/README.md
  - docs/RELEASE_QUALIFICATION.md
  - pyproject.toml
  - src/cacheness/config.py
  - src/cacheness/handlers.py
  - src/cacheness/storage/handlers/__init__.py
  - tests/packaging/test_wheel_matrix.py
  - tests/qualification/test_phase8_platform.py
  - tests/qualification/test_phase8_quality_workflow.py
  - tests/qualification/test_phase8_release.py
  - tests/test_full_suite_environment.py
  - tests/test_phase10_sqlcache_removal.py
  - tests/test_phase1_quality_gates.py
  - tests/test_phase9_documentation.py
  - tests/test_phase9_evidence_metadata.py
  - tests/test_phase9_quality_workflow.py
  - tools/phase8_evidence.py
  - tools/run_phase8_local_gates.py
  - tools/run_phase8_packaging.py
  - tools/run_phase8_platform_gates.py
findings:
  critical: 1
  warning: 1
  info: 0
  total: 2
status: issues_found
---

# Phase 11: Code Review Report

**Reviewed:** 2026-09-19T22:51:20Z  
**Depth:** standard  
**Files Reviewed:** 23  
**Status:** issues_found

## Summary

Reviewed the 23 surviving non-planning files changed since `e144899`, including the TensorFlow cutover, package/qualification tooling, CI, tests, and current guidance. Deleted supplemental documents and the deleted TensorFlow test/guide were checked as removal targets, not counted as surviving source files. The focused package, platform, documentation, and evidence tests pass, but the current release guide publishes a suite command that fails on live tests, and the normal CI path no longer runs the optional-extra wheel qualification that the removed TensorFlow job previously hosted.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: Published complete-suite command selects unconfigured live tests [BLOCKER]

**File:** `docs/RELEASE_QUALIFICATION.md:43-52` (also asserted by `tests/test_full_suite_environment.py:77-83`)  
**Issue:** The new canonical guide calls `uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false` the complete local regression command, but omits the repository's live-marker exclusion. It selects `tests/integration/test_postgresql_authority.py`, `test_s3_generation.py`, and `test_remote_topology.py`. These use `live_qualification_resources`, defined only in `tests/qualification/conftest.py`, which is not visible to sibling `tests/integration/`. Running the exact command against the PostgreSQL test errors at setup with `fixture 'live_qualification_resources' not found` (exit 1). Thus the published local command cannot be a green acceptance gate even without service credentials; the Phase 11 recorded green suite used the different, correctly filtered `-m 'not (live_postgresql or live_aws_s3 or live_remote)'` command.  
**Fix:** Document the same frozen non-live command used by the Phase 11 gate, and update the command contract accordingly:

```bash
uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false \
  -m 'not (live_postgresql or live_aws_s3 or live_remote)'
```

Keep protected live qualification as a separate, explicitly provisioned workflow.

## Warnings

### WR-01: Ordinary CI no longer qualifies published optional extras [WARNING]

**File:** `.github/workflows/quality.yml:51-54` and `.github/workflows/quality.yml:184-226`  
**Issue:** Deleting `tensorflow-compatible` also deletes the only pull-request/push invocation of `tools/run_phase8_local_gates.py packaging` (formerly on Python 3.12). The surviving `linux-stable` job runs `platform`, `core`, and example checks, but `core` explicitly excludes packaging; the only remaining packaging job is gated by `github.event_name == 'workflow_dispatch'`. A regression in wheel membership, optional-extra metadata, or a source-free extra probe can therefore pass normal CI despite five extras remaining published. The new wheel tests cover absence and local round trips, but they do not restore the full five-extra qualification to PR/push CI.  
**Fix:** Run the retained packaging gate on the pinned primary stable Linux row in normal CI (or add an equivalent non-advisory job), while keeping the manual exact-SHA release-candidate packaging job and its provenance controls.

---

_Reviewed: 2026-09-19T22:51:20Z_  
_Reviewer: the agent (gsd-code-reviewer)_  
_Depth: standard_
