---
id: SEED-007
status: dormant
planted: 2026-09-15
planted_during: v1.0 Phase 08
trigger_when: when preparing the first remotely qualified immutable release
scope: medium
---

# SEED-007: Qualify real PostgreSQL and Amazon S3, then publish an immutable release

## Why This Matters

Phase 8 intentionally closes on verified local readiness so Cacheness can be used
locally without making remote-service or release claims that have not been proved.
Real PostgreSQL and Amazon S3 qualification still matters before advertising those
topologies or publishing the first immutable release. Keeping that work together
preserves the exact-commit evidence chain without blocking local adoption.

## When to Surface

**Trigger:** when preparing the first remotely qualified immutable release

Surface this seed when a milestone includes real PostgreSQL/Amazon S3 support,
protected live qualification, remote workflow publication, or immutable GitHub
release publication.

## Scope Estimate

**Medium** — configure protected credentials and services, run the existing
exact-SHA qualification workflow, inspect and aggregate the signed evidence, and
publish the immutable release only after every non-deferred gate passes.

## Breadcrumbs

- `.planning/phases/08-production-gates-and-performance-stabilization/08-11-PLAN.md`
- `.planning/phases/08-production-gates-and-performance-stabilization/08-12-PLAN.md`
- `.github/workflows/live_qualification.yml`
- `.github/workflows/quality.yml`
- `tools/run_phase8_qualification.py`
- `tools/verify_phase8_release.py`
- `docs/RELEASE_QUALIFICATION.md`
- `.planning/REQUIREMENTS.md` (`BACK-05`)

## Notes

- Mocked S3, local emulators, and configuration-only preflight are not substitutes
  for real PostgreSQL and Amazon S3 qualification.
- Until this seed is completed, those remote topologies remain `NOT_QUALIFIED` and
  no immutable-release claim may be made.
- Preserve the existing qualification tooling and evidence schemas; this deferral
  changes the milestone gate, not the integrity standard.
