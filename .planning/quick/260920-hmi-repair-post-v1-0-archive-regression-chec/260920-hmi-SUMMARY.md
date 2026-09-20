---
quick_id: 260920-hmi
status: complete
date: 2026-09-20
source_commit: 21579fd4c9d40e0e34cdf72e7f62e873509b881c
---

# Post-archive regression checks repaired

Updated current verifiers and tests to read immutable v1.0 phase, roadmap,
requirements, and audit artifacts from `.planning/milestones/`; fixed the
current release-qualification link. The Phase 3 tag test now checks that the
retired scheduler is absent from the tagged tree, not from the tag's entire
ancestry. Historical Phase 11 provenance remains bound to the immutable
v1.0 snapshot, while fixture tests still reject changes to their qualified
source trees. No historical evidence was rewritten and no compatibility
directory or test exclusion was added.

Validation on a clean checkout of `21579fd4c9d40e0e34cdf72e7f62e873509b881c`:

- `uv run --isolated --all-extras --group dev --frozen pytest -q -o
  log_cli=false -m 'not (live_postgresql or live_aws_s3 or live_remote)'`: PASS
  (three expected environment/platform skips).
- Scoped Ruff over all touched Python files: PASS.
- `tools/run_phase8_packaging.py`: PASS for base wheel and all five declared
  extras, exact revision above; evidence in
  `/private/tmp/cacheness-postarchive-21579fd-packaging.json`.

Live PostgreSQL/S3, controlled-Linux performance, native Windows, and
immutable publication remain unqualified/nonpublished. This repair commit is
not itself the final squashed integration SHA.
