# Deferred Items

## Task 1 — TensorFlow runtime/config/export removal

- `src/cacheness/storage/handlers/__init__.py` retains three pre-existing Ruff
  `F401` findings for conditional pandas and Polars re-exports. They are outside
  the TensorFlow removal scope and the Task 1 diff does not touch those imports.

## Plan 11-07 — Cross-plan validation normalization

- The frozen non-live suite now passes
  `test_phase11_seed_resolution_is_canonical` after SEED-005 was fulfilled, but
  fails `test_phase11_phase_1_5_6_validations_are_canonical`,
  `test_phase11_phase_7_8_9_validations_are_canonical`, and
  `test_phase11_validation_discovery_and_seed_resolution_are_canonical` because
  the Phase 01 and Phase 07 validation frontmatter still reads `status: complete`
  rather than `status: validated`. Plan 11-08 exclusively owns those records;
  no historical validation evidence was changed by Plan 11-07.
