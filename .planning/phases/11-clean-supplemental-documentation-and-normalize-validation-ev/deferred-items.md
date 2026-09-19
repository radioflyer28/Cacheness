# Deferred Items

## Task 1 — TensorFlow runtime/config/export removal

- `src/cacheness/storage/handlers/__init__.py` retains three pre-existing Ruff
  `F401` findings for conditional pandas and Polars re-exports. They are outside
  the TensorFlow removal scope and the Task 1 diff does not touch those imports.
