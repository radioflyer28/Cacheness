# Deferred Items

## Targeted Ruff baseline

- **Status:** open, pre-existing repository lint debt
- **Command:** `uv run ruff check src/cacheness/config.py src/cacheness/__init__.py src/cacheness/storage/path_security.py src/cacheness/storage/guarded_handler_io.py src/cacheness/storage/manifest_repository.py src/cacheness/storage/operation_record.py src/cacheness/storage/operation_repository.py src/cacheness/storage/lifecycle.py src/cacheness/storage/reconciliation.py src/cacheness/storage/coordination.py src/cacheness/storage/blob_store.py src/cacheness/storage/clear_recovery.py src/cacheness/storage/__init__.py src/cacheness/error_handling.py tests/test_config_validation.py tests/test_manifest_repository_cas.py tests/test_blob_store_atomic_lifecycle.py tests/test_blob_store_reconciliation.py tests/test_blob_store_concurrency.py tests/test_blob_store_close_contract.py`
- **Evidence:** exits 1 with 23 F401/F841 findings, all in `src/cacheness/__init__.py`, `src/cacheness/config.py`, and `tests/test_config_validation.py`.
- **Scope proof:** `git diff --name-only a5fe465..HEAD -- src/cacheness/__init__.py src/cacheness/config.py tests/test_config_validation.py` produced no paths; none of the three files changed in 03-10.
- **Disposition:** do not expand this final lifecycle verification plan into unrelated lint cleanup. Lint on all 03-10 changed tests passes.

## SQLite backend interpreter-shutdown destructor warning

- **Status:** open, pre-existing reliability capture candidate
- **Evidence:** the completed full suite logs `SqliteBackend.__del__` attempting cleanup after Python has cleared `sys.meta_path`, producing a non-failing `ImportError` during interpreter shutdown.
- **Scope proof:** the warning occurred after a successful 749-passing Plan 03-07 suite and is unrelated to retired scheduler reachability.
- **Disposition:** retain as a future reliability item; do not mix destructor behavior into the scheduler-retirement change.
