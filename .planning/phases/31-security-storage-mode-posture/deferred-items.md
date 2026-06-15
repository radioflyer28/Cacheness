# Phase 31 Deferred Items

## 31-01 Out-of-Scope Verification Findings

- The exact plan pytest command expands to the broader suite because of repository pytest configuration and currently fails in `tests/test_compress_pickle.py` when `blosc2` is unavailable. The scoped SEC-01 command with addopts cleared passes for `tests/test_cache_signing.py` and `tests/test_key_rotation_api.py`.
- `uv run --python 3.12 ty check ...` on the plan's touched-file list still reports pre-existing diagnostics in `src/cacheness/_verification_mixin.py`, `src/cacheness/config.py`, and `src/cacheness/storage/blob_store.py` around mixin attributes, typed dict construction, and existing Path/string annotations. Introduced test typing diagnostics were resolved before commit.

## 31-05 Out-of-Scope Verification Findings

- `uv run --python 3.12 pytest -o addopts='' tests/test_storage_mode.py tests/test_core.py -x -q --ignore=tests/test_tensorflow_handler.py` reaches the new STRG-01 warning tests successfully, then fails on the pre-existing same-key overwrite rollback regression `tests/test_storage_mode.py::TestNoAutoDelete::test_failed_same_key_overwrite_preserves_previous_blob`. This maps to overwrite/blob preservation work, not the warning-first STRG-01 behavior, and Phase 31 Plan 05 explicitly does not reopen Phase 30 overwrite semantics.
- `uv run --python 3.12 ty check src/cacheness/core.py src/cacheness/_storage_mode_mixin.py tests/test_storage_mode.py tests/test_core.py tests/test_cache_integrity.py` still reports pre-existing mixin attribute and legacy test typing diagnostics, including `_storage_mode_mixin.py` unresolved attributes and older `tests/test_core.py` / `tests/test_storage_mode.py` optional-entry annotations. Ruff validation passes and the new warning behavior tests pass.

## 31-02 Out-of-Scope Verification Findings

- The literal plan pytest command `uv run --python 3.12 pytest tests/test_blob_store.py tests/test_encryption_at_rest.py tests/test_handler_bytes_protocol.py -x -q --ignore=tests/test_tensorflow_handler.py` fails before collection in this Windows environment because xdist tries to create worker temp directories under `C:\Users\akriz\AppData\Local\Temp\pytest-of-akriz`, which is ACL-denied. The controlled equivalent with addopts cleared, pytest cache disabled, and `--basetemp .uv-cache\pytest-31-02-plan-final` passes: 87 passed, 3 skipped.
- `uv run --python 3.12 ty check src/cacheness/storage/blob_store.py tests/test_blob_store.py tests/test_encryption_at_rest.py tests/test_handler_bytes_protocol.py` still reports pre-existing diagnostics in `blob_store.py` signing/config typing and legacy test metadata/skip typing. The new encrypted-read helper-specific diagnostics were resolved before commit; ruff format/check pass.
