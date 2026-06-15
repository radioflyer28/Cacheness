# Phase 31 Deferred Items

## 31-01 Out-of-Scope Verification Findings

- The exact plan pytest command expands to the broader suite because of repository pytest configuration and currently fails in `tests/test_compress_pickle.py` when `blosc2` is unavailable. The scoped SEC-01 command with addopts cleared passes for `tests/test_cache_signing.py` and `tests/test_key_rotation_api.py`.
- `uv run --python 3.12 ty check ...` on the plan's touched-file list still reports pre-existing diagnostics in `src/cacheness/_verification_mixin.py`, `src/cacheness/config.py`, and `src/cacheness/storage/blob_store.py` around mixin attributes, typed dict construction, and existing Path/string annotations. Introduced test typing diagnostics were resolved before commit.
