# Phase 7 Deferred Items

## Pre-existing deterministic-suite observation

- Observed: tests/test_blob_store_concurrency.py::test_clear_and_delete_converge_after_an_exact_snapshot raised CacheBlobLifecycleConflictError during the first all-extras suite attempt on 2026-09-10.
- Classification: pre-existing Phase 3 concurrency evidence; the exact repeated fixed non-live suite passed 1311 tests with zero failures/errors.
- Disposition: do not modify lifecycle coordination or reopen a race-patch loop in Phase 7. ADR 0001 and the approved Wave 1 boundary prohibit it. Preserve this observation for Phase 8 fault/performance qualification.

