# Phase 7 Deferred Items

## Pre-existing deterministic-suite observation

- **Status:** acknowledged
- Observed: tests/test_blob_store_concurrency.py::test_clear_and_delete_converge_after_an_exact_snapshot raised CacheBlobLifecycleConflictError during the first all-extras suite attempt on 2026-09-10.
- Classification: pre-existing Phase 3 concurrency evidence; the exact repeated fixed non-live suite passed 1311 tests with zero failures/errors.
- Disposition: do not modify lifecycle coordination or reopen a race-patch loop in Phase 7. ADR 0001 and the approved Wave 1 boundary prohibit it. Preserve this observation for Phase 8 fault/performance qualification.

## Current direct-suite observation

- **Status:** acknowledged
- Observed: after the all-mode fixed verifier exited zero on 2026-09-11, a separate direct non-live suite invocation reported `tests/test_blob_store_concurrency.py::test_clear_and_delete_converge_after_an_exact_snapshot` as `CacheBlobLifecycleConflictError` (1344 passed, 9 skipped, 1 failed).
- Classification: same Phase 3 concurrency observation; it is not a Plan 07-20 through 07-22 migration/rebuild selector failure.
- Disposition: keep the failed direct invocation visible in `07-VALIDATION.md`; do not retry it for a better result and do not add lifecycle coordination in Phase 7. Phase 8 owns fault/performance qualification under ADR 0001.

## WR-02 — future handler-registration API-contract decision

- **Status:** acknowledged
- Observed: `HandlerRegistry.register_handler(..., name=...)` uses `name=` for duplicate validation/logging but does not persist it; lookup, list, and unregister operate on `handler.data_type`.
- Scope: unrelated to the three verified Phase 7 migration/rebuild blockers and intentionally not fixed or claimed by Plans 07-20 through 07-22.
- Future work: decide whether to remove `name=` from the public registration API or make it a real persisted alias. Whichever contract is selected must define duplicate-`data_type` behavior and add lookup, list, unregister, alias-removal, and duplicate-data-type-under-different-alias tests.
