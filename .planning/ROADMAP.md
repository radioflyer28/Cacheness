# Roadmap: Cacheness

## Milestones

- ✅ **v0.7.0 Cleanup & Hardening** — Phases 1-6 (shipped 2026-04-02) — [archive](milestones/v0.7.0-ROADMAP.md)
- 🔄 **v0.8.0 API & Robustness** — Phases 7-10

## Phases

<details>
<summary>✅ v0.7.0 Cleanup & Hardening (Phases 1-6) — SHIPPED 2026-04-02</summary>

- [x] Phase 1: Handler Package Split — completed 2026-04-02
- [x] Phase 2: Metadata Package Split — completed 2026-04-02
- [x] Phase 3: Core Mixin Decomposition — completed 2026-04-02
- [x] Phase 4: Exception Handling — completed 2026-04-02
- [x] Phase 5: Security Hardening — completed 2026-04-02
- [x] Phase 6: Test Gaps & Handler Robustness — completed 2026-04-02

</details>

### v0.8.0 API & Robustness

- [ ] **Phase 7: Concurrency Foundation** — Fix SQLite Lock→RLock deadlock risk and document threading model
- [ ] **Phase 8: Crash-Safe Write Intent Logging** — Prevent orphaned blobs via write intent journal and startup recovery
- [ ] **Phase 9: Blob Integrity Validation** — Validate file_hash coverage under metadata HMAC or add blob HMAC if gap found
- [ ] **Phase 10: Prefix Deletion API** — Add `delete_by_prefix()` with backend-level SQL LIKE fast path

## Phase Details

### Phase 7: Concurrency Foundation
**Goal**: SQLite metadata backend is safe under re-entrant calls — no deadlocks from composed operations
**Depends on**: Nothing (foundational fix)
**Requirements**: CONC-01
**Success Criteria** (what must be TRUE):
  1. `SqliteBackend` uses `threading.RLock` — a re-entrant call path (e.g., `put()` calling metadata methods that re-acquire the lock) completes without deadlock
  2. Existing thread-safety tests pass unchanged with the RLock swap
  3. A new stress test exercises re-entrant lock acquisition (nested backend calls from a single thread) and completes without hanging
**Plans:** 1 plan
Plans:
- [ ] 07-01-PLAN.md — Lock→RLock swap, lock consistency audit, re-entrancy tests

### Phase 8: Crash-Safe Write Intent Logging
**Goal**: Incomplete writes are automatically detected and cleaned up — no orphaned blobs accumulate silently after crashes
**Depends on**: Phase 7 (safe locking needed for cleanup-on-init)
**Requirements**: INTG-01
**Success Criteria** (what must be TRUE):
  1. A "pending" marker (intent journal entry) is written before blob write begins and removed after metadata commit succeeds
  2. On cache init (when `cleanup_on_init=True`), stale intent entries older than a configurable threshold are detected and their orphaned blobs deleted
  3. A simulated crash (kill between blob write and metadata commit) leaves an intent entry that is cleaned up on next cache init
  4. Normal successful writes leave no residual intent entries
**Plans**: TBD

### Phase 9: Blob Integrity Validation
**Goal**: Blob content integrity is validated end-to-end — users can trust that stored blobs have not been tampered with or corrupted
**Depends on**: Phase 7 (safe locking for verification paths)
**Requirements**: INTG-02
**Success Criteria** (what must be TRUE):
  1. The existing `file_hash` field is confirmed to be covered by the metadata HMAC signature (i.e., tampering with blob content changes `file_hash`, which invalidates the signed metadata entry)
  2. `verify_integrity()` detects a blob whose content has been modified on disk (hash mismatch) when `verify_hashes=True`
  3. If a gap is found (file_hash not in signed fields, or xxhash insufficiency for authentication), an explicit `blob_hmac` field is added and verified — with backward compatibility for entries without it
**Plans**: TBD

### Phase 10: Prefix Deletion API
**Goal**: Users can bulk-delete cache entries by key prefix in a single call, with backend-optimized performance
**Depends on**: Phase 7 (safe locking for batch deletes), Phase 8 (crash safety awareness for bulk operations)
**Requirements**: MGMT-01
**Success Criteria** (what must be TRUE):
  1. `cache.delete_by_prefix("myapp/models/")` deletes all entries whose cache key starts with the given prefix, returning the count of deleted entries
  2. SQLite and PostgreSQL backends use SQL `LIKE` (or equivalent) for the prefix match — not Python-side iteration over all keys
  3. JSON backend falls back to Python-side prefix filtering (documented as slower for large caches)
  4. Deleting by prefix also removes the corresponding blob files, not just metadata
  5. Calling `delete_by_prefix()` with a prefix that matches no entries returns 0 without error
**Plans**: TBD

## Progress

| Phase | Milestone | Status | Completed |
|-------|-----------|--------|-----------|
| 1. Handler Package Split | v0.7.0 | Complete | 2026-04-02 |
| 2. Metadata Package Split | v0.7.0 | Complete | 2026-04-02 |
| 3. Core Mixin Decomposition | v0.7.0 | Complete | 2026-04-02 |
| 4. Exception Handling | v0.7.0 | Complete | 2026-04-02 |
| 5. Security Hardening | v0.7.0 | Complete | 2026-04-02 |
| 6. Test Gaps & Handler Robustness | v0.7.0 | Complete | 2026-04-02 |
| 7. Concurrency Foundation | v0.8.0 | Not started | - |
| 8. Crash-Safe Write Intent Logging | v0.8.0 | Not started | - |
| 9. Blob Integrity Validation | v0.8.0 | Not started | - |
| 10. Prefix Deletion API | v0.8.0 | Not started | - |
