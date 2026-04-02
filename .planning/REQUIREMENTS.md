# Cacheness v0.8.0 — API & Robustness Requirements

## Milestone Requirements

### Concurrency

- [ ] **CONC-01**: SQLite metadata backend uses RLock (not Lock) to prevent deadlock with re-entrant UnifiedCache methods

### Data Integrity

- [ ] **INTG-01**: Failed writes leave a "pending" marker in metadata; orphaned blobs from incomplete writes are detected and cleaned on cache init
- [ ] **INTG-02**: Blob content integrity is validated end-to-end — verify that file_hash covered by metadata HMAC provides sufficient blob tamper detection (or add explicit blob HMAC if gap found)

### Management APIs

- [ ] **MGMT-01**: User can delete all cache entries matching a key prefix via `delete_by_prefix()`, using backend-level SQL LIKE on SQL backends

## Future Requirements

- Document threading model and concurrency boundaries
- Concurrent stress tests for put/get/delete
- Configurable stale-intent threshold for orphan cleanup
- `put_batch()` / `get_batch()` with backend-level transactions
- Add explicit blob HMAC signing if INTG-02 validation reveals a gap

## Out of Scope

| Feature | Reason |
|---------|--------|
| ReadWriteLock | Needs call-chain audit to avoid deadlocks; deferred to future milestone |
| JSON backend concurrency | Documented limitation — "use SQLite for concurrent access" |
| Blob content encryption at rest | Feature addition, not hardening |
| Async/await support | Large feature addition, separate milestone |
| Advanced eviction policies | Large feature addition, separate milestone |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CONC-01 | — | Not started |
| INTG-01 | — | Not started |
| INTG-02 | — | Not started |
| MGMT-01 | — | Not started |
