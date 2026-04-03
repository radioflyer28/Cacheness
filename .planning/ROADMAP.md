# Roadmap: Cacheness

## Milestones

- ✅ **v0.7.0 Cleanup & Hardening** — Phases 1-6 (shipped 2026-04-02) — [archive](milestones/v0.7.0-ROADMAP.md)
- ✅ **v0.8.0 API & Robustness** — Phases 7-10 (shipped 2026-04-03) — [archive](milestones/v0.8.0-ROADMAP.md)
- 🔄 **v0.9.0 Completeness & Hardening** — Phases 11-14 (in progress)

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

<details>
<summary>✅ v0.8.0 API & Robustness (Phases 7-10) — SHIPPED 2026-04-03</summary>

- [x] Phase 7: Concurrency Foundation — completed 2026-04-02
- [x] Phase 8: Crash-Safe Write Intent Logging — completed 2026-04-02
- [x] Phase 9: Blob Integrity Validation — completed 2026-04-02
- [x] Phase 10: Prefix Deletion API — completed 2026-04-02

</details>

### v0.9.0 Completeness & Hardening

#### Phase 11: put_batch() API
**Goal:** Implement the missing `put_batch()` batch operation on UnifiedCache, completing the batch operations surface alongside existing `get_batch()`, `delete_batch()`, and `touch_batch()`.
**Requirements:** MGMT-01
**Success Criteria:**
- `put_batch()` method exists on UnifiedCache and accepts a list of (key_kwargs, data) tuples
- Backend-level transaction support for SQLite
- Tests cover JSON and SQLite backends, multiple data types, partial failure handling
- Status: Not Started

#### Phase 12: Threading Model Documentation
**Goal:** Fix contradictory concurrency documentation and accurately document the post-v0.8.0 threading model. API_REFERENCE.md claims "thread-safe for all operations" while TROUBLESHOOTING.md says "not thread-safe" — both are outdated.
**Requirements:** DOC-01
**Success Criteria:**
- API_REFERENCE.md Thread Safety section documents actual RLock behavior and backend-level protections
- TROUBLESHOOTING.md is consistent with API_REFERENCE.md
- Concurrency boundaries clearly documented per backend
- Status: Not Started

#### Phase 13: Deserialization Security
**Goal:** Harden pickle/dill deserialization paths with defense-in-depth documentation and verification that existing protections (HMAC + file_hash) cover blob tampering.
**Requirements:** SEC-01
**Success Criteria:**
- SECURITY.md documents the layered defense model (HMAC metadata signing + file_hash blob verification)
- Code comments at all 7 deserialization sites reference the security model
- Test verifies that blob tampering is detected when verify_hashes=True (default)
- Status: Not Started

#### Phase 14: Windows Key File Permissions
**Goal:** Replace the no-op `chmod(0o600)` on Windows with `icacls`-based ACL restriction so the signing key file is actually protected.
**Requirements:** SEC-02
**Success Criteria:**
- Windows: `icacls` restricts key file to current user
- Graceful fallback on `icacls` failure (log warning, continue)
- Test covers the Windows permission code path
- SECURITY.md documents Windows key file behavior
- Status: Not Started

## Progress

| Phase | Milestone | Status | Completed |
|-------|-----------|--------|-----------|
| 1. Handler Package Split | v0.7.0 | Complete | 2026-04-02 |
| 2. Metadata Package Split | v0.7.0 | Complete | 2026-04-02 |
| 3. Core Mixin Decomposition | v0.7.0 | Complete | 2026-04-02 |
| 4. Exception Handling | v0.7.0 | Complete | 2026-04-02 |
| 5. Security Hardening | v0.7.0 | Complete | 2026-04-02 |
| 6. Test Gaps & Handler Robustness | v0.7.0 | Complete | 2026-04-02 |
| 7. Concurrency Foundation | v0.8.0 | Complete | 2026-04-02 |
| 8. Crash-Safe Write Intent Logging | v0.8.0 | Complete | 2026-04-02 |
| 9. Blob Integrity Validation | v0.8.0 | Complete | 2026-04-02 |
| 10. Prefix Deletion API | v0.8.0 | Complete | 2026-04-02 |
| 11. put_batch() API | v0.9.0 | Not Started | — |
| 12. Threading Model Documentation | v0.9.0 | Not Started | — |
| 13. Deserialization Security | v0.9.0 | Not Started | — |
| 14. Windows Key File Permissions | v0.9.0 | Not Started | — |
