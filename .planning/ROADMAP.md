# Roadmap: Cacheness

## Milestones

- ✅ **v0.7.0 Cleanup & Hardening** — Phases 1-6 (shipped 2026-04-02) — [archive](milestones/v0.7.0-ROADMAP.md)
- ✅ **v0.8.0 API & Robustness** — Phases 7-10 (shipped 2026-04-03) — [archive](milestones/v0.8.0-ROADMAP.md)
- ✅ **v0.9.0 Completeness & Hardening** — Phases 11-14 (shipped 2026-04-03) — [archive](milestones/v0.9.0-ROADMAP.md)
- ✅ **v0.10.0 Security & Architecture** — Phases 15-22 (shipped 2026-04-06) — [archive](milestones/v0.10.0-ROADMAP.md)
- ✅ **v0.11.0 Cross-Backend Hardening** — Phases 23-27 (shipped 2026-04-07) — [archive](milestones/v0.11.0-ROADMAP.md)

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

<details>
<summary>✅ v0.9.0 Completeness & Hardening (Phases 11-14) — SHIPPED 2026-04-03</summary>

- [x] Phase 11: put_batch() API — completed 2026-04-03
- [x] Phase 12: Threading Model Documentation — completed 2026-04-03
- [x] Phase 13: Deserialization Security — completed 2026-04-03
- [x] Phase 14: Windows Key File Permissions — completed 2026-04-03

</details>

<details>
<summary>✅ v0.10.0 Security & Architecture (Phases 15-22) — SHIPPED 2026-04-06</summary>

- [x] Phase 15: Configurable Key Fallback — completed 2026-04-03
- [x] Phase 16: HKDF Key Derivation — completed 2026-04-03
- [x] Phase 17: Key Rotation API — completed 2026-04-03
- [x] Phase 18: Encryption at Rest — completed 2026-04-03
- [x] Phase 19: Core Decomposition II — completed 2026-04-03
- [ ] Phase 20: Concurrency & Atomics Verification — superseded by Phase 22
- [x] Phase 21: Retroactive Verification & Docs Cleanup — completed 2026-04-06
- [x] Phase 22: Concurrency & Integration Testing — completed 2026-04-06

</details>

<details>
<summary>✅ v0.11.0 Cross-Backend Hardening (Phases 23-27) — SHIPPED 2026-04-07</summary>

- [x] Phase 23: Encryption Schema & Storage — completed 2026-04-06
- [x] Phase 24: Cross-Backend Test Parity — completed 2026-04-06
- [x] Phase 25: Inline Blob Encryption — completed 2026-04-06
- [x] Phase 26: Integration & Hardening — completed 2026-04-07
- [x] Phase 27: Retroactive Verification & Doc Cleanup — completed 2026-04-07

</details>

## Progress

| Phase | Milestone | Status | Completed |
|-------|-----------|--------|-----------|
| 1-6 | v0.7.0 | Complete | 2026-04-02 |
| 7-10 | v0.8.0 | Complete | 2026-04-03 |
| 11-14 | v0.9.0 | Complete | 2026-04-03 |
| 15-22 | v0.10.0 | Complete | 2026-04-06 |
| 23-27 | v0.11.0 | Complete | 2026-04-07 |

## Backlog

Items sourced from the 2026-06-12 code review (`docs/CODE_REVIEW_FINDINGS.md`). Execution specs with acceptance criteria live in `docs/CODE_REVIEW_ACTIONS.md` (TASK-N references below). Wave-1 items (silent data loss) are NOT here — they're captured as pending todos for immediate work.

### Phase 999.1: TTL & eviction consistency (BACKLOG)

**Goal:** One coherent TTL/eviction story — per-entry `expires_at` honored end-to-end, cleanup paths delete blob files, counters survive overwrites, remote blobs evicted.
**Requirements:** Code review Wave 2 — TASK-5 (R5 per-entry TTL), TASK-6 (R6 init-cleanup orphans blobs), TASK-7 (R9/R10 preserve access_count/created_at), TASK-8 (R13 remote blob eviction)
**Plans:** 0 plans

Plans:
- [ ] TBD (promote with /gsd-review-backlog when ready)

### Phase 999.2: Multi-process & parity hardening (BACKLOG)

**Goal:** Same observable behavior across backends and safe concurrent/multi-process operation — no silent metadata loss on SQLite/PG, no temp-file races, non-destructive overwrites, integrity checks that see all blobs.
**Requirements:** Code review Wave 3 — TASK-9 (R11 unique temp names), TASK-10 (U2 user-metadata parity), TASK-11 (R7 non-destructive overwrite, incl. `_storage_mode_put`), TASK-12 (R12 backend-driven blob enumeration)
**Plans:** 0 plans

Plans:
- [ ] TBD (promote with /gsd-review-backlog when ready)

### Phase 999.3: Security posture hardening (BACKLOG)

**Goal:** Signing and encryption guarantees that hold under the documented threat model — no signature downgrade paths, no plaintext on disk during encrypted reads, crash-safe key rotation.
**Requirements:** Code review Wave 4 — TASK-13 (S1 min signature version + unsigned-entry docs), TASK-14 (S2/S3 in-memory decryption, backend-routed reads), TASK-15 (S4 two-phase rotate_key)
**Plans:** 0 plans

Plans:
- [ ] TBD (promote with /gsd-review-backlog when ready)

### Phase 999.4: Code review small fixes (BACKLOG)

**Goal:** Batch of low-risk independent fixes from the code review — each one commit, no design decisions needed.
**Requirements:** TASK-16 (metadata dict mutation), TASK-17 (sanitize-key collisions), TASK-18 (double read in get), TASK-19 (pyproject version 0.6.0→current), TASK-20 (absolute blob_id rejection), TASK-21 (SQLite PRAGMA placement), TASK-22 (S3 delete failure reporting), TASK-23 (root re-export of UnifiedCache)
**Plans:** 0 plans

Plans:
- [ ] TBD (promote with /gsd-review-backlog when ready)

