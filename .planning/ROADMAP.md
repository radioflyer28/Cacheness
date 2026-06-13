# Roadmap: Cacheness

## Milestones

- ✅ **v0.7.0 Cleanup & Hardening** — Phases 1-6 (shipped 2026-04-02) — [archive](milestones/v0.7.0-ROADMAP.md)
- ✅ **v0.8.0 API & Robustness** — Phases 7-10 (shipped 2026-04-03) — [archive](milestones/v0.8.0-ROADMAP.md)
- ✅ **v0.9.0 Completeness & Hardening** — Phases 11-14 (shipped 2026-04-03) — [archive](milestones/v0.9.0-ROADMAP.md)
- ✅ **v0.10.0 Security & Architecture** — Phases 15-22 (shipped 2026-04-06) — [archive](milestones/v0.10.0-ROADMAP.md)
- ✅ **v0.11.0 Cross-Backend Hardening** — Phases 23-27 (shipped 2026-04-07) — [archive](milestones/v0.11.0-ROADMAP.md)
- 🟡 **v0.12.0 Reliability Remediation** — Phases 28-32 (planned 2026-06-12)

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

<details open>
<summary>🟡 v0.12.0 Reliability Remediation (Phases 28-32) — PLANNED 2026-06-12</summary>

- [ ] Phase 28: Silent Data-Loss Remediation — Wave 1 fixes plus cache-key stability and property tests
- [ ] Phase 29: TTL & Eviction Consistency — per-entry expiry, cleanup parity, counters/timestamps, remote eviction
- [ ] Phase 30: Multi-Process & Backend Parity — temp files, metadata parity, non-destructive overwrites, blob enumeration
- [ ] Phase 31: Security & Storage-Mode Posture — signatures, encrypted reads, key rotation, storage-mode guards, durability contract
- [ ] Phase 32: Small Fixes & Release Polish — independent low-risk fixes and release/version polish

</details>

## Progress

| Phase | Milestone | Status | Completed |
|-------|-----------|--------|-----------|
| 1-6 | v0.7.0 | Complete | 2026-04-02 |
| 7-10 | v0.8.0 | Complete | 2026-04-03 |
| 11-14 | v0.9.0 | Complete | 2026-04-03 |
| 15-22 | v0.10.0 | Complete | 2026-04-06 |
| 23-27 | v0.11.0 | Complete | 2026-04-07 |
| 28-32 | v0.12.0 | Planned | — |

## v0.12.0 Phase Details

### Phase 28: Silent Data-Loss Remediation

**Goal:** Eliminate the highest-risk data-loss and broken-guarantee defects before broader hardening work begins.
**Requirements:** REL-01, REL-02, REL-03, REL-04, REL-05, REL-06, KEY-01, KEY-02
**Source:** Pending todos for TASK-1 through TASK-4, property-based cache-key todo, SEED-006
**Plans:** 5 plans

Success criteria:
1. `clear_all()` and `clear_all_namespaces()` remove namespace blob files without deleting reserved metadata or key files.
2. Stale write-intent cleanup resolves relative paths under `cache_dir`, preserves committed entries, and works in storage mode.
3. JSON backend data-critical save failures surface to callers and corrupt metadata files are preserved.
4. Cache-key fallback behavior is stable across subprocesses and protected by property-based tests.
5. Tier-1 tests from TASK-1 through TASK-4 pass with the TensorFlow tests ignored on Windows.

Plans:
- [ ] 28-01-PLAN.md - TASK-1 clear_all namespace blob cleanup (REL-01)
- [ ] 28-02-PLAN.md - TASK-2 write-intent path resolution, committed-entry guard, storage-mode cleanup (REL-02, REL-03, REL-04)
- [ ] 28-03-PLAN.md - SEED-006 record write intent before blob write follow-up (REL-02, REL-03, REL-04)
- [ ] 28-04-PLAN.md - TASK-3 JSON backend persistence failures and corrupt-file preservation (REL-05, REL-06)
- [ ] 28-05-PLAN.md - TASK-4 stable cache-key fallbacks and property tests (KEY-01, KEY-02)

### Phase 29: TTL & Eviction Consistency

**Goal:** Make TTL and eviction behavior coherent end-to-end across metadata backends, blob files, stats, and remote blob storage.
**Requirements:** TTL-01, TTL-02, TTL-03, TTL-04
**Source:** Backlog Phase 999.1, TASK-5 through TASK-8
**Plans:** 2/4 plans executed

Success criteria:
1. Stored `expires_at` controls per-entry expiry when present; global TTL applies only as fallback.
2. Init-time cleanup uses the public cleanup path and removes both metadata and blob files.
3. Metadata-only updates preserve `created_at`, signatures, and access-count semantics.
4. Eviction routes remote blob deletion through the blob backend and verifies S3/memory-style URI cleanup.

Plans:
**Wave 1**

- [x] 29-01-PLAN.md - TASK-5 stored expires_at read and cleanup semantics (TTL-01)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 29-02-PLAN.md - TASK-6 init cleanup through public cleanup path (TTL-02)
- [ ] 29-03-PLAN.md - TASK-7 access-count, created_at, TTL field, and signature preservation (TTL-03)

**Wave 3** *(blocked on Wave 2 completion)*

- [ ] 29-04-PLAN.md - TASK-8 remote URI blob deletion during size eviction (TTL-04)

### Phase 30: Multi-Process & Backend Parity

**Goal:** Ensure local and remote metadata/blob backends expose the same observable behavior under realistic overwrite, concurrency, and integrity-check paths.
**Requirements:** PAR-01, PAR-02, PAR-03, PAR-04
**Source:** Backlog Phase 999.2, TASK-9 through TASK-12
**Plans:** 0 plans

Success criteria:
1. Filesystem blob writes use unique temp files and avoid deterministic cross-process collisions.
2. SQLite and PostgreSQL preserve custom user metadata with parity against JSON.
3. Failed same-key overwrites restore or preserve the previous committed value in cache and storage modes.
4. Integrity and cleanup enumeration sees custom-handler blobs and excludes only reserved files.

Plans:
- [ ] TBD via `$gsd-discuss-phase 30` / `$gsd-plan-phase 30`

### Phase 31: Security & Storage-Mode Posture

**Goal:** Strengthen signing, encryption, rotation, and storage-mode durability guarantees without silently changing public API semantics.
**Requirements:** SEC-01, SEC-02, SEC-03, SEC-04, STRG-01, STRG-02, STRG-03
**Source:** Backlog Phase 999.3, TASK-13 through TASK-15, SEED-001, SEED-004, SEED-005, SEED-006
**Plans:** 0 plans

Success criteria:
1. Signature downgrade hardening and unsigned-entry risk documentation are tested and documented.
2. Encrypted reads use backend-routed reads and prefer in-memory handler paths over plaintext temp files.
3. Key rotation is two-phase enough that interrupted rotation leaves the old key and entries usable.
4. UnifiedCache and BlobStore signing converge on a shared canonical field strategy with compatibility handling.
5. Storage-mode destructive APIs and fsync/durability behavior are explicit, tested, and documented.

Plans:
- [ ] TBD via `$gsd-discuss-phase 31` / `$gsd-plan-phase 31`

### Phase 32: Small Fixes & Release Polish

**Goal:** Land low-risk independent code-review fixes and polish the package surface for the v0.12.0 release.
**Requirements:** POL-01, POL-02, POL-03, POL-04, POL-05, POL-06, POL-07, POL-08
**Source:** Backlog Phase 999.4, TASK-16 through TASK-23
**Plans:** 0 plans

Success criteria:
1. Each small fix is independently tested and committed with minimal blast radius.
2. Package version metadata and root imports match user expectations.
3. S3 and SQLite lifecycle/reporting fixes have targeted tests or explicit mocked verification.
4. Full suite passes once before v0.12.0 completion.

Plans:
- [ ] TBD via `$gsd-discuss-phase 32` / `$gsd-plan-phase 32`

## Backlog

Items sourced from the 2026-06-12 code review (`docs/CODE_REVIEW_FINDINGS.md`). Execution specs with acceptance criteria live in `docs/CODE_REVIEW_ACTIONS.md` (TASK-N references below). Wave-1 items (silent data loss) are NOT here — they're captured as pending todos for immediate work.

### Phase 999.1: TTL & eviction consistency (PROMOTED to Phase 29)

**Goal:** One coherent TTL/eviction story — per-entry `expires_at` honored end-to-end, cleanup paths delete blob files, counters survive overwrites, remote blobs evicted.
**Requirements:** Code review Wave 2 — TASK-5 (R5 per-entry TTL), TASK-6 (R6 init-cleanup orphans blobs), TASK-7 (R9/R10 preserve access_count/created_at), TASK-8 (R13 remote blob eviction)
**Plans:** 0 plans

Plans:
- [ ] TBD (promote with /gsd-review-backlog when ready)

### Phase 999.2: Multi-process & parity hardening (PROMOTED to Phase 30)

**Goal:** Same observable behavior across backends and safe concurrent/multi-process operation — no silent metadata loss on SQLite/PG, no temp-file races, non-destructive overwrites, integrity checks that see all blobs.
**Requirements:** Code review Wave 3 — TASK-9 (R11 unique temp names), TASK-10 (U2 user-metadata parity), TASK-11 (R7 non-destructive overwrite, incl. `_storage_mode_put`), TASK-12 (R12 backend-driven blob enumeration)
**Plans:** 0 plans

Plans:
- [ ] TBD (promote with /gsd-review-backlog when ready)

### Phase 999.3: Security posture hardening (PROMOTED to Phase 31)

**Goal:** Signing and encryption guarantees that hold under the documented threat model — no signature downgrade paths, no plaintext on disk during encrypted reads, crash-safe key rotation.
**Requirements:** Code review Wave 4 — TASK-13 (S1 min signature version + unsigned-entry docs), TASK-14 (S2/S3 in-memory decryption, backend-routed reads), TASK-15 (S4 two-phase rotate_key)
**Plans:** 0 plans

Plans:
- [ ] TBD (promote with /gsd-review-backlog when ready)

### Phase 999.4: Code review small fixes (PROMOTED to Phase 32)

**Goal:** Batch of low-risk independent fixes from the code review — each one commit, no design decisions needed.
**Requirements:** TASK-16 (metadata dict mutation), TASK-17 (sanitize-key collisions), TASK-18 (double read in get), TASK-19 (pyproject version 0.6.0→current), TASK-20 (absolute blob_id rejection), TASK-21 (SQLite PRAGMA placement), TASK-22 (S3 delete failure reporting), TASK-23 (root re-export of UnifiedCache)
**Plans:** 0 plans

Plans:
- [ ] TBD (promote with /gsd-review-backlog when ready)
