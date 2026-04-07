# Roadmap: Cacheness

## Milestones

- ✅ **v0.7.0 Cleanup & Hardening** — Phases 1-6 (shipped 2026-04-02) — [archive](milestones/v0.7.0-ROADMAP.md)
- ✅ **v0.8.0 API & Robustness** — Phases 7-10 (shipped 2026-04-03) — [archive](milestones/v0.8.0-ROADMAP.md)
- ✅ **v0.9.0 Completeness & Hardening** — Phases 11-14 (shipped 2026-04-03) — [archive](milestones/v0.9.0-ROADMAP.md)
- ✅ **v0.10.0 Security & Architecture** — Phases 15-22 (shipped 2026-04-06) — [archive](milestones/v0.10.0-ROADMAP.md)
- 🔄 **v0.11.0 Cross-Backend Hardening** — Phases 23-26 (in progress)

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

### v0.11.0 Cross-Backend Hardening (Phases 23-27)

- [x] **Phase 23: Encryption Schema & Storage** — Add encryption columns to SQLite/PG, migrations, fix put_entry/get_entry (completed)
- [x] **Phase 24: Cross-Backend Test Parity** — Parametrize encryption tests across all 3 backends (completed)
- [x] **Phase 25: Inline Blob Encryption** — Encrypt direct inline path, decrypt on read, key rotation for inline entries
 (completed 2026-04-06)
- [x] **Phase 26: Integration & Hardening** — Config validation, migration testing on existing databases
 (completed 2026-04-07)
- [x] **Phase 27: Retroactive Verification & Doc Cleanup** — Create missing VERIFICATION.md/SUMMARY.md for Phases 23-25, update REQUIREMENTS.md traceability, fix tech debt (completed 2026-04-07)

## Phase Details

### Phase 23: Encryption Schema & Storage
**Goal**: Encrypted blobs stored via SQLite or PostgreSQL backends can be read back without data loss
**Depends on**: Nothing (first phase of milestone)
**Requirements**: ENC-01, ENC-02
**Success Criteria** (what must be TRUE):
  1. `encryption_algorithm` and `encryption_iv` columns exist in SQLite and PostgreSQL schemas after migration
  2. `put_entry()` preserves encryption metadata fields when writing to SQLite and PostgreSQL backends
  3. `get_entry()` returns encryption metadata fields from SQLite and PostgreSQL backends
  4. Encrypted blob roundtrip (put→get) returns original data with all 3 metadata backends
**Plans**: 2 plans
Plans:
- [x] 23-01-PLAN.md — SQLite encryption schema, field handling, and tests
- [x] 23-02-PLAN.md — PostgreSQL encryption schema, field handling, and tests

### Phase 24: Cross-Backend Test Parity
**Goal**: Encryption test coverage is equal across all metadata backends, not just JSON
**Depends on**: Phase 23
**Requirements**: ENC-03
**Success Criteria** (what must be TRUE):
  1. Every existing encryption test runs against JSON, SQLite, and PostgreSQL backends
  2. No encryption test is hardcoded to a single backend
  3. PostgreSQL encryption tests are grouped with `@pytest.mark.xdist_group("docker")`
**Plans**: 1 plan
Plans:
- [x] 24-01-PLAN.md — Parametrize 12 encryption integration tests + fix outdated comment

### Phase 25: Inline Blob Encryption
**Goal**: Inline blobs (stored directly in metadata) are encrypted at rest, just like file-backed blobs
**Depends on**: Phase 23
**Requirements**: INLINE-01, INLINE-02, INLINE-03
**Success Criteria** (what must be TRUE):
  1. `_try_direct_inline()` encrypts data before storing in metadata when encryption is enabled
  2. `_read_inline_blob()` decrypts ciphertext before passing to handler when entry has encryption metadata
  3. `rotate_key()` decrypts inline entries with old key and re-encrypts with new key
  4. Inline encrypted roundtrip produces correct data for all supported types
**Plans**: 2 plans
Plans:
- [x] 25-01-PLAN.md — Encrypt inline write path + decrypt inline read path
- [x] 25-02-PLAN.md — Key rotation inline branch + inline encryption tests

### Phase 26: Integration & Hardening
**Goal**: Known-bad configuration combinations fail loudly at init, and schema migrations work on real databases
**Depends on**: Phase 23, Phase 25
**Requirements**: HARD-01, HARD-02
**Success Criteria** (what must be TRUE):
  1. Initializing UnifiedCache with encryption enabled but no key file raises a clear error at construction time
  2. Known-bad config combos (e.g., encryption + unsigned mode) are rejected with actionable error messages
  3. Schema migration from v3→v4 succeeds on a SQLite database containing existing cached entries
  4. Schema migration preserves all existing entries and their metadata
**Plans**: 2 plans

Plans:
- [x] 26-01-PLAN.md — Config validation for known-bad combinations
- [x] 26-02-PLAN.md — Real-data v3→v4 migration tests (SQLite + PostgreSQL)

### Phase 27: Retroactive Verification & Doc Cleanup
**Goal**: All v0.11.0 phases have formal verification artifacts and REQUIREMENTS.md traceability is accurate
**Depends on**: Phase 23, Phase 24, Phase 25, Phase 26
**Requirements**: ENC-01, ENC-02, ENC-03, INLINE-01, INLINE-02, INLINE-03, HARD-01, HARD-02
**Gap Closure**: Closes all gaps from v0.11.0 audit
**Success Criteria** (what must be TRUE):
  1. Phase 23 has VERIFICATION.md and SUMMARY.md files with evidence of requirement satisfaction
  2. Phase 24 has VERIFICATION.md with evidence of requirement satisfaction
  3. Phase 25 has VERIFICATION.md with evidence of requirement satisfaction
  4. All 8 REQUIREMENTS.md checkboxes are checked and traceability table shows Complete
  5. SUMMARY frontmatters include `requirements_completed` field
  6. Misleading comment in `_inline_blob_mixin.py` L130 is fixed
**Plans**: 3 plans

Plans:
- [x] 27-01-PLAN.md — Phase 23 retroactive verification & summaries (ENC-01, ENC-02)
- [x] 27-02-PLAN.md — Phases 24 & 25 retroactive verification (ENC-03, INLINE-01/02/03)
- [x] 27-03-PLAN.md — Comment fix, Phase 26 SUMMARYs, REQUIREMENTS.md finalization (HARD-01/02)

## Progress

| Phase | Milestone | Status | Completed |
|-------|-----------|--------|-----------|
| 1-6 | v0.7.0 | Complete | 2026-04-02 |
| 7-10 | v0.8.0 | Complete | 2026-04-03 |
| 11-14 | v0.9.0 | Complete | 2026-04-03 |
| 15-22 | v0.10.0 | Complete | 2026-04-06 |
| 23-27 | v0.11.0 | In progress | — |
