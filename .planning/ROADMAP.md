# Roadmap: Cacheness

## Milestones

- ✅ **v0.7.0 Cleanup & Hardening** — Phases 1-6 (shipped 2026-04-02) — [archive](milestones/v0.7.0-ROADMAP.md)
- ✅ **v0.8.0 API & Robustness** — Phases 7-10 (shipped 2026-04-03) — [archive](milestones/v0.8.0-ROADMAP.md)
- ✅ **v0.9.0 Completeness & Hardening** — Phases 11-14 (shipped 2026-04-03) — [archive](milestones/v0.9.0-ROADMAP.md)
- 🔄 **v0.10.0 Security & Architecture** — Phases 15-20

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

### v0.10.0 Security & Architecture (Phases 15-20)

- [x] **Phase 15: Configurable Key Fallback** — User-controlled key fallback policy (raise/warn/fallback)
- [x] **Phase 16: HKDF Key Derivation** — Per-namespace key derivation from master key (completed 2026-04-03)
- [ ] **Phase 17: Key Rotation API** — Manual rotate_key() with re-sign and old-key handling
- [ ] **Phase 18: Encryption at Rest** — AES-GCM blob encryption, disabled by default
- [ ] **Phase 19: Core Decomposition II** — Extract eviction, namespace, and init logic into mixins
- [ ] **Phase 20: Concurrency & Atomics Verification** — Thread safety and atomic write correctness

## Phase Details

### Phase 15: Configurable Key Fallback
**Goal**: Users control what happens when signing key is unavailable — no silent behavior
**Depends on**: Nothing (foundational)
**Requirements**: SEC-02
**Success Criteria** (what must be TRUE):
  1. User can set key fallback policy to `raise`, `warn`, or `fallback` via config
  2. `raise` policy causes `CacheSecurityError` when key file is missing
  3. `warn` policy logs a warning and falls back to in-memory key (current default behavior)
  4. `fallback` policy silently uses in-memory key without warning
**Plans:** 1 plan
Plans:
- [x] 15-01-PLAN.md — Config, security, core, blob_store changes + tests + docs (3 tasks, wave 1)

### Phase 16: HKDF Key Derivation
**Goal**: Each namespace derives its own signing key from the master key — cryptographic isolation
**Depends on**: Phase 15 (key fallback must be configurable before adding derivation complexity)
**Requirements**: SEC-01
**Success Criteria** (what must be TRUE):
  1. Namespace-specific signing keys are derived via HKDF-SHA256 from master key + namespace ID
  2. Entries signed with the shared master key (pre-HKDF) still verify successfully (migration path)
  3. New entries in different namespaces produce different signatures for identical data
  4. Disabling HKDF reverts to shared-key behavior (opt-out path)
**Plans:** 1/1 plans complete
Plans:
- [x] 16-01-PLAN.md — HKDF derivation, config, wiring, tests, docs (3 tasks, wave 1)

### Phase 17: Key Rotation API
**Goal**: Users can rotate signing keys and re-sign existing entries without data loss
**Depends on**: Phase 16 (rotation needs HKDF to re-derive namespace keys)
**Requirements**: SEC-04
**Success Criteria** (what must be TRUE):
  1. `rotate_key(new_key_file)` re-derives namespace keys and re-signs all entries
  2. After rotation, old entries verify successfully with new key
  3. Deleting old key file and restarting cache handles old-key entries gracefully (no crash)
  4. Rotation is atomic — partial failure leaves entries in a consistent state
**Plans**: TBD

### Phase 18: Encryption at Rest
**Goal**: Cached blob data can be encrypted on disk — disabled by default, signing unaffected
**Depends on**: Nothing (independent, but benefits from key infrastructure in Phases 15-17)
**Requirements**: SEC-03
**Success Criteria** (what must be TRUE):
  1. `encryption_enabled=True` encrypts blob content with AES-GCM before writing to disk
  2. Encrypted entries decrypt transparently on `get()` — no API change for callers
  3. Encryption is disabled by default — existing users see zero behavior change
  4. Configuration exposed via `CacheMetadataConfig` (`encryption_enabled`, `encryption_key_file`)
  5. Unencrypted entries remain readable when encryption is later enabled (migration path)
**Plans**: TBD

### Phase 19: Core Decomposition II
**Goal**: core.py reduced from ~2500 to ~1500 lines via mixin/delegate extraction
**Depends on**: Nothing (independent architectural work)
**Requirements**: ARCH-01
**Success Criteria** (what must be TRUE):
  1. core.py is ≤1500 lines after extraction
  2. Extracted concerns live in focused mixin or delegate files (following v0.7.0 pattern)
  3. All existing tests pass unchanged — no public API changes
  4. `from cacheness.core import UnifiedCache` continues to work
**Plans**: TBD

### Phase 20: Concurrency & Atomics Verification
**Goal**: Verify thread safety and atomic write correctness through testing — fix bugs if found
**Depends on**: Nothing (independent test-and-fix work)
**Requirements**: TEST-01, TEST-02
**Success Criteria** (what must be TRUE):
  1. Concurrent `put()`/`get()` from 8+ threads produces no data races or corruption (JSON and SQLite)
  2. No deadlocks observed under sustained concurrent access (60s stress test)
  3. `shutil.move()` atomic writes verified correct on Windows same-volume scenarios
  4. Concurrent writes to the same key produce no file corruption
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
| 7. Concurrency Foundation | v0.8.0 | Complete | 2026-04-02 |
| 8. Crash-Safe Write Intent Logging | v0.8.0 | Complete | 2026-04-02 |
| 9. Blob Integrity Validation | v0.8.0 | Complete | 2026-04-02 |
| 10. Prefix Deletion API | v0.8.0 | Complete | 2026-04-02 |
| 11. put_batch() API | v0.9.0 | Complete | 2026-04-03 |
| 12. Threading Model Documentation | v0.9.0 | Complete | 2026-04-03 |
| 13. Deserialization Security | v0.9.0 | Complete | 2026-04-03 |
| 14. Windows Key File Permissions | v0.9.0 | Complete | 2026-04-03 |
| 15. Configurable Key Fallback | v0.10.0 | Not started | - |
| 16. HKDF Key Derivation | 1/1 | Complete   | 2026-04-03 |
| 17. Key Rotation API | v0.10.0 | Not started | - |
| 18. Encryption at Rest | v0.10.0 | Not started | - |
| 19. Core Decomposition II | v0.10.0 | Not started | - |
| 20. Concurrency & Atomics Verification | v0.10.0 | Not started | - |
