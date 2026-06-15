---
gsd_state_version: 1.0
milestone: v0.12.0
milestone_name: Phase Details
status: executing
last_updated: "2026-06-15T18:36:45.015Z"
last_activity: 2026-06-15 -- Phase 32 Plan 02 complete
progress:
  total_phases: 9
  completed_phases: 4
  total_plans: 29
  completed_plans: 23
  percent: 44
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-12)

**Core value:** Improve reliability, security, and maintainability of Cacheness without changing public API semantics
**Current focus:** Phase 32 — Small Fixes & Release Polish

## Current Position

Phase: 32
Plan: 03
Status: Ready to execute next plan
Last activity: 2026-06-15 -- Phase 32 Plan 02 complete

## Accumulated Context

### Pending Todos (6)

1. Fix clear_all() not deleting blob files (storage) — code review R1, 🔴
2. Fix write-intent journal path resolution and safety checks (storage) — code review R2/R17, 🔴
3. JSON backend: propagate save failures, preserve corrupt files (database) — code review R3/R4, 🔴
4. Stabilize cache keys: remove unstable hash()/str() fallbacks (general) — code review U1, 🔴
5. Property-based stress testing for cache key serialization (testing) — pairs with #4
6. Tiered pull-through cache (general) — prerequisites noted from code review

*Note: "Store cacheness version in metadata" and "Encryption at rest" closed 2026-06-12 (shipped in Phase 23; residual gaps tracked in code-review tasks).*

### Code Review (2026-06-12)

Full review in `docs/CODE_REVIEW_FINDINGS.md`; execution specs in `docs/CODE_REVIEW_ACTIONS.md`.

- Wave 1 (silent data loss) → pending todos above
- Waves 2–4 + small fixes → backlog phases 999.1–999.4 in ROADMAP.md
- Decision-gated items → seeds SEED-001…006 in .planning/seeds/

## Performance Metrics

| Phase | Plan | Duration | Notes |
|-------|------|----------|-------|
| Phase 29-ttl-eviction-consistency P01 | 38min | 2 tasks | 12 files |
| Phase 29-ttl-eviction-consistency P02 | 8min | 2 tasks | 2 files |
| Phase 29-ttl-eviction-consistency P03 | 17min | 2 tasks | 9 files |
| Phase 29-ttl-eviction-consistency P04 | 2h 53m | 2 tasks | 2 files |
| Phase 31-security-storage-mode-posture P01 | 12min | 2 tasks | 8 files |
| Phase 31-security-storage-mode-posture P05 | 9min | 2 tasks | 4 files |
| Phase 31-security-storage-mode-posture P02 | 12min | 2 tasks | 5 files |
| Phase 31-security-storage-mode-posture P07 | 7min | 2 tasks | 2 files |
| Phase 31-security-storage-mode-posture P04 | 8min | 2 tasks | 6 files |
| Phase 31-security-storage-mode-posture P03 | 14min | 2 tasks | 8 files |
| Phase 31-security-storage-mode-posture P06 | 11min | 2 tasks | 15 files |
| Phase 31-security-storage-mode-posture P08 | 11min | 2 tasks | 6 files |
| Phase 32 P01 | 9min | 1 tasks | 2 files |
| Phase 32 P02 | 8min | 1 tasks | 2 files |

## Decisions

- [Phase 29-ttl-eviction-consistency]: Stored expires_at is authoritative for cache-mode reads and cleanup; fallback TTL applies only when expires_at is absent. — Completed by Phase 29 Plan 01 / TTL-01.
- [Phase 29-ttl-eviction-consistency]: Read-path expired-entry deletion remained outside Plan 29-02 scope per D-10.
- [Phase 29-ttl-eviction-consistency]: TTL-02 init cleanup delegates to public cleanup_expired so constructor cleanup deletes blobs and invokes on_evict.
- [Phase 29-ttl-eviction-consistency]: TTL-03 metadata-only backend updates preserve created_at, ttl_seconds, and expires_at by default; content updates pass explicit timestamps. — Completed by Phase 29 Plan 03.
- [Phase 29-ttl-eviction-consistency]: SQLite and PostgreSQL same-key overwrites preserve existing access_count. — Completed by Phase 29 Plan 03.
- [Phase 29-ttl-eviction-consistency]: TTL-04 size eviction delegates URI actual_path deletion to self._blob_store.blob_backend.delete_blob(actual_path), and backend delete exceptions are warning-only. — Completed by Phase 29 Plan 04.
- [Phase 31-security-storage-mode-posture]: SEC-01 keeps minimum_signature_version defaulting to 1 for old-cache compatibility. — Completed by Phase 31 Plan 01.
- [Phase 31-security-storage-mode-posture]: New deployments should use minimum_signature_version=3 with allow_unsigned_entries=False when metadata may be attacker-writable. — Completed by Phase 31 Plan 01.
- [Phase 31-security-storage-mode-posture]: Minimum-version rejection is enforced in CacheEntrySigner.verify_entry so UnifiedCache and BlobStore share the policy. — Completed by Phase 31 Plan 01.
- [Phase 31-security-storage-mode-posture]: STRG-01 keeps storage-mode destructive APIs warning-first by default rather than hard-refusal.
- [Phase 31-security-storage-mode-posture]: Implicit storage-mode TTL, eviction, and invalid-entry deletion behavior remains disabled.
- [Phase 31-security-storage-mode-posture]: Storage-mode destructive warnings use both cacheness.core logger.warning and RuntimeWarning.
- [Phase 31-security-storage-mode-posture]: Encrypted BlobStore reads preserve backend abstraction by using blob_backend.read_blob for ciphertext.
- [Phase 31-security-storage-mode-posture]: Encrypted reads try handler.get_bytes on decrypted plaintext before any temp-file fallback.
- [Phase 31-security-storage-mode-posture]: Temp fallback is limited to handlers that raise NotImplementedError and uses mkstemp under cache_dir with best-effort POSIX 0600 permissions.
- [Phase 31-security-storage-mode-posture]: STRG-03 production code remained unchanged because regressions passed against the existing pre-blob write-intent implementation.
- [Phase 31-security-storage-mode-posture]: Task 2 was recorded with an empty verification commit to preserve the plan's per-task commit trail without source churn.
- [Phase 31-security-storage-mode-posture]: Minimum signature-version policy remains signer-level, so legacy compatibility cannot bypass a stricter configured minimum.
- [Phase 31-security-storage-mode-posture]: BlobStore new writes use canonical fields; old flattened BlobStore signatures are accepted only through an explicit legacy verifier.
- [Phase 31-security-storage-mode-posture]: SEC-04 canonical signing fields live in src/cacheness/signing_fields.py and are shared by UnifiedCache and BlobStore.
- [Phase 31-security-storage-mode-posture]: SEC-03 writes new key bytes to <keyfile>.new and replaces the active key only after rotation succeeds.
- [Phase 31-security-storage-mode-posture]: UnifiedCache and BlobStore verify existing signed entries with the old signer before re-signing with the staged signer.
- [Phase 31-security-storage-mode-posture]: Local encrypted blob rotation writes ciphertext to <blob>.rotating and publishes with os.replace.
- [Phase 31-security-storage-mode-posture]: Leftover <keyfile>.new files are logged as interrupted rotations on startup; full resume remains out of scope.
- [Phase 31-security-storage-mode-posture]: STRG-02 keeps fsync_on_write defaulting to False so cache-mode and storage-mode write performance is unchanged unless users opt in.
- [Phase 31-security-storage-mode-posture]: Local file fsync failures propagate when fsync_on_write is enabled; parent directory fsync is best-effort for Windows and filesystems that do not support it.
- [Phase 31-security-storage-mode-posture]: The fsync policy is local-only; remote blob stores and database backends rely on their own durability contracts.
- [Phase 31-security-storage-mode-posture]: Interrupted rotation fallback accepts staged signatures only while the protected sibling <keyfile>.new still exists. — Completed by Phase 31 Plan 08 / SEC-03 gap closure.
- [Phase 31-security-storage-mode-posture]: Active signer and active encryption key remain the normal read path; staged signer/decryption are fallback-only. — Completed by Phase 31 Plan 08 / SEC-03 gap closure.
- [Phase 31-security-storage-mode-posture]: Startup does not replace keys, delete <keyfile>.new, rewrite metadata, or complete rotation implicitly. — Completed by Phase 31 Plan 08 / SEC-03 gap closure.
- [Phase 32]: BlobStore.put preserves the existing nested metadata contract by copying metadata with dict(metadata or {}) before adding internal fields. — POL-01 requires caller-owned metadata dictionaries to remain unchanged while stored metadata still carries Cacheness internal fields and signatures.
- [Phase 32]: BlobStore._sanitize_key remains the single key-normalization point; safe keys stay unchanged and transformed keys receive a stable original-key xxh3_64 suffix. — Completed by Phase 32 Plan 02 / POL-02.
