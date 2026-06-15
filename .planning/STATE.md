---
gsd_state_version: 1.0
milestone: v0.12.0
milestone_name: Reliability Remediation
status: executing
last_updated: "2026-06-15T02:50:27.330Z"
last_activity: 2026-06-15 -- Phase 31 Plan 07 complete
progress:
  total_phases: 9
  completed_phases: 3
  total_plans: 20
  completed_plans: 17
  percent: 85
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-12)

**Core value:** Improve reliability, security, and maintainability of Cacheness without changing public API semantics
**Current focus:** v0.12.0 Reliability Remediation

## Current Position

Phase: 31 — Security & Storage-Mode Posture
Plan: 31-01, 31-02, 31-05, and 31-07 complete; remaining Phase 31 plans pending
Status: Executing
Last activity: 2026-06-15 -- Phase 31 Plan 07 complete

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
