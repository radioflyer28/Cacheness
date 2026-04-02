# Project Research Summary

**Project:** Cacheness v0.8.0 "API & Robustness"
**Domain:** Python disk caching library — management APIs, concurrency, signing, crash safety
**Researched:** 2026-04-02
**Confidence:** HIGH

## Executive Summary

Cacheness v0.8.0 is a **hardening milestone**: no new dependencies, no architectural refactors, no public API revolutions. All four features — management APIs, concurrency safety, HMAC blob signing, and orphaned blob prevention — are behavioral additions to a well-decomposed codebase that emerged from v0.7.0's structural cleanup. The entire milestone runs on Python stdlib (`hmac`, `hashlib`, `threading`, `pathlib`) plus already-vendored dependencies (`xxhash`, `SQLAlchemy`). This is the ideal outcome for a robustness-focused release.

The most important finding from cross-referencing all four research files is that **the existing codebase is closer to v0.8.0's goals than the original requirements assumed.** Architecture research revealed that `put()`/`get()` already acquire the RLock, `update_data()` and `touch()` already exist, `get_metadata()` is implemented, and `delete_where`/`delete_batch`/`touch_batch` are already in core.py. The true gaps are narrower: `delete_by_prefix()`, `put_batch()`/`get_batch()`, blob HMAC signing, and crash-safe orphan prevention. Concurrency work is primarily documentation and testing, not new locking infrastructure.

The key risk is **scope creep** — the temptation to over-engineer each feature (ReadWriteLock, file-based WAL, full blob HMAC on every read) when simpler solutions suffice. The pitfalls research identified 14 specific traps, with 4 rated critical. The build order is clear: fix the SQLite Lock→RLock mismatch first (P10 — potential deadlock in batch ops), then orphan prevention and concurrency docs in parallel, then HMAC signing, then management APIs last since they depend on all three.

## Key Findings

### Recommended Stack

No new dependencies. All four features use Python stdlib or existing project dependencies. (See [STACK.md](STACK.md))

**Core technologies (all existing):**
- `threading.RLock` — already in core.py; extend usage documentation, not mechanism
- `hmac` + `hashlib` (stdlib) — extend existing `CacheEntrySigner` for blob content HMAC
- `xxhash` (existing dep) — fast non-crypto blob integrity; HMAC adds crypto authentication on top
- `SQLAlchemy` (existing dep) — SQL `LIKE`/`IN` for backend-level bulk operations
- `pathlib` + `json` (stdlib) — write-ahead intent journal for orphan prevention

### Expected Features

(See [FEATURES.md](FEATURES.md))

**Already implemented (narrowing scope):**
- `update_data()` — exists at core.py:2219
- `touch()` — exists at core.py:2435
- `get_metadata()` — exists at core.py:1545
- `delete_where()` / `delete_matching()` / `delete_batch()` / `touch_batch()` — all exist
- Thread locking on `put()`/`get()` — already acquired via `self._lock`

**Must have (truly missing — table stakes):**
- `delete_by_prefix()` — bulk cleanup by key prefix, needs backend-level SQL fast path
- `get_batch()` / `put_batch()` — batch CRUD for reduced metadata round-trips
- Concurrency documentation — clear thread-safe vs process-safe boundary definitions
- Orphan prevention — crash recovery beyond `verify_integrity(repair=True)`

**Should have (differentiators):**
- HMAC-SHA256 blob signing — cryptographic tamper detection for stored blobs
- `verify_integrity()` grace period (`min_age_seconds`) to avoid deleting in-progress blobs
- `get_metadata_batch()` / `touch_batch()` refinements

**Defer (v2+):**
- Async `put()`/`get()` — separate milestone
- ReadWriteLock — premature optimization; profile first
- File-level cross-process locking — SQLite WAL already handles this
- Encryption at rest — separate feature, not hardening
- `copy()`/`move()` entry operations — compose from existing CRUD

### Architecture Approach

(See [ARCHITECTURE.md](ARCHITECTURE.md))

These are **behavioral additions, not structural changes.** No new classes beyond `WriteJournal`. All management APIs go directly on `UnifiedCache` (not a new mixin — too coupled to put/get internals). Blob HMAC extends `CacheEntrySigner` with ~30 lines. Orphan prevention adds a single new file (`write_journal.py`).

**Modified components (10 files, ~200 lines total):**
1. `core.py` — management API methods, journal integration in `_PutCleanup`
2. `security.py` — `compute_blob_hmac()`, `verify_blob_hmac()`, signature v3
3. `_verification_mixin.py` — blob HMAC verification step
4. `storage/blob_store.py` — HMAC computation at write time
5. `config.py` — `verify_blob_hmac`, `enable_write_journal` flags
6. `interfaces.py` — `blob_hmac` in `SignableFields`
7. `metadata/base.py` — `delete_by_prefix()` in ABC
8. `metadata/sqlite_backend.py` — SQL fast path for prefix delete
9. `metadata/json_backend.py` — Python fallback for prefix delete

**New component (1 file):**
- `write_journal.py` — append-only intent journal for crash recovery

### Critical Pitfalls

(See [PITFALLS.md](PITFALLS.md))

1. **P1 (CRITICAL): HMAC TOCTOU gap** — Compute HMAC from persisted blob bytes, not in-memory serialization output. A truncated write would pass HMAC if signed from memory. Sign *after* `_write_blob()` succeeds.
2. **P2 (CRITICAL): JSON backend globally unsafe under concurrent writers** — Do NOT fix it. Document as single-process only. Adding file locks creates cross-platform pain for a backend already documented as "NOT safe for concurrency."
3. **P10 (HIGH): SQLite Lock vs RLock mismatch** — `SqliteBackend` uses non-reentrant `threading.Lock()` while `UnifiedCache` uses `RLock`. Batch management APIs that compose multiple backend calls will deadlock. Fix: change to `RLock` before implementing batch ops.
4. **P8 (HIGH): `verify_integrity(repair=True)` races with active writes** — Add `min_age_seconds=60` parameter. Only consider blobs orphaned if older than the threshold.
5. **P9 (HIGH): Signature version migration trap** — Keep `blob_hmac` as a separate field, not part of signature versioning. Avoids O(n × blob_size) migration for existing caches.

## Cross-Researcher Consensus

All four research files agree on these points:

| Topic | Consensus |
|-------|-----------|
| New dependencies | Zero — stdlib + existing deps are sufficient |
| JSON concurrency | Document as unsupported, don't fix |
| Lock type | Keep RLock for v0.8.0, defer ReadWriteLock |
| Blob HMAC | Use `hmac` + `hashlib` stdlib, not `cryptography` package |
| Management APIs | Add to `core.py` directly, not a new mixin |
| Backend fast paths | SQL `LIKE`/`IN` for SQLite/PG, Python iteration for JSON |
| Build order | Foundational safety first, then signing, then APIs last |

## Cross-Researcher Conflicts

| Topic | Conflict | Resolution |
|-------|----------|------------|
| **Orphan prevention mechanism** | STACK/ARCHITECTURE recommend file-based intent journal. PITFALLS (P3) warns about Windows file locking and startup latency, suggests metadata "pending" flag instead. | **Use intent journal.** The "pending" flag reverses write order (metadata-first) which creates dangling metadata — arguably worse than orphaned blobs. The journal is append-only with no fsync requirement (best-effort), avoiding the Windows complexity PITFALLS warns about. |
| **Blob HMAC in signature versioning** | ARCHITECTURE proposes adding `blob_hmac` to signature v3 field list. PITFALLS (P9) explicitly warns this forces O(n × blob_size) migration and recommends keeping it separate. | **Keep `blob_hmac` separate from `entry_signature`.** Store it alongside the signature, verify independently. New entries get it automatically; old entries treat `None` as "skip verification." No migration needed. |
| **Is blob HMAC even necessary?** | PITFALLS (P6) raises that if metadata HMAC already covers `file_hash`, tampering with the blob changes the hash which invalidates the metadata signature. STACK/ARCHITECTURE assume it's needed. | **Decision needed.** If `file_hash` (xxhash) is in the signed metadata fields AND `verify_hashes=True` compares the actual blob hash to stored `file_hash`, then blob HMAC is redundant for integrity. HMAC adds *authentication* (proves who wrote it) but the signing key is on the same filesystem. Recommend: validate existing coverage first, then decide if the additional HMAC layer is worth the performance cost. |
| **Concurrency safety scope** | STACK proposes a `thread_safe=True` config flag. ARCHITECTURE found `put()`/`get()` already hold the lock unconditionally (no config needed). | **No new config flag.** The lock is already always-on. The v0.8.0 scope is documentation + testing, not new locking infrastructure. |

## Decisions Needed Before Implementation

1. **Blob HMAC: build or skip?** Validate whether the existing `file_hash` (in signed metadata v2) + `verify_hashes=True` already provides sufficient integrity. If yes, HMAC is redundant and the phase can focus on ensuring `file_hash` is *always* populated and verified. If no (because xxhash is non-cryptographic), add `blob_hmac` as a separate field.

2. **Intent journal: when to recover?** Options: (a) always on `__init__`, (b) only when `cleanup_on_init=True`, (c) separate explicit method. Recommendation: (b) — piggyback on existing cleanup behavior.

3. **`delete_by_prefix()`: return type?** Return count of deleted entries (int) or list of deleted keys (List[str])? Count is cheaper; key list is more debuggable. Recommendation: return count for consistency with `delete_batch()`.

## Implications for Roadmap

Based on combined research, suggested phase structure:

### Phase 1: Concurrency Foundation
**Rationale:** Fix the SQLite Lock→RLock mismatch (P10) before any batch operations. Document threading model. This is a prerequisite for safe batch management APIs.
**Delivers:** RLock fix in SQLite backend, concurrency documentation, thread-safety stress tests.
**Addresses:** Concurrency safety (table stakes), P10 deadlock prevention, P2/P5 documentation.
**Avoids:** P10 deadlock in batch operations, P5 thread-vs-process confusion.
**Estimated scope:** ~50 lines code + docs + tests.

### Phase 2: Orphaned Blob Prevention
**Rationale:** Independent of other features. Adds crash safety to `put()`. The write journal is a new file with clear boundaries — low integration risk.
**Delivers:** `WriteJournal` class, `_PutCleanup` journal awareness, startup recovery, `min_age_seconds` in `verify_integrity()`.
**Addresses:** Orphan prevention (table stakes), P8 grace period for active writes.
**Avoids:** P3 Windows complexity (append-only, no fsync), P8 in-progress blob deletion.
**Estimated scope:** ~150 lines new code + tests.

### Phase 3: HMAC Blob Signing
**Rationale:** Depends on decision from "Decisions Needed" #1. If blob HMAC is needed, it must be in place before `update_blob_data()` so updates produce signed blobs. If file_hash coverage is sufficient, this phase shrinks to "ensure file_hash is always populated + add verification tests."
**Delivers:** `blob_hmac` field (if needed), streaming HMAC computation, backward-compatible verification.
**Addresses:** HMAC blob signing (differentiator), P1 TOCTOU prevention, P9 migration safety.
**Avoids:** P1 signing in-memory bytes, P6 performance regression (streaming HMAC), P9 forced migration.
**Estimated scope:** ~80 lines if full HMAC, ~30 lines if file_hash validation only.

### Phase 4: Management APIs
**Rationale:** Last because it depends on all three preceding phases — needs safe concurrency (Phase 1), crash-safe writes (Phase 2), and blob signing aware updates (Phase 3).
**Delivers:** `delete_by_prefix()`, `put_batch()`, `get_batch()`, `get_metadata_batch()`, possibly `update_blob_data()` alias.
**Addresses:** Management APIs (table stakes + differentiators), P4 O(n) re-signing prevention, P7 backend bulk operations.
**Avoids:** P4 re-sign trap (backend-level ops), P7 O(n²) JSON penalty (batch `_save_to_disk`), P12 `get_metadata` blob_data leak.
**Estimated scope:** ~150 lines code + backend extensions + tests.

### Phase Ordering Rationale

- **Phase 1 first** because P10 (SQLite deadlock) will bite immediately when batch operations are tested. It's also the smallest phase (~50 lines).
- **Phases 1-2 could run in parallel** since orphan prevention doesn't touch locking code.
- **Phase 3 before Phase 4** because `update_blob_data()` and `put_batch()` should produce signed blobs from day one. Bolting on signing after management APIs are shipped means re-testing everything.
- **Phase 4 last** because it composes all preceding guarantees (lock safety, crash recovery, signing) into user-facing convenience methods.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2 (Orphan Prevention):** Intent journal design needs careful thought about S3 backend interaction and recovery semantics. The journal format, staleness threshold, and cleanup strategy affect all blob backends.
- **Phase 3 (HMAC Signing):** Requires a concrete decision on whether blob HMAC is needed given existing file_hash coverage. Run a threat model analysis before planning.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Concurrency Foundation):** Lock→RLock swap is mechanical. Documentation follows existing patterns in `docs/SECURITY.md`.
- **Phase 4 (Management APIs):** Follows established backend-fast-path + generic-fallback pattern already used by `query_meta()`.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | **HIGH** | Zero new deps — all stdlib or existing. Verified against codebase imports. |
| Features | **HIGH** | Feature research identified many APIs already exist, narrowing true scope. Cross-referenced with actual core.py line numbers. |
| Architecture | **HIGH** | All analysis from direct codebase inspection post-v0.7.0. Integration points are specific and verified. |
| Pitfalls | **HIGH** | 14 pitfalls derived from actual code paths. Critical pitfalls (P1, P2, P10) confirmed via source inspection. |

**Overall confidence:** HIGH

### Gaps to Address

- **Blob HMAC necessity:** Needs threat model validation. If file_hash in signed metadata already prevents blob swaps, HMAC is redundant overhead. Validate during Phase 3 planning.
- **Intent journal + S3 interaction:** The journal records local intent but S3 uploads are eventually consistent (for compatible stores). Recovery logic needs S3-specific handling (HEAD request to verify existence).
- **PostgreSQL `delete_by_prefix()`:** Not analyzed in detail — assumed to follow SQLite's SQL `LIKE` pattern, but PG-specific optimizations (e.g., `text_pattern_ops` index) may be needed for large-scale use.
- **Batch operation error semantics:** Not decided — all-or-nothing (transactional) vs best-effort (partial success). Recommendation: best-effort with error list return, matching `delete_batch()` existing behavior.

## Sources

### Primary (HIGH confidence)
- Codebase inspection: `core.py`, `security.py`, `blob_store.py`, `config.py`, `metadata/*.py` — verified line numbers and existing implementations
- `docs/MISSING_MANAGEMENT_API.md` — existing API design proposals
- `docs/TRANSACTION_GUARANTEES.md` — crash recovery and concurrency model
- `docs/SECURITY.md` — HMAC signing architecture
- `.planning/codebase/CONCERNS.md` — v0.7.0 security and feature gap audit

### Secondary (MEDIUM confidence)
- diskcache, joblib.Memory, shelve, Redis — competitive analysis for feature expectations
- LevelDB/RocksDB — WAL and write-ahead log patterns for crash recovery

### Tertiary (LOW confidence)
- S3-compatible store consistency guarantees — varies by implementation (MinIO vs Ceph vs Garage)
- ReadWriteLock performance impact — theoretical; needs profiling under real workloads

---
*Research completed: 2026-04-02*
*Ready for roadmap: yes*
