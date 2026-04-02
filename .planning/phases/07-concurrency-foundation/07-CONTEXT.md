# Phase 7: Concurrency Foundation - Context

**Gathered:** 2026-04-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix the SQLite metadata backend's non-reentrant lock (threading.Lock → threading.RLock) and audit all lock acquisition points for consistency. This prevents deadlocks from composed operations and brings SqliteBackend in line with JsonBackend's existing RLock pattern.

</domain>

<decisions>
## Implementation Decisions

### Lock Type Fix Scope
- **D-01:** Swap `threading.Lock()` → `threading.RLock()` in `SqliteBackend.__init__()` (line 236 of `src/cacheness/metadata/sqlite_backend.py`)
- **D-02:** Audit ALL SqliteBackend methods for lock consistency — `iter_entry_summaries()` currently acquires no lock while every other public method does. Fix unprotected methods to acquire the lock.
- **D-03:** This is a "consistent" scope — NOT a thorough lock-sharing review. The `blob_store._lock = self._lock` sharing pattern (core.py line 406) is out of scope for this phase.

### Agent's Discretion
- Whether to add the lock to `iter_entry_summaries()` as `with self._lock:` wrapping the entire generator, or per-yield — agent decides based on performance and correctness trade-offs.
- Re-entrancy stress test design: a new test exercising nested backend calls from a single thread is required by the success criteria. Agent decides scope and approach.
- Documentation: minimal code-level comments explaining the RLock choice are sufficient. No separate docs file needed for this phase.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Concurrency Architecture
- `src/cacheness/metadata/sqlite_backend.py` — SqliteBackend with `threading.Lock()` at line 236; all lock acquisition points
- `src/cacheness/metadata/json_backend.py` — JsonBackend with `threading.RLock()` at line 32; reference pattern for consistency
- `src/cacheness/core.py` — UnifiedCache with `threading.RLock()` at line 173; lock sharing at line 406

### Testing
- `tests/test_thread_safety.py` — Existing concurrent put/get tests for both backends
- `.planning/research/PITFALLS.md` — P10 identifies the Lock→RLock deadlock risk

### Research
- `.planning/research/ARCHITECTURE.md` — Integration analysis for concurrency changes
- `.planning/research/FEATURES.md` — Concurrency feature analysis and existing lock usage

</canonical_refs>

<code_context>
## Existing Code Insights

### Lock Usage Summary
- **SqliteBackend:** `threading.Lock()` (non-reentrant) — 10+ methods acquire it via `with self._lock, self.SessionLocal() as session:`
- **JsonBackend:** `threading.RLock()` (reentrant) — all public methods acquire it; `cleanup_by_size()` → `get_stats()` re-entrant pattern works correctly
- **UnifiedCache:** `threading.RLock()` (reentrant) — shared with BlobStore via `self._blob_store._lock = self._lock`

### Unprotected Methods (Audit Targets)
- `SqliteBackend.iter_entry_summaries()` — no lock acquired; yields results from SQLAlchemy session
- Possible others — full audit needed during implementation

### Established Patterns
- `with self._lock, self.SessionLocal() as session:` — combined lock + session pattern used consistently in SqliteBackend
- `with self._lock:` — simpler pattern in JsonBackend and UnifiedCache

### Integration Points
- No changes to UnifiedCache or BlobStore needed — only SqliteBackend internal lock type changes
- Existing thread-safety tests should pass unchanged with the RLock swap

</code_context>

<specifics>
## Specific Ideas

No specific requirements — straightforward Lock→RLock swap with consistency audit.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 07-concurrency-foundation*
*Context gathered: 2026-04-02*
