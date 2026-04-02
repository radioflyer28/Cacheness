# Phase 8: Crash-Safe Write Intent Logging - Context

**Gathered:** 2026-04-02
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure phase — discuss skipped)

<domain>
## Phase Boundary

Prevent orphaned blobs from accumulating silently after crashes. A "pending" marker (intent journal entry) is written before blob write begins and removed after metadata commit succeeds. On cache init (when cleanup_on_init=True), stale intent entries older than a configurable threshold are detected and their orphaned blobs deleted.

</domain>

<decisions>
## Implementation Decisions

### Agent's Discretion
All implementation choices are at the agent's discretion — pure infrastructure phase. Use ROADMAP phase goal, success criteria, and codebase conventions to guide decisions.

Key design considerations:
- Intent journal storage mechanism (file-based vs metadata table column vs separate SQLite table)
- Threshold for "stale" intent entries (configurable via config)
- Integration point in core.py put() method (between _write_blob and put_entry)
- Cleanup integration in __init__ alongside existing _cleanup_expired()

</decisions>

<code_context>
## Existing Code Insights

### Write Path (core.py put())
- `_PutCleanup` class (line 71) already tracks blob paths for rollback on exception
- Write flow: `_write_blob()` → `put_entry()` → `cleanup.commit()` — crash between steps 1-2 orphans blob
- `_cleanup_expired()` runs on init when `cleanup_on_init=True` (line 195)

### Storage Layer
- `BlobStore._write_blob()` in `storage/blob_store.py` writes blob, returns `WriteBlobResult`
- Blobs stored via `blob_backend.write_blob_from_path()` — supports local filesystem + S3
- `_PutCleanup.rollback()` handles in-process failures but NOT crash recovery

### Config
- `cleanup_on_init: bool = True` in `CacheStorageConfig` (config.py line 41)
- `cache_dir` determines where blob files live
- RLock now in place (Phase 7) — safe for cleanup-on-init operations

</code_context>

<specifics>
## Specific Ideas

No specific requirements — infrastructure phase. Refer to ROADMAP phase description and success criteria.

</specifics>

<deferred>
## Deferred Ideas

None — infrastructure phase.

</deferred>
