# Phase 17: Key Rotation API - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-03
**Phase:** 17-key-rotation-api
**Areas discussed:** API surface & scope, Atomicity & failure handling, Old-key entry behavior, BlobStore (storage mode) support

---

## API Surface & Scope

### Where should rotate_key() live?

| Option | Description | Selected |
|--------|-------------|----------|
| UnifiedCache.rotate_key() only | Cache mode gets the API. BlobStore handled separately if needed. | |
| Both UnifiedCache and BlobStore | Both modes get rotate_key(). BlobStore has its own signer and entry iteration. | ✓ |
| Standalone function | rotate_key(cache_dir, ...) that works without instantiating a cache. | |

**User's choice:** Both UnifiedCache and BlobStore (recommended)
**Notes:** Both modes have independent signer wiring and need rotation support.

### How does the user provide the new key?

| Option | Description | Selected |
|--------|-------------|----------|
| new_key_file path | User provides path to a new key file (already generated). | ✓ |
| Auto-generate | rotate_key() generates a new key automatically and replaces the old one. | |
| Both options | new_key_file is optional. If omitted, auto-generate. If provided, use that file. | |

**User's choice:** new_key_file path (recommended)
**Notes:** User asked for recommended defaults across all areas.

### Should rotate_key() also handle v2→v3 migration?

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, combined | rotate_key() re-signs all entries (v1/v2/v3) with the new key as v3. One API. | ✓ |
| Separate methods | rotate_key() for rotation, migrate_signatures() for v2→v3. More explicit. | |
| No v2 migration | Only re-sign with new key, keep whatever version entries currently have. | |

**User's choice:** Yes, combined (recommended)
**Notes:** Phase 16 D-04 deferred this here.

### What should rotate_key() return?

| Option | Description | Selected |
|--------|-------------|----------|
| Structured result object | RotationResult with counts (total, re-signed, failed, skipped). | ✓ |
| Simple count | Return int count of re-signed entries. | |
| Bool only | Return bool (success/failure). | |

**User's choice:** Structured result object (recommended)

---

## Atomicity & Failure Handling

### What if re-signing fails partway through?

| Option | Description | Selected |
|--------|-------------|----------|
| Best-effort with progress report | If entry N fails, log it, skip it, keep going. Report at end. | ✓ |
| Full rollback | Roll back ALL re-signs to old key. Requires saving old signatures. | |
| Skip-and-continue | Same as best-effort but less structured reporting. | |

**User's choice:** Best-effort with progress report (recommended)
**Notes:** User initially chose full rollback, then asked for recommended defaults.

### How should concurrent access be handled?

| Option | Description | Selected |
|--------|-------------|----------|
| Hold lock for entire rotation | Acquire self._lock for the entire rotation. Simple, safe. | ✓ |
| Per-entry locking | Lock per-entry during re-sign. More complex. | |
| No extra locking | Caller responsible for quiescing the cache. | |

**User's choice:** Hold lock for entire rotation (recommended)

### What if the process crashes mid-rotation?

| Option | Description | Selected |
|--------|-------------|----------|
| Forward-compatible (verify both keys) | Signer routes v2→master, v3→derived. Mixed state handled. | ✓ |
| Resumable rotation journal | Write state file, detect incomplete on next init. | |
| No crash recovery | Best effort, user re-runs rotate_key(). | |

**User's choice:** Forward-compatible (recommended)

---

## Old-Key Entry Behavior

**User's choice:** Existing version routing handles mixed state (recommended)
**Notes:** `verify_entry()` already routes v1/v2→master_key, v3→derived_key. No special dual-key logic needed. `rotate_key()` is idempotent.

---

## BlobStore (Storage Mode) Support

**User's choice:** BlobStore gets its own `rotate_key()` with same semantics (recommended)
**Notes:** BlobStore has independent signer wiring via `_init_signer()`.

---

## Agent's Discretion

- `RotationResult` dataclass location
- Crash-interrupted rotation detection flag
- Exact log message wording
- Test file structure

## Deferred Ideas

None — discussion stayed within phase scope.
