# Phase 29: TTL & Eviction Consistency - Research

**Researched:** 2026-06-13
**Domain:** Python cache metadata TTL, cleanup, signing, and blob eviction consistency
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

## Implementation Decisions

### TTL Semantics

- **D-01:** Implement the approved TASK-5 direction: stored `expires_at` wins when present; caller/config TTL applies only when an entry has no stored expiry.
- **D-02:** `_is_expired()` must parse `entry["expires_at"]` when set. String values use `datetime.fromisoformat`; naive datetimes are treated as UTC.
- **D-03:** When `expires_at` is present, `_is_expired()` ignores the `ttl_seconds` argument for that entry.
- **D-04:** Backend `cleanup_expired(ttl_seconds)` implementations for JSON, SQLite, and PostgreSQL must delete entries where `expires_at < now` when `expires_at` is set, or where `expires_at` is null and `created_at < cutoff`.
- **D-05:** Do not remove the v3 TTL columns. Phase 29 chooses to honor them, not to drop them.
- **D-06:** Storage mode must remain protected from cache expiry behavior. `_storage_mode_get` must not call `_is_expired()`, and a storage-mode read of an entry with past `expires_at` must still return the data.

### Expired Cleanup Path

- **D-07:** Implement TASK-6 by changing `UnifiedCache._cleanup_expired()` to call the public `self.cleanup_expired(ttl_seconds)` path instead of `self.metadata_backend.cleanup_expired(ttl_seconds)`.
- **D-08:** Init-time expired cleanup must remove both metadata and blob files and must preserve the public cleanup path's hook behavior, including `on_evict`.
- **D-09:** Confirm `_cleanup_expired()` is called after the instance lock exists and while that lock is not already held. Existing RLock behavior may be relied on only after reading current `UnifiedCache.__init__`.
- **D-10:** The read-path expired-entry issue from R6 is not mandatory Phase 29 scope. It may be added only if it is a small, well-tested extension that does not blur TASK-6's init-cleanup acceptance.

### Metadata Counters, Timestamps, TTL, and Signatures

- **D-11:** Implement TASK-7 R9 for SQLite and PostgreSQL: same-key overwrites must not reset existing `access_count`.
- **D-12:** SQLite `put_entry` should move away from `INSERT OR REPLACE` for overwrites and use conflict-update semantics that omit `access_count` from the update set so existing access counts survive. New inserts still initialize `access_count` from input/default values.
- **D-13:** PostgreSQL `_upsert_entry` must apply the same access-count preservation rule as SQLite.
- **D-14:** Implement TASK-7 R10 for JSON, SQLite, and PostgreSQL: `update_entry_metadata()` must not reset `created_at`.
- **D-15:** Metadata-only updates must not alter `expires_at` or `ttl_seconds` unless the caller explicitly updates those fields through an existing supported path.
- **D-16:** Because `created_at` is signed, update paths must preserve or recompute signatures using the preserved `created_at`. Read `_update_mixin.py` before implementing metadata update changes.
- **D-17:** `update_data()` is the content-update path and may continue to refresh content metadata according to existing semantics, but metadata-only updates must not masquerade as content rewrites.
- **D-18:** Storage mode amplifies R10: users rely on `created_at` as durable provenance. Do not introduce storage-mode behavior that resets provenance timestamps during metadata-only operations.

### Remote Blob Eviction

- **D-19:** Implement TASK-8 in `UnifiedCache._enforce_size_limit()`: evicted entries whose `actual_path` contains `://` must be deleted through `self._blob_store.blob_backend.delete_blob(actual_path)`.
- **D-20:** Local filesystem blob deletion may keep the existing local unlink behavior, but remote URI deletion must no longer be skipped.
- **D-21:** Remote deletion failures should be caught and logged with `logger.warning`; they must not crash size-limit enforcement unless the existing local path behavior already does so.
- **D-22:** Use `InMemoryBlobBackend` and its `memory://` URIs as the fast local regression for remote-style deletion. S3 behavior is the production rationale, but tests do not need real S3 if the backend-abstraction contract is covered.
- **D-23:** Storage mode size limits are forced off, so TASK-8 must not add storage-mode eviction behavior.

### Planning and Test Packaging

- **D-24:** Preserve Phase 29 as Wave 2 consistency work. Do not silently include Phase 30 parity work, Phase 31 security/storage-mode policy work, or Phase 32 polish items.
- **D-25:** Plans should keep TASK-5 through TASK-8 traceable. The preferred granularity is four atomic plans unless the planner finds a strong reason to combine adjacent low-risk changes.
- **D-26:** Verify-first checks should be used for each task where the current bug is readily reproducible.
- **D-27:** Use `uv` for all commands. On Windows, always include `--ignore=tests/test_tensorflow_handler.py`.
- **D-28:** For changed Python files, quality gates are `ruff format`, `ruff check --fix`, `ruff check`, and `ty check` on touched files. Existing baseline `ty` diagnostics should be documented rather than hidden if unrelated.

### the agent's Discretion

- Whether to add read-path expired-entry deletion on `get()` as part of TASK-6, provided TASK-6's required init-cleanup behavior remains clear and independently verified.
- Whether to place PostgreSQL-specific coverage in backend unit tests, parity tests, or existing Docker-skipping PostgreSQL tests, as long as PostgreSQL semantics are planned explicitly.
- Whether to use direct backend tests or full `UnifiedCache` integration tests for access-count preservation, as long as at least one integration path proves cache-mode behavior.
- Whether to add a small helper for parsing `expires_at` if it reduces duplication across `_is_expired()` and backend cleanup code.

### Deferred Ideas (OUT OF SCOPE)

## Deferred Ideas

- Phase 30: PAR-01 through PAR-04, including unique temp blob names, SQLite/PostgreSQL user metadata parity, non-destructive overwrite, and backend-driven blob enumeration.
- Phase 31: SEC-01 through SEC-04 and STRG-01 through STRG-03, including storage-mode destructive API policy and fsync/durability contract.
- Phase 32: POL-01 through POL-08 release polish and small fixes.
- R11, R12, R7, U2, S1 through S4, and POL items are out of Phase 29 unless needed only as tiny compatibility adjustments to land TASK-5 through TASK-8 safely.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TTL-01 | User can set per-entry TTL and have stored `expires_at` honored by reads and cleanup across JSON, SQLite, and PostgreSQL semantics. | Implement `_is_expired`, JSON/SQLite/PG `cleanup_expired`, and parity tests for stored expiry precedence. [CITED: .planning/REQUIREMENTS.md] [VERIFIED: codebase grep] |
| TTL-02 | User can rely on init-time expired-entry cleanup deleting both metadata and blob files through the same path as public cleanup. | Route `UnifiedCache._cleanup_expired()` through public `cleanup_expired()` after lock/blob-store init. [CITED: .planning/REQUIREMENTS.md] [VERIFIED: codebase grep] |
| TTL-03 | User can overwrite or update metadata without unexpectedly resetting access counters, provenance timestamps, expiry semantics, or signatures. | Replace SQLite overwrite SQL, adjust PG `_upsert_entry`, remove `created_at` reset in metadata-only updates, and preserve signing field consistency. [CITED: .planning/REQUIREMENTS.md] [VERIFIED: codebase grep] |
| TTL-04 | User can rely on size/eviction cleanup deleting remote blobs through the configured blob backend instead of leaking S3 or memory-backed objects. | Update `_enforce_size_limit()` to call `blob_backend.delete_blob()` for URI paths and test with `memory://`. [CITED: .planning/REQUIREMENTS.md] [VERIFIED: codebase grep] |
</phase_requirements>

## Summary

Phase 29 should be planned as four traceable TASK-5 through TASK-8 units, because each acceptance item maps to a distinct implementation path: expiry predicate/metadata cleanup, init cleanup routing, metadata update/overwrite semantics, and remote URI deletion during size eviction. [CITED: .planning/phases/29-ttl-eviction-consistency/29-CONTEXT.md] [VERIFIED: codebase grep]

The current code already stores `ttl_seconds` and computes `expires_at` in JSON, SQLite, and PostgreSQL put paths, but reads and cleanup still fall back to global `created_at + ttl_seconds` cutoff behavior. [VERIFIED: codebase grep] `UnifiedCache.put()` also populates per-entry `ttl_seconds` from the configured default before signing and `put_entry()`, so backend cleanup and `_is_expired()` must treat stored expiry as the authoritative value when present. [VERIFIED: codebase grep]

The highest-risk planning detail is sequencing TASK-7: `created_at` is in the signed field set, `update_data()` intentionally refreshes `created_at`, but metadata-only backend updates currently reset it as an implementation side effect. [VERIFIED: codebase grep] The plan should preserve `update_data()` and `touch()` semantics while narrowing backend `update_entry_metadata()` to metadata-only fields. [VERIFIED: codebase grep]

**Primary recommendation:** Create four plans: TTL semantics, init cleanup routing, metadata overwrite/update preservation, and remote eviction cleanup, with one targeted verify-first regression per task. [CITED: docs/CODE_REVIEW_ACTIONS.md] [VERIFIED: codebase grep]

## Project Constraints (from AGENTS.md)

- `AGENTS.md` redirects all actionable rules to `.github/copilot-instructions.md`. [CITED: AGENTS.md]
- Use `uv` for Python/test commands; do not use `python`, `pip`, or `python -m pytest` directly. [CITED: .github/copilot-instructions.md]
- On Windows, include `--ignore=tests/test_tensorflow_handler.py` in pytest commands because TensorFlow tests hang. [CITED: .github/copilot-instructions.md]
- Full suite command is `uv run pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py`; targeted tests are preferred during development. [CITED: .github/copilot-instructions.md]
- Import `UnifiedCache` in tests from `cacheness.core`, not from the package root. [CITED: .github/copilot-instructions.md]
- The user specifically asked to ignore beads for this research; do not add beads work tracking to the plan. [CITED: user prompt]

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|--------------|----------------|-----------|
| Per-entry read expiry | API / Backend | Database / Storage | `UnifiedCache.get()` owns cache-mode read policy and calls `_is_expired()` before blob read. [VERIFIED: codebase grep] |
| Backend expired cleanup | Database / Storage | API / Backend | JSON/SQLite/PostgreSQL backends own metadata deletion, while public `cleanup_expired()` owns blob deletion and hooks. [VERIFIED: codebase grep] |
| Init-time expired cleanup | API / Backend | Database / Storage | `UnifiedCache.__init__` calls `_cleanup_expired()` after lock/backend/blob-store setup; it should route through public cleanup. [VERIFIED: codebase grep] |
| Access-count preservation | Database / Storage | API / Backend | SQLite `put_entry()` and PG `_upsert_entry()` currently overwrite counters at metadata persistence time. [VERIFIED: codebase grep] |
| Metadata-only timestamp/signature preservation | Database / Storage | API / Backend | Backend `update_entry_metadata()` currently resets `created_at`; `_update_mixin.py` re-signs after metadata updates. [VERIFIED: codebase grep] |
| Remote blob eviction | API / Backend | Blob Storage | `_enforce_size_limit()` owns evicted-entry blob deletion, and blob backends own URI-specific deletion. [VERIFIED: codebase grep] |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python | 3.12.10 available through `uv run --python 3.12` | Runtime for verification commands | Project supports Python `>=3.11`, and local probe confirmed Python 3.12.10. [VERIFIED: uv probe] [CITED: pyproject.toml] |
| pytest | 8.4.1 | Test runner | Project config and probe both indicate pytest 8.4.1. [VERIFIED: uv probe] [CITED: .planning/codebase/TESTING.md] |
| pytest-xdist | >=3.8.0 | Parallel test execution | Project pytest addopts use `-n auto --dist loadgroup`. [CITED: pyproject.toml] |
| SQLAlchemy | >=2.0.0 | SQLite/PostgreSQL metadata backend ORM/query layer | SQLite and PG backends use SQLAlchemy sessions, selects, updates, deletes, and text SQL. [CITED: pyproject.toml] [VERIFIED: codebase grep] |
| psycopg | >=3.1.0 | PostgreSQL optional backend driver | PostgreSQL extras declare `psycopg[binary]>=3.1.0`. [CITED: pyproject.toml] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Docker | 28.1.1 CLI present | Optional PostgreSQL/S3 integration services | Use only for Docker-gated PG/S3 tests; availability warning showed Docker config access is restricted. [VERIFIED: docker probe] |
| ruff | >=0.12.8 | Format/lint touched Python files | Run after code changes per project quality gate. [CITED: pyproject.toml] [CITED: .github/copilot-instructions.md] |
| ty | >=0.0.15 | Type checking touched Python files | Run after code changes and document unrelated baseline findings. [CITED: pyproject.toml] [CITED: .github/copilot-instructions.md] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| New TTL library | Custom helper inside `core.py` or small shared internal function | Use an internal parser only if it reduces duplication; no external dependency is justified. [CITED: .planning/phases/29-ttl-eviction-consistency/29-CONTEXT.md] |
| Real S3 regression for TASK-8 | `InMemoryBlobBackend` `memory://` regression | Memory backend covers the URI deletion contract without Docker/S3 flakiness. [CITED: docs/CODE_REVIEW_ACTIONS.md] [VERIFIED: codebase grep] |

**Installation:**

No new packages are required for Phase 29. [VERIFIED: codebase grep]

## Package Legitimacy Audit

No external packages should be installed for this phase, so the package legitimacy gate is not applicable. [CITED: docs/CODE_REVIEW_ACTIONS.md] [VERIFIED: codebase grep]

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| none | - | - | - | - | OK | No install planned. [VERIFIED: codebase grep] |

**Packages removed due to [SLOP] verdict:** none.
**Packages flagged as suspicious [SUS]:** none.

## Current Code Path Inventory

| Area | Exact Symbol / Location | Current Behavior | Planning Implication |
|------|-------------------------|------------------|----------------------|
| Read expiry | `UnifiedCache._is_expired()` in `src/cacheness/core.py:705` | Fetches entry and checks `created_at + ttl_seconds`; it ignores stored `expires_at`. [VERIFIED: codebase grep] | Add stored-expiry parse before TTL fallback. |
| Read path guard | `UnifiedCache.get()` in `src/cacheness/core.py:1017` | Storage mode returns `_storage_mode_get(cache_key)` before `_is_expired()`. [VERIFIED: codebase grep] | Keep this guard intact for TTL-01. |
| Put TTL storage | `UnifiedCache.put()` in `src/cacheness/core.py:941` | Copies config `default_ttl_seconds` into entry `ttl_seconds`; backend computes `expires_at`. [VERIFIED: codebase grep] | Backend cleanup must honor the stored fields already being written. |
| Init cleanup | `UnifiedCache._cleanup_expired()` in `src/cacheness/core.py:765` | Calls `metadata_backend.cleanup_expired(ttl_seconds)` directly. [VERIFIED: codebase grep] | Replace with public `self.cleanup_expired(ttl_seconds)` for TTL-02. |
| Public cleanup | `UnifiedCache.cleanup_expired()` in `src/cacheness/core.py:1225` | Scans `iter_entry_summaries()`, deletes local non-URI blob files, invokes `on_evict`, then calls backend cleanup. [VERIFIED: codebase grep] | Reuse this path at init; update its expired scan if stored `expires_at` changes the target set. |
| Size eviction | `UnifiedCache._enforce_size_limit()` in `src/cacheness/core.py:1091` | Deletes only paths without `://`; URI blobs are skipped. [VERIFIED: codebase grep] | Route URI paths through `self._blob_store.blob_backend.delete_blob(actual_path)`. |
| JSON cleanup | `JsonBackend.cleanup_expired()` in `src/cacheness/metadata/json_backend.py:390` | Deletes entries whose `created_at` is older than global cutoff. [VERIFIED: codebase grep] | Add `expires_at < now` branch and fallback cutoff branch. |
| JSON metadata update | `JsonBackend.update_entry_metadata()` in `src/cacheness/metadata/json_backend.py:191` | Resets `created_at` to now for metadata-only updates. [VERIFIED: codebase grep] | Remove reset and do not touch TTL fields unless explicitly supplied. |
| SQLite put | `SqliteBackend.put_entry()` in `src/cacheness/metadata/sqlite_backend.py:732` | Uses `INSERT OR REPLACE`; new values include `access_count`. [VERIFIED: codebase grep] | Replace with conflict update that omits `access_count` for existing rows. |
| SQLite cleanup | `SqliteBackend.cleanup_expired()` in `src/cacheness/metadata/sqlite_backend.py:1161` | Deletes by `created_at < cutoff_time`. [VERIFIED: codebase grep] | Add OR predicate for non-null `expires_at`. |
| SQLite metadata update | `SqliteBackend.update_entry_metadata()` in `src/cacheness/metadata/sqlite_backend.py:866` | Resets `entry.created_at = now`. [VERIFIED: codebase grep] | Preserve created timestamp and expiry fields. |
| PG upsert | `PostgresBackend._upsert_entry()` in `src/cacheness/storage/backends/postgresql_backend.py:951` | Existing-row update assigns `access_count=access_count_val`. [VERIFIED: codebase grep] | Preserve existing count on update. |
| PG cleanup | `PostgresBackend.cleanup_expired()` in `src/cacheness/storage/backends/postgresql_backend.py:1418` | Deletes by global cutoff and returns 0 when TTL is <= 0. [VERIFIED: codebase grep] | Implement stored-expiry predicate while preserving zero-TTL no-op policy. |
| PG metadata update | `PostgresBackend.update_entry_metadata()` in `src/cacheness/storage/backends/postgresql_backend.py:1187` | Resets `entry.created_at = now`. [VERIFIED: codebase grep] | Preserve created timestamp and expiry fields. |
| Signing fields | `VerificationMixin._extract_signable_fields()` and `CacheEntrySigner.SIGNED_FIELDS_BY_VERSION` | `created_at` and `file_hash` are signable fields. [VERIFIED: codebase grep] | Metadata-only update tests must verify signed entries still read. |
| Memory URI backend | `InMemoryBlobBackend.write_blob/delete_blob()` in `blob_backends.py:424` | Stores blobs under `memory://{blob_id}` and deletes by exact URI. [VERIFIED: codebase grep] | Use for fast TTL-04 regression. |
| S3 URI backend | `S3BlobBackend.delete_blob()` in `s3_backend.py:354` | Accepts S3 URI or key and deletes through the S3 client. [VERIFIED: codebase grep] | Production rationale for URI routing; mocked tests can cover failures. |

## Architecture Patterns

### System Architecture Diagram

```text
Cache-mode put()
  -> UnifiedCache.put()
  -> entry_data includes ttl_seconds when default TTL exists
  -> backend put_entry()
     -> JSON / SQLite / PG stores ttl_seconds and expires_at

Cache-mode get()
  -> UnifiedCache.get()
  -> storage_mode? yes -> _storage_mode_get() and no expiry
  -> storage_mode? no -> _is_expired()
     -> if stored expires_at exists: compare to now
     -> else: compare created_at + requested/config TTL
  -> verify signature/hash
  -> blob backend read
  -> update access time

Cleanup on init / public cleanup
  -> UnifiedCache._cleanup_expired()
  -> UnifiedCache.cleanup_expired()
     -> find entries using same stored-expiry semantics
     -> invoke on_evict
     -> delete local blob file or URI through blob backend
     -> metadata_backend.cleanup_expired()

Size eviction
  -> UnifiedCache._enforce_size_limit()
  -> metadata_backend.cleanup_by_size()
  -> for each removed entry:
     -> on_evict
     -> local path: unlink
     -> URI path: blob_backend.delete_blob()
```

### Recommended Plan Split

| Plan | Scope | Requirements | Why This Split |
|------|-------|--------------|----------------|
| 29-01 | TASK-5 stored `expires_at` read and cleanup semantics | TTL-01 | Touches `_is_expired()` plus three metadata backends and parity tests. [CITED: docs/CODE_REVIEW_ACTIONS.md] |
| 29-02 | TASK-6 init cleanup through public cleanup path | TTL-02 | Small `core.py` change with hook/blob-file acceptance. [CITED: docs/CODE_REVIEW_ACTIONS.md] |
| 29-03 | TASK-7 access-count, `created_at`, TTL fields, signatures | TTL-03 | Highest coupling: SQLite SQL, PG upsert, all backend metadata updates, signing tests. [CITED: docs/CODE_REVIEW_ACTIONS.md] [VERIFIED: codebase grep] |
| 29-04 | TASK-8 remote URI deletion during size eviction | TTL-04 | Isolated `core.py` eviction loop plus memory URI regression. [CITED: docs/CODE_REVIEW_ACTIONS.md] |

### Pattern 1: Stored Expiry Precedence

**What:** Parse `expires_at` first; only fall back to caller/config TTL when no stored expiry exists. [CITED: .planning/phases/29-ttl-eviction-consistency/29-CONTEXT.md]

**When to use:** `_is_expired()`, `UnifiedCache.cleanup_expired()` expired-entry scan, and JSON/SQLite/PostgreSQL `cleanup_expired()`. [VERIFIED: codebase grep]

**Example:**

```python
# Source: docs/CODE_REVIEW_ACTIONS.md TASK-5; adapt into shared helper if useful.
expires_at = entry.get("expires_at")
if expires_at:
    expires_dt = datetime.fromisoformat(expires_at) if isinstance(expires_at, str) else expires_at
    if expires_dt.tzinfo is None:
        expires_dt = expires_dt.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) > expires_dt
```

### Pattern 2: Public Cleanup Owns Blob Deletion

**What:** Metadata backends delete rows; `UnifiedCache.cleanup_expired()` deletes blobs and invokes hooks. [VERIFIED: codebase grep]

**When to use:** Init-time cleanup should call `self.cleanup_expired(ttl_seconds)` after `self._lock`, metadata backend, blob store, and write journal are initialized. [VERIFIED: codebase grep]

### Pattern 3: Metadata-Only Updates Must Not Rewrite Provenance

**What:** Backend `update_entry_metadata()` should update derived metadata fields but preserve `created_at`, `ttl_seconds`, and `expires_at` by default. [CITED: .planning/phases/29-ttl-eviction-consistency/29-CONTEXT.md] [VERIFIED: codebase grep]

**When to use:** JSON, SQLite, and PG metadata-only update methods; do not change `update_data()` or `touch()` semantics unless tests require explicit compatibility changes. [VERIFIED: codebase grep]

### Anti-Patterns to Avoid

- **Duplicating cleanup decisions in init:** Direct backend cleanup deletes metadata without public blob deletion and hooks. [VERIFIED: codebase grep]
- **Treating `expires_at` as advisory:** The locked decision says stored expiry wins when present. [CITED: .planning/phases/29-ttl-eviction-consistency/29-CONTEXT.md]
- **Using `INSERT OR REPLACE` for same-key overwrite:** SQLite replacement recreates the row and resets counters. [VERIFIED: codebase grep] [CITED: docs/CODE_REVIEW_FINDINGS.md]
- **Skipping URI paths in eviction:** This leaks remote objects such as S3 and memory-backed blobs. [CITED: docs/CODE_REVIEW_FINDINGS.md] [VERIFIED: codebase grep]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SQL conflict handling | Manual delete then insert | SQLite `INSERT ... ON CONFLICT(cache_key) DO UPDATE` or SQLAlchemy update preserving columns | Avoids resetting counters and avoids extra race windows. [VERIFIED: codebase grep] |
| Remote blob deletion | URI string parsing in `core.py` | Existing `blob_backend.delete_blob(actual_path)` | S3 and memory backends already own URI-specific behavior. [VERIFIED: codebase grep] |
| Signature field selection | New signing field list | Existing `_extract_signable_fields()` and `CacheEntrySigner` | `created_at` normalization is centralized there. [VERIFIED: codebase grep] |
| PostgreSQL service orchestration | Ad hoc local service assumptions | Existing Docker-gated PG fixtures and skips | Test docs define Docker grouping and skip behavior. [CITED: .planning/codebase/TESTING.md] |

**Key insight:** The phase is about making existing stored metadata authoritative and routing through existing abstractions; new mechanisms would increase parity risk. [VERIFIED: codebase grep]

## Common Pitfalls

### Pitfall 1: Public Cleanup Scan and Backend Cleanup Disagree

**What goes wrong:** `UnifiedCache.cleanup_expired()` may delete blobs for a different set of entries than backend `cleanup_expired()` deletes if one uses `expires_at` and the other uses only `created_at`. [VERIFIED: codebase grep]
**How to avoid:** Update both the public scan and backend cleanup predicates in the same plan and test both directions: past `expires_at` with recent `created_at`, and future `expires_at` with old `created_at`. [CITED: docs/CODE_REVIEW_ACTIONS.md]

### Pitfall 2: `ttl_seconds=None` Means No Cleanup

**What goes wrong:** Existing cleanup returns early when no TTL is configured; storage mode forces default TTL to `None`. [VERIFIED: codebase grep]
**How to avoid:** Keep storage-mode no-expiry behavior and only apply stored `expires_at` semantics to cache-mode read/cleanup paths specified by Phase 29. [CITED: .planning/phases/29-ttl-eviction-consistency/29-CONTEXT.md]

### Pitfall 3: Signature Regression Hidden by Missing `test_security.py`

**What goes wrong:** Project docs mention `tests/test_security.py`, but this checkout has no such file. [VERIFIED: rg --files]
**How to avoid:** Use `tests/test_cache_signing.py`, `tests/test_namespace_signing.py`, and `tests/test_cache_integrity_verification.py` for signature coverage. [VERIFIED: rg --files]

### Pitfall 4: PostgreSQL Coverage Is Optional/Docker-Gated

**What goes wrong:** Plans may require PG tests unconditionally and fail on machines without running services. [CITED: .planning/codebase/TESTING.md]
**How to avoid:** Put PG semantics in code review checklist and Docker-skipping tests; planners should accept skip when fixtures report unavailable. [CITED: .planning/codebase/TESTING.md]

### Pitfall 5: Memory URI Tests Need `UnifiedCache`, Not Only `BlobStore`

**What goes wrong:** Existing memory backend tests exercise `BlobStore.verify_integrity()`, not `UnifiedCache._enforce_size_limit()`. [VERIFIED: codebase grep]
**How to avoid:** Build a `UnifiedCache` configured with `blob_backend="memory"` or patch its blob backend, force eviction, and assert the evicted URI no longer exists. [CITED: docs/CODE_REVIEW_ACTIONS.md] [VERIFIED: codebase grep]

## Code Examples

### SQLite Conflict Update Shape

```sql
-- Source: derived from SqliteBackend.put_entry current INSERT OR REPLACE block.
INSERT INTO "cache_entries" (...)
VALUES (...)
ON CONFLICT(cache_key) DO UPDATE SET
  description = excluded.description,
  data_type = excluded.data_type,
  file_size = excluded.file_size,
  -- omit access_count so the existing row value survives
  ttl_seconds = excluded.ttl_seconds,
  expires_at = excluded.expires_at;
```

### Remote Eviction Deletion Shape

```python
# Source: docs/CODE_REVIEW_ACTIONS.md TASK-8.
if actual_path and "://" in actual_path:
    try:
        if self._blob_store.blob_backend.delete_blob(actual_path):
            blobs_deleted += 1
    except Exception as exc:  # cleanup must not crash eviction
        logger.warning("Failed to delete remote blob %s during size enforcement: %s", actual_path, exc)
```

### Metadata-Only Update Preservation Assertion

```python
# Source: proposed TTL-03 regression from context.
before = cache.get_metadata(cache_key=cache_key)
assert cache.update_metadata(cache_key, {"description": "new"}) is True
after = cache.get_metadata(cache_key=cache_key)
assert after["created_at"] == before["created_at"]
assert cache.get(cache_key=cache_key) is not None
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Schema v3 added TTL columns but cleanup ignored them | Phase 29 should honor stored `expires_at` end-to-end | v0.12.0 remediation planning on 2026-06-13 | Makes per-entry TTL coherent. [CITED: docs/CODE_REVIEW_FINDINGS.md] |
| Init cleanup deletes metadata only | Phase 29 should reuse public cleanup | v0.12.0 remediation planning on 2026-06-13 | Prevents blob orphans and preserves hooks. [CITED: docs/CODE_REVIEW_ACTIONS.md] |
| Eviction skips URI paths | Phase 29 should use blob backend deletion | v0.12.0 remediation planning on 2026-06-13 | Prevents remote object leaks. [CITED: docs/CODE_REVIEW_FINDINGS.md] |

**Deprecated/outdated:**
- `tests/test_security.py` as a target command is outdated for this checkout; use existing signing/integrity test files. [VERIFIED: rg --files]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `blob_backend="memory"` is usable in `UnifiedCache` configuration for the TASK-8 integration regression; if not, patching `cache._blob_store.blob_backend` with `InMemoryBlobBackend` is the fallback. [ASSUMED] | Validation Architecture | Planner may need a small setup spike for the memory eviction test. |

## Open Questions

1. **Should read-path expired entries be actively deleted during Phase 29?**
   - What we know: Context says it is discretionary and not mandatory. [CITED: .planning/phases/29-ttl-eviction-consistency/29-CONTEXT.md]
   - What's unclear: Whether adding deletion during `get()` would expand TASK-6 beyond init cleanup. [CITED: .planning/phases/29-ttl-eviction-consistency/29-CONTEXT.md]
   - Recommendation: Do not include unless it is a separately tested low-risk add-on after TASK-6 passes. [CITED: .planning/phases/29-ttl-eviction-consistency/29-CONTEXT.md]

2. **Should PG TTL/access-count tests be direct backend tests or parity tests?**
   - What we know: Context leaves placement to the agent, but PG semantics must be planned explicitly. [CITED: .planning/phases/29-ttl-eviction-consistency/29-CONTEXT.md]
   - What's unclear: Docker availability during execution. [CITED: .planning/codebase/TESTING.md]
   - Recommendation: Add direct PG backend tests in existing Docker-skipping files, plus JSON/SQLite always-on parity tests. [VERIFIED: codebase grep] [CITED: .planning/codebase/TESTING.md]

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| uv | All commands | yes | 0.11.19 | none. [VERIFIED: uv probe] |
| Python | Test runtime | yes | 3.12.10 via `uv run --python 3.12` | Use `--python 3.12` with project-local UV cache. [VERIFIED: uv probe] |
| pytest | Targeted validation | yes | 8.4.1 | Use documented test commands; one later `pytest --version` probe timed out after env recreation. [VERIFIED: uv probe] |
| Docker CLI | Optional PG/S3 tests | partial | 28.1.1 | Docker config access warning means PG/S3 tests may skip or need user environment repair. [VERIFIED: docker probe] |
| Graphify graph | Semantic graph context | no | - | Direct code/test grep used. [VERIFIED: gsd-tools graphify status] |

**Missing dependencies with no fallback:**
- None for JSON/SQLite/core Phase 29 planning. [VERIFIED: codebase grep]

**Missing dependencies with fallback:**
- Docker service/config access may block real PostgreSQL/S3 tests; fallback is skip-aware existing fixtures and memory URI tests. [VERIFIED: docker probe] [CITED: .planning/codebase/TESTING.md]

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.4.1 with pytest-xdist and JSON report. [VERIFIED: uv probe] [CITED: .planning/codebase/TESTING.md] |
| Config file | `pyproject.toml` under `[tool.pytest.ini_options]`. [CITED: pyproject.toml] |
| Quick run command | `uv run --python 3.12 pytest tests/test_core.py -x -q --ignore=tests/test_tensorflow_handler.py` [CITED: .github/copilot-instructions.md] |
| Full suite command | `uv run --python 3.12 pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py` [CITED: .github/copilot-instructions.md] |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| TTL-01 | `_is_expired()` honors stored past/future `expires_at` before fallback TTL; JSON/SQLite cleanup parity honors stored expiry; storage mode ignores expired `expires_at`. | unit + integration + parity | `uv run --python 3.12 pytest tests/test_core.py tests/test_metadata.py tests/test_backend_parity.py tests/test_storage_mode.py -x -q --ignore=tests/test_tensorflow_handler.py` | Existing files yes; new tests needed. [VERIFIED: rg --files] |
| TTL-02 | Re-init with init cleanup deletes expired metadata and blob files through public cleanup and invokes `on_evict`. | integration | `uv run --python 3.12 pytest tests/test_core.py -x -q --ignore=tests/test_tensorflow_handler.py` | Existing file yes; new regression needed. [VERIFIED: rg --files] |
| TTL-03 | Same-key overwrite preserves access count for SQLite/PG; metadata-only update preserves `created_at`, `ttl_seconds`, `expires_at`, and signed reads still verify. | backend + integration + signing | `uv run --python 3.12 pytest tests/test_metadata.py tests/test_update_operations.py tests/test_backend_parity.py tests/test_cache_signing.py tests/test_cache_integrity_verification.py -x -q --ignore=tests/test_tensorflow_handler.py` | Existing files yes; new tests needed. [VERIFIED: rg --files] |
| TTL-04 | Size eviction deletes URI blobs through configured blob backend, using `memory://` as fast regression. | integration | `uv run --python 3.12 pytest tests/test_core.py tests/test_blob_store.py -x -q --ignore=tests/test_tensorflow_handler.py` | Existing files yes; new regression needed. [VERIFIED: rg --files] |

### Sampling Rate

- **Per task commit:** Run the task-specific command from `docs/CODE_REVIEW_ACTIONS.md`, substituting `tests/test_cache_signing.py tests/test_cache_integrity_verification.py` for missing `tests/test_security.py`. [CITED: docs/CODE_REVIEW_ACTIONS.md] [VERIFIED: rg --files]
- **Per wave merge:** `uv run --python 3.12 pytest tests/test_core.py tests/test_metadata.py tests/test_backend_parity.py tests/test_update_operations.py tests/test_storage_mode.py tests/test_blob_store.py tests/test_cache_signing.py tests/test_cache_integrity_verification.py -x -q --ignore=tests/test_tensorflow_handler.py`. [VERIFIED: rg --files]
- **Phase gate:** Full suite green before `$gsd-verify-work`. [CITED: .github/copilot-instructions.md]

### Wave 0 Gaps

- [ ] Add stored-expiry precedence tests to `tests/test_core.py`, `tests/test_metadata.py`, and/or `tests/test_backend_parity.py` for TTL-01. [VERIFIED: codebase grep]
- [ ] Add init cleanup blob deletion regression in `tests/test_core.py` for TTL-02. [VERIFIED: codebase grep]
- [ ] Add access-count overwrite and metadata-only signature preservation regressions in `tests/test_update_operations.py`, backend tests, and signing/integrity tests for TTL-03. [VERIFIED: codebase grep]
- [ ] Add `memory://` size eviction regression in `tests/test_core.py` or `tests/test_blob_store.py` for TTL-04. [VERIFIED: codebase grep]

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | no | No user authentication in this cache-library phase. [VERIFIED: codebase grep] |
| V3 Session Management | no | No session state in this phase. [VERIFIED: codebase grep] |
| V4 Access Control | no | Phase does not add authorization boundaries. [CITED: .planning/REQUIREMENTS.md] |
| V5 Input Validation | yes | Parse `expires_at` defensively, treat invalid timestamps consistently with existing cleanup behavior. [VERIFIED: codebase grep] |
| V6 Cryptography | yes | Preserve existing `CacheEntrySigner` and `_extract_signable_fields()` behavior; do not hand-roll signatures. [VERIFIED: codebase grep] |

### Known Threat Patterns for This Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Signature invalidation from `created_at` mutation | Tampering | Preserve `created_at` on metadata-only updates and re-sign with existing helper when updated metadata changes signed fields. [VERIFIED: codebase grep] |
| Remote storage leakage on eviction | Information Disclosure | Delete URI blobs through configured blob backend during size enforcement. [CITED: docs/CODE_REVIEW_FINDINGS.md] [VERIFIED: codebase grep] |
| Storage-mode accidental expiry | Tampering | Keep storage-mode read path outside `_is_expired()` and keep size limits disabled. [VERIFIED: codebase grep] |

## Recommended Planning Notes

- Plan 29-01 should update `_is_expired()`, public cleanup scan, and all backend `cleanup_expired()` predicates together so blob deletion and metadata deletion agree. [VERIFIED: codebase grep]
- Plan 29-02 should be deliberately small: change `_cleanup_expired()` to call `cleanup_expired(ttl_seconds)` and test init-time blob deletion plus `on_evict`. [CITED: docs/CODE_REVIEW_ACTIONS.md]
- Plan 29-03 should run after 29-01 because preserving `expires_at` and `ttl_seconds` during metadata-only updates depends on the finalized TTL semantics. [CITED: .planning/phases/29-ttl-eviction-consistency/29-CONTEXT.md]
- Plan 29-04 can run independently after 29-02 because it touches `_enforce_size_limit()` and blob backend routing, not TTL parsing. [VERIFIED: codebase grep]

## Sources

### Primary (HIGH confidence)

- `.planning/phases/29-ttl-eviction-consistency/29-CONTEXT.md` - locked Phase 29 decisions, scope, and test ideas.
- `.planning/REQUIREMENTS.md` - TTL-01 through TTL-04 wording.
- `docs/CODE_REVIEW_FINDINGS.md` - R5, R6, R9/R10, R13 findings.
- `docs/CODE_REVIEW_ACTIONS.md` - TASK-5 through TASK-8 acceptance and commands.
- `src/cacheness/core.py` - `_is_expired`, `_cleanup_expired`, `cleanup_expired`, `_enforce_size_limit`, `get`, and `put`.
- `src/cacheness/metadata/json_backend.py` - JSON TTL/access/update behavior.
- `src/cacheness/metadata/sqlite_backend.py` - SQLite TTL/access/update behavior.
- `src/cacheness/storage/backends/postgresql_backend.py` - PG TTL/access/update behavior.
- `src/cacheness/_update_mixin.py` and `src/cacheness/_verification_mixin.py` - update and signing behavior.
- `src/cacheness/storage/backends/blob_backends.py` and `src/cacheness/storage/backends/s3_backend.py` - URI deletion abstractions.
- `.planning/codebase/TESTING.md` and `pyproject.toml` - test framework and commands.

### Secondary (MEDIUM confidence)

- Local environment probes for `uv`, Python 3.12.10, pytest 8.4.1, Docker CLI 28.1.1, and graph absence.

### Tertiary (LOW confidence)

- None beyond A1 in the assumptions log.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - drawn from pyproject, testing docs, and local probes.
- Architecture: HIGH - based on direct code symbol inspection.
- Pitfalls: HIGH - based on current code behavior and existing tests.

**Research date:** 2026-06-13
**Valid until:** 2026-07-13 for this codebase snapshot; re-run symbol grep if Phase 28 or other branches land first.
