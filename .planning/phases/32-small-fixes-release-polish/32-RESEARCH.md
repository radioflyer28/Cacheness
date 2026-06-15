# Phase 32: Small Fixes & Release Polish - Research

**Researched:** 2026-06-15
**Domain:** Python package polish, cache/blob lifecycle hardening, release metadata
**Confidence:** HIGH

## User Constraints

- No CONTEXT.md exists for this phase; roadmap, requirements, `docs/CODE_REVIEW_FINDINGS.md`, and `docs/CODE_REVIEW_ACTIONS.md` are the planning source of truth. [VERIFIED: user prompt]
- Do not invent extra product decisions; Phase 32 is limited to low-risk independent fixes from Backlog Phase 999.4 / TASK-16 through TASK-23. [VERIFIED: user prompt; .planning/ROADMAP.md]
- Ignore beads for this phase: do not require `bd`, do not create beads issues, and do not make beads usage a prerequisite. [VERIFIED: user prompt]
- Continue to honor `uv`-only commands, TensorFlow test ignore on Windows, and minimal-blast-radius public API guidance. [VERIFIED: user prompt; .github/copilot-instructions.md]

## Project Constraints (from AGENTS.md)

- `AGENTS.md` redirects all actionable instructions to `.github/copilot-instructions.md`. [VERIFIED: AGENTS.md]
- Use `uv` for Python, tests, and tooling; do not call `python`, `pip`, or `python -m pytest` directly. [VERIFIED: .github/copilot-instructions.md]
- Full test command is `uv run pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py`; Windows runs must ignore TensorFlow tests because they can hang. [VERIFIED: .github/copilot-instructions.md]
- For Python changes, run targeted tests, then `uv run ruff format <files>`, `uv run ruff check --fix <files>`, `uv run ruff check <files>`, and `uv run ty check <files>`. [VERIFIED: .github/copilot-instructions.md]
- Tests that need the class name should normally import `UnifiedCache` from `cacheness.core`; Phase 32 POL-08 intentionally changes package-root ergonomics and must add a root-import regression test. [VERIFIED: .github/copilot-instructions.md; docs/CODE_REVIEW_ACTIONS.md]
- Preserve public API signatures unless a task explicitly says otherwise. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md]
- If a task's verify-first behavior differs from the finding, stop and report instead of improvising. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md]

## Summary

Phase 32 is a release-polish batch, not a new feature phase. It should be planned as eight narrow commits mapped one-to-one to POL-01 through POL-08 / TASK-16 through TASK-23, with shared BlobStore and backend work sequenced to reduce conflicts. [VERIFIED: .planning/ROADMAP.md; .planning/REQUIREMENTS.md; docs/CODE_REVIEW_ACTIONS.md]

The highest-conflict area is BlobStore/filesystem backend polish: POL-01, POL-02, and POL-05 all touch blob storage paths or metadata in `src/cacheness/storage/blob_store.py` and `src/cacheness/storage/backends/blob_backends.py`. POL-04 and POL-08 are package-surface release polish and can be grouped late so version/import expectations are verified once. [VERIFIED: source grep; src/cacheness/storage/blob_store.py; src/cacheness/storage/backends/blob_backends.py; src/cacheness/__init__.py; pyproject.toml]

**Primary recommendation:** Plan Phase 32 in three waves: BlobStore safety, backend/reporting polish, then package-surface release polish, with verify-first probes at the start of every task. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md]

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|--------------|----------------|-----------|
| Caller metadata immutability | API / Backend | Browser / Client: none | `BlobStore.put()` owns the mutation boundary before metadata persistence. [VERIFIED: src/cacheness/storage/blob_store.py] |
| Blob key sanitization collision resistance | API / Backend | Database / Storage | BlobStore chooses public blob keys; filesystem backend persists resulting IDs. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md; src/cacheness/storage/blob_store.py] |
| Hot-path metadata read reduction | API / Backend | Database / Storage | `UnifiedCache.get()` already has the metadata entry and should avoid asking the backend again for expiry. [VERIFIED: src/cacheness/core.py] |
| Package version/import ergonomics | Package Surface | API / Backend | `pyproject.toml` and `src/cacheness/__init__.py` define installed metadata and root imports. [VERIFIED: pyproject.toml; src/cacheness/__init__.py] |
| Absolute blob ID rejection | Database / Storage | API / Backend | `FilesystemBlobBackend._get_blob_path()` converts blob IDs into paths and must reject escape inputs before joining. [VERIFIED: src/cacheness/storage/backends/blob_backends.py] |
| SQLite PRAGMA lifecycle | Database / Storage | API / Backend | SQLite connection events and `close()` own PRAGMA timing. [VERIFIED: src/cacheness/metadata/sqlite_backend.py] |
| S3 namespace delete reporting | Database / Storage | API / Backend | `S3BlobBackend.delete_namespace_blobs()` owns S3 bulk-delete behavior and failure reporting. [VERIFIED: src/cacheness/storage/backends/s3_backend.py] |

## Phase Requirements

| ID | Task | Requirement | Research Support |
|----|------|-------------|------------------|
| POL-01 | TASK-16 | User metadata dictionaries passed to `BlobStore.put()` remain unchanged. | `custom_metadata = metadata or {}` is still present and is mutated with storage/signing fields. [VERIFIED: .planning/REQUIREMENTS.md; src/cacheness/storage/blob_store.py] |
| POL-02 | TASK-17 | Sanitized blob keys cannot silently collide. | `_sanitize_key()` strips characters and truncates to 64 chars without adding a hash. [VERIFIED: .planning/REQUIREMENTS.md; src/cacheness/storage/blob_store.py] |
| POL-03 | TASK-18 | `get()` avoids redundant metadata reads when checking expiry. | `get()` fetches `entry`, then `_is_expired(cache_key, ttl)` calls `metadata_backend.get_entry(cache_key)` again. [VERIFIED: .planning/REQUIREMENTS.md; src/cacheness/core.py] |
| POL-04 | TASK-19 | Package version metadata matches the shipped changelog/version line. | `CHANGELOG.md` has `[0.12.0] - Unreleased`; `pyproject.toml` and `cacheness.__version__` still say `0.6.0`. [VERIFIED: CHANGELOG.md; pyproject.toml; src/cacheness/__init__.py] |
| POL-05 | TASK-20 | Absolute blob IDs are rejected before path construction. | `_get_blob_path()` sanitizes traversal separators but does not explicitly reject absolute IDs before joining with `base_dir`. [VERIFIED: src/cacheness/storage/backends/blob_backends.py] |
| POL-06 | TASK-21 | SQLite PRAGMA behavior runs at the correct lifecycle point. | `PRAGMA optimize` and `PRAGMA page_size` run on connect; `close()` only disposes the engine. [VERIFIED: src/cacheness/metadata/sqlite_backend.py; docs/CODE_REVIEW_FINDINGS.md] |
| POL-07 | TASK-22 | S3 namespace deletion reports per-object failures. | `delete_namespace_blobs()` uses `Delete={"Objects": objects, "Quiet": True}` and counts requested deletes, not returned failures. [VERIFIED: src/cacheness/storage/backends/s3_backend.py] |
| POL-08 | TASK-23 | `from cacheness import UnifiedCache` works. | Root package imports `UnifiedCache as cacheness` and does not export the `UnifiedCache` name. [VERIFIED: src/cacheness/__init__.py; .github/copilot-instructions.md] |

## Standard Stack

No new runtime or test packages are required for Phase 32. Use existing project dependencies and tools only: `uv`, `pytest`, `ruff`, `ty`, `xxhash`, SQLAlchemy/SQLite, and existing S3 test mocking where available. [VERIFIED: pyproject.toml; docs/CODE_REVIEW_ACTIONS.md]

**Installation:** None. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md]

## Package Legitimacy Audit

Not applicable. Phase 32 should install no external packages. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md; pyproject.toml]

## File/Test Impact Map

| Task | Primary Files | Likely Tests | Notes |
|------|---------------|--------------|-------|
| TASK-16 / POL-01 | `src/cacheness/storage/blob_store.py` | `tests/test_blob_store.py` | Add test asserting caller `metadata` dict is unchanged after `put()`. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md] |
| TASK-17 / POL-02 | `src/cacheness/storage/blob_store.py` | `tests/test_blob_store.py`, maybe `tests/test_blob_namespace.py` | Use `xxhash.xxh3_64(key.encode()).hexdigest()[:16]` when sanitization changes key; round-trip both colliding examples. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md] |
| TASK-18 / POL-03 | `src/cacheness/core.py` | `tests/test_core.py` | Change `_is_expired()` to accept optional existing entry; test with a counting/mock backend if practical. [VERIFIED: src/cacheness/core.py] |
| TASK-19 / POL-04 | `pyproject.toml`, `src/cacheness/__init__.py`, maybe `CHANGELOG.md` | import/version smoke command or a small packaging/import test | Decide whether release polish should set `0.12.0` directly or align to current changelog line. Do not update unrelated changelog content. [VERIFIED: CHANGELOG.md; pyproject.toml] |
| TASK-20 / POL-05 | `src/cacheness/storage/backends/blob_backends.py` | `tests/test_blob_namespace.py` or `tests/test_blob_store.py` | Add direct `FilesystemBlobBackend.write_blob()` or `_get_blob_path()` regression for absolute IDs. [VERIFIED: src/cacheness/storage/backends/blob_backends.py] |
| TASK-21 / POL-06 | `src/cacheness/metadata/sqlite_backend.py` | `tests/test_sqlite_schema_versioning.py`, `tests/test_metadata.py` | Move/trigger `PRAGMA optimize` during close; remove or document per-connect `page_size`. Existing tests may be enough plus a mocked cursor/unit check. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md] |
| TASK-22 / POL-07 | `src/cacheness/storage/backends/s3_backend.py` | `tests/test_s3_blob_backend.py` or `tests/test_s3_orphan_cleanup.py` | Mock `delete_objects` response with `Deleted` and `Errors`; acceptance asks for per-object failure logging and `(deleted, failed)` counts. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md] |
| TASK-23 / POL-08 | `src/cacheness/__init__.py` | `tests/test_core.py` or new import smoke test | Export `UnifiedCache` while keeping existing `cacheness` alias for compatibility. [VERIFIED: src/cacheness/__init__.py] |

## Recommended Plan Slicing and Waves

### Wave 1 - BlobStore caller-facing safety

1. TASK-16 / POL-01: copy caller metadata before adding internal fields. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md]
2. TASK-17 / POL-02: hash-suffix sanitized keys when the public key changes. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md]

These should be sequenced in one wave because both edit `BlobStore.put()`-adjacent key/metadata behavior and both primarily use `tests/test_blob_store.py`. [VERIFIED: src/cacheness/storage/blob_store.py]

### Wave 2 - Backend and hot-path polish

3. TASK-18 / POL-03: pass the existing entry into expiry logic to remove the double metadata read. [VERIFIED: src/cacheness/core.py]
4. TASK-20 / POL-05: reject absolute blob IDs in filesystem backend path construction. [VERIFIED: src/cacheness/storage/backends/blob_backends.py]
5. TASK-21 / POL-06: move or document SQLite PRAGMA lifecycle behavior. [VERIFIED: src/cacheness/metadata/sqlite_backend.py]
6. TASK-22 / POL-07: report S3 per-object delete failures. [VERIFIED: src/cacheness/storage/backends/s3_backend.py]

These are independent by file, but they should run after Wave 1 so BlobStore tests are not repeatedly editing the same area. [VERIFIED: source grep]

### Wave 3 - Release/package surface

7. TASK-19 / POL-04: align package version with the v0.12.0 release line. [VERIFIED: CHANGELOG.md; pyproject.toml]
8. TASK-23 / POL-08: re-export `UnifiedCache` from package root while preserving `cacheness`. [VERIFIED: src/cacheness/__init__.py]

Keep package-surface changes last so the final smoke command can validate both `cacheness.__version__` and root imports together. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md]

## Verify-First Gates

| Task | Verify First | Stop Condition |
|------|--------------|----------------|
| TASK-16 | Reproduce that `metadata` dict gains `actual_path`/format/hash/signature fields after `BlobStore.put()`. [VERIFIED: src/cacheness/storage/blob_store.py] | If caller metadata is already unchanged, stop and re-scope to tests only. |
| TASK-17 | Show `_sanitize_key("a:b")` and `_sanitize_key("ab")` or equivalent changed/raw collision examples still collide. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md] | If sanitization already appends a stable hash, stop and only add missing tests. |
| TASK-18 | Count backend `get_entry()` calls during one cache hit. [VERIFIED: src/cacheness/core.py] | If only one read occurs, stop and update finding rather than changing expiry semantics. |
| TASK-19 | Run `uv run python -c "import cacheness; print(cacheness.__version__)"`. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md] | If it already matches the release line, skip code change and add/adjust smoke coverage. |
| TASK-20 | Call `FilesystemBlobBackend(...)._get_blob_path()` or `write_blob()` with an absolute blob ID. [VERIFIED: src/cacheness/storage/backends/blob_backends.py] | If it already raises `ValueError`, add/keep regression only. |
| TASK-21 | Check connect hook and close behavior for `PRAGMA optimize`. [VERIFIED: src/cacheness/metadata/sqlite_backend.py] | If optimize already runs on close and page-size is no longer per-connect, skip code change. |
| TASK-22 | Mock S3 `delete_objects` returning an `Errors` list. [VERIFIED: src/cacheness/storage/backends/s3_backend.py] | If current method already logs/returns failures, align tests with current contract. |
| TASK-23 | Run `uv run python -c "from cacheness import UnifiedCache; print(UnifiedCache)"`. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md] | If import already works, add a regression test only. |

## Risks and Pitfalls

### Sanitized Key Compatibility

Changing `_sanitize_key()` changes storage keys only for user-provided keys that need sanitization or truncation. This is acceptable for POL-02 because the old behavior could silently collide, but the plan should mention it in the commit summary and avoid broad migration work. [VERIFIED: docs/CODE_REVIEW_FINDINGS.md; docs/CODE_REVIEW_ACTIONS.md]

### Double-Read Fix Must Preserve TTL Semantics

`_is_expired()` currently honors stored `expires_at` before falling back to TTL; POL-03 must preserve that Phase 29 behavior while accepting an already-fetched entry. [VERIFIED: src/cacheness/core.py; .planning/REQUIREMENTS.md]

### S3 Return-Type Contract

TASK-22 says return `(deleted, failed)` counts, which is a behavior/API change for `delete_namespace_blobs()` from an integer count. The planner should make downstream call sites explicit and keep the change limited to namespace deletion reporting. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md; src/cacheness/storage/backends/s3_backend.py]

### Release Version Scope

`CHANGELOG.md` currently marks `0.12.0` as Unreleased. POL-04 should align package metadata to that release line, but should not mark the release date or ship notes unless the plan explicitly includes release finalization. [VERIFIED: CHANGELOG.md]

### Existing User Changes

The working tree already has uncommitted changes in `.planning/config.json` and several tests. The Phase 32 executor must not revert those changes and should isolate edits to task files. [VERIFIED: git status]

## Out of Scope

- Storage-mode destructive API hardening, unsigned-entry default changes, JSON write batching, signing scheme unification, and new fsync policy are explicitly out of scope in `docs/CODE_REVIEW_ACTIONS.md`. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md]
- Phase 31 security/storage-mode fixes are complete and should not be re-opened unless a Phase 32 test exposes a regression. [VERIFIED: .planning/phases/31-security-storage-mode-posture/31-VERIFICATION.md]
- Phase 30 backend enumeration and unique temp filesystem writes appear already implemented; do not re-plan those as Phase 32 work. [VERIFIED: src/cacheness/storage/backends/blob_backends.py; .planning/ROADMAP.md]

## Architecture Patterns

### System Architecture Diagram

```text
User call / import / release check
  |
  +--> BlobStore.put(data, key, metadata)
  |      |
  |      +--> sanitize public key --> handler writes payload --> blob backend persists bytes
  |      +--> copy caller metadata --> add internal metadata --> metadata backend stores entry
  |
  +--> UnifiedCache.get(cache_key)
  |      |
  |      +--> metadata_backend.get_entry once --> expiry check uses same entry --> blob read
  |
  +--> FilesystemBlobBackend.write_blob(blob_id)
  |      |
  |      +--> reject absolute/path-escape IDs --> shard path under base_dir/namespace
  |
  +--> SQLiteBackend lifecycle
  |      |
  |      +--> connection PRAGMAs on connect --> PRAGMA optimize on close
  |
  +--> S3BlobBackend.delete_namespace_blobs(namespace)
  |      |
  |      +--> list objects --> delete_objects with visible errors --> report deleted/failed
  |
  +--> Package root
         |
         +--> __version__ and UnifiedCache export match release expectations
```

### Recommended Project Structure

```text
src/cacheness/
  __init__.py                         # package exports and __version__
  core.py                             # UnifiedCache hot-path get/expiry
  metadata/sqlite_backend.py          # SQLite PRAGMA lifecycle
  storage/blob_store.py               # BlobStore put metadata and key sanitization
  storage/backends/blob_backends.py   # Filesystem blob ID path validation
  storage/backends/s3_backend.py      # S3 namespace deletion reporting
tests/
  test_blob_store.py                  # POL-01/POL-02
  test_core.py                        # POL-03/POL-08 smoke if no new file
  test_blob_namespace.py              # POL-05 direct backend path tests
  test_sqlite_schema_versioning.py    # POL-06
  test_s3_blob_backend.py             # POL-07
```

### Pattern: Minimal Local Regression

**What:** Each task should add one targeted regression test for the specific code-review finding before or alongside the fix. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md]

**When to use:** All POL tasks; they are low-risk independent fixes and should not broaden into redesign. [VERIFIED: .planning/ROADMAP.md]

**Example:**

```python
# Source: docs/CODE_REVIEW_ACTIONS.md TASK-16
metadata = {"experiment": "x42"}
store.put({"value": 1}, key="sample", metadata=metadata)
assert metadata == {"experiment": "x42"}
```

### Anti-Patterns to Avoid

- **Broad release refactor:** Do not combine version/import polish with storage lifecycle changes in one commit. [VERIFIED: .planning/ROADMAP.md]
- **Re-opening completed Phase 31 contracts:** Keep staged-key, encryption, write-intent, storage-mode warnings, and fsync behavior unchanged unless directly broken by a POL task. [VERIFIED: 31-VERIFICATION.md]
- **Changing root import by removing alias:** `from cacheness import cacheness` must continue to work while adding `UnifiedCache`. [VERIFIED: src/cacheness/__init__.py]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Hash suffix for sanitized keys | Custom ad-hoc checksum | Existing `xxhash.xxh3_64` dependency | Project already uses xxhash and TASK-17 specifies it. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md; pyproject.toml] |
| S3 delete test infrastructure | Real AWS calls | Existing moto/mock or direct mocked boto3 client | S3 tests already use moto availability guards and mocks. [VERIFIED: tests/test_s3_blob_backend.py] |
| Version source | New version file | Existing `pyproject.toml` and `src/cacheness/__init__.py` | Current package metadata is defined there. [VERIFIED: pyproject.toml; src/cacheness/__init__.py] |

## Code Examples

### Metadata Copy Pattern

```python
# Source: docs/CODE_REVIEW_ACTIONS.md TASK-16
custom_metadata = dict(metadata or {})
custom_metadata["actual_path"] = self._to_relative_path(final_path)
```

### Sanitized Key Hash Pattern

```python
# Source: docs/CODE_REVIEW_ACTIONS.md TASK-17
if safe_key != key:
    digest = xxhash.xxh3_64(key.encode()).hexdigest()[:16]
    return f"{safe_key[:48]}_{digest}"
```

### Root Export Pattern

```python
# Source: src/cacheness/__init__.py current alias plus TASK-23 target
from .core import CacheConfig, UnifiedCache, get_cache

cacheness = UnifiedCache
```

## Common Pitfalls

### Pitfall 1: Treating Small Fixes as No-Test Changes

**What goes wrong:** Tiny-looking edits change public key identity, package exports, or backend deletion behavior without a regression. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md]
**How to avoid:** One focused test per POL task plus the task-specific Tier 1 command. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md]

### Pitfall 2: Returning S3 Counts Without Updating Consumers

**What goes wrong:** `delete_namespace_blobs()` callers may expect an int. [VERIFIED: src/cacheness/storage/backends/s3_backend.py]
**How to avoid:** Search call sites before changing the return shape; adapt only direct callers and tests. [VERIFIED: source grep]

### Pitfall 3: Version Bump Without Import Smoke

**What goes wrong:** `pyproject.toml` and `cacheness.__version__` can diverge because both currently contain `0.6.0`. [VERIFIED: pyproject.toml; src/cacheness/__init__.py]
**How to avoid:** Verify `import cacheness; print(cacheness.__version__)` and `from cacheness import UnifiedCache` together in Wave 3. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md]

## State of the Art

| Old Approach | Current/Target Approach | When Changed | Impact |
|--------------|-------------------------|--------------|--------|
| Root package exports `cacheness` alias only | Add `UnifiedCache` root export while preserving alias | Phase 32 target | Removes recurring import foot-gun. [VERIFIED: .github/copilot-instructions.md; docs/CODE_REVIEW_ACTIONS.md] |
| Sanitized keys strip/truncate only | Sanitized keys include original-key hash suffix when changed | Phase 32 target | Prevents silent collisions for distinct user keys. [VERIFIED: docs/CODE_REVIEW_FINDINGS.md] |
| S3 namespace delete uses quiet bulk deletion | Surface per-object failures | Phase 32 target | Avoids hidden remote cleanup failures. [VERIFIED: docs/CODE_REVIEW_FINDINGS.md] |
| Filesystem temp writes used deterministic `.tmp` names | Unique temp names already implemented | Phase 30 complete/current source | Do not re-plan TASK-9. [VERIFIED: src/cacheness/storage/backends/blob_backends.py; .planning/ROADMAP.md] |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `0.12.0` is the target package version for POL-04 because the changelog has `[0.12.0] - Unreleased` and the milestone is v0.12.0. [ASSUMED] | Phase Requirements / Risks | Planner may need a human release-date/version decision before editing metadata. |

## Open Questions

1. **Should POL-04 set `0.12.0` now or only prepare the release metadata?**
   - What we know: changelog top line is `[0.12.0] - Unreleased`; current package metadata is `0.6.0`. [VERIFIED: CHANGELOG.md; pyproject.toml]
   - What's unclear: whether the release date should be filled during Phase 32 or later.
   - Recommendation: Plan the version bump to `0.12.0` but leave changelog date unchanged unless the executor is explicitly doing release finalization. [ASSUMED]

2. **Should S3 `delete_namespace_blobs()` return type change be treated as public?**
   - What we know: TASK-22 requests `(deleted, failed)` counts. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md]
   - What's unclear: whether external callers use this method directly.
   - Recommendation: Search call sites during the task and document the behavior change in the commit. [VERIFIED: source grep]

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| `uv` | all commands | yes | 0.11.19 | none [VERIFIED: command probe] |
| `pytest` | tests | yes, but probe timed out after printing version | 8.4.1 observed; lock/spec also say 8.4.1 | Use normal project command; if cache errors recur, set project-local `UV_CACHE_DIR`. [VERIFIED: command probe; pyproject.toml; uv.lock] |
| `ruff` | quality gates | configured; command probe hit local uv cache/interpreter issue | pyproject >=0.12.8, lock 0.12.9 | Use project-local UV cache or existing approved project test command pattern. [VERIFIED: pyproject.toml; uv.lock; command probe] |
| `ty` | type gate | configured; command probe hit local uv cache/interpreter issue | pyproject/lock 0.0.15 | Use project-local UV cache or run after environment settles. [VERIFIED: pyproject.toml; uv.lock; command probe] |
| moto/boto3 | S3 tests | optional via tests | not probed | If unavailable, use mocked client unit test for TASK-22. [VERIFIED: tests/test_s3_blob_backend.py] |

**Missing dependencies with no fallback:** None identified for planning. [VERIFIED: pyproject.toml]

**Missing dependencies with fallback:** S3 integration dependencies may be skipped; TASK-22 can use direct mocks. [VERIFIED: tests/test_s3_blob_backend.py]

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.4.1 observed/configured [VERIFIED: command probe; pyproject.toml] |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` [VERIFIED: pyproject.toml] |
| Quick run command | `uv run pytest <targeted tests> -x -q --ignore=tests/test_tensorflow_handler.py` [VERIFIED: .github/copilot-instructions.md] |
| Full suite command | `uv run pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py` [VERIFIED: .github/copilot-instructions.md] |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| POL-01 | Caller metadata dict unchanged after `BlobStore.put()` | unit | `uv run pytest tests/test_blob_store.py -x -q --ignore=tests/test_tensorflow_handler.py` | yes |
| POL-02 | Sanitized colliding keys get distinct storage keys and round-trip | unit | `uv run pytest tests/test_blob_store.py tests/test_blob_namespace.py -x -q --ignore=tests/test_tensorflow_handler.py` | yes |
| POL-03 | Cache hit uses one metadata read for expiry | unit/integration | `uv run pytest tests/test_core.py -x -q --ignore=tests/test_tensorflow_handler.py` | yes |
| POL-04 | Package version metadata matches release line | smoke | `uv run python -c "import cacheness; print(cacheness.__version__)"` | n/a |
| POL-05 | Absolute blob IDs rejected before path join | unit | `uv run pytest tests/test_blob_namespace.py tests/test_blob_store.py -x -q --ignore=tests/test_tensorflow_handler.py` | yes |
| POL-06 | SQLite PRAGMA optimize lifecycle corrected | unit/integration | `uv run pytest tests/test_sqlite_schema_versioning.py tests/test_metadata.py -x -q --ignore=tests/test_tensorflow_handler.py` | yes |
| POL-07 | S3 namespace deletion reports failures | unit | `uv run pytest tests/test_s3_blob_backend.py tests/test_s3_orphan_cleanup.py -x -q --ignore=tests/test_tensorflow_handler.py` | yes |
| POL-08 | `from cacheness import UnifiedCache` works | smoke/unit | `uv run python -c "from cacheness import UnifiedCache; print(UnifiedCache.__name__)"` | n/a |

### Sampling Rate

- Per task commit: run the task's targeted test command and quality gates for changed Python files. [VERIFIED: .github/copilot-instructions.md]
- Per wave merge: run all tests touched by that wave. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md]
- Phase gate: run full suite once before v0.12.0 completion. [VERIFIED: .planning/ROADMAP.md; .github/copilot-instructions.md]

### Wave 0 Gaps

- Add or extend focused tests for POL-01 through POL-08 before claiming each fix complete. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md]
- Environment note: local `uv run ruff/ty` probes failed due a UV cache/interpreter access issue; executor should use normal project workflow and, if needed, set `UV_CACHE_DIR` to a project-local path. [VERIFIED: command probe]

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | no | No auth surface in Phase 32. [VERIFIED: phase scope] |
| V3 Session Management | no | No session surface in Phase 32. [VERIFIED: phase scope] |
| V4 Access Control | no | No permission model changes in Phase 32. [VERIFIED: phase scope] |
| V5 Input Validation | yes | Reject absolute blob IDs and collision-prone sanitized keys. [VERIFIED: POL-02; POL-05] |
| V6 Cryptography | no direct change | Preserve Phase 31 signing/encryption behavior; POL-01 must not mutate metadata in a way that breaks signing. [VERIFIED: 31-VERIFICATION.md; src/cacheness/storage/blob_store.py] |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Path traversal or absolute path escape via blob ID | Tampering | Explicit absolute-path rejection before constructing filesystem path. [VERIFIED: docs/CODE_REVIEW_FINDINGS.md] |
| Sanitized-key collision overwrite | Tampering | Stable hash suffix when sanitization changes the original key. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md] |
| Hidden S3 delete failure | Repudiation | Log/report per-object failures from S3 bulk-delete responses. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md] |

## Sources

### Primary (HIGH confidence)

- `.planning/ROADMAP.md` - Phase 32 scope, backlog 999.4 mapping, success criteria.
- `.planning/REQUIREMENTS.md` - POL-01 through POL-08 definitions and traceability.
- `docs/CODE_REVIEW_FINDINGS.md` - original findings U3, U6, U7, U8, U9, S6, R15, S3 deletion reporting.
- `docs/CODE_REVIEW_ACTIONS.md` - TASK-16 through TASK-23 acceptance criteria and global execution rules.
- `.planning/phases/31-security-storage-mode-posture/31-VERIFICATION.md` - completed Phase 31 behavior to avoid re-opening.
- `.github/copilot-instructions.md` - uv/test/quality gate and public API instructions.
- Current source files under `src/cacheness/` and tests under `tests/` cited above.

### Secondary (MEDIUM confidence)

- Command probes for `uv`, `pytest`, `ruff`, and `ty`; `uv` and `pytest` produced useful version data, while `ruff`/`ty` probes hit environment issues.

### Tertiary (LOW confidence)

- None used for implementation recommendations.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - no new packages; existing project tools verified in pyproject and instructions.
- Architecture: HIGH - based on direct source reads and prior phase verification.
- Pitfalls: HIGH - derived from code-review action specs and current source inspection.

**Research date:** 2026-06-15
**Valid until:** 2026-07-15, or sooner if Phase 32 source files change before planning.
