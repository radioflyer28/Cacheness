---
phase: 32-small-fixes-release-polish
verified: 2026-06-15T20:59:12Z
status: passed
score: 12/12 must-haves verified
overrides_applied: 0
---

# Phase 32: Small Fixes & Release Polish Verification Report

**Phase Goal:** Land low-risk independent code-review fixes and polish the package surface for the v0.12.0 release.
**Verified:** 2026-06-15T20:59:12Z
**Status:** passed
**Re-verification:** No - initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | POL-01: User metadata dictionaries passed to `BlobStore.put()` remain unchanged by Cacheness. | VERIFIED | `src/cacheness/storage/blob_store.py:632` copies with `dict(metadata or {})`; `tests/test_blob_store.py:75` verifies caller dict unchanged and stored internal metadata preserved. |
| 2 | POL-02: User blob keys that require sanitization cannot silently collide with distinct original keys. | VERIFIED | `src/cacheness/storage/blob_store.py:1641` preserves safe keys and appends an `xxh3_64` digest for transformed keys; `tests/test_blob_store.py:88` and `:99` cover sanitized-character and truncation collisions. |
| 3 | POL-03: Hot-path `get()` calls avoid redundant metadata reads when checking expiry. | VERIFIED | `src/cacheness/core.py:844` accepts an optional entry and `src/cacheness/core.py:1201` passes the already-fetched entry; `tests/test_core.py:158` asserts one `get_entry()` call on a cache hit. |
| 4 | POL-04: User-visible package version metadata matches the shipped version line. | VERIFIED | `pyproject.toml:3` and `src/cacheness/__init__.py:95` both report `0.12.0`; smoke command printed `0.12.0 UnifiedCache`. |
| 5 | POL-05: User-provided absolute blob IDs are rejected before path construction. | VERIFIED | `src/cacheness/storage/backends/blob_backends.py:441` rejects `Path(blob_id).is_absolute()` before separator replacement and base-dir join; `tests/test_blob_namespace.py:69` verifies no escaped write. |
| 6 | POL-06: SQLite PRAGMA behavior runs in the correct lifecycle location. | VERIFIED | Connect hook no longer runs `PRAGMA optimize`/`PRAGMA page_size`; `src/cacheness/metadata/sqlite_backend.py:1347` runs `PRAGMA optimize` in `close()` before `dispose()` and preserves disposal in `finally`; `tests/test_sqlite_schema_versioning.py:89` covers ordering and failure. |
| 7 | POL-07: S3 namespace deletion reports per-object failures instead of hiding them. | VERIFIED | `src/cacheness/storage/backends/s3_backend.py:634` returns `(deleted, failed)`, removes quiet bulk delete, inspects `Deleted`/`Errors`, and logs failed keys; tests at `tests/test_s3_blob_backend.py:569` and `tests/test_namespace_integration.py:540` cover failure and call-site compatibility. |
| 8 | POL-08: Users can import `UnifiedCache` from the package root without losing the legacy alias. | VERIFIED | `src/cacheness/__init__.py:32` imports `UnifiedCache`, `:100` preserves `cacheness = UnifiedCache`, and `:222` includes both exports; `tests/test_core.py:21` verifies identity and `__all__`. |
| 9 | Roadmap SC1: Each small fix is independently tested and committed with minimal blast radius. | VERIFIED | Implementation commits exist for 32-01 through 32-08: `18b2629`, `4e034ac`, `3a471f8`, `b46033a`, `a9e2cc7`, `aa96c9f`, `02299c0`, `bd2f2f5`; `4d6e0d1` is a focused final checkpoint regression fix. |
| 10 | Roadmap SC2: Package version metadata and root imports match user expectations. | VERIFIED | POL-04 and POL-08 evidence above plus independent smoke command: `uv run python -c ...` printed `0.12.0 UnifiedCache`. |
| 11 | Roadmap SC3: S3 and SQLite lifecycle/reporting fixes have targeted tests or mocked verification. | VERIFIED | SQLite lifecycle tests exist at `tests/test_sqlite_schema_versioning.py:89`; S3 direct mocked response test exists at `tests/test_s3_blob_backend.py:569`. |
| 12 | Roadmap SC4: Full suite passes once before v0.12.0 completion. | VERIFIED | Orchestrator final checkpoint passed: `uv run pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py` -> `1857 passed, 125 skipped, 27 warnings in 77.77s`. Final regression fix commit: `4d6e0d1`; recovery docs: `e88219e`. |

**Score:** 12/12 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `src/cacheness/storage/blob_store.py` | POL-01 metadata copy and POL-02 transformed-key digest | VERIFIED | Substantive source changes present and regression-tested. |
| `src/cacheness/core.py` | POL-03 entry reuse plus final checkpoint TTL/read behavior | VERIFIED | Hot path passes entry to `_is_expired`; final checkpoint regressions fixed in `4d6e0d1`. |
| `src/cacheness/storage/backends/blob_backends.py` | POL-05 absolute ID rejection | VERIFIED | Raw absolute ID guard precedes path construction. |
| `src/cacheness/metadata/sqlite_backend.py` | POL-06 close-time optimize | VERIFIED | Close-time `PRAGMA optimize` guarded before `engine.dispose()`. |
| `src/cacheness/storage/backends/s3_backend.py` | POL-07 per-object delete reporting | VERIFIED | Tuple return and `Deleted`/`Errors` accounting implemented. |
| `pyproject.toml`, `uv.lock`, `src/cacheness/__init__.py` | POL-04/POL-08 release/import surface | VERIFIED | Version and root export smoke passed. |
| Phase 32 regression tests | Focused tests for POL-01..POL-08 | VERIFIED | Targeted smoke command passed 7 representative Phase 32 nodes. |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `BlobStore.put` | metadata immutability test | caller metadata copy | WIRED | `test_put_does_not_mutate_caller_metadata` exercises the exact public API path. |
| `BlobStore._sanitize_key` | `xxhash.xxh3_64` | transformed-key suffix | WIRED | Manual grep verified digest use at `blob_store.py:1647`; tests assert expected digest-bearing keys. |
| `UnifiedCache.get` | `_is_expired` | existing `entry=entry` argument | WIRED | `core.py:1201` passes the fetched metadata entry. |
| `FilesystemBlobBackend.write_blob` | `_get_blob_path` | absolute-ID `ValueError` | WIRED | Regression calls `write_blob()` and observes rejection before file creation. |
| `SqliteBackend.close` | `PRAGMA optimize` | best-effort before dispose | WIRED | Tests verify optimize precedes dispose and dispose still happens on optimize failure. |
| `S3BlobBackend.delete_namespace_blobs` | boto3 `delete_objects` response | `Deleted`/`Errors` count inspection | WIRED | Mock test verifies response accounting and failed-key logging. |
| `src/cacheness/__init__.py` | `cacheness.core.UnifiedCache` | root re-export and alias | WIRED | Smoke and test assert `UnifiedCache is cacheness`. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|---|---|---|---|---|
| `BlobStore.put()` metadata | `custom_metadata` | caller `metadata` copied before internal fields | Yes | VERIFIED |
| `_sanitize_key()` | `key` -> `safe_key`/digest | caller key and deterministic xxhash digest | Yes | VERIFIED |
| `UnifiedCache.get()` | `entry` | `metadata_backend.get_entry(cache_key)` reused for expiry | Yes | VERIFIED |
| `SqliteBackend.close()` | connection lifecycle | `self.engine.connect()` then `PRAGMA optimize` and `dispose()` | Yes | VERIFIED |
| `S3BlobBackend.delete_namespace_blobs()` | `Deleted`/`Errors` | boto3 `delete_objects` response | Yes | VERIFIED |
| Package root exports | `UnifiedCache`, `cacheness`, `__version__` | runtime imports and static metadata | Yes | VERIFIED |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| Version metadata and root import smoke | `uv run python -c "import pathlib, tomllib, cacheness; from cacheness import UnifiedCache, cacheness as alias; ..."` | Printed `0.12.0 UnifiedCache` | PASS |
| Representative Phase 32 regression nodes | `uv run pytest -o addopts='' ... -q --ignore=tests/test_tensorflow_handler.py` | `7 passed in 4.48s` | PASS |
| Final release checkpoint | `uv run pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py` | Orchestrator evidence: `1857 passed, 125 skipped, 27 warnings in 77.77s` | PASS |
| Schema drift gate | Orchestrator drift gate | `drift_detected: false`, `blocking: false` | PASS |

### Probe Execution

| Probe | Command | Result | Status |
|---|---|---|---|
| Conventional shell probes | `rg --files scripts | rg "probe-.*\.sh$"` | No probes found | SKIPPED |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|---|---|---|---|---|
| POL-01 | 32-01 | Caller metadata dictionaries passed to `BlobStore.put()` remain unchanged. | SATISFIED | Source copy and regression test verified. |
| POL-02 | 32-02 | Sanitized keys cannot silently collide. | SATISFIED | Stable digest suffix and collision tests verified. |
| POL-03 | 32-03 | Hot-path `get()` avoids redundant metadata reads. | SATISFIED | Entry reuse and one-read test verified. |
| POL-04 | 32-07 | Package version metadata matches release line. | SATISFIED | pyproject/runtime/lock alignment and smoke verified. |
| POL-05 | 32-04 | Absolute blob IDs are rejected before path construction. | SATISFIED | Source guard and backend-boundary test verified. |
| POL-06 | 32-05 | SQLite PRAGMA behavior occurs in correct lifecycle location. | SATISFIED | Close-time optimize and lifecycle tests verified. |
| POL-07 | 32-06 | S3 namespace deletion reports failures. | SATISFIED | Tuple counts, failed-key logging, and direct call-site update verified. |
| POL-08 | 32-08 | Root `UnifiedCache` import works. | SATISFIED | Root export, alias preservation, `__all__`, smoke, and regression verified. |

No orphaned Phase 32 requirements were found in `.planning/REQUIREMENTS.md`; POL-01 through POL-08 are all mapped to Phase 32.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---|---|---|---|
| Touched Phase 32 files | n/a | Debt markers / placeholders / empty user-facing implementations | None | Scan found no `TBD`, `FIXME`, `XXX`, `TODO`, `HACK`, placeholder text, or console-only implementations in Phase 32 change paths. Routine helper `return {}` / `return []` matches were existing non-stub backend helpers. |

### Human Verification Required

None. Phase 32 is backend/package behavior with automated and smoke verification.

### Residual Risks

- `ROADMAP.md` and `STATE.md` still contain stale wording saying the final full-suite checkpoint has non-POL-08 failures pending follow-up. Code, summaries, and commits show those failures were fixed in `4d6e0d1`, documented in `e88219e`, and the final checkpoint passed. This is planning metadata staleness, not a Phase 32 implementation blocker.
- Pre-existing `ty` diagnostics and unrelated dirty files are documented in `deferred-items.md` and current `git status`. They do not invalidate POL-01..POL-08 or the final full-suite pass.
- Existing dirty user work in `.planning/config.json` and unrelated tests was not reverted or modified by this verification.

### Gaps Summary

No blocking gaps found. Phase 32 delivers the promised low-risk release-polish fixes, has targeted regression coverage, and has final full-suite pass evidence.

---

_Verified: 2026-06-15T20:59:12Z_
_Verifier: the agent (gsd-verifier)_
