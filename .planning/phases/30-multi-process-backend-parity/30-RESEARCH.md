# Phase 30: Multi-Process & Backend Parity - Research

**Researched:** 2026-06-14
**Domain:** Python cache storage reliability, local filesystem blob writes, SQLite/PostgreSQL metadata parity, rollback safety, integrity enumeration
**Confidence:** HIGH for repo behavior and phase scope; MEDIUM for live environment availability because uv execution is blocked in this sandbox

<user_constraints>
## User Constraints (from CONTEXT.md)

All items in this section are copied from `.planning/phases/30-multi-process-backend-parity/30-CONTEXT.md`; treat them as locked planning input. [VERIFIED: .planning/phases/30-multi-process-backend-parity/30-CONTEXT.md]

### Locked Decisions

#### Multi-Process Blob Writes

- **D-01:** Implement TASK-9 for `FilesystemBlobBackend.write_blob`: deterministic `<final>.tmp` temp paths must be replaced with unique temp file names.
- **D-02:** Unique temp files should use a `tempfile.mkstemp(dir=blob_path.parent, suffix=".tmp")` style approach, write through the returned file descriptor, and publish with `os.replace(tmp, blob_path)`.
- **D-03:** Failed writes must clean up their unique temp file in an exception path without deleting the previously committed final blob.
- **D-04:** PAR-01 verification must prove repeated same-blob writes leave the final content equal to the last successful write. If practical, include a concurrency-style regression; at minimum, cover the sequential same-blob overwrite acceptance from TASK-9.
- **D-05:** Reuse existing JSON backend atomic-save patterns as a reference for unique temp file behavior, but do not change JSON metadata save semantics in this phase unless a tiny shared helper is clearly lower risk.

#### SQLite/PostgreSQL User Metadata Parity

- **D-06:** Implement TASK-10 for SQLite `put_entry` and PostgreSQL `_upsert_entry`: leftover nested `metadata` keys must not be silently discarded after known fields are popped.
- **D-07:** If leftover user metadata exists and `metadata_dict_value` is absent, serialize the leftovers into `metadata_dict` using the project's JSON utility patterns.
- **D-08:** If `metadata_dict_value` already exists, merge leftover user metadata into it while preserving core-provided `metadata_dict_value` keys on conflicts.
- **D-09:** SQLite and PostgreSQL read paths must expose deserialized `metadata_dict` values in the nested `metadata` dict in the same observable shape JSON provides.
- **D-10:** PAR-02 verification must prove `BlobStore.put(key, data, metadata={"experiment": "x42"})`, `get_metadata(key)["experiment"] == "x42"`, and `list_keys(metadata_filter={"experiment": "x42"}) == [key]` across JSON and SQLite. PostgreSQL coverage should be planned explicitly behind the existing availability/skip gates.

#### Non-Destructive Same-Key Overwrite Failure

- **D-11:** Implement TASK-11 for both cache mode `put()` and storage mode `_storage_mode_put()`. The storage-mode variant is mandatory because R7 is durable-data loss there.
- **D-12:** A failed same-key overwrite must leave the previous committed value readable and its blob file present.
- **D-13:** Use the minimal-risk snapshot strategy from `docs/CODE_REVIEW_ACTIONS.md`: before writing a new blob that resolves to the same local path as an existing entry, move the old blob to `<path>.prev`.
- **D-14:** On successful metadata commit and cleanup commit, delete `<path>.prev`; on rollback, restore `<path>.prev` to the original path.
- **D-15:** Extend `_PutCleanup` with explicit previous-blob tracking and rollback restoration instead of scattering ad hoc cleanup across `put()` and `_storage_mode_put()`.
- **D-16:** Skip previous-blob snapshot/restore handling for remote URI paths and inline entries; this phase's approved TASK-11 fix is for same-local-path overwrite rollback.
- **D-17:** PAR-03 verification must cover cache mode and `storage_mode=True`: put key with value A, force the next metadata write to raise, attempt value B, assert the exception occurs, then assert `get(key)` returns A and the old blob still exists.

#### Backend-Driven Blob Enumeration

- **D-18:** Implement TASK-12 in `FilesystemBlobBackend.list_blobs`: replace hardcoded extension whitelist enumeration with directory enumeration under namespace blob directories.
- **D-19:** Enumeration must include custom-handler files and inline fallback extensions such as `.custom` or `.bin`.
- **D-20:** Enumeration must exclude reserved files and directories, including `.intents/`, metadata JSON/DB files, signing/key files, and temp files such as `*.tmp`.
- **D-21:** Prefer a shared `RESERVED_FILE_PATTERNS` or similar module-level constant if it reduces duplication with Phase 28 clear-all cleanup rules, but keep the Phase 30 change scoped to enumeration/integrity behavior.
- **D-22:** PAR-04 verification must create a fake custom-extension blob in a namespace directory and assert `verify_integrity()` reports it as orphaned; repair behavior may be included if already covered by the existing integrity test pattern.

#### Scope and Risk Fences

- **D-23:** Do not re-open Phase 29 TTL/eviction behavior except for test compatibility with Phase 30 changes. TTL-01 through TTL-04 are already verified complete.
- **D-24:** Do not implement Phase 31 security/storage-mode policy decisions in Phase 30. The storage-mode destructive API policy, fsync/durability contract, signature downgrade policy, encryption read path, and key rotation work remain Phase 31 scope.
- **D-25:** Do not implement Phase 32 small polish items in Phase 30, including metadata dict mutation, sanitized key collision hardening, hot-path double-read cleanup, package version metadata, absolute blob id rejection, SQLite PRAGMA lifecycle, S3 delete failure reporting, or root `UnifiedCache` re-export.
- **D-26:** Use verify-first checks for each TASK-9 through TASK-12 item where the current bug is readily reproducible.
- **D-27:** Plans should keep TASK-9 through TASK-12 traceable. Preferred granularity is four atomic plans unless the planner finds a strong reason to combine adjacent low-risk test-only or helper-only work.
- **D-28:** Use `uv` for all commands. On Windows, always include `--ignore=tests/test_tensorflow_handler.py`.
- **D-29:** For changed Python files, quality gates are `ruff format`, `ruff check --fix`, `ruff check`, and `ty check` on touched files. Existing baseline `ty` diagnostics should be documented rather than hidden if unrelated.
- **D-30:** Preserve existing PostgreSQL Docker/service availability behavior. Plan PostgreSQL parity tests explicitly, but do not require a local PostgreSQL service when the existing test suite would skip it.

### the agent's Discretion

- Whether TASK-9 should add only sequential same-blob regression coverage or a true multi-thread/multi-process stress test, provided the deterministic temp-name race is prevented in source.
- Whether TASK-10 user metadata parity tests live in `tests/test_blob_store.py`, `tests/test_backend_parity.py`, or backend-specific files, provided JSON and SQLite behavior are compared directly and PostgreSQL semantics are represented.
- Whether the previous-blob restore helper for TASK-11 lives in `_PutCleanup`, a small adjacent helper, or existing put-path code, provided both cache mode and storage mode use the same rollback semantics.
- Whether TASK-12 extracts reserved-file matching into a shared constant now or leaves a narrowly-scoped exclusion helper in `blob_backends.py`.

### Deferred Ideas (OUT OF SCOPE)

- Phase 31: SEC-01 through SEC-04 and STRG-01 through STRG-03, including storage-mode destructive API policy, fsync/durability contract, signature downgrade hardening, encryption read path, and two-phase key rotation.
- Phase 32: POL-01 through POL-08 release polish and small fixes.
- Phase 28/29 work is complete or separately tracked; do not fold silent-data-loss or TTL/eviction consistency work into Phase 30 except where directly required for PAR-01 through PAR-04.
- True S3 integration changes are not required for Phase 30 unless needed by existing blob enumeration abstractions. Remote deletion behavior was Phase 29 scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PAR-01 | User can write blobs from concurrent or repeated writers without deterministic temp-file collisions corrupting the final blob. | Use unique same-directory temp files in `FilesystemBlobBackend.write_blob` and `write_blob_stream`, publish with `os.replace`, and regression-test repeated same-blob writes. [VERIFIED: src/cacheness/storage/backends/blob_backends.py; CITED: https://docs.python.org/3.12/library/tempfile.html#tempfile.mkstemp; CITED: https://docs.python.org/3.12/library/os.html#os.replace] |
| PAR-02 | User can store and filter custom user metadata with SQLite and PostgreSQL backends the same way JSON already supports it. | Merge leftover nested metadata into `metadata_dict` in SQLite `put_entry` and PostgreSQL `_upsert_entry`, then expose it through nested metadata and list/filter APIs. [VERIFIED: src/cacheness/metadata/sqlite_backend.py; VERIFIED: src/cacheness/storage/backends/postgresql_backend.py; VERIFIED: tests/test_backend_parity.py] |
| PAR-03 | User can retry or fail a same-key overwrite without losing the previous committed value in cache mode or storage mode. | Extend `_PutCleanup` to track a previous local blob snapshot and restore it on rollback in both `UnifiedCache.put` and `_storage_mode_put`. [VERIFIED: src/cacheness/core.py; VERIFIED: src/cacheness/_storage_mode_mixin.py; VERIFIED: src/cacheness/_put_cleanup.py] |
| PAR-04 | User can run integrity checks and namespace cleanup over all backend-visible blobs, including custom handler extensions and inline fallback files. | Replace extension whitelist enumeration with namespace directory enumeration that excludes reserved/temp files; `verify_integrity()` already consumes `blob_backend.list_blobs()`. [VERIFIED: src/cacheness/storage/backends/blob_backends.py; VERIFIED: src/cacheness/storage/blob_store.py; VERIFIED: tests/test_cache_integrity_verification.py] |
</phase_requirements>

## Summary

Phase 30 is not a new subsystem; it is four contained reliability fixes on existing seams: filesystem blob publication, metadata backend serialization/readback, put rollback cleanup, and blob inventory enumeration. [VERIFIED: .planning/ROADMAP.md; VERIFIED: .planning/phases/30-multi-process-backend-parity/30-CONTEXT.md] The highest planning risk is TASK-11 because it crosses cache mode, storage mode, old-blob cleanup, write-intent cleanup, inline blob handling, and remote URI exclusions. [VERIFIED: src/cacheness/core.py; VERIFIED: src/cacheness/_storage_mode_mixin.py; VERIFIED: src/cacheness/_put_cleanup.py]

TASK-9 and TASK-12 are likely single-file changes plus tests. [VERIFIED: src/cacheness/storage/backends/blob_backends.py; VERIFIED: tests/test_blob_namespace.py; VERIFIED: tests/test_cache_integrity_verification.py] TASK-10 needs coordinated SQLite and PostgreSQL changes, but PostgreSQL execution must stay behind existing availability skips. [VERIFIED: tests/test_backend_parity.py; VERIFIED: tests/test_postgresql_backend.py] TASK-11 should be planned after TASK-9 because unique temp files prevent temp-name collisions, while previous-blob snapshot/restore prevents rollback from deleting the old committed value. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md; VERIFIED: src/cacheness/core.py]

**Primary recommendation:** Plan four atomic tasks in TASK-9 through TASK-12 order, with TASK-11 receiving the most design and regression-test budget. [VERIFIED: .planning/phases/30-multi-process-backend-parity/30-CONTEXT.md; VERIFIED: docs/CODE_REVIEW_ACTIONS.md]

## Project Constraints (from AGENTS.md)

- `AGENTS.md` redirects all actionable project instructions to `.github/copilot-instructions.md`. [VERIFIED: AGENTS.md; VERIFIED: .github/copilot-instructions.md]
- Use `uv` for project commands; do not use direct `python`, `pip`, or `python -m pytest`. [VERIFIED: .github/copilot-instructions.md]
- On Windows, test commands must include `--ignore=tests/test_tensorflow_handler.py` because TensorFlow tests are documented as hanging. [VERIFIED: .github/copilot-instructions.md]
- Tests run with pytest-xdist by default via `-n auto --dist loadgroup`; Docker/PostgreSQL tests use `@pytest.mark.xdist_group("docker")`. [VERIFIED: pyproject.toml; VERIFIED: .github/copilot-instructions.md]
- Changed Python files require `ruff format`, `ruff check --fix`, `ruff check`, and `ty check` on touched files. [VERIFIED: .github/copilot-instructions.md]
- Imports in tests should use `from cacheness.core import UnifiedCache`, not `from cacheness import UnifiedCache`. [VERIFIED: .github/copilot-instructions.md]
- Project instructions require beads for ordinary work tracking, but the user explicitly instructed this GSD work to ignore beads/bd; planner should follow the user instruction for this phase. [VERIFIED: .github/copilot-instructions.md; VERIFIED: user prompt]
- Preserve the unrelated `.planning/config.json` newline change; do not stage or commit it. [VERIFIED: user prompt; VERIFIED: git status --short]

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|--------------|----------------|-----------|
| Unique temp blob writes | Storage backend | OS/filesystem | `FilesystemBlobBackend.write_blob` owns local file publication and already maps blob IDs to namespace/shard paths. [VERIFIED: src/cacheness/storage/backends/blob_backends.py] |
| User metadata parity | Metadata backend | BlobStore public API | SQLite and PostgreSQL decide how nested metadata maps to dedicated columns and `metadata_dict`; `BlobStore` exposes `get_metadata` and metadata filtering. [VERIFIED: src/cacheness/metadata/sqlite_backend.py; VERIFIED: src/cacheness/storage/backends/postgresql_backend.py; VERIFIED: src/cacheness/storage/blob_store.py] |
| Non-destructive overwrite rollback | Core write orchestration | `_PutCleanup` helper and storage backend | `UnifiedCache.put` and `_storage_mode_put` orchestrate blob write then metadata commit; `_PutCleanup` is the existing rollback home. [VERIFIED: src/cacheness/core.py; VERIFIED: src/cacheness/_storage_mode_mixin.py; VERIFIED: src/cacheness/_put_cleanup.py] |
| Blob enumeration for integrity | Blob backend | BlobStore integrity verifier | `BlobStore.verify_integrity` delegates blob inventory to `blob_backend.list_blobs`, so enumeration belongs in the backend. [VERIFIED: src/cacheness/storage/blob_store.py; VERIFIED: src/cacheness/storage/backends/blob_backends.py] |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python stdlib `tempfile` and `os` | Python 3.12 target in Ruff; project supports Python >=3.11 | Race-free temp file creation and atomic final replacement for filesystem blobs | `tempfile.mkstemp` creates the file immediately and returns an fd; `os.replace` replaces the destination atomically when source and destination are on the same filesystem. [VERIFIED: pyproject.toml; CITED: https://docs.python.org/3.12/library/tempfile.html#tempfile.mkstemp; CITED: https://docs.python.org/3.12/library/os.html#os.replace] |
| SQLAlchemy | 2.0.43 in `uv.lock`, >=2.0.0 in project metadata | SQLite/PostgreSQL ORM/Core access and JSONB columns | Existing metadata backends already use SQLAlchemy sessions, Core `select`, `text`, and PostgreSQL JSONB handling. [VERIFIED: uv.lock; VERIFIED: pyproject.toml; VERIFIED: src/cacheness/metadata/sqlite_backend.py; VERIFIED: src/cacheness/storage/backends/postgresql_backend.py] |
| Project `json_utils` | local module | Consistent JSON serialization/deserialization for metadata_dict | SQLite already imports `json_dumps`/`json_loads`; PostgreSQL has fallback helpers and `_ensure_jsonb_value`. [VERIFIED: src/cacheness/json_utils.py; VERIFIED: src/cacheness/metadata/sqlite_backend.py; VERIFIED: src/cacheness/storage/backends/postgresql_backend.py] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | 8.4.1 in `uv.lock`, >=8.4.1 in project metadata | Regression and parity tests | Use existing file-level tests plus `monkeypatch` or `unittest.mock.patch.object` to force metadata failures. [VERIFIED: uv.lock; VERIFIED: pyproject.toml; VERIFIED: tests/test_fault_injection.py; CITED: https://docs.pytest.org/en/stable/how-to/monkeypatch.html] |
| pytest-xdist | 3.8.0 in `uv.lock`, >=3.8.0 in project metadata | Parallel test execution with Docker grouping | Keep PostgreSQL tests in xdist docker group and skip when PG URL/backend is unavailable. [VERIFIED: uv.lock; VERIFIED: pyproject.toml; VERIFIED: tests/test_backend_parity.py] |
| ruff and ty | ruff 0.12.9, ty 0.0.15 in `uv.lock` | Formatting, lint, type checks | Run on touched Python files per project instructions. [VERIFIED: uv.lock; VERIFIED: .github/copilot-instructions.md] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `tempfile.mkstemp` plus `os.replace` | `NamedTemporaryFile(delete=False)` | `mkstemp` matches the locked phase decision and returns an OS fd that can be written and closed before replacement; `NamedTemporaryFile` has Windows reopen/delete caveats. [VERIFIED: .planning/phases/30-multi-process-backend-parity/30-CONTEXT.md; CITED: https://docs.python.org/3.12/library/tempfile.html] |
| `_PutCleanup` extension | Ad hoc rollback code in `put()` and `_storage_mode_put()` | `_PutCleanup` is already the shared rollback guard, and D-15 requires previous-blob tracking there. [VERIFIED: src/cacheness/_put_cleanup.py; VERIFIED: .planning/phases/30-multi-process-backend-parity/30-CONTEXT.md] |
| Backend directory enumeration | Handler extension registry | D-18 through D-20 require backend-driven enumeration, and current integrity code already consumes backend output. [VERIFIED: .planning/phases/30-multi-process-backend-parity/30-CONTEXT.md; VERIFIED: src/cacheness/storage/blob_store.py] |

**Installation:** No new packages are required for Phase 30. [VERIFIED: pyproject.toml; VERIFIED: .planning/phases/30-multi-process-backend-parity/30-CONTEXT.md]

## Package Legitimacy Audit

This phase should not install external packages. [VERIFIED: pyproject.toml; VERIFIED: .planning/phases/30-multi-process-backend-parity/30-CONTEXT.md]

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| None | N/A | N/A | N/A | N/A | N/A | No install planned. [VERIFIED: pyproject.toml] |

**Packages removed due to [SLOP] verdict:** none. [VERIFIED: pyproject.toml]
**Packages flagged as suspicious [SUS]:** none. [VERIFIED: pyproject.toml]

## Architecture Patterns

### System Architecture Diagram

```text
Caller put()/BlobStore.put()
  -> key/path planning
  -> handler serialization
  -> FilesystemBlobBackend writes unique temp in namespace dir
  -> os.replace publishes final blob
  -> metadata backend put_entry/_upsert_entry
       -> known fields into columns
       -> leftover user metadata merged into metadata_dict
  -> commit cleanup removes previous snapshot

Failure before metadata commit
  -> _PutCleanup.rollback()
       -> delete newly written local/remote blob
       -> restore previous local snapshot when present
       -> leave committed metadata readable

verify_integrity()/cleanup inventory
  -> blob_backend.list_blobs()
       -> enumerate namespace directory recursively
       -> exclude metadata, intent, signing, and temp files
  -> compare backend-visible blobs with metadata entry paths
```

### Recommended Project Structure

```text
src/cacheness/storage/backends/blob_backends.py   # TASK-9 unique temp writes and TASK-12 enumeration
src/cacheness/metadata/sqlite_backend.py          # TASK-10 SQLite leftover metadata merge/readback
src/cacheness/storage/backends/postgresql_backend.py # TASK-10 PostgreSQL metadata merge/readback
src/cacheness/_put_cleanup.py                     # TASK-11 previous blob snapshot rollback
src/cacheness/core.py                             # TASK-11 cache-mode integration
src/cacheness/_storage_mode_mixin.py              # TASK-11 storage-mode integration
tests/test_blob_namespace.py                      # TASK-9 backend write regression
tests/test_backend_parity.py or test_blob_store.py # TASK-10 JSON/SQLite/PG parity
tests/test_fault_injection.py and test_storage_mode.py # TASK-11 rollback regressions
tests/test_cache_integrity_verification.py        # TASK-12 orphan enumeration regression
```

### Pattern 1: Unique Same-Directory Temp Publish

**What:** Create a temp file in `blob_path.parent`, write via the fd, close it, then publish with `os.replace`. [CITED: https://docs.python.org/3.12/library/tempfile.html#tempfile.mkstemp; CITED: https://docs.python.org/3.12/library/os.html#os.replace]

**When to use:** Local filesystem blob writes and stream writes that publish to a deterministic final blob path. [VERIFIED: src/cacheness/storage/backends/blob_backends.py]

**Example:**

```python
# Source: Python stdlib docs plus existing FilesystemBlobBackend shape.
fd, tmp_name = tempfile.mkstemp(dir=blob_path.parent, suffix=".tmp")
tmp_path = Path(tmp_name)
try:
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp_path, blob_path)
except Exception:
    try:
        tmp_path.unlink()
    except FileNotFoundError:
        pass
    raise
```

### Pattern 2: Metadata Leftovers Merge

**What:** After popping known backend fields from nested `metadata`, merge remaining user keys into `metadata_dict`, preserving existing `metadata_dict` keys on conflict. [VERIFIED: .planning/phases/30-multi-process-backend-parity/30-CONTEXT.md; VERIFIED: src/cacheness/metadata/sqlite_backend.py]

**When to use:** SQLite `put_entry` and PostgreSQL `_upsert_entry` after all dedicated-column fields are extracted. [VERIFIED: src/cacheness/metadata/sqlite_backend.py; VERIFIED: src/cacheness/storage/backends/postgresql_backend.py]

**Example:**

```python
# Source: Phase D-07/D-08 and json_utils patterns.
leftovers = metadata.copy()
existing = json_loads(metadata_dict_value) if isinstance(metadata_dict_value, str) else (metadata_dict_value or {})
merged = {**leftovers, **existing}
metadata_dict_value = json_dumps(merged)  # SQLite
# PostgreSQL: jsonb_metadata = _ensure_jsonb_value(merged)
```

### Pattern 3: Previous Blob Snapshot in Rollback Guard

**What:** Move the old local blob to `<path>.prev` before same-path overwrite; rollback restores it, commit deletes the snapshot. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md; VERIFIED: .planning/phases/30-multi-process-backend-parity/30-CONTEXT.md]

**When to use:** Only for same-local-path overwrites in cache mode and storage mode; skip inline entries and remote URIs. [VERIFIED: .planning/phases/30-multi-process-backend-parity/30-CONTEXT.md; VERIFIED: src/cacheness/core.py; VERIFIED: src/cacheness/_storage_mode_mixin.py]

### Anti-Patterns to Avoid

- **Deterministic `<final>.tmp`:** Two writers to the same blob share a temp path, which is the R11 race this phase fixes. [VERIFIED: src/cacheness/storage/backends/blob_backends.py; VERIFIED: docs/CODE_REVIEW_FINDINGS.md]
- **Discarding leftover metadata after `.pop()`:** SQLite and PostgreSQL currently pop known fields and leave custom keys unused, causing JSON/SQLite/PG parity failure. [VERIFIED: src/cacheness/metadata/sqlite_backend.py; VERIFIED: src/cacheness/storage/backends/postgresql_backend.py; VERIFIED: docs/CODE_REVIEW_FINDINGS.md]
- **Rollback deleting the final path without restoring old value:** Current `_PutCleanup.rollback()` unlinks `blob_path`, which destroys the old blob after same-path overwrite failure. [VERIFIED: src/cacheness/_put_cleanup.py; VERIFIED: docs/CODE_REVIEW_FINDINGS.md]
- **Extension whitelist inventory:** Current `list_blobs()` only finds known extensions, so custom handler files are invisible to integrity repair. [VERIFIED: src/cacheness/storage/backends/blob_backends.py; VERIFIED: docs/CODE_REVIEW_FINDINGS.md]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Unique temp file naming | Random string concatenation or `mktemp` | `tempfile.mkstemp(dir=..., suffix=".tmp")` | Official docs state `mkstemp` creates the file immediately and avoids race conditions when the platform implements `O_EXCL`. [CITED: https://docs.python.org/3.12/library/tempfile.html#tempfile.mkstemp] |
| Final file publication | Manual copy/truncate into destination | `os.replace(tmp, final)` | It preserves the existing atomic replace discipline used elsewhere in the project. [CITED: https://docs.python.org/3.12/library/os.html#os.replace; VERIFIED: docs/CODE_REVIEW_FINDINGS.md] |
| JSON serialization for metadata | Ad hoc `json.dumps` imports in each backend | `cacheness.json_utils.dumps/loads` and PostgreSQL `_ensure_jsonb_value` | Project already centralizes JSON behavior and has backend-specific JSONB conversion. [VERIFIED: src/cacheness/json_utils.py; VERIFIED: src/cacheness/storage/backends/postgresql_backend.py] |
| Rollback orchestration | Separate try/except cleanup blocks in two put paths | Extend `_PutCleanup` | Existing cleanup helper already tracks new local blobs and remote blobs. [VERIFIED: src/cacheness/_put_cleanup.py] |

**Key insight:** The planner should preserve existing orchestration seams instead of introducing new abstractions; the current seams are correct, but they need more complete state tracking. [VERIFIED: src/cacheness/core.py; VERIFIED: src/cacheness/_storage_mode_mixin.py; VERIFIED: src/cacheness/_put_cleanup.py]

## Common Pitfalls

### Pitfall 1: Treating TASK-9 as Sufficient for TASK-11

**What goes wrong:** Unique temp files prevent temp-file collision but do not stop rollback from deleting the old final blob after the new blob has been published. [VERIFIED: src/cacheness/storage/backends/blob_backends.py; VERIFIED: src/cacheness/_put_cleanup.py]

**Why it happens:** `FilesystemBlobBackend.write_blob` publishes before metadata commit, and `_PutCleanup.rollback` deletes the final path. [VERIFIED: src/cacheness/storage/backends/blob_backends.py; VERIFIED: src/cacheness/core.py]

**How to avoid:** Plan TASK-11 as explicit old-blob snapshot/restore after TASK-9. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md]

### Pitfall 2: Fixing Cache Mode but Missing Storage Mode

**What goes wrong:** Storage mode has an almost identical `_storage_mode_put` flow and is higher risk because its contract is durable storage. [VERIFIED: src/cacheness/_storage_mode_mixin.py; VERIFIED: docs/CODE_REVIEW_FINDINGS.md]

**How to avoid:** Every TASK-11 implementation task must touch or audit both `core.py` and `_storage_mode_mixin.py`, and its test command must include `tests/test_storage_mode.py`. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md]

### Pitfall 3: Adding Metadata Write Parity Without Read/Filter Parity

**What goes wrong:** Storing leftovers in `metadata_dict` is not enough if `get_entry`, `list_entries`, or `BlobStore.list(metadata_filter=...)` do not expose/filter the same shape as JSON. [VERIFIED: src/cacheness/storage/blob_store.py; VERIFIED: src/cacheness/metadata/sqlite_backend.py; VERIFIED: src/cacheness/storage/backends/postgresql_backend.py]

**How to avoid:** Tests must use the public `BlobStore.put`, `get_metadata`, and list/filter API across JSON and SQLite, with PostgreSQL behind skip gates. [VERIFIED: .planning/phases/30-multi-process-backend-parity/30-CONTEXT.md; VERIFIED: tests/test_backend_parity.py]

### Pitfall 4: Enumerating Too Broadly

**What goes wrong:** Directory enumeration can accidentally include `.intents`, metadata DB/JSON files, signing keys, or temp files unless explicit exclusions are added. [VERIFIED: .planning/phases/30-multi-process-backend-parity/30-CONTEXT.md]

**How to avoid:** Keep enumeration rooted under the namespace blob directory and exclude reserved names/patterns and `*.tmp`. [VERIFIED: src/cacheness/storage/backends/blob_backends.py; VERIFIED: .planning/phases/30-multi-process-backend-parity/30-CONTEXT.md]

## Dependency Ordering and Task Map

| Task | Requirement | Context Decisions | Plan Dependency | Notes |
|------|-------------|-------------------|-----------------|-------|
| TASK-9 unique temp names | PAR-01 | D-01 through D-05 | First | Low blast radius; update both `write_blob` and likely `write_blob_stream` because both use deterministic temp paths. [VERIFIED: src/cacheness/storage/backends/blob_backends.py] |
| TASK-10 metadata parity | PAR-02 | D-06 through D-10, D-30 | Can run after TASK-9 or in parallel in a separate branch | Tests should compare JSON and SQLite directly; PostgreSQL must skip when unavailable. [VERIFIED: tests/test_backend_parity.py] |
| TASK-11 non-destructive overwrite | PAR-03 | D-11 through D-17, D-24 | After TASK-9 preferred | Highest risk; shared rollback state must cover both put paths and avoid remote/inline scope creep. [VERIFIED: src/cacheness/core.py; VERIFIED: src/cacheness/_storage_mode_mixin.py] |
| TASK-12 blob enumeration | PAR-04 | D-18 through D-22 | Independent, but coordinate exclusions with TASK-9 temp suffix | Do after or with TASK-9 so temp suffix exclusion matches implementation. [VERIFIED: src/cacheness/storage/backends/blob_backends.py] |
| Scope fences | All | D-23 through D-30 | Always | Do not pull in Phase 31 security/storage policy or Phase 32 polish. [VERIFIED: .planning/phases/30-multi-process-backend-parity/30-CONTEXT.md] |

## Code Examples

### Cache-Mode Rollback Test Shape

```python
# Source: tests/test_fault_injection.py uses patch.object on metadata_backend.put_entry.
cache.put("value A", cache_key="same")
with patch.object(cache.metadata_backend, "put_entry", side_effect=RuntimeError("boom")):
    with pytest.raises(RuntimeError):
        cache.put("value B", cache_key="same")
assert cache.get(cache_key="same") == "value A"
```

### Integrity Enumeration Regression

```python
# Source: tests/test_cache_integrity_verification.py orphan tests.
cache = _make_cache(tmp_path)
ns_dir = tmp_path / "default"
ns_dir.mkdir(exist_ok=True)
orphan = ns_dir / "orphan.custom"
orphan.write_bytes(b"custom data")
report = cache.verify_integrity()
assert os.path.normpath(str(orphan)) in report["orphaned_blobs"]
```

### PostgreSQL Availability Gate

```python
# Source: tests/test_backend_parity.py
def _skip_if_pg_unavailable(backend):
    if backend == "postgresql":
        if not _HAS_PG or not _get_pg_url():
            pytest.skip("PostgreSQL not available")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Deterministic temp path `<final>.tmp` | Unique temp file per write in destination directory | Phase 30 planned | Prevents writers from sharing the same temp file. [VERIFIED: docs/CODE_REVIEW_FINDINGS.md; CITED: https://docs.python.org/3.12/library/tempfile.html#tempfile.mkstemp] |
| SQLite `INSERT OR REPLACE` access reset | `ON CONFLICT DO UPDATE` preserving access count is already present | Phase 29 completed before this research | Planner should not redo R9; keep Phase 30 focused on leftover user metadata. [VERIFIED: src/cacheness/metadata/sqlite_backend.py; VERIFIED: .planning/STATE.md] |
| Extension whitelist blob inventory | Backend directory enumeration with exclusions | Phase 30 planned | Integrity checks will see custom handler and inline fallback files. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md] |

**Deprecated/outdated:**
- `tempfile.mktemp`: official docs deprecate it because another process can create the returned filename before use. [CITED: https://docs.python.org/3.12/library/tempfile.html]
- Hardcoded extension whitelist for integrity inventory: it misses custom handler extensions by construction. [VERIFIED: src/cacheness/storage/backends/blob_backends.py; VERIFIED: docs/CODE_REVIEW_FINDINGS.md]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | If a true multi-process stress test proves flaky on Windows, the minimum acceptable PAR-01 regression is sequential repeated same-blob writes plus source inspection that deterministic temp names are gone. [ASSUMED] | Validation Architecture | Planner may over- or under-invest in concurrency stress. |
| A2 | PostgreSQL live service will usually be unavailable locally and should remain skip-gated. [ASSUMED] | Environment Availability | Planner might schedule PG live validation when the environment cannot satisfy it. |

## Open Questions (RESOLVED)

1. **Should TASK-9 include a true process-level race test?**
   - What we know: D-04 says include concurrency-style regression if practical, but sequential same-blob overwrite is minimum acceptable. [VERIFIED: .planning/phases/30-multi-process-backend-parity/30-CONTEXT.md]
   - What's unclear: Whether Windows process scheduling and xdist make a process-level race test stable enough for the suite. [ASSUMED]
   - RESOLVED: TASK-9 will use a deterministic sequential same-blob write regression plus a direct failed-write cleanup regression. Phase 30 planning does not require a true process-level race test because D-04 explicitly allows the sequential same-blob acceptance path when a concurrency-style regression is not practical. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md; VERIFIED: .planning/phases/30-multi-process-backend-parity/30-CONTEXT.md]

2. **Where should TASK-10 public parity tests live?**
   - What we know: Context allows `tests/test_blob_store.py`, `tests/test_backend_parity.py`, or backend-specific files. [VERIFIED: .planning/phases/30-multi-process-backend-parity/30-CONTEXT.md]
   - What's unclear: Which location will keep the smallest fixtures and cleanest PostgreSQL skip behavior. [ASSUMED]
   - RESOLVED: TASK-10 public parity coverage belongs in `tests/test_backend_parity.py`, with direct JSON and SQLite coverage and PostgreSQL coverage behind the existing availability skip gates. Do not require a local PostgreSQL service for Phase 30 planning. [VERIFIED: tests/test_backend_parity.py; VERIFIED: .planning/phases/30-multi-process-backend-parity/30-CONTEXT.md]

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| uv | All commands | Yes | 0.11.19 | None. [VERIFIED: uv --version] |
| Python via uv | Test execution | Blocked in sandbox | Project requires >=3.11 and targets py312; live `uv run --python 3.12` failed with access denied. | Planner should use project command pattern with workspace-local uv cache if needed. [VERIFIED: pyproject.toml; VERIFIED: shell probe] |
| pytest | Validation | Declared/locked, live probe blocked | 8.4.1 in `uv.lock` | Use `uv run pytest ...` once interpreter access works. [VERIFIED: uv.lock; VERIFIED: pyproject.toml] |
| PostgreSQL service | TASK-10 PostgreSQL parity | Not confirmed | Existing tests require URL/backend availability | Existing skip gates via `_skip_if_pg_unavailable`; do not require local PG. [VERIFIED: tests/test_backend_parity.py; VERIFIED: tests/test_postgresql_backend.py] |
| ruff/ty | Quality gates | Declared/locked, live probe blocked | ruff 0.12.9, ty 0.0.15 in `uv.lock` | Document unrelated baseline diagnostics if any. [VERIFIED: uv.lock; VERIFIED: .github/copilot-instructions.md] |

**Missing dependencies with no fallback:**
- None for planning. [VERIFIED: pyproject.toml]

**Missing dependencies with fallback:**
- Live PostgreSQL service may be absent; use existing skip-gated PostgreSQL tests. [VERIFIED: tests/test_backend_parity.py; VERIFIED: .planning/phases/30-multi-process-backend-parity/30-CONTEXT.md]
- Live `uv run` execution is blocked in this sandbox by interpreter access errors; planner should still emit standard `uv` commands because project instructions require them. [VERIFIED: shell probe; VERIFIED: .github/copilot-instructions.md]

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.4.1 with pytest-xdist 3.8.0 from `uv.lock`. [VERIFIED: uv.lock] |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]`. [VERIFIED: pyproject.toml] |
| Quick run command | `uv run pytest tests/test_blob_store.py tests/test_blob_namespace.py tests/test_backend_parity.py tests/test_fault_injection.py tests/test_cache_integrity_verification.py tests/test_storage_mode.py -x -q --ignore=tests/test_tensorflow_handler.py` [VERIFIED: docs/CODE_REVIEW_ACTIONS.md; VERIFIED: .github/copilot-instructions.md] |
| Full suite command | `uv run pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py` [VERIFIED: .github/copilot-instructions.md] |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| PAR-01 | Repeated same-blob writes use unique temp files and final content equals last successful write. | unit/regression | `uv run pytest tests/test_blob_namespace.py tests/test_blob_store.py -x -q --ignore=tests/test_tensorflow_handler.py` | Yes. [VERIFIED: tests/test_blob_namespace.py; VERIFIED: tests/test_blob_store.py] |
| PAR-02 | `BlobStore.put(..., metadata={"experiment": "x42"})` round-trips and filters across JSON/SQLite, with PG skip-gated. | parity/integration | `uv run pytest tests/test_backend_parity.py tests/test_blob_store.py tests/test_metadata.py -x -q --ignore=tests/test_tensorflow_handler.py` | Yes. [VERIFIED: tests/test_backend_parity.py; VERIFIED: tests/test_blob_store.py; VERIFIED: tests/test_metadata.py] |
| PAR-03 | Failed same-key overwrite preserves previous value and blob in cache mode and storage mode. | fault-injection/regression | `uv run pytest tests/test_core.py tests/test_fault_injection.py tests/test_cache_integrity.py tests/test_storage_mode.py -x -q --ignore=tests/test_tensorflow_handler.py` | Yes. [VERIFIED: tests/test_core.py; VERIFIED: tests/test_fault_injection.py; VERIFIED: tests/test_storage_mode.py] |
| PAR-04 | `verify_integrity()` detects custom-extension orphan blobs and repair can delete them if included. | integrity/regression | `uv run pytest tests/test_cache_integrity_verification.py tests/test_blob_store.py -x -q --ignore=tests/test_tensorflow_handler.py` | Yes. [VERIFIED: tests/test_cache_integrity_verification.py; VERIFIED: tests/test_blob_store.py] |

### Sampling Rate

- **Per task commit:** Run the task-specific Tier-1 command from `docs/CODE_REVIEW_ACTIONS.md`. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md]
- **Per wave merge:** Run the combined Phase 30 quick command above. [VERIFIED: docs/CODE_REVIEW_ACTIONS.md]
- **Phase gate:** Run full suite once with TensorFlow ignored on Windows. [VERIFIED: .github/copilot-instructions.md]

### Wave 0 Gaps

- [ ] Add a custom-extension orphan regression to `tests/test_cache_integrity_verification.py` for PAR-04. [VERIFIED: tests/test_cache_integrity_verification.py]
- [ ] Add a JSON/SQLite/PG public BlobStore metadata parity regression to `tests/test_backend_parity.py` or `tests/test_blob_store.py` for PAR-02. [VERIFIED: tests/test_backend_parity.py; VERIFIED: .planning/phases/30-multi-process-backend-parity/30-CONTEXT.md]
- [ ] Add cache-mode and storage-mode same-key metadata-failure regressions for PAR-03. [VERIFIED: tests/test_fault_injection.py; VERIFIED: tests/test_storage_mode.py]
- [ ] Add same-blob repeated write regression for PAR-01. [VERIFIED: tests/test_blob_namespace.py]

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | No | Phase 30 does not add authentication flows. [VERIFIED: .planning/phases/30-multi-process-backend-parity/30-CONTEXT.md] |
| V3 Session Management | No | Phase 30 does not manage user sessions. [VERIFIED: .planning/phases/30-multi-process-backend-parity/30-CONTEXT.md] |
| V4 Access Control | No | Phase 30 does not add authorization boundaries. [VERIFIED: .planning/phases/30-multi-process-backend-parity/30-CONTEXT.md] |
| V5 Input Validation | Yes | Validate/exclude reserved filesystem artifacts during enumeration; preserve path resolution through existing backend path helpers. [VERIFIED: src/cacheness/storage/backends/blob_backends.py; VERIFIED: .planning/phases/30-multi-process-backend-parity/30-CONTEXT.md] |
| V6 Cryptography | No direct change | Phase 31 owns signature/encryption policy; Phase 30 must not alter crypto semantics. [VERIFIED: .planning/phases/30-multi-process-backend-parity/30-CONTEXT.md] |

### Known Threat Patterns for This Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Temp file collision during concurrent writes | Tampering | Race-free unique temp creation plus atomic replacement. [VERIFIED: docs/CODE_REVIEW_FINDINGS.md; CITED: https://docs.python.org/3.12/library/tempfile.html#tempfile.mkstemp] |
| Integrity inventory includes reserved state | Tampering/Denial of service | Root enumeration under namespace blob directory and exclude `.intents`, metadata, signing keys, and temp files. [VERIFIED: .planning/phases/30-multi-process-backend-parity/30-CONTEXT.md] |
| Metadata parity silently drops user fields | Repudiation/Integrity | Preserve user metadata in `metadata_dict` and test public filter/read behavior. [VERIFIED: docs/CODE_REVIEW_FINDINGS.md; VERIFIED: src/cacheness/metadata/sqlite_backend.py] |

## Sources

### Primary (HIGH confidence)

- `.planning/ROADMAP.md` - Phase 30 goal, requirements, success criteria, task mapping. [VERIFIED: .planning/ROADMAP.md]
- `.planning/REQUIREMENTS.md` - PAR-01 through PAR-04 wording. [VERIFIED: .planning/REQUIREMENTS.md]
- `.planning/phases/30-multi-process-backend-parity/30-CONTEXT.md` - decisions D-01 through D-30, discretion, deferred scope. [VERIFIED: .planning/phases/30-multi-process-backend-parity/30-CONTEXT.md]
- `docs/CODE_REVIEW_FINDINGS.md` and `docs/CODE_REVIEW_ACTIONS.md` - R7/R11/R12/U2 and TASK-9 through TASK-12 details. [VERIFIED: docs/CODE_REVIEW_FINDINGS.md; VERIFIED: docs/CODE_REVIEW_ACTIONS.md]
- Source files named in the phase prompt and inspected in this session. [VERIFIED: src/cacheness/storage/backends/blob_backends.py; VERIFIED: src/cacheness/storage/blob_store.py; VERIFIED: src/cacheness/metadata/sqlite_backend.py; VERIFIED: src/cacheness/storage/backends/postgresql_backend.py; VERIFIED: src/cacheness/core.py; VERIFIED: src/cacheness/_storage_mode_mixin.py; VERIFIED: src/cacheness/_put_cleanup.py; VERIFIED: src/cacheness/json_utils.py]
- Test files named in the phase prompt and inspected in this session. [VERIFIED: tests/test_blob_store.py; VERIFIED: tests/test_blob_namespace.py; VERIFIED: tests/test_backend_parity.py; VERIFIED: tests/test_fault_injection.py; VERIFIED: tests/test_cache_integrity_verification.py; VERIFIED: tests/test_storage_mode.py; VERIFIED: tests/test_postgresql_backend.py]

### Secondary (MEDIUM confidence)

- Python 3.12 docs for `tempfile.mkstemp` and `os.replace`. [CITED: https://docs.python.org/3.12/library/tempfile.html#tempfile.mkstemp; CITED: https://docs.python.org/3.12/library/os.html#os.replace]
- pytest monkeypatch docs. [CITED: https://docs.pytest.org/en/stable/how-to/monkeypatch.html]
- SQLAlchemy PostgreSQL JSON type docs. [CITED: https://docs.sqlalchemy.org/en/20/dialects/postgresql.html#json-types]

### Tertiary (LOW confidence)

- Assumption A1 about concurrency stress practicality on Windows. [ASSUMED]
- Assumption A2 about local PostgreSQL availability. [ASSUMED]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - project dependencies and source usage are local and verified; no new packages are recommended. [VERIFIED: pyproject.toml; VERIFIED: uv.lock]
- Architecture: HIGH - all four capabilities map to existing code seams inspected in this session. [VERIFIED: src/cacheness/storage/backends/blob_backends.py; VERIFIED: src/cacheness/core.py; VERIFIED: src/cacheness/_storage_mode_mixin.py]
- Pitfalls: HIGH - each pitfall is grounded in source or code review findings, except stress-test practicality which is explicitly assumed. [VERIFIED: docs/CODE_REVIEW_FINDINGS.md; ASSUMED]
- Environment availability: MEDIUM - project config is verified, but live `uv run` probes failed under sandbox access constraints. [VERIFIED: pyproject.toml; VERIFIED: shell probe]

**Research date:** 2026-06-14
**Valid until:** 2026-07-14 for codebase planning assumptions, or sooner if Phase 28/30 source changes land before planning.

## Research Tooling Notes

- `gsd-tools query research-plan --input ...` failed because the local bridge reported `Unknown command: research-plan`; this research therefore used direct source inspection plus official docs. [VERIFIED: shell probe]
- `gsd-tools query classify-confidence ...` failed because the local bridge reported `Unknown command: classify-confidence`; confidence labels are assigned from observed source provenance instead of the unavailable seam. [VERIFIED: shell probe]
- `uv run --python 3.12 ...` failed in this sandbox with interpreter access errors after the global uv cache path also failed; test commands are documented but not executed. [VERIFIED: shell probe]
