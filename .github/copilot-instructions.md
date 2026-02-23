# GitHub Copilot Instructions for Cacheness

## Project Overview

**Cacheness** is a Python disk caching library with pluggable metadata backends and handler-based serialization. Supports JSON/SQLite/PostgreSQL backends and type-aware handlers (DataFrames, NumPy, TensorFlow tensors, etc.). A standalone storage backend allows using Cacheness as a persistent key-value store with rich metadata capabilities (cache eviction disabled).

**What makes it special:** Unlike joblib.Memory or diskcache, Cacheness provides **type-aware serialization** (optimized parquet for DataFrames, blosc2 for NumPy), **pluggable metadata backends** (JSON for simple caches, SQLite for large caches, PostgreSQL for distributed), and **cryptographic signing** for cache integrity.

**Detailed docs:** See [`docs/README.md`](../docs/README.md) for comprehensive guides.
**Common issues:** See [`docs/TROUBLESHOOTING.md`](../docs/TROUBLESHOOTING.md) for troubleshooting guide.


## Golden Rules

These constraints apply to EVERY task. Violating any of them is a bug.

**Package manager:**
- **ALWAYS use `uv`** — never `python`, `pip`, or `python -m pytest` directly
- `uv run pytest ...` / `uv run python ...` / `uv sync` / `uv add <pkg>`

**Testing:**
- **Full suite:** `uv run pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py`
- **Sequential run:** `uv run pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py -p no:xdist` (disables parallel execution)
- **Windows:** Always add `--ignore=tests/test_tensorflow_handler.py` — TF tests hang
- **Baseline:** 1427 passed, 102 skipped, 0 failures (~48s parallel, ~237s sequential)
- **Parallel execution:** Enabled by default via `pytest-xdist` (`-n auto --dist loadgroup`). Tests sharing external resources (Docker PostgreSQL/S3) are grouped via `@pytest.mark.xdist_group("docker")`.
- **Incremental testing:** During development, run only targeted tests (see [Test Suite](#test-suite) for details). Full suite runs only once — right before push.

**Imports:**
- `from cacheness import UnifiedCache` does NOT work — it's exported as `cacheness`. Use `from cacheness.core import UnifiedCache` in tests.
- `# noqa: E402` on the closing paren of a multi-line import does NOT suppress the error. Collapse to a single-line import instead.

**Windows (beads):**
- Start daemon once per session: `uv run bd daemon start` (required after reboot/logout)

**Work tracking:**
- File a beads issue for ANY work, even small fixes
- Every code change goes through the Mandatory Workflow below
- **Critical issues** discovered during work (not related to the current issue) → file immediately (don't lose context)
- **Non-critical issues** discovered during work → note them, propose them when closing out the current issue


## Mandatory Workflow

**This is the step-by-step protocol. Follow it for every task.**

### Session Start Hygiene

```bash
git worktree list          # Check for orphaned worktrees from prior sessions
git worktree prune         # Clean up stale references
```

### Workflow A: Feature/Fix Work (uses worktree)

Use this for any code change, test addition, or issue-tracked work.

1. **File/find the issue** — `mcp_beads_ready` or `mcp_beads_show`
2. **Create worktree & claim issue:**
   ```bash
   git worktree add worktrees/beads-<hash> -b beads-<hash>-<desc> dev
   cd worktrees/beads-<hash>
   uv sync --all-groups
   ```
   - Mark issue in-progress: beads MCP `update`, or CLI `uv run bd update <id> --claim`
   - Create `WORKTREE.md` with issue details
3. **Do the work** — code, tests, docs updates
4. **Critical issues** discovered during work (not related to the current issue) → file immediately (don't lose context)
5. **Quality gates** (if .py files changed):
   ```powershell
   # PowerShell (Windows)
   $files = git diff --name-only --diff-filter=ACMR HEAD -- '*.py'
   uv run ruff format $files; uv run ruff check --fix $files   # Phase 1: auto-fix
   uv run ruff check $files; uv run ty check $files            # Phase 2: validate
   ```
   ```bash
   # bash (Unix/Mac)
   uv run ruff format $(git diff --name-only --diff-filter=ACMR HEAD -- '*.py')
   uv run ruff check --fix $(git diff --name-only --diff-filter=ACMR HEAD -- '*.py')
   uv run ruff check $(git diff --name-only --diff-filter=ACMR HEAD -- '*.py')
   uv run ty check $(git diff --name-only --diff-filter=ACMR HEAD -- '*.py')
   ```
6. **Incremental tests** (after each code change):
   ```bash
   # Tier 1 — directly affected tests only (~seconds)
   uv run pytest tests/test_blob_store.py tests/test_core.py -x -q --ignore=tests/test_tensorflow_handler.py
   # Tier 2 — add regression-risk tests if Tier 1 passes (~30-60s)
   uv run pytest tests/test_blob_store.py tests/test_core.py tests/test_cache_integrity.py -x -q --ignore=tests/test_tensorflow_handler.py
   ```
   Select test files based on what you changed — see [Test Suite](#test-suite).
7. **Full test suite** (once, right before push):
   `uv run pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py`
8. **Update docs** if changing public API
9. **Push feature branch:** `git push -u origin beads-<hash>-<desc>`
10. **Integrate to dev:**
   ```bash
   cd ../..                                          # Back to Cacheness/
   git checkout dev
   git pull origin dev && git merge beads-<hash>-<desc>
   # If fast-forward: skip tests (already passed in worktree)
   # If real merge: uv run pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py
   git push origin dev
   ```
11. **Cleanup:**
   ```bash
   git worktree remove worktrees/beads-<hash> --force   # Always needs --force (.venv/pycache)
   git branch -d beads-<hash>-<desc>
   git push origin --delete beads-<hash>-<desc>
   ```
   - Close issue: beads MCP `close`, or CLI `uv run bd close <id> --force` (needed for child issues of epics)
12. **Verify** — `git status` confirms "up to date with origin"

### Workflow B: Direct-to-dev (no worktree)

Use this for docs-only changes, config tweaks, or trivial fixes that don't need isolation.

1. Ensure you're on `dev` and up to date: `git checkout dev && git pull origin dev`
2. Make changes, commit directly
3. `git push origin dev`

### Completion

- **CRITICAL:** Work is NOT complete until `git push` succeeds (both feature branch AND dev branch). NEVER stop before pushing. If push fails, resolve and retry until it succeeds.
- **Non-critical issues** discovered during work → note them, propose them when closing out the current issue
- **Handoff:** when handing off at session end, provide context for the next session.


## Git Worktree Details

Reference for worktree mechanics — the Mandatory Workflow above tells you *when* to use these.

**Structure:**
```
Cacheness/              # Main folder (on dev branch)
├── .venv/              # Main venv (NOT shared with worktrees)
├── worktrees/          # Feature worktrees container
│   ├── beads-abc123/   # Branch: beads-abc123-description (has own .venv)
│   └── beads-def456/   # Branch: beads-def456-description (has own .venv)
```

**Naming:** `beads-<hash>` format (e.g., `beads-a1b2c3d`) to clearly identify beads issues.

**Key rules:**
- Each worktree gets its own `.venv` — run `uv sync --all-groups` on first use
- Clean up promptly after merge — orphaned worktrees accumulate `.venv`/`__pycache__` bloat
- Each VS Code window has independent MCP instances
- `git worktree remove` always needs `--force` due to `.venv`/`__pycache__`


## Code Quality & Formatting

Reference for quality gate mechanics — the Mandatory Workflow step 4 tells you *when* to run these.

**Two-phase quality gates:** Phase 1 auto-fixes (never fails), Phase 2 validates (may fail).

```bash
# Full-repo check (scripts)
.\scripts\quality-check.ps1   # Windows
./scripts/quality-check.sh    # Unix/Linux/Mac

# Manual full-repo check
uv run ruff format . && uv run ruff check --fix .  # Phase 1
uv run ruff check . && uv run ty check             # Phase 2
```

**Pre-commit hook:** Auto-runs Phase 1 on every commit, logs Phase 2 errors to `.quality-errors.log`. Also installs bd hooks (pre-push, post-merge, etc.) via `--chain` so the quality check runs first.

**Install all hooks (quality + bd):**
```bash
.\scripts\install-hooks.ps1   # Windows
./scripts/install-hooks.sh    # Unix/Linux/Mac
```
Chain order: `pre-commit.old` (quality checks) → `pre-commit` (bd JSONL flush) → `pre-push` (bd stale guard)

**Config:** Ruff settings in `pyproject.toml`, Python 3.12+, line length 88.


## Test Suite

**Baseline:** 1427 passed, 102 skipped, 0 failures
**Full command:** `uv run pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py`

### Incremental Testing Strategy

During development, use **tiered testing** to minimize feedback time:

| Tier | When | What to run | Time |
|------|------|-------------|------|
| **Tier 1** | After each code change | Tests that directly exercise modified code | ~5-15s |
| **Tier 2** | After all planned changes for rapid debug before full suite | Add tests for likely regression areas | ~30-60s |
| **Full suite** | Once before push | All tests | ~33s parallel |

**Selecting Tier 1 tests:** Match changed source files to their primary test files:

| Source file changed | Primary test files |
|---------------------|--------------------|
| `core.py` | `test_core.py`, `test_storage_mode.py` |
| `blob_store.py` | `test_blob_store.py`, `test_blob_namespace.py` |
| `blob_backends.py` | `test_blob_store.py`, `test_blob_namespace.py`, `test_s3_blob_backend.py` |
| `s3_backend.py` | `test_s3_blob_backend.py` |
| `handlers/*.py` | `test_handlers.py` |
| `metadata.py` / backends | `test_metadata.py`, `test_sqlite_schema_versioning.py` |
| `size_utils.py` / `config.py` | `test_size_utils.py`, `test_core.py` |
| `security.py` | `test_security.py` |
| Path/namespace logic | `test_directory_sharding.py`, `test_namespace_config.py` |

**Selecting Tier 2 tests:** Add tests that cover cross-cutting concerns:
- Changed `core.py`? → Add `test_cache_integrity.py`, `test_update_operations.py`, `test_fault_injection.py`
- Changed blob layer? → Add `test_cache_integrity_verification.py`
- Changed metadata? → Add `test_backend_parity.py`, `test_backend_compatibility.py`

**Example workflow:**
```bash
# Editing core.py and blob_store.py — Tier 1
uv run pytest tests/test_core.py tests/test_blob_store.py tests/test_storage_mode.py -x -q --ignore=tests/test_tensorflow_handler.py
# Tier 2 (after Tier 1 passes)
uv run pytest tests/test_core.py tests/test_blob_store.py tests/test_storage_mode.py tests/test_cache_integrity.py tests/test_update_operations.py tests/test_cache_integrity_verification.py -x -q --ignore=tests/test_tensorflow_handler.py
# Full suite (once, before push)
uv run pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py
```

**Key test files:** `test_core.py`, `test_update_operations.py`, `test_handlers.py`, `test_metadata.py`, `test_security.py`, `test_schema_versioning.py`, `test_sqlite_schema_versioning.py`, `test_json_schema_versioning.py`, `test_pg_schema_versioning.py`, `test_namespace_config.py`


## MCP Tools Reference

**beads** — Issue tracking: `ready`, `show`, `create`, `update`, `close` (prefer MCP; failover to CLI for `--claim`, `--force` close, etc.)
**GitKraken** — Git ops: `status`, `add_or_commit`, `push`, `log_or_diff` (or use git CLI at discretion)
**language-server** — LSP navigation: `definition`, `references`, `hover`, `diagnostics`, `rename_symbol`, `edit_file`

**Git Operations:** Use git CLI or GitKraken MCP at discretion, whichever is more convenient/robust/safe for the task.


## Coding Gotchas

1. **`get()` is destructive on errors:** Auto-deletes entries that fail to load (except transient IO errors)
2. **Named params in `_create_cache_key()`:** Strip `prefix`, `description`, `custom_metadata`, `ttl_seconds` before hashing (critical bug pattern)
3. **Use `iter_entry_summaries()` for batch ops:** Lightweight flat dicts, not full ORM objects (major performance difference)
4. **Backend fast paths:** Check `hasattr(self.metadata_backend, 'query_meta')` for SQLite-specific optimizations

**Backend selection:** JSON (<200 entries, dev only, NOT safe for concurrency), SQLite (200+, production, multi-process), PostgreSQL (distributed)
See [`docs/BACKEND_SELECTION.md`](../docs/BACKEND_SELECTION.md) for details.


## Maintaining This Document

**Living Document Philosophy:** Update this file as the project, tools, and workflows evolve. When a better approach is identified, updating this doc should be proposed.

**When to update:** MCP tools added/replaced/removed, workflows improve, test baselines shift, patterns emerge, instructions cause confusion

**How to update:**
1. Edit `.github/copilot-instructions.md` directly
2. Commit: `docs: update copilot instructions - <reason>`
3. Push to dev branch (can use Workflow B — no worktree needed)

**Deprecating tools:** Document issues inline, suggest alternatives, disable in `.vscode/mcp.json`, remove after grace period

**Feedback loop:** Update instructions immediately when they fail or cause confusion. Propose improvements when discovering better approaches. Use git history as changelog.
