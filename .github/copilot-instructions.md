# GitHub Copilot Instructions for Cacheness

## Project Overview

**Cacheness** is a Python disk caching library with pluggable metadata backends and handler-based serialization. Supports JSON/SQLite/PostgreSQL backends and type-aware handlers (DataFrames, NumPy, TensorFlow tensors, etc.).

**What makes it special:** Unlike joblib.Memory or diskcache, Cacheness provides **type-aware serialization** (optimized parquet for DataFrames, blosc2 for NumPy), **pluggable metadata backends** (JSON for simple caches, SQLite for large caches, PostgreSQL for distributed), and **cryptographic signing** for cache integrity.

**Detailed docs:** See [`docs/README.md`](../docs/README.md) for comprehensive guides.
**Common issues:** See [`docs/TROUBLESHOOTING.md`](../docs/TROUBLESHOOTING.md) for troubleshooting guide.

## Quick Start

```bash
# Run tests
uv run pytest tests/ -x -q

# Quality check before push
.\scripts\quality-check.ps1  # Windows
./scripts/quality-check.sh   # Unix/Linux/Mac

# Check logged errors
cat .quality-errors.log

# Work on issue (see Git Worktree Workflow)
git worktree add worktrees/beads-<hash> -b beads-<hash>-<desc> dev
```

**Windows: Start beads daemon once per session**
```powershell
uv run bd daemon start
```
(Required after reboot/logout - beads-mcp auto-start doesn't work on Windows)

## Package Manager: uv

**ALWAYS use `uv` for Python operations** — never use `python`, `pip`, or `python -m pytest` directly.

```bash
uv run pytest tests/ -x -q          # Run tests
uv run python benchmarks/script.py  # Run benchmarks
uv sync                              # Install dependencies
uv add package-name                  # Add dependency
```


## Code Quality & Formatting

**Two-phase quality gates:** Phase 1 auto-fixes (never fails), Phase 2 validates (may fail).

```bash
# Run full check
.\scripts\quality-check.ps1   # Windows
./scripts/quality-check.sh    # Unix/Linux/Mac

# Or manually
uv run ruff format . && uv run ruff check --fix .  # Phase 1
uv run ruff check . && uv run ty check             # Phase 2
```

**Pre-commit hook:** Auto-runs Phase 1 on every commit, logs Phase 2 errors to `.quality-errors.log`.

**Config:** Ruff settings in `pyproject.toml`, Python 3.12+, line length 88.


## MCP Tools Reference

**beads** — Issue tracking: `ready`, `show`, `create`, `update`, `close` (don't use CLI)
**GitKraken** — Git ops: `status`, `add_or_commit`, `push`, `log_or_diff` (or use git CLI at discretion)
**memory** — Knowledge graph: `create_entities`, `create_relations`, `add_observations`, `search_nodes`, `read_graph`
**language-server** — LSP navigation: `definition`, `references`, `hover`, `diagnostics`, `rename_symbol`, `edit_file`
**memalot** — Memory leak detection: `list_reports`, `get_report` (requires instrumenting code)
**code-checker** — Quality tools: `run_pytest_check`, `run_pylint_check`, `run_mypy_check` (prefer `ruff`/`ty` directly)

**Git Operations:** Use git CLI or GitKraken MCP at discretion, whichever is more convenient/robust/safe for the task.


## Test Suite

**Baseline:** 848 passed, 65 skipped, 0 failures
**Command:** `uv run pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py` (use `-x` to stop on first failure)
**Windows:** Always add `--ignore=tests/test_tensorflow_handler.py` — TensorFlow tests hang on Windows.
**Key files:** `test_core.py`, `test_update_operations.py`, `test_handlers.py`, `test_metadata.py`, `test_security.py`

## Coding Gotchas

1. **`get()` is destructive on errors:** Auto-deletes entries that fail to load (except transient IO errors)
2. **Named params in `_create_cache_key()`:** Strip `prefix`, `description`, `custom_metadata`, `ttl_seconds` before hashing (critical bug pattern)
3. **Use `iter_entry_summaries()` for batch ops:** Lightweight flat dicts, not full ORM objects (major performance difference)
4. **Backend fast paths:** Check `hasattr(self.metadata_backend, 'query_meta')` for SQLite-specific optimizations

**Backend selection:** JSON (<200 entries, dev only, NOT safe for concurrency), SQLite (200+, production, multi-process), PostgreSQL (distributed)
See [`docs/BACKEND_SELECTION.md`](../docs/BACKEND_SELECTION.md) for details.

## Git Worktree Workflow

**Parallel development** using git worktrees nested inside the main repo to share dependencies.

**Structure:** 
```
Cacheness/              # Main folder (on dev branch)
├── .venv/              # Main venv (NOT shared with worktrees)
├── worktrees/          # Feature worktrees container
│   ├── beads-abc123/   # Branch: beads-abc123-description (has own .venv)
│   └── beads-def456/   # Branch: beads-def456-description (has own .venv)
```

**Naming:** Use `beads-<hash>` format (✅ `beads-a1b2c3d`) to clearly identify beads issues

**Setup (one-time):**
```bash
# Add worktrees/ to .gitignore
echo "/worktrees/" >> .gitignore
```

**Create feature worktree:**
```bash
# Get issue hash from beads, then:
git worktree add worktrees/beads-<hash> -b beads-<hash>-<desc> dev
cd worktrees/beads-<hash>
uv sync --all-groups   # Each worktree gets its own .venv
# Create WORKTREE.md with issue details, mark issue in-progress
```

**Note:** Nested worktrees do **not** share the parent's `.venv`. Each needs `uv sync --all-groups` on first use.

**Integration workflow:**
```bash
# Test and push from worktree
cd worktrees/beads-<hash>
uv run pytest tests/ -x -q && git push -u origin beads-<hash>-<desc>

# Integrate to dev (from main repo root)
cd ../..  # Back to Cacheness/
git checkout dev
git pull origin dev && git merge beads-<hash>-<desc>
uv run pytest tests/ -x -q && git push origin dev

# Cleanup
git worktree remove worktrees/beads-<hash>
git branch -d beads-<hash>-<desc> && git push origin --delete beads-<hash>-<desc>
# Close issue in beads
```

**Rules:** Each worktree has its own `.venv` (run `uv sync --all-groups` on first use). Clean up promptly after merge. Each VS Code window has independent MCP instances.

**beads gotcha:** Closing a child issue of an epic requires `--force` (parent-child dep blocks close). Use CLI: `uv run bd close CACHE-xxx --force`

## Sessions / Workflows

**MANDATORY WORKFLOW:**

1. **File issues (in beads) for remaining work** — Create issues for anything that needs follow-up
2. **Create feature worktree** (if working on specific issue):
   - Get issue hash from beads: `mcp_beads_show` or `mcp_beads_ready`
   - Create worktree: `git worktree add worktrees/beads-<hash> -b beads-<hash>-<desc> dev`
   - Open in new VS Code window: `code worktrees/beads-<hash>`
   - Create `WORKTREE.md` with issue details
   - Update beads: Mark issue as in-progress
3. **Run quality gates** (if code changed):
   - Quality check (changed files only, BEFORE testing):
     ```bash
     # Phase 1: auto-fix changed files
     uv run ruff format $(git diff --name-only --diff-filter=ACMR HEAD -- '*.py')
     uv run ruff check --fix $(git diff --name-only --diff-filter=ACMR HEAD -- '*.py')
     # Phase 2: validate changed files
     uv run ruff check $(git diff --name-only --diff-filter=ACMR HEAD -- '*.py')
     ```
   - Tests: `uv run pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py`
   - **Update docs:** If changing public API, update relevant files in `docs/` before committing
4. **Push feature branch** — `git push -u origin beads-<hash>-description`
5. **Integrate to dev**:
   - Return to main folder: `cd ../..` (from worktree back to Cacheness/)
   - Ensure on dev: `git checkout dev`
   - Merge: `git pull origin dev && git merge beads-<hash>-description`
   - Test: `uv run pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py`
     - **Skip if fast-forward merge:** If merge was fast-forward (no conflicts, no diverged dev changes), tests already passed in worktree — skip re-running
   - Push: `git push origin dev`
6. **Cleanup worktree**:
   - Remove: `git worktree remove worktrees/beads-<hash>`
   - Delete branch: `git branch -d beads-<hash>-description && git push origin --delete beads-<hash>-description`
   - Update beads: Close issue
7. **Verify** — Check git status confirms "up to date with origin"
8. **Hand off** — Provide context for next session

**CRITICAL:** Work is NOT complete until `git push` succeeds (both feature branch AND dev branch). NEVER stop before pushing. If push fails, resolve and retry until it succeeds.

## Maintaining This Document

**Living Document Philosophy:** Update this file as the project, tools, and workflows evolve.

**When to update:** MCP tools change, workflows improve, test baselines shift, patterns emerge, instructions cause confusion

**How to update:**
1. Edit `.github/copilot-instructions.md` directly
2. Commit: `docs: update copilot instructions - <reason>`
3. Push to dev branch

**Deprecating tools:** Document issues inline, suggest alternatives, disable in `.vscode/mcp.json`, remove after grace period

**Feedback loop:** Update instructions immediately when they fail or cause confusion. Propose improvements when discovering better approaches. Use git history as changelog.
