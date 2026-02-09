# GitHub Copilot Instructions for Cacheness

## Project Overview

**Cacheness** is a Python disk caching library with pluggable metadata backends and handler-based serialization. It provides persistent caching for expensive computations with support for multiple storage backends (JSON, SQLite, PostgreSQL) and type-aware serialization handlers (DataFrames, NumPy arrays, TensorFlow tensors, etc.).

## Package Manager: uv

**ALWAYS use `uv` for Python operations** — it's already available in PATH.

```bash
# Run tests
uv run pytest tests/ -x -q

# Run specific test file
uv run pytest tests/test_core.py -v

# Run benchmarks
uv run python benchmarks/management_ops_benchmark.py

# Install dependencies
uv sync

# Add new dependency
uv add package-name

# Run Python scripts
uv run python script.py
```

**Do NOT use:** `python`, `pip`, `python -m pytest` — use `uv run` instead.


## Code Quality & Formatting

**Primary Tools:** Use `ruff` for linting/formatting and `ty` for type checking.

### Quality Gates

This project uses a two-phase quality gate system.

**Phase 1: Auto-Fix (Always Runs, Never Fails)**
```bash
uv run ruff format . && uv run ruff check --fix .
```
- Formats all code to project standards
- Fixes safe lint violations automatically
- Modifies files in place
- Never exits with error

**Phase 2: Validation (Check Only, May Fail)**
```bash
uv run ruff check . && uv run ty check
```
- Reports unfixable lint issues
- Reports type errors
- Does not modify files
- Exits non-zero if issues remain

### Quick Commands

```bash
# Run full quality check (both phases)
.\scripts\quality-check.ps1   # Windows
./scripts/quality-check.sh    # Unix/Linux/Mac

# Or manually:
uv run ruff format . && uv run ruff check --fix .  # Phase 1: Auto-fix
uv run ruff check . && uv run ty check                # Phase 2: Validate
```

### Pre-commit Hook

A pre-commit hook is installed at `.git/hooks/pre-commit` that:
1. ✅ Runs Phase 1 auto-fixes on every commit
2. 📝 Logs Phase 2 errors to `.quality-errors.log` (if any)
3. ✅ Allows commit to proceed (warnings only)

**Check for logged errors:**
```bash
cat .quality-errors.log  # View errors from last commit
```

### Configuration

- Ruff settings in `pyproject.toml` under `[tool.ruff]`
- Target Python: 3.12+
- Line length: 88
- Respects `.gitignore` by default

### When to Use

**Auto-fix (Phase 1):**
- Automatically runs on every commit (via pre-commit hook)
- Run manually before pushing: `uv run ruff format . && uv run ruff check --fix .`

**Validation (Phase 2):**
- Before creating PR
- After implementing new features
- When fixing bugs or refactoring
- As part of CI/CD validation
- Run: `.\scripts\quality-check.ps1` or check `.quality-errors.log`


## Issue Tracking

This project uses **beads-mcp** for issue tracking. See the beads section in MCP Tools Reference below for available tools.


## MCP Tools Reference

This project has several MCP servers configured in `.vscode/mcp.json`. Use these tools appropriately:

### beads (Issue Tracking, Development session todos, Project Management)
**Purpose:** Track work items, bugs, and features
- `mcp_beads_ready` — List issues ready for work
- `mcp_beads_show` — View issue details
- `mcp_beads_create` — Create new issues
- `mcp_beads_update` — Update issue content/metadata
- `mcp_beads_close` — Close completed work
- **Do NOT:** Run `bd` or `beads` CLI commands via terminal

### GitKraken (Git Operations)
**Purpose:** All git operations (commit, push, status, etc.)
- `mcp_gitkraken_git_status` — Check repo state
- `mcp_gitkraken_git_add_or_commit` — Stage and commit changes
- `mcp_gitkraken_git_push` — Push to remote
- `mcp_gitkraken_git_log_or_diff` — View history/diffs
- **Do NOT:** Run `git` CLI commands via terminal

### memory (Persistent Knowledge Graph)
**Purpose:** Remember project context across sessions
- `create_entities` — Store project concepts, patterns, decisions
- `create_relations` — Link entities (e.g., "UnifiedCache" → "uses" → "SQLiteBackend")
- `add_observations` — Add facts to existing entities
- `search_nodes` — Find remembered information
- `read_graph` — See all stored knowledge
- **When to use:** Store architectural decisions, common patterns, user preferences

### language-server (Semantic Code Navigation)
**Purpose:** Navigate Python codebase with LSP intelligence
- `definition` — Jump to symbol definition
- `references` — Find all usages of a symbol
- `hover` — Get type info and documentation
- `diagnostics` — List errors/warnings in a file
- `rename_symbol` — Safely rename across codebase
- `edit_file` — Make precise line-based edits
- **When to use:** Refactoring, understanding code flow, finding usage patterns

### memalot (Memory Leak Detection)
**Purpose:** Detect Python memory leaks during testing
- `list_reports` — List saved leak analysis reports
- `get_report` — Retrieve specific leak report details
- **When to use:** Investigating memory issues, analyzing long-running cache operations
- **Note:** Requires instrumenting code with `memalot.start_leak_monitoring()`

### code-checker (Code Quality)
**Purpose:** Run pytest, pylint, and mypy with AI-friendly output
- `run_pytest_check` — Run tests with structured failure reports
- `run_pylint_check` — Get linting issues with fix suggestions
- `run_mypy_check` — Type checking with error explanations
- `run_all_checks` — Combined quality analysis
- **When to use:** Before committing, investigating test failures, code review
- **Note:** For direct linting/formatting, prefer `ruff` commands; for type checking, prefer `ty` (see Code Quality & Formatting section)


## Test Suite

- **Location:** `tests/` directory
- **Current baseline:** 787 passed, 70 skipped, 0 failures
- **Run command:** `uv run pytest tests/ -x -q`
- **Conventions:**
  - Use `-x` flag to stop on first failure
  - Use `-q` for quiet output, `-v` for verbose
  - Fixtures defined in `tests/conftest.py`

### Key Test Files
- `test_core.py` — Main cache operations
- `test_update_operations.py` — Management operations (update, touch, delete_where, batch ops)
- `test_handlers.py` — Serialization handlers
- `test_metadata.py` — Metadata backend tests
- `test_security.py` — HMAC signing and verification

## Architecture

### Storage Layer Separation (Phase 1 — Complete)

```
UnifiedCache (core.py)
    ↓
├── Handlers (handlers.py) — Type-specific serialization
│   ├── DataFrameHandler
│   ├── NumpyHandler
│   ├── TensorFlowHandler
│   └── PickleHandler (fallback)
│
├── Metadata Backends (metadata.py)
│   ├── JsonBackend
│   ├── SQLiteBackend
│   ├── PostgresBackend
│   └── MemoryCacheWrapper
│
└── Blob Storage (compress_pickle.py)
    └── Atomic file writes with .tmp + rename
```

### Key Components

1. **`src/cacheness/core.py`** — Main `UnifiedCache` class (1,689 lines)
   - Single coordination layer
   - Delegates to handlers and backends
   - **IMPORTANT:** Has a `_lock` field that is created but **never acquired** — not thread-safe

2. **`src/cacheness/handlers.py`** — Type-aware serialization registry
   - Handlers registered via `@register_handler` decorator
   - Priority system for handler selection
   - `can_handle(obj)` → `put(obj, path)` → `get(path)` pattern

3. **`src/cacheness/metadata.py`** — Metadata backend implementations
   - `MetadataBackend` abstract base class
   - Each backend implements: `put_entry()`, `get_entry()`, `list_entries()`, `delete_entry()`
   - **NEW:** `iter_entry_summaries()` for lightweight iteration (added for batch operations optimization)

4. **`src/cacheness/compress_pickle.py`** — Blob compression/decompression
   - **CRITICAL:** Always reads entire file with `f.read()` (fixed from 2GB buffer issue)
   - Supports blosc2, lz4, zstd, gzip

## Coding Conventions

### 1. Always Check Backend Capabilities
```python
# Backend-specific optimizations exist
if hasattr(self.metadata_backend, 'query_meta'):
    # SQLite fast path with JSON column search
    matching_keys = self.metadata_backend.query_meta(**kwargs)
else:
    # Generic path — iterate all entries
    for entry in self.metadata_backend.list_entries():
        # ... filter manually
```

### 2. Use `iter_entry_summaries()` for Batch Operations
```python
# ❌ BAD — Loads full ORM objects
for entry in self.metadata_backend.list_entries():
    if some_filter(entry):
        keys.append(entry['cache_key'])

# ✅ GOOD — Lightweight flat dicts, no ORM hydration
for entry in self.metadata_backend.iter_entry_summaries():
    if some_filter(entry):
        keys.append(entry['cache_key'])
```

### 3. Error Handling in `get()`
`get()` is **destructive** on errors — it auto-deletes entries that fail to load:
```python
try:
    obj = handler.get(actual_path)
except Exception:
    # WARNING: This deletes the metadata entry permanently
    self.metadata_backend.delete_entry(cache_key)
    return None
```

### 4. Named Parameters in `_create_cache_key()`
**Critical bug pattern:** Named parameters (`prefix`, `description`, `custom_metadata`, `ttl_seconds`) must be stripped before hashing:
```python
# ✅ CORRECT
kwargs_for_key = {k: v for k, v in kwargs.items() 
                  if k not in ('prefix', 'description', 'custom_metadata')}
cache_key = self._create_cache_key(**kwargs_for_key)
```

### 5. Backend Thread Safety
- **JSON:** Protected by `threading.Lock()`
- **SQLite:** `check_same_thread=False` + WAL mode + per-operation locks
- **PostgreSQL:** `ThreadPoolExecutor` around all operations
- **UnifiedCache:** Protected by `threading.RLock()` around all public methods

## Common Patterns

### Adding a New Handler
```python
@register_handler(priority=100)
class MyTypeHandler(Handler):
    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, MyType)
    
    def put(self, obj: MyType, file_path: str) -> Dict[str, Any]:
        # Serialize and return metadata
        return {"storage_format": "myformat"}
    
    def get(self, file_path: str) -> MyType:
        # Deserialize and return object
        return MyType(...)
```

### Adding a Management Operation
```python
def my_operation(self, **filter_kwargs) -> int:
    """New management operation."""
    count = 0
    # Use iter_entry_summaries for performance
    for entry in self.metadata_backend.iter_entry_summaries():
        if self._matches_filters(entry, filter_kwargs):
            # Perform operation
            count += 1
    return count
```

## Known Issues & Gotchas

1. **~~Race condition in `put()`:~~** FIXED — All public methods now protected by `threading.RLock()`
2. **~~Auto-deletion on read errors:~~** FIXED — `get()` preserves metadata on transient `IOError`/`OSError`
3. **Orphaned blobs on crash:** `put()` cleans up on exception, but a hard crash between blob write and metadata write can still leak. Use `verify_integrity(repair=True)` to detect/fix.
4. **~~`invalidate()` doesn't delete blob files:~~** FIXED — `invalidate()` now deletes the blob file before removing metadata. All delete operations (`delete_where`, `delete_matching`, `delete_batch`) delegate to `invalidate()` and benefit from this fix.
5. **SQLite "database is locked":** 30-second timeout can be exceeded under heavy concurrent writes
6. **JSON backend scales O(n²):** Each write re-serializes entire JSON file

## Benchmarks

Run benchmarks to validate performance:
```bash
uv run python benchmarks/management_ops_benchmark.py
uv run python benchmarks/handler_benchmark.py
uv run python benchmarks/serialization_benchmark.py
```

Expected baseline (200 entries):
- `touch_batch`: ~0.79ms (SQLite)
- `delete_where`: ~44ms (SQLite)
- `list_entries`: ~20ms (SQLite), ~0.40ms (JSON)

## Documentation

- **API Reference:** `docs/API_REFERENCE.md`
- **Performance:** `docs/PERFORMANCE.md`
- **Backend Selection:** `docs/BACKEND_SELECTION.md`

## When Modifying Core Logic

1. **Always run tests after changes:** `uv run pytest tests/ -x -q`
2. **Check for regressions:** Baseline is 787 passed, 70 skipped
3. **Run quality gates:** `.\scripts\quality-check.ps1` (or `./scripts/quality-check.sh`)
4. **Run relevant benchmark:** Ensure no performance degradation
5. **Update documentation:** If changing public API

## Dependencies

All dependencies managed in `pyproject.toml`:
- **Core:** `cachetools`, `xxhash`
- **Recommended:** `numpy`, `blosc2`, `pandas`, `sqlalchemy`, `dill`, `orjson`
- **Optional:** `tensorflow`, `torch`, `psycopg[binary]`, `boto3`
- **Dev:** `pytest`, `pytest-cov`, `hypothesis`, `ruff`, `ty`, `beads-mcp`

Install with: `uv sync --all-groups`

## Git Operations

Use Git CLI or GitKraken MCP at descretion, whichever is more convenient/robust/safe for the task.

## Git Worktree Workflow

This project uses **git worktrees** for parallel development, allowing work on multiple issues simultaneously without branch switching conflicts.

### Directory Structure

```
C:\Users\akriz\code\
├── Cacheness\              # Main worktree (main branch) - stable releases
├── Cacheness-dev\          # Integration worktree (dev branch) - where features merge
└── Cacheness-issue-<hash>\ # Feature worktrees (one per beads issue)
```

### Worktree Naming Convention

**Use beads issue hash names** to avoid naming conflicts:
- ✅ CORRECT: `Cacheness-issue-a1b2c3d` (uses beads hash)
- ❌ WRONG: `Cacheness-issue-42` (sequential numbering conflicts with beads)

Get the issue hash from beads commands like `mcp_beads_show`, `mcp_beads_ready`, or `mcp_beads_list`.

### Creating a Feature Worktree

When starting work on a beads issue:

```bash
# 1. Get issue details (note the hash, e.g., "a1b2c3d")
uv run bd show <issue-hash>

# 2. Create worktree with beads hash naming
cd C:\Users\akriz\code\Cacheness
git worktree add -b issue-<hash>-<short-description> ../Cacheness-issue-<hash> dev

# Example:
git worktree add -b issue-a1b2c3d-handler-perf ../Cacheness-issue-a1b2c3d dev

# 3. Open in new VS Code window
code ../Cacheness-issue-<hash>

# 4. Create WORKTREE.md with issue details
# (Do this from within the new worktree's VS Code window using mcp_beads_show)

# 5. Update beads to mark issue as in-progress
# (Do this from within the new worktree's VS Code window)
```

### WORKTREE.md Documentation

Each feature worktree should have a `WORKTREE.md` file at the root documenting the issue:

**Purpose:**
- Provides context about what's being worked on
- Tracks progress and decisions made during work
- Useful for parallel agent work (each agent sees their task)
- Helps resume work after context switches

**Template:**
```markdown
# Worktree: Issue <hash>

**Branch:** `issue-<hash>-<description>`
**Status:** In Progress
**Assigned:** <agent-name or your-name>
**Created:** <date>

## Issue Details

<Copy from beads: title, description, acceptance criteria>

## Work Log

- [ ] Task 1
- [ ] Task 2
- [ ] Task 3

## Notes

- Any important decisions or findings
- Links to relevant documentation
- Dependencies or blockers

## Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Definition of Done

- [ ] Code complete and committed
- [ ] Tests pass: `uv run pytest tests/ -x -q`
- [ ] Quality gates pass: `.\scripts\quality-check.ps1` (or `./scripts/quality-check.sh`)
- [ ] Feature branch pushed
- [ ] Ready for integration to dev
```

**Creating WORKTREE.md:**
```bash
# In your new worktree
cd ../Cacheness-issue-<hash>

# Use beads to get issue details
mcp_beads_show --id <hash>

# Create WORKTREE.md with the issue information
# (Agent will do this automatically based on beads output)
```

**Important:** `WORKTREE.md` is git-ignored (worktree-specific metadata, not committed)

### Working in Feature Worktree

Once in your feature worktree:

```bash
# Verify you're in the correct worktree
pwd                                    # Should show: .../Cacheness-issue-<hash>
git branch --show-current              # Should show: issue-<hash>-description

# Check WORKTREE.md for issue context and tasks
cat WORKTREE.md

# Normal development workflow
# ... make changes ...
# Update WORKTREE.md work log as you progress
git add .
git commit -m "fix: description of changes"

# Run quality gates
uv run pytest tests/ -x -q
.\scripts\quality-check.ps1

# Push your feature branch
git push -u origin issue-<hash>-description
```

### Integration Workflow

When feature is complete and pushed:

```bash
# Switch to dev worktree for integration
cd C:\Users\akriz\code\Cacheness-dev

# Pull latest changes
git pull origin dev

# Merge feature branch
git merge issue-<hash>-description

# Run full test suite in integration environment
uv run pytest tests/ -x -q
.\scripts\quality-check.ps1

# If tests pass, push to dev
git push origin dev

# Return to main worktree for cleanup
cd C:\Users\akriz\code\Cacheness

# Remove feature worktree
git worktree remove ../Cacheness-issue-<hash>

# Delete feature branch (locally and remotely)
git branch -d issue-<hash>-description
git push origin --delete issue-<hash>-description

# Update beads to close issue
# (will be done from main worktree's VS Code)
```

### Worktree Management Commands

```bash
# List all worktrees
git worktree list

# Remove a worktree (must be run from main worktree)
git worktree remove ../Cacheness-issue-<hash>

# Remove worktree with uncommitted changes (force)
git worktree remove --force ../Cacheness-issue-<hash>

# Prune stale worktree references
git worktree prune

# Move a worktree to different location
git worktree move ../Cacheness-issue-<hash> ../new-location
```

### Worktree Rules & Best Practices

1. **Main worktree** (`Cacheness/`) stays on `main` branch - for releases and stable work
2. **Dev worktree** (`Cacheness-dev/`) stays on `dev` branch - integration only, no feature development
3. **Feature worktrees** are temporary - create for issue, delete when merged
4. **Always use beads hash** in worktree names to match issue IDs
5. **One issue per worktree** - don't work on multiple issues in same worktree
6. **Push before integrating** - always push feature branch before merging to dev
7. **Clean up promptly** - remove worktrees after successful integration
8. **MCP servers are isolated** - each worktree's VS Code window has independent MCP instances

### Switching Between Worktrees

**Do NOT use `git checkout`** to switch between features. Instead:
- Open different VS Code windows for different worktrees
- Use Windows Terminal tabs, each cd'd to different worktree
- Each worktree is a separate physical directory

### When to Use Worktrees

**Use worktrees for:**
- Working on multiple beads issues simultaneously
- Keeping a stable environment for running benchmarks while developing
- Emergency hotfixes without interrupting feature work
- Reviewing/testing PRs without stashing current work

**Skip worktrees for:**
- Quick documentation updates (use main or dev directly)
- Single-issue workflows (traditional branch switching is fine)

### Troubleshooting

**Issue:** Can't remove worktree - "worktree contains modified or untracked files"
```bash
cd ../Cacheness-issue-<hash>
git add . && git commit -m "wip: save progress"
git push -u origin issue-<hash>-description
cd ../Cacheness
git worktree remove ../Cacheness-issue-<hash>
```

**Issue:** Worktree removed but branch still exists
```bash
git branch -d issue-<hash>-description       # Delete local
git push origin --delete issue-<hash>-description  # Delete remote
```

**Issue:** Lost track of which worktrees exist
```bash
git worktree list
# Then navigate to each and check git status
```

## Sessions / Workflows

**MANDATORY WORKFLOW:**

1. **File issues (in beads) for remaining work** — Create issues for anything that needs follow-up
2. **Create `WORKTREE.md` with issue details from beads
   - Create feature worktree** (if working on specific issue):
   - Get issue hash from beads: `mcp_beads_show` or `mcp_beads_ready`
   - Create worktree: `git worktree add -b issue-<hash>-<desc> ../Cacheness-issue-<hash> dev`
   - Open in new VS Code window: `code ../Cacheness-issue-<hash>`
   - Update beads: Mark issue as in-progress
3. **Run quality gates** (if code changed):
   - Tests: `uv run pytest tests/ -x -q`
   - Quality check: `.\scripts\quality-check.ps1` (or `./scripts/quality-check.sh`)
   - Check errors: `cat .quality-errors.log` (if file exists)
4. **Push feature branch** — `git push -u origin issue-<hash>-description`
5. **Integrate to dev worktree**:
   - Switch to dev worktree: `cd ../Cacheness-dev`
   - Merge: `git merge issue-<hash>-description`
   - Test: `uv run pytest tests/ -x -q`
   - Push: `git push origin dev`
6. **Cleanup worktree**:
   - Remove: `git worktree remove ../Cacheness-issue-<hash>`
   - Delete branch: `git branch -d issue-<hash>-description`
   - Update beads: Close issue
7. **Verify** — Use `mcp_gitkraken_git_status` to confirm "up to date with origin"
8. **Hand off** — Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds (both feature branch AND dev branch)
- NEVER stop before pushing — that leaves work stranded locally
- NEVER say "ready to push when you are" — YOU must push
- If push fails, resolve and retry until it succeeds
- Use beads hash names for worktrees to match issue IDs

### Session Completion

**When ending a work session**, you MUST complete ALL steps in the session. Work is NOT complete until `git push` succeeds.


---

## Maintaining This Document

This document is a **living document** — it should evolve as the project, tools, and workflows change.

### When to Update

**Update this document when:**
- ✏️ Adding or removing MCP tools
- ✏️ Changing development workflows
- ✏️ Discovering better practices
- ✏️ Tools become broken or unreliable
- ✏️ Test baselines change significantly
- ✏️ New patterns emerge from actual usage
- ✏️ Instructions cause confusion or errors

### How to Update

**Process:**
1. Identify what needs to change (broken tool, outdated workflow, etc.)
2. Edit `.github/copilot-instructions.md` directly
3. Update "Last Updated" date at bottom
4. Commit with descriptive message: `docs: update copilot instructions - <reason>`
5. Push to dev branch

**Example updates:**
```bash
# Remove broken tool
# 1. Remove from MCP Tools Reference section
# 2. Remove from .vscode/mcp.json
# 3. Update workflow steps that reference it

# Add new practice
# 1. Add to relevant section (Coding Conventions, Common Patterns, etc.)
# 2. Include example code if applicable
# 3. Explain why this practice is recommended

# Update workflow
# 1. Modify Sessions/Workflows section
# 2. Test the new workflow
# 3. Update any affected sections (e.g., Git Worktree Workflow)
```

### Deprecating Tools

**When a tool becomes problematic:**

1. **Document the issue** in comments:
   ```markdown
   ### tool-name (Currently Disabled - See Note)
   **Note:** This tool has issues with <problem>. Consider using <alternative> instead.
   ```

2. **Disable in MCP config**:
   - Comment out in `.vscode/mcp.json`
   - Don't delete immediately (keeps history)

3. **Update workflow instructions**:
   - Change from "Use tool X" to "Use alternative Y"
   - Add note explaining why

4. **Remove after grace period**:
   - After confirming alternative works
   - Remove from both this document and MCP config

**Example - Replacing GitKraken MCP:**
```markdown
### XYZ Operations
~~Use XYZ MCP for git operations~~

**Update 2026-02-09:** XYZ MCP has reliability issues. Use XYZ CLI instead:
- `xyz status` instead of `mcp_gitkraken_git_status`
- `xyz add` and `xyz commit` instead of `mcp_xyz_add_or_commit`
- `xyz push` instead of `mcp_xyz_push`
```

### Feedback Loop

**Encourage continuous improvement:**

- 🔄 **When instructions fail:** Update them immediately after finding a solution
- 🔄 **When tools break:** Document workarounds, then update for long-term fix
- 🔄 **When patterns emerge:** Codify them as conventions
- 🔄 **When confused:** Clarify the confusing section right away

**Agents should:**
- Suggest updates when encountering unclear instructions
- Report tools that consistently fail or cause issues
- Propose better workflows when discovering them
- Update test baselines after major test suite changes

### Change Tracking

**Use git history as changelog:**
```bash
# View instruction changes
git log --oneline -- .github/copilot-instructions.md

# See specific change
git show <commit-hash>:.github/copilot-instructions.md
```

**Commit message format:**
```
docs: update copilot instructions - <what changed>

- Specific change 1
- Specific change 2
- Reason for change
```

### Known Issues with Tools

**Current tool status:**
- ✅ **beads-mcp:** Working well, core dependency
- ✅ **memory:** Stable, useful for cross-session context
- ✅ **language-server:** Helpful for navigation
- ⚠️ **GitKraken MCP:** Use with caution, prefer git CLI for reliability
- ⚠️ **code-checker:** Useful but prefer direct `ruff`/`ty` commands
- ✅ **memalot:** Specialized tool, works when needed

**Note:** This status list should be updated as tools evolve.

---

**Document Version:** See git history for changes
**Contributing:** All agents and developers are encouraged to improve this document
