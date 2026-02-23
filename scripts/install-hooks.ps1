# Install pre-commit hooks for Cacheness
# Usage: .\scripts\install-hooks.ps1

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

Write-Host "Installing pre-commit hooks..." -ForegroundColor Cyan

# Use git rev-parse --git-dir to find the correct hooks directory.
# This works in both regular repos and worktrees.
try {
    $GitDir = (git -C $RepoRoot rev-parse --git-dir 2>$null).Trim()
    if (-not [System.IO.Path]::IsPathRooted($GitDir)) {
        $GitDir = Join-Path $RepoRoot $GitDir
    }
} catch {
    Write-Host "Error: Not in a git repository" -ForegroundColor Red
    exit 1
}

$HooksDir = Join-Path $GitDir "hooks"
if (-not (Test-Path $HooksDir)) {
    New-Item -ItemType Directory -Path $HooksDir -Force | Out-Null
}

# Step 1: Install our quality check as the pre-commit hook
$SourceHook = Join-Path $ScriptDir "hooks\pre-commit"
$DestHook = Join-Path $HooksDir "pre-commit"

Copy-Item -Path $SourceHook -Destination $DestHook -Force
Write-Host "  ✓ Quality check hook written" -ForegroundColor Gray

# Step 2: Layer bd hooks on top with --chain.
# bd will rename our pre-commit to pre-commit.old and call it first,
# then flush JSONL. All other bd hooks (pre-push, post-merge, etc.) are
# also installed here.
Write-Host "Installing bd (beads) hooks with --chain..." -ForegroundColor Cyan
uv run bd hooks install --chain

Write-Host ""
Write-Host "✅ All hooks installed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Hook chain on commit:" -ForegroundColor Yellow
Write-Host "  1. pre-commit.old  → Cacheness quality checks (ruff + ty)"
Write-Host "  2. pre-commit      → bd JSONL flush"
Write-Host "  3. pre-push        → bd stale-JSONL guard"
