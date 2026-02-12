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

# Copy pre-commit hook
$SourceHook = Join-Path $ScriptDir "hooks\pre-commit"
$DestHook = Join-Path $HooksDir "pre-commit"

Copy-Item -Path $SourceHook -Destination $DestHook -Force

Write-Host ""
Write-Host "✅ Pre-commit hook installed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "The hook will:" -ForegroundColor Yellow
Write-Host "  - Auto-fix code formatting and safe lint issues (Phase 1)"
Write-Host "  - Log validation errors to .quality-errors.log (Phase 2)"
Write-Host "  - Allow commits to proceed (warnings only)"
