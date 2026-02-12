#!/usr/bin/env pwsh
# Quality Check Script for Cacheness
# Runs formatting, linting, and type checking in two phases

$ErrorActionPreference = "Continue"

Write-Host "🔧 Phase 1: Auto-fixing..." -ForegroundColor Cyan
Write-Host ""

# Phase 1: Auto-fix (never fails)
Write-Host "  → Running ruff format..." -ForegroundColor Gray
uv run ruff format .

Write-Host "  → Running ruff check --fix..." -ForegroundColor Gray
uv run ruff check --fix .

Write-Host ""
Write-Host "✅ Auto-fixes applied" -ForegroundColor Green
Write-Host ""
Write-Host "🔍 Phase 2: Validation..." -ForegroundColor Cyan
Write-Host ""

# Phase 2: Validation (may fail)
$ruffFailed = $false
$tyFailed = $false

Write-Host "  → Running ruff check..." -ForegroundColor Gray
uv run ruff check .
if ($LASTEXITCODE -ne 0) {
    $ruffFailed = $true
}

Write-Host "  → Running ty..." -ForegroundColor Gray
uv run ty check
if ($LASTEXITCODE -ne 0) {
    $tyFailed = $true
}

Write-Host ""

if ($ruffFailed -or $tyFailed) {
    Write-Host "❌ Quality gate failed - please fix remaining issues" -ForegroundColor Red
    if ($ruffFailed) {
        Write-Host "   • Ruff found unfixable lint issues" -ForegroundColor Yellow
    }
    if ($tyFailed) {
        Write-Host "   • Type checking errors found" -ForegroundColor Yellow
    }
    exit 1
}

Write-Host "✅ All quality checks passed" -ForegroundColor Green
exit 0
