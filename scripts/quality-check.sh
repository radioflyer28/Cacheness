#!/bin/bash
# Quality Check Script for Cacheness
# Runs formatting, linting, and type checking in two phases

set +e  # Don't exit on error

echo "🔧 Phase 1: Auto-fixing..."
echo ""

# Phase 1: Auto-fix (never fails)
echo "  → Running ruff format..."
uv run ruff format .

echo "  → Running ruff check --fix..."
uv run ruff check --fix .

echo ""
echo "✅ Auto-fixes applied"
echo ""
echo "🔍 Phase 2: Validation..."
echo ""

# Phase 2: Validation (may fail)
ruff_failed=0
ty_failed=0

echo "  → Running ruff check..."
uv run ruff check . || ruff_failed=$?

echo "  → Running ty..."
uv run ty check || ty_failed=$?

echo ""

if [ $ruff_failed -ne 0 ] || [ $ty_failed -ne 0 ]; then
    echo "❌ Quality gate failed - please fix remaining issues"
    [ $ruff_failed -ne 0 ] && echo "   • Ruff found unfixable lint issues"
    [ $ty_failed -ne 0 ] && echo "   • Type checking errors found"
    exit 1
fi

echo "✅ All quality checks passed"
exit 0
