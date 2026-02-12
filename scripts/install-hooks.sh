#!/bin/bash
# Install pre-commit hooks for Cacheness
# Usage: ./scripts/install-hooks.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Installing pre-commit hooks..."

# Use git rev-parse --git-dir to find the correct hooks directory.
# This works in both regular repos and worktrees.
GIT_DIR="$(git -C "$REPO_ROOT" rev-parse --git-dir 2>/dev/null)" || {
    echo "Error: Not in a git repository"
    exit 1
}

# Make absolute if relative
case "$GIT_DIR" in
    /*) ;;
    *)  GIT_DIR="$REPO_ROOT/$GIT_DIR" ;;
esac

HOOKS_DIR="$GIT_DIR/hooks"
mkdir -p "$HOOKS_DIR"

# Copy pre-commit hook
cp "$SCRIPT_DIR/hooks/pre-commit" "$HOOKS_DIR/pre-commit"
chmod +x "$HOOKS_DIR/pre-commit"

echo "✅ Pre-commit hook installed successfully!"
echo ""
echo "The hook will:"
echo "  - Auto-fix code formatting and safe lint issues (Phase 1)"
echo "  - Log validation errors to .quality-errors.log (Phase 2)"
echo "  - Allow commits to proceed (warnings only)"
