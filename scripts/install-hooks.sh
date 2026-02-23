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

# Step 1: Install our quality check as the pre-commit hook
cp "$SCRIPT_DIR/hooks/pre-commit" "$HOOKS_DIR/pre-commit"
chmod +x "$HOOKS_DIR/pre-commit"
echo "  ✓ Quality check hook written"

# Step 2: Layer bd hooks on top with --chain.
# bd will rename our pre-commit to pre-commit.old and call it first,
# then flush JSONL. All other bd hooks (pre-push, post-merge, etc.) are
# also installed here.
echo "Installing bd (beads) hooks with --chain..."
uv run bd hooks install --chain

echo ""
echo "✅ All hooks installed successfully!"
echo ""
echo "Hook chain on commit:"
echo "  1. pre-commit.old  → Cacheness quality checks (ruff + ty)"
echo "  2. pre-commit      → bd JSONL flush"
echo "  3. pre-push        → bd stale-JSONL guard"
