# Default recipe
default: sync

# Show available recipes
help:
    @just --list

# Sync dependencies for all workspace packages
sync:
    uv sync

# Auto-format all packages
format:
    @echo "==> Formatting all packages"
    uv run ruff check --fix
    uv run ruff format

# Run linting and type checks for all packages
check:
    @echo "==> Checking all packages with prek"
    uv run prek run --all-files

# Run all tests
test:
    @echo "==> Running all tests"
    uv run pytest

# Format a specific package
format-pkg pkg:
    @echo "==> Formatting {{pkg}}"
    uv run ruff check --fix packages/{{pkg}}
    uv run ruff format packages/{{pkg}}

# Check a specific package
check-pkg pkg:
    @echo "==> Checking {{pkg}} (ruff + ty)"
    uv run ruff check packages/{{pkg}}
    uv run ruff format --check packages/{{pkg}}
    uv run ty check packages/{{pkg}}/src packages/{{pkg}}/tests

# Test a specific package
test-pkg pkg:
    @echo "==> Testing {{pkg}}"
    uv run pytest packages/{{pkg}}/tests -vv

# Build all packages
build:
    @echo "==> Building all packages"
    uv build --package kcastle-agent --no-sources --out-dir dist/agent
    uv build --package kcastle --no-sources --out-dir dist/tui

# Clean build artifacts
clean:
    rm -rf dist .ruff_cache .pytest_cache
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Install Git hooks
hooks:
    uv run prek install
