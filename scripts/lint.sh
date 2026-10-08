#!/usr/bin/env bash
# Format check, lint, and strict type check: the same gates as CI.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run ruff format --check .
uv run ruff check .
uv run pyright
