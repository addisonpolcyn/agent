#!/usr/bin/env bash
# Unit and offline integration tests. Live Claude tests: uv run pytest -m live
set -euo pipefail
cd "$(dirname "$0")/.."
uv run pytest "$@"
