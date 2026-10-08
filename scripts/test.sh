#!/usr/bin/env bash
# Unit and offline integration tests. Live tests: AGENTLAB_LIVE_TESTS=1 uv run pytest -m live
set -euo pipefail
cd "$(dirname "$0")/.."
uv run pytest "$@"
