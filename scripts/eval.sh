#!/usr/bin/env bash
# Run the eval suite and record a flywheel iteration. Offline by default; pass --live for Claude.
set -euo pipefail
cd "$(dirname "$0")/.."
args=(--offline)
if [[ "${1:-}" == "--live" ]]; then args=(); fi
uv run agentlab flywheel "${args[@]}"
