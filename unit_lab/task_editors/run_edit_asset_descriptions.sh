#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PYTHON="/home/agxi/miniconda3/envs/issac/bin/python"
PYTHON_BIN="${GENIESIM_EDITOR_PYTHON:-$DEFAULT_PYTHON}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/edit_asset_descriptions.py" "$@"
