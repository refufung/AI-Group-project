#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
ENET_ROOT="$(cd "$ROOT/.." && pwd)"
VENV="$ENET_ROOT/.venv/bin/activate"

if [[ ! -f "$VENV" ]]; then
  echo "Missing venv at $ENET_ROOT/.venv — create it first."
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV"
pip install -q -r "$ROOT/backend/requirements.txt"

cd "$ROOT/frontend"
if [[ ! -d node_modules ]]; then
  npm install
fi
if [[ ! -d dist ]]; then
  npm run build
fi

cd "$ROOT/backend"
exec uvicorn app:app --host 127.0.0.1 --port 8000
