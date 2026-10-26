#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
CNN_ROOT="$(cd "$ROOT/.." && pwd)"
VENV="$CNN_ROOT/.venv/bin/activate"

if [[ ! -f "$VENV" ]]; then
  echo "Missing venv at $CNN_ROOT/.venv — create it first."
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV"
pip install -q -r "$ROOT/backend/requirements.txt"

cd "$ROOT/backend"
uvicorn app:app --host 127.0.0.1 --port 8000 &
API_PID=$!

cleanup() {
  kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT

sleep 2
echo "API: http://127.0.0.1:8000/health"

cd "$ROOT/frontend"
if [[ ! -d node_modules ]]; then
  npm install
fi
npm run dev -- --host 127.0.0.1 --port 5173
