#!/usr/bin/env bash
# Paradev teardown: stop the Jupyter server that run.sh started for this workspace.
set -uo pipefail
cd "${SUPERSET_WORKSPACE_PATH:-$(git rev-parse --show-toplevel)}" || exit 0
if [[ -f .paradev/.run-port ]]; then
  PORT=$(cat .paradev/.run-port)
  pids=$(lsof -ti "tcp:$PORT" -sTCP:LISTEN 2>/dev/null || true)
  [[ -n "$pids" ]] && kill $pids 2>/dev/null && echo "stopped Jupyter on port $PORT"
  rm -f .paradev/.run-port
fi
# Fallback: anything still running out of this workspace's venv.
pkill -f "$PWD/.venv/bin/" 2>/dev/null || true
exit 0
