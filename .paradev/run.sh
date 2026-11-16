#!/usr/bin/env bash
# Paradev run: launch Jupyter on a free port unique to this workspace.
set -euo pipefail
cd "${SUPERSET_WORKSPACE_PATH:-$(git rev-parse --show-toplevel)}"
[[ -x .venv/bin/jupyter ]] || ./.paradev/setup.sh

# Pick the first free port at or above $JUPYTER_PORT (default 8888) so parallel workspaces don't collide.
PORT=$(.venv/bin/python - "${JUPYTER_PORT:-8888}" <<'PY'
import socket, sys
p = int(sys.argv[1])
while True:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", p)); break
        except OSError:
            p += 1
print(p)
PY
)
echo "Starting Jupyter for workspace '${SUPERSET_WORKSPACE_NAME:-local}' on http://localhost:$PORT"
echo "$PORT" > .paradev/.run-port
exec .venv/bin/jupyter lab --no-browser --ip=127.0.0.1 --port="$PORT" --port-retries=0 \
  --ServerApp.root_dir="$PWD" --ServerApp.token='' --ServerApp.password=''
