#!/usr/bin/env bash
# Paradev workspace setup: copy untracked local files from the root repo, then build .venv.
set -euo pipefail
cd "${SUPERSET_WORKSPACE_PATH:-$(git rev-parse --show-toplevel)}"
ROOT="${SUPERSET_ROOT_PATH:-}"

# 1. Copy untracked, gitignored local files (secrets, local overrides) from the root checkout.
if [[ -n "$ROOT" && -d "$ROOT" && "$ROOT" != "$PWD" ]]; then
  for f in .env .env.local .env.development .env.test .paradev/config.local.json; do
    if [[ -f "$ROOT/$f" && ! -e "$f" ]]; then
      mkdir -p "$(dirname "$f")"
      cp "$ROOT/$f" "$f"
      echo "copied $f from root repo"
    fi
  done
fi

# 2. Create a virtualenv and install the dependencies listed in pyproject.toml.
PY="${PYTHON:-python3}"
if [[ ! -x .venv/bin/python ]]; then
  if command -v uv >/dev/null 2>&1; then uv venv .venv; else "$PY" -m venv .venv; fi
fi
deps=$(.venv/bin/python -c 'import tomllib;print("\n".join(tomllib.load(open("pyproject.toml","rb"))["project"]["dependencies"]))')
if command -v uv >/dev/null 2>&1; then
  echo "$deps" | xargs uv pip install --python .venv/bin/python
else
  .venv/bin/python -m pip install --quiet --upgrade pip
  echo "$deps" | xargs .venv/bin/python -m pip install --quiet
fi
echo "setup complete: activate with 'source .venv/bin/activate'"
