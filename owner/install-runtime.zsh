#!/bin/zsh
set -eu
root=${0:A:h:h}
cd "$root"
[[ $(uname -m) == arm64 ]] || { print -u2 'Use a native arm64 shell.'; exit 1; }
bootstrap=$(mktemp -d)
trap 'rm -rf -- "$bootstrap"' EXIT
python3 -m pip install --disable-pip-version-check --target "$bootstrap/uv" 'uv==0.9.30'
export UV_PYTHON_INSTALL_DIR="$root/agent/runtime/python"
export UV_PYTHON_BIN_DIR="$bootstrap/bin"
export UV_CACHE_DIR="$bootstrap/cache"
uv="$bootstrap/uv/bin/uv"
"$uv" python install 3.12.12
[[ -x agent/runtime/venv/bin/python ]] || "$uv" venv --python 3.12.12 agent/runtime/venv
"$uv" pip sync --python agent/runtime/venv/bin/python owner/requirements.lock
