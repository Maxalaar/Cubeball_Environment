#!/usr/bin/env bash
set -euo pipefail

# Installs the project's Python dependencies into the CubeballEnvironment conda env.
# Required on NixOS: precompiled wheels (torch, etc.) need the FHS environment
# provided by conda-shell to find their dynamic libs (ld-linux, libc...).

ENV_NAME="CubeballEnvironment"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

conda-shell -c "conda run -n ${ENV_NAME} --no-capture-output pip install -e '${SCRIPT_DIR}'"
