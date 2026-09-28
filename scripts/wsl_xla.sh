#!/usr/bin/env bash
# Run a script under CPU torch_xla inside WSL. From PowerShell in the repo root:
#
#     wsl -d Ubuntu -- bash scripts/wsl_xla.sh scripts/show_xla_graph.py --weights /mnt/c/Users/varun/models/voxtral-mini-3b
#
# The venv is ~/xla-venv inside WSL: Python 3.11 from uv, torch 2.9.0 CPU, torch_xla 2.9.0.
# uv's Python keeps libpython in its own folder, so the loader has to be pointed at it.
set -euo pipefail
PY=~/xla-venv/bin/python
export LD_LIBRARY_PATH="$($PY -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PJRT_DEVICE=CPU
exec "$PY" -u "$@"
