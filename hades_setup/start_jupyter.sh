#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[start_jupyter] %s\n' "$*"
}

die() {
  printf '[start_jupyter] ERROR: %s\n' "$*" >&2
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [[ -f "${ENV_FILE}" ]]; then
  log "Loading environment overrides from ${ENV_FILE}"
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-medgemma-hades}"
JUPYTER_HOST="${JUPYTER_HOST:-127.0.0.1}"
JUPYTER_PORT="${1:-${JUPYTER_PORT:-8888}}"

if command -v conda >/dev/null 2>&1; then
  CONDA_EXE="$(command -v conda)"
elif [[ -x "${CONDA_DIR}/bin/conda" ]]; then
  CONDA_EXE="${CONDA_DIR}/bin/conda"
else
  die "conda was not found. Run bash ${SCRIPT_DIR}/setup_env.sh first."
fi

CONDA_BASE="$("${CONDA_EXE}" info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  die "Conda environment ${ENV_NAME} does not exist. Run bash ${SCRIPT_DIR}/setup_env.sh first."
fi

conda activate "${ENV_NAME}"

log "Starting Jupyter Lab on ${JUPYTER_HOST}:${JUPYTER_PORT}"
exec jupyter lab --no-browser --ip="${JUPYTER_HOST}" --port="${JUPYTER_PORT}"
