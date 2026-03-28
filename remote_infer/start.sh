#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[remote_infer:start] %s\n' "$*"
}

die() {
  printf '[remote_infer:start] ERROR: %s\n' "$*" >&2
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/hades_setup/.env"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-medgemma-hades}"

if [[ -f "${ENV_FILE}" ]]; then
  log "Loading environment overrides from ${ENV_FILE}"
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

HF_HOME="${HF_HOME:-${REPO_ROOT}/hf_cache}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}}"
MODELS_DIR="${MODELS_DIR:-${REPO_ROOT}/models}"
REMOTE_INFER_HOST="${REMOTE_INFER_HOST:-127.0.0.1}"
REMOTE_INFER_PORT="${REMOTE_INFER_PORT:-8009}"

export HF_HOME HUGGINGFACE_HUB_CACHE MODELS_DIR REMOTE_INFER_HOST REMOTE_INFER_PORT

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${MODELS_DIR}"

DEFAULT_TOKEN_PATH="${HOME}/.cache/huggingface/token"
ACTIVE_TOKEN_PATH="${HF_HOME}/token"
if [[ ! -s "${ACTIVE_TOKEN_PATH}" ]] && [[ -s "${DEFAULT_TOKEN_PATH}" ]]; then
  cp "${DEFAULT_TOKEN_PATH}" "${ACTIVE_TOKEN_PATH}"
  chmod 600 "${ACTIVE_TOKEN_PATH}"
  log "Synced Hugging Face token into workspace cache at ${ACTIVE_TOKEN_PATH}"
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_EXE="$(command -v conda)"
elif [[ -x "${CONDA_DIR}/bin/conda" ]]; then
  CONDA_EXE="${CONDA_DIR}/bin/conda"
else
  die "conda was not found. Run bash ${REPO_ROOT}/bootstrap.sh first."
fi

CONDA_BASE="$("${CONDA_EXE}" info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  die "Conda environment ${ENV_NAME} does not exist."
fi

conda activate "${ENV_NAME}"
cd "${REPO_ROOT}"

log "Starting uvicorn on ${REMOTE_INFER_HOST}:${REMOTE_INFER_PORT}"
exec uvicorn remote_infer.app:app --host "${REMOTE_INFER_HOST}" --port "${REMOTE_INFER_PORT}"
