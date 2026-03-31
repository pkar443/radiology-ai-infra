#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[run_medgemma_raw_image_test] %s\n' "$*"
}

die() {
  printf '[run_medgemma_raw_image_test] ERROR: %s\n' "$*" >&2
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-medgemma-hades}"
HF_HOME_DEFAULT="${WORKSPACE_DIR}/hf_cache"
MODELS_DIR_DEFAULT="${WORKSPACE_DIR}/models"

if [[ -f "${ENV_FILE}" ]]; then
  log "Loading environment overrides from ${ENV_FILE}"
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

HF_HOME="${HF_HOME:-${HF_HOME_DEFAULT}}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}}"
MODELS_DIR="${MODELS_DIR:-${MODELS_DIR_DEFAULT}}"
MEDGEMMA_MODEL_ID="${MEDGEMMA_MODEL_ID:-google/medgemma-1.5-4b-it}"
MEDGEMMA_DEVICE_MAP="${MEDGEMMA_DEVICE_MAP:-single}"
export HF_HOME HUGGINGFACE_HUB_CACHE MODELS_DIR MEDGEMMA_MODEL_ID MEDGEMMA_DEVICE_MAP

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
  die "conda was not found. Run bash ${SCRIPT_DIR}/setup_env.sh first."
fi

CONDA_BASE="$("${CONDA_EXE}" info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  die "Conda environment ${ENV_NAME} does not exist."
fi

conda activate "${ENV_NAME}"

log "Running raw MedGemma image test with environment ${ENV_NAME} (device_map=${MEDGEMMA_DEVICE_MAP})"
exec python "${SCRIPT_DIR}/test_medgemma_raw_image.py" "$@"
