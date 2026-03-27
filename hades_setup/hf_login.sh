#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[hf_login] %s\n' "$*"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
HF_HOME_DEFAULT="${WORKSPACE_DIR}/hf_cache"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

HF_HOME="${HF_HOME:-${HF_HOME_DEFAULT}}"
DEFAULT_TOKEN_PATH="${HOME}/.cache/huggingface/token"
ACTIVE_TOKEN_PATH="${HF_HOME}/token"

mkdir -p "${HF_HOME}"

if [[ -s "${ACTIVE_TOKEN_PATH}" ]]; then
  log "Hugging Face token found at ${ACTIVE_TOKEN_PATH}."
  log "Login helper check passed."
  exit 0
fi

if [[ -s "${DEFAULT_TOKEN_PATH}" ]]; then
  cp "${DEFAULT_TOKEN_PATH}" "${ACTIVE_TOKEN_PATH}"
  chmod 600 "${ACTIVE_TOKEN_PATH}"
  log "Hugging Face token found at ${DEFAULT_TOKEN_PATH}."
  log "Synced token into workspace cache at ${ACTIVE_TOKEN_PATH}."
  log "Login helper check passed."
  exit 0
fi

log "No Hugging Face token file was found."
log "Run the following command to log in interactively:"
printf '  hf auth login\n'
log "After login, rerun this script to confirm."
