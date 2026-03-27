#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[bootstrap] %s\n' "$*"
}

die() {
  printf '[bootstrap] ERROR: %s\n' "$*" >&2
  exit 1
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_DIR="${ROOT_DIR}/hades_setup"
SETUP_SCRIPT="${SETUP_DIR}/setup_env.sh"

if [[ ! -d "${SETUP_DIR}" ]]; then
  die "Expected setup directory at ${SETUP_DIR}"
fi

if [[ ! -x "${SETUP_SCRIPT}" ]]; then
  die "Expected executable setup script at ${SETUP_SCRIPT}"
fi

log "Running reproducible Hades setup from ${ROOT_DIR}"
bash "${SETUP_SCRIPT}"

printf '\n'
log "Bootstrap complete."
log "Next steps:"
log "  1. Optional: cp ${SETUP_DIR}/env.example ${SETUP_DIR}/.env"
log "  2. bash ${SETUP_DIR}/check_gpu.sh"
log "  3. bash ${SETUP_DIR}/hf_login.sh"
log "  4. bash ${SETUP_DIR}/run_medgemma_test.sh"
