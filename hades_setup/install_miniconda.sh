#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[install_miniconda] %s\n' "$*"
}

die() {
  printf '[install_miniconda] ERROR: %s\n' "$*" >&2
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
ARCH="$(uname -m)"
HELPER_BIN_DIR="${SCRIPT_DIR}/bin"

if command -v conda >/dev/null 2>&1; then
  log "conda is already available at $(command -v conda); skipping Miniconda install."
  exit 0
fi

if [[ -x "${CONDA_DIR}/bin/conda" ]]; then
  log "Miniconda is already installed at ${CONDA_DIR}; skipping reinstall."
  exit 0
fi

case "${ARCH}" in
  x86_64)
    INSTALLER_NAME="Miniconda3-latest-Linux-x86_64.sh"
    ;;
  aarch64)
    INSTALLER_NAME="Miniconda3-latest-Linux-aarch64.sh"
    ;;
  *)
    die "Unsupported architecture: ${ARCH}"
    ;;
esac

INSTALLER_PATH="${SCRIPT_DIR}/${INSTALLER_NAME}"
INSTALLER_URL="https://repo.anaconda.com/miniconda/${INSTALLER_NAME}"

if ! df -Pk "${HOME}" >/dev/null 2>&1; then
  log "System df is unavailable; using the bundled user-space df helper."
  export PATH="${HELPER_BIN_DIR}:${PATH}"
fi

if [[ -d "${CONDA_DIR}" ]] && [[ -z "$(ls -A "${CONDA_DIR}")" ]]; then
  log "Removing empty target directory left by a previous failed install: ${CONDA_DIR}"
  rmdir "${CONDA_DIR}"
fi

if [[ ! -f "${INSTALLER_PATH}" ]]; then
  log "Downloading ${INSTALLER_NAME} to ${INSTALLER_PATH}"
  if command -v curl >/dev/null 2>&1; then
    curl -L "${INSTALLER_URL}" -o "${INSTALLER_PATH}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${INSTALLER_PATH}" "${INSTALLER_URL}"
  else
    die "Neither curl nor wget is available for downloading Miniconda."
  fi
else
  log "Installer already exists at ${INSTALLER_PATH}; reusing it."
fi

log "Installing Miniconda into ${CONDA_DIR}"
bash "${INSTALLER_PATH}" -b -p "${CONDA_DIR}"

if [[ ! -x "${CONDA_DIR}/bin/conda" ]]; then
  die "Miniconda installation finished but ${CONDA_DIR}/bin/conda was not found."
fi

log "Miniconda installation complete."
"${CONDA_DIR}/bin/conda" --version
