#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[install_gh] %s\n' "$*"
}

die() {
  printf '[install_gh] ERROR: %s\n' "$*" >&2
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-$HOME/.local/gh-cli}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
ARCH="$(uname -m)"

case "${ARCH}" in
  x86_64)
    GH_ARCH="amd64"
    ;;
  aarch64|arm64)
    GH_ARCH="arm64"
    ;;
  *)
    die "Unsupported architecture: ${ARCH}"
    ;;
esac

if command -v gh >/dev/null 2>&1; then
  log "gh is already available at $(command -v gh)"
  gh --version | head -n 1
  exit 0
fi

if [[ -x "${BIN_DIR}/gh" ]]; then
  log "gh is already installed at ${BIN_DIR}/gh"
  "${BIN_DIR}/gh" --version | head -n 1
  exit 0
fi

mkdir -p "${INSTALL_ROOT}" "${BIN_DIR}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

log "Resolving latest GitHub CLI release for linux ${GH_ARCH}"
RELEASE_JSON="${TMP_DIR}/latest.json"
curl -fsSL "https://api.github.com/repos/cli/cli/releases/latest" -o "${RELEASE_JSON}"

read -r GH_TAG GH_URL <<EOF
$(python3 - "${RELEASE_JSON}" "${GH_ARCH}" <<'PY'
import json
import sys

release_json = sys.argv[1]
arch = sys.argv[2]
suffix = f"linux_{arch}.tar.gz"

with open(release_json, "r", encoding="utf-8") as handle:
    payload = json.load(handle)

tag = payload["tag_name"]
url = ""
for asset in payload.get("assets", []):
    candidate = asset.get("browser_download_url", "")
    if candidate.endswith(suffix):
        url = candidate
        break

if not url:
    raise SystemExit(f"Could not find release asset ending with {suffix}")

print(tag, url)
PY
)
EOF

if [[ -z "${GH_TAG}" || -z "${GH_URL}" ]]; then
  die "Failed to resolve a GitHub CLI release URL."
fi

ARCHIVE_PATH="${TMP_DIR}/gh_${GH_TAG#v}_linux_${GH_ARCH}.tar.gz"
EXTRACT_DIR="${TMP_DIR}/extract"
TARGET_DIR="${INSTALL_ROOT}/${GH_TAG}"

if [[ -x "${TARGET_DIR}/bin/gh" ]]; then
  log "Latest GitHub CLI ${GH_TAG} is already installed at ${TARGET_DIR}"
else
  log "Downloading ${GH_URL}"
  curl -fsSL "${GH_URL}" -o "${ARCHIVE_PATH}"

  mkdir -p "${EXTRACT_DIR}"
  tar -xzf "${ARCHIVE_PATH}" -C "${EXTRACT_DIR}"

  EXTRACTED_SUBDIR="$(find "${EXTRACT_DIR}" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
  if [[ -z "${EXTRACTED_SUBDIR}" ]]; then
    die "Failed to locate extracted GitHub CLI directory."
  fi

  rm -rf "${TARGET_DIR}"
  mv "${EXTRACTED_SUBDIR}" "${TARGET_DIR}"
fi

ln -sfn "${TARGET_DIR}/bin/gh" "${BIN_DIR}/gh"

log "Installed gh to ${BIN_DIR}/gh"
"${BIN_DIR}/gh" --version | head -n 1

case ":${PATH}:" in
  *":${BIN_DIR}:"*)
    ;;
  *)
    log "${BIN_DIR} is not currently on PATH."
    log "Add it for this shell with: export PATH=\"${BIN_DIR}:\$PATH\""
    ;;
esac
