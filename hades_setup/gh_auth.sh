#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[gh_auth] %s\n' "$*"
}

die() {
  printf '[gh_auth] ERROR: %s\n' "$*" >&2
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GH_BIN="${GH_BIN:-$HOME/.local/bin/gh}"

if command -v gh >/dev/null 2>&1; then
  GH_BIN="$(command -v gh)"
elif [[ ! -x "${GH_BIN}" ]]; then
  log "gh is not installed yet. Installing it in user space first."
  bash "${SCRIPT_DIR}/install_gh.sh"
fi

if [[ ! -x "${GH_BIN}" ]]; then
  die "Could not find a usable gh binary."
fi

log "Using gh at ${GH_BIN}"

if "${GH_BIN}" auth status >/dev/null 2>&1; then
  log "GitHub CLI is already authenticated."
  "${GH_BIN}" auth status
  exit 0
fi

log "GitHub CLI is not authenticated yet."
log "Run this interactive command next:"
printf '  %s auth login --git-protocol https --web\n' "${GH_BIN}"
log "After login, run:"
printf '  %s auth setup-git\n' "${GH_BIN}"
log "Then confirm with:"
printf '  %s auth status\n' "${GH_BIN}"
