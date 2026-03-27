#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[check_gpu] %s\n' "$*"
}

die() {
  printf '[check_gpu] ERROR: %s\n' "$*" >&2
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

printf '\n=== nvidia-smi ===\n'
nvidia-smi

printf '\n=== Python and Torch ===\n'
python - <<'PY'
import platform
import sys
import torch

print(f"python version: {platform.python_version()} ({sys.executable})")
print(f"torch version: {torch.__version__}")
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

count = torch.cuda.device_count()
print(f"torch.cuda.device_count(): {count}")

for idx in range(count):
    print(f"gpu[{idx}]: {torch.cuda.get_device_name(idx)}")
PY
