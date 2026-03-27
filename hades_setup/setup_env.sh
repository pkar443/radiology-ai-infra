#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[setup_env] %s\n' "$*"
}

die() {
  printf '[setup_env] ERROR: %s\n' "$*" >&2
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
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
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
HF_HOME="${HF_HOME:-${WORKSPACE_DIR}/hf_cache}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}}"
MODELS_DIR="${MODELS_DIR:-${WORKSPACE_DIR}/models}"

ensure_conda() {
  if command -v conda >/dev/null 2>&1; then
    CONDA_EXE="$(command -v conda)"
    return 0
  fi

  if [[ -x "${CONDA_DIR}/bin/conda" ]]; then
    CONDA_EXE="${CONDA_DIR}/bin/conda"
    return 0
  fi

  log "conda is not available; installing Miniconda into ${CONDA_DIR}"
  bash "${SCRIPT_DIR}/install_miniconda.sh"
  CONDA_EXE="${CONDA_DIR}/bin/conda"
}

activate_conda() {
  local conda_base
  conda_base="$("${CONDA_EXE}" info --base)"
  # shellcheck disable=SC1091
  source "${conda_base}/etc/profile.d/conda.sh"
}

accept_conda_tos() {
  log "Accepting conda Terms of Service for the default Anaconda channels"
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null
}

env_exists() {
  conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"
}

ensure_conda
activate_conda
accept_conda_tos

log "Workspace base: ${WORKSPACE_DIR}"
log "Using conda executable: ${CONDA_EXE}"
log "Ensuring workspace cache/model directories exist"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${MODELS_DIR}"

if env_exists; then
  log "Conda environment ${ENV_NAME} already exists; reusing it."
else
  log "Creating conda environment ${ENV_NAME} with Python ${PYTHON_VERSION}"
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

conda activate "${ENV_NAME}"

log "Python in active environment: $(python --version 2>&1)"
log "Upgrading base packaging tools inside ${ENV_NAME}"
python -m pip install --upgrade pip setuptools wheel

log "Installing CUDA-enabled PyTorch wheels from the official PyTorch index"
python -m pip install \
  "torch==2.9.1" \
  "torchvision==0.24.1" \
  "torchaudio==2.9.1" \
  --index-url https://download.pytorch.org/whl/cu128

log "Installing transformers and supporting packages"
python -m pip install \
  "transformers>=4.50.0,<5" \
  accelerate \
  sentencepiece \
  safetensors \
  protobuf \
  pillow \
  numpy \
  scipy \
  pydantic \
  fastapi \
  uvicorn \
  jupyter \
  ipykernel \
  "huggingface-hub[cli]>=0.34,<1.0"

log "Verifying installed package consistency"
python -m pip check

log "Registering the environment as a Jupyter kernel"
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (${ENV_NAME})"

log "Installed package versions:"
python - <<'PY'
from importlib.metadata import version

packages = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("torchaudio", "torchaudio"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("sentencepiece", "sentencepiece"),
    ("safetensors", "safetensors"),
    ("protobuf", "protobuf"),
    ("pillow", "Pillow"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pydantic", "pydantic"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("jupyter", "jupyter"),
    ("ipykernel", "ipykernel"),
    ("huggingface_hub", "huggingface_hub"),
]

for label, dist_name in packages:
    print(f"  {label}: {version(dist_name)}")
PY

log "Environment setup complete."
log "Next steps:"
log "  0. Optional: cp ${SCRIPT_DIR}/env.example ${SCRIPT_DIR}/.env"
log "  1. bash ${SCRIPT_DIR}/check_gpu.sh"
log "  2. bash ${SCRIPT_DIR}/hf_login.sh"
log "  3. bash ${SCRIPT_DIR}/run_medgemma_test.sh"
