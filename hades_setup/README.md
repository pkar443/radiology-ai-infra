# Hades MedGemma Setup

This directory contains the reusable setup logic that powers the repository-level `bootstrap.sh`. All scripts are written to work with relative paths so the whole folder can be cloned onto a fresh Linux machine and run in user space.

## Directory Contents

- `install_miniconda.sh`: installs Miniconda into `/home/pkar443/miniconda3` if `conda` is not already available.
- `setup_env.sh`: creates or reuses the `medgemma-hades` conda environment with Python 3.11 and installs the requested package stack.
- `check_gpu.sh`: prints `nvidia-smi`, Python details, Torch details, CUDA availability, device count, and GPU names.
- `smoke_test_torch.py`: verifies Torch import and runs a small matrix multiplication.
- `smoke_test_transformers.py`: imports Torch and Transformers, downloads `distilgpt2` on first run, and performs a tiny text-generation test.
- `hf_login.sh`: checks for a Hugging Face login token and guides you through `hf auth login` when needed.
- `install_gh.sh`: installs GitHub CLI in user space under `~/.local/gh-cli` and symlinks `~/.local/bin/gh`.
- `gh_auth.sh`: checks GitHub CLI authentication and prints the exact `gh auth login` flow for Hades.
- `test_medgemma.py`: loads MedGemma 1.5 from `MEDGEMMA_MODEL_ID` or `MEDGEMMA_MODEL_PATH` and runs a text prompt test with timing.
- `run_medgemma_test.sh`: activates the conda environment, loads `.env`, and runs `test_medgemma.py`.
- `start_jupyter.sh`: starts Jupyter Lab on localhost with a configurable port.
- `env.example`: template for local workstation environment variables.

## Recommended Run Order

Run these commands from the repository root:

```bash
bash bootstrap.sh
cp hades_setup/env.example hades_setup/.env
bash hades_setup/install_gh.sh
bash hades_setup/gh_auth.sh
bash hades_setup/check_gpu.sh
bash hades_setup/hf_login.sh
bash hades_setup/run_medgemma_test.sh
```

To start a notebook server on localhost port `8890`:

```bash
bash hades_setup/start_jupyter.sh 8890
```

## Notes About MedGemma 1.5

- The example environment file defaults `MEDGEMMA_MODEL_ID` to `google/medgemma-1.5-4b-it`.
- The example environment file computes `REPO_ROOT` dynamically and keeps `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, and `MODELS_DIR` inside the cloned repository workspace.
- MedGemma 1.5 on Hugging Face uses the Transformers ecosystem and the model card explicitly calls out `accelerate`.
- The smoke test intentionally uses `distilgpt2` so setup validation stays fast and lightweight.
- When you are ready to load MedGemma itself, keep the dedicated environment and cache settings from `.env` so model downloads stay in user space.
- GitHub CLI installation and authentication are optional for setup, but they are helpful if you want to push from Hades directly.

## Troubleshooting

- If `setup_env.sh` says it cannot find `conda`, rerun it. The script will install Miniconda into `/home/pkar443/miniconda3` if needed.
- If you want a single-command fresh-machine flow, use `bash bootstrap.sh` from the repository root.
- If `git push` fails for lack of credentials, run `bash hades_setup/install_gh.sh`, then `bash hades_setup/gh_auth.sh`, then complete the interactive `gh auth login --git-protocol https --web` flow.
- If PyTorch imports but `torch.cuda.is_available()` is `False`, verify that `nvidia-smi` still works in the same SSH session and rerun `bash hades_setup/check_gpu.sh`.
- If the Transformers smoke test hangs on first run, it is usually downloading model files. Check network access to Hugging Face and the cache directories in `.env`.
- If MedGemma loading fails with a 401/403-style Hugging Face error, run `bash hades_setup/hf_login.sh` and make sure your account has accepted the model terms for `google/medgemma-1.5-4b-it`.
- If Jupyter starts but you cannot connect from VS Code, confirm the SSH port forward points to the same localhost port you passed to `start_jupyter.sh`.
- If you want both RTX PRO 6000 GPUs visible in the environment, change `CUDA_VISIBLE_DEVICES=0` to `CUDA_VISIBLE_DEVICES=0,1` in `hades_setup/.env`.

## Assumptions

- Linux `x86_64`
- `bash`, `curl` or `wget`, and `nvidia-smi` are already present
- Outbound network access is available for Miniconda, PyTorch wheels, PyPI, and Hugging Face model downloads
