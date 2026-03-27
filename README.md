# Radiology AI Infra

Reproducible, user-space setup for MedGemma experimentation on the Hades GPU workstation. The repository is organized so a fresh machine can be prepared with one command and without root access.

## Quick Start (Fresh Machine)

```bash
git clone https://github.com/pkar443/radiology-ai-infra.git
cd radiology-ai-infra
bash bootstrap.sh
```

After bootstrap finishes:

```bash
cp hades_setup/env.example hades_setup/.env
bash hades_setup/check_gpu.sh
bash hades_setup/hf_login.sh
bash hades_setup/run_medgemma_test.sh
```

## What Bootstrap Does

- Detects Miniconda and installs it into `/home/pkar443/miniconda3` only if missing
- Initializes conda in-process without changing the system
- Creates or reuses the `medgemma-hades` environment with Python 3.11
- Installs the required PyTorch, Transformers, FastAPI, Jupyter, and Hugging Face packages
- Keeps `huggingface_hub` pinned below `1.0` for Transformers `v4.x` compatibility
- Creates workspace-local cache and model directories under `medgemma_workspace`

## Repository Layout

```text
bootstrap.sh
README.md
.gitignore
hades_setup/
  README.md
  env.example
  install_miniconda.sh
  setup_env.sh
  check_gpu.sh
  hf_login.sh
  run_medgemma_test.sh
  start_jupyter.sh
  smoke_test_torch.py
  smoke_test_transformers.py
  test_medgemma.py
  bin/df
```

## Notes

- Everything stays in user space. No `sudo`, no system package changes.
- Runtime caches, downloaded models, tokens, and Miniconda installers are intentionally ignored by git.
- MedGemma itself may still require approved Hugging Face access before `run_medgemma_test.sh` can download the model.
- Detailed script usage lives in [hades_setup/README.md](./hades_setup/README.md).
