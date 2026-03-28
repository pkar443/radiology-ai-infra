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
bash hades_setup/install_gh.sh
bash hades_setup/gh_auth.sh
bash hades_setup/hf_login.sh
bash hades_setup/run_medgemma_test.sh
bash remote_infer/start.sh
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
remote_infer/
  README.md
  app.py
  auth.py
  config.py
  model_loader.py
  schemas.py
  utils.py
  start.sh
  test_request.sh
  tests/
```

## Remote Inference Service

After model access and setup are working, you can start the minimal FastAPI service on Hades:

```bash
cd /home/pkar443/medgemma_workspace
source /home/pkar443/miniconda3/etc/profile.d/conda.sh
conda activate medgemma-hades
bash remote_infer/start.sh
```

Then test it locally on Hades:

```bash
bash remote_infer/test_request.sh
```

For laptop access later, tunnel the local-only API port:

```bash
ssh -N -L 8009:127.0.0.1:8009 pkar443@10.104.147.2
curl.exe http://127.0.0.1:8009/health
```

Service details and SSH tunnel usage are documented in [remote_infer/README.md](./remote_infer/README.md).

## Notes

- Everything stays in user space. No `sudo`, no system package changes.
- Runtime caches, downloaded models, tokens, and Miniconda installers are intentionally ignored by git.
- GitHub CLI can be installed in user space with `bash hades_setup/install_gh.sh`; it installs to `~/.local/gh-cli` and symlinks `~/.local/bin/gh`.
- MedGemma itself may still require approved Hugging Face access before `run_medgemma_test.sh` can download the model.
- Detailed script usage lives in [hades_setup/README.md](./hades_setup/README.md).
