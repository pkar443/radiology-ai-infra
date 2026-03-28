# Remote Infer Service

This directory contains a minimal FastAPI service that loads MedGemma once at startup on the Hades workstation and exposes simple localhost HTTP endpoints for text inference and radiology-style report drafting.

This first version is intended as a stable remote inference scaffold for manual testing and later integration. The HTTP service, startup path, health checks, and model loading are the primary goals here. Prompt quality for text-only report drafting may still require iteration because MedGemma 1.5 is a multimodal medical model.

## File Structure

- `app.py`: FastAPI app and HTTP endpoints
- `model_loader.py`: one-time model loading and generation logic
- `schemas.py`: Pydantic request and response schemas
- `config.py`: environment-backed runtime configuration
- `auth.py`: optional bearer-token check
- `utils.py`: logging, env bootstrapping, token sync, and prompt helpers
- `start.sh`: activates the conda env and starts uvicorn on localhost
- `test_request.sh`: sends sample requests to the running service

## Required Environment Variables

These can be exported in the shell or placed in `hades_setup/.env`:

- `MEDGEMMA_MODEL_ID`
- `MEDGEMMA_MODEL_PATH`
- `HF_HOME`
- `HUGGINGFACE_HUB_CACHE`
- `CUDA_VISIBLE_DEVICES`
- `REMOTE_INFER_HOST`
- `REMOTE_INFER_PORT`
- `REMOTE_INFER_AUTH_TOKEN`
- `MEDGEMMA_MAX_NEW_TOKENS`
- `MEDGEMMA_DEFAULT_TEMPERATURE`
- `MEDGEMMA_DEFAULT_TOP_P`

Notes:

- If both `MEDGEMMA_MODEL_PATH` and `MEDGEMMA_MODEL_ID` are set, the local path is preferred.
- `REMOTE_INFER_HOST` defaults to `127.0.0.1`.
- `REMOTE_INFER_PORT` defaults to `8009`.
- If `REMOTE_INFER_AUTH_TOKEN` is empty, bearer auth is bypassed.

## Start the Service on Hades

From the repository root:

```bash
cp hades_setup/env.example hades_setup/.env
bash remote_infer/start.sh
```

The service will load MedGemma during startup and then serve:

- `GET /health`
- `POST /infer-text`
- `POST /infer-report-test`

If startup succeeds but generations still need quality tuning, treat that as prompt/model iteration work rather than an API availability issue.

## Test Locally on Hades

In another shell on Hades:

```bash
cd /home/pkar443/medgemma_workspace
bash remote_infer/test_request.sh
```

If you prefer manual curls:

```bash
curl http://127.0.0.1:8009/health
```

```bash
curl -X POST http://127.0.0.1:8009/infer-text \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Summarize this abdominal CT report in two concise sentences.",
    "max_new_tokens": 120,
    "temperature": 0.0,
    "top_p": 1.0,
    "do_sample": false
  }'
```

## Access Later from Your Laptop via SSH Tunnel

Bind the server to localhost on Hades and tunnel the port from your laptop:

```bash
ssh -L 8009:127.0.0.1:8009 user@HADES_IP
```

Then call the service from your laptop against `http://127.0.0.1:8009`.

Example local curl through the tunnel:

```bash
curl -X POST http://127.0.0.1:8009/infer-report-test \
  -H "Content-Type: application/json" \
  -d '{
    "modality": "CT",
    "body_part": "Abdomen",
    "clinical_context": "Abdominal pain.",
    "findings_input": "No bowel obstruction. Mild hepatic steatosis. No ascites."
  }'
```

If `REMOTE_INFER_AUTH_TOKEN` is set, add:

```bash
-H "Authorization: Bearer YOUR_TOKEN"
```
