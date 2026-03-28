# Remote Infer Service

This directory contains a minimal FastAPI service that loads MedGemma once at startup on the Hades workstation and exposes simple localhost HTTP endpoints for text inference and report generation.

This first version is intended as a stable remote inference scaffold for manual testing and later integration. The HTTP service, startup path, health checks, model loading, and response normalization are the primary goals here.

## File Structure

- `app.py`: FastAPI app and HTTP endpoints
- `model_loader.py`: one-time model loading and generation logic
- `schemas.py`: Pydantic request and response schemas
- `config.py`: environment-backed runtime configuration
- `auth.py`: optional bearer-token check
- `utils.py`: logging, env bootstrapping, token sync, prompt passthrough, and response normalization
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

## Run and Verify

Start the Hades inference service from a Hades terminal:

```bash
cd /home/pkar443/medgemma_workspace

# activate conda
source /home/pkar443/miniconda3/etc/profile.d/conda.sh
conda activate medgemma-hades

# start the service
bash remote_infer/start.sh
```

Leave that terminal running.

Verify on Hades from another terminal:

```bash
curl http://127.0.0.1:8009/health
```

If needed, run the local smoke requests:

```bash
bash remote_infer/test_request.sh
```

From your laptop, create the SSH tunnel:

```bash
ssh -N -L 8009:127.0.0.1:8009 pkar443@10.104.147.2
```

Then test through the tunnel:

```bash
curl.exe http://127.0.0.1:8009/health
```

## Prompt Ownership

The main app is the source of truth for report prompt construction.

- The preferred `POST /infer-report-test` flow is for the caller to send a full plain-text `prompt`.
- Hades owns inference and optional response normalization only.
- A legacy compatibility path still accepts `study_id`, `modality`, `body_part`, `clinical_context`, and `findings_input`, but Hades now combines them in a thin passthrough form only.
- Hades no longer carries a second detailed radiology prompt template.

## Report Draft Response Shape

`POST /infer-report-test` uses the caller-provided prompt as-is when `prompt` is present. If the generated text includes labeled `Technique`, `Findings`, and `Impression` sections, the service parses them. If not, it falls back to treating the full generated text as `findings`.

The service then normalizes the generated text and returns both the full `report_text` and parsed section fields:

```json
{
  "request_id": "ct-abd-smoke-001",
  "report_text": "Technique:\n\nFindings:\nMild diffuse hepatic steatosis. No bowel obstruction.\n\nImpression:\nNo acute abdominopelvic abnormality.",
  "technique": "",
  "findings": "Mild diffuse hepatic steatosis. No bowel obstruction.",
  "impression": "No acute abdominopelvic abnormality.",
  "model_id": "google/medgemma-1.5-4b-it",
  "device": "cuda:0",
  "inference_time_ms": 1234
}
```

Normalization notes:

- Header matching is case-insensitive and tolerates minor spacing variations.
- If `Findings:` and `Impression:` are present, those sections are parsed directly.
- If the model returns plain free text without report headers, the full text is kept in `findings` and `report_text` is rewritten into the labeled section format.
- A missing technique line does not fail the request.

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
    "request_id": "ct-abd-001",
    "prompt": "Draft a concise radiology report with exactly these sections and labels: Technique, Findings, Impression. Do not use markdown or bullet points. Technique may be blank if unknown. Modality: CT. Body part: Abdomen. Clinical context: Abdominal pain. Source findings: No bowel obstruction. Mild hepatic steatosis. No ascites."
  }'
```

Legacy compatibility request shape:

```json
{
  "study_id": "ct-abd-001",
  "modality": "CT",
  "body_part": "Abdomen",
  "clinical_context": "Abdominal pain.",
  "findings_input": "No bowel obstruction. Mild hepatic steatosis. No ascites."
}
```

For legacy field payloads, Hades only concatenates the supplied values with simple labels before inference. It does not inject a detailed radiology instruction template.

If `REMOTE_INFER_AUTH_TOKEN` is set, add:

```bash
-H "Authorization: Bearer YOUR_TOKEN"
```
