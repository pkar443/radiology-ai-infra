# Remote Infer Service

This directory contains a minimal FastAPI service that loads MedGemma once at startup on the Hades workstation and exposes simple localhost HTTP endpoints for text inference, legacy text-only report generation, and the current Version 2 CT image-aware workflow.

This first version is intended as a stable remote inference scaffold for manual testing and later integration. The HTTP service, startup path, health checks, model loading, and response normalization are the primary goals here.

## File Structure

- `app.py`: FastAPI app and HTTP endpoints
- `model_loader.py`: one-time model loading plus text-only and image-text MedGemma generation logic
- `schemas.py`: Pydantic request and response schemas
- `config.py`: environment-backed runtime configuration and generation defaults
- `auth.py`: optional bearer-token check
- `utils.py`: logging, env bootstrapping, token sync, data URL decoding, MedGemma message assembly, prompt passthrough, and response normalization
- `start.sh`: activates the conda env and starts uvicorn on localhost
- `test_request.sh`: sends sample health, text, report, and image-aware requests to the running service

## Required Environment Variables

These can be exported in the shell or placed in `hades_setup/.env`:

- `MEDGEMMA_MODEL_ID`
- `MEDGEMMA_MODEL_PATH`
- `MEDGEMMA_DEVICE_MAP`
- `HF_HOME`
- `HUGGINGFACE_HUB_CACHE`
- `CUDA_VISIBLE_DEVICES`
- `REMOTE_INFER_HOST`
- `REMOTE_INFER_PORT`
- `REMOTE_INFER_AUTH_TOKEN`
- `REMOTE_INFER_DEBUG`
- `MEDGEMMA_MAX_NEW_TOKENS`
- `MEDGEMMA_DEFAULT_TEMPERATURE`
- `MEDGEMMA_DEFAULT_TOP_P`
- `MEDGEMMA_REPORT_MAX_NEW_TOKENS`
- `MEDGEMMA_REPORT_DO_SAMPLE`
- `MEDGEMMA_REPORT_TEMPERATURE`
- `MEDGEMMA_REPORT_TOP_P`
- `MEDGEMMA_IMAGE_REPORT_MAX_NEW_TOKENS`
- `MEDGEMMA_IMAGE_REPORT_DO_SAMPLE`
- `MEDGEMMA_IMAGE_REPORT_TEMPERATURE`
- `MEDGEMMA_IMAGE_REPORT_TOP_P`

Notes:

- If both `MEDGEMMA_MODEL_PATH` and `MEDGEMMA_MODEL_ID` are set, the local path is preferred.
- `MEDGEMMA_DEVICE_MAP` defaults to `single`, which loads the full MedGemma 4B model on the first visible GPU. Set `MEDGEMMA_DEVICE_MAP=auto` only if you intentionally want Hugging Face device sharding.
- `REMOTE_INFER_HOST` defaults to `127.0.0.1`.
- `REMOTE_INFER_PORT` defaults to `8009`.
- If `REMOTE_INFER_AUTH_TOKEN` is empty, bearer auth is bypassed.
- `REMOTE_INFER_DEBUG=true` keeps the request/output logs more verbose while still truncating previews.
- `/infer-report-test` uses `MEDGEMMA_REPORT_MAX_NEW_TOKENS=240`, `MEDGEMMA_REPORT_DO_SAMPLE=false`, `MEDGEMMA_REPORT_TEMPERATURE=0.0`, and `MEDGEMMA_REPORT_TOP_P=1.0` by default unless you override them in `hades_setup/.env`.
- `/infer-image-report` uses `MEDGEMMA_IMAGE_REPORT_MAX_NEW_TOKENS=900`, `MEDGEMMA_IMAGE_REPORT_DO_SAMPLE=false`, `MEDGEMMA_IMAGE_REPORT_TEMPERATURE=0.0`, and `MEDGEMMA_IMAGE_REPORT_TOP_P=1.0` by default unless you override them in `hades_setup/.env`.

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
- `POST /infer-image-report`

For this workstation, keeping `MEDGEMMA_DEVICE_MAP=single` is the safer default. We observed valid MedGemma generation on a single visible GPU and unusable pad-only continuations when the same model was sharded with `device_map="auto"` across both GPUs.

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

If you want deeper request/output visibility while debugging, set this in `hades_setup/.env` before starting the service:

```bash
REMOTE_INFER_DEBUG=true
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
- The preferred `POST /infer-image-report` flow is for the caller to send the already-selected `n-1`, `n`, and `n+1` CT slice images plus compact `instruction` and `query` text.
- Hades owns inference and optional response normalization only.
- A legacy compatibility path still accepts `study_id`, `modality`, `body_part`, `clinical_context`, and `findings_input`, but Hades now combines them in a thin passthrough form only.
- Hades no longer carries a second detailed radiology prompt template and does not re-select slices or rebuild report business logic.

## Image-Aware CT Endpoint

`POST /infer-image-report` is the Hades-side MedGemma image-aware endpoint for the current RadPilot Automatic CT workflow. The main app selects anchor groups and owns the instruction/query text. Hades trusts that ordered payload, decodes the images, runs MedGemma in notebook-style image-text mode, and normalizes the response.

Request shape:

```json
{
  "request_id": "req-123",
  "study_id": "123",
  "series_uid": "1.2.840....",
  "modality": "CT",
  "body_part": "Abdomen",
  "clinical_context": "Right upper quadrant pain",
  "instruction": "Instruction: Review the provided explainable CT anchor groups in order.",
  "query": "Return one JSON object only with report_text, technique, findings, impression, explanation_summary, structured_findings, and limitations.",
  "selection_strategy": "deterministic-uniform-non-overlapping-triplets",
  "anchor_group_count": 2,
  "anchor_groups": [
    {
      "anchor_id": "A01",
      "anchor_label": "A01 n",
      "center_slice_index": 120,
      "center_sop_instance_uid": "1.2.840....120",
      "slice_indices": [119, 120, 121],
      "slices": [
        {
          "slice_index": 119,
          "relative_position": "n-1",
          "anchor_label": "A01 n-1",
          "sop_instance_uid": "1.2.840....119",
          "image_data_url": "data:image/jpeg;base64,..."
        },
        {
          "slice_index": 120,
          "relative_position": "n",
          "anchor_label": "A01 n",
          "sop_instance_uid": "1.2.840....120",
          "image_data_url": "data:image/jpeg;base64,..."
        },
        {
          "slice_index": 121,
          "relative_position": "n+1",
          "anchor_label": "A01 n+1",
          "sop_instance_uid": "1.2.840....121",
          "image_data_url": "data:image/jpeg;base64,..."
        }
      ]
    }
  ],
  "slices": [
    {
      "slice_index": 119,
      "relative_position": "n-1",
      "anchor_label": "A01 n-1",
      "sop_instance_uid": "1.2.840....119",
      "anchor_id": "A01",
      "center_slice_index": 120,
      "image_data_url": "data:image/jpeg;base64,..."
    }
  ]
}
```

Response shape:

```json
{
  "request_id": "req-123",
  "report_text": "Technique:\nCT abdomen without contrast\n\nFindings:\nNo acute abnormality on the provided slices.\n\nImpression:\nNo acute abnormality on this limited slice review.",
  "technique": "CT abdomen without contrast",
  "findings": "No acute abnormality on the provided slices.",
  "impression": "No acute abnormality on this limited slice review.",
  "explanation_summary": "The selected anchor groups do not show a focal acute abnormality in this limited explainable CT context set.",
  "structured_findings": [
    {
      "id": "finding-1",
      "organ": "abdomen",
      "label": "No focal acute abnormality in reviewed anchor groups",
      "summary": "No focal acute abnormality is visible in the reviewed local slice groups.",
      "explanation": "The center slice A01 n is used as the primary anchor, with adjacent slices used as local confirmation only.",
      "anchor_slice_index": 120,
      "anchor_label": "A01 n",
      "supporting_anchors": ["A01 n-1", "A01 n+1"],
      "confidence": "low",
      "evidence": "No focal acute abnormality is visible across the provided local triplet around A01.",
      "abnormal": false
    }
  ],
  "limitations": "This is a selected anchor-group review, not a full-volume interpretation.",
  "model_id": "google/medgemma-1.5-4b-it",
  "device": "cuda:0",
  "inference_time_ms": 1234
}
```

Image handling notes:

- `image_data_url` must be a `data:image/<type>;base64,...` string.
- `anchor_groups` is the primary request structure. Hades flattens the groups in order and ignores the compatibility copy for inference ordering after validation.
- The flattened `slices` list must be an ordered compatibility copy of `anchor_groups`.
- Hades decodes each slice with Pillow, converts it to RGB, preserves caller order exactly, and raises a readable 422 error if any slice cannot be decoded.
- The current Automatic CT selector cap is 6 anchor groups, which means at most 18 provided slices per request.
- Hades validates structured JSON model output against the provided anchor labels and center slice indices, then falls back to text-section parsing if the model does not return valid JSON.

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
- If the model returns plain free text without report headers, the full cleaned text is returned in both `report_text` and `findings`.
- If parsing ever fails but cleaned raw text still exists, the service falls back to `report_text=<cleaned raw text>`, `technique=""`, `findings=<cleaned raw text>`, and `impression=""`.
- If cleanup empties the decoded text but the raw decoded continuation still contains usable English report text, the service preserves that raw text and logs `normalization_mode="raw_preserved_after_cleanup"`.
- A missing technique line does not fail the request.
- Whitespace-only, token-only, header-only, or otherwise unusable output returns HTTP 500 with a concise inference error because there is no caller-safe report text to return.

## Text-Only MedGemma Formatting

For text-only MedGemma 1.5 4B IT inference, Hades does not tokenize the caller prompt as a raw plain string. It wraps the caller prompt as a single user turn and adds the generation prompt for the model turn before `generate()`:

```python
messages = [
    {
        "role": "user",
        "content": [{"type": "text", "text": caller_prompt}],
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
```

If tokenized chat-template application is unavailable, Hades falls back to:

```python
prompt_text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = processor(prompt_text, return_tensors="pt")
```

## Notebook-Style Image-Text Formatting

`POST /infer-image-report` mirrors the reference notebook as closely as practical:

1. Build one single `user` message.
2. Assemble ordered content as:
   `instruction text -> image -> SLICE label -> image -> SLICE label -> ... -> final query text`
3. Keep the caller slice order exactly as received.
4. Use `AutoProcessor` plus `AutoModelForImageTextToText`.
5. Apply the chat template with `add_generation_prompt=True` and `continue_final_message=False`.
6. Run deterministic generation by default.
7. Decode with `processor.post_process_image_text_to_text(...)`.
8. Strip echoed prompt text if the decoded prompt appears at the start of the generated output.

Current slice-label assembly:

```python
content = [{"type": "text", "text": instruction}]
for slice_item in slices:
    content.append({"type": "image", "image": slice_item.image})
    content.append(
        {
            "type": "text",
            "text": f"SLICE {slice_item.relative_position} (index {slice_item.slice_index})",
        }
    )
content.append({"type": "text", "text": query})

messages = [{"role": "user", "content": content}]
```

Current generation path:

```python
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    continue_final_message=False,
    return_tensors="pt",
    tokenize=True,
    return_dict=True,
)

inputs = {
    name: tensor.to(device=model.device, dtype=model.dtype)
    if torch.is_floating_point(tensor)
    else tensor.to(device=model.device)
    for name, tensor in dict(inputs).items()
}

with torch.inference_mode():
    generated_sequence = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=300,
    )

decoded_output = processor.post_process_image_text_to_text(
    generated_sequence.cpu(),
    skip_special_tokens=True,
)[0]
decoded_prompt = processor.post_process_image_text_to_text(
    inputs["input_ids"].cpu(),
    skip_special_tokens=True,
)[0]
```

This keeps Hades focused on MedGemma-specific multimodal inference and normalization while the main app remains responsible for slice selection, CT workflow policy, and prompt ownership.

## Test Locally on Hades

In another shell on Hades:

```bash
cd /home/pkar443/medgemma_workspace
bash remote_infer/test_request.sh
```

The script sends:

- a very small known-good prompt
- a more realistic radiology prompt
- a tiny synthetic three-slice image-aware CT payload for `/infer-image-report`

Compare the API responses with the service logs in the terminal running `bash remote_infer/start.sh`.

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

For a local image-aware smoke test, use the helper script because it builds tiny inline JPEG data URLs for you:

```bash
bash remote_infer/test_request.sh
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
    "request_id": "ct-abd-known-good-001",
    "prompt": "Return a concise English radiology report using exactly these labels only: Technique, Findings, Impression. Technique may be blank. Findings should say mild hepatic steatosis. Impression should say no acute abdominal abnormality."
  }'
```

Example successful response:

```json
{
  "request_id": "ct-abd-known-good-001",
  "report_text": "Technique:\n\nFindings:\nMild hepatic steatosis.\n\nImpression:\nNo acute abdominal abnormality.",
  "technique": "",
  "findings": "Mild hepatic steatosis.",
  "impression": "No acute abdominal abnormality.",
  "model_id": "google/medgemma-1.5-4b-it",
  "device": "cuda:0",
  "inference_time_ms": 1187
}
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

## Debug Logs to Inspect

For `/infer-report-test`, inspect these structured log events in the terminal where the service is running:

- `event=infer_report_request`
- `event=infer_report_raw_output`
- `event=infer_report_cleaned_output`
- `event=infer_report_normalized`
- `event=infer_report_normalization_fallback`
- `event=infer_report_success`

For `/infer-image-report`, inspect:

- `event=infer_image_report_request`
- `event=infer_image_report_message_prepared`
- `event=infer_image_report_raw_output`
- `event=infer_image_report_cleaned_output`
- `event=infer_image_report_normalized`
- `event=infer_image_report_normalization_fallback`
- `event=infer_image_report_success`

The most useful fields are:

- `prompt_chars`
- `prompt_preview`
- `max_new_tokens`
- `temperature`
- `top_p`
- `do_sample`
- `slice_count`
- `ordered_slices`
- `raw_output_chars`
- `raw_output_preview`
- `cleaned_output_chars`
- `cleaned_output_preview`
- `text_before_cleanup_chars`
- `text_before_cleanup_preview`
- `text_after_cleanup_chars`
- `text_after_cleanup_preview`
- `text_source`
- `preserved_raw_text`
- `prompt_echo_removed`
- `generated_token_ids_head`
- `normalization_mode`
- `technique_preview`
- `findings_preview`
- `impression_preview`
