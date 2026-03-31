#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/hades_setup/.env"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

REMOTE_INFER_HOST="${REMOTE_INFER_HOST:-127.0.0.1}"
REMOTE_INFER_PORT="${REMOTE_INFER_PORT:-8009}"
BASE_URL="http://${REMOTE_INFER_HOST}:${REMOTE_INFER_PORT}"

AUTH_ARGS=()
if [[ -n "${REMOTE_INFER_AUTH_TOKEN:-}" ]]; then
  AUTH_ARGS=(-H "Authorization: Bearer ${REMOTE_INFER_AUTH_TOKEN}")
fi

pretty_print() {
  python3 -m json.tool
}

build_image_report_payload() {
  python3 <<'PY'
import base64
import io
import json

from PIL import Image


def image_data_url(color):
    image = Image.new("RGB", (16, 16), color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_anchor_group(anchor_id, center_slice_index, colors):
    triplet = [center_slice_index - 1, center_slice_index, center_slice_index + 1]
    return {
        "anchor_id": anchor_id,
        "anchor_label": f"{anchor_id} n",
        "center_slice_index": center_slice_index,
        "center_sop_instance_uid": f"1.2.840.0.1.2.3.{center_slice_index}",
        "slice_indices": triplet,
        "slices": [
            {
                "slice_index": triplet[0],
                "relative_position": "n-1",
                "anchor_label": f"{anchor_id} n-1",
                "sop_instance_uid": f"1.2.840.0.1.2.3.{triplet[0]}",
                "image_data_url": image_data_url(colors[0]),
            },
            {
                "slice_index": triplet[1],
                "relative_position": "n",
                "anchor_label": f"{anchor_id} n",
                "sop_instance_uid": f"1.2.840.0.1.2.3.{triplet[1]}",
                "image_data_url": image_data_url(colors[1]),
            },
            {
                "slice_index": triplet[2],
                "relative_position": "n+1",
                "anchor_label": f"{anchor_id} n+1",
                "sop_instance_uid": f"1.2.840.0.1.2.3.{triplet[2]}",
                "image_data_url": image_data_url(colors[2]),
            },
        ],
    }


def flatten_anchor_groups(anchor_groups):
    flattened = []
    for anchor_group in anchor_groups:
        for slice_item in anchor_group["slices"]:
            flattened.append(
                {
                    **slice_item,
                    "anchor_id": anchor_group["anchor_id"],
                    "center_slice_index": anchor_group["center_slice_index"],
                }
            )
    return flattened


anchor_groups = [
    build_anchor_group("A01", 120, [(64, 96, 128), (96, 128, 160), (128, 160, 192)]),
    build_anchor_group("A02", 150, [(96, 80, 64), (128, 112, 96), (160, 144, 128)]),
]

payload = {
    "request_id": "image-report-smoke-test",
    "study_id": "ct-image-smoke-001",
    "series_uid": "1.2.840.0.1.2.3",
    "modality": "CT",
    "body_part": "Abdomen",
    "clinical_context": "Right upper quadrant pain",
    "instruction": (
        "Instruction: You are reviewing an explainable CT context set from one study. "
        "Each anchor group is ordered as n-1, n, n+1. The center slice n is primary evidence. "
        "Adjacent slices are local confirmation only. Base your answer only on the provided slices "
        "and do not claim full-volume review."
    ),
    "query": (
        "Return one JSON object only with report_text, technique, findings, impression, "
        "explanation_summary, structured_findings, and limitations."
    ),
    "selection_strategy": "deterministic-uniform-non-overlapping-triplets",
    "anchor_group_count": len(anchor_groups),
    "anchor_groups": anchor_groups,
    "slices": flatten_anchor_groups(anchor_groups),
}

print(json.dumps(payload))
PY
}

printf '\n=== /health ===\n'
curl -sS "${BASE_URL}/health" | pretty_print

printf '\n=== /infer-text ===\n'
curl -sS -X POST \
  "${BASE_URL}/infer-text" \
  -H "Content-Type: application/json" \
  "${AUTH_ARGS[@]}" \
  -d '{
    "prompt": "Summarize the purpose of a CT abdomen report in two concise sentences.",
    "max_new_tokens": 120,
    "temperature": 0.0,
    "top_p": 1.0,
    "do_sample": false,
    "request_id": "text-smoke-test"
  }' | pretty_print

printf '\n=== /infer-report-test ===\n'
curl -sS -X POST \
  "${BASE_URL}/infer-report-test" \
  -H "Content-Type: application/json" \
  "${AUTH_ARGS[@]}" \
  -d '{
    "request_id": "smoke-raw-1",
    "prompt": "Return plain English text with exactly these labels: Technique, Findings, Impression. Findings should say right renal calculus. Impression should say no hydronephrosis.",
    "study_id": "3",
    "modality": "CT",
    "body_part": "Abdomen"
  }' | pretty_print

printf '\n=== /infer-report-test (realistic radiology prompt) ===\n'
curl -sS -X POST \
  "${BASE_URL}/infer-report-test" \
  -H "Content-Type: application/json" \
  "${AUTH_ARGS[@]}" \
  -d '{
    "study_id": "ct-abd-smoke-001",
    "prompt": "Draft a concise radiology report with exactly these sections and labels: Technique, Findings, Impression. Do not use markdown or bullet points. Do not invent demographics. Technique may be blank if unknown. Modality: CT. Body part: Abdomen. Clinical context: Abdominal pain. Source findings: Mild diffuse hepatic steatosis. No focal liver lesion. Gallbladder is unremarkable. No biliary ductal dilatation. Pancreas, spleen, adrenal glands, and kidneys are without acute abnormality. No bowel obstruction. No ascites.",
    "request_id": "report-smoke-test"
  }' | pretty_print

printf '\n=== /infer-image-report ===\n'
curl -sS -X POST \
  "${BASE_URL}/infer-image-report" \
  -H "Content-Type: application/json" \
  "${AUTH_ARGS[@]}" \
  -d "$(build_image_report_payload)" | pretty_print

printf '\nInspect the terminal running remote_infer/start.sh for:\n'
printf '  event=infer_report_request\n'
printf '  event=infer_report_raw_output\n'
printf '  event=infer_report_cleaned_output\n'
printf '  event=infer_report_normalized\n'
printf '  event=infer_image_report_request\n'
printf '  event=infer_image_report_raw_output\n'
printf '  event=infer_image_report_cleaned_output\n'
printf '  event=infer_image_report_normalized\n'
