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
    "study_id": "ct-abd-smoke-001",
    "modality": "CT",
    "body_part": "Abdomen",
    "clinical_context": "Abdominal pain.",
    "findings_input": "Mild diffuse hepatic steatosis. No focal liver lesion. Gallbladder is unremarkable. No biliary ductal dilatation. Pancreas, spleen, adrenal glands, and kidneys are without acute abnormality. No bowel obstruction. No ascites.",
    "request_id": "report-smoke-test"
  }' | pretty_print
