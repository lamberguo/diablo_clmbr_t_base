#!/usr/bin/env bash
set -euo pipefail

# Predict task value/label at a specific prediction time.
# Usage: bash bash_scripts/run_predict_next_value.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

TASK="guo_los"
ADAPTER_DIR=""
BASE_MODEL_NAME="StanfordShahLab/clmbr-t-base"
HF_TOKEN="${HF_TOKEN:-}"
HF_TOKEN_ENV="HF_TOKEN"
PATIENT_JSON="dataset/patient_template.json"
PREDICTION_TIME="2020-01-03 08:30:00"
OUTPUT_JSON="outputs/prediction_result.json"

CMD=(python inference/predict_next_value.py \
  --task "${TASK}" \
  --base_model_name "${BASE_MODEL_NAME}" \
  --hf_token_env "${HF_TOKEN_ENV}" \
  --patient_json "${PATIENT_JSON}" \
  --prediction_time "${PREDICTION_TIME}" \
  --output_json "${OUTPUT_JSON}")

if [[ -n "${ADAPTER_DIR}" ]]; then
  CMD+=(--adapter_dir "${ADAPTER_DIR}")
fi

if [[ -n "${HF_TOKEN}" ]]; then
  CMD+=(--hf_token "${HF_TOKEN}")
fi

"${CMD[@]}"

