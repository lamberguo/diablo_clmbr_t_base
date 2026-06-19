#!/usr/bin/env bash
set -euo pipefail

# Convert dataset/data/data.parquet -> dataset_pt/patients.pt
# Usage: bash bash_scripts/run_build_patient_cache.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET_PATH="dataset"
MAX_PATIENTS="20000"
OUTPUT_PT_PATH="dataset_pt/patients.pt"

python data_preprocessing/build_patient_cache.py \
  --dataset_path "${DATASET_PATH}" \
  --max_patients "${MAX_PATIENTS}" \
  --output_pt_path "${OUTPUT_PT_PATH}"

