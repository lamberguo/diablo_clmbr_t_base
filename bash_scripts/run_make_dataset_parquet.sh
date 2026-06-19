#!/usr/bin/env bash
set -euo pipefail

# Convert user CSV -> dataset/data/data.parquet
# Usage: bash bash_scripts/run_make_dataset_parquet.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

INPUT_CSV="dataset/raw_events.csv"
OUTPUT_DIR="dataset"
PATIENT_ID_COL="patient_id"
TIME_COL="event_time"
CODE_COL="code"
NUMERIC_VALUE_COL="value_num"
TEXT_VALUE_COL="value_text"
TIME_FORMAT=""

python data_preprocessing/make_dataset_parquet.py \
  --input_csv "${INPUT_CSV}" \
  --output_dir "${OUTPUT_DIR}" \
  --patient_id_col "${PATIENT_ID_COL}" \
  --time_col "${TIME_COL}" \
  --code_col "${CODE_COL}" \
  --numeric_value_col "${NUMERIC_VALUE_COL}" \
  --text_value_col "${TEXT_VALUE_COL}" \
  --time_format "${TIME_FORMAT}"

