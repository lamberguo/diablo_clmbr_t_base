#!/usr/bin/env bash
set -euo pipefail

# List available local adapter tasks for inference.
# Usage: bash bash_scripts/run_list_tasks.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

python inference/predict_next_value.py --list_tasks

