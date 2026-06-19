#!/usr/bin/env bash
set -euo pipefail

# Show metadata for one task (labels, problem type, notes).
# Usage: bash bash_scripts/run_describe_task.sh [task_name]

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

TASK="${1:-guo_los}"

python inference/predict_next_value.py --task "${TASK}" --describe_task
